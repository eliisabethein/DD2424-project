import json
import nltk
import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse

from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import MWETokenizer
from vae import VAE

import csv
from gensim.models import Word2Vec
import os.path
import tarfile
import requests

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda:0')
else:
    print("Using CPU")
    device = torch.device('cpu')

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_data(filename, max_sentence_len, with_labels=False):
    # the tokenizer splits <unk> so we use MWETokenizer to re-merge it
    data_original = []
    data_padded = []
    with open(filename, encoding="utf8") as f:
        for line in f:
            sentence, padded_sentence = tokenize_sentence(line, max_sentence_len, with_labels)
            data_original.append(sentence)
            data_padded.append(padded_sentence)

    return data_original, data_padded

def tokenize_sentence(string, max_sentence_len, with_labels=False):
    merger = MWETokenizer([('<', 'unk', '>')], separator = '')
    sentence = word_tokenize(string.strip())       # tokenize sentence
    sentence = merger.tokenize(sentence)         # merge <unk>
    if with_labels:
        sentence = sentence[1:]
    sentence = [token.lower() for token in sentence]
    sentence = sentence[:max_sentence_len - 2]   # cut sentence at max_sentence_length
    sentence = ['<sos>'] + sentence + ['<eos>']  # add start and end-of-sentence tags

    # pad the rest of the sentence
    padded_sentence = sentence.copy()
    padded_sentence.extend(['<pad>']*(max_sentence_len - len(sentence)))

    return sentence, padded_sentence

def get_batches_text(data, data_padded, batch_size, pad_index, word2vec_model, unk_word='<unk>'):
    inputs = []
    targets = []
    lengths = []
    for i in range(len(data) // batch_size):
        # take batch_size sentences from the data each time
        batch_sentences = data[i*batch_size:(i+1)*batch_size]
        batch_sentence_lens = [len(x) for x in batch_sentences]

        # sentences in a batch have to be sorted in decreasing order of length (for pack_padded_sentence)
        sorted_pairs = sorted(zip(batch_sentence_lens,batch_sentences), reverse=True)
        batch_sentences = [sentence for length, sentence in sorted_pairs]
        batch_sentence_lens = [length-1 for length, sentence in sorted_pairs]

        # each input and target is a (batch_size x max_sentence_len-1 x 1) matrix
        # initially filled with the index for padditng tag <pad>
        input_batch = np.ones((batch_size, len(data_padded[0])-1, 1)) * pad_index
        target_batch = np.ones((batch_size, len(data_padded[0])-1, 1)) * pad_index

        # for each sentence in the batch, fill the corresponding row in current_batch
        # with the indexed of the words in the sentence (except for <pad>)
        for j, sentence in enumerate(batch_sentences):
            word_indexes = np.array([word2vec_model.wv.vocab[word].index if word in word2vec_model.wv.vocab else word2vec_model.wv.vocab[unk_word].index for word in sentence])
            input_batch[j,0:len(sentence)-1,0] = word_indexes[:-1]
            target_batch[j,0:len(sentence)-1,0] = word_indexes[1:]

        # make the matrices into torch tensors and append
        inputs.append(input_batch)
        targets.append(target_batch)
        lengths.append(batch_sentence_lens)
    return inputs, targets, lengths

def loss_function(outputs, labels, mean, log_variance, seq_length, device, annealing_args=None, mask=None):
    if mask is not None:
        BCE = torch.zeros(mean.shape[1] * (seq_length - 1), device=device)
        BCE[mask] = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
    else:
        BCE = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
    BCE = BCE.view(mean.shape[1], -1).sum(-1)
    KLD = -0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp()).permute(1, 0, 2).sum(-1).squeeze(-1)
    if annealing_args is not None:
        kl_weight = kl_annealing_weight(annealing_args['type'], annealing_args['step'], annealing_args['k'], annealing_args['first_step'])
    else:
        kl_weight = 1.0
    weighted_KLD = kl_weight * KLD
    loss = BCE + weighted_KLD
    return loss, BCE, KLD, weighted_KLD, kl_weight

def kl_annealing_weight(annealing_type, step, k, first_step):
    if annealing_type == 'logistic':
        return float(1/(1+np.exp(-k*(step-first_step))))
    elif annealing_type == 'linear':
        return min(1, step/first_step)

def train(vae, inputs, targets, validation_inputs, validation_targets, epochs, vocab_size, hidden_size, latent_size, max_sentence_length,
          device, input_lens=None, val_input_lens=None, synthetic=False,
          num_layers=1, step=1.0, learning_rate=0.001, tracked_inputs=None, tracked_targets=None, annealing_args=None,
          is_aggressive=False, verbose=True):

    opt_dict = {"not_improved": 0, "lr": 1.0, "best_loss": 1e4}
    decay_epoch = 2
    lr_decay = 0.5
    max_decay = 5

    enc_optimizer = torch.optim.SGD(vae.encoder.parameters(), lr=learning_rate)
    stoch_enc_optimizer = torch.optim.SGD(vae.stochastic_encoder.parameters(), lr=learning_rate)
    stoch_dec_optimizer = torch.optim.SGD(vae.stochastic_decoder.parameters(), lr=learning_rate)
    dec_optimizer = torch.optim.SGD(vae.decoder.parameters(), lr=learning_rate)

    if annealing_args is not None:
        kl_terms = []
        kl_weights = []

    iteration = decay_cnt = 0

    total_epoch_losses = []
    total_kl_losses = []
    total_mi = []
    val_total_epoch_losses = []
    val_total_kl_losses = []
    val_total_mi = []

    previous_mi = -1

    for epoch in range(epochs):
        for i in np.random.permutation(len(inputs)):

            inner_iter = 1
            random_i = i

            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            while is_aggressive and inner_iter < 100:
                x = inputs[random_i]
                y = torch.tensor(targets[random_i].reshape(-1), dtype=torch.long, device=device)
                x_lens = input_lens[random_i] if not synthetic else None

                enc_optimizer.zero_grad()
                stoch_enc_optimizer.zero_grad()
                stoch_dec_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                if synthetic:
                    burn_batch_size, burn_sents_len, _ = x.shape
                    burn_num_words += burn_sents_len * burn_batch_size
                else:
                    burn_num_words = np.sum(x_lens)

                mask = None
                mean, log_variance, outputs = vae(x, x_lens=x_lens)
                if not synthetic:
                    mask = (y < padding_index)
                    outputs = outputs[mask]
                    y = y[mask]

                loss_summary = loss_function(outputs, y, mean, log_variance, max_sentence_length, device, annealing_args=annealing_args, mask=mask)

                loss = loss_summary[0]
                burn_cur_loss += loss.sum().item()

                loss = loss.mean(dim=-1)
                loss.backward()

                clip_grad_norm_(vae.parameters(), 5.0)

                stoch_enc_optimizer.step()
                enc_optimizer.step()

                random_i = np.random.randint(0, len(inputs)- 1)
                if inner_iter % 15 == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0
                inner_iter += 1

            x = inputs[i]
            y = torch.tensor(targets[i].reshape(-1), dtype=torch.long, device=device)
            x_lens = input_lens[i] if not synthetic else None

            mask = None
            mean, log_variance, outputs = vae(x, x_lens=x_lens)

            if not synthetic:
                mask = (y < padding_index)
                outputs = outputs[mask]
                y = y[mask]

            enc_optimizer.zero_grad()
            stoch_enc_optimizer.zero_grad()
            stoch_dec_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            loss_summary = loss_function(outputs, y, mean, log_variance, max_sentence_length,  device, annealing_args=annealing_args, mask=mask)

            loss = loss_summary[0]
            loss = loss.mean(dim=-1)

            if annealing_args is not None:
                kl_terms.append(np.mean(loss_summary[2].data.numpy()))
                kl_weights.append(loss_summary[4])

            loss.backward()
            clip_grad_norm_(vae.parameters(), 5.0)

            if not is_aggressive:
                stoch_enc_optimizer.step()
                enc_optimizer.step()

            dec_optimizer.step()
            stoch_dec_optimizer.step()

            if (iteration % 100 == 0) and verbose:
                print('epoch {} iteration {} loss {:.3f} CE {:.3f} KL {:.3f} weighted KL: {:.3f} weight {:.3f}'.format(epoch+1,
                            iteration, loss, loss_summary[1].mean(dim=-1).data.item(), \
                            loss_summary[2].mean(dim=-1).data.item(), \
                            loss_summary[3].mean(dim=-1).data.item(), loss_summary[4]))

            iteration += 1

            if annealing_args is not None:
                annealing_args['step'] = iteration

        if is_aggressive:
            vae.eval()
            current_mi = calc_mi(vae, validation_inputs)
            vae.train()
            print('current_mi:', current_mi)
            if current_mi - previous_mi < 0:
                is_aggressive = False
                print("STOP AGGRESSIVE")

            previous_mi = current_mi

        # Validation
        vae.eval()
        with torch.no_grad():
            val_loss, val_kl, val_ppl, val_mi = test_vae(vae, validation_inputs, validation_targets, val_input_lens, device, synthetic, annealing_args)
            loss, kl, ppl, mi = test_vae(vae, inputs, targets, input_lens, device, synthetic, annealing_args)
            total_epoch_losses.append(loss)
            total_kl_losses.append(kl)
            total_mi.append(mi)
            val_total_epoch_losses.append(val_loss)
            val_total_kl_losses.append(val_kl)
            val_total_mi.append(val_mi)
            if verbose:
                print ('Epoch [{}/{}], Training Loss: {:.4f},  Training KL: {:.4f}, Training Perplexity: {:5.2f}, Validation Loss: {:.4f}, KL {:.4f}, Val Perplexity: {:5.2f}\n'
                       .format(epoch + 1, epochs, loss, kl, ppl, val_loss, val_kl, val_ppl))

            if val_loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= decay_epoch:
                    opt_dict["best_loss"] = val_loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    #vae.load_state_dict(torch.load(args.save_path))
                    print('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1

                    enc_optimizer = torch.optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"])
                    stoch_enc_optimizer = torch.optim.SGD(vae.stochastic_encoder.parameters(), lr=opt_dict["lr"])
                    stoch_dec_optimizer = torch.optim.SGD(vae.stochastic_decoder.parameters(), lr=opt_dict["lr"])
                    dec_optimizer = torch.optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"])
            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = val_loss

            if decay_cnt == max_decay:
                break
        vae.train()
    return total_epoch_losses, total_kl_losses, val_total_epoch_losses, val_total_kl_losses

def test_vae(model, inputs, targets, input_lens, device, synthetic=False, annealing_args=None):
    kl_loss = ce_loss = 0
    num_words = num_sents = 0
    for i in np.random.permutation(len(inputs)):
        x = inputs[i]
        y = torch.tensor(targets[i].reshape(-1), dtype=torch.long, device=device)
        x_lens = input_lens[i] if not synthetic else None

        batch_size, sents_len, _ = x.shape
        if synthetic:
            num_words += batch_size * sents_len
        else:
            num_words = np.sum(x_lens)

        num_sents += batch_size

        mask = None
        mean, log_variance, outputs = vae(x, x_lens=x_lens)
        if not synthetic:
            mask = (y < padding_index)
            outputs = outputs[mask]
            y = y[mask]

        loss_summary = loss_function(outputs, y, mean, log_variance, max_sentence_length,  device, annealing_args=annealing_args, mask=mask)

        loss_rc = np.sum(loss_summary[1].data.numpy())
        loss_kl = np.sum(loss_summary[3].data.numpy())

        ce_loss += loss_rc.item()
        kl_loss += loss_kl.item()

    mutual_info = calc_mi(vae, inputs)

    loss = (kl_loss + ce_loss) / num_sents
    kl = kl_loss / num_sents
    ppl = np.exp(loss * num_sents / num_words)

    return loss, kl, ppl, mutual_info


def calc_mi(model, test_data_batch):
    mi = 0
    num_examples = 0
    for batch_data in test_data_batch:
        batch_size = batch_data.shape[0]
        num_examples += batch_size
        mutual_info = model.calc_mi(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples

if not os.path.isdir("yelp_data"):
    destination = "destination.tar.gz"
    download_file_from_google_drive("1FT49oLNV8syhmGXEgiK6XTjEfMNqqEJJ", destination)
    tar = tarfile.open(destination, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(destination)

max_sentence_length = 50
yelp_train_data_original, yelp_train_data_padded = load_data("yelp_data/yelp.train.txt", max_sentence_length, with_labels=True)
yelp_test_data_original, yelp_test_data_padded = load_data("yelp_data/yelp.test.txt", max_sentence_length, with_labels=True)
yelp_val_dat_original, yelp_val_data_padded = load_data("yelp_data/yelp.valid.txt", max_sentence_length, with_labels=True)

embedding_size = 512
epochs_w2v = 100
word2vec_model_name = "word2vec_yelp.model"

if os.path.isfile(word2vec_model_name):
    print('Loading word2vec model:', word2vec_model_name)
    word2vec_yelp = Word2Vec.load(word2vec_model_name)
else:
    print('Training word2vec model')
    word2vec_yelp = Word2Vec(yelp_train_data_original, min_count=1, size=embedding_size, window=5)
    word2vec_yelp.train(yelp_train_data_original, epochs=epochs_w2v, total_examples=word2vec_yelp.corpus_count)
    word2vec_yelp.save(word2vec_model_name)
    print('Saved word2vec model:', word2vec_model_name)

# make the word embeddings into a pythorch tensor
embedding_weights = word2vec_yelp.wv.vectors
embedding_weights = np.vstack((embedding_weights, np.zeros((1,embedding_size))))  # add zero vector for <pad>
embedding_weights = torch.tensor(embedding_weights, device=device)

batch_size = 16
vocabulary_size = len(word2vec_yelp.wv.vocab)
padding_index = vocabulary_size

yelp_train_inputs, yelp_train_targets, yelp_train_lengths = \
                get_batches_text(yelp_train_data_original, yelp_train_data_padded, batch_size, padding_index, word2vec_yelp, '_unk')

yelp_test_inputs, yelp_test_targets, yelp_test_lengths = \
                get_batches_text(yelp_test_data_original, yelp_test_data_padded, batch_size, padding_index, word2vec_yelp, '_unk')

yelp_val_inputs, yelp_val_targets, yelp_val_lengths = \
                get_batches_text(yelp_val_dat_original, yelp_val_data_padded, batch_size, padding_index, word2vec_yelp, '_unk')

hidden_size = 1024
latent_size = 1
num_layers = 1
step = 0.25
learning_rate = 1.0
epochs = 5

vae = VAE(hidden_size, num_layers, embedding_weights, latent_size, max_sentence_length, device, synthetic=True).to(device)

total_epoch_losses, total_kl_losses, val_total_epoch_losses, val_total_kl_losses = train(vae, yelp_train_inputs, yelp_train_targets,
      yelp_val_inputs, yelp_val_targets, epochs, vocabulary_size,
      hidden_size, latent_size, max_sentence_length, device, yelp_train_lengths, learning_rate=learning_rate,
      synthetic=False, step=step, tracked_inputs=None, tracked_targets=None, verbose=True)
