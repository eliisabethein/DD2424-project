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
import pickle

from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import MWETokenizer

import csv
from gensim.models import Word2Vec
import os.path
import tarfile
import requests
from tqdm import tqdm

class Util:
    def __init__(self, device):
        self.device = device
        print("Loaded Util")

    def download_file_from_google_drive(self, id, destination):
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = self.get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        self.save_response_content(response, destination)

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    def load_data(self, filename, max_sentence_len, with_labels=False):
        # the tokenizer splits <unk> so we use MWETokenizer to re-merge it
        data_original = []
        data_padded = []
        with open(filename, encoding="utf8") as f:
            for line in f:
                sentence, padded_sentence = self.tokenize_sentence(line, max_sentence_len, with_labels)
                data_original.append(sentence)
                data_padded.append(padded_sentence)

        return data_original, data_padded

    def tokenize_sentence(self, string, max_sentence_len, with_labels=False):
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

    def get_batches_text(self, data, data_padded, batch_size, pad_index, word2vec_model, unk_word='<unk>'):
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

    def baseline_loss_function(self, outputs, labels, seq_length, batch_size, mask=None):
        if mask is not None:
            BCE = torch.zeros(batch_size * (seq_length - 1), device=self.device)
            BCE[mask] = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        else:
            BCE = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        BCE = BCE.view(batch_size, -1).sum(-1)
        return BCE

    def test_baseline(self, model, inputs, targets, input_lens, max_sentence_length, synthetic=False):
            total_loss = 0
            num_words = num_sents = 0
            for i in np.random.permutation(len(inputs)):
                x = inputs[i]
                y = torch.tensor(targets[i].reshape(-1), dtype=torch.long, device=self.device)
                x_lens = input_lens[i] if not synthetic else None

                batch_size, sents_len, _ = x.shape
                if synthetic:
                    num_words += batch_size * sents_len
                else:
                    num_words = np.sum(x_lens)
                num_sents += batch_size

                mask = None
                outputs = model(x, x_lens=x_lens)
                if not synthetic:
                    mask = (y < padding_index)
                    outputs = outputs[mask]
                    y = y[mask]
                curr_loss = self.baseline_loss_function(outputs, y, max_sentence_length, batch_size, mask=mask)
                curr_loss = np.sum(curr_loss.cpu().data.numpy())
                total_loss += curr_loss.item()

            total_loss /= num_sents
            ppl = np.exp(total_loss * num_sents / num_words)
            return total_loss, ppl

    def train_baseline(self, model, inputs, targets, val_inputs, val_targets, epochs, vocab_size, hidden_size,
                       max_sentence_length, input_lens=None, val_input_lens=None, synthetic=False,
                       num_layers=1, learning_rate=0.001, verbose_level=1):

        opt_dict = {"not_improved": 0, "lr": learning_rate, "best_loss": 1e4}

        decay_epoch = 2
        lr_decay = 0.5
        max_decay = 5
        decay_cnt = 0

        enc_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
        dec_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)

        iteration = 0
        total_epoch_losses = []
        val_total_epoch_losses = []

        for epoch in range(epochs):
            for i in np.random.permutation(len(inputs)):
                x = inputs[i]
                y = torch.tensor(targets[i].reshape(-1), dtype=torch.long, device=self.device)
                x_lens = input_lens[i] if not synthetic else None

                batch_size, _, _ = x.shape

                mask = None
                # do the forward pass
                outputs = model(x, x_lens=x_lens)

                if not synthetic:
                    mask = (y < padding_index)
                    outputs = outputs[mask]
                    y = y[mask]

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                # compute the cross entropy loss
                loss = self.baseline_loss_function(outputs, y, max_sentence_length, batch_size, mask=mask)
                loss = loss.mean(dim=-1) # take the mean (same as divide by batch size?)

                # backward pass
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                enc_optimizer.step()
                dec_optimizer.step()

                if (iteration % 100 == 0) and (verbose_level == 2):
                    print('epoch {}\titeration {}\ttraining loss {:.3f}'.format(epoch+1, iteration, loss))

                iteration += 1

            # evaluate on the validation data
            model.eval()
            with torch.no_grad():
                train_loss, train_ppl = self.test_baseline(model, inputs, targets, input_lens, max_sentence_length, synthetic)
                val_loss, val_ppl = self.test_baseline(model, val_inputs, val_targets, val_input_lens, max_sentence_length, synthetic)
                total_epoch_losses.append(train_loss)
                val_total_epoch_losses.append(val_loss)
                if verbose_level > 0:
                    print ('Epoch [{}/{}], Training Loss: {:.4f} Perplexity: {:.4f}, Validation Loss: {:.4f} Validation Perplexity {:.4f}'
                           .format(epoch+1, epochs, train_loss, train_ppl, val_loss, val_ppl))

                # are we still decaying with the same logic?
                if val_loss > opt_dict["best_loss"]:
                    opt_dict["not_improved"] += 1
                    if opt_dict["not_improved"] >= decay_epoch:
                        opt_dict["best_loss"] = val_loss
                        opt_dict["not_improved"] = 0
                        opt_dict["lr"] = opt_dict["lr"] * lr_decay
                        #vae.load_state_dict(torch.load(args.save_path))
                        print('new lr: %f' % opt_dict["lr"])
                        decay_cnt += 1

                        enc_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=opt_dict["lr"])
                        dec_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=opt_dict["lr"])
                else:
                    opt_dict["not_improved"] = 0
                    opt_dict["best_loss"] = val_loss

                if decay_cnt == max_decay:
                    break
            model.train()

        return np.array(total_epoch_losses), np.array(val_total_epoch_losses)
