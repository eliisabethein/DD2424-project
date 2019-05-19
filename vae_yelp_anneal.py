print("starting application: yelp data, vae with annealing")
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
from vae import VAE
from vae_util import Util
import vae_util

import csv
from gensim.models import Word2Vec
import os.path
import tarfile
import requests

nltk.download('punkt')

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda:0')
else:
    print("Using CPU")
    device = torch.device('cpu')

if not os.path.exists("results-vae-anneal-32"):
    os.mkdir("results-vae-anneal-32")
    print("Directory 'results-vae-anneal-32' created ")
else:
    print("Directory 'results-vae-anneal-32' already exists")

util = Util()

if not os.path.isdir("yelp_data"):
    destination = "destination.tar.gz"
    util.download_file_from_google_drive("1FT49oLNV8syhmGXEgiK6XTjEfMNqqEJJ", destination)
    tar = tarfile.open(destination, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(destination)

max_sentence_length = 50
yelp_train_data_original, yelp_train_data_padded = util.load_data("yelp_data/yelp.train.txt", max_sentence_length, with_labels=True)
yelp_test_data_original, yelp_test_data_padded = util.load_data("yelp_data/yelp.test.txt", max_sentence_length, with_labels=True)
yelp_val_dat_original, yelp_val_data_padded = util.load_data("yelp_data/yelp.valid.txt", max_sentence_length, with_labels=True)

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
                util.get_batches_text(yelp_train_data_original, yelp_train_data_padded, batch_size, padding_index, word2vec_yelp, '_unk')

yelp_test_inputs, yelp_test_targets, yelp_test_lengths = \
                util.get_batches_text(yelp_test_data_original, yelp_test_data_padded, batch_size, padding_index, word2vec_yelp, '_unk')

yelp_val_inputs, yelp_val_targets, yelp_val_lengths = \
                util.get_batches_text(yelp_val_dat_original, yelp_val_data_padded, batch_size, padding_index, word2vec_yelp, '_unk')

hidden_size = 1024
latent_size = 32
num_layers = 1
step = 0.25
learning_rate = 0.001
epochs = 50

vae = VAE(hidden_size, num_layers, embedding_weights, latent_size, max_sentence_length, device, synthetic=True).to(device)

annealing_args = {'type': 'logistic', 'step': 0, 'k': 0.0025 / 6, 'first_step': 6 * 2500}

total_epoch_losses, total_kl_losses, total_mi, val_total_epoch_losses, val_total_kl_losses, val_total_mi = util.train(
        vae, yelp_train_inputs, yelp_train_targets, yelp_val_inputs, yelp_val_targets,
        epochs, vocabulary_size, hidden_size, latent_size, max_sentence_length, device,
        yelp_train_lengths, yelp_val_lengths, learning_rate=learning_rate, annealing_args=annealing_args,
        synthetic=False, step=step, tracked_inputs=None, tracked_targets=None, verbose=True)

pickle.dump(total_epoch_losses, open("results-vae-anneal-32/yelp_total_epoch_losses_vae_anneal_32.pkl", "wb"))
pickle.dump(total_kl_losses, open("results-vae-anneal-32/yelp_total_kl_losses_vae_anneal_32.pkl", "wb"))
pickle.dump(total_mi, open("results-vae-anneal-32/yelp_total_mi_vae_anneal_32.pkl", "wb"))

pickle.dump(val_total_epoch_losses, open("results-vae-anneal-32/yelp_val_total_epoch_losses_vae_anneal_32.pkl", "wb"))
pickle.dump(val_total_kl_losses, open("results-vae-anneal-32/yelp_val_total_kl_losses_vae_anneal_32.pkl", "wb"))
pickle.dump(val_total_mi, open("results-vae-anneal-32/yelp_val_total_mi_vae_anneal_32.pkl", "wb"))

torch.save(vae.state_dict(), "results-vae-anneal-32/yelp-vae-anneal-32.pwf")
