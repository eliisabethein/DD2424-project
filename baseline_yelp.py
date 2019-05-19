print("starting application: yelp data, baseline")
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
from baseline import Baseline
from baseline_util import Util

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

if not os.path.exists("results-baseline"):
    os.mkdir("results-baseline")
    print("Directory 'results-baseline' created ")
else:
    print("Directory 'results-baseline' already exists")

util = Util(device)

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

# without annealing
hidden_size = 1024
num_layers = 1
learning_rate = 0.001
epochs = 50

baseline_model = Baseline(hidden_size, num_layers, embedding_weights, max_sentence_length, device, synthetic=True).to(device)

verbose_level = 2

total_epoch_losses, val_total_epoch_losses = util.train_baseline(baseline_model, yelp_train_inputs, yelp_train_targets, yelp_val_inputs,
                yelp_val_targets, epochs, vocabulary_size, hidden_size, max_sentence_length,
                learning_rate=learning_rate, synthetic=True, verbose_level=verbose_level)


pickle.dump(total_epoch_losses, open("results-baseline/yelp_total_epoch_losses_baseline.pkl", "wb"))

pickle.dump(val_total_epoch_losses, open("results-baseline/yelp_val_total_epoch_losses_baseline.pkl", "wb"))

torch.save(baseline_model.state_dict(), "results-baseline/yelp-baseline.pwf")
