import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Baseline(nn.Module):
    # just a basic encoder-decoder with no VAE layers in the middle
    def __init__(self, hidden_dim, num_layers, embedding_weights, max_sentence_length, device, synthetic=False):
        super(Baseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = True

        self.encoder = Encoder(self.hidden_dim, num_layers, embedding_weights, max_sentence_length, device, synthetic)
        self.decoder = Decoder(self.hidden_dim, num_layers, embedding_weights, max_sentence_length, device, synthetic)

        if synthetic:
            for param in self.parameters():
                nn.init.uniform_(param, -0.01, 0.01)
            nn.init.uniform_(self.encoder.embed.weight, -0.1, 0.1)
            nn.init.uniform_(self.decoder.embed.weight, -0.1, 0.1)

    def encode(self, x, x_lens=None):
        batch_size, max_len, _ = x.shape
        hidden = self.encoder.init_hidden(batch_size)
        _, hidden = self.encoder.forward(x, hidden, x_lens)
        return hidden

    def decode(self, hidden, x, x_lens=None, train=True):
        outputs, _ = self.decoder.forward(x, hidden, x_lens, train)
        return outputs

    def forward(self, x, x_lens=None):
        # the last hidden state of the encoder is the first hidden state of the decoder
        hidden = self.encode(x, x_lens)
        outputs = self.decode(hidden, x, x_lens)
        return outputs
