import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_weights, max_sentence_length, device, synthetic=False):
        super(Decoder, self).__init__()
        # parameters
        self.vocabulary_size = embedding_weights.shape[0]
        self.embedding_size = embedding_weights.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = True
        self.device = device
        self.max_sentence_length = max_sentence_length

        # layers
        self.embed = nn.Embedding.from_pretrained(embedding_weights)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=self.batch_first)
        self.linear = nn.Linear(self.hidden_size, self.vocabulary_size)

    def forward(self, x, hidden, x_lens=None, train=True):
        batch_size, max_len, _ = x.shape

        x = torch.tensor(x, dtype=torch.long, device=self.device)  # make the input into a torch tensor
        x = self.embed(x).view(batch_size, max_len, self.embedding_size)

        if x_lens is not None and train:
            x_lens = torch.tensor(x_lens, dtype=torch.long, device=self.device)
            x = pack_padded_sequence(x, x_lens, batch_first=self.batch_first)

        h, c = hidden
        hidden = (h.contiguous(), c.contiguous())

        output, hidden = self.lstm(x.float(), hidden)

        if x_lens is not None and train:
            output, output_lens = pad_packed_sequence(output, batch_first=self.batch_first,
                                                      total_length=self.max_sentence_length-1)

        output = output.reshape(output.size(0)*output.size(1), output.size(2))
        output = self.linear(output)

        return output, hidden

class StochasticDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, latent_dim, device, synthetic=False):
        super(StochasticDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_first = True
        self.device = device

        self.latent_to_hidden = nn.Linear(latent_dim, 2 * self.hidden_dim * num_layers, self.batch_first)

    def forward(self, z):
        hidden_concatenated = self.latent_to_hidden(z)
        return hidden_concatenated
