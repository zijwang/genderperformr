'''
    Rewritten model with new APIs
    Used to train the model described in Supplementary Material
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *
from .data.consts import EMB


class GenderPerformrModel(nn.Module):
    def __init__(self, batch_size=256, is_bidirection=False,
                 emb_out_size=32, lstm_hidden_size=512,
                 lstm_layers=2, lstm_dropout=0.2, lstm_out_size=128, device=torch.device('cpu'), **kwargs
                 ):
        super(GenderPerformrModel, self).__init__()
        self.batch_size = batch_size
        self.is_bidirection = True if is_bidirection == 1 else False
        self.emb_out_size = emb_out_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_out_size = lstm_out_size
        self.device = device
        self.username_embed = nn.Embedding(len(EMB) + 1, self.emb_out_size, padding_idx=0, max_norm=1)

        self.username_lstm = nn.LSTM(input_size=self.emb_out_size, hidden_size=self.lstm_hidden_size,
                                     num_layers=self.lstm_layers, batch_first=True, bidirectional=self.is_bidirection,
                                     dropout=self.lstm_dropout)

        num_features = self.username_lstm.hidden_size * (1 + self.is_bidirection)
        self.out_bn = nn.BatchNorm1d(num_features)
        self.out_dense = nn.Linear(in_features=num_features, out_features=self.lstm_out_size)

        self.final_dense = nn.Linear(in_features=lstm_out_size, out_features=2)
        self._init_dense(self.out_dense)
        self._init_dense(self.final_dense)

    @staticmethod
    def _init_dense(layer):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.normal_(layer.bias)

    def _init_hidden(self):
        self.username_h0 = torch.zeros((1 + self.is_bidirection) * self.lstm_layers, self.batch_size,
                                       self.username_lstm.hidden_size, device=self.device)
        self.username_c0 = torch.zeros((1 + self.is_bidirection) * self.lstm_layers, self.batch_size,
                                       self.username_lstm.hidden_size, device=self.device)

    def forward(self, data):

        username, lens = data
        self.batch_size = len(lens)
        self._init_hidden()

        username_embed = self.username_embed(username)
        lens_sorted, idx = lens.sort(0, descending=True)
        packed = pack_padded_sequence(username_embed[idx], lens_sorted.data.tolist(), batch_first=True)
        username_out, _ = self.username_lstm(packed, (self.username_h0, self.username_c0))
        username_out_unpacked, _ = pad_packed_sequence(username_out, batch_first=True)
        _, orig_idx = idx.sort(0)
        username_out_unpacked = username_out_unpacked[orig_idx]

        if self.is_bidirection:
            username_select = torch.cat([username_out_unpacked[torch.arange(0, self.batch_size, dtype=torch.long),
                                         lens - 1, :self.lstm_hidden_size],
                                         username_out_unpacked[torch.arange(0, self.batch_size, dtype=torch.long),
                                         torch.zeros(self.batch_size, dtype=torch.long), self.lstm_hidden_size:]], 1)
        else:
            username_select = username_out_unpacked[torch.arange(0, self.batch_size, dtype=torch.long), lens - 1]

        username_bn = self.out_bn(username_select)

        dense_out = self.out_dense(F.relu(username_bn, inplace=True))
        final_out = self.final_dense(dense_out)
        final_out = F.log_softmax(final_out, dim=1)

        return final_out
