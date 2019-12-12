# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(self, data_size, hidden_size, output_size,
                 dropout, hidden_dim=128, gpu_enable=False):
        super(LSTMClassifier, self).__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gpu_enable = gpu_enable
        self.model = nn.LSTM(data_size, hidden_size, batch_first=True).double()
        self.hidden2out = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        if self.gpu_enable:
            return (
                Variable(torch.zeros(1, batch_size, self.hidden_size).double().cuda()),
                Variable(torch.zeros(1, batch_size, self.hidden_size).double().cuda())
            )
        else:
            return (
                Variable(torch.zeros(1, batch_size, self.hidden_size).double()),
                Variable(torch.zeros(1, batch_size, self.hidden_size).double())
            )

    def forward(self, X):
        hidden = self.init_hidden(batch_size=len(X))
        outputs, (h_n, c_n) = self.model(X.double(), hidden)
        # return self.softmax(self.hidden2out(outputs))
        return self.hidden2out(h_n[0])


class GRUClassifier(nn.Module):
    def __init__(self, data_size, hidden_size, output_size, dropout,
                 gpu_enable=False):
        super(GRUClassifier, self).__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gpu_enable = gpu_enable
        self.model = nn.GRU(data_size, hidden_size, batch_first=True).double()
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        if self.gpu_enable:
            return Variable(torch.zeros(1, batch_size, self.hidden_size).double().cuda())
        else:
            return Variable(torch.zeros(1, batch_size, self.hidden_size).double())

    def forward(self, X):
        hidden = self.init_hidden(batch_size=len(X))
        outputs, (h_n, c_n) = self.model(X.double(), hidden)
        return self.hidden2out(h_n[0])


class EnDecoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(EnDecoder, self).__init__()
        self.linear_1 = nn.Linear(D_in, H)
        self.linear_2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        return F.relu(self.linear_2(x))


class VAE(nn.Module):
    def __init__(self, encoder, decoder, encode_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encode_dim = encode_dim
        self.latent_dim = latent_dim
        self.__enc_mu = nn.Linear(encode_dim, latent_dim)
        self.__enc_log_sigma = nn.Linear(encode_dim, latent_dim)

    def __sample_latent(self, h_enc):
        mu = self.__enc_mu(h_enc)
        log_sigma = self.__enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).double()
        self.z_mean = mu
        self.z_sigma = sigma
        return mu + sigma * Variable(std_z, requires_grad=False)

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self.__sample_latent(h_enc=h_enc)
        return self.decoder(z)


class MLP(nn.Module):
    def __init__(self, data_size, hidden_size, output_size, n_class=2):
        super(MLP, self).__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer = nn.Linear(data_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(output_size, n_class)

    def forward(self, x):
        x = x.view(self.batch_size, self.data_size)
        return self.out(F.relu(self.output_layer(F.relu(self.hidden_layer(x)))))
