import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, latent_size=32, seq_len=30):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size, seq_len)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        latent = self.linear(h_n[-1])
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, latent):
        hidden = self.linear(latent)
        repeated = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.lstm(repeated)
        out = self.output_layer(decoded)
        return out