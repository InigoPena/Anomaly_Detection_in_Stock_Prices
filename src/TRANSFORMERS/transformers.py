import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # Make sure PE dimensions are correct
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        # pe shape: [1, max_len, d_model]
        return x + self.pe[:, :x.size(1), :].to(x.device)


# ---- STEP 3: Transformer Autoencoder ----
class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len=150, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]

        # Encoder input is now correctly shaped
        memory = self.encoder(x)  # [batch, seq_len, d_model]

        # For decoder in Transformer, we need a different target sequence
        # Here, we're using same sequence as input (autoencoder)
        out = self.decoder(x, memory)  # [batch, seq_len, d_model]

        return self.output_proj(out)  # [batch, seq_len, 1]



