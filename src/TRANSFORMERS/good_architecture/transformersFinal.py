# model.py

import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len=30, d_model=64, nhead=4, num_layers=2, input_dim=5):
        super(TransformerAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        x = self.input_projection(src)  # (batch_size, seq_len, d_model)
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding

        memory = self.encoder(x)
        out = self.decoder(x, memory)

        out = self.output_projection(out)  # (batch_size, seq_len, input_dim)
        return out

