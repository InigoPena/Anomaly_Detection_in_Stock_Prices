import torch
import torch.nn as nn

class GRUAutoencoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, hidden = self.encoder(x)

        decoder_input = torch.zeros(x.size(0), x.size(1), self.encoder.hidden_size).to(x.device)
        decoded_output, _ = self.decoder(decoder_input, hidden)
        out = self.output_layer(decoded_output)

        return out

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)

    def forward(self, x):
        output, hidden = self.gru(x)
        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.2):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        return output, hidden