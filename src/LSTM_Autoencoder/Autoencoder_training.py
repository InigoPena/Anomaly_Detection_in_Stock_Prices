from LSTM_Autoencoder import LSTMAutoencoder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path

####
# Load Data as data loader
####
sequence_dir = "data/sequences_ready"
tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
seq_len = 30
num_features = 5
batch_size = 64
window_size = 30

sequences = {}

for ticker in tickers:
    file_path = f"{sequence_dir}/{ticker}_data_sequences.csv"
    df = pd.read_csv(file_path)
    arr = df.values.reshape(-1, window_size, num_features) # Ecapsue the days in a window by its 5 features
    sequences[ticker.lower() + "_seq"] = arr

###
# Init and train models
###

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for trainig: {device}")
loss_fn = nn.MSELoss()

current_dir = Path(__file__).resolve().parent
performance_dir = current_dir / "model_performances"

for ticker in tickers:
    print(f"Training {ticker} model...")
    
    model = LSTMAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Usa el dataloader correspondiente
    data = DataLoader(TensorDataset(torch.tensor(sequences[ticker.lower() + "_seq"], dtype=torch.float32)), 
                      batch_size=batch_size, shuffle=True)
    
    loss_history = []

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in data:
            x = batch[0].to(device)
            output = model(x)
            loss = loss_fn(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(data)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/10 - Loss: {total_loss / len(data):.6f}")
    
    # Save The Loss History
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), loss_history, marker='o', label='Loss')
    plt.title(f"Training Loss - {ticker}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(performance_dir / f"{ticker.lower()}_loss.png")
    plt.close()
    print(f"Loss plot saved to model_performances/{ticker.lower()}_loss.png\n")

    # Save The Model
    torch.save(model.state_dict(), f"models/autoencoders/{ticker.lower()}_lstm_autoencoder.pth")
    print("\nModel saved\n")


