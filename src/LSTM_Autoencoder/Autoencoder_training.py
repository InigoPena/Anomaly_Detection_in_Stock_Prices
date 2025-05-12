from LSTM_Autoencoder import LSTMAutoencoder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn

####
# Load Data as data loader
####
sequence_dir = "data/sequences_ready"
tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
seq_len = 30
num_features = 1
batch_size = 64
window_size = 30

sequences = {}

for ticker in tickers:
    file_path = f"{sequence_dir}/{ticker}_data_sequences.csv"
    df = pd.read_csv(file_path)
    arr = df.values.reshape(-1, window_size, num_features)
    sequences[ticker.lower() + "_seq"] = arr
    
appl_tensor = torch.tensor(sequences["aapl_seq"], dtype=torch.float32)
appl_data = DataLoader(TensorDataset(appl_tensor), batch_size=batch_size, shuffle=True)

tsla_tensor = torch.tensor(sequences["tsla_seq"], dtype=torch.float32)
tsla_data = DataLoader(TensorDataset(tsla_tensor), batch_size=batch_size, shuffle=True)

googl_tensor = torch.tensor(sequences["googl_seq"], dtype=torch.float32)
googl_data = DataLoader(TensorDataset(googl_tensor), batch_size=batch_size, shuffle=True)

msft_tensor = torch.tensor(sequences["msft_seq"], dtype=torch.float32)
msft_data = DataLoader(TensorDataset(msft_tensor), batch_size=batch_size, shuffle=True)

nvda_tensor = torch.tensor(sequences["nvda_seq"], dtype=torch.float32)
nvda_data = DataLoader(TensorDataset(nvda_tensor), batch_size=batch_size, shuffle=True)

###
# Init and train models
###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training Apple model...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in appl_data:
        x = batch[0].to(device)
        output = model(x)
        loss = loss_fn(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss: {total_loss / len(appl_data):.6f}")

# Save model
torch.save(model.state_dict(), "src/Saved_models/apple_lstm_autoencoder.pth")
print(f"\nModel saved")