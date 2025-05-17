import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from transformers import TransformerAutoencoder

def train_on_dataset(csv_path, model_save_path, epochs=20, batch_size=64, lr=1e-4):
    # Load and prepare data
    data = pd.read_csv(csv_path).values
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(data)
    train_size = int(0.8 * len(dataset))
    train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Train loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            output = model(batch)
            loss = loss_fn(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{csv_path}] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

company_files = [
    "AAPL_data_sequences.csv",
    "GOOGL_data_sequences.csv",
    "MSFT_data_sequences.csv",
    "NVDA_data_sequences.csv",
    "TSLA_data_sequences.csv"
]

for file in company_files:
    name = file.split("_")[0]
    train_on_dataset(
        csv_path=f"../../data/sequences_ready/{file}",
        model_save_path=f"../../models/transformers/{name}_transformer.pth"
    )
