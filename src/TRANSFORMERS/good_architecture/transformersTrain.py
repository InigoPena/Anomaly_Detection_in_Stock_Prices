import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
from transformersFinal import TransformerAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def main():
    company_files = [
        "AAPL_data_sequences.csv",
        "GOOGL_data_sequences.csv",
        "MSFT_data_sequences.csv",
        "NVDA_data_sequences.csv",
        "TSLA_data_sequences.csv"
    ]

    data_path = "../../../data/sequences_ready/"
    model_save_path = "../../../models/transformers/Good_model"
    plot_save_path = "../good_architecture/plots/"
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(plot_save_path, exist_ok=True)

    for file in company_files:
        company = file.split("_")[0]
        print(f"\nTraining model for {company}...")

        df = pd.read_csv(os.path.join(data_path, file))
        data = df.values.reshape(-1, 30, 5)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)

        # Train/Validation split
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        model = TransformerAutoencoder(input_dim=5, seq_len=30).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion)

        model_file = os.path.join(model_save_path, f"{company.lower()}_transformer_autoencoder.pth")
        torch.save(model.state_dict(), model_file)
        print(f"Model saved to {model_file}")

        # Plot loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", marker='o')
        plt.plot(val_losses, label="Val Loss", marker='x')
        plt.title(f"Training and Validation Loss for {company}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plot_file = os.path.join(plot_save_path, f"lossFunction_{company}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Loss plot saved to {plot_file}")

if __name__ == "__main__":
    main()

