import os
import pandas as pd
import numpy as np

# Rutas relativas dentro de tu proyecto
INPUT_DIR = "data/processed_data"
OUTPUT_DIR = "data/sequences_ready"
WINDOW_SIZE = 30

# Asegura que la carpeta de salida existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columnas normalizadas que se usarán
normalized_columns = [
    "Close_Normalized",
    "Open_Normalized",
    "High_Normalized",
    "Low_Normalized",
    "Volume_Normalized"
]

def create_multivariate_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        sequences.append(seq)
    return np.array(sequences)

# Procesamiento por archivo
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        filepath = os.path.join(INPUT_DIR, file)
        df = pd.read_csv(filepath)

        if not all(col in df.columns for col in normalized_columns):
            print(f"Saltando {file}: columnas normalizadas incompletas")
            continue

        data = df[normalized_columns].values
        sequences = create_multivariate_sequences(data, WINDOW_SIZE)
        sequences_flat = sequences.reshape(sequences.shape[0], -1)

        output_filename = file.replace(".csv", f"_sequences.csv")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        pd.DataFrame(sequences_flat).to_csv(output_path, index=False)

        print(f"✓ {file} → {output_filename} ({sequences.shape[0]} secuencias)")