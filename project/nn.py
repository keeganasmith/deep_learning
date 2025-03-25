import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
# Load the DataFrame

# Dataset Class
class MHeightDataset(Dataset):
    def __init__(self, df, scaler=None):
        self.features = []
        self.labels = []
        for _, row in df.iterrows():
            n, k, m = row['n'], row['k'], row['m']
            G = row['G']
            P = G[:, k:]  # Extract only the P part
            flat_P = P.flatten()
            x = np.concatenate([flat_P, [n, k, m]])
            self.features.append(x)
            self.labels.append(row['result'])
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32).reshape(-1, 1)

        if scaler:
            self.features = scaler.transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

def extract_flat_P(row):
    return row["G"][:, row["k"]:].flatten().astype(np.float32)


df = None

try:
    print("loading flattened df")
    df = joblib.load("flattened_df.pkl")
except:
    print("loading results df")
    df = joblib.load("results_dataframe.pkl")
    print("flattening G")
    df["flat_P"] = Parallel(n_jobs=192)(
        delayed(extract_flat_P)(row) for _, row in df.iterrows()
    )
    df["flat_P"] = pd.Series(df["flat_P"])
    joblib.dump(df, "flattened_df.pkl")



# Convert back to DataFrame column

# Split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print("got past split")

# Parallel construction of training feature matrix
print("generating training features (in parallel)")
def build_feature_vector(row):
    return np.concatenate([row["flat_P"], [row["n"], row["k"], row["m"]]], dtype=np.float32)

sample_features = Parallel(n_jobs=192)(
    delayed(build_feature_vector)(row) for _, row in train_df.iterrows()
)

sample_features = np.stack(sample_features)

# Fit scaler
scaler = StandardScaler().fit(sample_features)
print("finished scaler")

# Datasets and loaders
train_dataset = MHeightDataset(train_df, scaler)
val_dataset = MHeightDataset(val_df, scaler)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)
print("finished creating datasets")
# Model
class MHeightMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

input_dim = train_dataset[0][0].shape[0]
model = MHeightMLP(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("setting optimizer")
# Training loop
for epoch in range(10):
    print("epoch 1")
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item() * x_batch.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

