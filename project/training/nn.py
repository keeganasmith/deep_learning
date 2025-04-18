import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt
import copy
from m_height import calculate_m_height  # (x, m, p)

# Custom collate to pad random_x and P columns into a unified token sequence
def collate_pad(batch):
    random_x_list, P_list, heights_list, targets = zip(*batch)
    ks = [x.shape[1] for x in random_x_list]
    nk_dims = [P.shape[1] for P in P_list]
    max_k = max(ks)
    max_nk = max(nk_dims)

    B = len(batch)
    x_pad = torch.zeros(B, 10, max_k, dtype=torch.float32)
    P_pad = torch.zeros(B, max_nk, max_k, dtype=torch.float32)
    h_pad = torch.zeros(B, 10,     dtype=torch.float32)

    for i, (x_np, P_np, h_np, tgt) in enumerate(batch):
        k = x_np.shape[1]
        nk = P_np.shape[1]
        x_pad[i, :, :k] = torch.from_numpy(x_np)
        P_full = torch.zeros(max_k, max_nk, dtype=torch.float32)
        P_full[:k, :nk] = torch.from_numpy(P_np)
        P_pad[i] = P_full[:max_k, :max_nk].T
        h_pad[i] = torch.from_numpy(h_np)
    x_tokens = torch.cat([x_pad, P_pad], dim=1)  # shape (B, 10+max_nk, max_k)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return x_tokens, h_pad, y


class RawResultsDataset(Dataset):
    """Dataset that returns raw random_x, P, m_heights, and target."""
    def __init__(self, df):
        self.random_x = [np.array(x, dtype=np.float32) for x in df['random_x']]
        self.P = []
        for idx, row in df.iterrows():
            k = row['k']
            n = row['n']
            P_arr = np.array(row['P'], dtype=np.float32).reshape((k, n - k))
            self.P.append(P_arr)
        self.m_heights = [np.array(h, dtype=np.float32) for h in df['m_heights']]
        self.targets = df['result'].astype(np.float32).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.random_x[idx], self.P[idx], self.m_heights[idx], self.targets[idx]


class Log2Loss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1)
        y_true = torch.clamp(y_true, min=1)
        return torch.mean((torch.log2(y_pred) - torch.log2(y_true)) ** 2)


class TransformerNetNoPE(nn.Module):
    def __init__(self, feature_dim, d_model=512, nhead=8, num_layers=6, mlp_hidden=2048, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=mlp_hidden,
            dropout=dropout, activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model + 10, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.BatchNorm1d(mlp_hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x_tokens, heights):
        # x_tokens: (B, seq_len, feature_dim)
        B, L, F = x_tokens.shape
        x = self.input_proj(x_tokens)        # (B, L, d_model)
        x = x.permute(1, 0, 2)               # (L, B, d_model)
        x = self.transformer_encoder(x)      # (L, B, d_model)
        x = x.permute(1, 0, 2)               # (B, L, d_model)
        pooled = x.mean(dim=1)               # (B, d_model)
        out = torch.cat([pooled, heights], dim=1)  # (B, d_model+10)
        return self.mlp_head(out)


def create_dataset(df, n_lower, n_upper, k_lower, k_upper, m_lower):
    print("Creating datasets")
    random_x_list = []
    m_heights_list = []

    for idx, row in df.iterrows():
        k = row["k"]
        n = row["n"]
        m = row["m"]
        row_P = np.array(row["P"], dtype=np.float32).reshape((k, n - k))
        x_dim = k

        curr_random_x = []
        curr_m_heights = []
        for _ in range(10):
            x_rand = np.random.uniform(low=-10, high=10, size=(x_dim,))
            curr_random_x.append(x_rand)
            m_height = calculate_m_height(x_rand, m, row_P)
            curr_m_heights.append(m_height)
            
        random_x_list.append(curr_random_x)
        m_heights_list.append(curr_m_heights)

    df["random_x"] = random_x_list
    df["m_heights"] = m_heights_list
    dataset = RawResultsDataset(df)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=512, collate_fn=collate_pad, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, collate_fn=collate_pad, num_workers=8, pin_memory=True)
    feature_dim = k_upper
    seq_len = 10 + (n_upper - k_lower)
    return {'train_loader': train_loader, 'val_loader': val_loader, 'feature_dim': feature_dim}


def train(datasets, num_epochs, learning_rate):
    train_loader = datasets['train_loader']
    val_loader   = datasets['val_loader']
    feature_dim  = datasets['feature_dim']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TransformerNetNoPE(feature_dim=feature_dim).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    criterion = Log2Loss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_sigma = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for x_tokens, heights, y in train_loader:
            x_tokens, heights, y = x_tokens.to(device), heights.to(device), y.to(device)
            optimizer.zero_grad()
            preds = net(x_tokens, heights)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} Train loss: {total_loss/len(train_loader):.4f}")
        net.eval()
        sigmas = []
        with torch.no_grad():
            for x_tokens, heights, y in val_loader:
                x_tokens, heights, y = x_tokens.to(device), heights.to(device), y.to(device)
                preds = net(x_tokens, heights).clamp(min=1)
                log_pred = torch.log2(preds)
                log_true = torch.log2(torch.clamp(y, min=1))
                sigmas.append(((log_pred - log_true)**2).mean().item())
        sigma = min(sigmas)
        print(f"Validation sigma: {sigma:.4f}")
        if sigma < best_sigma:
            best_sigma = sigma
            best_model = copy.deepcopy(net.state_dict())
    torch.save(best_model, "best_model.pt")
    print("Saved best model with sigma =", best_sigma)


def main():
    df = joblib.load("results_subset_1M.pkl")
    datasets = create_dataset(df, 9, 10, 4, 6, 2)
    train(datasets, num_epochs=50, learning_rate=1e-4)

if __name__ == "__main__":
    main()
