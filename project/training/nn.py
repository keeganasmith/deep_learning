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
    x_list, P_list, h_list, y_list, n_list, m_list, k_list = zip(*batch)

    k_vals  = [x.shape[0] for x in x_list]
    nk_vals = [P.shape[0] for P in P_list]
    max_k, max_nk = max(k_vals), max(nk_vals)
    B = len(batch)

    x_pad = torch.zeros(B, max_k,    10,    dtype=torch.float32)
    P_pad = torch.zeros(B, max_nk,   max_k, dtype=torch.float32)
    h_pad = torch.stack(h_list, dim=0)         # (B, 10)
    y     = torch.cat(y_list,     dim=0)       # (B, 1)

    n_t = torch.tensor(n_list, dtype=torch.float32).unsqueeze(1)
    m_t = torch.tensor(m_list, dtype=torch.float32).unsqueeze(1)
    k_t = torch.tensor(k_list, dtype=torch.float32).unsqueeze(1)

    for i, (x_seq, P_seq) in enumerate(zip(x_list, P_list)):
        k_i, nk_i = x_seq.shape[0], P_seq.shape[0]
        x_pad[i, :k_i, :]    = x_seq
        P_pad[i, :nk_i, :k_i] = P_seq

    lengths = [x.shape[0] + P.shape[0] for x, P in zip(x_list, P_list)]
    max_len  = max_k + max_nk
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    for i, L in enumerate(lengths):
        pad_mask[i, :L] = False   # False = attend, True = ignore

    return x_pad, P_pad, h_pad, y, n_t, m_t, k_t, pad_mask

class ResultsDataset(Dataset):
    def __init__(self, df):
        self.random_x = []
        for x_list in df['random_x']:
            x_arr = np.stack(x_list, axis=0)
            self.random_x.append(torch.from_numpy(x_arr.T).float())
        
        self.P = []
        for raw_P, k, n in zip(df['P'], df['k'], df['n']):
            arr = np.array(raw_P, dtype=np.float32).reshape((k, n - k))
            P_seq = torch.from_numpy(arr.T).float()          # → (n-k, k)
            self.P.append(P_seq)

        self.heights = [
            torch.from_numpy(np.array(h)).float()
            for h in df['m_heights']
        ]
        self.targets = torch.tensor(
            df['result'].values, dtype=torch.float32
        ).unsqueeze(1)

        # per-sample ints
        self.n_list = df['n'].astype(int).tolist()
        self.m_list = df['m'].astype(int).tolist()
        self.k_list = df['k'].astype(int).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.random_x[idx],   # (k_i, 10)
            self.P[idx],          # (n_i-k_i, k_i)
            self.heights[idx],    # (10,)
            self.targets[idx],    # (1,)
            self.n_list[idx],     # int
            self.m_list[idx],     # int
            self.k_list[idx]      # int
        )
    
class Log2Loss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1)
        y_true = torch.clamp(y_true, min=1)
        return torch.mean((torch.log2(y_pred) - torch.log2(y_true)) ** 2)
class TransformerWithP(nn.Module):
    def __init__(self, max_k, max_nk,
                 d_model=512, nhead=8, num_layers=6,
                 mlp_hidden=1024, dropout=0.1):
        super().__init__()
        self.x_proj = nn.Linear(10, d_model)          # random_x tokens
        self.P_proj = nn.Linear(max_k, d_model)       # P tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model* 2 + 10 + 3, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.BatchNorm1d(mlp_hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        ) 

    def forward(self, x_seq, P_seq, heights, n, m, k, pad_mask):
        """
        x_seq:   (B, max_k,   10)
        P_seq:   (B, max_nk,  max_k)
        heights: (B, 10)
        n, m, k: (B, 1)
        pad_mask:(B, max_k+max_nk)
        """
        x_emb = self.x_proj(x_seq)    # (B, max_k,   d_model)
        P_emb = self.P_proj(P_seq)    # (B, max_nk,  d_model)

        seq = torch.cat([x_emb, P_emb], dim=1)   # (B, S, d_model)
        enc = self.transformer(seq, src_key_padding_mask=pad_mask)

        k_len = x_emb.size(1)
        x_enc = enc[:, :k_len, :]    # (B, max_k,   d_model)
        P_enc = enc[:, k_len:, :]    # (B, max_nk,  d_model)

        pooled_x = x_enc.mean(dim=1)   # mean-pool over x columns -> (B, d_model)
        pooled_P = P_enc.mean(dim=1) # max-pool over P columns  -> (B, d_model)

        scalar_feats = torch.cat([n, m, k], dim=1)          # (B, 3)
        all_feats    = torch.cat([pooled_x, pooled_P, heights, scalar_feats], dim=1)

        return self.mlp_head(all_feats)  # (B, 1)


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
    dataset = ResultsDataset(df)
    train_idx, val_idx = train_test_split(list(range(len(dataset))),test_size=0.2, random_state=69)
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
    net = TransformerWithP(max_k = 6, max_nk = 6).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    criterion = Log2Loss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_sigma = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for x_tokens, P_seq, heights, y, n, m, k, mask in train_loader:
            x_tokens, P_seq, heights, y, n, m, k, mask = x_tokens.to(device), P_seq.to(device), heights.to(device), y.to(device), n.to(device),m.to(device),k.to(device), mask.to(device)
            optimizer.zero_grad()
            preds = net(x_tokens,P_seq, heights, n, m, k, mask)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} Train loss: {total_loss/len(train_loader):.4f}")
        net.eval()
        sigma = 0.0
        with torch.no_grad():
            for x_tokens, P_seq, heights, y, n, m, k, mask in val_loader:
                # move to GPU (or CPU)
                x_tokens, P_seq, heights, y, n, m, k, mask = (
                    x_tokens.to(device), P_seq.to(device),
                    heights.to(device), y.to(device),
                    n.to(device), m.to(device), k.to(device),
                    mask.to(device)
                )

                preds = net(x_tokens, P_seq, heights, n, m, k, mask)
                # Log2Loss already does clamp(>=1) + mean((log2ŷ-log2y)²)
                sigma += criterion(preds, y).item()

        # average over batches
        print("length of val_loader: ", len(val_loader))
        sigma /= len(val_loader)
        print(f"Validation sigma: {sigma:.4f}")

        if sigma < best_sigma:
            best_sigma = sigma
            best_model = copy.deepcopy(net.state_dict())
    torch.save(best_model, "best_model.pt")
    print("Saved best model with sigma =", best_sigma)


def main():
    df = joblib.load("results_dataframe.pkl")
    datasets = create_dataset(df, 9, 10, 4, 6, 2)
    train(datasets, num_epochs=50, learning_rate=1e-4)

if __name__ == "__main__":
    main()
