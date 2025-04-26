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
def collate_pad(batch):
    P_list, y_list, n_list, m_list, k_list = zip(*batch)

    nk_vals = [P.shape[0] for P in P_list]
    k_vals = [P.shape[1] for P in P_list]
    max_k = max(k_vals)
    max_nk = max(nk_vals)
    B = len(batch)

    P_pad = torch.zeros(B, max_nk,  max_k, dtype=torch.float32)
    y = torch.cat(y_list, dim=0)

    n_t = torch.tensor(n_list, dtype=torch.float32).unsqueeze(1)
    m_t = torch.tensor(m_list, dtype=torch.float32).unsqueeze(1)
    k_t = torch.tensor(k_list, dtype=torch.float32).unsqueeze(1)
    pad_mask = torch.ones(B, max_nk, dtype=torch.bool)

    for i, P_seq in enumerate(P_list):
        nk_i = P_seq.shape[0]
        k_i = P_seq.shape[1]
        P_pad[i, :nk_i, :k_i] = P_seq
        pad_mask[i, :nk_i] = False
        
    return  P_pad, y, n_t, m_t, k_t, pad_mask

class ResultsDataset(Dataset):
    def __init__(self, df):
       
        self.P = []
        for raw_P, k, n in zip(df['P'], df['k'], df['n']):
            arr = np.array(raw_P, dtype=np.float32).reshape((k, n - k))
            P_seq = torch.from_numpy(arr.T).float()     
            self.P.append(P_seq)

        
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
            self.P[idx],          # (n_i-k_i, k_i)
            self.targets[idx],    # (1,)
            self.n_list[idx],     # int
            self.m_list[idx],     # int
            self.k_list[idx]      # int
        )
    
class Log2Loss(nn.Module):
    def forward(self, y_pred, y_true):
        eps = 1e-6
        log_p = torch.log2(y_pred.clamp(min=eps))   # clamp for numerical safety
        log_t = torch.log2(y_true.clamp(min=eps))
        return torch.mean((log_p - log_t) ** 2)
    
class TransformerWithP(nn.Module):
    def __init__(self, max_k, max_nk,
                 d_model=256, nhead=4, num_layers=2,
                 mlp_hidden=512, dropout=0.1):
        super().__init__()
        # 1) embed each P row into d_model
        self.P_proj  = nn.Linear(max_k,   d_model)
        # 2) positional embeddings for up to max_nk rows
        self.pos_emb = nn.Embedding(max_nk, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 3) one-dim scoring for attention pooling
        self.attn_pool = nn.Linear(d_model, 1, bias=False)

        # 4) final MLP head: (pooled_d + 3 scalars) → 1
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model + 3, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, P_seq, n, m, k, pad_mask):
        """
        P_seq:    (B, max_nk, max_k)
        n,m,k:    (B, 1)
        pad_mask: (B, max_nk)    # True = padding rows
        """
        B, S, _ = P_seq.shape

        #––– 1) token + position embedding
        P_emb = self.P_proj(P_seq)   # (B, S, d_model)
        ids   = torch.arange(S, device=P_emb.device)  \
                    .unsqueeze(0).expand(B, -1)      # (B, S)
        P_emb = P_emb + self.pos_emb(ids)            # add pos info

        #––– 2) Transformer encoder (skips padded rows)
        enc = self.transformer(P_emb, src_key_padding_mask=pad_mask)
        # enc: (B, S, d_model)

        #––– 3) Attention pooling over rows
        scores = self.attn_pool(enc).squeeze(-1)      # (B, S)
        scores = scores.masked_fill(pad_mask, -1e9)   # ignore padding
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, S, 1)
        pooled  = (enc * weights).sum(dim=1)          # (B, d_model)

        #––– 4) concat your 3 scalars
        scalars = torch.cat([n, m, k], dim=1)         # (B, 3)
        feats   = torch.cat([pooled, scalars], dim=1) # (B, d_model+3)

        #––– 5) MLP to one final output
        return self.mlp_head(feats)                   # (B, 1)

       
def create_dataset(df, n_lower, n_upper, k_lower, k_upper, m_lower):
    print("Creating datasets")

    for idx, row in df.iterrows():
        k = row["k"]
        n = row["n"]
        m = row["m"]
        row_P = np.array(row["P"], dtype=np.float32).reshape((k, n - k))
        x_dim = k


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
        for P_seq, y, n, m, k, mask in train_loader:
            P_seq, y, n, m, k, mask = P_seq.to(device), y.to(device), n.to(device),m.to(device),k.to(device), mask.to(device)
            optimizer.zero_grad()
            preds = net(P_seq, n, m, k, mask)
            
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} Train loss: {total_loss/len(train_loader):.4f}")
        net.eval()
        sigma = 0.0
        with torch.no_grad():
            for P_seq, y, n, m, k, mask in val_loader:
                # move to GPU (or CPU)
                P_seq, y, n, m, k, mask = (
                    P_seq.to(device),
                    y.to(device),
                    n.to(device), m.to(device), k.to(device),
                    mask.to(device)
                )

                preds = net(P_seq, n, m, k, mask)
                sigma += criterion(preds, y).item()

        print("length of val_loader: ", len(val_loader))
        sigma /= len(val_loader)
        print(f"Validation sigma: {sigma:.4f}")

        if sigma < best_sigma:
            best_sigma = sigma
            best_model = copy.deepcopy(net.state_dict())
    torch.save(best_model, "best_model.pt")
    print("Saved best model with sigma =", best_sigma)


def main():
    df = joblib.load("results_subset_1M.pkl")
    datasets = create_dataset(df, 9, 10, 4, 6, 2)
    train(datasets, num_epochs=50, learning_rate=.001)

if __name__ == "__main__":
    main()
