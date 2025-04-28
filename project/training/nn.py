import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from m_height import calculate_m_height  # (x, m, p)

# Limit CPU threads per process to avoid oversubscription
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def collate_pad(batch):
    P_list, y_list, n_list, m_list, k_list = zip(*batch)

    nk_vals = [P.shape[0] for P in P_list]
    k_vals  = [P.shape[1] for P in P_list]
    max_k   = max(k_vals)
    max_nk  = max(nk_vals)
    B       = len(batch)

    P_pad    = torch.zeros(B, max_nk, max_k, dtype=torch.float32)
    y        = torch.stack(y_list, dim=0)
    n_t      = torch.tensor(n_list, dtype=torch.float32).unsqueeze(1)
    m_t      = torch.tensor(m_list, dtype=torch.float32).unsqueeze(1)
    k_t      = torch.tensor(k_list, dtype=torch.float32).unsqueeze(1)
    pad_mask = torch.ones(B, max_nk, dtype=torch.bool)

    for i, P_seq in enumerate(P_list):
        nk_i = P_seq.shape[0]
        k_i  = P_seq.shape[1]
        P_pad[i, :nk_i, :k_i] = P_seq
        pad_mask[i, :nk_i]    = False

    return P_pad, y, n_t, m_t, k_t, pad_mask

class ResultsDataset(Dataset):
    def __init__(self, df):
        self.raw_P   = df['P'].tolist()
        self.n_list  = df['n'].astype(int).tolist()
        self.m_list  = df['m'].astype(int).tolist()
        self.k_list  = df['k'].astype(int).tolist()
        self.targets = torch.tensor(df['result'].values,
                                    dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        raw_P = self.raw_P[idx]
        k     = self.k_list[idx]
        n     = self.n_list[idx]

        arr   = np.array(raw_P, dtype=np.float32).reshape((k, n - k))
        P_seq = torch.from_numpy(arr.T).float()

        return P_seq, self.targets[idx], n, self.m_list[idx], k

class Log2Loss(nn.Module):
    def forward(self, y_pred, y_true):
        eps   = 1e-6
        log_p = torch.log2(y_pred.clamp(min=eps))
        log_t = torch.log2(y_true.clamp(min=eps))
        return torch.mean((log_p - log_t) ** 2)

class TransformerWithP(nn.Module):
    def __init__(self, max_k, max_nk,
                 d_model=64, nhead=8, num_layers=2,
                 mlp_hidden=64, dropout=0.1):
        super().__init__()
        self.P_proj = nn.Linear(max_k, d_model)
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
            nn.Linear(2 * d_model + 3, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, P_seq, n, m, k, pad_mask):
        P_emb     = self.P_proj(P_seq)
        enc       = self.transformer(P_emb, src_key_padding_mask=pad_mask)
        valid     = (~pad_mask).unsqueeze(-1).float()
        enc_mask  = enc * valid
        sum_enc   = enc_mask.sum(dim=1)
        counts    = valid.sum(dim=1).clamp(min=1)
        mean_pool = sum_enc / counts
        neg_inf   = torch.finfo(enc.dtype).min
        enc_max   = enc.masked_fill(pad_mask.unsqueeze(-1), neg_inf)
        max_pool, _ = enc_max.max(dim=1)
        pooled    = torch.cat([mean_pool, max_pool], dim=1)
        scalars   = torch.cat([n, m, k], dim=1)
        all_feats = torch.cat([pooled, scalars], dim=1)
        return self.mlp_head(all_feats)


def create_dataloaders(df, batch_size):
    dataset     = ResultsDataset(df)
    train_idx, val_idx = train_test_split(list(range(len(dataset))),
                                          test_size=0.2,
                                          random_state=69)
    train_ds    = torch.utils.data.Subset(dataset, train_idx)
    val_ds      = torch.utils.data.Subset(dataset, val_idx)

    train_sampler = DistributedSampler(train_ds)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_pad,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_pad,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    return train_loader, val_loader


def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def train_loop(model, train_loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    for P_seq, y, n, m, k, mask in train_loader:
        P_seq, y, n, m, k = [t.to(device, non_blocking=True) for t in (P_seq, y, n, m, k)]
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            preds = model(P_seq, n, m, k, mask)
            loss  = criterion(preds, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    sigma = 0.0
    with torch.no_grad():
        for P_seq, y, n, m, k, mask in val_loader:
            P_seq, y, n, m, k = [t.to(device, non_blocking=True) for t in (P_seq, y, n, m, k)]
            mask = mask.to(device, non_blocking=True)
            preds = model(P_seq, n, m, k, mask)
            sigma += criterion(preds, y).item()
    return sigma / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data_path', type=str, default='large_results_dataframe.pkl')
    args = parser.parse_args()

    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")

    df = joblib.load(args.data_path)
    train_loader, val_loader = create_dataloaders(df, args.batch_size)

    model = TransformerWithP(max_k=6, max_nk=6).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.compile(model)

    criterion = Log2Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = GradScaler()

    best_sigma = float('inf')
    best_state = None

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        train_loss = train_loop(model, train_loader, optimizer, scaler, criterion, device)
        val_sigma  = validate(model, val_loader, criterion, device)

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{args.epochs} — Train loss: {train_loss:.4f}, Val sigma: {val_sigma:.4f}")
            if val_sigma < best_sigma:
                best_sigma = val_sigma
                best_state = model.module.state_dict()

    if dist.get_rank() == 0 and best_state is not None:
        torch.save(best_state, "best_model.pt")
        print(f"Saved best model with sigma={best_sigma:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
