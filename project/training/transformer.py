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
import torch.distributed as dist
import datetime
import os
from nn import Log2Loss
from nn import ResultsDataset
from nn import create_datasets

class TransformerNet(nn.Module):
    def __init__(self, p_rows, p_cols, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.3):
        """
        Args:
            p_rows: Number of rows in the original P matrix.
            p_cols: Number of columns in the original P matrix.
            d_model: Embedding dimension for tokens.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
        """
        super(TransformerNet, self).__init__()
        self.p_rows = p_rows
        self.p_cols = p_cols

        # Column embedding: each column (a vector of length p_rows) is mapped to a d_model-dimensional token.
        self.col_embedding = nn.Linear(p_rows, d_model)
        
        # Transformer encoder for the column tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Process the scalar features (n, k, m) into a representation of size d_model.
        self.scalar_fc = nn.Linear(3, d_model)
        
        # Additional fully-connected layers for deeper processing.
        self.fc1 = nn.Linear(2 * d_model, 2 * d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * d_model, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, total_features]
               where total_features = 3 + (p_rows * p_cols).
        Returns:
            out: Output tensor of shape [batch_size, 1].
        """
        batch_size = x.shape[0]
        
        # Split the input into scalars and the flattened P.
        scalars = x[:, :3]  # shape: [batch_size, 3]
        P_flat = x[:, 3:]   # shape: [batch_size, p_rows * p_cols]
        
        # Reshape P_flat to its original matrix shape: [batch_size, p_rows, p_cols]
        P = P_flat.view(batch_size, self.p_rows, self.p_cols)
        
        # For attention across columns, treat each column as a token.
        # Transpose to get shape [batch_size, p_cols, p_rows].
        tokens = P.transpose(1, 2)
        
        # Embed each column (token) into a d_model-dimensional vector.
        tokens = self.col_embedding(tokens)  # shape: [batch_size, p_cols, d_model]
        
        # Transformer encoder expects shape: [sequence_length, batch_size, d_model]
        tokens = tokens.transpose(0, 1)  # shape: [p_cols, batch_size, d_model]
        
        # Pass through the Transformer encoder.
        tokens = self.transformer_encoder(tokens)
        
        # Transpose back to shape: [batch_size, p_cols, d_model].
        tokens = tokens.transpose(0, 1)
        
        # Aggregate token representations via mean pooling over the column tokens.
        col_repr = tokens.mean(dim=1)  # shape: [batch_size, d_model]
        
        # Process the scalar features.
        scalar_repr = self.scalar_fc(scalars)  # shape: [batch_size, d_model]
        
        # Combine the representations.
        combined = torch.cat([col_repr, scalar_repr], dim=1)  # shape: [batch_size, 2 * d_model]
        
        # Deeper processing: additional fully connected layers.
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

# Training function with distributed training.
def train(datasets, num_epochs, learning_rate):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sigmas = []
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    idx = 0
    for n in datasets:
        for k in datasets[n]:
            for m in datasets[n][k]:
                if idx % world_size == rank:
                    element = datasets[n][k][m]
                    dataset = element["dataset"]
                    train_loader = element["train_loader"]
                    val_loader = element["val_loader"]
                    
                    # input_size = 3 + (p_rows * p_cols)
                    input_size = dataset.inputs.shape[1]
                    
                    # Instantiate the Transformer-based model.
                    net = TransformerNet(k, n - k, d_model=512, dim_feedforward=2048)
                    min_val_loss = float('inf')
                    best_model_state = None
                    net.to(device)
                    criterion = Log2Loss()
                    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                    train_losses = []
                    val_losses = []
                    sigma = float('inf')
                    for epoch in range(num_epochs):
                        net.train()
                        running_loss = 0.0

                        for inputs, targets in train_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            optimizer.zero_grad()
                            outputs = net(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()

                        avg_train_loss = running_loss / len(train_loader)
                        train_losses.append(avg_train_loss)
                        net.eval()
                        val_loss = 0.0
                        all_preds = []
                        all_targets = []
                        with torch.no_grad():
                            for inputs, targets in val_loader:
                                inputs, targets = inputs.to(device), targets.to(device)
                                preds = net(inputs)
                                loss = criterion(preds, targets)
                                val_loss += loss.item()
                                all_preds.append(preds.cpu())
                                all_targets.append(targets.cpu())

                        avg_val_loss = val_loss / len(val_loader)
                        val_losses.append(avg_val_loss)
                        if avg_val_loss < min_val_loss:
                            min_val_loss = avg_val_loss
                            best_model_state = copy.deepcopy(net.state_dict())
                        preds = torch.cat(all_preds).clamp(min=1.0)
                        targets = torch.cat(all_targets)
                        log_pred = torch.log2(preds)
                        log_true = torch.log2(targets)
                        sigma = min(((log_pred - log_true) ** 2).mean().item(), sigma)
                        print(f"{n}, {m}, {k} - Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}, Val log cost: {sigma:.4f}")

                    print("Final best validation sigma: ", sigma)
                    sigmas.append(sigma)
                    torch.save(best_model_state, f"../transformer_models/{n}-{k}-{m}_model.pt")

                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label="Training Loss")
                    plt.plot(val_losses, label="Validation Loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Training and Validation Loss")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"../transformer_plots/{n}-{k}-{m}_training_and_validation_loss.png")
                    print("Saved training plot for", n, k, m)
                idx += 1

    local_sum = torch.tensor([sum(sigmas)], device=device, dtype=torch.float32)
    local_count = torch.tensor([len(sigmas)], device=device, dtype=torch.float32)
    global_sum = local_sum.clone()
    global_count = local_count.clone()
    dist.barrier()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    overall_average = (global_sum / global_count).item()

    if dist.get_rank() == 0:
        print("Overall average sigma:", overall_average)

def main():
    print("Got to main")
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=60000))
    print("Finished initializing process group")
    df = joblib.load("results_subset_1M.pkl")
    print("Finished loading dataset")
    
    datasets = create_datasets(df, 9, 10, 4, 6, 2)
    train(datasets, num_epochs=50, learning_rate=0.001)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
