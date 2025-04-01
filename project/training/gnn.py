import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PyTorch Geometric imports
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Load data (assumes results_subset_1M.pkl is in the working directory)
print("loading df")
df = joblib.load("results_subset_1M.pkl")
max_k = int(df["k"].max())

class GraphResultsDataset(Dataset):
    def __init__(self, df):
        super(GraphResultsDataset, self).__init__()
        self.data_list = []
        for idx, row in df.iterrows():
            # Extract scalar parameters
            n = int(row["n"])
            k = int(row["k"])
            m = int(row["m"])
            # Process P: row["P"] should represent the systematic generator matrix's P part,
            # which is a flattened vector for a k x (n-k) matrix.
            P = np.array(row["P"], dtype=np.float32)
            desired_size = k * (n - k)
            if P.size != desired_size:
                if P.size < desired_size:
                    P = np.pad(P, (0, desired_size - P.size), mode="constant")
                else:
                    P = P[:desired_size]
            P = P.reshape((k, n - k))
            
            node_features = []
            # Nodes 0 to k-1: Identity columns (one-hot padded to length max_k)
            for i in range(k):
                vec = np.zeros(max_k, dtype=np.float32)
                vec[i] = 1.0
                node_features.append(vec)
            # Nodes k to n-1: Columns from P (each is originally length k, pad to max_k)
            for j in range(n - k):
                vec = P[:, j]
                if k < max_k:
                    pad_width = max_k - k
                    vec = np.pad(vec, (0, pad_width), mode="constant")
                node_features.append(vec)
            # Convert node features to a tensor of shape [n, max_k]
            x = torch.tensor(np.stack(node_features), dtype=torch.float)
            
            # Create a fully connected graph (excluding self-loops)
            src = []
            dst = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            
            # Global features: store [n, k, m] as a tensor with shape [1, 3]
            global_features = torch.tensor([n, k, m], dtype=torch.float).unsqueeze(0)
            
            # Target value (m-height)
            y = torch.tensor([row["result"]], dtype=torch.float)
            
            # Create a PyG Data object and attach global features
            data = Data(x=x, edge_index=edge_index, y=y)
            data.global_features = global_features  # shape: [1, 3]
            self.data_list.append(data)
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

 
full_dataset = None
if(not os.path.exists("./graph_dataset.pkl")):
    print("generating graph dataset")
    full_dataset = GraphResultsDataset(df)
    print("dumping graph_dataset in file")
    joblib.dump(full_dataset, "./graph_dataset.pkl")
else:
    print("loading graph dataset")
    full_dataset = joblib.load("./graph_dataset.pkl")
    print("finished loading graph dataset")

# Use the maximum k in the dataset to pad node features (so all node features have the same dimension)

# Custom Loss Function
class Log2Loss(nn.Module):
    def __init__(self):
        super(Log2Loss, self).__init__()
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1e-6)
        y_true = torch.clamp(y_true, min=1e-6)
        log_pred = torch.log2(y_pred)
        log_true = torch.log2(y_true)
        return torch.mean((log_pred - log_true) ** 2)

# Custom PyG Dataset
           
# Create the dataset and split into training and validation sets
indices = list(range(len(full_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = [full_dataset.get(i) for i in train_indices]
val_subset = [full_dataset.get(i) for i in val_indices]

# Use PyTorch Geometric's DataLoader
train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=512, shuffle=False)

# Define the GNN Model
class GNNModel(nn.Module):
    def __init__(self, node_feature_dim, global_feature_dim, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # MLP for combining pooled node features and global features
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels + global_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        global_feats = data.global_features
        x = torch.cat([x, global_feats], dim=1)
        x = self.mlp(x)
        return x

# Since node features are padded to length max_k, set node_feature_dim = max_k.
node_feature_dim = max_k         # e.g. 6 if max(k) == 6
global_feature_dim = 3           # for [n, k, m]
hidden_channels = 128

model = GNNModel(node_feature_dim, global_feature_dim, hidden_channels)

# Move the model to GPU if available (and enable DataParallel if using multiple GPUs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training setup
criterion = Log2Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50
losses = []
costs = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = criterion(preds, batch.y)
            val_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(batch.y.cpu())
    avg_val_loss = val_loss / len(val_loader)
    preds = torch.cat(all_preds).clamp(min=1.0)
    targets = torch.cat(all_targets)
    log_pred = torch.log2(preds)
    log_true = torch.log2(targets)
    sigma = ((log_pred - log_true) ** 2).mean().item()
    costs.append(sigma)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - σ (log cost): {sigma:.6f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pt")

# Plot training loss and log cost
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Training Loss")
plt.plot(costs, label="Log Cost σ")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Log Cost (σ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_and_accuracy.png")
print("Saved training plot as 'training_loss_and_accuracy.png'")

