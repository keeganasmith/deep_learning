import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt

# Load data
df = joblib.load("results_subset_1M.pkl")

class Log2Loss(nn.Module):
    def __init__(self):
        super(Log2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1e-6)
        y_true = torch.clamp(y_true, min=1e-6)
        log_pred = torch.log2(y_pred)
        log_true = torch.log2(y_true)
        return torch.mean((log_pred - log_true) ** 2)

# Custom Dataset
class ResultsDataset(Dataset):
    def __init__(self, df):
        self.inputs = []
        self.targets = []

        max_p_len = max(len(p) for p in df["P"])

        for _, row in df.iterrows():
            scalars = [row["n"], row["k"], row["m"]]
            p_vector = np.array(row["P"], dtype=np.float32)
            padded_p = np.pad(p_vector, (0, max_p_len - len(p_vector)), mode="constant")
            x = np.concatenate([scalars, padded_p])
            self.inputs.append(x)
            self.targets.append(row["result"])

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Dataset and DataLoader
full_dataset = ResultsDataset(df)
train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)

train_subset = torch.utils.data.Subset(full_dataset, train_indices)
val_subset = torch.utils.data.Subset(full_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=512, shuffle=False)
# Define model

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

input_size = len(full_dataset[0][0])
net = Net(input_size)

# Use all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    net = nn.DataParallel(net)
net.to(device)

# Training setup
criterion = Log2Loss()
optimizer = optim.Adam(net.parameters(), lr=.001)
epochs = 50
losses = []
costs = []

# Training loop
# Training loop
for epoch in range(epochs):
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
    losses.append(avg_train_loss)

    # Validation evaluation
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
    preds = torch.cat(all_preds).clamp(min=1.0)
    targets = torch.cat(all_targets)

    log_pred = torch.log2(preds)
    log_true = torch.log2(targets)
    sigma = ((log_pred - log_true) ** 2).mean().item()
    costs.append(sigma)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - σ (log cost): {sigma:.6f}")

# Save model
torch.save(net.state_dict(), "trained_model.pt")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Training Loss")
plt.plot(costs, label="Log Cost σ")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Accuracy (σ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_and_accuracy.png")
print("Saved training plot as 'training_loss_and_accuracy.png'")

