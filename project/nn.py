import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt
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
        for _, row in df.iterrows():
            scalars = [row["n"], row["k"], row["m"]]
            p_vector = np.array(row["P"], dtype=np.float32)
            x = np.concatenate([scalars, p_vector])
            self.inputs.append(x)
            self.targets.append(row["result"])

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



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

# Load data
def create_datasets(df, n_lower, n_upper, k_lower, k_upper, m_lower):
    datasets = {}
    for n in range(n_lower, n_upper + 1):
        for k in range(k_lower, k_upper + 1):
            for m in range(m_lower, n - k + 1):
                my_df = df.loc[(df["k"] == k) & (df["n"] == n) & (df["m"] == m)]
                # Dataset and DataLoader
                full_dataset = ResultsDataset(my_df)
                train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)

                train_subset = torch.utils.data.Subset(full_dataset, train_indices)
                val_subset = torch.utils.data.Subset(full_dataset, val_indices)

                train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=512, shuffle=False)
                if(not datasets.get(n, None)):
                    datasets[n] = {}
                if(not datasets[n].get(k, None)):
                    datasets[n][k] = {}
                datasets[n][k][m] = {"train_loader" : train_loader, "val_loader": val_loader, "full_dataset": full_dataset}
    return datasets

def train(datasets, num_epochs, learning_rate):
    for n in datasets:
        for k in datasets[n]:
            for m in datasets[n][k]:
                element = datasets[n][k][m]
                dataset = element["dataset"]
                train_loader = element["train_loader"]
                val_loader = element["val_loader"]
                input_size = len(dataset[0][0])
                net = Net(input_size)

                # Use all available GPUs
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs")
                    net = nn.DataParallel(net)
                net.to(device)

                # Training setup
                criterion = Log2Loss()
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                epochs = num_epochs
                train_losses = []
                val_losses = []

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
                    train_losses.append(avg_train_loss)

                    # Validation evaluation
                    net.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            preds = net(inputs)
                            loss = criterion(preds, targets)
                            val_loss += loss.item()

                    avg_val_loss = val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)

                    print(f"{n}, {m}, {k} - Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

                # Save model
                torch.save(net.state_dict(), f"./models/{n}-{k}-{m}_model.pt")

                # Plot train and val loss
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label="Training Loss")
                plt.plot(val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"./model/{n}-{k}-{m}_training_and_validation_loss.png")
                print("Saved training plot as 'training_and_validation_loss.png'")

                print("Saved training plot as 'training_loss_and_accuracy.png'")

def main():
    df = joblib.load("results_subset_1M.pkl")
    datasets = create_datasets(df, 9, 10, 4, 6, 2)
    train(datasets, 50, .001)


if __name__ == "__main__":
    main()