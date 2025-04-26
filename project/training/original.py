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
from m_height import calculate_m_height 
def avg(sigmas):
    total = 0
    for val in sigmas:
        total += val
    return total / len(sigmas)
class Log2Loss(nn.Module):
    def __init__(self):
        super(Log2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=.00001)
        y_true = torch.clamp(y_true, min=.00001)
        log_pred = torch.log2(y_pred)
        log_true = torch.log2(y_true)
        return torch.mean((log_pred - log_true) ** 2)

class ResultsDataset(Dataset):
    def __init__(self, df):
        self.inputs = []
        self.targets = []
        for _, row in df.iterrows():
            scalars = [row["n"], row["k"], row["m"]]
            p_vector = np.array(row["P"], dtype=np.float32)
            x_vector = np.concatenate(row["random_x"])
            x_vector = np.array(x_vector, dtype = np.float32)
            height_vector = np.array(row["height"], dtype=np.float32)
            x = np.concatenate([scalars, p_vector, height_vector])
            self.inputs.append(x)
            self.targets.append(row["result"])

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



class Net(nn.Module):
    def __init__(self, input_size, width=512, depth=5, drop_p=0.1):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = input_size if i == 0 else width
            layers += [
                nn.Linear(in_dim, width),
                nn.SELU(),
                nn.AlphaDropout(drop_p)
            ]
        layers.append(nn.Linear(width, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#eturn datasets
def create_datasets(df, n_lower, n_upper, k_lower, k_upper, m_lower):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # make sure P is float32
    df["P"] = df["P"].apply(lambda v: np.asarray(v, dtype=np.float32))

    # build a flat list of all (n,k,m) combos
    combos = [
        (n, k, m)
        for n in range(n_lower, n_upper + 1)
        for k in range(k_lower, k_upper + 1)
        for m in range(m_lower, n - k + 1)
    ]

    # split the combos evenly
    my_combos = combos[rank::world_size]

    # each rank only loops over its share
    my_datasets = {}
    for n, k, m in my_combos:
        print(f"[rank {rank}] creating dataset for (n={n}, k={k}, m={m})")
        subdf = df.loc[(df["k"] == k) & (df["n"] == n) & (df["m"] == m)].copy()

        random_x_list, m_heights_list = [], []
        for _, row in subdf.iterrows():
            P_mat = row["P"].reshape(k, n - k)
            xs, hs = [], []
            for _ in range(10):
                x_rand = np.random.uniform(-10, 10, size=(k,)).astype(np.float32)
                xs.append(x_rand)
                h = calculate_m_height(x_rand, m, P_mat).astype(np.float32)
                hs.append(h)
            random_x_list.append(xs)
            m_heights_list.append(hs)

        subdf["random_x"] = random_x_list
        subdf["height"]   = m_heights_list

        full_dataset = ResultsDataset(subdf)
        train_idx, val_idx = train_test_split(
            list(range(len(full_dataset))),
            test_size=0.2,
            random_state=42
        )
        train_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, train_idx),
            batch_size=512, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, val_idx),
            batch_size=512, num_workers=8, pin_memory=True
        )

        my_datasets.setdefault(n, {}).setdefault(k, {})[m] = {
            "train_loader": train_loader,
            "val_loader":   val_loader,
            "dataset":      full_dataset
        }

        print(f"[rank {rank}] done (n={n}, k={k}, m={m})")

    # now gather everyone's partial dict
    all_dicts = [None for _ in range(world_size)]
    dist.all_gather_object(all_dicts, my_datasets)

    # merge them into one `datasets`
    datasets = {}
    for d in all_dicts:
        for n, kd in d.items():
            datasets.setdefault(n, {})
            for k, md in kd.items():
                datasets[n].setdefault(k, {})
                for m, v in md.items():
                    datasets[n][k][m] = v

    return datasets

def train(datasets, num_epochs, learning_rate):
    world_size = dist.get_world_size()
    sigmas = []
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    idx =0
    for n in datasets:
        for k in datasets[n]:
            for m in datasets[n][k]:
                if(idx % world_size == local_rank):
                    element = datasets[n][k][m]
                    dataset = element["dataset"]
                    train_loader = element["train_loader"]
                    val_loader = element["val_loader"]
                    input_size = len(dataset[0][0])
                    net = Net(input_size)
                    min_val_loss = 10000000
                    best_model_state = None
                    net.to(device)
                    criterion = Log2Loss()
                    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                    epochs = num_epochs
                    train_losses = []
                    val_losses = []
                    sigma = 10000000
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
                        if(avg_val_loss < min_val_loss):
                            min_val_loss = avg_val_loss
                            best_model_state = copy.deepcopy(net.state_dict())
                        preds = torch.cat(all_preds).clamp(min=1.0)
                        targets = torch.cat(all_targets)

                        log_pred = torch.log2(preds)
                        log_true = torch.log2(targets)
                        sigma = min(((log_pred - log_true) ** 2).mean().item(), sigma)
                        print(f"{n}, {m}, {k} - Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}, Val log cost: ", sigma)

                    print("final best validation sigma: ", sigma)
                    sigmas.append(sigma)
                    torch.save(best_model_state, f"../models/{n}-{k}-{m}_model.pt")

                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label="Training Loss")
                    plt.plot(val_losses, label="Validation Loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Training and Validation Loss")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"../plots/{n}-{k}-{m}_training_and_validation_loss.png")
                    print("Saved training plot as 'training_and_validation_loss.png'")
                    print("Saved training plot as 'training_loss_and_accuracy.png'")
                idx += 1
    
    local_sum = torch.tensor([sum(sigmas)], device=device, dtype=torch.float32)
    local_count = torch.tensor([len(sigmas)], device=device, dtype=torch.float32)

    global_sum = local_sum.clone()
    global_count = local_count.clone()
    dist.barrier()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

    overall_average = (global_sum / global_count).item()

    if local_rank == 0:
        print("Overall average sigma:", overall_average)


def main():
    print("got to main")
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=60000))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print("finished initializing process group")
    df = joblib.load("results_dataframe.pkl")
    print("finished loading dataset")
    datasets = create_datasets(df, 9, 10, 4, 6, 2)
    train(datasets, 25, .0005)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
