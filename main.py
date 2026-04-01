from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# configs
DATA_DIR = Path("data")
INPUT_SIZE = 310
OUTPUT_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset
class EEGTrainDataset(Dataset):
    def __init__(self, dir: Path = DATA_DIR):
        self.feats = np.vstack([np.load(f)["train_data"] for f in dir.glob("*.npz")])
        self.labels = np.concatenate(
            [np.load(f)["train_label"] for f in dir.glob("*.npz")]
        )

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


class EEGTestDataset(Dataset):
    def __init__(self, dir: Path = DATA_DIR):
        self.feats = np.vstack([np.load(f)["test_data"] for f in dir.glob("*.npz")])
        self.labels = np.vstack([np.load(f)["test_label"] for f in dir.glob("*.npz")])

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


# hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 3

mlp_baseline = nn.Sequential(
    nn.Linear(INPUT_SIZE, 128),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(128, OUTPUT_SIZE),
).to(DEVICE)


def train_one_epoch(
    model: nn.Module,
    train_data: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device = DEVICE,
):
    model.train()
    size, total_acc, total_loss = 0, 0, 0.0

    for X, y in tqdm(train_data, leave=False):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        total_acc += (preds == y).sum().item()
        size += X.size(0)
    return total_loss / size, total_acc / size


def evaluate(
    model: nn.Module,
    test_data: DataLoader,
    device: torch.device = DEVICE,
):
    model.eval()
    size, total_acc, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * X.size(0)

            preds = logits.argmax(dim=1)
            total_acc += (preds == y).sum().item()
            size += X.size(0)
    return total_loss / size, total_acc / size


if __name__ == "__main__":
    train_data = EEGTrainDataset()
    test_data = EEGTestDataset()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = mlp_baseline

    print("Start training...")
    print(f"Using device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(
            model,
            train_loader,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
        )
        v_loss, v_acc = evaluate(model, test_loader)
        print(
            f"Epoch {epoch}: train_loss={t_loss:.4f} train_acc={t_acc:.4f} | "
            f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}"
        )
