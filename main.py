import argparse
import logging
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score as sklearn_f1
from tqdm import tqdm

from dataloader import EEGDataset
from models import BandSpatialCNN, TopoCNN, FactorizedCNN, BandGraphCNN

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
OUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS = 2
BATCH_SIZE = 32

MODEL_CHOICES = ["mlp", "band", "topo", "factorized", "band_graph"]
MODE_CHOICES = ["per-subject", "loocv"]


def build_model(name: str) -> nn.Module:
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(62 * 5, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
    if name == "band":
        return BandSpatialCNN()
    if name == "topo":
        return TopoCNN()
    if name == "factorized":
        return FactorizedCNN()
    if name == "band_graph":
        return BandGraphCNN()
    raise ValueError(f"Unknown model: {name}")


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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_data: DataLoader,
    device: torch.device = DEVICE,
):
    model.eval()
    size, total_acc, total_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    for X, y in test_data:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * X.size(0)

        preds = logits.argmax(dim=1)
        total_acc += (preds == y).sum().item()
        size += X.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    f1 = sklearn_f1(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / size, total_acc / size, f1


@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
):
    model.eval()
    all_X, all_preds, all_labels = [], [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        all_X.append(X.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    X_np = torch.cat(all_X).numpy()        # (N, 62, 5)
    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    mean_de = np.stack([
        X_np[labels_np == c].mean(axis=0) if (labels_np == c).any()
        else np.zeros((62, 5), dtype=np.float32)
        for c in range(5)
    ])  # (5, 62, 5): class, channel, band
    return mean_de, preds_np, labels_np


def run_per_subject(
    all_files: list,
    model_name: str,
    device: torch.device = DEVICE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    subj_accs, subj_f1s = [], []

    for f in all_files:
        train_loader = DataLoader(
            EEGDataset(f, split="train"), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            EEGDataset(f, split="test"), batch_size=batch_size, shuffle=False
        )

        model = build_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        logger.info("Start training %s", f.name)
        v_acc, v_f1 = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_f1 = evaluate(model, test_loader, device)
            logger.info(
                "Epoch %d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch, t_loss, t_acc, v_loss, v_acc, v_f1,
            )

        mean_de, preds, labels = collect_outputs(model, test_loader)
        out_path = OUT_DIR / "per_subject" / f"{f.stem}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, mean_de=mean_de, preds=preds, labels=labels,
                 acc=v_acc, f1=v_f1)

        subj_accs.append(v_acc)
        subj_f1s.append(v_f1)
        logger.info("Subject %s: acc=%.4f f1=%.4f", f.stem, v_acc, v_f1)

    if subj_accs:
        logger.info(
            "Subject-dep FINAL: acc=%.4f±%.4f f1=%.4f±%.4f",
            sum(subj_accs) / len(subj_accs),
            statistics.stdev(subj_accs) if len(subj_accs) > 1 else 0.0,
            sum(subj_f1s) / len(subj_f1s),
            statistics.stdev(subj_f1s) if len(subj_f1s) > 1 else 0.0,
        )


def run_loocv(
    all_files: list,
    model_name: str,
    device: torch.device = DEVICE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    loocv_accs, loocv_f1s = [], []

    for i, test_file in enumerate(all_files):
        train_ds = ConcatDataset(
            [
                EEGDataset(f, s)
                for j, f in enumerate(all_files)
                if j != i
                for s in ("train", "test")
            ]
        )
        test_ds = ConcatDataset([EEGDataset(test_file, s) for s in ("train", "test")])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = build_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        v_acc, v_f1 = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_f1 = evaluate(model, test_loader, device)
            logger.info(
                "LOOCV fold %d epoch %d: train_loss=%.4f train_acc=%.4f "
                "val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                i, epoch, t_loss, t_acc, v_loss, v_acc, v_f1,
            )

        mean_de, preds, labels = collect_outputs(model, test_loader)
        out_path = OUT_DIR / "loocv" / f"fold_{test_file.stem}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, mean_de=mean_de, preds=preds, labels=labels,
                 acc=v_acc, f1=v_f1)

        loocv_accs.append(v_acc)
        loocv_f1s.append(v_f1)
        logger.info("LOOCV fold %d (%s): acc=%.4f f1=%.4f", i, test_file.stem, v_acc, v_f1)

    logger.info(
        "LOOCV FINAL: acc=%.4f±%.4f f1=%.4f±%.4f",
        sum(loocv_accs) / len(loocv_accs),
        statistics.stdev(loocv_accs),
        sum(loocv_f1s) / len(loocv_f1s),
        statistics.stdev(loocv_f1s),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODE_CHOICES, default="per-subject")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="mlp")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%d%m%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{args.model}_{args.mode}.log"

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger.info(
        "Hyperparameters: lr=%s wd=%s label_smoothing=%s epochs=%d batch_size=%d",
        LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING, EPOCHS, BATCH_SIZE,
    )
    logger.info("Using device: %s  mode: %s  model: %s", DEVICE, args.mode, args.model)

    all_files = sorted(DATA_DIR.glob("*.npz"), key=lambda p: int(p.stem))

    if args.mode == "per-subject":
        run_per_subject(all_files, args.model)
    else:
        run_loocv(all_files, args.model)
