import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score as sklearn_f1
from tqdm import tqdm

from dataloader import EEGDataset

# configs
DATA_DIR = Path("data")
LOG_DIR = Path("logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

# hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS = 2
BATCH_SIZE = 32


class EEGMLP(nn.Module):
    CHAN_SIZE = 62
    BAND_SIZE = 5

    def __init__(self, w_chan: int, w_band: int):
        super().__init__()
        self.proj_band = nn.Parameter(torch.empty(self.BAND_SIZE, w_band))
        self.proj_chan = nn.Parameter(torch.empty(self.CHAN_SIZE, w_chan))
        nn.init.xavier_normal_(self.proj_band)
        nn.init.xavier_normal_(self.proj_chan)
        self.head = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(w_band * w_chan, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.proj_band  # (batch, 62, w_band)
        x = x.transpose(1, 2) @ self.proj_chan  # (batch, w_band, w_chan)
        return self.head(x)


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


def run_loocv(all_files, device=DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Leave-one-out cross-validation across subjects.

    DO NOT CALL from __main__. Run explicitly when needed.
    """
    loocv_accs, loocv_f1s = [], []
    for i, test_file in enumerate(all_files):
        train_ds = ConcatDataset(
            [EEGDataset(f, s) for j, f in enumerate(all_files) if j != i for s in ("train", "test")]
        )
        test_ds = ConcatDataset([EEGDataset(test_file, s) for s in ("train", "test")])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = EEGMLP(12, 3).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_f1 = evaluate(model, test_loader, device)
            logger.info(
                "LOOCV fold %d epoch %d: train_loss=%.4f train_acc=%.4f "
                "val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                i, epoch, t_loss, t_acc, v_loss, v_acc, v_f1,
            )

        loocv_accs.append(v_acc)
        loocv_f1s.append(v_f1)
        logger.info("LOOCV fold %d (%s): acc=%.4f f1=%.4f", i, test_file.stem, v_acc, v_f1)

    import statistics
    logger.info(
        "LOOCV FINAL: acc=%.4f±%.4f f1=%.4f±%.4f",
        sum(loocv_accs) / len(loocv_accs),
        statistics.stdev(loocv_accs),
        sum(loocv_f1s) / len(loocv_f1s),
        statistics.stdev(loocv_f1s),
    )


if __name__ == "__main__":
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now().strftime('%Y%d%m%H%M%S')}.log"

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
    logger.info("Using device: %s", DEVICE)

    all_files = sorted(DATA_DIR.glob("*.npz"), key=lambda p: int(p.stem))
    subj_accs, subj_f1s = [], []

    for f in all_files:
        train_data = EEGDataset(f, split="train")
        test_data = EEGDataset(f, split="test")
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        model = EEGMLP(12, 3).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        logger.info("Start training %s", f.name)

        for epoch in range(1, EPOCHS + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss, v_acc, v_f1 = evaluate(model, test_loader)
            logger.info(
                "Epoch %d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch, t_loss, t_acc, v_loss, v_acc, v_f1,
            )

        subj_accs.append(v_acc)
        subj_f1s.append(v_f1)
        logger.info("Subject %s: acc=%.4f f1=%.4f", f.stem, v_acc, v_f1)

    if subj_accs:
        import statistics
        logger.info(
            "Subject-dep FINAL: acc=%.4f±%.4f f1=%.4f±%.4f",
            sum(subj_accs) / len(subj_accs),
            statistics.stdev(subj_accs) if len(subj_accs) > 1 else 0.0,
            sum(subj_f1s) / len(subj_f1s),
            statistics.stdev(subj_f1s) if len(subj_f1s) > 1 else 0.0,
        )
