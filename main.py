import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
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

mlp_baseline = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(62 * 5, 128),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(128, 5),
)


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


@torch.no_grad
def evaluate(
    model: nn.Module,
    test_data: DataLoader,
    device: torch.device = DEVICE,
):
    model.eval()
    size, total_acc, total_loss = 0, 0, 0.0

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
    logging.basicConfig(
        filename=LOG_DIR / f"{datetime.now().strftime('%Y%d%m%H%M%S')}.log",
        level=logging.DEBUG,
    )

    logger.debug(f"Using device: {DEVICE}")

    for f in DATA_DIR.glob("*.npz"):
        train_data = EEGDataset(f, split="train")
        test_data = EEGDataset(f, split="test")
        train_loader = DataLoader(train_data, shuffle=True)
        test_loader = DataLoader(test_data, shuffle=False)

        model = EEGMLP(12, 3).to(DEVICE)
        # model = mlp_baseline.to(DEVICE)
        for name, p in model.named_parameters():
            print(name, p.shape)

        logger.info(f"Start training {f.name}")

        for epoch in range(1, EPOCHS + 1):
            t_loss, t_acc = train_one_epoch(
                model,
                train_loader,
                criterion=nn.CrossEntropyLoss(
                    label_smoothing=LABEL_SMOOTHING,
                ),
                optimizer=optim.Adam(
                    model.parameters(),
                    lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                ),
            )
            v_loss, v_acc = evaluate(model, test_loader)
            logger.info(
                f"Epoch {epoch}: train_loss={t_loss:.4f} train_acc={t_acc:.4f} "
                f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}"
            )
