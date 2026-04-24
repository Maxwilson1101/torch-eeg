from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGTrainDataset(Dataset):
    def __init__(self, file: Path):
        d = np.load(file)

        # normalize data
        x = d["train_data"]
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x = (x - mean) / (std + 1e-6)

        self.feats = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(d["train_label"]).long()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


class EEGTestDataset(Dataset):
    def __init__(self, file: Path):
        d = np.load(file)

        # normalize data
        x = d["test_data"]
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x = (x - mean) / (std + 1e-6)

        self.feats = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(d["test_label"]).long()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]
