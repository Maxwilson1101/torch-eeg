from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Dataset


Split = Literal["train", "test"]


class EEGDataset(Dataset):
    """
    Build EEG dataset from npz file

    Warning: npz files are guaranteed in some format
    """

    KEYS = {
        "train": ("train_data", "train_label"),
        "test": ("test_data", "test_label"),
    }

    def __init__(self, npz_file: Path, split: Split = "train"):
        d = np.load(npz_file)
        data_key, label_key = self.KEYS[split]

        self.data = np.astype(d[data_key], np.float32)
        self.label = np.astype(d[label_key], np.int64)

    def __getitem__(self, idx: int) -> tuple[ArrayLike, ArrayLike]:
        return self.data[idx], self.label[idx]

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    from pprint import pprint

    ROOT = Path("data")
    ds = EEGDataset(ROOT / "1.npz")
    pprint(ds)

    pprint(ds[0])
