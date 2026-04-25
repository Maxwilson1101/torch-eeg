from math import sin, cos, radians
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BandSpatialCNN  (TSception-inspired)
# Input: (B, 62, 5)
# ---------------------------------------------------------------------------

class BandSpatialCNN(nn.Module):
    def __init__(self, F: int = 16, num_S: int = 32, num_classes: int = 5):
        super().__init__()
        self.F = F

        # Band-temporal branches (operate on the 5-band dimension)
        self.band1 = nn.Conv2d(1, F, kernel_size=(1, 1))
        self.band2 = nn.Conv2d(1, F, kernel_size=(1, 2))   # used with F.pad(x,(0,1))
        self.band3 = nn.Conv2d(1, F, kernel_size=(1, 3), padding=(0, 1))

        self.band_bn = nn.BatchNorm2d(3 * F)

        # Spatial branches (operate on the 62-channel dimension)
        self.spatial_full = nn.Conv2d(3 * F, num_S, kernel_size=(62, 1))
        self.spatial_hemi = nn.Conv2d(3 * F, num_S, kernel_size=(31, 1), stride=(31, 1))

        self.spatial_bn = nn.BatchNorm2d(num_S)

        self.head = nn.Linear(num_S * 3 * 5, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 62, 5)
        x = x.unsqueeze(1)  # (B, 1, 62, 5)

        # Band stage
        b1 = self.band1(x)                                  # (B, F, 62, 5)
        b2 = self.band2(F.pad(x, (0, 1)))                  # (B, F, 62, 5)
        b3 = self.band3(x)                                  # (B, F, 62, 5)
        x = torch.cat([b1, b2, b3], dim=1)                  # (B, 3F, 62, 5)
        x = F.leaky_relu(self.band_bn(x))

        # Spatial stage
        s_full = self.spatial_full(x)                       # (B, num_S, 1, 5)
        s_hemi = self.spatial_hemi(x)                       # (B, num_S, 2, 5)
        x = torch.cat([s_full, s_hemi], dim=2)              # (B, num_S, 3, 5)
        x = F.leaky_relu(self.spatial_bn(x))

        x = x.flatten(1)                                    # (B, num_S*15)
        return self.head(x)


# ---------------------------------------------------------------------------
# TopoCNN  (topology-preserving 9×9 grid projection)
# Input: (B, 62, 5)
# ---------------------------------------------------------------------------

def _compute_grid_indices() -> torch.Tensor:
    """Read channel_62_pos.locs and map each electrode to a flat 9×9 grid index."""
    locs_path = Path(__file__).parent / "channel_62_pos.locs"
    xs, ys = [], []
    for line in locs_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        angle = float(parts[1])
        radius = float(parts[2])
        xs.append(radius * sin(radians(angle)))
        ys.append(radius * cos(radians(angle)))

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    flat_indices = []
    for xi, yi in zip(xs, ys):
        col = round((xi - x_min) / (x_max - x_min) * 8)
        row = round((yi - y_min) / (y_max - y_min) * 8)
        flat_indices.append(row * 9 + col)

    return torch.tensor(flat_indices, dtype=torch.long)  # (62,)


class TopoCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.register_buffer("grid_idx", _compute_grid_indices())  # (62,)

        self.conv_stack = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 62, 5)
        B = x.size(0)
        x = x.permute(0, 2, 1)  # (B, 5, 62)

        # Scatter channels onto 9×9 grid
        grid = torch.zeros(B, 5, 81, device=x.device, dtype=x.dtype)
        idx = self.grid_idx.view(1, 1, 62).expand(B, 5, 62)
        grid.scatter_add_(2, idx, x)                        # (B, 5, 81)
        grid = grid.view(B, 5, 9, 9)

        x = self.conv_stack(grid)                           # (B, 128, 9, 9)
        x = self.pool(x).flatten(1)                        # (B, 128)
        return self.head(x)


# ---------------------------------------------------------------------------
# FactorizedCNN  (EEGNet-inspired depthwise-separable)
# Input: (B, 62, 5)
# ---------------------------------------------------------------------------

class FactorizedCNN(nn.Module):
    def __init__(self, pointwise_filters: int = 128, num_classes: int = 5):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(62, 62, kernel_size=3, groups=62, padding=1, bias=False),
            nn.BatchNorm1d(62),
            nn.ELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(62, pointwise_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(pointwise_filters),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(pointwise_filters, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 62, 5)
        x = self.depthwise(x)   # (B, 62, 5)
        x = self.pointwise(x)   # (B, pointwise_filters, 5)
        x = self.pool(x)        # (B, pointwise_filters, 1)
        x = x.flatten(1)        # (B, pointwise_filters)
        return self.head(x)
