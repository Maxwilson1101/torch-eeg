# EEG Emotion Classification

5-class emotion classification on SEED-IV style EEG data using CNNs.  
Two evaluation protocols: **subject-dependent** (16 independent models) and **cross-subject LOOCV** (leave-one-out across 16 subjects).

---

## Data

### Format

Each subject's data is stored as a `.npz` file under `data/` named `{1..16}.npz`.

| Key | Shape | dtype | Description |
|---|---|---|---|
| `train_data` | `(N_train, 310)` | float32 | Flattened DE features, reshaped to `(N, 62, 5)` |
| `train_label` | `(N_train,)` | int64 | Emotion class 0–4 |
| `test_data` | `(N_test, 310)` | float32 | Same format |
| `test_label` | `(N_test,)` | int64 | Emotion class 0–4 |

After loading, `EEGDataset` reshapes each sample to `(62, 5)`:
- **62** — EEG electrode channels (62-lead cap, positions in `channel_62_pos.locs`)
- **5** — Differential entropy (DE) features over 5 frequency bands: δ (1–4 Hz), θ (4–8 Hz), α (8–13 Hz), β (13–30 Hz), γ (30–50 Hz)

### Labels

| Class | Emotion |
|---|---|
| 0 | Neutral |
| 1 | Sad |
| 2 | Fear |
| 3 | Happy |
| 4 | Disgust |

### Temporal structure

Within each `.npz` file, samples sharing the same label appear consecutively — each contiguous block is one **trial** (a continuous EEG segment recorded during a single emotional stimulus). Trials are assumed independent of each other.

### Electrode layout

- Channel ordering: `Channel Order.xlsx`
- Electrode positions (angle, radius in polar coords): `channel_62_pos.locs`
- Format: `index  angle  radius  name` (one electrode per line)

The `TopoCNN` model uses `channel_62_pos.locs` to project channels onto a 9×9 scalp topology grid.

---

## Models

Three CNN architectures are implemented in `models.py`:

| Model | Params | Description |
|---|---|---|
| `BandSpatialCNN` | ~146K | TSception-inspired; parallel band kernels + spatial branches |
| `TopoCNN` | ~95K | Projects electrodes onto 9×9 topology grid, then 2D CNN |
| `FactorizedCNN` | ~9K | EEGNet-inspired depthwise-separable 1D conv |

The `EEGMLP` baseline (bilinear projection head) is also in `main.py`.

---

## Training

### Setup

```bash
uv sync
```

### Subject-dependent (16 models)

Trains one model per subject, averages accuracy and macro-F1 across all 16:

```bash
uv run python main.py
```

Logs are written to `logs/` and also printed to the console.  
To swap the model, edit the `model = EEGMLP(12, 3).to(DEVICE)` line in `main.py` (replace with `BandSpatialCNN()`, `TopoCNN()`, or `FactorizedCNN()`).

### Cross-subject LOOCV

The `run_loocv()` function is defined in `main.py` but **not called automatically**. To run it:

```python
# in a script or notebook
from pathlib import Path
from main import run_loocv, DATA_DIR

all_files = sorted(DATA_DIR.glob("*.npz"), key=lambda p: int(p.stem))
run_loocv(all_files)
```

Each fold trains on all 15 other subjects (train + test splits concatenated) and evaluates on the held-out subject.

### Key hyperparameters

| Name | Default | Location |
|---|---|---|
| `LEARNING_RATE` | `1e-4` | `main.py` |
| `WEIGHT_DECAY` | `1e-4` | `main.py` |
| `LABEL_SMOOTHING` | `0.1` | `main.py` |
| `EPOCHS` | `2` | `main.py` |
| `BATCH_SIZE` | `32` | `main.py` |

---

## Tests

Smoke tests (output shape + backward pass) for all three CNN models:

```bash
uv run pytest tests/test_models.py -v
```

---

## Notes

- Electrode scalp maps can be visualized with [MNE-Python](https://mne.tools) or EEGLAB using `channel_62_pos.locs`.
- t-SNE visualization of subject data distributions is recommended to observe cross-subject variability.
- Submission deadline: **2026-04-27 23:59** via Canvas — pack as `姓名_学号_第二次作业.zip`.
