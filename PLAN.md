# EEG Emotion Classification — Improvements Plan

## Context
Homework task: 5-class emotion classification on SEED-IV style EEG data (16 subjects, 62 channels, 5 DE frequency band features). Two evaluation protocols: subject-dependent (16 models) and cross-subject LOOCV (16 folds). Current code has a minimal MLP, broken logging, missing F1 metric, optimizer reset bug, and no cross-subject setup.

---

## Critical Files
- `main.py` — train loop, logging, metrics
- `models.py` — **new** — three CNN architectures
- `tests/test_models.py` — **new** — smoke tests
- `dataloader.py` — reused as-is
- `channel_62_pos.locs` — electrode positions for TopoCNN

---

## Step 1 — Fix Optimizer Bug (main.py:131)
Move `criterion` and `optimizer` instantiation **before** the `for epoch` loop. Currently they're recreated each epoch, discarding Adam's momentum buffers.

Also add `BATCH_SIZE = 32` to the hyperparameters block and pass it to `DataLoader`.

---

## Step 2 — Add F1 Metric to `evaluate()`
Add `from sklearn.metrics import f1_score as sklearn_f1` import. Accumulate `all_preds` / `all_labels` lists across batches, then call `sklearn_f1(..., average="macro", zero_division=0)` once after the loop. Return `(loss, acc, f1)` — a triple instead of a pair.

---

## Step 3 — Improve Logging (main.py `__main__`)
Replace `logging.basicConfig(filename=...)` with dual-handler setup:
```python
file_handler = logging.FileHandler(log_path)
console_handler = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
# attach fmt to both, add both to root logger
```
Log all hyperparameters (`lr`, `wd`, `label_smoothing`, `epochs`, `batch_size`) once before the subject loop.

---

## Step 4 — Three CNN Architectures (`models.py` — new file)

### 4a. BandSpatialCNN (TSception-inspired)
Input `(B, 62, 5)` → unsqueeze → `(B, 1, 62, 5)`

**Band stage** — 3 parallel Conv2d kernels along band dim, padded to same width, concat → BN:
- `Conv2d(1, F, (1,1))`
- `F.pad(x,(0,1))` then `Conv2d(1, F, (1,2))`
- `Conv2d(1, F, (1,3), padding=(0,1))`
- Concat → `(B, 3F, 62, 5)` → `BatchNorm2d(3F)` + `LeakyReLU`

**Spatial stage** — two branches, concat on height dim → BN:
- `spatial_full = Conv2d(3F, num_S, (62,1))` → `(B, num_S, 1, 5)`
- `spatial_hemi = Conv2d(3F, num_S, (31,1), stride=(31,1))` → `(B, num_S, 2, 5)`
- Concat → `(B, num_S, 3, 5)` → `BatchNorm2d(num_S)` + `LeakyReLU`

**Head**: `flatten → Linear(num_S*15, 5)` with defaults `F=16, num_S=32` (~20K params).

---

### 4b. TopoCNN (topology-preserving projection)
Precompute 9×9 grid indices from `.locs` at module load:
```python
x = radius * sin(radians(angle)); y = radius * cos(radians(angle))
grid_col = round((x - x_min)/(x_max - x_min) * 8)
grid_row = round((y - y_min)/(y_max - y_min) * 8)
flat_idx = grid_row * 9 + grid_col   # shape (62,) registered as buffer
```
Multiple channels may share a cell (6 pairs) — handled via `scatter_add_` (sum).

Input `(B, 62, 5)` → permute to `(B, 5, 62)` → `scatter_add_` into zeros `(B, 5, 81)` → view `(B, 5, 9, 9)`

**Conv stack**: `Conv2d(5→32, k=3, pad=1)+BN+ReLU` × 3 (32→64→128) → `AdaptiveAvgPool2d(1)` → flatten `(B, 128)` → `Linear(128, 5)`.

Note: `_compute_grid_indices()` reads `channel_62_pos.locs` via `Path(__file__).parent`.

---

### 4c. FactorizedCNN (EEGNet-inspired)
Input `(B, 62, 5)` — two-stage depthwise-separable 1D convolution over the band dimension:

- **Depthwise**: `Conv1d(62, 62, kernel=3, groups=62, padding=1, bias=False)` → `BN1d(62)` + `ELU` → `(B, 62, 5)`
- **Pointwise**: `Conv1d(62, 128, kernel=1, bias=False)` → `BN1d(128)` + `ELU` → `(B, 128, 5)`
- `AdaptiveAvgPool1d(1)` → `(B, 128, 1)` → flatten → `Dropout(0.5)` → `Linear(128, 5)`

Default `pointwise_filters=128`.

---

## Step 5 — Smoke Tests (`tests/test_models.py` — new file)
Run via `uv run pytest tests/test_models.py -v`. For each model (`TestBandSpatialCNN`, `TestTopoCNN`, `TestFactorizedCNN`):

```python
def test_output_shape(self):
    out = Model()(torch.randn(4, 62, 5))
    assert out.shape == (4, 5)

def test_backward(self):
    model = Model()
    loss = model(torch.randn(4, 62, 5)).sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
```

**Execute smoke tests after creating the files to verify correctness.**

---

## Step 6 — Updated Train Loop (main.py `__main__`)

### 6a. Subject-Dependent
```python
subj_accs, subj_f1s = [], []
for f in sorted(DATA_DIR.glob("*.npz"), key=lambda p: int(p.stem)):
    # build loaders, model, criterion (before epoch loop), optimizer (before epoch loop)
    for epoch in range(1, EPOCHS+1):
        t_loss, t_acc = train_one_epoch(...)
        v_loss, v_acc, v_f1 = evaluate(...)
        logger.info("Epoch %d: ... val_f1=%.4f", epoch, ..., v_f1)
    subj_accs.append(v_acc); subj_f1s.append(v_f1)
    logger.info("Subject %s: acc=%.4f f1=%.4f", f.stem, v_acc, v_f1)
logger.info("Subject-dep FINAL: acc=%.4f±%.4f f1=%.4f±%.4f", ...)
```

### 6b. Cross-Subject LOOCV (defined as function, NOT called from `__main__`)
```python
def run_loocv(all_files, device=DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """DO NOT CALL from __main__. Run explicitly when needed."""
    for i, test_file in enumerate(all_files):
        # Training set: ConcatDataset of both splits from all other 15 subjects
        train_ds = ConcatDataset([EEGDataset(f, s) for j,f in enumerate(all_files)
                                   if j!=i for s in ("train","test")])
        # Test set: subject i's full data (both splits)
        test_ds = ConcatDataset([EEGDataset(test_file, s) for s in ("train","test")])
        # train + evaluate, collect v_acc, v_f1
    logger.info("LOOCV FINAL: acc=%.4f±%.4f f1=%.4f±%.4f", ...)
```
Uses `from torch.utils.data import ConcatDataset` — no changes to `dataloader.py`.

---

## Verification
1. Run smoke tests: `uv run pytest tests/test_models.py -v` — all 6 tests pass
2. Check model param counts: `BandSpatialCNN ~20K`, `TopoCNN ~140K`, `FactorizedCNN ~9K`
3. Single-subject sanity run (2 epochs): verify F1 appears in logs and console
4. Verify optimizer is NOT recreated inside epoch loop
5. Verify `run_loocv` is defined but never called in `__main__`
