# EEG Emotion Classification — Experiment Report

**Model:** BandSpatialCNN  
**Dataset:** SEED-IV style, 16 subjects, 5-class emotion recognition  
**Protocols:** Subject-dependent & Cross-subject LOOCV

---

## 1. Task

Five-class EEG emotion recognition (Neutral / Sad / Fear / Happy / Disgust) on SEED-IV style data.  
Each sample is a `(62, 5)` feature matrix: 62 scalp electrodes × 5 differential entropy (DE) frequency bands (δ, θ, α, β, γ).  
Two evaluation protocols are used:

- **Subject-dependent** — one model trained and evaluated per subject using that subject's own train/test split.
- **Cross-subject LOOCV** — 16-fold leave-one-out: each fold trains on 15 subjects (train+test combined) and evaluates on the held-out subject.

Chance level for 5-class classification is **20%**.

---

## 2. Model: BandSpatialCNN

BandSpatialCNN is a TSception-inspired architecture that factorises spatial and spectral learning into two sequential stages.

### 2.1 Band-Temporal Stage

Three parallel `Conv2d` kernels operate across the 5-band dimension to capture multi-scale frequency combinations:

| Branch | Kernel | Captures |
|---|---|---|
| `band1` | `(1, 1)` | Single-band activations |
| `band2` | `(1, 2)` | Adjacent-band interactions |
| `band3` | `(1, 3)` | Broad-band patterns |

Outputs are concatenated along the filter axis → `(B, 3F, 62, 5)` with `F=16`, giving 48 band feature maps per electrode.  
Batch normalisation + Leaky ReLU follow.

### 2.2 Spatial Stage

Two `Conv2d` operations aggregate the band features across electrodes:

| Branch | Kernel / Stride | Covers |
|---|---|---|
| `spatial_full` | `(62, 1)` | All 62 channels at once (global) |
| `spatial_hemi` | `(31, 1)` stride `(31,1)` | Left / Right hemisphere separately |

Their outputs are concatenated → `(B, num_S, 3, 5)` with `num_S=32`.  
Batch normalisation + Leaky ReLU follow.

### 2.3 Classifier Head

The `(B, 32, 3, 5)` tensor is flattened to `(B, 480)` and passed through a single linear layer to 5 logits.

**Total parameters: ~146K**

---

## 3. Experimental Setup

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Label smoothing | 0.1 |
| Epochs | 2 |
| Batch size | 32 |
| Loss | CrossEntropyLoss |
| Device | CUDA |

---

## 4. Results

### 4.1 Subject-Dependent

Each subject's own train split is used for training; the held-out test split for evaluation.

| Subject | Accuracy | Macro-F1 |
|:---:|:---:|:---:|
| 1  | 0.8782 | 0.8728 |
| 2  | 0.6554 | 0.6425 |
| 3  | 0.5689 | 0.5152 |
| 4  | 0.8478 | 0.8059 |
| 5  | 0.6715 | 0.6467 |
| 6  | 0.7244 | 0.7029 |
| 7  | 0.7452 | 0.7550 |
| 8  | 0.8109 | 0.7664 |
| 9  | 0.9696 | 0.9642 |
| 10 | 0.5064 | 0.4753 |
| 11 | 0.7035 | 0.6489 |
| 12 | 0.6811 | 0.5339 |
| 13 | 0.8429 | 0.8295 |
| 14 | 0.4599 | 0.3997 |
| 15 | 0.6651 | 0.6858 |
| 16 | 0.6042 | 0.5924 |
| **Mean** | **0.7084 ± 0.1382** | **0.6773 ± 0.1527** |

Best subject: **S9** (acc=0.9696). Worst subject: **S14** (acc=0.4599).  
The large standard deviation (±0.14) indicates substantial inter-subject variability.

### 4.2 Cross-Subject LOOCV

Training on 15 subjects, testing on the held-out subject.

| Subject (held out) | Accuracy | Macro-F1 |
|:---:|:---:|:---:|
| 1  | 0.5979 | 0.5789 |
| 2  | 0.5491 | 0.5360 |
| 3  | 0.6308 | 0.6381 |
| 4  | 0.6286 | 0.6099 |
| 5  | 0.5683 | 0.5836 |
| 6  | 0.3171 | 0.3126 |
| 7  | 0.4915 | 0.4830 |
| 8  | 0.6923 | 0.6912 |
| 9  | 0.5304 | 0.5414 |
| 10 | 0.3406 | 0.3295 |
| 11 | 0.5793 | 0.5903 |
| 12 | 0.4054 | 0.3898 |
| 13 | 0.7005 | 0.6911 |
| 14 | 0.4833 | 0.4825 |
| 15 | 0.6166 | 0.6161 |
| 16 | 0.5930 | 0.5829 |
| **Mean** | **0.5453 ± 0.1132** | **0.5411 ± 0.1152** |

Best fold: **S13** (acc=0.7005). Worst fold: **S6** (acc=0.3171).

### 4.3 Protocol Comparison

| Protocol | Accuracy | Macro-F1 |
|---|---|---|
| Subject-dependent | 0.7084 ± 0.1382 | 0.6773 ± 0.1527 |
| Cross-subject LOOCV | 0.5453 ± 0.1132 | 0.5411 ± 0.1152 |
| Chance | 0.2000 | — |

Both protocols substantially exceed chance. The ~16 pp drop from subject-dependent to cross-subject reflects the well-known EEG domain-shift problem: DE band-power distributions differ meaningfully across individuals.

---

## 5. Analysis

### 5.1 Overfitting

In both protocols, training accuracy reaches ≥ 99.9% by epoch 2 while validation loss continues to rise — a clear sign of overfitting within just 2 epochs. The label-smoothing (0.1) and weight decay (1e-4) provide only partial regularisation at this training speed.

### 5.2 Subject Outliers

Three subjects underperform consistently across both protocols:

| Subject | Subj-dep acc | LOOCV acc |
|:---:|:---:|:---:|
| S6  | 0.7244 | **0.3171** |
| S10 | 0.5064 | **0.3406** |
| S12 | 0.6811 | **0.4054** |

In subject-dependent mode these subjects are acceptable (the model adapts to their own distribution). In LOOCV the model trained on the remaining 15 fails to generalise to them, suggesting their DE feature distributions lie outside the convex hull of the training population. Possible causes include different emotional expressivity, electrode contact variation, or systematic labelling differences for those sessions.

### 5.3 High-Variance Subjects

The per-subject standard deviation of 0.14 (subject-dependent) and 0.11 (LOOCV) is large relative to the mean. This reflects genuine between-subject heterogeneity in EEG-based emotion representation rather than model instability, as the training loss trajectory is consistent across all folds.

---

## 6. Visualisations

### 6.1 Training Curves (LOOCV)

Per-fold training and validation loss / accuracy curves:

![Training curves](outputs/figures/training_20262704111218_band_loocv.png)

Each panel shows train loss (blue solid), val loss (blue dashed), train acc (red solid), val acc (red dashed). The universal pattern of converging train loss with diverging val loss confirms overfitting across all folds.

### 6.2 DE Band-Power Topomaps

Mean DE band-power per emotion class across 62 electrodes, visualised with MNE. Each figure is a 5×5 grid (rows = emotion class, columns = δ θ α β γ).

**Subject-dependent examples:**

| Subject 9 (best, acc=0.9696) | Subject 14 (worst, acc=0.4599) |
|---|---|
| ![S9 topomap](outputs/figures/per_subject/9_topomap.png) | ![S14 topomap](outputs/figures/per_subject/14_topomap.png) |

**LOOCV examples:**

| Fold 13 (best, acc=0.7005) | Fold 10 (worst, acc=0.3406) |
|---|---|
| ![fold13 topomap](outputs/figures/loocv/fold_13_topomap.png) | ![fold10 topomap](outputs/figures/loocv/fold_10_topomap.png) |

Full topomap figures for all 16 subjects and all 16 LOOCV folds are in `outputs/figures/`.

---

## 7. Conclusion

BandSpatialCNN achieves **70.84%** accuracy under the subject-dependent protocol and **54.53%** under cross-subject LOOCV on 5-class EEG emotion recognition — well above the 20% chance baseline. The dual-stage architecture (multi-scale band kernels + hemisphere-aware spatial convolution) provides an effective inductive bias for DE feature data.

The primary limitation is **cross-subject generalisation**: the 16 pp gap between protocols highlights the need for domain adaptation or subject-invariant feature learning. Additionally, the rapid saturation of training accuracy (≈100% by epoch 2) suggests the model capacity exceeds what 2 epochs of label-smoothed training can regularise effectively; longer training with early stopping or dropout tuning would likely improve LOOCV performance.
