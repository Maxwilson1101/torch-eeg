---
source: "[[1511.06448v3.pdf]]"
code: https://github.com/slavarepositories/EEG_RNN_Conv_Learning
task:
  - cognitive load classification
method-name:
mechanism: topology-preserving multi-spectral EEG image sequences + recurrent-convolutional model
architecture:
  - Azimuthal Equidistant Projection (3D→2D electrode layout)
  - Clough-Tocher interpolation (32×32 mesh)
  - 3-channel spectral image (theta / alpha / beta FFT bands)
  - VGG-style ConvNet (stacked 3×3, weight-shared across frames)
  - LSTM (1 layer, 128 cells)
key-claims:
  - "@architectural topology-preserving EEG-to-image transform preserves spatial / spectral / temporal structure jointly"
  - "@architectural per-frame ConvNet + LSTM captures temporal evolution of brain activity"
  - "@architectural weight-shared ConvNet across frames reduces parameter count"
  - "@empirical reduces error from 15.3% (prior SOTA) to 8.9% on cognitive load classification"
datasets:
---

## Pipeline (with shapes)

```
raw features: (N, 192*7=1344, +1 label) # FeatureMat_timeWin.mat, 7 windows × 192 feats/window
→ split into 7 windows of 192 features # n_windows=7, window_size=192
→ (N, 7, 192) → within each window, 192 = 3 bands × 64 electrodes (channel_size=64)
→ AEP project 64 electrode 3D coords to 2D # fixed, computed once at load time
→ Delaunay triangulation + barycentric weights
→ (vertices, weights) # fixed, precomputed → Clough-Tocher-style barycentric interp onto 32×32 grid per band
→ (N, 7, 3, 32, 32)
→ transpose to channels-last
→ (N, 7, 32, 32, 3)
→ per-channel StandardScaler (fit on train)
→ same shape, NaNs → 0 → TimeDistributed VGG-D ConvNet (shared across 7 frames): 4×Conv3-32 (ReLU, padding=same)
→ (N, 7, 32, 32, 32) MaxPool 2×2 stride 2
→ (N, 7, 16, 16, 32) 2×Conv3-64
→ (N, 7, 16, 16, 64) MaxPool
→ (N, 7, 8, 8, 64) Conv3-128
→ (N, 7, 8, 8, 128) MaxPool
→ (N, 7, 4, 4, 128) Flatten per frame
→ (N, 7, 2048)
→ LSTM(128, tanh, kernel_constraint=clip(±100))
→ (N, 128) # last timestep only
→ Dropout(0.5)
→ Dense(512, ReLU)
→ Dropout(0.5)
→ Dense(n_classes=4, softmax)
→ (N, 4)
```

## Fixed vs. learned

- Fixed: AEP (cartesian → spherical → polar) projection, Delaunay triangulation, barycentric interpolation weights (precomputed once — code comment notes this is the speedup vs. `scipy.griddata`'s per-call recompute), 32×32 grid resolution, 3-band split structure, NaN-to-zero imputation, per-channel StandardScaler statistics (fit on train fold)
- Learned: VGG-D ConvNet weights (shared across 7 frames via `TimeDistributed`), LSTM (128 units, tanh, with weight-clipping constraint), Dense-512 head, Dense-4 classifier

## Swap points

- AEP + Delaunay/barycentric interp → graph adjacency from electrode coords (skip image step entirely), learned 2D electrode embedding, RBF/distance-weighted scatter
- 3-band power features (theta/alpha/beta sum, computed upstream of this repo) → STFT spectrogram, wavelet, learned 1D conv over raw signal (EEGNet-style)
- VGG-D backbone → ResNet-18, ConvNeXt-tiny, ViT patch encoder over the 32×32×3 frame
- LSTM(128) aggregator → Transformer encoder, Mamba SSM, temporal max-pool (already implemented as `conv_net_max_pool` baseline), 1D temporal conv (already implemented as `conv_net_1d_conv`)
- TimeDistributed per-frame CNN → 3D conv over (T=7, H=32, W=32), ConvLSTM2D
- WeightClip(±100) hard constraint on LSTM kernel → spectral norm, weight decay, gradient clipping (the standard substitute)

## Hyperparameters

- input window: 7 frames × 192 features per frame = 1344 features per trial
- features per frame: 192 = 3 bands × 64 electrodes (channel_size=64, n_frequencies=3)
- image grid: 32×32, 3 channels (one per band)
- ConvNet (VGG-D): 4×Conv3-32, 2×Conv3-64, Conv3-128, three 2×2 maxpools (stride 2)
- LSTM: 1 layer, 128 units, tanh activation, kernel_constraint=WeightClip(c=100)
- FC head: Dense-512 (paper: 512 — matches), Dense-256 in conv_net_max_pool/1d_conv variants (paper: 512)
- dropout: 0.5 on every Dense input (twice in the head)
- optimizer: Adam, lr=1e-3, β1=0.9, β2=0.999
- batch size: 32 (paper: 20)
- epochs: 10 in the `__main__` script (paper: ~5 with early stopping; no early-stopping callback in this repo's main)
- num classes: 4 (inferred from `np.unique(labels)` at runtime; the dataset is the cognitive-load 4-way task)
- cross-validation: leave-one-subject-out, 13 folds (= 13 subjects)
- data: `Sample data/FeatureMat_timeWin.mat` (precomputed band-power features, not raw EEG), `Neuroscan_locs_orig.mat` (3D electrode coords), `trials_subNums.mat` (subject IDs)
- preprocessing scaler: per-channel StandardScaler, fit on train fold only, NaN-fill = 0
- LSTM weight clip threshold c=100 (not in paper)
- BatchNorm layers throughout `conv_net_lstm_1d_conv` (not in paper)
- `conv_net_lstm_1d_conv` head: Concatenate(LSTM-128, Conv1D-64), then Dense-256, BN, Dropout, Dense-classes (entire variant not in paper)

## Negative results

- Paper's max-pool aggregator and 1D temporal-conv aggregator are present in code (`conv_net_max_pool`, `conv_net_1d_conv`) — paper §2.2.3 reports both worse than LSTM
- Paper's deeper VGG configs (beyond D) — author stopped at D, matching paper's diminishing-returns finding (Table 1)
- The repo's `conv_net_lstm_1d_conv` (LSTM ⊕ 1D-conv concat with BatchNorm) is an extension not in the paper — no reported result in code or README, so unclear whether it helped
- Data augmentation by Gaussian noise — paper §2.3 reports it hurt; not implemented in this repo
- (No commented-out code blocks or TODOs in the pasted files indicating other dropped experiments)

## PyTorch skeleton

```python
class ConvNetLSTM(nn.Module):
    def __init__(self, n_classes=4, T=7, lstm_hidden=128, fc_hidden=512):
        super().__init__()
        self.cnn = VGG_D(in_ch=3)         # 4×C3-32 → MP → 2×C3-64 → MP → C3-128 → MP, ReLU+same-pad
        self.flatten_frame = nn.Flatten(start_dim=1)             # per-frame, after CNN
        self.lstm = nn.LSTM(2048, lstm_hidden, batch_first=True) # 4*4*128 = 2048
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(lstm_hidden, fc_hidden), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):                  # (B, T=7, 3, 32, 32)
        B, T = x.shape[:2]
        f = self.cnn(x.flatten(0, 1))      # (B*T, 128, 4, 4)
        f = self.flatten_frame(f).view(B, T, -1)   # (B, T, 2048)
        _, (h, _) = self.lstm(f)           # h: (1, B, 128) — last timestep
        return self.head(h.squeeze(0))     # (B, n_classes)
```

A few notes on what this revealed about the codebase versus the paper:

- The repo treats input not as raw EEG but as **precomputed band-power features** in `FeatureMat_timeWin.mat`. The FFT step from the paper happens upstream and isn't in this code at all — important if you want to reimplement end-to-end. The pipeline section reflects this faithfully (the FFT is implicit, not a step here).
- Two notable deviations from the paper, both flagged in Hyperparameters: batch size 32 vs. paper's 20, and 10 fixed epochs with no early stopping (paper used early stopping at ~5).
- The repo adds two things the paper didn't have: `WeightClip(±100)` on the LSTM kernel and `conv_net_lstm_1d_conv` (a fused LSTM+Conv1D variant with BatchNorm). Both are in `Swap points` or `Hyperparameters` rather than misrepresented as paper-faithful.
- The author kept all four single-frame VGG configs (A/B/C/D) and all three multi-frame aggregators (max-pool / temporal conv / LSTM) as separate factory methods — the repo *is* essentially a re-running of the paper's ablation grid, which is why `Negative results` could pull from "implemented but worse" rather than "tried and discarded."

One observation about how the skill performed: the SKILL.md instruction "focus on the paper's headline method" worked correctly here (picked `conv_net_lstm`), but the user gave me a file containing 8 model variants, which is a slightly unusual case. I suspect for repos that publish ablations as code, the `Negative results` section ends up doing more work than `Swap points`. That's probably fine — both sections feed idea generation differently — but worth knowing.
