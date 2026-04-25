---
source: "[[4543_REST_Efficient_and_Accele.pdf]]"
code: https://github.com/arshiaafzal/REST
task:
  - seizure detection
  - seizure classification
method-name: REST
mechanism: graph-based residual state update (gating-free recurrent)
architecture:
  - linear recurrent state mapping
  - graph convolution (Kipf-stype, 2-filter)
  - binary random make (continuous dropout)
  - shared-weight multi-updatae
key-claims:
  - "@architectural 1.29ms inference latency"
  - "@efficiency 37KB memory footprint"
  - "@efficiency 9x faster than SOTA"
  - "@efficiency 14x smaller than smallest competitive baseline"
  - "@empirical <10K parameters matches SOTA accuracy"
datasets:
  - TUH-EEG v2.0.0
  - CHB-MIT
---


## Pipeline (with shapes)

```
raw EDF: per-channel time series @ original fs (varies, often 250 Hz) # data_utils.py
→ pick 19 standard 10-20 channels (FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ)
→ resample to to_freq=200 Hz # fixed
→ window into T 1-second segments per clip (T=12, 60, etc.)
→ (T, 19, 200)
→ FFT along time axis, take real magnitude of first 100 freq bins, log(|fft|+1e-30) # fixed, in train_rest()
→ (T, 19, 100)
→ transpose
→ (19, T, 100) # nodes-first for PyG → wrap as PyG Data(x, edge_index, edge_weight, y, batch) edge_index/edge_weight: from adj_mat.mat (precomputed, distance-based, # fixed lookup same as eeg-gnn-ssl repo)
→ 19 nodes per clip
→ batch into PyG Data: x is (B*19, T, 100), edge_index tiled per clip

REST.update(x_t, edge_index, edge_weight, s_t, fire_rate): per timestep t in [0..T-1]: x_t: (B_19, 100) if s_t is None: s_t = Linear(100→32, bias=False)(x_t) # learned, l1 else: s_t = l1(x_t) + Linear(32→32)(s_t) # learned, l2 ds = GraphConv(32→32)(s_t, ei, ew).relu() # learned, gc1 ds = GraphConv(32→32)(ds, ei, ew) # learned, gc2 mask = bernoulli(fire_rate=0.5) over ds # stochastic, training-only ds = ds * mask s_t = s_t + ds # residual update → s_T: (B_19, 32)

→ Linear(32→1) per node
→ (B*19, 1)
→ global_mean_pool over 19 nodes per clip
→ (B, 1)
→ sigmoid
→ MSE vs binary label # loss in training_step (logit-MSE, not BCE — BCE line is commented)
```

## Fixed vs. learned

- Fixed: 19-channel selection, resample to 200 Hz, FFT + first-100-bin log magnitude, per-clip z-score normalization (mean/std along time axis), adjacency matrix and edge weights (loaded from `adj_mat.mat`, distance-based, copied from `tsy935/eeg-gnn-ssl`), Bernoulli fire mask (parameterless dropout), graph topology (19 nodes, undirected, weighted)
- Learned: `l1` (Linear 100→32, no bias, input projection), `l2` (Linear 32→32, recurrent state-mixing), `gc1` (GraphConv 32→32), `gc2` (GraphConv 32→32), `fc` (Linear 32→1 readout)

## Swap points

- `update` residual recurrence (l1+l2 → GraphConv stack → masked add) → gated GRU/LSTM cell over graph features, Mamba-style SSM update, ConvLSTM on the graph node stream, GRES-style residual block
- GraphConv (Kipf-style spatial) → GAT (attention over neighbors), GraphSAGE, ChebConv (spectral), learned dynamic adjacency (per-clip or per-timestep edge weights)
- Fixed distance-based adjacency → fully learned adjacency from electrode coords, kNN over feature similarity, multi-scale (band-specific) adjacencies stacked
- Bernoulli fire mask (continuous dropout) → standard dropout, DropEdge (drop edges instead of features), no masking + weight decay
- Sequential per-timestep update (T forward passes) → batched temporal convolution over T, single Transformer pass over the (T, 19) sequence, Mamba SSM scan
- log-magnitude FFT (first 100 bins) → STFT spectrogram per timestep, learned 1D conv over raw signal, wavelet decomposition
- MSE on sigmoid output → BCE-with-logits (already commented in code), focal loss for class imbalance

## Hyperparameters

- input window per clip: T 1-second segments (codebase notebook uses T=60 in `Clip_TUH.ipynb`; default in `Train_Rest.py` is whatever `EEG.shape[1]` provides; paper experiments span T ∈ {12, 60, 120})
- channels: 19 (10-20 montage subset)
- resample target: 200 Hz
- FFT bins kept: first 100 (real magnitude, log + 1e-30 floor)
- node feature dim per timestep: 100
- hidden state dim: 32
- l1: Linear(100→32, bias=False); l2: Linear(32→32, bias=True)
- gc1, gc2: GraphConv(32→32) (PyG default Kipf-style with self-loops + add aggregation)
- readout: Linear(32→1)
- fire_rate (Bernoulli mask probability): 0.5
- conv_type: 'gconv' (string flag in `train_rest()`; suggests other variants exist but only this one is wired up here)
- multi (stochastic per-timestep update count): True ⇒ N ~ Uniform{1..9} per timestep; False ⇒ N=1 (paper compares both)
- optimizer: Adam, lr=5e-4
- LR schedule: MultiStepLR, milestones [1000, 2000], gamma 0.3 (epoch interval)
- batch_size (in training): 1024 (default in `Train_Rest.py:train_rest`; paper: not specified)
- max_epochs: 100 (default)
- precision: bf16-mixed
- strategy: DDP (Lightning, `ddp_notebook_find_unused_parameters_false`)
- devices: [1, 2, 3] (3 GPUs by default in script)
- train ratio: 0.75 (rest is held out)
- random seed: 123 (set in fold split, parsing, and `torch.manual_seed`)
- loss: MSE on sigmoid(out) (paper text suggests BCE; the BCE line is present-but-commented — flag this divergence)
- pooling: `global_mean_pool` over 19 nodes
- adjacency source: `adj_mat.mat` from `tsy935/eeg-gnn-ssl` (distance-based)

## Negative results

- BCE loss on logits: present-but-commented in `training_step` (`F.binary_cross_entropy_with_logits(...)`). The MSE-on-sigmoid version is what runs. Either it underperformed or was a leftover — the README doesn't say. Worth ablating.
- Real-part-only FFT vs. log-magnitude: code comment `# Just real part might also be useful` suggests they tried or considered `fft(...).real` instead of `log(|fft|+ε)` and chose the magnitude version
- The repo's experimental ablation grid (single vs. multi-step update, fire rate variations, T variations) is exposed via the `multi` and `fire_rate` flags but only one config is run in `train_rest`. Paper §4 lists these as ablations; results not pulled into the codebase
- (No commented-out alternative architectures, dead modules, or `# didn't work` notes beyond the above)

## PyTorch skeleton

```python
class REST(nn.Module):
    def __init__(self, in_dim=100, hid=32, fire_rate=0.5, T=60):
        super().__init__()
        self.T, self.fire_rate = T, fire_rate
        self.l1  = nn.Linear(in_dim, hid, bias=False)     # input projection
        self.l2  = nn.Linear(hid, hid)                    # recurrent mix
        self.gc1 = GraphConv(hid, hid)                    # PyG Kipf-style
        self.gc2 = GraphConv(hid, hid)
        self.fc  = nn.Linear(hid, 1)

    def update(self, x_t, ei, ew, s_t):
        s_t = self.l1(x_t) if s_t is None else self.l1(x_t) + self.l2(s_t)
        ds  = self.gc2(self.gc1(s_t, ei, ew).relu(), ei, ew)
        if self.training:
            ds = ds * (torch.rand_like(ds) <= self.fire_rate).float()
        return s_t + ds                                   # residual

    def forward(self, data):                              # PyG Data
        x = data.x.float()                                # (B*N, T, 100)
        x = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True) + 1e-10)
        s_t = None
        for t in range(self.T):                           # sequential
            s_t = self.update(x[:, t], data.edge_index, data.edge_weight.float(), s_t)
        out = self.fc(s_t)                                # (B*N, 1)
        return global_mean_pool(out, data.batch)          # (B, 1)
```

A few things worth flagging on the codebase ↔ paper alignment:

- The README's `update` snippet drops the `multi` loop and shows the canonical single-step recurrence. The full `forward` has an inner stochastic loop `N ~ U{1..9}` per timestep when `multi=True` — this is the "Neural Cellular Automata" inspiration the README brags about, and it doubles the meaningful axis of variation in the model.
- Loss: paper claims BCE-style binary classification, code runs MSE on sigmoid output. This is the kind of small thing that matters for reimplementation — pick one, but know which.
- The dropped-mask-during-eval behavior is implicit (the original `update` always applies the mask via `torch.rand(...).cuda() <= fire_rate` regardless of `self.training`). My PyTorch skeleton gates it on `self.training` because that's almost certainly what you'd want — but if you reimplement faithfully, leave it always-on (it's continuous dropout, not standard dropout).
- The T value drives memory hard: at T=60 with batch 1024, you have 60 sequential GraphConv passes per forward. The repo's reliance on bf16-mixed and DDP across 3 GPUs is not optional at that batch/T combination.

For idea generation, the highest-leverage swaps are probably (1) replace the Bernoulli fire mask with something with a known-good alternative (DropEdge is a clean drop-in), (2) replace the sequential T-step update with a Mamba scan, and (3) make the adjacency learned per-clip (the EvoBrain move).