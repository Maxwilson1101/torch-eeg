---
source: "[[2104.02935v4.pdf]]"
code: https://github.com/deepBrains/TSception
task:
  - emotion recognition
method-name: TSception
mechanism: multi-scale temporal kernels + asymmetric hemisphere/global spatial kernels
architecture:
  - dynamic temporal layer (multi-scale 1D conv, lengths = {0.5, 0.25, 0.125} × fs)
  - asymmetric spatial layer (hemisphere kernel + global kernel)
  - high-level fusion layer (1D conv (3,1) over spatial dim)
  - LeakyReLU + average pooling + batch norm
  - global average pooling
  - fully connected classifier (softmax)
key-claims:
  - "@architectural multi-scale temporal kernels tied to sampling rate capture dynamic time-frequency EEG representations"
  - "@architectural weight-shared hemisphere kernel exploits left/right emotional asymmetry"
  - "@architectural high-level fusion of hemisphere + global representations makes network compact for online use"
  - "@empirical outperforms SVM, KNN, FBFgMDM, FBTSC, DeepConvNet, ShallowConvNet, EEGNet on accuracy and F1 in most settings"
  - "@empirical saliency maps localize learned features to frontal, temporal, and parietal regions (consistent with emotion neuroscience)"
  - "@empirical evaluated under generalized cross-validation to avoid data leakage"
datasets:
  - DEAP
  - MAHNOB-HCI
---

## Pipeline (with shapes)

```

raw EEG (per subject, multi-trial): (T_trials, C, S_total) # C channels, S_total time points
→ expand_dims to add freq channel
→ (T_trials, 1, C, S_total)
→ relabel {1.0, 2.0} → {0.0, 1.0} # binary, in PreProcess.py → split into overlapping windows: segment_length=4s @ 256Hz, overlap=0.975 step = 1024 * (1-0.975) = 25 samples; window = 1024 samples
→ (Subj, T_trials, N_seg, 1, C, 1024)
→ flatten over Subj-fold + within-trial reshuffle (split() in TrainModel.py)
→ DataLoader yields
→ (B, 1, C, 1024)

TSception forward(x): # x: (B, 1, C, T=1024) Tception1: Conv2d(1, num_T=9, k=(1, 0.5*256=128), s=1, p=0)
→ (B, 9, C, 1024-128+1=897) LeakyReLU
→ AvgPool(k=(1,8), s=(1,8))
→ (B, 9, C, 112) Tception2: Conv2d(1, 9, k=(1, 64))
→ AvgPool(1,8)
→ (B, 9, C, ~120) Tception3: Conv2d(1, 9, k=(1, 32))
→ AvgPool(1,8)
→ (B, 9, C, ~123) cat along dim=-1 (time)
→ (B, 9, C, 355) BatchNorm2d(9)
→ same shape

Sception1: Conv2d(9, num_S=6, k=(C,1), s=1) # collapses all channels
→ (B, 6, 1, 355) LeakyReLU
→ AvgPool(k=(1,2), s=(1,2))
→ (B, 6, 1, 177) Sception2: Conv2d(9, 6, k=(C/2,1), s=(C/2,1)) # 2 hemisphere groups
→ (B, 6, 2, 355) LeakyReLU
→ AvgPool(k=(1,2), s=(1,2))
→ (B, 6, 2, 177) cat along dim=2 (spatial)
→ (B, 6, 3, 177) BatchNorm2d(6) → same shape

flatten
→ (B, 6_3_177=3186) FC: Linear(3186, 128)
→ ReLU
→ Dropout(0.3)
→ Linear(128, num_classes=2)
→ (B, 2)
→ CrossEntropy + L1 reg (Lambda=1e-6 on all weights)
```

(Pool factors `int(self.pool*0.25)` = `int(8*0.25)` = 2, so spatial pool is `(1,2)` not `(1,8)`. The temporal pool stays at `(1,8)`. The exact post-Tception time dim depends on which kernel branch — they're concatenated despite slightly different output lengths because each path pools by 8.)

## Fixed vs. learned

- Fixed: 4-second window length, 0.975 overlap, 256 Hz sampling rate, channel layout (C-channel input split into top/bottom half by Sception2's stride trick, an architectural prior — no learning), 1.0/2.0 → 0/1 relabeling, expand_dims, train/val 80/20 split inside training trials
- Learned: Tception1/2/3 (3 parallel temporal Conv2d kernels, each Conv2d(1→9)), Sception1 (Conv2d(9→6) full-channel kernel), Sception2 (Conv2d(9→6) half-channel kernel with stride), 2× BatchNorm2d (T-stream after temporal cat, S-stream after spatial cat), FC head (3186→128→num_classes)

## Swap points

- Multi-scale temporal kernels (3 fixed inception-window ratios: 0.5/0.25/0.125) → learned 1D conv with dilation, dilated TCN, Mamba SSM, multi-head temporal self-attention, learnable scale gate over a fixed kernel bank
- Hard hemisphere prior in Sception2 (k=(C/2,1), s=(C/2,1)) → learned spatial attention, GCN over electrode graph, learned spatial mask, region-wise mixture of experts
- Conv2d-as-1D-conv (k=(1, L)) → native nn.Conv1d, depthwise-separable Conv2d (EEGNet-style)
- AvgPool over time → max-pool, attention-pooling, learned strided conv
- Concatenation of T-streams along time axis → channel-axis concat with broadcasted positional encoding, sum-fusion, gated fusion (a la Mixture-of-Depths)
- Cross-entropy + L1 reg → BCE-with-logits + dropout/weight-decay (L1 with Lambda=1e-6 is mild and may not be doing much), focal loss for imbalance
- Adam lr=1e-3 → AdamW with cosine schedule, SGD+momentum (the loss is small and noisy with batch=128)

## Hyperparameters

- input shape: (1, C, 1024) — C is dataset-dependent, time = 4s @ 256Hz
- sampling_rate: 256 Hz
- segment length: 4s (1024 samples); overlap: 0.975 (step = 25 samples ≈ 0.1s)
- num_T (temporal kernels per branch): 9 (the README/code suggest T=S=15, hidden=32 for cross-dataset transfer; defaults here are different)
- num_S (spatial kernels per branch): 6
- inception_window ratios (temporal): [0.5, 0.25, 0.125] → kernel sizes [128, 64, 32] @ 256Hz
- pool factor `self.pool`: 8 (temporal AvgPool); spatial pool = int(8*0.25) = 2
- hidden_node (FC): 128
- num_classes: 2
- dropout: 0.3 (FC only, after first Linear)
- Lambda (L1 weight on all params): 1e-6
- optimizer: Adam, lr=1e-3 (paper: same)
- epochs: 200 (paper: same)
- batch_size: 128
- early stopping patience: 4 epochs on val acc
- random_seed: 42 (model init); split shuffle uses np.random.seed(0)
- weight init: PyTorch default (Kaiming-uniform for Conv2d / Linear, no explicit init)
- bias: True everywhere (Conv2d default, Linear default)
- activation: LeakyReLU after each Conv2d, ReLU in the FC head (mixed — paper: not specified)
- BatchNorm: 2× (one after T-cat, one after S-cat); affine, default momentum=0.1
- training-mode-only: dropout, BN running stats, L1 reg (always on)
- cross-validation: leave-one-session-out (sessions = trials grouped in pairs); 80/20 train/val split per fold

## Negative results

- 4 EEG channel input is the implicit minimum: `__main__` test in Models.py uses `(4, 1024)`. Sception2 stride `int(input_size[-2]*0.5)` = 2 with 4 channels gives 2 spatial groups. With C=2 the spatial split degenerates (paper Table on TSception variants). Not flagged as a failure but is a structural floor.
- L1 regularization with Lambda=1e-6 on *all* parameters: the value is so small it's almost certainly negligible. Either tuned away from a higher value during dev (and no longer load-bearing) or vestigial — the paper claims it matters but the magnitude says otherwise.
- The Sception-only and Tception-only ablations are present in code (`Sception` and `Tception` classes are full nn.Module ablations of the full TSception). Paper §4.2 reports both worse than TSception.
- Default `T=9, S=6` differs from the paper's recommended-for-other-datasets `T=S=15, hidden=32` — code comment in `__main__` flags this.
- (No commented-out experiments, no negative result branches in repo)

## PyTorch skeleton

```python
class TSception(nn.Module):
    def __init__(self, num_classes=2, C=4, T=1024, fs=256, num_T=9, num_S=6, hidden=128, p=0.3):
        super().__init__()
        # Inception-style temporal branches with kernel sizes 0.5/0.25/0.125 of fs
        ks = [int(r*fs) for r in (0.5, 0.25, 0.125)]
        def t_branch(k): return nn.Sequential(
            nn.Conv2d(1, num_T, (1,k)), nn.LeakyReLU(),
            nn.AvgPool2d((1,8), (1,8)))
        self.t1, self.t2, self.t3 = t_branch(ks[0]), t_branch(ks[1]), t_branch(ks[2])
        self.bn_t = nn.BatchNorm2d(num_T)

        # Spatial branches with hard hemisphere prior
        def s_branch(rows, stride): return nn.Sequential(
            nn.Conv2d(num_T, num_S, (rows,1), (stride,1)), nn.LeakyReLU(),
            nn.AvgPool2d((1,2), (1,2)))
        self.s1 = s_branch(C, 1)            # collapse all channels
        self.s2 = s_branch(C//2, C//2)      # split into 2 hemisphere groups
        self.bn_s = nn.BatchNorm2d(num_S)

        flat = self._infer_flat(C, T)
        self.fc = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, num_classes))

    def forward(self, x):                              # (B, 1, C, T)
        t = torch.cat([self.t1(x), self.t2(x), self.t3(x)], dim=-1)   # cat over time
        t = self.bn_t(t)                                              # (B, num_T, C, T')
        s = torch.cat([self.s1(t), self.s2(t)], dim=2)                # cat over spatial
        s = self.bn_s(s)                                              # (B, num_S, 3, T'')
        return self.fc(s.flatten(1))
```

Notes on what the codebase reveals beyond the paper:

- The "TSception" name is doing a lot of work — the actual architectural commitment is **multi-scale temporal Inception + hard hemisphere prior + standard FC head**. The "Inception" framing matters: it's not just a multi-branch CNN, it's specifically Inception-style branching with fixed kernel ratios tied to sampling rate. That ratio choice (0.5/0.25/0.125 of fs) is the load-bearing prior, and is fixed-not-learned, which makes it an obvious swap point.
- The Sception2 hard hemisphere split (`k=(C/2,1), s=(C/2,1)`) is interesting and probably underrated — it imposes "left brain vs. right brain" inductive bias by construction. Modern equivalents (GAT, learned masks) are obvious replacements but might lose performance if the prior is genuinely useful for emotion recognition (which is where this paper lives).
- Code says LeakyReLU in the conv blocks but ReLU in the FC head. Inconsistent; flagging because someone might want to unify.
- The temporal output dimensions of the three Tception branches don't match exactly (each conv has a different output length), but they're concatenated along the time axis after pooling. This works because they all pool by 8x — the post-pool lengths come out close enough that concat-along-time is semantically reasonable but not strictly aligned. Worth knowing if you swap the pooling.
- Train file uses `cv='Leave_one_session_out'` but the README's `set_parameter` docstring also lists `Leave_one_subject_out` and `K_fold` — those CV functions aren't actually defined in the file you pasted. The repo presumably has them elsewhere or they were never finished. Flagging in case you want full CV results from this code (you'd need to implement them).

For idea generation, the highest-leverage swaps are: (1) replace fixed-ratio temporal kernels with learned scales or Mamba, (2) replace the hemisphere prior with a learned spatial graph, (3) ditch the L1 reg and add proper weight decay. The Inception-style multi-branch + small temporal kernels is also a clean fit for a Conv-Mamba hybrid where each branch becomes a Mamba scan at a different scale.
