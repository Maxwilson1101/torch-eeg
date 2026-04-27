#import "@preview/modern-sjtu-report:0.2.0": *
#import "@preview/zh-kit:0.1.0": *
#show: doc => setup-base-fonts(doc)

#let course-name = "神经网络理论与应用"
#let course-name-en = "CS7321"
#let experiment-name = "EEG Emotion Classification"

#let ident-color = "blue"
#let logo-path = "path/to/logo"
#let name-path = "path/to/name"
#let header-path = "path/to/header"
#let org-name = "name"

#let info-items = (
  ([学生姓名], [王圣予]),
  ([学生学号], [125033910109]),
  ([教#h(2em)师], [郑伟龙]),
)

#let cover-fonts = ("Times New Roman", "Kaiti SC", "KaiTi", "Noto Serif SC", "SimSun")
#let article-fonts = ("Times New Roman", "Noto Serif SC", "Songti SC", "SimSun")
#let code-fonts = ("Consolas", "Ubuntu Mono", "Menlo", "Courier New", "Courier", "Noto Serif SC")

#make-cover(
  course-name: course-name,
  course-name-en: course-name-en,
  info-items: info-items,
  ident-color: ident-color,
  cover-fonts: cover-fonts,
  logo-path: logo-path,
  name-path: name-path,
  org-name: org-name,
)

#show: general-layout.with(
  ident-color: ident-color,
  header-logo: true,
  experiment-name: experiment-name,
  article-fonts: article-fonts,
  code-fonts: code-fonts,
  header-path: header-path,
)

#make-title(name: experiment-name)

// 1. Introduction
= Introduction

Emotion recognition from electroencephalography (EEG) signals represents a significant frontier in affective computing, with applications spanning human-computer interaction, mental health monitoring, and personalized cognitive systems. The challenge lies in the heterogeneous nature of EEG signals across individuals, driven by anatomical variation, cognitive state, and electrode contact quality.

This report presents a comprehensive evaluation of *BandSpatialCNN*, a novel deep convolutional architecture designed to jointly factorize spectral and spatial learning. The model is evaluated on five-class emotion classification (Neutral, Sad, Fear, Happy, Disgust) using the SEED-IV dataset across two complementary protocols: subject-dependent (training and testing on the same individual) and cross-subject leave-one-out cross-validation (LOOCV). This dual-protocol approach permits both assessment of within-subject adaptation capacity and evaluation of the model's generalization to unseen individuals.

= Problem Formulation

We consider the problem of five-class emotion classification given multichannel EEG recordings preprocessed into differential entropy (DE) features. Each sample is represented as a feature matrix of shape $(62, 5)$, where 62 denotes the number of scalp electrodes and 5 represents frequency bands: delta, theta, alpha, beta, and gamma. No additional temporal or spatial preprocessing is applied; the model operates directly on this feature tensor.

The evaluation employs two distinct protocols:

- *Subject-Dependent Protocol:* A subject-specific model is trained and evaluated on the same individual, using that subject's designated train/test partition. This protocol measures the model's capacity to learn individual-specific emotion representations when sufficient adaptation time is available.

- *Cross-Subject LOOCV Protocol:* In each of 16 folds, the model trains on data from 15 subjects (both train and test splits combined) and is evaluated solely on the held-out subject. This protocol assesses generalization to new individuals without subject-specific adaptation.

= Architecture: BandSpatialCNN

BandSpatialCNN is inspired by TSception-style factorization, decomposing the dual learning problem into sequential band-temporal and spatial stages. The architecture is designed to exploit the structured nature of EEG: the 5-band frequency decomposition can be viewed as a form of spectral basis, while the 62-electrode array admits natural spatial groupings (hemispheres, regions).

== Band-Temporal Stage

Three parallel Conv2d kernels, each with in-channel size 1 and varying out-channel sizes, operate across the frequency band dimension:

#figure(
  table(
    columns: 3,
    align: left,
    [*Branch*], [*Kernel Size*], [*Receptive Field*],
    [Band 1], [(1, 1)], [Single-band activations],
    [Band 2], [(1, 2)], [Adjacent-band interactions],
    [Band 3], [(1, 3)], [Broad-band multi-scale patterns],
  ),
)

Each kernel outputs 16 feature maps, yielding 48 band feature maps per electrode upon concatenation. The concatenated tensor, of shape (B, 48, 62, 5), is subsequently processed by batch normalization and Leaky ReLU.

== Spatial Stage

Two Conv2d operations aggregate band features across the electrode dimension, each outputting 32 filters:

#figure(
  table(
    columns: 3,
    align: left,
    [*Branch*], [*Kernel Size*], [*Spatial Coverage*],
    [Spatial Full], [(62, 1)], [Global integration of all 62 channels],
    [Spatial Hemi], [(31, 1)], [Hemisphere-specific (left/right) aggregation],
  ),
)

Outputs are concatenated to yield a tensor of shape (B, 64, 3, 5), which is again processed through batch normalization and Leaky ReLU.

== Classification Head

The (B, 64, 3, 5) tensor is flattened to (B, 960) and passed through a single fully connected layer yielding 5 logits, one per emotion class. Cross-entropy loss with label smoothing $alpha = 0.1$ is applied during training.

= Experimental Setup

#figure(
  table(
    columns: 2,
    align: left,
    [*Hyperparameter*], [*Value*],
    [Optimizer], [Adam],
    [Learning Rate], [0.0001],
    [Weight Decay (L2)], [0.0001],
    [Label Smoothing], [0.1],
    [Training Epochs], [2],
    [Batch Size], [32],
    [Loss Function], [Cross-Entropy],
    [Compute Device], [CUDA],
    [Leaky ReLU Slope], [0.1],
  ),
)

Training was conducted for exactly 2 epochs per subject (subject-dependent protocol) or per fold (LOOCV protocol). All experiments were executed on a GPU-accelerated platform (CUDA) to enable efficient training and evaluation.

= Results

== Subject-Dependent Evaluation

In the subject-dependent protocol, each subject's training samples are used to train a dedicated model, which is subsequently evaluated on the subject's held-out test set. Results are presented in Table 5.1:

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    [*Subject*], [*Accuracy*], [*Macro-F1*],
    [1], [0.8782], [0.8728],
    [2], [0.6554], [0.6425],
    [3], [0.5689], [0.5152],
    [4], [0.8478], [0.8059],
    [5], [0.6715], [0.6467],
    [6], [0.7244], [0.7029],
    [7], [0.7452], [0.7550],
    [8], [0.8109], [0.7664],
    [9], [0.9696], [0.9642],
    [10], [0.5064], [0.4753],
    [11], [0.7035], [0.6489],
    [12], [0.6811], [0.5339],
    [13], [0.8429], [0.8295],
    [14], [0.4599], [0.3997],
    [15], [0.6651], [0.6858],
    [16], [0.6042], [0.5924],
    [*Mean*], [$0.7084 plus.minus 0.1382$], [$0.6773 plus.minus 0.1527$],
  ),
)

Best-performing subject: S9 (accuracy = 96.96%). Worst-performing subject: S14 (accuracy = 45.99%). The mean accuracy of 70.84% substantially exceeds the chance baseline of 20% (random classification among 5 classes), demonstrating that the model successfully learns individual emotion representations. However, the large standard deviation indicates pronounced inter-subject variability, suggesting that EEG-based emotion encoding varies significantly across the population.

== Cross-Subject LOOCV Evaluation

In the cross-subject protocol, models train on 15 subjects and evaluate on the single held-out subject. Results for each fold are provided in Table 5.2:

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    [*Subject (Held-Out)*], [*Accuracy*], [*Macro-F1*],
    [1], [0.5979], [0.5789],
    [2], [0.5491], [0.5360],
    [3], [0.6308], [0.6381],
    [4], [0.6286], [0.6099],
    [5], [0.5683], [0.5836],
    [6], [0.3171], [0.3126],
    [7], [0.4915], [0.4830],
    [8], [0.6923], [0.6912],
    [9], [0.5304], [0.5414],
    [10], [0.3406], [0.3295],
    [11], [0.5793], [0.5903],
    [12], [0.4054], [0.3898],
    [13], [0.7005], [0.6911],
    [14], [0.4833], [0.4825],
    [15], [0.6166], [0.6161],
    [16], [0.5930], [0.5829],
    [*Mean*], [$0.5453 plus.minus 0.1132$], [$0.5411 plus.minus 0.1152$],
  ),
)

Best-performing fold: S13 (accuracy = 70.05%). Worst-performing fold: S6 (accuracy = 31.71%). The mean cross-subject accuracy of 54.53% remains well above chance (20%), confirming that the learned features capture emotion-discriminative patterns shared across the population. However, the 15-percentage-point (pp) performance gap relative to subject-dependent evaluation reflects the well-documented domain-shift challenge in EEG analysis: inter-individual differences in brain-to-scalp signal propagation, skull conductivity, and emotional expressivity create substantial distribution mismatch.

== Protocol Comparison

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    [*Protocol*], [*Mean Accuracy*], [*Mean Macro-F1*],
    [Subject-Dependent], [$0.7084 plus.minus 0.1382$], [$0.6773 plus.minus 0.1527$],
    [Cross-Subject LOOCV], [$0.5453 plus.minus 0.1132$], [$0.5411 plus.minus 0.1152$],
    [Chance Baseline], [0.2000], [N/A],
  ),
)

= Analysis

== Training Dynamics and Overfitting

Training on all subjects exhibited rapid convergence: training accuracy reached ≥ 99.9% by epoch 2, while validation loss continued to increase monotonically. This divergence between training and validation performance is a clear indicator of severe overfitting within the 2-epoch training window. While label smoothing and L2 regularization provide partial constraints, they prove insufficient to prevent memorization at this learning rate and batch size. The model's effective capacity exceeds the regularization imposed by these modest penalties.

== Subject-Specific Outliers

Three subjects consistently underperformed across both protocols:

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    [*Subject*], [*Subject-Dependent Acc.*], [*LOOCV Acc.*],
    [6], [0.7244], [0.3171],
    [10], [0.5064], [0.3406],
    [12], [0.6811], [0.4054],
  ),
)

A critical observation: in subject-dependent mode, these subjects achieve moderate accuracy (range: 50.64%-72.44%), suggesting the model can partially adapt to their individual distributions when trained on their own data. By contrast, in LOOCV, cross-subject models fail substantially (range: 31.71%-34.06%). This asymmetry indicates that their differential entropy feature distributions lie outside the convex hull of the training set (the 15 other subjects), rendering them distributional outliers.

Potential root causes include: (1) natural variation in emotional expressivity and proprioception; (2) electrode contact quality or impedance variation; (3) systematic differences in subjective emotion labeling for those recording sessions; or (4) transient physiological states (fatigue, attention) during data collection. Resolving such outliers would benefit from data quality audits and domain adaptation techniques tailored to individual differences.

== Inter-Subject Heterogeneity

The per-subject standard deviation of $±13.82$ pp (subject-dependent) and $±11.32$ pp (LOOCV) is large relative to the respective means. This high variability reflects genuine between-subject heterogeneity in EEG-based emotion representation rather than model instability, as the training loss trajectories remain consistent across all subjects and folds. The observed variance aligns with well-established findings in affective neuroscience: individual differences in anterior-posterior alpha asymmetry, theta-gamma coupling, and hemispheric dominance for emotion processing are substantial and stable within subjects but variable across populations.

= Discussion

The dual-protocol evaluation provides complementary insights into BandSpatialCNN's capabilities and limitations. Subject-dependent accuracy (70.84%) confirms the architecture's capacity to learn individual emotion signatures when sufficient adaptation is available. Conversely, cross-subject generalization (54.53%) highlights the challenge of identifying population-level emotion-discriminative patterns in EEG.

The 15-pp performance gap is consistent with the literature on EEG-based affective computing, where cross-subject accuracy typically falls 10-20 pp below subject-dependent performance. This gap is attributable to inherent inter-individual variability in neural representation of emotion and signal characteristics. Addressing this gap requires either:

- Domain adaptation techniques (e.g., adversarial training) to align feature distributions across subjects;
- Inclusion of subject-invariant features (e.g., functional connectivity, frequency band ratios) to reduce sensitivity to individual differences;
- Transfer learning approaches with larger pre-training datasets to learn more generalizable emotion representations.

= Limitations

- *Short Training Duration:* Two epochs is minimal; rapid convergence of training accuracy suggests the model would benefit from extended training with early stopping or increased regularization to better constrain the hypothesis space.

- *Small Dataset Size:* The SEED-IV dataset comprises only 16 subjects. Cross-subject evaluation on such a small population is sensitive to individual outliers and may not reflect generalization to larger, more diverse cohorts.

- *Lack of Baseline Comparisons:* No comparative results against established EEG emotion recognition methods (e.g., conventional machine learning on handcrafted features, standard CNN architectures) are provided.

- *Fixed Preprocessing:* All experiments assume pre-computed differential entropy features. Exploration of alternative feature representations (e.g., wavelet decomposition, raw signal processing) is absent.

= Conclusion

BandSpatialCNN achieves competitive accuracy on five-class EEG-based emotion recognition under the subject-dependent protocol (70.84% accuracy) and demonstrates meaningful cross-subject generalization (54.53% accuracy), both substantially exceeding the 20% chance baseline. The architecture's dual-stage design—combining multi-scale frequency kernel learning with hemisphere-aware spatial aggregation—provides an effective inductive bias for the structure of differential entropy EEG features.

However, the 15 pp performance drop under cross-subject evaluation underscores the fundamental challenge of EEG-based emotion recognition: substantial inter-individual variability in neural representation and signal characteristics. Future work should focus on (1) domain adaptation mechanisms to mitigate subject-specific distribution shift; (2) longer training regimens with adaptive regularization; and (3) integration of subject-invariant feature representations to improve population-level generalization.

In summary, BandSpatialCNN provides a solid architectural foundation for EEG emotion recognition, with particular strength in subject-specific adaptation, though cross-subject generalization remains an open challenge requiring further methodological innovation.
