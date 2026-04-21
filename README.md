# CSI WiFi-Based Human Activity Recognition

## Overview

This repository implements deep learning models for **passive Human Activity Recognition (HAR)** using **Channel State Information (CSI)** extracted from commodity WiFi signals. The project benchmarks a ResNet-18 baseline against a novel hybrid CNN–Transformer architecture, both trained on the publicly available **UT-HAR** dataset.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
   - [What is CSI?](#what-is-csi)
   - [Why WiFi Sensing?](#why-wifi-sensing)
   - [The Classification Task](#the-classification-task)
2. [Existing Solutions](#existing-solutions)
   - [Traditional Machine Learning Approaches](#traditional-machine-learning-approaches)
   - [Deep Learning Approaches](#deep-learning-approaches)
   - [Limitations of Prior Work](#limitations-of-prior-work)
3. [Solution Deployed in This Repository](#solution-deployed-in-this-repository)
   - [Dataset: UT-HAR](#dataset-ut-har)
   - [Data Pipeline & Normalisation](#data-pipeline--normalisation)
   - [Model 1: CSIBaseline (ResNet-18)](#model-1-csibaseline-resnet-18)
   - [Model 2: CSIHybrid (CNN + Transformer)](#model-2-csihybrid-cnn--transformer)
   - [Training Strategy](#training-strategy)

---

## Problem Statement

### What is CSI?

Modern WiFi systems based on the IEEE 802.11n/ac/ax standards use **OFDM (Orthogonal Frequency Division Multiplexing)**, which splits the radio channel into many parallel narrow-band sub-carriers. For each sub-carrier, the wireless NIC can report the **Channel State Information (CSI)** — a complex-valued measurement that describes how the physical environment transforms the transmitted signal before it arrives at the receiver.

Formally, for a transmitted signal **X** and received signal **Y**, the relationship is:

```
Y = H * X + N
```

where **H** is the channel matrix and **N** is noise. CSI is an estimate of **H**, providing both **amplitude** (signal attenuation) and **phase** (signal delay) information for each sub-carrier and each antenna pair. A single CSI snapshot is therefore a matrix of complex numbers of shape `[antennas × sub-carriers]`.

When collected over time (multiple packets per second), CSI forms a **3-D tensor** `[time × antennas × sub-carriers]`, creating a rich spatio-temporal fingerprint of the physical channel.

### Why WiFi Sensing?

The key insight behind passive WiFi-based HAR is that **human motion perturbs the radio channel**. A person walking between a WiFi transmitter and receiver will cause small but consistent fluctuations in the CSI measurements. Different activities (walking, sitting, waving, falling) produce characteristically different disturbance patterns in amplitude and phase across the sub-carriers and over time.

This makes WiFi sensing attractive for a number of reasons compared to camera- or wearable-based systems:

| Property | Camera-based | Wearable-based | WiFi CSI |
|---|---|---|---|
| **Privacy** | Low (visual) | Medium | High (no images) |
| **User burden** | None | Must wear device | None |
| **Coverage** | Line-of-sight | Requires on-body | Through-wall capable |
| **Infrastructure cost** | Cameras required | Devices required | Existing WiFi reused |
| **Lighting dependence** | High | None | None |

WiFi signals can penetrate walls and do not require dedicated sensors beyond a standard WiFi NIC that supports CSI extraction, making the approach both privacy-preserving and infrastructure-efficient.

### The Classification Task

Given a windowed segment of CSI measurements, the task is to classify which of **seven human activities** is being performed:

| Label | Activity |
|---|---|
| 0 | Lie Down |
| 1 | Fall |
| 2 | Pick Up |
| 3 | Run |
| 4 | Sit Down |
| 5 | Stand Up |
| 6 | Walk |

This is a standard multi-class classification problem. The challenge lies in the signal's subtlety: activities such as "sit down" and "stand up" produce structurally similar CSI perturbations that differ mainly in their temporal ordering, while environmental noise and non-stationarity of the wireless channel can obscure discriminative features.

---

## Existing Solutions

### Traditional Machine Learning Approaches

Early WiFi sensing systems relied on hand-crafted features extracted from CSI amplitudes, followed by classical classifiers:

- **RSSI-based methods**: Older work used coarse Received Signal Strength Indicator (RSSI) rather than per-subcarrier CSI. RSSI is a single scalar per packet, providing insufficient resolution to distinguish fine-grained activities.
- **Statistical feature extraction**: Mean, variance, skewness, and spectral energy of CSI amplitude across sub-carriers were used as input to SVMs, k-NN, or Random Forests. These approaches are brittle across environments and require careful feature engineering.
- **PCA / PCA-based de-noising**: Principal Component Analysis was applied to reduce CSI dimensionality and isolate the human-motion components from static multipath reflections. The resulting lower-dimensional features were then fed to classifiers.
- **DTW (Dynamic Time Warping)**: Used to compare the temporal shape of CSI traces, particularly for gesture recognition, but computationally expensive and environment-sensitive.

These classical approaches generally achieve acceptable accuracy in controlled, single-environment settings but degrade sharply when the environment changes (furniture moved, different room, different transmitter position) because the hand-crafted features do not generalise.

### Deep Learning Approaches

The availability of labelled CSI datasets and the success of deep learning in signal processing motivated several neural network architectures:

**CNN-based methods**: Treating the 2-D CSI heatmap (time × sub-carriers) as an image and applying standard convolutional networks (e.g., ResNet, VGG) was an early and effective baseline. CNNs can automatically learn local spatial patterns corresponding to specific activity signatures without manual feature engineering.

**RNN / LSTM-based methods**: Since CSI is a time series, Recurrent Neural Networks and Long Short-Term Memory (LSTM) networks were applied to model temporal dependencies across packet timestamps. The original **UT-HAR** paper (Yang et al.) proposed a **Bidirectional LSTM (BiLSTM)** as the primary model, arguing that activity recognition requires capturing both forward and backward context in the time sequence.

**CNN-LSTM hybrids**: Several works combine a CNN front-end (to extract per-timestep spatial features from sub-carriers) with an LSTM back-end (to model temporal evolution of those features). This two-stage design aims to decouple spatial and temporal feature extraction.

**Attention-based models**: More recent approaches apply self-attention directly to CSI sequences, exploiting the Transformer's ability to model long-range temporal dependencies without the gradient vanishing issues of RNNs.

**Transformer-only models**: Patch-based Vision Transformers (ViT) have been adapted to treat the CSI matrix as a sequence of patches, processing the full spatio-temporal tensor with multi-head self-attention.

### Limitations of Prior Work

Despite the progress above, several challenges remain:

1. **Local vs. global temporal context**: Pure CNN models excel at detecting local spatio-temporal patterns but struggle to model the global sequential structure of an activity (e.g., the transition from raising to lowering an arm). RNNs handle sequential order but process features extracted independently at each timestep.
2. **Long-range dependencies in RNNs**: LSTMs suffer from vanishing gradients over very long sequences and process tokens strictly sequentially, preventing parallelisation during training.
3. **Lack of positional awareness in CNNs**: When a CNN processes the full CSI heatmap as a 2-D image, it treats the time and sub-carrier axes symmetrically, even though they carry fundamentally different types of information (temporal dynamics vs. frequency-domain channel response).
4. **Environment sensitivity**: Models trained in one room or with one device often fail in new settings, a problem that exists across all approaches and motivates cross-environment generalisation research.

---

## Solution Deployed in This Repository

### Dataset: UT-HAR

This repository uses the **UT-HAR** dataset (University of Texas Human Activity Recognition), a publicly available benchmark collected using commodity Intel 5300 NICs. It provides pre-segmented CSI amplitude matrices of shape `[250 time-steps × 90 sub-carriers]` for each sample, along with integer activity labels (0–6). The dataset ships with canonical train, validation, and test splits stored as CSV files.

The 250-step window at typical CSI collection rates (roughly 100 packets per second) corresponds to approximately 2.5 seconds of activity data per sample. The 90 sub-carriers come from 30 sub-carriers × 3 antenna pairs commonly available on the Intel 5300.

### Data Pipeline & Normalisation

Raw CSI data from the UT-HAR CSV files is preprocessed by the `DataLoad/DataLoader.py` pipeline:

1. **Per-sample normalisation**: Each sample (a 250 × 90 matrix) is independently standardised (zero mean, unit variance, with ε = 1e-8 for numerical stability). This removes the effect of path-loss and coarse-scale channel variation that differs between environments, making the model focus on the relative pattern within each window rather than its absolute magnitude.
2. **Zarr caching**: Normalised tensors are stored in a Zarr array on disk. Zarr is a chunked, compressed array format that allows efficient random-access loading during training without reading the entire dataset into RAM. On subsequent runs, the preprocessed cache is reused directly.
3. **Dask-backed computation**: During cache construction, Dask arrays are used to compute per-sample statistics in a memory-efficient chunked manner, enabling the pipeline to handle arbitrarily large datasets without loading everything into memory at once.
4. **Channel dimension expansion**: Each 2-D sample `(250, 90)` is expanded to `(1, 250, 90)` before being passed to the model, treating the CSI heatmap as a single-channel image analogous to a grayscale photograph.

### Model 1: CSIBaseline (ResNet-18)

`Model/CSIBaseline.py` implements the **baseline model** — a modified **ResNet-18** from the torchvision model zoo.

**Architecture overview**:

ResNet-18 is a residual convolutional network consisting of 8 residual blocks (pairs of 3×3 convolutions with skip connections), batch normalisation, and a global average pooling layer before the final fully connected classifier. The skip connections (`x + F(x)`) allow gradients to flow directly through the network during backpropagation, enabling training of deeper networks without severe vanishing gradient problems.

**Modification for CSI**:

The standard ResNet-18 expects 3-channel (RGB) input. Since CSI heatmaps are single-channel, the first convolutional layer (`conv1`) is replaced with an equivalent layer accepting **1 input channel** while preserving all other hyperparameters (64 output channels, 7×7 kernel, stride 2). The final fully connected layer is replaced with a linear layer mapping from 512 features to **7 output classes**.

**Rationale**: The baseline tests whether treating CSI heatmaps as 2-D "images" and applying a powerful, well-studied image classification backbone (with residual learning) provides a competitive starting point. It also serves as an upper-bound reference for purely spatial (non-temporal-aware) modelling.

**Limitation**: ResNet treats the time axis and the sub-carrier axis symmetrically as two spatial dimensions of an image. This means a kernel that spans 3 timesteps and 3 sub-carriers is treated identically to one spanning 3 sub-carriers and 3 timesteps. The architecture has no explicit mechanism to encode that the time axis is ordered and sequential.

### Model 2: CSIHybrid (CNN + Transformer)

`Model/CSIHybrid.py` implements the **proposed hybrid architecture**, designed to overcome the baseline's lack of temporal awareness.

**Design philosophy**: The model separates two complementary aspects of the problem:
- **Spatial (frequency-domain) feature extraction** — understanding how activity affects the pattern across sub-carriers at each moment.
- **Temporal modelling** — understanding how those spatial features evolve over time during an activity.

**Architecture in detail**:

**Stage 1 — CNN front-end (spatial feature extractor)**:
```
Input: (B, 1, 250, 90)   # batch, channels, time, sub-carriers

Conv2d(1→16, 3×3, padding=1)  →  BatchNorm2d  →  ReLU
MaxPool2d(1×2)                                      # 90 → 45 sub-carriers; time unchanged

Conv2d(16→32, 3×3, padding=1) →  BatchNorm2d  →  ReLU
MaxPool2d(1×3)                                      # 45 → 15 sub-carriers; time unchanged

Output: (B, 32, 250, 15)
```

The CNN pools **only along the sub-carrier (frequency) axis**, deliberately preserving the full 250-step temporal resolution. Each output timestep is described by a 32-channel feature map across 15 remaining frequency bins, yielding a `32 × 15 = 480`-dimensional feature vector per timestep.

**Stage 2 — Sequence reshaping**:
```
(B, 32, 250, 15)  →  permute  →  (B, 250, 32, 15)  →  reshape  →  (B, 250, 480)
```

The tensor is reshaped into a sequence of 250 tokens, each of dimension 480, which is the format expected by the Transformer encoder.

**Stage 3 — Positional encoding**:
```
x = x + pos_encoder   # pos_encoder is a learnable (1, 250, 480) parameter
```

Because the Transformer's self-attention mechanism is permutation-equivariant (it has no built-in notion of order), a learnable positional encoding is added to each token to inject temporal position information. Unlike the fixed sinusoidal encodings in the original Transformer paper, learnable positional encodings allow the model to discover the most useful positional representation from data.

**Stage 4 — Transformer encoder**:
```
TransformerEncoderLayer:
  - d_model = 480
  - nhead = 8          # 8 attention heads (60 dimensions each)
  - dim_feedforward = 1024
  - dropout = 0.1

TransformerEncoder: 3 stacked layers
```

The Transformer encoder applies **multi-head self-attention** over all 250 timesteps simultaneously. Each of the 8 attention heads can independently learn to focus on different temporal relationships — for instance, one head might learn to correlate the beginning and end of an activity, while another focuses on short-range motion dynamics. Stacking 3 encoder layers allows the model to build increasingly abstract temporal representations through successive rounds of attention.

Self-attention operates in O(n²) time with respect to sequence length n, which for n=250 is computationally tractable. The feed-forward sublayer within each encoder block applies a two-layer MLP (480→1024→480) independently to each timestep, adding non-linear transformations of the attended features.

**Stage 5 — Global average pooling & classifier**:
```
x = x.mean(dim=1)       # (B, 250, 480) → (B, 480)  [temporal mean pooling]

Linear(480 → 128) → ReLU → Dropout(0.5) → Linear(128 → 7)
```

The 250 Transformer output tokens are averaged to produce a single fixed-length representation of the entire activity window. This is then passed through a two-layer MLP classifier with dropout regularisation to produce the 7-class logit scores.

**Why this design outperforms a pure CNN or pure RNN**:

- Unlike the ResNet baseline, the hybrid model **explicitly decouples spatial and temporal processing**, applying convolutions along the frequency axis and self-attention along the time axis.
- Unlike LSTM-based approaches, the Transformer encoder **processes all timesteps in parallel** during training, enabling faster convergence on GPU hardware and avoiding vanishing gradient issues over the 250-step sequence length.
- The **multi-head attention mechanism** can capture dependencies between any two timesteps regardless of their distance in the sequence, which is particularly valuable for activities where the early and late phases are both diagnostic (e.g., "fall" involves a rapid transition followed by a static phase).

### Training Strategy

Both models are trained with the following configuration in `Training/Training_baseline.py`:

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Optimiser** | AdamW | Adam with decoupled weight decay; better regularisation than vanilla Adam |
| **Learning rate** | 0.001 | Standard starting point for AdamW on classification tasks |
| **Weight decay** | 1e-4 | L2 regularisation to prevent overfitting on the fixed dataset |
| **Loss function** | Cross-Entropy | Standard for multi-class classification; compatible with softmax outputs |
| **Batch size** | 64 | Balances gradient noise and GPU memory utilisation |
| **Epochs** | 20 | Sufficient for convergence on UT-HAR's dataset size |
| **Model selection** | Best validation accuracy | Prevents overfitting to train set; final evaluation on held-out test set |

The training loop evaluates the model on both the **validation** and **test** splits after each epoch. The checkpoint with the highest validation accuracy is saved to `checkpoints/baseline_best.pt`. This ensures the reported test accuracy corresponds to the model selected by the validation criterion, following standard evaluation protocol and avoiding test-set leakage into hyperparameter decisions.

---

## Summary

This repository explores the use of WiFi CSI as a privacy-preserving, infrastructure-reusing medium for passive human activity recognition. The classical CSI-image baseline (ResNet-18) provides a strong spatial feature extractor, while the proposed hybrid CNN–Transformer architecture additionally models the sequential temporal structure inherent in human motion data, addressing a fundamental limitation of treating CSI as a static 2-D image.
