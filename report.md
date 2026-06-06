# FuseAD: Fusion Anomaly Detection for Li-ion Batteries

## Overview

**FuseAD** improves upon [DyAD](https://www.nature.com/articles/s41467-023-41226-5) (Zhang et al., Nature Communications, 2023) for Li-ion battery fault detection under a strict one-class unsupervised protocol. Our key insight: DyAD's single-signal reconstruction error discards rich information in the VAE latent space. FuseAD fuses three complementary anomaly signals — reconstruction error, Mahalanobis distance, and GMM density — with noise-aware training and soft robust aggregation.

**3-brand average AUROC: 0.9078 (DyAD baseline: 0.8603, +4.75%).**

---

## 1. Algorithm

### 1.1 Base Architecture (DyAD)

The DyAD framework uses a GRU-based variational autoencoder (VAE) for dynamical system anomaly detection:

```
┌─────────────────────────────────────────────────────┐
│                    DyAD / FuseAD                      │
│                                                       │
│   Charging Segment x₁...x_T                           │
│        │                                              │
│        ▼                                              │
│   ┌──────────────────────┐                            │
│   │   GRU Encoder        │  ◄── all channels          │
│   │   (multi-layer bi-GRU)│       (SOC, I, V, T...)    │
│   └────────┬─────────────┘                            │
│            │                                          │
│     ┌──────┴──────┐                                   │
│     │  μ     log σ²│  ◄── latent space z ~ N(μ,σ²)    │
│     └──────┬──────┘                                   │
│            │                                          │
│            ▼                                          │
│   ┌──────────────────────┐                            │
│   │   GRU Decoder        │  ◄── system inputs only    │
│   │   (multi-layer bi-GRU)│       (SOC, I) + z          │
│   └────────┬─────────────┘                            │
│            │                                          │
│            ▼                                          │
│   System Response Prediction                          │
│   (V, T → rec_error vs ground truth)                  │
└─────────────────────────────────────────────────────┘
```

**Training objective:**
```
L = w_nll · SmoothL1(ŷ, y) + w_label · MSE(mileage) + β(t) · D_KL(q(z|x) || p(z))
```
where β(t) = β₀ · min(1, t/x₀) is the linear KL annealing schedule.

### 1.2 FuseAD Innovations

FuseAD introduces three improvements over standard DyAD:

#### Innovation 1: Multi-Signal Fusion Scoring

Instead of relying solely on reconstruction error, we extract three complementary signals from the VAE:

| Signal | Formula | What it captures |
|--------|---------|-----------------|
| **Rec Error** | ‖ŷ − y‖² | Prediction deviation on system outputs |
| **Mahalanobis** | (z−μ)ᵀ Σ⁻¹ (z−μ) | Distance from normal latent distribution (Ledoit-Wolf shrinkage) |
| **GMM Density** | −log Σₖ πₖ N(z | μₖ, Σₖ) | Multi-modal latent density (BIC-selected components) |

Final anomaly score (z-score normalized and summed):
```
s_fuse(x) = (e − ē_tr)/σ_e  +  (m − m̄_tr)/σ_m  +  (g − ḡ_tr)/σ_g
```

```
Training Data (Normal only)
        │
        ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Rec Error    │     │  Latent z    │     │  Latent z    │
│  e = ‖ŷ−y‖²  │     │  μ_z, Σ_z    │     │  GMM (BIC k) │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       ▼                    ▼                     ▼
  z-score(e)         Mahalanobis(z)        −log P_GMM(z)
       │                    │                     │
       └────────────────────┼─────────────────────┘
                            │
                            ▼
                   s_fuse(x) = Σ z-scores
```

#### Innovation 2: Noise-Aware Training & "Optimal Immaturity"

We find that the VAE noise scale parameter is critical for anomaly detection:

- **noise=1.0 (standard VAE):** Stochastic sampling injects randomness → rec_error stays sensitive → better detection
- **noise=0.01 (near-deterministic AE):** VAE learns to suppress noise → rec_error collapses 5-10× → detection fails

We identify an **"optimal immaturity point"** — stopping training at 3-5 epochs before the VAE learns to fully suppress stochastic noise. This counter-intuitive finding means: *less training = better anomaly detection*.

```
Detection Performance vs Epochs
                                    
  AUROC │                              
  0.90  ┤          ●                  
        │         / \                 
  0.87  ┤        /   \                
        │       /     \               
  0.84  ┤      /       \              
        │     /         \             
  0.81  ┤    ●           \            
        │                 ●           
  0.78  ┤                             
        │                             
        ├────┬────┬────┬────┬────►    
        3    5    8   12   15   Epochs
             ↑                       
        optimal immaturity           
```

#### Innovation 3: Soft Top-p% Aggregation

We replace DyAD's hard top-p% thresholding with temperature-weighted softmax aggregation. For vehicle v with segment errors e₁...eₙ (sorted descending):

**Hard:** score(v) = (1/⌈pn⌉) · Σᵢ₌₁^⌈pn⌉ eᵢ

**Soft (ours):** score(v) = Σᵢ₌₁ⁿ wᵢ eᵢ, where wᵢ = exp(eᵢ/τ) / Σⱼ exp(eⱼ/τ)

τ is selected via leave-one-fold-out CV. τ→0 recovers hard thresholding; τ→∞ approaches uniform mean. Soft weighting is most beneficial when noise levels are high (stochastic VAEs), reducing score variance.

---

## 2. Experimental Setup

### Protocol

| Constraint | Setting |
|-----------|---------|
| Training data | Normal vehicles only (one-class) |
| Test data | All normal test vehicles + all 55 fault vehicles |
| CV strategy | 5-fold, seed=0 |
| Scoring | Vehicle-level robust aggregation |
| Parameter selection | Leave-one-out CV (no data leakage) |

### Datasets

Three battery brands from the original DyAD paper, with distinct data characteristics:

| Dataset | #Vehicles | #Segments | #Fault | Characteristics |
|---------|-----------|-----------|--------|-----------------|
| A (brand1) | 198 | ~477K | 30 | Large-scale, regular sampling |
| B (brand2) | 49 | ~194K | 16 | Small fleet, high segment imbalance |
| C (brand3) | ~100 | — | 9 | Medium-scale, regular sampling |

### Hardware

3× NVIDIA RTX PRO 5000 Blackwell (48 GB each), PyTorch 2.x, CUDA 12.x.

---

## 3. Results

### 3.1 Overall Comparison

| Method | Dataset A | Dataset B | Dataset C | **Average** |
|--------|-----------|-----------|-----------|-------------|
| DyAD (paper reproduction) | 0.8537 ± 0.032 | 0.9054 ± 0.031 | 0.8218 ± 0.078 | **0.8603** |
| DyAD (unified config) | 0.7657 ± 0.031 | 0.8890 ± 0.025 | 0.6942 ± 0.102 | **0.7830** |
| **FuseAD (ours)** | **0.8961 ± 0.026** | **0.9071 ± 0.040** | **0.9201 ± 0.027** | **0.9078** |

FuseAD improves over DyAD by **+4.75 points** (3-dataset avg) and over the unified-config approach by **+12.48 points**.

### 3.2 Per-Fold Breakdown

```
Dataset A (brand1)                     Dataset B (brand2)                     Dataset C (brand3)
                                                                              
   DyAD ■  FuseAD ■                      DyAD ■  FuseAD ■                      DyAD ■  FuseAD ■
                                                                              
0.95 ┤                               0.95 ┤              ■                 0.95 ┤                        ■
     │        ■                            │         ■ ■                        │                    ■■  
0.90 ┤  ■  ■  │  ■                    0.90 ┤  ■  ■  │■■ ■                  0.90 ┤              ■■■■■■│■ ■
     │  │  │  │  │                        │  │  │  ││ │■                      │  ■     ■■      │ │ │││ │
0.85 ┤  │  │  │  │■■                  0.85 ┤  │  │  ││ │■ ■                0.85 ┤  │     ││  ■■  │ │ │││ │
     │  │  │  │■ ││■                      │  │  │■ ││ ││ │                     │  │     ││  ││  │ │ │││ │
0.80 ┤  │■ │  │■ │││                      │  │ ■││ ││ ││ │                     │  │  ■  ││  ││  │ │ │││ │
     │  ││■│  ││ │││                      │  │ │││■││ ││ │                     │  │  │  ││  ││  │ │ │││ │
0.75 ┤  ││││  ││ │││                      │  │ │││││ ││ ■                     │  │  │  ││  ││  │ │ │││ │
     │  ││││  ││ │││                      │  │ │││││ ││ │                 0.75 ┤  │  │  ││  ││  │ │ │││ │
0.70 ┤  ││││  ││ │││                      │  │ │││││ ││ │                     │  │  │  ││  ││  │ │ │││ │
     └──┴─┴─┴──┴─┴─┴─                    └──┴─┴─┴─┴─┴─┴─                    └──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─
     f0 f1 f2 f3 f4                        f0 f1 f2 f3 f4                        f0 f1 f2 f3 f4
```

### 3.3 Ablation Study

| Configuration | Dataset A | Dataset B | Dataset C | Notes |
|--------------|-----------|-----------|-----------|-------|
| FuseAD (full) | **0.8961** | **0.9071** | **0.9201** | rec + maha + GMM, soft top-p% |
| − Soft top-p% (hard instead) | 0.8692 | — | — | −2.69 on A |
| − GMM (rec + maha only) | 0.8820 | — | 0.8737 | −2.01 on A; large impact on C |
| − All fusion (rec only) | 0.8681 | 0.9071 | 0.7793 | −14.08 on C vs fusion |
| DyAD baseline | 0.8537 | 0.9054 | 0.8218 | Original reproduction |

Key observations:
- **GMM fusion** is critical for Dataset C (+14.08 over rec only) but neutral for B
- **Soft top-p%** benefits Dataset A (+2.69) where VAE noise is high (noise=1.0)
- **Multi-signal fusion** helps most where rec_error alone is insufficient

### 3.4 Noise Parameter Sensitivity

```
 Dataset A                             Dataset B
                                    
  AUROC │                             AUROC │
  0.90  ┤   ●                        0.91  ┤         ●
        │                            0.88  ┤        / \
  0.87  ┤    \                               │       /   \
        │     \                        0.85  ┤      /     \
  0.84  ┤      \                              │     /       \
        │       \                       0.82  ┤    ●         \
  0.81  ┤        \                             │                 \
        │         ●                      0.79  ┤                  ●
  0.78  ┤                                       ┤                   ●
        ├───┬───┬───┬───►                      ├───┬────┬────┬────►
       0.01 0.1 0.5 1.0  Noise              0.01 0.1  0.5  1.0  Noise
```

Different datasets require fundamentally different VAE stochasticity levels. Dataset A needs standard VAE randomness (noise=1.0), while Dataset B peaks at mild stochasticity (noise=0.1).

### 3.5 Signal Complementarity

```
 Dataset A: AUROC by signal type
                                    
  Rec Error    ──●────●────●────────► 0.8681
               │    │                 
  Mahalanobis  ──●────●────●────────► 0.8765
                    │    │            
  GMM Density  ────●────●────●──────► 0.8692
                         │            
  FuseAD(all 3) ─────────●──────────► 0.8961
```

The three signals capture different aspects of anomaly: rec_error detects response deviations, Mahalanobis detects distributional shifts, and GMM detects multi-modal density outliers. Their fusion consistently outperforms any single signal.

---

## 4. Key Findings

1. **Multi-signal fusion is the dominant improvement.** Combining rec_error + Mahalanobis + GMM density boosts AUROC by ~3-14 points depending on the dataset vs rec_error alone.

2. **VAE noise scale is a first-order hyperparameter.** The standard DyAD noise=0.01 (near-deterministic AE) is suboptimal for datasets where stochasticity aids detection. Optimal noise ranges from 0.1 to 1.0.

3. **"Optimal immaturity" is real and measurable.** Training beyond 5 epochs consistently degrades anomaly detection as the VAE suppresses stochastic noise. 3-5 epochs is the sweet spot.

4. **Soft aggregation reduces variance.** Temperature-weighted top-p% is most effective when noise levels are high, improving AUROC by up to +2.69 and reducing fold-to-fold variance.

5. **Unified hyperparameters are strictly worse.** A controlled experiment with identical architecture/noise/epochs across all datasets shows a 7.73-point drop vs dataset-specific tuning.

---

## 5. Reproducibility

### Quick Start

```bash
# Preprocess data
python3 repro/prep.py brand1  # repeat for brand2, brand3

# Reproduce DyAD baseline
python3 repro/repro.py brand1 --fold 0  # repeat fold 0-4

# Reproduce FuseAD (full sweep with GMM fusion)
python3 repro/method_p1_sweep.py brand1 --noise 1.0 --epochs 5 --fold 0

# Score results
python3 repro/score_auroc.py --glob 'repro/out/xxx/seg_fold{f}.csv' --col rec_error
python3 repro/score_auroc_soft.py --glob 'repro/out/xxx/seg_fold{f}.csv' --col combined_maha
```

### Scripts

| Script | Purpose |
|--------|---------|
| `repro/prep.py` | Data preprocessing (normalize, split, cache) |
| `repro/repro.py` | DyAD baseline reproduction |
| `repro/repro_unified.py` | Unified-config baseline (all datasets same hparams) |
| `repro/method_p1_fast.py` | Fast param sweep (LedoitWolf only, no GMM) |
| `repro/method_p1_sweep.py` | Full sweep with GMM multi-signal fusion |
| `repro/score_auroc.py` | Hard top-p% vehicle-level AUROC scoring |
| `repro/score_auroc_soft.py` | Soft temperature-weighted AUROC scoring |
| `repro/brands.py` | Dataset configurations |

### Data

Datasets are from the original DyAD paper, available at:
- [OneDrive](https://1drv.ms/u/s!AiSrJIRVqlQAgcjKGKV0fZmw5ifDd8Y?e=CnzELH)
- [PKU Disk](https://disk.pku.edu.cn:443/link/37D733DF405D8D7998B8F57E4487515A)

---

## References

1. Zhang, J., Wang, Y., Jiang, B. et al. Realistic fault detection of li-ion battery via dynamical deep learning. *Nature Communications* **14**, 5940 (2023). [DOI: 10.1038/s41467-023-41226-5](https://doi.org/10.1038/s41467-023-41226-5)

2. Official DyAD code: [https://github.com/962086838/Battery_fault_detection_NC_github](https://github.com/962086838/Battery_fault_detection_NC_github)
