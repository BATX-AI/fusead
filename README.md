# FuseAD: Fusion Anomaly Detection for Li-ion Batteries

[![AUROC](https://img.shields.io/badge/AUROC-0.9078-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

**FuseAD** improves Li-ion battery fault detection by fusing multiple anomaly signals from a variational autoencoder (VAE). Building on [DyAD](https://www.nature.com/articles/s41467-023-41226-5) (Zhang et al., *Nature Communications*, 2023), FuseAD combines reconstruction error, Mahalanobis distance, and GMM latent density with noise-aware training and soft robust aggregation.

> **3-brand average AUROC: 0.9078** (DyAD: 0.8603, **+4.75%**)

## Key Innovations

| Innovation | Description | Impact |
|-----------|-------------|--------|
| **Multi-Signal Fusion** | Combine rec_error + Mahalanobis + GMM density in latent space | +3.5 AUROC avg |
| **Noise-Aware Training** | Identify optimal VAE noise scale and "immaturity point" (3-5 epochs) | Prevents detection collapse |
| **Soft Top-p% Aggregation** | Temperature-weighted scoring instead of hard thresholding | +2.7 on high-noise settings |

## Architecture

```
Charging Segment ──► GRU Encoder ──► Latent z ~ N(μ,σ²) ──► GRU Decoder ──► Response Prediction
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
               Rec Error            Mahalanobis(z)         GMM Density
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                                   FuseAD Score
```

## Results

| Method | Dataset A | Dataset B | Dataset C | **Average** |
|--------|-----------|-----------|-----------|-------------|
| DyAD (paper repro) | 0.8537 ± 0.032 | 0.9054 ± 0.031 | 0.8218 ± 0.078 | **0.8603** |
| **FuseAD (ours)** | **0.8961 ± 0.026** | **0.9071 ± 0.040** | **0.9201 ± 0.027** | **0.9078** |

## Quick Start

```bash
# Clone
git clone https://github.com/BATX-AI/fusead.git
cd fusead

# Preprocess
python3 repro/prep.py brand1  # brand2, brand3

# Run baseline
python3 repro/repro.py brand1 --fold 0  # folds 0-4

# Run FuseAD with GMM fusion
python3 repro/method_p1_sweep.py brand1 --noise 1.0 --epochs 5 --fold 0

# Score
python3 repro/score_auroc.py --glob 'repro/out/xxx/seg_fold{f}.csv' --col rec_error
```

## Data

Battery charging data from 347 EVs across 3 brands, provided by the DyAD authors:
- [OneDrive](https://1drv.ms/u/s!AiSrJIRVqlQAgcjKGKV0fZmw5ifDd8Y?e=CnzELH)
- [PKU Disk](https://disk.pku.edu.cn:443/link/37D733DF405D8D7998B8F57E4487515A)

Place extracted data under `data/battery_brand1/`, `data/battery_brand2/`, `data/battery_brand3/`.

## Citation

If you use this code, please cite the original DyAD paper:

```bibtex
@article{zhang2023realistic,
  title={Realistic fault detection of li-ion battery via dynamical deep learning},
  author={Zhang, Jingzhao and Wang, Yanan and Jiang, Benben and He, Haowei and Huang, Shaobo and Wang, Chen and Zhang, Yang and Han, Xuebing and Guo, Dongxu and He, Guannan and Ouyang, Minggao},
  journal={Nature Communications},
  volume={14},
  pages={5940},
  year={2023},
  doi={10.1038/s41467-023-41226-5}
}
```

Original DyAD repository: [https://github.com/962086838/Battery_fault_detection_NC_github](https://github.com/962086838/Battery_fault_detection_NC_github)

## License

MIT
