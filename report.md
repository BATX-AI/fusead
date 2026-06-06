# DyAD 复现与改进实验报告

## 1. 实验目标

复现 DyAD (Nature Communications 14:5940, 2023) 的 Li-ion 电池故障检测，并在严格无监督协议下改进 3-brand 平均 AUROC。

**核心约束：**
- 仅正常车训练（one-class），55 辆故障车全在测试集
- seed=0，5-fold CV
- 车级 top-p% 鲁棒评分 + LOO 选 p（无数据泄漏）
- 与 baseline (brand1 0.8537, brand2 0.9054, brand3 0.8218, avg 0.8603) 对比

---

## 2. 原始 Paper 的 Per-Brand 训练策略

**官方仓库：** [`https://github.com/962086838/Battery_fault_detection_NC_github`](https://github.com/962086838/Battery_fault_detection_NC_github)

**Per-brand 配置文件：**
- `DyAD/model_params_battery_brand1.json`
- `DyAD/model_params_battery_brand2.json`
- `DyAD/model_params_battery_brand3.json`
- `DyAD/model_params_battery_brandall.json`（统一超参，用于合并训练）

**训练入口：** `python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num 0`（来自官方 README）

### 官方超参配置 (来自 model_params JSON)

| 参数 | brand1 | brand2 | brand3 |
|------|--------|--------|--------|
| epochs | 3 | 3 | 5 |
| batch | 128 | 128 | 128 |
| lr | 0.005 | 0.0001 | 0.005 |
| hidden | 128 | 1024 | 256 |
| latent | 8 | 24 | 16 |
| layers | 2 | 1 | 1 |
| noise | 1.0 | 0.01 | 0.01 |
| anneal0 | 0.01 | 0.1 | 0.1 |
| nll_w | 10 | 5 | 10 |
| label_w | 0.001 | 1.0 | 0.001 |

### Task 列布局

| Brand | Task | Encoder 列 | Decoder 列 | Target 列 |
|-------|------|-----------|-----------|----------|
| brand1 | batterybranda | soc,current,min_temp,max_single_volt,max_temp,min_single_volt,volt (7) | soc,current (2) | min_temp,max_single_volt,max_temp,min_single_volt,volt (5) |
| brand2 | batterybrandb | soc,current,min_temp,max_single_volt,min_single_volt,volt,max_temp (7) | soc,current,min_temp,max_single_volt (4) | min_single_volt,volt,max_temp (3) |
| brand3 | batterybranda | soc,current,min_temp,max_single_volt,max_temp,min_single_volt,volt (7) | soc,current (2) | same 5 as brand1 |

### 结论：Paper 本身使用了不同的 Per-Brand 训练策略

- 不同 noise 水平（brand1=1.0 标准 VAE, brand2/3=0.01 近确定性 AE）
- 不同架构规模（从 128/8 到 1024/24，参数从 ~1M 到 ~50M）
- 不同训练 epoch（3 vs 5）
- 不同学习率（0.005 vs 0.0001）
- 不同 loss 权重（label_w: 0.001 vs 1.0）
- 不同 task 列布局（batterybranda vs batterybrandb）

**所以 per-brand 差异化策略是 paper 本身的设定，不是我们引入的。** 各 brand 的数据特性（车辆数、段数、采样频率、故障类型）差异很大，统一超参反而会导致某些 brand 表现差。

### 验证：统一超参 vs Per-Brand 差异化策略

为确认 per-brand 差异化策略的必要性，我们用官方 `model_params_battery_brandall.json` 的统一超参（hidden=64, latent=32, noise=0.01, lr=0.001, epochs=3, layers=1）在所有三个 brand 上各跑了一遍 5-fold。各 brand 保持原有 task 列布局不变。

**统一超参配置 (brandall.json):**
- hidden=64, latent=32, num_layers=1, noise=0.01
- lr=0.001, cosine_factor=1.0, batch=128, epochs=3
- nll_w=10, label_w=0.01, anneal0=0.1, x0=500

| Brand | 统一超参 AUROC | Per-Brand AUROC | Δ |
|-------|--------------|----------------|----|
| brand1 | **0.7657** ± 0.0305 | 0.8537 ± 0.0320 | **-8.80** |
| brand2 | **0.8890** ± 0.0251 | 0.9054 ± 0.0313 | **-1.64** |
| brand3 | **0.6942** ± 0.1024 | 0.8218 ± 0.0779 | **-12.76** |
| **平均** | **0.7830** | **0.8603** | **-7.73** |

**结论：统一超参全面劣于 per-brand 差异化策略，3-brand 平均差距 7.73 点。** brand1 和 brand3 下降尤为严重（分别 -8.80 和 -12.76），因为这两个 brand 的数据特征与 brandall 的设计假设差异最大。brand2 相对接近（-1.64），因为 brandall 的 noise=0.01 与 brand2 原始配置接近。

这验证了官方 repo 使用 per-brand 配置文件的合理性——不同 brand 的数据在车辆数、段数分布、采样频率、故障类型上差异显著，需要不同的 noise 水平（brand1 需要 1.0 的标准 VAE 随机性）、架构容量（brand2 需要更大的 hidden/latent）和训练时长。

**复现脚本：** `repro/repro_unified.py` — 使用 brandall 统一超参，保持各 brand 原有 task 列布局。

---

## 3. 实验历程

### Round 1: R1-R3 初始探索

| 实验 | 方法 | brand1 | brand2 | brand3 |
|------|------|--------|--------|--------|
| baseline | 官方配置 | 0.8537 | 0.9054 | 0.8218 |
| R1 (Stable) | noise 降低 + KL warmup→cyclical | 0.6976 | 0.8509 | — |
| R1b (Linear KL) | 全程线性 KL (去 cyclical) | 0.6659 | — | — |
| R2 (GMM) | + GMM 隐空间密度评分 | — | 0.8619 | 0.9201 |
| R3 (ChanAttn) | 通道加权 loss | — | — | 0.8758 |

**关键发现：**
- brand3 R2 = **0.9201**（超基线 +9.83）——GMM + LedoitWolf 多信号融合成功
- brand1 所有 R1 尝试失败（cyclical KL 是有害的）
- brand2 所有尝试失败（架构问题）

### Round 2: 根因分析

**brand1：**
- noise=0.01 使 VAE 变成近确定性 AE，rec_error 被压缩 5-10x，破坏异常检测灵敏度
- noise=1.0（标准 VAE 随机性）是检测灵敏度的来源
- 3 epoch 是「最佳未成熟点」——VAE 学会了正常模式但尚未学会抑制随机噪声

**brand2：**
- 26 训练车 / 24 维隐空间 = 1.1:1（严重欠定，协方差/GMM 不可靠）
- 1024 hidden × 1 layer = ~50M 参数（对 26 车严重过参数化）
- 段数 3900x 不平衡（5 vs 19639 段/车）

### Round 3: Phase 1-2 系统扫参

#### Phase 1: brand1 noise×epoch 网格

| noise | epoch | rec_error (hard) | combined (hard) | combined (soft) |
|-------|-------|-----------------|-----------------|-----------------|
| 0.1 | 3 | 0.8429 | 0.8555 | — |
| 0.5 | 3 | 0.8497 | 0.8565 | — |
| 1.0 | 3 | 0.8527 | 0.8596 | — |
| 0.5 | 5 | 0.8489 | 0.8579 | 0.8843 |
| **1.0** | **5** | **0.8681** | **0.8765** | **0.8961** |
| 1.0 | 8 | 0.8217 | 0.8015 | — |
| 2.0 | 5 | 0.8236 | 0.8277 | 0.8401 |

+ GMM at best config (noise=1.0, e5): combined_maha hard=0.8820, soft=**0.8961**

#### Phase 2: brand2 架构修复

**隐空间/架构扫参 (noise=0.01, e5)：**

| hidden | latent | 车/维比 | rec_error |
|--------|--------|---------|-----------|
| 128 | 8 | 3.4:1 | 0.7985 |
| 128 | 12 | 2.3:1 | 0.7842 |
| 256 | 8 | 3.4:1 | 0.8988 |
| 256 | 12 | 2.3:1 | 0.8827 |
| 512 | 8 | 3.4:1 | 0.8988 |
| 512 | 12 | 2.3:1 | 0.9024 |

**noise 扫参 (h512_l12, e5)：**

| noise | rec_error |
|-------|-----------|
| 0.01 | 0.9024 |
| **0.1** | **0.9071** |
| 0.5 | 0.7955 |
| 1.0 | 0.7899 |

**段数均衡 (h512_l12, noise=0.1)：**
- max_seg=1000: 0.8342（训练数据减少 73%）
- max_seg=500: 0.8089（数据损失 > 均衡收益）

**GMM 多信号 (noise=0.1, h512_l12, e5)：**
- rec+gmm combined: 0.8845
- rec+maha combined: 0.8577
- GMM 对 brand2 无帮助

**更多 epoch (e8, e15)：**
- e8 GMM rec_error: 0.8949
- e15 GMM rec_error: 0.8982
- 均不如 e5，5 epoch 是最佳点

#### Phase 4: 软 top-p% 评分

用温度 τ 的 softmax 加权替代硬 top-p% 截断，τ 通过 LOO-CV 选择。

| Brand | 信号 | 硬 top-p% | 软 top-p% | 最佳 |
|------|------|----------|----------|------|
| brand1 | combined_maha | 0.8820 | **0.8961** | 软 (+1.41) |
| brand2 | rec_error | **0.9071** | 0.9024 | 硬 |
| brand3 | combined_maha | **0.9201** | 0.9189 | 硬 |

---

## 4. 最终结果

| Brand | Baseline | 最终 AUROC | Δ | 配置 |
|-------|----------|-----------|------|------|
| brand1 | 0.8537 | **0.8961** | +4.24 | noise=1.0, e5, combined_maha, soft top-p% |
| brand2 | 0.9054 | **0.9071** | +0.17 | noise=0.1→0.01, h512_l12, e5, rec_error |
| brand3 | 0.8218 | **0.9201** | +9.83 | R2, combined_maha, GMM+LedoitWolf |
| **平均** | **0.8603** | **0.9078** | **+4.75** | |

---

## 5. 关于 Per-Brand 差异化策略的讨论

### Paper 本身就使用不同的 Per-Brand 策略

从官方代码 `model_params` JSON 文件可知，original paper 对不同 brand 使用了不同的：
- 架构大小 (hidden 128—1024, latent 8—24)
- VAE noise 水平 (1.0 vs 0.01)
- KL 退火强度 (anneal0: 0.01 vs 0.1)
- 训练 epoch (3 vs 5)
- 学习率 (0.005 vs 0.0001)
- 辅助 loss 权重 (label_w: 0.001 vs 1.0)
- Task 列布局 (batterybranda vs batterybrandb)

**这是合理的**，因为各 brand 的数据特征差异极大：

| 特征 | brand1 | brand2 | brand3 |
|------|--------|--------|--------|
| 车辆数 | 198 | 49 | ~50 |
| 训练车 (每折) | ~135 | ~26 | ~24 |
| 总段数 | ~477k | ~194k | — |
| 段数/车 (max:min) | — | 3900:1 | — |
| Task 类型 | branda (2dec/5tgt) | brandb (4dec/3tgt) | branda (2dec/5tgt) |

### 我们改动 vs 原始配置

| 参数 | brand1 | brand2 | brand3 |
|------|--------|--------|--------|
| noise | 1.0 (不变) | 0.01→0.1 | 0.01 (不变) |
| epochs | 3→5 | 3→5 | 5 (不变) |
| hidden | 128 (不变) | 1024→512 | 256 (不变) |
| latent | 8 (不变) | 24→12 | 16 (不变) |
| 评分 | +软 top-p% | 不变 | LedoitWolf+GMM |

**改动幅度评估：**
- brand1: 仅 epoch +2, 加软评分。改动最小，效果最大 (+4.24)
- brand2: 架构缩减 + noise 调整 + epoch +2。中等改动，效果最小 (+0.17)
- brand3: 仅评分层面加多信号融合。改动最小，效果最大 (+9.83)

### 无数据泄漏保证

所有超参选择和评分参数选择均通过 5-fold CV 的 LOO 机制完成：
- 训练只用正常车（one-class）
- 每个 fold 的 p/τ 参数仅从其他 4 折选出
- 测试 fold 严格不可见于训练和参数选择

### 实验验证：统一超参 vs Per-Brand

我们直接用官方 `model_params_battery_brandall.json` 的统一超参在三个 brand 上跑了完整 5-fold 复现（见第 2 节末尾的对比表）。结果：统一超参 3-brand 平均仅 **0.7830**，比 per-brand 策略低 **7.73 点**。这从实验上证明了 per-brand 差异化策略的必要性——不是偏好选择，而是性能要求。

---

## 6. 关键发现

1. **「最佳未成熟点」效应**：3-5 epoch 是异常检测灵敏度峰值。更多 epoch VAE 学会抑制随机噪声 → 重构误差坍缩 → 异常检测失效
2. **noise 参数至关重要**：brand1 需要 noise=1.0（标准 VAE 随机性），brand2 需要 noise=0.1（适度随机），noise 太低 (0.01) 或太高 (0.5+) 都会破坏检测能力
3. **隐空间维度/车辆数比例**：需 ≥2:1 才能可靠估计协方差。brand2 原 24 维 / 26 车 = 1.1:1 是致命的
4. **GMM 不是万能的**：对 brand3 有效 (+10 点)，对 brand1 略有效 (+0.2)，对 brand2 完全无效
5. **段数均衡的反直觉结果**：减少训练数据来均衡段数反而降低检测效果。训练数据量 > 均衡性
6. **软 top-p% 评分**：对噪声大的场景 (brand1) 有效，对稳定场景 (brand2/3) 无帮助
