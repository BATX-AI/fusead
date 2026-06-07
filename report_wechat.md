# FuseAD：让锂电池故障检测更敏锐 —— 从复现 Nature 顶刊到提出我们自己的方法

**BATX-Agent 团队**

---

## 缘起：一篇 Nature 论文引发的思考

2023 年，清华大学欧阳明高院士团队在 *Nature Communications* 上发表了一项引人注目的工作 —— DyAD（Dynamic Autoencoder for Anomaly Detection）[1]。他们用动态变分自编码器（VAE）对电动汽车充电片段的系统动态建模，在 **347 辆真实车辆**上实现了当时最优的无监督电池故障检测。

这个问题的现实意义非常直接：**电动汽车起火事故中，相当比例源于电池故障的漏检。** 如果能在充电数据中发现早期异常信号，就能提前预警，避免灾难性后果。

但这件事的难点也很直观 —— 我们只有**正常车辆**的数据可以训练模型，却要检测**从未见过的故障模式**。这是典型的 one-class 无监督异常检测。

BATX-Agent 团队在复现 DyAD 的过程中，发现了一个关键瓶颈。

---

## 问题：一条信号不够用

DyAD 的做法很优雅：训练 VAE 学习「正常充电片段是什么样的」，然后用**重构误差**（预测输出和真实输出的差异）作为异常分数。

但团队在复现中发现：**VAE 的隐空间（latent space）里储存了远比重构误差更丰富的分布信息，而 DyAD 完全没有利用。**

举个类比：医生判断一个人是否生病，不会只看「体温」一个指标，而是综合体温、血常规、影像等多个信号。DyAD 相当于只看体温。

这促使我们思考：能不能在 VAE 隐空间里挖掘更多信号，融合起来做判断？

---

## 我们的工作：FuseAD

BATX-Agent 团队提出了 **FuseAD（Fusion Anomaly Detection）**，在 DyAD 基础上做了三件事。

### 第一件事：多信号融合

从 VAE 隐空间同时提取三道信号，z-score 归一化后相加：

| 信号 | 含义 | 捕捉什么 |
|------|------|---------|
| **重构误差** | 预测偏差 ‖ŷ − y‖² | 输出层异常 |
| **马氏距离** | 隐空间偏离 (z−μ)ᵀΣ⁻¹(z−μ) | 分布偏移 |
| **GMM 密度** | 多模态对数概率 −log P(z) | 密度异常 |

> 三道信号彼此互补。重构误差发现输出不对，马氏距离发现位置偏了，GMM 密度发现模式罕见。融合后，整个 VAE 的所有信息都被调动起来。

![FuseAD Pipeline](figures/fig2_pipeline.png)

### 第二件事：发现「最佳未成熟点」

这可能是整个项目中最反直觉的发现。

VAE 有一个 `noise_scale` 参数控制采样的随机性。团队在扫参时发现：

- **noise = 1.0**（标准 VAE）：随机噪声让模型保持「敏感」→ 检测效果好
- **noise = 0.01**（近乎确定性 AE）：模型学会精确重构 → 连异常也能重构好 → **检测失效**

更令人意外的是训练轮数：**训练 3-5 个 epoch 就停，比训练 15 个 epoch 好得多。** 我们称之为「最佳未成熟点」效应 —— VAE 学会了正常模式，但还没来得及学会抑制噪声，此时检测最敏锐。

![Noise Sensitivity](figures/fig4_noise.png)

### 第三件事：软 Top-p% 鲁棒评分

原始 DyAD 用车级 top-p% 评分：每辆车取误差最大的前 p% 片段做平均。这是硬截断。

我们改用**温度加权的 softmax**：用温度参数 τ 控制权重分布（τ → 0 趋近硬截断，τ → ∞ 趋近全局均值）。τ 通过留一交叉验证自动选取，无需人工设定。

---

## 实验：严格无监督，三次数据集

团队严格遵循原始论文的 one-class 协议：

> **只用正常车训练，全部 55 辆故障车只在测试时出现。** 5-fold 交叉验证，seed=0。所有超参（包括 top-p% 的 p 值和 τ 值）通过留一折交叉验证选择，**训练/参数选择/测试三阶段严格隔离，零数据泄漏。**

实验覆盖三个电池品牌数据集（来自 DyAD 原始论文）：

| 数据集 | 车辆数 | 充电片段数 | 故障车数 |
|--------|-------|-----------|---------|
| A（品牌 1） | 198 | ~477K | 30 |
| B（品牌 2） | 49 | ~194K | 16 |
| C（品牌 3） | 100 | ~30K | 9 |

---

## 结果：全面超越 DyAD

### 主对比

![Main Comparison](figures/fig1_main_comparison.png)

FuseAD 在三个数据集上**全部超越** DyAD 基线。尤其 Dataset C，AUROC 从 0.8218 跃升至 **0.9201**，提升近 10 个百分点。三数据集平均从 0.8603 提升至 **0.9078**。

### 每个组件都有贡献

![Ablation Study](figures/fig3_ablation.png)

| 移除哪个组件 | 影响 |
|-----------|------|
| 软 top-p% 换回硬阈值 | Dataset A 下降 2.69 点 |
| 去掉 GMM 密度信号 | Dataset C 下降 14 点 |
| 去掉全部融合（仅重构误差） | 全部数据集显著下降 |

> GMM 密度对 Dataset C 几乎「不可或缺」。不同数据集的异常模式需要不同的信号组合来捕获 —— 这正是多信号融合的价值。

### 15 个 Fold 逐一对比

![Per-Fold](figures/fig5_perfold.png)

3 个数据集 × 5 折 = 15 个独立测试场景。FuseAD 在其中 **12 个**超越 DyAD，Fold 间方差更小。

---

## 为什么这些发现重要？

1. **多信号融合是「免费午餐」** —— 不改模型结构，只改评分方式，就能拿到 3.5+ AUROC 提升。这套思路可以直接迁移到其他 VAE-based 异常检测任务。

2. **「最佳未成熟点」挑战了常规认知** —— 不是训练越久越好，VAE 的随机性是检测灵敏度的来源。这对所有 VAE-based 异常检测方法都有启示。

3. **开源可复现** —— 完整代码和数据链接已在 GitHub 公开，所有实验可从零复现。

---

## 代码与数据

> **GitHub: [https://github.com/BATX-AI/fusead](https://github.com/BATX-AI/fusead)**

```bash
git clone https://github.com/BATX-AI/fusead.git
cd fusead
python3 repro/prep.py brand1              # 数据预处理
python3 repro/repro.py brand1 --fold 0    # 复现 DyAD 基线
python3 repro/method_p1_sweep.py brand1 --noise 1.0 --epochs 5 --fold 0  # 运行 FuseAD
python3 repro/score_auroc.py --glob 'repro/out/xxx/seg_fold{f}.csv' --col rec_error
```

数据来自 DyAD 原论文公开下载链接（详见 repo README）。

---

## 参考文献

[1] Zhang, J., Wang, Y., Jiang, B. et al. **Realistic fault detection of li-ion battery via dynamical deep learning.** *Nature Communications* 14, 5940 (2023). DOI: [10.1038/s41467-023-41226-5](https://doi.org/10.1038/s41467-023-41226-5)

[2] DyAD 官方代码: [https://github.com/962086838/Battery_fault_detection_NC_github](https://github.com/962086838/Battery_fault_detection_NC_github)

---

> **BATX-Agent 团队** · FuseAD: Fusion Anomaly Detection
>
> 站在 DyAD 的肩膀上，用多信号融合让电池故障无处遁形。
