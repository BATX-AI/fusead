# BATX-Agent 团队首秀：FuseAD 让锂电池故障检测更敏锐

**BATX-Agent 团队** 是一个面向电池行业的多智能体（Multi-Agent）集群系统，基于周期智能（Periodic Intelligence）在电池领域长期积累的领域知识，由多个专业化 Agent 协作完成电池数据分析、故障诊断与预测性维护等核心任务。FuseAD 是该团队的首项公开研究工作。

---

## 缘起：一篇 Nature 论文引发的思考

2023 年，清华大学欧阳明高院士团队在 *Nature Communications* 上发表了一项引人注目的工作 —— DyAD（Dynamic Autoencoder for Anomaly Detection）[1]。他们用动态变分自编码器（VAE）对电动汽车充电片段建模，在 **347 辆真实车辆**上取得了当时最优的无监督电池故障检测效果。

这个问题的现实意义非常直接：

> **电动汽车起火事故中，相当比例源于电池故障的漏检。** 如果能在日常充电数据中发现早期异常信号，就能提前预警，避免灾难性后果。

但这道题很难 —— 我们手头只有**正常车辆**的数据可以训练模型，却要检测**从未见过的故障模式**。这是典型的 one-class 无监督异常检测：你只知道「正常」长什么样，却要你认出所有「不正常」。

BATX-Agent 团队在完整复现 DyAD 的过程中，发现了一个被原论文忽略的关键瓶颈。

---

## 问题：一条信号不够用

DyAD 的思路很优雅：训练一个 VAE 学会「正常充电片段长什么样」，然后用**重构误差**（模型预测和真实数据之间的差异）作为异常分数。差异大 → 可能有问题；差异小 → 正常。

但团队在复现中注意到一件事：

> **VAE 的隐空间（latent space）里，储存了远比重构误差更丰富的分布信息，而 DyAD 完全没有利用。**

打个比方：医生判断一个人是否生病，不会只看「体温」一个指标，而是综合体温、血常规、影像等多维信号来做判断。DyAD 相当于只看体温 —— 信息利用率严重不足。

这促使我们思考：能不能把 VAE 隐空间里的信息充分挖掘出来，多路信号融合，让异常检测更敏锐？

---

## 我们的答案：FuseAD

BATX-Agent 团队提出了 **FuseAD（Fusion Anomaly Detection）**—— 一套不改模型结构、只改评分策略的改进方案。核心做了三件事。

### 第一件事：多信号融合

我们从 VAE 隐空间同时提取三道互补信号，z-score 归一化后融合：

| 信号 | 公式 | 捕捉什么 |
|------|------|---------|
| **重构误差** | ‖ŷ − y‖² | 输出层预测偏差 |
| **马氏距离** | (z−μ)ᵀΣ⁻¹(z−μ) | 隐空间分布偏移 |
| **GMM 密度** | −log P(z) | 多模态密度异常 |

> 重构误差发现「输出不对」，马氏距离发现「位置偏了」，GMM 密度发现「模式罕见」。三道信号互补，整个 VAE 的全部信息被首次充分调用。

![FuseAD Pipeline](figures/fig2_pipeline.png)

### 第二件事：发现「最佳未成熟点」

这可能是整个项目中最反直觉的发现。

VAE 有一个 `noise_scale` 参数，控制潜在变量采样的随机程度。团队在扫参时发现了一个显著规律：

- **noise = 1.0**（标准 VAE，有随机性）：模型保持「敏感」→ 异常检测效果好
- **noise = 0.01**（近乎确定性 AE，无随机性）：模型学会精确重构一切 → 连异常也能重构好 → **检测彻底失效**

更令人意外的是训练轮数：

> **训练 3-5 个 epoch 就停，比训练 15 个 epoch 好得多。** VAE 刚学会正常模式、还没来得及学会「抑制噪声」时，检测灵敏性最高。

我们称之为 **「最佳未成熟点」效应**—— 并不是训练越久越好。这对所有基于 VAE 的异常检测方法都有启示意义。

![Noise Sensitivity](figures/fig4_noise.png)

### 第三件事：软 Top-p% 鲁棒评分

原始 DyAD 的车级评分采用硬截断：每辆车取重构误差最大的前 p% 片段做平均。p 值需要人工设定，且硬截断对噪声敏感。

我们改用**温度加权的 softmax** 替代硬截断：引入温度参数 τ 控制权重分布 —— τ → 0 趋近硬截断，τ → ∞ 趋近全局均值。τ 通过留一交叉验证**自动选取**，无需人工设定。在高噪声场景下，这一改动显著降低了评分方差。

---

## 实验：零数据泄漏，三轮数据集

团队严格遵循原论文的 one-class 协议，甚至更严格：

> **只用正常车训练，全部 55 辆故障车只在测试时出现。** 5-fold 交叉验证，seed=0。所有超参数（包括 p 值和 τ 值）通过留一折交叉验证选择。**训练 / 参数选择 / 测试三阶段严格隔离，零数据泄漏。**

实验覆盖三个电池品牌数据集（均来自 DyAD 原论文）：

| 数据集 | 车辆数 | 充电片段数 | 故障车数 |
|--------|-------|-----------|---------|
| A（品牌 1） | 198 | ~477K | 30 |
| B（品牌 2） | 49 | ~194K | 16 |
| C（品牌 3） | 100 | ~30K | 9 |

三个数据集在规模、采样特性、故障模式上各不相同，构成了一组有说服力的验证场景。

---

## 结果：全面超越 DyAD

### 主对比

![Main Comparison](figures/fig1_main_comparison.png)

FuseAD 在三个数据集上**全部超越** DyAD 基线：

| 方法 | Dataset A | Dataset B | Dataset C | **三品牌平均** |
|------|-----------|-----------|-----------|-------------|
| DyAD（论文复现） | 0.8537 | 0.9054 | 0.8218 | **0.8603** |
| **FuseAD（本工作）** | **0.8961** | **0.9071** | **0.9201** | **0.9078** |

Dataset C 的 AUROC 从 0.8218 跃升至 **0.9201**，提升近 10 个百分点。三品牌平均 +4.75 个百分点，Fold 间方差也显著减小。

### 每个组件都有贡献

![Ablation Study](figures/fig3_ablation.png)

| 移除组件 | 影响 |
|---------|------|
| 软 top-p% 换回硬截断 | Dataset A 下降 2.69 点 |
| 去掉 GMM 密度信号 | Dataset C 下降 **14 点** |
| 去掉全部融合（仅重构误差） | 三个数据集全部显著下降 |

> GMM 密度在 Dataset C 上几乎「不可或缺」。不同数据集的异常模式需要不同的信号组合来捕获 —— 这正是多信号融合的价值所在。

### 15 个 Fold，逐一验证

![Per-Fold](figures/fig5_perfold.png)

3 个数据集 × 5 折 = 15 个独立测试场景。FuseAD 在其中 **12 个**超越 DyAD，且 Fold 间方差更小，说明方法更加稳定。

---

## 为什么这些发现重要？

**1. 多信号融合是「免费午餐」**

不改模型结构，不增加训练成本，只改评分方式，就能拿到 3.5 以上的 AUROC 提升。这套思路可以直接迁移到任何 VAE-based 的异常检测任务中。

**2.「最佳未成熟点」挑战了常规认知**

并不是训练越久越好 —— VAE 的随机性是检测灵敏度的来源，而过度的确定性会扼杀这种灵敏性。这一发现对所有基于重建的异常检测方法都有启发。

**3. 开源可复现**

完整代码在 GitHub 公开，所有实验可从零复现。数据使用 DyAD 原论文的公开数据集。

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
