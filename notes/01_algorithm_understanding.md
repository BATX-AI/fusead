# DyAD 算法深度拆解

> 论文: *Realistic fault detection of li-ion battery via dynamical deep learning*, Nature Communications 14:5940 (2023).
> 本文档基于 **PDF 正文 + 官方源码** (`code/DyAD/`) 双向核对而成，标注了论文与代码不一致之处。

---

## 1. 问题设定 (one-class / 弱监督异常检测)

- **输入**: 电动车 BMS 上报的"充电片段 (charging snippet)"。每个片段是一段定长时间序列 (源码里 `seq_len = 128`)，每个时间步含若干通道：
  `soc, current, volt, max_single_volt, min_single_volt, max_temp, min_temp`（不同 brand 列略有差异）。
- **标签粒度**: **车级 (vehicle-level)**，不是片段级。元数据 `label=='00'` 表示正常车 (ind)，否则为故障车 (ood)。故障是工程师根据"析锂/续航骤降/温度异常/电压异常"逐例确认的，**无法用规则表达**。
- **数据规模**: 三个厂商 (化名 Dahu/Socea/Naobop ≈ brand1/2/3)
  | brand | 片段数 | 正常车 | 故障车 |
  |---|---|---|---|
  | 1 | 476,739 | 168 | 30 |
  | 2 | 194,245 | 33 | 16 |
  | 3 | 29,598 | 91 | 9 |
  | 合计 | ≈700,582 | **292** | **55** |
- **训练/测试划分** (`dataset.py` + `five_fold_train_test_split.ipynb`): 对**正常车**做 5 折；每折训练集 = 4/5 正常车，测试集 = 1/5 正常车 + **全部故障车**。
  → 关键: **训练只见正常车** → 这是**单类异常检测**，故障样本极少且只在测试期出现。这也是论文反复强调"limited anomaly samples"的根因。

---

## 2. 核心思想: 动态自编码器 (Dynamical AutoEncoder)

普通 AE/VAE 把整条数据 `x` 压成 `z` 再重建 `x`，对"罕见但正常"的充电模式会误报。DyAD 换了个**因果/动力系统**视角：

> 把电池视为一个随机动力系统：**系统响应 = f(系统输入, 系统参数)**。
> - 系统输入 (System Input) `x_in` = 司机/充电桩能控制的量：`soc, current`
> - 系统响应 (System Response) `x_out` = 电池自身决定的量：`volt, 温度, 单体电压` 等
> - 系统参数 `z` = 表征这块电池"健康状态"的隐变量

**编码器**看到全部通道 → 推断系统参数 `z`；**解码器**只拿"系统输入 + z"去**预测系统响应**。
异常分数 = 响应的**重建/预测误差**。直觉：健康电池在给定 (soc,current) 下电压/温度的响应是可预测的；故障电池的响应偏离 → 误差大。这比"重建整条曲线"更聚焦故障机理，且对罕见但正常的充电模式鲁棒。

### 列的角色 (`model/tasks.py`, brand1 的 `EvTask`)
```
encoder 输入 (6 通道): [soc, current, max_temp, max_single_volt, min_single_volt, volt]
decoder 输入 (前2):    [soc, current]            ← 系统输入
target/预测目标 (后4): [max_temp, max_single_volt, min_single_volt, volt]  ← 系统响应
```
(brand2: enc7/dec4/out3；brand3: enc7/dec2/out5，列顺序见 `tasks.py`)

---

## 3. 网络结构 (`model/dynamic_vae.py`)

```
input_sequence ──encoder_filter──► [B,128,6] ──GRU(enc)──► hidden
   hidden ─► hidden2mean ─► mean (μ, 32维)
   hidden ─► hidden2log_v ─► log_v
   z = μ + ε·σ·noise_scale   (训练; ε~N(0,I), noise_scale=0.01)
   z = μ                     (推理)
   mean ─► mean2latent(MLP 32→32→1) ─► mean_pred   # 里程预测头(弱监督)
   z ─► latent2hidden ─► 作为 decoder 初始隐状态
input_sequence ──decoder_filter──► [B,128,2] ──GRU(dec, init=z)──► outputs
   outputs ─► outputs2embedding(Linear) ─► log_p (预测的系统响应, [B,128,4])
```
- RNN: **GRU**, `hidden_size=64`, `num_layers=1`, **bidirectional=True**, `latent_size=32`。
- 注意论文正文写 encoder/decoder 是 "GCN/GRU"，**实际源码是 GRU**（`rnn_type="gru"`），且是双向。论文 Methods 写重建用 **MSE**，**源码实际用 `SmoothL1Loss`**（见下）。这两处是论文-代码差异，复现时以代码为准。

---

## 4. 损失函数 (`train.py: loss_fn`)

总损失（每个 batch）：
```
loss = nll_weight · L_recon  +  latent_label_weight · L_mileage  +  kl_weight(t) · L_KL / B
```
1. **L_recon (重建/预测损失)**: `SmoothL1Loss(log_p, target)`，只对**系统响应通道**算。`nll_weight = 10`。
2. **L_mileage (里程弱监督)**: `MSE(mean_pred, 归一化里程)`。里程用前 50 个样本的 min/max 归一化到 [0,1]。`latent_label_weight = 0.01`。
   - 作用: 强迫隐空间 `μ` 保留"行驶里程/老化程度"信息 → 论文消融显示能提升 AUROC。
3. **L_KL (变分正则)**: 标准 VAE KL `-0.5·Σ(1+log_v-μ²-exp(log_v))`，**线性退火** `kl_weight = anneal0·min(1, step/x0)`，`anneal0=0.1, x0=500`。
- 优化器 **AdamW** (lr=1e-3, wd=1e-6) + **CosineAnnealingLR**；`batch=64`，**仅 10 epochs**。

---

## 5. 鲁棒评分: 片段误差 → 车级分数 (`evaluate.py` + 论文式(5))

1. 训练后对每个片段算重建 MSE 误差 `rec_error`（`extract()` → `save_features_info`）。
2. **车级聚合 (robust scoring)**: 对一辆车的所有片段误差，
   - 阈值 `τ`：先把误差按 `τ` 截断/筛选；
   - 取**前 p 百分位 (top-p%)** 的误差求平均 → 该车的 `Vehicle Error`；
   - `Vehicle Error > τ` 判定为故障车。
   `τ, p` 在训练集上调。直觉：一辆故障车不是每个片段都异常，用 top-p% 抓"最像故障"的片段，比用均值鲁棒（抗个别噪声片段、抗正常车偶发高误差）。
3. **指标**: 车级 **AUROC**（扫阈值得 TPR/FPR 曲线）。

---

## 6. 论文报告的基准成绩 (Table 1, 待超越的靶子)

| 算法 | AUROC (%) | 平均直接成本 (10⁴ CNY/车/年) |
|---|---|---|
| **DyAD (本文)** | **88.6 ± 2.9** | **0.085** |
| GDN | 70.3 ± 5.5 | 0.126 |
| AE (普通自编码器) | 72.8 ± 13.4 | 0.133 |
| SVDD | 51.5 ± 8.3 | 0.152 |
| GP (高斯过程) | 66.6 | 0.162 |
| VE (变分评估, 非深度) | 55.6 | 0.169 |

- 经济成本模型 (式(6)): `cost = p·[(1-q_TP)·c_f] + [p·q_TP + (1-p)·q_FP]·c_t`
  其中 `p`=故障率(0.038%~0.075%)，`c_f`=单车故障成本(100~500万 CNY)，`c_t`=单车检测/检修成本(0.8~5.5万 CNY)，`q_TP/q_FP`=真/假阳率。
- **关键洞察**: 因为故障率 `p` 极低 (~万分之几)，**FPR 比 TPR 对总成本影响更大** → 优化目标不只是 AUROC，而是 ROC 上**经济最优工作点**。这是论文"social & financial factors"卖点的本质。

---

## 7. DyAD 的可攻击弱点 (后续优化的切入口)

| # | 弱点 | 说明 |
|---|---|---|
| W1 | **每个 brand 单独训练** | 没有跨厂商迁移/统一模型；小样本 brand (brand3 仅 9 故障车) 易过拟合。 |
| W2 | **GRU + 仅 10 epochs + latent 32** | 容量/训练都偏小；时序建模可被更强结构 (TCN/Transformer/SSM/Mamba) 取代。 |
| W3 | **里程归一化用前 50 样本 min/max** | 统计脆弱、易受批次顺序影响，是明显工程瑕疵。 |
| W4 | **`τ, p` 两个超参靠训练集网格调** | 评分规则不可学、对分布漂移敏感；可用可学习/分布式评分替代。 |
| W5 | **重建误差是唯一异常信号** | 没用上隐空间密度、预测不确定性、物理残差等互补信号。 |
| W6 | **未利用故障车的弱标签做对比/度量学习** | 纯单类，55 辆故障车的信息基本浪费。 |
| W7 | **片段独立、丢弃时间顺序** | 同一车多个片段间的演化趋势(老化轨迹)没建模。 |
| W8 | **无物理先验** | 纯数据驱动，没融入等效电路/电化学/产热方程等强先验。 |
| W9 | **不确定性/标定缺失** | 输出无置信度，难支撑"经济最优阈值"的稳健决策。 |

→ 这 9 个弱点逐一对应 `02_optimization_directions.md` 的优化方向。
