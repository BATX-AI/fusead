# DyAD 复现并超越 · 执行交接计划 (HANDOFF)

> 论文: *Realistic fault detection of li-ion battery via dynamical deep learning*, Nature Communications 14:5940 (2023).
> 目标: 在**与论文完全一致的协议与数据划分**下, 先**忠实复现** DyAD (三-brand 平均车级 AUROC ≈ **88.6%**, 平均直接成本 ≈ **0.085 万元/车/年**), 再用更好的算法**超越**之。
> 适用机器: **有公网(无需代理) + 一块 GPU**。本计划自带可直接运行的脚本(已在开发机用 CPU 验证逻辑正确)。

---

## 0. 一页速览 (TL;DR)

```bash
# 0) 取代码与数据(无代理直连) + 装环境
bash download_direct.sh                      # 下载并解压 brand1/2/3 到 ./data
pip install torch numpy pandas scipy scikit-learn tqdm   # GPU 机器装对应 CUDA 版 torch

# 1) 预处理(建缓存+五折划分, 与官方逐字节一致)
for b in brand1 brand2 brand3; do python repro/prep.py $b; done

# 2) 忠实复现 DyAD (每 brand 5 折), 得到基线
for b in brand1 brand2 brand3; do for f in 0 1 2 3 4; do
  python repro/repro.py $b --fold $f --outdir repro/out/${b}_base; done; done
for b in brand1 brand2 brand3; do
  python repro/score_auroc.py --glob "repro/out/${b}_base/seg_fold{f}.csv" --col rec_error; done
# 三个 brand 的 5 折 AUROC 取平均 → 对齐论文 88.6%

# 3) 运行 DyAD++ (已验证的改进) 并对比
for b in brand1 brand2 brand3; do for f in 0 1 2 3 4; do
  python repro/improve.py $b --fold $f --outdir repro/out/${b}_pp; done; done
for b in brand1 brand2 brand3; do
  python repro/score_auroc.py --glob "repro/out/${b}_pp/seg_fold{f}.csv" --col combined; done
```
GPU 上单 brand 5 折训练通常几分钟。开发机已用 brand3 验证: 基线 **0.835 → DyAD++ 0.888** (+5.3 点, 方差 −40%, 逐折全胜)。

---

## 1. 已完成 & 已验证 (目标机无需重做)

| 事项 | 状态 |
|---|---|
| 论文 PDF / 官方源码 | ✅ `pdf/`, `code/` |
| 算法拆解(代码+论文双核对) | ✅ `notes/01_algorithm_understanding.md` (含论文-代码差异: 正文说GCN/MSE, 实为GRU/SmoothL1) |
| 优化方向(3类13个) | ✅ `notes/02_optimization_directions.md` |
| **可运行的忠实复现+改进代码** | ✅ `repro/`(见第 8 节文件清单), 已在 CPU 跑通全流程 |
| **brand3 实测: 复现 0.835 / DyAD++ 0.888** | ✅ `repro/RESULTS.md` |
| 数据 brand2/brand3 | ✅ 已下载(开发机); 目标机用 `download_direct.sh` 重下即可 |

**已踩过的坑(目标机注意)**:
1. **torch ≥ 2.6**: `torch.load` 默认 `weights_only=True`, 读 pkl(含 numpy) 会报错。`prep.py` 已在文件头 monkeypatch 成 `weights_only=False`。官方 `code/` 若直接跑需同样处理。
2. **官方 `code/DyAD/model/tasks.py` 的 `Label.loss` 硬编码 `.to("cuda")`**: GPU 机器上 OK; 纯 CPU 会崩。我的 `repro/` 已规避(自写 loss, `cuda if available else cpu`)。
3. **五折划分依赖的 `all_car_dict` 存的是绝对路径**: 官方仓库自带的 `YOUR_all_car_dict.npz.npy` 是占位, **必须用本地数据重新生成**(`prep.py` 已自动按本地路径构建, 且 seed=0 与官方逐字节一致, brand3 已核对 `ind_sorted`/`ood_sorted` 完全吻合)。
4. **CPU 上 hidden=1024(brand2 原配置)极慢**(>12min/epoch)。**GPU 上无此问题**, 可忠实使用论文原超参。

---

## 2. 目标机执行步骤(详)

### 2.1 取代码与数据(无代理)
- 代码: 直接 `git clone https://github.com/962086838/Battery_fault_detection_NC_github.git` (有公网即可)。或直接用本交接包里的 `code/` + `repro/`。
- 数据: `bash download_direct.sh` —— 直连 figshare(`ndownloader.figshare.com` → 302 → AWS S3), 满速下载并解压、校验大小+gzip。三个 file_id 与字节数已写死在脚本里(brand1=1191991306, brand2=236920600, brand3=55469605)。

### 2.2 环境
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按本机 CUDA 选 cu121/cu124/cu128
pip install numpy pandas scipy scikit-learn tqdm
```
代码无需改动即自动用 GPU(`repro/` 内 `device = cuda if available else cpu`)。

### 2.3 预处理(缓存 + 五折划分)
```bash
for b in brand1 brand2 brand3; do python repro/prep.py $b; done
```
产出 `repro/cache/{brand}.npz`(全片段张量+元数据) 与 `repro/cache/{brand}_split.npy`(seed=0 的 ind/ood 车号顺序)。

### 2.4 忠实复现 DyAD(对齐 88.6%)
两条路, 推荐 A:

**A. 用本交接包的 `repro/`(已验证、干净、brand 通用)**
```bash
for b in brand1 brand2 brand3; do for f in 0 1 2 3 4; do
  python repro/repro.py $b --fold $f --outdir repro/out/${b}_base; done; done
```
`repro.py` 复用**官方 `DynamicVAE` 模型** + 各 brand **官方超参**(见 `repro/brands.py`, 见第 3 节表) + **官方划分与损失**, 仅规避了 CPU/新版 torch 的兼容坑 → 与论文同口径。

**B. 用官方 `code/`(交叉验证用)**
- 先跑 `code/data/five_fold_train_test_split.ipynb` 生成 `code/five_fold_utils/{all_car_dict, ind_odd_dict{1,2,3}}.npz.npy`(用本地路径)。
- 改 `code/DyAD/model/dataset.py` 默认的 `ind_ood_car_dict_path` 指向对应 brand 的 dict; 处理上面第 1/2 个坑。
- 跑 `code/DyAD/main_five_fold.py --config_path model_params_battery_brandX.json --fold k`(train→extract→evaluate)。
- 车级 AUROC 用 `code/notebooks/dyad_eval_fivefold-threshold.ipynb`(注意该 notebook 有 brand4 路径/NameError 等 bug, 我已在 `repro/score_auroc.py` 写了干净等价实现)。

### 2.5 车级评分 + AUROC
```bash
for b in brand1 brand2 brand3; do
  python repro/score_auroc.py --glob "repro/out/${b}_base/seg_fold{f}.csv" --col rec_error; done
```
`score_auroc.py` 忠实复刻官方鲁棒评分: 每车取 **top-p%** 重建误差均值为车级分数, p 用**留一折交叉**选(无测试泄漏), 输出 5 折 AUROC 均值±std。**三个 brand 的均值** = 对齐论文 88.6% 的数字。

### 2.6 跑 DyAD++ 并对比
```bash
for b in brand1 brand2 brand3; do for f in 0 1 2 3 4; do
  python repro/improve.py $b --fold $f --outdir repro/out/${b}_pp; done; done
for b in brand1 brand2 brand3; do
  python repro/score_auroc.py --glob "repro/out/${b}_pp/seg_fold{f}.csv" --col combined; done
```

---

## 3. 复现关键参数(忠实, 来自官方 config)

| 项 | brand1 | brand2 | brand3 |
|---|---|---|---|
| task(列布局) | batterybranda | batterybrandb | batterybranda |
| epochs | 3 | 3 | 5 |
| hidden_size | 128 | **1024** | 256 |
| num_layers | 2 | 1 | 1 |
| latent_size | 8 | 24 | 16 |
| bidirectional | true | true | true |
| lr | 0.005 | 0.0001 | 0.005 |
| noise_scale | 1.0 | 0.01 | 0.01 |
| nll_weight | 10 | 5 | 10 |
| latent_label_weight | 0.001 | 1.0 | 0.001 |
| anneal0 (KL) | 0.01 | 0.1 | 0.1 |

- **列布局**(列名→实际 column.pkl 索引, 见 `repro/brands.py`):
  - batterybranda: encoder=[soc,current,min_temp,max_single_volt,max_temp,min_single_volt,volt], decoder输入=前2(soc,current), 预测目标=后5。
  - batterybrandb: encoder=[soc,current,min_temp,max_single_volt,min_single_volt,volt,max_temp], decoder输入=前4, 预测目标=后3。
- 公共: AdamW(wd=1e-6) + CosineAnnealingLR, batch=128, 重建损失=SmoothL1, KL 线性退火(x0=500), 里程弱监督=MSE。
- **划分**: 正常车(ind)按 seed=0 shuffle 后 5 折; 每折 train=4/5 正常车, test=1/5 正常车 + **全部故障车(ood)**; **训练只用正常车**(单类)。
- **数据规模**: brand1 476,739段/168正常+30故障; brand2 194,245段/33+16; brand3 29,598段/91+9。合计 292 正常 + 55 故障 = 347 车。

---

## 4. 超越路线 (全部协议合规: 只用正常车训练, 故障车只在测试)

> 锚点: 已验证 **多信号融合**在 brand3 上 0.835→0.888。下面按"已验证→快赢→高上限"推进, 每步只改一个变量, 用第 5 节协议统一评测 + bootstrap 显著性。

### Tier 1 — 已验证 / 立即可做 (预期三-brand 平均 +2~4 点)
- **T1.1 多信号异常分数(已验证)**: 重建误差 + **隐空间密度(对训练正常隐向量拟合的马氏距离)**, z-score 融合 = `combined`。已在 `repro/improve.py` 实现。
  - 实测(brand3,30ep): rec=0.828, maha=0.847, **combined=0.888**, 逐折 5/5 胜基线。
- **T1.2 cyclical-KL 退火 + robust 里程归一化**: 已在 `improve.py`。塑造更规整的隐空间, 使马氏距离更可分。
- **T1.3 充分训练**: 论文 epochs 偏少(3~5)。GPU 上拉到收敛(+early-stopping), 已见重建与隐空间信号在 30ep 才变互补。

### Tier 2 — 快赢 (预期再 +1~3 点 / 降方差)
- **T2.1 更强的隐空间密度**: 用 **GMM / normalizing flow** 替代单高斯马氏, 建更精细的正常隐分布, 取 NLL 为异常分数。
- **T2.2 可学习融合**: 现在是等权 z-score; 改成在留出正常车上拟合的轻量加权 / OOD 校准, 或可微 top-k 评分替代 (τ,p) 网格。
- **T2.3 通道标准化的重建误差**: 各响应通道(电压/温度量纲不同)分别标准化后再聚合, 提升重建信号质量。
- **T2.4 更强主干**: GRU → **TCN / Mamba(SSM)**, 长序列更稳更快, 保留"系统输入→响应"的动态自编码结构。

### Tier 3 — 高上限 / 差异化贡献
- **T3.1 跨厂商统一模型 + 域适应**: 共享主干 + brand 条件(FiLM) + CORAL/DANN, 用全部 70 万片段联合训练; 重点救小样本 brand3。**训练仍只用正常车**, 不破坏协议。
- **T3.2 条件生成式似然异常**: 条件扩散/flow 学 `p(响应|输入)`, 异常=负对数似然(信息量>MSE, 自带标定)。
- **T3.3 物理信息混合(PINN/灰盒)**: 解码器嵌入等效电路/产热方程, 输出可解释物理参数漂移(内阻↑/产热↑)+残差异常。
- **T3.4 成本敏感端到端**: 直接以论文式(6)期望成本/ROC 凸包为目标, 输出标定概率解析求最优阈值 → 直接打"省钱"这个论文最强指标。

### 跨数据集自监督(可选底座)
- 充电片段做掩码重建/对比(TS2Vec/TF-C)预训练, 再微调异常头, 尤其救 brand3。

---

## 5. 评测协议 (与论文一致, 必须统一)

1. **主指标**: 车级 5 折 AUROC(均值±std), 三-brand 再平均。用 `repro/score_auroc.py`(top-p% 鲁棒评分, p 留一折交叉选)。
2. **经济指标**(论文最看重, 建议补上): 期望直接成本
   `cost = p·(1−q_TP)·c_f + [p·q_TP+(1−p)·q_FP]·c_t`,
   扫 `p∈[0.038%,0.075%]`, `c_f∈[1M,5M]`, `c_t∈[8k,55k]`(万元), 给成本曲线与 ROC 最优工作点。低故障率下 **FPR 主导成本** → 控 FPR 比堆 TPR 更省钱。
3. **显著性**: 对每方法做 bootstrap(车级重采样)置信区间 + 逐折配对; 只认 CI 不重叠 / 逐折一致的提升。
4. **稳健性 & 差异化**: 跨 brand 泛化; (论文未量化的)**早期预警提前量**——故障车最早被判异常距确认故障的提前天数; 参数量/推理延迟。

---

## 6. 实验矩阵 · 里程碑 · 预期

| 里程碑 | 内容 | 判据 |
|---|---|---|
| **M0 复现** | `repro.py` 跑三 brand 5 折 | 三-brand 平均 AUROC 落在 88.6±2.9 附近 |
| **M1 多信号(已验证)** | `improve.py` combined 三 brand | ≥2/3 brand 的 bootstrap CI 高于基线 |
| **M2 Tier2** | flow密度 + 可学习融合 + 强主干 | 三-brand 平均再 +1~3 点, 方差收窄 |
| **M3 Tier3** | 跨厂商统一 / 扩散似然 / 物理 / 成本 | 至少一条线: AUROC 再升 或 成本更低 或 可解释/早期预警 |
| **M4 整合** | 最优组合 + 成本曲线 + 写作 | 三-brand 平均 AUROC 显著 >88.6%, 成本 <0.085 |

预期: T1 已验证可达单 brand +5 点(brand3)。三-brand 平均稳超 88.6% 的最稳路径 = **T1.1 多信号 + T1.3 充分训练**, 再叠加 T2.1(flow 密度) 与 T3.1(跨厂商降方差)。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 复现数字与 88.6 有偏差 | 以 `repro.py` 为主、官方 `code/` 交叉验证; 检查划分(seed=0)、列布局、超参逐项对齐 |
| 改进"涨点"靠运气/调参 | bootstrap CI + 逐折配对 + 多种子; 单变量消融归因(brand3 已示范: 30ep 下 rec/maha/combined 拆开看) |
| 小样本 brand 过拟合 | 协议禁止用故障车训练 → 改进必须是无监督/自监督(T1/T2/T3.1~3.3 均满足) |
| 经济成本对 p 极敏感 | 报成本曲线而非单点, 强调 FPR 控制 |
| GPU 显存/速度 | brand2 hidden=1024 在 GPU 上无压力; 必要时 batch 调整 |

---

## 8. 文件清单 (拷到目标机)

```
download_direct.sh              # 无代理直连下载三 brand
read_torch_pkl.py               # 免 torch 读 pkl(可选; prep.py 已用 torch)
repro/
  brands.py                     # 各 brand 数据布局/列/超参(忠实官方)
  prep.py                       # 建缓存+五折划分(seed=0, 与官方一致)
  repro.py                      # 忠实复现 DyAD (通用 brand)  —— 产 rec_error
  improve.py                    # DyAD++ (cyclicalKL+robust里程+多信号) —— 产 rec_error/maha/combined
  score_auroc.py                # 车级 top-p% 鲁棒评分 + 5折 AUROC (--col 选信号)
  RESULTS.md                    # 已验证结果记录(brand3)
code/                           # 官方源码(交叉验证)
notes/01_*.md, notes/02_*.md    # 算法拆解 / 优化方向
pdf/paper.pdf                   # 原文
```

执行顺序: `download_direct.sh` → `prep.py`(×3 brand) → `repro.py`(复现) → `score_auroc.py` → `improve.py`(超越) → `score_auroc.py --col combined` → 按 Tier 2/3 迭代。
```
