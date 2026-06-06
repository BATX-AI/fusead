"""
车级鲁棒评分 + 5 折 AUROC (忠实复刻官方 charge_to_car 的 top-p% 聚合)。
车级分数 = 该车 rec_error 最高的 top-p% 片段的均值。
p 的选取: 留一折交叉 (对 fold i, 在其余折上选使平均 AUROC 最大的 p, 再用到 fold i) —— 无测试泄漏。
用法: python repro/score_auroc.py [--glob 'repro/scores/seg_fold{f}.csv']
"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

P_GRID = [h/1000 for h in range(50, 1000, 50)]   # 官方网格: 5%..95%

COL = 'rec_error'

def vehicle_scores(df, p):
    rows = []
    for car, g in df.groupby('car'):
        e = np.sort(g[COL].values.astype(float))[::-1]   # 降序
        k = max(round(p*len(e)), 1)
        rows.append((int(car), int(g['label'].iloc[0]), float(e[:k].mean())))
    return pd.DataFrame(rows, columns=['car','label','score'])

def fold_auroc(df, p):
    vs = vehicle_scores(df, p)
    if vs['label'].nunique() < 2: return np.nan
    return roc_auc_score(vs['label'], vs['score'])

def main(pattern, nfold=5):
    print(f"[评分列 = {COL}]")
    folds = []
    for f in range(nfold):
        try: folds.append(pd.read_csv(pattern.format(f=f)))
        except FileNotFoundError: folds.append(None)
    avail = [f for f in range(nfold) if folds[f] is not None]
    print(f"可用折: {avail}")
    # 每折 × 每个 p 的 AUROC
    A = {f: {p: fold_auroc(folds[f], p) for p in P_GRID} for f in avail}
    # AUROC-vs-p (各折平均) 便于看敏感性
    print("\nAUROC vs p (各折均值):")
    for p in P_GRID:
        vals = [A[f][p] for f in avail if not np.isnan(A[f][p])]
        print(f"  p={p:.2f}: {np.mean(vals):.4f}")
    # 留一折交叉选 p
    chosen, aurocs = [], []
    for f in avail:
        others = [o for o in avail if o != f]
        best_p = max(P_GRID, key=lambda p: np.nanmean([A[o][p] for o in others]) if others else A[f][p])
        a = A[f][best_p]; chosen.append((f, best_p, a)); aurocs.append(a)
        print(f"fold{f}: 选 p={best_p:.2f} -> AUROC={a:.4f}")
    print(f"\n>>> 5折 AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}  (留一折选 p, 无泄漏)")
    # 同时给"全局最优 p"上界 (作参考, 略乐观)
    best_global = max(P_GRID, key=lambda p: np.nanmean([A[f][p] for f in avail]))
    ub = [A[f][best_global] for f in avail]
    print(f"    参考(固定全局最优 p={best_global:.2f}): {np.mean(ub):.4f} ± {np.std(ub):.4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='repro/scores/seg_fold{f}.csv')
    ap.add_argument('--col', default='rec_error')
    a = ap.parse_args()
    COL = a.col
    main(a.glob)
