"""软 top-p% 评分: 用温度 τ 的 softmax 加权替代硬 top-p% 截断。
τ 通过留一折交叉验证选择（无泄漏）。
用法: python repro/score_auroc_soft.py [--glob 'path/seg_fold{f}.csv'] [--col rec_error]"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

TAU_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

COL = 'rec_error'

def vehicle_scores_soft(df, tau):
    """用 temperature-weighted softmax 替代硬 top-p%。
    tau→0: 趋向 hard argmax; tau→∞: 趋向 uniform mean."""
    rows = []
    for car, g in df.groupby('car'):
        e = np.sort(g[COL].values.astype(float))[::-1]  # 降序
        # Normalize to [0,1] for numerical stability
        e_range = e.max() - e.min()
        if e_range < 1e-8:
            score = float(e.mean())
        else:
            e_norm = (e - e.min()) / e_range
            w = np.exp(np.clip(e_norm / max(tau, 1e-8), -100, 100))
            w /= w.sum()
            if np.isnan(w).any():
                score = float(e.mean())
            else:
                score = float(np.sum(e * w))
        rows.append((int(car), int(g['label'].iloc[0]), score))
    return pd.DataFrame(rows, columns=['car', 'label', 'score'])

def fold_auroc_soft(df, tau):
    vs = vehicle_scores_soft(df, tau)
    if vs['label'].nunique() < 2:
        return np.nan
    return roc_auc_score(vs['label'], vs['score'])

def main(pattern, nfold=5):
    print(f"[评分列 = {COL}]")
    folds = []
    for f in range(nfold):
        try:
            folds.append(pd.read_csv(pattern.format(f=f)))
        except FileNotFoundError:
            folds.append(None)
    avail = [f for f in range(nfold) if folds[f] is not None]
    print(f"可用折: {avail}")

    # 每折 × 每个 τ 的 AUROC
    A = {f: {tau: fold_auroc_soft(folds[f], tau) for tau in TAU_GRID} for f in avail}

    print("\nAUROC vs tau (各折均值):")
    for tau in TAU_GRID:
        vals = [A[f][tau] for f in avail if not np.isnan(A[f][tau])]
        print(f"  tau={tau:.3f}: {np.mean(vals):.4f}")

    # 留一折交叉选 τ
    chosen, aurocs = [], []
    for f in avail:
        others = [o for o in avail if o != f]
        best_tau = max(TAU_GRID, key=lambda tau: np.nanmean([A[o][tau] for o in others]) if others else A[f][tau])
        a = A[f][best_tau]
        chosen.append((f, best_tau, a))
        aurocs.append(a)
        print(f"fold{f}: 选 tau={best_tau:.3f} -> AUROC={a:.4f}")

    print(f"\n>>> 5折 AUROC (soft) = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}  (留一折选 tau)")

    # 全局最优 τ (参考)
    best_global = max(TAU_GRID, key=lambda tau: np.nanmean([A[f][tau] for f in avail]))
    ub = [A[f][best_global] for f in avail]
    print(f"    参考(固定全局最优 tau={best_global:.3f}): {np.mean(ub):.4f} ± {np.std(ub):.4f}")

    # 同时对比硬 top-p% (同脚本内方便对比)
    from score_auroc import P_GRID, vehicle_scores, fold_auroc
    print("\n--- 对比: 硬 top-p% ---")
    A_hard = {f: {p: fold_auroc(folds[f], p) for p in P_GRID} for f in avail}
    hard_aurocs = []
    for f in avail:
        others = [o for o in avail if o != f]
        best_p = max(P_GRID, key=lambda p: np.nanmean([A_hard[o][p] for o in others]) if others else A_hard[f][p])
        hard_aurocs.append(A_hard[f][best_p])
    print(f"    硬 top-p%: {np.mean(hard_aurocs):.4f} ± {np.std(hard_aurocs):.4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='repro/scores/seg_fold{f}.csv')
    ap.add_argument('--col', default='rec_error')
    a = ap.parse_args()
    COL = a.col
    main(a.glob)
