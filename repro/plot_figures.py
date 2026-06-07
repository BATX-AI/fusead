#!/usr/bin/env python3
"""FuseAD 论文图表生成。输出到 figures/ 目录。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os, sys

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# Color palette (colorblind-friendly)
C_DYAD   = '#7f7f7f'  # grey
C_FUSEAD = '#d62728'  # red
C_ABLATE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
DATASETS = ['Dataset A\n(brand1)', 'Dataset B\n(brand2)', 'Dataset C\n(brand3)']

# ── Data ──────────────────────────────────────────────────────
dyad_mean  = np.array([0.8537, 0.9054, 0.8218])
dyad_std   = np.array([0.0320, 0.0313, 0.0779])
fuse_mean  = np.array([0.8961, 0.9071, 0.9201])
fuse_std   = np.array([0.0261, 0.0403, 0.0274])

# Per-fold data
dyad_folds = {
    'A': [0.8263, 0.8686, 0.8061, 0.8784, 0.8892],
    'B': [0.9271, 0.9107, 0.8854, 0.9464, 0.8571],
    'C': [0.7099, 0.8580, 0.8519, 0.7593, 0.9298],
}
fuse_folds = {
    'A': [0.8929, 0.9039, 0.8475, 0.9167, 0.9196],
    'B': [0.9479, 0.9107, 0.9271, 0.9196, 0.8304],
    'C': [0.8951, 0.8951, 0.9383, 0.9074, 0.9649],
}

# Ablation data: [Dataset A, Dataset B, Dataset C]
ablations = {
    'Rec Error only':       np.array([0.8681, 0.9071, 0.7793]),
    'Rec + Mahalanobis':    np.array([0.8820, 0.8577, 0.8737]),
    'Rec + Maha + GMM\n(FuseAD hard)': np.array([0.8820, 0.8845, 0.9201]),
    'FuseAD full\n(+ Soft top-p%)':    np.array([0.8961, 0.9071, 0.9201]),
}

# Noise sensitivity data
noise_vals = [0.01, 0.1, 0.5, 1.0, 2.0]
noise_data = {
    'Dataset A': [0.8429, 0.8497, 0.8565, 0.8681, 0.8236],
    'Dataset B': [0.9024, 0.9071, 0.7955, 0.7899, np.nan],
}


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Main comparison bar chart
# ═══════════════════════════════════════════════════════════════
def fig1_main_comparison():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(DATASETS))
    w = 0.32

    bars1 = ax.bar(x - w/2, dyad_mean, w, yerr=dyad_std, capsize=4,
                   color=C_DYAD, edgecolor='white', linewidth=0.5, label='DyAD')
    bars2 = ax.bar(x + w/2, fuse_mean, w, yerr=fuse_std, capsize=4,
                   color=C_FUSEAD, edgecolor='white', linewidth=0.5, label='FuseAD (ours)')

    # Annotate values on bars
    for bar, val in zip(bars1, dyad_mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=C_DYAD, fontweight='bold')
    for bar, val in zip(bars2, fuse_mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, color=C_FUSEAD, fontweight='bold')

    # Delta annotations
    for i in range(3):
        delta = fuse_mean[i] - dyad_mean[i]
        mid = (dyad_mean[i] + fuse_mean[i]) / 2
        ax.annotate(f'+{delta:.2%}'.replace('%',''), xy=(x[i] + w/2, fuse_mean[i] + 0.04),
                    fontsize=8, ha='center', color='#d62728', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0', edgecolor='none', alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.60, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Avg inset
    avg_dyad = dyad_mean.mean()
    avg_fuse = fuse_mean.mean()
    ax.text(0.98, 0.12, f'Avg: {avg_dyad:.4f} → {avg_fuse:.4f}\n(+{avg_fuse-avg_dyad:.2%})'.replace('%',''),
            transform=ax.transAxes, ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc', alpha=0.9))

    ax.set_title('FuseAD vs DyAD: 3-Dataset Comparison', fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig1_main_comparison.png')
    fig.savefig(f'{OUTDIR}/fig1_main_comparison.pdf')
    plt.close(fig)
    print('[1/5] Main comparison saved.')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Architecture / Pipeline Flow Diagram
# ═══════════════════════════════════════════════════════════════
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def box(x, y, w, h, text, color='#e8e8e8', text_color='black', fontsize=10, bold=False):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle='round,pad=0.15', facecolor=color,
                              edgecolor='#555', linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight=weight, zorder=3)

    def arrow(x1, y1, x2, y2, lw=1.5, color='#555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                   connectionstyle='arc3,rad=0'), zorder=1)

    def label(x, y, text, fontsize=8, color='#666', ha='center'):
        ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color)

    # ── Encoder block ──
    box(2.5, 3.8, 2.6, 0.9, 'GRU Encoder\n(Bi-directional)', color='#d4e6f1', bold=True)
    label(0.5, 3.8, 'Charging\nSegments', ha='center', fontsize=8, color='#333')
    arrow(0.9, 3.8, 1.2, 3.8)

    # ── Latent block ──
    box(5.2, 3.8, 1.6, 0.9, 'Latent z\n μ, log σ²', color='#f9e79f', bold=True)
    arrow(3.8, 3.8, 4.4, 3.8)

    # ── Decoder block ──
    box(7.8, 3.8, 2.4, 0.9, 'GRU Decoder\n+ Response Pred.', color='#d5f5e3', bold=True)
    arrow(6.0, 3.8, 6.6, 3.8)

    # ── Rec Error output ──
    box(9.4, 3.8, 0.9, 0.65, 'Rec\nError', color='#fadbd8', fontsize=8)
    arrow(9.0, 3.8, 9.0, 3.8, lw=1.2)

    # ── 3 Fusion branches ──
    branch_y = 2.0
    for i, (txt, clr) in enumerate([
        ('Reconstruction\n  Error (MSE)', '#fadbd8'),
        ('Mahalanobis\n  Distance (LW)', '#d4e6f1'),
        ('GMM Density\n  (BIC-k)', '#d5f5e3'),
    ]):
        bx = 2.5 + i * 2.5
        box(bx, branch_y, 2.0, 0.75, txt, color=clr, fontsize=8)
        if i == 0:
            arrow(5.2, 3.35, bx, branch_y + 0.38, lw=1.0, color='#999')
        else:
            arrow(5.2, 3.35, bx, branch_y + 0.38, lw=1.0, color='#999')

    # ── Fusion combiner ──
    box(5.2, 0.75, 3.0, 0.65, 'Z-score Normalize\n  +  Sum (FuseAD Score)', color='#e8daef', fontsize=9, bold=True)
    for i in range(3):
        bx = 2.5 + i * 2.5
        arrow(bx, branch_y - 0.38, 5.2, 0.75 + 0.33, lw=1.0, color='#999')

    label(1.0, 0.75, 'Anomaly\nScore', fontsize=8, color='#333', ha='center')
    arrow(3.7, 0.75, 2.5, 0.75, lw=1.5, color='#d62728')

    # ── Title ──
    ax.text(5, 4.9, 'FuseAD  Architecture', ha='center', fontsize=16,
            fontweight='bold', color='#2c3e50')
    ax.text(5, 4.55, 'Multi-Signal Fusion from VAE Latent Space',
            ha='center', fontsize=10, color='#7f8c8d')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig2_pipeline.png')
    fig.savefig(f'{OUTDIR}/fig2_pipeline.pdf')
    plt.close(fig)
    print('[2/5] Pipeline diagram saved.')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Ablation Study
# ═══════════════════════════════════════════════════════════════
def fig3_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)

    for di, (ds_name, ax) in enumerate(zip(DATASETS, axes)):
        names = list(ablations.keys())
        vals = [ablations[n][di] for n in names]
        colors = C_ABLATE

        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace('\n',' ') for n in names], fontsize=8)
        ax.set_title(ds_name.replace('\n',' '), fontweight='bold', fontsize=11)
        ax.set_xlim(0.72, 0.94 if di < 2 else 0.97)
        ax.axvline(x=dyad_mean[di], color=C_DYAD, linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(dyad_mean[di] + 0.003, len(names)-0.3, f'DyAD\n{dyad_mean[di]:.3f}',
                fontsize=7, color=C_DYAD, va='top')

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    axes[0].set_xlabel('AUROC', fontsize=11)
    fig.suptitle('Ablation Study: Component-wise Contribution', fontweight='bold', y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig3_ablation.png')
    fig.savefig(f'{OUTDIR}/fig3_ablation.pdf')
    plt.close(fig)
    print('[3/5] Ablation study saved.')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Noise Sensitivity
# ═══════════════════════════════════════════════════════════════
def fig4_noise():
    fig, ax = plt.subplots(figsize=(7, 4.2))

    colors = {'Dataset A': '#d62728', 'Dataset B': '#1f77b4'}
    markers = {'Dataset A': 'o', 'Dataset B': 's'}
    for label, data in noise_data.items():
        xs = noise_vals[:len(data)]
        ys = data
        # Remove NaN for plotting line
        mask = ~np.isnan(ys)
        ax.plot(np.array(xs)[mask], np.array(ys)[mask], '-', color=colors[label],
                marker=markers[label], markersize=8, linewidth=2, label=label,
                markerfacecolor='white', markeredgewidth=1.5)
        # Annotate
        for x, y in zip(np.array(xs)[mask], np.array(ys)[mask]):
            ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                       xytext=(0, 12), ha='center', fontsize=8, color=colors[label],
                       fontweight='bold')

    ax.set_xlabel('VAE Noise Scale')
    ax.set_ylabel('AUROC')
    ax.set_xticks([0.01, 0.1, 0.5, 1.0])
    ax.set_xticklabels(['0.01\n(near-AE)', '0.1', '0.5', '1.0\n(standard VAE)'])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(frameon=True, fancybox=True, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Annotate sweet spots
    ax.annotate('Optimal\nnoise=1.0', xy=(1.0, 0.8681), xytext=(1.3, 0.85),
                arrowprops=dict(arrowstyle='->', color='#d62728'), fontsize=8,
                color='#d62728', fontweight='bold')
    ax.annotate('Optimal\nnoise=0.1', xy=(0.1, 0.9071), xytext=(0.25, 0.88),
                arrowprops=dict(arrowstyle='->', color='#1f77b4'), fontsize=8,
                color='#1f77b4', fontweight='bold')

    ax.set_title('Noise Scale Sensitivity: Different Datasets Need Different VAE Stochasticity',
                 fontweight='bold', pad=12, fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig4_noise.png')
    fig.savefig(f'{OUTDIR}/fig4_noise.pdf')
    plt.close(fig)
    print('[4/5] Noise sensitivity saved.')


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Per-Fold Detailed Comparison
# ═══════════════════════════════════════════════════════════════
def fig5_perfold():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    ds_keys = ['A', 'B', 'C']

    for di, (key, ds_name, ax) in enumerate(zip(ds_keys, DATASETS, axes)):
        folds = np.arange(5)
        w = 0.3

        ax.bar(folds - w/2, dyad_folds[key], w, color=C_DYAD, edgecolor='white',
               linewidth=0.5, label='DyAD')
        ax.bar(folds + w/2, fuse_folds[key], w, color=C_FUSEAD, edgecolor='white',
               linewidth=0.5, label='FuseAD')

        # Delta per fold
        for f in folds:
            delta = fuse_folds[key][f] - dyad_folds[key][f]
            color = '#2ca02c' if delta > 0 else '#d62728'
            sign = '+' if delta > 0 else ''
            ax.text(f, max(dyad_folds[key][f], fuse_folds[key][f]) + 0.015,
                    f'{sign}{delta:.2%}'.replace('%',''), ha='center', fontsize=6.5,
                    color=color, fontweight='bold')

        ax.set_xticks(folds)
        ax.set_xticklabels([f'F{i+1}' for i in folds])
        ax.set_title(ds_name.replace('\n',' '), fontweight='bold', fontsize=11)
        ax.set_ylim(0.65, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if di == 0:
            ax.legend(frameon=True, fontsize=8)

    axes[0].set_ylabel('AUROC')
    fig.suptitle('5-Fold Cross-Validation: Per-Fold Comparison', fontweight='bold', y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig5_perfold.png')
    fig.savefig(f'{OUTDIR}/fig5_perfold.pdf')
    plt.close(fig)
    print('[5/5] Per-fold comparison saved.')


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fig1_main_comparison()
    fig2_pipeline()
    fig3_ablation()
    fig4_noise()
    fig5_perfold()
    print(f'\nAll figures saved to {OUTDIR}/')
