#!/usr/bin/env python3
"""FuseAD publication-quality figures. Output to figures/ directory."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ── Color palette (colorblind-friendly, Wong 2011) ──────────────
C_DYAD      = '#999999'  # grey
C_FUSEAD    = '#E69F00'  # orange
C_BLUE      = '#56B4E9'  # sky blue
C_GREEN     = '#009E73'  # bluish green
C_RED       = '#D55E00'  # vermillion
C_PURPLE    = '#CC79A7'  # reddish purple
C_ABLATE    = ['#CCCCCC', '#56B4E9', '#E69F00', '#D55E00']
DS_LABELS   = ['Dataset A\n(198 cars)', 'Dataset B\n(49 cars)', 'Dataset C\n(~100 cars)']
DS_SHORT    = ['Dataset A', 'Dataset B', 'Dataset C']

# ── Data ─────────────────────────────────────────────────────────
dyad_mean  = np.array([0.8537, 0.9054, 0.8218])
dyad_std   = np.array([0.0320, 0.0313, 0.0779])
fuse_mean  = np.array([0.8961, 0.9071, 0.9201])
fuse_std   = np.array([0.0261, 0.0403, 0.0274])
deltas     = fuse_mean - dyad_mean

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

ablations = {
    'Rec Error only':         np.array([0.8681, 0.9071, 0.7793]),
    '+ Mahalanobis':          np.array([0.8820, 0.8577, 0.8737]),
    '+ GMM density':          np.array([0.8820, 0.8845, 0.9201]),
    '+ Soft top-p% (FuseAD)': np.array([0.8961, 0.9071, 0.9201]),
}

noise_vals = [0.01, 0.1, 0.5, 1.0, 2.0]
noise_data = {
    'Dataset A (brand1)': [0.8429, 0.8497, 0.8565, 0.8681, 0.8236],
    'Dataset B (brand2)': [0.9024, 0.9071, 0.7955, 0.7899, np.nan],
}


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Main comparison — grouped bar chart
# ═══════════════════════════════════════════════════════════════════
def fig1_main_comparison():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(3)
    w = 0.30

    b1 = ax.bar(x - w/2, dyad_mean, w, yerr=dyad_std, capsize=4, error_kw={'linewidth':1.2},
                color=C_DYAD, edgecolor='white', linewidth=0.5, label='DyAD (baseline)')
    b2 = ax.bar(x + w/2, fuse_mean, w, yerr=fuse_std, capsize=4, error_kw={'linewidth':1.2},
                color=C_FUSEAD, edgecolor='white', linewidth=0.5, label='FuseAD (ours)')

    # Value labels on bars
    for bars, vals, c in [(b1, dyad_mean, C_DYAD), (b2, fuse_mean, C_FUSEAD)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.006,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8.5, color=c, fontweight='bold')

    # Delta badges above each pair
    for i in range(3):
        ax.annotate(f'+{deltas[i]:.3f}', xy=(x[i], max(dyad_mean[i], fuse_mean[i]) + 0.06),
                    fontsize=9, ha='center', color='#D55E00', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF5EB', edgecolor='#E69F00', alpha=0.9, linewidth=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(DS_LABELS, fontsize=9)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_ylim(0.60, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Legend
    legend = ax.legend(frameon=True, fancybox=True, framealpha=0.95, loc='upper left',
                       handlelength=1.0, handleheight=0.7)
    legend.get_frame().set_linewidth(0.5)

    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, zorder=0)

    # Summary box
    avg_d, avg_f = dyad_mean.mean(), fuse_mean.mean()
    ax.text(0.98, 0.08,
            f'3-dataset average\n'
            f'DyAD:    {avg_d:.4f}\n'
            f'FuseAD:  {avg_f:.4f}\n'
            f'Gain:      +{avg_f-avg_d:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            fontfamily='monospace', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8F8', edgecolor='#DDD', alpha=0.95, linewidth=0.6))

    ax.set_title('FuseAD vs DyAD: Main Comparison', fontweight='bold', pad=14)
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(f'{OUTDIR}/fig1_main_comparison.{fmt}')
    plt.close(fig)
    print('[1/5] Main comparison done.')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Algorithm pipeline diagram
# ═══════════════════════════════════════════════════════════════════
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def box(x, y, w, h, text, color='#E8E8E8', tc='black', fs=9, bold=False, pad=0.12):
        r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=f'round,pad={pad}',
                           facecolor=color, edgecolor='#888', linewidth=1.0, zorder=3)
        ax.add_patch(r)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
                fontweight='bold' if bold else 'normal', zorder=4)

    def arrow(x1, y1, x2, y2, lw=1.3, color='#666', style='simple'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=2,
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle='arc3,rad=0'))

    def curved_arrow(x1, y1, x2, y2, lw=1.0, color='#999', rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle=f'arc3,rad={rad}'))

    # ── Top row: Encoder → Latent → Decoder ──
    box(2.2, 4.2, 2.4, 0.85, 'GRU Encoder\n(all channels)', color='#D6EAF8', bold=True, fs=9)
    box(5.5, 4.2, 1.6, 0.85, 'Latent Space\nz ~ N(μ, σ²)', color='#FCF3CF', bold=True, fs=9)
    box(8.8, 4.2, 2.6, 0.85, 'GRU Decoder\n(system inputs + z)', color='#D5F5E3', bold=True, fs=9)
    arrow(3.4, 4.2, 4.7, 4.2)
    arrow(6.3, 4.2, 7.5, 4.2)

    # Input / output labels
    ax.text(0.35, 4.2, 'Charging\nSegments', ha='center', fontsize=7.5, color='#444', fontstyle='italic')
    arrow(0.7, 4.2, 1.0, 4.2, lw=1.0, color='#999')

    # Rec Error direct output
    box(11.0, 4.2, 1.2, 0.65, 'Rec Error\n(MSE)', color='#FADBD8', fs=7.5, pad=0.08)
    arrow(10.1, 4.2, 10.4, 4.2, lw=1.0, color='#999')

    # ── Bottom row: Three fusion branches ──
    branches = [
        (2.5, 'Reconstruction\nError (MSE)', '#FADBD8'),
        (5.5, 'Mahalanobis\nDistance (LW)', '#D6EAF8'),
        (8.5, 'GMM Density\n(BIC-selected k)', '#D5F5E3'),
    ]
    for bx, txt, clr in branches:
        box(bx, 2.2, 2.2, 0.75, txt, color=clr, fs=7.5)

    # Connect latent to each branch
    for bx, _, _ in branches:
        curved_arrow(5.5, 3.78, bx, 2.58, lw=0.8, color='#BBB', rad=-0.15)

    # ── Fusion combiner ──
    box(5.5, 0.85, 3.5, 0.65, 'Z-score Normalize + Sum  →  FuseAD Score',
        color='#E8DAEF', bold=True, fs=9, pad=0.1)

    for bx, _, _ in branches:
        curved_arrow(bx, 1.83, 5.5, 1.18, lw=0.8, color='#BBB', rad=0.15)

    # Output arrow
    ax.text(2.8, 0.85, 'Vehicle\nAnomaly\nScore', ha='center', fontsize=8, color='#444', fontweight='bold')
    curved_arrow(3.35, 0.85, 3.75, 0.85, lw=1.5, color='#D55E00', rad=0)

    # ── Section labels (subtle background) ──
    for x_pos, lbl in [(5.5, 5.05)]:
        ax.text(x_pos, 5.05, 'DyAD Base Architecture', ha='center', fontsize=9,
                color='#888', fontstyle='italic')
    ax.text(5.5, 3.15, 'FuseAD Multi-Signal Fusion', ha='center', fontsize=9,
            color='#888', fontstyle='italic')

    ax.set_title('FuseAD Architecture Overview', fontweight='bold', fontsize=15, pad=10, color='#2C3E50')

    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(f'{OUTDIR}/fig2_pipeline.{fmt}')
    plt.close(fig)
    print('[2/5] Pipeline diagram done.')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Ablation study — horizontal grouped bars
# ═══════════════════════════════════════════════════════════════════
def fig3_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.5), sharey=False)

    for di, (ds_name, ax) in enumerate(zip(DS_SHORT, axes)):
        names = list(ablations.keys())
        vals = [ablations[n][di] for n in names]

        colors = C_ABLATE
        ypos = range(len(names))
        bars = ax.barh(ypos, vals, height=0.55, color=colors, edgecolor='white', linewidth=0.5)

        # DyAD baseline line
        ax.axvline(x=dyad_mean[di], color='#999999', linestyle='--', linewidth=1.5, alpha=0.8, zorder=5)
        ax.text(dyad_mean[di] + 0.003, len(names) - 0.25, f'DyAD baseline\n{dyad_mean[di]:.4f}',
                fontsize=7, color='#666', va='top', fontstyle='italic')

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8.5, fontweight='bold', color='#333')

        ax.set_yticks(list(ypos))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(ds_name, fontweight='bold', fontsize=11, color='#2C3E50')
        ax.set_xlim(0.72, (dyad_mean[di] + 0.08) if di < 2 else 0.95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='x', labelsize=8)

    axes[0].set_xlabel('AUROC', fontsize=11)
    fig.suptitle('Ablation Study: Incremental Contribution of Each FuseAD Component',
                 fontweight='bold', fontsize=13, y=1.02, color='#2C3E50')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(f'{OUTDIR}/fig3_ablation.{fmt}')
    plt.close(fig)
    print('[3/5] Ablation study done.')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Noise sensitivity — dual-line plot with annotations
# ═══════════════════════════════════════════════════════════════════
def fig4_noise():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors_line = {'Dataset A (brand1)': C_RED, 'Dataset B (brand2)': C_BLUE}
    markers_line = {'Dataset A (brand1)': 'o', 'Dataset B (brand2)': 's'}

    for label, data in noise_data.items():
        xs = np.array(noise_vals[:len(data)])
        ys = np.array(data)
        mask = ~np.isnan(ys)
        ax.plot(xs[mask], ys[mask], '-', color=colors_line[label],
                marker=markers_line[label], markersize=9, linewidth=2.2,
                markerfacecolor='white', markeredgewidth=1.8, label=label, zorder=4)

        # Value annotations at each point
        for x, y in zip(xs[mask], ys[mask]):
            offset = 14 if x < 0.5 else -18
            ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                       xytext=(0, offset), ha='center', fontsize=7.5, color=colors_line[label],
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7))

    # Highlight optimal regions
    ax.axvspan(0.005, 0.055, alpha=0.06, color='#999', label='_nolegend_')
    ax.axvspan(0.055, 0.18, alpha=0.06, color='#E69F00', label='_nolegend_')
    ax.axvspan(0.7, 1.3, alpha=0.06, color='#D55E00', label='_nolegend_')

    ax.annotate('Optimal: noise=1.0', xy=(1.0, 0.8681), xytext=(1.45, 0.872),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2,
                                connectionstyle='arc3,rad=0.2'),
                fontsize=8.5, color=C_RED, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_RED, alpha=0.8, linewidth=0.6))

    ax.annotate('Optimal: noise=0.1', xy=(0.1, 0.9071), xytext=(0.35, 0.92),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2,
                                connectionstyle='arc3,rad=-0.2'),
                fontsize=8.5, color=C_BLUE, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=C_BLUE, alpha=0.8, linewidth=0.6))

    ax.set_xlabel('VAE Noise Scale (noise_scale parameter)', fontsize=11)
    ax.set_ylabel('AUROC', fontsize=11)
    ax.set_xticks([0.01, 0.1, 0.5, 1.0, 2.0])
    ax.set_xticklabels(['0.01\n(deterministic)', '0.1', '0.5', '1.0\n(standard VAE)', '2.0'], fontsize=8.5)
    ax.set_xlim(-0.05, 2.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    legend = ax.legend(frameon=True, fancybox=True, framealpha=0.95, loc='lower left',
                       handlelength=1.5)
    legend.get_frame().set_linewidth(0.5)
    legend.set_zorder(10)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)

    ax.set_title('Noise Scale Sensitivity', fontweight='bold', fontsize=13, pad=12, color='#2C3E50')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(f'{OUTDIR}/fig4_noise.{fmt}')
    plt.close(fig)
    print('[4/5] Noise sensitivity done.')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Per-fold detailed comparison
# ═══════════════════════════════════════════════════════════════════
def fig5_perfold():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.5), sharey=True)
    ds_keys = ['A', 'B', 'C']

    for di, (key, ds_name, ax) in enumerate(zip(ds_keys, DS_SHORT, axes)):
        folds = np.arange(5)
        w = 0.30

        ax.bar(folds - w/2, dyad_folds[key], w, color=C_DYAD, edgecolor='white',
               linewidth=0.5, label='DyAD', zorder=3)
        ax.bar(folds + w/2, fuse_folds[key], w, color=C_FUSEAD, edgecolor='white',
               linewidth=0.5, label='FuseAD', zorder=3)

        # Per-fold delta (absolute AUROC difference)
        for f in folds:
            delta = fuse_folds[key][f] - dyad_folds[key][f]
            clr = '#009E73' if delta > 0 else '#D55E00'
            sign = '+' if delta > 0 else ''
            y_pos = max(dyad_folds[key][f], fuse_folds[key][f])
            ax.text(f, y_pos + 0.012, f'{sign}{delta:.3f}', ha='center', fontsize=7,
                    color=clr, fontweight='bold')

        # Dataset mean lines
        ax.axhline(y=dyad_mean[di], color=C_DYAD, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
        ax.axhline(y=fuse_mean[di], color=C_FUSEAD, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)

        ax.set_xticks(folds)
        ax.set_xticklabels([f'Fold {i+1}' for i in folds], fontsize=8)
        ax.set_title(ds_name, fontweight='bold', fontsize=11, color='#2C3E50')
        ax.set_ylim(0.68, 1.01)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5, zorder=0)

        if di == 0:
            legend = ax.legend(frameon=True, fancybox=True, framealpha=0.95, fontsize=8.5,
                               handlelength=1.0)
            legend.get_frame().set_linewidth(0.5)

    axes[0].set_ylabel('AUROC', fontsize=11)
    fig.suptitle('5-Fold Cross-Validation: Per-Fold Detail', fontweight='bold',
                 fontsize=13, y=1.02, color='#2C3E50')
    fig.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(f'{OUTDIR}/fig5_perfold.{fmt}')
    plt.close(fig)
    print('[5/5] Per-fold comparison done.')


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    fig1_main_comparison()
    fig2_pipeline()
    fig3_ablation()
    fig4_noise()
    fig5_perfold()
    print(f'\nAll figures saved to {OUTDIR}/')
