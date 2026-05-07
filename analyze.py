"""
Visualization and correlation analysis for grokking investigation runs.

Usage:
  python analyze.py <results_dir> [output_plot.png]
  python analyze.py results/run1
  python analyze.py results/run1 results/run1/plot.png
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PHASE_COLORS = {'pre': '#ffcccc', 'transition': '#ffffcc', 'post': '#ccffcc'}


def load_results(results_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(results_dir) / 'metrics.csv')


def add_phase_shading(ax, df: pd.DataFrame):
    """Color background by training phase."""
    if 'phase' not in df.columns:
        return
    epochs = df['epoch'].values
    phases = df['phase'].values
    i = 0
    while i < len(phases):
        j = i + 1
        while j < len(phases) and phases[j] == phases[i]:
            j += 1
        end = epochs[j - 1] if j <= len(phases) else epochs[-1]
        ax.axvspan(epochs[i], end, alpha=0.12,
                   color=PHASE_COLORS.get(phases[i], 'white'), zorder=0)
        i = j


def plot_all(results_dir: str, save_path: str = None):
    df = load_results(results_dir)
    epochs = df['epoch'].values

    weight_l2_cols    = [c for c in df.columns if c.startswith('l2_')]
    eff_rank_cols     = [c for c in df.columns if c.startswith('eff_rank_')]
    cond_cols         = [c for c in df.columns if c.startswith('condition_number_')]
    dead_cols         = [c for c in df.columns if c.startswith('dead_frac_')]
    sparsity_cols     = [c for c in df.columns if c.startswith('sparsity_')]
    fourier_ent_cols  = [c for c in df.columns if c.startswith('fourier_mean_entropy_')]
    fourier_frac_cols = [c for c in df.columns if c.startswith('fourier_mean_frac_explained_')]
    dist_init_cols    = [c for c in df.columns if c.startswith('weight_dist_init_')]
    sign_stab_cols    = [c for c in df.columns if c.startswith('sign_stability_')]

    n_panels = 9
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    fig.suptitle(f'Grokking Investigation — {results_dir}', fontsize=12, y=1.002)

    # Panel 1: Loss (log scale)
    ax = axes[0]
    ax.semilogy(epochs, df['train_loss'], label='Train Loss', color='steelblue')
    ax.semilogy(epochs, df['test_loss'],  label='Test Loss',  color='crimson')
    ax.set_ylabel('Cross-Entropy Loss'); ax.legend(); ax.set_title('Loss Curves (log scale)')
    add_phase_shading(ax, df)

    # Panel 2: Accuracy + grokking score
    ax = axes[1]
    ax.plot(epochs, df['train_acc'], label='Train Acc', color='steelblue')
    ax.plot(epochs, df['test_acc'],  label='Test Acc',  color='crimson')
    if 'grokking_score' in df.columns:
        ax.plot(epochs, df['grokking_score'], label='Grokking Score', color='green', linestyle='--', alpha=0.7)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Value'); ax.legend(); ax.set_title('Accuracy & Grokking Score')
    add_phase_shading(ax, df)

    # Panel 3: Weight L2 norms
    ax = axes[2]
    for col in weight_l2_cols:
        ax.plot(epochs, df[col], label=col.replace('l2_', ''))
    ax.set_ylabel('L2 Norm'); ax.legend(fontsize=7); ax.set_title('Weight L2 Norms per Layer')
    add_phase_shading(ax, df)

    # Panel 4: Effective rank + condition number
    ax = axes[3]
    for col in eff_rank_cols:
        ax.plot(epochs, df[col].dropna(), label=col.replace('eff_rank_', ''))
    ax.set_ylabel('Effective Rank (solid)'); ax.legend(fontsize=7)
    ax2 = ax.twinx()
    for col in cond_cols:
        ax2.semilogy(epochs, df[col].dropna(), linestyle='--', alpha=0.6,
                     label=col.replace('condition_number_', ''))
    ax2.set_ylabel('Condition Number (dashed, log)')
    ax.set_title('Effective Rank & Condition Number of Weight Matrices')
    add_phase_shading(ax, df)

    # Panel 5: Weight distance from initialization + sign stability
    ax = axes[4]
    for col in dist_init_cols:
        ax.plot(epochs, df[col], label=col.replace('weight_dist_init_', ''))
    ax.set_ylabel('Dist from Init (solid)'); ax.legend(fontsize=7)
    ax2 = ax.twinx()
    for col in sign_stab_cols:
        ax2.plot(epochs, df[col], linestyle='--', alpha=0.7,
                 label=col.replace('sign_stability_', ''))
    ax2.set_ylabel('Sign Stability (dashed)')
    ax.set_title('Weight Drift from Initialization & Sign Stability')
    add_phase_shading(ax, df)

    # Panel 6: Dead neuron fraction & activation sparsity
    ax = axes[5]
    for col in dead_cols:
        ax.plot(epochs, df[col], label=f'dead {col.replace("dead_frac_", "")}')
    for col in sparsity_cols:
        ax.plot(epochs, df[col], linestyle='--', label=f'sparse {col.replace("sparsity_", "")}')
    ax.set_ylabel('Fraction'); ax.legend(fontsize=7); ax.set_title('Dead Neurons & Activation Sparsity')
    add_phase_shading(ax, df)

    # Panel 7: Representation quality — margin + test entropy
    ax = axes[6]
    if 'margin' in df.columns:
        ax.plot(epochs, df['margin'], label='Prediction Margin', color='purple')
    ax.set_ylabel('Margin (solid)'); ax.legend(fontsize=7)
    ax2 = ax.twinx()
    if 'test_entropy' in df.columns:
        ax2.plot(epochs, df['test_entropy'], color='orange', linestyle='--', label='Test Entropy')
    if 'cka_vs_prev' in df.columns:
        ax2.plot(epochs, df['cka_vs_prev'], color='teal', linestyle=':', label='CKA vs prev')
    ax2.set_ylabel('Entropy / CKA (dashed)')
    ax2.legend(fontsize=7)
    ax.set_title('Prediction Margin & Test Entropy & CKA Change')
    add_phase_shading(ax, df)

    # Panel 8: Sharpness (if computed)
    ax = axes[7]
    if 'sharpness' in df.columns:
        sub = df[['epoch', 'sharpness']].dropna()
        ax.semilogy(sub['epoch'], sub['sharpness'], color='brown', label='Sharpness')
        ax.set_ylabel('Sharpness (log)'); ax.legend()
    ax.set_title('Loss Landscape Sharpness')
    add_phase_shading(ax, df)

    # Panel 9: Fourier structure
    ax = axes[8]
    for col in fourier_ent_cols:
        sub = df[['epoch', col]].dropna()
        ax.plot(sub['epoch'], sub[col], label=f'entropy {col.split("_")[-1]}')
    ax.set_ylabel('Fourier Entropy (solid)'); ax.legend(fontsize=7)
    ax2 = ax.twinx()
    for col in fourier_frac_cols:
        sub = df[['epoch', col]].dropna()
        ax2.plot(sub['epoch'], sub[col], linestyle='--', alpha=0.7,
                 label=f'frac_expl {col.split("_")[-1]}')
    ax2.set_ylabel('Frac Explained by Top Freq (dashed)')
    ax.set_title('Fourier Structure of Neuron Activations')
    ax.set_xlabel('Epoch')
    add_phase_shading(ax, df)

    # Phase legend
    patches = [mpatches.Patch(color=PHASE_COLORS[ph], alpha=0.4, label=ph)
               for ph in ('pre', 'transition', 'post')]
    fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig(Path(results_dir) / 'plot.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to {Path(results_dir) / 'plot.png'}")
    plt.close()


def correlation_analysis(results_dir: str):
    """Rank metrics by correlation with test accuracy and pre-grokking predictiveness."""
    df = load_results(results_dir)

    grok_mask = df['test_acc'] >= 0.95
    if not grok_mask.any():
        print("Grokking (test_acc >= 0.95) did not occur -- skipping correlation analysis.")
        return

    grok_idx   = int(grok_mask.idxmax())
    grok_epoch = int(df.loc[grok_idx, 'epoch'])
    print(f"\nGrokking onset (test_acc >= 0.95): epoch {grok_epoch}")
    print(f"Total epochs logged: {len(df)}")

    skip_cols = {'epoch', 'phase', 'train_acc', 'test_acc', 'acc_gap', 'grokking_score'}
    numeric_cols = [c for c in df.select_dtypes(include=[float, int]).columns
                    if c not in skip_cols]

    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        aligned = df.loc[series.index, 'test_acc']
        corr = float(series.corr(aligned))

        total_change = series.iloc[-1] - series.iloc[0]
        pre_grok = series.loc[:grok_idx]
        if len(pre_grok) > 0 and abs(total_change) > 1e-12:
            pre_change = pre_grok.iloc[-1] - series.iloc[0]
            frac_before = float(pre_change / total_change)
        else:
            frac_before = float('nan')

        results.append({
            'metric':                   col,
            'corr_with_test_acc':       round(corr, 4),
            'frac_change_before_grok':  round(frac_before, 4) if not np.isnan(frac_before) else float('nan'),
        })

    corr_df = pd.DataFrame(results).sort_values('corr_with_test_acc',
                                                  key=abs, ascending=False)
    print("\nTop metrics correlated with test accuracy:")
    print(corr_df.head(20).to_string(index=False))

    out_path = Path(results_dir) / 'correlation_analysis.csv'
    corr_df.to_csv(out_path, index=False)
    print(f"\nFull analysis saved to {out_path}")
    return corr_df


if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/run'
    save_path   = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading results from: {results_dir}")
    plot_all(results_dir, save_path)
    correlation_analysis(results_dir)
