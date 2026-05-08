"""
Side-by-side comparison of grokking vs. non-grokking runs across key metrics.
Usage: python compare_plot.py [output.png]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── run registry ─────────────────────────────────────────────────────────────

RUNS = [
    {
        'label':     'add (MLP)',
        'path':      'results/full50k/metrics.csv',
        'color':     '#2166ac',   # blue
        'linestyle': '-',
        'grokked':   True,
        # canonical key → actual CSV column name
        'test_acc':     'test_acc',
        'test_loss':    'test_loss',
        'fourier_frac': 'fourier_mean_frac_explained_hidden_0',
        'fourier_ent':  'fourier_mean_entropy_hidden_0',
        'eff_rank':     'eff_rank_layers_0_weight',
        'cond_num':     'condition_number_layers_0_weight',
    },
    {
        'label':     'add (Transformer)',
        'path':      'results/transformer/metrics.csv',
        'color':     '#4dac26',   # green
        'linestyle': '-',
        'grokked':   True,
        'test_acc':     'test_acc',
        'test_loss':    'test_loss',
        'fourier_frac': 'fourier_mean_frac_explained_mlp',
        'fourier_ent':  'fourier_mean_entropy_mlp',
        'eff_rank':     'eff_rank_W_in_weight',
        'cond_num':     'condition_number_W_in_weight',
    },
    {
        'label':     'x3xy2y (MLP)',
        'path':      'results/x3xy2y_mlp/metrics.csv',
        'color':     '#d6604d',   # red
        'linestyle': '--',
        'grokked':   False,
        'test_acc':     'test_acc',
        'test_loss':    'test_loss',
        'fourier_frac': 'fourier_mean_frac_explained_hidden_0',
        'fourier_ent':  'fourier_mean_entropy_hidden_0',
        'eff_rank':     'eff_rank_layers_0_weight',
        'cond_num':     'condition_number_layers_0_weight',
    },
    {
        'label':     'x3xy2y (Transformer)',
        'path':      'results/x3xy2y_transformer/metrics.csv',
        'color':     '#f4a582',   # orange
        'linestyle': '--',
        'grokked':   False,
        'test_acc':     'test_acc',
        'test_loss':    'test_loss',
        'fourier_frac': 'fourier_mean_frac_explained_mlp',
        'fourier_ent':  'fourier_mean_entropy_mlp',
        'eff_rank':     'eff_rank_W_in_weight',
        'cond_num':     'condition_number_W_in_weight',
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load(run):
    df = pd.read_csv(run['path'])
    return df

def plot_metric(ax, df, run, col, dropna=True, **kw):
    if col not in df.columns:
        return
    sub = df[['epoch', col]].dropna() if dropna else df[['epoch', col]]
    ax.plot(sub['epoch'], sub[col],
            color=run['color'], linestyle=run['linestyle'],
            linewidth=1.8, **kw)

def grok_epoch(df):
    mask = df['test_acc'] >= 0.95
    if mask.any():
        return int(df.loc[mask.idxmax(), 'epoch'])
    return None

# ── figure ────────────────────────────────────────────────────────────────────

# All panels use alias resolution (key → run-specific column name)
PANELS = [
    ('Test Accuracy',                                        'test_acc',     None),
    ('Test Loss (log scale)',                                'test_loss',    'log'),
    ('Fourier Frac Explained\n(first hidden layer / MLP block)', 'fourier_frac', None),
    ('Fourier Entropy\n(first hidden layer / MLP block)',    'fourier_ent',  None),
    ('Effective Rank\n(first weight matrix)',                'eff_rank',     None),
    ('Condition Number\n(first weight matrix)',              'cond_num',     None),
]

fig, axes = plt.subplots(len(PANELS), 1, figsize=(13, 4.2 * len(PANELS)), sharex=False)
fig.suptitle('Grokking vs. Non-Grokking: Metric Comparison Across Operations & Architectures',
             fontsize=13, y=1.002)

dfs = [load(r) for r in RUNS]

for ax, (title, key, yscale) in zip(axes, PANELS):
    ax.set_title(title, fontsize=10, loc='left', pad=4)
    ax.set_xlabel('Epoch')

    for df, run in zip(dfs, RUNS):
        col = run.get(key, key)   # alias → actual CSV column name
        plot_metric(ax, df, run, col)

        if run['grokked']:
            ge = grok_epoch(df)
            if ge is not None:
                ax.axvline(ge, color=run['color'], linestyle=':', alpha=0.5, linewidth=1.2)

    if yscale == 'log':
        ax.set_yscale('log')

    ax.grid(True, alpha=0.25)

# ── legend ────────────────────────────────────────────────────────────────────

legend_handles = []
for run in RUNS:
    legend_handles.append(
        mlines.Line2D([], [], color=run['color'], linestyle=run['linestyle'],
                      linewidth=2, label=run['label'])
    )
legend_handles.append(
    mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1.2,
                  label='Grokking onset (test_acc=0.95)')
)

fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True)

plt.tight_layout()

out = sys.argv[1] if len(sys.argv) > 1 else 'results/comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
plt.close()
