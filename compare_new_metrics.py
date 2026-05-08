"""
Comparison plot for new metrics across grokking vs. non-grokking MLP runs.
Usage: python compare_new_metrics.py [output.png]
"""
import sys
import math

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RUNS = [
    {
        'label':     'add mod 113 (grokks ~ep 16800)',
        'path':      'results/add_wfourier/metrics.csv',
        'color':     '#2166ac',
        'linestyle': '-',
        'grokked':   True,
    },
    {
        'label':     'x3xy2y mod 113 (no grokking)',
        'path':      'results/x3_wfourier/metrics.csv',
        'color':     '#d6604d',
        'linestyle': '--',
        'grokked':   False,
    },
]

# (panel title, csv column, y-scale)
PANELS = [
    ('Test Accuracy',
     'test_acc',                                        None),
    ('Weight Fourier Frac (first input layer, mean of x/y halves)\n*** earliest signal: diverges by epoch 500 ***',
     'weight_fourier_frac_mean_layers_0_weight',        None),
    ('Activation Fourier Frac Explained\n(first hidden layer)',
     'fourier_mean_frac_explained_hidden_0',            None),
    ('Output Logit Fourier Frac\n(logit grid over all p^2 inputs)',
     'logit_fourier_frac',                              None),
    ('Effective Rank\n(first weight matrix)',
     'eff_rank_layers_0_weight',                        None),
    ('Group Associativity Fraction\n(fraction of (a,b,c) triples satisfying T[T[a,b],c]=T[a,T[b,c]])',
     'group_assoc_frac',                                None),
]

def load(run):
    return pd.read_csv(run['path'])

def grok_epoch(df):
    mask = df['test_acc'] >= 0.95
    if mask.any():
        return int(df.loc[mask.idxmax(), 'epoch'])
    return None

def plot_metric(ax, df, run, col):
    if col not in df.columns:
        return
    sub = df[['epoch', col]].dropna()
    if sub.empty:
        return
    ax.plot(sub['epoch'], sub[col],
            color=run['color'], linestyle=run['linestyle'],
            linewidth=1.8)

dfs = [load(r) for r in RUNS]

fig, axes = plt.subplots(len(PANELS), 1,
                         figsize=(13, 4.0 * len(PANELS)),
                         sharex=False)
fig.suptitle('Grokking vs. Non-Grokking: New Metric Comparison (MLP, mod 113)',
             fontsize=13, y=1.002)

for ax, (title, col, yscale) in zip(axes, PANELS):
    ax.set_title(title, fontsize=9, loc='left', pad=4)
    ax.set_xlabel('Epoch')

    for df, run in zip(dfs, RUNS):
        plot_metric(ax, df, run, col)

        if run['grokked']:
            ge = grok_epoch(df)
            if ge is not None:
                ax.axvline(ge, color=run['color'], linestyle=':', alpha=0.6, linewidth=1.2)

    if yscale == 'log':
        ax.set_yscale('log')

    ax.grid(True, alpha=0.25)

# Legend
handles = [
    mlines.Line2D([], [], color=r['color'], linestyle=r['linestyle'],
                  linewidth=2, label=r['label'])
    for r in RUNS
]
handles.append(
    mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1.2,
                  label='Grokking onset (test_acc=0.95)')
)
fig.legend(handles=handles, loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True)

plt.tight_layout()

out = sys.argv[1] if len(sys.argv) > 1 else 'results/comparison_new_metrics.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
plt.close()
