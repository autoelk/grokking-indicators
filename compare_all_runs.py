"""
Comparison plot for all 8 runs: 4 operations × 2 model types.
Usage: python compare_all_runs.py [output.png]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# color = operation, linestyle = model type
RUNS = [
    {'label': 'MLP add (grokks ~15400)',      'path': 'results/mlp_add/metrics.csv',      'color': '#2166ac', 'linestyle': '-',  'grokked': True},
    {'label': 'MLP multiply (grokks ~17100)', 'path': 'results/mlp_multiply/metrics.csv', 'color': '#1a9850', 'linestyle': '-',  'grokked': True},
    {'label': 'MLP x3xy2y (no grokking)',     'path': 'results/mlp_x3xy2y/metrics.csv',   'color': '#d6604d', 'linestyle': '-',  'grokked': False},
    {'label': 'MLP x2xyy2 (no grokking)',     'path': 'results/mlp_x2xyy2/metrics.csv',   'color': '#e08214', 'linestyle': '-',  'grokked': False},
    {'label': 'TFM add (grokks ~14900)',       'path': 'results/tfm_add/metrics.csv',      'color': '#2166ac', 'linestyle': '--', 'grokked': True},
    {'label': 'TFM multiply (grokks ~20900)', 'path': 'results/tfm_multiply/metrics.csv', 'color': '#1a9850', 'linestyle': '--', 'grokked': True},
    {'label': 'TFM x3xy2y (no grokking)',     'path': 'results/tfm_x3xy2y/metrics.csv',   'color': '#d6604d', 'linestyle': '--', 'grokked': False},
    {'label': 'TFM x2xyy2 (partial)',         'path': 'results/tfm_x2xyy2/metrics.csv',   'color': '#e08214', 'linestyle': '--', 'grokked': False},
]

# Each panel: (title, [col_candidates_in_priority_order], yscale)
PANELS = [
    ('Test Accuracy',
     ['test_acc'], None),
    ('Total Weight L2 Squared (sum of all w²)\ndriven down by weight decay — expected to fall as grokking nears',
     ['total_weight_sq'], 'log'),
    ('Weight Fourier Frac (first input layer, MLP only)\n*** earliest signal: diverges by epoch 500 ***',
     ['weight_fourier_frac_mean_layers_0_weight'], None),
    ('Activation Fourier Frac Explained (first hidden layer)\nMLP=hidden_0, TFM=mlp',
     ['fourier_mean_frac_explained_hidden_0', 'fourier_mean_frac_explained_mlp'], None),
    ('Output Logit Fourier Frac\n(logit grid over all p² inputs)',
     ['logit_fourier_frac'], None),
    ('Effective Rank (first hidden weight matrix)\nMLP=layers.0, TFM=W_in',
     ['eff_rank_layers_0_weight', 'eff_rank_W_in_weight'], None),
    ('Group Associativity Fraction\n(fraction of (a,b,c) triples satisfying T[T[a,b],c]=T[a,T[b,c]])',
     ['group_assoc_frac'], None),
]


def load(run):
    return pd.read_csv(run['path'])


def grok_epoch(df):
    mask = df['test_acc'] >= 0.95
    if mask.any():
        return int(df.loc[mask.idxmax(), 'epoch'])
    return None


def plot_metric(ax, df, run, cols):
    for col in cols:
        if col in df.columns:
            sub = df[['epoch', col]].dropna()
            if not sub.empty:
                ax.plot(sub['epoch'], sub[col],
                        color=run['color'], linestyle=run['linestyle'],
                        linewidth=1.6, alpha=0.85)
            return


dfs = [load(r) for r in RUNS]

fig, axes = plt.subplots(len(PANELS), 1,
                         figsize=(14, 4.0 * len(PANELS)),
                         sharex=False)
fig.suptitle('Grokking vs. Non-Grokking: All 8 Runs (4 ops × MLP/Transformer, mod 113)',
             fontsize=13, y=1.002)

for ax, (title, cols, yscale) in zip(axes, PANELS):
    ax.set_title(title, fontsize=9, loc='left', pad=4)
    ax.set_xlabel('Epoch')

    for df, run in zip(dfs, RUNS):
        plot_metric(ax, df, run, cols)

        if run['grokked']:
            ge = grok_epoch(df)
            if ge is not None:
                ax.axvline(ge, color=run['color'], linestyle=':', alpha=0.5, linewidth=1.0)

    if yscale == 'log':
        ax.set_yscale('log')

    ax.grid(True, alpha=0.25)

# Legend: operations by color, model type by linestyle
op_handles = [
    mlines.Line2D([], [], color='#2166ac', linewidth=2, label='add'),
    mlines.Line2D([], [], color='#1a9850', linewidth=2, label='multiply'),
    mlines.Line2D([], [], color='#d6604d', linewidth=2, label='x3xy2y'),
    mlines.Line2D([], [], color='#e08214', linewidth=2, label='x2xyy2'),
    mlines.Line2D([], [], color='gray',    linestyle='-',  linewidth=2, label='MLP (solid)'),
    mlines.Line2D([], [], color='gray',    linestyle='--', linewidth=2, label='Transformer (dashed)'),
    mlines.Line2D([], [], color='gray',    linestyle=':',  linewidth=1.2, label='Grokking onset (test_acc=0.95)'),
]
fig.legend(handles=op_handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.015), fontsize=10, frameon=True)

plt.tight_layout()

out = sys.argv[1] if len(sys.argv) > 1 else 'results/comparison_all_runs.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
plt.close()
