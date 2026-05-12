"""
Comprehensive comparison plot — all metrics × all 8 runs.
Usage: python compare_all_runs.py [output.png]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RUNS = [
    {'label': 'MLP add',      'path': 'results/mlp_add/metrics.csv',      'color': '#2166ac', 'linestyle': '-',  'grokked': True},
    {'label': 'MLP multiply', 'path': 'results/mlp_multiply/metrics.csv', 'color': '#1a9850', 'linestyle': '-',  'grokked': True},
    {'label': 'MLP x3xy2y',   'path': 'results/mlp_x3xy2y/metrics.csv',   'color': '#d6604d', 'linestyle': '-',  'grokked': False},
    {'label': 'MLP x2xyy2',   'path': 'results/mlp_x2xyy2/metrics.csv',   'color': '#e08214', 'linestyle': '-',  'grokked': False},
    {'label': 'TFM add',      'path': 'results/tfm_add/metrics.csv',      'color': '#2166ac', 'linestyle': '--', 'grokked': True},
    {'label': 'TFM multiply', 'path': 'results/tfm_multiply/metrics.csv', 'color': '#1a9850', 'linestyle': '--', 'grokked': True},
    {'label': 'TFM x3xy2y',   'path': 'results/tfm_x3xy2y/metrics.csv',   'color': '#d6604d', 'linestyle': '--', 'grokked': False},
    {'label': 'TFM x2xyy2',   'path': 'results/tfm_x2xyy2/metrics.csv',   'color': '#e08214', 'linestyle': '--', 'grokked': False},
]

# (panel title, [col candidates in priority order], y-scale)
# First matching column in a run's CSV is used; missing = silently skipped.
PANELS = [
    # ── Basic ──────────────────────────────────────────────────────────────
    ('Test Accuracy',
     ['test_acc'], None),
    ('Train Accuracy',
     ['train_acc'], None),
    ('Train Loss (log)',
     ['train_loss'], 'log'),
    ('Test Loss (log)',
     ['test_loss'], 'log'),
    ('Grokking Score  (test_acc / train_acc)',
     ['grokking_score'], None),
    ('Accuracy Gap  (train − test)',
     ['acc_gap'], None),

    # ── Total weight norm ──────────────────────────────────────────────────
    ('Total Weight L2²  (sum of all w²,  log)\ndriven down by weight decay throughout training',
     ['total_weight_sq'], 'log'),

    # ── Per-layer L2 norms ─────────────────────────────────────────────────
    ('L2 Norm — first weight layer\nMLP: layers.0.weight  |  TFM: W_in',
     ['l2_layers_0_weight', 'l2_W_in_weight'], None),
    ('L2 Norm — output weight layer\nMLP: layers.2.weight  |  TFM: unembed',
     ['l2_layers_2_weight', 'l2_unembed_weight'], None),

    # ── Effective rank ─────────────────────────────────────────────────────
    ('Effective Rank — first layer\nMLP: layers.0  |  TFM: W_in',
     ['eff_rank_layers_0_weight', 'eff_rank_W_in_weight'], None),
    ('Effective Rank — second layer\nMLP: layers.1  |  TFM: W_out',
     ['eff_rank_layers_1_weight', 'eff_rank_W_out_weight'], None),
    ('Effective Rank — output layer\nMLP: layers.2  |  TFM: unembed',
     ['eff_rank_layers_2_weight', 'eff_rank_unembed_weight'], None),
    ('Effective Rank — attention Q  (TFM only)',
     ['eff_rank_W_Q_weight'], None),
    ('Effective Rank — attention K  (TFM only)',
     ['eff_rank_W_K_weight'], None),

    # ── Condition number ───────────────────────────────────────────────────
    ('Condition Number — first layer  (log)\nMLP: layers.0  |  TFM: W_in',
     ['condition_number_layers_0_weight', 'condition_number_W_in_weight'], 'log'),
    ('Condition Number — output layer  (log)\nMLP: layers.2  |  TFM: unembed',
     ['condition_number_layers_2_weight', 'condition_number_unembed_weight'], 'log'),

    # ── Weight dynamics ────────────────────────────────────────────────────
    ('Weight Dist from Init — first layer\nMLP: layers.0  |  TFM: W_in',
     ['weight_dist_init_layers_0_weight', 'weight_dist_init_W_in_weight'], None),
    ('Weight Dist from Init — output layer\nMLP: layers.2  |  TFM: unembed',
     ['weight_dist_init_layers_2_weight', 'weight_dist_init_unembed_weight'], None),
    ('Sign Stability — first layer\nfraction of weights with same sign as at init',
     ['sign_stability_layers_0_weight', 'sign_stability_W_in_weight'], None),
    ('Update / Weight Ratio — first layer\n||Δθ|| / ||θ|| per step',
     ['update_weight_ratio_layers_0_weight', 'update_weight_ratio_W_in_weight'], None),

    # ── Gradient norms ─────────────────────────────────────────────────────
    ('Gradient Norm — first layer  (log)\nMLP: layers.0.weight  |  TFM: W_in.weight',
     ['grad_norm_layers_0_weight', 'grad_norm_W_in_weight'], 'log'),
    ('Gradient Alignment\ncosine similarity of consecutive gradient vectors',
     ['grad_alignment'], None),

    # ── Neuron structure ───────────────────────────────────────────────────
    ('Dead Neuron Fraction — first hidden layer\nMLP: hidden_0  |  TFM: mlp',
     ['dead_frac_hidden_0', 'dead_frac_mlp'], None),
    ('Activation Sparsity — first hidden layer\nMLP: hidden_0  |  TFM: mlp',
     ['sparsity_hidden_0', 'sparsity_mlp'], None),

    # ── Representation quality ─────────────────────────────────────────────
    ('Prediction Margin\nmin(top1 − top2 logit) over training set',
     ['margin'], None),
    ('Test Entropy\nmean softmax entropy on test set  (low = confident)',
     ['test_entropy'], None),
    ('Test Confidence\nmean (top1 − top2) logit gap on test set',
     ['test_confidence'], None),
    ('CKA vs Previous Checkpoint\nrepresentation similarity across log steps',
     ['cka_vs_prev'], None),

    # ── Sharpness ──────────────────────────────────────────────────────────
    ('Sharpness  (log)\nfinite-diff Hessian-vector approx  (sparse: every 1000 ep)',
     ['sharpness'], 'log'),

    # ── Fourier — activations ──────────────────────────────────────────────
    ('Activation Fourier Frac Explained — h0 / mlp\nMLP: hidden_0  |  TFM: mlp',
     ['fourier_mean_frac_explained_hidden_0', 'fourier_mean_frac_explained_mlp'], None),
    ('Activation Fourier Frac Explained — h1  (MLP only)',
     ['fourier_mean_frac_explained_hidden_1'], None),
    ('Activation Fourier Entropy — h0 / mlp\nMLP: hidden_0  |  TFM: mlp',
     ['fourier_mean_entropy_hidden_0', 'fourier_mean_entropy_mlp'], None),
    ('Activation Fourier Entropy — h1  (MLP only)',
     ['fourier_mean_entropy_hidden_1'], None),
    ('Activation Intrinsic Dimensionality — h0 / mlp\nPCA components for 90% variance',
     ['idim_hidden_0', 'idim_mlp'], None),
    ('Activation Intrinsic Dimensionality — h1  (MLP only)',
     ['idim_hidden_1'], None),

    # ── Fourier — weights ──────────────────────────────────────────────────
    ('Weight Fourier Frac — input layer  (MLP only)\n*** earliest signal: diverges ~ep 500 ***',
     ['weight_fourier_frac_mean_layers_0_weight'], None),
    ('Weight Fourier Frac — x-half  (MLP only)',
     ['weight_fourier_frac_x_layers_0_weight'], None),
    ('Weight Fourier Frac — y-half  (MLP only)',
     ['weight_fourier_frac_y_layers_0_weight'], None),

    # ── Fourier — logits ───────────────────────────────────────────────────
    ('Logit Fourier Frac\nFourier concentration of output logit grid over all p² inputs',
     ['logit_fourier_frac'], None),
    ('Logit Fourier Entropy',
     ['logit_fourier_entropy'], None),

    # ── Group structure ────────────────────────────────────────────────────
    ('Group Associativity Fraction\nfrac of (a,b,c) with T[T[a,b],c] = T[a,T[b,c]]',
     ['group_assoc_frac'], None),
    ('Group Has Identity\n1 = identity element exists in prediction table',
     ['group_has_identity'], None),
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
                        linewidth=1.5, alpha=0.85)
            return


dfs = [load(r) for r in RUNS]

n = len(PANELS)
fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=False)
fig.suptitle('All Metrics × All 8 Runs  (4 operations × MLP / Transformer, mod 113)',
             fontsize=13, y=1.001)

for ax, (title, cols, yscale) in zip(axes, PANELS):
    ax.set_title(title, fontsize=8, loc='left', pad=3)
    ax.set_xlabel('Epoch', fontsize=7)
    ax.tick_params(labelsize=7)

    for df, run in zip(dfs, RUNS):
        plot_metric(ax, df, run, cols)
        if run['grokked']:
            ge = grok_epoch(df)
            if ge is not None:
                ax.axvline(ge, color=run['color'], linestyle=':', alpha=0.45, linewidth=0.9)

    if yscale == 'log':
        ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

# Legend
handles = [
    mlines.Line2D([], [], color='#2166ac', linewidth=2, label='add'),
    mlines.Line2D([], [], color='#1a9850', linewidth=2, label='multiply'),
    mlines.Line2D([], [], color='#d6604d', linewidth=2, label='x3xy2y'),
    mlines.Line2D([], [], color='#e08214', linewidth=2, label='x2xyy2'),
    mlines.Line2D([], [], color='gray', linestyle='-',  linewidth=2, label='MLP (solid)'),
    mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Transformer (dashed)'),
    mlines.Line2D([], [], color='gray', linestyle=':',  linewidth=1.2, label='Grokking onset'),
]
fig.legend(handles=handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.005), fontsize=9, frameon=True)

plt.tight_layout()

out = sys.argv[1] if len(sys.argv) > 1 else 'results/comparison_all_runs.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
plt.close()
