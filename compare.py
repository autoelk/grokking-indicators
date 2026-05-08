import pandas as pd

add = pd.read_csv('results/full50k/metrics.csv')
x3  = pd.read_csv('results/x3xy2y_mlp/metrics.csv')

cols = [
    'epoch', 'test_acc', 'test_loss',
    'eff_rank_layers_0_weight', 'eff_rank_layers_2_weight',
    'condition_number_layers_0_weight',
    'fourier_mean_entropy_hidden_0', 'fourier_mean_frac_explained_hidden_0',
    'sign_stability_layers_2_weight', 'margin',
]

def show(df, label, checkpoints):
    print(f'=== {label} ===')
    for ep in checkpoints:
        row = df[df['epoch'] <= ep].iloc[-1]
        ep_val   = int(row['epoch'])
        tacc     = f"{row['test_acc']:.4f}"
        tloss    = f"{row['test_loss']:.2f}"
        er0      = f"{row['eff_rank_layers_0_weight']:.2f}" if pd.notna(row.get('eff_rank_layers_0_weight')) else 'N/A'
        er2      = f"{row['eff_rank_layers_2_weight']:.2f}" if pd.notna(row.get('eff_rank_layers_2_weight')) else 'N/A'
        cond0    = f"{row['condition_number_layers_0_weight']:.1f}" if pd.notna(row.get('condition_number_layers_0_weight')) else 'N/A'
        fent     = f"{row['fourier_mean_entropy_hidden_0']:.3f}" if pd.notna(row.get('fourier_mean_entropy_hidden_0')) else 'N/A'
        ffrac    = f"{row['fourier_mean_frac_explained_hidden_0']:.4f}" if pd.notna(row.get('fourier_mean_frac_explained_hidden_0')) else 'N/A'
        sign     = f"{row['sign_stability_layers_2_weight']:.4f}" if pd.notna(row.get('sign_stability_layers_2_weight')) else 'N/A'
        margin   = f"{row['margin']:.3f}" if pd.notna(row.get('margin')) else 'N/A'
        print(f"  ep {ep_val:6d}: test_acc={tacc}  test_loss={tloss:5s}  eff_rank_0={er0}  eff_rank_2={er2}  cond_0={cond0:7s}  fourier_ent={fent}  fourier_frac={ffrac}  sign_stab={sign}  margin={margin}")
    print()

show(add, 'ADD mod 113  (grokked at epoch ~16800)', [0, 2000, 5000, 10000, 14000, 16800, 19000])
show(x3,  'X3XY2Y mod 113  (no grokking, 50k epochs)', [0, 2000, 5000, 10000, 20000, 30000, 49800])
