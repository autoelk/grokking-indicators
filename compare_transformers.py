import pandas as pd

add = pd.read_csv('results/transformer/metrics.csv')
x3  = pd.read_csv('results/x3xy2y_transformer/metrics.csv')

def show(df, label, checkpoints):
    print(f'=== {label} ===')
    for ep in checkpoints:
        row = df[df['epoch'] <= ep].iloc[-1]
        ep_val = int(row['epoch'])
        tacc   = f"{row['test_acc']:.4f}"
        tloss  = f"{row['test_loss']:.2f}"

        def g(col):
            v = row.get(col)
            return f"{v:.3f}" if v is not None and pd.notna(v) else 'N/A'

        print(
            f"  ep {ep_val:6d}: test_acc={tacc}  test_loss={tloss:5s}"
            f"  eff_rank_W_in={g('eff_rank_W_in_weight')}"
            f"  eff_rank_unembed={g('eff_rank_unembed_weight')}"
            f"  cond_W_in={g('condition_number_W_in_weight')}"
            f"  fourier_ent={g('fourier_mean_entropy_mlp')}"
            f"  fourier_frac={g('fourier_mean_frac_explained_mlp')}"
            f"  sign_unembed={g('sign_stability_unembed_weight')}"
        )
    print()

show(add, 'ADD Transformer  (grokked ~ep 15500)', [0, 2000, 5000, 10000, 14000, 15500, 16000])
show(x3,  'X3XY2Y Transformer  (no grokking, 50k epochs)', [0, 2000, 5000, 10000, 20000, 35000, 49500])
