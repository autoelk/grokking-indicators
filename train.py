"""
Main training loop for the grokking investigation.

Example usage:
  python train.py --n_epochs 10000 --log_every 100 --out_dir results/run1
  python train.py --model_type transformer --operation multiply --out_dir results/transformer
"""
import argparse
import json
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd

from data import make_dataset, make_full_dataset, make_token_dataset
from model import GrokMLP, GrokTransformer
from metrics import (
    compute_all_metrics, detect_phase,
    snapshot_params, compute_weight_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description='Grokking investigation trainer')
    p.add_argument('--p',             type=int,   default=113,     help='Modulus')
    p.add_argument('--operation',     type=str,   default='add',
                   choices=['add', 'subtract', 'multiply', 'x2xyy2', 'x3xy2y'])
    p.add_argument('--train_frac',    type=float, default=0.3,     help='Fraction of data for training')
    p.add_argument('--model_type',    type=str,   default='mlp',   choices=['mlp', 'transformer'])
    p.add_argument('--hidden_sizes',  type=int,   nargs='+',       default=[200, 200])
    p.add_argument('--n_epochs',      type=int,   default=10000)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1.0,
                   help='Weight decay; 1.0 reliably causes grokking (Nanda et al. 2023)')
    p.add_argument('--beta1',         type=float, default=0.9)
    p.add_argument('--beta2',         type=float, default=0.98)
    p.add_argument('--log_every',     type=int,   default=100,     help='Epochs between metric logs')
    p.add_argument('--fourier_every', type=int,   default=500,     help='Epochs between Fourier metric logs')
    p.add_argument('--sharp_every',   type=int,   default=1000,    help='Epochs between sharpness computation (expensive)')
    p.add_argument('--warmup_steps',  type=int,   default=10,      help='Linear LR warmup epochs')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--device',        type=str,   default='cpu')
    p.add_argument('--out_dir',       type=str,   default='results/run')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Data ---
    if args.model_type == 'mlp':
        train_X, train_y, test_X, test_y = make_dataset(
            p=args.p, operation=args.operation,
            train_fraction=args.train_frac, seed=args.seed)
        all_X, _ = make_full_dataset(p=args.p, operation=args.operation)
        model = GrokMLP(p=args.p, hidden_sizes=tuple(args.hidden_sizes)).to(device)
    else:
        train_X, train_y, test_X, test_y = make_token_dataset(
            p=args.p, operation=args.operation,
            train_fraction=args.train_frac, seed=args.seed)
        all_X_oh, _ = make_full_dataset(p=args.p, operation=args.operation)
        # For Fourier analysis we need full one-hot for the transformer's
        # activation grid — pass all integer tokens instead
        all_tokens = torch.tensor(
            [[x, y, args.p] for x in range(args.p) for y in range(args.p)],
            dtype=torch.long)
        all_X = all_tokens
        model = GrokTransformer(p=args.p).to(device)

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    test_X  = test_X.to(device)
    test_y  = test_y.to(device)
    all_X   = all_X.to(device)

    print(f"Task: {args.operation} mod {args.p}")
    print(f"Model: {args.model_type}  |  Train: {len(train_X)}  Test: {len(test_X)}")
    print(f"Optimizer: AdamW lr={args.lr} wd={args.weight_decay} betas=({args.beta1},{args.beta2})")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / args.warmup_steps))

    # Snapshots for dynamics metrics
    theta_0    = snapshot_params(model)   # initial parameters, never updated
    prev_theta = snapshot_params(model)   # updated each log step
    prev_activations = None               # for CKA

    # --- Logging ---
    history = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(train_X)
        loss   = F.cross_entropy(logits, train_y)
        loss.backward()

        # Capture gradient norms BEFORE optimizer.step() wipes them
        do_log = (epoch % args.log_every == 0)
        if do_log:
            grad_snapshot = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    safe = name.replace('.', '_')
                    grad_snapshot[f'grad_norm_{safe}'] = (
                        param.grad.detach().pow(2).sum().sqrt().item())
            # Snapshot θ before step for update/weight ratio
            pre_step_theta = snapshot_params(model)

        optimizer.step()
        scheduler.step()

        # --- Metric logging ---
        if do_log:
            compute_fourier  = (epoch % args.fourier_every == 0)
            compute_sharpness = (epoch % args.sharp_every == 0)

            row_metrics, curr_acts = compute_all_metrics(
                model, train_X, train_y, test_X, test_y,
                all_X, args.p,
                theta_0=theta_0,
                prev_theta=pre_step_theta,
                prev_activations=prev_activations,
                compute_fourier=compute_fourier,
                compute_sharpness_metric=compute_sharpness,
            )

            row = {'epoch': epoch}
            row.update(grad_snapshot)
            row.update(row_metrics)
            row['phase'] = detect_phase(history)

            history.append(row)
            prev_activations = curr_acts
            prev_theta = snapshot_params(model)

            print(
                f"Ep {epoch:6d} | "
                f"loss {row['train_loss']:.4f}/{row['test_loss']:.4f} | "
                f"acc {row['train_acc']:.3f}/{row['test_acc']:.3f} | "
                f"{row['phase']}"
            )

            # Early stop once fully grokked
            if row['test_acc'] >= 0.999 and row['train_acc'] >= 0.999:
                if epoch > 500:
                    print("Grokking complete — stopping early.")
                    break

    # --- Save results ---
    df = pd.DataFrame(history)
    df.to_csv(out_dir / 'metrics.csv', index=False)

    config = vars(args)
    config.update({'n_train': len(train_X), 'n_test': len(test_X)})
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    torch.save(model.state_dict(), out_dir / 'model_final.pt')

    print(f"\nResults saved to {out_dir}/")
    if len(history) > 0:
        print(f"Final test accuracy: {history[-1]['test_acc']:.4f}")
        grok_rows = [r for r in history if r['test_acc'] >= 0.95]
        if grok_rows:
            print(f"Grokking onset (test_acc >= 0.95): epoch {grok_rows[0]['epoch']}")
        else:
            print("Grokking did not occur in this run.")


if __name__ == '__main__':
    main()
