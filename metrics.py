"""
All metric computation for the grokking investigation.
Functions return flat dicts of scalar values for easy logging.
"""
import math
from collections import Counter
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def compute_basic_metrics(model, train_X, train_y, test_X, test_y) -> dict:
    model.eval()
    with torch.no_grad():
        train_logits = model(train_X)
        test_logits  = model(test_X)

    train_loss = F.cross_entropy(train_logits, train_y).item()
    test_loss  = F.cross_entropy(test_logits,  test_y).item()
    train_acc  = (train_logits.argmax(1) == train_y).float().mean().item()
    test_acc   = (test_logits.argmax(1)  == test_y).float().mean().item()

    return {
        'train_loss': train_loss, 'test_loss': test_loss,
        'train_acc':  train_acc,  'test_acc':  test_acc,
        'acc_gap': train_acc - test_acc,
        'grokking_score': test_acc / max(train_acc, 1e-8),
    }


# ---------------------------------------------------------------------------
# Weight structure
# ---------------------------------------------------------------------------

def _effective_rank(W: torch.Tensor) -> float:
    """Effective rank via singular-value entropy (Roy & Vetterli 2007)."""
    try:
        S = torch.linalg.svdvals(W.float())
    except Exception:
        return float('nan')
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    H = -(p * p.log()).sum()
    return H.exp().item()


def _condition_number(W: torch.Tensor) -> float:
    try:
        S = torch.linalg.svdvals(W.float())
    except Exception:
        return float('nan')
    S = S[S > 1e-10]
    if len(S) < 2:
        return float('nan')
    return (S[0] / S[-1]).item()


def compute_weight_metrics(model, theta_0: Optional[Dict] = None,
                           prev_theta: Optional[Dict] = None) -> dict:
    """
    theta_0:    initial parameter snapshot (for drift / sign_stability metrics).
    prev_theta: parameter snapshot from the step just before optimizer.step()
                (for update_to_weight_ratio).
    Both should be {name: tensor} dicts of detached CPU tensors.
    """
    metrics = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        safe = name.replace('.', '_')
        W = param.detach()

        metrics[f'l2_{safe}']  = W.pow(2).sum().sqrt().item()

        if param.grad is not None:
            metrics[f'grad_norm_{safe}'] = param.grad.detach().pow(2).sum().sqrt().item()

        if W.dim() == 2:
            metrics[f'eff_rank_{safe}']       = _effective_rank(W)
            metrics[f'condition_number_{safe}'] = _condition_number(W)

        if theta_0 is not None and name in theta_0:
            diff = W - theta_0[name].to(W.device)
            metrics[f'weight_dist_init_{safe}'] = diff.pow(2).sum().sqrt().item()
            # Fraction of weights that still have the same sign as at init
            same_sign = (W.sign() == theta_0[name].to(W.device).sign()).float().mean().item()
            metrics[f'sign_stability_{safe}'] = same_sign

        if prev_theta is not None and name in prev_theta:
            delta = W - prev_theta[name].to(W.device)
            w_norm = W.pow(2).sum().sqrt().item()
            metrics[f'update_weight_ratio_{safe}'] = (
                delta.pow(2).sum().sqrt().item() / max(w_norm, 1e-12)
            )

    return metrics


def snapshot_params(model) -> Dict[str, torch.Tensor]:
    """Snapshot current parameters as detached CPU tensors."""
    return {name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def snapshot_grads(model) -> Optional[torch.Tensor]:
    """Concatenate all current gradients into a single flat vector (CPU)."""
    parts = [p.grad.detach().cpu().flatten()
             for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    return torch.cat(parts) if parts else None


def compute_gradient_alignment(curr_grads: Optional[torch.Tensor],
                                prev_grads: Optional[torch.Tensor]) -> dict:
    """Cosine similarity between consecutive gradient vectors."""
    if curr_grads is None or prev_grads is None:
        return {}
    cos = F.cosine_similarity(curr_grads.unsqueeze(0),
                               prev_grads.unsqueeze(0)).item()
    return {'grad_alignment': cos}


# ---------------------------------------------------------------------------
# Neuron structure
# ---------------------------------------------------------------------------

def compute_neuron_metrics(model, train_X, threshold=1e-3) -> dict:
    model.eval()
    with torch.no_grad():
        model(train_X, capture=True)

    metrics = {}
    for name, acts in model.activations.items():
        # acts: [N, hidden_size]
        dead = (acts.abs().max(dim=0).values <= threshold).float().mean().item()
        sparse = (acts <= threshold).float().mean().item()
        metrics[f'dead_frac_{name}']  = dead
        metrics[f'sparsity_{name}']   = sparse
    return metrics


# ---------------------------------------------------------------------------
# Representation quality
# ---------------------------------------------------------------------------

def compute_representation_metrics(model, train_X, train_y,
                                   test_X, test_y,
                                   prev_activations: Optional[Dict] = None) -> dict:
    model.eval()
    with torch.no_grad():
        train_logits = model(train_X, capture=True)
        curr_acts = {k: v.clone() for k, v in model.activations.items()}
        test_logits = model(test_X)

    # Prediction margin on training set
    top2 = train_logits.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).min().item()

    # Mean prediction entropy on test set (low = confident)
    test_probs = torch.softmax(test_logits, dim=1)
    test_entropy = -(test_probs * (test_probs + 1e-12).log()).sum(dim=1).mean().item()

    metrics = {'margin': margin, 'test_entropy': test_entropy}

    # CKA change: similarity between current and previous representations
    if prev_activations is not None:
        cka_changes = []
        for name in curr_acts:
            if name in prev_activations:
                A = curr_acts[name]          # [N, H]
                B = prev_activations[name].to(A.device)
                if A.shape == B.shape:
                    cka_changes.append(_linear_cka(A, B))
        if cka_changes:
            metrics['cka_vs_prev'] = sum(cka_changes) / len(cka_changes)

    return metrics, curr_acts


def _linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    """Linear CKA similarity between representation matrices A and B ([N, H])."""
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    dot_AB = (A.T @ B).pow(2).sum()
    dot_AA = (A.T @ A).pow(2).sum()
    dot_BB = (B.T @ B).pow(2).sum()
    denom = (dot_AA * dot_BB).sqrt()
    if denom < 1e-12:
        return float('nan')
    return (dot_AB / denom).item()


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------

def compute_sharpness(model, train_X, train_y, epsilon=1e-3) -> dict:
    """
    Approximate sharpness via finite-difference Hessian-vector product.
    sharpness ≈ ||∇L(θ + ε·v) - ∇L(θ)|| / ε  for a random unit vector v.
    """
    model.train()

    # Compute reference gradient at θ
    model.zero_grad()
    F.cross_entropy(model(train_X), train_y).backward()
    grad_ref = {name: p.grad.detach().clone()
                for name, p in model.named_parameters() if p.grad is not None}

    # Generate random unit perturbation v
    v = {name: torch.randn_like(p.data) for name, p in model.named_parameters()}
    v_norm = sum(t.pow(2).sum() for t in v.values()).sqrt()
    for t in v.values():
        t.div_(v_norm)

    # Perturb parameters by +ε·v
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.add_(epsilon * v[name])

    # Compute gradient at θ + ε·v
    model.zero_grad()
    F.cross_entropy(model(train_X), train_y).backward()
    grad_perturbed = {name: p.grad.detach().clone()
                      for name, p in model.named_parameters() if p.grad is not None}

    # Restore parameters
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.sub_(epsilon * v[name])

    # ||∇L(θ+εv) - ∇L(θ)|| / ε
    diff_sq = sum(
        (grad_perturbed[n] - grad_ref[n]).pow(2).sum()
        for n in grad_ref if n in grad_perturbed
    )
    sharpness = (diff_sq.sqrt() / epsilon).item()

    model.zero_grad()
    return {'sharpness': sharpness}


# ---------------------------------------------------------------------------
# Fourier analysis
# ---------------------------------------------------------------------------

def make_fourier_basis(p: int) -> torch.Tensor:
    """
    Orthonormal Fourier basis, shape [p, p].
    Row 0: constant 1/sqrt(p).
    Rows 2k-1, 2k: cos/sin at frequency k (normalized).
    """
    basis = [torch.ones(p) / p ** 0.5]
    for k in range(1, p // 2 + 1):
        angles = 2 * math.pi * k * torch.arange(p) / p
        cos_v = torch.cos(angles)
        sin_v = torch.sin(angles)
        basis.append(cos_v / cos_v.norm())
        basis.append(sin_v / sin_v.norm())
    # If p is even, we may have one extra row – trim to exactly p rows
    basis = basis[:p]
    return torch.stack(basis)  # [p, p]


def compute_fourier_metrics(model, all_X, p: int) -> dict:
    """
    all_X: FloatTensor [p^2, 2p] in canonical order x=0..p-1, y=0..p-1.
    Computes Fourier structure of MLP hidden-layer activations.
    """
    fourier_basis = make_fourier_basis(p).to(all_X.device)

    model.eval()
    with torch.no_grad():
        model(all_X, capture=True)

    metrics = {}
    for name, acts in model.activations.items():
        # acts: [p^2, H]
        H = acts.shape[1]

        # Center per-neuron
        acts_c = acts - acts.mean(dim=0, keepdim=True)

        # Reshape to [p, p, H]
        acts_2d = acts_c.view(p, p, H)

        # 2D DFT: fourier[fx, fy, h] = Σ_{x,y} basis[fx,x] * basis[fy,y] * acts[x,y,h]
        fourier = torch.einsum('xyh,fx,gy->fgh', acts_2d, fourier_basis, fourier_basis)
        # fourier: [p, p, H]

        power = fourier.pow(2)  # [p, p, H]

        top_freqs, fracs, entropies = [], [], []
        for ni in range(H):
            np_power = power[:, :, ni]  # [p, p]
            total = np_power.sum().item()
            if total < 1e-12:
                top_freqs.append(0); fracs.append(0.0); entropies.append(0.0)
                continue

            # Find dominant frequency k by looking at the 3x3 submatrix
            # at indices [0, 2k-1, 2k] x [0, 2k-1, 2k]
            best_freq, best_frac = 0, 0.0
            for k in range(1, p // 2):
                idx = [0, 2 * k - 1, 2 * k]
                sub = np_power[[[i] * 3 for i in idx], [idx] * 3]
                frac = sub.sum().item() / total
                if frac > best_frac:
                    best_frac = frac
                    best_freq = k

            top_freqs.append(best_freq)
            fracs.append(best_frac)

            # Entropy of flattened power spectrum
            p_dist = np_power.flatten() / total
            p_dist = p_dist.clamp(min=1e-12)
            H_val = -(p_dist * p_dist.log()).sum().item()
            entropies.append(H_val)

        metrics[f'fourier_mean_top_freq_{name}']      = sum(top_freqs) / H
        metrics[f'fourier_mean_frac_explained_{name}'] = sum(fracs) / H
        metrics[f'fourier_mean_entropy_{name}']        = sum(entropies) / H

        freq_counts = Counter(top_freqs)
        metrics[f'fourier_key_freqs_{name}'] = str([f for f, _ in freq_counts.most_common(3)])

    return metrics


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def detect_phase(history: list, window: int = 5) -> str:
    """
    Classify current training phase from the logged history list.
    Each entry is a dict with at least 'train_acc' and 'test_acc'.
    """
    if len(history) < 2:
        return 'pre'

    recent = history[-window:]
    train_acc = sum(r['train_acc'] for r in recent) / len(recent)
    test_acc  = sum(r['test_acc']  for r in recent) / len(recent)

    delta_test = 0.0
    if len(history) > window:
        delta_test = history[-1]['test_acc'] - history[-(window + 1)]['test_acc']

    if train_acc > 0.95 and test_acc > 0.95:
        return 'post'
    elif train_acc > 0.95 and delta_test > 0.05:
        return 'transition'
    return 'pre'


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def compute_all_metrics(model, train_X, train_y, test_X, test_y,
                        all_X, p,
                        theta_0=None, prev_theta=None,
                        prev_activations=None,
                        compute_fourier=True,
                        compute_sharpness_metric=False) -> tuple:
    """
    Returns (metrics_dict, curr_activations).
    curr_activations can be passed as prev_activations next time for CKA tracking.
    """
    metrics = {}
    metrics.update(compute_basic_metrics(model, train_X, train_y, test_X, test_y))
    metrics.update(compute_weight_metrics(model, theta_0=theta_0, prev_theta=prev_theta))
    metrics.update(compute_neuron_metrics(model, train_X))

    rep_metrics, curr_acts = compute_representation_metrics(
        model, train_X, train_y, test_X, test_y, prev_activations)
    metrics.update(rep_metrics)

    if compute_sharpness_metric:
        metrics.update(compute_sharpness(model, train_X, train_y))

    if compute_fourier:
        metrics.update(compute_fourier_metrics(model, all_X, p))

    return metrics, curr_acts
