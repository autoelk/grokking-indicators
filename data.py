import torch
from torch.utils.data import TensorDataset

OPERATIONS = {
    'add':      lambda x, y, p: (x + y) % p,
    'subtract': lambda x, y, p: (x - y) % p,
    'multiply': lambda x, y, p: (x * y) % p,
    'x2xyy2':  lambda x, y, p: (x**2 + x*y + y**2) % p,
    'x3xy2y' : lambda x, y, p: (x**3 + (x*y)**2 + y) % p,
}


def _build_full(p, operation):
    op_fn = OPERATIONS[operation]
    inputs, labels = [], []
    for x in range(p):
        for y in range(p):
            x_oh = torch.zeros(p); x_oh[x] = 1.0
            y_oh = torch.zeros(p); y_oh[y] = 1.0
            inputs.append(torch.cat([x_oh, y_oh]))
            labels.append(op_fn(x, y, p))
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)


def make_full_dataset(p=113, operation='add'):
    """Full p^2 dataset in canonical (x,y) order. Used for Fourier metric computation."""
    inputs, labels = _build_full(p, operation)
    return inputs, labels


def make_token_dataset(p=113, operation='add', train_fraction=0.3, seed=42):
    """Integer token format [x, y, p] for the Transformer model."""
    op_fn = OPERATIONS[operation]
    tokens, labels = [], []
    for x in range(p):
        for y in range(p):
            tokens.append(torch.tensor([x, y, p], dtype=torch.long))
            labels.append(op_fn(x, y, p))
    tokens = torch.stack(tokens)
    labels = torch.tensor(labels, dtype=torch.long)

    torch.manual_seed(seed)
    perm = torch.randperm(len(tokens))
    n_train = int(train_fraction * len(tokens))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return (tokens[train_idx], labels[train_idx],
            tokens[test_idx],  labels[test_idx])


def make_dataset(p=113, operation='add', train_fraction=0.3, seed=42):
    """One-hot encoded dataset split into train/test. Primary format for MLP."""
    inputs, labels = _build_full(p, operation)
    torch.manual_seed(seed)
    perm = torch.randperm(len(inputs))
    n_train = int(train_fraction * len(inputs))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return (inputs[train_idx], labels[train_idx],
            inputs[test_idx],  labels[test_idx])
