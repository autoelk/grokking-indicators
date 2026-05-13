import itertools
import torch
from torch.utils.data import TensorDataset

OPERATIONS = {
    'add':      lambda x, y, p: (x + y) % p,
    'subtract': lambda x, y, p: (x - y) % p,
    'multiply': lambda x, y, p: (x * y) % p,
    'x2xyy2':  lambda x, y, p: (x**2 + x*y + y**2) % p,
    'x3xy2y' : lambda x, y, p: (x**3 + (x*y)**2 + y) % p,
}

S5_SIZE = 120  # |S_5| = 5! = 120

_s5_table = None  # module-level cache


def _build_s5_table():
    """120×120 composition table for S_5. (pi ∘ pj)[k] = pi[pj[k]]."""
    perms = list(itertools.permutations(range(5)))
    idx = {p: i for i, p in enumerate(perms)}
    table = [[0] * 120 for _ in range(120)]
    for i, pi in enumerate(perms):
        for j, pj in enumerate(perms):
            table[i][j] = idx[tuple(pi[pj[k]] for k in range(5))]
    return table


def _get_s5_table():
    global _s5_table
    if _s5_table is None:
        _s5_table = _build_s5_table()
    return _s5_table


def _build_full_s5():
    table = _get_s5_table()
    n = S5_SIZE
    inputs, labels = [], []
    for i in range(n):
        for j in range(n):
            x_oh = torch.zeros(n); x_oh[i] = 1.0
            y_oh = torch.zeros(n); y_oh[j] = 1.0
            inputs.append(torch.cat([x_oh, y_oh]))
            labels.append(table[i][j])
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)


def _build_full(p, operation):
    if operation == 's5':
        return _build_full_s5()
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
    """Full n^2 dataset in canonical (x,y) order. Used for Fourier metric computation."""
    inputs, labels = _build_full(p, operation)
    return inputs, labels


def make_token_dataset(p=113, operation='add', train_fraction=0.3, seed=42):
    """Integer token format [x, y, sep] for the Transformer model."""
    if operation == 's5':
        p = S5_SIZE
        table = _get_s5_table()
        tokens = [torch.tensor([i, j, p], dtype=torch.long)
                  for i in range(p) for j in range(p)]
        labels = [table[i][j] for i in range(p) for j in range(p)]
    else:
        op_fn = OPERATIONS[operation]
        tokens = [torch.tensor([x, y, p], dtype=torch.long)
                  for x in range(p) for y in range(p)]
        labels = [op_fn(x, y, p) for x in range(p) for y in range(p)]

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
