import torch
import torch.nn as nn
from typing import Dict, List, Optional


class GrokMLP(nn.Module):
    def __init__(self, p=113, hidden_sizes=(200, 200)):
        super().__init__()
        self.p = p
        sizes = [2 * p] + list(hidden_sizes) + [p]
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        self.act = nn.ReLU()
        self.activations: Dict[str, torch.Tensor] = {}

    def forward(self, x, capture=False):
        if capture:
            self.activations = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                if capture:
                    self.activations[f'hidden_{i}'] = x.detach()
        return x  # raw logits [batch, p]


class GrokTransformer(nn.Module):
    """Single-layer Transformer. Input: integer tokens [x, y, =] where = has index p."""

    def __init__(self, p=113, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.p = p
        d_vocab = p + 1  # p values + '=' token

        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = nn.Parameter(torch.randn(3, d_model) * d_model ** -0.5)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_in  = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.act = nn.ReLU()

        self.unembed = nn.Linear(d_model, p, bias=False)

        self.mlp_activations: Optional[torch.Tensor] = None  # [batch, 3, d_mlp]
        self.activations: Dict[str, torch.Tensor] = {}

    def _attn(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.W_Q(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = self.W_K(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, Dh).transpose(1, 2)
        scale = Dh ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # [B,H,T,T]
        # causal mask: position i can only attend to j <= i
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)

    def forward(self, tokens, capture=False):
        # tokens: [batch, 3]
        x = self.embed(tokens) + self.pos_embed
        x = x + self._attn(x)

        pre_mlp = self.W_in(x)
        post_mlp = self.act(pre_mlp)
        if capture:
            self.mlp_activations = post_mlp.detach()
            self.activations = {'mlp': post_mlp[:, 2, :].detach()}  # at '=' position
        x = x + self.W_out(post_mlp)

        return self.unembed(x[:, 2, :])  # [batch, p] logits at '=' position
