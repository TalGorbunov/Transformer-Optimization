"""condmask GateNet: input-conditioned per-(layer, head) gate logits + hard ST-Gumbel.

The mask decision is conditioned on the ENTIRE input: masked-mean-pooled input
embeddings -> MLP -> logits [L, H]. Final layer weight is ZERO-init and its bias carries
the ablation init (+1.0 blockwise / U(-1,1) random), so gates are input-independent at
step 0 and the conditioning grows with training. Barrier = |init bias| = 1.0 — small on
purpose (the S0 audit: a 2.0 barrier vs a ~2.3 travel ceiling made flips near-impossible).

st_gumbel forward values are EXACTLY 0/1 via the Sterbenz-exact pattern copied from
gnnformer/learnmask.py:371-381 ((hard - soft.detach()) + soft).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GateNet(nn.Module):
    """pooled input [B, d] -> gate logits [B, L, H]. 1 = blockwise, 0 = full causal."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, hidden: int = 256,
                 init: str = "blockwise", init_bias: float = 1.0, seed: int = 0,
                 norm_input: bool = False):
        super().__init__()
        if init not in ("blockwise", "random"):
            raise ValueError(f"unknown init {init!r} (known: blockwise, random)")
        self.n_layers, self.n_heads, self.init = n_layers, n_heads, init
        self.norm_input = bool(norm_input)
        # norm_input: LayerNorm on the pooled input — bounds logit drift speed (the
        # vlm smoke measured max|Δlogit| 574 without it). Default False here so v1
        # text-grid ckpts stay loadable; enable for v2 runs via --gate-norm-input.
        pre = [nn.LayerNorm(d_model)] if self.norm_input else []
        self.net = nn.Sequential(
            *pre, nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, n_layers * n_heads))
        last = self.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            if init == "blockwise":
                last.bias.fill_(init_bias)
            else:
                g = torch.Generator().manual_seed(seed)
                last.bias.copy_((torch.rand(last.bias.shape, generator=g) * 2 - 1)
                                * init_bias)
        # init snapshot: the step-0 hard decision (input-independent), for flip telemetry
        self.register_buffer("init_logits", last.bias.detach().clone().view(n_layers,
                                                                            n_heads))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled.float()).view(-1, self.n_layers, self.n_heads)


def st_gumbel(logits: torch.Tensor, tau: float, mode: str = "train") -> torch.Tensor:
    """Hard 0/1 gates. mode='train': logistic-noise ST-Gumbel, gradient through the soft
    relaxation, forward values exactly {0,1}. mode='hard': deterministic 1[logit>0], no
    grad — deploy semantics (still input-conditioned: logits come from GateNet(input))."""
    if mode == "hard":
        with torch.no_grad():
            return (logits > 0).float()
    if mode != "train":
        raise ValueError(f"unknown mode {mode!r}")
    u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
    noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((logits + noise) / tau)
    return (soft > 0.5).float() - soft.detach() + soft


def pool_embeddings(emb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B,S,D] x [B,S] bool -> [B,D] fp32 masked mean, DETACHED (gradients reach GateNet
    through its own parameters only, never back into the embedding table)."""
    w = valid.float().unsqueeze(-1)
    return ((emb.float() * w).sum(1) / w.sum(1).clamp(min=1.0)).detach()


def tau_schedule(step: int, total: int, tau0: float = 2.0, tau1: float = 0.5,
                 hold_frac: float = 0.3) -> float:
    """Hold tau0 for the first hold_frac of training (exploration), then exp-anneal."""
    hold = int(total * hold_frac)
    if step <= hold or total <= hold + 1:
        return tau0
    t = (step - hold) / max(total - 1 - hold, 1)
    return tau0 * (tau1 / tau0) ** t
