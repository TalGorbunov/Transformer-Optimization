"""condmask mask algebra — ONE formula everywhere:

    mask[b, h] = base + g[b, h] * K * template          (per layer)

base      [S,S] fp32: 0 on open cells, MASK_MIN on causal-forbidden + padding.
template  [S,S] bool: True exactly on frame-row cross-block cells (q in block_i,
          k in block_j, i != j, both inside the frame region, k <= q). Prefix columns
          and tail rows are NEVER in the template — blockwise frame tokens keep the
          question prefix, and the tail stays fully causal by design.
g         [B,H] in {0,1} (or ST-Gumbel relaxed): 1 = blockwise, 0 = full causal.
K         SOFT_FORBID (-30) during training (gradient sanity), MASK_MIN at eval.

Fixed arms are the g≡0 / g≡1 special cases of the same assembly, so "fixed blockwise"
vs "all-gates-blockwise" parity is bit-exact and testable.

Padding (right-pad to S_max): pad columns closed for every row; pad rows attend ONLY
themselves (NaN guard). Mask dtype = query dtype (bf16 for a bf16 text model; MASK_MIN
survives bf16 by design — gnnformer/constants.py:7-8).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import MASK_MIN  # -65504.0

SOFT_FORBID = -30.0  # same value/rationale as gnnformer.learnmask.SOFT_FORBID


def layout_mask_parts(rec: Dict[str, Any], S: int,
                      device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (base [S,S] fp32, template [S,S] bool) for one sample padded to S.

    Works unchanged for decode-extended records: appended tokens raise rec["seq"] but
    sit at positions >= rec["fin"], i.e. tail rows — fully causal, never in template.
    """
    seq = rec["seq"]
    if seq > S:
        raise ValueError(f"seq {seq} > padded size {S}")
    pos = torch.arange(S, device=device)
    q, k = pos.view(-1, 1), pos.view(1, -1)
    real_q, real_k = q < seq, k < seq

    causal_open = (k <= q) & real_q & real_k
    base = torch.full((S, S), MASK_MIN, dtype=torch.float32, device=device)
    base[causal_open] = 0.0
    # pad rows: self only (NaN guard) — pad cols stay closed for real rows
    base[~causal_open.any(dim=1)] = MASK_MIN
    pad_diag = pos[seq:]
    base[pad_diag, pad_diag] = 0.0

    blk = torch.full((S,), -1, dtype=torch.long, device=device)
    for i, (a, b) in enumerate(rec["blocks"]):
        blk[a:b] = i
    in_frames_q, in_frames_k = (blk[q.view(-1)] >= 0).view(-1, 1), (blk[k.view(-1)] >= 0).view(1, -1)
    cross = blk.view(-1, 1) != blk.view(1, -1)
    template = in_frames_q & in_frames_k & cross & (k <= q)
    return base, template


def batch_mask_parts(recs: List[Dict[str, Any]], S: int, device: Any = "cpu"
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack per-sample parts -> (bases [B,S,S] fp32, templates [B,S,S] bool)."""
    parts = [layout_mask_parts(r, S, device) for r in recs]
    return (torch.stack([p[0] for p in parts]),
            torch.stack([p[1] for p in parts]))


def assemble_batch_masks(bases: torch.Tensor, templates: torch.Tensor,
                         g: torch.Tensor, K: float,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[B,S,S] + [B,H] -> [B,H,S,S] additive mask, differentiable in g.

    Structurally-closed cells stay MASK_MIN regardless of K (repo law: non-learnable
    cells never soften); only template cells move with the gates."""
    B, H = g.shape
    m = bases.unsqueeze(1) + g.view(B, H, 1, 1) * K * templates.unsqueeze(1).float()
    return m.to(dtype)


def fixed_gates(B: int, H: int, regime: str, device: Any = "cpu") -> torch.Tensor:
    """g for the fixed arms/reference rows: 'full' = all-open, 'blockwise' = all-fenced."""
    if regime == "full":
        return torch.zeros(B, H, device=device)
    if regime == "blockwise":
        return torch.ones(B, H, device=device)
    raise ValueError(f"unknown regime {regime!r} (known: full, blockwise)")
