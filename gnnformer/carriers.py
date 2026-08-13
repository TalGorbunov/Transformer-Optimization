"""Carrier-layer machinery: the lo/hi mask pair, truncation column algebra, the
hand-rolled LoRA, and checkpoint I/O.

Architecture (one forward, frozen 4-bit backbone):
  layers 0..L_OPEN-1 : fenced blocks — carrier_i reads only {prefix+question, frame_i, self}
  layers L_OPEN..end : fence OPENS between carriers (causal) + tail attends all carriers
  trainable          : e_c (one shared carrier embedding) + LoRA(q/k/v/o) on layers >= L_OPEN

Checkpoint schema (carrier_layer_best.pt, unchanged from legacy so all existing ckpts load):
  e_c [D] · lora {"<layer>.<proj>": (A [r,din], B [dout,r])} · l_open · rank · [alpha]
  · epoch · acc · [scratchpad, scratchpad_format, running_tally, pos_couple, jitter_gap,
  truncate_at]. NOTE: the legacy cached trainer never stored alpha — default it to 16.
Distilled-carrier schema (carrier_best.pt): e_c · [e_extra] · head_w · head_b · dprime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .constants import MASK_MIN, ROOMS
from .fencing import build_block_mask

Span = Tuple[int, int]
DEFAULT_LORA_ALPHA = 16.0


# ----------------------------------------------------------------------- lo/hi masks

def make_masks(seq: int, blocks: Sequence[Span], cpos: Sequence[int], fin_start: int):
    """(mask_lo, mask_hi): lo = fenced blocks (carrier reads prefix+question, own frame,
    itself); hi = lo + carrier_i attends earlier carriers (causal) + tail attends all
    carriers. Rebuild lazily per step in training (two seq^2 masks per sample don't fit RAM
    at pooled-data scale)."""
    mask_lo = build_block_mask(seq, blocks, hide_cols=list(cpos))
    mask_hi = mask_lo.clone()
    ct = torch.tensor(list(cpos), dtype=torch.long)
    for i in range(1, len(cpos)):
        mask_hi[cpos[i], ct[:i]] = 0.0
    mask_hi[fin_start:, ct] = 0.0
    return mask_lo, mask_hi


def keep_cols(seq: int, blocks: Sequence[Span], cpos: Sequence[int]) -> List[int]:
    """Surviving columns for physical frame-drop: [prefix+question]+[carriers]+[tail].
    Everything outside frame blocks survives; inside a block only its carrier does."""
    inblk = [False] * seq
    for a, b in blocks:
        for i in range(a, b):
            inblk[i] = True
    cset = set(cpos)
    return [i for i in range(seq) if (not inblk[i]) or i in cset]


def frame_cols(seq: int, blocks: Sequence[Span], cpos: Sequence[int]) -> List[int]:
    """Complement of keep_cols within the blocks: the frame tokens (incl. vision markers)
    that --drop-frame-kv hides from decoded rows."""
    cset = set(cpos)
    out: List[int] = []
    for a, b in blocks:
        out.extend(i for i in range(a, b) if i not in cset)
    return out


def truncated_masks(keep: Sequence[int], cpos: Sequence[int]):
    """Direct construction of the truncated (lo_t, hi_t) over the keep sequence — no dense
    seq^2 intermediate. hi_t == plain causal; lo_t == causal with carrier COLUMNS hidden
    from every other row (each carrier still sees itself). Equal to index-selecting the
    dense masks — pinned by tests/test_carrier_masks.py."""
    k = len(keep)
    idx = {p: j for j, p in enumerate(keep)}
    car = torch.tensor([idx[c] for c in cpos], dtype=torch.long)
    m = torch.zeros(k, k, dtype=torch.float16)
    m.masked_fill_(torch.triu(torch.ones(k, k, dtype=torch.bool), 1), MASK_MIN)
    hi_t = m.clone()
    lo_t = m
    lo_t[:, car] = MASK_MIN
    lo_t[car, car] = 0.0
    return lo_t, hi_t


def ext_mask(m: torch.Tensor, e: int) -> torch.Tensor:
    """Append e decoded/teacher-forced rows: each reads like the last tail row + causal
    over the appended rows."""
    if e == 0:
        return m
    s0 = m.shape[0]
    big = torch.full((s0 + e, s0 + e), MASK_MIN, dtype=m.dtype)
    big[:s0, :s0] = m
    for j in range(e):
        r = s0 + j
        big[r, :s0] = m[s0 - 1]
        big[r, s0 : r + 1] = 0.0
    return big


# ----------------------------------------------------------------------------- LoRA

@dataclass
class Lora:
    """Hand-rolled LoRA over q/k/v/o_proj of layers >= l_open (hook-based; B zero-init so
    step 0 == no-LoRA). Keep hold of `handles` and call remove() to detach."""

    params: Dict[Tuple[int, str], Tuple[nn.Parameter, nn.Parameter]]
    handles: List[Any]
    l_open: int
    rank: int
    alpha: float

    def parameters(self) -> List[nn.Parameter]:
        return [p for ab in self.params.values() for p in ab]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            f"{li}.{nm}": (A.detach().cpu(), B.detach().cpu())
            for (li, nm), (A, B) in self.params.items()
        }

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def attach_lora(
    layers: Any,
    l_open: int,
    *,
    rank: int,
    alpha: float = DEFAULT_LORA_ALPHA,
    device: Any,
    state: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Lora:
    """Attach LoRA hooks to q/k/v/o_proj of layers l_open..end. `state` restores a saved
    checkpoint's A/B; otherwise A~N(0, 0.01), B=0."""
    scale = alpha / rank
    params: Dict[Tuple[int, str], Tuple[nn.Parameter, nn.Parameter]] = {}
    handles: List[Any] = []
    for li in range(l_open, len(layers)):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(layers[li].self_attn, nm)
            if state is not None:
                A0, B0 = state[f"{li}.{nm}"]
                A = nn.Parameter(A0.float().to(device))
                B = nn.Parameter(B0.float().to(device))
            else:
                A = nn.Parameter(torch.randn(rank, mod.in_features, device=device) * 0.01)
                B = nn.Parameter(torch.zeros(mod.out_features, rank, device=device))
            params[(li, nm)] = (A, B)

            def mk(A=A, B=B):
                def hook(_m, inp, o):
                    x = inp[0]
                    return o + (scale * (x.float() @ A.T) @ B.T).to(o.dtype)

                return hook

            handles.append(mod.register_forward_hook(mk()))
    return Lora(params=params, handles=handles, l_open=l_open, rank=rank, alpha=alpha)


# ------------------------------------------------------------------ checkpoint I/O

@dataclass
class CarrierLayerCkpt:
    e_c: torch.Tensor
    lora_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    l_open: int
    rank: int
    alpha: float
    scratchpad: bool
    scratchpad_format: str
    pos_couple: bool
    epoch: Optional[int] = None
    acc: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def load_carrier_layer_ckpt(path: Path) -> CarrierLayerCkpt:
    ck = torch.load(path, map_location="cpu")
    known = {"e_c", "lora", "l_open", "rank", "alpha", "epoch", "acc", "scratchpad",
             "scratchpad_format", "pos_couple"}
    return CarrierLayerCkpt(
        e_c=ck["e_c"],
        lora_state=ck["lora"],
        l_open=int(ck["l_open"]),
        rank=int(ck["rank"]),
        alpha=float(ck.get("alpha", DEFAULT_LORA_ALPHA)),  # cached trainer never stored it
        scratchpad=bool(ck.get("scratchpad")),
        scratchpad_format=ck.get("scratchpad_format") or "poslist",
        pos_couple=bool(ck.get("pos_couple")),
        epoch=ck.get("epoch"),
        acc=ck.get("acc"),
        extra={k: v for k, v in ck.items() if k not in known},
    )


def save_carrier_layer_ckpt(
    path: Path,
    *,
    e_c: torch.Tensor,
    lora: Lora,
    epoch: int,
    acc: float,
    scratchpad: bool,
    scratchpad_format: str,
    **extra: Any,
) -> None:
    torch.save(
        {
            "e_c": e_c.detach().cpu(),
            "lora": lora.state(),
            "l_open": lora.l_open,
            "rank": lora.rank,
            "alpha": lora.alpha,
            "epoch": epoch,
            "acc": acc,
            "scratchpad": scratchpad,
            "scratchpad_format": scratchpad_format,
            **extra,
        },
        path,
    )


def load_distilled_carrier(path: Path) -> Dict[str, Any]:
    """carrier_best.pt from the token-distill trainer (e_c/e_extra/head_w/head_b/dprime)."""
    return torch.load(path, map_location="cpu")


def room_mean_embedding(tokenizer: Any, embed_tokens: Any, rooms: Sequence[str] = ROOMS) -> torch.Tensor:
    """Default e_c init: mean input embedding of the room words (last token each)."""
    rows = []
    for r in rooms:
        tid = tokenizer(" " + r, add_special_tokens=False).input_ids
        rows.append(embed_tokens.weight[tid[-1]].float())
    return torch.stack(rows).mean(0)
