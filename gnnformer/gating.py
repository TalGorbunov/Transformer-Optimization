"""Gated attention (arXiv:2505.06708) as a detachable adapter on the frozen backbone.

Standalone module — nothing else in `gnnformer/` imports it, and it imports nothing from
`fencing`/`engine`/`carriers`, so attaching a gate cannot perturb THE method's masks.

The paper's gate is `Y' = Y * sigmoid(X W_theta)` with X the hidden states *after
pre-normalization*. In a Qwen2/Qwen2.5-VL decoder layer that is exactly the input to
`self_attn` (the layer applies `input_layernorm` before calling it), so X is taken from a
pre-hook on `self_attn` and never re-normalized.

Positions (their numbering), all five implemented:
  G1  after SDPA, before o_proj — gates what the QUERY reads out of the collapsed sum
  G2  on the value path        — gates what each SOURCE token writes into the sum
  G3  on the key path          — gates how attractive each source token is to attend to
  G4  on the query path        — gates what the reader matches on
  G5  after o_proj             — gates the whole attention block's contribution

G1/G2/G5 only rescale outputs or values: they cannot change WHICH tokens are attended to,
because a layer's output gate cannot alter that same layer's attention weights. **G3 and
G4 can** — keys and queries feed q.k, so gating them reshapes the softmax itself. G3 in
particular lets an irrelevant source token suppress its own attractiveness, which reduces
effective fan-in, and is therefore the one position that could plausibly act on a
CAPACITY-shaped bottleneck rather than an interference-shaped one. The paper found G3/G4/G5
worthless for LM loss (6.016 / 5.981 / 6.017 vs 6.026 baseline); that says nothing about
an aggregation-limited task.

Granularity is confounded with position unless you group by it — these are the
granularity-matched comparisons the arm design otherwise cannot make:
    512 scores/token : g2_literal (value)  vs  g3_key (key)
   3584 scores/token : g1_elementwise (post-SDPA) vs g4_query vs g5_output

Variants (Qwen2.5-VL-7B: hidden 3584, 28 q heads, 4 kv heads, head_dim 128):

    g1_headwise      [3584, 28]    one scalar per query head, after SDPA      100 k/layer
    g1_headshared    [3584, 1]     one scalar for all heads (their weak arm)  3.6 k/layer
    g1_elementwise   [3584, 3584]  per channel, after SDPA                    12.85 M/layer
    g2_literal       [3584, 512]   on v_proj's output — forcibly shared by    1.84 M/layer
                                   the 7 query heads of each KV head (GQA)

`g2_expanded` (a per-query-head write gate, after `repeat_kv`) needs a forward patch on
the attention module and is P4 of the campaign — deliberately not implemented here until
that phase is approved.

IDENTITY INIT IS MANDATORY and the naive version silently breaks learning:

    g = sigmoid(x @ W.T + b) / sigmoid(b)      with W = 0, b = +2.0

W=0 makes g exactly 1.0 at init (bit-for-bit identity forward), and b=+2 keeps
sigma'(2) = 0.105 so the gate has a healthy gradient. b=+6 gives sigma'(6) = 0.0025 and
the gate cannot learn inside our step budget. `b` is a fixed buffer, not a parameter:
with W=0 the ratio is identically 1 for any b, so b's gradient is 0 at init and it can
only ever act as the curvature knob it is. The chosen b0 is recorded in every checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

GATE_VARIANTS = ("g1_headwise", "g1_headshared", "g1_elementwise", "g2_literal",
                 "g3_key", "g4_query", "g5_output")
DEFAULT_GATE_B0 = 2.0

# Which module each variant hooks, and whether the gate reads that module's OWN input
# (which is already X) or needs X stashed from the attention pre-hook.
#   g2/g3/g4 hook v_proj/k_proj/q_proj — their input IS X, so they are self-contained.
#   g1 gates o_proj's INPUT (the SDPA output); g5 gates o_proj's OUTPUT. Both need the stash.
_SELF_INPUT_HOOK = {"g2_literal": "v_proj", "g3_key": "k_proj", "g4_query": "q_proj"}


def gate_out_features(variant: str, *, hidden: int, n_heads: int, n_kv: int, head_dim: int) -> int:
    """Number of gate scores produced per token by `variant`."""
    if variant == "g1_headwise":
        return n_heads
    if variant == "g1_headshared":
        return 1
    if variant == "g1_elementwise":
        return n_heads * head_dim
    if variant == "g2_literal":
        return n_kv * head_dim
    if variant == "g3_key":
        return n_kv * head_dim          # k_proj is KV-width, same as v_proj
    if variant == "g4_query":
        return n_heads * head_dim       # q_proj is full width
    if variant == "g5_output":
        return hidden                   # after o_proj
    raise ValueError(f"unknown gate variant {variant!r} (known: {GATE_VARIANTS})")


def gate_scores(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """sigmoid(x W^T + b) / sigmoid(b) — exactly 1.0 everywhere when W == 0."""
    return torch.sigmoid(x.float() @ W.T + b) / torch.sigmoid(b)


def apply_gate(y: torch.Tensor, g: torch.Tensor, variant: str, *, head_dim: int) -> torch.Tensor:
    """Multiply `y` [B, T, F] by gate scores `g` [B, T, out], broadcasting per variant."""
    if variant in ("g1_headwise", "g1_headshared"):
        b, t, f = y.shape
        return (y.view(b, t, -1, head_dim) * g.unsqueeze(-1)).view(b, t, f).to(y.dtype)
    return (y * g).to(y.dtype)


@dataclass
class Gate:
    """Attached gate: `.parameters()` for the optimizer, `.state()`/`.remove()` like Lora.

    `mean_scores()` is the mandatory instrumentation — a gate sitting at ~1.0 learned
    nothing, and an arm whose gate never moved is VOID, not a null result."""

    variant: str
    params: Dict[int, nn.Parameter]
    buffers: Dict[int, torch.Tensor]
    handles: List[Any]
    layer_ids: List[int]
    b0: float
    dims: Dict[str, int]
    _stats: Dict[int, List[float]] = field(default_factory=dict)

    def parameters(self) -> List[nn.Parameter]:
        return [self.params[li] for li in self.layer_ids]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "b0": self.b0,
            "layer_ids": list(self.layer_ids),
            "dims": dict(self.dims),
            "W": {str(li): self.params[li].detach().cpu() for li in self.layer_ids},
        }

    def reset_stats(self) -> None:
        # clear() and NOT `self._stats = {}` — the hooks close over THIS dict object, so
        # rebinding orphans it and every subsequent gate score is written somewhere
        # mean_scores() can never see it ("gate: no forwards recorded" for a gate that was
        # in fact training fine — caught in the 2026-08-07 P3 smokes).
        self._stats.clear()

    def mean_scores(self) -> Dict[int, float]:
        """Mean gate score per layer since the last reset ({} if no forward ran)."""
        return {li: (sum(v) / len(v)) for li, v in sorted(self._stats.items()) if v}

    def stats_line(self) -> str:
        ms = self.mean_scores()
        if not ms:
            return "gate: no forwards recorded"
        vals = list(ms.values())
        return ("gate mean/layer " + " ".join(f"L{li}:{v:.4f}" for li, v in ms.items())
                + f" | min {min(vals):.4f} max {max(vals):.4f} span {max(vals)-min(vals):.4f}")

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def attach_gate(
    layers: Any,
    layer_ids: Sequence[int],
    variant: str,
    *,
    hidden: int,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    device: Any = "cpu",
    b0: float = DEFAULT_GATE_B0,
    state: Optional[Dict[str, Any]] = None,
    track_stats: bool = True,
) -> Gate:
    """Attach `variant` gates to `layer_ids` of a decoder layer list.

    G1 needs two hooks per layer: a pre-hook on `self_attn` that stashes X (the
    post-pre-norm hidden states) and a pre-hook on `self_attn.o_proj` that gates its
    input (the SDPA output). G2 needs one forward hook on `self_attn.v_proj`, whose own
    input already IS X. `state` restores a saved `Gate.state()`."""
    if variant not in GATE_VARIANTS:
        raise ValueError(f"unknown gate variant {variant!r} (known: {GATE_VARIANTS})")
    n_out = gate_out_features(variant, hidden=hidden, n_heads=n_heads, n_kv=n_kv,
                              head_dim=head_dim)
    layer_ids = list(layer_ids)
    if state is not None:
        if state["variant"] != variant:
            raise ValueError(f"state is {state['variant']!r}, asked for {variant!r}")
        b0 = float(state.get("b0", b0))
    params: Dict[int, nn.Parameter] = {}
    buffers: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []
    stats: Dict[int, List[float]] = {}
    gate = Gate(variant=variant, params=params, buffers=buffers, handles=handles,
                layer_ids=layer_ids, b0=float(b0),
                dims={"hidden": hidden, "n_heads": n_heads, "n_kv": n_kv,
                      "head_dim": head_dim, "n_out": n_out},
                _stats=stats)

    for li in layer_ids:
        if state is not None:
            W = nn.Parameter(state["W"][str(li)].float().to(device))
        else:
            W = nn.Parameter(torch.zeros(n_out, hidden, device=device))
        b = torch.full((n_out,), float(b0), device=device)
        params[li] = W
        buffers[li] = b
        attn = layers[li].self_attn

        def record(li=li):
            def _rec(g: torch.Tensor) -> None:
                if track_stats:
                    stats.setdefault(li, []).append(float(g.detach().mean()))
            return _rec

        if variant in _SELF_INPUT_HOOK:
            # v_proj / k_proj / q_proj: the module's own input already IS X, so one
            # forward hook suffices and nothing has to be stashed.
            def mk_self(li=li, W=W, b=b, rec=record()):
                def hook(_m, inp, o):
                    g = gate_scores(inp[0], W, b)
                    rec(g)
                    return apply_gate(o, g, variant, head_dim=head_dim)
                return hook

            handles.append(getattr(attn, _SELF_INPUT_HOOK[variant])
                           .register_forward_hook(mk_self()))
        elif variant == "g5_output":
            # after o_proj: the gate still reads X, but o_proj's input is the SDPA
            # output, so X has to come from the attention pre-hook.
            stash5: Dict[str, torch.Tensor] = {}

            def mk_stash5(stash=stash5):
                def pre(_m, args, kwargs):
                    stash["x"] = args[0] if args else kwargs["hidden_states"]
                    return None
                return pre

            def mk_out(li=li, W=W, b=b, stash=stash5, rec=record()):
                def hook(_m, _inp, o):
                    x = stash.get("x")
                    if x is None:
                        return None
                    g = gate_scores(x, W, b)
                    rec(g)
                    return apply_gate(o, g, variant, head_dim=head_dim)
                return hook

            handles.append(attn.register_forward_pre_hook(mk_stash5(), with_kwargs=True))
            handles.append(attn.o_proj.register_forward_hook(mk_out()))
        else:
            stash: Dict[str, torch.Tensor] = {}

            def mk_stash(stash=stash):
                def pre(_m, args, kwargs):
                    x = args[0] if args else kwargs["hidden_states"]
                    stash["x"] = x
                    return None
                return pre

            def mk_o(li=li, W=W, b=b, stash=stash, rec=record()):
                def pre(_m, args):
                    y = args[0]
                    x = stash.get("x")
                    if x is None:  # o_proj called outside a full attention forward
                        return None
                    g = gate_scores(x, W, b)
                    rec(g)
                    return (apply_gate(y, g, variant, head_dim=head_dim),) + tuple(args[1:])
                return pre

            handles.append(attn.register_forward_pre_hook(mk_stash(), with_kwargs=True))
            handles.append(attn.o_proj.register_forward_pre_hook(mk_o()))
    return gate


# ------------------------------------------------------------------ checkpoint I/O

def save_gate_ckpt(path: Path, gate: Gate, **extra: Any) -> None:
    torch.save({"gate": gate.state(), **extra}, path)


def load_gate_state(path: Path) -> Dict[str, Any]:
    ck = torch.load(path, map_location="cpu")
    return ck["gate"] if "gate" in ck else ck
