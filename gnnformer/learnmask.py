"""LEARNMASK: learned discrete attention mask from the fence init
(outputs/learnmask/CAMPAIGN_BRIEF.md).

Gate logits attach to **(relation-type, layer)** — never to absolute positions — so a
mask learned at one N assembles at ANY length by lookup. Every (query q, key k) cell of
the carrier layout (prefix | per-frame blocks(vision+carrier) | tail = final question +
decode) is classified into exactly one relation:

  R1 block token -> own block (causal)             fence ON    1 channel
  R2 block token -> other block (incl. its carrier) fence OFF  6 Δ-bucket channels
  R3 carrier    -> own block                        fence ON    1
  R4 carrier    -> other-block carrier              fence OFF   6 Δ-buckets
  R5 carrier    -> other-block content              fence OFF   6 Δ-buckets
  R6 tail       -> frame content (TRUNC P0.1)       fence ON    1
  R7 tail       -> carriers (hide_cols)             fence OFF   1
  R8 any -> prefix, tail -> tail (causal), diag     anchor, never learnable

22 learnable channels/layer. Δ-buckets (block distance) {1, 2, 3-4, 5-8, 9-16, 17+}.

Parity contract (pinned by tests/test_fencing.py, bit-for-bit):
  - fence init, hard mode  == carriers.make_masks lo == fencing.build_block_mask(
      seq, blocks, hide_cols=cpos) at EVERY layer;
  - the deployed hand design (R4+R7 open at layers >= L_OPEN) is a point in gate space:
      hard gates hand_open(li, l_open) == make_masks lo (li < l_open) / hi (>= l_open);
  - both hold with e teacher-forced/decoded rows appended (== carriers.ext_mask).

Gradient-scale gotcha (brief): while TRAINING, closed learnable cells use the soft
forbid K=SOFT_FORBID=-30 (kills softmax mass in bf16, keeps d/dlogit sane); MASK_MIN
only in the hard-frozen eval mask. Non-learnable relations always use MASK_MIN.

The per-layer mask path lives HERE (gated_stack_logits), not in engine.py/fencing.py:
gates are trainable at all 28 layers so the cached-lo-phase trainer path cannot apply,
and the anchored engine code stays untouched. The forward mirrors
CarrierEngine.forward_logits exactly (extra-row positions, e_c injection order, fp32
masks, SDPA backend list); the P1 in-run parity check compares the two on real samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .constants import L_OPEN, MASK_MIN

Span = Tuple[int, int]

# ------------------------------------------------------------------ channel registry

DELTA_BUCKETS: Tuple[Tuple[int, Optional[int]], ...] = (
    (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, None))
SOFT_FORBID = -30.0
CELL_ANCHOR, CELL_FORBID = -1, -2


def delta_bucket(d: int) -> int:
    """Block-distance d>=1 -> bucket index 0..5 (log-ish, length-generalizing)."""
    if d < 1:
        raise ValueError(f"block distance must be >= 1, got {d}")
    for i, (lo, hi) in enumerate(DELTA_BUCKETS):
        if d >= lo and (hi is None or d <= hi):
            return i
    raise AssertionError("unreachable")


def _bucket_name(b: int) -> str:
    lo, hi = DELTA_BUCKETS[b]
    return f"Δ{lo}" if hi == lo else (f"Δ{lo}+" if hi is None else f"Δ{lo}-{hi}")


@dataclass(frozen=True)
class Channel:
    name: str        # e.g. "R4[Δ3-4]"
    rel: str         # "R1".."R7"
    bucket: Optional[int]
    fence_on: bool   # init value in the hand fence (build_block_mask semantics)


def _mk_channels() -> List[Channel]:
    out: List[Channel] = []
    for rel, bucketed, on in (("R1", False, True), ("R2", True, False),
                              ("R3", False, True), ("R4", True, False),
                              ("R5", True, False), ("R6", False, True),
                              ("R7", False, False)):
        if bucketed:
            out.extend(Channel(f"{rel}[{_bucket_name(b)}]", rel, b, on)
                       for b in range(len(DELTA_BUCKETS)))
        else:
            out.append(Channel(rel, rel, None, on))
    return out


CHANNELS: List[Channel] = _mk_channels()
N_CH = len(CHANNELS)                      # 22
_CH0 = {rel: min(i for i, c in enumerate(CHANNELS) if c.rel == rel)
        for rel in ("R1", "R2", "R3", "R4", "R5", "R6", "R7")}
FENCE_ON = torch.tensor([c.fence_on for c in CHANNELS])

# Arms (brief): S1 tail-only, S2 +carriers (headline), S3 free. S0 free-table is a
# separate P3 diagnostic (per-cell logits), not a relation-gate arm.
ARM_RELATIONS: Dict[str, Tuple[str, ...]] = {
    "s1": ("R6", "R7"),
    "s2": ("R4", "R5", "R6", "R7"),
    "s3": ("R1", "R2", "R3", "R4", "R5", "R6", "R7"),
}


def arm_learn_mask(arm: str) -> torch.Tensor:
    """[N_CH] bool: which channels train in this arm. S1=2, S2=14, S3=22 per layer."""
    rels = ARM_RELATIONS[arm]
    return torch.tensor([c.rel in rels for c in CHANNELS])


def fence_open() -> torch.Tensor:
    """[N_CH] bool: the fence-init hard gate values (== build_block_mask)."""
    return FENCE_ON.clone()


def hand_open(layer: int, l_open: int = L_OPEN) -> torch.Tensor:
    """[N_CH] bool: the deployed hand design at `layer` — fence + R4/R7 open >= l_open
    (== make_masks lo below l_open, hi at/above)."""
    o = FENCE_ON.clone()
    if layer >= l_open:
        o[_CH0["R4"]: _CH0["R4"] + len(DELTA_BUCKETS)] = True
        o[_CH0["R7"]] = True
    return o


# ------------------------------------------------------------------ cell classification

def _norm_spans(readers: Sequence[Any]) -> List[Span]:
    """Reader spec -> spans. Carriers are single positions (int -> (c, c+1));
    question replicas are multi-token (a, b) spans. One code path for both."""
    out: List[Span] = []
    for r in readers:
        if isinstance(r, int):
            out.append((r, r + 1))
        else:
            a, b = r
            out.append((int(a), int(b)))
    return out


def readers_of(d: Dict[str, Any]) -> List[Span]:
    """Sample record -> reader spans ('readers' if present, else carrier 'cpos')."""
    return _norm_spans(d["readers"] if "readers" in d else d["cpos"])


def relation_cell_map(
    seq: int,
    blocks: Sequence[Span],
    readers: Sequence[Any],
    fin_start: int,
    e: int = 0,
    row_chunk: int = 4096,
) -> torch.Tensor:
    """[seq+e, seq+e] int16 cell map: learnable channel id 0..N_CH-1, CELL_ANCHOR for
    always-open cells (R8 prefix cols, tail->tail causal), CELL_FORBID above the
    diagonal. `readers` = per-block reader spec: single carrier positions (ints) or
    question-replica (a, b) spans — reader_i must sit inside block_i. Appended rows
    (teacher-forced targets / decode) are tail rows, matching carriers.ext_mask
    semantics. Row-chunked so P4-scale maps stay bounded."""
    S = seq + e
    a0 = blocks[0][0]
    blk = torch.full((S,), -1, dtype=torch.long)
    for i, (a, b) in enumerate(blocks):
        blk[a:b] = i
    if not bool((blk[a0:fin_start] >= 0).all()):
        raise ValueError("blocks must tile [blocks[0][0], fin_start) contiguously")
    spans = _norm_spans(readers)
    car = torch.full((S,), -1, dtype=torch.long)
    for i, (ra, rb) in enumerate(spans):
        if int(blk[ra]) != i or int(blk[rb - 1]) != i:
            raise ValueError(f"reader {i} span ({ra},{rb}) is not inside block {i}")
        car[ra:rb] = i
    pos = torch.arange(S)
    is_pre, is_tail = pos < a0, pos >= fin_start
    is_car = car >= 0
    is_cont = (blk >= 0) & ~is_car
    nb = len(blocks)
    dtab = torch.tensor([0] + [delta_bucket(d) for d in range(1, nb)], dtype=torch.long)

    cm = torch.full((S, S), CELL_FORBID, dtype=torch.int16)
    kblk, kcar, kcont = blk.view(1, -1), is_car.view(1, -1), is_cont.view(1, -1)
    ktail, kpre = is_tail.view(1, -1), is_pre.view(1, -1)
    for r0 in range(0, S, row_chunk):
        r1 = min(r0 + row_chunk, S)
        out = torch.full((r1 - r0, S), CELL_FORBID, dtype=torch.int16)
        qblk = blk[r0:r1].view(-1, 1)
        qcar = is_car[r0:r1].view(-1, 1)
        qcont = is_cont[r0:r1].view(-1, 1)
        qtail = is_tail[r0:r1].view(-1, 1)
        both = (qblk >= 0) & (kblk >= 0)
        same = both & (qblk == kblk)
        other = both & (qblk != kblk)
        bidx = dtab[(qblk - kblk).clamp(min=0, max=nb - 1)].to(torch.int16)
        out.masked_fill_(kpre.expand(r1 - r0, S), CELL_ANCHOR)
        out.masked_fill_(qtail & ktail, CELL_ANCHOR)
        out.masked_fill_(qtail & kcar, _CH0["R7"])
        out.masked_fill_(qtail & kcont, _CH0["R6"])
        out.masked_fill_(qcar & same, _CH0["R3"])
        out = torch.where(qcar & other & kcar, _CH0["R4"] + bidx, out)
        out = torch.where(qcar & other & kcont, _CH0["R5"] + bidx, out)
        out.masked_fill_(qcont & same, _CH0["R1"])
        out = torch.where(qcont & other, _CH0["R2"] + bidx, out)
        lower = pos.view(1, -1) <= pos[r0:r1].view(-1, 1)
        cm[r0:r1] = torch.where(lower, out, torch.tensor(CELL_FORBID, dtype=torch.int16))
    return cm


# ------------------------------------------------------------------ mask assembly

def mask_parts(
    cell_map: torch.Tensor,
    learn: torch.Tensor,
    frozen_open: torch.Tensor,
    forbid: float = MASK_MIN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (base fp32, lidx int64), both [S, S].

    base carries the NON-learnable structure: 0 on anchors, frozen-open channels and
    all learnable cells; `forbid` (MASK_MIN — non-learnable relations never soften) on
    causal-forbidden cells and frozen-closed channels. lidx maps each learnable cell to
    its slot in the arm's learnable-channel vector, and everything else to the sentinel
    slot n_learn (a constant 1.0 = open, zero additive contribution)."""
    cm = cell_map.long()
    n_learn = int(learn.sum())
    slot = torch.full((N_CH,), n_learn, dtype=torch.long)
    slot[learn] = torch.arange(n_learn)
    is_ch = cm >= 0
    ch = cm.clamp(min=0)
    lidx = torch.where(is_ch, slot[ch], torch.full_like(cm, n_learn))
    frozen_off = is_ch & ~learn[ch] & ~frozen_open[ch]
    closed = (cm == CELL_FORBID) | frozen_off
    base = torch.where(closed, torch.tensor(forbid, dtype=torch.float32),
                       torch.tensor(0.0, dtype=torch.float32))
    return base, lidx


def assemble_mask(base: torch.Tensor, lidx: torch.Tensor, g_learn: torch.Tensor,
                  K: float) -> torch.Tensor:
    """base + (1 - g)[cell]·K — differentiable in g_learn ([n_learn] floats in [0,1]).
    Train: K=SOFT_FORBID with sampled/soft gates. Hard-frozen eval: K=MASK_MIN with
    hard 0/1 gates reproduces deploy semantics exactly (0 + 1·MASK_MIN is exact fp32)."""
    g_ext = torch.cat([g_learn, g_learn.new_ones(1)])
    return base + (1.0 - g_ext[lidx]) * K


def hard_mask(cell_map: torch.Tensor, open_ch: torch.Tensor,
              forbid: float = MASK_MIN) -> torch.Tensor:
    """Deploy-semantics fp32 mask for a hard [N_CH] bool gate assignment."""
    base, _ = mask_parts(cell_map, torch.zeros(N_CH, dtype=torch.bool), open_ch,
                         forbid=forbid)
    return base


def hard_mask_lut(cell_map_dev: torch.Tensor, open_ch: torch.Tensor,
                  forbid: float = MASK_MIN) -> torch.Tensor:
    """GPU-friendly hard mask straight from a device-resident int16 cell map — no
    int64 lidx, no CPU->GPU mask transfer. For large-N transfer evals (N=64 masks are
    ~1.8 GB fp32 each; assemble per layer on the fly, never hold 28). Values match
    mask_parts/hard_mask exactly (0 / forbid). The transient .long() is the largest
    allocation (8 bytes/cell); chunk rows upstream if a layout ever exceeds memory."""
    lut = torch.full((N_CH + 2,), forbid, dtype=torch.float32,
                     device=cell_map_dev.device)
    lut[1] = 0.0  # CELL_ANCHOR (-1) -> index 1
    lut[2:][open_ch.to(cell_map_dev.device)] = 0.0
    return lut[(cell_map_dev.long() + 2)]


def hard_masks_by_layer(cell_map: torch.Tensor, open_table: torch.Tensor
                        ) -> List[torch.Tensor]:
    """Per-layer hard masks for an [N_CH, n_layers] bool table (memoized on distinct
    columns — the hand design has only two)."""
    memo: Dict[Tuple[bool, ...], torch.Tensor] = {}
    out = []
    for li in range(open_table.shape[1]):
        key = tuple(open_table[:, li].tolist())
        if key not in memo:
            memo[key] = hard_mask(cell_map, open_table[:, li])
        out.append(memo[key])
    return out


def hand_open_table(n_layers: int, l_open: int = L_OPEN) -> torch.Tensor:
    """[N_CH, n_layers] bool: the deployed hand design (fence + R4/R7 >= l_open)."""
    return torch.stack([hand_open(li, l_open) for li in range(n_layers)], dim=1)


def make_masks_spans(seq: int, blocks: Sequence[Span], readers: Sequence[Any],
                     fin_start: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(lo, hi) for span readers — the span generalization of carriers.make_masks:
    lo = build_block_mask with ALL reader positions hidden; hi = lo + reader_i rows
    read earlier readers' tokens + tail rows read all reader tokens. For single-token
    spans this equals carriers.make_masks BIT-FOR-BIT (pinned in tests) — the anchored
    carrier construction is the special case."""
    from .fencing import build_block_mask

    spans = _norm_spans(readers)
    hide = [p for a, b in spans for p in range(a, b)]
    lo = build_block_mask(seq, blocks, hide_cols=hide)
    hi = lo.clone()
    cols = [torch.arange(a, b) for a, b in spans]
    for i in range(1, len(spans)):
        rows = torch.arange(spans[i][0], spans[i][1])
        earlier = torch.cat(cols[:i])
        hi[rows.unsqueeze(1), earlier.unsqueeze(0)] = 0.0
    if cols:
        hi[fin_start:, torch.cat(cols)] = 0.0
    return lo, hi


def fence_open_table(n_layers: int) -> torch.Tensor:
    """[N_CH, n_layers] bool: the fence init at every layer (== build_block_mask)."""
    return fence_open().view(-1, 1).repeat(1, n_layers)


# ------------------------------------------------------------------ gate module

ESTIMATORS = ("ste", "soft", "st-gumbel", "hard-concrete")  # E1..E4 (brief menu)


class MaskGates(nn.Module):
    """Relation×layer gate logits + estimator (E1 det-STE / E2 soft-anneal /
    E3 ST-Gumbel default / E4 hard-concrete). Only the arm's learnable channels are
    parameters; frozen channels are handled by mask_parts' base (always MASK_MIN
    semantics). Init: ±init_logit by fence value, so hard mode at init == the fence."""

    def __init__(self, n_layers: int, arm: str = "s2", estimator: str = "st-gumbel",
                 init_logit: float = 2.0, init_open: bool = False):
        super().__init__()
        if arm not in ARM_RELATIONS:
            raise ValueError(f"unknown arm {arm!r} (known: {sorted(ARM_RELATIONS)})")
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator!r} (known: {ESTIMATORS})")
        self.arm, self.estimator = arm, estimator
        self.n_layers, self.init_logit = n_layers, float(init_logit)
        self.init_open = bool(init_open)
        learn = arm_learn_mask(arm)
        self.register_buffer("learn", learn)
        self.register_buffer("fence_on_learn", FENCE_ON[learn].clone())
        self.channel_names = [c.name for c, m in zip(CHANNELS, learn.tolist()) if m]
        # init_open=True: prune-from-open — ALL learnable gates start open and the
        # deviation penalty prunes; surviving edges = the necessity answer. Chosen
        # after the 2026-08-14 sweep showed a COORDINATION BARRIER at the fence
        # init (independent per-gate samples can't discover jointly-valuable
        # openings; gates retreat into the fence).
        sign = (torch.ones_like(self.fence_on_learn, dtype=torch.float32)
                if self.init_open else torch.where(self.fence_on_learn, 1.0, -1.0))
        init = (self.init_logit * sign).view(-1, 1).repeat(1, n_layers)
        self.logits = nn.Parameter(init)
        self.register_buffer("init_sign", sign.clone())

    @property
    def n_learn(self) -> int:
        return self.logits.shape[0]

    def gate_table(self, tau: float = 1.0, mode: str = "train") -> torch.Tensor:
        """[n_learn, n_layers] gate values in [0,1]. mode='hard': deterministic
        1[logit>0], no grad (the hard-frozen eval mask). mode='train': estimator
        forward (E1/E3 emit hard 0/1 values with straight-through gradients)."""
        l = self.logits
        if mode == "hard":
            with torch.no_grad():
                return (l > 0).float()
        if mode != "train":
            raise ValueError(f"unknown mode {mode!r}")
        if self.estimator == "soft":                      # E2
            return torch.sigmoid(l / tau)
        # ST forward values must be EXACTLY 0/1: (hard - soft.detach()) + soft is exact
        # (Sterbenz: 1-s exact for s in (0.5, 1], -s exact for hard=0), unlike
        # hard + soft - soft.detach() which leaves fp rounding residue.
        if self.estimator == "ste":                       # E1
            soft = torch.sigmoid(l)
            return (l > 0).float() - soft.detach() + soft
        u = torch.rand_like(l).clamp_(1e-6, 1 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)            # logistic = Gumbel diff
        if self.estimator == "st-gumbel":                 # E3 (Jang et al. '17 §2.2)
            soft = torch.sigmoid((l + noise) / tau)
            return (soft > 0.5).float() - soft.detach() + soft
        gamma, zeta = -0.1, 1.1                           # E4 (Louizos '17)
        s = torch.sigmoid((l + noise) / tau)
        return (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)

    def p_open(self) -> torch.Tensor:
        """sigma(logits) [n_learn, n_layers] — penalty term + heatmap."""
        return torch.sigmoid(self.logits)

    def deviation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(sum p_open over fence-OFF learnable, sum 1-p_open over fence-ON learnable):
        opening must pay for itself; closing (S3) must pay too."""
        p = self.p_open()
        off = ~self.fence_on_learn
        return p[off].sum(), (1.0 - p[self.fence_on_learn]).sum()

    def full_p_open(self) -> torch.Tensor:
        """[N_CH, n_layers] P(open) with frozen channels at their exact fence value —
        the headline heatmap (figure 1)."""
        out = FENCE_ON.to(self.logits.device).float().view(-1, 1).repeat(1, self.n_layers)
        out[self.learn] = self.p_open().detach()
        return out

    def hard_open_table(self) -> torch.Tensor:
        """[N_CH, n_layers] bool: the hard-frozen mask (frozen channels at fence,
        learnable at 1[logit>0]) — feed to hard_masks_by_layer for deploy-semantics
        eval/decode (memoizes distinct layer columns)."""
        full = fence_open().view(-1, 1).repeat(1, self.n_layers)
        full[self.learn.cpu()] = self.logits.detach().cpu() > 0
        return full

    def flips(self) -> int:
        """Learnable (channel, layer) gates whose hard value differs from the fence."""
        return int(((self.logits > 0) != (self.init_sign.view(-1, 1) > 0)).sum())

    def stats_line(self) -> str:
        p = self.p_open().detach()
        rels = sorted({c.split("[")[0] for c in self.channel_names})
        per = " ".join(
            f"{r}:{p[[i for i, c in enumerate(self.channel_names) if c.split('[')[0] == r]].mean():.2f}"
            for r in rels)
        dmax = float((self.logits.detach() - self.init_logit * self.init_sign.view(-1, 1))
                     .abs().max())
        return f"p_open[{per}] flips {self.flips()}/{self.logits.numel()} max|Δlogit| {dmax:.2f}"

    def state(self) -> Dict[str, Any]:
        return {"logits": self.logits.detach().cpu(), "arm": self.arm,
                "estimator": self.estimator, "init_logit": self.init_logit,
                "init_open": self.init_open,
                "n_layers": self.n_layers, "channel_names": list(self.channel_names)}

    @classmethod
    def from_state(cls, st: Dict[str, Any]) -> "MaskGates":
        g = cls(int(st["n_layers"]), arm=st["arm"], estimator=st["estimator"],
                init_logit=float(st["init_logit"]),
                init_open=bool(st.get("init_open", False)))
        with torch.no_grad():
            g.logits.copy_(st["logits"])
        return g


# ------------------------------------------------------------------ S0 free table

class FreeTableGates(nn.Module):
    """S0 diagnostic (brief): PER-CELL logits, per layer, on ONE fixed token layout —
    the oracle upper bound for any mask over the same edge scope + the vocabulary
    audit. Learnable cells = the cells covered by `scope_arm`'s relations (default
    s2), everything else frozen at the fence. Requires every sample to share the
    layout bit-for-bit (mmred_hf@512 seq8 layouts are uniform — verified 2026-08-12);
    class target only (e=0). Its numbers are NEVER a headline: the table cannot
    transfer across lengths and can memorize positions.

    Duck-types the MaskGates surface the trainer uses (gate_table / p_open /
    deviation / full_p_open / hard_open_table is N/A -> hard masks come from
    hard_layer_masks_ft). Audit heatmap = per-cell p_open averaged per declared
    relation channel."""

    def __init__(self, cell_map: torch.Tensor, n_layers: int, scope_arm: str = "s2",
                 estimator: str = "st-gumbel", init_logit: float = 2.0,
                 share_layers: bool = False):
        super().__init__()
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator!r}")
        self.estimator, self.init_logit = estimator, float(init_logit)
        self.n_layers = n_layers
        self.share_layers = bool(share_layers)  # ONE mask for all layers (prof.
        self.arm = (f"s0shared[{scope_arm}]"    # suggestion 2026-08-15): /28 params,
                    if share_layers else f"s0[{scope_arm}]")  # depth-independent
        learn_ch = arm_learn_mask(scope_arm)
        cm = cell_map.long()
        sel = (cm >= 0) & learn_ch[cm.clamp(min=0)]
        self.register_buffer("cell_map", cell_map)
        self.register_buffer("flat_idx", sel.view(-1).nonzero(as_tuple=True)[0])
        ch = cm.view(-1)[self.flat_idx]
        self.register_buffer("cell_ch", ch)  # declared relation channel per cell
        self.register_buffer("fence_on_learn", FENCE_ON[ch].clone())
        # base: fence values on frozen cells, 0 on learnable cells
        frozen_open = FENCE_ON.clone()
        base, _ = mask_parts(cell_map, learn_ch, frozen_open)
        self.register_buffer("base", base)
        sign = torch.where(self.fence_on_learn, 1.0, -1.0)
        cols = 1 if self.share_layers else n_layers
        self.logits = nn.Parameter(
            (self.init_logit * sign).view(-1, 1).repeat(1, cols))
        self.channel_names = [f"cell[{c.name}]" for c in CHANNELS]

    @property
    def n_learn(self) -> int:
        return self.logits.shape[0]

    def gate_table(self, tau: float = 1.0, mode: str = "train") -> torch.Tensor:
        t = MaskGates.gate_table(self, tau=tau, mode=mode)  # same estimators
        # layer-shared: ONE draw per cell, the identical mask at every layer
        # (expand keeps autograd — layer gradients accumulate into the one logit)
        return t.expand(-1, self.n_layers) if self.share_layers else t

    def p_open(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def deviation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.p_open()
        off = ~self.fence_on_learn
        return p[off].sum(), (1.0 - p[self.fence_on_learn]).sum()

    def layer_mask(self, g_col: torch.Tensor, K: float) -> torch.Tensor:
        """base + scatter((1-g)·K) at the learnable cells — differentiable in g_col."""
        S = self.base.shape[0]
        m = self.base.clone().view(-1)
        m = m.index_put((self.flat_idx,), (1.0 - g_col) * K, accumulate=True)
        return m.view(S, S)

    def full_p_open(self) -> torch.Tensor:
        """Audit heatmap: mean per-cell P(open) grouped by declared relation."""
        out = FENCE_ON.to(self.logits.device).float().view(-1, 1).repeat(1, self.n_layers)
        p = self.p_open().detach()
        for c in range(N_CH):
            sel = self.cell_ch == c
            if bool(sel.any()):
                out[c] = p[sel].mean(0)
        return out

    def flips(self) -> int:
        sign = torch.where(self.fence_on_learn, 1.0, -1.0)
        return int(((self.logits > 0) != (sign.view(-1, 1) > 0)).sum())

    def stats_line(self) -> str:
        p = self.p_open().detach()
        on, off = self.fence_on_learn, ~self.fence_on_learn
        return (f"cells {self.n_learn} p_open[fence-on:{p[on].mean():.2f} "
                f"fence-off:{p[off].mean():.2f}] flips {self.flips()}/{self.logits.numel()}")

    def state(self) -> Dict[str, Any]:
        return {"logits": self.logits.detach().cpu(), "arm": self.arm,
                "estimator": self.estimator, "init_logit": self.init_logit,
                "share_layers": self.share_layers,
                "n_layers": self.n_layers, "cell_map": self.cell_map.cpu(),
                "flat_idx": self.flat_idx.cpu()}


def layout_key(d: Dict[str, Any]) -> Tuple:
    """Hashable layout identity — S0 requires every sample to share it exactly."""
    return (d["seq"], tuple(d["blocks"]), tuple(_norm_spans(readers_of(d))), d["fin"])


def freetable_tf_logits(eng: Any, d: Dict[str, Any], ft: FreeTableGates, *,
                        tau: float, mode: str, grad_ckpt: bool = False) -> torch.Tensor:
    """Class-target forward (e=0) under the per-cell table. K per the same
    gradient-scale rule: SOFT_FORBID in train, MASK_MIN hard."""
    gt = ft.gate_table(tau=tau, mode=mode)
    K = MASK_MIN if mode == "hard" else SOFT_FORBID

    def mask_fn(li: int, S: int) -> torch.Tensor:
        return ft.layer_mask(gt[:, li], K)

    return gated_stack_logits(eng, d, [], mask_fn, grad_ckpt=grad_ckpt)


# ------------------------------------------------------------------ gated forward

def gated_stack_logits(
    eng: Any,
    d: Dict[str, Any],
    tgt_ids: Sequence[int],
    layer_mask_fn: Callable[[int, int], torch.Tensor],
    *,
    grad_ckpt: bool = False,
    return_h: bool = False,
) -> Any:
    """Full-stack teacher-forced forward with a per-layer mask -> logits for rows
    seq-1 .. seq+e (e+1 rows: row k predicts tgt_ids[k] for k < e; the FINAL row is
    the post-target continuation — the row CarrierEngine.forward_logits(extra=tgt)
    returns, kept so parity compares identical rows of identical-shape forwards;
    CE callers slice [:-1]). return_h=True additionally returns the normed hidden
    states [1, S, D] — the parity instrument re-heads the last row 1-D (GEMV), because
    the engine's h[0, -1] head call and this function's row-matrix head call hit
    different matmul kernels whose bf16 reduction order differs (~1e-2 logit noise).

    layer_mask_fn(li, S) returns the ADDITIVE 2D fp32 mask for layer li (assembled on
    the right device; called inside the checkpointed block so recompute re-assembles).
    Geometry mirrors CarrierEngine.forward_logits: e_c injected (detached — gates-only
    training) after target embeddings are appended; appended rows get incremental
    positions; masks fp32; SDPA backend list EFFICIENT->MATH."""
    from .engine import SDPA_BACKENDS
    from torch.nn.attention import sdpa_kernel

    dev = eng.dev
    e = len(tgt_ids)
    seq = d["seq"]
    S = seq + e
    emb = d["emb"].to(dev).unsqueeze(0)
    if e:
        ext = eng.text_model.embed_tokens(torch.tensor([list(tgt_ids)], device=dev))
        emb = torch.cat([emb, ext.to(emb.dtype)], dim=1)
    emb = emb.clone()
    if d.get("cpos") and eng.e_c is not None:  # carrier scaffold only; replicas
        stack = eng.e_c.unsqueeze(0).repeat(len(d["cpos"]), 1).to(torch.bfloat16)
        emb[0, torch.tensor(d["cpos"], device=dev)] = stack.detach()
    pos = d["pos"].to(dev)
    if e:
        inc = torch.arange(1, e + 1, device=dev).view(1, 1, e)
        pos = torch.cat([pos, pos[:, :, -1:] + inc], dim=2)
    cos_, sin_ = eng.text_model.rotary_emb(emb, pos)
    pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))

    def _blk(hh: torch.Tensor, li: int) -> torch.Tensor:
        m = layer_mask_fn(li, S).view(1, 1, S, S)
        with sdpa_kernel(SDPA_BACKENDS):
            return eng.layers[li](hh, attention_mask=m, position_embeddings=pe)[0]

    h = emb
    use_ckpt = grad_ckpt and torch.is_grad_enabled()
    for li in range(eng.n_layers):
        if use_ckpt:
            h = torch.utils.checkpoint.checkpoint(_blk, h, li, use_reentrant=False)
        else:
            h = _blk(h, li)
    h = eng.text_model.norm(h)
    lg = eng.head(h[0, seq - 1: S])
    return (lg, h) if return_h else lg


def gated_tf_logits(
    eng: Any,
    d: Dict[str, Any],
    tgt_ids: Sequence[int],
    gates: MaskGates,
    *,
    tau: float,
    mode: str,
    grad_ckpt: bool = False,
) -> torch.Tensor:
    """Teacher-forced logits under the LEARNED gates. mode='train': one estimator draw
    per (channel, layer) per step, soft forbid K=SOFT_FORBID; mode='hard': frozen
    1[logit>0] gates with K=MASK_MIN (deploy semantics)."""
    e = len(tgt_ids)
    cm = relation_cell_map(d["seq"], d["blocks"], readers_of(d), d["fin"], e=e)
    base, lidx = mask_parts(cm, gates.learn.cpu(), fence_open())
    base, lidx = base.to(eng.dev), lidx.to(eng.dev)
    gt = gates.gate_table(tau=tau, mode=mode)
    K = MASK_MIN if mode == "hard" else SOFT_FORBID

    def mask_fn(li: int, S: int) -> torch.Tensor:
        return assemble_mask(base, lidx, gt[:, li], K)

    return gated_stack_logits(eng, d, tgt_ids, mask_fn, grad_ckpt=grad_ckpt)


def locate_replica_layout(ids: List[int], tok: Any, question: str, NF: int
                          ) -> Optional[Tuple[List[Span], int]]:
    """-> (replica_spans[NF], fin_start) or None on any mismatch.

    Replicas = the FIRST NF occurrences of the bare-question needle — each follows
    <|vision_end|>, a clean token boundary. fin = the END of the last replica: the
    tail (canonical count prompt / final question + decode) is everything after it.
    The tail text must be SPACE-separated from the last replica: a "\\n" separator
    fuses with the replica's '?' into one token ('?\\n') and breaks its match
    (measured 2026-08-12). Inside the canonical tail the question reads
    "Question: How ..." whose ' How' token differs from the bare needle's 'How', so
    the tail adds no spurious bare matches."""
    from .fencing import find_subseq

    n0 = tok(question, add_special_tokens=False).input_ids
    occ0 = find_subseq(ids, n0)
    if len(occ0) < NF:
        return None
    fin = occ0[NF - 1] + len(n0)
    if any(o < fin for o in occ0[NF:]) or fin >= len(ids):
        return None
    return [(o, o + len(n0)) for o in occ0[:NF]], fin


def prepare_sample_replicas(
    eng: Any,
    frames: List[Any],
    question: str,
    *,
    gold: Any,
    task: str,
    resize: int = 512,
    tail_style: str = "canonical",
    answer_hint: str = "",
    answer_prime: str = "",
) -> Optional[Dict[str, Any]]:
    """Replica-scaffold sample prep: [frame_i + question-replica] x N + final question
    — the A3 one-forward supply construction with ZERO trained components (no carrier
    token, no e_c, no LoRA) and, per Tal 2026-08-12, NO leading question: per-frame
    conditioning comes from the replicas themselves.

    Layout: prefix = chat preamble only | block_i = [vision tokens + replica_i] |
    tail = ' '+final question + decode (see locate_replica_layout for why the
    separator is a space, not a newline). Positions: fencing.reset_positions verbatim
    (no carrier sequential override — that was carrier-specific). Fence init hides
    replica spans (hide_cols), exactly the legacy replica construction. Returns the
    sample record (readers, no cpos) or None on any structural check failure (caller
    counts a skip).

    tail_style 'canonical' (default): the tail is data.build_count_prompt — the
    PROMPT-CRITICAL wording ('Respond with a single integer ... Question: {q}
    Answer: ') that makes the FROZEN model emit the digit at the read position;
    without it em = 0.000 in every regime (2026-08-12 ep0 rows — the model opens a
    sentence instead). 'plain': just ' '+question (the failed variant, kept for
    ablation). answer_hint appends a user-side sentence after the tail text;
    answer_prime appends assistant-side text after the generation prompt."""
    from .data import build_count_prompt
    from .fencing import frame_blocks, reset_positions
    from .runtime import image_token_groups

    NF = len(frames)
    if resize > 0:
        frames = [f.resize((resize, resize)) for f in frames]
    content: List[Dict[str, Any]] = []
    for f in frames:
        content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": question})
    if tail_style == "canonical":
        fin_text = " " + build_count_prompt(question, NF)
    elif tail_style == "plain":
        fin_text = " " + question
    else:
        raise ValueError(f"unknown tail_style {tail_style!r}")
    if answer_hint:
        fin_text += " " + answer_hint
    content.append({"type": "text", "text": fin_text})
    inputs = eng.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(eng.dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if answer_prime:
        pids = eng.tok(answer_prime, add_special_tokens=False).input_ids
        ext = torch.tensor([pids], device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], ext], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(ext)], dim=1)
    ids = inputs["input_ids"][0].tolist()
    seq = len(ids)
    fg = image_token_groups(
        inputs["input_ids"][0].cpu(), expected_num_frames=NF, processor=eng.processor
    )
    vstarts = [p for p, t in enumerate(ids) if t == eng.vision_start_id]
    loc = locate_replica_layout(ids, eng.tok, question, NF)
    if len(fg) != NF or len(vstarts) != NF or loc is None:
        return None
    readers, fin_start = loc
    blocks = frame_blocks(vstarts, fin_start)
    for i, (ra, rb) in enumerate(readers):  # replica_i must sit inside block_i
        a, b = blocks[i]
        if not (a <= ra and rb <= b):
            return None
    with torch.no_grad():
        base_pos, _ = eng.rope_fn(
            inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        pos = reset_positions(base_pos, blocks, fin_start).clone()
        emb = eng.text_model.embed_tokens(inputs["input_ids"])
        img = eng.model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
        im_mask = inputs["input_ids"][0] == eng.model.config.image_token_id
        emb = emb.clone()
        emb[0, im_mask] = img.to(emb.dtype)
    return {
        "emb": emb[0].to(torch.bfloat16),
        "ids": ids,
        "pos": pos,
        "readers": readers,
        "blocks": blocks,
        "fin": fin_start,
        "seq": seq,
        "gold": gold,
        "task": task,
    }


def gated_greedy_digits(
    eng: Any,
    d: Dict[str, Any],
    layer_mask_fn_full: Callable[[int], torch.Tensor],
    *,
    max_tokens: int = 6,
) -> Tuple[Optional[int], int, str]:
    """Greedy EMITTED digit decode under per-layer masks (the campaign metric after
    Tal's 2026-08-12 call: no scratchpad anywhere, score the number the model emits;
    nothing in the context contains the answer, so it cannot be solved by copying).

    layer_mask_fn_full(li) returns the layer's 2D mask built at S_max = seq+max_tokens;
    each step slices to the current length (cell classification is prefix-stable, so
    the slice equals the smaller map — appended rows are tail rows). Recompute-from-
    scratch per step, mirroring engine.decode_answer's stop rule (first non-digit).
    -> (parsed int or None, 0-9-restricted first-token argmax, decoded text)."""
    toks: List[int] = []
    first_digit = -1
    with torch.no_grad():
        for step in range(max_tokens):
            def mfn(li: int, S: int) -> torch.Tensor:
                return layer_mask_fn_full(li)[:S, :S]

            lg = gated_stack_logits(eng, d, toks, mfn)[-1]
            if step == 0:
                first_digit = int(torch.stack([lg[t] for t in eng.digit_ids]).argmax())
            t = int(lg.argmax())
            if not eng.tok.decode([t]).strip().isdigit():
                break
            toks.append(t)
    text = eng.tok.decode(toks).strip()
    return (int(text) if text.isdigit() else None), first_digit, text


def handfence_tf_logits(eng: Any, d: Dict[str, Any], tgt_ids: Sequence[int],
                        l_open: int = L_OPEN, return_h: bool = False) -> Any:
    """Teacher-forced logits under the deployed hand fence expressed as hard gates
    (the P1 reference row + the in-run parity instrument vs engine.forward_logits)."""
    e = len(tgt_ids)
    cm = relation_cell_map(d["seq"], d["blocks"], readers_of(d), d["fin"], e=e)
    masks = hard_masks_by_layer(cm, hand_open_table(eng.n_layers, l_open))
    masks = [m.to(eng.dev) for m in masks]
    return gated_stack_logits(eng, d, tgt_ids, lambda li, S: masks[li],
                              return_h=return_h)
