"""Co-Former: two-pass select-and-repack — the Co-GNN adaptation that rewires BOTH
edges and positions (the transformer-specific piece GNNs get for free).

Pass 1 (selector): frozen backbone's layer-k states under full attention -> per-frame
features -> RelGateTF-style head -> KEEP logit per frame (ST-Gumbel hard).
Pass 2 (answer): dropped frames are cut from the attention graph in BOTH directions
(self-attend only) AND parked at far positions; KEPT frames get canonical COMPACTED
positions (their rank among kept frames), the tail follows contiguously — the
computation over kept content is geometrically identical to a short sequence
(measured basis: oracle-deleted 1.00 vs oracle-masked-in-place 0.25 @N=64).

keep-all == plain full attention with the ORIGINAL positions (frames are contiguous),
so the A2 control is the keep-all special case — parity assertable.

In two-pass repacking, Co-GNN's 4-action space collapses to KEEP/DROP per frame
(a dropped frame has no pass-2 self-updates to preserve); LISTEN/BROADCAST asymmetry
belongs to the in-pass variant.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import MASK_MIN

PARK = 20000  # dropped frames parked here (Qwen2.5 context 32k — safe)


class Selector(nn.Module):
    """[q, f_1..f_F] summaries -> transformer -> KEEP logit per frame.
    init_bias +1.0 => keep-all at step 0 (== full attention, parity-assertable)."""

    def __init__(self, d_model: int, d_g: int = 256, depth: int = 2,
                 init_bias: float = 1.0, seed: int = 0):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_g))
        enc = nn.TransformerEncoderLayer(d_g, nhead=4, dim_feedforward=2 * d_g,
                                         batch_first=True, norm_first=True,
                                         dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, num_layers=depth)
        self.head = nn.Linear(d_g, 1)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(init_bias)

    def forward(self, q_feat: torch.Tensor, f_feats: torch.Tensor,
                f_valid: torch.Tensor) -> torch.Tensor:
        B = f_feats.shape[0]
        toks = torch.cat([self.proj(q_feat.float()).unsqueeze(1),
                          self.proj(f_feats.float())], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                     device=f_valid.device), ~f_valid], dim=1)
        h = self.tf(toks, src_key_padding_mask=pad)
        return self.head(h[:, 1:]).squeeze(-1)          # [B, F] keep logits


def repack_positions(rec: Dict[str, Any], keep: torch.Tensor, S: int,
                     device: Any) -> torch.Tensor:
    """Canonical positions for the kept computational graph: prefix unchanged; kept
    frame (rank r) placed contiguously after the prefix; dropped frames parked at
    PARK + original offset; tail follows the last kept token; appended decode rows
    continue the tail. keep-all reproduces arange exactly (frames are contiguous)."""
    pos = torch.zeros(S, dtype=torch.long, device=device)
    p = rec["prefix_end"]
    pos[:p] = torch.arange(p, device=device)
    cur = p
    for i, (a, b) in enumerate(rec["blocks"]):
        w = b - a
        if bool(keep[i]):
            pos[a:b] = torch.arange(cur, cur + w, device=device)
            cur += w
        else:
            pos[a:b] = torch.arange(PARK + a, PARK + b, device=device)
    tail_w = S - rec["fin"]
    pos[rec["fin"]:] = torch.arange(cur, cur + tail_w, device=device)
    return pos


def repack_mask(base: torch.Tensor, region: torch.Tensor, col_frame: torch.Tensor,
                keep: torch.Tensor, K: float,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[S,S] additive mask: dropped frames cut from the graph in both directions
    (OR-field over 1-keep), kept frames + prefix + tail untouched. Differentiable in
    keep (ST). -> [1, S, S] (head-uniform)."""
    Fn = keep.shape[0]
    gpad = torch.cat([1.0 - keep, keep.new_zeros(1)])
    idx = col_frame.clamp(min=-1) % (Fn + 1)
    gc = gpad[idx]                                       # [S]
    orf = gc.unsqueeze(0) + gc.unsqueeze(1) - gc.unsqueeze(0) * gc.unsqueeze(1)
    return (base + K * orf * region.float()).unsqueeze(0).to(dtype)


def oracle_keep(rec: Dict[str, Any], device: Any) -> torch.Tensor:
    """[F] 1.0 for relevant frames (precomputed question-derived rec['rel'])."""
    rel = rec.get("rel") or [c == rec["q_char"] for c in rec["frame_chars"]]
    return torch.tensor([1.0 if r else 0.0 for r in rel], device=device)


class PairSelector(nn.Module):
    """N-INVARIANT selector: per-frame pairwise (question x frame) MLP on
    position-free features — no cross-frame attention, no N-dependence anywhere.
    The TF Selector's keep(rel) degraded 0.99 -> 0.66 as N grew (measured 2026-08-18):
    the selection problem inherited the length-generalization problem. Relevance here
    is a pairwise property, so the N-invariant parameterization loses nothing."""

    def __init__(self, d_model: int, hidden: int = 256, init_bias: float = 1.0,
                 seed: int = 0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(3 * d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(init_bias)

    def forward(self, q_feat: torch.Tensor, f_feats: torch.Tensor,
                f_valid: torch.Tensor) -> torch.Tensor:
        B, F, _ = f_feats.shape
        q = self.ln_q(q_feat.float()).unsqueeze(1).expand(-1, F, -1)
        f = self.ln_f(f_feats.float())
        return self.net(torch.cat([q, f, q * f], dim=-1)).squeeze(-1)   # [B, F]


def shared_positions(rec: Dict[str, Any], S: int, device: Any) -> torch.Tensor:
    """LABEL-FREE single-pass variant (user proposal): every frame positioned AS IF IT
    WERE THE FIRST — per-frame encoding becomes length-invariant by construction, no
    selector needed. Pairs with the blockwise mask (frames isolated, tail reads all):
    aliased positions are safe only because cross-frame edges are closed. Order
    information across frames is destroyed — acceptable for order-invariant reductions
    (counting), not for first/last-style questions. The readout still aggregates N
    items, so the fan-in capacity limit remains — expect good-not-oracle at long N."""
    pos = torch.zeros(S, dtype=torch.long, device=device)
    p = rec["prefix_end"]
    pos[:p] = torch.arange(p, device=device)
    w_max = 0
    for a, b in rec["blocks"]:
        w = b - a
        pos[a:b] = torch.arange(p, p + w, device=device)
        w_max = max(w_max, w)
    tail_w = S - rec["fin"]
    pos[rec["fin"]:] = torch.arange(p + w_max, p + w_max + tail_w, device=device)
    return pos


def answer_row_frame_mass(layer, h: torch.Tensor, pe, blocks_list, S: int,
                          row: torch.Tensor = None) -> torch.Tensor:
    """Attention mass from the answer row (last position) to each frame, computed
    manually from layer's q/k (SDPA never exposes weights). h [B,S,D] = input to this
    layer. -> [B, F_max] mean-over-heads mass. Differentiable (the aux loss trains the
    attention itself)."""
    hn = layer.input_layernorm(h)
    at = layer.self_attn
    B, S_, _ = hn.shape
    hd = pe[0].shape[-1]        # RoPE cos last dim == head_dim, model-agnostic
    nh = at.q_proj.weight.shape[0] // hd
    q = at.q_proj(hn).view(B, S_, nh, hd).transpose(1, 2)
    k = at.k_proj(hn).view(B, S_, -1, hd).transpose(1, 2)
    cos, sin = pe

    def rope(x):
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    if row is None:
        qa = q[:, :, -1]      # LEGACY: last row — a PAD row under %8 padding.
        # Trained AS arms are self-consistent (aux + eval read the same row:
        # a learned end-position read). Pass `row`=ans for the TRUE answer row
        # (required for zero-shot/training-free readoffs and probes).
    else:
        qa = q[torch.arange(B, device=h.device), :, row]
    sc = (qa.unsqueeze(2) * k).sum(-1) / hd ** 0.5          # [B, H, S]
    w = torch.softmax(sc.float(), dim=-1)
    Fm = max(len(bl) for bl in blocks_list)
    mass = torch.zeros(B, Fm, device=h.device)
    for b, bl in enumerate(blocks_list):
        for i, (a, e) in enumerate(bl):
            mass[b, i] = w[b, :, a:e].sum(-1).mean()
    return mass


def answer_row_token_mass(layer, h: torch.Tensor, pe) -> torch.Tensor:
    """Per-TOKEN attention mass from the answer row: (mean-heads [B,S], per-head
    [B,H,S]). Same manual q/k computation as answer_row_frame_mass — token granularity
    (not all tokens of a frame are equally important; user hypothesis 2026-08-19)."""
    hn = layer.input_layernorm(h)
    at = layer.self_attn
    B, S, _ = hn.shape
    hd = pe[0].shape[-1]        # RoPE cos last dim == head_dim, model-agnostic
    nh = at.q_proj.weight.shape[0] // hd
    q = at.q_proj(hn).view(B, S, nh, hd).transpose(1, 2)
    k = at.k_proj(hn).view(B, S, -1, hd).transpose(1, 2)
    cos, sin = pe

    def rope(x):
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * cos.unsqueeze(1) + torch.cat([-x2, x1], dim=-1) * sin.unsqueeze(1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    qa = q[:, :, -1]
    sc = (qa.unsqueeze(2) * k).sum(-1) / hd ** 0.5
    w = torch.softmax(sc.float(), dim=-1)                    # [B, H, S]
    return w.mean(1), w


def token_repack(keep_t: torch.Tensor) -> torch.Tensor:
    """Order-preserving position compaction at TOKEN level: kept tokens get contiguous
    positions (cumsum-1), dropped tokens parked. [S] bool -> [S] long."""
    pos = torch.cumsum(keep_t.long(), dim=0) - 1
    park = PARK + torch.arange(keep_t.shape[0], device=keep_t.device)
    return torch.where(keep_t, pos.clamp(min=0), park)


def token_mask(keep_t: torch.Tensor, K: float, dtype: torch.dtype,
               base: torch.Tensor, headwise: torch.Tensor = None) -> torch.Tensor:
    """Dropped tokens become inert (self-only, both directions). Optional per-head
    refinement: additional column cuts per head (mask-only — safe ON TOP of the global
    repack, which already fixes the geometry). -> [1 or H, S, S]."""
    S = keep_t.shape[0]
    drop = (1.0 - keep_t.float())
    orf = drop.unsqueeze(0) + drop.unsqueeze(1) - drop.unsqueeze(0) * drop.unsqueeze(1)
    eye = torch.eye(S, device=keep_t.device)
    m = base + K * orf * (1.0 - eye)
    if headwise is None:
        return m.unsqueeze(0).to(dtype)
    # headwise [H, S]: 1 = this head also cuts this column (for kept tokens only)
    hw = headwise * keep_t.unsqueeze(0)          # never re-cut what global handles
    mh = m.unsqueeze(0) + K * hw.unsqueeze(1) * (1.0 - eye).unsqueeze(0)
    return mh.to(dtype)


def soft_repack_positions(rec: Dict[str, Any], g: torch.Tensor, S: int,
                          device: Any) -> torch.Tensor:
    """DIFFERENTIABLE repack: float positions = prefix + cumsum of gate-weighted
    chunk widths. This is the gradient path masks never had — d(position)/d(gate)
    != 0, so 'dropping junk moves evidence closer together' is finally visible to
    backprop (the measured payoff: masked-in-place .25 vs repacked 1.00 @N=64).
    At g in {0,1}, prefix/kept/tail positions match repack_positions EXACTLY;
    dropped chunks differ (parked there, left in flow here) but their columns are
    masked, so the forward is identical — the relaxation is exact at the vertices."""
    pos = torch.zeros(S, dtype=torch.float32, device=device)
    p = rec["prefix_end"]
    pos[:p] = torch.arange(p, device=device, dtype=torch.float32)
    cur = torch.zeros((), device=device) + float(p)
    for i, (a, b) in enumerate(rec["blocks"]):
        w = b - a
        pos[a:b] = cur + torch.arange(w, device=device, dtype=torch.float32)
        cur = cur + g[i] * w
    tail_w = S - rec["fin"]
    pos[rec["fin"]:] = cur + torch.arange(tail_w, device=device,
                                          dtype=torch.float32)
    return pos


def soft_rope(rotary, pos_f: torch.Tensor, out_dtype: torch.dtype):
    """cos/sin WITH gradient through positions — HF's rotary.forward is
    @torch.no_grad(). Mirrors the HF layout exactly (cat(freqs, freqs),
    * attention_scaling); parity vs rotary() asserted at trainer startup."""
    inv = rotary.inv_freq.to(device=pos_f.device, dtype=torch.float32)
    scale = float(getattr(rotary, "attention_scaling", 1.0))
    freqs = pos_f.float().unsqueeze(-1) * inv            # [B, S, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos() * scale).to(out_dtype), (emb.sin() * scale).to(out_dtype)


class SoftGate(nn.Module):
    """Two scalars: g_i = sigmoid((mass_i * n_frames - a) / tau). The model's own
    answer-row attention IS the router; training tunes the operating point (and,
    through undetached mass, the attention itself). a0=0.5 = AS3's fixed threshold,
    so step 0 sits at the known-good operating point."""

    def __init__(self, a0: float = 0.5, tau0: float = 0.25):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a0)))
        self.log_tau = nn.Parameter(torch.tensor(float(tau0)).log())

    def forward(self, mass_n: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((mass_n - self.a) / self.log_tau.exp().clamp(min=0.02))


def log_gate_mask(base: torch.Tensor, region: torch.Tensor,
                  col_frame: torch.Tensor, g: torch.Tensor, K: float,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Soft-gate mask, log parameterization: reading frame j is biased by log(g_j)
    on region cells — attention to a half-open frame scales ~g_j. The -K*(1-g)
    OR-field is effectively binary below g~0.9 (g=0.65 -> -10.5 -> e^-10.5:
    invisible); log keeps the gradient alive across the whole (0,1) range.
    Exact at vertices: g=1 -> base; g=0 -> base + K*region (== repack_mask cols)."""
    Fn = g.shape[0]
    lg = torch.where(g > 0, g.clamp(min=1e-30).log().clamp(min=K),
                     torch.full_like(g, K))
    gpad = torch.cat([lg, g.new_zeros(1)])
    idx = col_frame.clamp(min=-1) % (Fn + 1)
    return (base + gpad[idx].unsqueeze(0) * region.float()).unsqueeze(0).to(dtype)
