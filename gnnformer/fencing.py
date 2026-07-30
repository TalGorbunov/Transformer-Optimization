"""The fencing mechanism: block-diagonal attention fences, per-block M-RoPE position
reset, hook plumbing, and per-frame message recompute.

This is the ONE canonical implementation of the thesis intervention (Exp A3:
replicas/carriers + blockfence + posreset, d' 6.34 -> 13.54 @n900). It unifies two
legacy copies that had drifted apart:
  - legacy/experiments/glstm/carrier_token_distill.py  (build_block_mask, reset_positions)
  - legacy/experiments/glstm/replica_carrier_probe.py  (inline incremental mask rules)
tests/test_fencing.py proves the two constructions agree on the deployed config and
pins the mask invariants (incl. the *documented* semantics that tail/decode rows still
see frames — the TRUNC campaign's P0.1 finding).

Mask convention: additive float mask, 0 = allowed, MASK_MIN = forbidden, shaped
[seq, seq] (row = query, col = key); view(1, 1, seq, seq) before injection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .constants import MASK_MIN

Span = Tuple[int, int]  # half-open [start, end)


# --------------------------------------------------------------------------- masks

def build_block_mask(seq: int, blocks: Sequence[Span], hide_cols: Sequence[int]) -> torch.Tensor:
    """Canonical fence mask: causal + full block-diagonal fence + globally hidden columns.

    blocks: per-frame spans [vision_start_i, next vision_start / final-question start) —
            each contains the frame's vision markers, image tokens, and its replica/carrier.
    hide_cols: positions hidden from every row OUTSIDE their own block (replica spans or
            carrier positions), so the joint computation is undisturbed by them.
    Rows outside all blocks (prefix, final question, decode tail) stay causal over
    everything except hide_cols — i.e. the tail DOES see all frames (by design; the
    readout reads frames at decode — RESULTS.md [2026-07-25] TRUNC).
    """
    m = torch.zeros(seq, seq, dtype=torch.float32)
    m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MASK_MIN)
    if hide_cols:
        m[:, torch.tensor(sorted(hide_cols), dtype=torch.long)] = MASK_MIN
    for i, (a, b) in enumerate(blocks):
        rows = torch.arange(a, b)
        # own block: plain causal (re-opens hidden columns within the block)
        causal = torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1)
        blk = torch.zeros(b - a, b - a)
        blk.masked_fill_(causal, MASK_MIN)
        m[a:b, a:b] = blk
        for j, (a2, b2) in enumerate(blocks):
            if j != i:
                m[rows.unsqueeze(1), torch.arange(a2, b2).unsqueeze(0)] = MASK_MIN
    return m


def build_replica_probe_mask(
    seq: int,
    rep_spans: Sequence[Span],
    vis_by_frame: Sequence[torch.Tensor],
    *,
    fence_frames: bool = False,
    fence_blocks: bool = False,
    blocks: Optional[Sequence[Span]] = None,
) -> torch.Tensor:
    """The probe's incremental mask construction (kept for the ablation ladder).

    Base: replicas invisible to non-replica rows; replica_i rows forbid other frames'
    image tokens and other replicas. +fence_frames: frame rows see only {prefix, own
    frame}. +fence_blocks: full block-diagonal over `blocks` (closes the vision-marker
    leak). With both flags this equals build_block_mask(seq, blocks, hide_cols=replica
    spans) — asserted by tests/test_fencing.py.
    """
    if fence_blocks and (not fence_frames or blocks is None):
        raise ValueError("fence_blocks requires fence_frames and explicit blocks")
    m = torch.zeros(seq, seq, dtype=torch.float32)
    m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MASK_MIN)
    rep_tok = torch.zeros(seq, dtype=torch.bool)
    for a, b in rep_spans:
        rep_tok[a:b] = True
    nonrep_rows = (~rep_tok).nonzero(as_tuple=True)[0]
    m[nonrep_rows.unsqueeze(1), rep_tok.nonzero(as_tuple=True)[0].unsqueeze(0)] = MASK_MIN
    all_vis = torch.cat(list(vis_by_frame))
    for i, (a, b) in enumerate(rep_spans):
        rows = torch.arange(a, b)
        own_vis = set(vis_by_frame[i].tolist())
        forb = torch.tensor([int(p) for p in all_vis.tolist() if p not in own_vis], dtype=torch.long)
        if forb.numel():
            m[rows.unsqueeze(1), forb.unsqueeze(0)] = MASK_MIN
        for j, (a2, b2) in enumerate(rep_spans):
            if j != i:
                m[rows.unsqueeze(1), torch.arange(a2, b2).unsqueeze(0)] = MASK_MIN
    if fence_frames:
        for i in range(len(vis_by_frame)):
            rows_f = vis_by_frame[i]
            own = set(vis_by_frame[i].tolist())
            forb_f = torch.tensor([int(p) for p in all_vis.tolist() if p not in own], dtype=torch.long)
            if forb_f.numel():
                m[rows_f.unsqueeze(1), forb_f.unsqueeze(0)] = MASK_MIN
    if fence_blocks:
        for i, (ai, bi) in enumerate(blocks):
            rows_b = torch.arange(ai, bi)
            for j, (aj, bj) in enumerate(blocks):
                if j != i:
                    m[rows_b.unsqueeze(1), torch.arange(aj, bj).unsqueeze(0)] = MASK_MIN
    return m


# ----------------------------------------------------------------------- positions

def reset_positions(base_pos: torch.Tensor, blocks: Sequence[Span], fin_start: int) -> torch.Tensor:
    """Per-block M-RoPE reset (Exp A2): every fenced block gets block 0's position ids
    (PCW-style reuse — safe because blocks cannot attend each other), and the final
    question continues right after block 0's max position.

    base_pos: (3, 1, seq) from the model's get_rope_index. Returns a new tensor.
    """
    pos = base_pos.clone()
    s0, e0 = blocks[0]
    for (si, ei) in blocks[1:]:
        pos[:, :, si:ei] -= int(base_pos[0, 0, si]) - int(base_pos[0, 0, s0])
    blk0_max = int(pos[:, :, s0:e0].max())
    pos[:, :, fin_start:] -= int(base_pos[0, 0, fin_start]) - (blk0_max + 1)
    return pos


# -------------------------------------------------------------------- token layout

def find_subseq(hay: List[int], needle: List[int]) -> List[int]:
    """All start indices where `needle` occurs in `hay`."""
    out, n = [], len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            out.append(i)
    return out


def find_question_spans(
    ids: List[int], tokenizer: Any, question: str, expected_occurrences: int
) -> Optional[List[Span]]:
    """Locate all occurrences of the question in the token ids (replicas + final).

    Tokenization is context-dependent, so the question is retokenized with a small set
    of leading-context variants until the expected occurrence count matches.
    """
    for pre in ("", " ", "\n"):
        needle = tokenizer(pre + question, add_special_tokens=False).input_ids
        occ = find_subseq(ids, needle)
        if len(occ) == expected_occurrences:
            return [(o, o + len(needle)) for o in occ]
    return None


def locate_word_token(ids: List[int], tokenizer: Any, word: str, span: Span) -> Optional[int]:
    """Last token position within `span` whose decode contains word[:4] (the message locus:
    room word / concept word / second character name inside a replica)."""
    a, b = span
    key = word[:4].lower()
    for p in range(b - 1, a - 1, -1):
        if key in tokenizer.decode([ids[p]]).strip().lower():
            return p
    return None


def frame_blocks(vision_starts: Sequence[int], fin_start: int) -> List[Span]:
    """Block i = [vision_start_i, vision_start_{i+1}) ; last block ends at the final question."""
    n = len(vision_starts)
    return [(vision_starts[i], vision_starts[i + 1] if i + 1 < n else fin_start) for i in range(n)]


# ------------------------------------------------------------------------- hooks

class FenceHooks:
    """Hook kit for fenced forwards on the frozen model:
    - injects a 4D additive mask into every decoder layer (`set_mask`/`clear_mask`),
    - captures q/k/v projection outputs at the given layers (`.qkv[L][name]`),
    - captures rotary position embeddings (`.cos`/`.sin`).

    Use as a context manager; hooks are removed on exit.
    """

    def __init__(self, layers: Any, capture_layers: Sequence[int] = ()):
        self._layers = layers
        self._capture = list(capture_layers)
        self._holder: Dict[str, Optional[torch.Tensor]] = {"mask": None}
        self._posemb: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self.qkv: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in self._capture}

    # -- mask control
    def set_mask(self, mask_2d: torch.Tensor, device: Any) -> None:
        seq = mask_2d.shape[-1]
        self._holder["mask"] = mask_2d.view(1, 1, seq, seq).to(device)

    def clear_mask(self) -> None:
        self._holder["mask"] = None

    @property
    def cos(self) -> torch.Tensor:
        return self._posemb["cos"]

    @property
    def sin(self) -> torch.Tensor:
        return self._posemb["sin"]

    # -- lifecycle
    def install(self) -> "FenceHooks":
        holder = self._holder

        def mask_pre(_m, hargs, hkwargs):
            mk = holder["mask"]
            if mk is None:
                return hargs, hkwargs
            hs = hargs[0] if hargs else hkwargs.get("hidden_states")
            if hs is not None and mk.dtype != hs.dtype:
                mk = mk.to(hs.dtype)
                holder["mask"] = mk
            if len(hargs) >= 2:
                return (hargs[0], mk) + tuple(hargs[2:]), hkwargs
            hkwargs = dict(hkwargs)
            hkwargs["attention_mask"] = mk
            return hargs, hkwargs

        for ly in self._layers:
            self._handles.append(ly.register_forward_pre_hook(mask_pre, with_kwargs=True))

        def mk_qkv(L, nm):
            def hook(_m, _i, o):
                self.qkv[L][nm] = o.detach()

            return hook

        for L in self._capture:
            for nm in ("q_proj", "k_proj", "v_proj"):
                self._handles.append(
                    getattr(self._layers[L].self_attn, nm).register_forward_hook(mk_qkv(L, nm))
                )

        if self._capture:
            posemb = self._posemb

            def pe_hook(_m, a_, k_):
                pe = k_.get("position_embeddings")
                if pe is None and len(a_) >= 1 and isinstance(a_[-1], tuple):
                    pe = a_[-1]
                if pe is not None:
                    posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()

            self._handles.append(
                self._layers[self._capture[0]].self_attn.register_forward_pre_hook(
                    pe_hook, with_kwargs=True
                )
            )
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "FenceHooks":
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.remove()


# --------------------------------------------------------------- message recompute

def recompute_messages(
    *,
    seq: int,
    mask_full: torch.Tensor,
    carrier_positions: Sequence[int],
    vis_by_frame: Sequence[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    dims: Dict[str, Any],
    w_o: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    differentiable: bool = False,
):
    """Per-frame attention message into each carrier position at one layer.

    For carrier i: softmax over its (masked) logits restricted to frame i's image-token
    keys, context-summed over values, projected through o_proj — "what this carrier
    reads from its own frame". Returns [NF, hidden] (torch if differentiable else numpy).
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
    )

    n_heads, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    q = q_proj.view(1, seq, n_heads, hd).transpose(1, 2)
    k = k_proj.view(1, seq, n_kv, hd).transpose(1, 2)
    v = v_proj.view(1, seq, n_kv, hd).transpose(1, 2)
    qr, kr = apply_multimodal_rotary_pos_emb(
        q.float(), k.float(), cos.float(), sin.float(), dims["mrope_section"]
    )
    kr = repeat_kv(kr, n_heads // n_kv)[0]
    vv = repeat_kv(v, n_heads // n_kv)[0].float()
    qr = qr[0]
    W = w_o.to(qr.device)
    msgs = []
    for i, c in enumerate(carrier_positions):
        lg = torch.einsum("hd,htd->ht", qr[:, c], kr) / (hd ** 0.5)
        lg = lg + mask_full[c].to(qr.device)
        wgt = torch.softmax(lg, -1)
        fidx = vis_by_frame[i].to(qr.device)
        ctx = torch.einsum("ht,htd->hd", wgt[:, fidx], vv[:, fidx]).reshape(-1)
        msgs.append(W @ ctx)
    st = torch.stack(msgs)
    return st if differentiable else st.detach().cpu().numpy()
