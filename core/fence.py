"""The fence: block-diagonal attention over per-frame blocks, per-block position reset,
the per-frame slot, and the hook that injects a mask into every decoder layer.

Twin: legacy/v1/gnnformer/fencing.py (316 lines). Dropped: build_replica_probe_mask (the
probe-era incremental construction) and recompute_messages (carrier-era). Added:
slot_positions and hide_cols_for.

Mask convention (unchanged, pinned bit-for-bit by tests/test_fence.py::test_legacy_parity):
additive float, 0 = allowed, MASK_MIN = forbidden, shape [seq, seq] (row = query,
col = key); view(1, 1, seq, seq) before injection.

Geometry of a fenced sequence:
    [prefix: system + question] [block_1] [block_2] ... [block_N] [tail: answer prompt]
    block_i = [<|vision_start|>, image tokens, <|vision_end|>]  — half-open span (a, b)
Rules the mask must satisfy (each is one test):
    causal everywhere;
    a block's rows may attend the prefix and their own block, never another block;
    hidden columns (the gate's decision) are forbidden to every row outside their own block;
    rows outside all blocks (prefix, tail) are plain causal over everything except hidden columns
    — i.e. the tail DOES see every kept block.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .constants import MASK_MIN

Span = Tuple[int, int]  # half-open [start, end)


# SDPA backends that accept a 4-D additive mask (FLASH does not). Wrap fenced forwards in
# torch.nn.attention.sdpa_kernel(FENCED_SDPA).
from torch.nn.attention import SDPBackend

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


# --------------------------------------------------------------------------- masks

def build_block_mask(seq: int, blocks: Sequence[Span], hide_cols: Sequence[int]) -> torch.Tensor:
    """Causal + block-diagonal fence + globally hidden columns. Twin: fencing.py:29 (verbatim).
    hide_cols are re-opened inside their own block (a block always sees itself)."""
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


def hide_cols_for(blocks: Sequence[Span], keep: Sequence[bool]) -> List[int]:
    """The gate's output as mask input: every column of every block whose keep flag is False.
    Pinned by tests/test_fence.py::test_gate_semantics."""
    assert len(blocks) == len(keep), (len(blocks), len(keep))
    return [c for (a, b), k in zip(blocks, keep) if not k for c in range(a, b)]


# ----------------------------------------------------------------------- positions

def reset_positions(base_pos: torch.Tensor, blocks: Sequence[Span], fin_start: int) -> torch.Tensor:
    """Per-block M-RoPE reset. Twin: fencing.py:111 (verbatim).
    base_pos: (3, 1, seq) from the model's get_rope_index. Every block gets block 0's position
    ids (blocks cannot attend each other, so reuse is safe); the tail continues right after
    block 0's max position. Returns a new tensor."""
    pos = base_pos.clone()
    if not blocks:
        return pos
    s0, e0 = blocks[0]
    for (si, ei) in blocks[1:]:
        pos[:, :, si:ei] -= int(base_pos[0, 0, si]) - int(base_pos[0, 0, s0])
    blk0_max = int(pos[:, :, s0:e0].max())
    pos[:, :, fin_start:] -= int(base_pos[0, 0, fin_start]) - (blk0_max + 1)
    return pos


# -------------------------------------------------------------------- token layout

def frame_blocks(vision_starts: Sequence[int], vision_ends: Sequence[int]) -> List[Span]:
    """Block i = [vision_start_i, vision_end_i + 1). NOTE the twin (fencing.py:165) closed a
    block at the NEXT vision_start, swallowing any replica text between frames; with no
    replicas the block is exactly the image span plus its two markers."""
    assert len(vision_starts) == len(vision_ends), (len(vision_starts), len(vision_ends))
    blocks = [(int(vs), int(ve) + 1) for vs, ve in zip(vision_starts, vision_ends)]
    assert all(a < b for a, b in blocks), blocks
    return blocks


def slot_positions(input_ids_1d: torch.Tensor, vision_end_id: int) -> List[int]:
    """The per-frame slot = each <|vision_end|> position, in frame order. This is where the
    gate reads (core.__doc__). Pinned by tests/test_fence.py::test_slots."""
    return (input_ids_1d == vision_end_id).nonzero().flatten().tolist()


def find_subseq(hay: List[int], needle: List[int]) -> List[int]:
    """All start indices where `needle` occurs in `hay`. Twin: fencing.py:129 (verbatim)."""
    out, n = [], len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            out.append(i)
    return out


def find_question_spans(ids: List[int], tokenizer: Any, question: str,
                        expected_occurrences: int) -> Optional[List[Span]]:
    """Locate the question's token span(s) (1 in the paper layouts, N+1 with replicas).
    Tokenization is context-dependent: retokenize with a few leading-context variants until
    the occurrence count matches. Twin: fencing.py:138 (verbatim)."""
    for pre in ("", " ", "\n"):
        needle = tokenizer(pre + question, add_special_tokens=False).input_ids
        occ = find_subseq(ids, needle)
        if len(occ) == expected_occurrences:
            return [(o, o + len(needle)) for o in occ]
    return None


# ------------------------------------------------------------------------- hooks

class FenceHooks:
    """Injects a 4-D additive mask into every decoder layer (set_mask / clear_mask) and can
    capture hidden states at chosen layers (for the gate). Context manager; hooks are removed
    on exit. Twin: fencing.py:173 (there it also captured q/k/v for message recompute).

    Capture convention: `capture_layers` are 0-based indices into `layers`; `self.hidden[L]` is
    the OUTPUT hidden state of decoder module L. The legacy scripts indexed the model's
    `output_hidden_states` tuple, where entry L is the output of module L-1; translate when
    comparing to a legacy layer number."""

    def __init__(self, layers: Any, capture_layers: Sequence[int] = ()):
        self._layers = layers
        self._capture = list(capture_layers)
        self._holder: Dict[str, Optional[torch.Tensor]] = {"mask": None}
        self._handles: List[Any] = []
        self.hidden: Dict[int, torch.Tensor] = {}

    def set_mask(self, mask_2d: torch.Tensor, device: Any) -> None:
        seq = mask_2d.shape[-1]
        self._holder["mask"] = mask_2d.view(1, 1, seq, seq).to(device)

    def clear_mask(self) -> None:
        self._holder["mask"] = None

    def install(self) -> "FenceHooks":
        """Register a forward_pre_hook on every layer that swaps in the held mask (positional
        or kwarg attention_mask, cast to the hidden-state dtype), and a forward hook on each
        capture layer that stores its output hidden state in self.hidden[L]."""
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

        def mk_capture(L: int):
            def hook(_m, _i, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                self.hidden[L] = hs.detach()
            return hook

        for L in self._capture:
            self._handles.append(self._layers[L].register_forward_hook(mk_capture(L)))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "FenceHooks":
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.remove()


# ------------------------------------------------------------ layout -> blocks (runtime)

def layout_blocks(input_ids_1d: torch.Tensor, layout: str, sid: Dict[str, int], *,
                  im_end_id: int) -> Tuple[List[Span], int]:
    """(blocks, fin_start) for a tokenised prompt built by core.prompt.build_messages.

    paper / question-first: a block is exactly the image span plus its two markers
    (frame_blocks); the question sits in the shared prefix (question-first) or in the tail
    (paper). replica: block i runs from its <|vision_start|> to the next one so the per-frame
    question copy sits INSIDE its frame's block; the last block ends at the <|im_end|> that
    closes the user turn. Under frame_blocks alone the replica text would fall outside every
    block and read every earlier frame — a joint reader, not the S1–S11 layout.
    fin_start = end of the last block (where the tail begins)."""
    ids = input_ids_1d
    vs = (ids == sid["vision_start"]).nonzero().flatten().tolist()
    ve = slot_positions(ids, sid["vision_end"])
    if not vs or len(vs) != len(ve):
        raise ValueError(f"vision markers mismatch: {len(vs)} starts, {len(ve)} ends")
    if layout in ("paper", "question-first"):
        blocks = frame_blocks(vs, ve)
    elif layout == "replica":
        closes = [p for p in (ids == im_end_id).nonzero().flatten().tolist() if p > ve[-1]]
        if not closes:
            raise ValueError("no <|im_end|> after the last frame")
        blocks = [(vs[i], vs[i + 1]) for i in range(len(vs) - 1)] + [(vs[-1], closes[0])]
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return blocks, blocks[-1][1]


def fenced_setup(inputs: Dict[str, Any], blocks: Sequence[Span], fin_start: int,
                 keep: Optional[Sequence[bool]], rope_fn: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """(mask_2d, position_ids) for one forward: the fence mask with the gate's hidden columns
    (keep=None or all-True = no gate) and the per-block position reset over the model's own
    M-RoPE positions. Shared by train.py and evaluate.py so the val metric IS the test metric."""
    ids = inputs["input_ids"]
    seq = int(ids.shape[1])
    hide = hide_cols_for(blocks, keep) if keep is not None else []
    mask = build_block_mask(seq, blocks, hide)
    with torch.no_grad():
        base_pos, _ = rope_fn(ids, image_grid_thw=inputs.get("image_grid_thw"),
                              attention_mask=inputs.get("attention_mask"))
    return mask, reset_positions(base_pos, blocks, fin_start)


def greedy_decode(model: Any, hooks: "FenceHooks", inputs: Dict[str, Any], setup_fn: Any, *,
                  tokenizer: Any, max_new: int, eos_id: int, stop_fn: Any = None) -> str:
    """Cache-free greedy decoding under the fence: every step re-forwards the whole sequence
    with mask + positions rebuilt by setup_fn(cur_inputs) -> (mask_2d, position_ids). Exact
    (no KV cache to keep consistent with a changing 4-D mask); cost grows with max_new.
    Stops at eos or when stop_fn(decoded_text) is true. Returns the decoded text."""
    from torch.nn.attention import sdpa_kernel

    ids = inputs["input_ids"]
    out: List[int] = []
    for _ in range(max_new):
        cur: Dict[str, Any] = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs:
                cur[k] = inputs[k]
        mask, pos = setup_fn(cur)
        cur.pop("attention_mask", None)
        hooks.set_mask(mask, ids.device)
        try:
            with sdpa_kernel(FENCED_SDPA):
                logits = model(**cur, position_ids=pos.to(ids.device), use_cache=False).logits
        finally:
            hooks.clear_mask()
        nxt = int(logits[0, -1].argmax())
        if nxt == eos_id:
            break
        out.append(nxt)
        text = tokenizer.decode(out, skip_special_tokens=True)
        if stop_fn is not None and stop_fn(text):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tokenizer.decode(out, skip_special_tokens=True)
