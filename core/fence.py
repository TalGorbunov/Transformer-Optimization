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


# --------------------------------------------------------------------------- masks

def build_block_mask(seq: int, blocks: Sequence[Span], hide_cols: Sequence[int]) -> torch.Tensor:
    """Causal + block-diagonal fence + globally hidden columns. Twin: fencing.py:29.
    hide_cols are re-opened inside their own block (a block always sees itself)."""
    raise NotImplementedError


def hide_cols_for(blocks: Sequence[Span], keep: Sequence[bool]) -> List[int]:
    """The gate's output as mask input: every column of every block whose keep flag is False.
    Pinned by tests/test_fence.py::test_gate_semantics."""
    raise NotImplementedError


# ----------------------------------------------------------------------- positions

def reset_positions(base_pos: torch.Tensor, blocks: Sequence[Span], fin_start: int) -> torch.Tensor:
    """Per-block M-RoPE reset. Twin: fencing.py:111.
    base_pos: (3, 1, seq) from the model's get_rope_index. Every block gets block 0's position
    ids (blocks cannot attend each other, so reuse is safe); the tail continues right after
    block 0's max position. Returns a new tensor."""
    raise NotImplementedError


# -------------------------------------------------------------------- token layout

def frame_blocks(vision_starts: Sequence[int], vision_ends: Sequence[int]) -> List[Span]:
    """Block i = [vision_start_i, vision_end_i + 1). NOTE the twin (fencing.py:165) closed a
    block at the NEXT vision_start, swallowing any replica text between frames; with no
    replicas the block is exactly the image span plus its two markers."""
    raise NotImplementedError


def slot_positions(input_ids_1d: torch.Tensor, vision_end_id: int) -> List[int]:
    """The per-frame slot = each <|vision_end|> position, in frame order. This is where the
    gate reads (core.__doc__). Pinned by tests/test_fence.py::test_slots."""
    raise NotImplementedError


def find_subseq(hay: List[int], needle: List[int]) -> List[int]:
    """All start indices where `needle` occurs in `hay`. Twin: fencing.py:129."""
    raise NotImplementedError


def find_question_spans(ids: List[int], tokenizer: Any, question: str,
                        expected_occurrences: int) -> Optional[List[Span]]:
    """Locate the question's token span(s) (1 in the paper layouts, N+1 with replicas).
    Tokenization is context-dependent: retokenize with a few leading-context variants until
    the occurrence count matches. Twin: fencing.py:138."""
    raise NotImplementedError


# ------------------------------------------------------------------------- hooks

class FenceHooks:
    """Injects a 4-D additive mask into every decoder layer (set_mask / clear_mask) and can
    capture hidden states at chosen layers (for the gate). Context manager; hooks are removed
    on exit. Twin: fencing.py:173 (there it also captured q/k/v for message recompute)."""

    def __init__(self, layers: Any, capture_layers: Sequence[int] = ()):
        self._layers = layers
        self._capture = list(capture_layers)
        self._holder: Dict[str, Optional[torch.Tensor]] = {"mask": None}
        self._handles: List[Any] = []
        self.hidden: Dict[int, torch.Tensor] = {}

    def set_mask(self, mask_2d: torch.Tensor, device: Any) -> None:
        raise NotImplementedError

    def clear_mask(self) -> None:
        self._holder["mask"] = None

    def install(self) -> "FenceHooks":
        """Register a forward_pre_hook on every layer that swaps in the held mask (positional
        or kwarg attention_mask, cast to the hidden-state dtype), and a forward hook on each
        capture layer that stores its output hidden state in self.hidden[L]."""
        raise NotImplementedError

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "FenceHooks":
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.remove()
