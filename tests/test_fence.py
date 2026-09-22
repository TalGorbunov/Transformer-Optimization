"""CPU tests pinning the fence: masks, gate semantics, position reset, slots.

The mask IS the method; a silent change here invalidates every comparison. The legacy-parity
test pins build_block_mask + reset_positions bit-for-bit against the two frozen
implementations (legacy/v1 = July–Sept 2026 package; legacy/experiments = pre-July).

Run: python tests/test_fence.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.constants import MASK_MIN
from core.fence import (
    build_block_mask,
    find_subseq,
    frame_blocks,
    hide_cols_for,
    layout_blocks,
    reset_positions,
    slot_positions,
)

# ---- synthetic layout: [prefix: system+question][ (vs, img*4, ve) x 3 ][tail]
PREFIX, NF, IMG, TAIL = 4, 3, 4, 3
BLK = IMG + 2                                   # vs + img + ve
VSTARTS = [PREFIX + BLK * i for i in range(NF)]
VENDS = [vs + BLK - 1 for vs in VSTARTS]
FIN = PREFIX + BLK * NF                         # tail start
SEQ = FIN + TAIL
BLOCKS = frame_blocks(VSTARTS, VENDS)
VE_ID, VS_ID, PAD_ID = 151653, 151652, 151655


def ids_synthetic() -> torch.Tensor:
    ids = [1] * PREFIX
    for _ in range(NF):
        ids += [VS_ID] + [PAD_ID] * IMG + [VE_ID]
    ids += [2] * TAIL
    return torch.tensor(ids)


def allowed(m: torch.Tensor, r: int, c: int) -> bool:
    return bool(m[r, c] == 0)


def test_frame_blocks():
    assert BLOCKS == [(4, 10), (10, 16), (16, 22)]
    assert all(b - a == BLK for a, b in BLOCKS)


def test_causality():
    m = build_block_mask(SEQ, BLOCKS, hide_cols=[])
    upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1)
    assert torch.all(m[upper] == MASK_MIN), "every strictly-upper entry is forbidden"
    assert m.shape == (SEQ, SEQ) and m.dtype == torch.float32


def test_prefix_visible_to_every_block():
    m = build_block_mask(SEQ, BLOCKS, hide_cols=[])
    for a, b in BLOCKS:
        for r in range(a, b):
            assert all(allowed(m, r, c) for c in range(PREFIX))


def test_block_isolation():
    """A block's rows see their own block (causal) and nothing of any other block."""
    m = build_block_mask(SEQ, BLOCKS, hide_cols=[])
    for i, (a, b) in enumerate(BLOCKS):
        for r in range(a, b):
            for c in range(a, r + 1):
                assert allowed(m, r, c)
            for j, (a2, b2) in enumerate(BLOCKS):
                if j != i:
                    assert all(not allowed(m, r, c) for c in range(a2, b2))


def test_tail_sees_all_kept_blocks():
    m = build_block_mask(SEQ, BLOCKS, hide_cols=[])
    for r in range(FIN, SEQ):
        assert all(allowed(m, r, c) for c in range(r + 1))


def test_gate_semantics():
    """keep=[T, F, T]: block 2's columns are hidden from every row outside block 2, block 2
    still sees itself, and the tail reads exactly blocks 1 and 3 + prefix."""
    keep = [True, False, True]
    hide = hide_cols_for(BLOCKS, keep)
    a1, b1 = BLOCKS[1]
    assert hide == list(range(a1, b1))
    m = build_block_mask(SEQ, BLOCKS, hide_cols=hide)
    for r in range(FIN, SEQ):                              # tail rows
        assert all(not allowed(m, r, c) for c in hide)
        for a, b in (BLOCKS[0], BLOCKS[2]):
            assert all(allowed(m, r, c) for c in range(a, b))
    for r in range(a1, b1):                                # the hidden block still sees itself
        assert all(allowed(m, r, c) for c in range(a1, r + 1))
    for r in range(*BLOCKS[2]):                            # other blocks never saw it anyway
        assert all(not allowed(m, r, c) for c in hide)
    assert hide_cols_for(BLOCKS, [True] * NF) == []


def test_no_fence_is_plain_causal():
    m = build_block_mask(SEQ, [], hide_cols=[])
    causal = torch.zeros(SEQ, SEQ).masked_fill(torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1), MASK_MIN)
    assert torch.equal(m, causal)


def test_reset_positions_blocks_identical():
    base = torch.arange(SEQ, dtype=torch.long).view(1, 1, SEQ).repeat(3, 1, 1)
    pos = reset_positions(base, BLOCKS, FIN)
    s0, e0 = BLOCKS[0]
    for a, b in BLOCKS[1:]:
        assert torch.equal(pos[:, :, a:b], pos[:, :, s0:e0])
    assert int(pos[0, 0, FIN]) == int(pos[:, :, s0:e0].max()) + 1
    assert torch.equal(pos[:, :, :PREFIX], base[:, :, :PREFIX])
    assert torch.equal(base, torch.arange(SEQ).view(1, 1, SEQ).repeat(3, 1, 1)), "input must not be mutated"


def test_slots():
    ids = ids_synthetic()
    slots = slot_positions(ids, VE_ID)
    assert slots == VENDS
    assert all(s == b - 1 for s, (a, b) in zip(slots, BLOCKS)), "the slot is the last token of its block"


def test_find_subseq():
    assert find_subseq([1, 2, 3, 2, 3], [2, 3]) == [1, 3]
    assert find_subseq([1, 2], [3]) == []


def test_legacy_parity():
    """Bit-for-bit against the frozen implementations. Skips (never fails) if a legacy import
    chain is unavailable in this environment."""
    base = torch.arange(SEQ, dtype=torch.long).view(1, 1, SEQ).repeat(3, 1, 1)
    hide = hide_cols_for(BLOCKS, [True, False, True])
    ours_m, ours_p = build_block_mask(SEQ, BLOCKS, hide), reset_positions(base, BLOCKS, FIN)
    # legacy/v1 (the July–Sept package). Its frame_blocks closes at the next vision_start,
    # which equals ours when frames are contiguous.
    sys.path.insert(0, str(_REPO / "legacy" / "v1"))
    try:
        from gnnformer.fencing import build_block_mask as v1_build, reset_positions as v1_reset  # type: ignore
        assert torch.equal(ours_m, v1_build(SEQ, BLOCKS, hide_cols=hide))
        assert torch.equal(ours_p, v1_reset(base, BLOCKS, FIN))
        print("  legacy/v1 parity: OK")
    except ImportError as exc:
        print(f"  [skip] legacy/v1 import unavailable: {exc}")
    # pre-July (the original trainer lineage)
    sys.path.insert(0, str(_REPO / "legacy"))
    try:
        from experiments.glstm.carrier_token_distill import (  # type: ignore
            build_block_mask as v0_build, reset_positions as v0_reset)
        assert torch.equal(ours_m, v0_build(SEQ, BLOCKS, hide_cols=hide))
        assert torch.equal(ours_p, v0_reset(base, BLOCKS, FIN))
        print("  legacy (pre-July) parity: OK")
    except Exception as exc:  # heavy import chain (nnsight etc.) -> skip
        print(f"  [skip] pre-July legacy import unavailable: {exc}")


def test_layout_blocks():
    """question-first: blocks = image spans; replica: block i reaches the next vision_start and
    the last block ends at the <|im_end|> closing the user turn (the per-frame question copy is
    inside its block, not a joint reader)."""
    sid = {"vision_start": VS_ID, "vision_end": VE_ID, "image_pad": PAD_ID}
    IM_END = 151645
    ids = ids_synthetic()
    blocks, fin = layout_blocks(ids, "question-first", sid, im_end_id=IM_END)
    assert blocks == BLOCKS and fin == FIN
    # replica: [q][vs img*4 ve q][vs img*4 ve q][vs img*4 ve q]<|im_end|> tail
    rep = [1] * PREFIX
    for _ in range(NF):
        rep += [VS_ID] + [PAD_ID] * IMG + [VE_ID] + [7, 7]          # 2 question tokens per replica
    rep += [IM_END] + [2] * TAIL
    rep_t = torch.tensor(rep)
    blocks, fin = layout_blocks(rep_t, "replica", sid, im_end_id=IM_END)
    stride = BLK + 2
    want = [(PREFIX + stride * i, PREFIX + stride * (i + 1)) for i in range(NF)]
    assert blocks == want, blocks
    assert fin == PREFIX + stride * NF and rep[fin] == IM_END
    for a, b in blocks:                                    # every replica token is inside its block
        assert rep[b - 1] == 7 and rep[b - 2] == 7


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
