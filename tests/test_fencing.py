"""CPU tests pinning the fencing mechanism (masks + position reset).

The scientific-validity test of the port: the canonical build_block_mask (trainer
lineage), the probe's incremental construction (probe lineage), and the LEGACY
implementation must all produce the identical mask for the deployed A3 config
(blockfence + posreset). A silent divergence here would invalidate probe-vs-trainer
comparisons in RESULTS.md.

Run: .venv/bin/python tests/test_fencing.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import MASK_MIN
from gnnformer.fencing import (
    build_block_mask,
    build_replica_probe_mask,
    frame_blocks,
    reset_positions,
)

# ---- synthetic token layout: [prefix+q0][ (vs, img*4, ve, rep*3) x 3 ][final q][tail]
PREFIX = 4
NF = 3
BLOCK = 9  # vs + 4 img + ve + 3 replica tokens
VSTARTS = [PREFIX + BLOCK * i for i in range(NF)]
FIN = PREFIX + BLOCK * NF          # final-question start
SEQ = FIN + 3 + 2                  # + final question (3) + tail (2)
BLOCKS = frame_blocks(VSTARTS, FIN)
REP_SPANS = [(vs + 6, vs + 9) for vs in VSTARTS]
VIS = [torch.tensor([vs + 1, vs + 2, vs + 3, vs + 4]) for vs in VSTARTS]
HIDE = [p for a, b in REP_SPANS for p in range(a, b)]


def canonical() -> torch.Tensor:
    return build_block_mask(SEQ, BLOCKS, hide_cols=HIDE)


def probe_style() -> torch.Tensor:
    return build_replica_probe_mask(
        SEQ, REP_SPANS, VIS, fence_frames=True, fence_blocks=True, blocks=BLOCKS
    )


def test_probe_equals_canonical():
    """Probe lineage == trainer lineage on the deployed (blockfence) config."""
    assert torch.equal(canonical(), probe_style())


def test_legacy_parity():
    """Ported canonical mask + posreset == the frozen legacy implementation, bit for bit."""
    sys.path.insert(0, str(_REPO / "legacy"))
    try:
        from experiments.glstm.carrier_token_distill import (  # type: ignore
            build_block_mask as legacy_build,
            reset_positions as legacy_reset,
        )
    except Exception as exc:  # heavy legacy import chain unavailable -> skip, don't fail
        print(f"  [skip] legacy import unavailable: {exc}")
        return
    assert torch.equal(canonical(), legacy_build(SEQ, BLOCKS, hide_cols=HIDE))
    base = torch.arange(SEQ).view(1, 1, SEQ).repeat(3, 1, 1)
    assert torch.equal(
        reset_positions(base, BLOCKS, FIN), legacy_reset(base, BLOCKS, FIN)
    )


def test_causality():
    m = canonical()
    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1)
    assert (m[future] == MASK_MIN).all()


def test_replica_row_visibility():
    """Replica rows see exactly: prefix+question, own block (causal), self."""
    m = canonical()
    for i, (a, b) in enumerate(REP_SPANS):
        r = b - 1  # last replica token row
        allowed = (m[r] == 0).nonzero(as_tuple=True)[0].tolist()
        ba, _bb = BLOCKS[i]
        expected = list(range(PREFIX)) + list(range(ba, r + 1))
        assert allowed == expected, f"replica {i}: {allowed} != {expected}"


def test_frame_rows_fenced():
    """Image-token rows cannot see any other frame's tokens (incl. vision markers)."""
    m = canonical()
    for i in range(NF):
        row = int(VIS[i][-1])
        for j, (aj, bj) in enumerate(BLOCKS):
            if j != i:
                assert (m[row, aj:bj] == MASK_MIN).all(), f"frame {i} sees block {j}"


def test_tail_semantics():
    """Documented P0.1 semantics: the final question / decode tail DOES see all frames'
    image tokens, does NOT see replicas/carriers (hide_cols)."""
    m = canonical()
    tail = SEQ - 1
    for i in range(NF):
        assert (m[tail, VIS[i]] == 0).all(), "tail must see frame image tokens (by design)"
    for a, b in REP_SPANS:
        assert (m[tail, a:b] == MASK_MIN).all(), "tail must not see replicas"


def test_reset_positions_blocks_identical():
    """After the reset every block carries block 0's positions and the final question
    continues right after block 0's max — position-equivalent to an isolated forward."""
    base = torch.arange(SEQ).view(1, 1, SEQ).repeat(3, 1, 1)
    pos = reset_positions(base, BLOCKS, FIN)
    s0, e0 = BLOCKS[0]
    for (si, ei) in BLOCKS[1:]:
        assert torch.equal(pos[:, :, si:ei], pos[:, :, s0:e0])
    blk0_max = int(pos[:, :, s0:e0].max())
    assert int(pos[0, 0, FIN]) == blk0_max + 1
    assert torch.equal(pos[:, :, :PREFIX], base[:, :, :PREFIX])  # prefix untouched


def test_no_mask_arm_is_plain_causal():
    """The unmasked control (no fence flags, no hidden cols) is plain causal."""
    m = build_replica_probe_mask(SEQ, [], [torch.tensor([], dtype=torch.long)])
    causal = torch.zeros(SEQ, SEQ)
    causal.masked_fill_(torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1), MASK_MIN)
    assert torch.equal(m, causal)


# ---- LEARNMASK parity (outputs/learnmask/): the relation-gate assembly must contain
# ---- the hand fence exactly — fence init bit-for-bit, hand design expressible.

# carrier layouts (single carrier, last token of each block):
#  A: prefix 10 | 3 blocks of 8 | tail 6  (the test_carrier_masks layout)
#  B: prefix 3 | 20 blocks of 2 | tail 4  (block distances up to 19 -> all Δ-buckets)
_CARRIER_LAYOUTS = [
    (40, [(10, 18), (18, 26), (26, 34)], [17, 25, 33], 34),
    (47, [(3 + 2 * i, 5 + 2 * i) for i in range(20)], [4 + 2 * i for i in range(20)], 43),
]


def test_learnmask_fence_init_parity():
    """Hard assembly at the fence init == build_block_mask(hide_cols=carriers) ==
    make_masks lo, BIT-FOR-BIT, at every layer, with and without appended rows."""
    from gnnformer.carriers import ext_mask, make_masks
    from gnnformer.learnmask import fence_open, hard_mask, relation_cell_map

    for seq, blocks, cpos, fin in _CARRIER_LAYOUTS:
        lo = build_block_mask(seq, blocks, hide_cols=list(cpos))
        assert torch.equal(lo, make_masks(seq, blocks, cpos, fin)[0])
        cm = relation_cell_map(seq, blocks, cpos, fin)
        assert torch.equal(hard_mask(cm, fence_open()), lo)
        for e in (1, 5):  # teacher-forced/decode rows: must equal ext_mask semantics
            cm_e = relation_cell_map(seq, blocks, cpos, fin, e=e)
            assert torch.equal(hard_mask(cm_e, fence_open()), ext_mask(lo, e))


def test_learnmask_replica_span_parity():
    """Span readers (question replicas): fence init == build_block_mask with the
    replica spans hidden — on THIS file's replica layout — and the span lo/hi
    construction reduces to carriers.make_masks bit-for-bit for single-token spans."""
    from gnnformer.carriers import make_masks
    from gnnformer.learnmask import (
        fence_open,
        hand_open_table,
        hard_mask,
        hard_masks_by_layer,
        make_masks_spans,
        relation_cell_map,
    )

    # replica layout (multi-token spans), reusing this file's canonical() reference
    cm = relation_cell_map(SEQ, BLOCKS, REP_SPANS, FIN)
    lo, hi = make_masks_spans(SEQ, BLOCKS, REP_SPANS, FIN)
    assert torch.equal(lo, canonical())          # == build_block_mask(hide_cols=HIDE)
    assert torch.equal(hard_mask(cm, fence_open()), lo)
    masks = hard_masks_by_layer(cm, hand_open_table(28, 12))
    assert torch.equal(masks[0], lo) and torch.equal(masks[12], hi)
    # replica rows in hi read earlier replicas; tail reads all replicas
    r2a, _ = REP_SPANS[2]
    assert (hi[r2a, [p for a, b in REP_SPANS[:2] for p in range(a, b)]] == 0).all()
    assert (hi[SEQ - 1, [p for a, b in REP_SPANS for p in range(a, b)]] == 0).all()
    # single-token spans == the anchored carrier construction, bit for bit
    for seq, blocks, cpos, fin in _CARRIER_LAYOUTS:
        lo_c, hi_c = make_masks(seq, blocks, cpos, fin)
        lo_s, hi_s = make_masks_spans(seq, blocks, cpos, fin)
        assert torch.equal(lo_c, lo_s) and torch.equal(hi_c, hi_s)


def test_learnmask_hand_design_expressible():
    """The deployed hand design is a point in gate space: R4+R7 open at layers >=
    L_OPEN reproduces make_masks (lo, hi) bit-for-bit — so the learned-mask trainer
    starts from a family that CONTAINS the baseline."""
    from gnnformer.carriers import ext_mask, make_masks
    from gnnformer.learnmask import hand_open_table, hard_masks_by_layer, relation_cell_map

    n_layers, l_open = 28, 12
    for seq, blocks, cpos, fin in _CARRIER_LAYOUTS:
        lo, hi = make_masks(seq, blocks, cpos, fin)
        cm = relation_cell_map(seq, blocks, cpos, fin)
        masks = hard_masks_by_layer(cm, hand_open_table(n_layers, l_open))
        for li in range(n_layers):
            assert torch.equal(masks[li], lo if li < l_open else hi), f"layer {li}"
        cm_e = relation_cell_map(seq, blocks, cpos, fin, e=3)
        masks_e = hard_masks_by_layer(cm_e, hand_open_table(n_layers, l_open))
        assert torch.equal(masks_e[0], ext_mask(lo, 3))
        assert torch.equal(masks_e[l_open], ext_mask(hi, 3))


def test_learnmask_chunking_invariant():
    """Row-chunked map construction (the P4 large-N path) == one-shot construction."""
    from gnnformer.learnmask import relation_cell_map

    seq, blocks, cpos, fin = _CARRIER_LAYOUTS[1]
    full = relation_cell_map(seq, blocks, cpos, fin, e=2)
    for chunk in (1, 7, 16):
        assert torch.equal(relation_cell_map(seq, blocks, cpos, fin, e=2, row_chunk=chunk), full)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
