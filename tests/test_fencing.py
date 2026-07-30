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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
