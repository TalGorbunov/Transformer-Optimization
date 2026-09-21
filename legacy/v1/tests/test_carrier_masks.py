"""CPU tests for the carrier lo/hi mask family and the truncation column algebra
(port of legacy/experiments/glstm/trunc_mask_smoke.py, the TRUNC P0.3 gate), plus
bit-for-bit parity with the frozen legacy implementations.

Run: .venv/bin/python tests/test_carrier_masks.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import ext_mask, frame_cols, keep_cols, make_masks, truncated_masks
from gnnformer.constants import MASK_MIN

# synthetic layout: prefix+question [0..9], 3 blocks of 8 (frame 7 + carrier last), tail [34..39]
SEQ = 40
BLOCKS = [(10, 18), (18, 26), (26, 34)]
CPOS = [17, 25, 33]
FIN = 34
LO, HI = make_masks(SEQ, BLOCKS, CPOS, FIN)


def vis(m, r):
    return set(torch.nonzero(m[r] == 0).flatten().tolist())


def test_keep_frame_split():
    fcols = set(frame_cols(SEQ, BLOCKS, CPOS))
    kcols = keep_cols(SEQ, BLOCKS, CPOS)
    assert sorted(kcols + sorted(fcols)) == list(range(SEQ))
    assert kcols == sorted(set(range(10)) | set(CPOS) | set(range(34, 40)))


def test_tail_rows():
    """P0.1 truth: tail rows attend ALL frame cols in lo AND hi; carriers only in hi."""
    fcols = set(frame_cols(SEQ, BLOCKS, CPOS))
    for r in range(FIN, SEQ):
        v_lo, v_hi = vis(LO, r), vis(HI, r)
        assert fcols <= v_lo and fcols <= v_hi, f"tail row {r} must see frames (P0.1)"
        assert not (set(CPOS) & v_lo), f"tail row {r} must NOT see carriers in lo"
        assert set(CPOS) <= v_hi, f"tail row {r} must see all carriers in hi"


def test_carrier_rows():
    """Carrier rows: own frame + prefix+question + self in lo; + earlier carriers in hi."""
    for i, c in enumerate(CPOS):
        v_lo, v_hi = vis(LO, c), vis(HI, c)
        a, _b = BLOCKS[i]
        assert v_lo == set(range(10)) | set(range(a, c + 1)), f"carrier {i} lo vis {v_lo}"
        assert v_hi == v_lo | set(CPOS[:i]), f"carrier {i} hi vis"


def test_frame_rows():
    for r in range(10, 17):
        assert vis(LO, r) == set(range(10)) | set(range(10, r + 1))


def test_truncated_submask_edges():
    kcols = keep_cols(SEQ, BLOCKS, CPOS)
    kt = torch.tensor(kcols)
    hi_t = HI.index_select(0, kt).index_select(1, kt)
    pos_c = {p: j for j, p in enumerate(kcols)}
    car_t = [pos_c[c] for c in CPOS]
    for r in range(FIN, SEQ):
        j = pos_c[r]
        v = vis(hi_t, j)
        assert set(car_t) <= v, "truncated tail must still see all carriers"
        assert v == {pos_c[c] for c in vis(HI, r) if c in pos_c}, "submask edge mismatch"
    for i, (j, c) in enumerate(zip(car_t, CPOS)):
        # own-frame edge GONE (dropped cols); earlier carriers + prefix+question + self remain
        assert vis(hi_t, j) == set(range(10)) | set(car_t[: i + 1]), f"trunc carrier {i}"


def test_positions_never_renumbered():
    kcols = keep_cols(SEQ, BLOCKS, CPOS)
    pos = torch.arange(SEQ).view(1, 1, SEQ).expand(3, 1, SEQ)
    pos_t = pos.index_select(2, torch.tensor(kcols))
    assert pos_t[0, 0].tolist() == kcols, "positions must be original ids"


def test_ext_mask_and_dropkv():
    e = 3
    fcols = set(frame_cols(SEQ, BLOCKS, CPOS))
    fc = torch.tensor(sorted(fcols), dtype=torch.long)
    big = ext_mask(HI.clone(), e)
    big[SEQ:, fc] = MASK_MIN
    for j in range(e):
        r = SEQ + j
        v = vis(big, r)
        assert not (fcols & v), "appended row must not see frames after dropkv edit"
        assert set(CPOS) <= v and set(range(10)) <= v and set(range(FIN, SEQ + j + 1)) <= v
    big_lo = ext_mask(LO.clone(), e)
    big_lo[SEQ:, fc] = MASK_MIN
    assert not (set(CPOS) & vis(big_lo, SEQ)), "appended lo rows must not see carriers"
    # ext of the TRUNCATED mask == truncated ext (fast-decode step-mask consistency)
    kcols = keep_cols(SEQ, BLOCKS, CPOS)
    kt = torch.tensor(kcols)
    hi_t = HI.index_select(0, kt).index_select(1, kt)
    big_t = ext_mask(hi_t.clone(), e)
    kt_e = torch.tensor(kcols + list(range(SEQ, SEQ + e)))
    assert torch.equal(big.index_select(0, kt_e).index_select(1, kt_e), big_t)


def test_truncated_masks_direct_build():
    kcols = keep_cols(SEQ, BLOCKS, CPOS)
    kt = torch.tensor(kcols)
    lo_t2, hi_t2 = truncated_masks(kcols, CPOS)
    lo_sel = LO.index_select(0, kt).index_select(1, kt).to(torch.float16)
    hi_sel = HI.index_select(0, kt).index_select(1, kt).to(torch.float16)
    assert torch.equal(hi_t2, hi_sel) and torch.equal(lo_t2, lo_sel)


def test_legacy_parity():
    """Ported mask family == the frozen legacy implementation, bit for bit."""
    sys.path.insert(0, str(_REPO / "legacy"))
    try:
        from experiments.glstm.carrier_layer_lora import (  # type: ignore
            frame_cols as l_fc,
            keep_cols as l_kc,
            make_masks as l_mm,
            truncated_masks as l_tm,
        )
        from experiments.glstm.carrier_layer_cached import ext_mask as l_ext  # type: ignore
    except Exception as exc:
        print(f"  [skip] legacy import unavailable: {exc}")
        return
    lo_l, hi_l = l_mm(SEQ, BLOCKS, CPOS, FIN)
    assert torch.equal(LO, lo_l) and torch.equal(HI, hi_l)
    assert keep_cols(SEQ, BLOCKS, CPOS) == l_kc(SEQ, BLOCKS, CPOS)
    assert frame_cols(SEQ, BLOCKS, CPOS) == l_fc(SEQ, BLOCKS, CPOS)
    kcols = keep_cols(SEQ, BLOCKS, CPOS)
    lo_t, hi_t = truncated_masks(kcols, CPOS)
    lo_tl, hi_tl = l_tm(kcols, CPOS)
    assert torch.equal(lo_t, lo_tl) and torch.equal(hi_t, hi_tl)
    assert torch.equal(ext_mask(HI.clone(), 4), l_ext(hi_l.clone(), 4))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
