#!/usr/bin/env python3
"""TRUNC campaign P0.3 CPU smoke (2026-07-24): executable checks of the mask/keep/position
plumbing on synthetic shapes — no model, no GPU. Asserts the P0.1 code truth (tail rows CAN
see frames) and the truncation/drop-kv edits' invariants. Run: python -u this_file."""
from __future__ import annotations
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.glstm.carrier_layer_lora import (make_masks, keep_cols, frame_cols,
                                                  truncated_masks, MIN)
from experiments.glstm.carrier_layer_cached import ext_mask

# synthetic layout: prefix+question [0..9], 3 blocks of 8 (frame 7 + carrier last),
# tail [34..39]
seq = 40
blocks = [(10, 18), (18, 26), (26, 34)]
cpos = [17, 25, 33]
fin = 34
lo, hi = make_masks(seq, blocks, cpos, fin)
vis = lambda m, r: set(torch.nonzero(m[r] == 0).flatten().tolist())

# --- P0.1 truth: tail rows attend ALL frame cols in lo AND hi; carriers only in hi
fcols = set(frame_cols(seq, blocks, cpos))
kcols = keep_cols(seq, blocks, cpos)
assert sorted(kcols + sorted(fcols)) == list(range(seq))
assert kcols == sorted(set(range(10)) | set(cpos) | set(range(34, 40))), kcols
for r in range(fin, seq):
    v_lo, v_hi = vis(lo, r), vis(hi, r)
    assert fcols <= v_lo and fcols <= v_hi, f"tail row {r} must see frames (P0.1)"
    assert not (set(cpos) & v_lo), f"tail row {r} must NOT see carriers in lo"
    assert set(cpos) <= v_hi, f"tail row {r} must see all carriers in hi"

# --- carrier rows: own frame + prefix+question + self in lo; + earlier carriers in hi
for i, c in enumerate(cpos):
    v_lo, v_hi = vis(lo, c), vis(hi, c)
    a, b = blocks[i]
    assert v_lo == set(range(10)) | set(range(a, c + 1)), f"carrier {i} lo vis {v_lo}"
    assert v_hi == v_lo | set(cpos[:i]), f"carrier {i} hi vis"

# --- frame rows never see other blocks or any carrier except own-block causal
for r in range(10, 17):
    assert vis(lo, r) == set(range(10)) | set(range(10, r + 1))

# --- truncated submask: index-select keeps exactly the surviving edges
kt = torch.tensor(kcols)
hi_t = hi.index_select(0, kt).index_select(1, kt)
pos_c = {p: j for j, p in enumerate(kcols)}          # original -> truncated coord
tail_t = [pos_c[r] for r in range(fin, seq)]
car_t = [pos_c[c] for c in cpos]
for j, r in zip(tail_t, range(fin, seq)):
    v = vis(hi_t, j)
    assert set(car_t) <= v, "truncated tail must still see all carriers"
    assert v == {pos_c[c] for c in vis(hi, r) if c in pos_c}, "submask edge mismatch"
for i, (j, c) in enumerate(zip(car_t, cpos)):
    v = vis(hi_t, j)
    # own-frame edge GONE (dropped cols), earlier carriers + prefix+question + self remain
    assert v == set(range(10)) | set(car_t[:i + 1]), f"trunc carrier {i} vis {v}"

# --- position preservation: index-select keeps ORIGINAL ids, never renumbers
pos = torch.arange(seq).view(1, 1, seq).expand(3, 1, seq)
pos_t = pos.index_select(2, kt)
assert pos_t[0, 0].tolist() == kcols, "positions must be original ids"

# --- ext_mask + dropkv edit: appended rows read like last tail row minus frame cols
e = 3
big = ext_mask(hi.clone(), e)
fc = torch.tensor(sorted(fcols), dtype=torch.long)
big[seq:, fc] = MIN
for j in range(e):
    r = seq + j
    v = vis(big, r)
    assert not (fcols & v), "appended row must not see frames after dropkv edit"
    assert set(cpos) <= v and set(range(10)) <= v and set(range(fin, seq + j + 1)) <= v
big_lo = ext_mask(lo.clone(), e)
big_lo[seq:, fc] = MIN
assert not (set(cpos) & vis(big_lo, seq)), "appended lo rows must not see carriers"
# ext of the TRUNCATED mask == truncated ext (fast-decode step mask consistency)
big_t = ext_mask(hi_t.clone(), e)
kt_e = torch.tensor(kcols + list(range(seq, seq + e)))
assert torch.equal(big.index_select(0, kt_e).index_select(1, kt_e), big_t), \
    "ext(submask) must equal submask(ext with dropkv)"

# --- E5 direct-built truncated masks == index-selected dense (fp16 compare)
lo_t2, hi_t2 = truncated_masks(kcols, cpos)
lo_sel = lo.index_select(0, kt).index_select(1, kt).to(torch.float16)
hi_sel = hi.index_select(0, kt).index_select(1, kt).to(torch.float16)
assert torch.equal(hi_t2, hi_sel), "hi_t direct-build != index-selected dense"
assert torch.equal(lo_t2, lo_sel), "lo_t direct-build != index-selected dense"

print("TRUNC CPU SMOKE: ALL ASSERTS PASSED")
print(f"  seq={seq} keep={len(kcols)} fcols={len(fcols)}; P0.1 confirmed executably: "
      f"tail rows see frames in lo and hi; truncated masks/positions consistent")
