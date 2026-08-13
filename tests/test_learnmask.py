"""CPU tests for gnnformer/learnmask.py — the relation partition, Δ-buckets, arm
sizes, gate estimators (E1-E4) and the differentiable mask assembly.

The bit-for-bit fence parity lives in tests/test_fencing.py (the campaign's P0 gate);
these tests pin everything else the trainer relies on.

Run: .venv/bin/python tests/test_learnmask.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import MASK_MIN
from gnnformer.learnmask import (
    CELL_ANCHOR,
    CELL_FORBID,
    CHANNELS,
    DELTA_BUCKETS,
    N_CH,
    SOFT_FORBID,
    MaskGates,
    arm_learn_mask,
    assemble_mask,
    delta_bucket,
    fence_open,
    hand_open,
    mask_parts,
    relation_cell_map,
)

# the test_carrier_masks carrier layout: prefix 10 | 3 blocks of 8 (carrier last) | tail 6
SEQ, BLOCKS, CPOS, FIN = 40, [(10, 18), (18, 26), (26, 34)], [17, 25, 33], 34
CM = relation_cell_map(SEQ, BLOCKS, CPOS, FIN)
_CH = {c.name: i for i, c in enumerate(CHANNELS)}


def test_channel_registry():
    """22 channels in the brief's order; fence init per the relation table."""
    assert N_CH == 22
    on = {c.name for c in CHANNELS if c.fence_on}
    assert on == {"R1", "R3", "R6"}  # R2/R4/R5 buckets + R7 all fence-OFF
    assert [c.rel for c in CHANNELS] == (
        ["R1"] + ["R2"] * 6 + ["R3"] + ["R4"] * 6 + ["R5"] * 6 + ["R6", "R7"])


def test_delta_buckets():
    edges = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 8: 3, 9: 4, 16: 4, 17: 5, 100: 5}
    for d, b in edges.items():
        assert delta_bucket(d) == b, (d, b)
    lo0 = [lo for lo, _ in DELTA_BUCKETS]
    assert lo0 == [1, 2, 3, 5, 9, 17]


def test_partition_total():
    """Every cell classified exactly once: below/on the diagonal there is NO forbid
    (each cell belongs to a relation or the anchor); above it everything is forbid."""
    q = torch.arange(SEQ).view(-1, 1)
    k = torch.arange(SEQ).view(1, -1)
    lower = k <= q
    assert not (CM[lower] == CELL_FORBID).any()
    assert (CM[~lower] == CELL_FORBID).all()
    assert int(CM.max()) < N_CH and int(CM.min()) == CELL_FORBID


def _lower(n: int) -> torch.Tensor:
    return torch.tril(torch.ones(n, n, dtype=torch.bool))


def test_cell_classes():
    """Spot-check every relation on the 3-block layout."""
    a0 = BLOCKS[0][0]
    assert (CM[:a0, :a0][_lower(a0)] == CELL_ANCHOR).all()         # prefix causal
    assert (CM[FIN:, :a0] == CELL_ANCHOR).all()                    # R8: any -> prefix
    tail = CM[FIN:, FIN:]
    assert (tail[_lower(SEQ - FIN)] == CELL_ANCHOR).all()          # tail -> tail causal
    assert int(CM[FIN, CPOS[0]]) == _CH["R7"]                      # tail -> carrier
    assert int(CM[FIN, BLOCKS[0][0] + 1]) == _CH["R6"]             # tail -> frame content
    assert int(CM[CPOS[0], CPOS[0]]) == _CH["R3"]                  # carrier self
    assert int(CM[CPOS[1], BLOCKS[1][0]]) == _CH["R3"]             # carrier -> own block
    assert int(CM[CPOS[1], CPOS[0]]) == _CH["R4[Δ1]"]              # carrier -> carrier Δ1
    assert int(CM[CPOS[2], CPOS[0]]) == _CH["R4[Δ2]"]              # carrier -> carrier Δ2
    assert int(CM[CPOS[1], BLOCKS[0][0] + 2]) == _CH["R5[Δ1]"]     # carrier -> content Δ1
    assert int(CM[BLOCKS[1][0] + 3, BLOCKS[1][0]]) == _CH["R1"]    # block own causal
    assert int(CM[BLOCKS[2][0] + 1, BLOCKS[0][0] + 1]) == _CH["R2[Δ2]"]  # block -> block
    assert int(CM[BLOCKS[1][0] + 1, CPOS[0]]) == _CH["R2[Δ1]"]     # other-block carrier col
    # R4[Δ1] appears once per adjacent carrier pair; R4[Δ2] once
    assert int((CM == _CH["R4[Δ1]"]).sum()) == 2
    assert int((CM == _CH["R4[Δ2]"]).sum()) == 1


def test_arm_sizes():
    """Brief logit counts per layer: S1=2 (56 @28L), S2=14 (392), S3=22 (616)."""
    assert int(arm_learn_mask("s1").sum()) == 2
    assert int(arm_learn_mask("s2").sum()) == 14
    assert int(arm_learn_mask("s3").sum()) == 22


def test_train_assembly_soft_forbid_only_on_learnable():
    """Train-mode mask at the fence init: learnable-closed cells carry SOFT_FORBID=-30
    (gradient-sane), frozen-closed cells carry MASK_MIN, open cells 0."""
    learn = arm_learn_mask("s2")
    base, lidx = mask_parts(CM, learn, fence_open())
    g = MaskGates(1, arm="s2", estimator="ste").gate_table(mode="train")[:, 0]
    m = assemble_mask(base, lidx, g, SOFT_FORBID)
    assert float(m[FIN, CPOS[0]]) == SOFT_FORBID            # R7: learnable, closed
    assert float(m[CPOS[1], CPOS[0]]) == SOFT_FORBID        # R4: learnable, closed
    assert float(m[BLOCKS[1][0] + 1, BLOCKS[0][0] + 1]) == MASK_MIN  # R2: frozen in s2
    assert float(m[FIN, BLOCKS[0][0] + 1]) == 0.0           # R6: learnable, open
    assert float(m[CPOS[0], BLOCKS[0][0]]) == 0.0           # R3: frozen-open
    assert float(m[BLOCKS[0][0], BLOCKS[0][0] + 1]) == MASK_MIN  # causality


def test_hard_at_init_is_fence_for_every_estimator():
    for est in ("ste", "soft", "st-gumbel", "hard-concrete"):
        g = MaskGates(4, arm="s3", estimator=est)
        hard = g.gate_table(mode="hard")
        want = g.fence_on_learn.float().view(-1, 1).expand_as(hard)
        assert torch.equal(hard, want), est


def test_estimator_ranges_and_st_hardness():
    torch.manual_seed(0)
    for est in ("ste", "soft", "st-gumbel", "hard-concrete"):
        g = MaskGates(6, arm="s3", estimator=est)
        t = g.gate_table(tau=1.0, mode="train").detach()
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0, est
        if est in ("ste", "st-gumbel"):  # ST estimators emit HARD forward values
            assert set(t.detach().unique().tolist()) <= {0.0, 1.0}, est


def test_soft_anneal_approaches_hard():
    g = MaskGates(3, arm="s1", estimator="soft")
    t = g.gate_table(tau=1e-3, mode="train")
    assert torch.allclose(t, g.gate_table(mode="hard"), atol=1e-6)


def test_gradients_flow_to_gates():
    """CE-shaped loss through the assembled mask reaches the logits for every
    estimator (the straight-through contract)."""
    learn = arm_learn_mask("s2")
    base, lidx = mask_parts(CM, learn, fence_open())
    for est in ("ste", "soft", "st-gumbel", "hard-concrete"):
        torch.manual_seed(1)
        g = MaskGates(2, arm="s2", estimator=est)
        gt = g.gate_table(tau=1.0, mode="train")
        loss = sum(assemble_mask(base, lidx, gt[:, li], SOFT_FORBID).sum()
                   for li in range(2))
        loss.backward()
        assert g.logits.grad is not None and float(g.logits.grad.abs().sum()) > 0, est


def test_deviation_penalty_at_init():
    g = MaskGates(28, arm="s2", init_logit=2.0)
    open_pen, close_pen = (t.detach() for t in g.deviation())
    n_off = int((~g.fence_on_learn).sum())  # R4x6 + R5x6 + R7 = 13
    n_on = int(g.fence_on_learn.sum())      # R6 = 1
    assert n_off == 13 and n_on == 1
    assert abs(float(open_pen) - torch.sigmoid(torch.tensor(-2.0)) * n_off * 28) < 1e-3
    assert abs(float(close_pen) - torch.sigmoid(torch.tensor(-2.0)) * n_on * 28) < 1e-3


def test_flips_and_heatmap():
    g = MaskGates(4, arm="s1")
    assert g.flips() == 0
    full = g.full_p_open()
    assert full.shape == (N_CH, 4)
    assert torch.equal(full[0], torch.ones(4))   # R1 frozen-open -> exact 1
    assert torch.equal(full[1], torch.zeros(4))  # R2[Δ1] frozen-closed -> exact 0
    with torch.no_grad():
        g.logits[-1, 0] = 3.0  # open R7 at layer 0
    assert g.flips() == 1


def test_hard_open_table():
    """At init the hard-frozen table is the fence at every layer; a flipped logit
    shows up as the channel opening at exactly that layer."""
    from gnnformer.learnmask import fence_open as _fo, hand_open_table

    g = MaskGates(4, arm="s2")
    want = _fo().view(-1, 1).repeat(1, 4)
    assert torch.equal(g.hard_open_table(), want)
    with torch.no_grad():
        g.logits[-1, 2] = 3.0  # R7 opens at layer 2
    t = g.hard_open_table()
    r7 = N_CH - 1
    assert t[r7].tolist() == [False, False, True, False]
    # the deployed hand design == fence + R4/R7 columns >= l_open
    ht = hand_open_table(4, l_open=2)
    assert torch.equal(ht[:, 0], _fo()) and bool(ht[r7, 2]) and bool(ht[r7, 3])


def test_hard_mask_lut_matches_hard_mask():
    """The device-friendly LUT assembly (large-N transfer path) == mask_parts-based
    hard_mask, bit for bit, for fence/hand/nofence tables."""
    from gnnformer.learnmask import hard_mask, hard_mask_lut

    for open_ch in (fence_open(), hand_open(20, 12),
                    torch.ones(N_CH, dtype=torch.bool)):
        assert torch.equal(hard_mask_lut(CM, open_ch), hard_mask(CM, open_ch))


def test_freetable_s0():
    """S0 per-cell table: hard init reproduces the fence bit-for-bit; gradients reach
    the cell logits; the audit heatmap groups by declared relation."""
    from gnnformer.learnmask import FreeTableGates, SOFT_FORBID, hard_mask

    ft = FreeTableGates(CM, n_layers=3, scope_arm="s2", estimator="ste")
    lo = hard_mask(CM, fence_open())
    gt = ft.gate_table(mode="hard")
    for li in range(3):
        assert torch.equal(ft.layer_mask(gt[:, li], MASK_MIN), lo)
    # gradient flows through the scatter
    gt_tr = ft.gate_table(tau=1.0, mode="train")
    loss = ft.layer_mask(gt_tr[:, 0], SOFT_FORBID).sum()
    loss.backward()
    assert ft.logits.grad is not None and float(ft.logits.grad.abs().sum()) > 0
    # audit heatmap: frozen R1 row exactly 1, learnable rows = per-cell means
    full = ft.full_p_open()
    assert full.shape == (N_CH, 3)
    assert torch.equal(full[_CH["R1"]], torch.ones(3))
    assert 0.0 < float(full[_CH["R7"], 0]) < 0.5  # fence-OFF learnable cells


def test_state_roundtrip():
    g = MaskGates(5, arm="s2", estimator="st-gumbel", init_logit=1.5)
    with torch.no_grad():
        g.logits.add_(torch.randn_like(g.logits) * 0.3)
    g2 = MaskGates.from_state(g.state())
    assert torch.equal(g.logits, g2.logits)
    assert (g2.arm, g2.estimator, g2.init_logit) == ("s2", "st-gumbel", 1.5)
    assert g2.channel_names == g.channel_names


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
