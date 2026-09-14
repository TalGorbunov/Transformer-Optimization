"""CPU tests for scripts/condmask — run after ANY change there (repo law).

python tests/test_condmask.py

Covers, in order:
  1. framewise-token integrity (the text-variant "frame tokens" check)
  2. mask semantics: fixed-vs-gated bit-exact parity + post-softmax zero weight on
     closed cells (MASK_MIN exact 0, SOFT_FORBID < 1e-10)
  3. ST-Gumbel exactness ({0,1} forward values, nonzero grads, both mask dtypes)
  4. the plumbing CANARY: gates provably FLIP to a planted target pattern through the
     same gumbel+assembly path, from BOTH inits, within 300 Adam steps (anti-S0)
  5. padded rows produce no NaN through SDPA
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "condmask"))

from gnnformer.constants import MASK_MIN
from gatenet import GateNet, pool_embeddings, st_gumbel, tau_schedule
from masks import (SOFT_FORBID, assemble_batch_masks, batch_mask_parts, fixed_gates,
                   layout_mask_parts)
from textdata import build_segmented_prompt, majority_baseline, render_frame_line

TOK_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _mk_states(n: int):
    rooms = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]
    out = []
    for i in range(n):
        occ = {r: [] for r in rooms}
        occ[rooms[i % 3]] = ["Sandra"]
        occ["Garden"] = sorted(["John", "Mary"]) if i % 2 else ["John"]
        out.append(repr({"step_id": i + 1, "rooms": occ}))
    return out


def test_framewise_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    states = _mk_states(8)
    q = "How many steps did Sandra spend in the Kitchen?"
    rec = build_segmented_prompt(tok, states, q, gold=3, sid="t0")
    assert rec is not None
    # blocks tile [prefix_end, fin) exactly
    assert rec["blocks"][0][0] == rec["prefix_end"]
    for (a1, b1), (a2, _) in zip(rec["blocks"], rec["blocks"][1:]):
        assert b1 == a2, "blocks must tile contiguously"
    assert rec["blocks"][-1][1] == rec["fin"]
    # every block decodes back to exactly its frame line
    for (a, b), s in zip(rec["blocks"], states):
        assert tok.decode(rec["ids"][a:b]) == "\n" + render_frame_line(s)
    # tail carries the canonical PROMPT-CRITICAL ending; answer position = last token
    tail = tok.decode(rec["ids"][rec["fin"]:])
    assert tail.endswith("Answer: ") and tail.startswith("\nQuestion:")
    assert rec["seq"] == len(rec["ids"])
    m_cls, m_frac = majority_baseline([0, 0, 1, 2])
    assert m_cls == 0 and abs(m_frac - 0.5) < 1e-9
    print("  test_framewise_tokens OK")


def _toy_rec(seq=20, nb=3):
    blocks = [(4 + 4 * i, 8 + 4 * i) for i in range(nb)]
    return {"seq": seq, "prefix_end": 4, "blocks": blocks, "fin": blocks[-1][1]}


def test_mask_semantics():
    rec = _toy_rec()
    S = rec["seq"] + 2  # padded
    base, template = layout_mask_parts(rec, S)
    # hand-built blockwise reference
    ref = base.clone()
    blk = torch.full((S,), -1)
    for i, (a, b) in enumerate(rec["blocks"]):
        blk[a:b] = i
    for qi in range(S):
        for ki in range(qi + 1):
            if blk[qi] >= 0 and blk[ki] >= 0 and blk[qi] != blk[ki]:
                ref[qi, ki] = MASK_MIN
    bases, templates = base.unsqueeze(0), template.unsqueeze(0)
    m_block = assemble_batch_masks(bases, templates, fixed_gates(1, 2, "blockwise"),
                                   MASK_MIN)
    m_full = assemble_batch_masks(bases, templates, fixed_gates(1, 2, "full"), MASK_MIN)
    assert torch.equal(m_block[0, 0], ref), "g=1 must equal hand-built blockwise"
    assert torch.equal(m_block[0, 1], ref)
    assert torch.equal(m_full[0, 0], base), "g=0 must equal plain causal"
    # template never touches prefix cols, tail rows, or above-diagonal
    assert not template[:, : rec["prefix_end"]].any()
    assert not template[rec["fin"]:].any()
    assert not template.triu(1).any()
    # post-softmax weight on closed cells: exact 0 under MASK_MIN, < 1e-10 under -30
    torch.manual_seed(0)
    scores = torch.randn(S, S)
    for K, bound in ((MASK_MIN, 0.0), (SOFT_FORBID, 1e-10)):
        m = assemble_batch_masks(bases, templates, fixed_gates(1, 1, "blockwise"), K
                                 )[0, 0]
        w = torch.softmax(scores + m, dim=-1)
        closed = template & (base == 0)
        assert float(w[closed].max()) <= bound, f"K={K}: closed weight leaks"
    print("  test_mask_semantics OK")


def test_st_exact():
    torch.manual_seed(0)
    logits = torch.randn(64, requires_grad=True)
    g = st_gumbel(logits, tau=1.0, mode="train")
    assert set(g.detach().unique().tolist()) <= {0.0, 1.0}, "ST forward must be exact 0/1"
    g.sum().backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
    hard = st_gumbel(logits, tau=1.0, mode="hard")
    assert torch.equal(hard, (logits > 0).float())
    # gradient survives the cast to both mask dtypes
    rec = _toy_rec()
    base, template = layout_mask_parts(rec, rec["seq"])
    for dt in (torch.float32, torch.bfloat16):
        lg = torch.zeros(1, 2, requires_grad=True)
        gg = st_gumbel(lg + 0.3, tau=1.0, mode="train")
        m = assemble_batch_masks(base.unsqueeze(0), template.unsqueeze(0), gg,
                                 SOFT_FORBID, dtype=dt)
        m.float().sum().backward()
        assert lg.grad is not None and torch.isfinite(lg.grad).all()
        assert float(lg.grad.abs().sum()) > 0, f"no grad through {dt} mask"
    assert tau_schedule(0, 100) == 2.0 and tau_schedule(29, 100) == 2.0
    assert abs(tau_schedule(99, 100) - 0.5) < 1e-6
    print("  test_st_exact OK")


def test_canary_flips():
    """Anti-S0: the machinery must FLIP gates to a planted target within 300 steps,
    from BOTH inits, through the same gumbel+assembly path used in training."""
    rec = _toy_rec()
    L, H, B = 4, 2, 4
    base, template = layout_mask_parts(rec, rec["seq"])
    bases = base.unsqueeze(0).expand(B, -1, -1)
    templates = template.unsqueeze(0).expand(B, -1, -1)
    torch.manual_seed(0)
    x = torch.randn(B, 32)
    target = (torch.rand(L, H) > 0.5).float()
    tgt_masks = [assemble_batch_masks(bases, templates,
                                      target[l].expand(B, -1), SOFT_FORBID)
                 for l in range(L)]
    for init in ("blockwise", "random"):
        net = GateNet(32, L, H, hidden=64, init=init, seed=1)
        opt = torch.optim.Adam(net.parameters(), lr=3e-2)
        ok_at = -1
        for step in range(300):
            g = st_gumbel(net(x), tau=1.0, mode="train")
            loss = sum(F.mse_loss(
                assemble_batch_masks(bases, templates, g[:, l], SOFT_FORBID),
                tgt_masks[l]) for l in range(L))
            opt.zero_grad(); loss.backward(); opt.step()
            hard = st_gumbel(net(x), tau=1.0, mode="hard")
            if all(torch.equal(hard[b], target) for b in range(B)):
                ok_at = step
                break
        assert ok_at >= 0, f"canary FAILED from init={init}: gates never matched target"
        print(f"  test_canary_flips OK (init={init}, converged at step {ok_at})")


def test_pad_rows_no_nan():
    torch.manual_seed(0)
    recs = [_toy_rec(seq=20), _toy_rec(seq=14, nb=2)]
    S = 22
    bases, templates = batch_mask_parts(recs, S)
    g = torch.rand(2, 2).round()
    m = assemble_batch_masks(bases, templates, g, MASK_MIN)
    q = torch.randn(2, 2, S, 8)
    out = F.scaled_dot_product_attention(q, q, q, attn_mask=m)
    assert torch.isfinite(out).all(), "padded rows produced NaN/inf"
    print("  test_pad_rows_no_nan OK")


def _rec4():
    return {"prefix_end": 5, "blocks": [(5, 9), (9, 14), (14, 20)], "fin": 20,
            "seq": 26}


def test_soft_repack_vertex_parity():
    """soft_repack_positions at g in {0,1} == repack_positions on every position
    that participates in the computation (prefix, kept blocks, tail). Dropped
    blocks are allowed to differ — their columns are hard-masked."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/condmask"))
    from coformer import repack_positions, soft_repack_positions
    rec, S = _rec4(), 28
    for bits in range(8):
        keep = torch.tensor([(bits >> i) & 1 for i in range(3)], dtype=torch.float32)
        hard = repack_positions(rec, keep > 0.5, S, "cpu").float()
        soft = soft_repack_positions(rec, keep, S, "cpu")
        assert torch.equal(soft[:5], hard[:5])
        for i, (a, b) in enumerate(rec["blocks"]):
            if keep[i] > 0.5:
                assert torch.equal(soft[a:b], hard[a:b]), f"kept block {i} @ {bits}"
        assert torch.equal(soft[rec["fin"]:], hard[rec["fin"]:]), f"tail @ {bits}"
    print("  test_soft_repack_vertex_parity OK")


def test_soft_positions_carry_gradient():
    """The whole point of AS4: d(loss)/d(gate) through POSITIONS is nonzero."""
    from coformer import soft_repack_positions, soft_rope

    class _Rot:
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
        attention_scaling = 1.0

    rec, S = _rec4(), 28
    theta = torch.zeros(3, requires_grad=True)
    g = torch.sigmoid(theta)
    pos = soft_repack_positions(rec, g, S, "cpu")
    cos, sin = soft_rope(_Rot(), pos.unsqueeze(0), torch.float32)
    torch.manual_seed(0)
    (cos * torch.randn_like(cos) + sin * torch.randn_like(sin)).sum().backward()
    assert theta.grad is not None and float(theta.grad.abs().sum()) > 0, \
        "no gradient through soft positions"
    print("  test_soft_positions_carry_gradient OK")


def test_soft_rope_matches_hf():
    """soft_rope on integer positions == HF rotary (which is @torch.no_grad)."""
    from coformer import soft_rope
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    cfg = Qwen2Config(hidden_size=64, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=128,
                      num_hidden_layers=1)
    rot = Qwen2RotaryEmbedding(config=cfg)
    pos = torch.arange(12).unsqueeze(0)
    ch, sh = rot(torch.zeros(1, 12, 64), pos)
    cs, ss = soft_rope(rot, pos.float(), torch.float32)
    assert torch.allclose(ch.float(), cs, atol=1e-5), "cos mismatch vs HF"
    assert torch.allclose(sh.float(), ss, atol=1e-5), "sin mismatch vs HF"
    print("  test_soft_rope_matches_hf OK")


def test_babilong_segments():
    """BABILong builder: sentence spans tile [prefix_end, fin) exactly and decode back
    to their sentences; repack/noRP index construction is the identity when all kept;
    the official scorer accepts the templated answer and rejects a wrong label."""
    from transformers import AutoTokenizer
    from eval_babilong import build, split_sents
    from babilong_official.metrics import TASK_LABELS, compare_answers
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    ctx = ("It was a dark night.\nThe wind howled. John moved to the hallway. "
           "Nobody spoke, Mrs. Boyd least of all. Mary went to the garden.")
    sents = split_sents(ctx)
    assert "John moved to the hallway." in sents and "Mary went to the garden." in sents
    rec = build(tok, "qa1", "Where is John?", sents)
    bl = rec["blocks"]
    assert bl[0][0] == rec["prefix_end"] and bl[-1][1] == rec["fin"] < rec["seq"]
    assert all(bl[i][1] == bl[i + 1][0] for i in range(len(bl) - 1))
    for i, (a, b) in enumerate(bl):
        assert tok.decode(rec["ids"][a:b]) == ("" if i == 0 else " ") + sents[i]
    assert "Where is John?" in tok.decode(rec["ids"][rec["fin"]:])
    q = "Where is John?"
    assert compare_answers("hallway", "The most recent location of John is hallway.", q,
                           TASK_LABELS["qa1"])
    assert not compare_answers("hallway", "John is in the garden.", q, TASK_LABELS["qa1"])
    assert not compare_answers("hallway", "hallway or garden", q, TASK_LABELS["qa1"])
    # sub_rec: keeping everything is the identity; a subset keeps order and spans decode
    import eval_babilong as eb
    sr, idx = eb.sub_rec(rec, list(range(len(sents))))
    assert sr["ids"] == rec["ids"] and idx == list(range(rec["seq"])) and sr["blocks"] == bl
    sr, idx = eb.sub_rec(rec, [1, 3])
    assert [tok.decode(sr["ids"][a:b]).strip() for a, b in sr["blocks"]] == [sents[1], sents[3]]
    assert idx[sr["fin"]:] == list(range(rec["fin"], rec["seq"]))
    # chunked_scores: every sentence scored exactly once, in order, windows <= chunk
    seen = []
    def fake_scores(model, r, layer, rows, dev):
        assert r["seq"] <= chunk
        seen.append(len(r["blocks"]))
        return torch.tensor([float(b - a) for a, b in r["blocks"]])
    chunk = rec["prefix_end"] + (rec["seq"] - rec["fin"]) + 12
    eb.attn_scores, real = fake_scores, eb.attn_scores
    try:
        m = eb.chunked_scores(None, rec, 0, "tail", "cpu", chunk)
    finally:
        eb.attn_scores = real
    assert m.tolist() == [float(b - a) for a, b in bl] and len(seen) > 1 and sum(seen) == len(bl)


if __name__ == "__main__":
    test_babilong_segments()
    test_framewise_tokens()
    test_mask_semantics()
    test_st_exact()
    test_canary_flips()
    test_pad_rows_no_nan()
    test_soft_repack_vertex_parity()
    test_soft_positions_carry_gradient()
    test_soft_rope_matches_hf()
    print("ALL condmask tests passed")
