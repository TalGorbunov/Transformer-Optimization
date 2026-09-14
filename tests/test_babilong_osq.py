"""CPU checks for the BABILong over-squashing probe and the bounded-degree selectors.

  python tests/test_babilong_osq.py

1. osq_from_scores on PLANTED scores reproduces the closed forms exactly (mass = the
   dispersion law, gap = planted margin, m_eff = margin, fan-ins).
2. tournament_scores: finite exactly on the finalists, recursion when the union overflows,
   identical to attn_scores when the record fits one window.
3. two_round_rec: copies land in the tail, blocks untouched, chunking still works.
4. arm_positions: gap arms shift the right span; decode runs on gapped positions.
5. osq_probe: checkpointed and plain Jacobians agree; shares in [0, 1].
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "condmask"))

from osq import osq_from_scores, osq_probe, jac_share  # noqa: E402
import eval_babilong as eb  # noqa: E402

torch.manual_seed(0)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      attn_implementation="sdpa")
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def make_rec(F=12, L=5, prefix=3, tail=3):
    ids, blocks = list(range(100, 100 + prefix)), []
    for j in range(F):
        blocks.append((len(ids), len(ids) + L))
        ids += [200 + j] * L
    fin = len(ids)
    ids += list(range(300, 300 + tail))
    return {"ids": ids, "prefix_end": prefix, "blocks": blocks, "fin": fin, "seq": len(ids)}


def test_planted_scores():
    rec = make_rec(F=10, L=3, prefix=2, tail=2)
    facts, m, H = [2, 7], 1.5, 2
    sc = torch.zeros(H, rec["seq"])
    for j in facts:
        sc[:, rec["blocks"][j][0]: rec["blocks"][j][1]] = m
    o = osq_from_scores(sc, rec["blocks"], facts)
    kF, NT = 6, 30
    mass = kF * math.exp(m) / (kF * math.exp(m) + NT - kF)
    assert abs(o["mass"] - mass) < 1e-6 and abs(o["pred_mass"] - mass) < 1e-6
    assert abs(o["gap"] - m) < 1e-6 and abs(o["m_eff"] - m) < 1e-6
    # raw sides of the gap: logit(mass) = lse_e - lse_j; planted evidence max = m
    assert abs((o["lse_e"] - o["lse_j"]) - math.log(mass / (1 - mass))) < 1e-5 and abs(o["smax_e"] - m) < 1e-6
    z = kF * math.exp(m) + NT - kF
    keff_tok = 1 / (kF * (math.exp(m) / z) ** 2 + (NT - kF) * (1 / z) ** 2)
    keff_sent = 1 / (2 * (3 * math.exp(m) / z) ** 2 + 8 * (3 / z) ** 2)
    assert abs(o["keff_tok"] - keff_tok) < 1e-4 and abs(o["keff_sent"] - keff_sent) < 1e-4
    assert o["n_tok"] == NT and o["n_fact_tok"] == kF
    # oracle-like record (no junk): gap undefined, mass 1
    o2 = osq_from_scores(sc, rec["blocks"], list(range(10)))
    assert o2["gap"] is None and abs(o2["mass"] - 1) < 1e-6
    # planted Jacobian norms
    gn = torch.zeros(rec["seq"])
    gn[rec["blocks"][2][0]: rec["blocks"][2][1]] = 2.0
    gn[rec["blocks"][5][0]: rec["blocks"][5][1]] = 1.0
    js = jac_share(gn, rec["blocks"], facts)
    assert abs(js["infl"] - 6 / 9) < 1e-6 and abs(js["keff_jac"] - 1 / ((2 / 3) ** 2 + (1 / 3) ** 2)) < 1e-6


def test_tournament(model):
    rec = make_rec(F=12, L=5, prefix=3, tail=3)              # seq 66
    full = eb.attn_scores(model, rec, 1, "tail", "cpu").float()
    same = eb.tournament_scores(model, rec, 1, "tail", "cpu", chunk=100, k_local=2)
    assert torch.allclose(full, same), "one window must reduce to attn_scores"
    t = eb.tournament_scores(model, rec, 1, "tail", "cpu", chunk=30, k_local=2)
    # chunk 30 -> budget 24 -> groups of 4 sentences (3 groups) -> 6 finalists (seq 36 > 30)
    # -> recursion: 2 groups -> 4 finalists -> final window of 26 tokens
    assert t.shape == (12,) and torch.isfinite(t).sum() == 4
    fin = torch.isfinite(t).nonzero().flatten().tolist()
    sub = eb.attn_scores(model, eb.sub_rec(rec, fin)[0], 1, "tail", "cpu").float()
    assert torch.allclose(t[fin], sub), "finalist scores come from ONE final softmax"
    assert eb.topk_keep(t, 2) == sorted(fin[i] for i in sub.topk(2).indices.tolist())
    # stage trace: same scores; stage 1 = 3 windows -> 6 finalists, stage 2 = 2 windows -> the
    # 4 finalists of the final softmax; ranks are per-window (0..3 at stage 1, 0..2 at stage 2)
    tr = []
    assert torch.allclose(eb.tournament_scores(model, rec, 1, "tail", "cpu", chunk=30, k_local=2, trace=tr), t)
    assert [e["windows"] for e in tr] == [3, 2] and len(tr[0]["finalists"]) == 6 and tr[1]["finalists"] == fin
    assert set(tr[1]["finalists"]) < set(tr[0]["finalists"]) and set(tr[0]["rank"]) == set(range(12))
    assert set(tr[1]["rank"]) == set(tr[0]["finalists"]) and max(tr[0]["rank"].values()) == 3
    assert all(tr[0]["rank"][j] < 2 for j in tr[0]["finalists"]) and all(tr[1]["rank"][j] < 2 for j in fin)
    # windows too small to prune (k_local >= sentences per window): must terminate, = concat
    t2 = eb.tournament_scores(model, rec, 1, "tail", "cpu", chunk=12, k_local=2)
    assert torch.allclose(t2, eb.chunked_scores(model, rec, 1, "tail", "cpu", 12).float().cpu())


def test_two_round(model):
    import types
    rec = make_rec(F=12, L=5, prefix=3, tail=3)
    r2 = eb.two_round_rec(rec, [1, 5])
    assert r2["blocks"] == rec["blocks"] and r2["fin"] == rec["fin"]
    assert r2["seq"] == rec["seq"] + 10 and r2["rows2"] == (rec["fin"], rec["fin"] + 10)
    assert r2["ids"][rec["fin"]: rec["fin"] + 10] == [201] * 5 + [205] * 5
    assert r2["ids"][rec["fin"] + 10:] == rec["ids"][rec["fin"]:]
    sr, _ = eb.sub_rec(r2, [0, 1])
    assert sr["seq"] == 3 + 10 + 13 and sr["fin"] == 13 and sr["rows2"] == (13, 23)
    # copy rows score every sentence, in one window and chunked (the copies ride in the tail)
    s = eb.attn_scores(model, r2, 1, "copies", "cpu")
    sc = eb.chunked_scores(model, r2, 1, "copies", "cpu", chunk=40)
    assert s.shape == sc.shape == (12,) and torch.isfinite(s).all() and torch.isfinite(sc).all()
    # rounds: the round-1 kept keep their round-1 share, the rest gain the copy rows' share
    args = types.SimpleNamespace(sel_tournament=0, sel_chunk=0, sel_layer=1, sel_rows="tail", sel_rounds=2, k=4)
    m1 = eb.attn_scores(model, rec, 1, "tail", "cpu").float()
    m = eb.select_scores(model, rec, args, "cpu")
    kept = eb.topk_keep(m1, 4)
    assert abs(float(m.sum()) - 2) < 1e-4 and torch.allclose(m[kept], (m1 / m1.sum())[kept], atol=1e-5)
    args.sel_rounds = 1
    assert torch.allclose(eb.select_scores(model, rec, args, "cpu").float(), m1)


def test_positions(model):
    rec = make_rec(F=12, L=5, prefix=3, tail=3)
    sr, idx = eb.sub_rec(rec, [2, 7])
    assert eb.arm_positions("full", sr, idx) == list(range(sr["seq"]))
    assert eb.arm_positions("attn_norp", sr, idx) == idx
    g = eb.arm_positions("oracle_gap@1000", sr, idx)
    assert g[: sr["fin"]] == list(range(sr["fin"])) and g[sr["fin"]] == sr["fin"] + 1000
    pg = eb.arm_positions("oracle_pgap@1000", sr, idx)
    assert pg[: 3] == [0, 1, 2] and pg[3] == 1003 and pg[-1] == sr["seq"] - 1 + 1000
    out = eb.decode(model, sr["ids"], g, set(), "cpu", 2)
    assert len(out) == 2


def test_probe(model):
    rec = make_rec(F=12, L=5, prefix=3, tail=3)
    facts = [2, 7]
    pos = list(range(rec["seq"]))
    a = osq_probe(model, rec["ids"], pos, rec["blocks"], facts, 1, gold=7, dev="cpu", jac=True, ckpt=True)
    b = osq_probe(model, rec["ids"], pos, rec["blocks"], facts, 1, gold=7, dev="cpu", jac=True, ckpt=False)
    for k in ("mass", "keff_tok", "keff_sent", "gap", "pred_mass", "m_eff", "infl", "keff_jac"):
        assert a[k] is not None and math.isfinite(a[k]), k
        assert abs(a[k] - b[k]) < 1e-5, (k, a[k], b[k])
    assert 0 <= a["mass"] <= 1 and 0 <= a["infl"] <= 1 and a["keff_sent"] >= 1
    assert a["pred_tok"] == int(model(torch.tensor([rec["ids"]])).logits[0, -1].argmax())
    # the probe's mass at the last row == attn_scores' last-row sentence masses (renormalised)
    w = eb.sent_mass(eb._row_mass(model.model.layers[1], _h_at(model, rec, 1), _pe(model, rec), rec, "last"),
                     rec["blocks"])
    assert abs(a["mass"] - float(w[facts].sum() / w.sum())) < 1e-5


def _pe(model, rec):
    h = model.model.embed_tokens(torch.tensor([rec["ids"]]))
    c, s = model.model.rotary_emb(h, torch.arange(rec["seq"])[None])
    return (c, s)


def _h_at(model, rec, layer):
    h = model.model.embed_tokens(torch.tensor([rec["ids"]]))
    pos = torch.arange(rec["seq"])[None]
    pe = _pe(model, rec)
    with torch.no_grad():
        for li in range(layer):
            o = model.model.layers[li](h, attention_mask=None, position_ids=pos, position_embeddings=pe)
            h = o[0] if isinstance(o, tuple) else o
    return h


if __name__ == "__main__":
    test_planted_scores()
    m = tiny_model()
    test_tournament(m)
    test_two_round(m)
    test_positions(m)
    test_probe(m)
    print("test_babilong_osq: OK")
