"""CPU checks for the many-needle instrument (scripts/condmask/needles.py).

  python tests/test_needles.py

1. story(): K needles all match the question's (person, room); no distractor does; K + D lines.
2. build() with the counting PROMPT: blocks tile and each block decodes back to its sentence.
3. cand_logp (KV-cached, cache cropped between candidates) == teacher-forced full forward.
4. needle_metrics: per-needle mass sums to osq_probe's `mass`; Jacobian sums reproduce `infl`.
Tokenizer = cached Qwen2.5-0.5B-Instruct (chat template; digits are single tokens), model = a
random tiny Llama over that vocabulary.
"""
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "condmask"))
import needles as nd  # noqa: E402
from eval_babilong import build  # noqa: E402
from osq import osq_probe  # noqa: E402

torch.manual_seed(0)


def tiny(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      attn_implementation="sdpa")
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_story():
    rng = random.Random(0)
    for K, D in [(1, 0), (4, 64), (16, 48), (64, 0)]:
        sents, facts, q = nd.story(K, D, rng)
        assert len(sents) == K + D and len(facts) == K
        p, r = q.split("did ")[1].split(" go")[0], q.split("the ")[-1].rstrip("?")
        for j, s in enumerate(sents):
            assert (s.startswith(p) and s.endswith(f"the {r}.")) == (j in facts), (s, q)


def test_build_and_probe(tok, model):
    rng = random.Random(1)
    sents, facts, q = nd.story(3, 5, rng)
    rec = build(tok, "qa1", q, sents, nd.PROMPT)
    b = rec["blocks"]
    assert b[0][0] == rec["prefix_end"] and b[-1][1] == rec["fin"]
    assert all(b[j][1] == b[j + 1][0] for j in range(len(b) - 1))
    for (a, e), s in zip(b, sents):
        assert tok.decode(rec["ids"][a:e]).strip() == s
    assert "How many times did" in tok.decode(rec["ids"][rec["fin"]:])
    ids, pos = rec["ids"], list(range(rec["seq"]))
    ex = {}
    row = osq_probe(model, ids, pos, b, facts, 1, 7, "cpu", jac=True, ckpt=False, extra=ex)
    m = nd.needle_metrics(ex, rec, facts)
    assert abs(m["mass_needles"] - row["mass"]) < 1e-5, (m["mass_needles"], row["mass"])
    a0, b1 = b[0][0], b[-1][1]
    assert abs(m["jac_needle_sum"] / float(ex["gn"][a0:b1].sum()) - row["infl"]) < 1e-5
    assert 1 <= m["keff_needles"] <= 3 + 1e-6 and -1 <= m["cos_needles"] <= 1
    return ids


def test_cand_logp(tok, model, ids):
    cands = ["2", "3", "12"]
    got = nd.cand_logp(model, tok, ids, tok.eos_token_id, cands, "cpu")
    for c, g in zip(cands, got):
        t = tok(c, add_special_tokens=False).input_ids + [tok.eos_token_id]
        with torch.no_grad():
            lg = torch.log_softmax(model(input_ids=torch.tensor([ids + t])).logits[0].float(), -1)
        ref = sum(float(lg[len(ids) - 1 + i, t[i]]) for i in range(len(t)))
        assert abs(g - ref) < 1e-3, (c, g, ref)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = tiny(tok)
    test_story()
    ids = test_build_and_probe(tok, model)
    test_cand_logp(tok, model, ids)
    print("test_needles: OK")
