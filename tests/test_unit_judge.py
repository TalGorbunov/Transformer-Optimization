"""CPU test for experiments/unit_judge.py (the UNIT baseline's regime D) on a TINY randomly
initialised Qwen2.5-VL with the real processor: the judge line picks frame / clip, the yes / no
vocabulary is one token per spelling, score_unit returns finite log-odds with P(yes) + P(no) <= 1
and the prompt's token count, a one-frame prompt is shorter than a three-frame one, and
summarise() turns per-unit records into the gate_fit operating-point table (hard / easy / all).
Needs the Qwen2.5-VL config and processor in the HF cache (no weights); skips loudly otherwise.

Run: python tests/test_unit_judge.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import get_backbone
from core.constants import MODEL_ID
from experiments.unit_judge import JUDGE_TEXT, judge_text, judge_vocab, score_unit, summarise

_CACHE = {}


def _tiny():
    if "m" in _CACHE:
        return _CACHE["m"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        cfg = AutoConfig.from_pretrained(MODEL_ID)
        proc = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    except Exception as exc:
        print(f"  [skip] Qwen2.5-VL config / processor unavailable: {str(exc)[:120]}")
        _CACHE["m"] = None
        return None
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 2
    tc.num_attention_heads, tc.num_key_value_heads = 4, 2
    rs = dict(tc.rope_scaling or {})
    rs["mrope_section"] = [2, 3, 3]
    tc.rope_scaling = rs
    vc.depth, vc.hidden_size, vc.num_heads, vc.intermediate_size, vc.out_hidden_size = 2, 32, 2, 64, 64
    vc.fullatt_block_indexes = [1]
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    _CACHE["m"] = (model, proc)
    return _CACHE["m"]


def _frames(n, size=(112, 112), seed=0):
    rng = np.random.RandomState(seed)
    return [Image.fromarray(rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)) for _ in range(n)]


def test_judge_text():
    assert judge_text(1) == JUDGE_TEXT.format(what="frame") and judge_text(5) == JUDGE_TEXT.format(what="clip")
    assert judge_text(1).startswith("Does this frame contain evidence needed to answer the question above? Answer yes or no.")


def test_score_unit_tiny():
    t = _tiny()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("qwen2.5-vl-7b")
    vocab = judge_vocab(proc.tokenizer, spec.name)
    assert vocab["yes"] == [9454, 9693, 7414] and vocab["no"] == [2753, 2152, 2308]
    q = "How many times does the person pick up ginger?\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5"
    r1 = score_unit(spec, model, proc, proc.tokenizer, _frames(1), q + "\n\n" + judge_text(1), vocab, "cpu")
    r3 = score_unit(spec, model, proc, proc.tokenizer, _frames(3), q + "\n\n" + judge_text(3), vocab, "cpu")
    for r in (r1, r3):
        assert math.isfinite(r["score"]) and 0.0 < r["p_yes"] < 1.0 and 0.0 < r["p_no"] < 1.0 and r["p_yes"] + r["p_no"] <= 1.0 + 1e-6
        assert abs(r["score"] - (math.log(r["p_yes"]) - math.log(r["p_no"]))) < 1e-4
        assert isinstance(r["top1"], str) and r["tokens"] > 0
    assert r3["tokens"] > r1["tokens"], "three frames carry more tokens than one"
    r1b = score_unit(spec, model, proc, proc.tokenizer, _frames(1), q + "\n\n" + judge_text(1), vocab, "cpu")
    assert r1b["score"] == r1["score"], "deterministic"
    rs = score_unit(spec, model, proc, proc.tokenizer, _frames(1), q, vocab, "cpu", system_prompt="You are a judge.")
    assert rs["tokens"] > r1["tokens"] - 30 and math.isfinite(rs["score"])
    print(f"  tiny model: 1 frame {r1['tokens']} tok score {r1['score']:+.3f}; 3 frames {r3['tokens']} tok score {r3['score']:+.3f}")


def test_summarise():
    rng = np.random.RandomState(0)
    recs = []
    for task in ("AC", "FAM"):
        for i in range(40):
            recs.append({"qtype": task, "kind": "pos", "score": float(rng.normal(1.0, 1.0)), "p_yes": 0.6})
            recs.append({"qtype": task, "kind": "hard", "score": float(rng.normal(0.3, 1.0)), "p_yes": 0.4})
            recs.append({"qtype": task, "kind": "easy", "score": float(rng.normal(-1.0, 1.0)), "p_yes": 0.2})
    m = summarise(recs)
    keys = {(r["task"], r["neg"]) for r in m}
    assert keys == {(t, n) for t in ("AC", "FAM", "all") for n in ("hard", "easy", "all")}
    by = {(r["task"], r["neg"]): r for r in m}
    assert by[("all", "easy")]["auc"] > by[("all", "hard")]["auc"] > 0.5, "easy negatives are easier"
    assert by[("AC", "all")]["n_pos"] == 40 and by[("AC", "all")]["n_neg"] == 80 and by[("all", "all")]["n_neg"] == 160
    assert 0.0 <= by[("AC", "hard")]["recall@spec95"] <= 1.0 and by[("AC", "hard")]["p_yes_pos"] == 0.6
    # MMReD-style records: one negative kind "neg" -> the "all" table only
    mm = summarise([{"qtype": "x", "kind": "pos", "score": 1.0, "p_yes": 0.6}] * 25 + [{"qtype": "x", "kind": "neg", "score": -1.0, "p_yes": 0.1}] * 25)
    assert {(r["task"], r["neg"]) for r in mm} == {("x", "neg"), ("x", "all"), ("all", "neg"), ("all", "all")}
    assert mm[0]["auc"] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
