"""CPU tests for experiments/_diag_common.py: the DIAG pair protocols on synthetic states, the
rank AUC, the first-token margin, and row_attention's arithmetic against a direct softmax."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from core.constants import CHARS, ROOMS  # noqa: E402
from core.mmred import recompute_answer  # noqa: E402
from experiments import _diag_common as dc  # noqa: E402


def _row(qtype, question, answer, seq):
    return {"qid": "t", "qtype": qtype, "question": question, "answer": answer,
            "sequence": [{"step_id": i + 1, "rooms": {r: list(rooms.get(r, [])) for r in ROOMS}} for i, rooms in enumerate(seq)]}


def test_count_pair():
    seq = [{"Kitchen": ["Mary"]}, {"Garden": ["Mary"]}, {"Kitchen": ["Mary", "John"]}, {"Office": ["Mary"]}]
    row = _row("steps_in_room", "How many steps did Mary spend in the Kitchen?", "2", seq)
    p = dc.build_pair(row, random.Random(0))
    assert p["flip_kind"] == "evid" and p["gold"] == "2" and p["flip_gold"] == "3"
    assert p["flip_t"] in (1, 3) and p["evid"] == {0, 2}
    assert recompute_answer("steps_in_room", row["question"], p["flip_states"]) == "3"
    assert recompute_answer("steps_in_room", row["question"], p["ctrl_states"]) == "2"
    assert sum(a != b for a, b in zip(p["base_states"], p["flip_states"])) == 1


def test_needle_pairs():
    seq = [{"Kitchen": ["Mary", "John"]}, {"Garden": ["Mary"], "Office": ["John"]}, {"Bedroom": ["Mary", "John", "Sandra"]}]
    row = _row("char_at_frame", "In which room was Mary at step 2?", "Garden", seq)
    p = dc.build_pair(row, random.Random(1))
    assert p["flip_kind"] == "needle" and p["flip_t"] == 1 and p["evid"] == {1}
    assert p["flip_gold"] != "Garden" and recompute_answer("char_at_frame", row["question"], p["flip_states"]) == p["flip_gold"]
    assert recompute_answer("char_at_frame", row["question"], p["ctrl_states"]) == "Garden" and p["ctrl_t"] != 1
    row2 = _row("n_char_at_frame", "How many other characters were in the same room as Mary at step 3?", "2", seq)
    p2 = dc.build_pair(row2, random.Random(2))
    assert p2["flip_t"] == 2 and p2["flip_gold"] in ("1", "3")
    assert recompute_answer("n_char_at_frame", row2["question"], p2["ctrl_states"]) == "2"


def test_rank_auc_and_margin():
    assert dc.rank_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert abs(dc.rank_auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) - 0.5) < 1e-9
    m, pred = dc.margin_of(np.array([0.1, 2.0, 0.5]), 1)
    assert abs(m - 1.5) < 1e-9 and pred == 1
    assert dc.gold_index("steps_in_room", "12", [str(d) for d in range(10)]) is None
    assert dc.gold_index("char_at_frame", "Garden", list(ROOMS)) == 2
    assert dc.cond_tag(0, 12, 0) == "base" and dc.cond_tag(2.0, 12, 0) == "tau2L12" and dc.cond_tag(0, 0, 3398) == "logn3398"


def test_row_attention_matches_direct_softmax():
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv

    torch.manual_seed(0)
    T, nH, nKV, hd = 6, 2, 1, 4
    dims = {"n_heads": nH, "n_kv": nKV, "head_dim": hd, "mrope_section": [1, 1, 0]}
    q, k = torch.randn(1, T, nH * hd), torch.randn(1, T, nKV * hd)
    cos, sin = torch.randn(3, 1, T, hd), torch.randn(3, 1, T, hd)

    class _Att:  # stub with the one attribute row_attention reads
        scaling = 0.7

    class _Layer:
        self_attn = _Att()

    qk = dc.QKCapture.__new__(dc.QKCapture)
    qk.q, qk.k, qk.cos, qk.sin = {0: q}, {0: k}, cos, sin
    mask_row = torch.zeros(T); mask_row[3] = -65504.0
    w = dc.row_attention(qk, [_Layer()], 0, T - 1, mask_row, dims)
    qr, kr = apply_multimodal_rotary_pos_emb(q.view(1, T, nH, hd).transpose(1, 2), k.view(1, T, nKV, hd).transpose(1, 2), cos, sin, dims["mrope_section"])
    kr = repeat_kv(kr, nH)[0]
    ref = torch.softmax(torch.einsum("hd,htd->ht", qr[0][:, T - 1], kr) * 0.7 + mask_row, -1)
    assert torch.allclose(w, ref, atol=1e-6) and float(w[:, 3].max()) == 0.0
    bm = dc.block_masses(w, [(0, 2), (2, 5)])
    assert bm.shape == (nH, 2) and torch.allclose(bm[:, 0], w[:, 0:2].sum(-1))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(name)
    print("ALL OK")
