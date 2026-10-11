#!/usr/bin/env python3
"""The fitted gate of experiments/evaluate.py (--gate-npz) reproduces the legacy S8/S9 rule
(legacy/v1/scripts/sparse/train_sft_gated.py::infer_gate): block t is kept iff ((x_t - mu) / sd) . w + b > 0
on its slot state at the gate layer, module index = legacy layer - 1, from the batched path's
capture dict. Also: the npz that gate_fit.py --save-gate writes has exactly the keys load_gate reads."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from experiments.evaluate import load_gate  # noqa: E402

GATE_KEYS = ("layer", "mu", "sd", "w", "b")


def test_load_gate_matches_legacy_rule():
    rng = np.random.RandomState(0)
    H, N, L = 16, 7, 20
    mu, sd, w, b = rng.randn(H), rng.rand(H) + 0.5, rng.randn(H), 0.1
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gate.npz"
        np.savez(p, layer=L, mu=mu, sd=sd, w=w, b=b, arm="fenced_qfirst", features="slot")
        keep, layers = load_gate(p)
    assert layers == [L - 1], "legacy layer numbering: hidden_states[L] = output of decoder module L - 1"
    x = rng.randn(N, H).astype(np.float32)
    states = {L - 1: torch.tensor(x, dtype=torch.bfloat16), 3: torch.zeros(N, H)}     # the path captures in the model dtype
    got = keep(states, 0)
    xf = states[L - 1].float().numpy()
    want = (((xf - mu) / sd) @ w + b > 0).tolist()                                          # infer_gate: sc = ((x - mu) / sd) @ w + b; keep sc > 0
    assert got == want and len(got) == N and any(got) and not all(got), got
    assert all(isinstance(g, bool) for g in got)


def test_gate_fit_export_keys():
    src = (_REPO / "experiments" / "gate_fit.py").read_text()
    save = src.split("np.savez(args.save_gate,")[1].split(")")[0]
    for k in GATE_KEYS:
        assert f"{k}=" in save, f"gate_fit --save-gate must write {k!r}"


if __name__ == "__main__":
    test_load_gate_matches_legacy_rule()
    test_gate_fit_export_keys()
    print("ALL OK")
