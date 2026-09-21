"""CPU tests for core.metrics. Run: python tests/test_metrics.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.metrics import bootstrap_ci, dprime_pair, em_table, law_pred, split_by_answer


def test_dprime_synthetic():
    rng = np.random.RandomState(0)
    n, NF, H = 300, 4, 16
    y = (rng.rand(n, NF) < 0.5).astype(int)
    X = rng.randn(n, NF, H)
    X[..., 0] += 3.0 * y                       # separation 3σ along one axis
    d, sd, d_auc = dprime_pair(X, y)
    assert 2.4 < d < 3.6 and sd < 0.5 and 2.4 < d_auc < 3.6, (d, sd, d_auc)
    d0, _, _ = dprime_pair(X, rng.permutation(y.reshape(-1)).reshape(n, NF))
    assert d0 < 0.5, d0


def test_law_pred():
    assert law_pred(0.0, 8, [3]) == 0.0
    assert law_pred(0.0, 8, [0]) == 0.5           # boundary class at chance-half
    assert law_pred(4.0, 8, [3]) < law_pred(8.0, 8, [3]) < law_pred(8.0, 2, [3])


ROWS = [{"qtype": "steps_in_room", "answer": "3"}, {"qtype": "steps_in_room", "answer": "20"},
        {"qtype": "first_app", "answer": "Kitchen"}, {"qtype": "first_app", "answer": "Garden"}]
PREDS = ["3", "16", "Kitchen", None]


def test_em_table():
    t = em_table(ROWS, PREDS)
    assert t["steps_in_room"] == (0.5, 2) and t["first_app"] == (0.5, 2) and t["all"] == (0.5, 4)


def test_split_by_answer():
    s = split_by_answer(ROWS, PREDS, threshold=16)
    assert s["<=16"] == (1.0, 1) and s[">16"] == (0.0, 1)     # only numeric qtypes counted


def test_bootstrap_ci():
    lo, hi = bootstrap_ci([True] * 30)
    assert lo == hi == 1.0
    lo, hi = bootstrap_ci([True] * 60 + [False] * 40, seed=1)
    assert lo < 0.6 < hi and hi - lo < 0.25


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
