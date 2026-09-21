#!/usr/bin/env python3
"""Skew-aware accuracy reporting for the ninv campaign (shared by every instrument).

MMReD-HF's pair-count prior is ~0.78-0.81 zeros, so RAW accuracy is flattering:
predicting 0 everywhere already scores ~0.80. Every gate and calibration number in
this campaign must therefore be reported as four numbers, not one (DIRECTIVE
2026-08-09 item 4):

  raw       plain cell accuracy — comparable to the historical park numbers
  majority  what "always predict the majority class" scores on the SAME cells
  c2        recall on class 2 (both leaves carry evidence) — the class that actually
            exercises fan-2 aggregation, and the one that degrades first
  balanced  mean per-class recall — the honest headline under a skewed prior

Defined once here so no instrument can quietly omit them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def class_report(pred: np.ndarray, true: np.ndarray, n_classes: int = 3) -> dict:
    """-> dict(raw, majority, balanced, recall[list], support[list], n)."""
    pred, true = np.asarray(pred).reshape(-1), np.asarray(true).reshape(-1)
    support = [int((true == c).sum()) for c in range(n_classes)]
    recall = [float((pred[true == c] == c).mean()) if support[c] else float("nan")
              for c in range(n_classes)]
    present = [r for r, s in zip(recall, support) if s]
    return {"raw": float((pred == true).mean()),
            "majority": float(max(support) / max(len(true), 1)),
            "balanced": float(np.mean(present)) if present else float("nan"),
            "recall": recall, "support": support, "n": int(len(true))}


def format_report(rep: dict, prefix: str = "") -> str:
    """One-line rendering: raw / majority / c2 / balanced + per-class recall."""
    rec = " ".join(f"c{c} {r:.3f}" if r == r else f"c{c} n/a"
                   for c, r in enumerate(rep["recall"]))
    c2 = rep["recall"][2] if len(rep["recall"]) > 2 else float("nan")
    return (f"{prefix}raw {rep['raw']:.3f}  majority {rep['majority']:.3f}  "
            f"c2 {c2:.3f}  balanced {rep['balanced']:.3f}   "
            f"[{rec}; support {rep['support']}, n={rep['n']}]")


def cluster_bootstrap_ci(ok_2d: np.ndarray, n_boot: int = 5000, seed: int = 0,
                         pct: Sequence[float] = (2.5, 97.5)):
    """95% CI resampling SAMPLES (rows), not cells.

    Node predictions inside one sample are correlated — a naive binomial CI over
    cells is far too tight and would make a 0.94 look distinguishable from 0.95
    when it is not (the Phase 0 park lesson).
    """
    ok_2d = np.asarray(ok_2d)
    rng = np.random.default_rng(seed)
    n = len(ok_2d)
    bs = np.array([ok_2d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, pct)
    return float(lo), float(hi), bs
