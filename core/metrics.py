"""Scoring: exact match per question type (the benchmark metric), the d′ estimator used by
the supply probe, and bootstrap intervals.

Twin: legacy/v1/gnnformer/metrics.py (73 lines: dprime_pair, law_pred, gold_histogram).
tests/test_metrics.py pins dprime_pair on synthetic Gaussians and em_table on a toy set.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def em_table(rows: Sequence[dict], preds: Sequence[Optional[str]]) -> Dict[str, Tuple[float, int]]:
    """{qtype: (exact-match accuracy, n)} plus an 'all' entry; uses core.prompt.exact_match."""
    raise NotImplementedError


def split_by_answer(rows: Sequence[dict], preds: Sequence[Optional[str]], threshold: int = 16
                    ) -> Dict[str, Tuple[float, int]]:
    """For numeric qtypes: accuracy on answers <= threshold vs > threshold (the coverage split
    of plan §7.2; 16 = the largest count in the training lengths)."""
    raise NotImplementedError


def bootstrap_ci(correct: Sequence[bool], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95 % percentile bootstrap interval of the mean."""
    raise NotImplementedError


def dprime_pair(X: np.ndarray, y: np.ndarray, seeds: Sequence[int] = (0, 1, 2),
                max_lda: int = 4000) -> Tuple[float, float, float]:
    """Held-out, sample-disjoint (60/40), shrinkage-LDA whitened d′ between evidence and
    non-evidence per-frame states. X [n, NF, H], y [n, NF] in {0,1}.
    Returns (d′ mean over seeds, d′ std, AUC-derived d′ = sqrt(2)·Φ⁻¹(AUC)).
    Twin: legacy/v1/gnnformer/metrics.py:dprime_pair — THE d′ behind every supply number in
    RESULTS.md; port it verbatim (tests/test_metrics.py::test_dprime_synthetic)."""
    raise NotImplementedError


def law_pred(dprime: float, N: int, gold: Iterable[int]) -> float:
    """Zero-parameter exact-match prediction of a joint (unfenced) read from a single-frame d′:
    2·Φ(d′/(2√N)) − 1, boundary-aware (gold ∈ {0, N} uses Φ(d′/(2√N))). Twin: metrics.py:law_pred."""
    raise NotImplementedError
