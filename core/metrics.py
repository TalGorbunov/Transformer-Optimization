"""Scoring: exact match per question type (the benchmark metric), the d′ estimator used by
the supply probe, and bootstrap intervals.

Twin: legacy/v1/gnnformer/metrics.py (73 lines: dprime_pair, law_pred, gold_histogram).
tests/test_metrics.py pins dprime_pair on synthetic Gaussians and em_table on a toy set.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ND = NormalDist()


def _acc(flags: Sequence[bool]) -> Tuple[float, int]:
    n = len(flags)
    return (float(sum(flags)) / n if n else float("nan"), n)


def em_table(rows: Sequence[dict], preds: Sequence[Optional[str]]) -> Dict[str, Tuple[float, int]]:
    """{qtype: (exact-match accuracy, n)} plus an 'all' entry; uses core.prompt.exact_match."""
    from .prompt import exact_match  # local import: prompt never imports metrics

    assert len(rows) == len(preds), (len(rows), len(preds))
    by_q: Dict[str, List[bool]] = {}
    for row, pred in zip(rows, preds):
        by_q.setdefault(row["qtype"], []).append(exact_match(pred, str(row["answer"])))
    table = {q: _acc(flags) for q, flags in by_q.items()}
    table["all"] = _acc([f for flags in by_q.values() for f in flags])
    return table


def split_by_answer(rows: Sequence[dict], preds: Sequence[Optional[str]], threshold: int = 16
                    ) -> Dict[str, Tuple[float, int]]:
    """For numeric qtypes: accuracy on answers <= threshold vs > threshold (the coverage split
    of plan §7.2; 16 = the largest count in the training lengths)."""
    from .mmred import NUMERIC_QTYPES
    from .prompt import exact_match

    lo: List[bool] = []
    hi: List[bool] = []
    for row, pred in zip(rows, preds):
        if row["qtype"] not in NUMERIC_QTYPES:
            continue
        gold = str(row["answer"])
        (lo if int(gold) <= threshold else hi).append(exact_match(pred, gold))
    return {f"<={threshold}": _acc(lo), f">{threshold}": _acc(hi)}


def bootstrap_ci(correct: Sequence[bool], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95 % percentile bootstrap interval of the mean."""
    x = np.asarray(list(correct), dtype=np.float64)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def dprime_pair(X: np.ndarray, y: np.ndarray, seeds: Sequence[int] = (0, 1, 2),
                max_lda: int = 4000) -> Tuple[float, float, float]:
    """Held-out, sample-disjoint (60/40), shrinkage-LDA whitened d′ between evidence and
    non-evidence per-frame states. X [n, NF, H], y [n, NF] in {0,1}.
    Returns (d′ mean over seeds, d′ std, AUC-derived d′ = sqrt(2)·Φ⁻¹(AUC)).
    Twin: legacy/v1/gnnformer/metrics.py:dprime_pair — THE d′ behind every supply number in
    RESULTS.md; ported verbatim (tests/test_metrics.py::test_dprime_synthetic)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    n, NF, H = X.shape
    yf = y.reshape(-1).astype(int)
    samp = np.repeat(np.arange(n), NF)
    dws, das = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        perm = rng.permutation(n)
        tr_s = set(perm[: int(0.6 * n)].tolist())
        trf = np.array([i for i in range(len(yf)) if samp[i] in tr_s])
        tef = np.array([i for i in range(len(yf)) if samp[i] not in tr_s])
        Xf = X.reshape(-1, H).astype(np.float64)
        sub = rng.permutation(len(trf))[:max_lda]
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Xf[trf][sub], yf[trf][sub])
        w = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)
        p = Xf[tef] @ w
        yt = yf[tef]
        pE, pN = p[yt == 1], p[yt == 0]
        dws.append(abs(pE.mean() - pN.mean()) / (0.5 * (pE.std() + pN.std()) + 1e-12))
        try:
            auc = min(max(roc_auc_score(yt, p), 1e-4), 1 - 1e-4)
            das.append(np.sqrt(2) * _ND.inv_cdf(auc))
        except ValueError:
            pass
    return float(np.mean(dws)), float(np.std(dws)), float(np.mean(das)) if das else float("nan")


def law_pred(dprime: float, N: int, gold: Iterable[int]) -> float:
    """Zero-parameter exact-match prediction of a joint (unfenced) read from a single-frame d′:
    2·Φ(d′/(2√N)) − 1, boundary-aware (gold ∈ {0, N} uses Φ(d′/(2√N))). Twin: metrics.py:law_pred."""
    d_n = dprime / np.sqrt(N)
    p_int = max(2 * _ND.cdf(d_n / 2.0) - 1.0, 0.0)
    p_bnd = _ND.cdf(d_n / 2.0)
    return float(np.mean([p_bnd if g in (0, N) else p_int for g in gold]))
