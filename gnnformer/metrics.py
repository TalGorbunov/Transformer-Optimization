"""Thesis metrics: the held-out whitened d' estimator and the sqrt(N) readout law.

`dprime_pair` is THE d' behind every supply number in RESULTS.md (ported verbatim
from legacy/experiments/glstm/dprime_vs_n.py, where 12 scripts imported it from).
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

_ND = NormalDist()


def dprime_pair(
    X: np.ndarray,
    y: np.ndarray,
    seeds: Sequence[int] = (0, 1, 2),
    max_lda: int = 4000,
) -> Tuple[float, float, float]:
    """Held-out, group-split, shrinkage-LDA whitened d'.

    X [n, NF, H] per-frame message vectors; y [n, NF] binary evidence labels.
    Returns (d'_w mean, d'_w std, d'_auc mean) over sample-disjoint 60/40 splits.
    """
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
    """Zero-parameter exact-match prediction 2*Phi(d'/(2*sqrt(N)))-1, boundary-aware."""
    d_n = dprime / np.sqrt(N)
    p_int = max(2 * _ND.cdf(d_n / 2.0) - 1.0, 0.0)
    p_bnd = _ND.cdf(d_n / 2.0)
    return float(np.mean([p_bnd if g in (0, N) else p_int for g in gold]))


def gold_histogram(golds: Iterable[int]) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for g in golds:
        hist[g] = hist.get(g, 0) + 1
    return hist


def format_gold_histogram(golds: Iterable[int]) -> str:
    return " ".join(f"g{g}:{c}" for g, c in sorted(gold_histogram(golds).items()))
