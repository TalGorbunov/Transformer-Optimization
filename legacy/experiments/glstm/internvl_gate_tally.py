#!/usr/bin/env python3
"""P3b (p0p2 campaign, 2026-07-24): scaffold-level gate->tally on the InternVL2.5-8B
per-frame message cache (job 124280, multipass-qfirst bench — each frame SOLO in its own
forward with the question first; i.e. MULTIPASS-ISOLATED supply, not one-forward fenced).

Per layer: logistic gate on per-frame messages (sample-disjoint 60/40 split, 3 seeds),
tally = sum of per-frame binary verdicts per 8-frame sample -> exact-match vs gold.
Band (prereg): exact >= 0.90 => "the GNN scaffold structure ports" at the scaffold level.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CACHE = PROJECT_ROOT / "outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/bench_cache.pt"
NF = 8


def main() -> int:
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    y = np.asarray(c["labels"], int)
    n_fr = len(y); n_s = n_fr // NF
    gold = y.reshape(n_s, NF).sum(1)
    out = PROJECT_ROOT / "outputs/frame_axis/internvl/gate_tally" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"=== InternVL2.5-8B gate->tally (P3b; cache={CACHE.name} job 124280, "
             f"MULTIPASS-ISOLATED qfirst supply; n={n_s} samples x {NF} frames) ===",
             f"gold hist: {np.bincount(gold)} | majority {np.mean(gold == np.bincount(gold).argmax()):.3f}",
             f"perception_ok (cached digit-readout ref): {np.mean(np.asarray(c['perception_ok'])):.3f}"]
    res = {}
    for L in sorted(c["msgs"]):
        X = np.asarray(c["msgs"][L], np.float32)
        accs, gates = [], []
        for seed in (0, 1, 2):
            rng = np.random.RandomState(seed)
            sidx = rng.permutation(n_s)
            tr_s, te_s = sidx[: int(0.6 * n_s)], sidx[int(0.6 * n_s):]
            tr = (tr_s[:, None] * NF + np.arange(NF)).ravel()
            te = (te_s[:, None] * NF + np.arange(NF)).ravel()
            sc = StandardScaler().fit(X[tr])
            gate = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(X[tr]), y[tr])
            pred_fr = gate.predict(sc.transform(X[te])).reshape(len(te_s), NF)
            tally = pred_fr.sum(1)
            accs.append(float(np.mean(tally == gold[te_s])))
            gates.append(float(gate.score(sc.transform(X[te]), y[te])))
        res[L] = (float(np.mean(accs)), float(np.std(accs)), float(np.mean(gates)))
        lines.append(f"L{L}: gate->tally EXACT {res[L][0]:.3f} ± {res[L][1]:.3f}  "
                     f"(per-frame gate acc {res[L][2]:.3f}; seeds 0-2, sample-disjoint 60/40)")
    best = max(res, key=lambda L: res[L][0])
    lines.append(f"BEST L{best}: {res[best][0]:.3f} ± {res[best][1]:.3f}  "
                 f"(band >=0.90 => scaffold ports)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
    print("\n".join(lines)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
