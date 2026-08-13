#!/usr/bin/env python3
"""q_pad leak probe (Exp C follow-up, 2026-07-15): does the pad arm's per-frame query ITSELF
linearly encode the evidence label?

The broadcast-gate qcond arm hit eval d' ~29 — far beyond the physical mp x mp ceiling (~6.3-7.3),
so the gate must be broadcasting label information carried by its q_pad(frame) input feature
rather than repairing routing. This measures held-out d' (dprime_pair) directly on:
  q_pad  — the pad arm's per-frame query (the qcond gate feature)
  q_mp   — the mp arm's per-frame query (same question, multipass context)
  k_mean / v_mean — within-frame mean of the joint pre-rotary k/v (the content features)
If d'(q_pad) is large, the qcond arm is invalid by construction (feature leaks the label:
the pad forward lets the question attend the frame before the query is captured).
CPU only; same capture and eval protocol as the gate run.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from experiments.glstm.dprime_vs_n import dprime_pair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()
    cap = Path(args.capture); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    S = blob["samples"]; NF = int(blob["config"]["n_frames"]); L = int(args.layer)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(S)); n_tr = int(args.train_frac * len(S))
    ev = idx[n_tr:]

    feats = {c: [] for c in ("q_pad", "q_mp", "k_mean", "v_mean")}
    y = np.zeros((len(S), NF), dtype=np.int64)
    for si, rec in enumerate(S):
        A = rec["arms"]
        y[si] = np.asarray(rec["labels"], dtype=np.int64)
        for t in range(NF):
            feats["q_pad"].append(np.asarray(A["pad"][L]["q"][t], np.float32).reshape(-1))
            feats["q_mp"].append(np.asarray(A["mp"][L]["q"][t], np.float32).reshape(-1))
            feats["k_mean"].append(np.asarray(A["joint"][L]["k"][t], np.float32).mean(0))
            feats["v_mean"].append(np.asarray(A["joint"][L]["v"][t], np.float32).mean(0))
        if (si + 1) % 100 == 0:
            print(f"  {si+1}/{len(S)}", flush=True)

    lines = [f"=== Q_PAD LEAK PROBE (L{L}, dprime_pair, eval n={len(ev)} / full n={len(S)}) ==="]
    rows = ["feature,split,dprime_w,std,auc"]
    for c in feats:
        X = np.stack(feats[c]).reshape(len(S), NF, -1)
        for split_name, split_idx in (("eval", ev), ("full", np.arange(len(S)))):
            dw, ds, da = dprime_pair(X[split_idx], y[split_idx])
            lines.append(f"  {c:<8} [{split_name}] d'={dw:.2f}+-{ds:.2f} (auc {da:.2f})")
            rows.append(f"{c},{split_name},{dw:.4f},{ds:.4f},{da:.4f}")
            print(lines[-1], flush=True)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
