#!/usr/bin/env python3
"""#3 delta-direction stability across layers: is the count axis the SAME direction at L13..L27, or does
the model rotate it? Uses the multi-layer cache (cache_message_sum_layersweep.py). Per layer reports
||delta||/||mu_all||, per-frame SNR; then the cosine matrix of delta across layers (high off-diagonal =>
stable axis => inject anywhere; low => model rotates the count axis => inject where it's strongest)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="layersweep .pt")
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()
    cache = torch.load(args.cache, map_location="cpu")
    layers = sorted(next(iter(cache.values()))["reps_by_layer"].keys())
    deltas = {}
    print(f"{'layer':>5} {'|mu_all|':>9} {'|delta|':>8} {'sigma(eps)':>10} "
          f"{'|d|/|mu|':>9} {'nE':>5} {'nN':>5} {'per-frame SNR':>14}")
    for L in layers:
        ev, nv = [], []
        for v in cache.values():
            if int(v.get("seq_len", -1)) != args.seq_len or v.get("frame_labels") is None:
                continue
            reps = v["reps_by_layer"][L].float().numpy()
            lab = np.asarray([int(x) for x in v["frame_labels"]])
            ev.extend(reps[lab == 1]); nv.extend(reps[lab == 0])
        ev = np.stack(ev); nv = np.stack(nv)
        mu_all = np.concatenate([ev, nv]).mean(0)
        delta = (ev.mean(0) - nv.mean(0)) / 2.0
        dhat = delta / (np.linalg.norm(delta) + 1e-9)
        sig = 0.5 * ((ev @ dhat).std() + (nv @ dhat).std())
        snr = abs((ev.mean(0) - nv.mean(0)) @ dhat) / (sig + 1e-9)
        deltas[L] = dhat
        print(f"{L:>5} {np.linalg.norm(mu_all):>9.2f} {np.linalg.norm(delta):>8.3f} {sig:>10.3f} "
              f"{np.linalg.norm(delta)/np.linalg.norm(mu_all):>9.4f} {len(ev):>5} {len(nv):>5} {snr:>14.3f}")
    print("\ncosine(delta_L, delta_L') across layers:")
    print("      " + " ".join(f"L{L:<4}" for L in layers))
    for La in layers:
        row = " ".join(f"{float(deltas[La] @ deltas[Lb]):>5.2f}" for Lb in layers)
        print(f"L{La:<4} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
