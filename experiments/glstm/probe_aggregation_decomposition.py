#!/usr/bin/env python3
"""Quantitative decomposition of WHY S_all fails. Writes the actual numbers behind the math:
m_k = mu_all + s_k * delta + eps_k   (s_k=+1 evidence, -1 non-evidence).

Reports, per dataset (L19 cache):
  |mu_all|, |delta|, ratio |delta|/|mu_all|   (how tiny the discriminative axis is vs the shared mean)
  sigma_within (std of eps along delta-hat)    (the per-frame 'noise floor' on the count axis)
  per-frame SNR = |mu_E - mu_N| / sigma_within
  corr(||S_evid||, g)  vs  corr(||S_all||, g)  -- THE crux: does total magnitude encode g?
  corr(proj(S_all, delta-hat), g)              -- the count IS there along delta (low-SNR), just not in magnitude
  predicted S_all per-count SNR = 2|delta| / (sigma_within * sqrt(N))
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.glstm.probe_message_sum_decodability import load_cache, agg_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()
    for spec in args.caches:
        label, _, path = spec.partition(":")
        exs = load_cache(Path(path), args.seq_len)
        N = exs[0].reps.shape[0]
        ev = np.stack([e.reps[i] for e in exs for i in range(N) if e.labels[i] == 1])
        nv = np.stack([e.reps[i] for e in exs for i in range(N) if e.labels[i] == 0])
        allf = np.concatenate([ev, nv])
        mu_all = allf.mean(0); mu_E = ev.mean(0); mu_N = nv.mean(0)
        delta = (mu_E - mu_N) / 2.0
        dhat = delta / (np.linalg.norm(delta) + 1e-9)
        # within-class noise along delta-hat
        sig = 0.5 * ((ev @ dhat).std() + (nv @ dhat).std())
        snr_pf = abs((mu_E - mu_N) @ dhat) / (sig + 1e-9)
        gold = np.asarray([e.gold for e in exs])
        S_all = np.stack([agg_sum(e.reps) for e in exs])
        S_evid = np.stack([agg_sum(e.reps, np.where(e.labels == 1)[0]) for e in exs])
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1])
        mag_evid = np.linalg.norm(S_evid, axis=1)
        mag_all = np.linalg.norm(S_all, axis=1)
        proj_all = S_all @ dhat
        snr_pred = 2 * np.linalg.norm(delta) / (sig * np.sqrt(N) + 1e-9)
        print(f"\n===== {label} (N={N}, {len(exs)} ex) =====")
        print(f"  |mu_all|={np.linalg.norm(mu_all):.2f}   |delta|={np.linalg.norm(delta):.3f}   "
              f"|delta|/|mu_all|={np.linalg.norm(delta)/np.linalg.norm(mu_all):.4f}")
        print(f"  sigma_within(on delta axis)={sig:.3f}   per-frame SNR=|muE-muN|/sigma={snr_pf:.3f}")
        print(f"  -- the crux: does MAGNITUDE encode the count? --")
        print(f"  corr(||S_evid||, g) = {corr(mag_evid, gold):+.3f}   (count = #vectors summed -> magnitude axis)")
        print(f"  corr(||S_all||,  g) = {corr(mag_all, gold):+.3f}   (N fixed -> magnitude blind to g)")
        print(f"  corr(proj(S_all,delta), g) = {corr(proj_all, gold):+.3f}   (g IS linearly there on the tiny delta axis)")
        print(f"  predicted S_all per-count SNR = 2|delta|/(sigma*sqrt(N)) = {snr_pred:.3f}  "
              f"(<~2 => can't round to exact integer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
