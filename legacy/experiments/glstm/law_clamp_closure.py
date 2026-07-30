#!/usr/bin/env python3
"""P0d: LAW + CLAMP composed closure — zero fitted parameters.

The plain closed form 2Φ(d′/2√N)−1 over-predicts the frozen model increasingly with N because
the model's EMITTED range is clamped (an N-invariant answer distribution, RESULTS
[2026-07-10f]) on top of the d′-limited latent estimate. Both ingredients are measured:
  latent:  count estimate ~ gold + ε, σ = √N / d′  (the same noise model the law integrates)
  clamp :  the empirical emitted-value marginal at that N (from the behavioral rows)
Composition = RANK REMAP (monotone, distribution-preserving): the latent estimate is pushed
through F_emitted∘F_latent⁻¹ — i.e., the model reports the emitted-distribution value at its
latent estimate's quantile. This simultaneously preserves ordinal information (measured corr
~0.75) and reproduces the range clamp. Predicted exact-match = P(remapped == gold), Monte Carlo.

Inputs: --cells "NAME:N:DPRIME:ROWS_JSON,..." where ROWS_JSON has per-sample gold/pred.
Outputs: results.csv, report.txt, fig_b3_clamp.png (measured vs plain law vs law+clamp).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ND = NormalDist()


def plain_law(dprime, N, gold):
    d_n = dprime / np.sqrt(N)
    p_int = max(2 * ND.cdf(d_n / 2.0) - 1.0, 0.0)
    p_bnd = ND.cdf(d_n / 2.0)
    return float(np.mean([p_bnd if g in (0, N) else p_int for g in gold]))


def law_clamp(dprime, N, gold, emitted, mc=4000, seed=0):
    """Rank-remap MC: latent ~ g + N(0, (√N/d′)²); emitted value = empirical quantile of the
    measured emitted marginal at the latent's within-simulation quantile."""
    rng = np.random.RandomState(seed)
    sigma = np.sqrt(N) / max(dprime, 1e-6)
    g = np.repeat(np.asarray(gold, dtype=float), mc // len(gold) + 1)[: mc]
    latent = g + rng.randn(len(g)) * sigma
    ranks = latent.argsort().argsort() / (len(latent) - 1)          # [0,1] quantiles
    emitted_sorted = np.sort(np.asarray(emitted, dtype=float))
    remap = emitted_sorted[np.clip((ranks * (len(emitted_sorted) - 1)).round().astype(int),
                                   0, len(emitted_sorted) - 1)]
    return float(np.mean(np.round(remap) == g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True,
                    help="comma list NAME:N:DPRIME:rows.json (pred field = emitted values)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    rows_csv = ["cell,N,dprime,n,measured_exact,plain_law,law_clamp"]
    lines = ["=== LAW + CLAMP composed closure (rank-remap, zero fitted params) ==="]
    pts = []
    for part in args.cells.split(","):
        name, N, dp, rj = part.split(":", 3)
        N, dp = int(N), float(dp)
        rows = json.loads(Path(rj).read_text())
        gold = [r["gold"] for r in rows if r.get("pred") is not None]
        pred = [r["pred"] for r in rows if r.get("pred") is not None]
        measured = float(np.mean([p == g for p, g in zip(pred, gold)]))
        pl = plain_law(dp, N, gold)
        lc = law_clamp(dp, N, gold, pred)
        lines.append(f"  {name:<16s} N={N:<4d} d'={dp:.2f}  measured {measured:.3f}  "
                     f"plain-law {pl:.3f}  law+clamp {lc:.3f}")
        rows_csv.append(f"{name},{N},{dp},{len(gold)},{measured:.4f},{pl:.4f},{lc:.4f}")
        pts.append((name, N, measured, pl, lc))

    (out / "results.csv").write_text("\n".join(rows_csv) + "\n")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mmred = [(N, m, pl, lc) for name, N, m, pl, lc in pts if name.startswith("mmred")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        if mmred:
            mmred.sort()
            Ns = [x[0] for x in mmred]
            axes[0].plot(Ns, [x[1] for x in mmred], "ko-", label="measured (B3)")
            axes[0].plot(Ns, [x[2] for x in mmred], "s--", color="#1f77b4", label="plain law")
            axes[0].plot(Ns, [x[3] for x in mmred], "^-", color="#d62728", label="law + clamp")
            axes[0].set_xscale("log", base=2); axes[0].set_xlabel("N"); axes[0].set_ylabel("exact-match")
            axes[0].set_title("image-MMRED behavior vs N"); axes[0].legend(fontsize=8)
        others = [(f"{name}\nN={N}", m, pl, lc) for name, N, m, pl, lc in pts
                  if not name.startswith("mmred")]
        if others:
            x = np.arange(len(others)); w = 0.27
            axes[1].bar(x - w, [o[1] for o in others], w, color="k", label="measured")
            axes[1].bar(x, [o[2] for o in others], w, color="#1f77b4", label="plain law")
            axes[1].bar(x + w, [o[3] for o in others], w, color="#d62728", label="law + clamp")
            axes[1].set_xticks(x, [o[0] for o in others], fontsize=8)
            axes[1].set_title("transfer: real-video rungs"); axes[1].legend(fontsize=8)
        fig.suptitle("Law + emission-clamp composition (both ingredients measured, 0 fitted params)",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(out / "fig_b3_clamp.png", dpi=130)
        print("wrote fig_b3_clamp.png")
    except Exception as e:
        print("plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
