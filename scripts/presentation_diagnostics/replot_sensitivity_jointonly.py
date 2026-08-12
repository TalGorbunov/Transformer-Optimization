#!/usr/bin/env python3
"""Replot a probe_sensitivity run's JOINT panel only (no fenced panel).

Reads sensitivity.npz from an existing run dir and writes
sensitivity_jointonly.{png,pdf} next to it. CPU-only, seconds.

Usage:
  python scripts/presentation_diagnostics/replot_sensitivity_jointonly.py \
      --run outputs/presentation/sensitivity/20260731_211330 [--layer 16]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--title", default="joint attention (no fence)")
    args = ap.parse_args()

    run = Path(args.run)
    Sj = np.load(run / "sensitivity.npz")["S_joint"]          # (n, NF, NF)
    sh = Sj / np.maximum(Sj.sum(-1, keepdims=True), 1e-12)
    M = sh.mean(0)
    NF = M.shape[0]

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("source frame f′")
    ax.set_ylabel("read-out target for frame f")
    ax.set_title(args.title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label="sensitivity share")
    fig.suptitle("Jacobian sensitivity ‖∂(read-out signal for f)/∂(frame f′ embeddings)‖, "
                 f"L≤{args.layer}, N={NF}", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(run / f"sensitivity_jointonly.{ext}", dpi=300)
    print("wrote", run / "sensitivity_jointonly.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
