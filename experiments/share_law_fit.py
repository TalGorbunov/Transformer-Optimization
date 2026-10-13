#!/usr/bin/env python3
"""The share law fitted on the attention photograph (CPU, seconds).

At the answer row a head's attention splits over three token groups: the k needle frames, the N - k filler frames and
the rest of the prompt (text, markers, the sink). The share law weighs a needle frame e^s, a filler frame 1 and the
rest C filler-equivalents, so with Z = k e^s + (N - k) + C
    evid = k e^s / Z        nonevid = (N - k) / Z        other = C / Z.
Per layer, (s, C) are fitted by least squares on the head-mean evid and nonevid shares of every row of the plain
photograph cells (probe_attention.py mass.csv: one line per (qid, layer, head); one observation per (row, N)).
A second, C-free form is fitted next to it, the share of the FRAME mass only (frame_mass = evid + nonevid):
    evid / frame_mass = k e^s / (k e^s + N - k),
which holds whatever the rest of the prompt takes; the per-N table lists the measured frame mass so the two forms can
be told apart (a constant C makes the frame mass grow ~linearly in N; a sink that takes a fixed fraction keeps it flat).
The gated cells (haystack hidden, N - k = 0) are not fitted: the three-group law predicts their needle share as
k e^s / (k e^s + C) and the last table shows that prediction next to the measurement, per k.
Inputs: the week-2 photograph cells (177467 plain N=8..128, 177538 gated N=16/128; STATE 2026-10-12 09:30).
  python experiments/share_law_fit.py --out outputs/scaleup/vnbench/qwen2.5-vl-7b/s5_photo/SHARE_LAW.md
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

DEFAULT_ROOT = Path("outputs/scaleup/vnbench/qwen2.5-vl-7b/s5_photo/native_mp151200")
DEFAULT_LAYERS = (12, 16, 20, 24, 27)      # probe_attention's default --layers
START = (0.5, math.log(8.0))              # (s, log C) at the MMReD S10 values (s = 0.30, C = 8), roughly
REQUIRED_COLUMNS = ("qid", "N", "k", "layer", "head", "evid_mass", "nonevid_mass", "other_mass")


@dataclass(frozen=True)
class Observation:
    """One row of one photograph cell at one layer: the head-mean shares at the answer row."""
    qid: str
    N: int
    k: int
    layer: int
    evid: float
    nonevid: float
    other: float


@dataclass(frozen=True)
class Fit:
    s: float                 # three-group law (evid, nonevid, other with the rest at a constant weight C)
    C: float
    rmse_evid: float
    rmse_nonevid: float
    s_frame: float           # frame-share law (evid / (evid + nonevid), C-free)
    rmse_frame: float
    n_obs: int


def newest_mass_csv(cell: Path) -> Path:
    """The newest run dir of a cell that holds a mass.csv (empty dirs of failed attempts are skipped)."""
    runs = sorted(d for d in cell.glob("2*") if (d / "mass.csv").exists())
    if not runs:
        raise SystemExit(f"no run dir with mass.csv under {cell}")
    return runs[-1] / "mass.csv"


def load_observations(path: Path, layers: Sequence[int]) -> List[Observation]:
    """mass.csv -> head-mean shares per (qid, N, layer)."""
    sums: Dict[Tuple[str, int, int], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    k_of: Dict[Tuple[str, int, int], int] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} lacks columns {missing}")
        for row in reader:
            layer = int(row["layer"])
            if layer not in layers:
                continue
            key = (row["qid"], int(row["N"]), layer)
            acc = sums[key]
            acc[0] += float(row["evid_mass"])
            acc[1] += float(row["nonevid_mass"])
            acc[2] += float(row["other_mass"])
            acc[3] += 1
            k_of[key] = int(row["k"])
    return [Observation(qid, N, k_of[(qid, N, layer)], layer, e / n, ne / n, o / n)
            for (qid, N, layer), (e, ne, o, n) in sums.items()]


def predicted_shares(s: float, C: float, k: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(evid, nonevid, other) under the law for arrays of k and N."""
    needle = k * math.exp(s)
    Z = needle + (N - k) + C
    return needle / Z, (N - k) / Z, C / Z


def fit_layer(obs: Sequence[Observation]) -> Fit:
    """Least squares over the evid and nonevid shares jointly; other is their complement."""
    k = np.array([o.k for o in obs], dtype=float)
    N = np.array([o.N for o in obs], dtype=float)
    evid = np.array([o.evid for o in obs])
    nonevid = np.array([o.nonevid for o in obs])

    def residuals(params: np.ndarray) -> np.ndarray:
        pred_evid, pred_nonevid, _ = predicted_shares(params[0], math.exp(params[1]), k, N)
        return np.concatenate([pred_evid - evid, pred_nonevid - nonevid])

    result = least_squares(residuals, x0=np.array(START))
    s, C = float(result.x[0]), math.exp(float(result.x[1]))
    pred_evid, pred_nonevid, _ = predicted_shares(s, C, k, N)
    s_frame, rmse_frame = fit_frame_share(k, N, evid / (evid + nonevid))
    return Fit(s, C, float(np.sqrt(np.mean((pred_evid - evid) ** 2))),
               float(np.sqrt(np.mean((pred_nonevid - nonevid) ** 2))), s_frame, rmse_frame, len(obs))


def frame_share(s: float, k: np.ndarray, N: np.ndarray) -> np.ndarray:
    """The needles' share of the frame mass under the C-free law."""
    needle = k * math.exp(s)
    return needle / (needle + (N - k))


def fit_frame_share(k: np.ndarray, N: np.ndarray, share: np.ndarray) -> Tuple[float, float]:
    """(s, rmse) of the frame-share law by least squares."""
    result = least_squares(lambda params: frame_share(params[0], k, N) - share, x0=np.array([START[0]]))
    s = float(result.x[0])
    return s, float(np.sqrt(np.mean((frame_share(s, k, N) - share) ** 2)))


def by_layer(obs: Sequence[Observation]) -> Dict[int, List[Observation]]:
    groups: Dict[int, List[Observation]] = defaultdict(list)
    for o in obs:
        groups[o.layer].append(o)
    return dict(sorted(groups.items()))


def fit_table(fits: Dict[int, Fit]) -> List[str]:
    lines = ["| layer | three-group s | e^s | C (rest, filler-equivalents) | rmse evid | rmse nonevid | frame-share s | e^s | rmse | obs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for layer, fit in fits.items():
        lines.append(f"| L{layer} | {fit.s:.3f} | {math.exp(fit.s):.2f} | {fit.C:.1f} | {fit.rmse_evid:.4f} | "
                     f"{fit.rmse_nonevid:.4f} | {fit.s_frame:.3f} | {math.exp(fit.s_frame):.2f} | {fit.rmse_frame:.4f} | {fit.n_obs} |")
    return lines


def per_n_table(obs: Sequence[Observation], fit: Fit, n_list: Sequence[int]) -> List[str]:
    """Measured vs predicted mean shares at each N (predictions use every row's own k)."""
    lines = ["| N | rows | evid measured / law | nonevid measured / law | other measured / law | frame mass measured | needle share of frame mass measured / law |",
             "|---|---|---|---|---|---|---|"]
    for N in n_list:
        rows = [o for o in obs if o.N == N]
        if not rows:
            continue
        k = np.array([o.k for o in rows], float)
        n = np.full(len(rows), float(N))
        pred = predicted_shares(fit.s, fit.C, k, n)
        measured = (np.mean([o.evid for o in rows]), np.mean([o.nonevid for o in rows]), np.mean([o.other for o in rows]))
        frame_mass = np.array([o.evid + o.nonevid for o in rows])
        share = np.array([o.evid for o in rows]) / frame_mass
        lines.append(f"| {N} | {len(rows)} | " + " | ".join(f"{m:.4f} / {p.mean():.4f}" for m, p in zip(measured, pred))
                     + f" | {frame_mass.mean():.4f} | {share.mean():.3f} / {frame_share(fit.s_frame, k, n).mean():.3f} |")
    return lines


def gated_table(gated: Sequence[Observation], fit: Fit) -> List[str]:
    """The gated photograph (N - k = 0) per k: measured needle share vs k e^s / (k e^s + C) from the plain fit."""
    lines = ["| k | N | rows | gated evid measured | law k e^s / (k e^s + C) |", "|---|---|---|---|---|"]
    for (k, N) in sorted({(o.k, o.N) for o in gated}):
        rows = [o for o in gated if o.k == k and o.N == N]
        pred = k * math.exp(fit.s) / (k * math.exp(fit.s) + fit.C)
        lines.append(f"| {k} | {N} | {len(rows)} | {np.mean([o.evid for o in rows]):.4f} | {pred:.4f} |")
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo-root", type=Path, default=DEFAULT_ROOT, help="holds planted_N<N>/{plain,gated}/<run>/mass.csv")
    ap.add_argument("--n-list", nargs="+", type=int, default=[8, 16, 32, 64, 128], help="plain cells to fit")
    ap.add_argument("--gated-n-list", nargs="+", type=int, default=[16, 128], help="gated cells to check against the fit")
    ap.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    ap.add_argument("--out", type=Path, default=None, help="markdown report (default: print only)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    plain: List[Observation] = []
    sources: List[str] = []
    for N in args.n_list:
        path = newest_mass_csv(args.photo_root / f"planted_N{N}" / "plain")
        sources.append(str(path.parent))
        plain.extend(load_observations(path, args.layers))
    gated: List[Observation] = []
    for N in args.gated_n_list:
        path = newest_mass_csv(args.photo_root / f"planted_N{N}" / "gated")
        sources.append(str(path.parent))
        gated.extend(load_observations(path, args.layers))
    fits = {layer: fit_layer(obs) for layer, obs in by_layer(plain).items()}
    lines = ["# Share law on the VNBench photograph (head-mean shares at the answer row; plain planted cells fitted)", "",
             "Three-group law: evid = k e^s / Z, nonevid = (N - k) / Z, other = C / Z, Z = k e^s + (N - k) + C (least squares on evid "
             "and nonevid). Frame-share law: evid / (evid + nonevid) = k e^s / (k e^s + N - k), C-free.", "",
             "Sources:"] + [f"- `{s}`" for s in sources] + ["", "## Fits per layer", ""] + fit_table(fits)
    for layer, fit in fits.items():
        lines += ["", f"## L{layer}: measured vs law by N (plain)", ""] + per_n_table(by_layer(plain)[layer], fit, args.n_list)
        gated_layer = [o for o in gated if o.layer == layer]
        if gated_layer:
            lines += ["", f"## L{layer}: gated photograph vs the plain fit", ""] + gated_table(gated_layer, fit)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
