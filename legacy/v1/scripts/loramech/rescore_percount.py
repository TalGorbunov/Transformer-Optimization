#!/usr/bin/env python3
"""LORAMECH L6 — per-count anatomy from a run's longn_predictions.csv (CPU, no GPU).

For each eval source: per-count recall table, exact acc, parse-fail rate, MAE,
class distribution, MAJORITY-CLASS baseline (the honest floor), and the mid-range
vs extremes split that decides H4 (extremes-shortcut relapse vs live curve).

Usage:
  python scripts/loramech/rescore_percount.py <run_dir_or_csv> [more ...]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def rescore(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src, rs in by_src.items():
        golds = [int(r["gold"]) for r in rs]
        preds = [int(r["pred"]) if r["pred"] not in ("", None) else None for r in rs]
        n = len(rs)
        hits = sum(int(p == g) for p, g in zip(preds, golds))
        pf = sum(p is None for p in preds)
        mae_pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
        mae = (sum(abs(p - g) for p, g in mae_pairs) / len(mae_pairs)) if mae_pairs else float("nan")
        dist = Counter(golds)
        maj_gold, maj_n = dist.most_common(1)[0]
        print(f"\n=== {src} (n={n}) ===")
        print(f"acc={hits/n:.4f}  parse_fail={pf/n:.3f}  mae={mae:.2f}  "
              f"majority-baseline={maj_n/n:.4f} (always g{maj_gold})")
        print("class dist: " + " ".join(f"g{g}:{c}" for g, c in sorted(dist.items())))
        per = defaultdict(lambda: [0, 0])
        for p, g in zip(preds, golds):
            per[g][1] += 1
            per[g][0] += int(p == g)
        print("per-count recall:")
        for g in sorted(per):
            c, t = per[g]
            print(f"  g{g:>3}: {c:>3}/{t:<3} = {c/t:.3f}")
        # H4 split: extremes = the two lowest + two highest gold classes present
        klasses = sorted(per)
        if len(klasses) >= 5:
            extreme = set(klasses[:2] + klasses[-2:])
            mid = [g for g in klasses if g not in extreme]
            for name, ks in (("extremes", sorted(extreme)), ("mid-range", mid)):
                c = sum(per[g][0] for g in ks)
                t = sum(per[g][1] for g in ks)
                print(f"{name} ({','.join('g%d' % g for g in ks)}): "
                      f"{c}/{t} = {c/max(t,1):.3f}")


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        csv_path = p if p.suffix == ".csv" else p / "longn_predictions.csv"
        if not csv_path.exists():
            print(f"[skip] no longn_predictions.csv at {arg}")
            continue
        print(f"\n########## {csv_path} ##########")
        rescore(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
