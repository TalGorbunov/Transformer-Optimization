#!/usr/bin/env python3
"""Accuracy-vs-N tables from existing evaluate.py run dirs (eval.csv), per qtype and per true k.

A "cell" is a run-dir pattern with {N} in it, relative to outputs/scaleup/<dataset>/<backbone>/, e.g.
  s1_eval/native_mp151200/uniform_N{N}/plain      the frozen official-sampling curve
  s1_eval/native_mp151200/planted_N{N}/plain      needles guaranteed in view
  s3_oracle/native_mp151200/planted_N{N}/gated    fenced + oracle gate
The newest run dir with an eval.csv is taken per (cell, N). k per row = the number of evidence intervals in
json/test.json (the true needle count), so the by-k table is the same across protocols.

Usage: python experiments/curve_table.py --dataset vnbench --cells s1_eval/native_mp151200/uniform_N{N}/plain ... \
           [--n-list 8 16 32 64 128 256] [--out outputs/scaleup/vnbench/qwen2.5-vl-7b/CURVES.md]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
K_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 6))


def newest_eval(cell_dir: Path) -> Optional[Path]:
    runs = sorted(cell_dir.glob("*/eval.csv"))
    return runs[-1] if runs else None


def load_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def acc(rows: List[Dict[str, str]]) -> Tuple[float, int]:
    n = len(rows)
    return (sum(int(r["correct"]) for r in rows) / n if n else float("nan")), n


def fmt(a: Tuple[float, int]) -> str:
    return "—" if a[1] == 0 else f"{a[0]:.2f} ({a[1]})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="qwen2.5-vl-7b")
    ap.add_argument("--cells", nargs="+", required=True, help="run-dir patterns with {N}, relative to the dataset/backbone root")
    ap.add_argument("--n-list", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--data-root", type=Path, default=None, help="json/test.json for the true k (default data/<dataset>)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    base = _REPO / "outputs" / "scaleup" / args.dataset / args.backbone
    data_root = args.data_root or (_REPO / "data" / args.dataset)
    rows_json = json.loads((data_root / "json" / "test.json").read_text())
    k_of = {r["qid"]: len(r["evidence"]) for r in rows_json}
    ivs_of = {r["qid"]: [tuple(iv) for iv in r["evidence"]] for r in rows_json}
    probe_path = data_root / "json" / "video_probe.json"
    dur_of = {r["qid"]: json.loads(probe_path.read_text())[r["video_id"]]["duration"] for r in rows_json} if probe_path.exists() else {}

    def in_view(qid: str, n: int, every: bool = False) -> Optional[bool]:
        """Exact: does the official grid j * dur / n (j < n) sample a frame inside any (or every) evidence interval?"""
        if qid not in dur_of:
            return None
        times = [j * dur_of[qid] / n for j in range(n)]
        hits = [any(a <= t <= b for t in times) for a, b in ivs_of[qid]]
        return all(hits) if every else any(hits)

    tables: Dict[str, Dict[int, List[Dict[str, str]]]] = defaultdict(dict)
    for cell in args.cells:
        for n in args.n_list:
            p = newest_eval(base / cell.format(N=n))
            if p is not None:
                tables[cell][n] = load_rows(p)
    lines = [f"# {args.dataset} / {args.backbone}: accuracy vs N (newest run per cell; rows in parentheses)", ""]
    for cell, by_n in tables.items():
        ns = sorted(by_n)
        qtypes = sorted({r["qtype"] for rows in by_n.values() for r in rows})
        lines += [f"## `{cell}`", "", "| qtype | " + " | ".join(f"N={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for q in qtypes + ["ALL"]:
            cells = [fmt(acc([r for r in by_n[n] if q == "ALL" or r["qtype"] == q])) for n in ns]
            lines.append(f"| {q} | " + " | ".join(cells) + " |")
        lines += ["", "| k (true needles) | " + " | ".join(f"N={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for lo, hi in K_BUCKETS:
            label = f"k={lo}" if lo == hi else f"k={lo}-{hi}"
            cells = [fmt(acc([r for r in by_n[n] if lo <= k_of.get(r["qid"], -1) <= hi])) for n in ns]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if "uniform" in cell and dur_of:          # exact needle-in-view split for the official grid
            lines += ["", "| needle in view (exact) | " + " | ".join(f"N={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
            for label, every, want in (("none", False, False), ("at least one", False, True), ("every needle", True, True)):
                cells = [fmt(acc([r for r in by_n[n] if in_view(r["qid"], n, every) is want])) for n in ns]
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
