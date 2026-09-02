#!/usr/bin/env python3
"""LORAMECH exam dirs-file builder (CPU) — P4 contamination discipline by construction.

Draws a class-balanced, validity-gated exam set per cell with a stratified seed-1
shuffle (iter_sample_dirs_shuffled: every prefix is class-balanced — the K0-sorted-trap
fix), writes one dirs-file per cell + a report with the gold histogram. The N=8 and
N=16 files must then be passed to the trainers via --exclude-dirs-file; N=32 comes from
the disjoint park2 generation; nothing trains at N=64/128.

Usage:
  python scripts/loramech/build_exam_dirs.py --out outputs/loramech/examdirs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
)
from gnnformer.metrics import format_gold_histogram  # noqa: E402

CELLS = [
    ("exam_ff_N8", "data/mmred_images_park/seq_len_8/all_uniform", 150),
    ("exam_ff_N16", "data/mmred_longN_park/seq_len_16/all_uniform", 150),
    ("exam_ff_N32", "data/mmred_longN_park2/seq_len_32/all_uniform", 150),
    ("exam_ff_N64", "data/mmred_longN_park/seq_len_64/all_uniform", 150),
    ("exam_ff_N128", "data/mmred_longN_park/seq_len_128/all_uniform", 150),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/loramech/examdirs"))
    ap.add_argument("--seed", type=int, default=1,
                    help="shuffle seed (MUST differ from the trainer's fixed seed 0? "
                         "No — exclusion makes any draw safe; 1 documents the intent)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for name, root, k in CELLS:
        picked, golds, n_bad = [], [], 0
        for sd in iter_sample_dirs_shuffled(Path(root), args.seed):
            if len(picked) >= k:
                break
            try:
                _sid, _frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
            except Exception:
                n_bad += 1
                continue
            if parse_task_labels(q0, states, gold) is None:
                n_bad += 1
                continue
            picked.append(str(sd))
            golds.append(gold)
        (args.out / f"{name}.txt").write_text("\n".join(picked) + "\n")
        line = (f"{name}: {len(picked)} dirs (invalid skipped {n_bad}) from {root}\n"
                f"  gold-hist {format_gold_histogram(golds)}")
        print(line)
        report.append(line)
    (args.out / "REPORT.txt").write_text("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
