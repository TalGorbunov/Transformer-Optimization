#!/usr/bin/env python3
"""The official VNBench score from a vnbench4 run (eval.csv with all four option orders per video).
VideoNIAH evaluation/eval.py: a video is correct iff all four tries are answered correctly; per-task accuracy
over videos; family (retrieval / ordering / counting) = mean of its three subtasks; overall = mean of the
families. The single-try (try 0) accuracy is printed next to it: the number the try-0 cells report.
  python experiments/vnbench_circular.py <run dir | eval.csv> [--out X.md]
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

FAMILIES = {"retrieval": ("ret_edit1", "ret_insert1", "ret_insert2"), "ordering": ("ord_edit1", "ord_insert1", "ord_insert2"),
            "counting": ("cnt_edit1", "cnt_insert1", "cnt_edit2")}
TRY = re.compile(r"^(.*)_t([123])$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = args.run if args.run.suffix == ".csv" else sorted(args.run.glob("**/eval.csv"))[-1]
    tries = defaultdict(dict)                     # stem -> {try: correct}
    task_of = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            m = TRY.match(r["qid"])
            stem, t = (m.group(1), int(m.group(2))) if m else (r["qid"], 0)
            tries[stem][t] = bool(int(r["correct"]))
            task_of[stem] = r["qtype"]
    per = {}
    for q in sorted(set(task_of.values())):
        stems = [s for s, t in task_of.items() if t == q]
        circ = [len(tries[s]) == 4 and all(tries[s].values()) for s in stems]
        t0 = [tries[s].get(0, False) for s in stems]
        per[q] = (sum(circ) / len(stems), sum(t0) / len(stems), len(stems), sum(len(tries[s]) < 4 for s in stems))
    lines = [f"# official circular score: `{path}`", "", "| task | circular (all 4 tries) | try 0 only | videos | incomplete |", "|---|---|---|---|---|"]
    lines += [f"| {q} | {c:.3f} | {a:.3f} | {n} | {inc} |" for q, (c, a, n, inc) in per.items()]
    fam = {}
    for name, qs in FAMILIES.items():
        have = [per[q] for q in qs if q in per]
        if have:
            fam[name] = (sum(c for c, *_ in have) / len(have), sum(a for _, a, *_ in have) / len(have))
            lines.append(f"| **{name}** | **{fam[name][0]:.3f}** | {fam[name][1]:.3f} | | |")
    if fam:
        lines.append(f"| **overall** | **{sum(c for c, _ in fam.values()) / len(fam):.3f}** | {sum(a for _, a in fam.values()) / len(fam):.3f} | | |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
