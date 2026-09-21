#!/usr/bin/env python3
"""Aggregate the MMReD-HF eval grid into one tidy CSV (+ optional headline figure).

Scans:
  arm A  outputs/mmred_hf/frozen/grid_seq<N>_<split>*/<ts>/per_sample.csv
  arm B  outputs/mmred_hf/armB/noLora_seq<N>_steps/<ts>*/report.txt   (fresh-logistic exact)
  arm C  outputs/mmred_hf/exam/seq_len_<N>_<split>_<grp>/<ts>*/report.txt (per-task lines)

Writes outputs/mmred_hf/grid/grid.csv with rows:
  arm,split,seq_len,qtype,n,acc   (qtype = '_all' rows included for convenience)
Newest run dir per cell wins; cells not yet on disk are skipped with a note.

Usage: python scripts/mmred_hf/aggregate_grid.py [--out outputs/mmred_hf/grid]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gnnformer.mmred_hf import DC_QTYPES, NIAH_QTYPES  # noqa: E402

BASE = REPO / "outputs/mmred_hf"
SEQS = [8, 16, 32, 64, 128]


def newest(globbed):
    dirs = sorted(globbed)
    return dirs[-1] if dirs else None


def arm_a_rows():
    rows = []
    for split in ("test", "val"):
        for n in SEQS:
            hits = sorted(BASE.glob(f"frozen/grid_seq{n}_{split}*/**/per_sample.csv"))
            if not hits:
                continue
            per = defaultdict(lambda: [0, 0])
            with open(hits[-1]) as f:
                for r in csv.DictReader(f):
                    per[r["qtype"]][0] += int(r["hit"])
                    per[r["qtype"]][1] += 1
            tot = [sum(h for h, _ in per.values()), sum(c for _, c in per.values())]
            rows += [("A", split, n, qt, c, h / max(c, 1)) for qt, (h, c) in per.items()]
            rows.append(("A", split, n, "_all", tot[1], tot[0] / max(tot[1], 1)))
    return rows


def arm_b_rows():
    rows = []
    for n in SEQS:
        hits = sorted(BASE.glob(f"armB/noLora_seq{n}_steps/*/report.txt")) or \
               sorted(BASE.glob(f"armB/noLora_seq{n}_steps/*/*/report.txt"))
        if not hits:
            continue
        txt = hits[-1].read_text()
        m = re.search(r"fresh logistic \(5 seeds\): err [\d.]+, exact ([\d.]+)", txt)
        if m:
            rows.append(("B", "test", n, "steps_in_room", 50, float(m.group(1))))
    return rows


def arm_c_rows():
    rows = []
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (split,n) -> qtype -> [h,c]
    for split in ("test", "val"):
        for n in SEQS:
            for grp in ("niah", "dc", "steps"):
                pat = f"exam/seq_len_{n}_{split}_{grp}/*/report.txt"
                hits = sorted(BASE.glob(pat)) or sorted(BASE.glob("armB/" + pat.split("/", 1)[1]))
                if grp == "steps":  # steps-only runs live under armB run dirs
                    hits = sorted(BASE.glob(f"armB/seq_len_{n}_{split}_steps/*/report.txt"))
                if not hits:
                    continue
                txt = hits[-1].read_text()
                m = re.search(r"per-task acc: (.+)", txt)
                if not m:
                    continue
                for tok in m.group(1).split():
                    qt, hn = tok.rsplit(":", 1)
                    h, c = hn.split("/")
                    # NIAH/DC full-group runs supersede the small steps-only cell
                    if agg[(split, n)][qt][1] < int(c):
                        agg[(split, n)][qt] = [int(h), int(c)]
    for (split, n), per in agg.items():
        tot = [sum(h for h, _ in per.values()), sum(c for _, c in per.values())]
        rows += [("C", split, n, qt, c, h / max(c, 1)) for qt, (h, c) in per.items()]
        rows.append(("C", split, n, "_all", tot[1], tot[0] / max(tot[1], 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "grid"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = arm_a_rows() + arm_b_rows() + arm_c_rows()
    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "split", "seq_len", "qtype", "n", "acc"])
        w.writerows(rows)

    # coverage summary
    have = {(a, s, n) for a, s, n, qt, *_ in rows if qt == "_all" or a == "B"}
    want = [("A", "test", n) for n in SEQS] + [("A", "val", n) for n in (8, 16)] + \
           [("B", "test", n) for n in SEQS] + \
           [("C", "test", n) for n in SEQS] + [("C", "val", n) for n in (8, 16)]
    missing = [w_ for w_ in want if w_ not in have]
    print(f"wrote {out/'grid.csv'}: {len(rows)} rows")
    print("MISSING cells:", missing if missing else "none — grid complete")
    # quick curves
    for arm in "ABC":
        c = {n: f"{acc:.3f}" for a, s, n, qt, _, acc in rows
             if a == arm and s == "test" and (qt == "_all" or (a == "B"))}
        if c:
            print(f"arm {arm} (test, overall): " +
                  "  ".join(f"N={n}:{c[n]}" for n in SEQS if n in c))


if __name__ == "__main__":
    main()
