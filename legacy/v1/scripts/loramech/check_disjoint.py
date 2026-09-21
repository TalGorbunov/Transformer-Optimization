#!/usr/bin/env python3
"""Post-hoc contamination assert: exam dirs-file(s) vs a run's train_dirs.txt.

Usage: python scripts/loramech/check_disjoint.py <train_dirs.txt> <exam1.txt> [exam2.txt ...]
Exit 0 = disjoint; exit 1 = overlap (paths printed).
"""
import sys
from pathlib import Path


def norm(p):
    return str(Path(p.strip()).resolve())


def main() -> int:
    train = {norm(ln) for ln in Path(sys.argv[1]).read_text().splitlines() if ln.strip()}
    bad = 0
    for f in sys.argv[2:]:
        exam = {norm(ln) for ln in Path(f).read_text().splitlines() if ln.strip()}
        inter = train & exam
        print(f"{f}: {len(exam)} dirs, overlap with train = {len(inter)}")
        for p in sorted(inter):
            print(f"  CONTAMINATED: {p}")
        bad += len(inter)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
