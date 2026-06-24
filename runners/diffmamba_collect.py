#!/usr/bin/env python
"""Collect accuracy-by-split for the Diff/Mamba experiment matrix.
Scans outputs/diffmamba_job*/ run dirs' metrics.csv and prints a compact table per job."""
import csv, glob, os, sys
from collections import defaultdict

ROOTS = sys.argv[1:] or sorted(glob.glob("outputs/diffmamba_job*"))
SPLITS = ["iid_test", "length_ood_test", "composition_ood_test"]

def run_label(path):
    # run dir name like 20260620_..._j1_count_carrier_diff -> strip timestamp prefix
    b = os.path.basename(os.path.dirname(path))
    parts = b.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else b

for root in ROOTS:
    metric_files = sorted(glob.glob(os.path.join(root, "*", "metrics.csv")))
    if not metric_files:
        print(f"\n### {root}: (no metrics yet)")
        continue
    print(f"\n### {root}")
    print(f"{'run / variant':52s} {'iid':>7} {'len_ood':>8} {'comp_ood':>9} {'mae_iid':>8} {'memdis_iid':>10} {'n':>4}")
    rows = defaultdict(dict)   # label -> split -> (acc,mae,memdis,n)
    for mf in metric_files:
        lbl = run_label(mf)
        with open(mf) as fh:
            for r in csv.DictReader(fh):
                rows[lbl][r["split"]] = r
    for lbl in sorted(rows):
        d = rows[lbl]
        def g(split, key):
            try: return float(d[split][key])
            except Exception: return float("nan")
        iid = g("iid_test","accuracy"); lo = g("length_ood_test","accuracy"); co = g("composition_ood_test","accuracy")
        mae = g("iid_test","mae"); md = g("iid_test","memory_disabled_accuracy"); n = g("iid_test","n")
        print(f"{lbl:52s} {iid:7.3f} {lo:8.3f} {co:9.3f} {mae:8.3f} {md:10.3f} {int(n) if n==n else 0:4d}")
