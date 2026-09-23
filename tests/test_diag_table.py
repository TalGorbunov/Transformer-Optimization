"""CPU test for experiments/diag_table.py: run-dir discovery, arm/cond parsing, the gold<=16/>16 split, Wilson CI."""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import diag_table as dt  # noqa: E402


def _run(root, arm_dir, N, stamp, rows):
    d = os.path.join(root, arm_dir, f"N{N}", stamp)
    os.makedirs(d)
    with open(os.path.join(d, "eval.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "qtype", "seq_len", "atype", "gold", "raw", "pred", "correct", "parse_ok", "n_evid"])
        w.writeheader()
        for i, (q, gold, ok) in enumerate(rows):
            w.writerow(dict(qid=f"{i:07d}", qtype=q, seq_len=N, atype="number", gold=gold, raw="", pred="", correct=ok, parse_ok=1, n_evid=""))
    open(os.path.join(d, "summary.csv"), "w").write("group,n,correct,acc,ci_lo,ci_hi\n")
    return d


def test_collect_and_split():
    with tempfile.TemporaryDirectory() as root:
        rows32 = [("steps_in_room", 3, 1), ("steps_in_room", 12, 0), ("steps_in_room", 20, 0), ("steps_in_room", 17, 1),
                  ("char_at_frame", "kitchen", 1), ("char_at_frame", "hall", 1)]
        _run(root, "C_fenced", 32, "20260923_100000_x", rows32)
        _run(root, "C_fenced_tau2", 32, "20260923_100001_x", rows32[:2])
        _run(root, "C_fenced", 32, "20260923_090000_old", [("steps_in_room", 3, 0)])   # older stamp: superseded
        d_unfinished = os.path.join(root, "D_gated_oracle", "N64", "20260923_120000_x")
        os.makedirs(d_unfinished); open(os.path.join(d_unfinished, "eval.csv"), "w").write("qid\n")  # no summary.csv -> skipped
        rows = dt.collect(root)
        key = {(r["arm"], r["cond"], r["qtype"], r["split"], r["N"]): r for r in rows}
        assert set(r["arm"] for r in rows) == {"C_fenced"}, rows
        assert set(r["cond"] for r in rows) == {"base", "tau2"}
        r = key[("C_fenced", "base", "steps_in_room", "all", 32)]
        assert (r["n"], r["correct"], r["acc"]) == (4, 2, 0.5) and r["run"].endswith("20260923_100000_x")
        assert key[("C_fenced", "base", "steps_in_room", "gold<=16", 32)]["n"] == 2
        assert key[("C_fenced", "base", "steps_in_room", "gold>16", 32)]["correct"] == 1
        assert ("C_fenced", "base", "char_at_frame", "gold<=16", 32) not in key   # split only for count types
        assert key[("C_fenced", "tau2", "steps_in_room", "all", 32)]["n"] == 2
        md = dt.markdown(rows, "t")
        assert "## steps_in_room — gold>16" in md and "C  fenced-SFT read + tau2" in md and "0.50 (4)" in md


def test_wilson():
    lo, hi = dt.wilson(0, 50)
    assert lo == 0.0 and 0.05 < hi < 0.09
    lo, hi = dt.wilson(50, 50)
    assert hi == 1.0 and 0.91 < lo < 0.95
    assert all(v != v for v in dt.wilson(0, 0))  # nan, nan


if __name__ == "__main__":
    test_collect_and_split()
    test_wilson()
    print("ok")
