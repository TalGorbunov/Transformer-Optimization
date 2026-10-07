#!/usr/bin/env python3
"""UNIT baseline — collect the T / E / D run dirs of one dataset x backbone into the per-task table of
docs/UNIT_BASELINE_2026-10-06.md and its derived gaps (CPU; no model).

Columns per task: T (text-only) · E(frame lo) · E(frame hi) · E per clip unit @hi · P* = max_U E (with the
unit that attains it) · O(N) oracle gate (existing fenced + oracle-gate row, same rows) · F(N) official uniform
row (existing plain row) · wall = P* - T with a PAIRED bootstrap CI on the common rows · resolution gain =
E(frame hi) - E(frame lo) · temporal gain = P* - E(frame hi) · fence cost = E(frame hi) - O(N) · retrieval
room = P* - F(N) · D(U*) = the per-unit judge at the unit of P* (AUC vs hard / easy, recall @ spec 99, yes-rate
on evidence units). A second table gives E by k (k = 1 rows = pure perception) and the unit fallbacks.

Every cell is read from the NEWEST run dir of its cell directory; missing cells stay blank. Rows are
restricted to the T run's qids (or --qids-file), so every column is on the same rows.
Usage:
  python experiments/unit_table.py --dataset herbench --backbone qwen2.5-vl-7b
  python experiments/unit_table.py --dataset mmred --backbone qwen2.5-vl-7b --config seq_len_8 --oracle-glob '' --official-glob ''
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.data.videoqa import UNIT_OFFSETS  # noqa: E402

UNITS = ("frame", "clip3_d1", "clip5_d1", "clip5_d2", "clip9_d2")
assert set(UNITS) == set(UNIT_OFFSETS)


def _newest(pattern: str, config: Optional[str] = None) -> Optional[Path]:
    """The newest run dir matching the glob that holds results (run_logged also drops runner-*.log
    files into the cell dir; those must not win the sort). With `config` (MMReD seq_len_<N>), only
    runs whose config.json names that config / split count: the MMReD E and D cells of several
    configs share one cell directory."""
    runs = []
    for p in sorted(glob.glob(pattern)):
        d = Path(p)
        if not ((d / "eval.csv").exists() or (d / "metrics.csv").exists()):
            continue
        if config:
            try:
                cfg = json.loads((d / "config.json").read_text())
            except Exception:  # noqa: BLE001
                continue
            if cfg.get("config") != config and not str(cfg.get("split_name", "")).startswith(config + "_"):
                continue
        runs.append(d)
    return runs[-1] if runs else None


def read_eval(run_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """eval.csv -> {qid: row} (correct as int, k as int when present)."""
    if run_dir is None or not (run_dir / "eval.csv").exists():
        return {}
    out = {}
    for r in csv.DictReader(open(run_dir / "eval.csv")):
        r["correct"] = int(r["correct"])
        r["k"] = int(r["k"]) if r.get("k") not in (None, "") else None
        out[r["qid"]] = r
    return out


def read_metrics(run_dir: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """metrics.csv of unit_judge -> {(task, neg): row}."""
    if run_dir is None or not (run_dir / "metrics.csv").exists():
        return {}
    return {(r["task"], r["neg"]): r for r in csv.DictReader(open(run_dir / "metrics.csv"))}


def acc(rows: Dict[str, Dict[str, Any]], qids: Sequence[str]) -> Tuple[Optional[float], int]:
    sel = [rows[q]["correct"] for q in qids if q in rows]
    return (sum(sel) / len(sel) if sel else None), len(sel)


def paired_ci(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]], qids: Sequence[str], n_boot: int = 2000,
              seed: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """mean(a - b) and its 95 % paired bootstrap interval on the rows both have."""
    common = [q for q in qids if q in a and q in b]
    if not common:
        return None, None, None, 0
    d = np.array([a[q]["correct"] - b[q]["correct"] for q in common], dtype=np.float64)
    rng = np.random.RandomState(seed)
    means = d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(common)


def f(x: Optional[float], nd: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="herbench")
    ap.add_argument("--backbone", default="qwen2.5-vl-7b")
    ap.add_argument("--base", type=Path, default=None, help="default outputs/scaleup/<dataset>/<backbone>/s1u_unit")
    ap.add_argument("--t-glob", default=None, help="T run dirs (default <base>/T/*/*)")
    ap.add_argument("--config", default=None, help="MMReD: restrict T/E/D to runs of this config (seq_len_8)")
    ap.add_argument("--oracle-glob", default=None, help="existing fenced + oracle-gate eval run dirs, {N} for the frame count")
    ap.add_argument("--official-glob", default=None, help="existing official (plain, uniform) eval run dirs, {N}")
    ap.add_argument("--n-list", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--qids-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default <base>/TABLE.md (+ .csv)")
    args = ap.parse_args()
    base = args.base or Path("outputs/scaleup") / args.dataset / args.backbone / "s1u_unit"
    if args.dataset == "herbench":
        root = Path("outputs/scaleup") / args.dataset / args.backbone
        oracle_glob = args.oracle_glob if args.oracle_glob is not None else str(root / "s3_oracle/native_mp1003520/uniform_N{N}/gated/*")
        official_glob = args.official_glob if args.official_glob is not None else str(root / "s1_eval/native_mp1003520/uniform_N{N}/plain/*")
    else:
        oracle_glob, official_glob = args.oracle_glob or "", args.official_glob or ""
    tag = f"*{args.config}*" if args.config else "*"

    # ---- runs
    t_run = _newest(args.t_glob or str(base / "T" / tag / "*"), args.config)
    T = read_eval(t_run)
    E: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    E_run: Dict[Tuple[str, str], Optional[Path]] = {}
    for u in UNITS:
        for res in ("lo", "hi", "default"):
            run = _newest(str(base / f"E_{u}_{res}" / "*"), args.config)
            if run is not None:
                E[(u, res)] = read_eval(run)
                E_run[(u, res)] = run
    D: Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for u in UNITS:
        for res in ("lo", "hi", "default"):
            run = _newest(str(base / f"D_{u}_{res}" / "*"), args.config)
            if run is not None:
                D[(u, res)] = read_metrics(run)
    O = {N: read_eval(_newest(oracle_glob.format(N=N))) for N in args.n_list} if oracle_glob else {}
    F = {N: read_eval(_newest(official_glob.format(N=N))) for N in args.n_list} if official_glob else {}

    # ---- rows: the T run's qids (or the file), per task
    if args.qids_file is not None:
        qids = [ln.strip() for ln in args.qids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    elif T:
        qids = list(T)
    else:
        qids = sorted({q for e in E.values() for q in e})
    task_of: Dict[str, str] = {}
    for src in [T] + list(E.values()) + list(O.values()) + list(F.values()):
        for q, r in src.items():
            task_of.setdefault(q, r["qtype"])
    by_task: Dict[str, List[str]] = defaultdict(list)
    for q in qids:
        if q in task_of:
            by_task[task_of[q]].append(q)
    tasks = sorted(by_task)
    e_rows = [q for q in qids if any(q in e for e in E.values())]          # rows with evidence (RLPC etc. drop out)

    def res_of(u: str) -> str:
        for res in ("hi", "default", "lo"):
            if (u, res) in E and (u != "frame" or res != "lo"):
                return res
        return "hi"

    header = ["task", "n", "T", "E frame lo", "E frame hi"] + [f"E {u}" for u in UNITS[1:]] + \
             ["P*", "U*", "wall P*-T", "wall CI", "res gain", "temporal gain"] + \
             [f"O(N{N})" for N in args.n_list] + [f"F(N{N})" for N in args.n_list] + \
             [f"fence cost N{N}" for N in args.n_list] + [f"retrieval room N{N}" for N in args.n_list] + \
             ["D(U*) AUC hard", "D(U*) AUC easy", "D(U*) rec@spec99 (all)", "D(U*) yes-rate pos", "D(frame hi) AUC hard"]
    table: List[List[str]] = []
    for task in tasks + ["ALL-evidence"]:
        rows = [q for q in e_rows] if task == "ALL-evidence" else by_task[task]
        rows_e = [q for q in rows if q in e_rows] or rows
        t_acc, n_t = acc(T, rows)
        e_acc: Dict[Tuple[str, str], Optional[float]] = {key: acc(e, rows_e)[0] for key, e in E.items()}
        cand = [(v, key) for key, v in e_acc.items() if v is not None and key[1] != "lo"]
        if cand:
            p_star, (u_star, r_star) = max(cand)
            # the SMALLEST unit within 1 point of the best (the plan's rule, approximated: ties go to the smaller unit)
            for u in UNITS:
                key = (u, res_of(u))
                if key in e_acc and e_acc[key] is not None and e_acc[key] >= p_star - 0.01:
                    u_star, r_star, p_star = u, key[1], e_acc[key]
                    break
        else:
            p_star, u_star, r_star = None, "", ""
        fr_lo, fr_hi = e_acc.get(("frame", "lo")), e_acc.get(("frame", res_of("frame")))
        wall = ci_lo = ci_hi = None
        if p_star is not None and T:
            wall, ci_lo, ci_hi, _ = paired_ci(E[(u_star, r_star)], T, rows_e)
        line = [task, str(len(rows)), f(t_acc), f(fr_lo), f(fr_hi)] + [f(e_acc.get((u, res_of(u)))) for u in UNITS[1:]] + \
               [f(p_star), f"{u_star}@{r_star}" if u_star else "", f(wall), f"[{f(ci_lo)}, {f(ci_hi)}]" if wall is not None else "",
                f(fr_hi - fr_lo) if fr_hi is not None and fr_lo is not None else "",
                f(p_star - fr_hi) if p_star is not None and fr_hi is not None else ""]
        o_acc = {N: acc(O[N], rows)[0] for N in O}
        f_acc = {N: acc(F[N], rows)[0] for N in F}
        line += [f(o_acc.get(N)) for N in args.n_list] + [f(f_acc.get(N)) for N in args.n_list]
        line += [f(fr_hi - o_acc[N]) if fr_hi is not None and o_acc.get(N) is not None else "" for N in args.n_list]
        line += [f(p_star - f_acc[N]) if p_star is not None and f_acc.get(N) is not None else "" for N in args.n_list]
        dk = (u_star, r_star) if u_star else None
        d_task = "all" if task == "ALL-evidence" else task
        dm = D.get(dk, {}) if dk else {}
        d_fr = D.get(("frame", "hi"), {})
        line += [f(float(dm[(d_task, "hard")]["auc"])) if (d_task, "hard") in dm else "",
                 f(float(dm[(d_task, "easy")]["auc"])) if (d_task, "easy") in dm else "",
                 f(float(dm[(d_task, "all")]["recall@spec99"])) if (d_task, "all") in dm else "",
                 f(float(dm[(d_task, "all")]["yes_rate_pos"]), 2) if (d_task, "all") in dm else "",
                 f(float(d_fr[(d_task, "hard")]["auc"])) if (d_task, "hard") in d_fr else ""]
        table.append(line)

    # ---- by-k table (perception vs joint read) and fallbacks
    k_header = ["task", "unit@res", "n rows", "k=1 acc (n)", "k=2-4 acc (n)", "k>=5 acc (n)", "fallback rows"]
    k_table: List[List[str]] = []
    for (u, res), e in sorted(E.items()):
        for task in tasks:
            rows = [q for q in by_task[task] if q in e]
            if not rows:
                continue
            strata = []
            for lo, hi in ((1, 1), (2, 4), (5, 10 ** 9)):
                sel = [e[q]["correct"] for q in rows if e[q]["k"] is not None and lo <= e[q]["k"] <= hi]
                strata.append(f"{sum(sel) / len(sel):.3f} ({len(sel)})" if sel else "")
            fb = sum(1 for q in rows if e[q].get("unit_eff") not in ("", None, u))
            k_table.append([task, f"{u}@{res}", str(len(rows))] + strata + [str(fb)])

    out = args.out or base / "TABLE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# UNIT baseline table — {args.dataset} x {args.backbone} (generated by experiments/unit_table.py)", "",
          f"T run: `{t_run}`; E runs: " + ", ".join(f"{u}@{r}: `{p.name}`" for (u, r), p in sorted(E_run.items())) +
          (f"; oracle glob `{oracle_glob}`; official glob `{official_glob}`" if oracle_glob else ""), "",
          "Rows per task = the T run's qids (RLPC and other rows without evidence units drop out of the E / D columns). "
          "wall = P* - T on the common rows with a paired bootstrap 95 % CI; P* = the smallest unit within 1 point of the best E@hi.", "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in table]
    md += ["", "## E by number of evidence units k (k = 1 rows = pure perception)", "",
           "| " + " | ".join(k_header) + " |", "|" + "---|" * len(k_header)]
    md += ["| " + " | ".join(r) + " |" for r in k_table]
    out.write_text("\n".join(md) + "\n")
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(table)
    print("\n".join(md))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
