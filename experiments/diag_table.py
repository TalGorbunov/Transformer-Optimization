"""DIAG Tier 1 — the comparison table (arms x N, per qtype; counts split by gold <= 16 / > 16).

Reads every finished evaluate.py run dir under --root (default outputs/diag/eval/tier1), laid out as
<root>/<ARM>[_tau<t>][_logn<s>]/N<N>/<stamp>*/eval.csv, and writes
  <out>.csv   long form: arm, cond, qtype, split, N, n, correct, acc, ci_lo, ci_hi, run
  <out>.md    one wide table per qtype (rows = arms, columns = N); steps_in_room adds the
              "gold <= 16" / "gold > 16" rows (train seq_len <= 16, so counts > 16 are never seen in training)
Nothing is typed in: every cell traces to a run dir (the .csv carries the path).

    python experiments/diag_table.py                     # tier-1 dirs -> outputs/diag/tier1_table.{csv,md}
    python experiments/diag_table.py --root outputs/diag/eval/d3 --out outputs/_scratch/d3_table   # smoke on Tier-0 dirs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

ARM_ORDER = [
    "frozen_plain", "frozen_qfirst",
    "A_plain_paper", "B_plain_qfirst", "C_fenced", "D_gated_oracle", "E_fenced_logn",
]
ARM_LABEL = {
    "frozen_plain": "frozen, deployed (images then question)",
    "frozen_qfirst": "frozen, question-first",
    "A_plain_paper": "A  plain LoRA, deployed layout",
    "B_plain_qfirst": "B  plain LoRA, question-first",
    "C_fenced": "C  fenced-SFT read",
    "D_gated_oracle": "D  gated read (oracle gate)",
    "E_fenced_logn": "E  fenced-SFT + log-N training prior",
}
COUNT_SPLIT = {"steps_in_room", "crowd_count"}   # gold = a count that can exceed the training window
TRAIN_MAX = 16
COND_RE = re.compile(r"^(?P<arm>.*?)(?P<cond>(?:_tau[0-9.]+)?(?:_logn[0-9]+)?)$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def finished_runs(root: str):
    """Yield (arm, cond, N, run_dir) for every run dir that has eval.csv + summary.csv (finished)."""
    for d in sorted(glob.glob(os.path.join(root, "*", "N*", "2*"))):
        if not (os.path.exists(os.path.join(d, "eval.csv")) and os.path.exists(os.path.join(d, "summary.csv"))):
            continue
        arm_dir = os.path.basename(os.path.dirname(os.path.dirname(d)))
        n_dir = os.path.basename(os.path.dirname(d))
        m = COND_RE.match(arm_dir)
        arm, cond = (m.group("arm"), m.group("cond").lstrip("_") or "base") if m else (arm_dir, "base")
        try:
            N = int(n_dir[1:])
        except ValueError:
            continue
        yield arm, cond, N, d


def gold_int(v: str):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def collect(root: str):
    """Return rows: dict(arm, cond, qtype, split, N, n, correct, acc, ci_lo, ci_hi, run). Latest run per cell wins."""
    latest = {}
    for arm, cond, N, d in finished_runs(root):
        latest[(arm, cond, N)] = d           # sorted glob -> the latest stamp overwrites
    rows = []
    for (arm, cond, N), d in sorted(latest.items()):
        tally = defaultdict(lambda: [0, 0])  # (qtype, split) -> [correct, n]
        with open(os.path.join(d, "eval.csv")) as fh:
            for r in csv.DictReader(fh):
                q = r["qtype"]
                ok = 1 if str(r.get("correct", "0")).strip() in ("1", "True", "true") else 0
                tally[(q, "all")][0] += ok
                tally[(q, "all")][1] += 1
                if q in COUNT_SPLIT:
                    g = gold_int(r.get("gold"))
                    if g is not None:
                        s = f"gold<={TRAIN_MAX}" if g <= TRAIN_MAX else f"gold>{TRAIN_MAX}"
                        tally[(q, s)][0] += ok
                        tally[(q, s)][1] += 1
        for (q, s), (k, n) in sorted(tally.items()):
            lo, hi = wilson(k, n)
            rows.append(dict(arm=arm, cond=cond, qtype=q, split=s, N=N, n=n, correct=k,
                             acc=round(k / n, 4) if n else float("nan"), ci_lo=round(lo, 4), ci_hi=round(hi, 4), run=d))
    return rows


def arm_key(arm: str, cond: str):
    i = ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)
    return (i, arm, cond != "base", cond)


def fmt_cell(r) -> str:
    if r is None:
        return "·"
    return f"{r['acc']:.2f} ({r['n']})"


def markdown(rows, title: str) -> str:
    by = {(r["arm"], r["cond"], r["qtype"], r["split"], r["N"]): r for r in rows}
    Ns = sorted({r["N"] for r in rows})
    arms = sorted({(r["arm"], r["cond"]) for r in rows}, key=lambda ac: arm_key(*ac))
    qtypes = sorted({r["qtype"] for r in rows})
    out = [f"# {title}", "",
           f"Exact match on the official MMReD test split; cell = accuracy (n). Train window: seq_len <= {TRAIN_MAX}.",
           "Arms with a `tau`/`logn` suffix are eval-time interventions on that arm's adapter (no retraining).", ""]
    for q in qtypes:
        splits = ["all"] + ([f"gold<={TRAIN_MAX}", f"gold>{TRAIN_MAX}"] if q in COUNT_SPLIT else [])
        for s in splits:
            head = f"## {q}" + ("" if s == "all" else f" — {s}")
            out += [head, "", "| arm | " + " | ".join(f"N={N}" for N in Ns) + " |",
                    "|---|" + "---:|" * len(Ns)]
            for arm, cond in arms:
                cells = [by.get((arm, cond, q, s, N)) for N in Ns]
                if not any(cells):
                    continue
                label = ARM_LABEL.get(arm, arm) + ("" if cond == "base" else f" + {cond}")
                out.append(f"| {label} | " + " | ".join(fmt_cell(c) for c in cells) + " |")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/diag/eval/tier1")
    ap.add_argument("--out", default="outputs/diag/tier1_table", help="prefix; writes <out>.csv and <out>.md")
    ap.add_argument("--title", default="DIAG Tier 1 — comparison table")
    args = ap.parse_args()
    rows = collect(args.root)
    if not rows:
        print(f"no finished eval runs under {args.root}")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = markdown(rows, args.title)
    with open(args.out + ".md", "w") as fh:
        fh.write(md + "\n")
    cells = {(r["arm"], r["cond"], r["N"]) for r in rows}
    print(f"{len(rows)} rows, {len(cells)} (arm, cond, N) cells -> {args.out}.csv / .md")
    print(md)


if __name__ == "__main__":
    main()
