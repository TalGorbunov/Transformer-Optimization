#!/usr/bin/env python3
"""TREEFOLD ANSWER + SCORE — root note -> answer string -> armC-format report.

Scoring is armC's norm/EM verbatim (ints as ints, names/rooms
case-insensitive, comma-lists as sets). Every report carries: per-type x N EM
grid, ALL/SEEN/UNSEEN rows, COUNT-LIKE row, the majority baseline computed on
the SAME cells, per-N gold class distribution (K0-sorted-trap discipline), and
the separate failure counters — every failure kind counts as WRONG.

Importable; standalone main combines records.json files from several run dirs
(e.g. a cell split across jobs by N) into one report.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from tf_common import COUNT_LIKE, SEEN_TYPES, answer_prompt, norm

FAIL_KINDS = ("ask_parse", "not_combinable", "caption_missing", "leaf_parse",
              "merge_parse", "answer_empty")


def clean_answer(raw: str) -> str:
    """Light cleanup only: first line, strip quotes/period. No extraction
    heuristics — the prompt demands the bare answer; anything else is wrong."""
    s = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    return s.strip(" '\"`.").strip()


def run_answers(gen, samples: List[dict], merges: Dict[str, dict],
                max_new: int, out_path: Path,
                prompt_builder=answer_prompt) -> Dict[str, dict]:
    """samples carry 'ask'. Answers only for samples with a root note."""
    todo = [s for s in samples if merges[s["dir"]]["root"] is not None]
    prompts = [prompt_builder(s["ask"], s["q"], s["n_frames"],
                              merges[s["dir"]]["root"]) for s in todo]
    raws = gen.text(prompts, max_new=max_new, tag="answer", log_every=10)
    out = {s["dir"]: {"raw": r, "pred": clean_answer(r)}
           for s, r in zip(todo, raws)}
    out_path.write_text(json.dumps(out))
    return out


def build_records(tasks: List[dict], sample_fail: Dict[str, str],
                  answers: Dict[str, dict]) -> List[dict]:
    """One scored record per task. sample_fail: dir -> fail kind (pre-answer)."""
    records = []
    for t in tasks:
        d = t["dir"]
        rec = {k: t[k] for k in ("N", "type", "dir", "q", "gold")}
        if d in sample_fail:
            rec.update(fail=sample_fail[d], pred_raw=None, pred=None, em=0)
        else:
            a = answers.get(d)
            if a is None or not a["pred"]:
                rec.update(fail="answer_empty",
                           pred_raw=None if a is None else a["raw"],
                           pred=None, em=0)
            else:
                rec.update(fail=None, pred_raw=a["raw"], pred=a["pred"],
                           em=int(norm(a["pred"]) == norm(t["gold"])))
        records.append(rec)
    return records


def report(records: List[dict], title: str) -> str:
    ns = sorted({r["N"] for r in records})
    types = sorted({r["type"] for r in records})
    cell = lambda t, n: [r for r in records if r["type"] == t and r["N"] == n]
    lines = [title, f"n={len(records)}", "",
             f"{'type':<26} " + " ".join(f"N={n:<4}" for n in ns) + "  seen?"]
    for t in types:
        row = []
        for n in ns:
            sel = cell(t, n)
            row.append(f"{np.mean([r['em'] for r in sel]):.2f} " if sel
                       else "  -   ")
        lines.append(f"{t:<26} " + " ".join(row)
                     + ("  SEEN" if t in SEEN_TYPES else ""))
    lines.append("")
    groups = (("ALL", lambda t: True), ("SEEN", lambda t: t in SEEN_TYPES),
              ("UNSEEN", lambda t: t not in SEEN_TYPES),
              ("COUNT-LIKE", lambda t: t in COUNT_LIKE))
    for label, fn in groups:
        row = []
        for n in ns:
            sel = [r["em"] for r in records if fn(r["type"]) and r["N"] == n]
            row.append(f"{np.mean(sel):.2f} " if sel else "  -   ")
        lines.append(f"{label:<26} " + " ".join(row))
    # majority baseline on the same cells (mean over type-cells of the modal
    # gold's frequency — comparable to mean EM at equal per-type counts)
    row = []
    for n in ns:
        cells = []
        for t in types:
            golds = [norm(r["gold"]) for r in cell(t, n)]
            if golds:
                cells.append(Counter(golds).most_common(1)[0][1] / len(golds))
        row.append(f"{np.mean(cells):.2f} " if cells else "  -   ")
    lines.append(f"{'MAJORITY':<26} " + " ".join(row))
    # failure counters (each counted as WRONG above)
    lines += ["", "failure counters (all scored WRONG):"]
    for kind in FAIL_KINDS:
        per_n = [sum(1 for r in records if r["fail"] == kind and r["N"] == n)
                 for n in ns]
        if any(per_n):
            lines.append(f"  {kind:<16} " +
                         " ".join(f"N={n}:{c}" for n, c in zip(ns, per_n)))
    if not any(r["fail"] for r in records):
        lines.append("  none")
    # class distribution per N (top golds)
    lines += ["", "gold class distribution per N (top 5):"]
    for n in ns:
        c = Counter(norm(r["gold"]) for r in records if r["N"] == n)
        top = ", ".join(f"{g}:{k}" for g, k in c.most_common(5))
        lines.append(f"  N={n:<4} n={sum(c.values()):<4} {top}")
    return "\n".join(lines)


def write_cell_csv(records: List[dict], path: Path) -> None:
    rows = ["type,N,n,em,majority"]
    for t in sorted({r["type"] for r in records}):
        for n in sorted({r["N"] for r in records}):
            sel = [r for r in records if r["type"] == t and r["N"] == n]
            if not sel:
                continue
            golds = Counter(norm(r["gold"]) for r in sel)
            rows.append(f"{t},{n},{len(sel)},"
                        f"{np.mean([r['em'] for r in sel]):.4f},"
                        f"{golds.most_common(1)[0][1] / len(sel):.4f}")
    path.write_text("\n".join(rows) + "\n")


def fidelity_report(fid: Dict[str, dict], records: List[dict]) -> str:
    """Pooled p_leaf / per-level p_merge per (type, N) + predicted-vs-measured
    EM (prediction = p_leaf^N * p_merge^(N-1), count-like types only)."""
    em_by_dir = {r["dir"]: r["em"] for r in records}
    by_cell = defaultdict(list)
    for d, f in fid.items():
        by_cell[(f["type"], f["N"])].append((d, f))
    lines = ["fidelity (count-like; gold = v3 program on oracle subtree; "
             "field by max leaf agreement):",
             f"{'type':<26} {'N':>4} {'cov':>5} {'p_leaf':>7} {'p_merge':>8}"
             f" {'EM_pred':>8} {'EM_meas':>8}  per-level p_merge"]
    for (t, n), items in sorted(by_cell.items()):
        covered = [(d, f) for d, f in items if f["field"] and
                   f["p_leaf"] is not None]
        cov = len(covered) / len(items)
        if not covered:
            lines.append(f"{t:<26} {n:>4} {cov:>5.2f}      -        -"
                         "        -        -")
            continue
        p_leaf = float(np.mean([f["p_leaf"] for _, f in covered]))
        hits = defaultdict(lambda: [0, 0])
        for _, f in covered:
            for lv, v in f["levels"].items():
                hits[int(lv)][0] += v["hit"]
                hits[int(lv)][1] += v["n"]
        tot_hit = sum(h for h, _ in hits.values())
        tot_n = sum(m for _, m in hits.values())
        p_merge = tot_hit / tot_n if tot_n else float("nan")
        em_pred = (p_leaf ** n) * (p_merge ** max(n - 1, 0))
        em_meas = float(np.mean([em_by_dir[d] for d, _ in covered]))
        per_lv = " ".join(f"L{lv}:{h/m:.2f}({m})"
                          for lv, (h, m) in sorted(hits.items()) if m)
        lines.append(f"{t:<26} {n:>4} {cov:>5.2f} {p_leaf:>7.3f} "
                     f"{p_merge:>8.3f} {em_pred:>8.3f} {em_meas:>8.3f}  "
                     f"{per_lv}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="run dirs containing records.json, to combine")
    ap.add_argument("--title", default="TREEFOLD combined")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for d in args.run_dirs:
        records += json.load(open(Path(d) / "records.json"))
    rep = report(records, args.title)
    print(rep)
    (out / "report.txt").write_text(rep + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1))
    write_cell_csv(records, out / "cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
