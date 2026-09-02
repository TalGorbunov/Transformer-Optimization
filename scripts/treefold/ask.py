#!/usr/bin/env python3
"""TREEFOLD P0 — ASK: question -> note schema + leaf/merge/answer rules.

One call per sample; the model sees ONLY the question (+ fixed conventions) —
never a frame, never the type name, never N. Zero-shot (format + conventions)
is the pre-registered primary; few-shot (+ the 4 armC SEEN exemplars rewritten
as notes) is the control. Band P0: zero-shot parse rate >= 0.95 AND
combinable rate >= 0.95, else few-shot becomes primary for T1-T4 (decision
written in STATE before any tree run).

Outputs (audit-first, like armC's programs.json):
  asks.json   one record per task: raw generation + parsed ask + error
  report.txt  parse/combinable rates overall + per type
  audit.md    per type: one pretty example ask + observed field-name sets
  prompt.txt  the exact prompt template used (provenance)

Smoke:  python scripts/treefold/ask.py --mode zero --ns 16 --limit 3 \\
            --output outputs/_scratch/treefold_smoke/ask_zs
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from tf_common import (SEEN_TYPES, Gen, ask_prompt, build_tasks, extract_json,
                       parse_ns)

REQUIRED_KEYS = ("note", "leaf", "merge", "answer", "combinable")


def validate_ask(obj) -> str:
    """'' if usable, else the failure reason."""
    if not isinstance(obj, dict):
        return "not-a-dict"
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return f"missing:{','.join(missing)}"
    if obj["combinable"] is False:
        return ""                      # valid ask, declared out-of-class
    if not isinstance(obj["note"], dict) or not obj["note"]:
        return "empty-note"
    for k in ("leaf", "merge", "answer"):
        if not isinstance(obj[k], str) or not obj[k].strip():
            return f"empty-{k}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("zero", "few"), required=True)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--ns", default="16,32,64,128")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.per_type, parse_ns(args.ns), args.limit)
    prompts = [ask_prompt(t["q"], args.mode) for t in tasks]
    (out / "prompt.txt").write_text(prompts[0] if prompts else "")
    print(f"ASK mode={args.mode}: {len(tasks)} questions", flush=True)

    gen = Gen(batch=args.batch)
    t0 = time.time()
    raws = gen.text(prompts, max_new=args.max_new, tag=f"ask-{args.mode}",
                    log_every=5)

    records = []
    for t, raw in zip(tasks, raws):
        obj, err = extract_json(raw)
        why = err or validate_ask(obj)
        records.append({**{k: t[k] for k in ("N", "type", "dir", "q", "gold")},
                        "raw": raw, "ask": obj if not why else None,
                        "err": why or None,
                        "combinable": (bool(obj.get("combinable"))
                                       if obj and not why else None)})
    (out / "asks.json").write_text(json.dumps(records, indent=1))

    # ------------------------------------------------------------------ report
    n = len(records)
    parsed = [r for r in records if r["err"] is None]
    comb = [r for r in parsed if r["combinable"]]
    lines = [f"TREEFOLD ASK — mode={args.mode}  (7B nf4, question-only input)",
             f"n={n}  parse-rate {len(parsed)/max(n,1):.3f}  "
             f"combinable-rate(all) {len(comb)/max(n,1):.3f}  "
             f"wall {time.time()-t0:.0f}s", "",
             f"{'type':<26} {'n':>3} {'parsed':>7} {'comb':>6}  seen?"]
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    for t, rs in sorted(by_type.items()):
        p = sum(r["err"] is None for r in rs)
        c = sum(bool(r["combinable"]) for r in rs)
        lines.append(f"{t:<26} {len(rs):>3} {p/len(rs):>7.2f} {c/len(rs):>6.2f}"
                     + ("  SEEN" if t in SEEN_TYPES else ""))
    err_counts = defaultdict(int)
    for r in records:
        if r["err"]:
            err_counts[r["err"].split(":")[0]] += 1
    lines += ["", "failure kinds: " + (json.dumps(dict(err_counts))
                                       if err_counts else "none")]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")

    # ------------------------------------------------------------------- audit
    md = [f"# ASK audit — mode={args.mode}", ""]
    for t, rs in sorted(by_type.items()):
        md.append(f"## {t}" + ("  (SEEN)" if t in SEEN_TYPES else ""))
        fieldsets = sorted({tuple(sorted(r["ask"]["note"]))
                            for r in rs if r["ask"] and r["combinable"]})
        md.append(f"- note field-sets observed: "
                  f"{[list(f) for f in fieldsets] or 'NONE'}")
        ex = next((r for r in rs if r["ask"]), None)
        if ex:
            md += [f"- example (gold `{ex['gold']}`): {ex['q']}", "```json",
                   json.dumps(ex["ask"], indent=1), "```"]
        else:
            md.append("- NO parsed ask for this type")
        md.append("")
    (out / "audit.md").write_text("\n".join(md))
    print(f"wrote {out}/asks.json, report.txt, audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
