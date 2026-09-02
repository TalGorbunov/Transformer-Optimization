#!/usr/bin/env python3
"""armC Ask-Compile-Execute with THE campaign 7B as the compiler.

The "7B-only + exact executor" demonstration (Tal, 2026-08-26): the same frozen
Qwen2.5-VL-7B that failed as an N-fold tree EXECUTOR (TREEFOLD: per-op error ×
2N ops) is here invoked O(1) times per question — it compiles `def solve(frames)`
once; the sandbox executes it exactly over the N records. Everything except the
generating model is armC v3 verbatim (imported read-only): SCHEMA prompt,
sandbox/whitelist, POOLS + per-type stride, norm/EM, SEEN/UNSEEN split.

Difference to ask_compile_execute.py generation: the VL 7B is chat-templated
(completion-style continuation is a base-LM idiom), so the code is extracted
from the reply's first ``` fence (or the raw reply if unfenced).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[1]), str(_HERE.parent / "recagg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ask_compile_execute import (POOLS, SCHEMA, SEEN_TYPES, norm,  # noqa: E402
                                 run_program, type_of)
from tf_common import Gen, parse_qa  # noqa: E402

FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)(?:```|$)", re.DOTALL)


def extract_code(reply: str) -> str:
    m = FENCE_RE.search(reply)
    code = m.group(1) if m else reply
    return code.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="strided task cap (smoke)")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []   # (N, qtype, prompt, gold, frames)  — armC's exact loop
    for root, N in POOLS:
        by_type = defaultdict(list)
        for d in sorted(Path(root).iterdir()):
            if d.is_dir() and (d / "qa.txt").is_file():
                by_type[type_of(d.name)].append(d)
        for t, dirs in sorted(by_type.items()):
            step = max(1, len(dirs) // args.per_type)
            for d in dirs[::step][: args.per_type]:
                q, states, a = parse_qa(d / "qa.txt")
                prompt = SCHEMA + f"Question: {q}\n```python\n"
                tasks.append((N, t, prompt, str(a).strip(), states))
    if args.limit and args.limit < len(tasks):
        step = max(1, len(tasks) // args.limit)
        tasks = tasks[::step][: args.limit]
    print(f"{len(tasks)} tasks (7B-VL compiler, armC v3 protocol)", flush=True)

    gen = Gen(batch=args.batch)
    t0 = time.time()
    raws = gen.text([p for _, _, p, _, _ in tasks], max_new=args.max_new,
                    tag="compile7b", log_every=5)

    records = []
    for (N, t, _, gold, frames), raw in zip(tasks, raws):
        code = extract_code(raw)
        pred, err = run_program(code, frames)
        em = int(err is None and pred is not None and norm(pred) == norm(gold))
        records.append({"N": N, "type": t, "gold": gold,
                        "pred": None if pred is None else str(pred),
                        "err": err, "em": em, "code": code})

    # ------------------------------------------------- report (armC v3 format)
    ns = sorted({r["N"] for r in records})
    types = sorted({r["type"] for r in records})
    lines = ["ARM C Ask-Compile-Execute — Qwen2.5-VL-7B-Instruct COMPILER "
             "(oracle records; v3 prompt/sandbox verbatim)",
             f"per-type {args.per_type}; seen types: {sorted(SEEN_TYPES)}", ""]
    lines.append(f"{'type':<26} " + " ".join(f"N={n:<4}" for n in ns) + "  seen?")
    for t in types:
        row = []
        for n in ns:
            sel = [r["em"] for r in records if r["type"] == t and r["N"] == n]
            row.append(f"{np.mean(sel):.2f} " if sel else "  -   ")
        lines.append(f"{t:<26} " + " ".join(row)
                     + ("  SEEN" if t in SEEN_TYPES else ""))
    lines.append("")
    for label, fn in (("ALL", lambda t: True),
                      ("SEEN", lambda t: t in SEEN_TYPES),
                      ("UNSEEN", lambda t: t not in SEEN_TYPES)):
        row = []
        for n in ns:
            sel = [r["em"] for r in records if fn(r["type"]) and r["N"] == n]
            row.append(f"{np.mean(sel):.2f} " if sel else "  -   ")
        lines.append(f"{label:<26} " + " ".join(row))
    fail = float(np.mean([r["err"] is not None for r in records]))
    lines += ["", f"exec-failure rate {fail:.3f}   n={len(records)}   "
              f"wall {time.time() - t0:.0f}s"]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "programs.json").write_text(json.dumps(records, indent=1))
    print(f"wrote {out}/report.txt, programs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
