#!/usr/bin/env python3
"""RECAGG Arm C — Ask-Compile-Execute: the task-agnostic aggregation candidate.

The LM NEVER sees the sequence. It sees only the QUESTION (a length-free
language problem) and compiles it to a tiny python function over per-frame
records; exact execution over the N records does the aggregation — length-
invariant by the definition of execution. Records here are ORACLE (the
benchmark's own GT state dicts), so this measures the Compile+Execute layers
in isolation, exactly as Arm A measured frozen in-context aggregation.

Protocol (pre-registered 2026-08-17):
  - eval on the MIXED test pools seq_len_{16,32,64,128}_test — ALL 24 question
    types, per-type strided sampling;
  - few-shot compile prompt contains 4 SEEN types (steps_in_room,
    rooms_visited, where_spend, char_at_frame — the last pins the verified
    1-based step convention); the other 20 types are UNSEEN — reported
    separately (the task-generalization claim);
  - one program generated per SAMPLE (no type oracle anywhere);
  - sandboxed exec: whitelisted builtins, code rejected if it contains
    import/__/exec/eval/open/while; any exception or reject = exec-failure
    (reported separately, never silently wrong);
  - EM: ints compared as ints, names/rooms case-insensitively; comma-lists as
    sets. Every generated program is saved for audit (programs.json).
Prediction on the record: EM is ~FLAT in N wherever the compiled program is
right, so accuracy becomes a per-TYPE constant, not a length curve.

Runs in venv_arch. See slurm/recagg_ask_compile.sbatch.
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

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import parse_qa  # noqa: E402

POOLS = [(f"data/mmred_hf/dirs/seq_len_{n}_test", n) for n in (16, 32, 64, 128)]

SCHEMA = '''You translate questions about a sequence of camera frames into python.
`frames` is a list of dicts, one per step (step k = frames[k-1], 1-based).
Each frame is {"rooms": {room_name: [list of character names in that room]}}.
Rooms: Kitchen, Bathroom, Garden, Office, Bedroom, Hallway.
Write ONLY a function `def solve(frames):` returning the answer
(an int, or a room/character name string). If no one / no room matches,
return "Nobody". No imports (Counter and defaultdict are pre-loaded), no
while loops.
Conventions: a character "appears" in a room at every step they are in it;
their FIRST appearance is the earliest such step and their FINAL appearance
the latest such step (scan from the end for FINAL). "Together" means being in
the same room at the same step; "alone" means being the only person in that
room at that step.

Question: How many steps did John spend in the Bedroom?
```python
def solve(frames):
    return sum(1 for f in frames if "John" in f["rooms"]["Bedroom"])
```

Question: How many different rooms did Sandra visit?
```python
def solve(frames):
    return len({r for f in frames for r, who in f["rooms"].items() if "Sandra" in who})
```

Question: In which room did Michael spend the most time?
```python
def solve(frames):
    counts = {}
    for f in frames:
        for r, who in f["rooms"].items():
            if "Michael" in who:
                counts[r] = counts.get(r, 0) + 1
    return max(counts, key=counts.get)
```

Question: In which room was Daniel at step 11?
```python
def solve(frames):
    f = frames[10]
    for r, who in f["rooms"].items():
        if "Daniel" in who:
            return r
```

Question: In which room was Mary at the step where John made his final appearance in the Garden?
```python
def solve(frames):
    for f in reversed(frames):
        if "John" in f["rooms"]["Garden"]:
            for r, who in f["rooms"].items():
                if "Mary" in who:
                    return r
            return "Nobody"
    return "Nobody"
```

'''
SEEN_TYPES = {"steps_in_room", "rooms_visited", "where_spend", "char_at_frame"}

CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
FORBIDDEN = re.compile(r"__|import|exec|eval|open|while|input|getattr|globals")
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict)
                 else getattr(__builtins__, k)
                 for k in ("len", "sum", "max", "min", "sorted", "set", "list",
                           "dict", "tuple", "range", "enumerate", "abs", "str",
                           "int", "float", "all", "any", "zip", "reversed",
                           "round", "isinstance", "next", "iter", "filter",
                           "map", "bool", "frozenset")}


def type_of(dirname: str) -> str:
    m = re.match(r"([a-z_]+?)_[A-Z0-9]", dirname)
    return m.group(1) if m else dirname


def run_program(code: str, frames):
    if FORBIDDEN.search(code):
        return None, "forbidden"
    from collections import Counter, defaultdict
    env = {"__builtins__": SAFE_BUILTINS, "Counter": Counter,
           "defaultdict": defaultdict}
    try:
        exec(code, env)                                   # noqa: S102 (sandboxed)
        out = env["solve"](frames)
        return out, None
    except Exception as e:                                # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"[:80]


def norm(x) -> str:
    if isinstance(x, (list, tuple, set)):
        return ",".join(sorted(str(v).strip().lower() for v in x))
    s = str(x).strip().lower()
    return ",".join(sorted(p.strip() for p in s.split(","))) if "," in s else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--per-type", type=int, default=8,
                    help="samples per (type, N) — strided across each type")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-prompts", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []   # (N, qtype, prompt, gold, frames)
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
        print(f"N={N}: {sum(min(args.per_type, len(v)) for v in by_type.values())}"
              f" samples over {len(by_type)} types")
    if args.dry_prompts:
        print(tasks[0][2][-500:])
        print(f"[gold {tasks[0][3]}]  {len(tasks)} tasks total")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()
    print(f"loaded {args.model} in {time.time() - t0:.0f}s")

    records = []
    for i in range(0, len(tasks), args.batch):
        chunk = tasks[i:i + args.batch]
        enc = tok([p for _, _, p, _, _ in chunk], return_tensors="pt",
                  padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 stop_strings=["```"], tokenizer=tok)
        for j, (N, t, _, gold, frames) in enumerate(chunk):
            cont = tok.decode(gen[j, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            code = cont.split("```")[0].strip()
            pred, err = run_program(code, frames)
            em = int(err is None and pred is not None
                     and norm(pred) == norm(gold))
            records.append({"N": N, "type": t, "gold": gold,
                            "pred": None if pred is None else str(pred),
                            "err": err, "em": em, "code": code})
        if (i // args.batch) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(tasks)} {time.time() - t0:.0f}s",
                  flush=True)

    # ------------------------------------------------------------------ report
    ns = sorted({r["N"] for r in records})
    types = sorted({r["type"] for r in records})
    lines = [f"ARM C Ask-Compile-Execute — {args.model} (oracle records)",
             f"per-type {args.per_type}; seen types: {sorted(SEEN_TYPES)}", ""]
    hdr = f"{'type':<26} " + " ".join(f"N={n:<4}" for n in ns) + "  seen?"
    lines.append(hdr)
    per = {}
    for t in types:
        row = []
        for n in ns:
            sel = [r["em"] for r in records if r["type"] == t and r["N"] == n]
            row.append(float(np.mean(sel)) if sel else float("nan"))
        per[t] = row
        lines.append(f"{t:<26} " + " ".join(f"{v:.2f} " for v in row)
                     + ("  SEEN" if t in SEEN_TYPES else ""))
    lines.append("")
    for label, pred_fn in (("ALL", lambda t: True),
                           ("SEEN", lambda t: t in SEEN_TYPES),
                           ("UNSEEN", lambda t: t not in SEEN_TYPES)):
        row = []
        for n in ns:
            sel = [r["em"] for r in records
                   if pred_fn(r["type"]) and r["N"] == n]
            row.append(float(np.mean(sel)))
        lines.append(f"{label:<26} " + " ".join(f"{v:.2f} " for v in row))
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
