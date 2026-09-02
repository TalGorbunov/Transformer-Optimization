#!/usr/bin/env python3
"""ARMOR Exp C step 2 — compile + exact execution on MLVU-AC records (venv_arch).

Consumes step 1's records.json (per-frame {"time","desc","vis"} from the frozen
VLM), compiles one program per question with Qwen2.5-14B (the armC v3 route:
question -> `def solve(frames)`; the LM never sees the frames), executes in the
armC sandbox (run_program imported from scripts/recagg/ask_compile_execute —
read-only), maps the integer answer to the MCQ options by the [2026-07-24] P2b
prereg rule (nearest option; parse/exec-fail = wrong).

HONESTY NOTES (stated up front, reported in the run dir):
 - all 206 questions are ONE type ("how many instances of action X"), so the
   compile stage is near-degenerate here; what this cell measures is the
   COMPOSED system on a non-synthetic benchmark, not compile generality (armC
   already measured that across 24 types).
 - the frozen 0.282 baseline saw the MCQ options in-prompt; this route never
   sees the options (same asymmetry as P2b, prereg'd there).
 - the benchmark is evidence-delivery-limited at 32f ([2026-07-11c]: ~0.37
   visible frames per gold instance) — the sampling gap bounds any 32f arm.

Band (CAMPAIGN_BRIEF): composed >= 0.282 + 0.05 = 0.332 -> GO.

Usage:
  python scripts/armor/mlvu_compile_execute.py \
      --records outputs/armor/mlvu/<ts>_captions/records.json \
      --output outputs/armor/mlvu/<ts>_compiled
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/ninv"))
sys.path.insert(0, str(_REPO / "scripts/recagg"))

from ask_compile_execute import run_program  # noqa: E402  (armC sandbox, read-only)

SCHEMA = '''You translate a question about a video into python. The video was
uniformly subsampled: `frames` is a list of dicts, in time order, one per
SAMPLED frame (a sparse sample of the whole video). Each frame is
{"time": seconds_float, "desc": short_description, "vis": bool} where "vis"
says whether the queried action is visible in that frame.
Write ONLY a function `def solve(frames):` returning an integer.
Conventions: an "instance" of an action is one contiguous scene; consecutive
sampled frames with vis=True belong to the same instance, separated instances
have a vis=False frame (or a large time gap) between them. Count at least 1 if
the action is ever visible. No imports (Counter and defaultdict are
pre-loaded), no while loops.

Question: In this video, how many instances are there of the 'high jump' action scene in total?
```python
def solve(frames):
    runs = 0
    prev = False
    for f in frames:
        if f["vis"] and not prev:
            runs += 1
        prev = f["vis"]
    return runs
```

Question: In how many of the sampled frames is the 'push up' action visible?
```python
def solve(frames):
    return sum(1 for f in frames if f["vis"])
```

'''


def mcq_pick(pred, candidates):
    """P2b prereg rule: nearest option by |pred - option|; ties -> smaller
    option; non-int pred -> None (scored wrong)."""
    try:
        p = int(round(float(pred)))
    except (TypeError, ValueError):
        return None
    opts = sorted(int(c) for c in candidates)
    return min(opts, key=lambda o: (abs(o - p), o))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.records).read_text())
    qids = sorted(data)
    if args.limit:
        qids = qids[: args.limit]
    print(f"{len(qids)} questions from {args.records}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()
    print(f"loaded {args.model} in {time.time() - t0:.0f}s")

    rows = []
    for i in range(0, len(qids), args.batch):
        chunk = qids[i:i + args.batch]
        prompts = [SCHEMA + f"Question: {data[q]['question']}\n```python\n"
                   for q in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 stop_strings=["```"], tokenizer=tok)
        for j, q in enumerate(chunk):
            cont = tok.decode(gen[j, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            code = cont.split("```")[0].strip()
            pred, err = run_program(code, data[q]["records"])
            gold = int(data[q]["gold"])
            mcq = mcq_pick(pred, data[q]["candidates"]) if err is None else None
            rows.append({"qid": q, "gold": gold, "pred_open": None
                         if pred is None else str(pred), "err": err,
                         "mcq_pick": mcq, "mcq_ok": int(mcq == gold),
                         "n_vis": sum(r["vis"] for r in data[q]["records"]),
                         "code": code})
        if (i // args.batch) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(qids)} {time.time() - t0:.0f}s",
                  flush=True)

    # -------------------------------------------------------------------- report
    n = len(rows)
    mcq_acc = float(np.mean([r["mcq_ok"] for r in rows]))
    exec_fail = float(np.mean([r["err"] is not None for r in rows]))
    open_int = [(int(round(float(r["pred_open"]))), r["gold"]) for r in rows
                if r["err"] is None and r["pred_open"] is not None
                and r["pred_open"].lstrip("-").replace(".", "").isdigit()]
    open_em = float(np.mean([p == g for p, g in open_int])) if open_int else 0.0
    mae = float(np.mean([abs(p - g) for p, g in open_int])) if open_int else -1
    mean_pred = float(np.mean([p for p, _ in open_int])) if open_int else -1
    mean_gold = float(np.mean([r["gold"] for r in rows]))
    by_gold = defaultdict(list)
    for r in rows:
        by_gold[min(r["gold"], 6)].append(r["mcq_ok"])
    lines = [f"ARMOR C — MLVU-AC composed system ({args.model} over frozen-VLM records)",
             f"records: {args.records}",
             f"n={n}  MCQ nearest-option acc = {mcq_acc:.3f}  "
             f"(band: >= 0.332 = frozen 0.282 + 0.05 -> GO; chance 0.25)",
             f"open exact = {open_em:.3f}  MAE = {mae:.2f}  "
             f"mean pred = {mean_pred:.2f} vs mean gold = {mean_gold:.2f}",
             f"exec-fail rate = {exec_fail:.3f}",
             "by gold count (mcq acc): " + "  ".join(
                 f"g{g}:{np.mean(v):.2f}(n={len(v)})"
                 for g, v in sorted(by_gold.items())),
             f"wall {time.time() - t0:.0f}s"]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "programs.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}/report.txt, programs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
