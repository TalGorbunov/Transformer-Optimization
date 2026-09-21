#!/usr/bin/env python3
"""TREEFOLD 14B stage-swap control — the LEAF and ANSWER interfaces at 14B.

Runs in venv_arch (Qwen2.5-14B-Instruct bf16, armC loading pattern). Prompts
are byte-identical to the 7B cells (tf_common builders) — ONLY the generating
model changes, so any EM delta is attributable to model scale at that stage.
The merge stage stays with the 7B (measured sound: conditional fidelity 0.995).

Modes:
  leaves   oracle records -> 14B leaf notes (run_leaves schema, so run_cell
           consumes them via --leaf notes). Partial-saved, resumable.
  answers  roots from an existing merges.json -> 14B answers -> full armC-format
           re-score (build_records/report imported from answer_score).

Smokes:
  ... stages_14b.py --mode leaves  --asks <fs>/asks.json --ns 16 --limit 3 --output ...
  ... stages_14b.py --mode answers --asks <fs>/asks.json --merges <T1>/merges.json \\
        --output ...  (--ns/--per-type MUST match the merge run's task grid)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from answer_score import build_records, clean_answer, report, write_cell_csv
from leaf_text import oracle_states
from tf_common import (answer_prompt, build_tasks, extract_json,
                       leaf_text_prompt, parse_ns, render_state)

MODEL = "Qwen/Qwen2.5-14B-Instruct"
CHUNK = 2000


class Gen14B:
    def __init__(self, batch: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16).to("cuda").eval()
        self.batch = batch
        print(f"[gen14b] loaded {MODEL} in {time.time()-t0:.0f}s", flush=True)

    def text(self, prompts, max_new, tag="", log_every=10):
        import torch
        outs = []
        t0 = time.time()
        for i in range(0, len(prompts), self.batch):
            chunk = [self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True) for p in prompts[i:i + self.batch]]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new,
                                        do_sample=False,
                                        pad_token_id=self.tok.pad_token_id)
            for k in range(len(chunk)):
                outs.append(self.tok.decode(
                    g[k, enc["input_ids"].shape[1]:],
                    skip_special_tokens=True).strip())
            if log_every and (i // self.batch) % log_every == 0:
                print(f"  [{tag}] {len(outs)}/{len(prompts)} "
                      f"{time.time()-t0:.0f}s", flush=True)
        return outs


def join_asks(tasks, asks):
    """-> (samples-with-ask, sample_fail dir->kind), armC/run_cell semantics."""
    sample_fail, samples = {}, []
    for t in tasks:
        a = asks.get(t["dir"])
        if a is None or a["err"] is not None:
            sample_fail[t["dir"]] = "ask_parse"
        elif not a["combinable"]:
            sample_fail[t["dir"]] = "not_combinable"
        else:
            samples.append({**t, "ask": a["ask"]})
    return samples, sample_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("leaves", "answers"), required=True)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--merges", default=None, help="answers mode: merges.json")
    ap.add_argument("--ns", default="16,32,64,128")
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--title", default="TREEFOLD 14B stage swap")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    tasks = build_tasks(args.per_type, parse_ns(args.ns), args.limit)
    asks = {r["dir"]: r for r in json.load(open(args.asks))}
    samples, sample_fail = join_asks(tasks, asks)
    print(f"{args.title} [{args.mode}]: {len(tasks)} tasks, "
          f"{len(samples)} with usable asks", flush=True)

    if args.mode == "leaves":
        flat = []
        for s in samples:
            states = oracle_states(s)
            n = len(states)
            for t in range(n):
                flat.append((s["dir"],
                             leaf_text_prompt(s["ask"], t + 1, n,
                                              render_state(states[t], t + 1, n))))
        partial = out / "leaf_notes.partial.json"
        raws = json.load(open(partial)) if partial.is_file() else []
        if raws:
            print(f"[resume] {len(raws)}/{len(flat)}", flush=True)
        gen = Gen14B(batch=args.batch)
        while len(raws) < len(flat):
            raws += gen.text([p for _, p in flat[len(raws):len(raws) + CHUNK]],
                             max_new=args.max_new, tag="leaf14b")
            partial.write_text(json.dumps(raws))
            print(f"[leaf14b] {len(raws)}/{len(flat)}", flush=True)
        notes = {}
        for (d, _), raw in zip(flat, raws):
            rec = notes.setdefault(d, {"raws": [], "notes": [], "errs": []})
            obj, err = extract_json(raw)
            if obj is not None and not isinstance(obj, dict):
                obj, err = None, "not-a-dict"
            rec["raws"].append(raw)
            rec["notes"].append(obj)
            rec["errs"].append(err)
        (out / "leaf_notes.json").write_text(json.dumps(notes))
        partial.unlink(missing_ok=True)
        ok = sum(e is None for r in notes.values() for e in r["errs"])
        tot = sum(len(r["errs"]) for r in notes.values())
        print(f"DONE leaves: {ok}/{tot} parsed ({ok/max(tot,1):.4f})")
        return 0

    # ---------------------------------------------------------------- answers
    merges = json.load(open(args.merges))
    todo = []
    for s in samples:
        m = merges.get(s["dir"])
        if m is None:
            sample_fail[s["dir"]] = "caption_missing"     # not in merge run
        elif m["failed"]:
            sample_fail[s["dir"]] = m["failed"]["stage"]
        elif m["root"] is not None:
            todo.append(s)
    prompts = [answer_prompt(s["ask"], s["q"], s["n_frames"],
                             merges[s["dir"]]["root"]) for s in todo]
    gen = Gen14B(batch=args.batch)
    raws = gen.text(prompts, max_new=32, tag="answer14b")
    answers = {s["dir"]: {"raw": r, "pred": clean_answer(r)}
               for s, r in zip(todo, raws)}
    (out / "answers.json").write_text(json.dumps(answers))
    records = build_records(tasks, sample_fail, answers)
    rep = report(records, f"{args.title} — answers=14B merges={args.merges}")
    print("\n" + rep)
    (out / "report.txt").write_text(rep + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1))
    write_cell_csv(records, out / "cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
