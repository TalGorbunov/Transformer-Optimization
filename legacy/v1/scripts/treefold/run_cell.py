#!/usr/bin/env python3
"""TREEFOLD run one cell end-to-end: asks -> leaves -> fan-k tree -> answer
-> score. Resumable: every stage writes its JSON to the run dir and is loaded
back on restart (greedy decode makes resume exact). The orchestrator holds
ZERO task logic — it only routes questions, frames, step indices and notes.

Leaf sources:
  oracle    text-leaf over GT states (T1/T2/T5)
  captions  text-leaf over Arm B caption states (--captions, T4)
  notes     precomputed leaf notes, e.g. leaf_vlm.py output (--notes, T3)

Examples:
  T1:  run_cell.py --asks <zs>/asks.json --leaf oracle --fan 2 --output ...
  T2:  run_cell.py --asks <zs>/asks.json --leaf oracle --fan 8 --ns 64,128 \\
         --reuse-leaves <T1>/leaf_notes.json --output ...
  T4:  run_cell.py --asks <zs>/asks.json --leaf captions --per-type 4 \\
         --captions outputs/recagg/armB_captions/20260817_203538_full/captions.json ...
  T3:  run_cell.py --asks <zs>/asks.json --leaf notes --per-type 4 \\
         --notes <leaf_vlm_run>/leaf_notes.json --output ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from answer_score import (build_records, fidelity_report, report, run_answers,
                          write_cell_csv)
from leaf_text import caption_states, oracle_states, run_leaves
from tf_common import (COUNT_LIKE, Gen, build_tasks, load_v3_program_map,
                       parse_ns)
from tree_reduce import fidelity_hooks, run_tree


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--leaf", choices=("oracle", "captions", "notes"),
                    required=True)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--reuse-leaves", default=None,
                    help="leaf_notes.json from another run (e.g. T2 reusing T1)")
    ap.add_argument("--fan", type=int, default=2, help="0 = k=N (Arm A shape)")
    ap.add_argument("--ns", default="16,32,64,128")
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-style", choices=("canon", "minimal"),
                    default="canon",
                    help="minimal = clutter-stripped leaf+answer prompts "
                         "(prompt-repair arm, ablation 2026-08-25)")
    ap.add_argument("--max-new-note", type=int, default=96)
    ap.add_argument("--max-new-answer", type=int, default=32)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--title", default="TREEFOLD cell")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    t00 = time.time()

    tasks = build_tasks(args.per_type, parse_ns(args.ns), args.limit)
    assert len({t["dir"] for t in tasks}) == len(tasks), "dir-name collision"
    asks = {r["dir"]: r for r in json.load(open(args.asks))}

    sample_fail = {}                       # dir -> fail kind (pre-answer)
    samples = []                           # tasks that enter the pipeline
    for t in tasks:
        a = asks.get(t["dir"])
        if a is None or a["err"] is not None:
            sample_fail[t["dir"]] = "ask_parse"
        elif not a["combinable"]:
            sample_fail[t["dir"]] = "not_combinable"
        else:
            samples.append({**t, "ask": a["ask"]})
    print(f"{args.title}: {len(tasks)} tasks, {len(samples)} enter pipeline "
          f"({json.dumps({k: sum(1 for v in sample_fail.values() if v == k) for k in set(sample_fail.values())})} pre-failed)",
          flush=True)

    gen_holder = {}
    def gen():
        if "g" not in gen_holder:
            gen_holder["g"] = Gen(batch=args.batch)
        return gen_holder["g"]

    # ------------------------------------------------------------- leaf notes
    leaf_path = out / "leaf_notes.json"
    external = args.notes if args.leaf == "notes" else args.reuse_leaves
    if external:
        leaf_notes = json.load(open(external))
        print(f"[leaves] loaded {len(leaf_notes)} samples from {external}")
    elif leaf_path.is_file():
        leaf_notes = json.load(open(leaf_path))
        print(f"[resume] leaf_notes.json ({len(leaf_notes)} samples)")
    else:
        states_by_dir, missing = {}, []
        caps = json.load(open(args.captions)) if args.leaf == "captions" else None
        for s in samples:
            st = (caption_states(s, caps) if args.leaf == "captions"
                  else oracle_states(s))
            if st is None:
                missing.append(s["dir"])
            else:
                states_by_dir[s["dir"]] = st
        for d in missing:
            sample_fail[d] = "caption_missing"
        samples = [s for s in samples if s["dir"] in states_by_dir]
        from tf_common import leaf_text_prompt, minimal_leaf_prompt
        lb = (minimal_leaf_prompt if args.prompt_style == "minimal"
              else leaf_text_prompt)
        leaf_notes = run_leaves(gen(), samples, states_by_dir,
                                args.max_new_note, leaf_path,
                                prompt_builder=lb)
    if external:
        missing = [s["dir"] for s in samples if s["dir"] not in leaf_notes]
        for d in missing:
            sample_fail[d] = "caption_missing"      # source record missing
        samples = [s for s in samples if s["dir"] in leaf_notes]

    # ------------------------------------------------------------------- tree
    merge_path = out / "merges.json"
    if merge_path.is_file() and not Path(str(merge_path) + ".partial").is_file():
        merges = json.load(open(merge_path))
        print(f"[resume] merges.json")
    else:
        merges = run_tree(gen(), samples, leaf_notes, args.fan,
                          args.max_new_note, merge_path)
    for s in samples:
        f = merges[s["dir"]]["failed"]
        if f:
            sample_fail[s["dir"]] = f["stage"]

    # ---------------------------------------------------------------- answers
    ans_path = out / "answers.json"
    if ans_path.is_file():
        answers = json.load(open(ans_path))
        print(f"[resume] answers.json")
    else:
        from tf_common import answer_prompt, minimal_answer_prompt
        ab = (minimal_answer_prompt if args.prompt_style == "minimal"
              else answer_prompt)
        answers = run_answers(gen(), samples, merges, args.max_new_answer,
                              ans_path, prompt_builder=ab)

    # ------------------------------------------------------------------ score
    records = build_records(tasks, sample_fail, answers)
    rep = report(records, f"{args.title} — fan={args.fan or 'N'} "
                          f"leaf={args.leaf} asks={args.asks}")
    fid = {}
    if any(s["type"] in COUNT_LIKE for s in samples):
        pmap = load_v3_program_map()
        oracle_by_dir = {s["dir"]: oracle_states(s) for s in samples
                         if s["type"] in COUNT_LIKE and s["dir"] in pmap}
        fid = fidelity_hooks(
            [s for s in samples if s["dir"] in oracle_by_dir],
            leaf_notes, merges, pmap, oracle_by_dir)
        (out / "fidelity.json").write_text(json.dumps(fid, indent=1))
        rep += "\n\n" + fidelity_report(fid, records)
    rep += f"\n\nwall {time.time()-t00:.0f}s"
    print("\n" + rep)
    (out / "report.txt").write_text(rep + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1))
    write_cell_csv(records, out / "cells.csv")
    print(f"wrote {out}/report.txt, records.json, cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
