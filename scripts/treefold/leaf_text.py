#!/usr/bin/env python3
"""TREEFOLD MAP (text-leaf) — apply the ask's leaf rule to TEXT records.

Isolation arms: the record is either the ORACLE state (parse_qa, T1/T2/T5) or
Arm B's full-state VLM caption (captions.json, T4). One short generation per
frame; all frames of all samples flattened and batched together; partial
results saved every CHUNK prompts so a walltime kill resumes cheaply
(greedy decode => resume is exact).

Importable (run_cell.py drives it); standalone main is for smokes only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from tf_common import extract_json, leaf_text_prompt, parse_qa, render_state

CHUNK = 2000     # prompts per partial-save


def oracle_states(task: dict) -> List[dict]:
    """Room dicts from the benchmark's own qa.txt (GT)."""
    _, states, _ = parse_qa(Path(task["path"]) / "qa.txt")
    return [s["rooms"] for s in states]


def caption_states(task: dict, captions: dict) -> Optional[List[dict]]:
    """Room dicts from an Arm B captions.json record (None if not captioned)."""
    rec = captions.get(task["dir"])
    return None if rec is None else rec["states"]


def run_leaves(gen, samples: List[dict], states_by_dir: Dict[str, List[dict]],
               max_new: int, out_path: Path,
               prompt_builder=leaf_text_prompt) -> Dict[str, dict]:
    """samples carry 'ask'. Returns dir -> {raws, notes, errs}; saves partials
    to out_path.partial and the final result to out_path."""
    flat = []                                    # (dir, t, prompt)
    for s in samples:
        states = states_by_dir[s["dir"]]
        n = len(states)
        for t in range(n):
            flat.append((s["dir"], t,
                         prompt_builder(s["ask"], t + 1, n,
                                        render_state(states[t], t + 1, n))))
    partial = Path(str(out_path) + ".partial")
    raws: List[str] = json.load(open(partial)) if partial.is_file() else []
    if raws:
        print(f"[leaf] resuming from partial: {len(raws)}/{len(flat)}",
              flush=True)
    t0 = time.time()
    while len(raws) < len(flat):
        chunk = [p for _, _, p in flat[len(raws):len(raws) + CHUNK]]
        raws += gen.text(chunk, max_new=max_new, tag="leaf", log_every=10)
        partial.write_text(json.dumps(raws))
        print(f"[leaf] {len(raws)}/{len(flat)}  {time.time()-t0:.0f}s",
              flush=True)

    out: Dict[str, dict] = {}
    for (d, t, _), raw in zip(flat, raws):
        rec = out.setdefault(d, {"raws": [], "notes": [], "errs": []})
        note, err = extract_json(raw)
        if note is not None and not isinstance(note, dict):
            note, err = None, "not-a-dict"
        rec["raws"].append(raw)
        rec["notes"].append(note)
        rec["errs"].append(err)
    out_path.write_text(json.dumps(out))
    partial.unlink(missing_ok=True)
    return out


def main() -> int:                               # smoke-only entrypoint
    import argparse

    from tf_common import Gen, build_tasks, parse_ns
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--ns", default="16")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    asks = {r["dir"]: r for r in json.load(open(args.asks))}
    tasks = [t for t in build_tasks(8, parse_ns(args.ns), args.limit)
             if t["dir"] in asks and asks[t["dir"]]["ask"]
             and asks[t["dir"]]["combinable"]]
    for t in tasks:
        t["ask"] = asks[t["dir"]]["ask"]
    states = {t["dir"]: oracle_states(t) for t in tasks}
    gen = Gen(batch=args.batch)
    notes = run_leaves(gen, tasks, states, args.max_new,
                       out / "leaf_notes.json")
    ok = sum(e is None for r in notes.values() for e in r["errs"])
    tot = sum(len(r["errs"]) for r in notes.values())
    print(f"leaf smoke: {ok}/{tot} notes parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
