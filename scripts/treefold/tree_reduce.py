#!/usr/bin/env python3
"""TREEFOLD REDUCE — fan-k tree of model-executed merges + fidelity hooks.

Balanced fold over frame order: nodes chunked into groups of k per level, one
short generation per group (a group of 1 passes through with no call); all
groups of a level, across ALL samples, batched together. fan=0 means k=N (one
merge call over all leaf notes = the Arm A protocol with our notes). A merge
parse failure kills the sample (merge_parse_fail at that level) — failures are
never silently scored.

Fidelity hooks (count-like types, instrument only — EM never touches this):
gold partial for a subtree = the canonical armC v3 program (sandboxed,
read-only) run on that subtree's ORACLE frames; the note field is chosen once
per sample by max leaf agreement (audited; no field >= 0.5 => the sample is
counted out of instrument coverage, reported separately).

Importable; run_cell.py drives it. State saved after every level (resumable).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from tf_common import (COUNT_LIKE, choose_count_field, extract_json,
                       match_val, merge_prompt, run_program)


def run_tree(gen, samples: List[dict], leaf_notes: Dict[str, dict], fan: int,
             max_new: int, out_path: Path) -> Dict[str, dict]:
    """samples carry 'ask'. Returns dir -> {nodes, root, failed}."""
    partial = Path(str(out_path) + ".partial")
    if partial.is_file():
        st = json.load(open(partial))
        alive = {d: [tuple(x[:2]) + (x[2],) for x in v]
                 for d, v in st["alive"].items()}
        nodes, failed, level = st["nodes"], st["failed"], st["level"]
        print(f"[tree] resuming at level {level + 1}", flush=True)
    else:
        alive, nodes, failed, level = {}, {}, {}, 0
        for s in samples:
            ln = leaf_notes[s["dir"]]
            if any(e is not None for e in ln["errs"]):
                failed[s["dir"]] = {"stage": "leaf_parse", "level": 0}
                continue
            alive[s["dir"]] = [(t + 1, t + 1, note)
                               for t, note in enumerate(ln["notes"])]
            nodes[s["dir"]] = []
    ask_by_dir = {s["dir"]: s["ask"] for s in samples}

    t0 = time.time()
    while any(len(v) > 1 for v in alive.values()):
        level += 1
        queue = []                                    # (dir, gi, children)
        prompts = []
        next_state: Dict[str, list] = {}
        for d, ns in alive.items():
            if len(ns) == 1:
                next_state[d] = ns
                continue
            k = len(ns) if fan == 0 else fan
            groups = [ns[i:i + k] for i in range(0, len(ns), k)]
            next_state[d] = [None] * len(groups)      # filled below
            for gi, g in enumerate(groups):
                if len(g) == 1:
                    next_state[d][gi] = g[0]
                else:
                    queue.append((d, gi, g))
                    prompts.append(merge_prompt(ask_by_dir[d], g))
        print(f"[tree] level {level}: {len(prompts)} merges "
              f"({sum(len(v) > 1 for v in alive.values())} samples alive)",
              flush=True)
        raws = gen.text(prompts, max_new=max_new, tag=f"merge-L{level}",
                        log_every=10)
        for (d, gi, g), raw in zip(queue, raws):
            note, err = extract_json(raw)
            if note is not None and not isinstance(note, dict):
                note, err = None, "not-a-dict"
            nodes[d].append({"level": level, "start": g[0][0], "end": g[-1][1],
                             "raw": raw, "note": note, "err": err})
            if err:
                failed.setdefault(d, {"stage": "merge_parse", "level": level})
                next_state[d] = None
            elif next_state[d] is not None:
                next_state[d][gi] = (g[0][0], g[-1][1], note)
        alive = {d: v for d, v in next_state.items() if v is not None}
        partial.write_text(json.dumps(
            {"level": level, "alive": {d: [list(x) for x in v]
                                       for d, v in alive.items()},
             "nodes": nodes, "failed": failed}))
        print(f"[tree] level {level} done  {time.time()-t0:.0f}s", flush=True)

    out = {}
    for s in samples:
        d = s["dir"]
        root = alive[d][0][2] if d in alive and alive[d] else None
        out[d] = {"nodes": nodes.get(d, []), "root": root,
                  "failed": failed.get(d)}
    out_path.write_text(json.dumps(out))
    partial.unlink(missing_ok=True)
    return out


def fidelity_hooks(samples: List[dict], leaf_notes: Dict[str, dict],
                   merges: Dict[str, dict], program_map: Dict[str, str],
                   oracle_by_dir: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Per-sample fidelity record for count-like types. Golds always from
    ORACLE frames (GT), whatever produced the notes."""
    out = {}
    for s in samples:
        d = s["dir"]
        if s["type"] not in COUNT_LIKE or d not in program_map:
            continue
        code = program_map[d]
        frames = [{"rooms": r} for r in oracle_by_dir[d]]
        lgolds = [run_program(code, frames[t:t + 1])[0]
                  for t in range(len(frames))]
        notes = leaf_notes[d]["notes"]
        field, agree = choose_count_field(notes, lgolds)
        rec = {"type": s["type"], "N": s["N"], "field": field,
               "leaf_agree": agree, "p_leaf": None, "levels": {}}
        if field is not None:
            defined = [(n, g) for n, g in zip(notes, lgolds)
                       if isinstance(n, dict) and g is not None]
            rec["p_leaf"] = (sum(match_val(n.get(field), g)
                                 for n, g in defined) / len(defined)
                            if defined else None)
            for node in merges[d]["nodes"]:
                if node["note"] is None:
                    continue
                gold, err = run_program(code,
                                        frames[node["start"] - 1:node["end"]])
                if err:
                    continue                     # gold undefined on subtree
                lv = rec["levels"].setdefault(str(node["level"]),
                                              {"hit": 0, "n": 0})
                lv["n"] += 1
                lv["hit"] += int(match_val(node["note"].get(field), gold))
        out[d] = rec
    return out
