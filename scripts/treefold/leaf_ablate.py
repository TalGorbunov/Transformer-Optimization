#!/usr/bin/env python3
"""TREEFOLD leaf-prompt ablation — is the binding FP prompt-shaped or model-shaped?

Diagnosis instrument (NOT a method change): re-runs the LEAF stage only, on
steps_in_room samples (binary per-frame gold from the v3 program), under prompt
variants, and reports the confusion split that diagnosed the failure
(FP rate on gold-0 frames, split by queried-room-occupied).

Variants (V0 = canonical leaf_text_prompt, the measured baseline):
  V0 canon    conventions + plan framing + schema + rule   (as in T1)
  V1 minimal  record + rule + reply-JSON only (no conventions, no plan framing)
  V2 think    V0 + "First state where each mentioned character is, then the
              JSON on the last line." (reasoning before emission; JSON = last {...})
  V3 nudge    V0 + generic check line ("Check which room the character is
              actually in before answering; if the condition is not met use 0.")

Task logic stays in the ask's own leaf rule — variants only change FRAMING.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from leaf_text import oracle_states
from tf_common import (Gen, build_tasks, extract_json, leaf_text_prompt,
                       parse_ns, render_state)

THINK = ("\nBefore the JSON: first state in one short sentence which room each "
         "character mentioned in the instructions is actually in, then output "
         "the JSON note on the last line.")
NUDGE = ("\nCheck which room the character is actually in before answering; "
         "if the condition is not met, use 0.")


def minimal_prompt(ask, step, n, rendered):
    return "\n".join([
        f"Frame contents — {rendered}",
        f"Instructions: {ask['leaf']}",
        "Note fields: " + json.dumps(ask["note"]),
        "Reply with ONLY the JSON note for this frame.",
    ])


def last_json(text):
    """For V2: parse the LAST {...} block (reasoning precedes it)."""
    idx = text.rfind("{")
    while idx != -1:
        obj, err = extract_json(text[idx:])
        if obj is not None:
            return obj, None
        idx = text.rfind("{", 0, idx)
    return extract_json(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--arch14b", action="store_true",
                    help="run with Qwen2.5-14B (venv_arch) instead of the 7B")
    ap.add_argument("--variants", default="V0_canon,V1_minimal,V2_think,V3_nudge")
    ap.add_argument("--ns", default="16,32")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    asks = {r["dir"]: r for r in json.load(open(args.asks))}
    tasks = [t for t in build_tasks(8, parse_ns(args.ns))
             if t["type"] == "steps_in_room" and asks.get(t["dir"])
             and asks[t["dir"]]["ask"] and asks[t["dir"]]["combinable"]]
    print(f"{len(tasks)} steps_in_room samples", flush=True)

    frames = []           # (dir, t, char, room, gold, rendered)
    for s in tasks:
        m = re.search(r"did (\w+) spend in the (\w+)", s["q"])
        char, room = m.group(1), m.group(2)
        states = oracle_states(s)
        n = len(states)
        for t, st in enumerate(states):
            gold = 1 if char in st.get(room, []) else 0
            frames.append((s, t + 1, n, char, room, gold,
                           render_state(st, t + 1, n)))

    if args.arch14b:
        from stages_14b import Gen14B
        gen = Gen14B(batch=args.batch)
    else:
        gen = Gen(batch=args.batch)
    results = {}
    for variant in [v.strip() for v in
                    args.variants.replace(",", " ").split()]:
        prompts = []
        for s, step, n, char, room, gold, rendered in frames:
            ask = asks[s["dir"]]["ask"]
            if variant == "V1_minimal":
                p = minimal_prompt(ask, step, n, rendered)
            else:
                p = leaf_text_prompt(ask, step, n, rendered)
                if variant == "V2_think":
                    p += THINK
                elif variant == "V3_nudge":
                    p += NUDGE
            prompts.append(p)
        t0 = time.time()
        raws = gen.text(prompts, max_new=args.max_new, tag=variant, log_every=5)
        conf = Counter()
        fp_split = Counter()
        n_split = Counter()
        for (s, step, n, char, room, gold, rendered), raw in zip(frames, raws):
            obj, err = (last_json(raw) if variant == "V2_think"
                        else extract_json(raw))
            if not isinstance(obj, dict) or "count" not in obj:
                conf["parse_fail"] += 1
                continue
            try:
                pred = 1 if int(obj["count"]) >= 1 else 0
            except Exception:                              # noqa: BLE001
                conf["parse_fail"] += 1
                continue
            conf[("TP" if gold and pred else "FN" if gold else
                  "FP" if pred else "TN")] += 1
            if not gold:
                occ = len(oracle_states(s)[step - 1].get(room, [])) > 0
                n_split[occ] += 1
                if pred:
                    fp_split[occ] += 1
        results[variant] = {"confusion": dict(conf),
                            "fp_room_occupied": (fp_split[True],
                                                 n_split[True]),
                            "fp_room_empty": (fp_split[False], n_split[False]),
                            "raws_sample": raws[:4],
                            "wall_s": round(time.time() - t0)}
        tp, fn = conf["TP"], conf["FN"]
        fp, tn = conf["FP"], conf["TN"]
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        print(f"{variant}: acc {acc:.3f}  TP {tp} FN {fn} FP {fp} TN {tn} "
              f"parse_fail {conf['parse_fail']}  "
              f"FP|occ {fp_split[True]}/{n_split[True]}  "
              f"FP|empty {fp_split[False]}/{n_split[False]}", flush=True)
    (out / "ablation.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out}/ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
