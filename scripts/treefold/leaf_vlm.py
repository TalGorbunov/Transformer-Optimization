#!/usr/bin/env python3
"""TREEFOLD MAP (VLM leaf) — question-conditioned per-frame notes from pixels.

THE METHOD's perception stage (T3): per-frame single-image calls @512 (A3
multipass parity — the measured equivalent of the fenced forward), prompt =
frame + question + leaf rule + note schema -> JSON note. Sampling = the Arm B
subset (armC stride, per_type=4) so T3/T4/Arm B/armC compare on identical
samples. Output schema matches leaf_text.run_leaves (dir -> raws/notes/errs),
so run_cell.py consumes it via --leaf notes --notes <here>/leaf_notes.json.

Also logs per-frame leaf accuracy vs GT on count-like types (v3 program on the
single oracle frame; the p that Arm B lacked) — instrument only.

Resumable per sample (leaf_notes.partial.json).
Smoke:  ... leaf_vlm.py --asks <zs>/asks.json --ns 16 --limit 1 \\
            --output outputs/_scratch/treefold_smoke/leaf_vlm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tf_common import (COUNT_LIKE, Gen, build_tasks, choose_count_field,
                       extract_json, leaf_vlm_prompt, load_v3_program_map,
                       parse_ns, parse_qa, run_program)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--ns", default="16,32,64,128")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    from PIL import Image

    asks = {r["dir"]: r for r in json.load(open(args.asks))}
    tasks = build_tasks(args.per_type, parse_ns(args.ns), args.limit)
    skipped = {"ask_missing": 0, "ask_parse": 0, "not_combinable": 0}
    samples = []
    for t in tasks:
        a = asks.get(t["dir"])
        if a is None:
            skipped["ask_missing"] += 1
        elif a["err"] is not None:
            skipped["ask_parse"] += 1
        elif not a["combinable"]:
            skipped["not_combinable"] += 1
        else:
            samples.append({**t, "ask": a["ask"]})
    print(f"VLM leaves: {len(samples)} samples "
          f"({sum(t['n_frames'] for t in samples)} frames), skipped {skipped}",
          flush=True)

    partial = out / "leaf_notes.partial.json"
    results = json.load(open(partial)) if partial.is_file() else {}
    if results:
        print(f"[resume] {len(results)} samples already captioned", flush=True)
    pmap = load_v3_program_map()
    gen = Gen(batch=args.batch)
    t0 = time.time()
    for si, s in enumerate(samples):
        if s["dir"] in results:
            continue
        d = Path(s["path"])
        n = s["n_frames"]
        frames = [Image.open(d / f"{i:03d}.png").convert("RGB")
                  .resize((args.resize, args.resize)) for i in range(n)]
        prompts = [leaf_vlm_prompt(s["ask"], s["q"], t + 1, n)
                   for t in range(n)]
        raws = gen.vlm(frames, prompts, max_new=args.max_new)
        rec = {"N": s["N"], "type": s["type"], "raws": raws,
               "notes": [], "errs": [], "leaf_acc": None, "field": None}
        for raw in raws:
            note, err = extract_json(raw)
            if note is not None and not isinstance(note, dict):
                note, err = None, "not-a-dict"
            rec["notes"].append(note)
            rec["errs"].append(err)
        # per-frame leaf accuracy vs GT (count-like, instrument only)
        if s["type"] in COUNT_LIKE and s["dir"] in pmap:
            _, states, _ = parse_qa(d / "qa.txt")
            oracle = [{"rooms": st["rooms"]} for st in states]
            golds = [run_program(pmap[s["dir"]], oracle[t:t + 1])[0]
                     for t in range(n)]
            field, agree = choose_count_field(rec["notes"], golds)
            rec["field"], rec["leaf_acc"] = field, (agree if field else None)
        results[s["dir"]] = rec
        partial.write_text(json.dumps(results))
        if (si + 1) % 5 == 0 or si + 1 == len(samples):
            accs = [r["leaf_acc"] for r in results.values()
                    if r["leaf_acc"] is not None]
            pf = np.mean([e is None for r in results.values()
                          for e in r["errs"]])
            print(f"  {si + 1}/{len(samples)} samples  parse-ok {pf:.3f}  "
                  f"count-like leaf-acc {np.mean(accs):.3f} (n={len(accs)})  "
                  f"{time.time() - t0:.0f}s", flush=True)

    (out / "leaf_notes.json").write_text(json.dumps(results))
    partial.unlink(missing_ok=True)
    accs = [r["leaf_acc"] for r in results.values() if r["leaf_acc"] is not None]
    summary = {
        "n_samples": len(results),
        "n_frames": sum(len(r["raws"]) for r in results.values()),
        "parse_ok": float(np.mean([e is None for r in results.values()
                                   for e in r["errs"]])),
        "countlike_leaf_acc_mean": float(np.mean(accs)) if accs else None,
        "countlike_covered": len(accs), "skipped": skipped,
        "wall_s": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE {json.dumps(summary)}")
    print(f"wrote {out}/leaf_notes.json, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
