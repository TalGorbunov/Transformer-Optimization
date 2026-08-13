#!/usr/bin/env python3
"""P4k: DEDUP SEMANTICS probe on mmred_natural — does the model count FRAMES or INSTANCES?

Same frame sequences, two questions:
  frames : "In how many of the N frames does a dog appear?"        (gold = k, frame count)
  count  : "How many dogs are there in these N frames in total?"   (support-size reading?)

Registered prediction: if the model implicitly deduplicates repeated content, the two answers
DISSOCIATE on ident cells (k copies of ONE dog → 'count' reading collapses toward 1) and agree
on dist cells (k different dogs → both readings = k). Ties to rooms_visited support-size
semantics. Generation reader; per-cell exact-match vs frame-gold + answer distributions.
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.scripts.patch_importence import group_restoration_importance as gri

INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mmred_natural_v2")
    ap.add_argument("--cells", default="ident_far,dist_far,ident_near,dist_near")
    ap.add_argument("--limit", type=int, default=150, help="per cell")
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    from PIL import Image

    rows = []
    for cell in [c.strip() for c in args.cells.split(",")]:
        dirs = sorted(d for d in (Path(args.data_root) / cell).iterdir()
                      if (d / "meta.json").exists())
        random.Random(args.sample_seed).shuffle(dirs)
        dirs = dirs[: args.limit]
        for i, sd in enumerate(dirs):
            meta = json.loads((sd / "meta.json").read_text())
            gold = int(meta["answer"])
            NF = meta["n_frames"]
            frames = [Image.open(sd / f"frame_{t:02d}.jpg").convert("RGB") for t in range(NF)]
            n_unique = len({f["image_id"] for f in meta["frames"] if f["is_evidence"]})
            for proto, q in [
                ("frames", f"In how many of the {NF} frames does a dog appear?"),
                ("count", f"How many dogs are there in these {NF} frames in total?"),
            ]:
                prompt = (f"You will be shown {NF} frames.\n"
                          f"Respond with a single integer from 0 to {NF} (0 is allowed). "
                          f"Output only the integer.\nQuestion: {q}\nAnswer: ")
                try:
                    inputs = tgi.move_inputs_to_model_device(
                        tgi.build_inputs_from_prompt(frames, prompt))
                    plen = int(inputs["input_ids"].shape[1])
                    with torch.no_grad():
                        gen = model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                             pad_token_id=pad)
                    dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0]
                except Exception as exc:
                    print(f"{sd.name}/{proto} failed: {exc}", flush=True)
                    continue
                m = INT_RE.search(dec)
                pred = int(m.group(0)) if m else None
                rows.append({"cell": cell, "sid": sd.name, "proto": proto, "gold": gold,
                             "n_unique": n_unique, "pred": pred})
            if (i + 1) % 25 == 0:
                print(f"  [{cell}] {i+1}/{len(dirs)}", flush=True)
                (out / "rows.json").write_text(json.dumps(rows, indent=1))

    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    lines = ["=== DEDUP SEMANTICS (frames-question vs count-question) ==="]
    summary = {}
    for cell in sorted({r["cell"] for r in rows}):
        cl = {}
        for proto in ("frames", "count"):
            rr = [r for r in rows if r["cell"] == cell and r["proto"] == proto
                  and r["pred"] is not None]
            acc_frames = float(np.mean([r["pred"] == r["gold"] for r in rr]))
            acc_unique = float(np.mean([r["pred"] == r["n_unique"] for r in rr]))
            mp = float(np.mean([r["pred"] for r in rr]))
            mg = float(np.mean([r["gold"] for r in rr]))
            mu = float(np.mean([r["n_unique"] for r in rr]))
            cl[proto] = {"n": len(rr), "acc_vs_framegold": acc_frames,
                         "acc_vs_unique": acc_unique, "mean_pred": mp,
                         "mean_gold": mg, "mean_unique": mu}
            lines.append(f"  {cell:<11s} {proto:<7s} n={len(rr):<4d} "
                         f"acc-vs-frames {acc_frames:.3f}  acc-vs-unique {acc_unique:.3f}  "
                         f"mean pred {mp:.2f} (frame-gold {mg:.2f}, unique {mu:.2f})")
        summary[cell] = cl
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
