#!/usr/bin/env python3
"""P2b prep (p0p2 campaign, 2026-07-23): convert MLVU-AC question dirs (meta.json +
frame_XXX.jpg) into MMRED-format sample dirs (qa.txt + NNN.png) so the carrier eval
tooling (carrier_layer_lora.py --eval-only --dirs-file) consumes them unchanged.

Per question: uniformly subsample --n-frames of the stored 128 (same np.linspace rule as
experiments/mlvu/eval_ac_behavior.py), resize to --resize max-side, write NNN.png; qa.txt
carries one dummy state dict per frame (the loader counts frames = len(states)), the
open-form action question (same wording as the frozen behavior runs), and the gold count.
meta.json candidates are copied into the sample dir (mcq.json) for the nearest-option
mapping step at collection.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mlvu_ac")
    ap.add_argument("--n-frames", type=int, required=True)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    src = Path(args.data_root)
    out_base = Path(args.out_root) / f"seq_len_{args.n_frames}" / "all_uniform"
    out_base.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in src.iterdir() if (d / "meta.json").exists())
    n_done = 0
    for sd in dirs:
        meta = json.loads((sd / "meta.json").read_text())
        gold = int(meta["answer"])
        action = meta.get("action") or "the action"
        question = (f"In how many separate instances does the '{action}' action "
                    f"scene occur in the video?")
        idx = np.linspace(0, meta["n_frames"] - 1, args.n_frames).round().astype(int)
        od = out_base / sd.name
        od.mkdir(exist_ok=True)
        for j, t in enumerate(idx):
            img = Image.open(sd / f"frame_{t:03d}.jpg").convert("RGB")
            if args.resize:
                w, h = img.size
                sc = args.resize / max(w, h)
                img = img.resize((max(1, round(w * sc)), max(1, round(h * sc))))
            img.save(od / f"{j:03d}.png")
        states = "\n".join("{'step_id': %d}" % (j + 1) for j in range(args.n_frames))
        (od / "qa.txt").write_text(
            f"qid: {sd.name}\nqtype: mlvu_action_count\natype: integer\n"
            f"seq_len: {args.n_frames}\nquestion:\n{states}\n{question}\nanswer:\n{gold}\n",
            encoding="utf-8")
        (od / "mcq.json").write_text(json.dumps(
            {"candidates": meta["candidates"], "gold": gold}), encoding="utf-8")
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(dirs)}", flush=True)
    print(f"wrote {n_done} sample dirs -> {out_base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
