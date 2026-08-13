#!/usr/bin/env python3
"""LEARNMASK P0 data check (CPU, no model): tokens/frame + sequence geometry at the
campaign resolution 512 vs the legacy 392, on real MMReD-HF samples.

Builds the exact carrier prompt prepare_sample builds (question | [frame, carrier]xN |
question, chat template + generation prompt) with the exact runtime processor config
(use_fast=False), counts image tokens per frame, and projects the per-layer mask-buffer
cost the brief cares about (per-layer in-hook assembly, no 28x seq^2 buffers).

Run: .venv/bin/python scripts/learnmask/p0_token_check.py [--root DIR] [--n 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers import AutoProcessor

from gnnformer.constants import CARRIER_TOKEN, MODEL_7B
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample
from gnnformer.runtime import image_token_groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/mmred_hf/dirs/seq_len_8_train_steps_in_room")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--resize", type=int, nargs="+", default=[512, 392])
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL_7B, trust_remote_code=True, use_fast=False)
    dirs = iter_sample_dirs_shuffled(Path(args.root), 0)[: args.n]
    for sd in dirs:
        _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
        nf = len(frames)
        print(f"\n== {sd.name} (N={nf}, gold={a0}, frame {frames[0].size})")
        for rs in args.resize:
            fr = [f.resize((rs, rs)) for f in frames]
            content = [{"type": "text", "text": q0}]
            for f in fr:
                content.append({"type": "image", "image": f})
                content.append({"type": "text", "text": CARRIER_TOKEN})
            content.append({"type": "text", "text": q0})
            inputs = proc.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            ids = inputs["input_ids"][0]
            groups = image_token_groups(ids, expected_num_frames=nf, processor=proc)
            per = sorted({len(g) for g in groups})
            thw = inputs["image_grid_thw"][0].tolist()
            seq = int(ids.shape[0])
            tokf = per[0]
            blk = tokf + 3  # vision_start + tokens + vision_end + carrier
            print(f"  resize {rs}: grid_thw {thw} -> {tokf} img tok/frame "
                  f"(block {blk}), seq {seq} @N={nf}")
            over = seq - nf * blk
            for n2 in (16, 32, 64, 128):
                s2 = over + n2 * blk
                print(f"    projected N={n2:>3}: seq ~{s2:>6}  "
                      f"fp32 mask {s2 * s2 * 4 / 1e9:6.2f} GB/layer  "
                      f"int16 cellmap {s2 * s2 * 2 / 1e9:6.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
