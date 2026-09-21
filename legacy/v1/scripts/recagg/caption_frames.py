#!/usr/bin/env python3
"""RECAGG Arm B step 1 — per-frame structured captions from the frozen VLM.

Captions each frame of the Arm C sample set independently (single-image calls;
the A3 one-forward fenced supply has measured MULTIPASS PARITY, so per-frame
calls are the measured-equivalent of the fenced forward — the fenced variant is
an efficiency optimization, not a different measurement). Output = the same
records structure Arm C executes over, plus caption fidelity vs the GT states.

Sampling MUST mirror ask_compile_execute.py exactly (same pools, same per-type
stride) so Arm B can re-execute the SAME v3 programs on VLM records — the
controlled swap that isolates perception. --per-type <= the Arm C run's 8;
subsetting takes every (8/per_type)-th of each type group.

Usage: see slurm/recagg_caption_frames.sbatch. Smoke:
  ... caption_frames.py --ns 16 --per-type 1 --output outputs/_scratch/armB_smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import ROOMS_HF, parse_qa  # noqa: E402

NAMES = ("Mary", "Michael", "John", "Daniel", "Sandra")

CAPTION_PROMPT = (
    "Characters: Mary, Michael, John, Daniel, Sandra. "
    "Rooms: Kitchen, Bathroom, Garden, Office, Bedroom, Hallway. "
    "The room names are written on the floor plan. List who is in each room "
    "in this frame, in exactly this format: 'Room: Name, Name; Room: Name'. "
    "Only list occupied rooms. Answer:")


def type_of(dirname: str) -> str:
    m = re.match(r"([a-z_]+?)_[A-Z0-9]", dirname)
    return m.group(1) if m else dirname


def sample_dirs(root: Path, per_type: int, armc_per_type: int = 8):
    """The exact Arm C stride, then every (armc_per_type/per_type)-th of it."""
    by_type = defaultdict(list)
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "qa.txt").is_file():
            by_type[type_of(d.name)].append(d)
    out = []
    for t, dirs in sorted(by_type.items()):
        step = max(1, len(dirs) // armc_per_type)
        armc = dirs[::step][:armc_per_type]
        sub = max(1, armc_per_type // per_type)
        out += armc[::sub][:per_type]
    return out


_ROOM_RE = re.compile(rf"({'|'.join(ROOMS_HF)})\s*:\s*([A-Za-z ,&]+)",
                      re.IGNORECASE)


def parse_caption(text: str):
    """'Kitchen: Mary, Michael; Garden: John' -> rooms dict. Tolerates the
    model's literal 'Room: Kitchen: ...' prefix (smoke 133777 lesson: match the
    ROOM NAME anywhere in the segment, not the first token). Unknown names and
    rooms dropped; a name claimed twice keeps the first mention."""
    rooms = {r: [] for r in ROOMS_HF}
    placed = set()
    for part in re.split(r"[;\n]", text):
        m = _ROOM_RE.search(part)
        if not m:
            continue
        room = m.group(1).capitalize()
        for name in re.split(r"[,&]| and ", m.group(2)):
            name = name.strip().capitalize()
            if name in NAMES and name not in placed:
                rooms[room].append(name)
                placed.add(name)
    return {"rooms": rooms}


def fidelity(pred_states, gt_states):
    """Per-character placement accuracy + per-frame exact-state rate."""
    char_ok = frame_ok = 0
    n_char = len(gt_states) * len(NAMES)
    for p, g in zip(pred_states, gt_states):
        loc_p = {c: r for r, who in p["rooms"].items() for c in who}
        loc_g = {c: r for r, who in g["rooms"].items() for c in who}
        ok = sum(loc_p.get(c) == loc_g.get(c) for c in NAMES)
        char_ok += ok
        frame_ok += int(ok == len(NAMES))
    return char_ok / n_char, frame_ok / len(gt_states)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="16,32,64,128")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ns_want = [int(x) for x in args.ns.replace(",", " ").split()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image

    from gnnformer.runtime import load_runtime

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor = rt.model, rt.processor
    t0 = time.time()

    results = {}   # dir_name -> {"N", "captions", "states", ...}
    char_accs, frame_accs = [], []
    for N in ns_want:
        root = Path(f"data/mmred_hf/dirs/seq_len_{N}_test")
        dirs = sample_dirs(root, args.per_type)
        print(f"N={N}: captioning {len(dirs)} samples x {N} frames", flush=True)
        for d in dirs:
            _, gt_states, _ = parse_qa(d / "qa.txt")
            frames = [Image.open(d / f"{i:03d}.png").convert("RGB")
                      .resize((args.resize, args.resize))
                      for i in range(len(gt_states))]
            caps = []
            for b0 in range(0, len(frames), args.batch):
                imgs = frames[b0:b0 + args.batch]
                msgs = [[{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": CAPTION_PROMPT}]}] for _ in imgs]
                texts = [processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True) for m in msgs]
                enc = processor(text=texts, images=imgs, return_tensors="pt",
                                padding=True).to(rt.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=args.max_new,
                                         do_sample=False)
                for k in range(len(imgs)):
                    caps.append(processor.tokenizer.decode(
                        gen[k, enc["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip())
            states = [parse_caption(c) for c in caps]
            ca, fa = fidelity(states, gt_states)
            char_accs.append(ca)
            frame_accs.append(fa)
            results[d.name] = {"N": N, "captions": caps,
                               "states": [s["rooms"] for s in states],
                               "char_acc": ca, "frame_acc": fa}
            done = len(results)
            if done % 10 == 0:
                print(f"  {done} samples, char_acc {np.mean(char_accs):.4f} "
                      f"frame_acc {np.mean(frame_accs):.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)

    summary = {"char_acc_mean": float(np.mean(char_accs)),
               "frame_acc_mean": float(np.mean(frame_accs)),
               "n_samples": len(results), "args": vars(args)}
    (out / "captions.json").write_text(json.dumps(results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE {len(results)} samples  per-char placement acc "
          f"{summary['char_acc_mean']:.4f}  per-frame exact-state "
          f"{summary['frame_acc_mean']:.4f}  wall {time.time() - t0:.0f}s")
    print(f"wrote {out}/captions.json, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
