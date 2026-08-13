#!/usr/bin/env python3
"""Frozen-model baseline on the SAME samples the carrier runs use: plain prompt
(frames + question, no carriers/replicas/masks), digit-argmax at the answer position.

Anchor (RESULTS.md [2026-07-18] E1): full-prior N=8, n=900 -> acc 0.219.
--lora-ckpt runs the DRIFT arm: the trained adapter's hooks stay on while the plain
prompt runs (no-harm-on-task check; LoRA alpha now read from the ckpt, not hardcoded).

Usage:
  python scripts/eval_frozen.py --data_root data/mmred_images_park/seq_len_8/all_uniform \
      --limit 900 --shuffle-dirs 0 --output outputs/carrier/frozen
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.data import (
    collect_evidence_frame_indices,
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    read_dirs_file,
)
from gnnformer.runtime import get_layers, load_runtime, move_to_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--question-first", action="store_true")
    ap.add_argument("--lora-ckpt", default=None, help="drift arm: adapter hooks ON, plain prompt")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED")
    ap.add_argument("--dirs-file", default=None, help="same-items floor cells")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/carrier/frozen")
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    lora = None
    if args.lora_ckpt:
        ck = load_carrier_layer_ckpt(Path(args.lora_ckpt))
        lora = attach_lora(get_layers(model), ck.l_open, rank=ck.rank, alpha=ck.alpha,
                           device=rt.device, state=ck.lora_state)
        print(f"[drift-test] LoRA hooks active on PLAIN prompt (ckpt {args.lora_ckpt}, "
              f"trained acc {ck.acc}, alpha {ck.alpha})", flush=True)
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    hits = n_done = n_skip = 0
    mae = 0.0
    per: dict = {}
    if args.dirs_file:
        sample_dirs = read_dirs_file(Path(args.dirs_file))
        data_label = args.dirs_file
    else:
        sample_dirs = (iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs)
                       if args.shuffle_dirs is not None else iter_sample_dirs(Path(args.data_root)))
        data_label = args.data_root
    for sd in sample_dirs:
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            n_skip += 1
            continue
        evid = set(collect_evidence_frame_indices(q0, states))
        if not evid and states and isinstance(states[0], dict) and "natural" in states[0]:
            evid = {t for t, st in enumerate(states) if (st.get("natural", {}) or {}).get("evidence")}
        if not evid:
            mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
            if mm and states:  # co-occupancy floor cells
                nA, nB = mm.group(1), mm.group(2)
                evid = {t for t, st in enumerate(states)
                        if any(nA in (occ or []) and nB in (occ or [])
                               for occ in (st.get("rooms", {}) or {}).values())}
        if (not evid and gold != 0) or gold > 9:
            n_skip += 1
            continue
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}] if args.question_first else []
        for f in frames:
            content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, rt.device)
        with torch.inference_mode():
            lg = model(**inputs).logits[0, -1].float()
        dg = int(np.argmax([float(lg[t]) for t in digit_ids]))
        hits += dg == gold
        mae += abs(dg - gold)
        pg = per.setdefault(gold, [0, 0])
        pg[1] += 1
        pg[0] += dg == gold
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done} acc so far {hits/n_done:.3f}", flush=True)
    pc = " ".join(f"g{g}:{c}/{t}" for g, (c, t) in sorted(per.items()))
    line = (f"FROZEN BASELINE (qfirst={args.question_first}, lora={bool(args.lora_ckpt)}, "
            f"n={n_done}, data={data_label}): acc {hits/max(n_done,1):.3f}  "
            f"MAE {mae/max(n_done,1):.2f}  {pc}")
    (out / "report.txt").write_text(line + "\n")
    print(line)
    print("wrote", out)
    if lora:
        lora.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
