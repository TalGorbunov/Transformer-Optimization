#!/usr/bin/env python3
"""Frozen-model baseline on the SAME samples the carrier runs use (same iteration order, same
limit): plain prompt (frames + question, no carriers/replicas/masks), digit-argmax at the answer
position. Gives the same-prior baseline for tonight's N=8 exact-count numbers."""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import (iter_sample_dirs, iter_sample_dirs_shuffled,
                                       load_mmred_sample)
from evaluations.scripts.patch_importence import group_restoration_importance as gri


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--question-first", action="store_true")
    ap.add_argument("--lora-ckpt", default=None,
                    help="DRIFT TEST (2026-07-19): load a carrier_layer_best.pt and register its "
                         "LoRA hooks while running the PLAIN prompt (no carriers/masks) — "
                         "measures whether the trained adapter damages normal behavior. "
                         "Compare against the same run without this flag.")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED",
                    help="stratified deterministic shuffle of sample dirs (class-balanced "
                         "prefixes; REQUIRED whenever --limit < the full dir count)")
    ap.add_argument("--dirs-file", default=None,
                    help="P2a Arm 4 (2026-07-24): file of sample-dir paths — overrides "
                         "--data_root iteration (same-items floor cells)")
    ap.add_argument("--output", default="outputs/ladder/image_longN/frozen_baseline")
    args = ap.parse_args()

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    if args.lora_ckpt:
        from models.model import get_layers as _gl
        layers_ = _gl(model)
        ckl = torch.load(args.lora_ckpt, map_location="cpu")
        scale = 16.0 / int(ckl["rank"])                     # alpha=16 (trainer default)
        n_h = 0
        for key, (A, B) in ckl["lora"].items():
            li, nm = key.split("."); li = int(li)
            A = A.float().to(model.device); B = B.float().to(model.device)
            mod = getattr(layers_[li].self_attn, nm)

            def mk(A=A, B=B):
                def hook(_m, inp, o):
                    x = inp[0]
                    return o + (scale * (x.float() @ A.T) @ B.T).to(o.dtype)
                return hook
            mod.register_forward_hook(mk()); n_h += 1
        print(f"[drift-test] {n_h} LoRA hooks active on PLAIN prompt "
              f"(ckpt {args.lora_ckpt}, trained acc {ckl.get('acc'):.3f})", flush=True)
    tok = processor.tokenizer
    dev = model.device
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    hits = n_done = n_skip = 0
    mae = 0.0
    per = {}
    if args.dirs_file:
        sample_dirs = [Path(l.strip()) for l in open(args.dirs_file) if l.strip()]
    else:
        sample_dirs = (iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs)
                       if args.shuffle_dirs is not None else iter_sample_dirs(Path(args.data_root)))
    for sd in sample_dirs:
        if n_done >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            n_skip += 1; continue
        evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
        if not evid and states and isinstance(states[0], dict) and "natural" in states[0]:
            # P3a (2026-07-24): natural composed samples — evidence from judge-gated flags
            evid = {t for t, st in enumerate(states)
                    if (st.get("natural", {}) or {}).get("evidence")}
        if not evid:
            mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
            if mm and states:   # P2a Arm 4 (2026-07-24): co-occupancy floor cells
                nA, nB = mm.group(1), mm.group(2)
                evid = {t for t, st in enumerate(states)
                        if any(nA in (occ or []) and nB in (occ or [])
                               for occ in (st.get("rooms", {}) or {}).values())}
        if not evid and gold != 0:
            n_skip += 1; continue
        if gold > 9:
            n_skip += 1; continue
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = ([{"type": "text", "text": q0}] if args.question_first else [])
        for f in frames:
            content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            lg = model(**inputs).logits[0, -1].float()
        dg = int(np.argmax([float(lg[t]) for t in digit_ids]))
        hits += (dg == gold); mae += abs(dg - gold)
        per.setdefault(gold, [0, 0]); per[gold][1] += 1; per[gold][0] += (dg == gold)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done} acc so far {hits/n_done:.3f}", flush=True)
    pc = " ".join(f"g{g}:{c}/{t}" for g, (c, t) in sorted(per.items()))
    line = (f"FROZEN BASELINE (qfirst={args.question_first}, n={n_done}, data={args.data_root}): "
            f"acc {hits/max(n_done,1):.3f}  MAE {mae/max(n_done,1):.2f}  {pc}")
    (out / "report.txt").write_text(line + "\n")
    print(line); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
