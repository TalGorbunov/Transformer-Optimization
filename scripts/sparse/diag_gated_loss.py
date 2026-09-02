#!/usr/bin/env python3
"""SPARSE diagnostic — per-sample answer-token loss under the oracle gate, by k.

Frozen model (optionally + adapter), NO training: for each sample compute the
LM loss on the answer tokens under (a) the plain fence and (b) the oracle-gated
fence. Prints per-k mean/max loss. Purpose: locate the heavy tail that makes
gated SFT oscillate (suspect: k=0 — all blocks hidden, prior answer ~N).

Usage: python scripts/sparse/diag_gated_loss.py --limit 60 --output outputs/_scratch/sparse_smoke/diag_loss
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    rooms_to_room2chars,
)
from gnnformer.fencing import FenceHooks, build_block_mask, reset_positions  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

_LORAMECH = _REPO / "scripts" / "loramech"
if str(_LORAMECH) not in sys.path:
    sys.path.insert(0, str(_LORAMECH))
from train_sft_fenced import build_fenced_messages, parse_layout  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--peft-adapter", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    if args.peft_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.peft_adapter), is_trainable=False)
        model.eval()
    hooks = FenceHooks(layers).install()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    per = defaultdict(lambda: defaultdict(list))
    rows = ["sample,k,mode,loss,pred_tok"]
    n = 0
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), 0):
        if n >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            tr = target_of(Path(str(sd)), q0)
            char, room = tr
            evid = {t for t, st in enumerate(states)
                    if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
            assert len(evid) == gold
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        except Exception:
            continue
        full = processor.apply_chat_template(
            build_fenced_messages(frames, q0, answer=gold), add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt")
        prompt = processor.apply_chat_template(
            build_fenced_messages(frames, q0), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
        full = move_to_device(dict(full), rt.device)
        ids = full["input_ids"][0].tolist()
        parsed = parse_layout(ids, tok, q0, len(frames), vs_id)
        if parsed is None:
            continue
        blocks, fin_start = parsed
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :bnd] = -100
        with torch.no_grad():
            pos, _ = rope_fn(full["input_ids"], image_grid_thw=full.get("image_grid_thw"),
                             attention_mask=full.get("attention_mask"))
        pos = reset_positions(pos, blocks, fin_start)
        full.pop("attention_mask", None)
        for mode in ("plain", "gated"):
            hide = ([] if mode == "plain" else
                    [p for t, (a, b) in enumerate(blocks) if t not in evid
                     for p in range(int(a), int(b))])
            mask = build_block_mask(len(ids), blocks, hide_cols=hide)
            hooks.set_mask(mask, rt.device)
            try:
                with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
                    o = model(**full, position_ids=pos.to(rt.device), labels=labels)
                    lg = o.logits[0, bnd - 1]
            finally:
                hooks.clear_mask()
            loss = float(o.loss)
            ptok = tok.decode([int(lg.argmax())]).strip()
            per[mode][gold].append(loss)
            rows.append(f"{sd.name},{gold},{mode},{loss:.4f},{ptok}")
        n += 1
    hooks.remove()
    lines = []
    for mode in ("plain", "gated"):
        for k in sorted(per[mode]):
            v = np.array(per[mode][k])
            lines.append(f"{mode} k={k}: n={len(v)} mean {v.mean():.3f} "
                         f"median {np.median(v):.3f} max {v.max():.3f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "losses.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
