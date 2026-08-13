#!/usr/bin/env python3
"""Stage 1: per-frame ALONE-pass ceiling — one frame + the per-frame question,
answer-restricted first-token logit argmax (single forward per frame, no decode).

Tasks (one per family):
  roomofc  "In which room is {C}?"          labels: 6 rooms      (char_at_frame dirs)
  occofr   "Who is in the {R}?"             labels: 5 names+Nobody (room_at_frame dirs)
  trig     "Is {B} in the {R1}?"            labels: Yes/No       (char_on_char_first_app dirs)
  empty    "Is the {R} empty?" (R rotates)  labels: Yes/No       (room_empty dirs)

Usage: python scripts/mmred_hf/stage1_alone.py --task roomofc --limit 150
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.mmred_hf import CHAR_ORDER, ROOM_ORDER, _char_room, _match, _rooms  # noqa: E402
from gnnformer.runtime import load_runtime, move_to_device  # noqa: E402

TASKS = {
    "roomofc": ("seq_len_8_train_char_at_frame", "char_at_frame"),
    "occofr": ("seq_len_8_train_room_at_frame", "room_at_frame"),
    "trig": ("seq_len_8_train_char_on_char_first_app", "char_on_char_first_app"),
    "empty": ("seq_len_8_train_room_empty", "room_empty"),
}


def build(task, g, states, t):
    """-> (question, candidate answer words, gold index) for frame t."""
    if task == "roomofc":
        c = g[0]
        room = _char_room(states[t], c)
        return (f"In which room is {c}? Answer with the room name only.",
                ROOM_ORDER, ROOM_ORDER.index(room))
    if task == "occofr":
        r = g[0]
        occ = _rooms(states[t]).get(r, [])
        if len(occ) > 1:
            return None
        gold = "Nobody" if not occ else occ[0]
        cands = CHAR_ORDER + ["Nobody"]
        return (f"Who is in the {r}? Answer with the name, or Nobody.",
                cands, cands.index(gold))
    if task == "trig":
        _a, b, r1 = g
        gold = int(b in _rooms(states[t]).get(r1, []))
        return (f"Is {b} in the {r1}? Answer Yes or No.", ["No", "Yes"], gold)
    if task == "empty":
        r = ROOM_ORDER[t % 6]
        gold = int(not _rooms(states[t]).get(r, []))
        return (f"Is the {r} empty? Answer Yes or No.", ["No", "Yes"], gold)
    raise ValueError(task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--output", default="outputs/mmred_hf/stage1")
    args = ap.parse_args()

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    root_name, qtype = TASKS[args.task]
    dirs = sorted((_REPO / f"data/mmred_hf/dirs/{root_name}").iterdir())[: args.limit]

    def first_ids(words):
        ids = []
        for w in words:
            ids.append(tok(" " + w, add_special_tokens=False).input_ids[0])
        assert len(set(ids)) == len(ids), f"first-token collision: {words} -> {ids}"
        return ids

    hits = n = 0
    per_lab = {}
    t0 = time.time()
    for sd in dirs:
        try:
            _sid, frames, q0, states, _a = load_mmred_sample(sd)
            g = _match(qtype, q0)
        except Exception:
            continue
        for t, fr in enumerate(frames):
            b = build(args.task, g, states, t)
            if b is None:
                continue
            q, cands, gold = b
            cid = first_ids(cands)
            fr = fr.resize((args.resize, args.resize)) if args.resize else fr
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": fr},
                {"type": "text", "text": q + "\nAnswer:"}]}]
            inputs = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, rt.device)
            with torch.inference_mode():
                lg = model(**inputs).logits[0, -1]
            pred = int(torch.tensor([lg[i] for i in cid]).argmax())
            hits += pred == gold
            n += 1
            pl = per_lab.setdefault(gold, [0, 0])
            pl[0] += pred == gold
            pl[1] += 1
        if n and n % 200 < 8:
            print(f"  {n} frames {time.time()-t0:.0f}s acc {hits/n:.3f}", flush=True)

    line = (f"STAGE1 ALONE-PASS [{args.task}] n={n} acc {hits/max(n,1):.3f}  "
            f"per-label: " + " ".join(f"{k}:{a}/{b}" for k, (a, b) in sorted(per_lab.items())))
    out = _REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}.txt").write_text(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
