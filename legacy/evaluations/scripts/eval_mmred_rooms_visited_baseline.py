#!/usr/bin/env python3
"""Track A generalization diagnosis: does the frozen-Qwen aggregation bottleneck appear on a
NON-counting MMReD task? Task = "How many distinct rooms did <C> visit?" (set-cardinality, ans 0..6),
recomputed from the existing rendered samples' states (NO re-render). Reuses the proven model-load +
generation from eval_mmred_qwen25_vl_accuracy.py; only the question + gold differ.

Reports baseline accuracy by seq_len. If accuracy degrades as seq_len grows (like the counting task),
the over-squashing bottleneck generalizes beyond counting.
"""
from __future__ import annotations

import argparse
import ast
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen-Qwen baseline on the rooms-visited MMReD task (Track A).")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="2,4,6,8")
    p.add_argument("--task", default="rooms_visited", choices=["rooms_visited", "co_occupancy"],
                   help="rooms_visited = set-cardinality (ans 0..6); co_occupancy = count frames C&D share a room (ans 0..seq_len).")
    p.add_argument("--max-samples", type=int, default=120, help="cap per seq_len")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "eval_mmred_rooms_visited_baseline")
    return p.parse_args()


def states_of(qa_path: Path) -> List[Dict[str, Any]]:
    lines = qa_path.read_text(encoding="utf-8").splitlines()
    qi = next(i for i, l in enumerate(lines) if l.strip() == "question:")
    ai = next(i for i, l in enumerate(lines) if l.strip() == "answer:")
    return [ast.literal_eval(l.strip()) for l in lines[qi + 1 : ai] if l.strip().startswith("{")]


def rooms_visited(states: List[Dict[str, Any]], character: str) -> int:
    return len({room for st in states for room, occ in st["rooms"].items() if character in occ})


def present_characters(states: List[Dict[str, Any]]) -> List[str]:
    return sorted({c for st in states for occ in st["rooms"].values() for c in occ})


def co_occupancy(states: List[Dict[str, Any]], c1: str, c2: str) -> int:
    return sum(1 for st in states if any(c1 in occ and c2 in occ for occ in st["rooms"].values()))


def build_cooc_prompt(c1: str, c2: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: In how many of the {num_frames} frames were {c1} and {c2} in the same room?\n"
        "Answer: "
    )


def build_rooms_prompt(character: str, num_frames: int, num_rooms: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house with {num_rooms} rooms.\n"
        f"Respond with a single integer from 0 to {num_rooms} (0 is allowed). Output only the integer.\n"
        f"Question: How many distinct rooms did {character} visit across the {num_frames} frames?\n"
        "Answer: "
    )


@torch.inference_mode()
def generate_int(model: Any, processor: Any, frames: List[Image.Image], prompt: str, device: str, max_new_tokens: int) -> Optional[int]:
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in frames] + [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = base.move_inputs_to_device(dict(inputs), device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    pad = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad)
    decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    m = re.search(r"-?\d+", str(decoded))
    return int(m.group(0)) if m else None


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "run.log").open("w", encoding="utf-8")

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    emit(f"Loading {args.model_name} (4bit={args.load_in_4bit}) on {device}/{dtype}")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))

    overall: List[str] = ["seq_len,n,accuracy,mean_gold,mean_pred"]
    for sl in seq_lens:
        sample_root = args.data_root / f"seq_len_{sl}" / args.split
        if not sample_root.is_dir():
            emit(f"seq_len={sl}: missing {sample_root}, skip"); continue
        dirs = [d for d in sorted(sample_root.iterdir()) if (d / "qa.txt").is_file()]
        rng.shuffle(dirs)
        dirs = dirs[: int(args.max_samples)]
        correct = n = 0
        gold_sum = pred_sum = 0
        by_gold: Dict[int, List[int]] = {}
        for d in dirs:
            states = states_of(d / "qa.txt")
            chars = present_characters(states)
            if not chars:
                continue
            num_rooms = len(states[0]["rooms"]) if states else 6
            frames = [Image.open(d / f"{i:03d}.png").convert("RGB") for i in range(len(states))]
            if args.task == "co_occupancy":
                if len(chars) < 2:
                    continue
                C, D = rng.sample(chars, 2)
                gold = co_occupancy(states, C, D)
                prompt = build_cooc_prompt(C, D, len(frames))
            else:
                C = rng.choice(chars)
                gold = rooms_visited(states, C)
                prompt = build_rooms_prompt(C, len(frames), num_rooms)
            pred = generate_int(model, processor, frames, prompt, device, int(args.max_new_tokens))
            n += 1
            gold_sum += gold
            if pred is not None:
                pred_sum += pred
                if pred == gold:
                    correct += 1
            by_gold.setdefault(gold, [0, 0])
            by_gold[gold][1] += 1
            by_gold[gold][0] += int(pred == gold)
            if n % 25 == 0:
                emit(f"  seq_len={sl}: {n}/{len(dirs)} running acc={correct/max(1,n):.3f}")
        acc = correct / max(1, n)
        emit(f"seq_len={sl}: acc={acc:.3f} (n={n}) mean_gold={gold_sum/max(1,n):.2f} mean_pred={pred_sum/max(1,n):.2f}")
        for g in sorted(by_gold):
            c, t = by_gold[g]
            emit(f"    gold={g}: {c}/{t} = {c/max(1,t):.2f}")
        overall.append(f"{sl},{n},{acc:.4f},{gold_sum/max(1,n):.3f},{pred_sum/max(1,n):.3f}")

    (args.output / "accuracy_by_seq_len.csv").write_text("\n".join(overall) + "\n", encoding="utf-8")
    emit("Wrote accuracy_by_seq_len.csv")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
