#!/usr/bin/env python3
"""Text-frames baseline: feed the 7B model the MMReD frames as TEXT (no images at all),
then measure exact-match answer accuracy on the 3 tasks, binned by (gold count x seq_len),
and save a heatmap.

Each frame is a fully-structured room->occupants snapshot (from qa.txt / states.json), so we can
describe it in words instead of rendering it. This isolates "can the frozen LM aggregate the
per-frame facts" from "can the vision tower read the rendered frame" -- if accuracy is high here
but low with images, the bottleneck is perception; if it is low here too, it is aggregation.

Tasks (one per run, via --task):
  steps_in_room : "How many steps did C spend in room R?"      (question+gold taken from qa.txt)
  rooms_visited : "How many distinct rooms did C visit?"        (ans 0..num_rooms)
  co_occupancy  : "In how many frames were C and D in same room?" (ans 0..seq_len)

The "count" axis of the heatmap is the gold answer for that task.
Reuses model-load + device/dtype + state parsing from the existing image baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv

INTEGER_RE = re.compile(r"-?\d+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-only (frames-as-text) MMReD accuracy on the 3 tasks.")
    p.add_argument("--task", default="steps_in_room",
                   choices=["steps_in_room", "rooms_visited", "co_occupancy"])
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="1,2,3,4,5,6,7,8")
    p.add_argument("--max-samples", type=int, default=80, help="cap per seq_len")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--cot", action=argparse.BooleanOptionalAction, default=False,
                   help="Allow chain-of-thought: ask the model to reason then end with 'Answer: <int>'.")
    p.add_argument("--oracle-list", action=argparse.BooleanOptionalAction, default=False,
                   help="Feed ONLY the queried entities' per-frame rooms (no scene to scan): isolates "
                        "cross-frame fusion from retrieval.")
    p.add_argument("--log-predictions", action=argparse.BooleanOptionalAction, default=True,
                   help="Write a per-sample predictions.csv (seq_len,gold,pred,correct,raw).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "outputs" / "eval_mmred_text_frames_acc")
    return p.parse_args()


def frames_as_text(states: List[Dict[str, Any]]) -> str:
    """Render every frame's room->occupants mapping as plain text (the image, in words)."""
    lines: List[str] = []
    for i, st in enumerate(states, start=1):
        lines.append(f"Frame {i}:")
        for room, occ in st["rooms"].items():
            who = ", ".join(occ) if occ else "(empty)"
            lines.append(f"  {room}: {who}")
    return "\n".join(lines)


def room_of(state: Dict[str, Any], ch: str) -> str:
    for room, occ in state["rooms"].items():
        if ch in occ:
            return room
    return "not present"


def _occ(state: Dict[str, Any], room: str) -> List[str]:
    return state["rooms"].get(room, [])


# Additional Cat-1 tasks (answer = sum over frames of a per-frame binary predicate; ans 0..seq_len).
# Picked on the park dataset for genuine within-sequence count diversity (fire-rate ~0.5, low
# degeneracy); see scratch_cat1_diversity.py. Same DeepSets-sum shape as steps_in_room/co_occupancy.
NEW_CAT1 = ("room_busy", "char_accompanied", "char_alone")


def cat1_task(task: str, states: List[Dict[str, Any]], rng: random.Random):
    """A new Cat-1 task: choose the queried entity DETERMINISTICALLY via `rng` (so the eval, the
    extraction-ceiling Monte-Carlo, and the per-frame extraction probe all agree), then return
    (per_frame_evidence[0/1], question, oracle_body) or None to skip."""
    n = len(states)
    rooms = list(states[0]["rooms"].keys())
    chars = rv.present_characters(states)
    if task == "room_busy":
        R = rng.choice(rooms)
        ev = [int(len(_occ(s, R)) >= 1) for s in states]
        q = f"In how many of the {n} frames was the {R} occupied (had at least one person in it)?"
        oracle = "Per-frame occupancy:\n" + "\n".join(
            f"frame {i+1}: {R} holds {len(_occ(s, R))}" for i, s in enumerate(states))
        return ev, q, oracle
    if task == "char_accompanied":
        if not chars:
            return None
        C = rng.choice(chars)
        ev = [int(room_of(s, C) != "not present" and len(_occ(s, room_of(s, C))) >= 2) for s in states]
        q = f"In how many of the {n} frames was {C} in a room with at least one other person?"
        oracle = "Per-frame:\n" + "\n".join(
            f"frame {i+1}: {C} in {room_of(s, C)}"
            f" with {max(0, len(_occ(s, room_of(s, C))) - 1) if room_of(s, C) != 'not present' else 0} others"
            for i, s in enumerate(states))
        return ev, q, oracle
    if task == "char_alone":
        if not chars:
            return None
        C = rng.choice(chars)
        ev = [int(room_of(s, C) != "not present" and len(_occ(s, room_of(s, C))) == 1) for s in states]
        q = f"In how many of the {n} frames was {C} alone in their room (no one else in the room)?"
        oracle = "Per-frame:\n" + "\n".join(
            f"frame {i+1}: {C} in {room_of(s, C)}"
            f" ({len(_occ(s, room_of(s, C))) if room_of(s, C) != 'not present' else 0} total in room)"
            for i, s in enumerate(states))
        return ev, q, oracle
    return None


def build_prompt(task: str, states: List[Dict[str, Any]], question: str,
                 num_frames: int, num_rooms: int, hi: int, cot: bool,
                 oracle_body: Optional[str] = None) -> str:
    if oracle_body is not None:
        head = oracle_body + "\n\n"
    else:
        body = frames_as_text(states)
        head = f"You are given {num_frames} frames describing steps in a house, as text.\n{body}\n\n"
    if cot:
        return (
            head
            + "Reason step by step: go through the frames one at a time, note each frame that is "
            + "relevant to the question, then count them.\n"
            + f"The answer is an integer from 0 to {hi}. End your response with a line of exactly "
            + "'Answer: <integer>'.\n"
            + f"Question: {question}\n"
        )
    return (
        head
        + f"Respond with a single integer from 0 to {hi} (0 is allowed). Output only the integer.\n"
        + f"Question: {question}\n"
        + "Answer: "
    )


ANSWER_RE = re.compile(r"[Aa]nswer\s*:?\s*(-?\d+)")


def parse_pred(decoded: str, cot: bool) -> Optional[int]:
    """In CoT mode prefer the integer after the final 'Answer:'; else the last integer. In plain
    mode take the first integer."""
    text = str(decoded)
    if cot:
        matches = list(ANSWER_RE.finditer(text))
        if matches:
            return int(matches[-1].group(1))
        ints = INTEGER_RE.findall(text)
        return int(ints[-1]) if ints else None
    m = INTEGER_RE.search(text)
    return int(m.group(0)) if m else None


@torch.inference_mode()
def generate(model: Any, processor: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = base.move_inputs_to_device(dict(inputs), device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    pad = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad)
    return processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


def question_and_gold(task: str, d: Path, states: List[Dict[str, Any]],
                      rng: random.Random) -> Optional[Tuple[str, int, str]]:
    """Return (question, gold int, oracle_body) for the chosen task, or None to skip.

    oracle_body presents ONLY the per-frame evidence needed for the question (the queried
    entities' rooms), with no distractor rooms/characters and no scene to scan -- this isolates
    cross-frame *fusion* from *retrieval* (HERBench's oracle-frame study, in text)."""
    n = len(states)
    if task in NEW_CAT1:
        res = cat1_task(task, states, rng)
        if res is None:
            return None
        ev, question, oracle = res
        return question, int(sum(ev)), oracle
    if task == "steps_in_room":
        lines = (d / "qa.txt").read_text(encoding="utf-8").splitlines()
        ai = next((i for i, l in enumerate(lines) if l.strip() == "answer:"), -1)
        qi = next((i for i, l in enumerate(lines) if l.strip() == "question:"), -1)
        if ai < 0 or qi < 0:
            return None
        question = next((l.strip() for l in lines[qi + 1:ai]
                         if l.strip() and not l.strip().startswith("{")), None)
        gold_txt = next((l.strip() for l in lines[ai + 1:] if l.strip()), "")
        m = INTEGER_RE.search(gold_txt)
        if question is None or m is None:
            return None
        # entities from metadata.json (authoritative); fall back to regex on the question.
        C = R = None
        meta_path = d / "metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            C, R = meta.get("target_character"), meta.get("target_room")
        if not C or not R:
            mm = re.search(r"did (\w+) spend in the (\w+)", question)
            if mm:
                C, R = mm.group(1), mm.group(2)
        seq = [room_of(st, C) for st in states] if C else []
        oracle = (f"{C} was in these rooms across the {n} frames (in order): "
                  f"{', '.join(seq)}.")
        return question, int(m.group(0)), oracle

    chars = rv.present_characters(states)
    if not chars:
        return None
    # balanced datasets pin the queried entity in metadata (query_character / query_pair) so gold ==
    # the engineered count; legacy data has neither -> fall back to a deterministic random choice.
    qmeta = {}
    mp = d / "metadata.json"
    if mp.is_file():
        try:
            qmeta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            qmeta = {}
    if task == "rooms_visited":
        C = qmeta.get("query_character") or rng.choice(chars)
        gold = rv.rooms_visited(states, C)
        seq = [room_of(st, C) for st in states]
        oracle = (f"{C} was in these rooms across the {n} frames (in order): {', '.join(seq)}.")
        return f"How many distinct rooms did {C} visit across the {n} frames?", gold, oracle
    # co_occupancy
    if len(chars) < 2:
        return None
    qpair = qmeta.get("query_pair")
    if qpair and len(qpair) == 2:
        C, D = qpair
    else:
        C, D = rng.sample(chars, 2)
    gold = rv.co_occupancy(states, C, D)
    pairs = [f"frame {i+1}: {C} in {room_of(st, C)}, {D} in {room_of(st, D)}"
             for i, st in enumerate(states)]
    oracle = "Locations across the frames:\n" + "\n".join(pairs)
    return f"In how many of the {n} frames were {C} and {D} in the same room?", gold, oracle


def save_heatmap(grid: Dict[Tuple[int, int], Tuple[int, int]], task: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seq_lens = sorted({sl for sl, _ in grid})
    counts = sorted({c for _, c in grid})
    mat = np.full((len(seq_lens), len(counts)), np.nan)
    for r, sl in enumerate(seq_lens):
        for col, c in enumerate(counts):
            if (sl, c) in grid:
                cor, tot = grid[(sl, c)]
                if tot > 0:
                    mat[r, col] = cor / tot
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(1.2 * len(counts) + 2, 0.7 * len(seq_lens) + 2), dpi=150)
    im = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"Text-frames 7B accuracy: {task}")
    ax.set_xlabel("Gold count")
    ax.set_ylabel("Seq len")
    ax.set_xticks(range(len(counts)), [str(c) for c in counts])
    ax.set_yticks(range(len(seq_lens)), [str(s) for s in seq_lens])
    for r in range(len(seq_lens)):
        for col in range(len(counts)):
            v = mat[r, col]
            if math.isfinite(v):
                ax.text(col, r, f"{v*100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / args.task / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    emit(f"task={args.task} model={args.model_name} 4bit={args.load_in_4bit} device={device}/{dtype}")
    emit(f"data_root={args.data_root} split={args.split} seq_lens={seq_lens} "
         f"max_samples={args.max_samples} cot={args.cot} max_new_tokens={args.max_new_tokens}")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))

    grid: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (seq_len, gold) -> (correct, total)
    rows: List[str] = ["seq_len,gold,correct,total,accuracy"]
    pred_rows: List[str] = ["seq_len,sample_id,gold,pred,correct,raw"]
    # per-seq aggregates for bias analysis
    agg: Dict[int, Dict[str, float]] = {}  # sl -> {n, correct, gold_sum, pred_sum, pred_n, under, over}
    total_correct = total_n = 0

    for sl in seq_lens:
        sample_root = args.data_root / f"seq_len_{sl}" / args.split
        if not sample_root.is_dir():
            emit(f"seq_len={sl}: missing {sample_root}, skip"); continue
        dirs = [d for d in sorted(sample_root.iterdir()) if (d / "qa.txt").is_file()]
        rng.shuffle(dirs)
        dirs = dirs[: int(args.max_samples)]
        a = agg.setdefault(sl, dict(n=0, correct=0, gold_sum=0.0, pred_sum=0.0, pred_n=0, under=0, over=0))
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            if not states:
                continue
            num_rooms = len(states[0]["rooms"])
            qg = question_and_gold(args.task, d, states, rng)
            if qg is None:
                continue
            question, gold, oracle_body = qg
            hi = num_rooms if args.task == "rooms_visited" else len(states)
            prompt = build_prompt(args.task, states, question, len(states), num_rooms, hi,
                                  bool(args.cot), oracle_body if args.oracle_list else None)
            raw = generate(model, processor, prompt, device, int(args.max_new_tokens))
            pred = parse_pred(raw, bool(args.cot))
            ok = int(pred is not None and pred == gold)
            total_n += 1; total_correct += ok
            a["n"] += 1; a["correct"] += ok; a["gold_sum"] += gold
            if pred is not None:
                a["pred_sum"] += pred; a["pred_n"] += 1
                if pred < gold: a["under"] += 1
                elif pred > gold: a["over"] += 1
            key = (sl, gold)
            cor, tot = grid.get(key, (0, 0))
            grid[key] = (cor + ok, tot + 1)
            if args.log_predictions:
                clean = str(raw).replace("\n", " ").replace('"', "'")[:300]
                pred_rows.append(f'{sl},{d.name},{gold},{pred if pred is not None else ""},{ok},"{clean}"')
        mp = a["pred_sum"] / a["pred_n"] if a["pred_n"] else float("nan")
        mg = a["gold_sum"] / a["n"] if a["n"] else float("nan")
        emit(f"seq_len={sl}: acc={a['correct']/max(1,a['n']):.3f} (n={a['n']}) "
             f"mean_gold={mg:.2f} mean_pred={mp:.2f} bias={mp-mg:+.2f} "
             f"under={a['under']} over={a['over']} exact={a['correct']}")

    for (sl, g) in sorted(grid):
        cor, tot = grid[(sl, g)]
        rows.append(f"{sl},{g},{cor},{tot},{cor/max(1,tot):.4f}")
    (run_dir / "accuracy_by_count_seqlen.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if args.log_predictions:
        (run_dir / "predictions.csv").write_text("\n".join(pred_rows) + "\n", encoding="utf-8")
    save_heatmap(grid, args.task, run_dir / "heatmap_acc_count_seqlen.png")
    # overall bias summary
    tot_under = sum(int(a["under"]) for a in agg.values())
    tot_over = sum(int(a["over"]) for a in agg.values())
    emit(f"overall: acc={total_correct/max(1,total_n):.3f} (n={total_n}) "
         f"under={tot_under} over={tot_over} exact={total_correct}")
    emit(f"wrote {run_dir}/ (accuracy_by_count_seqlen.csv, predictions.csv, heatmap_acc_count_seqlen.png)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
