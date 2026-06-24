#!/usr/bin/env python3
"""Experiment 2: per-frame 'look-again' verification pass (extraction-axis CoT).

Instead of one forward over all N frames + aggregation, ask the model a focused SINGLE-FRAME binary
question per frame ('Is C in the R in this frame?'), read P(yes), then SUM across frames to get the
count. This raises the per-frame extraction term (one frame, one focused question = easier perception)
at N x compute. We report:
  (a) per-frame verification bal_acc / AUC  -> the 'raised' extraction ceiling (compare to joint-pass 0.94)
  (b) count exact-match (hard sum of yes/no, and soft sum of P(yes))  -> compare to single-pass adapter
      (steps 0.79). If (a) >> 0.94 and (b) >> 0.79, extraction was the binding constraint and a
      per-frame pass buys it back.

Tasks: steps_in_room (is C in R?) and co_occupancy (are C and D in the same room?).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from evaluations.scripts import probe_evidence_selection_image as pi
from evaluations.scripts import probe_evidence_selection_linear as pr  # auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-frame verification pass (extraction-axis CoT).")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=60, help="cap per seq_len")
    p.add_argument("--task", default="steps_in_room", choices=["steps_in_room", "co_occupancy"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "per_frame_verification")
    return p.parse_args()


def yesno_token_ids(tokenizer):
    yes, no = [], []
    for s in ["yes", " yes", "Yes", " Yes", "YES"]:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            yes.append(ids[0])
    for s in ["no", " no", "No", " No", "NO"]:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            no.append(ids[0])
    return sorted(set(yes)), sorted(set(no))


@torch.inference_mode()
def p_yes_single_frame(model, processor, frame, question, yes_ids, no_ids, device) -> float:
    """One focused forward on ONE frame; return P(yes) = softmax over (max yes-logit, max no-logit)."""
    prompt = f"{question}\nAnswer with a single word: yes or no."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": frame}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    inputs = base.move_inputs_to_device(dict(inputs), device)
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    ly = logits[yes_ids].max(); ln = logits[no_ids].max()
    return float(torch.softmax(torch.stack([ly, ln]), dim=0)[0])


def sample_query(task: str, meta: Dict[str, Any], states: List[Dict[str, Any]]):
    """Return (question_template_fn(state_unused), per_frame_label_fn, gold_count, desc) for the task."""
    if task == "steps_in_room":
        C, R = meta.get("target_character"), meta.get("target_room")
        if not C or not R:
            return None
        q = f"Is {C} in the {R} in this frame?"
        label_fn = lambda st: int(tf.room_of(st, C) == R)
        gold = sum(label_fn(st) for st in states)
        return q, label_fn, gold, f"{C}@{R}"
    else:  # co_occupancy
        freq = Counter(c for st in states for occ in st["rooms"].values() for c in occ)
        if len(freq) < 2:
            return None
        C, D = [c for c, _ in freq.most_common(2)]
        q = f"Are {C} and {D} in the same room in this frame?"

        def label_fn(st):
            rc, rd = tf.room_of(st, C), tf.room_of(st, D)
            return int(rc == rd and rc != "not present")
        gold = sum(label_fn(st) for st in states)
        return q, label_fn, gold, f"{C}&{D}"


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")
    def emit(m): print(m, flush=True); log.write(m + "\n"); log.flush()

    emit(f"PER-FRAME VERIFICATION | task={args.task} model={args.model_name} 4bit={args.load_in_4bit}")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    yes_ids, no_ids = yesno_token_ids(processor.tokenizer)
    emit(f"yes_ids={yes_ids} no_ids={no_ids}")

    pf_p: List[float] = []; pf_y: List[int] = []          # per-frame soft prob + label
    cnt_rows = ["seq_len,gold,count_hard,count_soft"]
    by_sl = defaultdict(lambda: [0, 0, 0])                  # seq_len -> [n, hard_ok, soft_ok]
    n_samples = 0
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            emit(f"seq_len={sl}: missing, skip"); continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        rng.shuffle(dirs); dirs = dirs[: int(args.max_samples)]
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            if not states:
                continue
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            sq = sample_query(args.task, meta, states)
            if sq is None:
                continue
            q, label_fn, gold, _ = sq
            try:
                frames = pi.load_frames(d, states, meta)
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if len(frames) != len(states):
                continue
            probs = []
            for st, fr in zip(states, frames):
                try:
                    py = p_yes_single_frame(model, processor, fr, q, yes_ids, no_ids, device)
                except Exception as exc:
                    emit(f"  frame skip {d.name}: {exc}"); py = 0.0
                probs.append(py); pf_p.append(py); pf_y.append(label_fn(st))
            c_hard = int(sum(1 for p in probs if p > 0.5))
            c_soft = int(round(sum(probs)))
            cnt_rows.append(f"{sl},{gold},{c_hard},{c_soft}")
            agg = by_sl[sl]; agg[0] += 1; agg[1] += int(c_hard == gold); agg[2] += int(c_soft == gold)
            n_samples += 1
        emit(f"seq_len={sl}: samples so far={n_samples}")

    # ---- per-frame verification ceiling ----
    yt = torch.tensor(pf_y); pp = torch.tensor(pf_p)
    pred = (pp > 0.5).long()
    accs = [float((pred[yt == c] == c).float().mean()) for c in (0, 1) if (yt == c).any()]
    bacc = sum(accs) / len(accs) if accs else 0.0
    auc = pr.auc_score(pp, yt)
    emit("")
    emit(f"PER-FRAME verification: bal_acc={bacc:.3f} auc={auc:.3f}  (frames={len(yt)}, pos={yt.float().mean():.2%})")
    emit(f"  -> compare to joint-pass extraction ceiling ~0.94 (is the focused per-frame pass sharper?)")

    # ---- count accuracy from summing per-frame answers ----
    tot = [0, 0, 0]
    emit("\ncount exact-match by seq_len (hard sum / soft sum):")
    for sl in sorted(by_sl):
        n, h, s = by_sl[sl]
        tot[0] += n; tot[1] += h; tot[2] += s
        emit(f"  seq_len={sl}: n={n:3d} hard_acc={h/max(1,n):.3f} soft_acc={s/max(1,n):.3f}")
    emit(f"OVERALL: n={tot[0]} hard_acc={tot[1]/max(1,tot[0]):.3f} soft_acc={tot[2]/max(1,tot[0]):.3f}")
    emit(f"  -> compare to single-pass adapter (steps 0.79). If higher, extraction was the binding constraint.")
    (run_dir / "counts.csv").write_text("\n".join(cnt_rows) + "\n", encoding="utf-8")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
