#!/usr/bin/env python3
"""Frozen Mamba-LM baseline on TEXT-MMRED (architecture comparison vs frozen Qwen).

Loads a pretrained frozen Mamba LM (transformers pure-PyTorch path, no mamba-ssm needed) and evaluates
it on the text-frame version of the MMRED tasks (frames rendered as text), so we can ask: does the
Mamba *architecture* aggregate per-frame evidence better than the Qwen transformer? (Caveat: confounded
by pretraining data / instruction-tuning, and it's text-only — Mamba LMs aren't multimodal.)

Compare per-task acc to frozen Qwen text numbers (steps 0.47 / rooms 0.39 / co-occ 0.34).
"""
from __future__ import annotations
import argparse, re, sys, time, random
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiiuae/falcon-mamba-7b-instruct")
    ap.add_argument("--data-root", default="data/mmred_images_park")
    ap.add_argument("--split", default="all_uniform")
    ap.add_argument("--seq-lens", default="4,6,8")
    ap.add_argument("--tasks", default="steps_in_room,rooms_visited,co_occupancy")
    ap.add_argument("--max-samples", type=int, default=40, help="per task (across seq_lens)")
    ap.add_argument("--output", default="outputs/frame_axis/mamba_lm_text")
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model} (pure-PyTorch Mamba path)...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = next(model.parameters()).device
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    tasks = [t for t in str(args.tasks).split(",") if t]

    @torch.inference_mode()
    def ask(states, question, n):
        prompt = (f"{tf.frames_as_text(states)}\n\n"
                  f"Question: {question}\n"
                  f"Reply with ONLY a single integer (0 to {n}) and nothing else. Do not explain. Final answer: ")
        msgs = [{"role": "user", "content": prompt}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
        except Exception:
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        gen = model.generate(ids, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        # take the LAST integer in [0,n] (the conclusion), ignoring the echoed "{n} frames" mid-reasoning
        cand = [int(x) for x in re.findall(r"\d+", txt) if 0 <= int(x) <= n]
        pred = cand[-1] if cand else -1
        if ask._dbg < 8:
            print(f"  [dbg] gold? raw={txt!r} -> pred={pred}", flush=True); ask._dbg += 1
        return pred
    ask._dbg = 0

    rows = ["task,n,acc,mae,mean_gold,mean_pred"]
    report = [f"=== FROZEN MAMBA-LM TEXT-MMRED ({args.model}) ==="]
    for task in tasks:
        rng = random.Random(0)
        nc = ncorr = 0; gs = ps = 0.0; absd = 0.0
        for sl in seq_lens:
            sr = Path(args.data_root) / f"seq_len_{sl}" / args.split
            if not sr.is_dir():
                continue
            dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
            rng.shuffle(dirs)
            per = max(1, args.max_samples // len(seq_lens))
            for d in dirs[:per]:
                states = rv.states_of(d / "qa.txt")
                if not states:
                    continue
                qg = tf.question_and_gold(task, d, states, random.Random(hash(d.name) & 0xffff))
                if qg is None:
                    continue
                question, gold, _ = qg
                pred = ask(states, question, len(states))
                nc += 1; ncorr += int(pred == gold); gs += gold; ps += max(pred, 0); absd += abs(max(pred, 0) - gold)
                if nc % 10 == 0:
                    print(f"  {task}: {nc} acc={ncorr/nc:.3f}", flush=True)
        acc = ncorr / max(1, nc)
        rows.append(f"{task},{nc},{acc:.4f},{absd/max(1,nc):.3f},{gs/max(1,nc):.3f},{ps/max(1,nc):.3f}")
        report.append(f"{task:16s}: acc={acc:.3f}  mae={absd/max(1,nc):.2f}  n={nc}  (Qwen text ref: steps .47/rooms .39/co-occ .34)")
    rep = "\n".join(report) + "\n"
    print("\n" + rep)
    (out / "by_task.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "report.txt").write_text(rep, encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
