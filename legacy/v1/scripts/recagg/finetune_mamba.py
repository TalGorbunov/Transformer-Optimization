#!/usr/bin/env python3
"""RECAGG adapter route — LoRA fine-tune a pretrained SSM LM to count,
train SHORT (N<=16), test LONG (32/64/128 via frozen_readers --adapter).

The question this answers (STATE 2026-08-17): does installing the counting
circuit by supervision in a pretrained SSM yield a mechanism that length-
extrapolates, or does it drift like the from-scratch R3 head (leak + readout
range)? Either outcome is thesis-grade.

Training data is SYNTHETIC caption streams in the exact frozen_readers direct
format (same INSTRUCTIONS header, same caption renderer, zero-shot form):
MMReD vocab (5 names x 6 HF rooms), sticky random-walk occupancy, and the
target (char, room) trajectory CONSTRUCTED to hit a count k ~ U{0..N} so the
count distribution is uniform (the HF pools are 0-skewed; a counting circuit
needs coverage). The real HF pools are NEVER touched -> eval stays clean.
Loss on answer tokens only. N ~ U{2..16} per example.

Runs in venv_arch. Usage: see slurm/recagg_finetune_mamba.sbatch (train job
chains straight into the frozen_readers eval with --adapter).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "scripts/ninv"))

from frozen_readers import INSTRUCTIONS, render  # noqa: E402
from load_hf_sample import ROOMS_HF  # noqa: E402

NAMES = ("Mary", "Michael", "John", "Daniel", "Sandra")


def make_example(rng: np.random.Generator, n_min: int, n_max: int):
    """One synthetic stream + question + gold, uniform-count by construction."""
    N = int(rng.integers(n_min, n_max + 1))
    rooms = list(ROOMS_HF)
    tgt_char = NAMES[rng.integers(len(NAMES))]
    tgt_room = rooms[rng.integers(len(rooms))]
    k = int(rng.integers(0, N + 1))
    in_frames = set(rng.choice(N, size=k, replace=False).tolist())
    pos = {c: rooms[rng.integers(len(rooms))] for c in NAMES}
    states = []
    for f in range(N):
        for c in NAMES:
            if rng.random() < 0.3:
                pos[c] = rooms[rng.integers(len(rooms))]
        if f in in_frames:
            pos[tgt_char] = tgt_room
        elif pos[tgt_char] == tgt_room:
            pos[tgt_char] = rooms[(rooms.index(tgt_room) + 1
                                   + int(rng.integers(len(rooms) - 1)))
                                  % len(rooms)]
        st = {"rooms": {r: [c for c in NAMES if pos[c] == r] for r in rooms}}
        states.append(st)
    q = f"How many steps did {tgt_char} spend in the {tgt_room}?"
    prompt = INSTRUCTIONS + render(states, q)
    return prompt, f" {k}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="state-spaces/mamba-2.8b-hf")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    # peft rejects out_proj (and conv1d) on mamba: LoRA there is incompatible
    # with the SSM block's merge semantics (learned on job 133671).
    lin_names = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                        if isinstance(m, torch.nn.Linear)
                        and "lm_head" not in n and "embed" not in n
                        and "out_proj" not in n})
    print("LoRA target linear layers:", lin_names)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=lin_names, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def batch_tensors(n):
        prompts, answers = zip(*(make_example(rng, args.n_min, args.n_max)
                                 for _ in range(n)))
        full = [p + a + tok.eos_token for p, a in zip(prompts, answers)]
        enc = tok(full, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, p in enumerate(prompts):
            n_p = len(tok(p)["input_ids"])
            labels[i, :n_p] = -100        # loss on answer tokens only
        return {k: v.to("cuda") for k, v in enc.items()}, labels.to("cuda")

    @torch.no_grad()
    def quick_val(N, n=20):
        """Greedy EM on n fresh synthetic streams of length exactly N."""
        ok = 0
        for _ in range(n):
            p, a = make_example(rng, N, N)
            enc = tok(p, return_tensors="pt").to("cuda")
            g = model.generate(**enc, max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            cont = tok.decode(g[0, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            import re
            m = re.search(r"-?\d+", cont)
            ok += int(m is not None and int(m.group()) == int(a))
        return ok / n

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        enc, labels = batch_tensors(args.batch)
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 100 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            v16, v32 = quick_val(16), quick_val(32)
            print(f"  [val@step{step}] EM N=16 {v16:.2f} | N=32 (extrap "
                  f"preview) {v32:.2f}", flush=True)
            model.train()

    model.save_pretrained(out / "adapter")
    print(f"saved adapter -> {out}/adapter   wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
