#!/usr/bin/env python3
"""RECAGG Arm A — frozen pretrained readers over ORACLE per-frame captions.

The task-agnostic question, isolated: can a frozen pretrained LM aggregate
(count) over a stream of clean per-frame statements, and how does that ability
scale with N — recurrent/hybrid architectures vs attention?

No VLM, no training, no images: per-frame captions are rendered from MMReD-HF's
own ground-truth state dicts (qa.txt), the question and gold answer are the
benchmark's own. Perception error is exactly zero by construction; every miss is
aggregation (or formatting, which is reported separately).

Protocol (pre-registered):
  - plain-text few-shot completion, SAME prompt format for every model (base and
    instruct alike — comparability over per-model tuning); exemplars are FIXED
    across all N and all models, drawn from the N=4 TRAIN pool (never eval).
  - greedy decoding, answer = first integer in the continuation; no integer =
    format failure (reported separately from wrong counts, never silently 0).
  - eval pools: N=8/16 train_steps_in_room (200 each; the whole pool),
    N=32/64/128 test pools (50 steps_in_room dirs each — thin, CIs shown).
  - runs in venv_arch (transformers 5.x, native Mamba/xLSTM/RWKV inference).

Usage (one model per invocation; see slurm/recagg_frozen_readers.sbatch):
  <venv_arch>/bin/python scripts/recagg/frozen_readers.py \
      --model NX-AI/xLSTM-7b --label xlstm7b \
      --output outputs/recagg/armA_readers/<ts>_xlstm7b
  --dry-prompts renders the prompts and exits (CPU, login-OK).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import evidence_bits, parse_qa  # noqa: E402  (read-only)

POOLS = [  # (root, N, limit)
    ("data/mmred_hf/dirs/seq_len_8_train_steps_in_room", 8, 200),
    ("data/mmred_hf/dirs/seq_len_16_train_steps_in_room", 16, 200),
    ("data/mmred_hf/dirs/seq_len_32_test", 32, 50),
    ("data/mmred_hf/dirs/seq_len_64_test", 64, 50),
    ("data/mmred_hf/dirs/seq_len_128_test", 128, 50),
]
EXEMPLAR_POOL = "data/mmred_hf/dirs/seq_len_4_train_steps_in_room"

INSTRUCTIONS = ("Below are logs of security-camera frames. Each frame lists who "
                "is in each room. Answer the question with a single number.\n\n")


def steps_dirs(root: str):
    """Sorted steps_in_room sample dirs (name carries K<count>; sorted = K0-first,
    so any slice below the pool size MUST stride — the K0-sorted-trap rule)."""
    r = Path(root)
    return sorted(p for p in r.iterdir()
                  if p.is_dir() and p.name.startswith("steps_in_room")
                  and (p / "qa.txt").is_file())


def take(dirs, limit):
    if limit <= 0:
        return []                 # shots=0 / no-op
    if limit >= len(dirs):
        return dirs
    step = len(dirs) / limit
    return [dirs[int(i * step)] for i in range(limit)]   # even stride, never head


def caption(state: dict) -> str:
    """One frame's ground-truth state -> caption naming every occupied room."""
    parts = []
    for room, chars in state["rooms"].items():
        if chars:
            names = " and ".join(chars)
            verb = "are" if len(chars) > 1 else "is"
            parts.append(f"{names} {verb} in the {room}")
    return "; ".join(parts) + "." if parts else "all rooms are empty."


def render(states, question) -> str:
    lines = [f"Frame {i + 1}: {caption(s)}" for i, s in enumerate(states)]
    return "\n".join(lines) + f"\nQuestion: {question}\nAnswer:"


def tally_answer(question: str, states: list, gold: int) -> str:
    """Exemplar answer in running-tally scratchpad form — the symbolic
    externalization: the count lives in the TOKEN STREAM, not in a hidden
    state, so it is exact at any N (this is the caption-scan mechanism)."""
    bits = evidence_bits(question, states)
    assert bits is not None and sum(bits) == gold
    c = 0
    parts = []
    for i, b in enumerate(bits):
        c += b
        parts.append(f"Frame {i + 1}: {'yes' if b else 'no'} (count={c})")
    return " " + ". ".join(parts) + f". Final answer: {gold}"


def build_prompt_prefix(shots: int, style: str) -> str:
    ex_dirs = take(steps_dirs(EXEMPLAR_POOL), shots)
    blocks = []
    for d in ex_dirs:
        q, states, a = parse_qa(d / "qa.txt")
        gold = int(str(a).strip())
        ans = (tally_answer(q, states, gold) if style == "tally"
               else f" {gold}")
        blocks.append(render(states, q) + ans)
    return INSTRUCTIONS + "\n\n".join(blocks) + "\n\n"


INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="override per-pool limit")
    ap.add_argument("--ns", default="8,16,32,64,128")
    ap.add_argument("--style", default="direct", choices=["direct", "tally"],
                    help="tally = running-count scratchpad exemplars (the "
                         "symbolic-externalization test); answer read from "
                         "'Final answer:' / last integer")
    ap.add_argument("--adapter", default=None,
                    help="path to a LoRA adapter dir (peft) to load on top")
    ap.add_argument("--chunk", type=int, default=0,
                    help="divide-and-conquer inference: split each stream into "
                         "<=chunk-frame windows, query each window separately, "
                         "SUM the integer answers (any window unparsable -> "
                         "format failure). 0 = whole-stream")
    ap.add_argument("--dry-prompts", action="store_true")
    args = ap.parse_args()
    ns_want = {int(x) for x in args.ns.replace(",", " ").split()}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = build_prompt_prefix(args.shots, args.style)

    tasks = []   # (N, prompt, gold, sample_key) — sample_key groups windows
    for root, N, limit in POOLS:
        if N not in ns_want:
            continue
        dirs = take(steps_dirs(root), args.limit or limit)
        golds = []
        for d in dirs:
            q, states, a = parse_qa(d / "qa.txt")
            assert len(states) == N, f"{d}: {len(states)} states != {N}"
            gold = int(str(a).strip())
            if args.chunk:
                for a0 in range(0, N, args.chunk):
                    win = states[a0:a0 + args.chunk]
                    tasks.append((N, prefix + render(win, q), gold,
                                  (N, d.name)))
            else:
                tasks.append((N, prefix + render(states, q), gold, (N, d.name)))
            golds.append(gold)
        c = {k: int(v) for k, v in
             zip(*np.unique(np.array(golds), return_counts=True))}
        print(f"N={N}: {len(dirs)} samples, gold dist {c}")

    if args.dry_prompts:
        print("\n----- sample prompt (last task) -----\n")
        print(tasks[-1][1][-2500:])
        print(f"\n[gold {tasks[-1][2]}]  prefix tokens ~{len(prefix.split())} words; "
              f"{len(tasks)} prompts total")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"loaded LoRA adapter: {args.adapter}")
    print(f"loaded {args.model} in {time.time() - t0:.0f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")

    FINAL_RE = re.compile(r"[Ff]inal answer[:\s]*(-?\d+)")
    preds, raws = [], []
    for i in range(0, len(tasks), args.batch):
        chunk = tasks[i:i + args.batch]
        max_new = (args.max_new if args.style == "direct"
                   else 12 * max(N for N, _, _, _ in chunk) + 32)
        enc = tok([p for _, p, _, _ in chunk], return_tensors="pt", padding=True,
                  padding_side="left").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 stop_strings=["\n\n"], tokenizer=tok)
        for j in range(len(chunk)):
            cont = tok.decode(gen[j, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            if args.style == "tally":
                m = FINAL_RE.search(cont)
                ints = INT_RE.findall(cont)
                preds.append(int(m.group(1)) if m
                             else int(ints[-1]) if ints else None)
            else:
                m = INT_RE.search(cont)
                preds.append(int(m.group()) if m else None)
            raws.append(cont.strip()[-80:] if args.style == "tally"
                        else cont.strip()[:60])
        if (i // args.batch) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(tasks)} {time.time() - t0:.0f}s",
                  flush=True)

    # ---------------------------------------------------------------- report
    lines = [f"ARM A frozen reader — {args.label} ({args.model})",
             f"shots={args.shots} greedy max_new={args.max_new} plain-completion "
             f"format (uniform across models)", "",
             f"{'N':>4} {'n':>4} {'EM':>6} {'ci95':>15} {'MAE':>6} {'fmtfail':>7} "
             f"{'majority':>8}"]
    # group window-tasks back into samples (whole-stream mode: 1 task/sample);
    # sample pred = sum of window answers, None if any window unparsable
    by_sample: dict = {}
    for (N, _, gold, key), p in zip(tasks, preds):
        ent = by_sample.setdefault(key, {"N": N, "gold": gold, "preds": []})
        ent["preds"].append(p)
    results = {}
    for root, N, limit in POOLS:
        if N not in ns_want:
            continue
        cells = [e for k, e in by_sample.items() if k[0] == N]
        g = np.array([e["gold"] for e in cells])
        p = [sum(e["preds"]) if all(x is not None for x in e["preds"]) else None
             for e in cells]
        n_here = len(cells)
        ok = np.array([x is not None and x == y for x, y in zip(p, g)], float)
        fmt = float(np.mean([x is None for x in p]))
        mae = float(np.mean([abs(x - y) for x, y in zip(p, g) if x is not None])
                    ) if any(x is not None for x in p) else float("nan")
        maj = float(np.bincount(g).max() / len(g))
        bs = [ok[np.random.default_rng(s).integers(0, len(ok), len(ok))].mean()
              for s in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"{N:>4} {n_here:>4} {ok.mean():>6.3f} [{lo:.3f},{hi:.3f}] "
                     f"{mae:>6.2f} {fmt:>7.3f} {maj:>8.3f}")
        results[N] = {"em": float(ok.mean()), "ci": [float(lo), float(hi)],
                      "mae": mae, "format_fail": fmt, "majority": maj,
                      "n": int(n_here)}
    lines += ["", f"wall {time.time() - t0:.0f}s"]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    (out / "raw_continuations.json").write_text(json.dumps(raws, indent=2))
    print(f"wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
