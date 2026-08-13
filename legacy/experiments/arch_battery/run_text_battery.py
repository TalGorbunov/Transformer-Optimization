#!/usr/bin/env python3
"""Cross-architecture text-MMRED battery (Exp 1, 2026-07-13): does an extensive-state
architecture escape the softmax counting collapse?

Model-agnostic wrapper: loads ANY HF causal LM (runs in .venv_arch, NOT the shared .venv),
presents the steps_in_room states-only task with the EXACT prompt template of
evaluations/scripts/eval_mmred_text_frames_acc.py (helpers below are verbatim copies -- that
module's import chain needs nnsight, which this venv intentionally lacks), greedy decoding,
generation reader (first integer in the decoded string; counts exceed 9 so no digit-argmax).

Per model x N: exact match, MAE, bias, emitted range (p95-p5 of predictions -- the clamp
signature), Spearman(pred, gold), majority baseline, parse failures. n=150, seed 2.

Anchor (do not re-run): Qwen2.5-VL-7B text EM 0.196/0.062/0.035/0.020 at N=8/16/24/40,
majority-locked from N=16.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random
import torch

INTEGER_RE = re.compile(r"-?\d+")


# ---- verbatim helpers from evaluations/scripts/eval_mmred_text_frames_acc.py ----
def states_of(qa_path: Path) -> List[Dict[str, Any]]:
    lines = qa_path.read_text(encoding="utf-8").splitlines()
    qi = next(i for i, l in enumerate(lines) if l.strip() == "question:")
    ai = next(i for i, l in enumerate(lines) if l.strip() == "answer:")
    return [ast.literal_eval(l.strip()) for l in lines[qi + 1: ai] if l.strip().startswith("{")]


def frames_as_text(states: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, st in enumerate(states, start=1):
        lines.append(f"Frame {i}:")
        for room, occ in st["rooms"].items():
            who = ", ".join(occ) if occ else "(empty)"
            lines.append(f"  {room}: {who}")
    return "\n".join(lines)


def build_prompt(states: List[Dict[str, Any]], question: str, num_frames: int, hi: int) -> str:
    body = frames_as_text(states)
    head = f"You are given {num_frames} frames describing steps in a house, as text.\n{body}\n\n"
    return (
        head
        + f"Respond with a single integer from 0 to {hi} (0 is allowed). Output only the integer.\n"
        + f"Question: {question}\n"
        + "Answer: "
    )


def parse_pred(decoded: str) -> Optional[int]:
    m = INTEGER_RE.search(str(decoded))
    return int(m.group(0)) if m else None


def question_and_gold(d: Path) -> Optional[Tuple[str, int]]:
    lines = (d / "qa.txt").read_text(encoding="utf-8").splitlines()
    ai = next((i for i, l in enumerate(lines) if l.strip() == "answer:"), -1)
    qi = next((i for i, l in enumerate(lines) if l.strip() == "question:"), -1)
    if ai < 0 or qi < 0:
        return None
    question = next((l.strip() for l in lines[qi + 1: ai]
                     if l.strip() and not l.strip().startswith("{")), None)
    gold_txt = next((l.strip() for l in lines[ai + 1:] if l.strip()), "")
    m = INTEGER_RE.search(gold_txt)
    if question is None or m is None:
        return None
    return question, int(m.group(0))
# ---- end verbatim helpers ----


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).statistic
        return float(r) if np.isfinite(r) else float("nan")
    except Exception:
        return float("nan")


def load_model(name: str, load_in_4bit: bool, trust_remote_code: bool, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    kw: Dict[str, Any] = dict(dtype=getattr(torch, dtype), device_map="auto",
                              trust_remote_code=trust_remote_code)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype))
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new_tokens: int, system: str = "") -> str:
    if tok.chat_template is not None:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    else:
        ids = tok(prompt, return_tensors="pt").input_ids
    if not torch.is_tensor(ids):                # transformers v5 returns BatchEncoding
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--arch-class", required=True,
                    help="softmax | hybrid-mamba2 | hybrid-gdn | mlstm | griffin | ...")
    ap.add_argument("--data", default=("8=data/mmred_images_park,16=data/mmred_text_longN,"
                                       "24=data/mmred_text_longN,40=data/mmred_text_longN,"
                                       "64=data/mmred_text_arch,128=data/mmred_text_arch"),
                    help="comma list N=data_root")
    ap.add_argument("--split", default="all_uniform")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--system", default="", help="optional system prompt (e.g. /no_think)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--limit-n", default="", help="optional comma subset of N values (smoke)")
    ap.add_argument("--output", required=True, help="run dir (created)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    entries = []
    for part in args.data.split(","):
        N, p = part.split("=", 1)
        entries.append((int(N), root / p))
    if args.limit_n:
        keep = {int(x) for x in args.limit_n.replace(",", " ").split()}
        entries = [e for e in entries if e[0] in keep]

    t0 = time.time()
    print(f"loading {args.model_name} (4bit={args.load_in_4bit}, dtype={args.dtype})", flush=True)
    model, tok = load_model(args.model_name, bool(args.load_in_4bit),
                            bool(args.trust_remote_code), args.dtype)
    import transformers
    print(f"loaded in {time.time()-t0:.0f}s | transformers {transformers.__version__} "
          f"| torch {torch.__version__} | chat_template={'yes' if tok.chat_template else 'no'}",
          flush=True)

    rows = ["model,arch_class,N,n,em,mae,bias,range_p95_p5,spearman,majority,parse_fail"]
    pred_rows = ["N,sample_id,gold,pred,correct,raw"]
    for N, data_root in sorted(entries):
        sample_root = data_root / f"seq_len_{N}" / args.split
        if not sample_root.is_dir():
            print(f"N={N}: missing {sample_root}, skip", flush=True)
            continue
        rng = random.Random(int(args.seed))
        dirs = [d for d in sorted(sample_root.iterdir()) if (d / "qa.txt").is_file()]
        rng.shuffle(dirs)
        dirs = dirs[: int(args.n)]
        golds, preds = [], []
        fails = 0
        tN = time.time()
        for j, d in enumerate(dirs):
            states = states_of(d / "qa.txt")
            qg = question_and_gold(d)
            if not states or qg is None:
                continue
            question, gold = qg
            prompt = build_prompt(states, question, len(states), len(states))
            raw = generate(model, tok, prompt, int(args.max_new_tokens), args.system)
            pred = parse_pred(raw)
            golds.append(gold); preds.append(pred)
            if pred is None:
                fails += 1
            clean = str(raw).replace("\n", " ").replace('"', "'")[:120]
            pred_rows.append(f'{N},{d.name},{gold},{pred if pred is not None else ""},'
                             f'{int(pred == gold)},"{clean}"')
            if (j + 1) % 25 == 0:
                print(f"  N={N}: {j+1}/{len(dirs)} ({time.time()-tN:.0f}s)", flush=True)
        g = np.array(golds, dtype=float)
        p_arr = np.array([x if x is not None else np.nan for x in preds], dtype=float)
        ok = ~np.isnan(p_arr)
        em = float(np.mean([pr == go for pr, go in zip(p_arr, g)])) if len(g) else float("nan")
        mae = float(np.nanmean(np.abs(p_arr - g))) if ok.any() else float("nan")
        bias = float(np.nanmean(p_arr) - np.mean(g)) if ok.any() else float("nan")
        rng95 = (float(np.nanpercentile(p_arr, 95) - np.nanpercentile(p_arr, 5))
                 if ok.any() else float("nan"))
        sp = spearman(p_arr[ok], g[ok]) if ok.sum() > 2 else float("nan")
        vals, cnts = np.unique(g, return_counts=True)
        majority = float(cnts.max() / len(g)) if len(g) else float("nan")
        rows.append(f"{args.model_name},{args.arch_class},{N},{len(g)},{em:.4f},{mae:.3f},"
                    f"{bias:+.3f},{rng95:.2f},{sp:.3f},{majority:.4f},{fails}")
        print(f"N={N}: em={em:.3f} mae={mae:.2f} bias={bias:+.2f} range={rng95:.1f} "
              f"spearman={sp:.3f} majority={majority:.3f} fails={fails} n={len(g)} "
              f"({time.time()-tN:.0f}s)", flush=True)
        # incremental write: a walltime/OOM kill must not lose completed N values
        (out / "results.csv").write_text("\n".join(rows) + "\n")
        (out / "predictions.csv").write_text("\n".join(pred_rows) + "\n")

    (out / "results.csv").write_text("\n".join(rows) + "\n")
    (out / "predictions.csv").write_text("\n".join(pred_rows) + "\n")
    (out / "run_config.json").write_text(json.dumps(vars(args) | {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "elapsed_s": round(time.time() - t0)}, indent=2, default=str))
    print(f"wrote {out} ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
