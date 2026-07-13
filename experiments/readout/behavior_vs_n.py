#!/usr/bin/env python3
"""B3: frozen-model behavioral accuracy vs sequence length N (generation-based answer reader).

Counts can exceed 9 at long N, so the single-token digit argmax used everywhere at N=8 is
insufficient — this reads the answer by greedy generation + first-integer parse (the same
reader validated in C1, where it recovered multi-digit counterfactuals at 0.95-0.99).

Data: the long-N steps_in_room sets (data/mmred_longN_park/seq_len_N/all_uniform), frames
resized at load (default 392px per B0a). Reports exact-match, MAE, and by-gold-count rows.
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri

INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=".../seq_len_N/all_uniform")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id

    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    rows = []
    n = 0
    t_start = time.time()
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            continue
        if not frames:
            continue
        NF = len(frames)
        if int(args.resize) > 0:
            frames = [f.resize((int(args.resize), int(args.resize))) for f in frames]
        prompt = (f"You will be shown {NF} frames describing steps in a house.\n"
                  f"Respond with a single integer from 0 to {NF} (0 is allowed). "
                  f"Output only the integer.\nQuestion: {q0}\nAnswer: ")
        try:
            t0 = time.time()
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt(frames, prompt))
            plen = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                gen = model.generate(**inputs, do_sample=False,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0]
            dt = time.time() - t0
        except Exception as exc:
            print(f"{sid} failed: {type(exc).__name__}: {exc}", flush=True)
            fail = globals().get("_fail", 0) + 1; globals()["_fail"] = fail
            if fail >= 10 and n == 0:
                raise
            continue
        m = INT_RE.search(dec)
        pred = int(m.group(0)) if m else None
        rows.append({"sid": sid, "gold": gold, "pred": pred, "raw": dec,
                     "n_frames": NF, "prompt_tokens": plen, "sec": round(dt, 2)})
        n += 1
        if n % 10 == 0:
            acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
            print(f"  {n}/{args.limit}  acc so far {acc:.3f}  ({dt:.1f}s/sample, {plen} tok)",
                  flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))

    acc = float(np.mean([r["pred"] == r["gold"] for r in rows])) if rows else float("nan")
    mae = float(np.mean([abs((r["pred"] if r["pred"] is not None else -99) - r["gold"])
                         for r in rows])) if rows else float("nan")
    parse_fail = float(np.mean([r["pred"] is None for r in rows])) if rows else float("nan")
    per = defaultdict(list)
    for r in rows:
        per[r["gold"]].append(int(r["pred"] == r["gold"]))
    by_count = {int(g): {"n": len(v), "acc": float(np.mean(v))} for g, v in sorted(per.items())}
    sec = float(np.mean([r["sec"] for r in rows])) if rows else float("nan")
    summary = {"data_root": str(args.data_root), "n": len(rows), "resize": int(args.resize),
               "exact_match": acc, "mae": mae, "parse_fail": parse_fail,
               "sec_per_sample": sec, "by_count": by_count,
               "wallclock_min": round((time.time() - t_start) / 60, 1)}
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"=== BEHAVIOR vs N (generation reader)  {args.data_root}  n={len(rows)} ===",
             f"exact-match {acc:.3f}   MAE {mae:.2f}   parse-fail {parse_fail:.3f}   "
             f"{sec:.1f}s/sample",
             "by gold count: " + " ".join(f"{g}:{d['acc']:.2f}(n{d['n']})"
                                          for g, d in by_count.items())]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
