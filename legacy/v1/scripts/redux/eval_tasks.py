#!/usr/bin/env python3
"""REDUX C1/C3 — plain frames-first behavioral ladder over {count, exists, majority}.

Loop and conventions from scripts/eval_frozen.py (loader, resize 392) but greedy
GENERATE + task-aware prompts/parsing from scripts/redux/tasks.py (the one label
module). Task inferred per dirs-file from its name (exam_{task}_N*.txt). Reports
acc, parse-fail, per-gold-class and per-k accuracy; per-sample predictions CSV
(source,sample_dir,task,n_frames,k,gold,pred).

Usage:
  python scripts/redux/eval_tasks.py --eval-dirs-file outputs/redux/examdirs/exam_exists_N8.txt \
      [--eval-dirs-file ...] [--peft-adapter DIR] --output outputs/redux/c1_plain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "redux"))

from tasks import (  # noqa: E402
    build_prompt,
    derive_gold,
    evidence_count,
    parse_answer,
    target_of,
    task_of_dirs_file,
)

from gnnformer.data import build_prompt_inputs, load_mmred_sample, read_dirs_file  # noqa: E402
from gnnformer.runtime import load_runtime, move_to_device  # noqa: E402

SDPA = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dirs-file", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole file")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--peft-adapter", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    if args.peft_adapter:  # structural refs not needed here; wrap-last rule moot but kept
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.peft_adapter), is_trainable=False)
        model.eval()
        print(f"peft adapter loaded (frozen): {args.peft_adapter}", flush=True)

    out = args.output / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    srows = ["source,task,n,accuracy,parse_fail"]
    prows = ["source,sample_dir,task,n_frames,k,gold,pred"]

    @torch.inference_mode()
    def predict(frames, prompt):
        inp = move_to_device(build_prompt_inputs(processor, frames, prompt), rt.device)
        with sdpa_kernel(SDPA):
            g = model.generate(**inp, max_new_tokens=5, do_sample=False)
        return tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    for f in args.eval_dirs_file:
        task = task_of_dirs_file(f)
        dirs = read_dirs_file(Path(f))
        if args.limit:
            dirs = dirs[: args.limit]
        n = ok = pf = 0
        per_gold = defaultdict(lambda: [0, 0])
        per_k = defaultdict(lambda: [0, 0])
        t0 = time.time()
        for sd in dirs:
            try:
                _sid, frames, q0, states, _a0 = load_mmred_sample(sd)
                tr = target_of(sd, q0)
                if tr is None:
                    continue
                c, r = tr
                if args.resize > 0:
                    frames = [fr.resize((args.resize, args.resize)) for fr in frames]
            except Exception:
                continue
            nf = len(frames)
            k = evidence_count(states, c, r)
            gold = derive_gold(task, states, c, r, nf)
            pred = parse_answer(task, predict(frames, build_prompt(task, c, r, nf, q0)))
            n += 1
            hit = int(pred == gold)
            ok += hit
            pf += int(pred is None)
            per_gold[gold][1] += 1
            per_gold[gold][0] += hit
            per_k[k][1] += 1
            per_k[k][0] += hit
            prows.append(f"{f},{sd},{task},{nf},{k},{gold},{'' if pred is None else pred}")
            if n % 25 == 0:
                print(f"  {f}: {n}/{len(dirs)} acc {ok/max(1,n):.3f} "
                      f"{time.time()-t0:.0f}s", flush=True)
        gold_dist = Counter(g for g, (_c, _t) in
                            ((g, v) for g, v in per_gold.items()) )
        pg = " ".join(f"{g}:{c_}/{t_}" for g, (c_, t_) in sorted(per_gold.items()))
        pk = " ".join(f"k{k_}:{c_}/{t_}" for k_, (c_, t_) in sorted(per_k.items()))
        maj = max((t_ for _c, t_ in per_gold.values()), default=0) / max(1, n)
        print(f"CELL {f} task={task}: n={n} acc={ok/max(1,n):.4f} "
              f"pf={pf/max(1,n):.3f} maj-baseline={maj:.3f}\n  per-gold {pg}\n  per-k {pk}",
              flush=True)
        srows.append(f"{f},{task},{n},{ok/max(1,n):.4f},{pf/max(1,n):.3f}")
        del gold_dist
    (out / "summary.csv").write_text("\n".join(srows) + "\n")
    (out / "predictions.csv").write_text("\n".join(prows) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
