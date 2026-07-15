#!/usr/bin/env python3
"""Q-first behavioral eval: the model's emitted answer under the QUESTION-FIRST layout
(fa.build_inputs — the same layout the Q-first probe caches used) for the parity-table regimes:
ns2/ns4/ns6 (park seq_len 2/4/6), crowd8 (park seq_len_8), coocBal (mmred_cooc_balanced seq_len_8).
Fills the 'model acc' column of the theory-background parity Table 2 (those regimes were probe-only)."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

import experiments.glstm.frame_axis_aggregator_adapter as fa
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base

REGIMES = [
    ("ns2", "steps_in_room", "data/mmred_images_park", 2),
    ("ns4", "steps_in_room", "data/mmred_images_park", 4),
    ("ns6", "steps_in_room", "data/mmred_images_park", 6),
    ("crowd8", "steps_in_room", "data/mmred_images_park", 8),
    ("coocBal", "co_occupancy", "data/mmred_cooc_balanced", 8),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", default="outputs/frame_axis/probes/qfirst_behavior")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    device = base.resolve_device("cuda")
    dtype = base.resolve_dtype("bfloat16", device)
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, True)
    pad = processor.tokenizer.pad_token_id
    if pad is None:
        pad = processor.tokenizer.eos_token_id

    lines = []
    for name, task, root, seq in REGIMES:
        splits = fa.declare_splits(Path(root), "all_uniform", [seq], [], 0.0, 0.0, 0, None, 12345)
        dirs = [Path(d) for d, _ in splits["train"]][: args.limit]
        rng = random.Random(0)
        n_ok = n_tot = 0
        preds = []
        t0 = time.time()
        for d in dirs:
            ex = fa.make_example(d, task, rng, eval_mode=True)
            if ex is None:
                continue
            frames, question, gold, nf, states = ex
            # same answer-format instruction as the frames-first behavioral protocol
            # (base.build_prompt), kept in the Q-first position so only layout differs
            q_instr = (f"Respond with a single integer from 0 to {len(frames)} (0 is allowed). "
                       f"Output only the integer.\n{question}\nAnswer with only the integer.")
            inputs = fa.build_inputs(processor, frames, q_instr, device)
            plen = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                oids = model.generate(**inputs, do_sample=False,
                                      max_new_tokens=args.max_new_tokens, pad_token_id=pad)
            txt = processor.batch_decode(oids[:, plen:], skip_special_tokens=True)[0]
            p = base.extract_first_integer(txt)
            preds.append({"dir": str(d), "gold": int(gold), "pred": p, "raw": txt})
            n_tot += 1
            n_ok += int(p == gold)
        acc = n_ok / max(1, n_tot)
        lines.append(f"{name:<8s} task={task:<14s} N={seq}  acc={acc:.3f}  n={n_tot}  "
                     f"({time.time()-t0:.0f}s)")
        print(lines[-1], flush=True)
        (out / f"{name}_preds.json").write_text(json.dumps(preds))
        (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
