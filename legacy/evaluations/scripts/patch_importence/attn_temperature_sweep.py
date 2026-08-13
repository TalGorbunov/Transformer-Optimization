#!/usr/bin/env python3
"""Causal test of the softmax-DISPERSION hypothesis for the aggregation bottleneck.

Hypothesis: the last token can't cleanly aggregate N frames because softmax attention DISPERSES as
the number of frames grows (Velickovic et al., "Softmax is not Enough", 2410.01104) -> each frame's
message is averaged with ~1/N weight -> blurry, count-blind consolidation.

Test (training-free, frozen model): rescale the pre-softmax attention logits by a factor beta
(self_attn.scaling *= beta). beta>1 SHARPENS (lower temperature), beta<1 DIFFUSES. If SHARPENING
raises MMRED accuracy -- especially MORE at high seq_len -- dispersion is causal. If diffusing hurts,
that confirms the direction. If nothing moves, dispersion is NOT the bottleneck (-> over-squashing/readout).

Readout = deterministic digit-logit argmax over {0..8} at the answer position (no generation).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from collections import defaultdict
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import torch
import experiments.glstm.frame_axis_aggregator_adapter as fa
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import get_layers


def digit_token_ids(processor):
    tok = processor.tokenizer
    ids = []
    for d in range(9):
        cand = tok(str(d), add_special_tokens=False).input_ids
        ids.append(cand[0])
    return ids


def predict_count(model, inputs, digit_ids):
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1]  # next-token distribution at the answer position
    sub = torch.stack([logits[i] for i in digit_ids])
    return int(sub.argmax().item())


def set_beta(layers, lo, hi, beta, base_scales):
    for i, lyr in enumerate(layers):
        if lo <= i < hi:
            lyr.self_attn.scaling = base_scales[i] * beta
        else:
            lyr.self_attn.scaling = base_scales[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="steps_in_room")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    ap.add_argument("--per-seqlen", type=int, default=25)
    ap.add_argument("--seq-lens", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--betas", default="0.5,1.0,2.0,3.0,4.0")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()
    device = base.resolve_device("cuda"); dtype = base.resolve_dtype("bfloat16", device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    bands = {"all": (0, 999), "late": (14, 999)}

    print(f"loading {args.model_name} ...", flush=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, True)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = get_layers(model)
    base_scales = [float(l.self_attn.scaling) for l in layers]
    digit_ids = digit_token_ids(processor)
    print(f"n_layers={len(layers)} base_scale={base_scales[0]:.5f} digit_ids={digit_ids}", flush=True)

    # gather eval samples (frames, question, gold, seq_len), capped per seq_len
    import random
    rng = random.Random(0)
    samples = []
    for sl in seq_lens:
        splits = fa.declare_splits(args.data_root, "all_uniform", [sl], [], 0.0, 0.0, 0, None, 12345)
        dirs = [d for d, _ in splits["train"]]; rng.shuffle(dirs)
        got = 0
        for dstr in dirs:
            ex = fa.make_example(Path(dstr), args.task, rng, eval_mode=True)
            if ex is None:
                continue
            frames, question, gold, nf, states = ex
            try:
                inputs = fa.build_inputs(processor, frames, question, device)
            except Exception:
                continue
            samples.append((inputs, int(gold), int(nf)))
            got += 1
            if got >= args.per_seqlen:
                break
    print(f"collected {len(samples)} samples across seq_lens {seq_lens}", flush=True)

    # sweep
    results = {}  # (band, beta) -> {"overall": acc, "by_sl": {sl: acc}}
    for band_name, (lo, hi_raw) in bands.items():
        hi = min(hi_raw, len(layers))
        for beta in betas:
            if band_name == "all" and beta == 1.0 and ("all", 1.0) in results:
                continue
            set_beta(layers, lo, hi, beta, base_scales)
            correct = 0; by = defaultdict(lambda: [0, 0])
            t0 = time.time()
            for inputs, gold, sl in samples:
                pred = predict_count(model, inputs, digit_ids)
                ok = int(pred == gold); correct += ok
                by[sl][0] += ok; by[sl][1] += 1
            acc = correct / max(1, len(samples))
            by_sl = {sl: by[sl][0] / by[sl][1] for sl in sorted(by)}
            results[(band_name, beta)] = {"overall": acc, "by_sl": by_sl}
            print(f"[band={band_name:4s} beta={beta:.1f}] overall={acc:.3f}  "
                  f"by_seqlen={{ {', '.join(f'{k}:{v:.2f}' for k,v in by_sl.items())} }}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    set_beta(layers, 0, len(layers), 1.0, base_scales)  # restore

    # summary tables
    print("\n===== OVERALL ACC by (band, beta) =====", flush=True)
    print("band     " + "  ".join(f"b={b:.1f}" for b in betas), flush=True)
    for band_name in bands:
        row = [results.get((band_name, b), {}).get("overall") for b in betas]
        print(f"{band_name:8s} " + "  ".join(f"{(x if x is not None else float('nan')):.3f}" for x in row), flush=True)

    print("\n===== seq-len-stratified (band=late): does sharpening help MORE at high N? =====", flush=True)
    print("beta   " + "  ".join(f"sl{sl}" for sl in seq_lens), flush=True)
    for b in betas:
        d = results.get(("late", b), {}).get("by_sl", {})
        print(f"{b:.1f}    " + "  ".join(f"{d.get(sl, float('nan')):.2f}" for sl in seq_lens), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
