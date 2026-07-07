#!/usr/bin/env python3
"""TARGETED softmax-dispersion test: temperature ONLY on the (question-token query -> frame-token key)
attention block. This isolates the exact aggregation step in the "frame -> carrier question token ->
last token" flow, instead of the blunt all-attention sweep.

Prompt is FRAMES-FIRST so the question tokens sit AFTER the frames and can causally attend to them
(question-first would make question->frame attention a no-op under the causal mask).

Intervention: a custom attention fn mirrors eager attention but multiplies the logit sub-block
[query in QUESTION positions, key in FRAME positions] by beta before softmax (beta>1 sharpens). Applied
only in a layer band. If sharpening THIS block raises MMRED accuracy (more at high seq_len), the
frame->carrier aggregation attention is dispersing and that is causal.

Readout = digit-logit argmax over {0..8} at the answer position (no generation).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from collections import defaultdict
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn
import experiments.glstm.frame_axis_aggregator_adapter as fa
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import get_layers, image_token_groups
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as mq
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

STATE = {"beta": 1.0, "lo": 0, "hi": 999, "q_pos": None, "k_pos": None}


def temp_frame_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = mq.repeat_kv(key, module.num_key_value_groups)
    value_states = mq.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    li = getattr(module, "layer_idx", -1)
    if (STATE["beta"] != 1.0 and STATE["lo"] <= li < STATE["hi"]
            and STATE["q_pos"] is not None and attn_weights.shape[-1] == attn_weights.shape[-2]):
        qp, kp = STATE["q_pos"], STATE["k_pos"]
        if len(qp) and len(kp):
            attn_weights[:, :, qp.unsqueeze(1), kp.unsqueeze(0)] *= STATE["beta"]
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def build_inputs_frames_first(processor, frames, question, device):
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": f"Question: {question}"})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    return base.move_inputs_to_device(dict(inputs), device)


def frame_and_question_positions(processor, input_ids, n_frames):
    ids_cpu = input_ids[0].detach().cpu()
    spans = image_token_groups(ids_cpu, n_frames, processor=processor)
    if len(spans) != n_frames:
        return None, None
    frame_pos = sorted(int(i) for sp in spans for i in sp)
    last_img = max(frame_pos)
    special = set(processor.tokenizer.all_special_ids)
    seq_len = int(input_ids.shape[1])
    # question tokens = post-frame text tokens (drop chat-template special tokens like im_end/assistant)
    q_pos = [i for i in range(last_img + 1, seq_len)
             if int(ids_cpu[i]) not in special]
    return frame_pos, q_pos


def digit_token_ids(processor):
    return [processor.tokenizer(str(d), add_special_tokens=False).input_ids[0] for d in range(9)]


def predict_count(model, inputs, digit_ids):
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1]
    return int(torch.stack([logits[i] for i in digit_ids]).argmax().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="steps_in_room")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    ap.add_argument("--per-seqlen", type=int, default=25)
    ap.add_argument("--seq-lens", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--betas", default="0.5,1.0,2.0,3.0,4.0")
    ap.add_argument("--bands", default="all,late")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()
    device = base.resolve_device("cuda"); dtype = base.resolve_dtype("bfloat16", device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    print(f"loading {args.model_name} ...", flush=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, True)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = get_layers(model)
    n_layers = len(layers)
    band_defs = {"all": (0, n_layers), "late": (14, n_layers)}
    bands = {b: band_defs[b] for b in args.bands.split(",")}
    # register the custom attention everywhere (it's a no-op when beta==1.0)
    ALL_ATTENTION_FUNCTIONS["temp_frame"] = temp_frame_attention
    model.config._attn_implementation = "temp_frame"
    for l in layers:
        l.self_attn.config._attn_implementation = "temp_frame"
    digit_ids = digit_token_ids(processor)
    print(f"n_layers={n_layers} digit_ids={digit_ids}", flush=True)

    import random
    rng = random.Random(0)
    samples = []  # (inputs, gold, seq_len, q_pos_tensor, k_pos_tensor)
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
                inputs = build_inputs_frames_first(processor, frames, question, device)
            except Exception:
                continue
            fp, qp = frame_and_question_positions(processor, inputs["input_ids"], nf)
            if fp is None or not qp:
                continue
            samples.append((inputs, int(gold), int(nf),
                            torch.tensor(qp, device=device), torch.tensor(fp, device=device)))
            got += 1
            if got >= args.per_seqlen:
                break
    print(f"collected {len(samples)} samples; e.g. n_qtok={len(samples[0][3])} n_frametok={len(samples[0][4])}", flush=True)

    results = {}
    for band_name, (lo, hi) in bands.items():
        for beta in betas:
            if beta == 1.0 and ("__base__", 1.0) in results:
                results[(band_name, 1.0)] = results[("__base__", 1.0)]; continue
            STATE["lo"], STATE["hi"], STATE["beta"] = lo, hi, beta
            correct = 0; by = defaultdict(lambda: [0, 0]); t0 = time.time()
            for inputs, gold, sl, qp, kp in samples:
                STATE["q_pos"], STATE["k_pos"] = qp, kp
                pred = predict_count(model, inputs, digit_ids)
                ok = int(pred == gold); correct += ok
                by[sl][0] += ok; by[sl][1] += 1
            acc = correct / max(1, len(samples))
            by_sl = {sl: by[sl][0] / by[sl][1] for sl in sorted(by)}
            results[(band_name, beta)] = {"overall": acc, "by_sl": by_sl}
            if beta == 1.0:
                results[("__base__", 1.0)] = results[(band_name, beta)]
            print(f"[band={band_name:4s} beta={beta:.1f}] overall={acc:.3f}  "
                  f"by_seqlen={{ {', '.join(f'{k}:{v:.2f}' for k,v in by_sl.items())} }}  ({time.time()-t0:.0f}s)", flush=True)
    STATE["beta"] = 1.0

    print("\n===== OVERALL ACC by (band, beta) =====", flush=True)
    print("band     " + "  ".join(f"b={b:.1f}" for b in betas), flush=True)
    for band_name in bands:
        row = [results.get((band_name, b), {}).get("overall") for b in betas]
        print(f"{band_name:8s} " + "  ".join(f"{(x if x is not None else float('nan')):.3f}" for x in row), flush=True)
    print("\n===== seq-len stratified (band=all): sharpening help MORE at high N? =====", flush=True)
    print("beta   " + "  ".join(f"sl{sl}" for sl in seq_lens), flush=True)
    for b in betas:
        d = results.get(("all", b), {}).get("by_sl", {})
        print(f"{b:.1f}    " + "  ".join(f"{d.get(sl, float('nan')):.2f}" for sl in seq_lens), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
