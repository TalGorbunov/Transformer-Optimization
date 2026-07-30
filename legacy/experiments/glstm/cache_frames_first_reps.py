#!/usr/bin/env python3
"""Re-extract per-frame L19 reps in the FRAMES-FIRST (deployed) layout: images, THEN the question
(base.build_prompt) -- so frames are query-BLIND (they precede the question), unlike the question-first
cache_minimal_frame_reps.py. Lets us check the aggregation decomposition replicates in the real layout.

Stores per example the same fields as cache_minimal: reps [N,H], query_rep [H] (last token), gold,
frame_labels (per-frame binary evidence), seq_len. Output file name mirrors minimal_L{layer}_{task}.pt.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

import experiments.glstm.frame_axis_aggregator_adapter as fa
from experiments.glstm.cache_minimal_frame_reps import frame_labels
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import image_token_groups


def build_inputs_frames_first(processor, frames, question, device):
    prompt = base.build_prompt(question, num_frames=len(frames))
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    return base.move_inputs_to_device(dict(inputs), device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="steps_in_room")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "message_sum" / "cache_framesfirst")
    return p.parse_args()


def main():
    args = parse_args()
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"minimal_L{args.read_layer}_{args.task}.pt"
    print(f"loading {args.model_name} ...", flush=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    for p_ in model.parameters():
        p_.requires_grad_(False)
    target = fa.get_layers(model)[int(args.read_layer)]

    st = {"spans": None, "cur_pos": -1, "frame_reps": None, "query_rep": None}

    def edit(hs):
        if st["spans"] is None or hs.shape[1] <= st["cur_pos"]:
            return hs
        st["frame_reps"] = torch.stack([hs[0, idx, :].float().mean(0).cpu() for idx in st["spans"]], 0)
        st["query_rep"] = hs[0, st["cur_pos"], :].float().cpu().clone()
        return hs

    def pre_hook(module, hargs, hkwargs):
        if len(hargs) >= 1:
            return (edit(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = edit(hkwargs["hidden_states"]); return hargs, hkwargs
    target.register_forward_pre_hook(pre_hook, with_kwargs=True)

    splits = fa.declare_splits(args.data_root, args.split, [args.seq_len], [], 0.0, 0.0, 0, None, args.split_seed)
    dirs = [d for d, _ in splits["train"]]
    if args.limit:
        dirs = dirs[:args.limit]
    import random
    rng = random.Random(0)
    cache = {}
    t0 = time.time()
    for i, dstr in enumerate(dirs):
        d = Path(dstr)
        ex = fa.make_example(d, args.task, rng, eval_mode=True)
        if ex is None:
            continue
        frames, question, gold, nf, states = ex
        try:
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        inputs = build_inputs_frames_first(processor, frames, question, device)
        ids = inputs["input_ids"]
        spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
        if len(spans) != len(frames):
            continue
        st["spans"] = spans; st["cur_pos"] = int(ids.shape[1]) - 1; st["frame_reps"] = None
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if st["frame_reps"] is None:
            continue
        cache[d.name] = {"reps": st["frame_reps"].half(), "query_rep": st["query_rep"].half(),
                         "gold": int(gold), "frame_labels": frame_labels(args.task, states, meta),
                         "task": args.task, "seq_len": int(nf)}
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(dirs)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(cache, out_path)
    golds = {}
    for v in cache.values():
        golds[v["gold"]] = golds.get(v["gold"], 0) + 1
    print(f"wrote {out_path}: {len(cache)} examples (FRAMES-FIRST); gold dist {dict(sorted(golds.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
