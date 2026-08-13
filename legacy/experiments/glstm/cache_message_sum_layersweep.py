#!/usr/bin/env python3
"""Cache per-frame reps at SEVERAL decoder layers in ONE joint forward per example, for the message-sum
decodability layer sweep (Stage 1, pushed across depth). Same object as cache_minimal_frame_reps.py
(per-frame mean-pooled vision reps + last-token rep + per-frame evidence labels + gold), but stored
per-layer so the CPU probe can ask "where across depth is the count decodable / where does the
evidence/non-evidence interference peak?".

Stores per example: reps_by_layer {L: [N,H] half}, query_by_layer {L: [H] half} (last-token rep),
gold, frame_labels (per-frame binary evidence for steps), seq_len.
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
from experiments.glstm.cache_frames_first_reps import build_inputs_frames_first
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import image_token_groups


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="steps_in_room")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--layers", default="13,15,17,19,21,23,25,27")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--frames-first", action="store_true",
                   help="deployed layout: images THEN question (frames query-blind, carriers exist)")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "message_sum" / "cache_layersweep")
    return p.parse_args()


def main():
    args = parse_args()
    layers_want = [int(x) for x in str(args.layers).replace(",", " ").split()]
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"layersweep_{args.task}.pt"
    print(f"loading {args.model_name} ...", flush=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    for p_ in model.parameters():
        p_.requires_grad_(False)
    all_layers = fa.get_layers(model)

    # shared state: spans + cur_pos set per example; each layer hook writes into st[layer]
    st = {"spans": None, "cur_pos": -1, "reps": {}, "query": {}}

    def make_hook(L):
        def edit(hs):
            if st["spans"] is None or hs.shape[1] <= st["cur_pos"]:
                return hs
            st["reps"][L] = torch.stack([hs[0, idx, :].float().mean(0).cpu() for idx in st["spans"]], 0)
            st["query"][L] = hs[0, st["cur_pos"], :].float().cpu().clone()
            return hs

        def pre_hook(module, hargs, hkwargs):
            if len(hargs) >= 1:
                return (edit(hargs[0]),) + tuple(hargs[1:]), hkwargs
            hkwargs = dict(hkwargs); hkwargs["hidden_states"] = edit(hkwargs["hidden_states"]); return hargs, hkwargs
        return pre_hook

    for L in layers_want:
        all_layers[L].register_forward_pre_hook(make_hook(L), with_kwargs=True)

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
        inputs = (build_inputs_frames_first(processor, frames, question, device)
                  if args.frames_first else fa.build_inputs(processor, frames, question, device))
        ids = inputs["input_ids"]
        spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
        if len(spans) != len(frames):
            continue
        st["spans"] = spans; st["cur_pos"] = int(ids.shape[1]) - 1
        st["reps"] = {}; st["query"] = {}
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if len(st["reps"]) != len(layers_want):
            continue
        cache[d.name] = {
            "reps_by_layer": {L: st["reps"][L].half() for L in layers_want},
            "query_by_layer": {L: st["query"][L].half() for L in layers_want},
            "gold": int(gold), "frame_labels": frame_labels(args.task, states, meta),
            "task": args.task, "seq_len": int(nf),
        }
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(dirs)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(cache, out_path)
    golds = {}
    for v in cache.values():
        golds[v["gold"]] = golds.get(v["gold"], 0) + 1
    print(f"wrote {out_path}: {len(cache)} examples; layers {layers_want}; gold dist {dict(sorted(golds.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
