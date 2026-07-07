#!/usr/bin/env python3
"""Cache per-frame L19 reps for a minimal-crowding MMRED task — ONCE — so all readout/aggregator
experiments (sum / deepsets / logic / classifier, IID & OOD count-holdout) become fast CPU fits.

Since we DON'T inject back into the LM (we read the count from the head directly), the frozen model is
used only to produce per-frame reps, which never change -> cache them.

For each example dir we store: per-frame mean-pooled L19 vision-token reps [N,H] (question-conditioned,
since the question precedes the frames), the query-position rep, the gold count, and per-frame labels
(binary evidence for steps/co-occ; room-of-queried-char for rooms -> enables the soft-OR readout).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

import experiments.glstm.frame_axis_aggregator_adapter as fa
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from models.model import image_token_groups


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["steps_in_room","rooms_visited","co_occupancy","distinct_visitors","distinct_companions"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--limit", type=int, default=0, help="0 = all dirs")
    p.add_argument("--multipass", action="store_true",
                   help="cache per-frame reps from N focused SINGLE-frame forwards (relieves vision crowding) "
                        "instead of one joint forward")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "cache")
    return p.parse_args()


def frame_labels(task, states, meta):
    """per-frame label: binary evidence (steps/cooc) or room-of-queried-char (rooms)."""
    if task == "steps_in_room":
        C, R = meta.get("target_character"), meta.get("target_room")
        return [int(tf.room_of(s, C) == R) for s in states] if C and R else None
    if task == "rooms_visited":
        C = meta.get("query_character")
        return [tf.room_of(s, C) for s in states] if C else None
    if task == "distinct_visitors":  # per-frame: chars present in the queried room R
        R = meta.get("query_room") or meta.get("target_room")
        return [sorted(s["rooms"].get(R, [])) for s in states] if R else None
    if task == "distinct_companions":  # per-frame: other chars in C's room
        C = meta.get("query_character") or meta.get("target_character")
        if not C:
            return None
        out = []
        for s in states:
            cr = tf.room_of(s, C)
            out.append(sorted(x for x in (s["rooms"].get(cr, []) if cr != "not present" else []) if x != C))
        return out
    qp = meta.get("query_pair")
    if qp and len(qp) == 2:
        C, D = qp
        return [int(tf.room_of(s, C) == tf.room_of(s, D)) for s in states]
    return None


def main():
    args = parse_args()
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"minimal_L{args.read_layer}_{args.task}{'_multipass' if getattr(args,'multipass',False) else ''}.pt"
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
        if getattr(args, "multipass", False):
            # MULTI-PASS: one focused single-frame forward per frame (query + that frame alone) ->
            # clean per-frame rep (~0.99, the model resolves entities); relieves the joint-pass crowding.
            fr_reps = []; qrep = None; ok = True
            for fi in range(len(frames)):
                si = fa.build_inputs(processor, [frames[fi]], question, device)
                sp = image_token_groups(si["input_ids"][0].detach().cpu(), 1, processor=processor)
                if len(sp) != 1:
                    ok = False; break
                st["spans"] = sp; st["cur_pos"] = int(si["input_ids"].shape[1]) - 1; st["frame_reps"] = None
                with torch.inference_mode():
                    model(**si, use_cache=False)
                if st["frame_reps"] is None:
                    ok = False; break
                fr_reps.append(st["frame_reps"][0]); qrep = qrep if qrep is not None else st["query_rep"]
            if not ok:
                continue
            st["frame_reps"] = torch.stack(fr_reps, 0); st["query_rep"] = qrep
        else:
            inputs = fa.build_inputs(processor, frames, question, device)
            ids = inputs["input_ids"]
            spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
            if len(spans) != len(frames):
                continue
            st["spans"] = spans; st["cur_pos"] = int(ids.shape[1]) - 1
            st["frame_reps"] = None
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
    print(f"wrote {out_path}: {len(cache)} examples; gold dist {dict(sorted(golds.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
