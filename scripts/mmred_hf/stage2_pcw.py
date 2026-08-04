#!/usr/bin/env python3
"""Stage 2: PCW message decodability — joint pass, fenced window per frame, replica =
the PER-FRAME question; capture each frame's MESSAGE to its replica's loci and probe.

Per sample: [q_pf][frame_1+q_pf]...[frame_N+q_pf] with blockfence + posreset (A3).
Features per frame x {L12, L16} x {last-token locus, span-mean of messages}.
Probes (5 seeds, 50/50) per task like stage 1; compare to the alone-pass ceiling.

Usage: python scripts/mmred_hf/stage2_pcw.py --task occofr --limit 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/mmred_hf"))

from stage1_alone import TASKS, build  # noqa: E402
from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.fencing import (  # noqa: E402
    FenceHooks, build_replica_probe_mask, find_question_spans, frame_blocks,
    recompute_messages, reset_positions,
)
from gnnformer.mmred_hf import ROOM_ORDER, _match  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    attention_dims, dequantize_linear_weight, get_layers, get_rope_index_fn,
    image_token_groups, load_runtime, move_to_device,
)


def pf_question(task, g, t):
    """The per-frame question text (constant across a sample's windows)."""
    if task == "roomofc":
        return f"In which room is {g[0]}? Answer with the room name only."
    if task == "occofr":
        return f"Who is in the {g[0]}? Answer with the name, or Nobody."
    if task == "trig":
        return f"Is {g[1]} in the {g[2]}? Answer Yes or No."
    if task == "empty":
        return None  # per-sample rotation handled by caller
    raise ValueError(task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--layers", default="12,16")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--output", default="outputs/mmred_hf/stage2")
    args = ap.parse_args()

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.split(",")]
    dims = attention_dims(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    w_o = {L: dequantize_linear_weight(layers[L].self_attn.o_proj) for L in Ls}
    hooks = FenceHooks(layers, capture_layers=Ls).install()

    root_name, qtype = TASKS[args.task]
    dirs = sorted((_REPO / f"data/mmred_hf/dirs/{root_name}").iterdir())[: args.limit]

    feats = {(L, v): [] for L in Ls for v in ("last", "mean")}
    ys = []
    n_done = n_skip = 0
    t0 = time.time()
    for si, sd in enumerate(dirs):
        try:
            _sid, frames, q0, states, _a = load_mmred_sample(sd)
            g = _match(qtype, q0)
        except Exception:
            n_skip += 1
            continue
        if args.task == "empty":
            room = ROOM_ORDER[si % 6]
            q_pf = f"Is the {room} empty? Answer Yes or No."
            g_eff = g
        else:
            q_pf = pf_question(args.task, g, 0)
        NF = len(frames)
        if args.resize:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q_pf}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q_pf})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, rt.device)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q_pf, NF + 1)
        if len(fg) != NF or spans is None:
            n_skip += 1
            continue
        rep_spans = spans[1:] if len(spans) == NF + 1 else spans[:NF]
        vis = [torch.tensor(sorted(int(p) for p in gset), dtype=torch.long) for gset in fg]
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(vstarts) != NF:
            n_skip += 1
            continue
        blocks = frame_blocks(vstarts, rep_spans[-1][1])
        m = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                     fence_blocks=True, blocks=blocks)
        hooks.set_mask(m, model.device)
        with torch.inference_mode():
            base_pos, _ = rope_fn(inputs["input_ids"],
                                  image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos_ids = reset_positions(base_pos, blocks, rep_spans[-1][1])
            model(**inputs, position_ids=pos_ids)
        hooks.clear_mask()

        # labels per frame (reuse stage-1 build; frame index drives 'empty' room too)
        labs = []
        keep = []
        for t in range(NF):
            gg = g if args.task != "empty" else g
            b = build(args.task, gg, states, t) if args.task != "empty" else None
            if args.task == "empty":
                from gnnformer.mmred_hf import _rooms
                labs.append(int(not _rooms(states[t]).get(room, [])))
                keep.append(t)
            elif b is not None:
                labs.append(b[2])
                keep.append(t)
            else:
                labs.append(None)

        last_c = [sp[1] - 1 for sp in rep_spans]
        for L in Ls:
            common = dict(seq=seq, cos=hooks.cos, sin=hooks.sin, dims=dims, w_o=w_o[L],
                          q_proj=hooks.qkv[L]["q_proj"], k_proj=hooks.qkv[L]["k_proj"],
                          v_proj=hooks.qkv[L]["v_proj"], mask_full=m, vis_by_frame=vis)
            m_last = recompute_messages(carrier_positions=last_c, **common)
            span_positions = [list(range(sp[0], sp[1])) for sp in rep_spans]
            flat = [p for sp in span_positions for p in sp]
            m_flat = recompute_messages(
                carrier_positions=flat, **{**common, "vis_by_frame":
                    [vis[i] for i, sp in enumerate(span_positions) for _ in sp]})
            # regroup span means
            out_mean, k = [], 0
            for sp in span_positions:
                out_mean.append(m_flat[k:k + len(sp)].mean(0))
                k += len(sp)
            for t in keep:
                feats[(L, "last")].append(np.asarray(m_last[t], dtype=np.float32))
                feats[(L, "mean")].append(np.asarray(out_mean[t], dtype=np.float32))
        ys += [labs[t] for t in keep]
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done} samples ({n_skip} skip) {time.time()-t0:.0f}s", flush=True)

    hooks.remove()
    y = np.array(ys)
    from sklearn.linear_model import LogisticRegression
    lines = [f"STAGE2 PCW [{args.task}] n_frames={len(y)} (samples={n_done})"]
    for (L, v), F in feats.items():
        X = np.stack(F).astype(np.float32)
        accs = []
        for seed in range(5):
            idx = np.random.default_rng(seed).permutation(len(y))
            h = len(y) // 2
            clf = LogisticRegression(max_iter=2000).fit(X[idx[:h]], y[idx[:h]])
            accs.append(float(clf.score(X[idx[h:]], y[idx[h:]])))
        maj = float(np.bincount(y).max()) / len(y)
        lines.append(f"  L{L} {v:5s}: acc {np.mean(accs):.3f}±{np.std(accs):.3f} (maj {maj:.3f})")
        print(lines[-1], flush=True)
    out = _REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
