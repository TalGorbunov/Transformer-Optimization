#!/usr/bin/env python3
"""Supply probe: per-frame question replicas behind the attention fence — measures the
one-forward supply d' (replica read) against the in-run JOINT anchor (final-question
read at the ANCHOR_OFFSET locus).

Anchor this script must reproduce (RESULTS.md [2026-07-18] E1): Q-first blockfence +
posreset at n=900, N=8 -> replica d' 13.54 +/- 0.27 (joint anchor 5.95); at n=300 the
A3 band is ~9.2-10.9. Also the source of the [2026-07-27] posreset dose-response.

Arms (the ablation ladder):
  --no-mask                       unmasked prompt-engineering control
  (default)                       replicas masked-invisible, frames shared
  --fence-frames                  + frame rows isolated
  --fence-frames --fence-blocks   + full block fence (A3; closes the marker leak)
  --reset-positions               + per-block M-RoPE reset
  --question-first                + shared question prefix (the deployed layout)

Usage:
  python scripts/probe_supply.py --data_root data/mmred_images_park/seq_len_8/all_uniform \
      --limit 300 --shuffle-dirs 0 --question-first --fence-frames --fence-blocks \
      --reset-positions --output outputs/carrier/probe
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import ANCHOR_OFFSET, ROOMS
from gnnformer.data import (
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    load_natural_sample,
    probe_evidence,
)
from gnnformer.fencing import (
    FenceHooks,
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    locate_word_token,
    reset_positions,
)
from gnnformer.metrics import dprime_pair, format_gold_histogram
from gnnformer.mmred_hf import probe_evidence_mmred
from gnnformer.runtime import (
    attention_dims,
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)
from gnnformer.fencing import recompute_messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", choices=("steps", "cooc", "mmred_niah"), default="steps",
                    help="mmred_niah: MMReD-HF materialized dirs, qtype from the dir-name "
                         "prefix, evidence via gnnformer.mmred_hf (gold may be non-numeric)")
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--fence-frames", action="store_true")
    ap.add_argument("--fence-blocks", action="store_true")
    ap.add_argument("--reset-positions", action="store_true")
    ap.add_argument("--question-first", action="store_true")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/carrier/probe")
    args = ap.parse_args()
    if args.reset_positions and (args.no_mask or not args.fence_frames):
        ap.error("--reset-positions requires --fence-frames")
    if args.fence_blocks and (args.no_mask or not args.fence_frames):
        ap.error("--fence-blocks requires --fence-frames")

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.split(",")]
    dims = attention_dims(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    w_o = {L: dequantize_linear_weight(layers[L].self_attn.o_proj) for L in Ls}

    hooks = FenceHooks(layers, capture_layers=Ls).install()
    feats_rep = {L: [] for L in Ls}
    feats_anc = {L: [] for L in Ls}
    labels_all, gold_all = [], []
    n_done = n_skip = 0

    if args.natural:
        dirs = sorted(d for d in Path(args.data_root).iterdir() if d.is_dir())
    elif args.shuffle_dirs is not None:
        dirs = iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs)
    else:
        dirs = iter_sample_dirs(Path(args.data_root))

    for sd in dirs:
        if n_done >= args.limit:
            break
        try:
            if args.natural:
                frames, q0, gold, evid, room = load_natural_sample(sd)
            else:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
                if args.task == "mmred_niah":
                    mq = re.match(r"(.+?)_(?:K\d+|A[A-Za-z]+)_\d+$", sd.name)
                    if mq is None:
                        n_skip += 1
                        continue
                    pe_ = probe_evidence_mmred(mq.group(1), q0, states)
                    if pe_ is None:
                        n_skip += 1
                        continue
                    evid, room = pe_
                    gold = len(evid)  # evidence count stands in for the histogram
                else:
                    gold = int(str(a0).strip())
                    pe_ = probe_evidence(args.task, q0, states, gold, ROOMS)
                    if pe_ is None:
                        n_skip += 1
                        continue
                    evid, room = pe_
        except Exception:
            n_skip += 1
            continue
        if not evid and gold != 0:
            n_skip += 1
            continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]

        content = []
        if args.question_first:
            content.append({"type": "text", "text": q0})
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, model.device)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        if len(fg) != NF:
            n_skip += 1
            continue
        exp_occ = NF + 2 if args.question_first else NF + 1
        spans = find_question_spans(ids, tok, q0, exp_occ)
        if spans is None:
            n_skip += 1
            continue
        if args.question_first:
            spans = spans[1:]
        rep_spans, fin_span = spans[:NF], spans[NF]
        rep_c = [locate_word_token(ids, tok, room, sp) for sp in rep_spans]
        if any(c is None for c in rep_c):
            n_skip += 1
            continue
        anc_c = seq - 1 - ANCHOR_OFFSET
        vis_by_frame = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]

        blocks = None
        if args.reset_positions or args.fence_blocks:
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            if len(vstarts) != NF:
                n_skip += 1
                continue
            blocks = frame_blocks(vstarts, fin_span[0])

        if args.no_mask:
            m = build_replica_probe_mask(seq, [], [torch.tensor([], dtype=torch.long)])
            hooks.clear_mask()  # plain causal forward; m used only for recompute rows
        else:
            m = build_replica_probe_mask(seq, rep_spans, vis_by_frame,
                                         fence_frames=args.fence_frames,
                                         fence_blocks=args.fence_blocks, blocks=blocks)
            hooks.set_mask(m, model.device)
        if n_done == 0 and not args.no_mask:
            alw = lambda r: int((m[r] == 0).sum())  # noqa: E731
            print(f"[mask-debug] seq={seq} allowed-keys: frame0 {alw(int(vis_by_frame[0][0]))} "
                  f"replica0 {alw(rep_spans[0][0])} replicaLast {alw(rep_spans[-1][0])} "
                  f"final-q {alw(anc_c)}", flush=True)

        pos_ids = None
        if args.reset_positions:
            with torch.inference_mode():
                base_pos, _ = rope_fn(inputs["input_ids"],
                                      image_grid_thw=inputs.get("image_grid_thw"),
                                      attention_mask=inputs.get("attention_mask"))
            pos_ids = reset_positions(base_pos, blocks, fin_span[0])
            if n_done == 0:
                s0, e0 = blocks[0]
                same_len = all(b - a == e0 - s0 for a, b in blocks)
                blocks_eq = same_len and all(
                    torch.equal(pos_ids[:, :, a:b], pos_ids[:, :, s0:e0]) for a, b in blocks[1:])
                print(f"[pos-debug] seq={seq} max_pos {int(base_pos.max())} -> "
                      f"{int(pos_ids.max())}, blocks_identical={blocks_eq}", flush=True)

        with torch.inference_mode():
            model(**inputs, position_ids=pos_ids) if pos_ids is not None else model(**inputs)
        hooks.clear_mask()

        for L in Ls:
            common = dict(seq=seq, cos=hooks.cos, sin=hooks.sin, dims=dims, w_o=w_o[L],
                          q_proj=hooks.qkv[L]["q_proj"], k_proj=hooks.qkv[L]["k_proj"],
                          v_proj=hooks.qkv[L]["v_proj"], mask_full=m,
                          vis_by_frame=vis_by_frame)
            rep_msgs = recompute_messages(carrier_positions=rep_c, **common)
            anc_msgs = recompute_messages(carrier_positions=[anc_c] * NF, **common)
            feats_rep[L].append(rep_msgs)
            feats_anc[L].append(anc_msgs)

        labels_all.append([1 if t in evid else 0 for t in range(NF)])
        gold_all.append(gold)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done} samples (skip {n_skip})", flush=True)

    hooks.remove()
    y = np.array(labels_all)
    print("[gold-hist] " + format_gold_histogram(gold_all), flush=True)
    variant = ("fenced" if args.fence_frames else "masked") if not args.no_mask else "unmasked"
    variant += ("+blockfence" if args.fence_blocks else "")
    variant += ("+posreset" if args.reset_positions else "")
    variant += ("+qfirst" if args.question_first else "")
    lines = [f"=== SUPPLY PROBE (n={n_done}, skip={n_skip}, variant={variant}, "
             f"data={args.data_root}) ==="]
    cache = {"labels": y, "gold": np.array(gold_all), "rep": {}, "anc": {}}
    for L in Ls:
        Xr = np.stack(feats_rep[L])
        Xa = np.stack(feats_anc[L])
        cache["rep"][L], cache["anc"][L] = Xr, Xa
        dr, sr, ar = dprime_pair(Xr, y)
        da, sa, aa = dprime_pair(Xa, y)
        lines.append(f"L{L}: REPLICA read d'={dr:.2f}±{sr:.2f} (auc {ar:.2f})   "
                     f"JOINT anchor d'={da:.2f}±{sa:.2f} (auc {aa:.2f})   "
                     f"ratio {dr/max(da,1e-9):.2f}x")
        per = []
        for i in range(Xr.shape[1]):
            try:
                di, _, _ = dprime_pair(Xr[:, [i], :], y[:, [i]])
                per.append(f"{di:.2f}")
            except Exception:
                per.append("--")
        lines.append(f"L{L}: per-copy d' (index 0..{Xr.shape[1]-1}): " + " ".join(per))
    torch.save(cache, out / "messages_cache.pt")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
