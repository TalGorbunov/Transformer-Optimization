#!/usr/bin/env python3
"""Position-symmetry check across fenced blocks — ALL THREE M-RoPE channels.

The canary discriminator (2026-08-10) found real, deterministic answer-logit
sensitivity to content-preserving frame swaps (T1 0.586, T2 0.621, T0 identity
0.000). Two candidate causes remain; this settles the structural one:

  Every posreset verification so far printed CHANNEL 0 ONLY. M-RoPE has three
  channels (t, h, w); reset_positions subtracts one scalar delta (computed from
  channel 0 at the block-start token) from all channels. If the per-block channel
  offsets are not uniform, blocks are NOT position-identical in h/w, frame order
  enters through RoPE phase, and the fence's order-invariance is broken — the
  h/w sibling of the Phase 0 node-position leak.

Prints, for one HF N=8 sample laid out EXACTLY as the trainer (block fence +
reset_positions + node posreset + canonical tail):
  per block i>0, per channel c: max |pos[c, block_i] - pos[c, block_0]|
  and the same for the replica spans and node spans.
All zeros = position symmetry holds (the leak story falls to FP-chaos, and the
canary's meaningful invariant becomes answer stability, not logit bits).
Nonzero = the leak, located.

Usage: python scripts/ninv/pos_symmetry_check.py --output outputs/ninv/<ts>_possym
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import evidence_bits, iter_hf_sample_dirs, load_hf_sample  # noqa: E402
from train_registers import canonical_positions, tree_levels_b2  # noqa: E402

from gnnformer.fencing import find_question_spans, frame_blocks  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/mmred_hf/dirs/seq_len_8_train_steps_in_room")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dev = model.device

    lines = []
    n_done = 0
    worst = 0
    for sd in iter_hf_sample_dirs(Path(args.root)):
        if n_done >= args.n_samples:
            break
        _sid, frames, q0, states, a0 = load_hf_sample(sd, resize=args.resize)
        if evidence_bits(q0, states) is None:
            continue
        NF = len(frames)
        levels = tree_levels_b2(NF)
        n_nodes = sum(len(g) for g in levels)
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content += [{"type": "image", "image": f}, {"type": "text", "text": q0}]
        content += [{"type": "text", "text": q0}] * n_nodes
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        fg = image_token_groups(inputs["input_ids"][0], expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            continue
        rep = spans[1 : NF + 1]
        sq = spans[NF + 1 :]
        blocks = frame_blocks(vstarts, sq[0][0])
        mv = move_to_device(inputs, dev)
        with torch.no_grad():
            base_pos, _ = rope_fn(mv["input_ids"],
                                  image_grid_thw=mv.get("image_grid_thw"),
                                  attention_mask=mv.get("attention_mask"))
        pos, _, _ = canonical_positions(base_pos.cpu(), blocks, sq, sq[-1],
                                        len(ids), 0)
        print(f"\n=== {sd.name}  seq={len(ids)}  blocks={blocks[:2]}... ===")
        b0 = blocks[0]
        w0 = pos[:, 0, b0[0]:b0[1]]
        for bi, (a, b) in enumerate(blocks[1:], start=1):
            if b - a != b0[1] - b0[0]:
                print(f"  block {bi}: LENGTH MISMATCH {b-a} vs {b0[1]-b0[0]}")
                worst = max(worst, 10 ** 9)
                continue
            d = (pos[:, 0, a:b] - w0).abs()
            per_ch = [int(d[c].max()) for c in range(3)]
            flag = "" if max(per_ch) == 0 else "   <-- ASYMMETRY"
            print(f"  block {bi} vs 0: max|dpos| t={per_ch[0]} h={per_ch[1]} "
                  f"w={per_ch[2]}{flag}")
            lines.append(f"{sd.name} block{bi} {per_ch}")
            worst = max(worst, max(per_ch))
        r0 = rep[0]
        w0r = pos[:, 0, r0[0]:r0[1]]
        for ri, (a, b) in enumerate(rep[1:], start=1):
            d = (pos[:, 0, a:b] - w0r).abs()
            per_ch = [int(d[c].max()) for c in range(3)]
            if max(per_ch):
                print(f"  replica {ri} vs 0: max|dpos| {per_ch}   <-- ASYMMETRY")
                worst = max(worst, max(per_ch))
        s0 = sq[0]
        w0s = pos[:, 0, s0[0]:s0[1]]
        for si, (a, b) in enumerate(sq[1:], start=1):
            d = (pos[:, 0, a:b] - w0s).abs()
            per_ch = [int(d[c].max()) for c in range(3)]
            if max(per_ch):
                print(f"  node {si} vs 0: max|dpos| {per_ch}   <-- ASYMMETRY")
                worst = max(worst, max(per_ch))
        n_done += 1
    verdict = ("POSITION ASYMMETRY FOUND — the order-sensitivity has a structural "
               "cause; the fence's order-invariance is broken at the RoPE level"
               if worst else
               "ALL CHANNELS SYMMETRIC across blocks/replicas/nodes — the "
               "deterministic T1/T2 sensitivity is NOT positional; remaining "
               "explanation is bf16 summation-order chaos amplified over 28 "
               "layers, under which the meaningful canary is ANSWER stability")
    print(f"\nVERDICT: {verdict}")
    (out / "report.txt").write_text("\n".join(lines + [f"worst={worst}", verdict]) + "\n")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
