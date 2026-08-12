#!/usr/bin/env python3
"""Attention-map figure: the graph rewiring, seen directly.

Same Q-first replica layout, two arms (plain causal vs full block fence + posreset).
At layer --layer the head-averaged attention matrix is computed from the captured
q/k projections, averaged into prompt segments (prefix+Q, frame_i, replica_i,
final Q), averaged over --limit samples, and rendered as two log-scale heatmaps:
joint = dense cross-frame mass; fenced = block-diagonal islands + question column.

Backs the logged "attention is block-sparse" claim (RESULTS.md [2026-07-17], pt 4).

Usage:
  python scripts/probe_attention_map.py --limit 8 --layer 16 \
      --output outputs/presentation/attnmap/<ts>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from gnnformer.constants import ROOMS
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence
from gnnformer.fencing import (
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    reset_positions,
)
from gnnformer.runtime import (
    attention_dims,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    L = args.layer
    dims = attention_dims(model)
    n_heads, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    acc = {"joint": None, "fenced": None}
    n_done = 0
    NF_ref = None
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence(args.task, q0, states, gold, ROOMS)
            if pe_ is None:
                continue
        except Exception:
            continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, dev)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 2)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            continue
        spans = spans[1:]
        rep_spans, fin_span = spans[:NF], spans[NF]
        blocks = frame_blocks(vstarts, fin_span[0])
        vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]

        # segments: [prefix+Q] then per frame [vision_f][replica_f] then [final]
        segs, labels = [(0, blocks[0][0])], ["Q"]
        for i, ((a, b), (ra, rb)) in enumerate(zip(blocks, rep_spans)):
            segs.append((int(vis[i][0]), int(vis[i][-1]) + 1))
            labels.append(f"f{i+1}")
            segs.append((ra, rb))
            labels.append(f"c{i+1}")
        segs.append((fin_span[0], seq))
        labels.append("fin")
        if NF_ref is None:
            NF_ref = NF
        elif NF != NF_ref:
            continue

        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos_reset = reset_positions(base_pos, blocks, fin_span[0]).clone()
            emb = text_model.embed_tokens(inputs["input_ids"])
            img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb = emb.clone()
            emb[0, im_mask] = img.to(emb.dtype)

            m_causal = build_replica_probe_mask(seq, [], [torch.tensor([], dtype=torch.long)])
            m_fence = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                               fence_blocks=True, blocks=blocks)
            for arm, mask2d, pos in (("joint", m_causal, base_pos),
                                     ("fenced", m_fence, pos_reset)):
                mask4 = mask2d.to(dev).to(emb.dtype).view(1, 1, seq, seq)
                cos_, sin_ = text_model.rotary_emb(emb, pos.to(dev))
                pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                h = emb
                for ly in layers[:L]:
                    h = ly(h, attention_mask=mask4, position_embeddings=pe)[0]
                ln = layers[L].input_layernorm(h)
                at = layers[L].self_attn
                q = at.q_proj(ln).view(1, seq, n_heads, hd).transpose(1, 2)
                k = at.k_proj(ln).view(1, seq, n_kv, hd).transpose(1, 2)
                qr, kr = apply_multimodal_rotary_pos_emb(
                    q.float(), k.float(), pe[0].float(), pe[1].float(), dims["mrope_section"])
                kr = repeat_kv(kr, n_heads // n_kv)[0]
                qr = qr[0]
                A = torch.zeros(seq, seq, device=dev)
                madd = mask2d.to(dev).to(torch.float32)
                for hh in range(n_heads):
                    lg = qr[hh] @ kr[hh].T / (hd ** 0.5) + madd
                    A += torch.softmax(lg, -1)
                A /= n_heads
                # segment-average attention mass: sum over cols in segment, mean over rows
                nseg = len(segs)
                M = np.zeros((nseg, nseg))
                for ri, (ra_, rb_) in enumerate(segs):
                    row = A[ra_:rb_]
                    for ci, (ca, cb) in enumerate(segs):
                        M[ri, ci] = float(row[:, ca:cb].sum(1).mean())
                acc[arm] = M if acc[arm] is None else acc[arm] + M
        n_done += 1
        print(f"  {n_done}/{args.limit} ({time.time()-t0:.0f}s)", flush=True)

    Mj, Mf = acc["joint"] / n_done, acc["fenced"] / n_done
    np.savez(out / "attn_segments.npz", joint=Mj, fenced=Mf, labels=np.array(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    vmin = -3.0
    for ax, M, ttl in [(axes[0], Mj, "joint attention (complete causal graph)"),
                       (axes[1], Mf, "fenced + posreset (rewired: islands + question)")]:
        im = ax.imshow(np.log10(np.maximum(M, 10 ** vmin)), cmap="Blues", vmin=vmin, vmax=0)
        ax.set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_xlabel("attended-to segment (keys)")
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, label="log10 attention mass")
    axes[0].set_ylabel("attending segment (queries)")
    fig.suptitle(f"Head-averaged attention @L{L}, averaged over {n_done} samples — "
                 "the rewiring, seen directly", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"attention_map.{ext}", dpi=300)

    (out / "ABOUT.md").write_text(
        "# Attention map — the graph rewiring, seen directly\n\n"
        "Same Q-first replica prompt, two forwards: plain causal (joint) vs block fence\n"
        "+ position reset. Head-averaged attention at layer "
        f"{L}, summed into prompt segments (Q prefix, frame_i vision, carrier/replica_i,\n"
        "final question), log scale. Joint: every frame attends across all earlier frames\n"
        "(dense lower triangle = the interference paths). Fenced: block-diagonal islands\n"
        "(frame_i+c_i pairs) that also read the leading question column — the graph the\n"
        "method actually runs on below L*. Backs 'attention is block-sparse'\n"
        f"(RESULTS.md [2026-07-17]). Data: {args.data_root}, n={n_done} samples.\n"
        "Artifacts: attention_map.png/pdf, attn_segments.npz, report in runner log.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
