#!/usr/bin/env python3
"""Attention map of the DEPLOYED two-phase graph (the hierarchical-coarsening figure).

Runs the deployed carrier stack (single learned carrier per frame, e_c + LoRA from a
carrier_layer ckpt, lo/hi masks, posreset) and renders the head-averaged attention at
one layer BELOW L* (fenced phase: frame+carrier islands + question) and one layer AT/
ABOVE L* (open phase: carriers attend each other, the final question reads carriers).
Together with probe_attention_map.py (probe layout) this photographs both levels of
the hierarchy: fine graph -> islands -> coarse graph over carrier supernodes.

Usage:
  python scripts/probe_attnmap_deployed.py --ckpt checkpoints/carrier_layer_fmt_caption_best.pt \
      --limit 8 --layers 8,20 --output outputs/presentation/attnmap_deployed/<ts>
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

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.engine import SDPA_BACKENDS
from torch.nn.attention import sdpa_kernel
from gnnformer.constants import ROOMS
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence
from gnnformer.engine import CarrierEngine
from gnnformer.runtime import attention_dims, get_layers, load_runtime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/carrier_layer_fmt_caption_best.pt")
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--layers", default="8,20", help="below-L*,at/above-L* capture layers")
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--no-lora", action="store_true", help="mask/graph only, frozen weights")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    layers = get_layers(rt.model)
    dims = attention_dims(rt.model)
    n_heads, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    dev = rt.model.device
    Ls = [int(x) for x in args.layers.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = load_carrier_layer_ckpt(Path(args.ckpt))
    lora = None
    if not args.no_lora:
        lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha,
                           device=dev, state=ck.lora_state)
    e_c = torch.as_tensor(ck.e_c).to(dev)
    eng = CarrierEngine(rt, l_open=ck.l_open, e_c=e_c)
    text_model = eng.text_model
    print(f"[ckpt] {args.ckpt}: L*={ck.l_open} rank={ck.rank} lora={'off' if args.no_lora else 'on'}",
          flush=True)

    acc = {L: None for L in Ls}
    labels = None
    n_done = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            if probe_evidence(args.task, q0, states, gold, ROOMS) is None:
                continue
        except Exception:
            continue
        d = eng.prepare_sample(frames, q0, gold=gold, task=args.task, with_masks=True)
        if d is None or (labels is not None and len(d["cpos"]) * 2 + 2 != len(labels)):
            continue
        seq, cpos, blocks, fin = d["seq"], d["cpos"], d["blocks"], d["fin"]
        segs = [(0, blocks[0][0])]
        seg_labels = ["Q"]
        for i, ((a, b), c) in enumerate(zip(blocks, cpos)):
            segs.append((a, c))
            seg_labels.append(f"f{i+1}")
            segs.append((c, c + 1))
            seg_labels.append(f"c{i+1}")
        segs.append((fin, seq))
        seg_labels.append("fin")
        if labels is None:
            labels = seg_labels

        with torch.no_grad():
            emb = d["emb"].to(dev).unsqueeze(0).clone()
            emb[0, torch.tensor(cpos, device=dev)] = e_c.to(torch.bfloat16)
            # engine-style: fp32 masks + EFFICIENT->MATH sdpa backends (fused kernels
            # reject fp32 bias with bf16 query; MATH/EFFICIENT accept it, and the LoRA
            # layers upcast queries to fp32 anyway)
            lo = d["lo"].to(dev).to(torch.float32).view(1, 1, seq, seq)
            hi = d["hi"].to(dev).to(torch.float32).view(1, 1, seq, seq)
            pos = d["pos"].to(dev)
            cos_, sin_ = text_model.rotary_emb(emb, pos)
            pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
            h = emb
            for li in range(max(Ls) + 1):
                mask4 = lo if li < eng.l_open else hi
                if li in Ls:
                    ln = layers[li].input_layernorm(h)
                    at = layers[li].self_attn
                    q = at.q_proj(ln).view(1, seq, n_heads, hd).transpose(1, 2)
                    k = at.k_proj(ln).view(1, seq, n_kv, hd).transpose(1, 2)
                    qr, kr = apply_multimodal_rotary_pos_emb(
                        q.float(), k.float(), pe[0].float(), pe[1].float(),
                        dims["mrope_section"])
                    kr = repeat_kv(kr, n_heads // n_kv)[0]
                    qr = qr[0]
                    A = torch.zeros(seq, seq, device=dev)
                    madd = mask4[0, 0].float()
                    for hh_ in range(n_heads):
                        lg = qr[hh_] @ kr[hh_].T / (hd ** 0.5) + madd
                        A += torch.softmax(lg, -1)
                    A /= n_heads
                    M = np.zeros((len(segs), len(segs)))
                    for ri, (ra_, rb_) in enumerate(segs):
                        row = A[ra_:rb_]
                        for ci, (ca, cb) in enumerate(segs):
                            M[ri, ci] = float(row[:, ca:cb].sum(1).mean())
                    acc[li] = M if acc[li] is None else acc[li] + M
                with sdpa_kernel(SDPA_BACKENDS):
                    h = layers[li](h, attention_mask=mask4, position_embeddings=pe)[0]
        n_done += 1
        print(f"  {n_done}/{args.limit} ({time.time()-t0:.0f}s)", flush=True)

    Ms = {L: acc[L] / n_done for L in Ls}
    np.savez(out / "attn_segments_deployed.npz", labels=np.array(labels),
             **{f"L{L}": Ms[L] for L in Ls})

    fig, axes = plt.subplots(1, len(Ls), figsize=(6.3 * len(Ls), 5.6))
    vmin = -3.0
    phase = {True: "fenced phase (islands + question)", False: "open phase (coarse graph)"}
    for ax, L in zip(np.atleast_1d(axes), Ls):
        im = ax.imshow(np.log10(np.maximum(Ms[L], 10 ** vmin)), cmap="Blues", vmin=vmin, vmax=0)
        ax.set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_xlabel("attended-to segment (keys)")
        ax.set_title(f"L{L} — {phase[L < ck.l_open]}", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, label="log10 attention mass")
    np.atleast_1d(axes)[0].set_ylabel("attending segment (queries)")
    fig.suptitle(f"DEPLOYED stack attention (e_c + LoRA {'off' if args.no_lora else 'on'}, "
                 f"L*={ck.l_open}, n={n_done}) — the two-level hierarchy", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"attnmap_deployed.{ext}", dpi=300)

    if lora is not None:
        lora.remove()
    (out / "ABOUT.md").write_text(
        "# Deployed-stack attention map — both levels of the hierarchy\n\n"
        "The deployed carrier stack (single learned carrier per frame, e_c + LoRA from\n"
        f"{args.ckpt}, lo/hi masks, posreset) run once per sample; head-averaged attention\n"
        f"captured below L* (L{Ls[0]}: each frame+carrier island reads itself + the leading\n"
        f"question) and at/above L* (L{Ls[1]}: the fence lifts — carriers attend each other\n"
        "and the final question reads the carriers). This is the graph-coarsening picture:\n"
        "fine token graph -> islands pool into carrier supernodes -> coarse graph over\n"
        "supernodes. Companion figure: probe_attention_map.py (probe layout, joint vs\n"
        f"fenced). Data: {args.data_root}, n={n_done} samples.\n"
        "Artifacts: attnmap_deployed.png/pdf, attn_segments_deployed.npz.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
