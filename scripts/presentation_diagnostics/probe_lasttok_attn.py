#!/usr/bin/env python3
"""Attention row of THE LAST TOKEN (and the anchor) across layers — plain forward.

The attnmap_plain figures average the whole final-question segment; this probe
extracts the single row that actually produces the answer logits: the last
position (seq-1) and the anchor (seq-1-ANCHOR_OFFSET), at several layers in ONE
forward per sample. Segments: sys | f1..fN | fin_text (question) | tail (anchor..end).

Outputs: lasttok.csv (N, layer, qpos, segment, mass_mean, std),
         lasttok_layers.png/pdf (frames-share vs layer + per-frame profile).

Usage:
  python probe_lasttok_attn.py --layers "16 20 24 27" --output outputs/presentation/attnmap_lasttok/<ts>
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from gnnformer.constants import ANCHOR_OFFSET, ROOMS
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence
from gnnformer.fencing import build_replica_probe_mask, frame_blocks
from gnnformer.runtime import (
    attention_dims,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NS = [(8, "data/mmred_images_park/seq_len_8/all_uniform", 8),
      (32, "data/mmred_longN_park/seq_len_32/all_uniform", 8)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="16 20 24 27")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Ls = sorted(int(x) for x in args.layers.replace(",", " ").split())

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    dims = attention_dims(model)
    n_heads, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # results[(N, L, qpos_name)] = list of per-sample dicts of segment masses
    results: dict = {}
    for NF_target, root, limit in NS:
        n_done = 0
        t0 = time.time()
        for sd in iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs):
            if n_done >= limit:
                break
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
                if probe_evidence(args.task, q0, states, gold, ROOMS) is None:
                    continue
            except Exception:
                continue
            NF = len(frames)
            if args.resize > 0:
                frames = [f.resize((args.resize, args.resize)) for f in frames]
            content = [{"type": "image", "image": f} for f in frames]
            content.append({"type": "text", "text": q0})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, dev)
            ids = inputs["input_ids"][0].tolist()
            seq = len(ids)
            fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                    processor=processor)
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            if len(fg) != NF or len(vstarts) != NF:
                continue
            vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
            fin_start = int(vis[-1][-1]) + 2
            blocks = frame_blocks(vstarts, fin_start)
            anc = seq - 1 - ANCHOR_OFFSET
            qpos = {"last": seq - 1, "anchor": anc}
            # segments: sys | frames | fin_text | tail
            segs = [("sys", 0, blocks[0][0])]
            for i in range(NF):
                segs.append((f"f{i+1}", int(vis[i][0]), int(vis[i][-1]) + 1))
            segs.append(("fin_text", fin_start, anc))
            segs.append(("tail", anc, seq))
            causal = build_replica_probe_mask(seq, [], [torch.tensor([], dtype=torch.long)])

            with torch.no_grad():
                base_pos, _ = rope_fn(inputs["input_ids"],
                                      image_grid_thw=inputs.get("image_grid_thw"),
                                      attention_mask=inputs.get("attention_mask"))
                emb = text_model.embed_tokens(inputs["input_ids"])
                img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
                img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
                im_mask = inputs["input_ids"][0] == model.config.image_token_id
                emb = emb.clone()
                emb[0, im_mask] = img.to(emb.dtype)
                mask4 = causal.to(dev).to(emb.dtype).view(1, 1, seq, seq)
                cos_, sin_ = text_model.rotary_emb(emb, base_pos.to(dev))
                pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                madd = causal.to(dev).to(torch.float32)
                h = emb
                for li in range(max(Ls) + 1):
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
                        for qname, qp in qpos.items():
                            row = torch.zeros(seq, device=dev)
                            for hh in range(n_heads):
                                row += torch.softmax(
                                    (qr[hh, qp] @ kr[hh].T) / (hd ** 0.5) + madd[qp], -1)
                            row /= n_heads
                            masses = {name: float(row[a:b].sum()) for name, a, b in segs}
                            results.setdefault((NF_target, li, qname), []).append(masses)
                    h = layers[li](h, attention_mask=mask4, position_embeddings=pe)[0]
            n_done += 1
            print(f"  N={NF_target}: {n_done}/{limit} ({time.time()-t0:.0f}s)", flush=True)

    # ---- CSV ----
    rows = [["N", "layer", "qpos", "segment", "mass_mean", "std"]]
    for (N, L, qname), lst in sorted(results.items()):
        keys = lst[0].keys()
        for kseg in keys:
            v = np.array([d[kseg] for d in lst])
            rows.append([N, L, qname, kseg, f"{v.mean():.5f}", f"{v.std():.5f}"])
    with open(out / "lasttok.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    def frames_total(N, L, qname):
        lst = results.get((N, L, qname), [])
        if not lst:
            return np.nan, np.nan
        v = np.array([sum(d[k] for k in d if k.startswith("f") and k != "fin_text")
                      for d in lst])
        return v.mean(), v.std()

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {8: "#2a78d6", 32: "#eb6834"}
    styles = {"last": "-", "anchor": "--"}
    for N in (8, 32):
        for qname in ("last", "anchor"):
            m = [frames_total(N, L, qname)[0] for L in Ls]
            s = [frames_total(N, L, qname)[1] for L in Ls]
            axes[0].errorbar(Ls, m, yerr=s, color=colors[N], ls=styles[qname],
                             marker="o", ms=4, label=f"N={N} {qname}")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("attention mass on ALL frames")
    axes[0].set_title("evidence share of the answer-producing row", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    for L, alpha in zip(Ls, np.linspace(0.35, 1.0, len(Ls))):
        lst = results.get((8, L, "last"), [])
        if not lst:
            continue
        prof = np.array([[d[f"f{i+1}"] for i in range(8)] for d in lst])
        axes[1].plot(range(1, 9), prof.mean(0), marker="o", ms=3, alpha=alpha,
                     color="#2a78d6", label=f"L{L}")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("last-token mass on frame")
    axes[1].set_title("per-frame profile of the LAST token (N=8)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("The last token's attention across layers — plain forward pass", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"lasttok_layers.{ext}", dpi=300)

    (out / "ABOUT.md").write_text(
        "# Last-token attention rows across layers (plain forward)\n\n"
        "Head-mean attention row of the true LAST position (seq-1) and the anchor\n"
        f"(seq-1-{ANCHOR_OFFSET}) at layers {Ls}, one forward per sample, plain joint\n"
        "prompt. Segments: sys | f1..fN | fin_text (final question text) | tail\n"
        "(anchor..end). CSV: lasttok.csv; figure: lasttok_layers.png.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
