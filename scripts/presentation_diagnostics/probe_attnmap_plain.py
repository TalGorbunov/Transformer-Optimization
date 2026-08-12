#!/usr/bin/env python3
"""Attention map of the PLAIN forward pass — query over-squashing, illustrated.

No carriers, no replicas, no fence, no posreset: the vanilla joint prompt
[frames][final question]. The final question (fin) is the ONLY reader. Two figures:

  attnmap_plain   segment-averaged head-mean attention @--layer at N=8 and N=32,
                  side by side, log scale — the fin row's budget visibly smearing
                  across N frames while frames also attend each other densely.
  fin_dilution    the fin row quantified: attention mass reaching each frame,
                  with the uniform 1/N reference — one softmax budget split N ways.

Usage:
  python probe_attnmap_plain.py --output outputs/presentation/attnmap_plain/<ts>
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

from gnnformer.constants import ROOMS
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
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    L = args.layer

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

    results = {}  # N -> dict(M=seg matrix mean, labels, fin_frame_mass=(n, NF))
    for NF_target, root, limit in NS:
        acc_M = None
        fin_mass = []
        pollution = []
        labels = None
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
            # PLAIN joint prompt: frames then the question. Nothing else.
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
            fin_start = int(vis[-1][-1]) + 2  # after the last vision block + its end token
            blocks = frame_blocks(vstarts, fin_start)
            segs = [(0, blocks[0][0])]
            slabels = ["sys"]
            for i, b in enumerate(blocks):
                segs.append((int(vis[i][0]), int(vis[i][-1]) + 1))
                slabels.append(f"f{i+1}")
            segs.append((fin_start, seq))
            slabels.append("Q (fin)")
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
                madd = causal.to(dev).to(torch.float32)
                A = torch.zeros(seq, seq, device=dev)
                for hh in range(n_heads):
                    A += torch.softmax(qr[hh] @ kr[hh].T / (hd ** 0.5) + madd, -1)
                A /= n_heads
                nseg = len(segs)
                M = np.zeros((nseg, nseg))
                for ri, (ra, rb) in enumerate(segs):
                    row = A[ra:rb]
                    for ci, (ca, cb) in enumerate(segs):
                        M[ri, ci] = float(row[:, ca:cb].sum(1).mean())
                acc_M = M if acc_M is None else acc_M + M
                fin_mass.append(M[-1, 1:-1].copy())  # fin row, frame segments only
                fr = M[1:-1, 1:-1]  # frame rows x frame cols
                cross = fr.sum(1) - np.diag(fr)  # mass a frame's tokens spend on OTHER frames
                pollution.append(np.stack([np.diag(fr), cross], 1))  # (NF, [own, cross])
                labels = slabels
            n_done += 1
            print(f"  N={NF_target}: {n_done}/{limit} ({time.time()-t0:.0f}s)", flush=True)
        results[NF_target] = dict(M=acc_M / n_done, labels=labels,
                                  fin=np.stack(fin_mass), pol=np.stack(pollution),
                                  n=n_done)

    # ---- figure 1: the two heatmaps ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    vmin = -3.0
    for ax, NF_target in zip(axes, [n for n, _, _ in NS]):
        r = results[NF_target]
        im = ax.imshow(np.log10(np.maximum(r["M"], 10 ** vmin)), cmap="Blues",
                       vmin=vmin, vmax=0)
        step = 1 if NF_target <= 8 else 4
        ticks = [0] + list(range(1, NF_target + 1, step)) + [NF_target + 1]
        ax.set_xticks(ticks, [r["labels"][t] for t in ticks], fontsize=6, rotation=90)
        ax.set_yticks(ticks, [r["labels"][t] for t in ticks], fontsize=6)
        ax.set_title(f"N={NF_target} (n={r['n']})", fontsize=11)
        ax.set_xlabel("attended-to segment (keys)")
    axes[0].set_ylabel("attending segment (queries)")
    fig.colorbar(im, ax=axes, shrink=0.75, label="log10 attention mass")
    fig.suptitle(f"Plain forward pass @L{L} — one reader (the final question), N frames",
                 fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"attnmap_plain.{ext}", dpi=300, bbox_inches="tight")

    # ---- figure 2: fin-row dilution ----
    fig2, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = {8: "#2a78d6", 32: "#eb6834"}
    rows = []
    for NF_target in [n for n, _, _ in NS]:
        r = results[NF_target]
        m = r["fin"].mean(0)
        s = r["fin"].std(0)
        frame_total = float(m.sum())
        ax.errorbar(range(1, NF_target + 1), m, yerr=s, color=colors[NF_target],
                    marker="o", ms=4, lw=1.5, capsize=2,
                    label=f"N={NF_target} (frames get {frame_total:.2f} of the budget)")
        ax.axhline(frame_total / NF_target, color=colors[NF_target], ls="--", lw=1)
        for i, (mi, si) in enumerate(zip(m, s)):
            rows.append([NF_target, i + 1, f"{mi:.5f}", f"{si:.5f}"])
    ax.set_yscale("log")
    ax.set_xlabel("frame index")
    ax.set_ylabel("fin-row attention mass on the frame (log)")
    ax.set_title("The reader's budget splits N ways (dashes = uniform share)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out / f"fin_dilution.{ext}", dpi=300)
    with open(out / "fin_dilution.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "frame", "fin_mass_mean", "std"], *rows])

    # ---- figure 3: value pollution (frame rows' cross-frame budget) ----
    fig3, ax = plt.subplots(figsize=(7.6, 4.4))
    prows = []
    for NF_target in [n for n, _, _ in NS]:
        r = results[NF_target]
        own = r["pol"][:, :, 0].mean(0)
        cross = r["pol"][:, :, 1].mean(0)
        cs = r["pol"][:, :, 1].std(0)
        ax.errorbar(range(1, NF_target + 1), cross, yerr=cs, color=colors[NF_target],
                    marker="o", ms=4, lw=1.5, capsize=2,
                    label=f"N={NF_target}: mass on OTHER frames "
                          f"(mean {cross.mean():.2f}; own-frame {own.mean():.2f})")
        for i in range(NF_target):
            prows.append([NF_target, i + 1, f"{own[i]:.4f}", f"{cross[i]:.4f}"])
    ax.set_xlabel("frame index")
    ax.set_ylabel("frame row's attention mass on OTHER frames")
    ax.set_title("Value pollution: frames absorb other frames while being encoded",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig3.tight_layout()
    for ext in ("png", "pdf"):
        fig3.savefig(out / f"value_pollution.{ext}", dpi=300)
    with open(out / "value_pollution.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "frame", "own_mass", "cross_mass"], *prows])

    (out / "ABOUT.md").write_text(
        "# Plain-forward attention map — query over-squashing, illustrated\n\n"
        "The vanilla joint prompt (frames + final question; NO carriers/replicas/fence/\n"
        f"posreset), head-averaged attention @L{L}, segment-averaged, log scale, at N=8\n"
        "and N=32. The final question (fin) is the only reader; its row shows one softmax\n"
        "budget smeared across N frames (fin_dilution quantifies it against the uniform\n"
        "1/N dashes), while the dense lower triangle is frames attending each other (the\n"
        "value-pollution paths). Companion to the fenced/rewired attnmap figures — this\n"
        "is the BEFORE picture of the pair.\n"
        "Artifacts: attnmap_plain.png/pdf, fin_dilution.png/pdf, fin_dilution.csv.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
