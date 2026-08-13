#!/usr/bin/env python3
"""Jacobian sensitivity probe (the GNN-literature over-squashing measure).

For the SAME Q-first replica layout, two arms:
  joint  — plain causal attention (no fence): target = the discriminative signal
           at the final-question anchor locus (the model's aggregation point),
           read per frame f as w_j . m_anchor,f (recompute_messages restricted to
           frame f's vision keys).
  fenced — full block fence + posreset (A3): target = each replica-carrier's own
           message, w_c . m_carrier,f.

For each target scalar s_f we backprop to the LAYER-0 merged input embeddings and
report S[f, f'] = || dS_f / d(emb rows of frame f') ||_2 — the sensitivity of the
read-out signal to each source frame. Joint should be diffuse (one locus splits
its sensitivity budget over N frames); fenced should be near-diagonal (each
carrier owns its frame; the fence zeroes cross-frame edges below L*).

w_j / w_c are logistic directions fit IN-JOB on a no-grad collection phase
(arm-matched messages, held-out protocol as scripts/gate_tally.py).

Usage:
  python scripts/probe_sensitivity.py --data_root data/mmred_images_park/seq_len_8/all_uniform \
      --fit-n 64 --grad-n 12 --layer 16 --output outputs/presentation/sensitivity/<ts>
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

from gnnformer.constants import ANCHOR_OFFSET, ROOMS
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence
from gnnformer.fencing import (
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    locate_word_token,
    recompute_messages,
    reset_positions,
)
from gnnformer.metrics import dprime_pair
from gnnformer.runtime import (
    attention_dims,
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--fit-n", type=int, default=64)
    ap.add_argument("--grad-n", type=int, default=12)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--target", choices=["message", "qstate", "anchor"], default="message",
                    help="message = frame-key-restricted carrier message (original); "
                         "qstate = full layer-L hidden state at each question replica's room token; "
                         "anchor = THE final question query's layer-L state (one row per sample, "
                         "count direction ridge-fit on gold; joint arm only)")
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
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_O = dequantize_linear_weight(layers[L].self_attn.o_proj)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def prep(sd):
        """Build the Q-first replica layout for one sample -> dict or None."""
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence(args.task, q0, states, gold, ROOMS)
            if pe_ is None:
                return None
            evid, room = pe_
        except Exception:
            return None
        if not evid and gold != 0:
            return None
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
            return None
        spans = spans[1:]
        rep_spans, fin_span = spans[:NF], spans[NF]
        rep_c = [locate_word_token(ids, tok, room, sp) for sp in rep_spans]
        if any(c is None for c in rep_c):
            return None
        blocks = frame_blocks(vstarts, fin_span[0])
        vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
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
        return dict(seq=seq, NF=NF, emb=emb, base_pos=base_pos, pos_reset=pos_reset,
                    vis=vis, rep_c=rep_c, anc=seq - 1 - ANCHOR_OFFSET,
                    m_causal=m_causal, m_fence=m_fence,
                    y=np.array([1 if t in evid else 0 for t in range(NF)]))

    def run_msgs(d, arm, emb_var):
        """Truncated forward layers[:L] + layer-L qkv -> differentiable messages [NF, H]."""
        seq = d["seq"]
        mask2d = d["m_causal"] if arm == "joint" else d["m_fence"]
        mask4 = mask2d.to(dev).to(emb_var.dtype).view(1, 1, seq, seq)
        pos = (d["base_pos"] if arm == "joint" else d["pos_reset"]).to(dev)
        cos_, sin_ = text_model.rotary_emb(emb_var, pos)
        pe = (cos_.to(emb_var.dtype), sin_.to(emb_var.dtype))
        h = emb_var
        for ly in layers[:L]:
            h = ly(h, attention_mask=mask4, position_embeddings=pe)[0]
        if args.target == "qstate":
            idx = torch.tensor(d["rep_c"], dtype=torch.long, device=h.device)
            return h[0, idx]
        if args.target == "anchor":
            return h[0, [d["anc"]]]
        ln = layers[L].input_layernorm(h)
        at = layers[L].self_attn
        carriers = [d["anc"]] * d["NF"] if arm == "joint" else d["rep_c"]
        return recompute_messages(seq=seq, mask_full=mask2d, carrier_positions=carriers,
                                  vis_by_frame=d["vis"], cos=pe[0], sin=pe[1], dims=dims,
                                  w_o=W_O, q_proj=at.q_proj(ln), k_proj=at.k_proj(ln),
                                  v_proj=at.v_proj(ln), differentiable=True)

    dirs = iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs)
    arms = ("joint",) if args.target == "anchor" else ("joint", "fenced")

    # ---- phase 1: no-grad message collection, fit the two directions ----
    feats = {"joint": [], "fenced": []}
    ys = []
    prepped = []
    t0 = time.time()
    for sd in dirs:
        if len(prepped) >= args.fit_n + args.grad_n:
            break
        d = prep(sd)
        if d is None:
            continue
        prepped.append(d)
        if len(prepped) <= args.fit_n:
            with torch.no_grad():
                for arm in arms:
                    feats[arm].append(run_msgs(d, arm, d["emb"]).detach().float().cpu().numpy())
            ys.append(d["y"])
            d["emb"] = None  # fit samples: free the embedding
        if len(prepped) % 20 == 0:
            print(f"  prep {len(prepped)} ({time.time()-t0:.0f}s)", flush=True)

    Y = np.stack(ys)
    W = {}
    lines = [f"=== SENSITIVITY PROBE (fit_n={len(ys)}, grad_n={args.grad_n}, L{L}, "
             f"target={args.target}, data={args.data_root}) ==="]
    from sklearn.linear_model import LogisticRegression, Ridge

    for arm in arms:
        X = np.stack(feats[arm])
        if args.target == "anchor":
            Xa = X[:, 0, :]                       # [n, H] final-query states
            gold = Y.sum(1)                       # gold count per sample
            reg = Ridge(alpha=1.0).fit(Xa, gold)
            W[arm] = torch.tensor(reg.coef_, dtype=torch.float32, device=dev)
            lines.append(f"[fit] {arm}: anchor count-ridge R2 {reg.score(Xa, gold):.3f} (n={len(gold)})")
        else:
            n, NF, H = X.shape
            dp, sdv, _ = dprime_pair(X, Y)
            clf = LogisticRegression(max_iter=2000).fit(X.reshape(-1, H), Y.reshape(-1))
            W[arm] = torch.tensor(clf.coef_[0], dtype=torch.float32, device=dev)
            lines.append(f"[fit] {arm}: message d' {dp:.2f}±{sdv:.2f} (direction from n={n})")
        print(lines[-1], flush=True)

    # ---- phase 2: gradient phase ----
    S = {"joint": [], "fenced": []}
    grad_samples = [d for d in prepped if d["emb"] is not None][: args.grad_n]
    for si, d in enumerate(grad_samples):
        for arm in arms:
            emb_var = d["emb"].detach().clone().requires_grad_(True)
            msgs = run_msgs(d, arm, emb_var)
            NF = d["NF"]
            R = msgs.shape[0]           # NF rows (message/qstate) or 1 row (anchor)
            mat = np.zeros((R, NF), dtype=np.float64)
            for f in range(R):
                s = (W[arm] * msgs[f].float()).sum()
                (g,) = torch.autograd.grad(s, emb_var, retain_graph=(f < R - 1))
                gr = g[0].float()
                for fp in range(NF):
                    mat[f, fp] = float(gr[d["vis"][fp].to(dev)].norm())
            S[arm].append(mat)
        print(f"  grad {si+1}/{len(grad_samples)} ({time.time()-t0:.0f}s)", flush=True)

    Sj = np.stack(S["joint"])   # (n, R, NF): read-out row x source frame
    np.savez(out / "sensitivity.npz", S_joint=Sj,
             **({} if args.target == "anchor" else {"S_fenced": np.stack(S["fenced"])}))

    def row_share(M):  # normalize each target row to shares over source frames
        return M / np.maximum(M.sum(-1, keepdims=True), 1e-12)

    if args.target == "anchor":
        NF = Sj.shape[2]
        sh = row_share(Sj)[:, 0, :]                       # (n, NF) share per source frame
        mean, sd = sh.mean(0), sh.std(0)
        lines += [f"[grad] joint anchor: per-frame shares {np.round(mean, 3).tolist()}",
                  f"[grad] joint anchor: max/min share {mean.max():.3f}/{mean.min():.3f} "
                  f"(uniform = {1/NF:.3f})"]
        fig, ax = plt.subplots(figsize=(8.4, 2.6))
        im = ax.imshow(mean[None, :], cmap="Blues", vmin=0, vmax=max(0.25, float(mean.max())))
        for fp in range(NF):
            ax.text(fp, 0, f"{mean[fp]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mean[fp] > 0.6 * max(0.25, float(mean.max())) else "#12161c")
        ax.set_xticks(np.arange(NF))
        ax.set_xlabel("source frame f′")
        ax.set_yticks([0])
        ax.set_yticklabels(["Q (fin)"])
        ax.set_title("joint attention (no fence)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, label="sensitivity share")
        fig.suptitle("Jacobian sensitivity ‖∂(final-query count signal)/∂(frame f′ embeddings)‖, "
                     f"L≤{L}, N={NF}", fontsize=12)
    else:
        Sf = np.stack(S["fenced"])
        shj, shf = row_share(Sj), row_share(Sf)
        diag_j = float(np.mean([m.diagonal().mean() for m in shj]))
        diag_f = float(np.mean([m.diagonal().mean() for m in shf]))
        lines += [f"[grad] joint: own-frame sensitivity share {diag_j:.3f} (uniform = {1/Sj.shape[1]:.3f})",
                  f"[grad] fenced: own-frame sensitivity share {diag_f:.3f}",
                  f"[grad] cross/within ratio: joint {(1-diag_j)/max(diag_j,1e-9):.2f}, "
                  f"fenced {(1-diag_f)/max(diag_f,1e-9):.2f}"]

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for ax, M, ttl in [(axes[0], shj.mean(0), "joint attention (no fence)"),
                           (axes[1], shf.mean(0), "fenced + posreset")]:
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
            ax.set_xlabel("source frame f′")
            ax.set_ylabel("read-out target for frame f")
            ax.set_title(ttl, fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.8, label="sensitivity share")
        tgt_label = ("read-out signal for f" if args.target == "message"
                     else "question-replica state for f")
        fig.suptitle(f"Jacobian sensitivity ‖∂({tgt_label})/∂(frame f′ embeddings)‖, "
                     f"L≤{L}, N={Sj.shape[1]}", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sensitivity.{ext}", dpi=300)

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "ABOUT.md").write_text(
        "# Jacobian sensitivity probe — over-squashing measured the GNN-literature way\n\n"
        "Both arms use the SAME Q-first replica prompt; only the attention graph differs.\n"
        "For each frame f we take the discriminative signal at the read-out locus (joint:\n"
        "final-question anchor restricted to frame f's keys; fenced: carrier f's message),\n"
        "backprop to the layer-0 merged embeddings, and report the gradient-norm share per\n"
        "source frame f'. Joint = one aggregation point splitting its sensitivity budget\n"
        "(off-diagonal mass = cross-frame interference); fenced = near-diagonal (each\n"
        "carrier owns its frame). Directions w fit in-job (held-out logistic on arm-matched\n"
        f"messages). Data: {args.data_root}. Artifacts: sensitivity.png/pdf, sensitivity.npz\n"
        "(raw norms), report.txt (headline shares + fit d').\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
