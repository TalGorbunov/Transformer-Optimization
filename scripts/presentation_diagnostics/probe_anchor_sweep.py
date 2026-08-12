#!/usr/bin/env python3
"""Anchor-choice sweep: WHERE inside the replica should the teacher message be read?

The distillation teacher reads the message received by the ROOM word's token inside
each per-frame question replica (a hand-picked, task-informed choice). This probe
sweeps that choice on the teacher configuration itself (Q-first replicas + full
block fence + posreset, the A3 layout) with NO training:

  room   the queried room word's token  (the method's current choice)
  char   the queried character name's token
  last   the last token of the replica span
  first  the first token of the replica span
  mean   average of the messages over every token in the replica span

For each variant: held-out d' + gate->tally (scripts/gate_tally.py protocol) on the
messages @L16. If a variant beats `room` clearly, it becomes the new distillation
anchor (one-line change in train_carrier_token); otherwise `room` is validated.

Usage:
  python scripts/probe_anchor_sweep.py --limit 150 --layer 16 \
      --output outputs/presentation/anchor_sweep/<ts>
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

from gnnformer.constants import ROOMS
from gnnformer.data import (
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_target_character_room,
    probe_evidence,
)
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

VARIANTS = ["room", "char", "last", "first", "mean"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=150)
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
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_O = dequantize_linear_weight(layers[L].self_attn.o_proj)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    feats = {v: [] for v in VARIANTS}
    ys, golds = [], []
    n_done = n_skip = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence(args.task, q0, states, gold, ROOMS)
            if pe_ is None:
                n_skip += 1
                continue
            evid, room = pe_
            parsed = parse_target_character_room(q0)
            if parsed is None:
                n_skip += 1
                continue
            character, _ = parsed
        except Exception:
            n_skip += 1
            continue
        if not evid and gold != 0:
            n_skip += 1
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
            n_skip += 1
            continue
        spans = spans[1:]
        rep_spans, fin_span = spans[:NF], spans[NF]
        pos_room = [locate_word_token(ids, tok, room, sp) for sp in rep_spans]
        pos_char = [locate_word_token(ids, tok, character, sp) for sp in rep_spans]
        if any(c is None for c in pos_room) or any(c is None for c in pos_char):
            n_skip += 1
            continue
        blocks = frame_blocks(vstarts, fin_span[0])
        vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
        m_fence = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                           fence_blocks=True, blocks=blocks)
        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos_r = reset_positions(base_pos, blocks, fin_span[0]).clone()
            emb = text_model.embed_tokens(inputs["input_ids"])
            img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb = emb.clone()
            emb[0, im_mask] = img.to(emb.dtype)
            mask4 = m_fence.to(dev).to(emb.dtype).view(1, 1, seq, seq)
            cos_, sin_ = text_model.rotary_emb(emb, pos_r.to(dev))
            pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
            h = emb
            for ly in layers[:L]:
                h = ly(h, attention_mask=mask4, position_embeddings=pe)[0]
            ln = layers[L].input_layernorm(h)
            at = layers[L].self_attn
            common = dict(seq=seq, mask_full=m_fence, vis_by_frame=vis, cos=pe[0], sin=pe[1],
                          dims=dims, w_o=W_O, q_proj=at.q_proj(ln), k_proj=at.k_proj(ln),
                          v_proj=at.v_proj(ln))
            feats["room"].append(recompute_messages(carrier_positions=pos_room, **common))
            feats["char"].append(recompute_messages(carrier_positions=pos_char, **common))
            feats["last"].append(recompute_messages(
                carrier_positions=[rb - 1 for _, rb in rep_spans], **common))
            feats["first"].append(recompute_messages(
                carrier_positions=[ra for ra, _ in rep_spans], **common))
            span_msgs = []
            for i, (ra, rb) in enumerate(rep_spans):
                ms = recompute_messages(carrier_positions=list(range(ra, rb)),
                                        vis_by_frame=[vis[i]] * (rb - ra),
                                        **{k_: v_ for k_, v_ in common.items()
                                           if k_ != "vis_by_frame"})
                span_msgs.append(ms.mean(0))
            feats["mean"].append(np.stack(span_msgs))
        ys.append([1 if t in evid else 0 for t in range(NF)])
        golds.append(gold)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{args.limit} (skip {n_skip}) {time.time()-t0:.0f}s", flush=True)

    from sklearn.linear_model import LogisticRegression

    Y = np.array(ys)
    G = np.array(golds)
    lines = [f"=== ANCHOR SWEEP (n={n_done}, skip={n_skip}, L{L}, teacher config = "
             f"Q-first blockfence+posreset, data={args.data_root}) ==="]
    rows = []
    for v in VARIANTS:
        X = np.stack(feats[v]).astype(np.float32)
        n, NF, H = X.shape
        dp, sdv, auc = dprime_pair(X, Y)
        accs, ferrs = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            tr, ev = idx[: n // 2], idx[n // 2:]
            clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
            pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
            ferrs.append(float((pr != Y[ev]).mean()))
            accs.append(float((pr.sum(1) == G[ev]).mean()))
        rows.append((v, dp, sdv, float(np.mean(accs)), float(np.std(accs)), float(np.mean(ferrs))))
        lines.append(f"{v:>6}: d' {dp:.2f}±{sdv:.2f} (auc {auc:.2f})  "
                     f"gate->tally {np.mean(accs):.3f}±{np.std(accs):.3f}  "
                     f"per-frame err {np.mean(ferrs):.4f}")
        print(lines[-1], flush=True)
    np.savez(out / "anchor_messages.npz", Y=Y, G=G,
             **{v: np.stack(feats[v]).astype(np.float16) for v in VARIANTS})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    names = [r[0] for r in rows]
    for ax, vals, errs, ylab in [
            (axes[0], [r[1] for r in rows], [r[2] for r in rows], "held-out d′"),
            (axes[1], [r[3] for r in rows], [r[4] for r in rows], "gate→tally exact")]:
        bars = ax.bar(names, vals, yerr=errs, capsize=3,
                      color=["#2a78d6" if nm == "room" else "#9ec5f4" for nm in names])
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.7)
        ax.set_axisbelow(True)
        for b, v_ in zip(bars, vals):
            ax.annotate(f"{v_:.2f}", (b.get_x() + b.get_width() / 2, v_),
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Where to read the replica's message @L{L} (n={n_done}, teacher config) — "
                 "dark bar = the method's current choice", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"anchor_sweep.{ext}", dpi=300)

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "ABOUT.md").write_text(
        "# Anchor-choice sweep — is the room-token read the right distillation target?\n\n"
        "The carrier is distilled to match the message received by the ROOM word's token\n"
        "inside each per-frame question replica — a hand-picked, task-informed choice that\n"
        "was never ablated. This probe reads the SAME teacher forward (Q-first replicas +\n"
        "full block fence + posreset) at five candidate anchors (room / character / last /\n"
        "first token of the replica, and the span-mean of all its messages) and scores each\n"
        "with held-out d' + gate->tally @L" + str(L) + ". No training anywhere. If an\n"
        "alternative wins clearly, it becomes the new distillation anchor (one-line change\n"
        "in train_carrier_token.py); if not, the room choice is validated as an ablation\n"
        f"winner rather than a guess. Data: {args.data_root}, n={n_done}.\n"
        "Artifacts: anchor_sweep.png/pdf, anchor_messages.npz, report.txt.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
