#!/usr/bin/env python3
"""Super-carrier probe: ONE reader over the N coarse nodes (two-level hierarchy).

Layout: Q-first replicas, full fence + posreset below L*=12 (the deployed phase-1).
Above L* the fence lifts (replicas visible). The READER is the final question's room
token; two arms differ only in what the reader's rows may attend above L*:
  coarse   reader sees {question, replica/carrier tokens} ONLY — softmax competition
           over ~N coarse nodes instead of N*m tokens (the proposal under test)
  full     reader sees everything (plain causal above L*) — the control

At --read-layers (default 16,20) the reader's message is decomposed per source
carrier (softmax under the arm's mask, context restricted to carrier f's key) ->
N features/sample -> held-out gate->tally + d'. Zero training anywhere.

Usage:
  python probe_supercarrier.py --output outputs/presentation/supercarrier/<ts>
  (loops N=8,16,32,64 with per-N limits; see NS below)
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

from gnnformer.constants import MASK_MIN, ROOMS
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

NS = [(8, "data/mmred_images_park/seq_len_8/all_uniform", 120),
      (16, "data/mmred_longN_park/seq_len_16/all_uniform", 120),
      (32, "data/mmred_longN_park/seq_len_32/all_uniform", 80),
      (64, "data/mmred_longN_park/seq_len_64/all_uniform", 40)]
w = 12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read-layers", default="16,20")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    read_layers = [int(x) for x in args.read_layers.split(",")]
    L_TOP = max(read_layers)

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    dims = attention_dims(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_O = {L: dequantize_linear_weight(layers[L].self_attn.o_proj) for L in read_layers}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression

    results = []  # (N, arm, L, tally, std, ferr, dprime)
    for NF_target, root, limit in NS:
        feats = {(arm, L): [] for arm in ("coarse", "full") for L in read_layers}
        ys, golds = [], []
        n_done = n_skip = 0
        t0 = time.time()
        for sd in iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs):
            if n_done >= limit:
                break
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
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
            rep_c = [locate_word_token(ids, tok, room, sp) for sp in rep_spans]
            fin_c = locate_word_token(ids, tok, room, fin_span)
            if any(c is None for c in rep_c) or fin_c is None:
                n_skip += 1
                continue
            blocks = frame_blocks(vstarts, fin_span[0])
            vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
            m1 = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                          fence_blocks=True, blocks=blocks)
            causal = build_replica_probe_mask(seq, [], [torch.tensor([], dtype=torch.long)])
            frame_cols = torch.cat(vis)
            m2 = {"full": causal, "coarse": causal.clone()}
            m2["coarse"][fin_span[0]:, frame_cols] = MASK_MIN  # reader rows lose frame keys

            with torch.no_grad():
                base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                      attention_mask=inputs.get("attention_mask"))
                pos = reset_positions(base_pos, blocks, fin_span[0]).clone().to(dev)
                emb = text_model.embed_tokens(inputs["input_ids"])
                img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
                img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
                im_mask = inputs["input_ids"][0] == model.config.image_token_id
                emb = emb.clone()
                emb[0, im_mask] = img.to(emb.dtype)
                cos_, sin_ = text_model.rotary_emb(emb, pos)
                pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                m1_4 = m1.to(dev).to(emb.dtype).view(1, 1, seq, seq)
                h = emb
                for ly in layers[:L_STAR]:
                    h = ly(h, attention_mask=m1_4, position_embeddings=pe)[0]
                for arm in ("coarse", "full"):
                    m2_4 = m2[arm].to(dev).to(emb.dtype).view(1, 1, seq, seq)
                    ha = h
                    for li in range(L_STAR, L_TOP + 1):
                        if li in read_layers:
                            ln = layers[li].input_layernorm(ha)
                            at = layers[li].self_attn
                            msgs = recompute_messages(
                                seq=seq, mask_full=m2[arm],
                                carrier_positions=[fin_c] * NF,
                                vis_by_frame=[torch.tensor([c]) for c in rep_c],
                                cos=pe[0], sin=pe[1], dims=dims, w_o=W_O[li],
                                q_proj=at.q_proj(ln), k_proj=at.k_proj(ln),
                                v_proj=at.v_proj(ln))
                            feats[(arm, li)].append(msgs)
                        if li < L_TOP:
                            ha = layers[li](ha, attention_mask=m2_4, position_embeddings=pe)[0]
            ys.append([1 if t in evid else 0 for t in range(NF)])
            golds.append(gold)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  N={NF_target}: {n_done}/{limit} (skip {n_skip}) {time.time()-t0:.0f}s",
                      flush=True)
        Y = np.array(ys)
        G = np.array(golds)
        for (arm, L), fs in feats.items():
            X = np.stack(fs).astype(np.float32)
            n, NF, H = X.shape
            dp, sdv, _ = dprime_pair(X, Y)
            accs, ferrs = [], []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                idx = rng.permutation(n)
                tr, ev = idx[: n // 2], idx[n // 2:]
                clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
                pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
                ferrs.append(float((pr != Y[ev]).mean()))
                accs.append(float((pr.sum(1) == G[ev]).mean()))
            results.append([NF_target, arm, L, float(np.mean(accs)), float(np.std(accs)),
                            float(np.mean(ferrs)), dp])
            print(f"[N={NF_target} {arm} L{L}] tally {np.mean(accs):.3f}±{np.std(accs):.3f} "
                  f"ferr {np.mean(ferrs):.4f} d' {dp:.2f}", flush=True)

    with open(out / "supercarrier.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "arm", "L", "tally", "std", "ferr", "dprime"], *results])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    styles = {("coarse", read_layers[0]): ("#2a78d6", "--"), ("coarse", read_layers[-1]): ("#2a78d6", "-"),
              ("full", read_layers[0]): ("#eb6834", "--"), ("full", read_layers[-1]): ("#eb6834", "-")}
    for (arm, L), (col, ls) in styles.items():
        pts = [(r[0], r[3], r[4]) for r in results if r[1] == arm and r[2] == L]
        if pts:
            xs, ac, sd = zip(*sorted(pts))
            ax.errorbar(xs, ac, yerr=sd, color=col, ls=ls, marker="o", ms=5, capsize=3,
                        lw=2, label=f"{arm} reader @L{L}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64], [8, 16, 32, 64])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("N frames")
    ax.set_ylabel("exact-count accuracy (gate→tally, held-out)")
    ax.set_title("ONE reader over N coarse nodes vs N (super-carrier probe)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"supercarrier.{ext}", dpi=300)
    (out / "ABOUT.md").write_text(
        "# Super-carrier probe — one reader over the N coarse nodes\n\n"
        "Two-level hierarchy test: fenced islands below L*=12 (deployed phase-1), fence\n"
        "lifts above; the final question's room token acts as a single reader whose rows,\n"
        "in the 'coarse' arm, may attend ONLY {question, replica/carrier tokens} — softmax\n"
        "competition over ~N coarse nodes instead of N*m tokens. 'full' arm = plain causal\n"
        "reader (control). Reader message decomposed per source carrier -> gate->tally.\n"
        "Zero training. If coarse ~ full or better and both decay slower than the joint\n"
        "anchor, a single aggregator over coarse nodes is viable at the supply level —\n"
        "the question is whether N-way softmax competition alone reintroduces the decay.\n"
        "Artifacts: supercarrier.png/pdf, supercarrier.csv, report in runner log.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
