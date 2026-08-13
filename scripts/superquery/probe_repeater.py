#!/usr/bin/env python3
"""REPEATER TREE: end-to-end exact counting in ONE forward, frozen model + quantizers.

The patch experiment (patch_n8) proved the law: frozen attention transfers VOCAB-coded
content losslessly (P2 lvl2 = 1.000) and state-coded content only as analog (~0.4).
This probe deploys the fix end-to-end at N=8, b=2 (levels 4-2-1):

  layers 0..20   fenced blocks + tree mask (unchanged)
  Q1 @ L20       quantize level-1: pre-fit linear probe reads each pair-SQ's count
                 (PREDICTED, not gold) -> span := digit-token embedding (norm-matched)
  layers 21..24  level-2 reads vocab codes (measured perfect in P2)
  Q2 @ L24       quantize level-2 the same way (probe fit on P2-arm states)
  layers 25..27  root reads vocab codes
  readout        linear probe on root span @L27 (fit/test split) -> exact gold count

Quantizer probes come from BANKED captures (feats_N8.npz lvl1@20; patch_feats.npz
P2 lvl2@24) — gold labels used only for fitting those probes and for scoring.
Reported: Q1 acc, Q2 acc (vs gold), root exact/±1/MAE, and the compounding chain.

Usage:
  python scripts/superquery/probe_repeater.py --output outputs/superquery/repeater_n8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))

from probe_tree import leaf_sets, tree_levels  # noqa: E402

from gnnformer.constants import MASK_MIN, ROOMS  # noqa: E402
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence  # noqa: E402
from gnnformer.fencing import (  # noqa: E402
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    reset_positions,
)
from gnnformer.runtime import (  # noqa: E402
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NF = 8
ROOT = "data/mmred_images_park/seq_len_8/all_uniform"
Q1_L, Q2_L, TOP_L = 20, 24, 27


def fit_head(X, y):
    from sklearn.linear_model import LogisticRegression
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    clf = LogisticRegression(max_iter=2000).fit((X - mu) / sd, y)
    return clf, mu, sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--lvl1-npz", default="outputs/superquery/capture_16_64_128790/feats_N8.npz")
    ap.add_argument("--lvl2-npz", default="outputs/superquery/patch_n8/patch_feats.npz")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    levels = tree_levels(NF, 2)
    lsets = leaf_sets(levels)
    n_nodes = sum(len(g) for g in levels)

    # ---- Q1 head: level-1 pair count from frozen states @L20 (banked capture)
    d1 = np.load(args.lvl1_npz)
    X1, Y1 = d1["b2|0|20|mean"].astype(np.float32), d1["Y"]
    c1 = np.stack([[int(Y1[s, g].sum()) for g in levels[0]] for s in range(len(Y1))])
    q1_head = fit_head(X1.reshape(-1, X1.shape[-1]), c1.reshape(-1))
    print(f"Q1 head: {len(c1)} samples, train-acc "
          f"{q1_head[0].score((X1.reshape(-1, X1.shape[-1]) - q1_head[1]) / q1_head[2], c1.reshape(-1)):.3f}",
          flush=True)

    # ---- Q2 head: level-2 count from QUANTIZED-CHILD states @L24 (P2 arm of patch run)
    d2 = np.load(args.lvl2_npz)
    X2, Y2 = d2["P2|1|24|mean"].astype(np.float32), d2["Y"]
    c2 = np.stack([[int(Y2[s, sorted(ls)].sum()) for ls in lsets[1]] for s in range(len(Y2))])
    q2_head = fit_head(X2.reshape(-1, X2.shape[-1]), c2.reshape(-1))
    print(f"Q2 head: {len(c2)} samples, train-acc "
          f"{q2_head[0].score((X2.reshape(-1, X2.shape[-1]) - q2_head[1]) / q2_head[2], c2.reshape(-1)):.3f}",
          flush=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_emb = text_model.embed_tokens.weight
    dig = {}
    for k in range(NF + 1):
        ids_k = tok(f"{k}", add_special_tokens=False).input_ids
        assert len(ids_k) == 1
        dig[k] = ids_k[0]

    def quantize(h, spans_counts):
        for (a, b), k in spans_counts:
            v = W_emb[dig[int(k)]].to(h.dtype)
            scale = h[0, a:b].norm(dim=-1).median() / (v.norm() + 1e-6)
            h[0, a:b] = v * scale
        return h

    rows = []
    root_feats, q1_accs, q2_accs = [], [], []
    golds, q1_preds_all, q2_preds_all = [], [], []
    n_done = n_skip = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence("steps", q0, states, gold, ROOMS)
            if pe_ is None:
                n_skip += 1
                continue
            evid, _room = pe_
        except Exception:
            n_skip += 1
            continue
        if (not evid and gold != 0) or len(frames) != NF:
            n_skip += 1
            continue
        frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        for _ in range(n_nodes):
            content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, dev)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            n_skip += 1
            continue
        rep_spans = spans[1 : NF + 1]
        sq_spans = spans[NF + 1 :]
        fin_start = sq_spans[0][0]
        blocks = frame_blocks(vstarts, fin_start)
        vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]

        m = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                     fence_blocks=True, blocks=blocks)
        prefix_cols = torch.arange(0, blocks[0][0])
        si = 0
        span_of = {}
        for li, groups in enumerate(levels):
            for gi, g in enumerate(groups):
                a, b = sq_spans[si]
                span_of[(li, gi)] = (a, b)
                si += 1
                child_spans = ([rep_spans[c] for c in g] if li == 0
                               else [span_of[(li - 1, c)] for c in g])
                rws = torch.arange(a, b)
                m[rws] = MASK_MIN
                cols = torch.cat([prefix_cols] +
                                 [torch.arange(ca, cb) for ca, cb in child_spans])
                m[rws.unsqueeze(1), cols.unsqueeze(0)] = 0.0
                blk = torch.zeros(b - a, b - a)
                blk.masked_fill_(torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1),
                                 MASK_MIN)
                m[a:b, a:b] = blk

        k1_gold = [int(sum(1 for c in g if c in evid)) for g in levels[0]]
        k2_gold = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]

        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"],
                                  image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos = reset_positions(base_pos, blocks, fin_start).clone().to(dev)
            emb = text_model.embed_tokens(inputs["input_ids"]).clone()
            img = model.model.get_image_features(inputs["pixel_values"],
                                                 inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb[0, im_mask] = img.to(emb.dtype)
            cos_, sin_ = text_model.rotary_emb(emb, pos)
            pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
            m4 = m.to(dev).to(emb.dtype).view(1, 1, seq, seq)

            h = emb
            for li in range(TOP_L + 1):
                h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
                if li == Q1_L:
                    # Q1: probe-predicted pair counts -> digit embeddings
                    clf, mu, sd_ = q1_head
                    feats1 = []
                    for gi in range(len(levels[0])):
                        a, b = span_of[(0, gi)]
                        feats1.append(h[0, a:b].mean(0).float().cpu().numpy())
                    p1 = clf.predict((np.stack(feats1) - mu) / sd_)
                    q1_accs.append(float((p1 == np.array(k1_gold)).mean()))
                    q1_preds_all.append(p1.tolist())
                    h = quantize(h, [(span_of[(0, gi)], p1[gi])
                                     for gi in range(len(levels[0]))])
                if li == Q2_L:
                    clf, mu, sd_ = q2_head
                    feats2 = []
                    for gi in range(len(levels[1])):
                        a, b = span_of[(1, gi)]
                        feats2.append(h[0, a:b].mean(0).float().cpu().numpy())
                    p2 = clf.predict((np.stack(feats2) - mu) / sd_)
                    q2_accs.append(float((p2 == np.array(k2_gold)).mean()))
                    q2_preds_all.append(p2.tolist())
                    h = quantize(h, [(span_of[(1, gi)], p2[gi])
                                     for gi in range(len(levels[1]))])
            a, b = span_of[(2, 0)]
            root_feats.append(np.concatenate([h[0, a:b].mean(0).float().cpu().numpy(),
                                              h[0, b - 1].float().cpu().numpy()]))
        golds.append(gold)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  {n_done}/{args.limit} (skip {n_skip}) {time.time()-t0:.0f}s",
                  flush=True)

    G = np.array(golds)
    n = len(G)
    X = np.stack(root_feats).astype(np.float32)
    H = X.shape[-1] // 2
    print(f"\nQ1 acc (per-node, vs gold): {np.mean(q1_accs):.3f}")
    print(f"Q2 acc (per-node, vs gold): {np.mean(q2_accs):.3f}")
    # symbolic composition baselines (no root read): sum of Q1 / sum of Q2 preds
    sum_q1 = np.array([sum(p) for p in q1_preds_all])
    sum_q2 = np.array([sum(p) for p in q2_preds_all])
    print(f"sum(Q1 preds) == gold: {float((sum_q1 == G).mean()):.3f}  "
          f"(the external-compose bound)")
    print(f"sum(Q2 preds) == gold: {float((sum_q2 == G).mean()):.3f}")
    from sklearn.linear_model import LogisticRegression
    for feat, sl in (("mean", slice(0, H)), ("last", slice(H, 2 * H)),
                     ("both", slice(0, 2 * H))):
        accs, maes, pm1 = [], [], []
        for seed in range(3):
            idx = np.random.default_rng(seed).permutation(n)
            tr, ev = idx[: n // 2], idx[n // 2:]
            Xtr, Xev = X[tr, sl], X[ev, sl]
            mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
            clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd_, G[tr])
            pr = clf.predict((Xev - mu) / sd_)
            accs.append(float((pr == G[ev]).mean()))
            pm1.append(float((np.abs(pr - G[ev]) <= 1).mean()))
            maes.append(float(np.abs(pr - G[ev]).mean()))
        print(f"[ROOT@L{TOP_L} {feat}] exact {np.mean(accs):.3f} +-1 {np.mean(pm1):.3f} "
              f"MAE {np.mean(maes):.2f}", flush=True)
        rows.append([feat, np.mean(accs), np.mean(pm1), np.mean(maes)])
    with open(out / "repeater.csv", "w", newline="") as f:
        csv.writer(f).writerows(
            [["feat", "root_exact", "pm1", "mae"], *rows,
             ["q1_acc", np.mean(q1_accs), "", ""], ["q2_acc", np.mean(q2_accs), "", ""],
             ["sum_q1", float((sum_q1 == G).mean()), "", ""],
             ["sum_q2", float((sum_q2 == G).mean()), "", ""]])
    np.savez_compressed(out / "repeater_feats.npz", X=X.astype(np.float16), G=G,
                        q1=np.array(q1_preds_all), q2=np.array(q2_preds_all))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
