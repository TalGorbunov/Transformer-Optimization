#!/usr/bin/env python3
"""REPEATER TREE v2: in-layout calibrated Q1 + hard-vs-soft quantization ablation.

v1 (repeater_n8) was quantizer-limited: Q1 0.887 because its head was fit on the
multi-arm capture layout (domain shift) and unregularized; the relay itself was
lossless (sum(Q1)=sum(Q2)=0.550, root 0.517).  v2 fixes Q1 and ablates the write:

  calib split (120 samples, this exact b2-only layout):
    pass A: forward 0..20, capture lvl1 span-means -> fit Q1 (standardized, C=0.5)
    pass B: forward 0..20, quantize lvl1 with Q1 preds (per arm), run 21..24,
            capture lvl2 span-means -> fit Q2_arm (per arm, on its own child code)
  eval split (120 fresh samples): full repeater per arm, root probe @L27 (3 seeds).

Arms:  hard  span := e(argmax count), norm-matched (v1 behavior, calibrated)
       soft  span := sum_k p_k e(k),  norm-matched (uncertainty-preserving)

Usage:
  python scripts/superquery/probe_repeater2.py --output outputs/superquery/repeater2_n8
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
ARMS = ("hard", "soft")


def fit_head(X, y, C=0.5):
    from sklearn.linear_model import LogisticRegression
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=C).fit((X - mu) / sd, y)
    return clf, mu, sd


def head_prob(head, X):
    clf, mu, sd = head
    return clf.predict_proba((X - mu) / sd), clf.classes_


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    levels = tree_levels(NF, 2)
    lsets = leaf_sets(levels)
    n_nodes = sum(len(g) for g in levels)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_emb = text_model.embed_tokens.weight
    dig_ids = {}
    for k in range(NF + 1):
        ids_k = tok(f"{k}", add_special_tokens=False).input_ids
        assert len(ids_k) == 1
        dig_ids[k] = ids_k[0]

    def prep(sd):
        """Load + tokenize + locate + mask one sample; None on any parse failure."""
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence("steps", q0, states, gold, ROOMS)
            if pe_ is None:
                return None
            evid, _room = pe_
        except Exception:
            return None
        if (not evid and gold != 0) or len(frames) != NF:
            return None
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
            return None
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
        k1 = [int(sum(1 for c in g if c in evid)) for g in levels[0]]
        k2 = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]
        return dict(inputs=inputs, m=m, seq=seq, span_of=span_of, blocks=blocks,
                    fin_start=fin_start, k1=k1, k2=k2, gold=gold)

    def fwd_to(rec, hi_layer, h=None, lo_layer=0):
        """Run layers lo..hi on rec (building emb/pos/mask once, cached in rec)."""
        if "pe" not in rec:
            inputs = rec["inputs"]
            base_pos, _ = rope_fn(inputs["input_ids"],
                                  image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos = reset_positions(base_pos, rec["blocks"], rec["fin_start"]).clone().to(dev)
            emb = text_model.embed_tokens(inputs["input_ids"]).clone()
            img = model.model.get_image_features(inputs["pixel_values"],
                                                 inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb[0, im_mask] = img.to(emb.dtype)
            cos_, sin_ = text_model.rotary_emb(emb, pos)
            rec["pe"] = (cos_.to(emb.dtype), sin_.to(emb.dtype))
            rec["m4"] = rec["m"].to(dev).to(emb.dtype).view(1, 1, rec["seq"], rec["seq"])
            rec["emb"] = emb
        if h is None:
            h = rec["emb"]
        for li in range(lo_layer, hi_layer + 1):
            h = layers[li](h, attention_mask=rec["m4"], position_embeddings=rec["pe"])[0]
        return h

    def span_feats(h, rec, lv):
        return np.stack([h[0, a:b].mean(0).float().cpu().numpy()
                         for gi in range(len(levels[lv]))
                         for a, b in [rec["span_of"][(lv, gi)]]])

    def write_code(h, rec, lv, probs, classes, arm):
        for gi in range(len(levels[lv])):
            a, b = rec["span_of"][(lv, gi)]
            p = probs[gi]
            if arm == "hard":
                v = W_emb[dig_ids[int(classes[int(np.argmax(p))])]].float()
            else:
                v = torch.zeros(W_emb.shape[1], device=dev)
                for ci, k in enumerate(classes):
                    v += float(p[ci]) * W_emb[dig_ids[int(k)]].float()
            v = v.to(h.dtype)
            scale = h[0, a:b].norm(dim=-1).median() / (v.norm() + 1e-6)
            h[0, a:b] = v * scale
        return h

    # ---------- gather samples
    dirs = []
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if len(dirs) >= args.n_calib + args.n_eval:
            break
        dirs.append(sd)
    calib_dirs, eval_dirs = dirs[: args.n_calib], dirs[args.n_calib :]

    t0 = time.time()
    # ---------- calibration pass A: lvl1 states -> Q1
    X1, y1 = [], []
    calib_recs = []
    with torch.no_grad():
        for sd in calib_dirs:
            rec = prep(sd)
            if rec is None:
                continue
            h20 = fwd_to(rec, Q1_L)
            X1.append(span_feats(h20, rec, 0))
            y1 += rec["k1"]
            rec["h20"] = h20   # kept (with pe/m4/emb) for pass B
            calib_recs.append(rec)
    X1 = np.concatenate(X1)
    q1_head = fit_head(X1, np.array(y1))
    print(f"[calib] Q1 fit on {len(y1)} nodes ({len(calib_recs)} samples) "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---------- calibration pass B: per-arm quantized children -> Q2_arm
    q2_head = {}
    X2 = {arm: [] for arm in ARMS}
    y2 = []
    with torch.no_grad():
        for rec in calib_recs:
            probs, classes = head_prob(q1_head, span_feats(rec["h20"], rec, 0))
            for arm in ARMS:
                hq = write_code(rec["h20"].clone(), rec, 0, probs, classes, arm)
                h24 = fwd_to(rec, Q2_L, h=hq, lo_layer=Q1_L + 1)
                X2[arm].append(span_feats(h24, rec, 1))
            y2 += rec["k2"]
            del rec["h20"], rec["pe"], rec["m4"], rec["emb"]
    for arm in ARMS:
        q2_head[arm] = fit_head(np.concatenate(X2[arm]), np.array(y2))
    print(f"[calib] Q2 fit per arm on {len(y2)} nodes {time.time()-t0:.0f}s", flush=True)

    # ---------- eval
    res = {arm: dict(q1=[], q2=[], sum1=[], sum2=[], root=[]) for arm in ARMS}
    golds = []
    n_done = 0
    with torch.no_grad():
        for sd in eval_dirs:
            rec = prep(sd)
            if rec is None:
                continue
            h20 = fwd_to(rec, Q1_L)
            f1 = span_feats(h20, rec, 0)
            probs1, cls1 = head_prob(q1_head, f1)
            p1 = np.array([cls1[int(np.argmax(p))] for p in probs1])
            for arm in ARMS:
                hq = write_code(h20.clone(), rec, 0, probs1, cls1, arm)
                h24 = fwd_to(rec, Q2_L, h=hq, lo_layer=Q1_L + 1)
                f2 = span_feats(h24, rec, 1)
                probs2, cls2 = head_prob(q2_head[arm], f2)
                p2 = np.array([cls2[int(np.argmax(p))] for p in probs2])
                hq2 = write_code(h24, rec, 1, probs2, cls2, arm)
                h27 = fwd_to(rec, TOP_L, h=hq2, lo_layer=Q2_L + 1)
                a, b = rec["span_of"][(2, 0)]
                res[arm]["root"].append(np.concatenate(
                    [h27[0, a:b].mean(0).float().cpu().numpy(),
                     h27[0, b - 1].float().cpu().numpy()]))
                res[arm]["q1"].append(float((p1 == np.array(rec["k1"])).mean()))
                res[arm]["q2"].append(float((p2 == np.array(rec["k2"])).mean()))
                res[arm]["sum1"].append(int(p1.sum()))
                res[arm]["sum2"].append(int(p2.sum()))
            golds.append(rec["gold"])
            del rec
            n_done += 1
            if n_done % 20 == 0:
                print(f"  eval {n_done} {time.time()-t0:.0f}s", flush=True)

    G = np.array(golds)
    n = len(G)
    from sklearn.linear_model import LogisticRegression
    rows = []
    for arm in ARMS:
        q1a = float(np.mean(res[arm]["q1"]))
        q2a = float(np.mean(res[arm]["q2"]))
        s1 = float((np.array(res[arm]["sum1"]) == G).mean())
        s2 = float((np.array(res[arm]["sum2"]) == G).mean())
        print(f"[{arm}] Q1 {q1a:.3f}  sum(Q1) {s1:.3f}  Q2 {q2a:.3f}  sum(Q2) {s2:.3f}",
              flush=True)
        X = np.stack(res[arm]["root"]).astype(np.float32)
        H = X.shape[-1] // 2
        for feat, sl in (("mean", slice(0, H)), ("last", slice(H, 2 * H))):
            accs, pm1, maes = [], [], []
            for seed in range(3):
                idx = np.random.default_rng(seed).permutation(n)
                tr, ev = idx[: n // 2], idx[n // 2:]
                Xtr, Xev = X[tr, sl], X[ev, sl]
                mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
                clf = LogisticRegression(max_iter=2000, C=0.5).fit((Xtr - mu) / sd_, G[tr])
                pr = clf.predict((Xev - mu) / sd_)
                accs.append(float((pr == G[ev]).mean()))
                pm1.append(float((np.abs(pr - G[ev]) <= 1).mean()))
                maes.append(float(np.abs(pr - G[ev]).mean()))
            print(f"[{arm} ROOT {feat}] exact {np.mean(accs):.3f} +-1 {np.mean(pm1):.3f} "
                  f"MAE {np.mean(maes):.2f}", flush=True)
            rows.append([arm, feat, q1a, s1, q2a, s2, np.mean(accs), np.mean(pm1),
                         np.mean(maes)])
    with open(out / "repeater2.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "feat", "q1", "sum_q1", "q2", "sum_q2",
                                  "root_exact", "pm1", "mae"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
