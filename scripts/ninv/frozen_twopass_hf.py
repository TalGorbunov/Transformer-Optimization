#!/usr/bin/env python3
"""FROZEN two-pass on MMReD-HF N=8 — the missing frozen headline (ninv campaign).

COPY of scripts/superquery/probe_repeater4d.py (the park v4d anchor: EMIT-EM 0.980
@N=8) with ONLY the data source and the reporting changed, so the number is directly
comparable to the anchor:
  - samples via scripts/ninv/load_hf_sample (HF room vocab, steps_in_room filter);
    calibration = seq_len_8_train_steps_in_room (200 dirs), eval =
    seq_len_8_test (the untouched 50-dir benchmark pool)
  - reporting adds the majority baseline (predict 0), per-gold breakdown, and
    conditional EM, per the campaign's skew-reporting directive
  - --resize is a knob (park anchor was 392; the 2026-08-10 margin finding says
    512 lifts HF verdict margins — run BOTH for the comparability x margin grid)
PIPELINE IS OTHERWISE BYTE-FAITHFUL, including NO node posreset (the anchor
predates it; heads are calibrated and evaluated at the same N, so node-position
drift cannot leak here). Zero training; ridge heads are calibration, not learning.

Pass 1 (vision, one forward): fenced blocks + b=2 tree, Q1@L20 quantizes pair
counts into digit tokens in-place, Q2@L24 reads the two half-counts.
Pass 2 (text, ~50 tokens): "Two partial counts are {a} and {b}. What is the total
count?" + forced "Answer: " -> the model ADDS AND EMITS. Scored on emitted EM.

Usage: python scripts/ninv/frozen_twopass_hf.py --resize 512 --output outputs/ninv/<ts>
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))

sys.path.insert(0, str(_REPO / "scripts/ninv"))
from probe_tree import leaf_sets, tree_levels  # noqa: E402

from load_hf_sample import evidence_bits, iter_hf_sample_dirs, load_hf_sample  # noqa: E402

from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.fencing import (  # noqa: E402
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    reset_positions,
)
from gnnformer.runtime import (  # noqa: E402
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NF = 8
CALIB_ROOT = "data/mmred_hf/dirs/seq_len_8_train_steps_in_room"
EVAL_ROOT = "data/mmred_hf/dirs/seq_len_8_test"
SUFFIX = " Answer: ("
Q1_L, Q2_L = 20, 24
P2_TMPL = "Two partial counts are {a} and {b}. What is the total count?"
P2_FORCE = "Answer: "


def fit_ridge(X, y, lo, hi):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    p = PCA(n_components=min(200, Xs.shape[0] - 1), random_state=0).fit(Xs)
    rg = Ridge(alpha=10.0).fit(p.transform(Xs), y)
    def predict(Xn):
        Xn = (Xn - mu) / sd
        return np.clip(np.round(rg.predict(p.transform(Xn))), lo, hi).astype(int)
    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-calib", type=int, default=200, help="strided from the train pool")
    ap.add_argument("--n-eval", type=int, default=50, help="the whole test pool")
    ap.add_argument("--resize", type=int, default=512,
                    help="park anchor was 392; 512 = the margin-finding arm")
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
    final_norm = text_model.norm
    dev = model.device
    W_U = dequantize_linear_weight(model.lm_head).float().to(dev)
    W_emb = text_model.embed_tokens.weight
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dig = {k: tok(f"{k}", add_special_tokens=False).input_ids[0] for k in range(NF + 1)}
    dig_ids = torch.tensor([dig[k] for k in range(NF + 1)], device=dev)
    dig_set = set(dig_ids.tolist())
    suf_ids_l = [tok(s, add_special_tokens=False).input_ids
                 for s in (SUFFIX, " Answer: ", " Answer:")]

    def prep(sd):
        try:
            # HF adapter path: own room vocab (Hallway, no Park); per-sample
            # re-assert sum(bits) == gold so a bad sample is skipped, not mislabelled
            _sid, frames, q0, states, a0 = load_hf_sample(sd)
            gold = int(str(a0).strip())
            bits = evidence_bits(q0, states)
            if bits is None or sum(bits) != gold:
                return None
            evid = {t for t, b in enumerate(bits) if b}
        except Exception:
            return None
        if len(frames) != NF:
            return None
        frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        for _ in range(n_nodes):
            content.append({"type": "text", "text": q0})
            content.append({"type": "text", "text": SUFFIX})
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
        sq_raw = spans[NF + 1 :]
        sq_spans, ok = [], True
        for a, b in sq_raw:
            ext = None
            for si_ in suf_ids_l:
                if ids[b : b + len(si_)] == si_:
                    ext = b + len(si_)
                    break
            if ext is None:
                ok = False
                break
            sq_spans.append((a, ext))
        if not ok:
            return None
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

    def write_digits(h, rec, lv, vals):
        for gi, k in enumerate(vals):
            a, b = rec["span_of"][(lv, gi)]
            v = W_emb[dig[int(k)]].to(h.dtype)
            scale = h[0, a:b].norm(dim=-1).median() / (v.norm() + 1e-6)
            h[0, a:b] = v * scale
        return h

    def forward(rec, stops, heads=None, collect=None, top_layer=27):
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
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        m4 = rec["m"].to(dev).to(emb.dtype).view(1, 1, rec["seq"], rec["seq"])
        h = emb
        preds = {}
        for li in range(top_layer + 1):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            act = stops.get(li)
            if act is None:
                continue
            kind, lv = act
            X = np.stack([h[0, a:b].mean(0).float().cpu().numpy()
                          for gi in range(len(levels[lv]))
                          for a, b in [rec["span_of"][(lv, gi)]]])
            if kind == "capture":
                collect.setdefault((lv, li), []).append(X)
            else:
                v = heads[(lv, li)](X)
                preds[lv] = v.tolist()
                h = write_digits(h, rec, lv, v)
        return preds

    def pass2_emit(a, b):
        q = P2_TMPL.format(a=a, b=b)
        it = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": q}]}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        it = move_to_device(it, dev)
        f2 = torch.tensor([tok(P2_FORCE, add_special_tokens=False).input_ids],
                          device=dev)
        it["input_ids"] = torch.cat([it["input_ids"], f2], 1)
        it["attention_mask"] = torch.cat([it["attention_mask"],
                                          torch.ones_like(f2)], 1)
        with torch.no_grad():
            for _s in range(4):
                h = model(**it, output_hidden_states=True).hidden_states[-1][0, -1]
                lg = (final_norm(h.unsqueeze(0)).float() @ W_U.T)[0]
                t = int(lg.argmax().item())
                if t in dig_set:
                    return t
                nt = torch.tensor([[t]], device=dev)
                it["input_ids"] = torch.cat([it["input_ids"], nt], 1)
                it["attention_mask"] = torch.cat([it["attention_mask"],
                                                  torch.ones_like(nt)], 1)
        return -1

    # ---------------- samples: calib from the TRAIN pool, eval from the untouched
    # TEST pool (disjoint by construction — no shared-pool split needed)
    pool = iter_hf_sample_dirs(Path(CALIB_ROOT))
    # stride, never head-slice: dirs sort K0-first (the 2026-08-09/10 trap x3)
    if 0 < args.n_calib < len(pool):
        pool = pool[:: max(1, len(pool) // args.n_calib)]
    calib_dirs = pool[: args.n_calib]
    eval_dirs = iter_hf_sample_dirs(Path(EVAL_ROOT))[: args.n_eval]
    print(f"[pools] calib {len(calib_dirs)} dirs from {CALIB_ROOT} | "
          f"eval {len(eval_dirs)} dirs from {EVAL_ROOT}", flush=True)

    # ---------------- calibration: Q1@20, then Q2@24 under quantized pairs
    t0 = time.time()
    heads = {}
    col: dict = {}
    y1_c, y2_c = [], []
    recs = []
    with torch.no_grad():
        for sd in calib_dirs:
            rec = prep(sd)
            if rec is None:
                continue
            forward(rec, {Q1_L: ("capture", 0)}, collect=col, top_layer=Q1_L)
            y1_c += rec["k1"]
            y2_c += rec["k2"]
            recs.append(rec)
    heads[(0, Q1_L)] = fit_ridge(np.concatenate(col[(0, Q1_L)]),
                                 np.array(y1_c).reshape(-1), 0, 2)
    print(f"[calib] Q1@{Q1_L} on {len(y1_c)} nodes {time.time()-t0:.0f}s", flush=True)
    col2: dict = {}
    with torch.no_grad():
        for rec in recs:
            forward(rec, {Q1_L: ("quantize", 0), Q2_L: ("capture", 1)},
                    heads=heads, collect=col2, top_layer=Q2_L)
    heads[(1, Q2_L)] = fit_ridge(np.concatenate(col2[(1, Q2_L)]),
                                 np.array(y2_c).reshape(-1), 0, 4)
    print(f"[calib] Q2@{Q2_L} {time.time()-t0:.0f}s", flush=True)

    # ---------------- eval: chain -> pass 2 -> emitted answer
    em = em_cond_n = em_cond = s1 = s2 = n = 0
    q1a, q2a = [], []
    mae = []
    ctr = Counter()
    per_gold: dict = {}
    golds_seen = []
    with torch.no_grad():
        for sd in eval_dirs:
            rec = prep(sd)
            if rec is None:
                continue
            preds = forward(rec, {Q1_L: ("quantize", 0), Q2_L: ("quantize", 1)},
                            heads=heads, top_layer=Q2_L)
            a, b = preds[1]
            t = pass2_emit(a, b)
            val = ({v: k for k, v in dig.items()}.get(t, -1))
            em += int(val == rec["gold"])
            mae.append(abs(val - rec["gold"]) if val >= 0 else 9)
            s1 += int(sum(preds[0]) == rec["gold"])
            s2 += int(a + b == rec["gold"])
            if a + b == rec["gold"]:
                em_cond_n += 1
                em_cond += int(val == rec["gold"])
            q1a.append(float((np.array(preds[0]) == np.array(rec["k1"])).mean()))
            q2a.append(float((np.array(preds[1]) == np.array(rec["k2"])).mean()))
            ctr[val] += 1
            golds_seen.append(rec["gold"])
            pg = per_gold.setdefault(rec["gold"], [0, 0])
            pg[1] += 1
            pg[0] += int(val == rec["gold"])
            n += 1
            if n % 20 == 0:
                print(f"  eval {n} {time.time()-t0:.0f}s", flush=True)

    # skew-aware reporting (campaign directive): raw EM alone flatters under a
    # zero-heavy gold prior — always show the predict-0 baseline + per-gold cells
    maj = float(np.mean([g == 0 for g in golds_seen]))
    nz = [(v, c) for v, c in per_gold.items() if v > 0]
    em_nz = (sum(c[0] for _, c in nz) / max(sum(c[1] for _, c in nz), 1)) if nz else float("nan")
    print(f"[twopass-hf @{args.resize}px] Q1 {np.mean(q1a):.3f} Q2 {np.mean(q2a):.3f} "
          f"sum1 {s1/n:.3f} sum2 {s2/n:.3f} "
          f"EMIT-EM {em/n:.3f} MAE {np.mean(mae):.2f} "
          f"cond-EM {em_cond/max(em_cond_n,1):.3f} (n={n}) "
          f"| majority(predict-0) {maj:.3f} EM-on-gold>0 {em_nz:.3f} "
          f"| park v4d anchor 0.980@392", flush=True)
    print("per-gold: " + " ".join(f"g{g}:{c[0]}/{c[1]}"
                                  for g, c in sorted(per_gold.items())), flush=True)
    print(f"emitted: {ctr.most_common(8)}", flush=True)
    with open(out / "twopass_hf.csv", "w", newline="") as f:
        csv.writer(f).writerows([["resize", "q1", "q2", "sum1", "sum2", "emit_em",
                                  "mae", "cond_em", "majority0", "em_gold_nz", "n"],
                                 [args.resize, np.mean(q1a), np.mean(q2a), s1 / n,
                                  s2 / n, em / n, np.mean(mae),
                                  em_cond / max(em_cond_n, 1), maj, em_nz, n]])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
