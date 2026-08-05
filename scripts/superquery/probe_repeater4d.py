#!/usr/bin/env python3
"""REPEATER v4d — FINAL two-pass pipeline: best quantizer chain + perfect pass-2.

Pass 1 (vision, one forward): fenced blocks + b=2 tree, Q1@L20 (pair rr 0.994 in
v4b), Q2@L24 -> two predicted half-counts (sum bound 0.980).
Pass 2 (text, ~50 tokens): "Two partial counts are {a} and {b}. What is the total
count?" + force "Answer: " -> model ADDS AND EMITS (template measured 1.000 in
addsanity). Scored on the emitted digit (EM). Model-emitted, end to end.

Usage: python scripts/superquery/probe_repeater4d.py --output outputs/superquery/repeater4d_n8
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
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NF = 8
ROOT = "data/mmred_images_park/seq_len_8/all_uniform"
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
    ap.add_argument("--n-calib", type=int, default=150)
    ap.add_argument("--n-eval", type=int, default=300)
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

    # ---------------- samples
    dirs = []
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if len(dirs) >= args.n_calib + args.n_eval:
            break
        dirs.append(sd)
    calib_dirs, eval_dirs = dirs[: args.n_calib], dirs[args.n_calib :]

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
            n += 1
            if n % 20 == 0:
                print(f"  eval {n} {time.time()-t0:.0f}s", flush=True)

    print(f"[v4d] Q1 {np.mean(q1a):.3f} Q2 {np.mean(q2a):.3f} "
          f"sum1 {s1/n:.3f} sum2 {s2/n:.3f} "
          f"EMIT-EM {em/n:.3f} MAE {np.mean(mae):.2f} "
          f"cond-EM {em_cond/max(em_cond_n,1):.3f} (n={n}) "
          f"emitted: {ctr.most_common(6)}", flush=True)
    with open(out / "repeater4d.csv", "w", newline="") as f:
        csv.writer(f).writerows([["q1", "q2", "sum1", "sum2", "emit_em", "mae",
                                  "cond_em", "n"],
                                 [np.mean(q1a), np.mean(q2a), s1 / n, s2 / n, em / n,
                                  np.mean(mae), em_cond / max(em_cond_n, 1), n]])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
