#!/usr/bin/env python3
"""COMPRESSED single-forward pipeline (Tal's schedule): Q1@16 -> Q2@17/18 (1-2 layer
relays) -> quad-count template codes into tail registers @18/19 -> model adds & emits.
All inside ONE forward. Measures the three unknowns + end-to-end:

  [pair-rr]   pair ridge-round acc at L16,17,18,19,20 (maturity curve gap-fill)
  [relay]     Q2 acc on quad states at +1/+2/+3 layers after Q1 writes @16
  [e2e]       arm A: Q1@16 -> Q2@17 -> registers@18;  arm B: Q1@16 -> Q2@18 ->
              registers@19. Registers = in-situ per-digit template codes (donor
              runs on calib samples, deadline2 recipe). EM on emitted total.

Reference points: two-pass v4d 0.980; deadline2 template acceptance 1.00@16/0.60@20.

Usage: python scripts/superquery/probe_compressed.py --output outputs/superquery/compressed_n8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
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
Q1_L = 16
PAIR_LS = (16, 17, 18, 19, 20)
RELAY_LS = (17, 18, 19)
ARMS = {"A_17_18": (17, 18), "B_18_19": (18, 19)}   # (Q2_L, REG_L)


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


def held_out_acc(X, y, lo, hi):
    accs = []
    for seed in range(3):
        idx = np.random.default_rng(seed).permutation(len(y))
        tr, ev = idx[: len(y) // 2], idx[len(y) // 2:]
        pred = fit_ridge(X[tr], y[tr], lo, hi)
        accs.append(float((pred(X[ev]) == y[ev]).mean()))
    return float(np.mean(accs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-calib", type=int, default=150)
    ap.add_argument("--n-eval", type=int, default=200)
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
    suf_ids_l = [tok(s, add_special_tokens=False).input_ids
                 for s in (SUFFIX, " Answer: ", " Answer:")]

    def prep(sd, a, b):
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
        tail = f" Counts: {a} and {b}. What is the total count? Answer: "
        tail_ids = tok(tail, add_special_tokens=False).input_ids
        force = torch.tensor([tail_ids], device=dev)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], force], 1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(force)], 1)
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
        for aa, bb in sq_raw:
            ext = None
            for si_ in suf_ids_l:
                if ids[bb : bb + len(si_)] == si_:
                    ext = bb + len(si_)
                    break
            if ext is None:
                ok = False
                break
            sq_spans.append((aa, ext))
        if not ok:
            return None
        tail_pos0 = seq - len(tail_ids)
        regs = [tail_pos0 + i for i, t in enumerate(tail_ids)
                if t in (dig[a], dig[b]) and i < len(tail_ids) - 8]
        if len(regs) != 2:
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
                aa, bb = sq_spans[si]
                span_of[(li, gi)] = (aa, bb)
                si += 1
                child_spans = ([rep_spans[c] for c in g] if li == 0
                               else [span_of[(li - 1, c)] for c in g])
                rws = torch.arange(aa, bb)
                m[rws] = MASK_MIN
                cols = torch.cat([prefix_cols] +
                                 [torch.arange(ca, cb) for ca, cb in child_spans])
                m[rws.unsqueeze(1), cols.unsqueeze(0)] = 0.0
                blk = torch.zeros(bb - aa, bb - aa)
                blk.masked_fill_(torch.triu(torch.ones(bb - aa, bb - aa,
                                                       dtype=torch.bool), 1), MASK_MIN)
                m[aa:bb, aa:bb] = blk
        ts = sq_spans[-1][1]
        rows = torch.arange(ts, seq)
        m[rows] = MASK_MIN
        m[rows.unsqueeze(1), prefix_cols.unsqueeze(0)] = 0.0
        blk = torch.zeros(seq - ts, seq - ts)
        blk.masked_fill_(torch.triu(torch.ones(seq - ts, seq - ts, dtype=torch.bool), 1),
                         MASK_MIN)
        m[ts:, ts:] = blk
        k1 = [int(sum(1 for c in g if c in evid)) for g in levels[0]]
        k2 = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]
        return dict(inputs=inputs, m=m, seq=seq, blocks=blocks, span_of=span_of,
                    fin_start=fin_start, regs=regs, k1=k1, k2=k2, gold=gold)

    def write_digits(h, rec, lv, vals):
        for gi, k in enumerate(vals):
            a, b = rec["span_of"][(lv, gi)]
            v = W_emb[dig[int(k)]].to(h.dtype)
            scale = h[0, a:b].norm(dim=-1).median() / (v.norm() + 1e-6)
            h[0, a:b] = v * scale
        return h

    def forward(rec, q1=None, q2=None, q2_L=None, cap_lvl1=(), cap_lvl2=(),
                cap_regs=(), reg_write=None, reg_L=None):
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
        caps1, caps2, capsr = {}, {}, {}
        preds = {}
        for li in range(28):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            def span_X(lv):
                return np.stack([h[0, a:b].mean(0).float().cpu().numpy()
                                 for gi in range(len(levels[lv]))
                                 for a, b in [rec["span_of"][(lv, gi)]]])
            if li == Q1_L and q1 is not None:
                v = q1(span_X(0))
                preds[0] = v.tolist()
                h = write_digits(h, rec, 0, v)
            if q2 is not None and li == q2_L:
                v = q2(span_X(1))
                preds[1] = v.tolist()
                h = write_digits(h, rec, 1, v)
            if li in cap_lvl1:
                caps1[li] = span_X(0)
            if li in cap_lvl2:
                caps2[li] = span_X(1)
            if li in cap_regs:
                capsr[li] = h[0, rec["regs"]].float().cpu()
            if reg_write is not None and li == reg_L:
                for j, p in enumerate(rec["regs"]):
                    h[0, p] = reg_write[j].to(dev).to(h.dtype)
        lg = (final_norm(h[0, -1:]).float() @ W_U.T)[0]
        return int(lg.argmax().item()), preds, caps1, caps2, capsr

    # ---------------- gather
    dirs = []
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if len(dirs) >= args.n_calib + args.n_eval:
            break
        dirs.append(sd)
    calib_dirs, eval_dirs = dirs[: args.n_calib], dirs[args.n_calib :]

    # ---------------- calibration
    t0 = time.time()
    X1 = {L: [] for L in PAIR_LS}
    y1 = []
    crecs = []
    with torch.no_grad():
        for sd in calib_dirs:
            rec = prep(sd, 0, 0)
            if rec is None:
                continue
            _t, _p, caps1, _c2, _cr = forward(rec, cap_lvl1=PAIR_LS)
            for L in PAIR_LS:
                X1[L].append(caps1[L])
            y1 += rec["k1"]
            crecs.append((sd, rec))
    y1_np = np.array(y1)
    rows_out = []
    q1_heads = {}
    for L in PAIR_LS:
        X = np.concatenate(X1[L])
        acc = held_out_acc(X, y1_np, 0, 2)
        q1_heads[L] = fit_ridge(X, y1_np, 0, 2)
        rows_out.append(["pair_rr", L, acc])
        print(f"[pair-rr L{L}] {acc:.3f}", flush=True)
    print(f"[calib] pair heads {time.time()-t0:.0f}s", flush=True)

    # relay: Q1 writes @16 -> lvl2 states at +1/+2/+3
    X2 = {L: [] for L in RELAY_LS}
    y2 = []
    with torch.no_grad():
        for sd, rec in crecs:
            _t, _p, _c1, caps2, _cr = forward(rec, q1=q1_heads[Q1_L],
                                              cap_lvl2=RELAY_LS)
            for L in RELAY_LS:
                X2[L].append(caps2[L])
            y2 += rec["k2"]
    y2_np = np.array(y2)
    q2_heads = {}
    for L in RELAY_LS:
        X = np.concatenate(X2[L])
        acc = held_out_acc(X, y2_np, 0, 4)
        q2_heads[L] = fit_ridge(X, y2_np, 0, 4)
        rows_out.append(["relay_q2", L, acc])
        print(f"[relay Q2@L{L} (+{L-Q1_L})] {acc:.3f}", flush=True)

    # donor templates for register writes at 18, 19 (real gold quad digits in tail)
    tmpl_sum = defaultdict(lambda: torch.zeros(3584))
    tmpl_cnt = defaultdict(int)
    with torch.no_grad():
        for sd, rec0 in crecs:
            a, b = rec0["k2"]
            recA = prep(sd, a, b)
            if recA is None:
                continue
            _t, _p, _c1, _c2, capsr = forward(recA, cap_regs=(18, 19))
            for L in (18, 19):
                for j, k in enumerate((a, b)):
                    tmpl_sum[(k, L)] += capsr[L][j]
                    tmpl_cnt[(k, L)] += 1
    print(f"[calib] templates {time.time()-t0:.0f}s", flush=True)

    # ---------------- eval: compressed end-to-end
    stats = {arm: [0, 0, 0] for arm in ARMS}   # em, sum2_ok, n
    with torch.no_grad():
        for sd in eval_dirs:
            rec = prep(sd, 0, 0)
            if rec is None:
                continue
            for arm, (q2l, regl) in ARMS.items():
                def regw(preds):
                    return [tmpl_sum[(int(k), regl)] / max(tmpl_cnt[(int(k), regl)], 1)
                            for k in preds]
                # run chain once to get preds, then rerun with register write
                # (single run: reg_write needs preds known at regl — preds[1] is
                #  computed at q2l < regl, so do it in ONE run via closure state)
                t, preds, *_ = forward(rec, q1=q1_heads[Q1_L], q2=q2_heads[q2l],
                                       q2_L=q2l)
                pv = preds.get(1)
                if pv is None:
                    continue
                t2, _p2, *_ = forward(rec, q1=q1_heads[Q1_L], q2=q2_heads[q2l],
                                      q2_L=q2l, reg_write=regw(pv), reg_L=regl)
                s = stats[arm]
                s[0] += int(t2 == dig[rec["gold"]])
                s[1] += int(sum(pv) == rec["gold"])
                s[2] += 1
            if sum(stats["A_17_18"][2:]) % 20 == 0:
                print(f"  eval {stats['A_17_18'][2]} {time.time()-t0:.0f}s", flush=True)

    for arm, (q2l, regl) in ARMS.items():
        em, s2, n = stats[arm]
        rows_out.append([f"e2e_{arm}", f"Q2@{q2l},reg@{regl}",
                         em / max(n, 1)])
        print(f"[e2e {arm}] EM {em/max(n,1):.3f} sum2 {s2/max(n,1):.3f} (n={n})",
              flush=True)
    with open(out / "compressed.csv", "w", newline="") as f:
        csv.writer(f).writerows([["metric", "layer", "value"], *rows_out])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
