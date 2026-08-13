#!/usr/bin/env python3
"""REPEATER v4c: emission via TAIL REGISTERS / TWO-PASS (v4b post-mortem).

v4b: quantized tree reached sum-bounds 0.967-0.980 in-forward, but the decode
position would not read codes written into FAR spans (EM ~0.13, emits '1').
Emission copying wants NEARBY, token-like sources. Arms:

  R2_gold  tail text carries two placeholder digits ("Counts: 0 and 0"); at L20
           they are overwritten with GOLD lvl2-count digit codes -> reader emits
           total @21-27. UPPER BOUND: isolates local-register emission itself.
  R2_pred  same, codes = PREDICTED lvl2 counts (Q1@16 -> Q2@20 chain).
  R1_pred  single register ("Count: 0") overwritten @24 with predicted ROOT total
           (fan-1 pure copy from 3 tokens away).
  TP_pred  TWO-PASS: prefill computes predicted lvl2 counts; pass 2 is a tiny TEXT
           prompt presenting them as REAL tokens ("first part: X, second part: Y")
           -> model sums and emits. The guaranteed-fallback deliverable.

Usage: python scripts/superquery/probe_repeater4c.py --output outputs/superquery/repeater4c_n8
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
Q1_L, Q2_L, R1_L = 16, 20, 24
TAIL2 = " Counts: 0 and 0. Answer: ( "     # two registers (positions of the 0s)
TAIL1 = " Count: 0. Answer: ( "            # one register


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

    def prep(sd, tail_txt):
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
        tail_ids = tok(tail_txt, add_special_tokens=False).input_ids
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
        # register positions: the '0' digit tokens inside the appended tail
        tail_pos0 = seq - len(tail_ids)
        regs = [tail_pos0 + i for i, t in enumerate(tail_ids) if t == dig[0]]
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
        # tail rows: prefix + own causal ONLY (codes live IN the tail registers)
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
        return dict(inputs=inputs, m=m, seq=seq, span_of=span_of, blocks=blocks,
                    fin_start=fin_start, regs=regs, k1=k1, k2=k2, gold=gold, q0=q0)

    def write_digit_at(h, p, k):
        v = W_emb[dig[int(k)]].to(h.dtype)
        scale = h[0, p].norm() / (v.norm() + 1e-6)
        h[0, p] = v * scale
        return h

    def forward(rec, stops, heads=None, collect=None, reg_vals=None, reg_layer=None,
                reg_from_lv=1):
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
        for li in range(28):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            act = stops.get(li)
            if act is not None:
                kind, lv = act
                X = np.stack([h[0, a:b].mean(0).float().cpu().numpy()
                              for gi in range(len(levels[lv]))
                              for a, b in [rec["span_of"][(lv, gi)]]])
                if kind == "capture":
                    collect.setdefault((lv, li), []).append(X)
                else:
                    v = heads[(lv, li)](X)
                    preds[lv] = v.tolist()
            if reg_layer is not None and li == reg_layer:
                vals = (reg_vals if reg_vals is not None
                        else preds.get(reg_from_lv, [0, 0]))
                for p, k in zip(rec["regs"], vals):
                    h = write_digit_at(h, p, k)
        lg = (final_norm(h[0, -1:]).float() @ W_U.T)[0]
        return lg, preds

    # ---------------- samples
    dirs = []
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if len(dirs) >= args.n_calib + args.n_eval:
            break
        dirs.append(sd)
    calib_dirs, eval_dirs = dirs[: args.n_calib], dirs[args.n_calib :]

    # ---------------- calibration (Q1@16, Q2@20 under quantized children, root@24)
    t0 = time.time()
    heads = {}
    col: dict = {}
    y1_c, y2_c, gold_c = [], [], []
    recs = []
    with torch.no_grad():
        for sd in calib_dirs:
            rec = prep(sd, TAIL2)
            if rec is None:
                continue
            forward(rec, {Q1_L: ("capture", 0)}, collect=col)
            y1_c += rec["k1"]
            y2_c += rec["k2"]
            gold_c.append(rec["gold"])
            recs.append(rec)
    heads[(0, Q1_L)] = fit_ridge(np.concatenate(col[(0, Q1_L)]),
                                 np.array(y1_c).reshape(-1), 0, 2)
    print(f"[calib] Q1@{Q1_L} on {len(y1_c)} nodes {time.time()-t0:.0f}s", flush=True)
    col2: dict = {}
    with torch.no_grad():
        for rec in recs:
            forward(rec, {Q1_L: ("quantize", 0), Q2_L: ("capture", 1),
                          R1_L: ("capture", 2)}, heads=heads, collect=col2)
    heads[(1, Q2_L)] = fit_ridge(np.concatenate(col2[(1, Q2_L)]),
                                 np.array(y2_c).reshape(-1), 0, 4)
    heads[(2, R1_L)] = fit_ridge(np.concatenate(col2[(2, R1_L)]),
                                 np.array(gold_c), 0, 8)
    print(f"[calib] Q2@{Q2_L} + rootQ@{R1_L} {time.time()-t0:.0f}s", flush=True)

    # ---------------- eval
    res = {a: dict(em=0, dig=0, n=0, s2=0) for a in
           ("R2_gold", "R2_pred", "R1_pred", "TP_pred")}
    ctr = {a: Counter() for a in res}
    def score(arm, lg, gold):
        r = res[arm]
        top1 = int(lg.argmax().item())
        r["em"] += int(top1 == dig[gold])
        r["dig"] += int(top1 in dig_set)
        r["n"] += 1
        ctr[arm][tok.decode([top1])] += 1
    n_done = 0
    with torch.no_grad():
        for sd in eval_dirs:
            rec = prep(sd, TAIL2)
            if rec is None:
                continue
            k2g = rec["k2"]
            # R2_gold: registers get GOLD lvl2 counts (no quantizers at all)
            lg, _ = forward(rec, {}, reg_vals=k2g, reg_layer=Q2_L)
            score("R2_gold", lg, rec["gold"])
            # R2_pred: full chain Q1@16 -> Q2@20 -> registers @20
            lg, preds = forward(rec, {Q1_L: ("quantize", 0), Q2_L: ("quantize", 1)},
                                heads=heads, reg_layer=Q2_L)
            score("R2_pred", lg, rec["gold"])
            res["R2_pred"]["s2"] += int(sum(preds[1]) == rec["gold"])
            # TP_pred: pass 2 = pure text with predicted counts as REAL tokens
            p2 = (f"In the first half of a video, a person appears in the room "
                  f"{preds[1][0]} times. In the second half, {preds[1][1]} times. "
                  f"How many times in total?")
            it = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": p2}]}],
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            it = move_to_device(it, dev)
            f2 = torch.tensor([tok("Answer: ( ", add_special_tokens=False).input_ids],
                              device=dev)
            it["input_ids"] = torch.cat([it["input_ids"], f2], 1)
            it["attention_mask"] = torch.cat([it["attention_mask"],
                                              torch.ones_like(f2)], 1)
            hh = model(**it, output_hidden_states=True).hidden_states[-1][0, -1]
            lg2 = (final_norm(hh.unsqueeze(0)).float() @ W_U.T)[0]
            score("TP_pred", lg2, rec["gold"])
            # R1_pred: single register, root code @24
            rec1 = prep(sd, TAIL1)
            if rec1 is not None:
                lg, _ = forward(rec1, {Q1_L: ("quantize", 0), Q2_L: ("quantize", 1),
                                       R1_L: ("quantize", 2)},
                                heads=heads, reg_layer=R1_L, reg_from_lv=2)
                score("R1_pred", lg, rec1["gold"])
            n_done += 1
            if n_done % 20 == 0:
                print(f"  eval {n_done} {time.time()-t0:.0f}s", flush=True)

    rows = []
    for arm in res:
        r = res[arm]
        n = max(r["n"], 1)
        tops = "; ".join(f"{w!r}:{c}" for w, c in ctr[arm].most_common(4))
        rows.append([arm, r["em"] / n, r["dig"] / n,
                     r["s2"] / n if arm == "R2_pred" else "", n])
        print(f"[{arm}] EMIT-EM {r['em']/n:.3f} top1-digit {r['dig']/n:.3f} "
              f"{'sum2 %.3f' % (r['s2']/n) if arm == 'R2_pred' else ''} (n={n}) "
              f"tops: {tops}", flush=True)
    with open(out / "repeater4c.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "emit_em", "top1_digit", "sum2", "n"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
