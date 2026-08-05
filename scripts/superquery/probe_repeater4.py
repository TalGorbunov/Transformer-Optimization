#!/usr/bin/env python3
"""REPEATER v4 "EMIT": the model SAYS the answer — no probe at the output.

Chain: fenced blocks -> quantized tree stages (ridge->round->digit-embedding writes,
calibrated in-layout, PREDICTED counts) -> the tail generation position (native
<|im_start|>assistant + forced "Answer: (") reads ONLY clean vocab codes within its
measured capacity and EMITS the answer token. Scored on the emitted token (EM).

Schedule arms (depth budget vs Q1 quality; pair rr: L16 0.885 / L18 ? / L20 0.983):
  A late : Q1@20 Q2@24, decode-as-root reads lvl2 codes @25-27 (3 layers)
  B early+rootQ: Q1@16 Q2@20 rootQ@24, decode COPIES root code @25-27 (fan-1)
  C early: Q1@16 Q2@20, decode-as-root @21-27 (7 layers)
  D mid  : Q1@18 Q2@22, decode-as-root @23-27 (5 layers)

Usage: python scripts/superquery/probe_repeater4.py --output outputs/superquery/repeater4_n8
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
FORCE_TXT = "Answer: ("
ARMS = {
    #        Q1_L  Q2_L  rootQ_L  tail_reads
    "A_late":  (20, 24, None, "lvl2"),
    "B_rootQ": (16, 20, 24, "root"),
    "C_early": (16, 20, None, "lvl2"),
    "D_mid":   (18, 22, None, "lvl2"),
}


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
    force_ids = tok(FORCE_TXT, add_special_tokens=False).input_ids

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
        # append the forced assistant prefix — emission happens at the LAST position
        force = torch.tensor([force_ids], device=dev)
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
        tail_start = sq_spans[-1][1]
        k1 = [int(sum(1 for c in g if c in evid)) for g in levels[0]]
        k2 = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]
        return dict(inputs=inputs, m=m, seq=seq, span_of=span_of, blocks=blocks,
                    fin_start=fin_start, tail_start=tail_start,
                    prefix_end=int(blocks[0][0]), k1=k1, k2=k2, gold=gold)

    def make_mask(rec, tail_reads):
        """Arm mask: tree rows from rec['m'] + restricted tail rows."""
        m = rec["m"].clone()
        seq = rec["seq"]
        ts = rec["tail_start"]
        rows = torch.arange(ts, seq)
        m[rows] = MASK_MIN
        tgt = ([rec["span_of"][(1, gi)] for gi in range(2)] if tail_reads == "lvl2"
               else [rec["span_of"][(2, 0)]])
        cols = torch.cat([torch.arange(0, rec["prefix_end"])] +
                         [torch.arange(a, b) for a, b in tgt])
        m[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
        blk = torch.zeros(seq - ts, seq - ts)
        blk.masked_fill_(torch.triu(torch.ones(seq - ts, seq - ts, dtype=torch.bool), 1),
                         MASK_MIN)
        m[ts:, ts:] = blk
        return m

    def write_digits(h, rec, lv, vals):
        for gi, k in enumerate(vals):
            a, b = rec["span_of"][(lv, gi)]
            v = W_emb[dig[int(k)]].to(h.dtype)
            scale = h[0, a:b].norm(dim=-1).median() / (v.norm() + 1e-6)
            h[0, a:b] = v * scale
        return h

    def forward(rec, m, stops, heads=None, arm_cfg=None, collect=None):
        """Run 0..27 with quantize interventions. stops: dict layer->action."""
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
        m4 = m.to(dev).to(emb.dtype).view(1, 1, rec["seq"], rec["seq"])
        h = emb
        preds = {}
        for li in range(28):
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
            else:  # quantize
                v = heads[(lv, li)](X)
                preds[lv] = v.tolist()
                h = write_digits(h, rec, lv, v)
        return h, preds

    # ---------------- gather sample dirs
    dirs = []
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if len(dirs) >= args.n_calib + args.n_eval:
            break
        dirs.append(sd)
    calib_dirs, eval_dirs = dirs[: args.n_calib], dirs[args.n_calib :]

    # ---------------- calibration
    t0 = time.time()
    heads = {}
    col: dict = {}
    y1_c, y2_c, gold_c = [], [], []
    calib_recs = []
    with torch.no_grad():
        for sd in calib_dirs:
            rec = prep(sd)
            if rec is None:
                continue
            m = make_mask(rec, "lvl2")
            forward(rec, m, {16: ("capture", 0), 18: ("capture", 0),
                             20: ("capture", 0)}, collect=col)
            y1_c += rec["k1"]
            y2_c += rec["k2"]
            gold_c.append(rec["gold"])
            calib_recs.append(rec)
    y1_np = np.array(y1_c)
    for L in (16, 18, 20):
        X = np.concatenate(col[(0, L)])
        heads[(0, L)] = fit_ridge(X, y1_np.reshape(-1), 0, 2)
    print(f"[calib] Q1 heads @16/18/20 on {len(y1_np)} nodes {time.time()-t0:.0f}s",
          flush=True)
    # chain passes: capture lvl2 (and root) under each schedule's Q1
    col2: dict = {}
    with torch.no_grad():
        for rec in calib_recs:
            m = make_mask(rec, "lvl2")
            # early chain: Q1@16 -> capture lvl2@20; Q2@20 -> capture root@24
            forward(rec, m, {16: ("quantize", 0), 20: ("capture", 1)}, heads=heads,
                    collect=col2)
            # mid chain: Q1@18 -> capture lvl2@22
            forward(rec, m, {18: ("quantize", 0), 22: ("capture", 1)}, heads=heads,
                    collect=col2)
            # late chain: Q1@20 -> capture lvl2@24
            forward(rec, m, {20: ("quantize", 0), 24: ("capture", 1)}, heads=heads,
                    collect=col2)
    y2_np = np.array(y2_c)
    for L in (20, 22, 24):
        heads[(1, L)] = fit_ridge(np.concatenate(col2[(1, L)]), y2_np.reshape(-1), 0, 4)
    print(f"[calib] Q2 heads @20/22/24 {time.time()-t0:.0f}s", flush=True)
    # rootQ (arm B): early chain with Q1@16+Q2@20 -> capture root@24
    col3: dict = {}
    with torch.no_grad():
        for rec in calib_recs:
            m = make_mask(rec, "root")
            forward(rec, m, {16: ("quantize", 0), 20: ("quantize", 1),
                             24: ("capture", 2)}, heads=heads, collect=col3)
    heads[(2, 24)] = fit_ridge(np.concatenate(col3[(2, 24)]),
                               np.array(gold_c), 0, 8)
    print(f"[calib] rootQ head @24 {time.time()-t0:.0f}s", flush=True)

    # ---------------- eval: emission per arm
    res = {a: dict(em=0, em_r=0, dig=0, n=0, q1=[], q2=[], s1=0, s2=0)
           for a in ARMS}
    top_ctr = {a: Counter() for a in ARMS}
    n_done = 0
    with torch.no_grad():
        for sd in eval_dirs:
            rec = prep(sd)
            if rec is None:
                continue
            for arm, (q1l, q2l, rql, tail_reads) in ARMS.items():
                m = make_mask(rec, tail_reads)
                stops = {q1l: ("quantize", 0), q2l: ("quantize", 1)}
                if rql is not None:
                    stops[rql] = ("quantize", 2)
                h, preds = forward(rec, m, stops, heads=heads)
                lg = (final_norm(h[0, -1:]).float() @ W_U.T)[0]
                top1 = int(lg.argmax().item())
                r = res[arm]
                r["em"] += int(top1 == dig[rec["gold"]])
                r["em_r"] += int(int(lg[dig_ids].argmax().item()) == rec["gold"])
                r["dig"] += int(top1 in dig_set)
                r["q1"].append(float((np.array(preds[0]) == np.array(rec["k1"])).mean()))
                r["q2"].append(float((np.array(preds[1]) == np.array(rec["k2"])).mean()))
                r["s1"] += int(sum(preds[0]) == rec["gold"])
                r["s2"] += int(sum(preds[1]) == rec["gold"])
                r["n"] += 1
                top_ctr[arm][tok.decode([top1])] += 1
            n_done += 1
            if n_done % 20 == 0:
                print(f"  eval {n_done} {time.time()-t0:.0f}s", flush=True)

    rows = []
    for arm in ARMS:
        r = res[arm]
        n = max(r["n"], 1)
        tops = "; ".join(f"{w!r}:{c}" for w, c in top_ctr[arm].most_common(4))
        rows.append([arm, np.mean(r["q1"]), np.mean(r["q2"]), r["s1"] / n, r["s2"] / n,
                     r["em"] / n, r["em_r"] / n, r["dig"] / n, n])
        print(f"[{arm}] Q1 {np.mean(r['q1']):.3f} Q2 {np.mean(r['q2']):.3f} "
              f"sum1 {r['s1']/n:.3f} sum2 {r['s2']/n:.3f} "
              f"EMIT-EM {r['em']/n:.3f} (restricted {r['em_r']/n:.3f}, "
              f"top1-digit {r['dig']/n:.3f}) tops: {tops}", flush=True)
    with open(out / "repeater4.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "q1", "q2", "sum_q1", "sum_q2", "emit_em",
                                  "emit_restricted", "top1_digit_frac", "n"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
