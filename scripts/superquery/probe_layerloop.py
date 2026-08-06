#!/usr/bin/env python3
"""LAYER LOOPING (Tal, 2026-08-06): repeat frozen layers to stretch the window
between information-readiness and the emission deadline.

Layer schedule variants (inference-time only, frozen weights):
  base      0..27 (reference)
  L16x2     ...,15,16,16,17,...      (+1 step in the aggregation window)
  L16x3     ...,15,16,16,16,17,...
  B1316x2   ...,12,[13,14,15,16]x2,17,...  (+4 steps)

Per variant: (a) pair rr at the exit of the repeated region (does maturity rise?),
(b) register template-acceptance when writing right after the repeats (does the
deadline follow layer INDEX or PROCESSING COUNT?), (c) compressed e2e:
Q1 at repeat-exit -> Q2 one step later -> registers one step after -> EM.
Also logs residual-norm ratio at repeat exit (explosion check).

Usage: python scripts/superquery/probe_layerloop.py --output outputs/superquery/layerloop_n8
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
# schedule: list of layer indices to run, and the STEP index right after the
# repeated region where Q1 fires; Q2 = next step; registers = step after that.
SCHEDULES = {
    "base":    (list(range(28)), 16),
    "L16x2":   (list(range(17)) + [16] + list(range(17, 28)), 17),
    "L16x3":   (list(range(17)) + [16, 16] + list(range(17, 28)), 18),
    "B1316x2": (list(range(17)) + [13, 14, 15, 16] + list(range(17, 28)), 20),
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
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=150)
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

    def forward(rec, sched, q1_step, q1=None, q2=None, cap1=(), cap2=(),
                cap_regs=(), reg_write=None, reg_step=None):
        """Run the layer SCHEDULE (list of layer indices). Steps are indices into
        the schedule; Q1 fires at step q1_step, Q2 at q1_step+1, regs at +2."""
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
        norm0 = None
        for step, li in enumerate(sched):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            def span_X(lv):
                return np.stack([h[0, a:b].mean(0).float().cpu().numpy()
                                 for gi in range(len(levels[lv]))
                                 for a, b in [rec["span_of"][(lv, gi)]]])
            if step == q1_step:
                norm0 = float(h[0].norm(dim=-1).median())
                if q1 is not None:
                    v = q1(span_X(0))
                    preds[0] = v.tolist()
                    h = write_digits(h, rec, 0, v)
                if cap1 == "q1":
                    caps1[step] = span_X(0)
            if q2 is not None and step == q1_step + 1:
                v = q2(span_X(1))
                preds[1] = v.tolist()
                h = write_digits(h, rec, 1, v)
            if cap2 == "q2" and step == q1_step + 1:
                caps2[step] = span_X(1)
            if cap_regs == "reg" and step == q1_step + 2:
                capsr[step] = h[0, rec["regs"]].float().cpu()
            if reg_write is not None and step == reg_step:
                for j, p in enumerate(rec["regs"]):
                    h[0, p] = reg_write[j].to(dev).to(h.dtype)
        lg = (final_norm(h[0, -1:]).float() @ W_U.T)[0]
        return int(lg.argmax().item()), preds, caps1, caps2, capsr, norm0

    # ---------------- gather
    dirs = []
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if len(dirs) >= args.n_calib + args.n_eval:
            break
        dirs.append(sd)
    calib_dirs, eval_dirs = dirs[: args.n_calib], dirs[args.n_calib :]

    rows_out = []
    t0 = time.time()
    for name, (sched, q1_step) in SCHEDULES.items():
        # calib: capture pair states at q1_step exit + templates + fit heads
        X1, y1, y2 = [], [], []
        crecs = []
        with torch.no_grad():
            for sd in calib_dirs:
                rec = prep(sd, 0, 0)
                if rec is None:
                    continue
                _t, _p, caps1, _c2, _cr, nrm = forward(
                    rec, sched, q1_step, cap1="q1")
                X1.append(caps1[q1_step])
                y1 += rec["k1"]
                y2 += rec["k2"]
                crecs.append((sd, rec))
        X1 = np.concatenate(X1)
        y1_np, y2_np = np.array(y1), np.array(y2)
        pair_acc = held_out_acc(X1, y1_np, 0, 2)
        q1_head = fit_ridge(X1, y1_np, 0, 2)
        # relay: Q2 states one step after Q1 writes
        X2 = []
        with torch.no_grad():
            for sd, rec in crecs:
                _t, _p, _c1, caps2, _cr, _n = forward(
                    rec, sched, q1_step, q1=q1_head, cap2="q2")
                X2.append(caps2[q1_step + 1])
        X2 = np.concatenate(X2)
        relay_acc = held_out_acc(X2, y2_np, 0, 4)
        q2_head = fit_ridge(X2, y2_np, 0, 4)
        # templates: donor runs (real digits) captured at reg step
        tmpl_sum = defaultdict(lambda: torch.zeros(3584))
        tmpl_cnt = defaultdict(int)
        with torch.no_grad():
            for sd, rec0 in crecs:
                a, b = rec0["k2"]
                recA = prep(sd, a, b)
                if recA is None:
                    continue
                _t, _p, _c1, _c2, capsr, _n = forward(
                    recA, sched, q1_step, cap_regs="reg")
                for j, k in enumerate((a, b)):
                    tmpl_sum[k] += capsr[q1_step + 2][j]
                    tmpl_cnt[k] += 1
        # eval: e2e
        em = s2 = n = 0
        with torch.no_grad():
            for sd in eval_dirs:
                rec = prep(sd, 0, 0)
                if rec is None:
                    continue
                _t, preds, *_ = forward(rec, sched, q1_step, q1=q1_head,
                                        q2=q2_head)
                pv = preds.get(1)
                if pv is None:
                    continue
                regw = [tmpl_sum[int(k)] / max(tmpl_cnt[int(k)], 1) for k in pv]
                t2, _p, *_ = forward(rec, sched, q1_step, q1=q1_head, q2=q2_head,
                                     reg_write=regw, reg_step=q1_step + 2)
                em += int(t2 == dig[rec["gold"]])
                s2 += int(sum(pv) == rec["gold"])
                n += 1
        rows_out.append([name, len(sched), q1_step, pair_acc, relay_acc,
                         s2 / max(n, 1), em / max(n, 1), n])
        print(f"[{name}] sched_len {len(sched)} q1_step {q1_step} "
              f"pair-rr {pair_acc:.3f} relay {relay_acc:.3f} "
              f"sum2 {s2/max(n,1):.3f} e2e-EM {em/max(n,1):.3f} (n={n}) "
              f"{time.time()-t0:.0f}s", flush=True)
    with open(out / "layerloop.csv", "w", newline="") as f:
        csv.writer(f).writerows([["schedule", "sched_len", "q1_step", "pair_rr",
                                  "relay", "sum2", "e2e_em", "n"], *rows_out])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
