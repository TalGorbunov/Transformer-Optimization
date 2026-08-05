#!/usr/bin/env python3
"""Quantized-hop PATCH experiment: why does the count die at the tree hop?

b=2 tree at N=8 (levels 4-2-1). Level-1 SQ states hold pair counts (0.95 linear
@L20); the level-2 read only transfers ~0.4. Four arms separate ANALOG NOISE from
OUT-OF-DISTRIBUTION CODE: at L_PATCH=20 the level-1 SQ spans are overwritten with
progressively cleaner codes (using GOLD pair counts — teacher-forced upper bound),
then layers 21..27 run and level-2/3 count decodability is probed at L24/27.

  P0  control, untouched                       (known: lvl2 ~0.4)
  P1  count-class CENTROID of real lvl1 states (same subspace, denoised)
      -> if this rescues lvl2: the hop problem is NOISE (fix: denoiser/adapter)
  P2  token EMBEDDING of the count word, norm-matched to L20
      -> if only this rescues: reads need VOCAB-space codes (fix: quantizer to
         token space — the in-model re-quantization repeater)
  P3  input-level text patch: child span embeddings = count-word embedding from
      layer 0 (ceiling/sanity — reading plain text must work)

Usage:
  python scripts/superquery/probe_patch.py --output outputs/superquery/patch_n8
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

from probe_tree import _std_fit, leaf_sets, tree_levels  # noqa: E402

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
L_PATCH = 20
READ = (24, 27)
ARMS_P = ("P0", "P1", "P2", "P3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--centroid-npz",
                    default="outputs/superquery/capture_16_64_128790/feats_N8.npz")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    levels = tree_levels(NF, 2)          # [4, 2, 1]
    lsets = leaf_sets(levels)
    n_nodes = sum(len(g) for g in levels)  # 7

    # ---- P1 centroids from the banked frozen capture (lvl1 span-mean @L20)
    d = np.load(args.centroid_npz)
    X0, Y0 = d["b2|0|20|mean"].astype(np.float32), d["Y"]
    cnt0 = np.stack([[int(Y0[s, g].sum()) for g in levels[0]] for s in range(len(Y0))])
    cent = {k: X0.reshape(-1, X0.shape[-1])[cnt0.reshape(-1) == k].mean(0)
            for k in (0, 1, 2)}
    print(f"centroids from {args.centroid_npz}: "
          f"{[int((cnt0 == k).sum()) for k in (0, 1, 2)]} nodes per class", flush=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_emb = text_model.embed_tokens.weight
    cnt_tok = {}
    for k in (0, 1, 2):
        ids_k = tok(f" {k}", add_special_tokens=False).input_ids
        assert len(ids_k) == 1, f"count word ' {k}' not a single token"
        cnt_tok[k] = ids_k[0]

    feats: dict = {}     # (P, lv, L, feat) -> list of (nodes, H)
    ys, golds = [], []
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
                rows = torch.arange(a, b)
                m[rows] = MASK_MIN
                cols = torch.cat([prefix_cols] +
                                 [torch.arange(ca, cb) for ca, cb in child_spans])
                m[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
                blk = torch.zeros(b - a, b - a)
                blk.masked_fill_(torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1),
                                 MASK_MIN)
                m[a:b, a:b] = blk

        k_pair = [int(sum(1 for c in g if c in evid)) for g in levels[0]]

        def capture(h, li, P):
            hf = h[0].float().cpu()
            for lv in (1, 2):
                for feat in ("mean", "last"):
                    fs = []
                    for gi in range(len(levels[lv])):
                        a, b = span_of[(lv, gi)]
                        fs.append(hf[a:b].mean(0) if feat == "mean" else hf[b - 1])
                    feats.setdefault((P, lv, li, feat), []).append(
                        torch.stack(fs).numpy())

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

            def run(h, lo, hi_, P):
                for li in range(lo, hi_ + 1):
                    h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
                    if li in READ:
                        capture(h, li, P)
                return h

            h20 = run(emb, 0, L_PATCH, "P0__pre")   # captures nothing (READ>20)
            # P0
            run(h20, L_PATCH + 1, max(READ), "P0")
            # P1 centroid / P2 token-embedding patches at L20
            for P in ("P1", "P2"):
                hp = h20.clone()
                for gi in range(len(levels[0])):
                    a, b = span_of[(0, gi)]
                    k = k_pair[gi]
                    if P == "P1":
                        v = torch.tensor(cent[k], device=dev, dtype=hp.dtype)
                    else:
                        v = W_emb[cnt_tok[k]].to(hp.dtype)
                        scale = hp[0, a:b].norm(dim=-1).median() / (v.norm() + 1e-6)
                        v = v * scale
                    hp[0, a:b] = v
                run(hp, L_PATCH + 1, max(READ), P)
            # P3 input-level text patch (full forward)
            emb3 = emb.clone()
            for gi in range(len(levels[0])):
                a, b = span_of[(0, gi)]
                emb3[0, a:b] = W_emb[cnt_tok[k_pair[gi]]].to(emb3.dtype)
            run(emb3, 0, max(READ), "P3")

        ys.append([1 if t in evid else 0 for t in range(NF)])
        golds.append(gold)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  {n_done}/{args.limit} (skip {n_skip}) {time.time()-t0:.0f}s "
                  f"seq={seq}", flush=True)

    Y = np.array(ys)
    G = np.array(golds)
    n = len(G)
    np.savez_compressed(out / "patch_feats.npz", Y=Y, G=G,
                        **{f"{P}|{lv}|{L}|{ft}": np.stack(v).astype(np.float16)
                           for (P, lv, L, ft), v in feats.items() if P != "P0__pre"})

    from sklearn.linear_model import LogisticRegression, Ridge
    splits = []
    for seed in range(3):
        idx = np.random.default_rng(seed).permutation(n)
        splits.append((idx[: n // 2], idx[n // 2:]))
    rows = []
    for P in ARMS_P:
        for lv in (1, 2):
            cnt = np.stack([[int(Y[s, sorted(ls)].sum()) for ls in lsets[lv]]
                            for s in range(n)])
            for L in READ:
                for feat in ("mean", "last"):
                    X = np.stack(feats[(P, lv, L, feat)]).astype(np.float32)
                    _, nn, H = X.shape
                    acc, pm1, r2s = [], [], []
                    for tr, ev in splits:
                        Xtr, Xev = _std_fit(X[tr].reshape(-1, H), X[ev].reshape(-1, H))
                        clf = LogisticRegression(max_iter=1000).fit(
                            Xtr, cnt[tr].reshape(-1))
                        pr = clf.predict(Xev).reshape(len(ev), nn)
                        acc.append(float((pr == cnt[ev]).mean()))
                        pm1.append(float((np.abs(pr - cnt[ev]) <= 1).mean()))
                        rg = Ridge(alpha=10.0).fit(Xtr, cnt[tr].reshape(-1))
                        pv = rg.predict(Xev).reshape(len(ev), nn)
                        sst = float(((cnt[ev] - cnt[ev].mean()) ** 2).sum())
                        r2s.append(1 - float(((pv - cnt[ev]) ** 2).sum()) / max(sst, 1e-9))
                    rows.append([P, lv + 1, L, feat, np.mean(acc), np.mean(pm1),
                                 np.mean(r2s)])
                    print(f"[{P} lvl{lv+1} L{L} {feat}] count {np.mean(acc):.3f} "
                          f"+-1 {np.mean(pm1):.3f} R2 {np.mean(r2s):.2f}", flush=True)
    with open(out / "patch.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "level", "L", "feat", "count_acc", "pm1", "r2"],
                                 *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
