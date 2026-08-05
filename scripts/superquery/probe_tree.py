#!/usr/bin/env python3
"""Tree-superquery probe: divide-and-conquer readout over fenced replica blocks.

Layout (the professor's proposal, 2026-08-05): [q0][frame_i + q_i]xN fenced blocks with
per-block posreset, FENCED AT ALL LAYERS (fence never lifts). After the blocks, one
question-replica per internal tree node ("superquery"): a level-1 SQ may attend ONLY
{prefix, its own tokens (causal), the q_i spans of its b children}; a level-l SQ only
its b child SQ spans. Arms = flat (one SQ, fan-in N) + trees b in {2,4,8}. All arms
ride ONE forward per sample: blocks are shared, each SQ row's mask encodes its arm.

Failure-signal ladder (zero training; linear probes on SQ hidden states, split by sample):
  nodes.csv  per (N, arm, level, layer, feat): subtree-count probe acc / +-1 / ridge R2,
             gate d' (count>0) — WHERE does aggregation die, at WHICH fan-in
  top.csv    per (N, arm, layer, feat): top-node gold regression R2/MAE + composed tally
             (sum of level-1 count predictions == gold)

Features are dumped to <output>/feats_N<k>.npz right after each N's capture (crash-safe;
fp16). Fits standardize features (lbfgs converges ~50 iters vs max-iter crawl unscaled).

Usage:
  python scripts/superquery/probe_tree.py --output outputs/superquery/<ts> [--ns 8,16]
  python scripts/superquery/probe_tree.py --fit-npz 'outputs/superquery/<ts>/feats_N*.npz' \
      --output <dir>   # CPU-only refit from dumps
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.metrics import dprime_pair

NS = [(8, "data/mmred_images_park/seq_len_8/all_uniform", 120),
      (16, "data/mmred_longN_park/seq_len_16/all_uniform", 120),
      (32, "data/mmred_longN_park/seq_len_32/all_uniform", 80),
      (64, "data/mmred_longN_park/seq_len_64/all_uniform", 40)]
BRANCHINGS = (2, 4, 8)


def tree_levels(n_leaves: int, b: int):
    """Bottom-up grouping: level l = groups of <=b nodes of level l-1, down to 1 node.
    Returns [[group child-index lists per node]] per level; ragged top groups allowed."""
    levels, cur = [], n_leaves
    while cur > 1:
        groups = [list(range(i, min(i + b, cur))) for i in range(0, cur, b)]
        levels.append(groups)
        cur = len(groups)
    return levels


def leaf_sets(levels):
    """Per level: each node's set of descendant leaf indices."""
    out, prev = [], None
    for li, groups in enumerate(levels):
        if li == 0:
            cur = [set(g) for g in groups]
        else:
            cur = [set().union(*(prev[c] for c in g)) for g in groups]
        out.append(cur)
        prev = cur
    return out


def build_arms(NF: int):
    arms = {"flat": [[list(range(NF))]]}
    for b in BRANCHINGS:
        if b < NF:
            arms[f"b{b}"] = tree_levels(NF, b)
    return arms


def _std_fit(Xtr, Xev, pca=True):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xev = (Xtr - mu) / sd, (Xev - mu) / sd
    if pca:
        from sklearn.decomposition import PCA
        k = min(512, Xtr.shape[0] - 1, Xtr.shape[1])
        p = PCA(n_components=k, random_state=0).fit(Xtr)
        Xtr, Xev = p.transform(Xtr), p.transform(Xev)
    return Xtr, Xev


def run_fits(NF_target, feats, Y, G, fit_layers, feat_kinds, node_rows, top_rows):
    """feats: dict (arm, lv, L, feat) -> (n, nodes, H) float array.
    Probes run on standardized PCA-512 features (linear probe on a linear projection
    is still a linear probe — a lower bound either way; ~15x faster than raw 3584d)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge

    arms = build_arms(NF_target)
    arm_leaves = {a: leaf_sets(lv) for a, lv in arms.items()}
    n = len(G)
    splits = []
    for seed in range(3):
        idx = np.random.default_rng(seed).permutation(n)
        splits.append((idx[: n // 2], idx[n // 2:]))

    for arm, levels in arms.items():
        lsets = arm_leaves[arm]
        for lv in range(len(levels)):
            cnt = np.stack([[int(Y[s, sorted(ls)].sum()) for ls in lsets[lv]]
                            for s in range(n)])            # (n, nodes)
            fan = float(np.mean([len(g) for g in levels[lv]]))
            for L in fit_layers:
                for feat in feat_kinds:
                    X = feats[(arm, lv, L, feat)].astype(np.float32)
                    _, nn, H = X.shape
                    dp, _, _ = dprime_pair(X, (cnt > 0).astype(np.int64))
                    acc, pm1, r2s, tally = [], [], [], []
                    for tr, ev in splits:
                        Xtr, Xev = _std_fit(X[tr].reshape(-1, H), X[ev].reshape(-1, H))
                        clf = LogisticRegression(max_iter=1000).fit(
                            Xtr, cnt[tr].reshape(-1))
                        pr = clf.predict(Xev).reshape(len(ev), nn)
                        acc.append(float((pr == cnt[ev]).mean()))
                        pm1.append(float((np.abs(pr - cnt[ev]) <= 1).mean()))
                        rg = Ridge(alpha=10.0).fit(Xtr, cnt[tr].reshape(-1))
                        pv = rg.predict(Xev).reshape(len(ev), nn)
                        sse = float(((pv - cnt[ev]) ** 2).sum())
                        sst = float(((cnt[ev] - cnt[ev].mean()) ** 2).sum())
                        r2s.append(1 - sse / max(sst, 1e-9))
                        tally.append(float((pr.sum(1) == G[ev]).mean()))
                    node_rows.append([NF_target, arm, lv + 1, fan, L, feat, nn,
                                      np.mean(acc), np.mean(pm1), np.mean(r2s), dp,
                                      np.mean(tally)])
                    print(f"[N={NF_target} {arm} lvl{lv+1} fan{fan:.1f} L{L} {feat}] "
                          f"count {np.mean(acc):.3f} +-1 {np.mean(pm1):.3f} "
                          f"R2 {np.mean(r2s):.2f} d' {dp:.2f} "
                          f"tally {np.mean(tally):.3f}", flush=True)
        # top node: gold regression
        lv_top = len(levels) - 1
        for L in fit_layers:
            for feat in feat_kinds:
                X = feats[(arm, lv_top, L, feat)].astype(np.float32)[:, 0]
                mae, r2s, acc = [], [], []
                for tr, ev in splits:
                    Xtr, Xev = _std_fit(X[tr], X[ev])
                    rg = Ridge(alpha=10.0).fit(Xtr, G[tr])
                    pv = rg.predict(Xev)
                    mae.append(float(np.abs(pv - G[ev]).mean()))
                    sst = float(((G[ev] - G[ev].mean()) ** 2).sum())
                    r2s.append(1 - float(((pv - G[ev]) ** 2).sum()) / max(sst, 1e-9))
                    clf = LogisticRegression(max_iter=1000).fit(Xtr, G[tr])
                    acc.append(float((clf.predict(Xev) == G[ev]).mean()))
                top_rows.append([NF_target, arm, L, feat, np.mean(acc), np.mean(mae),
                                 np.mean(r2s)])
                print(f"[N={NF_target} {arm} TOP L{L} {feat}] gold-acc "
                      f"{np.mean(acc):.3f} MAE {np.mean(mae):.2f} "
                      f"R2 {np.mean(r2s):.2f}", flush=True)


def write_csvs(out: Path, node_rows, top_rows):
    with open(out / "nodes.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "arm", "level", "fan_in", "L", "feat", "n_nodes",
                                  "count_acc", "pm1", "r2", "gate_dp", "tally"], *node_rows])
    with open(out / "top.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "arm", "L", "feat", "gold_acc", "mae", "r2"],
                                 *top_rows])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read-layers", default="12,16,20", help="layers captured to npz")
    ap.add_argument("--fit-layers", default="16,20",
                    help="layers actually probed; 'none' = capture+dump only")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--ns", default="8,16,32,64")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--no-prefix", action="store_true",
                    help="drop the leading q0 (no question-conditioning of image "
                         "tokens; blocks get the question only via their replica) — "
                         "layout-robustness ablation")
    ap.add_argument("--fit-npz", default=None,
                    help="glob of feats_N*.npz — CPU-only refit, no model load")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    read_layers = [int(x) for x in args.read_layers.split(",")]
    fit_layers = ([] if args.fit_layers == "none"
                  else [int(x) for x in args.fit_layers.split(",")])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    node_rows, top_rows = [], []

    if args.fit_npz:
        for p in sorted(_glob.glob(args.fit_npz)):
            d = np.load(p)
            NF_target = int(re.search(r"feats_N(\d+)", p).group(1))
            feats = {}
            for k in d.files:
                if "|" in k:
                    arm, lv, L, feat = k.split("|")
                    feats[(arm, int(lv), int(L), feat)] = d[k]
            avail = {kk[2] for kk in feats}
            run_fits(NF_target, feats, d["Y"], d["G"],
                     [L for L in fit_layers if L in avail],
                     sorted({kk[3] for kk in feats}), node_rows, top_rows)
        write_csvs(out, node_rows, top_rows)
        print("wrote", out)
        return 0

    from gnnformer.constants import MASK_MIN, ROOMS
    from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence
    from gnnformer.fencing import (
        build_replica_probe_mask,
        find_question_spans,
        frame_blocks,
        reset_positions,
    )
    from gnnformer.runtime import (
        get_layers,
        get_rope_index_fn,
        image_token_groups,
        load_runtime,
        move_to_device,
    )

    ns_want = {int(x) for x in args.ns.split(",")}
    L_TOP = max(read_layers)
    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)

    for NF_target, root, limit in [x for x in NS if x[0] in ns_want]:
        arms = build_arms(NF_target)
        n_nodes_total = sum(len(g) for lv in arms.values() for g in lv)

        feats: dict = {}
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
                evid, _room = pe_
            except Exception:
                n_skip += 1
                continue
            if not evid and gold != 0:
                n_skip += 1
                continue
            NF = len(frames)
            if NF != NF_target:
                n_skip += 1
                continue
            if args.resize > 0:
                frames = [f.resize((args.resize, args.resize)) for f in frames]
            content = [] if args.no_prefix else [{"type": "text", "text": q0}]
            for f in frames:
                content.append({"type": "image", "image": f})
                content.append({"type": "text", "text": q0})
            for _ in range(n_nodes_total):
                content.append({"type": "text", "text": q0})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, dev)
            ids = inputs["input_ids"][0].tolist()
            seq = len(ids)
            fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                    processor=processor)
            n_pre = 0 if args.no_prefix else 1
            spans = find_question_spans(ids, tok, q0, NF + n_pre + n_nodes_total)
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            if len(fg) != NF or spans is None or len(vstarts) != NF:
                n_skip += 1
                continue
            rep_spans = spans[n_pre : NF + n_pre]
            sq_spans = spans[NF + n_pre :]
            fin_start = sq_spans[0][0]
            blocks = frame_blocks(vstarts, fin_start)
            vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]

            m = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                         fence_blocks=True, blocks=blocks)
            # SQ rows: prefix + own span (causal) + child spans ONLY
            prefix_cols = torch.arange(0, blocks[0][0])
            si = 0
            span_of: dict = {}  # (arm, level, node) -> span
            for arm, levels in arms.items():
                for li, groups in enumerate(levels):
                    for gi, g in enumerate(groups):
                        a, bnd = sq_spans[si]
                        span_of[(arm, li, gi)] = (a, bnd)
                        si += 1
                        child_spans = ([rep_spans[c] for c in g] if li == 0
                                       else [span_of[(arm, li - 1, c)] for c in g])
                        rows = torch.arange(a, bnd)
                        m[rows] = MASK_MIN
                        cols = torch.cat([prefix_cols] +
                                         [torch.arange(ca, cb) for ca, cb in child_spans])
                        m[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
                        blk = torch.zeros(bnd - a, bnd - a)
                        blk.masked_fill_(torch.triu(torch.ones(bnd - a, bnd - a,
                                                               dtype=torch.bool), 1), MASK_MIN)
                        m[a:bnd, a:bnd] = blk

            with torch.no_grad():
                base_pos, _ = rope_fn(inputs["input_ids"],
                                      image_grid_thw=inputs.get("image_grid_thw"),
                                      attention_mask=inputs.get("attention_mask"))
                pos = reset_positions(base_pos, blocks, fin_start).clone().to(dev)
                emb = text_model.embed_tokens(inputs["input_ids"])
                img = model.model.get_image_features(inputs["pixel_values"],
                                                     inputs["image_grid_thw"])
                img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
                im_mask = inputs["input_ids"][0] == model.config.image_token_id
                emb = emb.clone()
                emb[0, im_mask] = img.to(emb.dtype)
                cos_, sin_ = text_model.rotary_emb(emb, pos)
                pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                m4 = m.to(dev).to(emb.dtype).view(1, 1, seq, seq)
                h = emb
                for li in range(L_TOP + 1):
                    h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
                    if li in read_layers:
                        hf = h[0].float().cpu()
                        for arm, levels in arms.items():
                            for lv in range(len(levels)):
                                for feat in ("mean", "last"):
                                    fs = []
                                    for gi in range(len(levels[lv])):
                                        a, bnd = span_of[(arm, lv, gi)]
                                        fs.append(hf[a:bnd].mean(0) if feat == "mean"
                                                  else hf[bnd - 1])
                                    feats.setdefault((arm, lv, li, feat), []).append(
                                        torch.stack(fs).numpy())
            ys.append([1 if t in evid else 0 for t in range(NF)])
            golds.append(gold)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  N={NF_target}: {n_done}/{limit} (skip {n_skip}) "
                      f"{time.time()-t0:.0f}s seq={seq}", flush=True)

        Y = np.array(ys)          # (n, NF) evidence bits
        G = np.array(golds)
        feats = {k: np.stack(v).astype(np.float16) for k, v in feats.items()}
        np.savez_compressed(out / f"feats_N{NF_target}.npz", Y=Y, G=G,
                            **{f"{a}|{lv}|{L}|{ft}": arr
                               for (a, lv, L, ft), arr in feats.items()})
        print(f"  dumped feats_N{NF_target}.npz "
              f"({(out / f'feats_N{NF_target}.npz').stat().st_size/2**20:.0f} MB)",
              flush=True)
        run_fits(NF_target, feats, Y, G, fit_layers, ["mean", "last"],
                 node_rows, top_rows)
        write_csvs(out, node_rows, top_rows)   # rewrite after every N (crash-safe)

    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
