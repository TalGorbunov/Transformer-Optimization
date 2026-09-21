#!/usr/bin/env python3
"""QGATE B1/A2/D3 fits on capture_qb output (CPU).

A2  state-only -> bit (with ALT relabelings: same state, different labels) — the
    question-necessity control, MUST fail (H-QA2: <= base rate + 0.10).
D3  bilinear(state, symbolic (char,room) one-hots) -> bit — are the question-
    independent occupancy facts linearly present in the blind encodings?
B1  interaction(state, model text-only q_vec) -> bit — the deployable scorer.
    Split by DIR (held-out dirs; ALT pairs make unseen-(C,R) automatic).
    LOO: fit on {count,exists} q_vecs, test on {majority} q_vecs.

Usage: python scripts/qgate/fit_qb.py --npz outputs/qgate/bcap2/capture.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

CHARS = ["Sandra", "Mary", "Michael", "John", "Daniel", "Laura", "Peter", "Emma", "Noah"]
ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]


def acc_auc(y, score):
    from sklearn.metrics import roc_auc_score

    pred = (score > 0).astype(int) if score.dtype != int else score
    acc = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        auc = float("nan")
    return acc, auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, default=1, help="0=L12, 1=L20")
    ap.add_argument("--chan", default="all", choices=("all", "mean", "last", "max"),
                    help="pooling channel when states are 3H concats")
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=False)
    S = z["block_states"][:, args.layer_idx]          # [B, H or 3H]
    if args.chan != "all" and S.shape[1] % 3 == 0:
        H = S.shape[1] // 3
        i = ("mean", "last", "max").index(args.chan)
        S = S[:, i * H:(i + 1) * H]
    bdir, bt = z["block_dir"], z["block_t"]
    labels = z["labels"]                              # rows (di, pi, t, bit)
    qv = z["qvecs"][:, args.layer_idx]                # [Q, H]
    qkeys = [s.split("|") for s in z["qvec_keys"]]    # di, pi, task
    pmeta = [s.split("|") for s in z["pair_meta"]]    # di, pi, c, r, k
    block_of = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(bdir, bt))}
    qv_of = {(int(d), int(p), t): i for i, (d, p, t) in enumerate(qkeys)}
    pair_of = {(int(d), int(p)): (c, r) for d, p, c, r, _k in pmeta}

    rng = np.random.default_rng(args.seed)
    dirs = np.unique(bdir)
    test_dirs = set(rng.choice(dirs, int(len(dirs) * args.test_frac), replace=False))
    print(f"blocks {len(S)}, labels {len(labels)}, qvecs {len(qv)}, dirs {len(dirs)} "
          f"(test {len(test_dirs)}), layer idx {args.layer_idx}")

    p_s = PCA(args.pca, random_state=0).fit(S)
    Sp = p_s.transform(S)
    p_q = PCA(min(args.pca, len(qv) - 1), random_state=0).fit(qv)
    Qp = p_q.transform(qv)

    rows = []
    for di, pi, t, bit in labels:
        bi = block_of.get((int(di), int(t)))
        if bi is None or (int(di), int(pi)) not in pair_of:
            continue
        rows.append((int(di), int(pi), int(t), int(bit), bi))
    rows = np.array(rows)
    is_test = np.array([r[0] in test_dirs for r in rows])
    y = rows[:, 3]
    base = max(y[is_test].mean(), 1 - y[is_test].mean())
    print(f"rows {len(rows)}, test base rate {base:.3f}")

    # ---- A2: state only
    X = Sp[rows[:, 4]]
    lr = LogisticRegression(max_iter=2000).fit(X[~is_test], y[~is_test])
    acc, auc = acc_auc(y[is_test], lr.decision_function(X[is_test]))
    print(f"A2 state-only:        acc {acc:.4f}  auc {auc:.4f}  "
          f"(H-QA2 pass = acc <= {base + 0.10:.3f})")

    # ---- D3: bilinear(state, symbolic pair)
    sym = np.zeros((len(rows), len(CHARS) + len(ROOMS)), dtype=np.float32)
    for i, (di, pi, *_rest) in enumerate(rows):
        c, r = pair_of[(int(di), int(pi))]
        if c in CHARS:
            sym[i, CHARS.index(c)] = 1.0
        if r in ROOMS:
            sym[i, len(CHARS) + ROOMS.index(r)] = 1.0
    Xd3 = np.einsum("bi,bj->bij", sym, Sp[rows[:, 4]]).reshape(len(rows), -1)
    lr3 = LogisticRegression(max_iter=3000).fit(Xd3[~is_test], y[~is_test])
    acc3, auc3 = acc_auc(y[is_test], lr3.decision_function(Xd3[is_test]))
    print(f"D3 state x symbolic:  acc {acc3:.4f}  auc {auc3:.4f}  "
          f"(facts-present iff >> base)")

    # ---- B1: interaction(state, model q_vec) — for each label row x task
    d_int = min(args.pca, Qp.shape[1])
    out = {}
    for split_name, fit_tasks, test_tasks in (
            ("heldout", ("count", "exists", "majority"), ("count", "exists", "majority")),
            ("LOO-majority", ("count", "exists"), ("majority",))):
        Xf, yf, Xt, yt = [], [], [], []
        for di, pi, t, bit, bi in rows:
            for task in ("count", "exists", "majority"):
                qi = qv_of.get((int(di), int(pi), task))
                if qi is None:
                    continue
                feat = np.concatenate([Sp[bi], Qp[qi],
                                       Sp[bi][:d_int] * Qp[qi][:d_int]])
                te = int(di) in test_dirs
                if not te and task in fit_tasks:
                    Xf.append(feat)
                    yf.append(bit)
                elif te and task in test_tasks:
                    Xt.append(feat)
                    yt.append(bit)
        lrb = LogisticRegression(max_iter=3000).fit(np.array(Xf), np.array(yf))
        accb, aucb = acc_auc(np.array(yt), lrb.decision_function(np.array(Xt)))
        out[split_name] = (accb, aucb, len(yt))
        print(f"B1 {split_name:14s}: acc {accb:.4f}  auc {aucb:.4f}  n_test {len(yt)}")
    Path(str(args.npz.parent / "fit_report.json")).write_text(json.dumps(
        {"A2": [acc, auc], "D3": [acc3, auc3],
         "B1": {k: list(v) for k, v in out.items()}, "base": base}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
