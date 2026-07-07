#!/usr/bin/env python3
"""Mechanism tests for the evidence/non-evidence interference (Stage 1 follow-up). CPU, on the L19 cache.

A) Distractor-dilution: hold the g evidence frames, add m non-evidence frames, decode g. Decorrelates
   "#distractors" from the count -> isolates pure interference (m=0 is S_evid=1.0; watch it fall).
B) K-hub capacity: partition the 8 frames into K contiguous hubs, sum each, concat, decode g, K in
   {1,2,4,8}. K=1=S_all (single hub), K=8=no summing. The over-squashing capacity curve + the
   quantitative version of the "multiple aggregation hubs" idea.
C) Separability: single-frame is-evidence linear acc (signal exists per frame?) and the CENTERED
   direction overlap cos(mu_e-mu_all, mu_n-mu_all) (raw cos is ~1 because a huge shared frame-mean
   dominates; the centered version is the discriminative overlap that actually sets the SNR).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.glstm.probe_message_sum_decodability import load_cache, fit_ridge, agg_sum  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402


def exp_A_dilution(exs, seeds, rng_seed=0):
    rows = []
    N = exs[0].reps.shape[0]
    rng = np.random.RandomState(rng_seed)
    for m in range(0, N):  # number of non-evidence distractors added
        X, y = [], []
        for e in exs:
            ev = np.where(e.labels == 1)[0]
            nv = np.where(e.labels == 0)[0]
            if len(ev) < 1 or len(nv) < m:
                continue
            pick = nv if m == 0 else nv[rng.permutation(len(nv))[:m]]
            S = agg_sum(e.reps, np.concatenate([ev, pick]).astype(int)) if m > 0 else agg_sum(e.reps, ev)
            X.append(S); y.append(e.gold)
        if len(set(y)) < 2 or len(y) < 50:
            continue
        r = fit_ridge(np.stack(X), np.asarray(y), seeds)
        r["m"] = m; r["n"] = len(y); rows.append(r)
    return rows


def exp_B_khub(exs, seeds):
    rows = []
    N = exs[0].reps.shape[0]
    gold = np.asarray([e.gold for e in exs])
    for K in (1, 2, 4, 8):
        if N % K != 0:
            continue
        sz = N // K
        feats = []
        for e in exs:
            hubs = [agg_sum(e.reps, np.arange(k * sz, (k + 1) * sz)) for k in range(K)]
            feats.append(np.concatenate(hubs))
        r = fit_ridge(np.stack(feats), gold, seeds)
        r["K"] = K; r["dim"] = feats[0].shape[0]; rows.append(r)
    return rows


def exp_C_separability(exs, seeds):
    # single-frame is-evidence linear balanced-acc
    X = np.stack([e.reps[i] for e in exs for i in range(e.reps.shape[0])])
    y = np.asarray([int(e.labels[i]) for e in exs for i in range(e.reps.shape[0])])
    baccs = []
    for s in seeds[:3]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=s, stratify=y)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(Xtr), ytr)
        baccs.append(balanced_accuracy_score(yte, clf.predict(sc.transform(Xte))))
    # centered discriminative direction overlap
    ev = np.stack([e.reps[i] for e in exs for i in range(e.reps.shape[0]) if e.labels[i] == 1])
    nv = np.stack([e.reps[i] for e in exs for i in range(e.reps.shape[0]) if e.labels[i] == 0])
    mu_all = np.concatenate([ev, nv]).mean(0)
    de = ev.mean(0) - mu_all; dn = nv.mean(0) - mu_all
    raw_cos = float(ev.mean(0) @ nv.mean(0) / (np.linalg.norm(ev.mean(0)) * np.linalg.norm(nv.mean(0))))
    cen_cos = float(de @ dn / (np.linalg.norm(de) * np.linalg.norm(dn) + 1e-9))
    # signal-to-noise of the discriminative axis (Fisher-ish): |mu_e-mu_n| / sqrt(within-class var on that axis)
    w = ev.mean(0) - nv.mean(0); w = w / (np.linalg.norm(w) + 1e-9)
    pe = ev @ w; pn = nv @ w
    snr = float(abs(pe.mean() - pn.mean()) / (0.5 * (pe.std() + pn.std()) + 1e-9))
    return {"perframe_isevidence_bacc": float(np.mean(baccs)), "raw_cos": raw_cos,
            "centered_cos": cen_cos, "perframe_snr": snr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "message_sum")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    run_dir = args.output / (time.strftime("%Y%m%d_%H%M%S") + "_mechanism")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def emit(m): print(m, flush=True); lines.append(m)

    csvA = ["label,m_distractors,n,acc,r2"]
    csvB = ["label,K_hubs,dim,acc,r2"]
    csvC = ["label,perframe_isevidence_bacc,raw_cos,centered_cos,perframe_snr"]
    for spec in args.caches:
        label, _, path = spec.partition(":")
        exs = load_cache(Path(path), args.seq_len)
        emit(f"\n===== {label}: {len(exs)} examples =====")
        emit("  (C) separability:")
        c = exp_C_separability(exs, seeds)
        emit(f"    per-frame is-evidence bacc={c['perframe_isevidence_bacc']:.3f}  "
             f"raw_cos={c['raw_cos']:.3f}  centered_cos={c['centered_cos']:.3f}  "
             f"per-frame SNR(|de|/within-std)={c['perframe_snr']:.2f}")
        csvC.append(f"{label},{c['perframe_isevidence_bacc']:.4f},{c['raw_cos']:.4f},{c['centered_cos']:.4f},{c['perframe_snr']:.4f}")
        emit("  (A) distractor-dilution (fix evidence, add m non-evidence -> decode g):")
        for r in exp_A_dilution(exs, seeds):
            emit(f"    m={r['m']}: acc={r['acc']:.3f} r2={r['r2']:.3f} (n={r['n']})")
            csvA.append(f"{label},{r['m']},{r['n']},{r['acc']:.4f},{r['r2']:.4f}")
        emit("  (B) K-hub capacity (partition into K summed hubs, concat -> decode g):")
        for r in exp_B_khub(exs, seeds):
            emit(f"    K={r['K']} (dim={r['dim']}): acc={r['acc']:.3f} r2={r['r2']:.3f}")
            csvB.append(f"{label},{r['K']},{r['dim']},{r['acc']:.4f},{r['r2']:.4f}")

    (run_dir / "exp_A_dilution.csv").write_text("\n".join(csvA) + "\n")
    (run_dir / "exp_B_khub.csv").write_text("\n".join(csvB) + "\n")
    (run_dir / "exp_C_separability.csv").write_text("\n".join(csvC) + "\n")
    (run_dir / "README.md").write_text("# message-sum mechanism tests\n\n```\n" + "\n".join(lines) + "\n```\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as _csv
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        A = list(_csv.DictReader(open(run_dir / "exp_A_dilution.csv")))
        B = list(_csv.DictReader(open(run_dir / "exp_B_khub.csv")))
        for label in sorted(set(r["label"] for r in A)):
            ra = [r for r in A if r["label"] == label]
            axes[0].plot([int(r["m_distractors"]) for r in ra], [float(r["acc"]) for r in ra], "o-", label=label)
            rb = [r for r in B if r["label"] == label]
            axes[1].plot([int(r["K_hubs"]) for r in rb], [float(r["acc"]) for r in rb], "o-", label=label)
        axes[0].set_xlabel("# non-evidence distractors added"); axes[0].set_ylabel("decode g acc")
        axes[0].set_title("(A) distractor dilution"); axes[0].legend(); axes[0].set_ylim(0, 1.02)
        axes[1].set_xlabel("# aggregation hubs K"); axes[1].set_ylabel("decode g acc")
        axes[1].set_title("(B) K-hub capacity"); axes[1].legend(); axes[1].set_ylim(0, 1.02)
        fig.tight_layout(); fig.savefig(run_dir / "mechanism.png", dpi=120); plt.close(fig)
    except Exception as exc:
        emit(f"[warn] plot failed: {exc}")
    print(f"\nwrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
