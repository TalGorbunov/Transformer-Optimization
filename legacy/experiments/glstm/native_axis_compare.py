#!/usr/bin/env python3
"""E6-style CPU comparison (reusable across ladder rungs): native reading axis vs delta-hat vs
whitened LDA axis, on a per-frame carrier message cache.

Inputs: native_axes.pt (native_axis_probe.py) + messages_cache.pt (probe_frame_to_carrier_message
--save-messages). Per (layer, offset): held-out d' along w* (shrinkage LDA), along delta-hat, and
along the NATIVE axis; cos(native, delta-hat), cos(native, w*); rho along each axis; the law
2*Phi(d'/2*sqrt(N))-1 (count-prior-mixed) evaluated at d'_native  ->  compare to the model's own
accuracy (from the cache's model_correct and/or --model-acc from a behavior run).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from statistics import NormalDist
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split

ND = NormalDist()
phi = ND.cdf


def dprime_heldout(X, y, axis=None, seeds=(0, 1, 2)):
    """d' of held-out projections. axis=None -> fit shrinkage-LDA on train; else fixed axis."""
    ds = []
    for s in seeds:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.4, random_state=s, stratify=y)
        if axis is None:
            Xe, Xn = X[tr][y[tr] == 1], X[tr][y[tr] == 0]
            lw = LedoitWolf().fit(np.concatenate([Xe - Xe.mean(0), Xn - Xn.mean(0)]))
            w = np.linalg.solve(lw.covariance_, Xe.mean(0) - Xn.mean(0))
            w /= (np.linalg.norm(w) + 1e-12)
        else:
            w = axis / (np.linalg.norm(axis) + 1e-12)
        p = X[te] @ w
        pE, pN = p[y[te] == 1], p[y[te] == 0]
        sd = 0.5 * (pE.std() + pN.std())
        ds.append(abs(pE.mean() - pN.mean()) / (sd + 1e-12))
    return float(np.mean(ds)), float(np.std(ds))


def lda_axis(X, y):
    Xe, Xn = X[y == 1], X[y == 0]
    lw = LedoitWolf().fit(np.concatenate([Xe - Xe.mean(0), Xn - Xn.mean(0)]))
    w = np.linalg.solve(lw.covariance_, Xe.mean(0) - Xn.mean(0))
    return w / (np.linalg.norm(w) + 1e-12)


def predict_acc(dprime, N, gold):
    d_n = dprime / np.sqrt(N)
    p_int = max(2 * phi(d_n / 2.0) - 1.0, 0.0)
    p_bnd = phi(d_n / 2.0)
    return float(np.mean([p_bnd if g in (0, N) else p_int for g in gold]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="native_axes.pt")
    ap.add_argument("--cache", required=True, help="messages_cache.pt")
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--layers", default="", help="default: all layers present in both files")
    ap.add_argument("--model-acc", type=float, default=None,
                    help="behavioral accuracy from a matched run (optional, reported alongside)")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    nat = torch.load(args.native, map_location="cpu", weights_only=False)
    c = torch.load(args.cache, map_location="cpu", weights_only=False)
    layers = ([int(x) for x in args.layers.replace(",", " ").split()] if args.layers
              else sorted(set(nat["axes"]).intersection(c["msgs"].keys())))
    off = int(args.offset)
    NF = int(c["n_frames"]); gold = np.asarray(c["gold"])
    model_acc_cache = float(np.mean(c["model_correct"])) if "model_correct" in c else float("nan")
    lines = [f"=== NATIVE AXIS COMPARE  cache={args.cache}  offset={off}  N={NF} "
             f"n={len(gold)}  model_acc(cache)={model_acc_cache:.3f}"
             + (f"  model_acc(behavior)={args.model_acc:.3f}" if args.model_acc else "") + "==="]
    lines.append(f"{'L':>3} {'coh':>5} | {'d_nat':>6} {'d_dhat':>6} {'d_w*':>6} | "
                 f"{'cos(n,dh)':>9} {'cos(n,w*)':>9} | {'rho_nat':>7} | {'pred@nat':>8} "
                 f"{'pred@w*':>8}")
    results = {}
    for L in layers:
        msgs = c["msgs"][L][off].astype(np.float32)          # [n, NF, H]
        labels = c["labels"].astype(int)                     # [n, NF]
        X = msgs.reshape(-1, msgs.shape[-1]); y = labels.reshape(-1)
        a = nat["axes"][L]["axis"].astype(np.float32); coh = nat["axes"][L]["coherence"]
        dh = X[y == 1].mean(0) - X[y == 0].mean(0); dh /= (np.linalg.norm(dh) + 1e-12)
        w = lda_axis(X, y)
        d_nat, _ = dprime_heldout(X, y, axis=a)
        d_dh, _ = dprime_heldout(X, y, axis=dh)
        d_w, _ = dprime_heldout(X, y)                         # held-out LDA
        cos_ndh = float(a @ dh); cos_nw = float(a @ w)
        # rho along native: frame-slot residual correlations
        muE, muN = X[y == 1].mean(0), X[y == 0].mean(0)
        mus = np.where(labels[..., None] == 1, muE, muN)
        P = ((msgs - mus) @ a)                               # [n, NF]
        C = np.corrcoef(P.T); rho_nat = float(np.nanmean(C[~np.eye(NF, dtype=bool)]))
        pred_nat = predict_acc(d_nat, NF, gold)
        pred_w = predict_acc(d_w, NF, gold)
        lines.append(f"{L:>3} {coh:>5.2f} | {d_nat:>6.2f} {d_dh:>6.2f} {d_w:>6.2f} | "
                     f"{cos_ndh:>9.3f} {cos_nw:>9.3f} | {rho_nat:>7.3f} | {pred_nat:>8.3f} "
                     f"{pred_w:>8.3f}")
        results[int(L)] = {"coherence": float(coh), "d_native": d_nat, "d_deltahat": d_dh,
                           "d_w": d_w, "cos_native_deltahat": cos_ndh, "cos_native_w": cos_nw,
                           "rho_native": rho_nat, "pred_at_native": pred_nat, "pred_at_w": pred_w}
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "native_compare.txt").write_text(report, encoding="utf-8")
        (out / "native_compare.json").write_text(json.dumps(
            {"results": results, "model_acc_cache": model_acc_cache,
             "model_acc_behavior": args.model_acc, "n": int(len(gold)), "N": NF,
             "offset": off, "native": str(args.native), "cache": str(args.cache)}, indent=1),
            encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
