#!/usr/bin/env python3
"""RECAGG P2 — length extrapolation: fit aggregator heads @N in {8,16},
zero-shot eval @N in {32,64}. THE headline experiment (H2).

Reuses the P0 arms/recipe from train_heads.py (canonical: epochs=20000,
PCA-128-whitened front-end, aux per-frame BCE, subset-count augmentation).
Hygiene per the brief: heads NEVER see a capture longer than the fit lengths;
scaler/PCA/probe fit on the train half of the fit pools only; park and HF are
never mixed (this script takes one domain per invocation — the P2 directive is
HF @512 only, Tal 2026-08-17).

Readout note (pre-registered expectation): the classification head's support is
0..max_fit_N (16) and CANNOT emit larger counts — its extrapolation failure at
N=32/64 is structural. The scalar-round readout carries H2; both are reported.

Usage:
  python scripts/recagg/eval_extrap.py \
      --fit  outputs/ninv/20260809_235142_hf8_leaf512/feats_N8.npz:8 \
             outputs/recagg/p1_captures/20260817_154347_hf16_512/feats_N16.npz:16 \
      --eval outputs/recagg/p1_captures/20260817_154347_hf32_512/feats_N32.npz:32 \
             outputs/recagg/p1_captures/20260817_154347_hf64_512/feats_N64.npz:64 \
      --label hf512 --output outputs/recagg/p2_extrap/<ts>_hf512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "scripts/ninv"))
sys.path.insert(0, str(_HERE))

from metrics_skew import class_report, cluster_bootstrap_ci  # noqa: E402
from train_heads import (  # noqa: E402
    AttnPoolHead,
    GRUHead,
    SelectiveSSMHead,
    dist_line,
    fit_frame_probe,
    load_capture,
    n_params,
)

from sklearn.decomposition import PCA  # noqa: E402

import torch.nn as nn  # noqa: E402


class NoleakSSMHead(SelectiveSSMHead):
    """R3b — the H2 mechanism ablation. Two drift sources pinned:
    (1) decay: A_log frozen at -20 -> exp(dt*A) = 1-eps, an EXACT integrator
        (the learned ~0.99/step decay was indistinguishable from 1.0 over 16
        training steps but leaked ~half the integral over 64);
    (2) readout range: count read by a PURE LINEAR map on h_N (a linear function
        of an unsaturated integral extrapolates exactly; the MLP readout was
        only fitted on outputs 0..16).
    Prediction (pre-registered in STATE 2026-08-17): recovers most of the
    zero-shot N=64 loss; failure would point back at the recurrence itself."""

    def __init__(self, h_in, n_max, proj=128, state=128):
        super().__init__(h_in, n_max, proj, state)
        with torch.no_grad():
            self.A_log.fill_(-20.0)
        self.A_log.requires_grad_(False)
        self.lin = nn.Linear(state, 1)

    def forward(self, x):
        u = self.proj(x)
        A = -torch.exp(self.A_log)
        h = x.new_zeros(x.shape[0], self.A_log.shape[0])
        ys = []
        for t in range(u.shape[1]):
            ut = u[:, t]
            dt = torch.sigmoid(self.w_dt(ut))
            h = torch.exp(dt * A) * h + dt * self.b
            ys.append(self.w_c(ut) * h)
        aux = self.aux(torch.stack(ys, dim=1)).squeeze(-1)
        cls_logits, _ = self.out(h)          # cls kept for training signal only
        return cls_logits, self.lin(h).squeeze(-1), aux


def train_multilen(model, groups, epochs, aux_w, lr=1e-3, wd=1e-4, seed=0):
    """P0 recipe generalized to several fit lengths: each epoch draws a random
    subset (random order, count = subset Y-sum) from EVERY group and sums the
    losses, so N=8 and N=16 sequences interleave every step.
    Subset indices are drawn on CPU (seeded Generator) and moved to the model's
    device — keeps runs reproducible across cpu/cuda."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for Xtr, Ytr in groups:
            B, N = Ytr.shape
            k = int(torch.randint(1, N + 1, (1,), generator=g))
            idx = torch.argsort(torch.rand(B, N, generator=g), dim=1)[:, :k].to(dev)
            Xb = Xtr[torch.arange(B, device=dev)[:, None], idx]
            Yb = Ytr[torch.arange(B, device=dev)[:, None], idx]
            Gb = Yb.sum(1)
            logits, scalar, frame_logits = model(Xb)
            loss = loss + (F.cross_entropy(logits, Gb)
                           + 0.1 * F.mse_loss(scalar, Gb.float())
                           + aux_w * F.binary_cross_entropy_with_logits(
                               frame_logits, Yb.float()))
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def predict(model, X, clamp_max):
    logits, scalar, _ = model(X.to(next(model.parameters()).device))
    return (logits.argmax(-1).cpu().numpy(),
            scalar.round().clamp(0, clamp_max).long().cpu().numpy())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True, metavar="NPZ:N")
    ap.add_argument("--eval", nargs="+", required=True, metavar="NPZ:N")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--label", default="extrap")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--canary-perms", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=0,
                    help="divide-and-conquer eval on zero-shot cells: run the "
                         "head per window of <=chunk frames (inside its trained "
                         "range), round each window's scalar (re-quantization), "
                         "sum across windows. 0 = off")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                       else args.device if args.device != "auto" else "cpu")
    print(f"device: {dev}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def parse(spec):
        p, n = spec.rsplit(":", 1)
        X, Y, G = load_capture(p, args.layer)
        assert X.shape[1] == int(n), f"{p}: N mismatch"
        return {"path": p, "n": int(n), "X": X, "Y": Y, "G": G}

    fits = [parse(s) for s in args.fit]
    evals = [parse(s) for s in args.eval]
    H = fits[0]["X"].shape[-1]
    n_fit_max = max(f["n"] for f in fits)
    for d in fits + evals:
        print(f"[{'fit' if d in fits else 'eval'}] N={d['n']} n={len(d['G'])} "
              f"{d['path']}\n    gold dist: {dist_line(d['G'], d['n'])}")

    arms = ["R1_sum_probe", "R2_gru", "R3_ssm", "R3b_noleak", "R4_attnpool"]
    # cells: inlength eval halves of fit Ns + zero-shot eval Ns (+ chunked zs)
    cells = [f"N{f['n']}_in" for f in fits] + [f"N{e['n']}_zs" for e in evals]
    if args.chunk:
        cells += [f"N{e['n']}_ch{args.chunk}" for e in evals]
    pool = {c: {a: {"cls": [], "reg": []} for a in arms} for c in cells}
    true = {c: [] for c in cells}
    canary = {c: {a: [] for a in arms} for c in cells}
    p_frame_cell = {c: [] for c in cells}

    for s in range(args.seeds):
        print(f"\n--- seed {s} ---")
        halves = []
        for f in fits:
            m = len(f["G"])
            idx = np.random.default_rng(s).permutation(m)
            tr, ev = idx[: m // 2], idx[m // 2:]
            halves.append((f, tr, ev))
            print(f"N={f['n']} TRAIN dist: {dist_line(f['G'][tr], f['n'])}")
            print(f"N={f['n']} EVAL  dist: {dist_line(f['G'][ev], f['n'])}")

        # scaler + PCA + frame probe on pooled TRAIN frames of the fit Ns only
        Ftr = np.concatenate([f["X"][tr].reshape(-1, H) for f, tr, _ in halves])
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        pca = PCA(n_components=min(args.pca, len(Ftr) - 1, H), whiten=True,
                  random_state=0).fit((Ftr - mu) / sd)
        tx = lambda X3: pca.transform(
            ((X3.reshape(-1, H) - mu) / sd)).reshape(*X3.shape[:2], -1).astype(np.float32)
        probe = fit_frame_probe(
            np.concatenate([f["X"][tr].reshape(1, -1, H) for f, tr, _ in halves], axis=1),
            np.concatenate([f["Y"][tr].reshape(1, -1) for f, tr, _ in halves], axis=1))

        groups = [(torch.from_numpy(tx(f["X"][tr])).to(dev),
                   torch.from_numpy(f["Y"][tr]).to(dev))
                  for f, tr, _ in halves]
        heads = {"R2_gru": GRUHead(args.pca, n_fit_max, args.proj, args.hidden),
                 "R3_ssm": SelectiveSSMHead(args.pca, n_fit_max, args.proj, args.hidden),
                 "R3b_noleak": NoleakSSMHead(args.pca, n_fit_max, args.proj, args.hidden),
                 "R4_attnpool": AttnPoolHead(args.pca, n_fit_max, args.proj)}
        heads = {k: m.to(dev) for k, m in heads.items()}
        for name, model in heads.items():
            if s == 0:
                print(f"{name}: {n_params(model):,} params")
            train_multilen(model, groups, args.epochs, args.aux_weight, seed=s)
            if s == 0:
                torch.save({"state_dict": model.state_dict(), "scaler_mu": mu,
                            "scaler_sd": sd, "pca_components": pca.components_,
                            "pca_mean": pca.mean_, "n_max": n_fit_max,
                            "layer": args.layer, "fit": args.fit},
                           out / f"{name}_seed0.pt")

        def eval_cell(cell, X, Y, G, N):
            bits = probe(X)
            p_frame_cell[cell].append(float((bits == Y).mean()))
            pool[cell]["R1_sum_probe"]["cls"].append(bits.sum(1))
            pool[cell]["R1_sum_probe"]["reg"].append(bits.sum(1))
            canary[cell]["R1_sum_probe"].append(0.0)
            Xt = torch.from_numpy(tx(X))
            for name, model in heads.items():
                pc, pr = predict(model, Xt, N)
                pool[cell][name]["cls"].append(pc)
                pool[cell][name]["reg"].append(pr)
                em0 = float((pr == G).mean())
                deltas = []
                for k in range(args.canary_perms):
                    prm = np.stack([np.random.default_rng(7000 + s * 91 + k * 13 + i)
                                    .permutation(N) for i in range(len(G))])
                    _, prk = predict(model, Xt[np.arange(len(G))[:, None], prm], N)
                    deltas.append(abs(float((prk == G).mean()) - em0))
                canary[cell][name].append(max(deltas))
            true[cell].append(G)

        def eval_cell_chunked(cell, X, Y, G, N, chunk):
            """Divide-and-conquer: head per <=chunk-frame window (inside its
            trained range), round each window scalar (re-quantization), sum.
            R1's sum is already windowless — recomputed identically as sanity."""
            bits = probe(X)
            p_frame_cell[cell].append(float((bits == Y).mean()))
            pool[cell]["R1_sum_probe"]["cls"].append(bits.sum(1))
            pool[cell]["R1_sum_probe"]["reg"].append(bits.sum(1))
            canary[cell]["R1_sum_probe"].append(0.0)
            Xt = torch.from_numpy(tx(X))
            starts = list(range(0, N, chunk))
            for name, model in heads.items():
                tot = np.zeros(len(G), dtype=np.int64)
                for a in starts:
                    w = min(chunk, N - a)
                    _, prw = predict(model, Xt[:, a:a + w], w)
                    tot += prw
                pool[cell][name]["cls"].append(np.minimum(tot, N))
                pool[cell][name]["reg"].append(np.minimum(tot, N))
                canary[cell][name].append(0.0)   # canary covered by the zs cell
            true[cell].append(G)

        for f, tr, ev in halves:
            eval_cell(f"N{f['n']}_in", f["X"][ev], f["Y"][ev], f["G"][ev], f["n"])
        for e in evals:
            eval_cell(f"N{e['n']}_zs", e["X"], e["Y"], e["G"], e["n"])
            if args.chunk:
                eval_cell_chunked(f"N{e['n']}_ch{args.chunk}", e["X"], e["Y"],
                                  e["G"], e["n"], args.chunk)

    # -------------------------------------------------------------------- report
    lines = [f"RECAGG P2 extrapolation — {args.label}",
             f"fit: {args.fit}", f"eval: {args.eval}",
             f"layer={args.layer} seeds={args.seeds} epochs={args.epochs} "
             f"pca={args.pca} aux={args.aux_weight} (canonical P0 recipe)",
             f"cls support capped at 0..{n_fit_max} (fit lengths) — its zero-shot "
             f"failure at larger N is structural; EM_reg (scalar-round, clamped to "
             f"eval N) carries H2.", ""]
    results = {}
    hdr = (f"{'cell':<8} {'arm':<14} {'EM_reg':>6} {'ci95':>15} {'EM_cls':>6} "
           f"{'major':>6} {'bal':>6} {'bound':>6} {'canaryΔ':>8}")
    lines.append(hdr)
    for c in cells:
        tr_ = np.concatenate(true[c])
        n_cell = int(c[1:].split("_")[0])
        pf = float(np.mean(p_frame_cell[c]))
        bound = pf ** n_cell
        for a in arms:
            pr = np.concatenate(pool[c][a]["reg"])
            pc = np.concatenate(pool[c][a]["cls"])
            rep = class_report(pr, tr_, n_classes=n_cell + 1)
            lo, hi, _ = cluster_bootstrap_ci((pr == tr_)[:, None])
            em_cls = float((pc == tr_).mean())
            can = max(canary[c][a])
            lines.append(f"{c:<8} {a:<14} {rep['raw']:>6.3f} [{lo:.3f},{hi:.3f}] "
                         f"{em_cls:>6.3f} {rep['majority']:>6.3f} "
                         f"{rep['balanced']:>6.3f} {bound:>6.3f} {can:>8.3f}")
            results.setdefault(c, {})[a] = {
                "em_reg": rep["raw"], "ci": [lo, hi], "em_cls": em_cls,
                "majority": rep["majority"], "balanced": rep["balanced"],
                "bound": bound, "p_frame": pf, "canary_max": can,
                "n": int(len(tr_))}
        lines.append("")
    lines.append(f"wall {time.time() - t0:.0f}s")
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    print(f"wrote {out}/report.txt, results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
