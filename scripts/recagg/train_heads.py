#!/usr/bin/env python3
"""RECAGG P0 — four aggregator heads over the SAME fenced per-frame leaf states.

Campaign: outputs/recagg/CAMPAIGN_BRIEF.md (H1–H4 pre-registered). This script is
the P0 instrument: it consumes an existing leaf-state capture npz (key
"leaf|0|<layer>|mean", (n, N, 3584) fp16) and trains/evaluates, on pooled 50/50
splits (leaf_probe.py convention):

  R1 sum-probe        logistic per-frame verdict probe -> external sum (control)
  R2 GRU head         proj -> GRU -> count logits (0..N) + scalar (report both)
  R3 minimal SSM      hand-rolled diagonal selective SSM (input-dep dt,B,C), pure torch
  R4 attention-pool   single trained-query softmax read (the Hahn negative control)

Guardrails wired in (brief §Guardrails): class distribution printed for EVERY
split incl. train; every EM reported next to majority baseline + balanced
(metrics_skew.class_report) + the p_frame^N perception bound; fp16 -> fp32 cast;
per-dim scaler fit on train only and saved with the ckpt; order canary (frame
permutation at eval) on every arm; heads <= ~1M params (printed); no installs.

One npz per invocation — park and HF are different domains and are never mixed.

Anchors: first run of the campaign — the canonical P0 numbers live in
outputs/recagg/p0_heads/ (see outputs/recagg/INDEX.md once canonical).

Usage:
  python scripts/recagg/train_heads.py \
      --npz outputs/ninv/20260810_000358_park8_leaf/feats_N8.npz \
      --n 8 --label park8 --output outputs/recagg/p0_heads/<ts>_park8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from metrics_skew import class_report, cluster_bootstrap_ci  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


# --------------------------------------------------------------------------- data


def load_capture(npz: str, layer: int):
    d = np.load(npz)
    lk = f"leaf|0|{layer}|mean"
    if lk not in d.files:
        raise SystemExit(f"{npz} has no '{lk}' (keys: {d.files[:6]}...)")
    X = d[lk].astype(np.float32)          # (n, N, H)  fp16 -> fp32 (guardrail)
    Y = d["Y"].astype(np.int64)           # (n, N) per-frame evidence bits
    G = d["G"].astype(np.int64)           # (n,) gold counts
    assert (Y.sum(1) == G).all(), "Y rowsum != G — capture inconsistent"
    return X, Y, G


def dist_line(g: np.ndarray, n_max: int) -> str:
    c = np.bincount(g, minlength=n_max + 1)
    return " ".join(f"{k}:{v}" for k, v in enumerate(c))


# ------------------------------------------------------------------- frame probe


def fit_frame_probe(X3: np.ndarray, y2: np.ndarray):
    """Per-frame verdict probe = leaf_probe.fit_leaf (logistic on std-PCA-512)."""
    H = X3.shape[-1]
    X, y = X3.reshape(-1, H), y2.reshape(-1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    p = PCA(n_components=min(512, X.shape[0] - 1, H), random_state=0).fit((X - mu) / sd)
    clf = LogisticRegression(max_iter=1000).fit(p.transform((X - mu) / sd), y)
    return lambda Z3: clf.predict(
        p.transform((Z3.reshape(-1, Z3.shape[-1]) - mu) / sd)).reshape(Z3.shape[:2])


# ------------------------------------------------------------------------- heads


class Readout(nn.Module):
    """Shared count readout: tiny MLP -> (class logits, scalar). Identical across
    arms so readout capacity is never the confound. Needed because an integrator
    encodes count as magnitude along a direction — a LINEAR class head must tile
    the count axis as an upper envelope of lines, which barely trains (v6: SSM
    cls 0.14 while its own scalar head read 0.38 off the same state)."""

    def __init__(self, d_in: int, n_max: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.cls = nn.Linear(hidden, n_max + 1)
        self.reg = nn.Linear(hidden, 1)

    def forward(self, h):
        z = self.net(h)
        return self.cls(z), self.reg(z).squeeze(-1)


class GRUHead(nn.Module):
    """R2: proj -> GRU -> last hidden -> count logits (0..n_max) + scalar.

    aux: per-step bit logit from the GRU output — supervision parity with R1's
    frame probe (v1 lesson: count-only labels on 100 sequences -> pure overfit).
    """

    def __init__(self, h_in: int, n_max: int, proj: int = 128, hidden: int = 128,
                 layers: int = 1):
        super().__init__()
        self.proj = nn.Linear(h_in, proj)
        self.gru = nn.GRU(proj, hidden, num_layers=layers, batch_first=True)
        self.out = Readout(hidden, n_max)
        self.aux = nn.Linear(hidden, 1)

    def forward(self, x):                          # x (B, N, H)
        z, _ = self.gru(self.proj(x))
        return (*self.out(z[:, -1]), self.aux(z).squeeze(-1))


class SelectiveSSMHead(nn.Module):
    """R3: minimal diagonal selective SSM, pure torch (brief: no mamba_ssm).

    u_t = proj(x_t);  dt_t = softplus(W_dt u_t)  (input-dependent step)
    A   = -exp(A_log) (learned, negative, diagonal)
    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B(u_t) * u_t
    y   = C(u_N) * h_N  -> count logits + scalar
    Sequential loop is fine — sequences are <= 128 steps.
    """

    def __init__(self, h_in: int, n_max: int, proj: int = 128, state: int = 128):
        super().__init__()
        self.proj = nn.Linear(h_in, proj)
        self.w_dt = nn.Linear(proj, state)
        self.w_c = nn.Linear(proj, state)
        # Input contribution is a learned CONSTANT vector: h can only encode how
        # often (and with what decay) the gate opened — a pure selective
        # integrator. Any content-bearing input map (v7 triple product, v8/v9
        # linear W·u) hands the readout a linear bag-of-frames fingerprint of
        # the train sample and it memorizes (trainEM 1.0, eval ~0.2) while the
        # GRU's saturating tanh happens to destroy that fingerprint. Selection
        # stays input-dependent through dt(u) — which is H4's actual question.
        self.b = nn.Parameter(torch.ones(state))
        # A_log=-4 -> decay exp(dt*A) ~ 0.99/step: near-integrator init. At the
        # naive A_log=0 the state halves per step and counting never trains.
        self.A_log = nn.Parameter(torch.full((state,), -4.0))
        self.out = Readout(state, n_max)
        self.aux = nn.Linear(state, 1)

    def forward(self, x):                          # x (B, N, H)
        u = self.proj(x)                           # (B, N, p)
        A = -torch.exp(self.A_log)                 # (s,)
        h = x.new_zeros(x.shape[0], self.A_log.shape[0])
        ys = []
        for t in range(u.shape[1]):
            ut = u[:, t]
            # bounded gate (sigmoid, like the GRU's): dt in (0,1) is a soft
            # "count this frame" decision; unbounded softplus let every channel
            # grow its own idiosyncratic accumulator (v8 memorization).
            dt = torch.sigmoid(self.w_dt(ut))
            h = torch.exp(dt * A) * h + dt * self.b
            ys.append(self.w_c(ut) * h)            # per-step read C_t * h_t
        # count readout from h_N directly (the integral); C_t*h_t is the per-step
        # aux read only — gating the final read by the LAST frame's C(u) was the
        # v4 failure (arbitrary content modulates the accumulated count).
        aux = self.aux(torch.stack(ys, dim=1)).squeeze(-1)
        return (*self.out(h), aux)


class AttnPoolHead(nn.Module):
    """R4: single trained-query softmax attention read (Hahn negative control)."""

    def __init__(self, h_in: int, n_max: int, proj: int = 128):
        super().__init__()
        self.k = nn.Linear(h_in, proj)
        self.v = nn.Linear(h_in, proj)
        self.q = nn.Parameter(torch.randn(proj) / proj ** 0.5)
        self.out = Readout(proj, n_max)
        self.aux = nn.Linear(proj, 1)

    def forward(self, x):                          # x (B, N, H)
        k, v = self.k(x), self.v(x)
        w = torch.softmax(k @ self.q / k.shape[-1] ** 0.5, dim=1)   # (B, N)
        pooled = (w.unsqueeze(-1) * v).sum(1)
        return (*self.out(pooled), self.aux(v).squeeze(-1))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------- training


def train_head(model: nn.Module, Xtr: torch.Tensor, Gtr: torch.Tensor,
               Ytr: torch.Tensor, epochs: int, aux_w: float, augment: bool,
               lr: float = 1e-3, wd: float = 1e-4, seed: int = 0) -> None:
    """Full-batch Adam. augment=True: each epoch trains on a random frame SUBSET
    in random ORDER, count target = sum of the subset's Y bits (legit — same
    supervision R1 uses). Combinatorial count labels instead of 100; imposes the
    permutation invariance of clean counting; varies train length. v2 lesson:
    without it every head memorizes the count labels (trainEM 1.0, eval ~0.2)
    despite reading the frame bit at 0.99 (head_pf)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    B, N = Ytr.shape
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for e in range(epochs):
        if augment:
            k = int(torch.randint(1, N + 1, (1,), generator=g))
            idx = torch.argsort(torch.rand(B, N, generator=g), dim=1)[:, :k]
            Xb = Xtr[torch.arange(B)[:, None], idx]
            Yb = Ytr[torch.arange(B)[:, None], idx]
            Gb = Yb.sum(1)
        else:
            Xb, Yb, Gb = Xtr, Ytr, Gtr
        opt.zero_grad()
        logits, scalar, frame_logits = model(Xb)
        loss = (F.cross_entropy(logits, Gb)
                + 0.1 * F.mse_loss(scalar, Gb.float())
                + aux_w * F.binary_cross_entropy_with_logits(frame_logits,
                                                             Yb.float()))
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, n_max: int):
    logits, scalar, frame_logits = model(X)
    return (logits.argmax(-1).numpy(),
            scalar.round().clamp(0, n_max).long().numpy(),
            (frame_logits > 0).long().numpy())


# -------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--label", default="capture")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--canary-perms", type=int, default=3)
    ap.add_argument("--pca", type=int, default=128,
                    help="PCA front-end fit on TRAIN frames (0 = raw 3584-dim; "
                         "v1 lesson: raw + count-only labels = pure overfit)")
    ap.add_argument("--aux-weight", type=float, default=1.0,
                    help="per-frame bit BCE weight (0 = count-only supervision; "
                         "supervision parity with R1's frame probe)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--augment", type=int, default=1,
                    help="1 = subset-count augmentation (random frame subset in "
                         "random order, target = subset Y-sum); 0 = fixed "
                         "sequences (v1/v2 recipe, memorizes)")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(torch.get_num_threads(), 4))
    t0 = time.time()

    X, Y, G = load_capture(args.npz, args.layer)
    m, N, H = X.shape
    n_max = args.n
    print(f"loaded {args.npz}: X {X.shape} Y {Y.shape} G {G.shape}  n_max={n_max}")
    print(f"POOL class dist   : {dist_line(G, n_max)}   (n={m})")

    arms = ["R1_sum_probe", "R2_gru", "R3_ssm", "R4_attnpool"]
    # pooled predictions across seeds: arm -> readout -> list of eval-half preds
    pool = {a: {"cls": [], "reg": []} for a in arms}
    pool_true: list[np.ndarray] = []
    # order canary: arm -> list of |EM(perm) - EM(identity)| per seed
    canary = {a: [] for a in arms}
    per_seed_em = {a: [] for a in arms}
    train_em = {a: [] for a in arms}   # underfit diagnostic (June-runs lesson)
    head_pf = {a: [] for a in arms}    # each head's own frame-bit acc on eval
    p_frames = []

    for s in range(args.seeds):
        idx = np.random.default_rng(s).permutation(m)
        tr, ev = idx[: m // 2], idx[m // 2:]
        print(f"\n--- seed {s} ---")
        print(f"TRAIN class dist  : {dist_line(G[tr], n_max)}   (n={len(tr)})")
        print(f"EVAL  class dist  : {dist_line(G[ev], n_max)}   (n={len(ev)})")

        # scaler on TRAIN frames only (guardrail)
        mu = X[tr].reshape(-1, H).mean(0)
        sd = X[tr].reshape(-1, H).std(0) + 1e-6
        Xs = (X - mu) / sd
        pca = None
        if args.pca:
            pca = PCA(n_components=min(args.pca, len(tr) * N - 1, H), whiten=True,
                      random_state=0).fit(Xs[tr].reshape(-1, H))
            Xs = pca.transform(Xs.reshape(-1, H)).reshape(m, N, -1).astype(np.float32)

        # frame probe -> p_frame + R1
        probe = fit_frame_probe(X[tr], Y[tr])       # probe standardizes internally
        bits_ev = probe(X[ev])
        p_frame = float((bits_ev == Y[ev]).mean())
        p_frames.append(p_frame)
        r1_cls = bits_ev.sum(1)
        print(f"p_frame (probe raw acc on eval frames) = {p_frame:.4f}   "
              f"p_frame^{N} = {p_frame ** N:.3f}")
        pool["R1_sum_probe"]["cls"].append(r1_cls)
        pool["R1_sum_probe"]["reg"].append(r1_cls)   # sum has one readout
        per_seed_em["R1_sum_probe"].append(float((r1_cls == G[ev]).mean()))
        canary["R1_sum_probe"].append(0.0)           # sum is invariant by construction
        train_em["R1_sum_probe"].append(float((probe(X[tr]).sum(1) == G[tr]).mean()))
        head_pf["R1_sum_probe"].append(p_frame)

        Xtr = torch.from_numpy(Xs[tr])
        Xev = torch.from_numpy(Xs[ev])
        Gtr = torch.from_numpy(G[tr])
        Ytr = torch.from_numpy(Y[tr])
        h_in = Xs.shape[-1]

        heads = {"R2_gru": GRUHead(h_in, n_max, args.proj, args.hidden),
                 "R3_ssm": SelectiveSSMHead(h_in, n_max, args.proj, args.hidden),
                 "R4_attnpool": AttnPoolHead(h_in, n_max, args.proj)}
        for name, model in heads.items():
            if s == 0:
                print(f"{name}: {n_params(model):,} params")
                assert n_params(model) < 1_200_000, f"{name} over the ~1M budget"
            train_head(model, Xtr, Gtr, Ytr, args.epochs, args.aux_weight,
                       bool(args.augment), lr=args.lr, wd=args.wd, seed=s)
            tc, _, _ = predict(model, Xtr, n_max)
            train_em[name].append(float((tc == G[tr]).mean()))
            pc, pr, pf = predict(model, Xev, n_max)
            head_pf[name].append(float((pf == Y[ev]).mean()))
            pool[name]["cls"].append(pc)
            pool[name]["reg"].append(pr)
            em0 = float((pc == G[ev]).mean())
            per_seed_em[name].append(em0)

            # order canary: clean counting is permutation-invariant
            deltas = []
            for k in range(args.canary_perms):
                prm = np.stack([np.random.default_rng(1000 + s * 97 + k * 7 + i)
                                .permutation(N) for i in range(len(ev))])
                Xp = Xev[np.arange(len(ev))[:, None], prm]
                pck, _, _ = predict(model, Xp, n_max)
                deltas.append(abs(float((pck == G[ev]).mean()) - em0))
            canary[name].append(max(deltas))

            if s == 0:   # save one ckpt per arm, scaler included (guardrail)
                torch.save({"state_dict": model.state_dict(),
                            "scaler_mu": mu, "scaler_sd": sd,
                            "pca_components": None if pca is None
                            else pca.components_,
                            "pca_mean": None if pca is None else pca.mean_,
                            "n_max": n_max, "proj": args.proj,
                            "hidden": args.hidden, "layer": args.layer,
                            "aux_weight": args.aux_weight,
                            "npz": args.npz, "seed": s},
                           out / f"{name}_seed0.pt")
        pool_true.append(G[ev])

    # ------------------------------------------------------------- pooled report
    true = np.concatenate(pool_true)
    lines = [f"RECAGG P0 — {args.label}  ({args.npz})",
             f"n={m} N={N} layer={args.layer} seeds={args.seeds} "
             f"epochs={args.epochs} proj={args.proj} hidden={args.hidden}",
             f"pool class dist: {dist_line(G, n_max)}",
             f"p_frame = {np.mean(p_frames):.4f} +- {np.std(p_frames):.4f}   "
             f"p_frame^{N} bound = {np.mean(p_frames) ** N:.3f}",
             ""]
    hdr = (f"{'arm':<14} {'EM':>6} {'ci95':>15} {'major':>6} {'bal':>6} "
           f"{'EM_reg':>6} {'trainEM':>7} {'head_pf':>7} "
           f"{'seedEM mean+-sd':>16} {'canaryΔ':>8}")
    lines.append(hdr)
    results = {}
    for a in arms:
        pc = np.concatenate(pool[a]["cls"])
        pr = np.concatenate(pool[a]["reg"])
        rep = class_report(pc, true, n_classes=n_max + 1)
        lo, hi, _ = cluster_bootstrap_ci((pc == true)[:, None])
        em_reg = float((pr == true).mean())
        se = per_seed_em[a]
        can = max(canary[a])
        tem = float(np.mean(train_em[a]))
        hpf = float(np.mean(head_pf[a]))
        lines.append(f"{a:<14} {rep['raw']:>6.3f} [{lo:.3f},{hi:.3f}] "
                     f"{rep['majority']:>6.3f} {rep['balanced']:>6.3f} "
                     f"{em_reg:>6.3f} {tem:>7.3f} {hpf:>7.4f} "
                     f"{np.mean(se):>8.3f}+-{np.std(se):.3f} {can:>8.3f}")
        results[a] = {"em_cls": rep["raw"], "em_ci": [lo, hi],
                      "majority": rep["majority"], "balanced": rep["balanced"],
                      "recall": rep["recall"], "support": rep["support"],
                      "em_reg": em_reg, "train_em": tem, "head_p_frame": hpf,
                      "seed_em": se, "canary_max_delta": can}
    lines.append("")
    for a in arms:
        r = results[a]
        rec = " ".join(f"c{c}:{v:.2f}" if v == v else f"c{c}:n/a"
                       for c, v in enumerate(r["recall"]))
        lines.append(f"{a} per-class recall: {rec}  support {r['support']}")
    lines += ["",
              f"supervision: aux_weight={args.aux_weight} (per-frame bit BCE, parity "
              f"with R1's probe), pca={args.pca} (0=raw), augment={args.augment} "
              f"(subset-count augmentation).",
              "head_pf = the arm's own per-frame bit accuracy on eval (R1: the probe).",
              "readout: EM = classification argmax (primary); EM_reg = scalar-round.",
              "canaryΔ = max |EM under frame permutation - EM| over seeds x perms "
              "(clean counting is permutation-invariant; large Δ = position artifact).",
              f"wall {time.time() - t0:.0f}s"]

    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "p_frame_mean": float(np.mean(p_frames)),
         "p_frame_bound": float(np.mean(p_frames) ** N),
         "results": results}, indent=2))
    print(f"\nwrote {out}/report.txt, results.json, *_seed0.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
