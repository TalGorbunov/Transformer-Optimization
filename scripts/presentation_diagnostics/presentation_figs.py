#!/usr/bin/env python3
"""Presentation figures (CPU, cache-only): the GNN-narrative panels.

Figures (--fig):
  pca         2D PCA of per-frame messages @L16 (N=8 caches), one panel per arm
              (joint / fenced replica / learned carrier), points colored by the
              per-frame ground-truth verdict; held-out linear d' annotated.
  curves      The SAME sum-readout (logistic per-frame gate -> tally, exactly the
              scripts/gate_tally.py methodology) on all three arms, vs N.
              Cache-backed cells only; every plotted number lands in curves.csv.
  saturation  Per-layer gate fit on the depth dump (carrier_depth_L12_N32):
              per-frame gate error and tally exact vs layer; the write window
              (L12-19) shaded.

Anchors these must be consistent with (RESULTS.md): joint d' ~2.0 @L16 N=8;
fenced qfirst 9.24; gate->tally 0.998 (fenced N=8) / 0.997 (carrier N=8);
joint gate ~0.09 @N=32; per-frame gate err 0.339@L12 -> 0.0051@L24 [2026-07-25].

Every run dir gets: ABOUT.md (plain-language summary + provenance), the figure
(png+pdf), and a CSV of every plotted number.

Usage:
  python scripts/presentation_figs.py --fig pca --output outputs/presentation/pca/<ts>
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------- caches (all frozen)
CK = Path("checkpoints")
LEG = Path("outputs_legacy/outputs/ladder/image_longN")
CACHES = {
    "joint": {  # probe format: msgs[layer][offset] = (n, NF, H); deployed locus L16/off9
        8: CK / "msgcache_joint_N8.pt",
        16: CK / "msgcache_joint_N16.pt",
        32: CK / "msgcache_joint_N32.pt",
        64: CK / "msgcache_joint_N64.pt",
        128: CK / "msgcache_joint_N128.pt",
    },
    "fenced": {  # rep[layer] = (n, NF, H); replicas + block fence + posreset + Q-first
        8: CK / "msgcache_replica_blockfence_qfirst_full900.pt",
        32: LEG / "replica_blockfence_qfirst_N32/20260717_210541/messages_cache.pt",
        128: LEG / "replica_blockfence_qfirst_N128/20260717_210452/messages_cache.pt",
    },
    "carrier": {  # npz X (n,NF,H), Y (n,NF) [,G]; proxy_room_k1 chain (the one with long-N evals)
        8: LEG / "carrier_token/20260717_201919_proxy_room_k1/messages_best.npz",
        32: LEG / "carrier_token_lengen_N32/20260718_010210_proxy_room_k1/messages_eval.npz",
        128: LEG / "carrier_token_lengen_N128/20260718_010126_proxy_room_k1/messages_eval.npz",
    },
    # canonical distilled carrier (checkpoints/ table) -- extra N=8 cell for the CSV
    "carrier_distill8": CK / "carrier_token_room_k1_messages_best.npz",
    "depth": LEG / "carrier_depth_L12_N32/20260727_220340_L12_r8_evalonly/carrier_states_cache.pt",
}

ARM_COLOR = {"joint": "#eb6834", "fenced": "#2a78d6", "carrier": "#1baf7a"}
ARM_LABEL = {
    "joint": "joint attention (no rewiring)",
    "fenced": "fenced replicas (rewired)",
    "carrier": "fenced + learned carrier",
}
VERDICT_COLOR = {0: "#eb6834", 1: "#2a78d6"}  # absent / present


def load_arm(arm: str, n_frames: int, layer: int = 16):
    """Return X (n, NF, H) float32, Y (n, NF) int, G (n,) int."""
    p = CACHES[arm][n_frames] if isinstance(CACHES[arm], dict) else CACHES[arm]
    if str(p).endswith(".npz"):
        z = np.load(p, allow_pickle=True)
        X = np.asarray(z["X"], dtype=np.float32)
        Y = np.asarray(z["Y"], dtype=int)
        G = np.asarray(z["G"], dtype=int) if "G" in z.files else Y.sum(1)
        return X, Y, G, str(p)
    c = torch.load(p, map_location="cpu", weights_only=False)
    if "msgs" in c:  # joint probe format
        X = np.asarray(c["msgs"][layer][9], dtype=np.float32)
    else:
        rep = c["rep"]
        if layer not in rep:
            layer = sorted(rep.keys())[-1]
            print(f"[warn] layer 16 absent in {p}, using L{layer}")
        X = np.asarray(rep[layer], dtype=np.float32)
    Y = np.asarray(c["labels"], dtype=int)
    G = np.asarray(c["gold"], dtype=int)
    return X, Y, G, str(p)


def gate_tally(X, Y, G, seeds=5, train_frac=0.5):
    """scripts/gate_tally.py methodology, verbatim: held-out logistic gate + sum."""
    n, NF, H = X.shape
    accs, ferrs, dps = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        ntr = int(train_frac * n)
        tr, ev = idx[:ntr], idx[ntr:]
        clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
        s = clf.decision_function(X[ev].reshape(-1, H))
        y = Y[ev].reshape(-1)
        m1, m0 = s[y == 1], s[y == 0]
        if len(m1) and len(m0):
            dps.append(float((m1.mean() - m0.mean()) / np.sqrt(0.5 * (m1.var() + m0.var()) + 1e-12)))
        pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
        ferrs.append(float((pr != Y[ev]).mean()))
        accs.append(float((pr.sum(1) == G[ev]).mean()))
    return (np.mean(accs), np.std(accs), np.mean(ferrs), np.std(ferrs),
            np.mean(dps), np.std(dps))


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e2", linewidth=0.7)
    ax.set_axisbelow(True)


def write_about(out: Path, title: str, body: str, inputs: list, artifacts: list):
    prov = "\n".join(
        f"- `{p}` ({datetime.datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d})"
        for p in inputs
    )
    arts = "\n".join(f"- `{a}`" for a in artifacts)
    (out / "ABOUT.md").write_text(
        f"# {title}\n\n{body.strip()}\n\n## Inputs (frozen caches)\n{prov}\n\n"
        f"## Artifacts\n{arts}\n\nGenerated by `scripts/presentation_figs.py` on "
        f"{datetime.date.today()}.\n"
    )


# ---------------------------------------------------------------------------- figures
def fig_pca(out: Path, seeds: int):
    arms = ["joint", "fenced", "carrier"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    rows, inputs = [], []
    for col, arm in enumerate(arms):
        ax, axh = axes[0, col], axes[1, col]
        X, Y, G, path = load_arm(arm, 8)
        inputs.append(path)
        n, NF, H = X.shape
        F = X.reshape(-1, H)
        F = F - F.mean(0)
        # PCA via SVD on (up to) 4000 sampled rows, project all
        rng = np.random.default_rng(0)
        sub = rng.choice(len(F), min(4000, len(F)), replace=False)
        _, _, Vt = np.linalg.svd(F[sub], full_matrices=False)
        P = F @ Vt[:2].T
        y = Y.reshape(-1)
        *_, dp, dps = gate_tally(X, Y, G, seeds=seeds)
        show = rng.choice(len(P), min(3000, len(P)), replace=False)
        for v, name in [(0, "frame without the event"), (1, "frame with the event")]:
            m = show[y[show] == v]
            ax.scatter(P[m, 0], P[m, 1], s=7, alpha=0.45, lw=0,
                       color=VERDICT_COLOR[v], label=name)
        ax.set_title(f"{ARM_LABEL[arm]}\nheld-out linear d′ = {dp:.1f}", fontsize=11)
        ax.set_xlabel("PC1")
        style_ax(ax)
        # bottom row: held-out projection onto the discriminant axis (seed-0 split)
        srng = np.random.default_rng(0)
        idx = srng.permutation(n)
        tr, ev = idx[: n // 2], idx[n // 2:]
        clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
        s = clf.decision_function(X[ev].reshape(-1, H))
        ye = Y[ev].reshape(-1)
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 45)
        for v in (0, 1):
            axh.hist(s[ye == v], bins=bins, density=True, alpha=0.55,
                     color=VERDICT_COLOR[v], lw=0)
        axh.set_xlabel("held-out projection onto the discriminant axis")
        style_ax(axh)
        rows.append([arm, n, NF, f"{dp:.2f}", f"{dps:.2f}", path])
    axes[0, 0].set_ylabel("PC2")
    axes[1, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Per-frame messages @L16, N=8 — same task, three graphs\n"
                 "top: unsupervised 2D view (PCA) · bottom: the 1D axis the readout actually uses",
                 fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pca_panel.{ext}", dpi=300)
    with open(out / "pca.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "n", "NF", "dprime", "dprime_std", "cache"], *rows])
    write_about(
        out, "PCA panel — what the readout sees, before and after graph rewiring",
        """Each point is ONE FRAME's message vector (the 3,584-dim hidden state the
readout must decode), projected to 2D with PCA and colored by whether the queried
event happens in that frame. Left: messages from plain joint attention (the
complete-graph baseline) — the two classes overlap (the over-squashing /
interference tax). Middle: same model, same prompt content, but with the
block-diagonal fence + position reset + question-first (graph rewiring) — the
classes separate. Right: the fenced graph with the single learned carrier token
per frame instead of 20-token question replicas. The annotated d′ is a held-out
linear discriminant on the full-dim messages (not the 2D projection), the same
fitting protocol as scripts/gate_tally.py. The BOTTOM row projects held-out
messages onto that discriminant axis (1D histograms per class) — the view the
readout actually uses; PCA (top) can understate separation when the top variance
directions are not the class direction (visible on the carrier arm). Note the d′
here is the logistic-axis held-out estimator; RESULTS.md headline d′ values
(e.g. fenced 9.24) use the probe's d′_w estimator — different scale, same
ordering. Expected anchors: joint ~2, fenced/carrier well-separated
(RESULTS.md [2026-07-17], [2026-07-17→18]).""",
        inputs, ["pca_panel.png", "pca_panel.pdf", "pca.csv"])


def fig_curves(out: Path, seeds: int):
    cells = {"joint": [8, 16, 32, 64, 128], "fenced": [8, 32, 128], "carrier": [8, 32, 128]}
    rows, inputs = [], []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arm, ns in cells.items():
        xs, ys, es = [], [], []
        for n_frames in ns:
            X, Y, G, path = load_arm(arm, n_frames)
            inputs.append(path)
            acc, acc_sd, ferr, ferr_sd, dp, _ = gate_tally(X, Y, G, seeds=seeds)
            print(f"[{arm} N={n_frames}] tally exact {acc:.3f}±{acc_sd:.3f} "
                  f"per-frame err {ferr:.4f} d′ {dp:.2f}")
            xs.append(n_frames), ys.append(acc), es.append(acc_sd)
            rows.append([arm, n_frames, f"{acc:.4f}", f"{acc_sd:.4f}",
                        f"{ferr:.4f}", f"{ferr_sd:.4f}", f"{dp:.2f}", path])
            del X
        dy = {"joint": 0, "fenced": 9, "carrier": -11}[arm]  # de-collide end labels
        ax.errorbar(xs, ys, yerr=es, color=ARM_COLOR[arm], lw=2, marker="o", ms=6,
                    capsize=3, label=ARM_LABEL[arm])
        ax.annotate(ARM_LABEL[arm].split(" (")[0], (xs[-1], ys[-1]),
                    xytext=(6, dy), textcoords="offset points",
                    color=ARM_COLOR[arm], fontsize=9, va="center")
    # extra CSV-only cell: canonical distilled carrier @N=8
    X, Y, G, path = load_arm("carrier_distill8", 8)
    inputs.append(path)
    acc, acc_sd, ferr, ferr_sd, dp, _ = gate_tally(X, Y, G, seeds=seeds)
    rows.append(["carrier_distill(csv-only)", 8, f"{acc:.4f}", f"{acc_sd:.4f}",
                f"{ferr:.4f}", f"{ferr_sd:.4f}", f"{dp:.2f}", path])
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128], [8, 16, 32, 64, 128])
    ax.set_xlim(7, 300)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("N frames in context")
    ax.set_ylabel("exact-count accuracy (held-out)")
    ax.set_title("Same sum-readout (logistic gate → tally), three graphs", fontsize=12)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"gate_tally_vs_N.{ext}", dpi=300)
    with open(out / "curves.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "N", "tally_exact", "std", "per_frame_err",
                                  "err_std", "dprime", "cache"], *rows])
    write_about(
        out, "Sum-readout accuracy vs N — the GNN before/after figure",
        """One readout, three graphs. The readout is identical everywhere: a held-out
logistic per-frame gate + a sum (a one-layer DeepSets/GIN-style aggregator, the
scripts/gate_tally.py protocol, 5 seeds). What changes is the graph that produced
the messages: plain joint attention (complete graph), the fenced replica graph
(rewiring only, zero trained params), and the fenced graph with the learned
carrier token (3.5k trained params). Joint messages cap the readout regardless of
N (the supply ceiling); the rewired graphs stay near ceiling to N=128. The
carrier arm is the proxy_room_k1 chain (the only one with long-N caches); the
canonical distilled carrier's N=8 cell is in curves.csv for reference. Anchors:
fenced 0.998 @N=8, joint gate ~0.09 @N=32 (RESULTS.md [2026-07-17/18]).""",
        inputs, ["gate_tally_vs_N.png", "gate_tally_vs_N.pdf", "curves.csv"])


def fig_saturation(out: Path, seeds: int):
    c = torch.load(CACHES["depth"], map_location="cpu", weights_only=False)
    Y = np.asarray(c["labels"], dtype=int)
    G = np.asarray(c["gold"], dtype=int)
    layers = sorted(c["rep"].keys())
    rows = []
    errs, accs = [], []
    for L in layers:
        X = np.asarray(c["rep"][L], dtype=np.float32)
        acc, acc_sd, ferr, ferr_sd, dp, _ = gate_tally(X, Y, G, seeds=seeds)
        print(f"[L{L}] per-frame err {ferr:.4f}±{ferr_sd:.4f} tally {acc:.3f} d′ {dp:.2f}")
        errs.append((ferr, ferr_sd)), accs.append((acc, acc_sd))
        rows.append([L, f"{ferr:.4f}", f"{ferr_sd:.4f}", f"{acc:.4f}", f"{acc_sd:.4f}", f"{dp:.2f}"])
        del X
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for ax, vals, ylab in [(ax1, errs, "per-frame gate error"),
                           (ax2, accs, "exact-count accuracy")]:
        m, s = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        ax.axvspan(12, 19, color="#f0efec", zorder=0)
        ax.errorbar(layers, m, yerr=s, color="#2a78d6", lw=2, marker="o", ms=6, capsize=3)
        ax.set_ylabel(ylab)
        style_ax(ax)
    ax1.axvline(12, color="#52514e", lw=1, ls="--")
    ax1.annotate("fence lifts (L*=12)", (12, ax1.get_ylim()[1] * 0.92), fontsize=9,
                 color="#52514e", xytext=(5, 0), textcoords="offset points")
    ax1.annotate("write window", (15.5, ax1.get_ylim()[1] * 0.8), fontsize=9,
                 color="#52514e", ha="center")
    ax2.set_xlabel("layer the carrier states are read at")
    ax1.set_title("Carrier saturation curve (deployed stack, N=32)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"carrier_saturation.{ext}", dpi=300)
    with open(out / "saturation.csv", "w", newline="") as f:
        csv.writer(f).writerows([["layer", "per_frame_err", "err_std", "tally_exact",
                                  "acc_std", "dprime"], *rows])
    meta = {k: c.get(k) for k in ("sd", "ckpt", "truncate_at") if k in c}
    write_about(
        out, "Carrier saturation curve — which layer holds the verdict",
        f"""The deployed L*=12 stack was run once at N=32 and the carrier hidden
states dumped at every even layer 2–24 ({CACHES['depth']}). At each layer the
same held-out logistic gate → tally readout is fit. The per-frame verdict is
NOT yet in the carrier when the fence lifts at L*=12 — it gets WRITTEN during
the open+LoRA phase (shaded, L12–19) and saturates by ~L20–24. This is the
per-model procedure for locating the best-supplied layer (the knee), and it
explains why L* must sit well before the end: the write needs runway. Anchor:
err 0.339@L12 → 0.0051@L24 (RESULTS.md [2026-07-25]). Dump metadata: {meta}.""",
        [str(CACHES["depth"])],
        ["carrier_saturation.png", "carrier_saturation.pdf", "saturation.csv"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", required=True, choices=["pca", "curves", "saturation"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 10, "text.color": "#0b0b0b",
                         "axes.labelcolor": "#0b0b0b"})
    {"pca": fig_pca, "curves": fig_curves, "saturation": fig_saturation}[args.fig](out, args.seeds)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
