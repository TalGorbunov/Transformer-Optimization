#!/usr/bin/env python3
"""SPARSE figures F1-F4 (+ CSV tables alongside) -> outputs/sparse/fig/.

Reads run dirs only; skips series whose cells haven't landed (warns). Palette =
the validated categorical reference order (dataviz skill): blue, orange, aqua,
yellow, magenta. One axis per chart; thin marks; direct labels; no dual axes.

F1 flat-line: acc on the k in 1..8 band vs N — frozen / P1b / P1b+logN /
   P1g-oracle / P1g-model.
F2 capacity: acc vs gold k at N=128 (P1g-oracle, S4) + frozen c(fan) overlay.
F3 alpha panel: read/verdict alpha with CIs, gated vs ungated; sensitivity vs k.
F4 S0 sweep: acc vs tau per N + tau2-all control.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "outputs/sparse/fig"
FIG.mkdir(parents=True, exist_ok=True)

C = {"frozen": "#2a78d6", "p1b": "#eb6834", "logn": "#eda100",
     "oracle": "#1baf7a", "model": "#e87ba4"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})
NS = [8, 16, 32, 64, 128]


def latest(pat):
    ds = sorted(REPO.glob(pat))
    return ds[-1] if ds else None


def band_acc_from_preds(run_glob, source_tag, lo=1, hi=8):
    """gold in [lo,hi] accuracy per N from a run dir's longn_predictions.csv."""
    d = latest(run_glob)
    if d is None:
        return {}
    f = d / "longn_predictions.csv"
    if not f.exists():
        return {}
    agg = {}
    for r in csv.DictReader(open(f)):
        if source_tag not in r["source"]:
            continue
        g = int(r["gold"])
        if not (lo <= g <= hi):
            continue
        N = int(r["n_frames"])
        a = agg.setdefault(N, [0, 0])
        a[1] += 1
        a[0] += int(r["pred"] != "" and int(r["pred"]) == g)
    return {N: v[0] / v[1] for N, v in agg.items() if v[1]}


def frozen_from_armor(lo=1, hi=8):
    """Frozen plain-arm acc on the gold<=8 probe slice: pred_base==gold in
    ARMOR-A logits.csv (evid rows), per N."""
    out = {}
    for N in NS:
        d = latest(f"outputs/armor/hahn/20260822_211457_N{N}")
        if d is None or not (d / "logits.csv").exists():
            continue
        ok = n = 0
        for r in csv.DictReader(open(d / "logits.csv")):
            if r["flip_kind"] != "evid":
                continue
            g = int(r["gold"])
            if not (lo <= g <= hi):
                continue
            ok += int(int(r["pred_base"]) == g)
            n += 1
        if n:
            out[N] = ok / n
    return out


def merged(*maps):
    out = {}
    for m in maps:
        out.update(m)
    return out


# --------------------------------------------------------------------------- F1
def f1():
    series = {
        "frozen (plain)": (C["frozen"], frozen_from_armor()),
        "P1b fenced": (C["p1b"], merged(
            band_acc_from_preds("outputs/loramech/p1b_fenced_ep10/2*", "exam_ff"),
            band_acc_from_preds("outputs/loramech/p1b_zs_64_128/2*", "exam_ff"))),
        "P1b + eval logN": (C["logn"], band_acc_from_preds(
            "outputs/loramech/n3_logn_fenced/2*", "exam_ff")),
        "P1g oracle gate": (C["oracle"], merged(
            band_acc_from_preds("outputs/sparse/s1_p1g_r3/2*", "exam_ff"),
            band_acc_from_preds("outputs/sparse/s1_ladder_64_128/2*", "exam_ff"))),
        "P1g model gate": (C["model"], merged(
            band_acc_from_preds("outputs/sparse/s2/eval_short/2*", "exam_ff"),
            band_acc_from_preds("outputs/sparse/s2/eval_long/2*", "exam_ff"))),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = ["series," + ",".join(f"N{N}" for N in NS)]
    for name, (col, m) in series.items():
        xs = [N for N in NS if N in m]
        if not xs:
            print(f"[F1] missing series: {name}", file=sys.stderr)
            continue
        ys = [m[N] for N in xs]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, label=name,
                zorder=3 if "P1g" in name else 2)
        if "P1g" in name:  # direct-label the headline series only; legend covers rest
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", color=col, fontsize=9, va="center")
        rows.append(name + "," + ",".join(f"{m.get(N, float('nan')):.3f}" for N in NS))
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("N (frames)")
    ax.set_ylabel("exact-match accuracy, gold 1–8 band")
    ax.set_title("F1 — the gate makes the small-count band flat in N (train N ≤ 16)")
    ax.axvspan(7, 17, color="0.85", alpha=0.35, lw=0, zorder=1)
    ax.text(11.3, -0.01, "train window", fontsize=8, color="0.35", ha="center")
    ax.legend(loc="center left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "F1_flatline.png")
    (FIG / "F1_flatline.csv").write_text("\n".join(rows) + "\n")
    plt.close(fig)


# --------------------------------------------------------------------------- F2
def f2():
    d = latest("outputs/sparse/s4/eval/2*")
    if d is None or not (d / "longn_predictions.csv").exists():
        print("[F2] S4 not landed", file=sys.stderr)
        return
    per = {64: {}, 128: {}}
    for r in csv.DictReader(open(d / "longn_predictions.csv")):
        N = int(r["n_frames"])
        g = int(r["gold"])
        a = per[N].setdefault(g, [0, 0])
        a[1] += 1
        a[0] += int(r["pred"] != "" and int(r["pred"]) == g)
    fan = [2, 4, 8, 16, 32, 64]
    rr = [0.98, 0.90, 0.65, 0.44, 0.21, 0.12]  # superquery frozen c(fan) anchors
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for N, col, mk in ((64, C["oracle"], "s"), (128, "#0d7a52", "o")):
        gs = sorted(g for g in per[N] if 1 <= g < N)  # k=N is the no-hidden special case
        ys = [per[N][g][0] / per[N][g][1] for g in gs]
        ax.plot(gs, ys, "-" + mk, color=col, lw=2, ms=6,
                label=f"P1g + oracle gate, N={N} (S4)")
        if N in per[N]:
            aN = per[N][N]
            ax.plot([N], [aN[0] / aN[1]], mk, color=col, ms=8, mfc="none", mew=1.8)
            ax.annotate("k=N (all frames,\nnothing hidden)", (N, aN[0] / aN[1]),
                        xytext=(-8, -22 if aN[0] else 12), textcoords="offset points",
                        fontsize=7.5, color=col, ha="right")
    ax.plot(fan, rr, "--^", color=C["frozen"], lw=1.6, ms=6,
            label="frozen c(fan) (superquery rr)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("k (true count / fan)")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title("F2 — capacity in k with N fixed (the wall is in k now)")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.text(1.05, 0.515, "0.5", fontsize=8, color="0.4")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "F2_capacity.png")
    rows = ["N,gold,acc,n"]
    for N in (64, 128):
        for g in sorted(per[N]):
            a = per[N][g]
            rows.append(f"{N},{g},{a[0]/a[1]:.3f},{a[1]}")
    (FIG / "F2_capacity.csv").write_text("\n".join(rows) + "\n")
    plt.close(fig)


# --------------------------------------------------------------------------- F3
def f3():
    # panel a: alpha dots with CI. Values measured in STATE (recomputed here from csvs
    # would duplicate the fit code; keep the logged numbers as the single source).
    alphas = [
        ("read L20\nungated (N2)", 0.802, 0.66, 0.97, C["p1b"]),
        ("read L20\nGATED (S3)", 0.066, -0.076, 0.195, C["oracle"]),
        ("verdict L20\nungated (N2)", 0.009, 0.001, 0.016, "#b8622f"),
        ("verdict L20\nGATED (S3)", -0.007, -0.030, 0.014, "#0d7a52"),
    ]
    ks = [1, 2, 4, 8, 16, 32]
    med20 = [41.9, 36.7, 20.3, 11.9, 4.1, 2.9]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    xs = np.arange(len(alphas))
    for i, (lab, v, lo, hi, col) in enumerate(alphas):
        a1.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="o", color=col,
                    ms=8, capsize=4, lw=2)
        a1.annotate(f"{v:+.2f}", (i, v), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, color=col)
    a1.axhline(0, color="0.5", lw=0.8)
    a1.set_xticks(xs)
    a1.set_xticklabels([a[0] for a in alphas], fontsize=8)
    a1.set_ylabel(r"$\alpha$  ($\Delta \propto N^{-\alpha}$)")
    a1.set_title("F3a — the gate removes the read's N-decay")
    a2.plot(ks, med20, "-o", color=C["oracle"], lw=2, ms=6, label="gated read L20 (S3)")
    kk = np.linspace(1, 32, 100)
    a2.plot(kk, med20[0] * ((ks[0] + 2.0) / (kk + 2.0)) ** 1.19, ":", color="0.4",
            lw=1.4, label=r"fit $\Delta \propto (k+2)^{-1.19}$")
    a2.axhspan(1.2, 1.4, color="0.8", alpha=0.5, lw=0)
    a2.text(1.05, 1.45, "bf16 floor", fontsize=8, color="0.4")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xticks(ks)
    a2.set_xticklabels(ks)
    a2.set_xlabel("base k (flip k → k+1, N = 64)")
    a2.set_ylabel("median ‖Δh‖ at the read locus")
    a2.set_title("F3b — sensitivity now decays in k, not N")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "F3_alpha.png")
    plt.close(fig)


# --------------------------------------------------------------------------- F4
def f4():
    taus = [1.0, 1.5, 2.0, 3.0, 4.0]
    accs = {}
    for t in taus:
        d = latest(f"outputs/sparse/s0_tau{t}_L12/2*")
        if d is None:
            continue
        for row in csv.DictReader(open(d / "longn_eval.csv")):
            N = int(Path(row["source"]).stem.split("N")[-1])
            accs.setdefault(N, {})[t] = float(row["accuracy"])
    d = latest("outputs/sparse/s0_tau2.0_L0/2*")
    all2 = {}
    if d:
        for row in csv.DictReader(open(d / "longn_eval.csv")):
            all2[int(Path(row["source"]).stem.split("N")[-1])] = float(row["accuracy"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for N, col in ((32, C["frozen"]), (64, C["p1b"]), (128, C["oracle"])):
        ts = sorted(accs.get(N, {}))
        ax.plot(ts, [accs[N][t] for t in ts], "-o", color=col, lw=2, ms=6,
                label=f"N={N}")
        if N in all2:
            ax.plot([2.0], [all2[N]], "D", color=col, ms=7, mfc="none", mew=1.6)
    ax.plot([], [], "D", color="0.3", mfc="none", label="τ=2 all-layers ctrl")
    ax.axvline(1.5, color="0.6", lw=0.9, ls=":")
    ax.text(1.52, 0.98, "over-sharpening\nthreshold", fontsize=8, color="0.4")
    ax.set_xlabel(r"sharpening factor $\tau$ (layers ≥ 12)")
    ax.set_ylabel("exact-match accuracy (100/cell)")
    ax.set_ylim(0, 1.02)
    ax.set_title("F4 — no eval-time temperature reaches the k-regime (S0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "F4_s0_sweep.png")
    plt.close(fig)


# --------------------------------------------------------------------------- F5
def f5():
    """Wave-3 flat-line: k in 1..8 band vs N — P1b / P1g / S8 vn / S9 nfree
    (oracle gates; S8-model dashed). S9 carries the prompt-variant label."""
    series = {
        "P1b fenced (no gate)": (C["p1b"], "-", merged(
            band_acc_from_preds("outputs/loramech/p1b_fenced_ep10/2*", "exam_ff"),
            band_acc_from_preds("outputs/loramech/p1b_zs_64_128/2*", "exam_ff"))),
        "P1g oracle gate (wave 2)": (C["frozen"], "-", merged(
            band_acc_from_preds("outputs/sparse/s1_p1g_r3/2*", "exam_ff"),
            band_acc_from_preds("outputs/sparse/s1_ladder_64_128/2*", "exam_ff"))),
        "S8 virtual-N + oracle gate": (C["oracle"], "-", merged(
            band_acc_from_preds("outputs/sparse/s8_vn/2*", "exam_ff"),
            band_acc_from_preds("outputs/sparse/s8_ladder_64_128/2*", "exam_ff"))),
        "S8 + model gate (deployed)": (C["oracle"], "--", merged(
            band_acc_from_preds("outputs/sparse/s8_mg/eval_short/2*", "exam_ff"),
            band_acc_from_preds("outputs/sparse/s8_mg/eval_long/2*", "exam_ff"))),
        "S9 N-free prompt + oracle gate*": (C["model"], ":", merged(
            band_acc_from_preds("outputs/sparse/s9_nfree/2*", "exam_ff"),
            band_acc_from_preds("outputs/sparse/s9_ladder_64_128/2*", "exam_ff"))),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    rows = ["series," + ",".join(f"N{N}" for N in NS)]
    for name, (col, ls, m) in series.items():
        xs = [N for N in NS if N in m]
        if not xs:
            print(f"[F5] missing series: {name}", file=sys.stderr)
            continue
        ys = [m[N] for N in xs]
        ax.plot(xs, ys, ls, marker="o", color=col, lw=2, ms=6, label=name,
                zorder=3 if "S8" in name or "S9" in name else 2)
        rows.append(name + "," + ",".join(f"{m.get(N, float('nan')):.3f}" for N in NS))
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("N (frames)")
    ax.set_ylabel("exact-match accuracy, gold 1–8 band")
    ax.set_title("F5 — full (k, N~) coverage makes the small-count band EXACT at every N")
    ax.axvspan(7, 17, color="0.85", alpha=0.35, lw=0, zorder=1)
    ax.text(11.3, -0.01, "train window", fontsize=8, color="0.35", ha="center")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.text(0.99, 0.97, "*new prompt anchor (N-free) — not comparable to N-prompt rows",
            transform=ax.transAxes, fontsize=6.5, color="0.4", ha="right", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "F5_flatline_w3.png")
    (FIG / "F5_flatline_w3.csv").write_text("\n".join(rows) + "\n")
    plt.close(fig)





# --------------------------------------------------------------------------- F6
def f6():
    """S11 three-panel: (a) log-log median dnorm at the read vs N — S9-gated
    (CI band) vs ungated-P1b (n2_hahn) vs frozen joint (ARMOR-A); floors shaded.
    (b) dnorm vs k with the (k+C)^-gamma fit. (c) margins vs N + flip rate."""
    rng = np.random.default_rng(0)

    def chain_meds(dir_pat, locus="final", layer="20"):
        out = {}
        for N in NS:
            p = REPO / dir_pat.format(N=N) / "pairs.csv"
            if not p.exists():
                continue
            v = [float(r["dnorm"]) for r in csv.DictReader(open(p))
                 if r["locus"] == locus and r["layer"] == layer and r["flip_kind"] == "evid"
                 and r["arm"] in ("p1fence", "plain")]
            if v:
                out[N] = v
        return out

    s11 = chain_meds("outputs/sparse/s11/gate_oracle_N{N}")
    n2 = chain_meds("outputs/loramech/n2_hahn/p1fence_ep10_N{N}")
    armor = chain_meds("outputs/armor/hahn/20260822_211457_N{N}")
    floors = []
    for N in NS:
        p = REPO / f"outputs/sparse/s11/gate_oracle_N{N}/controls.csv"
        if p.exists():
            floors += [float(r["dnorm"]) for r in csv.DictReader(open(p))
                       if r["kind"] == "ctrl" and r["locus"] == "final" and r["layer"] == "20"]
    fl_hi = np.percentile(floors, 75) if floors else 1.4

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for name, ch, col in (("S9 N-free + gate (S11)", s11, C["oracle"]),
                          ("P1b ungated (n2_hahn)", n2, C["p1b"]),
                          ("frozen joint (ARMOR-A)", armor, C["frozen"])):
        Ns = sorted(ch)
        if not Ns:
            continue
        med = [np.median(ch[N]) for N in Ns]
        a1.plot(Ns, med, "-o", color=col, lw=2, ms=6, label=name)
        if "S9" in name:
            los, his = [], []
            for N in Ns:
                v = np.array(ch[N])
                bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(500)]
                los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
            a1.fill_between(Ns, los, his, color=col, alpha=0.18, lw=0)
    a1.axhspan(0.5, fl_hi, color="0.85", alpha=0.6, lw=0)
    a1.text(8.2, fl_hi * 0.82, "measured floor (ctrl)", fontsize=7.5, color="0.4")
    a1.set_xscale("log", base=2); a1.set_yscale("log")
    a1.set_xticks(NS); a1.set_xticklabels(NS)
    a1.set_xlabel("N"); a1.set_ylabel("median ‖Δh‖, read locus L20")
    a1.set_title("F6a — α: text-controlled read is flat")
    a1.text(0.03, 0.10, "S9 α = +0.035 [−0.04, +0.12]\nP1b α = +0.80 [0.66, 0.97]\n"
            "frozen α = +0.72 [0.63, 0.78]", transform=a1.transAxes, fontsize=8)
    a1.legend(fontsize=7.5, loc="center right")

    ks = [1, 2, 4, 8, 16, 32]
    kd = {}
    for K in ks:
        p = REPO / f"outputs/sparse/s11/gate_oracle_N64_k{K}/pairs.csv"
        if p.exists():
            kd[K] = np.median([float(r["dnorm"]) for r in csv.DictReader(open(p))
                               if r["locus"] == "final" and r["layer"] == "20"])
    a2.plot(list(kd), list(kd.values()), "-o", color=C["oracle"], lw=2, ms=6,
            label="S9 gated read (S11)")
    kk = np.linspace(1, 32, 100)
    a2.plot(kk, kd[1] * ((1 + 5.0) / (kk + 5.0)) ** 1.19, ":", color="0.4", lw=1.5,
            label=r"fit $\Delta \propto (k+5)^{-1.19}$ [1.17,1.22]")
    a2.set_xscale("log", base=2); a2.set_yscale("log")
    a2.set_xticks(ks); a2.set_xticklabels(ks)
    a2.set_xlabel("base k (N = 64)"); a2.set_ylabel("median ‖Δh‖ L20")
    a2.set_title("F6b — γ: decay is in k")
    a2.legend(fontsize=7.5)

    Ns, margins, fliprate = [], [], []
    for N in NS:
        p = REPO / f"outputs/sparse/s11/gate_oracle_N{N}/logits.csv"
        if not p.exists():
            continue
        mb, fl, n = [], 0, 0
        for r in csv.DictReader(open(p)):
            if r["flip_kind"] != "evid":
                continue
            mb.append(float(r["margin_base"])); fl += int(r["pred_base"] != r["pred_flip"]); n += 1
        Ns.append(N); margins.append(np.median(mb)); fliprate.append(fl / n)
    a3.plot(Ns, margins, "-o", color=C["oracle"], lw=2, ms=6, label="median margin (digits)")
    a3.set_xscale("log", base=2)
    a3.set_xticks(NS); a3.set_xticklabels(NS)
    a3.set_ylim(0, 14)
    a3.set_xlabel("N"); a3.set_ylabel("margin (nats)")
    for N, fr in zip(Ns, fliprate):
        a3.annotate(f"{fr:.0%}", (N, 2.4), fontsize=8, ha="center", color="0.35")
    a3.text(0.5, 0.30, "flip-changes-answer rate", transform=a3.transAxes,
            fontsize=7.5, color="0.35", ha="center")
    a3.set_title("F6c — margins constant; flips land")
    a3.legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "F6_s9_alpha.png")
    rows = ["N,s11_med,n2_med,armor_med,margin,fliprate"]
    for N in NS:
        rows.append(f"{N},{np.median(s11[N]) if N in s11 else ''},"
                    f"{np.median(n2[N]) if N in n2 else ''},"
                    f"{np.median(armor[N]) if N in armor else ''},"
                    f"{margins[Ns.index(N)] if N in Ns else ''},"
                    f"{fliprate[Ns.index(N)] if N in Ns else ''}")
    (FIG / "F6_s9_alpha.csv").write_text("\n".join(rows) + "\n")
    plt.close(fig)





# --------------------------------------------------------------------------- F7
def f7():
    """S10 photograph: measured L20 evidence mass vs N (ungated, per k, with the
    fitted share law) and vs k under the gate (N-invariant)."""
    def cell_means(pat, Ns):
        out = {}
        for N in Ns:
            f = REPO / pat.format(N=N) / "mass.csv"
            if not f.exists():
                continue
            acc = {}
            for r in csv.DictReader(open(f)):
                if int(r["layer"]) != 20:
                    continue
                acc.setdefault(int(r["k"]), []).append(float(r["evid_mass"]))
            for k, v in acc.items():
                out[(N, k)] = float(np.mean(v))
        return out
    p1b = cell_means("outputs/sparse/s10/p1b_N{N}", (8, 16, 32, 64))
    gat = cell_means("outputs/sparse/s10/gated_N{N}", (8, 32, 128))
    s_fit, C_fit = 0.30, 8.0  # L20 fit from the S10 analysis (R2 0.969)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for k, col in ((2, C["frozen"]), (4, C["p1b"]), (8, C["oracle"])):
        Ns = [N for N in (8, 16, 32, 64) if (N, k) in p1b]
        a1.plot(Ns, [p1b[(N, k)] for N in Ns], "o", color=col, ms=7, label=f"measured k={k}")
        nn = np.linspace(8, 64, 100)
        a1.plot(nn, k * np.exp(s_fit) / (k * np.exp(s_fit) + (nn - k) + C_fit), "-",
                color=col, lw=1.4, alpha=0.7)
    a1.set_xscale("log", base=2)
    a1.set_xticks([8, 16, 32, 64]); a1.set_xticklabels([8, 16, 32, 64])
    a1.set_xlabel("N"); a1.set_ylabel("evidence attention mass (L20, head-mean)")
    a1.set_title("F7a — ungated: the share law, photographed\n"
                 r"lines: $k e^{s}/(k e^{s}+(N-k)+C)$, s=0.30, C=8, $R^2$=0.97")
    a1.legend(fontsize=8)
    for N, mk in ((8, "o"), (32, "s"), (128, "^")):
        ks = [k for k in (2, 4, 8) if (N, k) in gat]
        a2.plot(ks, [gat[(N, k)] for k in ks], "-" + mk, color=C["oracle"], lw=1.5,
                ms=7, alpha=0.55 + 0.15 * (N == 128), label=f"gated, N={N}")
    kk = np.linspace(2, 8, 50)
    a2.plot(kk, kk * np.exp(0.26) / (kk * np.exp(0.26) + 46.0), ":", color="0.4",
            lw=1.5, label="fit $k e^{s'}/(k e^{s'}+C')$, s'=0.26, C'=46")
    a2.set_xscale("log", base=2)
    a2.set_xticks([2, 4, 8]); a2.set_xticklabels([2, 4, 8])
    a2.set_xlabel("k"); a2.set_ylabel("evidence attention mass (L20)")
    a2.set_title("F7b — gated: N is gone\n(three N overlaid, ≤6% apart)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "F7_attn_photo.png")
    rows = ["arm,N,k,evid_mass_L20"]
    for (N, k), v in sorted(p1b.items()):
        rows.append(f"p1b,{N},{k},{v:.4f}")
    for (N, k), v in sorted(gat.items()):
        rows.append(f"gated,{N},{k},{v:.4f}")
    (FIG / "F7_attn_photo.csv").write_text("\n".join(rows) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    f1()
    f2()
    f3()
    f4()
    f5()
    f6()
    f7()
    print("wrote", FIG)
