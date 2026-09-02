#!/usr/bin/env python3
"""TREEFOLD figures F-A..F-D + CSVs (CPU; run after cells land).

F-A  fan-in law: EM vs k (log-x: 2,4,8,16,N) at N=128, count-like + ALL,
     majority line + quoted reference marks (chunk-16, Arm A). THE headline.
F-B  length curves on identical samples: T1 / T3 / quoted baselines, N=16..128.
F-C  per-level fidelity: p_leaf, p_merge; predicted vs measured EM (T1).
F-D  perception: T3 count-like per-frame leaf accuracy vs Arm B full-state.

Baselines are QUOTED from canonical recagg runs (never rerun):
  armC v3 ALL 0.89/0.92/0.85/0.90 (armC_compile/20260817_193946_qwen14b_v3)
  Arm B   ALL 0.79/0.81/0.72/0.72 (armB_execute/20260817_213301_v3progs)
  chunk-16 0.78/0.58/0.42 @32/64/128 (armB_adapter/*_mamba_ft_chunk16)
  Arm B caption fidelity 0.913/char, 0.637/frame-exact
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_NS = [16, 32, 64, 128]
ARMC_V3_ALL = {16: 0.89, 32: 0.92, 64: 0.85, 128: 0.90}
ARMB_ALL = {16: 0.79, 32: 0.81, 64: 0.72, 128: 0.72}
CHUNK16 = {32: 0.78, 64: 0.58, 128: 0.42}
ARMB_CHAR_ACC = 0.913


def em_of(run_dir, n=None, subset=None):
    """Mean EM over records.json, optionally at one N / one type subset."""
    recs = json.load(open(Path(run_dir) / "records.json"))
    sel = [r for r in recs if (n is None or r["N"] == n)
           and (subset is None or r["type"] in subset)]
    return float(np.mean([r["em"] for r in sel])) if sel else float("nan")


def majority_of(run_dir, n):
    from collections import Counter

    from tf_common import norm
    recs = [r for r in json.load(open(Path(run_dir) / "records.json"))
            if r["N"] == n]
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["type"]].append(norm(r["gold"]))
    return float(np.mean([Counter(g).most_common(1)[0][1] / len(g)
                          for g in by_type.values()]))


def main() -> int:
    from tf_common import COUNT_LIKE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1", help="T1 run dir (fan-2 oracle)")
    ap.add_argument("--t2", nargs="*", default=[],
                    help="k:dir pairs, e.g. 4:outputs/... 8:... N:...")
    ap.add_argument("--t3", help="T3 run dir")
    ap.add_argument("--t4", help="T4 run dir")
    ap.add_argument("--t3-leaves", help="leaf_vlm run dir (summary.json)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------ F-A fan law
    if args.t1 and args.t2:
        ks, em_all, em_cl = [2], [em_of(args.t1, 128)], \
            [em_of(args.t1, 128, COUNT_LIKE)]
        for spec in args.t2:
            k, d = spec.split(":", 1)
            ks.append(129 if k.upper() == "N" else int(k))
            em_all.append(em_of(d, 128))
            em_cl.append(em_of(d, 128, COUNT_LIKE))
        order = np.argsort(ks)
        ks = [ks[i] for i in order]
        labels = ["N" if k == 129 else str(k) for k in ks]
        em_all = [em_all[i] for i in order]
        em_cl = [em_cl[i] for i in order]
        maj = majority_of(args.t1, 128)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(range(len(ks)), em_all, "o-", label="ALL (24 types)")
        ax.plot(range(len(ks)), em_cl, "s-", label="count-like")
        ax.axhline(maj, ls=":", c="gray", label=f"majority {maj:.2f}")
        ax.axhline(CHUNK16[128], ls="--", c="tab:orange", alpha=0.6,
                   label=f"chunk-16 ref {CHUNK16[128]:.2f}")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("fan-in k (N=128)")
        ax.set_ylabel("exact match")
        ax.set_ylim(0, 1)
        ax.set_title("F-A  Fan-in law: model-merge EM vs k at N=128")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "FA_fanin_law.png", dpi=200)
        rows = ["k,em_all,em_countlike"]
        rows += [f"{l},{a:.4f},{c:.4f}"
                 for l, a, c in zip(labels, em_all, em_cl)]
        (out / "FA_fanin_law.csv").write_text("\n".join(rows) + "\n")

    # -------------------------------------------------------- F-B length curves
    if args.t1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        curves = [("T1 tree (oracle leaves)", args.t1, "o-"),
                  ("T3 tree (VLM leaves)", args.t3, "s-"),
                  ("T4 tree (Arm B captions)", args.t4, "^-")]
        for label, d, style in curves:
            if d:
                ax.plot(BASE_NS, [em_of(d, n) for n in BASE_NS], style,
                        label=label)
        ax.plot(BASE_NS, [ARMC_V3_ALL[n] for n in BASE_NS], "--", c="gray",
                label="armC v3 (oracle+Python)")
        ax.plot(BASE_NS, [ARMB_ALL[n] for n in BASE_NS], ":", c="gray",
                label="Arm B (VLM+Python)")
        ax.plot(list(CHUNK16), list(CHUNK16.values()), "-.", c="tab:orange",
                alpha=0.6, label="chunk-16 (tuned mamba)")
        ax.plot(BASE_NS, [majority_of(args.t1, n) for n in BASE_NS], ":",
                c="lightgray", label="majority")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BASE_NS)
        ax.set_xticklabels(BASE_NS)
        ax.set_xlabel("N (frames)")
        ax.set_ylabel("exact match (ALL 24 types)")
        ax.set_ylim(0, 1)
        ax.set_title("F-B  Length curves on identical samples")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "FB_length_curves.png", dpi=200)

    # ------------------------------------------------------- F-C fidelity (T1)
    if args.t1 and (Path(args.t1) / "fidelity.json").is_file():
        fid = json.load(open(Path(args.t1) / "fidelity.json"))
        agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # N -> lv -> [h,n]
        pl = defaultdict(list)
        for f in fid.values():
            if f["p_leaf"] is None:
                continue
            pl[f["N"]].append(f["p_leaf"])
            for lv, v in f["levels"].items():
                agg[f["N"]][int(lv)][0] += v["hit"]
                agg[f["N"]][int(lv)][1] += v["n"]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for n in sorted(agg):
            lvs = sorted(agg[n])
            ax.plot(lvs, [agg[n][l][0] / max(agg[n][l][1], 1) for l in lvs],
                    "o-", label=f"N={n} (p_leaf {np.mean(pl[n]):.3f})")
        ax.set_xlabel("tree level")
        ax.set_ylabel("p_merge (pooled, count-like)")
        ax.set_ylim(0, 1.02)
        ax.set_title("F-C  Per-level merge fidelity (T1)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "FC_level_fidelity.png", dpi=200)

    # ------------------------------------------------------- F-D perception
    if args.t3_leaves:
        summ = json.load(open(Path(args.t3_leaves) / "summary.json"))
        acc = summ.get("countlike_leaf_acc_mean")
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        bars = [("targeted\n(T3 leaf, count-like)", acc, "tab:blue"),
                ("full-state\n(Arm B, per-char)", ARMB_CHAR_ACC, "tab:gray")]
        ax.bar([b[0] for b in bars], [b[1] or 0 for b in bars],
               color=[b[2] for b in bars])
        for i, (_, v, _) in enumerate(bars):
            if v:
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("per-frame accuracy")
        ax.set_title("F-D  Targeted vs full-state perception")
        fig.tight_layout()
        fig.savefig(out / "FD_perception.png", dpi=200)

    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
