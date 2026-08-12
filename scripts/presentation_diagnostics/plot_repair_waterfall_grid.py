#!/usr/bin/env python3
"""Repair waterfall across N=8..128: small multiples, one staircase per length.

Rungs (same readout — held-out gate->tally — everywhere):
  1 joint model readout      curves.csv joint rows (single reader, full softmax)
  2 + a reader per frame     qkv qD_kvD   (query side 1: addressing/competition)
  3 + clean the readers      qkv qC_kvD   (query side 2: reader states)
  4 + clean the frames       qkv qC_kvC   (value noise)
  5 + posreset & Q-first     curves.csv fenced rows (N=8/32/128) or a gate fit on the
                             fenced_supply probe caches (N=16/64)
Carrier (1-token) annotated where a cell exists. Missing cells are skipped, not faked.

qkv runs are auto-discovered from outputs/presentation/qkv_swap/*/report.txt
(mapped to N via the data_root in the header; latest run per N wins).

Usage: python plot_repair_waterfall_grid.py [<output_dir>]
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES_CSV = "outputs/presentation/curves/20260731_202940/curves.csv"
QKV_GLOB = "outputs/presentation/qkv_swap/2*/report.txt"
FENCED_CACHES = {16: "outputs/presentation/fenced_supply/N16",
                 64: "outputs/presentation/fenced_supply/N64"}
NS = [8, 16, 32, 64, 128]
CELL_RE = re.compile(r"^(q[CD]_kv[CD]): .*gate->tally ([\d.]+)±([\d.]+)")
HEAD_RE = re.compile(r"n=(\d+).*data=\S*seq_len_(\d+)")
RUNGS = [("joint model\nreadout", "#9a9891"), ("+ reader\nper frame", "#86b6ef"),
         ("+ clean the\nreaders", "#2a78d6"), ("+ clean the\nframes", "#eb6834"),
         ("+ posreset\n& Q-first", "#1baf7a")]


def gate_tally_from_cache(cache_path, layer=16, seeds=5):
    import torch
    from sklearn.linear_model import LogisticRegression
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    X = np.asarray(c["rep"][layer], dtype=np.float32)
    Y = np.asarray(c["labels"], dtype=int)
    G = np.asarray(c["gold"], dtype=int)
    n, NF, H = X.shape
    accs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        tr, ev = idx[: n // 2], idx[n // 2:]
        clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
        pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
        accs.append(float((pr.sum(1) == G[ev]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "outputs/presentation/waterfall_grid/local")
    out.mkdir(parents=True, exist_ok=True)

    curves = {}
    with open(CURVES_CSV) as f:
        for row in csv.DictReader(f):
            curves[(row["arm"], int(row["N"]))] = (float(row["tally_exact"]), float(row["std"]))

    qkv = {}  # N -> {cell: (acc, sd)}, latest run per N
    for rp in sorted(Path(".").glob(QKV_GLOB)):
        txt = rp.read_text()
        hm = HEAD_RE.search(txt)
        if not hm:
            continue
        n_frames = int(hm.group(2))
        cells = {m.group(1): (float(m.group(2)), float(m.group(3)))
                 for m in (CELL_RE.match(ln.strip()) for ln in txt.splitlines()) if m}
        if len(cells) == 4:
            qkv[n_frames] = (cells, str(rp.parent))

    fenced_fit = {}
    for n_frames, root in FENCED_CACHES.items():
        caches = sorted(Path(root).glob("2*/messages_cache.pt"))
        if caches:
            fenced_fit[n_frames] = (gate_tally_from_cache(caches[-1]), str(caches[-1].parent))

    rows = []
    fig, axes = plt.subplots(1, len(NS), figsize=(15.5, 4.3), sharey=True)
    for ax, n_frames in zip(axes, NS):
        vals = [curves.get(("joint", n_frames))]
        cells = qkv.get(n_frames, ({}, ""))[0]
        vals += [cells.get("qD_kvD"), cells.get("qC_kvD"), cells.get("qC_kvC")]
        vals.append(curves.get(("fenced", n_frames)) or
                    (fenced_fit.get(n_frames, (None,))[0]))
        for i, ((lab, col), v) in enumerate(zip(RUNGS, vals)):
            if v is None:
                ax.annotate("n/a", (i, 0.03), ha="center", fontsize=7, color="#9a9891")
                rows.append([n_frames, lab.replace("\n", " "), "", ""])
                continue
            acc, sd = v
            ax.bar(i, acc, 0.68, yerr=sd, capsize=2, color=col)
            ax.annotate(f"{acc:.2f}", (i, acc + sd + 0.02), ha="center", fontsize=8)
            rows.append([n_frames, lab.replace("\n", " "), f"{acc:.3f}", f"{sd:.3f}"])
        cr = curves.get(("carrier", n_frames))
        ttl = f"N={n_frames}"
        if cr:
            ttl += f"  (carrier {cr[0]:.2f})"
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_ylim(0, 1.14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("exact-count accuracy\n(gate→tally, held-out)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in RUNGS]
    fig.legend(handles, [l.replace("\n", " ") for l, _ in RUNGS], ncol=5, frameon=False,
               fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("From the joint failure to the method, one repair at a time — across context length",
                 fontsize=12, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"repair_waterfall_grid.{ext}", dpi=300, bbox_inches="tight")

    with open(out / "waterfall_grid.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "rung", "acc", "std"], *rows])
    srcs = [f"- curves: `{CURVES_CSV}`"]
    srcs += [f"- qkv N={n}: `{d}`" for n, (_, d) in sorted(qkv.items())]
    srcs += [f"- fenced N={n}: `{d}`" for n, (_, d) in sorted(fenced_fit.items())]
    (out / "ABOUT.md").write_text(
        "# Repair waterfall across N — the joint failure decomposed at every length\n\n"
        "One panel per context length, five rungs each (same held-out gate->tally readout):\n"
        "joint single-locus readout -> per-frame readers (query side 1: addressing) ->\n"
        "clean readers (query side 2) -> clean frames (value noise) -> full fence\n"
        "(+posreset & Q-first). Carrier cell annotated in panel titles where measured.\n"
        "Rungs 2-4 use the q/kv-swap probe protocol (own-frame softmax) — a controlled\n"
        "decomposition stitched with the curves/fenced instruments; missing cells shown\n"
        "as n/a, never interpolated.\n\n## Sources\n" + "\n".join(srcs) +
        "\n\nArtifacts: repair_waterfall_grid.png/pdf, waterfall_grid.csv. Generated by\n"
        "`scripts/presentation_diagnostics/plot_repair_waterfall_grid.py`.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
