#!/usr/bin/env python3
"""DIAG figure set (docs/DIAGNOSTICS_2026-09-22.md §3): F1..F10 from CSV/JSON, CPU only.

Usage:
  python experiments/figs/diag_fig.py f8                    # per-qtype heatmap, official grid (drawable today)
  python experiments/figs/diag_fig.py f5                    # where the mass goes when you sharpen (legacy F11 csv)
  python experiments/figs/diag_fig.py all-available [--fits outputs/diag/fits] [--out outputs/diag/fig]
  subcommands: f1 f2 f3 f4 f5 f7 f8 f10 all-available   (--qtype / --layer narrow the JSON-driven ones)

Inputs that exist today
  F8  outputs/mmred_hf/armB_grid_v2_perlen/armB_grid_linear_ALL.csv   (seq_len,qtype,n,acc; legacy instrument)
      outputs/mmred_hf/frozen/grid_seq{8,16,32,64,128}_test/<latest>/report.txt   ("  qtype: a/b = acc" lines)
  F5  outputs/sparse/fig/F11_temperature_coupling.csv   (N,k,tau,competitor_mass,evidence_mass,...,other_mass,s0_accuracy)
Inputs written later by the DIAG fitters (<fits>/alpha.json, sharelaw.json, margins.json, headscan.json, and an
optional outputs/diag/eval_summary.csv with columns qtype,arm,cond,N,acc,n); schemas in the DIAG brief. A missing
input prints "inputs missing" and the figure is skipped — nothing is invented.

Every figure -> <out>/<name>.png (200 dpi) + .pdf + .csv (every plotted number) + one ABOUT line in <out>/README.md.
Design rules (Tal): the title is the claim; <= 4 series; one legend; N on a log2 x-axis (8..128); colors fixed
across the set; bf16 floor = light grey band; train window (N <= 16) shaded; exponents printed on the line.
Palette note: grey<->green is DeltaE 5.7 under deutan (validated), so every series also carries a marker shape,
a printed value/exponent and a legend entry — identity is never color-alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "outputs/diag/fig"
FITS_DEFAULT = REPO / "outputs/diag/fits"
LINEAR_CSV = REPO / "outputs/mmred_hf/armB_grid_v2_perlen/armB_grid_linear_ALL.csv"
FROZEN_ROOT = REPO / "outputs/mmred_hf/frozen"
F11_CSV = REPO / "outputs/sparse/fig/F11_temperature_coupling.csv"

NS = [8, 16, 32, 64, 128]
TRAIN_MAX = 16
COL = dict(frozen="#8a8f96", fenced="#1f9a6b", trained="#e0602a", gated="#2f6fd6", sink="#b48a2e",
           ink="#1c1e21", muted="#6b6f75", grid="#e4e6e9", floor="#e9eaec", train="#f3f3f0", surface="white")
MARK = dict(frozen="o", fenced="s", trained="^", gated="D")
RUNGS = [("frozen", "frozen read"), ("fenced", "fenced fact"), ("trained", "trained read"), ("gated", "gated read")]
NIAH = ["first_app", "final_app", "char_on_char_first_app", "char_on_char_final_app", "char_at_frame",
        "first_at_room", "last_at_room", "room_on_char_first_app", "room_on_char_final_app", "room_at_frame",
        "char_on_char_at_frame", "n_room_on_char_first_app", "n_room_on_char_final_app", "n_char_at_frame", "n_empty"]
DC = ["room_empty", "where_spend", "crowded_room", "who_spend", "spend_alone", "spend_together", "steps_in_room",
      "rooms_visited", "crowd_count"]


# ----------------------------------------------------------------------------- shared style & helpers
def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",          # its digits are tabular (equal width) by default
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": COL["muted"], "axes.linewidth": 0.8, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": COL["grid"], "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": COL["muted"], "ytick.color": COL["muted"], "axes.labelcolor": COL["ink"],
        "text.color": COL["ink"], "lines.linewidth": 2, "lines.markersize": 6, "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": COL["surface"], "figure.facecolor": COL["surface"],
        "savefig.facecolor": COL["surface"], "pdf.fonttype": 42, "figure.dpi": 100,
    })


def claim(fig, text: str, sub: str | None = None, y: float = 0.985) -> None:
    """The title is the claim: bold, left, wrapped to the figure width; subtitle in muted ink below it."""
    lines = textwrap.wrap(text, width=int(fig.get_figwidth() * 10.5))   # ~10.5 bold 11-pt chars per inch
    lh = 11 * 1.3 / 72 / fig.get_figheight()                   # one 11-pt line as a figure fraction
    fig.text(0.01, y, "\n".join(lines), ha="left", va="top", fontsize=11, fontweight="bold", color=COL["ink"])
    if sub:
        fig.text(0.01, y - lh * len(lines) - 0.004, sub, ha="left", va="top", fontsize=8.5, color=COL["muted"])


def footer(fig, text: str) -> None:
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=7.2, color=COL["muted"], wrap=True)


def xlog2(ax, ns=NS) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(ns[0] / 1.25, ns[-1] * 1.25)
    ax.set_xlabel("N (frames in context)")
    ax.tick_params(axis="x", which="minor", bottom=False)


def shade_train(ax) -> None:
    lo = ax.get_xlim()[0]
    ax.axvspan(lo, TRAIN_MAX * math.sqrt(2), color=COL["train"], zorder=0, lw=0)
    ax.text(lo * 1.05, 0.98, f"train N ≤ {TRAIN_MAX}", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=7.5, color=COL["muted"])


def floor_band(ax, hi: float, lo: float = 0.0, label: str = "bf16 floor") -> None:
    if hi is None or not np.isfinite(hi) or hi <= lo:
        return
    ax.axhspan(lo, hi, color=COL["floor"], zorder=0, lw=0)
    ax.text(ax.get_xlim()[1] / 1.03, hi, label, ha="right", va="top", fontsize=7.5, color=COL["muted"])


def exponent(ax, x, y, val, color, name="α", ci=None) -> None:
    """Print the fitted exponent on the line, just above its right end (stays inside the axes)."""
    s = f"{name} = {val:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
    ax.annotate(s, (x, y), xytext=(0, 7), textcoords="offset points", ha="right", va="bottom", fontsize=8,
                color=COL["ink"], bbox=dict(boxstyle="round,pad=0.15", fc=COL["surface"], ec=color, lw=0.8))


def shade(hexcol: str, f: float) -> tuple:
    """Blend a series color toward the surface (f = 0 -> color, 1 -> white) for an ordinal ramp."""
    r, g, b = mcolors.to_rgb(hexcol)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def rung_of(arm: str, locus: str | None = None) -> str | None:
    """Map a fit cell's (arm, locus) onto the four fixed rungs; None = not a diagnostic rung."""
    a = (arm or "").lower()
    if a.startswith("gated") or "gate" in a:
        return "gated"
    if any(k in a for k in ("trained", "sft", "lora", "p2", "adapter")):
        return "trained"
    if a.startswith("fenced") and (locus or "").lower() not in ("answer", "answer_row", "read"):
        return "fenced"
    return "frozen"


def save(fig, out: Path, name: str, rows: list[dict]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=200)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {png} (+ .pdf, .csv with {len(rows)} rows)")
    return png


def about(out: Path, key: str, name: str, message: str, data: str, regime: str) -> None:
    """One ABOUT line per figure in <out>/README.md (merged by figure key, other lines kept)."""
    readme = out / "README.md"
    head = "# outputs/diag/fig — the DIAG figure set\n\nOne line per figure: path · message · data source · regime. " \
           "Regenerate with `python experiments/figs/diag_fig.py <fig>` (every number comes from a file, none typed).\n\n"
    line = f"- **{key}** `{name}.png` (+ .pdf, .csv) — message: {message} — data: {data} — regime: {regime}"
    old = readme.read_text().splitlines() if readme.exists() else []
    kept = [l for l in old if l.strip() and not l.startswith("#") and not l.startswith(f"- **{key}**")
            and not l.startswith("One line per figure")]
    lines = sorted(kept + [line], key=lambda l: int(re.search(r"\*\*F(\d+)", l).group(1)) if re.search(r"\*\*F(\d+)", l) else 99)
    readme.write_text(head + "\n".join(lines) + "\n")


def rel(p: Path) -> str:
    """Repo-relative path for captions when inside the repo, else the absolute path."""
    return str(p.relative_to(REPO)) if REPO in p.resolve().parents else str(p)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def missing(fig: str, *paths: Path) -> bool:
    gone = [str(p) for p in paths if not p.exists()]
    if gone:
        print(f"{fig}: inputs missing — " + ", ".join(gone))
    return bool(gone)


def cond_tau(cond: str | None) -> float | None:
    m = re.search(r"tau[_=]?([\d.]+)", cond or "")
    return float(m.group(1)) if m else None


def by_layer(cells: list[dict], layer: int | None) -> list[dict]:
    if layer is None or not any("layer" in c for c in cells):
        return cells
    sel = [c for c in cells if c.get("layer") in (None, layer)]
    return sel or cells


def load_acc(fits: Path) -> list[dict]:
    """Accuracy rows (qtype, arm, cond, N, acc): eval_summary.csv if present, else margins.json cells."""
    p = REPO / "outputs/diag/eval_summary.csv"
    if p.exists():
        with open(p) as f:
            return [dict(r, N=int(r["N"]), acc=float(r["acc"])) for r in csv.DictReader(f)]
    m = load_json(fits / "margins.json")
    rows = []
    for c in (m or {}).get("cells", []):
        for n, a in zip(c["N"], c.get("acc", [])):
            rows.append(dict(qtype=c["qtype"], arm=c["arm"], cond=c.get("cond"), N=int(n), acc=a))
    return rows


# ----------------------------------------------------------------------------- F8 (drawable today)
def read_linear(path: Path) -> dict[tuple[str, int], float]:
    with open(path) as f:
        return {(r["qtype"], int(r["seq_len"])): float(r["acc"]) for r in csv.DictReader(f)}


def read_frozen(root: Path) -> tuple[dict[tuple[str, int], float], dict[int, Path]]:
    acc, src = {}, {}
    for n in NS:
        runs = sorted(p for p in (root / f"grid_seq{n}_test").glob("*/report.txt"))
        if not runs:
            continue
        src[n] = runs[-1]                                     # latest run dir that produced a report
        for line in runs[-1].read_text().splitlines():
            m = re.match(r"\s+([a-z_]+): (\d+)/(\d+) = ([\d.]+)", line)
            if m and not m.group(1).startswith("atype"):
                acc[(m.group(1), n)] = float(m.group(4))
    return acc, src


def heat(ax, M: np.ndarray, cmap, show_rows: bool) -> None:
    ax.grid(False)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=COL["train"], hatch="///",
                                       edgecolor="#c9cbcf", lw=0))
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=COL["muted"])
            else:
                lum = mcolors.rgb_to_hsv(cmap(v)[:3])[2]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                        color="white" if v > 0.62 or lum < 0.6 else COL["ink"])
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color=COL["surface"], lw=1.4)          # the 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(NIAH + DC if show_rows else [])
    ax.axhline(len(NIAH) - 0.5, color=COL["ink"], lw=1.4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("N (frames)")


def fig_f8(out: Path, **_) -> bool:
    if missing("F8", LINEAR_CSV, FROZEN_ROOT):
        return False
    lin = read_linear(LINEAR_CSV)
    frz, src = read_frozen(FROZEN_ROOT)
    if not frz:
        print("F8: inputs missing — no report.txt under", FROZEN_ROOT)
        return False
    rows_q = NIAH + DC
    L = np.array([[lin.get((q, n), np.nan) for n in NS] for q in rows_q])
    F = np.array([[frz.get((q, n), np.nan) for n in NS] for q in rows_q])
    base = matplotlib.colormaps["Blues"]                          # one hue, light -> dark (sequential)
    cmap = mcolors.LinearSegmentedColormap.from_list("blues_t", base(np.linspace(0.06, 1.0, 256)))

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 8.9), gridspec_kw=dict(wspace=0.08, left=0.26, right=0.90,
                                                                       top=0.885, bottom=0.10))
    heat(axs[0], L, cmap, True)
    heat(axs[1], F, cmap, False)
    axs[0].set_title("a  per-frame linear readout")
    axs[1].set_title("b  frozen model, exact match")
    for (y0, y1), lab in zip(((0, len(NIAH)), (len(NIAH), len(rows_q))), ("NIAH — needle (k = 1)", "DC — count (k grows)")):
        axs[0].text(-0.62, (y0 + y1 - 1) / 2, lab, transform=axs[0].get_yaxis_transform(), rotation=90,
                    ha="center", va="center", fontsize=8.5, fontweight="bold", color=COL["ink"])
    cax = fig.add_axes([0.915, 0.10, 0.016, 0.785])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label("accuracy (exact match)", fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    claim(fig, "Per-frame readout is flat, the answer collapses — across the whole benchmark",
          "24 official MMReD question types × N ∈ {8…128}: left, what a linear head reads off each frame; "
          "right, what the frozen model answers")
    footer(fig, "official MMReD test split, 50 rows per cell; frozen Qwen2.5-VL-7B, paper prompt (images then question); "
                "per-frame readout = linear probe on carrier states (legacy instrument, 2026-08).\n"
                "a: logistic heads fit on train seq 8/16 carrier states, composed per type by its symbolic reduction "
                "(armB_grid_linear_ALL.csv); '–' = no cell (two heads skipped in the seq-8 dump).  "
                "b: eval_frozen.py 2026-08-01 (frames resized to 392 px), latest run dir per N.")
    rows = [dict(panel=p, qtype=q, group="NIAH" if q in NIAH else "DC", N=n, acc=v, n=50,
                 source=str(rel(LINEAR_CSV)) if p == "linear_readout" else str(rel(src[n])))
            for p, M in (("linear_readout", L), ("frozen_exact_match", F))
            for q, row in zip(rows_q, M) for n, v in zip(NS, row) if not np.isnan(v)]
    save(fig, out, "F8_perqtype_heatmap", rows)
    about(out, "F8", "F8_perqtype_heatmap", "per-frame linear readout stays flat in N for most of the 24 types "
          "while frozen exact match collapses (NIAH softly, DC to ~0)",
          "outputs/mmred_hf/armB_grid_v2_perlen/armB_grid_linear_ALL.csv + outputs/mmred_hf/frozen/grid_seq*_test/<latest>/report.txt",
          "official MMReD (test split, 50/cell); the linear probe is a legacy instrument (2026-08), frozen grid at 392 px")
    return True


# ----------------------------------------------------------------------------- F5 (drawable today)
def fig_f5(out: Path, **_) -> bool:
    if missing("F5", F11_CSV):
        return False
    with open(F11_CSV) as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    cells = [(N, t) for N in (32, 128) for t in (1.0, 4.0)]
    pick = {}
    for N, t in cells:
        m = [r for r in rows if int(r["N"]) == N and r["tau"] == t]
        if m:
            pick[(N, t)] = m[0]
    if len(pick) < len(cells):
        print("F5: inputs missing — cells absent in F11 csv:", [c for c in cells if c not in pick])
        return False
    xs = [0, 1.3, 3.4, 4.7]
    stack = [("evidence_mass", "evidence frames (k = 2)", COL["fenced"]),
             ("competitor_mass", "competitor frames (N − k)", COL["frozen"]),
             ("other_mass", "prompt + sink", COL["sink"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), gridspec_kw=dict(left=0.09, right=0.70, top=0.80, bottom=0.25))
    out_rows = []
    for x, (N, t) in zip(xs, cells):
        r, bottom = pick[(N, t)], 0.0
        for key, lab, c in stack:
            v = r[key]
            ax.bar(x, v, bottom=bottom, width=0.62, color=c, edgecolor=COL["surface"], linewidth=1.6,
                   label=lab if x == xs[0] else None)
            if v >= 0.07:
                ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if key != "evidence_mass" or v > 0.1 else COL["ink"])
            bottom += v
        ax.text(x, bottom + 0.02, f"acc {r['s0_accuracy']:.2f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=COL["ink"])
        out_rows.append(dict(N=N, k=int(r["k"]), tau=t, layer=20, evidence_mass=r["evidence_mass"],
                             competitor_mass=r["competitor_mass"], prompt_sink_mass=r["other_mass"],
                             evid_per_frame=r["evid_per_frame"], nonevid_per_frame=r["nonevid_per_frame"],
                             conc_max_over_mean=r["conc"], s0_accuracy=r["s0_accuracy"]))
    ax.set_xticks(xs)                                   # the evidence segment is too thin to label inside:
    ax.set_xticklabels([f"τ = {t:g}\nevid. {pick[(N, t)]['evidence_mass']:.3f}" for N, t in cells])
    for xc, N in ((0.65, 32), (4.05, 128)):
        ax.text(xc, -0.24, f"N = {N}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=COL["ink"])
    ax.set_xlim(-0.65, 5.35)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("attention mass on the answer row, layer 20 (head median)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.02, 1.0), title="stack (top → bottom)",
              title_fontsize=8.5, alignment="left")
    claim(fig, "Where the mass goes when you sharpen",
          "raising τ frees competitor mass, but it flows to the prompt/sink — the evidence share falls and the count dies")
    footer(fig, "buckets are head-medians at layer 20 and do not sum to 1; legacy park data, P1b adapter, k = 2 evidence frames; "
                "τ applied on layers ≥ 12; acc = S0 exact match (outputs/sparse/s0_tau*_L12). Source: outputs/sparse/fig/F11_temperature_coupling.csv")
    save(fig, out, "F5_mass_when_sharpen", out_rows)
    about(out, "F5", "F5_mass_when_sharpen", "sharpening (τ 1→4) moves attention mass from competitor frames to the "
          "prompt/sink, not to the evidence; evidence mass and accuracy both fall at N = 32 and N = 128",
          "outputs/sparse/fig/F11_temperature_coupling.csv (S10b photograph + S0 τ sweep)", "legacy park data, P1b adapter, L20, k = 2")
    return True


# ----------------------------------------------------------------------------- F1 / F4 (alpha.json)
def fig_f1(out: Path, fits: Path, layer=20, qtype=None, **_) -> bool:
    p = fits / "alpha.json"
    if missing("F1", p):
        return False
    cells = by_layer([c for c in load_json(p)["cells"] if qtype in (None, c["qtype"])], layer)
    groups: dict[str, dict[str, dict]] = {}
    for c in cells:
        r = rung_of(c["arm"], c.get("locus"))
        g = groups.setdefault(c["qtype"], {})
        if r and (r not in g or (c.get("cond") or "base") == "base"):
            g[r] = c
    if not groups:
        print("F1: inputs missing — no cells in", p)
        return False
    qs = list(groups)
    fig, ax = plt.subplots(figsize=(max(5.2, 1.6 * len(qs) + 3.2), 4.4),
                           gridspec_kw=dict(left=0.12, right=0.98, top=0.80, bottom=0.16))
    w, rows = 0.19, []
    for i, q in enumerate(qs):
        for j, (r, lab) in enumerate(RUNGS):
            c = groups[q].get(r)
            if not c:
                continue
            x = i + (j - 1.5) * w
            lo, hi = c.get("ci") or (c["alpha"], c["alpha"])
            ax.bar(x, c["alpha"], width=w * 0.92, color=COL[r], label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, c["alpha"], yerr=[[c["alpha"] - lo], [hi - c["alpha"]]], fmt="none", ecolor=COL["ink"],
                        elinewidth=1, capsize=3, zorder=3)
            ax.text(x, hi + 0.02, f"{c['alpha']:.2f}", ha="center", va="bottom", fontsize=8, color=COL["ink"])
            rows.append(dict(qtype=q, rung=lab, arm=c["arm"], locus=c.get("locus"), cond=c.get("cond"),
                             layer=c.get("layer"), alpha=c["alpha"], ci_lo=lo, ci_hi=hi, r2=c.get("r2")))
    ax.axhline(0, color=COL["muted"], lw=0.8)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels(qs)
    ax.set_ylabel("α  (single-frame influence ∝ N^−α)")
    ax.set_ylim(top=max(r["ci_hi"] for r in rows) * 1.4)          # headroom for the one legend row
    ax.legend(ncol=4, loc="upper center")
    claim(fig, "One frame's influence on the answer: the read dilutes, the fact does not, the gate flattens",
          f"exponent of the flip response vs N with 95% CI; layer {layer}")
    footer(fig, f"source: {rel(p)} (D1 flip-influence ladder); bars = median-based power-law fit, whiskers = bootstrap CI")
    save(fig, out, "F1_alpha_ladder", rows)
    about(out, "F1", "F1_alpha_ladder", "the frozen/trained read decays as N^−α with α ≈ 0.7, the fenced fact and the gated read are flat",
          str(rel(p)), "official MMReD (DIAG D1/D1b)")
    return True


def fig_f4(out: Path, fits: Path, layer=20, qtype=None, **_) -> bool:
    p = fits / "alpha.json"
    if missing("F4", p):
        return False
    cells = by_layer(load_json(p)["cells"], layer)
    qtype = qtype or (cells[0]["qtype"] if cells else None)
    lines: dict[str, dict] = {}
    for c in cells:
        r = rung_of(c["arm"], c.get("locus"))
        if c["qtype"] == qtype and r in ("frozen", "fenced", "gated") and (r not in lines or (c.get("cond") or "base") == "base"):
            lines[r] = c
    if not lines:
        print("F4: inputs missing — no frozen/fenced/gated cells for", qtype)
        return False
    pos = (lines.get("frozen") or {}).get("by_position") or {}   # D6: α by flip position -> side panel
    fig, axs = plt.subplots(1, 2 if pos else 1, figsize=(7.2 if pos else 6.4, 4.4), squeeze=False,
                            gridspec_kw=dict(left=0.11, right=0.98, top=0.82, bottom=0.14, wspace=0.3,
                                             width_ratios=[3.2, 1] if pos else [1]))
    ax = axs[0, 0]
    xlog2(ax)
    ax.set_yscale("log")
    floors = [f.get("median", np.nan) for c in lines.values() for k, f in (c.get("floors") or {}).items() if k != "replay"]
    rows = []
    for r, lab in RUNGS:
        c = lines.get(r)
        if not c:
            continue
        N, med = np.array(c["N"], float), np.array(c["median"], float)
        if c.get("iqr_lo") and c.get("iqr_hi"):
            ax.fill_between(N, c["iqr_lo"], c["iqr_hi"], color=COL[r], alpha=0.12, lw=0)
        ax.plot(N, med, marker=MARK[r], color=COL[r], label=lab)
        exponent(ax, N[-1], med[-1], c["alpha"], COL[r], ci=c.get("ci"))
        rows += [dict(qtype=qtype, rung=lab, arm=c["arm"], N=int(n), median=m, iqr_lo=lo, iqr_hi=hi, alpha=c["alpha"])
                 for n, m, lo, hi in zip(N, med, c.get("iqr_lo") or [None] * len(N), c.get("iqr_hi") or [None] * len(N))]
    if floors and np.isfinite(np.nanmax(floors)):
        floor_band(ax, float(np.nanmax(floors)), lo=ax.get_ylim()[0])
    shade_train(ax)
    ax.set_ylabel("median ‖Δh‖ at the answer row (log)")
    ax.legend(loc="lower left")
    if pos:
        side, vals = axs[0, 1], [v.get("alpha", v) if isinstance(v, dict) else v for v in pos.values()]
        side.bar(range(len(pos)), vals, color=COL["frozen"], width=0.6)
        for i, v in enumerate(vals):
            side.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        side.set_xticks(range(len(pos)))
        side.set_xticklabels(list(pos), fontsize=7.5)
        side.set_title("α by flip position", fontsize=8, loc="center")
        side.set_xlabel("flipped frame")
        rows += [dict(qtype=qtype, rung="frozen read (by position)", position=k, alpha=v) for k, v in zip(pos, vals)]
    claim(fig, "Single-frame influence vs N, three lines", f"{qtype}, layer {layer}: frozen joint read · fenced per-frame fact · gated read")
    footer(fig, f"source: {rel(p)}; band = IQR over flip pairs; grey band = max non-replay floor (ctrl / perm); exponents from the power-law fit")
    save(fig, out, "F4_influence_vs_N", rows)
    about(out, "F4", "F4_influence_vs_N", "the frozen read's flip response decays to the floor by N = 128 while the fenced fact and gated read stay flat",
          str(rel(p)), "official MMReD (DIAG D1/D6)")
    return True


# ----------------------------------------------------------------------------- F2 / F3 (sharelaw.json)
def share(N, k, s, C):
    return k * math.exp(s) / (k * math.exp(s) + (N - k) + C)


def fig_f2(out: Path, fits: Path, layer=20, qtype=None, **_) -> bool:
    p = fits / "sharelaw.json"
    if missing("F2", p):
        return False
    fs = by_layer([f for f in load_json(p)["fits"] if qtype in (None, f["qtype"]) and cond_tau(f.get("cond")) in (None, 1.0)], layer)
    main = next((f for f in fs if rung_of(f["arm"]) != "gated"), None)
    gated = next((f for f in fs if rung_of(f["arm"]) == "gated"), None)
    if not main:
        print("F2: inputs missing — no ungated fit in", p)
        return False
    r = rung_of(main["arm"])
    ks = sorted({c["k"] for c in main["cells"]})[:3]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), gridspec_kw=dict(left=0.12, right=0.98, top=0.82, bottom=0.14))
    xlog2(ax)
    grid = np.exp2(np.linspace(math.log2(NS[0]), math.log2(NS[-1]), 60))
    rows = []
    for i, k in enumerate(ks):
        col = shade(COL[r], 0.55 * (1 - i / max(1, len(ks) - 1)))
        cells = sorted((c for c in main["cells"] if c["k"] == k), key=lambda c: c["N"])
        ax.plot([c["N"] for c in cells], [c["evid"] for c in cells], ls="none", marker=MARK[r], color=col, label=f"k = {k}")
        ax.plot(grid, [share(n, k, main["s"], main["C"]) for n in grid], color=col, lw=1.4)
        rows += [dict(arm=main["arm"], k=k, N=c["N"], evid=c["evid"], nonevid=c.get("nonevid"), n=c.get("n"),
                      fit=share(c["N"], k, main["s"], main["C"])) for c in cells]
    if gated:
        cells = sorted(gated["cells"], key=lambda c: c["N"])
        ax.plot([c["N"] for c in cells], [c["evid"] for c in cells], marker=MARK["gated"], color=COL["gated"], label="gated (N removed)")
        rows += [dict(arm=gated["arm"], k=c["k"], N=c["N"], evid=c["evid"], nonevid=c.get("nonevid"), n=c.get("n"), fit=None) for c in cells]
    shade_train(ax)
    ax.set_ylabel("attention mass on evidence frames, layer %d" % layer)
    ax.set_ylim(0, None)
    ax.text(0.98, 0.97, f"m = k·eˢ / (k·eˢ + (N−k) + C)\ns = {main['s']:.2f}, C = {main['C']:.1f}, R² = {main['r2']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=COL["ink"])
    ax.legend(loc="center right")
    claim(fig, "Attention obeys the share law; the gate removes N from it", f"{main['qtype']}, {main['arm']}: points = measured, curves = the fitted law")
    footer(fig, f"source: {rel(p)} (D2 photograph + NLS fit); head-median block mass; gated overlay = oracle gate")
    save(fig, out, "F2_share_law", rows)
    about(out, "F2", "F2_share_law", "evidence mass follows m = k·eˢ/(k·eˢ + (N−k) + C) with a small s; the gated read is N-invariant",
          str(rel(p)), "official MMReD (DIAG D2/D2b)")
    return True


def fig_f3(out: Path, fits: Path, layer=20, **_) -> bool:
    p = fits / "sharelaw.json"
    if missing("F3", p):
        return False
    fs = [f for f in by_layer(load_json(p)["fits"], layer) if cond_tau(f.get("cond")) is not None]
    acc = load_acc(fits)
    if not fs:
        print("F3: inputs missing — no τ conditions in", p)
        return False
    cls = lambda q: "needle (k = 1)" if q in NIAH else "count (k ≥ 2)"
    colr = {"needle (k = 1)": COL["fenced"], "count (k ≥ 2)": COL["trained"]}
    lsty = {32: "-", 128: "--"}
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2), gridspec_kw=dict(left=0.08, right=0.98, top=0.80, bottom=0.15, wspace=0.28))
    rows, seen = [], set()
    for f in fs:
        for c in f["cells"]:
            if c["N"] in lsty:
                rows.append(dict(panel="evidence_mass", qtype=f["qtype"], cls=cls(f["qtype"]), N=c["N"], tau=cond_tau(f["cond"]), y=c["evid"]))
    for a in acc:
        t = cond_tau(a.get("cond"))
        if t is not None and a["N"] in lsty:
            rows.append(dict(panel="accuracy", qtype=a["qtype"], cls=cls(a["qtype"]), N=a["N"], tau=t, y=a["acc"]))
    for ax, panel, ylab in ((axs[0], "evidence_mass", "attention mass on evidence, layer %d" % layer), (axs[1], "accuracy", "exact match")):
        for key in sorted({(r["cls"], r["N"]) for r in rows if r["panel"] == panel}):
            pts = sorted((r["tau"], r["y"]) for r in rows if r["panel"] == panel and (r["cls"], r["N"]) == key)
            lab = f"{key[0]}, N = {key[1]}"
            ax.plot(*zip(*pts), color=colr[key[0]], ls=lsty[key[1]], marker="o" if key[1] == 32 else "s",
                    label=lab if lab not in seen else None)
            seen.add(lab)
        ax.set_xlabel("τ (attention temperature, layers ≥ 12)")
        ax.set_ylabel(ylab)
    axs[0].set_title("a  mass on the evidence")
    axs[1].set_title("b  accuracy")
    axs[0].legend(loc="best")
    claim(fig, "One temperature rescues a needle and breaks a count", "frozen model, question-first prompt; solid N = 32, dashed N = 128")
    footer(fig, f"source: {rel(p)} + accuracy rows (eval_summary.csv or margins.json); DIAG D3")
    save(fig, out, "F3_tau_dissociation", rows)
    about(out, "F3", "F3_tau_dissociation", "sharpening raises needle mass and accuracy but lowers evidence mass and accuracy for counts",
          str(rel(p)) + " + accuracy rows", "official MMReD (DIAG D3)")
    return True


# ----------------------------------------------------------------------------- F7 (headscan.json) / F10 (margins.json)
def fig_f7(out: Path, fits: Path, qtype=None, **_) -> bool:
    p = fits / "headscan.json"
    if missing("F7", p):
        return False
    cells = [c for c in load_json(p)["cells"] if qtype in (None, c["qtype"])]
    acc = load_acc(fits)
    if not cells:
        print("F7: inputs missing — no cells in", p)
        return False
    qtype = qtype or cells[0]["qtype"]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2), gridspec_kw=dict(left=0.08, right=0.98, top=0.80, bottom=0.15, wspace=0.28))
    rows, head = [], cells[0].get("named_head", "named head")
    for c in cells:
        if c["qtype"] != qtype:
            continue
        r = rung_of(c["arm"])
        axs[0].plot(c["N"], c["named_auc"], marker=MARK[r], color=COL[r], label=dict(RUNGS)[r])
        rows += [dict(panel="auc", qtype=qtype, arm=c["arm"], N=n, y=v, head=head) for n, v in zip(c["N"], c["named_auc"])]
        pts = sorted((a["N"], a["acc"]) for a in acc if a["qtype"] == qtype and a["arm"] == c["arm"] and cond_tau(a.get("cond")) in (None, 1.0))
        if pts:
            axs[1].plot(*zip(*pts), marker=MARK[r], color=COL[r])
            rows += [dict(panel="acc", qtype=qtype, arm=c["arm"], N=n, y=v, head=None) for n, v in pts]
    for ax in axs:
        xlog2(ax)
        shade_train(ax)
        ax.set_ylim(0, 1.05)
    axs[0].axhline(0.5, color=COL["muted"], lw=0.8)
    axs[0].set_ylabel(f"evidence-vs-rest AUC of {head} mass")
    axs[1].set_ylabel("exact match")
    axs[0].set_title("a  the selector head ranks the evidence")
    axs[1].set_title("b  the answer still fails")
    axs[0].legend(loc="lower left")
    claim(fig, "The model ranks the evidence and still cannot count it", f"{qtype}: same x-axis, same runs")
    footer(fig, f"source: {rel(p)} (D2/D8 head scan) + accuracy rows")
    save(fig, out, "F7_rank_vs_count", rows)
    about(out, "F7", "F7_rank_vs_count", "a trained-in head separates evidence frames at AUC ≈ 1 at every N while accuracy collapses",
          str(rel(p)) + " + accuracy rows", "official MMReD (DIAG D2/D8)")
    return True


def fig_f10(out: Path, fits: Path, qtype=None, **_) -> bool:
    p = fits / "margins.json"
    if missing("F10", p):
        return False
    cells = [c for c in load_json(p)["cells"] if qtype in (None, c["qtype"]) and cond_tau(c.get("cond")) in (None, 1.0)]
    if not cells:
        print("F10: inputs missing — no cells in", p)
        return False
    qtype = qtype or cells[0]["qtype"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), gridspec_kw=dict(left=0.12, right=0.98, top=0.82, bottom=0.14))
    xlog2(ax)
    rows, seen = [], set()
    for c in cells:
        r = rung_of(c["arm"])
        if c["qtype"] != qtype or r in seen:
            continue
        seen.add(r)
        N, m = c["N"], c["margin_median"]
        if c.get("margin_iqr_lo") and c.get("margin_iqr_hi"):
            ax.fill_between(N, c["margin_iqr_lo"], c["margin_iqr_hi"], color=COL[r], alpha=0.12, lw=0)
        ax.plot(N, m, marker=MARK[r], color=COL[r], label=dict(RUNGS)[r])
        rows += [dict(qtype=qtype, rung=dict(RUNGS)[r], arm=c["arm"], N=n, margin_median=v, acc=a)
                 for n, v, a in zip(N, m, c.get("acc") or [None] * len(N))]
    ax.axhline(0, color=COL["ink"], lw=1)
    ax.text(ax.get_xlim()[1] / 1.02, 0, "decision boundary", ha="right", va="bottom", fontsize=7.5, color=COL["muted"])
    shade_train(ax)
    ax.set_ylabel("logit margin, gold − best rival (median)")
    ax.legend(loc="best")
    claim(fig, "The decision margin crosses zero where accuracy dies", f"{qtype}: frozen · trained · gated; band = IQR")
    footer(fig, f"source: {rel(p)} (D1 margins); accuracy per point in the csv")
    save(fig, out, "F10_margin_vs_N", rows)
    about(out, "F10", "F10_margin_vs_N", "the frozen/trained read's margin slides through zero with N; the gated read's margin is constant",
          str(rel(p)), "official MMReD (DIAG D1)")
    return True


# ----------------------------------------------------------------------------- CLI
FIGS = {"f1": fig_f1, "f2": fig_f2, "f3": fig_f3, "f4": fig_f4, "f5": fig_f5, "f7": fig_f7, "f8": fig_f8, "f10": fig_f10}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fig", choices=list(FIGS) + ["all-available"])
    ap.add_argument("--fits", type=Path, default=FITS_DEFAULT, help="dir with alpha/sharelaw/margins/headscan.json")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--layer", type=int, default=20, help="layer to draw for the JSON-driven figures")
    ap.add_argument("--qtype", default=None, help="restrict the JSON-driven figures to one question type")
    a = ap.parse_args(argv)
    style()
    todo = list(FIGS) if a.fig == "all-available" else [a.fig]
    done = {f: FIGS[f](out=a.out, fits=a.fits, layer=a.layer, qtype=a.qtype) for f in todo}
    print("drawn:", [f for f, ok in done.items() if ok] or "none", "| skipped:", [f for f, ok in done.items() if not ok] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
