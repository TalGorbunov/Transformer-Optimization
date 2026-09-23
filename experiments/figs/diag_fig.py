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
Inputs written by the DIAG fitters (experiments/diag_fit.py) under <fits>/:
  alpha.json      {"cells":[{qtype, arm, cond, model, layer, locus ∈ final|slot_t, N[], median[], iqr_lo[], iqr_hi[], n[],
                   alpha, r2, ci, floors{ctrl|perm|replay:{median,max,per_N}}, by_gold, by_position}]}        -> F1, F4
  sharelaw.json   {"fits":[{qtype, arm, cond, layer, cells[{N,k,evid,nonevid,other,n[,edge,s_frame,frame_share]}],
                   frame_law{N[],edge[],s_frame[],sink[],frame_share[],s_frame_mean,sink_mean}, ...}]}         -> F2
  sharelaw_d3.json  same shape, conds base | tauT[L12] | lognS (the D3 τ / log-N sweep, question-first)         -> F3
  headscan.json   {"groups":[{qtype, arm, cond, N[], auc{head:{N:auc}}, best_per_layer, population{N:{count,..}}, named}]} -> F7
  margins.json    {"cells":[{qtype, arm, cond, N[], margin_median[], iqr_lo[], iqr_hi[], accuracy[], flip_changes_answer[],
                   noise_floor[], n[]}]}                                                                        -> F10
  outputs/diag/eval_summary.csv   qtype,arm,cond,N,acc,n,ci_lo,ci_hi,run  (cond strings may lack the photo conds'
                   'L12' suffix: both sides are normalised by stripping 'L\\d+')                               -> F3, F7
frame_law is written by the CPU fit pass; when it is absent the figures fall back to the cells' mean masses and say so.
A missing input prints "inputs missing" and the figure is skipped — nothing is invented.

Arms (frozen model, official rows): plain = the DEPLOYED layout (images then question) = "deployed read";
qfirst = question-first read; fenced + locus slot_t = the per-frame FACT ("fenced fact"); fenced + locus final =
the frozen fenced read (sits at the floor; never a main series); gated + final = "gated read".

Every figure -> <out>/<name>.png (200 dpi) + .pdf + .csv (every plotted number) + one ABOUT line in <out>/README.md.
Design rules (Tal): the title is the claim; <= 4 series; one legend; N on a log2 x-axis (8..128); colors fixed
across the set; bf16 floor = light grey band; train window (N <= 16) shaded; exponents printed on the line;
official-vs-legacy regime named in the footer.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "outputs/diag/fig"
FITS_DEFAULT = REPO / "outputs/diag/fits"
LINEAR_CSV = REPO / "outputs/mmred_hf/armB_grid_v2_perlen/armB_grid_linear_ALL.csv"
FROZEN_ROOT = REPO / "outputs/mmred_hf/frozen"
F11_CSV = REPO / "outputs/sparse/fig/F11_temperature_coupling.csv"
EVAL_CSV = REPO / "outputs/diag/eval_summary.csv"

NS = [8, 16, 32, 64, 128]
TRAIN_MAX = 16
COL = dict(frozen="#8a8f96", plain="#8a8f96", qfirst="#5c6570", fenced="#1f9a6b", trained="#e0602a", gated="#2f6fd6",
           sink="#b48a2e", ink="#1c1e21", muted="#6b6f75", grid="#e4e6e9", floor="#e9eaec", train="#f3f3f0", surface="white")
MARK = dict(frozen="o", plain="o", qfirst="v", fenced="s", trained="^", gated="D")
# the four series of the frozen-model ladder: (arm, locus in alpha.json, legend label)
SERIES = [("plain", "final", "deployed read"), ("qfirst", "final", "question-first read"),
          ("fenced", "slot_t", "fenced fact"), ("gated", "final", "gated read")]
LABEL = {arm: lab for arm, _, lab in SERIES}
# where a series is a question type rather than an arm (F3, F7b, F10 panels): its own hue + marker, never the arm palette
QCOL = dict(char_at_frame="#7b4fb3", steps_in_room="#c2452d", where_spend="#2a7f9e")
QMARK = dict(char_at_frame="o", steps_in_room="s", where_spend="^")
QLAB = dict(char_at_frame="char_at_frame (needle, k = 1)", steps_in_room="steps_in_room (count)",
            where_spend="where_spend (count control)")
NIAH = ["first_app", "final_app", "char_on_char_first_app", "char_on_char_final_app", "char_at_frame",
        "first_at_room", "last_at_room", "room_on_char_first_app", "room_on_char_final_app", "room_at_frame",
        "char_on_char_at_frame", "n_room_on_char_first_app", "n_room_on_char_final_app", "n_char_at_frame", "n_empty"]
DC = ["room_empty", "where_spend", "crowded_room", "who_spend", "spend_alone", "spend_together", "steps_in_room",
      "rooms_visited", "crowd_count"]
OFFICIAL = "official MMReD test rows (HF ef1e43ce/mmred, native 512 px, paper prompt), frozen Qwen2.5-VL-7B nf4"


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


def exponent(ax, x, y, val, color, name="α", ci=None, fmt="{:.2f}"):
    """Print the fitted exponent on the line, just above its right end (stays inside the axes)."""
    s = f"{name} = {fmt.format(val)}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
    return ax.annotate(s, (x, y), xytext=(0, 7), textcoords="offset points", ha="right", va="bottom", fontsize=8,
                       color=COL["ink"], bbox=dict(boxstyle="round,pad=0.15", fc=COL["surface"], ec=color, lw=0.8))


def deoverlap(ax, anns) -> None:
    """Push exponent boxes that land on top of each other upward (in points) so every label stays legible."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = -np.inf
    for a in sorted(anns, key=lambda a: a.get_window_extent(r).y0):
        bb = a.get_window_extent(r)
        if bb.y0 < top + 3:
            ox, oy = a.xyann
            a.xyann = (ox, oy + (top + 3 - bb.y0) * 72 / fig.dpi)
            bb = a.get_window_extent(r)
        top = bb.y1


def shade(hexcol: str, f: float) -> tuple:
    """Blend a series color toward the surface (f = 0 -> color, 1 -> white) for an ordinal ramp."""
    r, g, b = mcolors.to_rgb(hexcol)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


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
    return str(p.resolve().relative_to(REPO)) if REPO in p.resolve().parents else str(p)


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


def cond_norm(cond: str | None) -> str:
    """'tau1.5L12' -> 'tau1.5', ''/None -> 'base': eval conds lack the layer suffix the photograph conds carry."""
    return re.sub(r"L\d+$", "", (cond or "base").strip()) or "base"


def cond_x(cond: str | None) -> tuple[str, float] | None:
    """('tau', 1.0) for base, ('tau', T) for tauT, ('logn', S) for lognS; None for anything else."""
    c = cond_norm(cond)
    if c == "base":
        return ("tau", 1.0)
    m = re.fullmatch(r"tau[_=]?([\d.]+)", c)
    if m:
        return ("tau", float(m.group(1)))
    m = re.fullmatch(r"logn[_=]?(\d+)", c)
    if m:
        return ("logn", float(m.group(1)))
    return None


def cond_tau(cond: str | None) -> float | None:
    x = cond_x(cond)
    return x[1] if x and x[0] == "tau" else None


def by_layer(cells: list[dict], layer: int | None) -> list[dict]:
    if layer is None or not any("layer" in c for c in cells):
        return cells
    sel = [c for c in cells if c.get("layer") in (None, layer)]
    return sel or cells


def pick(cells: list[dict], qtype: str, arm: str, locus: str | None = None, layer: int | None = None,
         cond: str = "base") -> dict | None:
    """The one fit cell for (qtype, arm, locus, layer) at the base condition; None if absent."""
    for c in cells:
        if c.get("qtype") == qtype and c.get("arm") == arm and cond_norm(c.get("cond")) == cond \
                and (locus is None or c.get("locus") == locus) and (layer is None or c.get("layer") in (None, layer)):
            return c
    return None


def n_range(ns) -> str:
    ns = [int(n) for n in ns]
    return f"N = {min(ns)}…{max(ns)}" if len(ns) > 1 else f"N = {ns[0]}"


def load_acc(fits: Path) -> list[dict]:
    """Accuracy rows (qtype, arm, cond, N, acc, run): eval_summary.csv if present, else margins.json cells."""
    if EVAL_CSV.exists():
        with open(EVAL_CSV) as f:
            return [dict(r, cond=cond_norm(r.get("cond")), N=int(r["N"]), acc=float(r["acc"]), run=r.get("run") or "")
                    for r in csv.DictReader(f)]
    m = load_json(fits / "margins.json")
    rows = []
    for c in (m or {}).get("cells", []):
        for n, a in zip(c["N"], c.get("accuracy") or c.get("acc") or []):
            rows.append(dict(qtype=c["qtype"], arm=c["arm"], cond=cond_norm(c.get("cond")), N=int(n), acc=a, run=""))
    return rows


def frame_law_of(fit: dict) -> tuple[dict, bool]:
    """fit['frame_law'] when the fitter wrote it (True), else the same per-N summary computed from the cells' mean
    masses (False): edge = (evid/k)/(nonevid/(N−k)), sink = other, k-cells pooled per N weighted by n."""
    fl = fit.get("frame_law") or {}
    if fl.get("N"):
        return fl, True
    by_n: dict[int, list] = {}
    for c in fit.get("cells", []):
        n, k, e, ne = int(c["N"]), int(c["k"]), c.get("evid"), c.get("nonevid")
        if 0 < k < n and e is not None and ne is not None and np.isfinite(e) and np.isfinite(ne) and ne > 0:
            by_n.setdefault(n, []).append(((e / k) / (ne / (n - k)), c.get("other", np.nan), e / (e + ne), c.get("n", 1)))
    out = {"N": sorted(by_n), "edge": [], "s_frame": [], "sink": [], "frame_share": []}
    for n in out["N"]:
        w = np.array([t[3] for t in by_n[n]], float)
        edge = float(np.average([t[0] for t in by_n[n]], weights=w))
        out["edge"].append(edge)
        out["s_frame"].append(math.log(edge) if edge > 0 else np.nan)
        out["sink"].append(float(np.average([t[1] for t in by_n[n]], weights=w)))
        out["frame_share"].append(float(np.average([t[2] for t in by_n[n]], weights=w)))
    if out["N"]:
        out["s_frame_mean"] = float(np.nanmean(out["s_frame"]))
        out["sink_mean"] = float(np.nanmean(out["sink"]))
    return out, False


def mass_at(fit: dict, N: int) -> tuple[float, float, float]:
    """(total evidence mass, non-evidence mass, prompt+sink share) at N, k-cells pooled weighted by n."""
    cs = [c for c in fit.get("cells", []) if int(c["N"]) == N]
    if not cs:
        return np.nan, np.nan, np.nan
    w = np.array([c.get("n", 1) for c in cs], float)
    f = lambda key: float(np.average([c.get(key, np.nan) for c in cs], weights=w))
    return f("evid"), f("nonevid"), f("other")


# ----------------------------------------------------------------------------- F8 (drawable today)
def read_linear(path: Path) -> dict[tuple[str, int], float]:
    with open(path) as f:
        return {(r["qtype"], int(r["seq_len"])): float(r["acc"]) for r in csv.DictReader(f)}


def faithful_report(n: int) -> list[Path]:
    """The DIAG faithful frozen grid (paper prompt, native 512 px, upstream parser, 2026-09-22): N=8 lives under
    outputs/port (the C1 faithful gate), N>=16 under outputs/diag/eval/grid."""
    if n == 8:
        return sorted((REPO / "outputs/port/evaluate/seq_len_8_test").glob("*/*_faithful/report.txt"))
    return sorted((REPO / f"outputs/diag/eval/grid/N{n}").glob("*_faithful/report.txt"))


def read_frozen(root: Path) -> tuple[dict[tuple[str, int], float], dict[int, Path]]:
    acc, src = {}, {}
    for n in NS:
        runs = faithful_report(n) or sorted(p for p in (root / f"grid_seq{n}_test").glob("*/report.txt"))
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
                + ("b: evaluate.py faithful runs 2026-09-22 (paper prompt, native 512 px, upstream parser; C1 gate + eval/grid)."
                   if all("faithful" in str(v) for v in src.values()) else
                   "b: eval_frozen.py 2026-08-01 (frames resized to 392 px), latest run dir per N."))
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
def default_qtype(items: list[dict], qtype: str | None) -> str | None:
    if qtype:
        return qtype
    qs = [c["qtype"] for c in items]
    return "char_at_frame" if "char_at_frame" in qs else (qs[0] if qs else None)


COUNT_FIXED = {"steps_in_room", "crowd_count"}   # count qtypes: F1 uses the fixed-count 0->1 stratum


def fig_f1(out: Path, fits: Path, layer=20, qtype=None, **_) -> bool:
    p = fits / "alpha.json"
    if missing("F1", p):
        return False
    cells = [c for c in load_json(p)["cells"] if qtype in (None, c["qtype"])]
    order = NIAH + DC
    qs = sorted({c["qtype"] for c in cells}, key=lambda q: order.index(q) if q in order else 99)
    picked = {(q, arm, locus): pick(cells, q, arm, locus, layer) for q in qs for arm, locus, _ in SERIES}
    if not any(picked.values()):
        print(f"F1: inputs missing — no base cells at layer {layer} in", p)
        return False
    fig, ax = plt.subplots(figsize=(max(5.6, 1.9 * len(qs) + 3.0), 4.6),
                           gridspec_kw=dict(left=0.10, right=0.98, top=0.80, bottom=0.17))
    w, rows, ranges, seen = 0.19, [], {}, set()
    for i, q in enumerate(qs):
        for j, (arm, locus, lab) in enumerate(SERIES):
            c = picked[(q, arm, locus)]
            if not c:
                continue
            fixed = False
            if q in COUNT_FIXED and locus == "final":
                # count types: the pooled slope mixes k (which grows with N) with N; use the fixed-count
                # 0 -> 1 flips when the fitter produced that stratum (by_gold["0"]: >=5 pairs per N, >=3 N)
                bg = (c.get("by_gold") or {}).get("0")
                if bg and bg.get("alpha") is not None and bg.get("ci"):
                    c = {**c, "alpha": bg["alpha"], "ci": bg["ci"], "N": bg.get("N", c["N"]), "r2": bg.get("r2"), "n": bg.get("n", c.get("n"))}
                    fixed = True
            x = i + (j - 1.5) * w
            lo, hi = c.get("ci") or (c["alpha"], c["alpha"])
            ax.bar(x, c["alpha"], width=w * 0.92, color=COL[arm], label=lab if lab not in seen else None, zorder=2)
            seen.add(lab)
            ax.errorbar(x, c["alpha"], yerr=[[c["alpha"] - lo], [hi - c["alpha"]]], fmt="none", ecolor=COL["ink"],
                        elinewidth=1, capsize=3, zorder=3)
            ax.text(x, max(hi, c["alpha"]) + 0.02, f"{c['alpha']:.2f}", ha="center", va="bottom", fontsize=7.8, color=COL["ink"])
            ranges.setdefault(n_range(c["N"]), []).append(f"{q}/{lab}")
            rows.append(dict(qtype=q, series=lab, arm=arm, locus=locus, cond=cond_norm(c.get("cond")), layer=c.get("layer"),
                             alpha=c["alpha"], ci_lo=lo, ci_hi=hi, r2=c.get("r2"), N_min=min(c["N"]), N_max=max(c["N"]),
                             n_N=len(c["N"]), n_rows_min=min(c.get("n") or [0]), fixed_count_0to1=fixed))
    ax.axhline(0, color=COL["muted"], lw=0.8)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([q + ("\n(fixed count 0→1)" if any(r["qtype"] == q and r["fixed_count_0to1"] for r in rows) else "") for q in qs])
    ax.set_ylabel("α  (single-frame influence ∝ N^−α)")
    ymax, ymin = max(r["ci_hi"] for r in rows), min(min(r["ci_lo"] for r in rows), 0.0)
    ax.set_ylim(ymin - 0.06, ymax * 1.35 + 0.05)          # headroom for the one legend row
    ax.legend(ncol=4, loc="upper center")
    claim(fig, "One frame's influence on the answer: the read dilutes, the fact does not, the gate flattens",
          f"exponent of the flip response vs N with 95% CI; layer {layer}; deployed read = images then question (the paper's layout)")
    nr = " / ".join(sorted(ranges)) if len(ranges) > 1 else next(iter(ranges))
    footer(fig, f"official test rows, frozen model; N ranges per cell: {nr} (count bars: fixed-count 0→1 stratum).  Source: {rel(p)} "
                "(D1); bars = power-law fit on medians, whiskers = bootstrap 95% CI; the frozen fenced answer row (at the floor) is not a bar.")
    save(fig, out, "F1_alpha_ladder", rows)
    about(out, "F1", "F1_alpha_ladder", "the deployed and question-first reads decay as N^−α (α ≈ 0.5–0.9); the fenced "
          "fact is flat; the gated read is flat for the needle types and still decays for steps_in_room",
          str(rel(p)), "official MMReD (DIAG D1/D1b), frozen model")
    return True


def fig_f4(out: Path, fits: Path, layer=20, qtype=None, **_) -> bool:
    p = fits / "alpha.json"
    if missing("F4", p):
        return False
    cells = load_json(p)["cells"]
    qtype = default_qtype(cells, qtype)
    lines = {(arm, locus): pick(cells, qtype, arm, locus, layer) for arm, locus, _ in SERIES}
    lines = {k: v for k, v in lines.items() if v}
    if not lines:
        print(f"F4: inputs missing — no base cells for {qtype} at layer {layer} in", p)
        return False
    dep = lines.get(("plain", "final")) or {}
    slot = lines.get(("fenced", "slot_t")) or {}
    fen_read = pick(cells, qtype, "fenced", "final", layer) or {}
    perm_src = slot if ((((slot.get("floors") or {}).get("perm") or {}).get("median")) or 0) > 0 else fen_read   # slot_t perm floor is often 0
    perm = ((perm_src.get("floors") or {}).get("perm") or {}).get("median")
    perm_max = ((perm_src.get("floors") or {}).get("perm") or {}).get("max")
    perm_locus = perm_src.get("locus")
    ctrl = ((dep.get("floors") or {}).get("ctrl") or {}).get("median")
    pos = dep.get("by_position") or {}                              # D6: α by flip position -> side panel (deployed arm)
    fig, axs = plt.subplots(1, 2 if pos else 1, figsize=(8.2 if pos else 6.4, 4.7), squeeze=False,
                            gridspec_kw=dict(left=0.09, right=0.985, top=0.80, bottom=0.20, wspace=0.28,
                                             width_ratios=[3.2, 1] if pos else [1]))
    ax = axs[0, 0]
    ns_all = sorted({int(n) for c in lines.values() for n in c["N"]})
    xlog2(ax, ns_all)
    ax.set_yscale("log")
    rows, anns, ys = [], [], []
    for arm, locus, lab in SERIES:
        c = lines.get((arm, locus))
        if not c:
            continue
        N, med = np.array(c["N"], float), np.array(c["median"], float)
        lo, hi = c.get("iqr_lo"), c.get("iqr_hi")
        if lo and hi:
            ax.fill_between(N, lo, hi, color=COL[arm], alpha=0.12, lw=0)
            ys += [v for v in lo + hi if v and v > 0]
        ys += [v for v in med if v > 0]
        ax.plot(N, med, marker=MARK[arm], color=COL[arm], label=lab)
        anns.append(exponent(ax, N[-1], med[-1], c["alpha"], COL[arm], ci=c.get("ci")))
        rows += [dict(qtype=qtype, series=lab, arm=arm, locus=locus, layer=c.get("layer"), N=int(n), median=m, iqr_lo=l, iqr_hi=h,
                      n=k, alpha=c["alpha"], ci_lo=(c.get("ci") or [None, None])[0], ci_hi=(c.get("ci") or [None, None])[1])
                 for n, m, l, h, k in zip(N, med, lo or [None] * len(N), hi or [None] * len(N), c.get("n") or [None] * len(N))]
    ymin = min(ys + ([perm] if perm else []))
    ymax = max(ys)
    ax.set_ylim(ymin / 1.8, ymax * 2.2)
    if perm and perm > 0:
        ax.axhspan(ax.get_ylim()[0], perm, color=COL["floor"], zorder=0, lw=0)
        ax.text(ax.get_xlim()[1] / 1.03, perm, "permutation floor (bf16 reorder)", ha="right", va="top", fontsize=7.5, color=COL["muted"])
        rows.append(dict(qtype=qtype, series=f"permutation floor (fenced arm, {perm_locus} locus)", N=None, median=perm, layer=perm_src.get("layer")))
    if ctrl and ctrl > 0:
        ax.axhline(ctrl, color=COL["muted"], ls=":", lw=1.3, zorder=1)
        ax.text(ax.get_xlim()[0] * 1.05, ctrl, "control edit elsewhere", ha="left", va="bottom", fontsize=7.5, color=COL["muted"])
        rows.append(dict(qtype=qtype, series="control floor (deployed arm)", N=None, median=ctrl, layer=dep.get("layer")))
    shade_train(ax)
    deoverlap(ax, anns)
    ax.set_ylabel("median ‖Δh‖ at the answer row (log)")
    handles, labels = ax.get_legend_handles_labels()
    if perm and perm > 0:
        handles.append(Patch(facecolor=COL["floor"], edgecolor="none")); labels.append("permutation floor (bf16 reorder)")
    if ctrl and ctrl > 0:
        handles.append(Line2D([], [], color=COL["muted"], ls=":", lw=1.3)); labels.append("control edit elsewhere")
    ax.legend(handles, labels, loc="lower left", fontsize=8)
    if pos:
        side = axs[0, 1]
        vals = [v.get("alpha", v) if isinstance(v, dict) else v for v in pos.values()]
        side.bar(range(len(pos)), vals, color=COL["plain"], width=0.6)
        for i, v in enumerate(vals):
            side.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        side.set_xticks(range(len(pos)))
        side.set_xticklabels(list(pos), fontsize=7.5)
        side.set_ylim(0, max(vals) * 1.25 + 0.02)
        side.set_title("α by flip position\n(deployed read)", fontsize=8, loc="center")
        side.set_xlabel("flipped frame")
        rows += [dict(qtype=qtype, series="deployed read (by position)", arm="plain", position=k, alpha=v) for k, v in zip(pos, vals)]
    claim(fig, "One frame's influence on the answer: the deployed read dilutes with N, the fenced fact and the gated read do not",
          f"{qtype}, layer {layer}: medians over flip pairs with IQR bands; α [95% CI] printed per series")
    omit = (f"; the frozen fenced read (fenced arm, answer row) is omitted: α = {fen_read['alpha']:.2f}, median "
            f"{min(fen_read['median']):.2f}–{max(fen_read['median'][2:] or fen_read['median']):.2f} at N ≥ 32 vs permutation floor max "
            f"{perm_max:.2f}") if fen_read and perm_max else ""
    footer(fig, f"{OFFICIAL}, {n_range(ns_all)}.  Source: {rel(p)} (D1 flip influence; D6 by position); grey band = permutation "
                f"floor (fenced arm, median, measured at the {perm_locus} locus); dotted = answer-preserving control edit (deployed arm, median){omit}.")
    save(fig, out, "F4_influence_vs_N", rows)
    about(out, "F4", "F4_influence_vs_N", "the deployed read's flip response decays toward the floor with N while the fenced "
          "fact and the gated read stay flat; question-first decays faster",
          str(rel(p)), "official MMReD (DIAG D1/D6), frozen model")
    return True


# ----------------------------------------------------------------------------- F2 / F3 (sharelaw*.json)
def frame_law(n, k, s, sink):
    """The frame law: the prompt+sink takes a constant share, the frames split the rest with edge e^s per evidence frame."""
    return (1 - sink) * k * math.exp(s) / (k * math.exp(s) + (n - k))


def fig_f2(out: Path, fits: Path, layer=20, qtype=None, **_) -> bool:
    p = fits / "sharelaw.json"
    if missing("F2", p):
        return False
    fs = by_layer([f for f in load_json(p)["fits"] if cond_norm(f.get("cond")) == "base"], layer)
    qtype = default_qtype(fs, qtype)
    fit = {arm: next((f for f in fs if f["qtype"] == qtype and f["arm"] == arm), None) for arm in ("plain", "qfirst", "fenced", "gated")}
    if not (fit["plain"] or fit["qfirst"]):
        print(f"F2: inputs missing — no plain/qfirst base fit for {qtype} at layer {layer} in", p)
        return False
    ns_all = sorted({int(c["N"]) for f in fit.values() if f for c in f["cells"]})
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 5.0), gridspec_kw=dict(left=0.07, right=0.985, top=0.77, bottom=0.19, wspace=0.26))
    a, b = axs
    xlog2(a, ns_all)
    xlog2(b, ns_all)
    grid = np.exp2(np.linspace(math.log2(ns_all[0]), math.log2(ns_all[-1]), 60))
    rows, notes, fallback = [], [], []
    for arm in ("plain", "qfirst"):
        f = fit[arm]
        if not f:
            continue
        cells = sorted(f["cells"], key=lambda c: (c["N"], c["k"]))
        law, real = frame_law_of(f)
        N = sorted({int(c["N"]) for c in cells})
        m = [mass_at(f, n)[0] for n in N]
        a.plot(N, m, ls="none", marker=MARK[arm], color=COL[arm], label=LABEL[arm], zorder=3)
        ks = sorted({int(c["k"]) for c in cells})
        k_law = ks[0] if len(ks) == 1 else float(np.average([c["k"] for c in cells], weights=[c.get("n", 1) for c in cells]))
        if real and law.get("s_frame_mean") is not None and np.isfinite(law["s_frame_mean"]):
            s, sink = law["s_frame_mean"], law["sink_mean"]
            a.plot(grid, [frame_law(n, k_law, s, sink) for n in grid], color=COL[arm], lw=1.4, zorder=2)
            notes.append(f"{LABEL[arm]}: eˢ = ×{math.exp(s):.2f}, sink share {sink:.2f}")
        else:
            fallback.append(LABEL[arm])
        rows += [dict(panel="a_mass", qtype=qtype, series=LABEL[arm], arm=arm, layer=f.get("layer"), N=n, k=k_law, evid_mass=mm,
                      law=(frame_law(n, k_law, law["s_frame_mean"], law["sink_mean"]) if real and law.get("s_frame_mean") is not None else None),
                      s_frame_mean=law.get("s_frame_mean"), sink_mean=law.get("sink_mean"), frame_law_from="fitter" if real else "cell means")
                 for n, mm in zip(N, m)]
    if fit["gated"]:
        f = fit["gated"]
        N = sorted({int(c["N"]) for c in f["cells"]})
        m = [mass_at(f, n)[0] for n in N]
        a.plot(N, m, ls="--", marker=MARK["gated"], color=COL["gated"], label="gated read (N removed)")
        rows += [dict(panel="a_mass", qtype=qtype, series=LABEL["gated"], arm="gated", layer=f.get("layer"), N=n, evid_mass=mm) for n, mm in zip(N, m)]
    shade_train(a)
    a.set_ylim(0, None)
    a.set_ylabel(f"mass on the evidence frame, layer {layer} (head-mean)")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=8)
    a.set_title("a  evidence mass: (1 − sink) · k·eˢ / (k·eˢ + N − k)")
    if notes:
        a.text(0.98, 0.44, "\n".join(notes + ["eˢ = needle vs one non-evidence frame"]), transform=a.transAxes, ha="right", va="top",
               fontsize=7.6, color=COL["ink"], bbox=dict(boxstyle="round,pad=0.3", fc=COL["surface"], ec=COL["grid"], lw=0.8))
    anns = []
    for arm in ("plain", "qfirst", "fenced"):
        f = fit[arm]
        if not f:
            continue
        law, real = frame_law_of(f)
        if not law.get("N"):
            continue
        N, edge = [int(n) for n in law["N"]], [float(e) for e in law["edge"]]
        b.plot(N, edge, marker=MARK[arm], color=COL[arm], label=LABEL[arm] if arm != "fenced" else "fenced fact (per-block photograph)")
        anns.append(exponent(b, N[-1], edge[-1], float(np.mean(edge)), COL[arm], name="mean eˢ", fmt="×{:.2f}"))
        rows += [dict(panel="b_edge", qtype=qtype, series=LABEL[arm], arm=arm, layer=f.get("layer"), N=n, edge=e,
                      sink=sk, frame_law_from="fitter" if real else "cell means") for n, e, sk in zip(N, edge, law["sink"])]
        if not real and LABEL[arm] not in fallback:
            fallback.append(LABEL[arm])
    b.axhline(1.0, color=COL["muted"], lw=0.9, ls="--")
    b.text(b.get_xlim()[0] * 1.05, 1.0, "no edge (eˢ = 1)", ha="left", va="bottom", fontsize=7.5, color=COL["muted"])
    shade_train(b)
    b.set_ylim(0, max([e for r in rows if r["panel"] == "b_edge" for e in [r["edge"]]] or [1]) * 1.3)   # headroom for the printed edges
    b.set_ylabel("per-frame edge  eˢ = (evid / k) / (nonevid / (N − k))")
    b.set_title("b  the edge is constant in N;\n    the (N − k) term does the diluting")
    b.legend(loc="lower left", fontsize=8)
    deoverlap(b, anns)
    claim(fig, "Attention on the evidence follows the frame law: a fixed per-frame edge, diluted by the N − k other frames; the gate removes N",
          f"{qtype}, layer {layer}: points = measured mass, curve = the frame law from the fitted edge and sink share (no free parameter per N)")
    fb = (f"  frame_law absent for {', '.join(dict.fromkeys(fallback))}: no curve drawn; edges are computed from the cells' mean masses "
          "(rerun after the CPU fit pass).") if fallback else ""
    footer(fig, f"{OFFICIAL}, {n_range(ns_all)}.  Source: {rel(p)} (D2 photograph; frame law from diag_fit.py); the 2-parameter "
                f"(s, C) fit is not shown; gated overlay = oracle gate.{fb}")
    save(fig, out, "F2_share_law", rows)
    about(out, "F2", "F2_share_law", "evidence mass follows (1 − sink)·k·eˢ/(k·eˢ + N − k) with a per-frame edge that is constant "
          "in N; the gated read is N-invariant", str(rel(p)), "official MMReD (DIAG D2/D2b), frozen model")
    return True


X_LOGN = 5.0                                                     # the log-N condition sits on its own tick, right of τ = 4


def tau_series(ax, pts: list[tuple[str, float, float]], color: str, marker: str, label: str, above: bool = True, fmt="{:.2f}") -> None:
    """pts = [(cond, x, y)]: τ points joined by a line, the log-N point isolated (hollow marker); values printed."""
    tau = sorted((x, y) for c, x, y in pts if cond_x(c)[0] == "tau")
    logn = [(x, y) for c, x, y in pts if cond_x(c)[0] == "logn"]
    if tau:
        ax.plot(*zip(*tau), marker=marker, color=color, label=label)
    for x, y in logn:
        ax.plot([x], [y], marker=marker, color=color, ls="none", mfc=COL["surface"], mec=color, markersize=7, mew=1.6)
    for x, y in tau + logn:
        ax.annotate(fmt.format(y), (x, y), xytext=(0, 6 if above else -6), textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=7, color=color)


def tau_axis(ax, taus: list[float], has_logn: bool) -> None:
    ticks = sorted(set(taus) | {1.0})
    ax.set_xticks(ticks + ([X_LOGN] if has_logn else []))
    ax.set_xticklabels([f"{t:g}" for t in ticks] + (["log-N"] if has_logn else []))
    ax.set_xlim(min(ticks) - 0.35, (X_LOGN if has_logn else max(ticks)) + 0.4)
    if has_logn:
        ax.axvline((max(ticks) + X_LOGN) / 2, color=COL["grid"], lw=1, ls="--")
    ax.set_xlabel("τ (temperature)" + ("  ·  log-N" if has_logn else ""))


def fig_f3(out: Path, fits: Path, layer=20, **_) -> bool:
    p = fits / "sharelaw_d3.json"
    if missing("F3", p):
        return False
    fs = [f for f in by_layer(load_json(p)["fits"], layer) if cond_x(f.get("cond"))]
    acc = [a for a in load_acc(fits) if a["arm"] == "qfirst" and cond_x(a["cond"])]
    if not fs:
        print("F3: inputs missing — no τ / log-N conditions in", p)
        return False
    conds_at = {}
    for f in fs:
        for c in f["cells"]:
            conds_at.setdefault(int(c["N"]), set()).add(cond_norm(f["cond"]))
    Ns = [n for n in sorted(conds_at) if len(conds_at[n]) >= 2]
    if not Ns:
        print("F3: inputs missing — no N with two or more conditions in", p)
        return False
    taus = sorted({cond_x(f["cond"])[1] for f in fs if cond_x(f["cond"])[0] == "tau"})
    has_logn = any(cond_x(f["cond"])[0] == "logn" for f in fs)
    xpos = lambda cond: cond_x(cond)[1] if cond_x(cond)[0] == "tau" else X_LOGN
    H = 3.5 * len(Ns) + 1.7
    fig, axs = plt.subplots(len(Ns), 3, figsize=(10.6, H), squeeze=False,
                            gridspec_kw=dict(left=0.065, right=0.95, top=1 - 1.25 / H, bottom=1.0 / H, wspace=0.5, hspace=0.42))
    rows, fallback, handles = [], set(), {}
    for r, N in enumerate(Ns):
        a, b, c = axs[r]
        # (a) per-frame edge at N: needle vs count
        for q in ("char_at_frame", "steps_in_room"):
            pts = []
            for f in fs:
                if f["qtype"] != q:
                    continue
                law, real = frame_law_of(f)
                if N in [int(n) for n in law.get("N", [])]:
                    i = [int(n) for n in law["N"]].index(N)
                    pts.append((f["cond"], xpos(f["cond"]), float(law["edge"][i])))
                    if not real:
                        fallback.add(q)
                    rows.append(dict(panel="a_edge", N=N, qtype=q, cond=cond_norm(f["cond"]), x=xpos(f["cond"]), y=float(law["edge"][i]),
                                     layer=f.get("layer"), frame_law_from="fitter" if real else "cell means"))
            if pts:
                tau_series(a, pts, QCOL[q], QMARK[q], QLAB[q], above=(q == "char_at_frame"))
        a.axhline(1.0, color=COL["muted"], lw=0.9, ls="--")
        a.set_ylabel(f"per-frame edge eˢ, layer {layer}")
        a.set_ylim(0, max([r["y"] for r in rows if r["panel"] == "a_edge" and r["N"] == N] or [1]) * 1.18)
        a.set_title(f"a  N = {N}: the needle's edge rises")
        # (b) needle: total evidence mass (left axis) + sink share (right axis)
        b2 = b.twinx()
        b2.grid(False)
        for spine in ("top",):
            b2.spines[spine].set_visible(False)
        pe, ps = [], []
        for f in fs:
            if f["qtype"] != "char_at_frame":
                continue
            law, real = frame_law_of(f)
            e, ne, other = mass_at(f, N)
            if np.isfinite(e):
                pe.append((f["cond"], xpos(f["cond"]), e))
            ln = [int(n) for n in law.get("N", [])]
            sk = float(law["sink"][ln.index(N)]) if N in ln else other
            if np.isfinite(sk):
                ps.append((f["cond"], xpos(f["cond"]), sk))
            rows.append(dict(panel="b_mass", N=N, qtype="char_at_frame", cond=cond_norm(f["cond"]), x=xpos(f["cond"]), evid_mass=e,
                             nonevid_mass=ne, sink_share=sk, layer=f.get("layer")))
        if pe:
            tau_series(b, pe, QCOL["char_at_frame"], QMARK["char_at_frame"], "needle evidence mass (left axis)", above=True, fmt="{:.3f}")
        if ps:
            tau_series(b2, ps, COL["sink"], "D", "sink share, prompt + sink (right axis)", above=False)
        b.set_ylim(0, max([y for _, _, y in pe] or [1]) * 1.6)
        b2.set_ylim(0, 1.0)
        b.set_ylabel("evidence mass (needle)", color=QCOL["char_at_frame"])
        b2.set_ylabel("sink share", color=COL["sink"])
        b.tick_params(axis="y", colors=QCOL["char_at_frame"])
        b2.tick_params(axis="y", colors=COL["sink"])
        b.set_title(f"b  N = {N}: the sink takes the mass")
        # (c) exact match at N, question-first, for the three qtypes
        for q in ("char_at_frame", "steps_in_room", "where_spend"):
            pts = [(x["cond"], xpos(x["cond"]), x["acc"]) for x in acc if x["qtype"] == q and x["N"] == N]
            if pts:
                tau_series(c, pts, QCOL[q], QMARK[q], QLAB[q], above=(q != "steps_in_room"))
                rows += [dict(panel="c_acc", N=N, qtype=q, cond=cond_norm(cn), x=xx, y=yy) for cn, xx, yy in pts]
        c.set_ylim(0, 1.05)
        c.set_ylabel("exact match (question-first)")
        c.set_title(f"c  N = {N}: exact match falls")
        for ax in (a, b, c):
            tau_axis(ax, taus, has_logn)
        for ax in (a, b, b2, c):
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
    if has_logn:
        handles["log-N (isolated hollow marker)"] = Line2D([], [], marker="o", ls="none", mfc=COL["surface"], mec=COL["ink"], mew=1.4, markersize=7)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper right", bbox_to_anchor=(0.985, 1 - 0.55 / H), ncol=3, fontsize=8)
    claim(fig, "Sharpening raises the needle's edge, the sink takes the mass, and both counts and needles lose accuracy",
          f"frozen model, question-first layout; τ = 1 is the base condition; one row per N present ({', '.join(map(str, Ns))})")
    per_n = "; ".join(f"N = {n}: {', '.join(sorted(conds_at[n], key=lambda s: (cond_x(s)[0] != 'tau', cond_x(s)[1])))}" for n in Ns)
    fb = f"  frame_law absent ({', '.join(sorted(fallback))}): edges from the cells' mean masses — rerun after the CPU fit pass." if fallback else ""
    footer(fig, f"{OFFICIAL}; QUESTION-FIRST layout (regime label: not the deployed images-then-question order). τ divides the attention "
                f"logits on decoder modules ≥ 12; log-N = logit scaling ln(seq)/ln(sref), sref = 3000 tokens.  Conditions present: {per_n}.  "
                f"Source: {rel(p)} (D3 photograph, layer {layer}) + eval_summary.csv (D3 accuracy).{fb}")
    save(fig, out, "F3_tau_dissociation", rows)
    about(out, "F3", "F3_tau_dissociation", "raising τ raises the needle's per-frame edge but the freed mass goes to the sink; "
          "needle and count accuracy both fall", str(rel(p)) + " + eval_summary.csv", "official MMReD (DIAG D3), frozen, question-first")
    return True


# ----------------------------------------------------------------------------- F7 (headscan.json) / F10 (margins.json)
def fig_f7(out: Path, fits: Path, qtype=None, **_) -> bool:
    p = fits / "headscan.json"
    if missing("F7", p):
        return False
    d = load_json(p)
    groups, thresh = d.get("groups") or [], float(d.get("thresh", 0.98))
    want = [q for q in ("char_at_frame", "steps_in_room") if qtype in (None, q)]
    sel = []
    for q in want:
        for arm in ("plain", "qfirst", "fenced"):
            g = next((g for g in groups if g["qtype"] == q and g["arm"] == arm and cond_norm(g.get("cond")) == "base"), None)
            if g and len(sel) < 4:
                sel.append(g)
    if not sel:
        print("F7: inputs missing — no plain/qfirst/fenced groups for", want, "in", p)
        return False
    acc = [a for a in load_acc(fits) if a["arm"] == "plain" and a["cond"] == "base" and "eval/grid" in a["run"] and a["qtype"] in want]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.5), gridspec_kw=dict(left=0.07, right=0.985, top=0.80, bottom=0.15, wspace=0.28))
    rows, best_all, n_over = [], 0.0, 0
    for g in sel:
        Ns = [int(n) for n in g["N"]]
        best = []
        for n in Ns:
            cand = [(a.get(str(n), a.get(n)), h) for h, a in g["auc"].items() if str(n) in a or n in a]
            v, h = max(cand) if cand else (np.nan, None)
            best.append((v, h))
        n_over += sum(int((g.get("population") or {}).get(str(n), {}).get("count", 0)) for n in Ns)
        best_all = max(best_all, max(v for v, _ in best))
        ls = "-" if g["qtype"] == "char_at_frame" else "--"
        axs[0].plot(Ns, [v for v, _ in best], marker=MARK[g["arm"]], color=COL[g["arm"]], ls=ls, label=f"{LABEL[g['arm']]} · {g['qtype']}")
        rows += [dict(panel="a_best_auc", qtype=g["qtype"], arm=g["arm"], series=LABEL[g["arm"]], N=n, y=v, head=h,
                      n_heads_over_thresh=int((g.get("population") or {}).get(str(n), {}).get("count", 0)))
                 for n, (v, h) in zip(Ns, best)]
    axs[0].axhline(thresh, color=COL["muted"], lw=0.9, ls=":")
    axs[0].text(axs[0].get_xlim()[1] if axs[0].get_xlim()[1] > 1 else NS[-1], thresh, f"AUC {thresh:g}", ha="right", va="bottom", fontsize=7.5, color=COL["muted"])
    axs[0].axhline(0.5, color=COL["muted"], lw=0.8)
    note = (f"no head ≥ {thresh:g} in any frozen arm (best {best_all:.2f})" if n_over == 0
            else f"{n_over} head×N cells ≥ {thresh:g} (best {best_all:.2f})")
    axs[0].text(0.98, 0.04, note, transform=axs[0].transAxes, ha="right", va="bottom", fontsize=8, color=COL["ink"],
                bbox=dict(boxstyle="round,pad=0.25", fc=COL["surface"], ec=COL["grid"]))
    for q in want:
        pts = sorted((a["N"], a["acc"]) for a in acc if a["qtype"] == q)
        if not pts:
            continue
        ls = "-" if q == "char_at_frame" else "--"
        axs[1].plot(*zip(*pts), marker=QMARK[q], color=COL["plain"], ls=ls)
        for n, v in pts:
            axs[1].annotate(f"{v:.2f}", (n, v), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=7.2)
        axs[1].annotate(q, pts[-1], xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=COL["ink"])
        rows += [dict(panel="b_exact_match", qtype=q, arm="plain", series="deployed read", N=n, y=v) for n, v in pts]
    for ax in axs:
        xlog2(ax)
        shade_train(ax)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("best single-head AUC, evidence vs the rest (per N)")
    axs[1].set_ylabel("exact match, deployed read (images then question)")
    axs[0].set_title("a  no single frozen head cleanly ranks the evidence")
    axs[1].set_title("b  and the answer falls with N")
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=8)
    if not acc:
        axs[1].text(0.5, 0.5, "no deployed-grid accuracy rows\n(eval_summary.csv, run ∋ eval/grid)", transform=axs[1].transAxes,
                    ha="center", va="center", fontsize=8, color=COL["muted"])
    claim(fig, f"No single frozen head separates the evidence (best AUC {best_all:.2f} < {thresh:g}) — and exact match falls with N",
          "a: max over 140 scanned heads (layers 12–27) of the evidence-vs-rest AUC of block mass, frozen arms; b: the faithful grid")
    footer(fig, f"{OFFICIAL}.  Source: {rel(p)} (D2/D8 head scan; AUC per head × N, best head per N in the csv) + eval_summary.csv "
                "rows with arm = plain, cond = base, run under outputs/diag/eval/grid (the deployed layout).")
    save(fig, out, "F7_rank_vs_count", rows)
    about(out, "F7", "F7_rank_vs_count", f"the best frozen head reaches AUC {best_all:.2f} (none ≥ {thresh:g}) at any N while exact "
          "match falls with N", str(rel(p)) + " + eval_summary.csv", "official MMReD (DIAG D2/D8), frozen model")
    return True


def fig_f10(out: Path, fits: Path, qtype=None, **_) -> bool:
    p = fits / "margins.json"
    if missing("F10", p):
        return False
    cells = [c for c in load_json(p)["cells"] if cond_norm(c.get("cond")) == "base"]
    qs = [q for q in ("char_at_frame", "steps_in_room") if qtype in (None, q) and any(c["qtype"] == q for c in cells)]
    if not qs and cells:
        qs = [default_qtype(cells, qtype)]
    if not qs:
        print("F10: inputs missing — no base cells in", p)
        return False
    fig, axs = plt.subplots(1, len(qs), figsize=(4.9 * len(qs) + 0.6, 4.7), squeeze=False,
                            gridspec_kw=dict(left=0.09 if len(qs) == 1 else 0.065, right=0.985, top=0.80, bottom=0.17, wspace=0.24))
    rows, handles = [], {}
    for i, (ax, q) in enumerate(zip(axs[0], qs)):
        sub = {arm: next((c for c in cells if c["qtype"] == q and c["arm"] == arm), None) for arm in ("plain", "qfirst", "gated")}
        sub = {k: v for k, v in sub.items() if v}
        Ns = sorted({int(n) for c in sub.values() for n in c["N"]})
        xlog2(ax, Ns)
        floor = [max((c["noise_floor"][c["N"].index(n)] for c in sub.values() if n in c["N"] and c.get("noise_floor")), default=0.0) for n in Ns]
        if any(floor):
            ax.fill_between(Ns, [-f for f in floor], floor, color=COL["floor"], lw=0, zorder=0)
            handles.setdefault("bf16 noise floor (± per N)", Patch(facecolor=COL["floor"], edgecolor="none"))
        anns = []
        for arm in ("plain", "qfirst", "gated"):
            c = sub.get(arm)
            if not c:
                continue
            N, m = [int(n) for n in c["N"]], [float(v) for v in c["margin_median"]]
            lo, hi = c.get("iqr_lo") or c.get("margin_iqr_lo"), c.get("iqr_hi") or c.get("margin_iqr_hi")
            if lo and hi:
                ax.fill_between(N, lo, hi, color=COL[arm], alpha=0.12, lw=0)
            ax.plot(N, m, marker=MARK[arm], color=COL[arm], label=LABEL[arm])
            anns.append(exponent(ax, N[-1], m[-1], m[-1], COL[arm], name="m", fmt="{:+.2f}"))
            rows += [dict(qtype=q, series=LABEL[arm], arm=arm, N=n, margin_median=v, iqr_lo=l, iqr_hi=h, accuracy=a, flip_changes_answer=fl,
                          noise_floor=nf, n=k)
                     for n, v, l, h, a, fl, nf, k in zip(N, m, lo or [None] * len(N), hi or [None] * len(N), c.get("accuracy") or [None] * len(N),
                                                          c.get("flip_changes_answer") or [None] * len(N), c.get("noise_floor") or [None] * len(N),
                                                          c.get("n") or [None] * len(N))]
        ax.axhline(0, color=COL["ink"], lw=1)
        ax.text(ax.get_xlim()[1] / 1.03, 0, "decision boundary", ha="right", va="bottom", fontsize=7.5, color=COL["muted"])
        shade_train(ax)
        ns = next((c.get("n") for c in sub.values() if c.get("n")), None)
        if ns and len(set(ns)) > 1:
            ax.text(0.98, 0.02, "rows per N: " + " · ".join(str(int(k)) for k in ns), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.2, color=COL["muted"])
        ax.set_ylabel("first-token logit margin, gold − best rival (median)")
        ax.set_title(f"{'ab'[i]}  {QLAB.get(q, q)}")
        deoverlap(ax, anns)
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axs[0, 0].legend(list(handles.values()), list(handles.keys()), loc="lower left", fontsize=8)   # the needle panel's lower left is empty
    claim(fig, "The first-token margin slides through zero with N; the gate holds it for the needle, not for the count",
          "median gold − best-rival logit margin at the first answer token; bands = IQR over rows; grey = bf16 noise floor")
    footer(fig, f"{OFFICIAL}.  Source: {rel(p)} (D1 margins); accuracy, flip_changes_answer and rows-per-N for every point are in the csv; "
                "the frozen fenced read is not drawn.")
    save(fig, out, "F10_margin_vs_N", rows)
    about(out, "F10", "F10_margin_vs_N", "the deployed and question-first margins slide through zero with N; the gated margin holds for "
          "the needle and collapses for steps_in_room", str(rel(p)), "official MMReD (DIAG D1), frozen model")
    return True


# ----------------------------------------------------------------------------- CLI
FIGS = {"f1": fig_f1, "f2": fig_f2, "f3": fig_f3, "f4": fig_f4, "f5": fig_f5, "f7": fig_f7, "f8": fig_f8, "f10": fig_f10}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fig", choices=list(FIGS) + ["all-available"])
    ap.add_argument("--fits", type=Path, default=FITS_DEFAULT, help="dir with alpha/sharelaw/sharelaw_d3/margins/headscan.json")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--layer", type=int, default=20, help="layer to draw for the JSON-driven figures")
    ap.add_argument("--qtype", default=None, help="restrict the JSON-driven figures to one question type")
    a = ap.parse_args(argv)
    style()
    todo = list(FIGS) if a.fig == "all-available" else [a.fig]
    done, failed = {}, []
    for f in todo:
        try:
            done[f] = FIGS[f](out=a.out, fits=a.fits, layer=a.layer, qtype=a.qtype)
        except Exception as e:                       # one figure's failure never aborts the set
            if a.fig != "all-available":
                raise
            tb = e.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            where = f" ({Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno})" if tb else ""
            print(f"{f.upper()}: failed — {type(e).__name__}: {e}{where}")
            done[f] = False
            failed.append(f)
            plt.close("all")
    print("drawn:", [f for f, ok in done.items() if ok] or "none", "| skipped:", [f for f, ok in done.items() if not ok and f not in failed] or "none",
          "| failed:", failed or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
