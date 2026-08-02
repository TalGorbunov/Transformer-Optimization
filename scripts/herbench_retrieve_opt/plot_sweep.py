#!/usr/bin/env python3
"""Phase-2 figures + verdict table for the HERBench retrieve-opt sweep.

Reads the per-arm probe report.json files (latest run each) under
outputs/herbench_retrieve_opt/probe/<arm>/<ts>/report.json and produces:
  Figure 1  d' vs delta, one line per resolution (pooled + per key-verb panels),
            A0/A1 as the delta=0 points, horizontal bar at the 2.5 decision line.
  Figure 2  per-verb d' bars for the best arm (sorted), bar at 2.5.
plus a verdict.md summarising peak-over-layer d' per arm/verb vs the bar.

d' reported = max over probed layers (peak read layer), which is what the decision bar is
about. CPU-only. Usage:
  python scripts/herbench_retrieve_opt/plot_sweep.py \
    --probe-base outputs/herbench_retrieve_opt/probe --out outputs/herbench_retrieve_opt/phase2
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# arm dir -> (delta, res, kind)
ARMS = {
    "A0_r448": (0.0, 448, "anchor"), "A1_r672": (0.0, 672, "anchor"),
    "B_d0.5_r448": (0.5, 448, "clip"), "B_d1_r448": (1.0, 448, "clip"),
    "B_d2_r448": (2.0, 448, "clip"),
    "C_d0.5_r672": (0.5, 672, "clip"), "C_d1_r672": (1.0, 672, "clip"),
    "C_d2_r672": (2.0, 672, "clip"),
}
BAR = 2.5
KEY_VERBS = ["open", "close", "pick", "put"]


def latest_report(base: Path, arm: str):
    ds = sorted(glob.glob(str(base / arm / "2*" / "report.json")))
    return json.load(open(ds[-1])) if ds else None


def peak_pooled(rep):
    vals = [bl["pooled"]["dprime"] for bl in rep["by_layer"].values()
            if bl["pooled"]["dprime"] == bl["pooled"]["dprime"]]
    return max(vals) if vals else float("nan")


def peak_verb(rep, verb):
    vals = []
    for bl in rep["by_layer"].values():
        pv = bl["per_verb"].get(verb)
        if pv and pv["dprime"] == pv["dprime"]:
            vals.append(pv["dprime"])
    return max(vals) if vals else float("nan")


def verb_npos(rep, verb):
    for bl in rep["by_layer"].values():
        pv = bl["per_verb"].get(verb)
        if pv:
            return pv["npos"]
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-base", default="outputs/herbench_retrieve_opt/probe")
    ap.add_argument("--out", default="outputs/herbench_retrieve_opt/phase2")
    args = ap.parse_args()
    base = Path(args.probe_base); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    reps = {arm: latest_report(base, arm) for arm in ARMS}
    reps = {a: r for a, r in reps.items() if r is not None}

    # ---- Figure 1: d' vs delta, line per resolution (pooled + per key verb) ----
    fig, axes = plt.subplots(1, 1 + len(KEY_VERBS), figsize=(4 * (1 + len(KEY_VERBS)), 4),
                             squeeze=False)
    panels = [("pooled", None)] + [(v, v) for v in KEY_VERBS]
    for ax_i, (title, verb) in enumerate(panels):
        ax = axes[0][ax_i]
        for res, mk in ((448, "o-"), (672, "s--")):
            xs, ys = [], []
            for arm, (d, r, kind) in sorted(ARMS.items(), key=lambda kv: kv[1][0]):
                if r != res or arm not in reps:
                    continue
                y = peak_pooled(reps[arm]) if verb is None else peak_verb(reps[arm], verb)
                if y == y:
                    xs.append(d); ys.append(y)
            if xs:
                ax.plot(xs, ys, mk, label=f"{res}px")
        ax.axhline(BAR, color="red", ls=":", lw=1, label="bar 2.5")
        ax.set_title(f"{title}"); ax.set_xlabel("±δ (s)"); ax.set_ylabel("peak d′")
        ax.set_ylim(0, max(3.0, BAR + 0.3)); ax.grid(alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("HERBench retrieve-opt: per-unit d′ vs temporal context ±δ (peak read layer)")
    fig.tight_layout()
    fig.savefig(out / "fig1_dprime_vs_delta.png", dpi=130)
    plt.close(fig)

    # ---- pick best arm = max over (well-powered verb peak) among clip arms ----
    def arm_score(arm):
        return max((peak_verb(reps[arm], v) for v in KEY_VERBS
                    if verb_npos(reps[arm], v) >= 40), default=float("nan"))
    clip_arms = [a for a in reps if ARMS[a][2] == "clip"]
    best_arm = max(clip_arms, key=lambda a: (arm_score(a) if arm_score(a) == arm_score(a) else -1))

    # ---- Figure 2: per-verb d' bars for best arm ----
    rep = reps[best_arm]
    verbs_all = set()
    for bl in rep["by_layer"].values():
        verbs_all |= set(bl["per_verb"].keys())
    rows = [(v, peak_verb(rep, v), verb_npos(rep, v)) for v in verbs_all]
    rows = [r for r in rows if r[1] == r[1]]
    rows.sort(key=lambda r: -r[1])
    fig2, ax = plt.subplots(figsize=(max(6, 0.7 * len(rows)), 4))
    labels = [f"{v}\n(n={n})" for v, _, n in rows]
    ax.bar(labels, [d for _, d, _ in rows],
           color=["C2" if d >= BAR else "C0" for _, d, _ in rows])
    ax.axhline(BAR, color="red", ls=":", lw=1)
    ax.set_ylabel("peak d′"); ax.set_title(f"Per-verb d′ — best arm {best_arm} (bar 2.5)")
    ax.grid(alpha=0.3, axis="y")
    fig2.tight_layout(); fig2.savefig(out / "fig2_per_verb_best_arm.png", dpi=130)
    plt.close(fig2)

    # ---- verdict table ----
    lines = ["# HERBench retrieve-opt — Phase 2 verdict\n",
             "Peak-over-layer whitened d′ (question-grouped). Decision bar d′≥2.5.\n",
             "## Pooled d′ by arm", "| arm | δ | res | peak pooled d′ |", "|---|---|---|---|"]
    for arm, (d, r, k) in sorted(ARMS.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        if arm in reps:
            lines.append(f"| {arm} | {d} | {r} | {peak_pooled(reps[arm]):.2f} |")
    lines += ["\n## Per-verb peak d′ (well-powered, n_pos≥40 unless noted)",
              "| arm | " + " | ".join(KEY_VERBS) + " |", "|---|" + "---|" * len(KEY_VERBS)]
    for arm, (d, r, k) in sorted(ARMS.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        if arm not in reps:
            continue
        cells = []
        for v in KEY_VERBS:
            dv = peak_verb(reps[arm], v); n = verb_npos(reps[arm], v)
            cells.append(f"{dv:.2f}{'*' if dv >= BAR else ''}{' (n=%d)' % n if n < 40 else ''}"
                         if dv == dv else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append(f"\nbest clip arm by well-powered verb peak: **{best_arm}** "
                 f"(peak {arm_score(best_arm):.2f})")
    (out / "verdict.md").write_text("\n".join(lines) + "\n")
    print("wrote", out / "fig1_dprime_vs_delta.png", out / "fig2_per_verb_best_arm.png",
          out / "verdict.md")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
