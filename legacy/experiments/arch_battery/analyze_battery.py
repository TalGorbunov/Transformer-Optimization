#!/usr/bin/env python3
"""Exp 1 Step 4: merge per-model battery results + the 2-panel figure (EM vs N, range vs N)."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing <model_short>/results.csv")
    ap.add_argument("--anchor", default="Qwen2.5-VL-7B-text:softmax:8=0.196,16=0.062,24=0.035,40=0.020",
                    help="name:class:N=em,... measured anchor row(s); '' to skip")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.output) if args.output else root
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(root.glob("*/results.csv")):
        with f.open() as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        print("no results.csv found under", root)
        return 1
    hdr = ["model", "arch_class", "N", "n", "em", "mae", "bias", "range_p95_p5",
           "spearman", "majority", "parse_fail"]
    with (out / "battery_all.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in hdr})
    print(f"merged {len(rows)} rows from {len(set(r['model'] for r in rows))} models")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    models = sorted({(r["model"], r["arch_class"]) for r in rows})
    cmap = plt.get_cmap("tab10")
    for i, (m, klass) in enumerate(models):
        rr = sorted((int(r["N"]), float(r["em"]), float(r["range_p95_p5"]), float(r["majority"]))
                    for r in rows if r["model"] == m)
        Ns = [x[0] for x in rr]
        style = "--" if klass == "softmax" else "-"
        label = f"{m.split('/')[-1]} ({klass})"
        axes[0].plot(Ns, [x[1] for x in rr], style, marker="o", color=cmap(i % 10), label=label)
        axes[1].plot(Ns, [x[2] for x in rr], style, marker="o", color=cmap(i % 10), label=label)
    # majority band (per-N max over models, they share data)
    Ns_all = sorted({int(r["N"]) for r in rows})
    maj = [max(float(r["majority"]) for r in rows if int(r["N"]) == N) for N in Ns_all]
    axes[0].fill_between(Ns_all, 0, maj, color="gray", alpha=0.25, label="majority baseline")
    if args.anchor:
        for a in args.anchor.split(";"):
            name, klass, pts = a.split(":", 2)
            pn = sorted((int(p.split("=")[0]), float(p.split("=")[1])) for p in pts.split(","))
            axes[0].plot([x[0] for x in pn], [x[1] for x in pn], ":", marker="s",
                         color="black", label=f"{name} (anchor, {klass})")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns_all, [str(n) for n in Ns_all])
        ax.set_xlabel("N (frames)")
    axes[0].set_ylabel("exact match")
    axes[0].set_title("steps_in_room text battery: EM vs N (dashed=softmax, solid=extensive-state)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("emitted range (p95−p5 of predictions)")
    axes[1].set_title("clamp signature: emitted range vs N")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_arch_battery.png", dpi=130)
    print(f"wrote {out}/battery_all.csv + fig_arch_battery.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
