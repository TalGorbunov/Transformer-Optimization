#!/usr/bin/env python3
"""Per-gold-count model accuracy vs extraction-bound ceiling, recomputed from each run's
predictions.csv (has seq_len+gold) and the measured L19 extraction p. Reproduces acc_per_count_iid.png
numerically so we can compare model vs ceiling per count."""
import csv, json, random
from pathlib import Path

ROOT = Path("/home/tal.gorbunov/projects/Transformer-Optimization")
RUNS = {
    "room_busy": "outputs/frame_axis_aggregator_adapter/cat1_room_busy/20260620_191327_deepsets",
    "char_accompanied": "outputs/frame_axis_aggregator_adapter/cat1_char_accompanied/20260620_191324_deepsets",
    "char_alone": "outputs/frame_axis_aggregator_adapter/cat1_char_alone/20260620_191324_deepsets",
}


def ceiling_for(n, k, p, sims=4000):
    """MonteCarlo exact-match ceiling for one sample: k ones + (n-k) zeros, each flipped w.p. (1-p),
    perfect sum; P(noisy sum == k)."""
    ev = [1] * k + [0] * (n - k)
    rng = random.Random(1234 + n * 100 + k)
    cor = 0
    for _ in range(sims):
        cor += int(sum((e if rng.random() < p else 1 - e) for e in ev) == k)
    return cor / sims


for task, rd in RUNS.items():
    rd = ROOT / rd
    p = json.loads((rd / "extraction_p.json").read_text())[task]
    rows = list(csv.DictReader((rd / "predictions.csv").open()))
    rows = [r for r in rows if r["split"] == "test_iid" and r["task"] == task]
    by_count = {}
    for r in rows:
        g = int(r["gold"]); by_count.setdefault(g, []).append((int(r["seq_len"]), int(r["pred"])))
    print(f"\n=== {task}  (measured L19 p={p:.3f}, n={len(rows)}) ===")
    print(f"{'gold':>4} {'n':>4} {'model_acc':>10} {'ceiling':>8} {'model-ceil':>11}")
    tot_m = tot_c = 0.0
    for g in sorted(by_count):
        items = by_count[g]
        m = sum(int(pr == g) for _, pr in items) / len(items)
        c = sum(ceiling_for(n, g, p) for n, _ in items) / len(items)
        tot_m += m * len(items); tot_c += c * len(items)
        print(f"{g:>4} {len(items):>4} {m:>10.3f} {c:>8.3f} {m-c:>+11.3f}")
    N = len(rows)
    print(f"{'ALL':>4} {N:>4} {tot_m/N:>10.3f} {tot_c/N:>8.3f} {(tot_m-tot_c)/N:>+11.3f}")
