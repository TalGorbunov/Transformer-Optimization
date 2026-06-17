#!/usr/bin/env python3
"""Decisive diagnostic for the rooms-visited (set-cardinality) task.

Question: is distinct-room cardinality *linearly readable* from the per-frame messages that the
adapter aggregates at the injection layer? If a linear probe CANNOT read it off the pooled messages,
then no readout (sum / max / PNA / slots) can recover it -- the dedup-relevant info (room identity)
has already been squashed out upstream. If it CAN, the limitation is the readout's expressivity.

We probe several pooled features of the raw per-frame messages, per injection layer:
  - raw_message_sum : sum over (valid) frames        -- what carrier_direct_sum sees
  - raw_message_max : max over (valid) frames        -- what carrier_max sees
  - raw_message_mean: mean over (valid) frames
  - read            : the adapter's actual read vector
  - injection       : the vector injected into the residual stream

Reads the diagnostic .npz dumps already written by layerwise_frame_message_glstm.py (no GPU, no
retrain). gold_count (= distinct rooms) is parsed from the npz filename (..._rooms<N>_...).
Reports, per (run, feature, layer): Ridge round-to-int accuracy + MAE, and a count-blind baseline
(always predict the train-set mean) for reference.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOMS_RE = re.compile(r"_rooms(\d+)_")
LAYER_RE = re.compile(r"_layer(\d+)\.npz$")


def parse_npz_name(name: str) -> Optional[Tuple[str, int, int]]:
    """Return (example_key, gold_count, layer) or None."""
    m_rooms = ROOMS_RE.search(name)
    m_layer = LAYER_RE.search(name)
    if not (m_rooms and m_layer):
        return None
    gold = int(m_rooms.group(1))
    layer = int(m_layer.group(1))
    key = LAYER_RE.sub("", name)  # strip _layerNN.npz -> per-example key
    return key, gold, layer


def pooled_features(npz_path: Path) -> Dict[str, np.ndarray]:
    d = np.load(npz_path)
    raw = np.asarray(d["raw_messages"])[0]   # (carriers, frames, dim)
    read = np.asarray(d["read"])[0]          # (carriers, dim_mem)
    inj = np.asarray(d["injection"])[0]      # (carriers, dim)
    valid = np.asarray(d["valid"])[0] if "valid" in d else None  # (carriers, frames)
    carriers, frames, dim = raw.shape
    # collapse carriers by mean (carrier dim is 1 here, but be safe)
    raw_c = raw.mean(axis=0)                  # (frames, dim)
    if valid is not None:
        v = valid.mean(axis=0) > 0.5         # (frames,)
        if not v.any():
            v = np.ones(frames, dtype=bool)
    else:
        v = np.ones(frames, dtype=bool)
    raw_v = raw_c[v]                          # (n_valid, dim)
    feats = {
        "raw_message_sum": raw_v.sum(axis=0).astype(np.float32),
        "raw_message_max": raw_v.max(axis=0).astype(np.float32),
        "raw_message_mean": raw_v.mean(axis=0).astype(np.float32),
        "read": read.mean(axis=0).astype(np.float32),
        "injection": inj.mean(axis=0).astype(np.float32),
    }
    return feats


def _probe_one(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, float, float, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=int(seed),
        stratify=y if len(set(y.tolist())) > 1 else None,
    )
    clf = Ridge(alpha=1.0, random_state=int(seed)).fit(x_train, y_train)
    pred_f = clf.predict(x_test)
    pred = np.rint(pred_f).astype(int)
    base_pred = int(round(float(np.mean(y_train))))
    return (
        float(accuracy_score(y_test, pred)),
        float(mean_absolute_error(y_test, pred_f)),
        float(accuracy_score(y_test, np.full_like(y_test, base_pred))),
        float(mean_absolute_error(y_test, np.full_like(y_test, base_pred, dtype=float))),
    )


def probe(x: np.ndarray, y: np.ndarray, seeds: List[int]) -> Dict[str, float]:
    accs, maes, bacc, bmae = [], [], [], []
    for s in seeds:
        a, m, ba, bm = _probe_one(x, y, s)
        accs.append(a); maes.append(m); bacc.append(ba); bmae.append(bm)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "mae": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "baseline_accuracy": float(np.mean(bacc)),
        "baseline_mae": float(np.mean(bmae)),
    }


def collect_run(run_dir: Path) -> Dict[Tuple[str, int], List[Tuple[Dict[str, np.ndarray], int]]]:
    diag = run_dir / "diagnostics"
    out: Dict[Tuple[str, int], List[Tuple[Dict[str, np.ndarray], int]]] = defaultdict(list)
    npz_files = sorted(diag.glob("*.npz"))
    for f in npz_files:
        parsed = parse_npz_name(f.name)
        if parsed is None:
            continue
        _key, gold, layer = parsed
        feats = pooled_features(f)
        for feat_name, vec in feats.items():
            out[(feat_name, layer)].append((vec, gold))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc linear probe: distinct-room count from pooled messages.")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="run dirs containing diagnostics/*.npz")
    ap.add_argument("--labels", nargs="+", default=None, help="optional label per run dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated seeds; probe averages over splits")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_rooms_visited_messages")
    args = ap.parse_args()

    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        print("labels must match run-dirs", file=sys.stderr)
        return 2
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[str] = ["run,feature,layer,n,probe_acc,probe_acc_std,probe_mae,probe_mae_std,baseline_acc,baseline_mae,lift_acc"]
    lines: List[str] = []
    for label, rd in zip(labels, args.run_dirs):
        rd = Path(rd)
        grouped = collect_run(rd)
        if not grouped:
            lines.append(f"[{label}] NO diagnostics npz found under {rd}/diagnostics")
            continue
        lines.append(f"\n=== {label}  ({rd}) ===")
        for (feature, layer) in sorted(grouped):
            data = grouped[(feature, layer)]
            x = np.stack([vec for vec, _ in data])
            y = np.asarray([g for _, g in data], dtype=int)
            if len(y) < 10 or len(set(y.tolist())) < 2:
                continue
            r = probe(x, y, seeds)
            lift = r["accuracy"] - r["baseline_accuracy"]
            lines.append(
                f"  {feature:18s} L{layer:<2d} n={r['n']:<4d} "
                f"acc={r['accuracy']:.3f}±{r['accuracy_std']:.3f} mae={r['mae']:.2f}±{r['mae_std']:.2f}  "
                f"(blind acc={r['baseline_accuracy']:.3f} mae={r['baseline_mae']:.2f})  lift={lift:+.3f}"
            )
            rows.append(
                f"{label},{feature},{layer},{r['n']},{r['accuracy']:.4f},{r['accuracy_std']:.4f},"
                f"{r['mae']:.4f},{r['mae_std']:.4f},{r['baseline_accuracy']:.4f},{r['baseline_mae']:.4f},{lift:.4f}"
            )
    report = "\n".join(lines) + "\n"
    print(report)
    (args.output / "probe_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (args.output / "probe_report.txt").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}/probe_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
