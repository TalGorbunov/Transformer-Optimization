#!/usr/bin/env python3
"""Layer sweep of message-sum decodability (Stage 1 across depth). Reads the multi-layer cache from
cache_message_sum_layersweep.py and, per layer, reports the headline metrics + a mechanism metric:

  S_all->g, S_evid->g, S_nonev->(N-g), MEAN_evid->g (g>=1), ref last-token->g   [acc + R^2]
  cos(mean_evid_dir, mean_nonev_dir)  -- if ~1, evidence and non-evidence sums share a direction, so
      S_all projects to g+(N-g)=N (constant) and g becomes unreadable. The mechanistic explanation of
      the S_all interference collapse.

Answers "what is R^2 at L27?" and "where across depth is the count decodable / interference worst?".
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from experiments.glstm.probe_message_sum_decodability import Example, fit_ridge, agg_sum, agg_mean  # noqa: E402


def load_layer(cache: dict, layer: int, seq_len: int) -> List[Example]:
    out = []
    for v in cache.values():
        if int(v.get("seq_len", -1)) != seq_len:
            continue
        fl = v.get("frame_labels")
        rbl = v.get("reps_by_layer")
        if fl is None or rbl is None or layer not in rbl:
            continue
        reps = rbl[layer].float().numpy()
        try:
            labels = np.asarray([int(x) for x in fl], dtype=int)
        except (TypeError, ValueError):
            continue
        if labels.shape[0] != reps.shape[0] or not set(labels.tolist()) <= {0, 1}:
            continue
        q = v["query_by_layer"][layer].float().numpy()
        out.append(Example(reps, labels, int(v["gold"]), q))
    return out


def mean_dir_cosine(exs: List[Example]) -> float:
    """cos between the mean (unit) evidence-frame rep and mean non-evidence-frame rep."""
    ev, nv = [], []
    for e in exs:
        ev.extend(e.reps[e.labels == 1])
        nv.extend(e.reps[e.labels == 0])
    if not ev or not nv:
        return float("nan")
    u = np.mean(np.stack(ev), 0); v = np.mean(np.stack(nv), 0)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True, help="LABEL:PATH (layersweep .pt)")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "message_sum")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    run_dir = args.output / (time.strftime("%Y%m%d_%H%M%S") + "_layersweep")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def emit(m): print(m, flush=True); lines.append(m)

    csv = ["label,layer,metric,acc,r2,baseline_acc"]
    cos_csv = ["label,layer,cos_evid_nonev"]
    per_label_layer: Dict[str, Dict[str, Dict[int, float]]] = {}

    for spec in args.caches:
        label, _, path = spec.partition(":")
        cache = torch.load(Path(path), map_location="cpu")
        layers = sorted(next(iter(cache.values()))["reps_by_layer"].keys())
        emit(f"\n===== {label}: {len(cache)} examples; layers {layers} =====")
        store = {m: {} for m in ("S_all", "S_evid", "MEAN_evid", "ref_lasttok", "cos")}
        for L in layers:
            exs = load_layer(cache, L, args.seq_len)
            if not exs:
                continue
            gold = np.asarray([e.gold for e in exs])
            S_all = np.stack([agg_sum(e.reps) for e in exs])
            S_evid = np.stack([agg_sum(e.reps, np.where(e.labels == 1)[0]) for e in exs])
            S_nonev = np.stack([agg_sum(e.reps, np.where(e.labels == 0)[0]) for e in exs])
            Xq = np.stack([e.query for e in exs])
            pos = np.where(gold >= 1)[0]
            mean_ev = np.stack([agg_mean(exs[i].reps, np.where(exs[i].labels == 1)[0]) for i in pos])
            metrics = {
                "S_all": fit_ridge(S_all, gold, seeds),
                "S_evid": fit_ridge(S_evid, gold, seeds),
                "S_nonev": fit_ridge(S_nonev, exs[0].reps.shape[0] - gold, seeds),
                "MEAN_evid": fit_ridge(mean_ev, gold[pos], seeds),
                "ref_lasttok": fit_ridge(Xq, gold, seeds),
            }
            cos = mean_dir_cosine(exs)
            for m, r in metrics.items():
                csv.append(f"{label},{L},{m},{r['acc']:.4f},{r['r2']:.4f},{r['baseline_acc']:.4f}")
            cos_csv.append(f"{label},{L},{cos:.4f}")
            for k in ("S_all", "S_evid", "MEAN_evid", "ref_lasttok"):
                store[k][L] = metrics[k]["r2"]
            store["cos"][L] = cos
            emit(f"  L{L:<2d}: S_all acc={metrics['S_all']['acc']:.3f}/r2={metrics['S_all']['r2']:.3f}  "
                 f"S_evid acc={metrics['S_evid']['acc']:.3f}  MEAN_evid acc={metrics['MEAN_evid']['acc']:.3f}  "
                 f"ref_lasttok acc={metrics['ref_lasttok']['acc']:.3f}/r2={metrics['ref_lasttok']['r2']:.3f}  "
                 f"cos(evid,nonev)={cos:.3f}")
        per_label_layer[label] = store

    (run_dir / "layersweep.csv").write_text("\n".join(csv) + "\n")
    (run_dir / "cos_evid_nonev.csv").write_text("\n".join(cos_csv) + "\n")
    (run_dir / "README.md").write_text("# message-sum layer sweep\n\n```\n" + "\n".join(lines) + "\n```\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for label, store in per_label_layer.items():
            Ls = sorted(store["S_all"].keys())
            axes[0].plot(Ls, [store["S_all"][L] for L in Ls], "o-", label=f"{label} S_all R²")
            axes[0].plot(Ls, [store["ref_lasttok"][L] for L in Ls], "s--", label=f"{label} last-tok R²")
            axes[1].plot(Ls, [store["cos"][L] for L in Ls], "o-", label=f"{label} cos(evid,nonev)")
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("R² (decode gold)")
        axes[0].set_title("Count decodability across depth"); axes[0].legend(fontsize=8)
        axes[1].set_xlabel("layer"); axes[1].set_ylabel("cos(mean evid dir, mean nonev dir)")
        axes[1].set_title("Evidence/non-evidence direction overlap (interference mechanism)")
        axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(run_dir / "layersweep.png", dpi=120); plt.close(fig)
    except Exception as exc:
        emit(f"[warn] plot failed: {exc}")
    print(f"\nwrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
