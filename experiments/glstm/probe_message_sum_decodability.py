#!/usr/bin/env python3
"""Stage 1: is the COUNT linearly decodable from SUMMED per-frame messages, and does it survive
the aggregation? Pure CPU post-processing on cached L19 per-frame reps (cache_minimal_frame_reps.py,
JOINT pass). No model, no GPU.

We construct the aggregate ourselves (S = Sigma m_k over cached query-conditioned L19 per-frame reps),
so this is independent of the model's attention routing (that is Stage 2). Two experiments:

EXP 1 - Superposition curve. For prefix length j=1..8, decode the running count over the first j
  frames from SUM_j = Sigma_{k<=j} m_k. Plot acc / MAE / R^2 vs j. Does packing more frames into one
  vector degrade linear count decodability? (NOTE: at FIXED j every example sums exactly j frames, so
  SUM_j and MEAN_j = SUM_j/j are linearly equivalent -- a linear probe cannot distinguish them. The
  normalization question therefore cannot be asked here; it is asked in EXP 2b where the divisor varies.)

EXP 2 - Evidence/non-evidence interference + the normalization test (seq_len 8 only).
  2a U-shape: decode gold count g from S_all = Sigma all 8 frames, S_evid = Sigma evidence frames,
     S_nonev = Sigma non-evidence frames. Report overall decode quality AND per-count accuracy. The
     hypothesis: S_all decode-error is U-shaped in g (clean at g=0,8 -> no mixing; worst at g=4),
     while S_evid / S_nonev stay flat+high -> mixing evidence with non-evidence in one sum is the cut.
  2b Normalization test: decode g from SUM of evidence frames (magnitude ~ g) vs MEAN of evidence
     frames (divide by g -> magnitude normalized away). Sum should work, mean should fail. This is the
     model-agnostic proxy for "the softmax denominator squashes the extensive count."

Reference line: decode gold from the real last-token L19 rep (query_rep) -- what the model actually holds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# ------------------------------- data loading -------------------------------
class Example:
    __slots__ = ("reps", "labels", "gold", "query")

    def __init__(self, reps: np.ndarray, labels: np.ndarray, gold: int, query: np.ndarray):
        self.reps = reps      # [N, H] float32
        self.labels = labels  # [N] int (0/1 per-frame evidence)
        self.gold = gold      # int
        self.query = query    # [H] float32 (last-token L19 rep)


def load_cache(path: Path, seq_len: int) -> List[Example]:
    cache = torch.load(path, map_location="cpu")
    out: List[Example] = []
    n_skip = 0
    for name, v in cache.items():
        if int(v.get("seq_len", -1)) != seq_len:
            n_skip += 1
            continue
        fl = v.get("frame_labels")
        reps = v.get("reps")
        if fl is None or reps is None:
            n_skip += 1
            continue
        reps = reps.float().numpy()
        # frame_labels for steps_in_room is a per-frame binary list (int)
        try:
            labels = np.asarray([int(x) for x in fl], dtype=int)
        except (TypeError, ValueError):
            n_skip += 1
            continue
        if labels.ndim != 1 or labels.shape[0] != reps.shape[0] or not set(labels.tolist()) <= {0, 1}:
            n_skip += 1
            continue
        query = v["query_rep"].float().numpy()
        out.append(Example(reps, labels, int(v["gold"]), query))
    print(f"  loaded {len(out)} examples (skipped {n_skip}) from {path.name}", flush=True)
    return out


# ------------------------------- probing core -------------------------------
def fit_ridge(X: np.ndarray, y: np.ndarray, seeds: List[int], alpha: float = 1.0) -> Dict[str, float]:
    """Standardize -> Ridge -> round-to-int acc + MAE + R^2, averaged over CV seeds.
    Baseline = always predict the train-set mean count (count-blind)."""
    accs, maes, r2s, b_acc, b_mae = [], [], [], [], []
    n_classes = len(set(y.tolist()))
    for s in seeds:
        strat = y if n_classes > 1 and np.bincount(y).min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=s, stratify=strat)
        sc = StandardScaler().fit(Xtr)
        clf = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
        pf = clf.predict(sc.transform(Xte))
        pred = np.rint(pf).astype(int)
        accs.append(accuracy_score(yte, pred))
        maes.append(mean_absolute_error(yte, pf))
        r2s.append(r2_score(yte, pf) if len(set(yte.tolist())) > 1 else 0.0)
        bp = int(round(float(np.mean(ytr))))
        b_acc.append(accuracy_score(yte, np.full_like(yte, bp)))
        b_mae.append(mean_absolute_error(yte, np.full_like(yte, bp, dtype=float)))
    return {
        "n": int(len(y)), "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "mae": float(np.mean(maes)), "r2": float(np.mean(r2s)),
        "baseline_acc": float(np.mean(b_acc)), "baseline_mae": float(np.mean(b_mae)),
    }


def fit_ridge_percount(X: np.ndarray, y: np.ndarray, seeds: List[int],
                       alpha: float = 1.0) -> Dict[int, Dict[str, float]]:
    """Per-true-count accuracy/MAE on held-out (one probe across all counts) -> the U-shape curve."""
    per: Dict[int, List[Tuple[int, float]]] = {}  # g -> list of (correct, abs_err)
    for s in seeds:
        strat = y if np.bincount(y).min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=s, stratify=strat)
        sc = StandardScaler().fit(Xtr)
        clf = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
        pf = clf.predict(sc.transform(Xte))
        pred = np.rint(pf).astype(int)
        for g, p, pc in zip(yte, pf, pred):
            per.setdefault(int(g), []).append((int(pc == g), abs(float(p) - g)))
    out: Dict[int, Dict[str, float]] = {}
    for g, lst in sorted(per.items()):
        cor = np.mean([c for c, _ in lst])
        mae = np.mean([e for _, e in lst])
        out[g] = {"n": len(lst), "acc": float(cor), "mae": float(mae)}
    return out


# ------------------------------- aggregations -------------------------------
def agg_sum(reps: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
    r = reps if idx is None else reps[idx]
    if r.shape[0] == 0:
        return np.zeros(reps.shape[1], dtype=np.float32)
    return r.sum(0).astype(np.float32)


def agg_mean(reps: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
    r = reps if idx is None else reps[idx]
    if r.shape[0] == 0:
        return np.zeros(reps.shape[1], dtype=np.float32)
    return r.mean(0).astype(np.float32)


# ------------------------------- experiments -------------------------------
def experiment1(exs: List[Example], seeds: List[int]) -> List[dict]:
    """Prefix superposition curve: decode running count over first j frames from SUM_j."""
    rows = []
    N = exs[0].reps.shape[0]
    for j in range(1, N + 1):
        X = np.stack([agg_sum(e.reps[:j]) for e in exs])
        y = np.asarray([int(e.labels[:j].sum()) for e in exs])
        r = fit_ridge(X, y, seeds)
        r.update({"j": j, "lift_acc": r["acc"] - r["baseline_acc"]})
        rows.append(r)
    return rows


def experiment2(exs: List[Example], seeds: List[int]) -> dict:
    """Evidence/non-evidence interference (2a) + normalization test (2b), seq_len 8."""
    gold = np.asarray([e.gold for e in exs])

    S_all = np.stack([agg_sum(e.reps) for e in exs])
    S_evid = np.stack([agg_sum(e.reps, np.where(e.labels == 1)[0]) for e in exs])
    S_nonev = np.stack([agg_sum(e.reps, np.where(e.labels == 0)[0]) for e in exs])

    overall = {
        "S_all->g": fit_ridge(S_all, gold, seeds),
        "S_evid->g": fit_ridge(S_evid, gold, seeds),
        "S_nonev->(N-g)": fit_ridge(S_nonev, (exs[0].reps.shape[0] - gold), seeds),
    }
    # 2b normalization test: decode g from SUM vs MEAN of EVIDENCE frames (divisor g varies).
    # restrict to g>=1 (mean undefined for g=0) and use the SAME example set for a fair sum-vs-mean.
    pos = np.where(gold >= 1)[0]
    exs_pos = [exs[i] for i in pos]
    gpos = gold[pos]
    sum_ev = np.stack([agg_sum(e.reps, np.where(e.labels == 1)[0]) for e in exs_pos])
    mean_ev = np.stack([agg_mean(e.reps, np.where(e.labels == 1)[0]) for e in exs_pos])
    norm_test = {
        "SUM_evid->g (g>=1)": fit_ridge(sum_ev, gpos, seeds),
        "MEAN_evid->g (g>=1)": fit_ridge(mean_ev, gpos, seeds),
    }
    # per-count U-shape (one probe across all g, eval per true g)
    percount = {
        "S_all": fit_ridge_percount(S_all, gold, seeds),
        "S_evid": fit_ridge_percount(S_evid, gold, seeds),
    }
    return {"overall": overall, "norm_test": norm_test, "percount": percount}


# ------------------------------- plotting -------------------------------
def plot_exp1(results: Dict[str, List[dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, rows in results.items():
        js = [r["j"] for r in rows]
        axes[0].plot(js, [r["acc"] for r in rows], "o-", label=f"{label} probe")
        axes[0].plot(js, [r["baseline_acc"] for r in rows], "--", alpha=0.4,
                     label=f"{label} blind")
        axes[1].plot(js, [r["mae"] for r in rows], "o-", label=label)
    axes[0].set_xlabel("prefix length j (frames summed)"); axes[0].set_ylabel("count acc")
    axes[0].set_title("EXP1: running-count decodability vs #summed frames"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("prefix length j"); axes[1].set_ylabel("MAE"); axes[1].set_title("EXP1: MAE vs j")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_exp2_ushape(results: Dict[str, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4), squeeze=False)
    for ax, (label, res) in zip(axes[0], results.items()):
        for feat in ("S_all", "S_evid"):
            pc = res["percount"][feat]
            gs = sorted(pc.keys())
            ax.plot(gs, [pc[g]["acc"] for g in gs], "o-", label=feat)
        ax.set_xlabel("true gold count g"); ax.set_ylabel("per-count acc")
        ax.set_title(f"{label}: U-shape (S_all) vs flat (S_evid)"); ax.legend(fontsize=8)
        ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ------------------------------- main -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 message-sum decodability probes (CPU).")
    p.add_argument("--caches", nargs="+", required=True,
                   help="LABEL:PATH pairs, e.g. crowded:outputs/.../cache_crowded/minimal_L19_steps_in_room.pt")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "message_sum")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import time
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def emit(m: str) -> None:
        print(m, flush=True); lines.append(m)

    datasets: Dict[str, List[Example]] = {}
    for spec in args.caches:
        label, _, path = spec.partition(":")
        exs = load_cache(Path(path), args.seq_len)
        if exs:
            datasets[label] = exs

    exp1_all: Dict[str, List[dict]] = {}
    exp2_all: Dict[str, dict] = {}
    csv1 = ["label,j,n,acc,acc_std,mae,r2,baseline_acc,lift_acc"]
    csv2 = ["label,feature,n,acc,acc_std,mae,r2,baseline_acc"]
    csvpc = ["label,feature,g,n,acc,mae"]

    for label, exs in datasets.items():
        gd = {}
        for e in exs:
            gd[e.gold] = gd.get(e.gold, 0) + 1
        emit(f"\n===== {label}: {len(exs)} examples; gold dist {dict(sorted(gd.items()))} =====")

        # reference: decode gold from the real last-token L19 rep
        Xq = np.stack([e.query for e in exs]); yq = np.asarray([e.gold for e in exs])
        ref = fit_ridge(Xq, yq, seeds)
        emit(f"  [ref] last-token L19 rep -> gold: acc={ref['acc']:.3f} mae={ref['mae']:.2f} "
             f"r2={ref['r2']:.3f} (blind acc={ref['baseline_acc']:.3f})")

        emit("  --- EXP1: running-count decodability vs prefix length j ---")
        rows = experiment1(exs, seeds)
        exp1_all[label] = rows
        for r in rows:
            emit(f"    j={r['j']}: acc={r['acc']:.3f}+/-{r['acc_std']:.3f} mae={r['mae']:.2f} "
                 f"r2={r['r2']:.3f}  (blind {r['baseline_acc']:.3f}, lift {r['lift_acc']:+.3f})")
            csv1.append(f"{label},{r['j']},{r['n']},{r['acc']:.4f},{r['acc_std']:.4f},{r['mae']:.4f},"
                        f"{r['r2']:.4f},{r['baseline_acc']:.4f},{r['lift_acc']:.4f}")

        emit("  --- EXP2a: evidence/non-evidence interference (seq_len 8) ---")
        res = experiment2(exs, seeds)
        exp2_all[label] = res
        for feat, r in res["overall"].items():
            emit(f"    {feat:20s}: acc={r['acc']:.3f} mae={r['mae']:.2f} r2={r['r2']:.3f} "
                 f"(blind {r['baseline_acc']:.3f})")
            csv2.append(f"{label},{feat},{r['n']},{r['acc']:.4f},{r['acc_std']:.4f},{r['mae']:.4f},"
                        f"{r['r2']:.4f},{r['baseline_acc']:.4f}")
        emit("  --- EXP2b: normalization test (SUM vs MEAN of evidence frames) ---")
        for feat, r in res["norm_test"].items():
            emit(f"    {feat:22s}: acc={r['acc']:.3f} mae={r['mae']:.2f} r2={r['r2']:.3f} "
                 f"(blind {r['baseline_acc']:.3f})")
            csv2.append(f"{label},{feat},{r['n']},{r['acc']:.4f},{r['acc_std']:.4f},{r['mae']:.4f},"
                        f"{r['r2']:.4f},{r['baseline_acc']:.4f}")
        emit("  --- EXP2a per-count (U-shape) ---")
        for feat, pc in res["percount"].items():
            cells = "  ".join(f"g{g}:{pc[g]['acc']:.2f}" for g in sorted(pc))
            emit(f"    {feat}: {cells}")
            for g in sorted(pc):
                csvpc.append(f"{label},{feat},{g},{pc[g]['n']},{pc[g]['acc']:.4f},{pc[g]['mae']:.4f}")

    # write outputs
    (run_dir / "exp1_prefix_curve.csv").write_text("\n".join(csv1) + "\n")
    (run_dir / "exp2_decode.csv").write_text("\n".join(csv2) + "\n")
    (run_dir / "exp2_percount.csv").write_text("\n".join(csvpc) + "\n")
    (run_dir / "README.md").write_text("# Stage 1 message-sum decodability\n\n```\n" + "\n".join(lines) + "\n```\n")
    try:
        if exp1_all:
            plot_exp1(exp1_all, run_dir / "exp1_prefix_curve.png")
        if exp2_all:
            plot_exp2_ushape(exp2_all, run_dir / "exp2_ushape.png")
    except Exception as exc:  # plotting must never block the numbers
        emit(f"[warn] plotting failed: {exc}")
    print(f"\nwrote {run_dir}/ (exp1_prefix_curve.csv, exp2_decode.csv, exp2_percount.csv, README.md, plots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
