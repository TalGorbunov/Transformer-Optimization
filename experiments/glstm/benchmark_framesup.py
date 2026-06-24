#!/usr/bin/env python3
"""Per-frame-SUPERVISED readout benchmark (the 'fix' lever) on cached per-frame L19 reps.

If the count-only additive/logic readout underperforms OOD, the likely cause is a weak per-frame
detector learned from count-only MSE. MMRED is self-generated, so we HAVE per-frame labels (cached) —
use them directly:
  - occurrence tasks (steps, co-occ): per-frame binary detector (BCE on frame evidence), count = Σ σ.
  - distinct task (rooms): per-frame ROOM classifier (CE on room id), distinct = Σ_room[1-Π(1-p_room)]
    (soft-OR over frames per room) — the structurally correct extensive operator for distinct-count.

Count-holdout: the per-frame detector is trained ONLY on frames from low-count (<holdout) examples, so
held-out HIGH counts are never seen; the sum / soft-OR then extrapolates by construction. CPU, fast.
"""
from __future__ import annotations
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch

torch.manual_seed(0); np.random.seed(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "cache")
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--tasks", default="steps_in_room,rooms_visited,co_occupancy")
    p.add_argument("--holdout-from", type=int, default=5)
    p.add_argument("--iid-frac", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "readout_benchmark")
    return p.parse_args()


def load(cache_dir, L, task):
    f = cache_dir / f"minimal_L{L}_{task}.pt"
    if not f.is_file():
        return None
    obj = torch.load(f, map_location="cpu", weights_only=False)
    out = []
    for v in obj.values():
        if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None:
            out.append((v["reps"].float(), int(v["gold"]), v["frame_labels"]))
    return out


def split(ex, holdout_from, iid_frac, rng):
    lo = [e for e in ex if e[1] < holdout_from]; ood = [e for e in ex if e[1] >= holdout_from]
    rng.shuffle(lo); n = int(iid_frac * len(lo))
    return lo[n:], lo[:n], ood


def per_count(items, pred_fn):
    by = defaultdict(list)
    for r, g, _ in items:
        by[g].append(pred_fn(r))
    return {g: (float(np.mean([round(p) == g for p in ps])), float(np.mean(ps)), len(ps)) for g, ps in sorted(by.items())}


def overall(items, pred_fn):
    return float(np.mean([round(pred_fn(r)) == g for r, g, _ in items])) if items else 0.0


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import random
    summary = ["task,method,iid_acc,ood_acc,ood_per_count"]
    for task in [t for t in args.tasks.split(",") if t]:
        ex = load(args.cache_dir, args.read_layer, task)
        if ex is None:
            print(f"##### {task}: cache/labels missing, skip #####"); continue
        D = ex[0][0].shape[1]
        tr, te_iid, te_ood = split(ex, args.holdout_from, args.iid_frac, random.Random(0))
        allf = torch.cat([r for r, _, _ in tr], 0); mu = allf.mean(0); sd = allf.std(0) + 1e-6
        nrm = lambda r: (r - mu) / sd
        print(f"\n##### {task} (frame-sup): train={len(tr)} iid={len(te_iid)} ood={len(te_ood)} "
              f"dist={dict(sorted(Counter(g for _,g,_ in ex).items()))} #####")

        rooms_task = isinstance(tr[0][2][0], str)  # rooms: categorical room labels
        if not rooms_task:
            # per-frame binary detector via BCE -> count = Σ σ
            Xf = torch.cat([nrm(r) for r, _, _ in tr], 0)
            yf = torch.tensor([float(l) for _, _, lab in tr for l in lab])
            w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=5e-3); bce = torch.nn.BCEWithLogitsLoss()
            for _ in range(args.steps):
                opt.zero_grad(); bce(Xf @ w + b, yf).backward(); opt.step()
            pred = lambda r: float(torch.sigmoid(nrm(r) @ w + b).sum().detach())
            method = "sum_framesup"
        else:
            rooms = sorted({l for _, _, lab in tr for l in lab}); ridx = {r: i for i, r in enumerate(rooms)}
            K = len(rooms)
            Xf = torch.cat([nrm(r) for r, _, _ in tr], 0)
            yf = torch.tensor([ridx[l] for _, _, lab in tr for l in lab])
            W = torch.zeros(D, K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
            opt = torch.optim.Adam([W, b], lr=5e-3); ce = torch.nn.CrossEntropyLoss()
            for _ in range(args.steps):
                opt.zero_grad(); ce(Xf @ W + b, yf).backward(); opt.step()
            def pred(r):  # distinct = Σ_room [1 - Π_frames (1 - p(room|frame))]
                p = torch.softmax(nrm(r) @ W + b, dim=-1).clamp(1e-4, 1 - 1e-4)  # [N,K]
                soft_or = 1 - torch.exp(torch.log1p(-p).sum(0))                  # [K]
                return float(soft_or.sum().detach())
            method = "softOR_framesup"

        iid = overall(te_iid, pred); ood = overall(te_ood, pred); pc = per_count(te_ood, pred)
        pcs = " ".join(f"{g}:{a:.2f}(mp{mp:.1f})" for g, (a, mp, n) in pc.items())
        print(f"  {method:<16} IID={iid:.3f} OOD={ood:.3f}   OOD per-count: {pcs}")
        summary.append(f"{task},{method},{iid:.4f},{ood:.4f},{pcs.replace(',',';')}")
    (args.output / "benchmark_framesup.csv").write_text("\n".join(summary) + "\n")
    print(f"\nwrote {args.output}/benchmark_framesup.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
