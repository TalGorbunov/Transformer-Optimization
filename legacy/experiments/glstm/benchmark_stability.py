#!/usr/bin/env python3
"""Stability check: is the OOD-extrapolation result seed-robust? Quantifies the variance that the
single-run benchmark hid. For each task, N seeds of:
  - sum_countonly : Σσ(w·rep), trained MSE-to-count   (suspected HIGH variance)
  - sum_framesup  : per-frame BCE detector then Σσ      (occurrence; suspected STABLE)
  - softOR_framesup (rooms): per-frame room CE then soft-OR  (distinct; suspected STABLE)
Reports OOD acc mean±std across seeds. CPU.
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "cache")
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--tasks", default="steps_in_room,rooms_visited,co_occupancy")
    p.add_argument("--holdout-from", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "readout_benchmark")
    return p.parse_args()


def load(cache_dir, L, task):
    f = cache_dir / f"minimal_L{L}_{task}.pt"
    if not f.is_file():
        return None
    obj = torch.load(f, map_location="cpu", weights_only=False)
    return [(v["reps"].float(), int(v["gold"]), v.get("frame_labels")) for v in obj.values() if torch.is_tensor(v["reps"])]


def split(ex, hf, rng):
    lo = [e for e in ex if e[1] < hf]; ood = [e for e in ex if e[1] >= hf]; rng.shuffle(lo)
    n = int(0.2 * len(lo)); return lo[n:], ood


def ov(items, fn):
    return float(np.mean([round(fn(r)) == g for r, g, _ in items])) if items else 0.0


def run_seed(task, ex, hf, seed, steps):
    torch.manual_seed(seed); rng = random.Random(seed)
    tr, ood = split(ex, hf, rng); D = ex[0][0].shape[1]
    allf = torch.cat([r for r, _, _ in tr], 0); mu = allf.mean(0); sd = allf.std(0) + 1e-6
    nrm = lambda r: (r - mu) / sd
    res = {}
    # count-only sum
    w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=5e-3)
    for _ in range(steps):
        opt.zero_grad(); batch = rng.sample(tr, min(64, len(tr))); loss = 0.0
        for r, g, _ in batch:
            loss = loss + (torch.sigmoid(nrm(r) @ w + b).sum() - g) ** 2
        (loss / len(batch)).backward(); opt.step()
    res["sum_countonly"] = ov(ood, lambda r: float(torch.sigmoid(nrm(r) @ w + b).sum().detach()))
    # frame-sup
    rooms_task = tr[0][2] is not None and isinstance(tr[0][2][0], str)
    if not rooms_task and tr[0][2] is not None:
        X = torch.cat([nrm(r) for r, _, _ in tr], 0); y = torch.tensor([float(l) for _, _, lab in tr for l in lab])
        w2 = torch.zeros(D, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w2, b2], lr=5e-3); bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(steps):
            opt.zero_grad(); bce(X @ w2 + b2, y).backward(); opt.step()
        res["sum_framesup"] = ov(ood, lambda r: float(torch.sigmoid(nrm(r) @ w2 + b2).sum().detach()))
    elif rooms_task:
        rooms = sorted({l for _, _, lab in tr for l in lab}); ridx = {r: i for i, r in enumerate(rooms)}; K = len(rooms)
        X = torch.cat([nrm(r) for r, _, _ in tr], 0); y = torch.tensor([ridx[l] for _, _, lab in tr for l in lab])
        W = torch.zeros(D, K, requires_grad=True); b2 = torch.zeros(K, requires_grad=True)
        opt = torch.optim.Adam([W, b2], lr=5e-3); ce = torch.nn.CrossEntropyLoss()
        for _ in range(steps):
            opt.zero_grad(); ce(X @ W + b2, y).backward(); opt.step()
        def fn(r):
            p = torch.softmax(nrm(r) @ W + b2, -1).clamp(1e-4, 1 - 1e-4)
            return float((1 - torch.exp(torch.log1p(-p).sum(0))).sum().detach())
        res["softOR_framesup"] = ov(ood, fn)
    return res


def main():
    args = parse_args()
    rows = ["task,method,ood_mean,ood_std,per_seed"]
    for task in [t for t in args.tasks.split(",") if t]:
        ex = load(args.cache_dir, args.read_layer, task)
        if ex is None:
            print(f"{task}: missing"); continue
        agg = {}
        for s in range(args.seeds):
            for k, v in run_seed(task, ex, args.holdout_from, s, args.steps).items():
                agg.setdefault(k, []).append(v)
        print(f"\n##### {task} (dist={dict(sorted(Counter(g for _,g,_ in ex).items()))}) #####")
        for k, vs in agg.items():
            print(f"  {k:<18} OOD = {np.mean(vs):.3f} ± {np.std(vs):.3f}   per-seed {[round(x,2) for x in vs]}")
            rows.append(f"{task},{k},{np.mean(vs):.4f},{np.std(vs):.4f},{';'.join(f'{x:.3f}' for x in vs)}")
    (args.output).mkdir(parents=True, exist_ok=True)
    (args.output / "stability.csv").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {args.output}/stability.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
