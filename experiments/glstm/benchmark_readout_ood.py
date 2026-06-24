#!/usr/bin/env python3
"""OOD count-extrapolation benchmark of readout/aggregator methods on cached per-frame L19 reps.

Reads `cache/minimal_L19_<task>.pt` (per-frame reps + gold + per-frame labels). Count-holdout:
train on low counts, test_iid = held-out low-count examples, test_ood = held-out HIGH counts (never
trained). Compares, all with a count read DIRECTLY from the head (no LM injection):
  - classifier : 9-way softmax on mean-pooled rep  (the CE cap baseline — should fail OOD)
  - sum        : count = Σ σ(w·repᵢ)               (pure extensive; right for occurrence-count)
  - deepsets   : φ(repᵢ) -> concat(Σ,mean,max) -> linear   (the "full" aggregator)
  - logic      : per-frame p=σ(W repᵢ); soft-sum Σp + soft-OR 1-Π(1-p) -> linear  (soft-OR = distinct)
Reports per-count accuracy + mean_pred, IID and OOD, per task. CPU, fast, reproducible.
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
    p.add_argument("--holdout-from", type=int, default=5, help="counts >= this are held out as test_ood")
    p.add_argument("--iid-frac", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "readout_benchmark")
    return p.parse_args()


def load(cache_dir, L, task):
    f = cache_dir / f"minimal_L{L}_{task}.pt"
    if not f.is_file():
        return None
    obj = torch.load(f, map_location="cpu", weights_only=False)
    return [(v["reps"].float(), int(v["gold"])) for v in obj.values() if torch.is_tensor(v["reps"])]


def split(ex, holdout_from, iid_frac, rng):
    lo = [e for e in ex if e[1] < holdout_from]
    ood = [e for e in ex if e[1] >= holdout_from]
    rng.shuffle(lo)
    n_iid = int(iid_frac * len(lo))
    return lo[n_iid:], lo[:n_iid], ood  # train, test_iid, test_ood


def evaluate(pred_fn, items):
    by = defaultdict(list)
    for r, g in items:
        by[g].append(pred_fn(r))
    rows = {}
    for g, ps in sorted(by.items()):
        rows[g] = (float(np.mean([round(p) == g for p in ps])), float(np.mean(ps)), len(ps))
    overall = float(np.mean([round(p) == g for r, g in items for p in [pred_fn(r)]])) if items else 0.0
    return overall, rows


def train_model(method, train, D, d, n_steps):
    norm_mu = torch.cat([r for r, _ in train], 0).mean(0)
    norm_sd = torch.cat([r for r, _ in train], 0).std(0) + 1e-6
    nrm = lambda r: (r - norm_mu) / norm_sd
    import random
    rng = random.Random(0)

    if method == "classifier":
        W = torch.zeros(D, 9, requires_grad=True); b = torch.zeros(9, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=5e-3); ce = torch.nn.CrossEntropyLoss()
        X = torch.stack([nrm(r).mean(0) for r, _ in train]); y = torch.tensor([g for _, g in train])
        for _ in range(n_steps):
            opt.zero_grad(); ce(X @ W + b, y).backward(); opt.step()
        return lambda r: int((nrm(r).mean(0) @ W + b).argmax().detach())

    if method == "sum":
        w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
        params = [w, b]; fwd = lambda r: torch.sigmoid(nrm(r) @ w + b).sum()
    elif method == "deepsets":
        phi = torch.nn.Sequential(torch.nn.Linear(D, d), torch.nn.GELU())
        head = torch.nn.Linear(3 * d, 1)
        params = list(phi.parameters()) + list(head.parameters())
        def fwd(r):
            h = phi(nrm(r))  # [N,d]
            feat = torch.cat([h.sum(0), h.mean(0), h.max(0).values])
            return head(feat).squeeze()
    else:  # logic
        Wp = torch.nn.Linear(D, d)
        head = torch.nn.Linear(2 * d + 2, 1)
        params = list(Wp.parameters()) + list(head.parameters())
        def fwd(r):
            p = torch.sigmoid(Wp(nrm(r))).clamp(1e-4, 1 - 1e-4)  # [N,d]
            s_sum = p.sum(0)                                     # occurrence count per channel
            s_or = 1 - torch.exp(torch.log1p(-p).sum(0))         # soft-union (distinct) per channel
            feat = torch.cat([s_sum, s_or, s_sum.sum().view(1), s_or.sum().view(1)])
            return head(feat).squeeze()

    opt = torch.optim.Adam(params, lr=5e-3)
    for _ in range(n_steps):
        opt.zero_grad(); batch = rng.sample(train, min(64, len(train))); loss = 0.0
        for r, g in batch:
            loss = loss + (fwd(r) - g) ** 2
        (loss / len(batch)).backward(); opt.step()
    return lambda r: float(fwd(r).detach())


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import random
    methods = ["classifier", "sum", "deepsets", "logic"]
    summary = ["task,method,iid_acc,ood_acc,ood_per_count"]
    for task in [t for t in args.tasks.split(",") if t]:
        ex = load(args.cache_dir, args.read_layer, task)
        if ex is None:
            print(f"##### {task}: cache missing, skip #####"); continue
        D = ex[0][0].shape[1]
        tr, te_iid, te_ood = split(ex, args.holdout_from, args.iid_frac, random.Random(0))
        print(f"\n##### {task}: train={len(tr)} test_iid={len(te_iid)} test_ood={len(te_ood)} "
              f"(holdout counts>={args.holdout_from}; dist={dict(sorted(Counter(g for _,g in ex).items()))}) #####")
        print(f"  {'method':<11} {'IID':>6} {'OOD':>6}   OOD per-count acc")
        for m in methods:
            pred = train_model(m, tr, D, args.d, args.steps)
            iid_acc, _ = evaluate(pred, te_iid)
            ood_acc, ood_rows = evaluate(pred, te_ood)
            pc = " ".join(f"{g}:{a:.2f}(mp{mp:.1f})" for g, (a, mp, n) in ood_rows.items())
            print(f"  {m:<11} {iid_acc:6.3f} {ood_acc:6.3f}   {pc}")
            summary.append(f"{task},{m},{iid_acc:.4f},{ood_acc:.4f},{pc.replace(',',';')}")
    (args.output / "benchmark.csv").write_text("\n".join(summary) + "\n")
    print(f"\nwrote {args.output}/benchmark.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
