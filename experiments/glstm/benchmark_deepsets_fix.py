#!/usr/bin/env python3
"""Diagnose + fix why DeepSets underperforms `sum` on OOD count-extrapolation.

Hypothesis: deepsets = concat(Σφ, meanφ, maxφ) -> linear. mean is count-INVARIANT, max SATURATES;
only Σφ is count-extensive. Trained on low counts, the readout leans on the non-extensive channels and
fails to extrapolate. Variants tested (count-holdout, cached reps, CPU):
  deepsets_full   : [sum,mean,max] -> linear            (the underperformer)
  deepsets_sumonly: [sum] -> linear                     (drop non-extensive channels = the fix)
  deepsets_meanmax: [mean,max] -> linear                (control: should NOT extrapolate)
  deepsets_L1     : [sum,mean,max] but L1 penalty on the mean/max readout weights (in-place fix)
"""
from __future__ import annotations
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
torch.manual_seed(0); np.random.seed(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "cache")
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--tasks", default="steps_in_room,rooms_visited,co_occupancy")
    p.add_argument("--holdout-from", type=int, default=5)
    p.add_argument("--iid-frac", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--l1", type=float, default=0.05)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "readout_benchmark")
    return p.parse_args()


def load(cache_dir, L, task):
    f = cache_dir / f"minimal_L{L}_{task}.pt"
    if not f.is_file():
        return None
    obj = torch.load(f, map_location="cpu", weights_only=False)
    return [(v["reps"].float(), int(v["gold"])) for v in obj.values() if torch.is_tensor(v["reps"])]


def split(ex, hf, iid_frac, rng):
    lo = [e for e in ex if e[1] < hf]; ood = [e for e in ex if e[1] >= hf]
    rng.shuffle(lo); n = int(iid_frac * len(lo)); return lo[n:], lo[:n], ood


def ov(items, fn):
    return float(np.mean([round(fn(r)) == g for r, g in items])) if items else 0.0


def pc(items, fn):
    by = defaultdict(list)
    for r, g in items:
        by[g].append(fn(r))
    return " ".join(f"{g}:{np.mean([round(p)==g for p in ps]):.2f}(mp{np.mean(ps):.1f})" for g, ps in sorted(by.items()))


def main():
    args = parse_args()
    summary = ["task,variant,iid,ood,ood_per_count"]
    for task in [t for t in args.tasks.split(",") if t]:
        ex = load(args.cache_dir, args.read_layer, task)
        if ex is None:
            print(f"##### {task}: missing #####"); continue
        D = ex[0][0].shape[1]
        tr, iid, ood = split(ex, args.holdout_from, args.iid_frac, random.Random(0))
        allf = torch.cat([r for r, _ in tr], 0); mu = allf.mean(0); sd = allf.std(0) + 1e-6
        nrm = lambda r: (r - mu) / sd
        print(f"\n##### {task}: train={len(tr)} iid={len(iid)} ood={len(ood)} "
              f"dist={dict(sorted(Counter(g for _,g in ex).items()))} #####")
        for variant in ["deepsets_full", "deepsets_sumonly", "deepsets_meanmax", "deepsets_L1"]:
            phi = torch.nn.Sequential(torch.nn.Linear(D, args.d), torch.nn.GELU())
            nin = {"deepsets_full": 3, "deepsets_sumonly": 1, "deepsets_meanmax": 2, "deepsets_L1": 3}[variant]
            head = torch.nn.Linear(nin * args.d, 1)
            params = list(phi.parameters()) + list(head.parameters())
            def feat(r):
                h = phi(nrm(r))
                s, m, x = h.sum(0), h.mean(0), h.max(0).values
                if variant == "deepsets_sumonly":
                    return s
                if variant == "deepsets_meanmax":
                    return torch.cat([m, x])
                return torch.cat([s, m, x])
            opt = torch.optim.Adam(params, lr=5e-3)
            for _ in range(args.steps):
                opt.zero_grad(); batch = random.sample(tr, min(64, len(tr))); loss = 0.0
                for r, g in batch:
                    loss = loss + (head(feat(r)).squeeze() - g) ** 2
                loss = loss / len(batch)
                if variant == "deepsets_L1":  # push readout off the non-extensive (mean/max) channels
                    loss = loss + args.l1 * head.weight[:, args.d:].abs().sum()
                loss.backward(); opt.step()
            fn = lambda r: float(head(feat(r)).squeeze().detach())
            iacc, oacc = ov(iid, fn), ov(ood, fn)
            print(f"  {variant:<18} IID={iacc:.3f} OOD={oacc:.3f}   {pc(ood, fn)}")
            summary.append(f"{task},{variant},{iacc:.4f},{oacc:.4f},{pc(ood,fn).replace(',',';')}")
    (args.output).mkdir(parents=True, exist_ok=True)
    (args.output / "deepsets_fix.csv").write_text("\n".join(summary) + "\n")
    print(f"\nwrote {args.output}/deepsets_fix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
