#!/usr/bin/env python3
"""Proper DeepSets ρ(Σ φ(x)) with multi-dim φ + nonlinear ρ — does it solve ALL perm-invariant tasks
OOD (the universality claim), in our setting? Multi-seed count-holdout on cached reps. Variants:
  ds_phiMLP_rhoLinear : φ=MLP(D->d), sum, ρ=Linear(d->1)   (extensive readout -> should extrapolate)
  ds_phiMLP_rhoMLP    : φ=MLP(D->d), sum, ρ=MLP(d->1)       (canonical DeepSets; test if ρ caps OOD)
Universality is about REPRESENTATION (in-distribution); this tests whether it also EXTRAPOLATES.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random

CACHE = PROJECT_ROOT / "outputs" / "frame_axis" / "cache"
TASKS = ["steps_in_room", "rooms_visited", "co_occupancy"]
HF, STEPS, SEEDS, D_MEM = 5, 500, 4, 128


def load(task):
    f = CACHE / f"minimal_L19_{task}.pt"
    if not f.is_file():
        return None
    o = torch.load(f, map_location="cpu", weights_only=False)
    return [(v["reps"].float(), int(v["gold"])) for v in o.values() if torch.is_tensor(v["reps"])]


def run(task, variant, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    ex = load(task); D = ex[0][0].shape[1]
    lo = [e for e in ex if e[1] < HF]; ood = [e for e in ex if e[1] >= HF]; rng.shuffle(lo)
    tr = lo[int(0.2 * len(lo)):]
    allf = torch.cat([r for r, _ in tr], 0); mu = allf.mean(0); sd = allf.std(0) + 1e-6
    nrm = lambda r: (r - mu) / sd
    phi = torch.nn.Sequential(torch.nn.Linear(D, D_MEM), torch.nn.GELU(), torch.nn.Linear(D_MEM, D_MEM), torch.nn.GELU())
    rho = (torch.nn.Linear(D_MEM, 1) if variant == "ds_phiMLP_rhoLinear"
           else torch.nn.Sequential(torch.nn.Linear(D_MEM, D_MEM), torch.nn.GELU(), torch.nn.Linear(D_MEM, 1)))
    params = list(phi.parameters()) + list(rho.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    pred = lambda r: rho(phi(nrm(r)).sum(0)).squeeze()
    for _ in range(STEPS):
        opt.zero_grad(); batch = rng.sample(tr, min(64, len(tr))); loss = 0.0
        for r, g in batch:
            loss = loss + (pred(r) - g) ** 2
        (loss / len(batch)).backward(); opt.step()
    with torch.no_grad():
        oacc = float(np.mean([round(float(pred(r))) == g for r, g in ood]))
        from collections import defaultdict
        by = defaultdict(list)
        for r, g in ood:
            by[g].append(float(pred(r)))
        pc = " ".join(f"{g}:{np.mean([round(p)==g for p in ps]):.2f}(mp{np.mean(ps):.1f})" for g, ps in sorted(by.items()))
    return oacc, pc


def main():
    rows = ["task,variant,ood_mean,ood_std,per_seed,last_per_count"]
    for task in TASKS:
        if load(task) is None:
            print(f"{task}: missing"); continue
        print(f"\n##### {task} (dist={dict(sorted(Counter(g for _,g in load(task)).items()))}) #####")
        for variant in ["ds_phiMLP_rhoLinear", "ds_phiMLP_rhoMLP"]:
            accs, lastpc = [], ""
            for s in range(SEEDS):
                a, pc = run(task, variant, s); accs.append(a); lastpc = pc
            print(f"  {variant:<22} OOD = {np.mean(accs):.3f} ± {np.std(accs):.3f}  per-seed {[round(x,2) for x in accs]}")
            print(f"       last-seed per-count: {lastpc}")
            rows.append(f"{task},{variant},{np.mean(accs):.4f},{np.std(accs):.4f},{';'.join(f'{x:.3f}' for x in accs)},{lastpc.replace(',',' ')}")
    (CACHE.parent / "readout_benchmark").mkdir(parents=True, exist_ok=True)
    (CACHE.parent / "readout_benchmark" / "deepsets_proper.csv").write_text("\n".join(rows) + "\n")
    print(f"\nwrote deepsets_proper.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
