#!/usr/bin/env python3
"""Is canonical DeepSets' OOD failure due to LATENT DIM (Wagstaff: dim>=max set size) or the LEARNED
READOUT? Sweep d in {64,256,512,1024} for rho(Sum phi(x)), count-only, rho=linear (best case for
extrapolation). Max set size here is 8, so d>=8 already satisfies Wagstaff -> predict no improvement
with d (=> it's the learned readout). Compare vs fixed-sum reference (d-independent, ~0.99).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
HF, STEPS, SEEDS = 5, 500, 3
TASKS=["steps_in_room","rooms_visited","co_occupancy"]
DIMS=[64,256,512,1024]
def load(t):
    o=torch.load(CACHE/f"minimal_L19_{t}.pt",map_location="cpu",weights_only=False)
    return [(v["reps"].float(),int(v["gold"])) for v in o.values() if torch.is_tensor(v["reps"])]
rows=["task,latent_dim,ood_mean,ood_std,per_seed"]
for task in TASKS:
    ex=load(task)
    print(f"\n##### {task} #####")
    for d in DIMS:
        accs=[]
        for seed in range(SEEDS):
            torch.manual_seed(seed); rng=random.Random(seed)
            lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo)
            tr=lo[int(0.2*len(lo)):]; D=ex[0][0].shape[1]
            allf=torch.cat([r for r,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6
            nrm=lambda r:(r-mu)/sd
            phi=torch.nn.Sequential(torch.nn.Linear(D,d),torch.nn.GELU(),torch.nn.Linear(d,d),torch.nn.GELU())
            rho=torch.nn.Linear(d,1)
            opt=torch.optim.Adam(list(phi.parameters())+list(rho.parameters()),lr=3e-3)
            pr=lambda r: rho(phi(nrm(r)).sum(0)).squeeze()
            for _ in range(STEPS):
                opt.zero_grad(); batch=rng.sample(tr,min(64,len(tr))); loss=0.0
                for r,g in batch: loss=loss+(pr(r)-g)**2
                (loss/len(batch)).backward(); opt.step()
            with torch.no_grad():
                accs.append(float(np.mean([round(float(pr(r)))==g for r,g in ood])))
        print(f"  d={d:<5} OOD={np.mean(accs):.3f} +/- {np.std(accs):.3f}  {[round(x,2) for x in accs]}")
        rows.append(f"{task},{d},{np.mean(accs):.4f},{np.std(accs):.4f},{';'.join(f'{x:.3f}' for x in accs)}")
(CACHE.parent/"readout_benchmark"/"dimsweep.csv").write_text("\n".join(rows)+"\n")
print("\nwrote dimsweep.csv")
