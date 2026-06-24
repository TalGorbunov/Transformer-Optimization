#!/usr/bin/env python3
"""Can per-frame supervision rescue CANONICAL DeepSets (multi-dim phi-MLP)? Occurrence tasks.
  variant = phi-MLP -> sum -> rho ; trained with count_MSE + lambda * per-frame BCE on an aux head.
  rho in {linear, MLP} to isolate whether the nonlinear readout caps OOD.
Multi-seed count-holdout. (rooms is the soft-OR case, already 1.000.)
"""
import sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
HF, STEPS, SEEDS, D_MEM, LAM = 5, 500, 4, 128, 1.0
def load(t):
    o=torch.load(CACHE/f"minimal_L19_{t}.pt",map_location="cpu",weights_only=False)
    return [(v["reps"].float(),int(v["gold"]),v.get("frame_labels")) for v in o.values()
            if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
rows=["task,variant,ood_mean,ood_std,per_seed,last_per_count"]
for task in ["steps_in_room","co_occupancy"]:
    ex=load(task)
    if isinstance(ex[0][2][0],str): continue
    print(f"\n##### {task} (canonical phi-MLP DeepSets + per-frame sup) #####")
    for variant in ["ds_framesup_rhoLinear","ds_framesup_rhoMLP"]:
        accs=[]; lastpc=""
        for seed in range(SEEDS):
            torch.manual_seed(seed); rng=random.Random(seed)
            lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo)
            tr=lo[int(0.2*len(lo)):]; D=ex[0][0].shape[1]
            allf=torch.cat([r for r,_,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6
            nrm=lambda r:(r-mu)/sd
            phi=torch.nn.Sequential(torch.nn.Linear(D,D_MEM),torch.nn.GELU(),torch.nn.Linear(D_MEM,D_MEM),torch.nn.GELU())
            aux=torch.nn.Linear(D_MEM,1)
            rho=(torch.nn.Linear(D_MEM,1) if "Linear" in variant
                 else torch.nn.Sequential(torch.nn.Linear(D_MEM,D_MEM),torch.nn.GELU(),torch.nn.Linear(D_MEM,1)))
            params=list(phi.parameters())+list(aux.parameters())+list(rho.parameters())
            opt=torch.optim.Adam(params,lr=3e-3); bce=torch.nn.BCEWithLogitsLoss()
            for _ in range(STEPS):
                opt.zero_grad(); batch=rng.sample(tr,min(64,len(tr))); loss=0.0
                for r,g,lab in batch:
                    h=phi(nrm(r))                      # [N,d]
                    loss=loss+(rho(h.sum(0)).squeeze()-g)**2
                    loss=loss+LAM*bce(aux(h).squeeze(-1),torch.tensor([float(x) for x in lab]))
                (loss/len(batch)).backward(); opt.step()
            with torch.no_grad():
                pr=lambda r: float(rho(phi(nrm(r)).sum(0)).squeeze())
                accs.append(float(np.mean([round(pr(r))==g for r,g,_ in ood])))
                by=defaultdict(list)
                for r,g,_ in ood: by[g].append(pr(r))
                lastpc=" ".join(f"{g}:{np.mean([round(p)==g for p in ps]):.2f}(mp{np.mean(ps):.1f})" for g,ps in sorted(by.items()))
        print(f"  {variant:<22} OOD={np.mean(accs):.3f} +/- {np.std(accs):.3f}  {[round(x,2) for x in accs]}")
        print(f"       last per-count: {lastpc}")
        rows.append(f"{task},{variant},{np.mean(accs):.4f},{np.std(accs):.4f},{';'.join(f'{x:.3f}' for x in accs)},{lastpc.replace(',',' ')}")
(CACHE.parent/"readout_benchmark"/"deepsets_framesup.csv").write_text("\n".join(rows)+"\n")
print("\nwrote deepsets_framesup.csv")
