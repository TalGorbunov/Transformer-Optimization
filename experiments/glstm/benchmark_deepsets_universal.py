#!/usr/bin/env python3
"""UNIVERSAL canonical DeepSets that EXTRAPOLATES: rho(Sum phi(x)) with
  phi : Linear(D->S) -> sigmoid  (S latent slots, per-frame soft state)
  pool: FIXED extensive channels  soft_sum = sum_i p_i  ,  soft_or = 1 - prod_i(1-p_i)   [each S-dim]
  rho : LINEAR over [soft_sum, soft_or]  (extensive -> extrapolates; LEARNS occurrence-vs-distinct)
Trained: count_MSE + lambda * per-frame BCE(p, per-frame-state-target). One architecture, all 3 tasks.
  steps/co-occ target: slot0 = evidence (binary).   rooms target: one-hot(room) over first slots.
Tests whether canonical DeepSets, fixed-extensive-rho + per-frame sup, covers every task OOD.
"""
import sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
HF, STEPS, SEEDS, S, LAM = 5, 500, 4, 8, 1.0
TASKS = ["steps_in_room","rooms_visited","co_occupancy"]
def load(t):
    o=torch.load(CACHE/f"minimal_L19_{t}.pt",map_location="cpu",weights_only=False)
    return [(v["reps"].float(),int(v["gold"]),v.get("frame_labels")) for v in o.values()
            if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
def target_for(task, lab, roomidx):
    T=torch.zeros(len(lab),S)
    if task=="rooms_visited":
        for i,r in enumerate(lab):
            if roomidx.get(r,99)<S: T[i,roomidx[r]]=1.0
    else:
        for i,v in enumerate(lab): T[i,0]=float(v)
    return T
rows=["task,ood_mean,ood_std,per_seed,last_per_count"]
for task in TASKS:
    ex=load(task)
    rooms=sorted({l for _,_,lab in ex for l in lab}) if isinstance(ex[0][2][0],str) else []
    roomidx={r:i for i,r in enumerate(rooms)}
    print(f"\n##### {task} (S={S} slots, rooms={len(rooms)}) #####")
    accs=[]; lastpc=""
    for seed in range(SEEDS):
        torch.manual_seed(seed); rng=random.Random(seed)
        lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo)
        tr=lo[int(0.2*len(lo)):]; D=ex[0][0].shape[1]
        allf=torch.cat([r for r,_,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6
        nrm=lambda r:(r-mu)/sd
        phi=torch.nn.Linear(D,S); rho=torch.nn.Linear(2*S,1)
        opt=torch.optim.Adam(list(phi.parameters())+list(rho.parameters()),lr=3e-3); bce=torch.nn.BCELoss()
        def feats(r):
            p=torch.sigmoid(phi(nrm(r))).clamp(1e-4,1-1e-4)   # [N,S]
            ss=p.sum(0); so=1-torch.exp(torch.log1p(-p).sum(0))
            return torch.cat([ss,so]), p
        for _ in range(STEPS):
            opt.zero_grad(); batch=rng.sample(tr,min(64,len(tr))); loss=0.0
            for r,g,lab in batch:
                f,p=feats(r)
                loss=loss+(rho(f).squeeze()-g)**2 + LAM*bce(p, target_for(task,lab,roomidx))
            (loss/len(batch)).backward(); opt.step()
        with torch.no_grad():
            pr=lambda r: float(rho(feats(r)[0]).squeeze())
            accs.append(float(np.mean([round(pr(r))==g for r,g,_ in ood])))
            by=defaultdict(list)
            for r,g,_ in ood: by[g].append(pr(r))
            lastpc=" ".join(f"{g}:{np.mean([round(p)==g for p in ps]):.2f}(mp{np.mean(ps):.1f})" for g,ps in sorted(by.items()))
    print(f"  universal_extensive_DS  OOD={np.mean(accs):.3f} +/- {np.std(accs):.3f}  {[round(x,2) for x in accs]}")
    print(f"     last per-count: {lastpc}")
    rows.append(f"{task},{np.mean(accs):.4f},{np.std(accs):.4f},{';'.join(f'{x:.3f}' for x in accs)},{lastpc.replace(',',' ')}")
(CACHE.parent/"readout_benchmark"/"deepsets_universal.csv").write_text("\n".join(rows)+"\n")
print("\nwrote deepsets_universal.csv")
