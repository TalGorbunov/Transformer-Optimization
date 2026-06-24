#!/usr/bin/env python3
"""How LITTLE per-frame supervision stabilizes the count-only sum? Sweep lambda in
loss = count_MSE + lambda * per_frame_BCE  (binary tasks). Multi-seed OOD mean+/-std vs lambda.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
HF, STEPS, SEEDS = 5, 400, 4
def load(t):
    o=torch.load(CACHE/f"minimal_L19_{t}.pt",map_location="cpu",weights_only=False)
    return [(v["reps"].float(),int(v["gold"]),v.get("frame_labels")) for v in o.values()
            if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
rows=["task,lambda,ood_mean,ood_std,per_seed"]
for task in ["steps_in_room","co_occupancy"]:
    ex=load(task)
    if isinstance(ex[0][2][0],str): continue
    print(f"\n##### {task} #####")
    for lam in [0.0,0.01,0.05,0.2,1.0]:
        accs=[]
        for seed in range(SEEDS):
            torch.manual_seed(seed); rng=random.Random(seed)
            lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo)
            tr=lo[int(0.2*len(lo)):]; D=ex[0][0].shape[1]
            allf=torch.cat([r for r,_,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6
            nrm=lambda r:(r-mu)/sd
            w=torch.zeros(D,requires_grad=True); b=torch.zeros(1,requires_grad=True)
            opt=torch.optim.Adam([w,b],lr=5e-3); bce=torch.nn.BCEWithLogitsLoss()
            for _ in range(STEPS):
                opt.zero_grad(); batch=rng.sample(tr,min(64,len(tr))); loss=0.0
                for r,g,lab in batch:
                    logit=nrm(r)@w+b
                    loss=loss+(torch.sigmoid(logit).sum()-g)**2
                    if lam>0: loss=loss+lam*bce(logit,torch.tensor([float(x) for x in lab]))
                (loss/len(batch)).backward(); opt.step()
            accs.append(float(np.mean([round(float(torch.sigmoid(nrm(r)@w+b).sum().detach()))==g for r,g,_ in ood])))
        print(f"  lambda={lam:<5} OOD={np.mean(accs):.3f} +/- {np.std(accs):.3f}  {[round(x,2) for x in accs]}")
        rows.append(f"{task},{lam},{np.mean(accs):.4f},{np.std(accs):.4f},{';'.join(f'{x:.3f}' for x in accs)}")
(CACHE.parent/"readout_benchmark"/"auxloss.csv").write_text("\n".join(rows)+"\n")
print("\nwrote auxloss.csv")
