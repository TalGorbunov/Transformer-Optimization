#!/usr/bin/env python3
"""Spot-check: is L19 the right cache layer? Run sum_countonly OOD on distinct_visitors at L17/19/21."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"; HF, STEPS, SEEDS = 5, 400, 3
for L in [17, 19, 21]:
    f = CACHE/f"minimal_L{L}_distinct_visitors.pt"
    if not f.is_file(): print(f"L{L}: cache missing"); continue
    o = torch.load(f, map_location="cpu", weights_only=False)
    ex = [(v["reps"].float(), int(v["gold"])) for v in o.values() if torch.is_tensor(v["reps"])]
    accs=[]
    for seed in range(SEEDS):
        torch.manual_seed(seed); rng=random.Random(seed)
        lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo); tr=lo[int(0.2*len(lo)):]
        D=ex[0][0].shape[1]; allf=torch.cat([r for r,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6; nrm=lambda r:(r-mu)/sd
        w=torch.zeros(D,requires_grad=True); b=torch.zeros(1,requires_grad=True); opt=torch.optim.Adam([w,b],lr=5e-3)
        for _ in range(STEPS):
            opt.zero_grad(); batch=rng.sample(tr,min(64,len(tr))); loss=sum((torch.sigmoid(nrm(r)@w+b).sum()-g)**2 for r,g in batch)/len(batch)
            loss.backward(); opt.step()
        accs.append(float(np.mean([round(float(torch.sigmoid(nrm(r)@w+b).sum().detach()))==g for r,g in ood])))
    print(f"  L{L}: sum_countonly OOD = {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
