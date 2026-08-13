#!/usr/bin/env python3
"""Readout ablation: HOLD the per-frame detector FIXED (one supervised p_i), vary ONLY the pooling/readout.
  sum         = sum_i p_i              (fixed, parameter-free, extensive)   -> extrapolates
  mean        = (sum_i p_i)/N          (fixed; divides count out)           -> count-blind*
  max         = max_i p_i              (fixed; saturates at 1)              -> count-blind
  learned_rho = MLP(sum_i p_i)         (learned decoder ON TOP of the correct sum, trained on counts<5)
                                                                            -> memorizes 0..4, FAILS 5..8
Occurrence tasks (count = #evidence frames). count-holdout: train<5, test_iid<5, test_ood>=5. Multi-seed.
(*fixed N=8 here, so mean = sum/N is a rescaled sum recoverable by xN; read directly it predicts the
 fraction -> count-blind. Its magnitude-blindness is fundamental only when N varies.)
"""
import sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
HF, STEPS, SEEDS = 5, 400, 3
def load(t):
    o=torch.load(CACHE/f"minimal_L19_{t}.pt",map_location="cpu",weights_only=False)
    return [(v["reps"].float(),int(v["gold"]),v.get("frame_labels")) for v in o.values()
            if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
def run(task, seed):
    torch.manual_seed(seed); rng=random.Random(seed)
    ex=load(task)
    lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo)
    n=int(0.2*len(lo)); tr=lo[n:]; iid=lo[:n]; D=ex[0][0].shape[1]
    allf=torch.cat([r for r,_,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6; nrm=lambda r:(r-mu)/sd
    # FIXED detector: per-frame BCE on evidence
    X=torch.cat([nrm(r) for r,_,_ in tr],0); y=torch.tensor([float(v) for _,_,lab in tr for v in lab])
    w=torch.zeros(D,requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=5e-3); bce=torch.nn.BCEWithLogitsLoss()
    for _ in range(STEPS): opt.zero_grad(); bce(X@w+b,y).backward(); opt.step()
    p=lambda r: torch.sigmoid(nrm(r)@w+b).detach()   # [N] per-frame probs (FIXED)
    reads={"sum":lambda r:float(p(r).sum()),
           "mean":lambda r:float(p(r).mean()),
           "max":lambda r:float(p(r).max())}
    # learned_rho: MLP on the (frozen) sum, trained MSE on counts<5
    rho=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.GELU(),torch.nn.Linear(32,1))
    o2=torch.optim.Adam(rho.parameters(),lr=5e-3)
    S_tr=[(float(p(r).sum()),g) for r,g,_ in tr]
    for _ in range(STEPS):
        o2.zero_grad(); loss=0.0
        for s,g in random.Random(seed).sample(S_tr,min(64,len(S_tr))): loss=loss+(rho(torch.tensor([s]))-g)**2
        (loss/64).backward(); o2.step()
    reads["learned_rho(sum)"]=lambda r: float(rho(torch.tensor([float(p(r).sum())])).detach())
    out={}
    for nm,fn in reads.items():
        out[nm]=(float(np.mean([round(fn(r))==g for r,g,_ in iid])), float(np.mean([round(fn(r))==g for r,g,_ in ood])))
    return out
for task in ["steps_in_room","co_occupancy"]:
    agg=defaultdict(lambda:([],[]))
    for s in range(SEEDS):
        for nm,(i,o) in run(task,s).items(): agg[nm][0].append(i); agg[nm][1].append(o)
    print(f"\n##### {task} (detector FIXED; only readout varies) #####")
    print(f"  {'readout':<18} {'IID':>12} {'OOD':>14}")
    for nm in ["sum","mean","max","learned_rho(sum)"]:
        i,o=agg[nm]; print(f"  {nm:<18} {np.mean(i):.3f}        {np.mean(o):.3f} +/- {np.std(o):.2f}")
