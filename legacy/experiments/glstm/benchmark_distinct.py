#!/usr/bin/env python3
"""Push SUM to its limit on DISTINCT-over-characters (tasks 12 distinct_visitors, 13 distinct_companions).
Compare, count-holdout (train<5, test>=5), multi-seed, on cached per-frame reps + per-frame char-list labels:
  sum_countonly  : scalar Sigma sigma(w.rep), MSE->distinct gold   (does scalar sum REGRESS distinct, like rooms 0.97?)
  softOR_framesup: per-char p(char present|frame), soft-OR over frames per char, SUM over chars = distinct
                   (the structurally-correct operator; per-frame multi-hot char supervision)
Reports OOD mean+/-std. (Distinct-over-chars needs ~9-char roster -> crowded; extraction-confounded, accepted.)
"""
import sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
HF, STEPS, SEEDS = 5, 400, 3
def load(t):
    f = CACHE/f"minimal_L19_{t}.pt"
    if not f.is_file(): return None
    o = torch.load(f, map_location="cpu", weights_only=False)
    return [(v["reps"].float(), int(v["gold"]), v.get("frame_labels")) for v in o.values()
            if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
for task in ["distinct_visitors","distinct_companions"]:
    ex = load(task)
    if ex is None: print(f"{task}: cache missing, skip"); continue
    chars = sorted({c for _,_,lab in ex for fr in lab for c in fr}); cidx={c:i for i,c in enumerate(chars)}; Kc=len(chars)
    print(f"\n##### {task}: {len(ex)} ex, {Kc} chars, dist={dict(sorted(Counter(g for _,g,_ in ex).items()))} #####")
    agg={"sum_countonly":[], "softOR_framesup":[]}
    for seed in range(SEEDS):
        torch.manual_seed(seed); rng=random.Random(seed)
        lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo); tr=lo[int(0.2*len(lo)):]
        D=ex[0][0].shape[1]; allf=torch.cat([r for r,_,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6; nrm=lambda r:(r-mu)/sd
        ovf=lambda fn: float(np.mean([round(fn(r))==g for r,g,_ in ood]))
        # sum_countonly
        w=torch.zeros(D,requires_grad=True); b=torch.zeros(1,requires_grad=True); opt=torch.optim.Adam([w,b],lr=5e-3)
        for _ in range(STEPS):
            opt.zero_grad(); batch=rng.sample(tr,min(64,len(tr))); loss=0.0
            for r,g,_ in batch: loss=loss+(torch.sigmoid(nrm(r)@w+b).sum()-g)**2
            (loss/len(batch)).backward(); opt.step()
        agg["sum_countonly"].append(ovf(lambda r: float(torch.sigmoid(nrm(r)@w+b).sum().detach())))
        # softOR per-char (multi-hot per-frame supervision)
        W=torch.zeros(D,Kc,requires_grad=True); b2=torch.zeros(Kc,requires_grad=True); opt=torch.optim.Adam([W,b2],lr=5e-3)
        bce=torch.nn.BCEWithLogitsLoss()
        X=torch.cat([nrm(r) for r,_,_ in tr],0)
        Y=torch.zeros(X.shape[0],Kc); row=0
        for r,_,lab in tr:
            for fr in lab:
                for c in fr: Y[row,cidx[c]]=1.0
                row+=1
        for _ in range(STEPS): opt.zero_grad(); bce(X@W+b2,Y).backward(); opt.step()
        def sor(r):
            p=torch.sigmoid(nrm(r)@W+b2).clamp(1e-4,1-1e-4)   # [N,Kc]
            return float((1-torch.exp(torch.log1p(-p).sum(0))).sum().detach())   # soft-OR per char, sum over chars
        agg["softOR_framesup"].append(ovf(sor))
    for k,vs in agg.items():
        print(f"  {k:<18} OOD = {np.mean(vs):.3f} +/- {np.std(vs):.3f}  per-seed {[round(x,2) for x in vs]}")
