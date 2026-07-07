#!/usr/bin/env python3
"""Is distinct-over-chars EXTRACTION-bound? Per-frame per-character detector AUC: can a linear probe tell
WHICH of the 9 chars is in room R (distinct_visitors) / in C's room (distinct_companions) from the frame rep?
Low AUC (~0.5-0.7) => extraction is the wall under 9-char crowding; high AUC => problem is elsewhere."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
def auc(s, y):
    s=np.asarray(s); y=np.asarray(y)
    if y.min()==y.max(): return float("nan")
    p=s[y==1]; n=s[y==0]; return float((p[:,None]>n[None,:]).mean())
for task in ["distinct_visitors","distinct_companions"]:
    f=CACHE/f"minimal_L19_{task}.pt"
    if not f.is_file(): print(f"{task}: cache missing"); continue
    o=torch.load(f,map_location="cpu",weights_only=False)
    rows=[]  # (frame_rep, set-of-relevant-chars) per frame
    chars=set()
    for v in o.values():
        if not torch.is_tensor(v["reps"]) or v.get("frame_labels") is None: continue
        for i,fr in enumerate(v["frame_labels"]):
            rows.append((v["reps"][i].float(), set(fr))); chars.update(fr)
    chars=sorted(chars); cidx={c:i for i,c in enumerate(chars)}; K=len(chars)
    rng=random.Random(0); rng.shuffle(rows); ntr=int(0.7*len(rows))
    tr=rows[:ntr]; te=rows[ntr:]
    X=torch.stack([r for r,_ in tr]); mu=X.mean(0); sd=X.std(0)+1e-6; nrm=lambda r:(r-mu)/sd
    Xn=torch.stack([nrm(r) for r,_ in tr]); D=Xn.shape[1]
    Y=torch.zeros(len(tr),K)
    for i,(_,s) in enumerate(tr):
        for c in s: Y[i,cidx[c]]=1.0
    W=torch.zeros(D,K,requires_grad=True); b=torch.zeros(K,requires_grad=True)
    opt=torch.optim.Adam([W,b],lr=5e-3); bce=torch.nn.BCEWithLogitsLoss()
    for _ in range(300): opt.zero_grad(); bce(Xn@W+b,Y).backward(); opt.step()
    with torch.no_grad():
        aucs=[]
        for ci,c in enumerate(chars):
            s=[float((nrm(r)@W[:,ci]+b[ci])) for r,_ in te]; y=[int(c in st) for _,st in te]
            a=auc(s,y);  aucs.append(a) if not np.isnan(a) else None
        print(f"{task}: per-char detection AUC = {np.nanmean(aucs):.3f} (mean over {K} chars) | range [{np.nanmin(aucs):.2f},{np.nanmax(aucs):.2f}]")
