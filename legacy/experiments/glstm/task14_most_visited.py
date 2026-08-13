#!/usr/bin/env python3
"""Task 14 most_visited_room: NAME answer via per-candidate SUM + argmax (read->reduce->DECODE template).
Reuses the rooms cache (per-frame reps + per-frame room-of-C labels). 
  [B] per-room detector p(room r | frame) = softmax(W rep)   (per-frame supervised by room-of-C)
  [C] score_r = SUM_frames p(r|frame)                        (per-candidate occupancy sum)
  [D] answer  = argmax_r score_r -> room name                (categorical decode)
gold = the room C is in most often (mode of the room sequence). argmax is magnitude-robust -> this tests
the TEMPLATE generalization to name answers (not a count-OOD test). Train/test split, multi-seed.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
from collections import Counter
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"/"minimal_L19_rooms_visited.pt"
STEPS, SEEDS = 400, 3
obj = torch.load(CACHE, map_location="cpu", weights_only=False)
ex = [(v["reps"].float(), v["frame_labels"]) for v in obj.values()
      if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
rooms = sorted({r for _,lab in ex for r in lab}); ridx={r:i for i,r in enumerate(rooms)}; K=len(rooms)
def gold_most(lab): return ridx[Counter(lab).most_common(1)[0][0]]
print(f"task14 most_visited_room: {len(ex)} examples, {K} rooms")
accs=[]
for seed in range(SEEDS):
    torch.manual_seed(seed); rng=random.Random(seed); idx=list(range(len(ex))); rng.shuffle(idx)
    n=int(0.25*len(ex)); te=[ex[i] for i in idx[:n]]; tr=[ex[i] for i in idx[n:]]
    D=ex[0][0].shape[1]
    allf=torch.cat([r for r,_ in tr],0); mu=allf.mean(0); sd=allf.std(0)+1e-6; nrm=lambda r:(r-mu)/sd
    # per-room classifier, per-frame supervised by room-of-C
    X=torch.cat([nrm(r) for r,_ in tr],0); y=torch.tensor([ridx[l] for _,lab in tr for l in lab])
    W=torch.zeros(D,K,requires_grad=True); b=torch.zeros(K,requires_grad=True)
    opt=torch.optim.Adam([W,b],lr=5e-3); ce=torch.nn.CrossEntropyLoss()
    for _ in range(STEPS): opt.zero_grad(); ce(X@W+b,y).backward(); opt.step()
    with torch.no_grad():
        ok=0
        for r,lab in te:
            score=torch.softmax(nrm(r)@W+b,-1).sum(0)   # [K] per-room occupancy
            ok += int(int(score.argmax())==gold_most(lab))
        accs.append(ok/len(te))
print(f"  most_visited_room accuracy = {np.mean(accs):.3f} +/- {np.std(accs):.3f}  per-seed {[round(x,2) for x in accs]}")
# baseline: random room = 1/K
print(f"  (chance = 1/{K} = {1/K:.3f})")
