#!/usr/bin/env python3
"""TASK-AGNOSTIC query-routed fixed-operator bank. ONE module:
  detector  phi: rep -> sigmoid -> p [N,S]   (query-conditioned; per-frame slot probs)
  fixed ops sum_op = sum_{frames,slots} p     (occurrence)   ;  or_op = sum_slot[1-prod_frames(1-p)] (distinct)
  router    softmax(Linear(query_rep)) -> [w_sum, w_or]   (picks the operator FROM THE QUERY, not the count)
  count = w_sum*sum_op + w_or*or_op
Per-frame supervised (slot0=evidence for occurrence; one-hot room for rooms). count-holdout (train<5,test>=5).
Configs: (A) train MIX of all 3 tasks -> OOD per task + router weights (shows auto-selection).
         (B) HELD-OUT TASK: train router+detector on steps+co-occ only, ZERO-SHOT rooms.
The router selects FIXED-extensive ops off the query -> agnostic AND extrapolates (no learned readout).
"""
import sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
CACHE = PROJECT_ROOT/"outputs"/"frame_axis"/"cache"
S, HF, STEPS, SEEDS, LAM = 8, 5, 700, 3, 1.0
ALL=["steps_in_room","rooms_visited","co_occupancy"]
def load(t):
    o=torch.load(CACHE/f"minimal_L19_{t}.pt",map_location="cpu",weights_only=False)
    return [(v["reps"].float(),int(v["gold"]),v["frame_labels"],v["query_rep"].float()) for v in o.values()
            if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None and "query_rep" in v]
def tgt(task,lab,ridx):
    T=torch.zeros(len(lab),S)
    if task=="rooms_visited":
        for i,r in enumerate(lab):
            if ridx.get(r,99)<S: T[i,ridx[r]]=1.0
    else:
        for i,v in enumerate(lab): T[i,0]=float(v)
    return T
def experiment(train_tasks, seed):
    torch.manual_seed(seed); rng=random.Random(seed)
    data={}; pool=[]
    for t in ALL:
        ex=load(t); rooms=sorted({l for _,_,lab,_ in ex for l in lab}) if isinstance(ex[0][2][0],str) else []
        ridx={r:i for i,r in enumerate(rooms)}
        lo=[e for e in ex if e[1]<HF]; ood=[e for e in ex if e[1]>=HF]; rng.shuffle(lo); tr=lo[int(0.2*len(lo)):]
        data[t]=(tr,ood,ridx)
        if t in train_tasks:
            for reps,g,lab,q in tr: pool.append((reps,g,tgt(t,lab,ridx),q))
    allf=torch.cat([r for r,_,_,_ in pool],0); mu=allf.mean(0); sd=allf.std(0)+1e-6
    allq=torch.stack([q for _,_,_,q in pool]); qmu=allq.mean(0); qsd=allq.std(0)+1e-6
    nrm=lambda r:(r-mu)/sd; qn=lambda q:(q-qmu)/qsd
    D=pool[0][0].shape[1]
    phi=torch.nn.Linear(D,S); router=torch.nn.Linear(D,2)
    opt=torch.optim.Adam(list(phi.parameters())+list(router.parameters()),lr=3e-3); bl=torch.nn.BCEWithLogitsLoss()
    def fwd(reps,q):
        lg=phi(nrm(reps)); p=torch.sigmoid(lg).clamp(1e-4,1-1e-4)
        sum_op=p.sum(); or_op=(1-torch.exp(torch.log1p(-p).sum(0))).sum()
        w=torch.softmax(router(qn(q)),-1)
        return w[0]*sum_op+w[1]*or_op, lg, w
    for _ in range(STEPS):
        opt.zero_grad(); batch=rng.sample(pool,min(64,len(pool))); loss=0.0
        for reps,g,T,q in batch:
            pred,lg,_=fwd(reps,q); loss=loss+(pred-g)**2+LAM*bl(lg,T)
        (loss/len(batch)).backward(); opt.step()
    out={}
    with torch.no_grad():
        for t in ALL:
            tr,ood,ridx=data[t]
            accs=[round(float(fwd(r,q)[0]))==g for r,g,_,q in ood]
            ws=np.mean([fwd(r,q)[2].numpy() for r,g,_,q in ood[:60]],0)
            out[t]=(float(np.mean(accs)), ws)  # (ood_acc, [w_sum,w_or])
    return out
def main():
    for cfg,tt in [("MIX_all3",ALL),("HELDOUT_rooms (train steps+cooc, zero-shot rooms)",["steps_in_room","co_occupancy"])]:
        print(f"\n############ {cfg} ############")
        agg=defaultdict(list); wagg=defaultdict(list)
        for s in range(SEEDS):
            for t,(a,w) in experiment(tt,s).items(): agg[t].append(a); wagg[t].append(w)
        for t in ALL:
            held=" (HELD-OUT)" if t not in tt else ""
            w=np.mean(wagg[t],0)
            print(f"  {t:<16}{held:<12} OOD={np.mean(agg[t]):.3f} +/- {np.std(agg[t]):.3f}  router[w_sum,w_or]=[{w[0]:.2f},{w[1]:.2f}]")
    return 0
if __name__=="__main__": raise SystemExit(main())
