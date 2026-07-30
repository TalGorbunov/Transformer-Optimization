#!/usr/bin/env python3
"""Can SUM do rooms (distinct-count)? Compare, on the rooms cache, count-holdout, multi-seed:
  count_only_sum       : Σσ(w·rep), MSE-to-count (no per-frame labels)        [agnostic, suspected noisy]
  framesup_sum_firstv  : per-frame BCE on 'first visit to this room' -> Σσ     [sum w/ the right label]
  framesup_softOR      : per-frame room CE -> Σ_room[1-Π(1-p)]                  [the soft-OR operator]
The first-visit label per frame = 1 iff C's room hasn't appeared in earlier frames.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
from collections import Counter

CACHE = PROJECT_ROOT / "outputs" / "frame_axis" / "cache" / "minimal_L19_rooms_visited.pt"
HF, STEPS, SEEDS = 5, 400, 4


def first_visit(rooms):  # [r0,r1,...] -> [1,0,1,...] (1 at first occurrence of each room)
    seen = set(); out = []
    for r in rooms:
        out.append(0 if r in seen else 1); seen.add(r)
    return out


def main():
    obj = torch.load(CACHE, map_location="cpu", weights_only=False)
    ex = [(v["reps"].float(), int(v["gold"]), v["frame_labels"]) for v in obj.values()
          if torch.is_tensor(v["reps"]) and v.get("frame_labels") is not None]
    print(f"rooms: {len(ex)} examples, dist={dict(sorted(Counter(g for _,g,_ in ex).items()))}")
    agg = {"count_only_sum": [], "framesup_sum_firstv": [], "framesup_softOR": []}
    for seed in range(SEEDS):
        torch.manual_seed(seed); rng = random.Random(seed)
        lo = [e for e in ex if e[1] < HF]; ood = [e for e in ex if e[1] >= HF]; rng.shuffle(lo)
        tr = lo[int(0.2 * len(lo)):]
        D = ex[0][0].shape[1]
        allf = torch.cat([r for r, _, _ in tr], 0); mu = allf.mean(0); sd = allf.std(0) + 1e-6
        nrm = lambda r: (r - mu) / sd
        ovf = lambda fn: float(np.mean([round(fn(r)) == g for r, g, _ in ood]))

        # count-only sum
        w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=5e-3)
        for _ in range(STEPS):
            opt.zero_grad(); batch = rng.sample(tr, min(64, len(tr))); loss = 0.0
            for r, g, _ in batch:
                loss = loss + (torch.sigmoid(nrm(r) @ w + b).sum() - g) ** 2
            (loss / len(batch)).backward(); opt.step()
        agg["count_only_sum"].append(ovf(lambda r: float(torch.sigmoid(nrm(r) @ w + b).sum().detach())))

        # frame-sup sum with first-visit labels
        X = torch.cat([nrm(r) for r, _, _ in tr], 0)
        y = torch.tensor([float(v) for _, _, lab in tr for v in first_visit(lab)])
        w2 = torch.zeros(D, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w2, b2], lr=5e-3); bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(STEPS):
            opt.zero_grad(); bce(X @ w2 + b2, y).backward(); opt.step()
        agg["framesup_sum_firstv"].append(ovf(lambda r: float(torch.sigmoid(nrm(r) @ w2 + b2).sum().detach())))

        # frame-sup soft-OR (per-frame room classifier)
        rooms = sorted({l for _, _, lab in tr for l in lab}); ridx = {r: i for i, r in enumerate(rooms)}; K = len(rooms)
        Xr = torch.cat([nrm(r) for r, _, _ in tr], 0); yr = torch.tensor([ridx[l] for _, _, lab in tr for l in lab])
        W = torch.zeros(D, K, requires_grad=True); b3 = torch.zeros(K, requires_grad=True)
        opt = torch.optim.Adam([W, b3], lr=5e-3); ce = torch.nn.CrossEntropyLoss()
        for _ in range(STEPS):
            opt.zero_grad(); ce(Xr @ W + b3, yr).backward(); opt.step()
        def sor(r):
            p = torch.softmax(nrm(r) @ W + b3, -1).clamp(1e-4, 1 - 1e-4)
            return float((1 - torch.exp(torch.log1p(-p).sum(0))).sum().detach())
        agg["framesup_softOR"].append(ovf(sor))

    print("\nrooms OOD (counts 5,6 held out) across seeds:")
    for k, vs in agg.items():
        print(f"  {k:<20} {np.mean(vs):.3f} ± {np.std(vs):.3f}   per-seed {[round(x,2) for x in vs]}")


if __name__ == "__main__":
    raise SystemExit(main())
