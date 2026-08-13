#!/usr/bin/env python3
"""VERIFY the `sum` OOD-extrapolation result is real, not an artifact.

Key worry: causal attention + question-first could leak the running/total count INTO individual frame
reps, so "sum extrapolates" might be trivial (the answer is already in the reps) rather than genuine
per-frame-evidence aggregation. Controls (count-holdout, same cache, CPU):
  - sum            : Σ σ(w·repᵢ)                         (the claim)
  - detector AUC   : does the learned per-frame score match the cached per-frame evidence labels?
  - reg_mean       : linear REGRESSOR on mean-pooled rep  (count-invariant pooling; if it extrapolates -> leak)
  - reg_last       : linear regressor on the LAST frame rep only (if it extrapolates -> single frame leaks total)
  - reg_first      : linear regressor on the FIRST frame rep only (sanity: must NOT know the total)
  - reg_max        : linear regressor on max-pooled rep
  - sum_shuffle    : sum with per-example frame order shuffled (perm-invariant -> must be identical to sum)
If reg_mean/reg_last extrapolate ~ as well as sum, the win is confounded. If they fail (cap) while sum
succeeds, sum is genuinely doing extensive aggregation -> the result is real.
"""
from __future__ import annotations
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, random
torch.manual_seed(0); np.random.seed(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "cache")
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--tasks", default="steps_in_room,rooms_visited,co_occupancy")
    p.add_argument("--holdout-from", type=int, default=5)
    p.add_argument("--iid-frac", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "readout_benchmark")
    return p.parse_args()


def load(cache_dir, L, task):
    f = cache_dir / f"minimal_L{L}_{task}.pt"
    if not f.is_file():
        return None
    obj = torch.load(f, map_location="cpu", weights_only=False)
    ex = []
    for name, v in obj.items():
        if torch.is_tensor(v["reps"]):
            ex.append((name, v["reps"].float(), int(v["gold"]), v.get("frame_labels")))
    return ex


def split(ex, hf, iid_frac, rng):
    lo = [e for e in ex if e[2] < hf]; ood = [e for e in ex if e[2] >= hf]
    rng.shuffle(lo); n = int(iid_frac * len(lo)); return lo[n:], lo[:n], ood


def auc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    if y.min() == y.max():
        return float("nan")
    pos = s[y == 1]; neg = s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean())


def ov(items, fn):  # items: (name, reps, gold, labels)
    return float(np.mean([round(fn(r)) == g for _, r, g, _ in items])) if items else 0.0


def pc(items, fn):
    by = defaultdict(list)
    for _, r, g, _ in items:
        by[g].append(fn(r))
    return " ".join(f"{g}:{np.mean([round(p)==g for p in ps]):.2f}(mp{np.mean(ps):.1f})" for g, ps in sorted(by.items()))


def fit_pool_regressor(tr, nrm, pool, D, steps):
    w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=5e-3)
    X = torch.stack([pool(nrm(r)) for _, r, _, _ in tr]); y = torch.tensor([float(g) for _, _, g, _ in tr])
    for _ in range(steps):
        opt.zero_grad(); ((X @ w + b - y) ** 2).mean().backward(); opt.step()
    return lambda r: float((pool(nrm(r)) @ w + b).detach())


def main():
    args = parse_args()
    out = ["task,probe,iid,ood,note"]
    for task in [t for t in args.tasks.split(",") if t]:
        ex = load(args.cache_dir, args.read_layer, task)
        if ex is None:
            print(f"##### {task}: missing #####"); continue
        D = ex[0][1].shape[1]
        tr, iid, ood = split(ex, args.holdout_from, args.iid_frac, random.Random(0))
        # split sanity
        names_tr = {n for n, *_ in tr} | {n for n, *_ in iid}; names_ood = {n for n, *_ in ood}
        leak = names_tr & names_ood
        g_tr = sorted(set(g for _, _, g, _ in tr + iid)); g_ood = sorted(set(g for _, _, g, _ in ood))
        print(f"\n##### {task}: train+iid golds={g_tr}  ood golds={g_ood}  dir-overlap={len(leak)} "
              f"(MUST be 0)  dist={dict(sorted(Counter(g for _,_,g,_ in ex).items()))} #####")
        allf = torch.cat([r for _, r, _, _ in tr], 0); mu = allf.mean(0); sd = allf.std(0) + 1e-6
        nrm = lambda r: (r - mu) / sd

        # sum + detector AUC
        w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=5e-3)
        for _ in range(args.steps):
            opt.zero_grad(); batch = random.sample(tr, min(64, len(tr))); loss = 0.0
            for _, r, g, _ in batch:
                loss = loss + (torch.sigmoid(nrm(r) @ w + b).sum() - g) ** 2
            (loss / len(batch)).backward(); opt.step()
        sum_fn = lambda r: float(torch.sigmoid(nrm(r) @ w + b).sum().detach())
        print(f"  sum           IID={ov(iid,sum_fn):.3f} OOD={ov(ood,sum_fn):.3f}   {pc(ood,sum_fn)}")
        out.append(f"{task},sum,{ov(iid,sum_fn):.4f},{ov(ood,sum_fn):.4f},the-claim")
        # detector AUC vs cached per-frame labels (binary tasks only)
        if tr[0][3] is not None and not isinstance(tr[0][3][0], str):
            scs, lbs = [], []
            for _, r, _, lab in ood:
                p = torch.sigmoid(nrm(r) @ w + b).detach().numpy()
                scs += list(p); lbs += list(lab)
            print(f"  -> per-frame detector AUC vs gold evidence (OOD frames): {auc(scs,lbs):.3f}")
            out.append(f"{task},detector_auc_ood,,{auc(scs,lbs):.4f},perfframe")

        # LEAKAGE probes: pooled regressors (if these extrapolate, the reps carry the answer)
        probes = {
            "reg_mean": lambda h: h.mean(0),
            "reg_max": lambda h: h.max(0).values,
            "reg_last": lambda h: h[-1],
            "reg_first": lambda h: h[0],
        }
        for nm, pool in probes.items():
            fn = fit_pool_regressor(tr, nrm, pool, D, args.steps)
            print(f"  {nm:<13} IID={ov(iid,fn):.3f} OOD={ov(ood,fn):.3f}   {pc(ood,fn)}")
            out.append(f"{task},{nm},{ov(iid,fn):.4f},{ov(ood,fn):.4f},leak-probe")

        # permutation invariance sanity: sum on shuffled frame order (must equal sum)
        def sum_shuf(r):
            idx = torch.randperm(r.shape[0]); return sum_fn(r[idx])
        d = abs(np.mean([sum_shuf(r) for _, r, _, _ in ood[:50]]) - np.mean([sum_fn(r) for _, r, _, _ in ood[:50]]))
        print(f"  sum_shuffle vs sum mean-pred diff (must be ~0): {d:.4f}")
        out.append(f"{task},sum_shuffle_diff,,{d:.4f},perm-invariant-check")
    (args.output).mkdir(parents=True, exist_ok=True)
    (args.output / "verify.csv").write_text("\n".join(out) + "\n")
    print(f"\nwrote {args.output}/verify.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
