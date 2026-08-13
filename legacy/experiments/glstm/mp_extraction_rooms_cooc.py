#!/usr/bin/env python3
"""Does MULTIPASS lift per-frame EXTRACTION for the low-acc tasks (co_occupancy, rooms_visited)?

co_occ task acc ~0.55 == its joint per-frame extraction_p (0.558 ~ chance for a binary "are C,D in the
same room this frame?"). rooms_visited needs which-room-of-C per frame (multiclass). If multipass
(focused single-frame reads) pushes these per-frame signals up like it did for steps (0.99->1.0), the
recipe for the hard tasks is "multipass extraction -> simple sum", no fancier aggregator needed.

Reads the existing caches in cache_mp_compare/ (joint + *_multipass), fits a LINEAR per-frame probe
(split by sample), reports:
  - co_occupancy: binary AUC + acc (chance 0.5)
  - rooms_visited: multiclass which-room acc (chance 1/n_rooms)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "outputs" / "frame_axis" / "cache_mp_compare"


def split_by_sample(n_samples, seed=0, frac=0.2):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g)
    n_te = max(1, int(frac * n_samples))
    te = set(perm[:n_te].tolist())
    return te


def fit_linear(Xtr, ytr, Xte, yte, n_cls, device="cpu", steps=400):
    lin = nn.Linear(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); lossf(lin(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        logit = lin(Xte)
        acc = (logit.argmax(1) == yte).float().mean().item()
        auc = float("nan")
        if n_cls == 2:
            s = logit[:, 1].numpy(); y = yte.numpy()
            p, ng = s[y == 1], s[y == 0]
            if len(p) and len(ng):
                auc = float((p[:, None] > ng[None, :]).mean())
    return acc, auc


def cooc_xy(cache):
    X, y, samp = [], [], []
    for si, v in enumerate(cache.values()):
        fl = v.get("frame_labels")
        if not torch.is_tensor(v["reps"]) or fl is None:
            continue
        reps = v["reps"].float()
        for i, lab in enumerate(fl):
            X.append(reps[i]); y.append(int(lab)); samp.append(si)
    return torch.stack(X), torch.tensor(y), torch.tensor(samp)


def rooms_xy(cache):
    X, rooms, samp = [], [], []
    for si, v in enumerate(cache.values()):
        fl = v.get("frame_labels")
        if not torch.is_tensor(v["reps"]) or fl is None:
            continue
        reps = v["reps"].float()
        for i, r in enumerate(fl):
            X.append(reps[i]); rooms.append(str(r)); samp.append(si)
    vocab = sorted(set(rooms)); ridx = {r: i for i, r in enumerate(vocab)}
    y = torch.tensor([ridx[r] for r in rooms])
    return torch.stack(X), y, torch.tensor(samp), len(vocab)


def run(task):
    joint = CACHE / f"minimal_L19_{task}.pt"
    mp = CACHE / f"minimal_L19_{task}_multipass.pt"
    print(f"\n#### {task} ####", flush=True)
    for tag, path in (("JOINT", joint), ("MULTIPASS", mp)):
        if not path.is_file():
            print(f"  {tag}: missing {path}"); continue
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if task == "co_occupancy":
            X, y, samp = cooc_xy(cache); n_cls = 2; chance = 0.5
        else:
            X, y, samp, n_cls = rooms_xy(cache); chance = 1.0 / n_cls
        te_ids = split_by_sample(int(samp.max()) + 1)
        te = torch.tensor([int(s) in te_ids for s in samp])
        mu = X[~te].mean(0, keepdim=True); sd = X[~te].std(0, keepdim=True) + 1e-6
        Xn = (X - mu) / sd
        acc, auc = fit_linear(Xn[~te], y[~te], Xn[te], y[te], n_cls)
        extra = f"  AUC={auc:.3f}" if n_cls == 2 else f"  ({n_cls} rooms)"
        print(f"  {tag:9s} per-frame acc={acc:.3f}  (chance={chance:.3f}){extra}  "
              f"[n_frames={len(y)}]", flush=True)


if __name__ == "__main__":
    print(f"cache dir: {CACHE}", flush=True)
    run("co_occupancy")
    run("rooms_visited")
