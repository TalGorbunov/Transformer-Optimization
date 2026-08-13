#!/usr/bin/env python3
"""Position-leakage probe: can a LINEAR head read a frame's INDEX straight off its per-frame L19 rep?

Motivation: the temporal runs (first/last/span) showed `sum` (permutation-invariant) nearly matching
`mamba`. The suspect: Qwen's positional encoding tags each frame's tokens with WHERE it sits in the
sequence, so an order-invariant Σφ(rep) becomes a *set-of-positions* multi-hot -> first/last/span
become decodable WITHOUT real order modelling. This probe quantifies that leakage.

Clean contrast (no new compute -- reuses existing caches):
  - JOINT cache: all frames in one forward -> frame k at sequence position p_k (varies) -> position
    is in the rep content -> linear probe SHOULD decode the index.
  - MULTIPASS cache: each frame forwarded ALONE (query + 1 frame) -> every frame at ~the same
    sequence position -> index should NOT be decodable (only content differs).
If joint >> multipass (and joint >> chance/shuffled), position-leakage is real and `sum` is not a
clean order-blind control for these temporal tasks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn


def load_xy(cache_path: Path, seq_len: int):
    cache = torch.load(cache_path)
    X, y, samp = [], [], []
    for i, (_k, v) in enumerate(cache.items()):
        if v is None:
            continue
        reps = v["reps"].float()  # [N, H]
        N = reps.shape[0]
        if N != seq_len:
            continue
        for j in range(N):
            X.append(reps[j]); y.append(j); samp.append(i)
    if not X:
        return None
    return torch.stack(X), torch.tensor(y), torch.tensor(samp)


def fit_probe(Xtr, ytr, Xte, yte, n_pos, device, steps=400, lr=1e-2, wd=1e-3):
    lin = nn.Linear(Xtr.shape[1], n_pos).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(lin(Xtr), ytr); loss.backward(); opt.step()
    with torch.no_grad():
        pred = lin(Xte).argmax(1)
    acc = (pred == yte).float().mean().item()
    # mean absolute index error (how FAR off, in frames) -- relevant for first/last decodability
    mae = (pred.float() - yte.float()).abs().mean().item()
    return acc, mae


def run_cache(name, path: Path, seq_len, device, seed=0):
    xy = load_xy(path, seq_len)
    if xy is None:
        print(f"## {name}: {path}  -> NO seq_len={seq_len} entries", flush=True); return
    X, y, samp = xy
    n_pos = seq_len
    g = torch.Generator().manual_seed(seed)
    uniq = samp.unique()
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    n_te = max(1, int(0.2 * len(uniq)))
    te_ids = set(perm[:n_te].tolist())
    te = torch.tensor([int(s.item()) in te_ids for s in samp])
    mu = X[~te].mean(0, keepdim=True); sd = X[~te].std(0, keepdim=True) + 1e-6
    Xn = ((X - mu) / sd).to(device)
    Xtr, ytr, Xte, yte = Xn[~te], y[~te].to(device), Xn[te], y[te].to(device)
    acc, mae = fit_probe(Xtr, ytr, Xte, yte, n_pos, device)
    ysh = ytr[torch.randperm(len(ytr), generator=g)]
    acc_sh, _ = fit_probe(Xtr, ysh, Xte, yte, n_pos, device)
    print(f"## {name}  (n_samples={len(uniq)}, n_frames={len(y)}, seq_len={seq_len})", flush=True)
    print(f"   frame-index linear-probe acc = {acc:.3f}   (chance={1.0/n_pos:.3f}, shuffled={acc_sh:.3f})", flush=True)
    print(f"   mean |pred-true| index error  = {mae:.2f} frames", flush=True)
    return acc, mae, acc_sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("outputs/frame_axis/cache_mp_compare"))
    ap.add_argument("--task", default="steps_in_room")
    ap.add_argument("--read-layer", type=int, default=19)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device={dev}", flush=True)
    joint = args.cache_dir / f"minimal_L{args.read_layer}_{args.task}.pt"
    mp = args.cache_dir / f"minimal_L{args.read_layer}_{args.task}_multipass.pt"
    res = {}
    if joint.is_file():
        res["joint"] = run_cache("JOINT (all frames, 1 forward)", joint, args.seq_len, dev)
    else:
        print(f"missing {joint}", flush=True)
    if mp.is_file():
        res["multipass"] = run_cache("MULTIPASS (each frame alone)", mp, args.seq_len, dev)
    else:
        print(f"missing {mp}", flush=True)
    if "joint" in res and "multipass" in res and res["joint"] and res["multipass"]:
        print(f"\n==> position leakage = joint - multipass acc = "
              f"{res['joint'][0] - res['multipass'][0]:+.3f}  "
              f"(large positive => sum can exploit position; 'sum can't do temporal' fails for joint reps)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
