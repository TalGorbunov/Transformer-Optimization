#!/usr/bin/env python3
"""#1 sigma-vs-crowding: does the per-frame noise (sigma_within on the count axis) grow with the number
of characters in a frame? If yes, crowding -> more noise -> worse count SNR, linking 'crowding' and
'over-squashing' quantitatively. Uses the L19 cache + the source states (chars/frame)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv


def chars_in_frame(state):
    return sum(len(v) for v in state["rooms"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data-root", required=True, help="e.g. data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()
    cache = torch.load(args.cache, map_location="cpu")
    root = Path(args.data_root)
    reps_l, lab_l, crowd_l = [], [], []
    miss = 0
    for name, v in cache.items():
        if int(v.get("seq_len", -1)) != args.seq_len or v.get("frame_labels") is None:
            continue
        states = rv.states_of(root / name / "qa.txt")
        reps = v["reps"].float().numpy()
        if not states or len(states) != reps.shape[0]:
            miss += 1; continue
        lab = [int(x) for x in v["frame_labels"]]
        for fi in range(reps.shape[0]):
            reps_l.append(reps[fi]); lab_l.append(lab[fi]); crowd_l.append(chars_in_frame(states[fi]))
    X = np.stack(reps_l); y = np.asarray(lab_l); c = np.asarray(crowd_l)
    print(f"frames={len(y)} (missed {miss} dirs); crowding dist {dict(zip(*np.unique(c, return_counts=True)))}")
    # global count axis
    dhat = (X[y == 1].mean(0) - X[y == 0].mean(0)); dhat /= (np.linalg.norm(dhat) + 1e-9)
    proj = X @ dhat
    print(f"\n{'chars/frame':>11} {'n':>5} {'sigma_within':>13} {'|dmu|':>8} {'SNR':>7}")
    for cv in sorted(set(c.tolist())):
        m = c == cv
        if m.sum() < 30 or len(set(y[m].tolist())) < 2:
            continue
        pe = proj[m & (y == 1)]; pn = proj[m & (y == 0)]
        sig = 0.5 * (pe.std() + pn.std())
        dmu = abs(pe.mean() - pn.mean())
        print(f"{cv:>11} {int(m.sum()):>5} {sig:>13.3f} {dmu:>8.3f} {dmu/(sig+1e-9):>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
