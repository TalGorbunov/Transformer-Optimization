#!/usr/bin/env python3
"""Per-task decodability suite: which per-frame facts do the conditioned carriers hold?

Generalizes content_probe.py to every fact type the arm-B GIN readout needs. For each
requested qtype: load samples of THAT qtype (their own questions condition the
carriers), extract frozen carrier states at --layer, label each frame with the task's
LOCAL fact (the same functions that build the gold scans), and fit linear probes
(5 seeds, 50/50). Output: one decodability row per (qtype, fact).

Fact heads:
  gate      binary: question's (C,R) match           steps_in_room, crowd_count(any-room)
  room6     6-way room-of-C                          char_at_frame-style
  occ7      occupant-of-R in {5 names, nobody, multi} first/last_at_room-style
  cnt       small count (0..5)                       n_char_at_frame / n_empty
  emptyK    binary room-empty (per fixed room)       room_empty-style (6 heads pooled)

Usage:
  python scripts/mmred_hf/probe_suite.py --qtypes char_at_frame n_empty room_at_frame \
      --config seq_len_8 --split train --limit 150
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.engine import CarrierEngine  # noqa: E402
from gnnformer.mmred_hf import (  # noqa: E402
    CHAR_ORDER, ROOM_ORDER, _char_room, _match, _rooms, qtype_from_dirname,
)
from gnnformer.runtime import load_runtime  # noqa: E402
sys.path.insert(0, str(_REPO / 'scripts/mmred_hf'))
from facts import frame_labels  # noqa: E402




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtypes", nargs="+", required=True)
    ap.add_argument("--config", default="seq_len_8")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--carrier-ckpt", default="checkpoints/carrier_token_room_k1_best.pt")
    ap.add_argument("--output", default="outputs/mmred_hf/probe_suite")
    args = ap.parse_args()

    rt = load_runtime()
    tk = torch.load(args.carrier_ckpt, map_location="cpu", weights_only=False)
    e_c = (tk["e_c"] if isinstance(tk, dict) else tk.e_c).float().to(rt.device)
    eng = CarrierEngine(rt, l_open=12, e_c=e_c)

    from sklearn.linear_model import LogisticRegression
    lines = []
    for qtype in args.qtypes:
        root = _REPO / f"data/mmred_hf/dirs/{args.config}_{args.split}_{qtype}"
        if not root.is_dir():
            root = _REPO / f"data/mmred_hf/dirs/{args.config}_{args.split}"
        dirs = [d for d in sorted(root.iterdir())
                if qtype_from_dirname(d.name) == qtype][: args.limit]
        X, y = [], []
        space = "?"
        for sd in dirs:
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
                labs, space = frame_labels(qtype, q0, states)
            except Exception:
                continue
            rec = eng.prepare_sample(frames, q0, gold=0, task=qtype,
                                     resize=args.resize, with_masks=True,
                                     with_trunc_cols=True)
            if rec is None:
                continue
            with torch.no_grad():
                caches, *_ = eng.prefill_capture(rec, 12)
            kk = rec["keep"]
            st_l = caches[min(args.layer, len(caches) - 1)]
            for t, c in enumerate([kk.index(p) for p in rec["cpos"]]):
                X.append(st_l[c].float().cpu().numpy())
                y.append(labs[t])
        if len(y) < 60:
            lines.append(f"{qtype:28s} [{space}] SKIP (n={len(y)})")
            continue
        X = np.stack(X).astype(np.float32)
        y = np.array(y)
        accs = []
        for seed in range(5):
            idx = np.random.default_rng(seed).permutation(len(y))
            h = len(y) // 2
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[idx[:h]], y[idx[:h]])
            accs.append(float(clf.score(X[idx[h:]], y[idx[h:]])))
        maj = float(np.bincount(y).max()) / len(y)
        lines.append(f"{qtype:28s} [{space}] n={len(y):5d} acc {np.mean(accs):.3f}"
                     f"±{np.std(accs):.3f} (maj {maj:.3f})")
        print(lines[-1], flush=True)

    out = _REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / f"suite_{args.config}_{args.split}.txt").write_text("\n".join(lines) + "\n")
    print("\n== DECODABILITY SUITE ==")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
