#!/usr/bin/env python3
"""v5 dense-read augmentation: synthetic per-step content QA from gold states.

The content probe (2026-08-03) showed room identity is linearly present in the
carriers (0.814) while the trained direct read lags (0.53 and climbing) — a
supervision-density gap: 1 labeled read per sample vs the probe's per-frame
labels. This generator densifies supervision at the DATA level: for each train
sequence, emit one synthetic direct-QA sample dir PER STEP (canonical
char_at_frame / room_at_frame phrasings so qtype dispatch works unchanged),
frames hardlinked. build_target_v4 turns them into ' answer: v END' targets.

Usage:
  python scripts/mmred_hf/gen_dense_qa.py --configs 2 4 8 16 --rows-per-config 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--rows-per-config", type=int, default=150)
    ap.add_argument("--out", default=str(REPO / "data/mmred_hf/dirs/aug_dense_qa"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    n_dirs = n_link = 0

    for cfg in args.configs:
        rows = json.loads((REPO / f"data/mmred_hf/json/seq_len_{cfg}_train.json").read_text())
        rng.shuffle(rows)
        images = REPO / f"data/mmred_hf/images/seq_len_{cfg}_train"
        used = 0
        for row in rows:
            if used >= args.rows_per_config:
                break
            src = images / row["qid"]
            frames = sorted(src.glob("frame_*.png"))
            if len(frames) != row["seq_len"]:
                continue
            states = [{"rooms": {r: list(c) for r, c in s["rooms"].items()}}
                      for s in row["sequence"]]
            chars = sorted({c for st in states for occ in st["rooms"].values() for c in occ})
            state_lines = [repr(st) for st in states]
            for t in range(row["seq_len"]):
                # char_at_frame variant: rotating character
                char = chars[(t + used) % len(chars)]
                room = next(r for r, occ in states[t]["rooms"].items() if char in occ)
                emit = [(f"char_at_frame_A{room}_{row['qid']}{t:03d}1",
                         f"In which room was {char} at step {t + 1}?", room)]
                # room_at_frame variant: a room with <=1 occupant (unique answer)
                cands = [(r, occ) for r, occ in states[t]["rooms"].items() if len(occ) <= 1]
                if cands:
                    r, occ = cands[(t + used) % len(cands)]
                    ans = occ[0] if occ else "Nobody"
                    emit.append((f"room_at_frame_A{ans}_{row['qid']}{t:03d}2",
                                 f"Who was in the {r} at step {t + 1}?", ans))
                for name, q, ans in emit:
                    d = out_root / name
                    d.mkdir(exist_ok=True)
                    (d / "qa.txt").write_text(
                        "\n".join(["question:"] + state_lines + [q, "answer:", ans]) + "\n",
                        encoding="utf-8")
                    for i, f in enumerate(frames):
                        dst = d / f"{i:03d}.png"
                        if not dst.exists():
                            os.link(f, dst)
                            n_link += 1
                    n_dirs += 1
            used += 1
    print(f"generated {n_dirs} synthetic QA dirs ({n_link} hardlinks) -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
