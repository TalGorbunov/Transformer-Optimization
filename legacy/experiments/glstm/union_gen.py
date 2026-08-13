#!/usr/bin/env python3
"""Discretionary composition eval (2026-07-19, pre-approved list): OR-UNION counting.

"How many frames was {C} in the {R1} or the {R2}?" — a reduction over TWO per-frame
predicates that was never in any training mixture; digit- and scratchpad-compatible
(evid = union frames; answer = |union|). Generated from existing states, frames symlinked,
answers greedily balanced over 0..8. Dir names union_N{N}_K{ans}_{i:04d}.
"""
from __future__ import annotations
import argparse, os, sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--out", default="data/mmred_union_or/seq_len_8/all_uniform")
    ap.add_argument("--per-answer", type=int, default=60)
    args = ap.parse_args()
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    made = Counter(); idx = 0
    for sd in iter_sample_dirs(Path(args.src)):
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        N = len(states)
        chars = sorted({c for st in states for occ in (st.get("rooms", {}) or {}).values()
                        for c in occ})
        rooms = sorted((states[0].get("rooms", {}) or {}).keys())
        cands = []
        for c in chars:
            per_room = {r: {t for t, st in enumerate(states)
                            if c in ((st.get("rooms", {}) or {}).get(r, []) or [])}
                        for r in rooms}
            for i1 in range(len(rooms)):
                for i2 in range(i1 + 1, len(rooms)):
                    u = per_room[rooms[i1]] | per_room[rooms[i2]]
                    cands.append((c, rooms[i1], rooms[i2], len(u)))
        if not cands:
            continue
        cands.sort(key=lambda x: made[x[3]])
        c, r1, r2, ans = cands[0]
        if made[ans] >= args.per_answer:
            continue
        made[ans] += 1
        name = f"union_N{N}_K{ans}_{idx:04d}"; idx += 1
        dd = out_root / name; dd.mkdir(exist_ok=True)
        for i in range(N):
            dst = dd / f"{i:03d}.png"
            if not dst.exists():
                os.symlink(os.path.relpath(sd / f"{i:03d}.png", dd), dst)
        q = f"How many frames was {c} in the {r1} or the {r2}?"
        lines = [f"qid: {name}", "qtype: union_or", "atype: integer",
                 f"seq_len: {N}", "question:"]
        lines += [repr(st) for st in states]
        lines += [q, "answer:", str(ans)]
        (dd / "qa.txt").write_text("\n".join(lines) + "\n")
    print(f"generated {sum(made.values())} samples: "
          + " ".join(f"K{a}:{made[a]}" for a in sorted(made)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
