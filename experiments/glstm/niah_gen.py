#!/usr/bin/env python3
"""C3 (2026-07-19): NIAH/which-frame task generation from existing MMRED states. CPU only.

For each source sample dir, find (char, room) pairs present in EXACTLY ONE frame and emit
"In which frame number (1-N) was {C} in the {R}?" with the 1-indexed frame as the integer
answer — digit-protocol compatible, and scratchpad-consistent (" frames k -> k"). Frames are
symlinked; qa.txt rewritten with the same states. Answers greedily balanced across 1..N.
Dir names niah_N{N}_K{ans}_{i:04d} so iter_sample_dirs_shuffled stratifies on K.
"""
from __future__ import annotations
import argparse, ast, os, sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--out", default="data/mmred_niah_which/seq_len_8/all_uniform")
    ap.add_argument("--per-answer", type=int, default=90)
    args = ap.parse_args()
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    made = Counter(); idx = 0
    for sd in iter_sample_dirs(Path(args.src)):
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        N = len(states)
        # (char, room) -> frames where present
        occ = {}
        for t, st in enumerate(states):
            for rname, chars in (st.get("rooms", {}) or {}).items():
                for c in chars:
                    occ.setdefault((c, rname), []).append(t)
        singles = [(c, r, fr[0]) for (c, r), fr in occ.items() if len(fr) == 1]
        if not singles:
            continue
        # pick the singleton whose answer class is currently least filled
        singles.sort(key=lambda x: (made[x[2] + 1], x[2]))
        c, r, t = singles[0]
        ans = t + 1
        if made[ans] >= args.per_answer:
            continue
        made[ans] += 1
        name = f"niah_N{N}_K{ans}_{idx:04d}"; idx += 1
        dd = out_root / name; dd.mkdir(exist_ok=True)
        for i in range(N):
            dst = dd / f"{i:03d}.png"
            if not dst.exists():
                os.symlink(os.path.relpath(sd / f"{i:03d}.png", dd), dst)
        q = f"In which frame number (1-{N}) was {c} in the {r}?"
        lines = [f"qid: {name}", "qtype: which_frame", "atype: integer",
                 f"seq_len: {N}", "question:"]
        lines += [repr(st) for st in states]
        lines += [q, "answer:", str(ans)]
        (dd / "qa.txt").write_text("\n".join(lines) + "\n")
        if all(made[a] >= args.per_answer for a in range(1, N + 1)):
            break
    print(f"generated {sum(made.values())} samples: "
          + " ".join(f"K{a}:{made[a]}" for a in sorted(made)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
