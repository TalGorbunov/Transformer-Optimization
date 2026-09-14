#!/usr/bin/env python3
"""Filtered-counting text task: question-dependent distractor frames.

Each frame is ONE sighting ("Frame k: <char> is in the <room>."); the question asks
"How many frames show <C> in the <R>?". Frames about characters != C are pure
distractors, and WHICH frames are distractors depends on the question — so no static
attention mask can be optimal, and a relational (question x frame) gate has provable
headroom. Gold is ALWAYS <= gold-max (default 8) at every N: longer sequences add
distractors only, so the length axis isolates distractor-filtering from answer-range
extrapolation. qa.txt grammar matches scripts/condmask/textdata.load_text_sample.
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]
CHARS = ["Sandra", "Mary", "Michael", "John", "Daniel", "Laura", "Peter", "Emma", "Noah"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("steps", "entity", "distinct", "cooc"),
                    default="steps")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--per-gold", type=int, default=100)
    ap.add_argument("--gold-max", type=int, default=8)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crowded", action="store_true",
                    help="steps only: EVERY frame contains all 5 characters (dense "
                         "counting — no frame is droppable; token-level testbed)")
    ap.add_argument("--render", action="store_true",
                    help="also render park-style PNGs (visual filtered-counting)")
    args = ap.parse_args()
    if args.render:
        import sys as _s
        _s.path.insert(0, "datasets/mmred")
        import render_mmred as RM
        RM.ROOMS = ROOMS
        globals()["RM"] = RM
    rng = random.Random(args.seed)
    N = args.seq_len
    gmax = min(args.gold_max, N)
    base = args.out_root / f"seq_len_{N}" / args.split
    made = 0
    gmin = 1 if args.task == "distinct" else 0
    for g in range(gmin, gmax + 1):
        for j in range(args.per_gold):
            C = rng.choice(CHARS)
            R = rng.choice(ROOMS)
            other_rooms = [r for r in ROOMS if r != R]
            others = [c for c in CHARS if c != C]
            frames = []   # list of (chars_tuple, room)
            if args.crowded:
                assert args.task == "steps"
                roster = CHARS[:5]
                if C not in roster:
                    C = rng.choice(roster)
                evid = set(rng.sample(range(N), g))
                states = []
                for k in range(N):
                    occ = {r: [] for r in ROOMS}
                    occ[R if k in evid else rng.choice(other_rooms)].append(C)
                    for o in roster:
                        if o != C:
                            occ[rng.choice(ROOMS)].append(o)
                    states.append(repr({"step_id": k + 1,
                                        "rooms": {r: sorted(occ[r])
                                                  for r in ROOMS if occ[r]}}))
                q = f"How many frames show {C} in the {R}?"
                sid = f"crowded_N{N}_K{g}_{j:04d}"
                d = base / sid
                d.mkdir(parents=True, exist_ok=True)
                (d / "qa.txt").write_text("question:\n" + "\n".join(states)
                                          + f"\n{q}\nanswer:\n{g}\n",
                                          encoding="utf-8")
                made += 1
                continue
            if args.task == "steps":
                n_rel = rng.randint(g, min(args.gold_max, N))
                frames += [((C,), R)] * g
                frames += [((C,), rng.choice(other_rooms))
                           for _ in range(n_rel - g)]
                q = f"How many frames show {C} in the {R}?"
            elif args.task == "entity":
                n_rel = g
                frames += [((C,), rng.choice(ROOMS)) for _ in range(g)]
                q = f"How many frames show {C}?"
            elif args.task == "distinct":
                if g > min(len(ROOMS), args.gold_max):
                    continue
                rooms_g = rng.sample(ROOMS, g)
                n_rel = rng.randint(g, min(args.gold_max, N))
                frames += [((C,), r) for r in rooms_g]
                frames += [((C,), rng.choice(rooms_g))
                           for _ in range(n_rel - g)]
                q = f"How many distinct rooms did {C} appear in across the frames?"
            else:  # cooc — relational relevance: BOTH characters must be present
                D = rng.choice([c for c in CHARS if c != C])
                n_rel = g
                frames += [(tuple(sorted((C, D))), rng.choice(ROOMS))
                           for _ in range(g)]
                q = f"In how many frames are {C} and {D} together?"
                others = [c for c in CHARS if c not in (C, D)]
            while len(frames) < N:   # distractors: singles or pairs of OTHER chars
                if args.task == "cooc" and rng.random() < 0.4:
                    pair = tuple(sorted(rng.sample(others + [C] if rng.random() < 0.3
                                                   else others, 2)))
                    frames.append((pair, rng.choice(ROOMS)))
                else:
                    frames.append(((rng.choice(others),), rng.choice(ROOMS)))
            frames = frames[:N]
            rng.shuffle(frames)
            states = [repr({"step_id": k + 1, "rooms": {room: list(chs)}})
                      for k, (chs, room) in enumerate(frames)]
            sid = f"{args.task}_N{N}_K{g}_{j:04d}"
            d = base / sid
            d.mkdir(parents=True, exist_ok=True)
            (d / "qa.txt").write_text(
                "question:\n" + "\n".join(states) + f"\n{q}\nanswer:\n{g}\n",
                encoding="utf-8")
            if args.render:
                for k, (chs, room) in enumerate(frames):
                    occ = {r: [] for r in ROOMS}
                    occ[room] = list(chs)
                    RM.render_frame(occ, k + 1, str(d / f"{k:03d}.png"))
            made += 1
        print(f"K={g}: {args.per_gold}", flush=True)
    print(f"DONE total={made} -> {base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
