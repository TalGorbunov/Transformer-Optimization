#!/usr/bin/env python3
"""Generate COUNT-BALANCED MMReD datasets for rooms_visited and co_occupancy.

The base generator only controls the steps `evidence_count`; co_occ / rooms_visited counts fall out of
random filler placement (→ heavily skewed, e.g. co_occ ~50% zeros). This script instead *targets the
derived count* and sweeps it UNIFORMLY, so every possible count has equal support:
  rooms_visited : designate target C, place C in exactly K distinct rooms across N frames, K∈1..min(6,N).
  co_occupancy  : designate pair (C,D), place them in the same room in exactly K of N frames, K∈0..N.
Other characters are placed at random each frame (they are not the queried entity). The same scenes are
emitted as BOTH text frames (qa.txt state dicts) and rendered PNG frames, so an image-vs-text comparison
differs only in modality.

The designated target is written to metadata (`query_character` / `query_pair`); a matching patch in
`eval_mmred_text_frames_acc.question_and_gold` reads it (falling back to random for legacy data) so gold
== the engineered count K. Output layout matches the park dataset: <out>/seq_len_N/<split>/<id>/.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for render_mmred
import render_mmred as R

PARK_ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]
# 9 visually-distinct (colored) characters — extended 2026-06-24 for distinct-over-character tasks
CHARACTERS = ["Sandra", "Mary", "Michael", "John", "Daniel", "Laura", "Peter", "Emma", "Noah"]


def states_rooms_visited(rng, seq_len, target, K, rooms, chars):
    """Target C visits exactly K distinct rooms across seq_len frames."""
    chosen = rng.sample(rooms, K)
    frame_rooms = list(chosen) + [rng.choice(chosen) for _ in range(seq_len - K)]
    rng.shuffle(frame_rooms)
    others = [c for c in chars if c != target]
    states = []
    for i in range(seq_len):
        occ = {r: [] for r in rooms}
        occ[frame_rooms[i]].append(target)
        for o in others:
            occ[rng.choice(rooms)].append(o)
        states.append({"step_id": i + 1, "rooms": {r: sorted(occ[r]) for r in rooms}})
    return states


def states_steps(rng, seq_len, C, R, K, rooms, chars):
    """Target C is in room R in exactly K of seq_len frames (in a non-R room otherwise)."""
    evidence = set(rng.sample(range(seq_len), K))
    others = [c for c in chars if c != C]
    other_rooms = [r for r in rooms if r != R]
    states = []
    for i in range(seq_len):
        occ = {r: [] for r in rooms}
        occ[R if i in evidence else rng.choice(other_rooms)].append(C)
        for o in others:
            occ[rng.choice(rooms)].append(o)
        states.append({"step_id": i + 1, "rooms": {r: sorted(occ[r]) for r in rooms}})
    return states


def states_co_occupancy(rng, seq_len, C, D, K, rooms, chars):
    """C and D share a room in exactly K of seq_len frames (different rooms in the rest)."""
    same = set(rng.sample(range(seq_len), K))
    others = [c for c in chars if c not in (C, D)]
    states = []
    for i in range(seq_len):
        occ = {r: [] for r in rooms}
        if i in same:
            occ[rng.choice(rooms)].extend([C, D])
        else:
            r1, r2 = rng.sample(rooms, 2)
            occ[r1].append(C); occ[r2].append(D)
        for o in others:
            occ[rng.choice(rooms)].append(o)
        states.append({"step_id": i + 1, "rooms": {r: sorted(occ[r]) for r in rooms}})
    return states


def states_distinct_visitors(rng, seq_len, R, K, rooms, roster):
    """Exactly K distinct characters EVER appear in room R (gold = K). Non-visitors never enter R."""
    visitors = rng.sample(roster, K); nonvis = [c for c in roster if c not in visitors]
    other_rooms = [r for r in rooms if r != R]
    vframes = {v: set(rng.sample(range(seq_len), rng.randint(1, seq_len))) for v in visitors}
    states = []
    for i in range(seq_len):
        occ = {r: [] for r in rooms}
        for v in visitors:
            occ[R if i in vframes[v] else rng.choice(other_rooms)].append(v)
        for o in nonvis:
            occ[rng.choice(other_rooms)].append(o)  # never in R
        states.append({"step_id": i + 1, "rooms": {r: sorted(occ[r]) for r in rooms}})
    return states


def states_distinct_companions(rng, seq_len, C, K, rooms, roster):
    """Exactly K distinct OTHER characters share C's room at some frame (gold = K)."""
    others = [c for c in roster if c != C]
    companions = rng.sample(others, K); noncomp = [c for c in others if c not in companions]
    c_rooms = [rng.choice(rooms) for _ in range(seq_len)]
    cframes = {comp: set(rng.sample(range(seq_len), rng.randint(1, seq_len))) for comp in companions}
    states = []
    for i in range(seq_len):
        occ = {r: [] for r in rooms}; occ[c_rooms[i]].append(C)
        notc = [r for r in rooms if r != c_rooms[i]]
        for comp in companions:
            occ[c_rooms[i] if i in cframes[comp] else rng.choice(notc)].append(comp)
        for nc in noncomp:
            occ[rng.choice(notc)].append(nc)  # never share C's room
        states.append({"step_id": i + 1, "rooms": {r: sorted(occ[r]) for r in rooms}})
    return states


def write_sample(out_dir, sample_id, task, states, gold, query, rooms, render):
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = len(states)
    frame_paths = [f"{i:03d}.png" for i in range(seq_len)]
    if task == "rooms_visited":
        C = query; nlq = f"How many distinct rooms did {C} visit across the {seq_len} frames?"
        meta_extra = {"query_character": C, "target_character": C}
    elif task == "steps_in_room":
        C, RT = query; nlq = f"How many steps did {C} spend in the {RT}?"
        meta_extra = {"target_character": C, "target_room": RT}
    elif task == "distinct_visitors":
        RT = query; nlq = f"How many distinct characters appeared in the {RT} across the {seq_len} frames?"
        meta_extra = {"query_room": RT, "target_room": RT}
    elif task == "distinct_companions":
        C = query; nlq = f"How many distinct other characters shared a room with {C} across the {seq_len} frames?"
        meta_extra = {"query_character": C, "target_character": C}
    else:
        C, D = query; nlq = f"In how many of the {seq_len} frames were {C} and {D} in the same room?"
        meta_extra = {"query_pair": [C, D], "target_character": C}
    ex = {
        "qid": sample_id, "qtype": task, "atype": "integer", "seq_len": seq_len,
        "question": "\n".join(repr(s) for s in states) + "\n" + nlq + "\n",
        "answer": gold,
    }
    R.write_qa_txt(str(out_dir), ex)
    meta = {"answer": gold, "seq_len": seq_len, "rooms": rooms, "split": "all_uniform",
            "sample_id": sample_id, "qtype": task, "frame_paths": frame_paths,
            "legacy_frame_paths": frame_paths, "question": nlq, **meta_extra}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if render:
        for i, st in enumerate(states):
            R.render_frame(st["rooms"], st["step_id"], str(out_dir / frame_paths[i]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rooms_visited", "co_occupancy", "steps_in_room",
                                       "distinct_visitors", "distinct_companions"], required=True)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--per-count", type=int, default=120)
    ap.add_argument("--n-chars", type=int, default=len(CHARACTERS),
                    help="total characters present per scene (>=1 for rooms/steps, >=2 for co_occ). Lower = less crowding.")
    ap.add_argument("--rooms", default=",".join(PARK_ROOMS),
                    help="comma-separated room subset (fewer rooms = simpler scene). Default = all 6 park rooms.")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--split", default="all_uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true", help="skip PNGs (text-only / format check)")
    args = ap.parse_args()

    ROOMS_USE = [r.strip() for r in str(args.rooms).split(",") if r.strip()]
    R.ROOMS = ROOMS_USE  # renderer reads module-global ROOMS for the grid/normalization
    rng = random.Random(args.seed)
    N = args.seq_len
    if args.task == "rooms_visited":
        counts = list(range(1, min(len(ROOMS_USE), N) + 1))
    elif args.task == "distinct_visitors":
        counts = list(range(0, min(len(CHARACTERS), N) + 1))        # 0..min(roster,N) distinct chars in R
    elif args.task == "distinct_companions":
        counts = list(range(0, min(len(CHARACTERS) - 1, N) + 1))    # 0..min(roster-1,N) distinct companions
    else:
        counts = list(range(0, N + 1))
    base = args.out_root / f"seq_len_{N}" / args.split
    from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
    n_made = 0
    for K in counts:
        for j in range(args.per_count):
            sid = f"{args.task}_N{N}_K{K}_{j:04d}"
            def chars_with(targets):  # targets + filler up to n_chars, clamped to roster
                rest = [c for c in CHARACTERS if c not in targets]
                k = max(0, min(args.n_chars - len(targets), len(rest)))
                return list(targets) + rng.sample(rest, k)
            if args.task == "steps_in_room":
                C = rng.choice(CHARACTERS); RT = rng.choice(ROOMS_USE)
                states = states_steps(rng, N, C, RT, K, ROOMS_USE, chars_with([C]))
                g = sum(1 for st in states if C in st["rooms"][RT])
                assert g == K, f"steps gold {g} != {K}"
                write_sample(base / sid, sid, args.task, states, K, (C, RT), ROOMS_USE, not args.no_render)
            elif args.task == "rooms_visited":
                C = rng.choice(CHARACTERS)
                states = states_rooms_visited(rng, N, C, K, ROOMS_USE, chars_with([C]))
                assert rv.rooms_visited(states, C) == K, f"rooms gold {rv.rooms_visited(states,C)} != {K}"
                write_sample(base / sid, sid, args.task, states, K, C, ROOMS_USE, not args.no_render)
            elif args.task == "distinct_visitors":
                RT = rng.choice(ROOMS_USE)
                states = states_distinct_visitors(rng, N, RT, K, ROOMS_USE, CHARACTERS)
                g = len({c for st in states for c in st["rooms"][RT]})
                assert g == K, f"distinct_visitors gold {g} != {K}"
                write_sample(base / sid, sid, args.task, states, K, RT, ROOMS_USE, not args.no_render)
            elif args.task == "distinct_companions":
                C = rng.choice(CHARACTERS)
                def comp_count(states, C):
                    s = set()
                    for st in states:
                        cr = next((r for r in ROOMS_USE if C in st["rooms"][r]), None)
                        if cr is not None: s.update(x for x in st["rooms"][cr] if x != C)
                    return len(s)
                states = states_distinct_companions(rng, N, C, K, ROOMS_USE, CHARACTERS)
                g = comp_count(states, C)
                assert g == K, f"distinct_companions gold {g} != {K}"
                write_sample(base / sid, sid, args.task, states, K, C, ROOMS_USE, not args.no_render)
            else:
                C, D = rng.sample(CHARACTERS, 2)
                states = states_co_occupancy(rng, N, C, D, K, ROOMS_USE, chars_with([C, D]))
                assert rv.co_occupancy(states, C, D) == K, f"cooc gold != {K}"
                write_sample(base / sid, sid, args.task, states, K, (C, D), ROOMS_USE, not args.no_render)
            n_made += 1
        print(f"K={K}: {args.per_count} samples", flush=True)
    print(f"DONE task={args.task} total={n_made} -> {base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
