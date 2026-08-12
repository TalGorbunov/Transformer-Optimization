#!/usr/bin/env python3
"""MMReD-HF sample adapter for the ninv campaign (CPU, login-node OK).

The campaign switched to the ORIGINAL benchmark at data/mmred_hf/dirs/ (DIRECTIVE
2026-08-09). This module parses one HF sample dir into the EXACT tuple contract of
`gnnformer.data.load_mmred_sample` — (sample_id, frames, question, states, answer) —
so the existing capture/probe machinery consumes HF data unchanged.

Why a separate adapter instead of reusing gnnformer.data:
  The HF room vocabulary is Kitchen/Bathroom/Garden/Office/Bedroom/HALLWAY, whereas
  `gnnformer.constants.ROOMS` carries PARK and no Hallway. Editing gnnformer is
  forbidden (campaign rule 2), so this file carries its own ROOMS_HF tuple and every
  entry point takes `rooms` explicitly. Nothing here imports gnnformer.

qa.txt layout (verified on disk 2026-08-09):
    question:
    {'rooms': {'Kitchen': ['Mary', 'Michael'], ..., 'Hallway': []}}   x N frames
    How many steps did Michael spend in the Office?
    answer:
    0

Evidence bit for frame f = "is the queried character in the queried room at frame f".
For steps_in_room the gold answer is exactly the number of set evidence bits, which is
what `self_check` verifies — the adapter is only trustworthy if the labels it derives
reproduce the benchmark's own answers.

Usage:
  python scripts/ninv/load_hf_sample.py --root data/mmred_hf/dirs/seq_len_8_train_steps_in_room
  python scripts/ninv/load_hf_sample.py --root data/mmred_hf/dirs/seq_len_64_test --n 20
  python scripts/ninv/load_hf_sample.py --all          # self-check every pool at once
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# HF room vocabulary — Hallway is present, Park is NOT. Deliberately local to this
# file; do not swap in gnnformer.constants.ROOMS, it is a different world.
ROOMS_HF: Tuple[str, ...] = ("Kitchen", "Bathroom", "Garden", "Office", "Bedroom",
                             "Hallway")

STEPS_RE = re.compile(r"How many steps did\s+(\w+)\s+spend in the\s+([\w ]+?)\s*\?",
                      re.IGNORECASE)

# Every campaign pool filters sample dirs by this prefix: the test/headfit/val dirs
# hold ALL 18 MMReD question types mixed together (1200 dirs, only 50 of them ours).
STEPS_PREFIX = "steps_in_room"


def parse_qa(qa_path: Path) -> Tuple[str, List[Dict[str, Any]], str]:
    """-> (question, states, answer_text). ast.literal_eval only, never eval/json."""
    lines = qa_path.read_text(encoding="utf-8").splitlines()
    q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "question:"), -1)
    a_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "answer:"), -1)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        raise RuntimeError(f"bad qa.txt layout: {qa_path}")
    states: List[Dict[str, Any]] = []
    question: Optional[str] = None
    for ln in lines[q_idx + 1 : a_idx]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
            continue
        question = s
        break
    if question is None:
        raise RuntimeError(f"no question line in {qa_path}")
    answer = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if answer is None:
        raise RuntimeError(f"no answer in {qa_path}")
    return question, states, answer


def load_hf_sample(sample_dir: Path, *, resize: Optional[int] = None):
    """-> (sample_id, frames[PIL], question, states, answer_text).

    Same 5-tuple as gnnformer.data.load_mmred_sample, so callers are interchangeable.
    Frame count is taken from the number of state dicts and cross-checked against the
    PNGs on disk — a mismatch means a truncated sample, which must not silently become
    a short video with stale labels.
    """
    from PIL import Image

    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)
    question, states, answer = parse_qa(sample_dir / "qa.txt")
    n_png = len(list(sample_dir.glob("[0-9][0-9][0-9].png")))
    if n_png != len(states):
        raise RuntimeError(f"{sample_dir.name}: {n_png} PNGs but {len(states)} states")
    frames = [Image.open(sample_dir / f"{i:03d}.png").convert("RGB")
              for i in range(len(states))]
    if resize:
        frames = [f.resize((resize, resize)) for f in frames]
    return sample_dir.name, frames, question, states, answer


def parse_steps_question(question: str, rooms: Sequence[str] = ROOMS_HF
                         ) -> Optional[Tuple[str, str]]:
    """'How many steps did Michael spend in the Office?' -> ('Michael', 'Office')."""
    m = STEPS_RE.search(question)
    if not m:
        return None
    character = m.group(1).strip()
    raw = m.group(2).strip()
    room = next((r for r in rooms if r.lower() == raw.lower()), None)
    return None if room is None else (character, room)


def evidence_bits(question: str, states: Sequence[Dict[str, Any]],
                  rooms: Sequence[str] = ROOMS_HF) -> Optional[List[int]]:
    """Per-frame 0/1: is the queried character in the queried room at that frame."""
    parsed = parse_steps_question(question, rooms)
    if parsed is None:
        return None
    character, room = parsed
    out = []
    for st in states:
        occupants = (st.get("rooms", {}) or {}).get(room, []) if isinstance(st, dict) else []
        out.append(1 if character in occupants else 0)
    return out


def iter_hf_sample_dirs(root: Path, prefix: str = STEPS_PREFIX):
    """Sorted sample dirs under `root` whose name starts with `prefix`.

    The prefix filter is mandatory for test/headfit/val pools: those hold every MMReD
    question type mixed together and only ~50 of 1200 dirs are steps_in_room.
    """
    root = Path(root)
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith(prefix) and (p / "qa.txt").is_file())


def self_check(root: Path, n: int = 10, prefix: str = STEPS_PREFIX, verbose: bool = True):
    """MANDATORY before any HF capture: recompute each answer from the evidence bits
    and assert it equals the qa.txt gold. Returns (n_ok, n_bad, failures, golds).

    Samples with an even STRIDE across the sorted pool, never the first n. Sample dirs
    are named ..._K<evidence_count>_<id>, so sorted order puts every K0 (gold 0) first:
    taking the head would check only all-zero samples, which an adapter that always
    returned zero bits would also pass. The stride guarantees nonzero golds are seen,
    and the caller asserts on the gold distribution.
    """
    pool = iter_hf_sample_dirs(root, prefix)
    if not pool:
        raise SystemExit(f"no '{prefix}*' sample dirs under {root}")
    step = max(1, len(pool) // max(n, 1))
    dirs = pool[::step][:n]
    n_ok, failures, golds = 0, [], []
    for sd in dirs:
        question, states, answer = parse_qa(sd / "qa.txt")
        bits = evidence_bits(question, states)
        if bits is None:
            failures.append((sd.name, "unparseable question", question, answer))
            continue
        recomputed, gold = sum(bits), int(str(answer).strip())
        golds.append(gold)
        if recomputed != gold:
            failures.append((sd.name, f"recomputed {recomputed} != gold {gold}",
                             question, "".join(str(b) for b in bits)))
        else:
            n_ok += 1
            if verbose:
                print(f"  OK {sd.name:<34} N={len(states):<4} gold={gold:<3} "
                      f"bits={''.join(str(b) for b in bits)}")
    return n_ok, len(failures), failures, golds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--prefix", default=STEPS_PREFIX)
    ap.add_argument("--all", action="store_true",
                    help="self-check every steps_in_room pool under data/mmred_hf/dirs")
    args = ap.parse_args()

    roots = ([p for p in sorted(Path("data/mmred_hf/dirs").iterdir())
              if p.is_dir() and iter_hf_sample_dirs(p, args.prefix)]
             if args.all else [Path(args.root)])
    if not args.all and args.root is None:
        raise SystemExit("pass --root or --all")

    total_ok = total_bad = 0
    degenerate = []
    for root in roots:
        pool = iter_hf_sample_dirs(root, args.prefix)
        print(f"\n=== {root}  ({len(pool)} {args.prefix}* samples) ===")
        ok, bad, failures, golds = self_check(root, args.n, args.prefix,
                                              verbose=not args.all)
        total_ok += ok
        total_bad += bad
        for name, why, q, extra in failures:
            print(f"  FAIL {name}: {why}\n       q={q!r}\n       {extra}")
        nz = sum(1 for g in golds if g > 0)
        # An all-zero check set is VACUOUS: an adapter that always returned zero bits
        # would pass it. Flag it rather than report a green tick.
        flag = "  <- VACUOUS: no nonzero gold checked" if golds and nz == 0 else ""
        print(f"  checked {ok + bad}: {ok} ok, {bad} MISMATCH   "
              f"golds {sorted(set(golds))} ({nz}/{len(golds)} nonzero){flag}")
        if golds and nz == 0:
            degenerate.append(str(root))
    print(f"\nTOTAL: {total_ok} ok, {total_bad} mismatch")
    if degenerate:
        print(f"VACUOUS pools (checked only gold=0): {degenerate}")
    if total_bad:
        print("SELF-CHECK FAILED — do not capture on this data; write STATE and stop.")
    return 1 if (total_bad or degenerate) else 0


if __name__ == "__main__":
    raise SystemExit(main())
