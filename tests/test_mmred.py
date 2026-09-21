"""CPU tests for core.mmred: the official benchmark's vocabulary, gold recomputation (parity
with the published answers), the evidence-label rule, and the loader's class interleaving.

Run: python tests/test_mmred.py   (gold parity needs data/mmred_hf/json; skips otherwise)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.constants import CHARS, NOBODY, ROOMS
from core.mmred import (
    DC_QTYPES,
    NIAH_QTYPES,
    QTYPES,
    evidence_frames,
    load_split,
    recompute_answer,
    states,
    stratified_order,
)

DATA = _REPO / "data" / "mmred_hf"


def _st(*placements):
    """One state per argument: {char: room}. Every char is in exactly one room (upstream vgen)."""
    out = []
    for pl in placements:
        rooms = {r: [] for r in ROOMS}
        for c, r in pl.items():
            rooms[r].append(c)
        out.append({"rooms": rooms})
    return out


# --- vocabulary -------------------------------------------------------------------------

def test_vocabulary_matches_upstream():
    try:
        from mmred.const import CHARS as U_CHARS, NOBODY as U_NOBODY, ROOMS as U_ROOMS  # type: ignore
    except ImportError as exc:
        print(f"  [skip] upstream mmred package not importable here: {exc}")
        return
    assert list(U_ROOMS) == ROOMS and list(U_CHARS) == CHARS and U_NOBODY == NOBODY


def test_qtypes_are_the_24_upstream_names():
    assert len(QTYPES) == 24 and len(set(QTYPES)) == 24
    assert len(NIAH_QTYPES) == 15 and len(DC_QTYPES) == 9
    try:
        from mmred.qgen.questions import QUESTIONS  # type: ignore
        assert set(QUESTIONS.keys()) == set(QTYPES)
    except ImportError as exc:
        print(f"  [skip] upstream questions registry not importable: {exc}")


# --- gold parity (the validity test of our reading of every task) -----------------------

def test_gold_parity():
    """recompute_answer must reproduce the PUBLISHED answer for every row of seq 8 + 16 test
    (50 per qtype each). 100 %, no tolerance."""
    files = [DATA / "json" / f"seq_len_{n}_test.json" for n in (8, 16)]
    if not all(f.exists() for f in files):
        print("  [skip] data/mmred_hf/json not present")
        return
    n = bad = 0
    for f in files:
        for row in json.loads(f.read_text()):
            n += 1
            got = recompute_answer(row["qtype"], row["question"], states(row))
            if str(got) != str(row["answer"]):
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH {row['qtype']} qid={row['qid']}: got {got!r} want {row['answer']!r}")
    assert bad == 0, f"{bad}/{n} mismatches"
    print(f"  gold parity: {n}/{n}")


# --- the evidence rule ------------------------------------------------------------------

S = _st(
    {"Daniel": "Kitchen", "John": "Garden", "Mary": "Kitchen", "Michael": "Office", "Sandra": "Hallway"},
    {"Daniel": "Garden", "John": "Garden", "Mary": "Kitchen", "Michael": "Office", "Sandra": "Hallway"},
    {"Daniel": "Kitchen", "John": "Kitchen", "Mary": "Kitchen", "Michael": "Bedroom", "Sandra": "Hallway"},
    {"Daniel": "Bathroom", "John": "Kitchen", "Mary": "Garden", "Michael": "Bedroom", "Sandra": "Hallway"},
)


def test_evidence_dense():
    assert evidence_frames("steps_in_room", "How many steps did Daniel spend in the Kitchen?", S) == {0, 2}
    assert evidence_frames("steps_in_room", "How many steps did Sandra spend in the Garden?", S) == set()
    assert evidence_frames("where_spend", "In which room did Daniel spend the most time?", S) == {0, 1, 2, 3}
    assert evidence_frames("rooms_visited", "How many different rooms did Mary visit?", S) == {0, 1, 2, 3}
    # crowd (3 or more in one room): only frame 2 (Kitchen has 3)
    assert evidence_frames("crowd_count", "For how many steps was there a crowd (3 or more people in one room)?", S) == {2}
    assert evidence_frames("crowded_room", "Which room was crowded (3 or more people in one room) for the most steps?", S) == {2}
    # who_spend(R): frames where R is occupied
    assert evidence_frames("who_spend", "Who spent the most time in the Garden?", S) == {0, 1, 3}


def test_evidence_positional():
    assert evidence_frames("char_at_frame", "In which room was Daniel at step 3?", S) == {2}
    assert evidence_frames("room_at_frame", "Who was in the Kitchen at step 1?", S) == {0}
    assert evidence_frames("n_empty", "How many rooms were empty at step 4?", S) == {3}
    assert evidence_frames("first_app", "In which room did Daniel first appear?", S) == {0}
    assert evidence_frames("final_app", "In which room was Daniel at the final step?", S) == {3}


def test_evidence_trigger():
    # "when John first appeared in the Kitchen": every frame with John in Kitchen; the readout picks the first
    assert evidence_frames("char_on_char_first_app",
                           "In which room was Daniel when John first appeared in the Kitchen?", S) == {2, 3}
    assert evidence_frames("n_room_on_char_final_app",
                           "How many characters were in the Kitchen when John made their final appearance in the Kitchen?", S) == {2, 3}
    assert evidence_frames("first_at_room", "Who was the first to appear in the Bathroom?", S) == {3}
    assert evidence_frames("last_at_room", "Who was the last to appear in the Bedroom?", S) == {2, 3}


def test_evidence_every_qtype_returns_a_set_or_none():
    q = {
        "first_app": "In which room did Mary first appear?", "final_app": "In which room was Mary at the final step?",
        "char_at_frame": "In which room was Mary at step 2?", "n_empty": "How many rooms were empty at step 1?",
        "steps_in_room": "How many steps did Mary spend in the Kitchen?", "where_spend": "In which room did Mary spend the least amount of time?",
        "spend_alone": "Who spent the most time alone in the rooms?", "room_empty": "Which room was empty for more steps than the other rooms?",
        "spend_together": "With whom did Mary spend the most time together in the same room?",
    }
    for qt, qq in q.items():
        ev = evidence_frames(qt, qq, S)
        assert ev is None or (isinstance(ev, set) and all(0 <= t < len(S) for t in ev)), qt


# --- loader ------------------------------------------------------------------------------

def test_load_split_filters_qtypes():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); (root / "json").mkdir()
        rows = [{"qid": f"{i:07d}", "seq_len": 2, "qtype": qt, "atype": "number", "question": "q", "answer": "0",
                 "sequence": [{"step_id": 1, "rooms": {r: [] for r in ROOMS}}, {"step_id": 2, "rooms": {r: [] for r in ROOMS}}]}
                for i, qt in enumerate(["steps_in_room", "n_empty", "steps_in_room"])]
        (root / "json" / "seq_len_2_test.json").write_text(json.dumps(rows))
        assert len(load_split("seq_len_2", "test", root)) == 3
        assert [r["qtype"] for r in load_split("seq_len_2", "test", root, qtypes=["n_empty"])] == ["n_empty"]
        assert len(states(rows[0])) == 2 and set(states(rows[0])[0]["rooms"]) == set(ROOMS)


def test_stratified_order():
    """HF rows arrive answer-sorted (all zeros first); any head slice must mix classes."""
    rows = [{"qid": i, "answer": "0"} for i in range(50)] + [{"qid": 50 + i, "answer": "1"} for i in range(10)] \
        + [{"qid": 60 + i, "answer": "2"} for i in range(5)]
    out = stratified_order(rows, seed=0)
    assert sorted(r["qid"] for r in out) == list(range(65)), "a permutation"
    head = [r["answer"] for r in out[:10]]
    assert len(set(head)) >= 2, f"head slice is single-class: {head}"
    assert stratified_order(rows, seed=0) == out, "deterministic in the seed"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
