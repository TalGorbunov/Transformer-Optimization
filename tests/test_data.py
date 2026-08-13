"""CPU tests for MMRED task parsing (gnnformer.data) on synthetic states, with
legacy-parity checks against the frozen implementations.

Run: .venv/bin/python tests/test_data.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import ROOMS
from gnnformer.data import frame_attr_labels, parse_task_labels, probe_evidence

# 8 frames; Alice: Kitchen@{1,4,7}, Garden@{0}; Bob: Kitchen@{4}, Office elsewhere
STATES = []
for t in range(8):
    rooms = {}
    if t in (1, 4, 7):
        rooms.setdefault("Kitchen", []).append("Alice")
    elif t == 0:
        rooms.setdefault("Garden", []).append("Alice")
    else:
        rooms.setdefault("Park", []).append("Alice")
    if t == 4:
        rooms.setdefault("Kitchen", []).append("Bob")
    else:
        rooms.setdefault("Office", []).append("Bob")
    STATES.append({"rooms": rooms})

CASES = [
    ("How many steps did Alice spend in the Kitchen?", 3, "steps", {1, 4, 7}),
    ("were Alice and Bob in the same room in these frames?", 1, "cooc", {4}),
    ("How many frames was Alice in the Kitchen or the Garden?", 4, "union", {0, 1, 4, 7}),
    ("In which frame number (1-8) was Bob in the Kitchen?", 5, "which", {4}),
    ("How many distinct rooms did Alice visit?", 3, "rooms", {0, 1, 2, 3, 4, 5, 6, 7}),
]


def test_parse_task_labels():
    for q, gold, want_task, want_evid in CASES:
        got = parse_task_labels(q, STATES, gold)
        assert got is not None, q
        task, evid, aux = got
        assert task == want_task and evid == want_evid, (q, got)
    # sanity gates: cooc/union/rooms verify derived evidence against gold (steps does NOT —
    # legacy behavior, pinned by test_legacy_parity); empty-evidence steps with gold>0 -> None
    assert parse_task_labels(CASES[1][0], STATES, 2) is None
    assert parse_task_labels(CASES[2][0], STATES, 3) is None
    assert parse_task_labels("How many steps did Carol spend in the Kitchen?", STATES, 2) is None


def test_frame_attr_labels():
    task, evid, _aux = parse_task_labels(CASES[0][0], STATES, 3)
    labels = frame_attr_labels(task, CASES[0][0], STATES, evid)
    assert labels == {1: "Kitchen", 4: "Kitchen", 7: "Kitchen"}
    task, evid, _aux = parse_task_labels(CASES[2][0], STATES, 4)
    labels = frame_attr_labels("union", CASES[2][0], STATES, evid)
    assert labels[0] == "Garden" and labels[1] == "Kitchen"


def test_probe_evidence():
    got = probe_evidence("steps", CASES[0][0], STATES, 3, ROOMS)
    assert got == ({1, 4, 7}, "Kitchen")
    got = probe_evidence("cooc", CASES[1][0], STATES, 1, ROOMS)
    assert got == ({4}, "Bob")
    assert probe_evidence("cooc", CASES[1][0], STATES, 2, ROOMS) is None  # gold mismatch


def test_legacy_parity():
    sys.path.insert(0, str(_REPO / "legacy"))
    try:
        from experiments.glstm.carrier_layer_lora import (  # type: ignore
            frame_attr_labels as l_fal,
            parse_task_labels as l_ptl,
        )
    except Exception as exc:
        print(f"  [skip] legacy import unavailable: {exc}")
        return
    for q, gold, _t, _e in CASES:
        ours = parse_task_labels(q, STATES, gold)
        theirs = l_ptl(q, STATES, gold)
        assert ours == theirs, (q, ours, theirs)
        task, evid, _aux = ours
        assert frame_attr_labels(task, q, STATES, evid) == l_fal(task, q, STATES, evid)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
