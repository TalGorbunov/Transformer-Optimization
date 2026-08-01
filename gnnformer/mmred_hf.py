"""Adapter for the ORIGINAL MMReD benchmark (HF `ef1e43ce/mmred` + Fr0do/mmred renders).

Bridges the upstream layout to our sample interface:
  JSON rows (scripts/mmred_hf/prep.py output, data/mmred_hf/json/<config>_<split>.json)
  + rendered frames (data/mmred_hf/images/<config>_<split>/<qid>/frame_%04d.png)
  -> (sample_id, frames[PIL], question, states, answer_text)

states use our convention: one dict per frame with {"rooms": {room: [chars]}} — all six
rooms explicit (empty list = empty room), directly compatible with rooms_to_room2chars.

`recompute_answer` re-derives the gold answer for ALL 24 upstream qtypes from the states
alone; the parity test (tests/test_mmred_hf_adapter.py) asserts it equals the published
answer — this validates both the state parsing and our reading of each task's semantics.
Dispatch is by the row's `qtype` (never by question-regex sniffing: several templates
are prefix-ambiguous, e.g. final_app vs char_at_frame).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

NOBODY = "Nobody"

# ------------------------------------------------------------------ loading

def load_index(json_path: Path) -> List[Dict[str, Any]]:
    """Rows of a prepped config_split JSON: {qid, seq_len, qtype, atype, question,
    answer, sequence:[{step_id, rooms}]}."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def row_states(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Upstream sequence -> our per-frame state dicts (step order asserted by prep)."""
    return [{"rooms": {r: list(c) for r, c in s["rooms"].items()}} for s in row["sequence"]]


def load_mmred_hf_sample(row: Dict[str, Any], images_root: Path):
    """-> (sample_id, frames[PIL], question, states, answer_text)."""
    states = row_states(row)
    qid = row["qid"]
    frames = [
        Image.open(Path(images_root) / qid / f"frame_{i + 1:04d}.png").convert("RGB")
        for i in range(len(states))
    ]
    return qid, frames, row["question"], states, str(row["answer"])


# ------------------------------------------------------------------ state helpers

def _rooms(st: Dict[str, Any]) -> Dict[str, List[str]]:
    return st.get("rooms", {}) or {}


def _char_room(st: Dict[str, Any], char: str) -> Optional[str]:
    for room, occ in _rooms(st).items():
        if char in occ:
            return room
    return None


def _room_names(states: Sequence[Dict[str, Any]]) -> List[str]:
    return list(_rooms(states[0]).keys()) if states else []


def _single_occupant(st: Dict[str, Any], room: str) -> Optional[str]:
    occ = _rooms(st).get(room, [])
    if len(occ) == 0:
        return NOBODY
    if len(occ) == 1:
        return occ[0]
    return None  # generator guarantees uniqueness; None -> parity mismatch


def _app_step(states, char: str, room: str, final: bool) -> Optional[int]:
    """First/last frame index where char is in room."""
    hits = [t for t, st in enumerate(states) if char in _rooms(st).get(room, [])]
    if not hits:
        return None
    return hits[-1] if final else hits[0]


def _argcmp(counts: Dict[str, int], want_max: bool) -> Optional[str]:
    """Unique arg-max/min or None on tie (generator guarantees uniqueness)."""
    if not counts:
        return None
    best = max(counts.values()) if want_max else min(counts.values())
    winners = [k for k, v in counts.items() if v == best]
    return winners[0] if len(winners) == 1 else None


# ------------------------------------------------------------------ question parsing

_P = {
    "first_app": re.compile(r"In which room did (\w+) first appear"),
    "final_app": re.compile(r"In which room was (\w+) at the final step"),
    "char_on_char_first_app": re.compile(r"In which room was (\w+) when (\w+) first appeared in the (\w+)"),
    "char_on_char_final_app": re.compile(r"In which room was (\w+) when (\w+) made their final appearance in the (\w+)"),
    "char_at_frame": re.compile(r"In which room was (\w+) at step (\d+)"),
    "first_at_room": re.compile(r"Who was the first to appear in the (\w+)"),
    # published HF data says "last to appear"; repo HEAD reworded to "last person seen"
    # (same computation) — accept both
    "last_at_room": re.compile(r"Who was the last (?:person seen|to appear) in the (\w+)"),
    "room_on_char_first_app": re.compile(r"Who was in the (\w+) when (\w+) first appeared in the (\w+)"),
    "room_on_char_final_app": re.compile(r"Who was in the (\w+) when (\w+) made their final appearance in the (\w+)"),
    "room_at_frame": re.compile(r"Who was in the (\w+) at step (\d+)"),
    "char_on_char_at_frame": re.compile(r"Who was in the same room as (\w+) at step (\d+)"),
    "n_room_on_char_first_app": re.compile(r"How many characters were in the (\w+) when (\w+) first appeared in the (\w+)"),
    "n_room_on_char_final_app": re.compile(r"How many characters were in the (\w+) when (\w+) made their final appearance in the (\w+)"),
    "n_char_at_frame": re.compile(r"How many other characters were in the same room as (\w+) at step (\d+)"),
    "n_empty": re.compile(r"How many rooms were empty at step (\d+)"),
    "room_empty": re.compile(r"Which room was empty for (more|fewer) steps than the other rooms"),
    "where_spend": re.compile(r"In which room did (\w+) spend the (most|least amount of) time"),
    "crowded_room": re.compile(r"Which room was crowded \((\d+) or more people in one room\) for the most steps"),
    # published wording says "time alone in the <R>" (reworded at HEAD, same
    # occupancy-count computation); "alone" optional. spend_alone is the one that
    # ends "in the rooms" — dispatch is by qtype so there is no clash.
    "who_spend": re.compile(r"Who spent the (most|least amount of) time (?:alone )?in the (\w+)"),
    "spend_alone": re.compile(r"Who spent the (most|least amount of) time alone in the rooms"),
    "spend_together": re.compile(r"With whom did (\w+) spend the (most|least amount of) time together in the same room"),
    "steps_in_room": re.compile(r"How many steps did (\w+) spend in the (\w+)"),
    "rooms_visited": re.compile(r"How many different rooms did (\w+) visit"),
    # published: "How many times did a crowd (...) appear?"; HEAD: "For how many steps
    # was there a crowd (...)". Same computation (steps with >=1 crowded room).
    "crowd_count": re.compile(
        r"(?:For how many steps was there a crowd|How many times did a crowd) "
        r"\((\d+) or more people in one room\)"),
}

NIAH_QTYPES = [
    "first_app", "final_app", "char_on_char_first_app", "char_on_char_final_app",
    "char_at_frame", "first_at_room", "last_at_room", "room_on_char_first_app",
    "room_on_char_final_app", "room_at_frame", "char_on_char_at_frame",
    "n_room_on_char_first_app", "n_room_on_char_final_app", "n_char_at_frame", "n_empty",
]
DC_QTYPES = [
    "room_empty", "where_spend", "crowded_room", "who_spend", "spend_alone",
    "spend_together", "steps_in_room", "rooms_visited", "crowd_count",
]


def _match(qtype: str, question: str):
    m = _P[qtype].search(question)
    if not m:
        raise ValueError(f"question does not match {qtype} template: {question!r}")
    return m.groups()


# ------------------------------------------------------------------ answer recompute

def recompute_answer(qtype: str, question: str, states: Sequence[Dict[str, Any]]):
    """Re-derive the gold answer from states (str for person/room, int for number).
    Returns None when a generator uniqueness guarantee does not hold (parity failure)."""
    g = _match(qtype, question)
    rooms = _room_names(states)

    if qtype == "first_app":
        return _char_room(states[0], g[0])
    if qtype == "final_app":
        return _char_room(states[-1], g[0])
    if qtype in ("char_on_char_first_app", "char_on_char_final_app"):
        a, b, room = g
        t = _app_step(states, b, room, final=qtype.endswith("final_app"))
        return None if t is None else _char_room(states[t], a)
    if qtype == "char_at_frame":
        return _char_room(states[int(g[1]) - 1], g[0])
    if qtype in ("first_at_room", "last_at_room"):
        room = g[0]
        hits = [t for t, st in enumerate(states) if _rooms(st).get(room, [])]
        if not hits:
            return NOBODY
        t = hits[-1] if qtype == "last_at_room" else hits[0]
        return _single_occupant(states[t], room)
    if qtype in ("room_on_char_first_app", "room_on_char_final_app"):
        room_0, char, room_1 = g
        t = _app_step(states, char, room_1, final=qtype.endswith("final_app"))
        return None if t is None else _single_occupant(states[t], room_0)
    if qtype == "room_at_frame":
        return _single_occupant(states[int(g[1]) - 1], g[0])
    if qtype == "char_on_char_at_frame":
        char, k = g[0], int(g[1])
        room = _char_room(states[k - 1], char)
        others = [c for c in _rooms(states[k - 1]).get(room, []) if c != char]
        if len(others) == 0:
            return NOBODY
        return others[0] if len(others) == 1 else None
    if qtype in ("n_room_on_char_first_app", "n_room_on_char_final_app"):
        room_0, char, room_1 = g
        t = _app_step(states, char, room_1, final=qtype.endswith("final_app"))
        return None if t is None else len(_rooms(states[t]).get(room_0, []))
    if qtype == "n_char_at_frame":
        char, k = g[0], int(g[1])
        room = _char_room(states[k - 1], char)
        return len(_rooms(states[k - 1]).get(room, [])) - 1
    if qtype == "n_empty":
        k = int(g[0])
        return sum(1 for occ in _rooms(states[k - 1]).values() if not occ)
    if qtype == "room_empty":
        counts = {r: sum(1 for st in states if not _rooms(st).get(r, [])) for r in rooms}
        return _argcmp(counts, want_max=(g[0] == "more"))
    if qtype == "where_spend":
        char, cmp_word = g
        counts = {r: 0 for r in rooms}
        for st in states:
            counts[_char_room(st, char)] += 1
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "crowded_room":
        n_crowd = int(g[0])
        counts = {r: sum(1 for st in states if len(_rooms(st).get(r, [])) >= n_crowd)
                  for r in rooms}
        return _argcmp(counts, want_max=True)
    if qtype == "who_spend":
        cmp_word, room = g
        chars = sorted({c for st in states for occ in _rooms(st).values() for c in occ})
        counts = {c: sum(1 for st in states if c in _rooms(st).get(room, [])) for c in chars}
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "spend_alone":
        cmp_word = g[0]
        chars = sorted({c for st in states for occ in _rooms(st).values() for c in occ})
        counts = {c: 0 for c in chars}
        for st in states:
            for occ in _rooms(st).values():
                if len(occ) == 1:
                    counts[occ[0]] += 1
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "spend_together":
        char, cmp_word = g
        chars = sorted({c for st in states for occ in _rooms(st).values() for c in occ})
        counts = {c: 0 for c in chars if c != char}
        for st in states:
            room = _char_room(st, char)
            for c in _rooms(st).get(room, []):
                if c != char:
                    counts[c] += 1
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "steps_in_room":
        char, room = g
        return sum(1 for st in states if char in _rooms(st).get(room, []))
    if qtype == "rooms_visited":
        return len({_char_room(st, g[0]) for st in states})
    if qtype == "crowd_count":
        n_crowd = int(g[0])
        return sum(
            1 for st in states
            if any(len(occ) >= n_crowd for occ in _rooms(st).values())
        )
    raise ValueError(f"unknown qtype: {qtype}")


# ------------------------------------------------------------------ probe evidence

def probe_evidence_mmred(qtype: str, question: str, states: Sequence[Dict[str, Any]]):
    """Supply-probe labels -> (evidence_frame_set, locus_word) or None.

    Labels are the LOCAL per-frame fact the carrier must supply — the first/last/
    positional selection is the READOUT's job and must not leak into the labels
    (2026-08-01 first_at_room probe: single-answer-frame labels gave pooled d' 0.98
    while index-0 per-copy read 4.33 — first-ness is global, occupancy is local).

    steps_in_room: frames with char-in-room, locus = room (our 'steps' analog).
    first/last_at_room: frames where the room is OCCUPIED, locus = room.
    *_on_char_*_app: frames where the trigger char is in the trigger room, locus = it.
    room_at_frame: positional query — no content label; returns None (unsuitable
    for a supply probe).
    """
    g = _match(qtype, question)
    if qtype == "steps_in_room":
        char, room = g
        evid = {t for t, st in enumerate(states) if char in _rooms(st).get(room, [])}
        return evid, room
    if qtype in ("char_on_char_first_app", "char_on_char_final_app"):
        _, b, room = g
        evid = {t for t, st in enumerate(states) if b in _rooms(st).get(room, [])}
        return (evid, room) if evid else None
    if qtype in ("room_on_char_first_app", "room_on_char_final_app",
                 "n_room_on_char_first_app", "n_room_on_char_final_app"):
        _, char, room_1 = g
        evid = {t for t, st in enumerate(states) if char in _rooms(st).get(room_1, [])}
        return (evid, room_1) if evid else None
    if qtype in ("first_at_room", "last_at_room"):
        room = g[0]
        evid = {t for t, st in enumerate(states) if _rooms(st).get(room, [])}
        return (evid, room) if evid else None
    return None
