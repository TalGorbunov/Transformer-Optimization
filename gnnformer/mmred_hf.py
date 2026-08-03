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


# ------------------------------------------------------------------ scan targets
# Gold caption-scan builders per outputs/mmred_hf/formats.md: one slot per frame,
# lowercase payload words, `-` = null, `+` joins multi-word slots, `(k)` running
# counters, `*` trigger marks, anchors total:/answer:/max:/min: + END.

ROOM_ORDER = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
CHAR_ORDER = ["Daniel", "John", "Mary", "Michael", "Sandra"]
ROOM_ABBR = {"Kitchen": "k", "Bathroom": "b", "Garden": "g", "Office": "o",
             "Bedroom": "be", "Hallway": "h"}
CHAR_ABBR = {"Daniel": "da", "John": "jo", "Mary": "ma", "Michael": "mi", "Sandra": "sa"}
_CANON = {w.lower(): w for w in ROOM_ORDER + CHAR_ORDER}
_CANON["nobody"] = NOBODY

_ALL_QTYPES = sorted(NIAH_QTYPES + DC_QTYPES, key=len, reverse=True)


def qtype_from_dirname(name: str) -> Optional[str]:
    """materialize_dirs names are <qtype>_[KA]<answer>_<qid>; longest-prefix match
    (several qtypes are prefixes of others, e.g. first_app / first_at_room share none,
    but n_room_on_char_first_app vs room_on_char_first_app do collide on substring)."""
    for qt in _ALL_QTYPES:
        if name.startswith(qt + "_"):
            return qt
    return None


def _join(words) -> str:
    return "+".join(w.lower() for w in words) if words else "-"


def build_scan_mmred(qtype: str, question: str, states: Sequence[Dict[str, Any]],
                     answer: str) -> Optional[str]:
    """-> ' scan: f1:<slot> ... | <anchor>: <value> END' or None when the scan's own
    reduction fails to reproduce the published answer (parse_task_labels-style sanity
    gate — the caller counts a skip). The scan is built from states ONLY."""
    g = _match(qtype, question)
    N = len(states)
    ans = str(answer)
    slots: List[str] = []
    anchor = "answer"

    if qtype == "steps_in_room":
        char, room = g
        k = 0
        for st in states:
            if char in _rooms(st).get(room, []):
                k += 1
                slots.append(f"{room.lower()}({k})")
            else:
                slots.append("-")
        anchor, value, tailval = "total", str(k), str(k)
    elif qtype == "crowd_count":
        m = int(g[0])
        k = 0
        for st in states:
            crowded = [r for r in ROOM_ORDER if len(_rooms(st).get(r, [])) >= m]
            if crowded:
                k += 1
                slots.append(f"{_join(crowded)}({k})")
            else:
                slots.append("-")
        anchor, value, tailval = "total", str(k), str(k)
    elif qtype == "rooms_visited":
        char = g[0]
        seen: List[str] = []
        for st in states:
            r = _char_room(st, char)
            if r not in seen:
                seen.append(r)
            slots.append(f"{r.lower()}({len(seen)})")
        anchor, value, tailval = "total", str(len(seen)), str(len(seen))
    elif qtype in ("first_app", "final_app", "char_at_frame"):
        char = g[0]
        rooms_c = [_char_room(st, char) for st in states]
        slots = [r.lower() for r in rooms_c]
        t = 0 if qtype == "first_app" else (N - 1 if qtype == "final_app" else int(g[1]) - 1)
        value = rooms_c[t]
        tailval = value.lower()
    elif qtype in ("first_at_room", "last_at_room", "room_at_frame"):
        room = g[0]
        occs = [_rooms(st).get(room, []) for st in states]
        slots = [_join(o) for o in occs]
        if qtype == "room_at_frame":
            o = occs[int(g[1]) - 1]
        else:
            hits = [t for t in range(N) if occs[t]]
            o = occs[hits[-1] if qtype == "last_at_room" else hits[0]] if hits else []
        if len(o) > 1:
            return None  # generator guarantees a single occupant at the selected frame
        value = o[0] if o else NOBODY
        tailval = value.lower()
    elif qtype == "char_on_char_at_frame":
        char, k = g[0], int(g[1])
        others_per = []
        for st in states:
            room = _char_room(st, char)
            others_per.append([c for c in _rooms(st).get(room, []) if c != char])
        slots = [_join(o) for o in others_per]
        o = others_per[k - 1]
        if len(o) > 1:
            return None
        value = o[0] if o else NOBODY
        tailval = value.lower()
    elif qtype == "n_char_at_frame":
        char, k = g[0], int(g[1])
        ns = []
        for st in states:
            room = _char_room(st, char)
            ns.append(len(_rooms(st).get(room, [])) - 1)
        slots = [str(n) for n in ns]
        value = tailval = str(ns[k - 1])
    elif qtype == "n_empty":
        k = int(g[0])
        ns = [sum(1 for occ in _rooms(st).values() if not occ) for st in states]
        slots = [str(n) for n in ns]
        value = tailval = str(ns[k - 1])
    elif qtype in ("char_on_char_first_app", "char_on_char_final_app",
                   "room_on_char_first_app", "room_on_char_final_app",
                   "n_room_on_char_first_app", "n_room_on_char_final_app"):
        final = qtype.endswith("final_app")
        if qtype.startswith("char_on_char"):
            a, trig_char, trig_room = g
            payloads = [_char_room(st, a).lower() for st in states]
        else:
            room0, trig_char, trig_room = g
            if qtype.startswith("n_room"):
                payloads = [str(len(_rooms(st).get(room0, []))) for st in states]
            else:
                payloads = [_join(_rooms(st).get(room0, [])) for st in states]
        trig = [trig_char in _rooms(st).get(trig_room, []) for st in states]
        slots = [p + ("*" if tr else "") for p, tr in zip(payloads, trig)]
        hits = [t for t in range(N) if trig[t]]
        if not hits:
            return None
        t = hits[-1] if final else hits[0]
        if qtype.startswith("char_on_char"):
            value = _char_room(states[t], a)
            tailval = value.lower()
        elif qtype.startswith("n_room"):
            value = tailval = payloads[t]
        else:
            o = _rooms(states[t]).get(room0, [])
            if len(o) > 1:
                return None
            value = o[0] if o else NOBODY
            tailval = value.lower()
    else:  # multi-counter arg-max/min family
        if qtype == "where_spend":
            char, cmp_w = g
            keys, abbr, want_max = ROOM_ORDER, ROOM_ABBR, cmp_w == "most"
            score = lambda st: [_char_room(st, char)]  # noqa: E731
        elif qtype == "who_spend":
            cmp_w, room = g
            keys, abbr, want_max = CHAR_ORDER, CHAR_ABBR, cmp_w == "most"
            score = lambda st: list(_rooms(st).get(room, []))  # noqa: E731
        elif qtype == "spend_alone":
            keys, abbr, want_max = CHAR_ORDER, CHAR_ABBR, g[0] == "most"
            score = lambda st: [occ[0] for occ in _rooms(st).values() if len(occ) == 1]  # noqa: E731
        elif qtype == "spend_together":
            char, cmp_w = g
            keys = [c for c in CHAR_ORDER if c != char]
            abbr, want_max = CHAR_ABBR, cmp_w == "most"
            score = lambda st: [c for c in _rooms(st).get(_char_room(st, char), [])  # noqa: E731
                                if c != char]
        elif qtype == "room_empty":
            keys, abbr, want_max = ROOM_ORDER, ROOM_ABBR, g[0] == "more"
            score = lambda st: [r for r in ROOM_ORDER if not _rooms(st).get(r, [])]  # noqa: E731
        elif qtype == "crowded_room":
            m = int(g[0])
            keys, abbr, want_max = ROOM_ORDER, ROOM_ABBR, True
            score = lambda st: [r for r in ROOM_ORDER if len(_rooms(st).get(r, [])) >= m]  # noqa: E731
        else:
            raise ValueError(f"unknown qtype: {qtype}")
        counts = {kx: 0 for kx in keys}
        for st in states:
            scored = [w for w in score(st) if w in counts]
            for w in scored:
                counts[w] += 1
            slots.append(_join(scored) + " "
                         + " ".join(f"{abbr[kx]}{counts[kx]}" for kx in keys))
        win = _argcmp(counts, want_max)
        if win is None:
            return None
        anchor = "max" if want_max else "min"
        value = win
        tailval = f"{win.lower()}({counts[win]})"

    if str(value) != ans:
        return None
    body = " ".join(f"f{t + 1}:{s}" for t, s in enumerate(slots))
    return f" scan: {body} | {anchor}: {tailval} END"


def parse_answer_mmred(text: str) -> Optional[str]:
    """Eval parser: anchor on the LAST total:/answer:/max:/min:, read the value before
    END, map back to canonical case (Kitchen/Sandra/Nobody) for EM scoring."""
    mm = list(re.finditer(r"\b(total|answer|max|min):", text))
    if not mm:
        return None
    kind = mm[-1].group(1)
    seg = text[mm[-1].end():].split("END")[0]
    if kind == "total":
        nums = re.findall(r"\d+", seg)
        return nums[-1] if nums else None
    m2 = re.search(r"[A-Za-z]+|\d+", seg)
    if not m2:
        return None
    w = m2.group(0)
    return _CANON.get(w.lower(), w)


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


# ------------------------------------------------------------------ v4 targets
# The caption scans demanded per-frame CONTENT identity from the carriers — the
# VoCo-style compression property our method does not claim (2026-08-03 finding:
# truncated free decode degenerates on exactly the content payloads, while the
# relevance bits read at 99%/frame). v4 asks carriers only for what the thesis
# claims they carry: verdict scans (relevance bit / question-conditioned count +
# running tally) for the aggregation qtypes, DIRECT short answers for everything
# else (single-fact read at the answer position — no generative cascade).

V4_VERDICT_QTYPES = {
    "steps_in_room", "crowd_count", "n_char_at_frame", "n_empty",
    "n_room_on_char_first_app", "n_room_on_char_final_app",
}


def _v4_scan(slots: List[str], tail: str) -> str:
    return " scan: " + " ".join(f"f{t + 1}:{s}" for t, s in enumerate(slots)) + tail


def build_target_v4(qtype: str, question: str, states: Sequence[Dict[str, Any]],
                    answer: str) -> Optional[str]:
    """v4 gold target: verdict scan (aggregation qtypes) or ' answer: <v> END'.
    None when the scan's own reduction fails to reproduce the published answer."""
    g = _match(qtype, question)
    ans = str(answer)

    if qtype in ("steps_in_room", "crowd_count"):
        k = 0
        slots = []
        for st in states:
            if qtype == "steps_in_room":
                hit = g[0] in _rooms(st).get(g[1], [])
            else:
                hit = any(len(occ) >= int(g[0]) for occ in _rooms(st).values())
            if hit:
                k += 1
                slots.append(f"x({k})")
            else:
                slots.append("-")
        return _v4_scan(slots, f" | total: {k} END") if str(k) == ans else None
    if qtype in ("n_char_at_frame", "n_empty"):
        if qtype == "n_char_at_frame":
            char, kq = g[0], int(g[1])
            ns = [len(_rooms(st).get(_char_room(st, char), [])) - 1 for st in states]
        else:
            kq = int(g[0])
            ns = [sum(1 for occ in _rooms(st).values() if not occ) for st in states]
        if str(ns[kq - 1]) != ans:
            return None
        return _v4_scan([str(n) for n in ns], f" | answer: {ns[kq - 1]} END")
    if qtype in ("n_room_on_char_first_app", "n_room_on_char_final_app"):
        room0, char, room1 = g
        ns = [len(_rooms(st).get(room0, [])) for st in states]
        trig = [char in _rooms(st).get(room1, []) for st in states]
        hits = [t for t in range(len(states)) if trig[t]]
        if not hits:
            return None
        t = hits[-1] if qtype.endswith("final_app") else hits[0]
        if str(ns[t]) != ans:
            return None
        slots = [f"{n}{'*' if tr else ''}" for n, tr in zip(ns, trig)]
        return _v4_scan(slots, f" | answer: {ns[t]} END")
    # direct answer for everything else (content/selection + set/argmax stretch cells)
    return f" answer: {ans if ans.isdigit() else ans.lower()} END"
