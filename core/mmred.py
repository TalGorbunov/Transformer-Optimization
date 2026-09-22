"""The official MMReD benchmark (HF ef1e43ce/mmred + Fr0do/mmred renders) — rows, frames,
gold recomputation, evidence labels.

Twin: legacy/v1/gnnformer/mmred_hf.py (645 lines). Kept: load_index → load_split,
row_states → states, load_mmred_hf_sample → frames, recompute_answer, the qtype lists,
probe_evidence_mmred → evidence_frames (extended from 9 to 24 qtypes). Dropped: the scan /
v4 / v5 target builders and the dirname parser (scratchpad era, qa.txt bridge).

Data layout (data/mmred_hf -> /rg/shocher_prj/lab_data/mmred):
    json/<config>_<split>.json          rows: {qid, seq_len, qtype, atype, question, answer,
                                              sequence:[{step_id, rooms:{room:[chars]}}]}
    images/<config>_<split>/<qid>/frame_%04d.png   official renders, 512 px, "Step k" printed
Prepared by experiments/prepare_data.py (HF download -> json; official renderer -> images).

Evidence-label rule (the gate's training labels; NEVER used at inference):
    a frame is evidence iff the question's LOCAL predicate holds there; first/last/count/argmax
    are the readout's job. For positional qtypes the predicate is "this is the named step".
    The generator's own relevant_map agrees with this rule for the dense qtypes and marks only
    the answer frame for the needle qtypes (see plan §2) — tests/test_mmred.py::test_evidence_*
    pin both facts on synthetic states.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from .constants import CHARS, NOBODY, ROOMS

Row = Dict[str, Any]
State = Dict[str, Any]  # {"rooms": {room: [chars]}} — all six rooms present, [] = empty

# 15 needle-in-a-haystack + 9 dense-context qtypes, exactly the upstream names.
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
QTYPES = NIAH_QTYPES + DC_QTYPES
NUMERIC_QTYPES = [q for q in QTYPES if q.startswith("n_") or q in ("steps_in_room", "rooms_visited", "crowd_count")]


# ------------------------------------------------------------------ loading

def load_split(config: str, split: str, data_root: Path, qtypes: Optional[Sequence[str]] = None) -> List[Row]:
    """Rows of data_root/json/<config>_<split>.json, optionally filtered to `qtypes`
    (the --qtypes flag every script takes). Twin: mmred_hf.py:30. Each row is stamped with
    `split_dir` = "<config>_<split>" so frames() can find its renders."""
    path = Path(data_root) / "json" / f"{config}_{split}.json"
    rows: List[Row] = json.loads(path.read_text(encoding="utf-8"))
    if qtypes is not None:
        want = set(qtypes)
        unknown = want - set(QTYPES)
        if unknown:
            raise ValueError(f"unknown qtypes: {sorted(unknown)}")
        rows = [r for r in rows if r["qtype"] in want]
    for r in rows:
        r.setdefault("split_dir", f"{config}_{split}")
    return rows


def states(row: Row) -> List[State]:
    """Upstream sequence -> per-frame state dicts (step order asserted). Twin: mmred_hf.py:36."""
    seq = row["sequence"]
    for i, s in enumerate(seq):
        assert int(s["step_id"]) == i + 1, (row.get("qid"), i, s["step_id"])
    return [{"rooms": {r: list(c) for r, c in s["rooms"].items()}} for s in seq]


def frames(row: Row, data_root: Path, split_dir: Optional[str] = None) -> List[Image.Image]:
    """The official renders for this row, native resolution, RGB. Twin: mmred_hf.py:41.
    `split_dir` defaults to the value load_split() stamped on the row."""
    sd = split_dir or row["split_dir"]
    n = len(row["sequence"])
    root = Path(data_root) / "images" / sd / str(row["qid"])
    return [Image.open(root / f"frame_{i + 1:04d}.png").convert("RGB") for i in range(n)]


def stratified_order(rows: Sequence[Row], seed: int) -> List[Row]:
    """A shuffle that interleaves answer classes. The HF rows arrive sorted so that any head
    slice is all-zeros (the K0 trap, RESULTS.md 2026-08-09/10); every loader must go through
    this. Pinned by tests/test_mmred.py::test_stratified_order."""
    rng = random.Random(seed)
    groups: Dict[str, List[Row]] = {}
    for r in rows:
        groups.setdefault(str(r["answer"]), []).append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out: List[Row] = []
    while any(groups[k] for k in keys):          # round-robin over classes
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop())
    return out


# ------------------------------------------------------------------ question parsing
# Twin: mmred_hf.py:97 (_P). Comments record wordings that differ between the published HF
# rows and the generator's HEAD; both are accepted, the computation is the same.

_P = {
    "first_app": re.compile(r"In which room did (\w+) first appear"),
    "final_app": re.compile(r"In which room was (\w+) at the final step"),
    "char_on_char_first_app": re.compile(r"In which room was (\w+) when (\w+) first appeared in the (\w+)"),
    "char_on_char_final_app": re.compile(r"In which room was (\w+) when (\w+) made their final appearance in the (\w+)"),
    "char_at_frame": re.compile(r"In which room was (\w+) at step (\d+)"),
    "first_at_room": re.compile(r"Who was the first to appear in the (\w+)"),
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
    "who_spend": re.compile(r"Who spent the (most|least amount of) time (?:alone )?in the (\w+)"),
    "spend_alone": re.compile(r"Who spent the (most|least amount of) time alone in the rooms"),
    "spend_together": re.compile(r"With whom did (\w+) spend the (most|least amount of) time together in the same room"),
    "steps_in_room": re.compile(r"How many steps did (\w+) spend in the (\w+)"),
    "rooms_visited": re.compile(r"How many different rooms did (\w+) visit"),
    "crowd_count": re.compile(
        r"(?:For how many steps was there a crowd|How many times did a crowd) "
        r"\((\d+) or more people in one room\)"),
}
assert set(_P) == set(QTYPES)


def parse_question(qtype: str, question: str) -> Tuple[str, ...]:
    """The (char / room / step) arguments of a question for its qtype, via the upstream
    templates. Twin: mmred_hf.py:97 (_P) + :145 (_match)."""
    m = _P[qtype].search(question)
    if not m:
        raise ValueError(f"question does not match {qtype} template: {question!r}")
    return m.groups()


# ------------------------------------------------------------------ state helpers (twin :54-92)

def _rooms(st: State) -> Dict[str, List[str]]:
    return st.get("rooms", {}) or {}


def _char_room(st: State, char: str) -> Optional[str]:
    for room, occ in _rooms(st).items():
        if char in occ:
            return room
    return None


def _room_names(states_: Sequence[State]) -> List[str]:
    return list(_rooms(states_[0]).keys()) if states_ else []


def _single_occupant(st: State, room: str) -> Optional[str]:
    occ = _rooms(st).get(room, [])
    if len(occ) == 0:
        return NOBODY
    if len(occ) == 1:
        return occ[0]
    return None  # generator guarantees uniqueness; None -> parity mismatch


def _app_step(states_: Sequence[State], char: str, room: str, final: bool) -> Optional[int]:
    """First/last frame index where char is in room."""
    hits = [t for t, st in enumerate(states_) if char in _rooms(st).get(room, [])]
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


# ------------------------------------------------------------------ gold + evidence

def _recompute(qtype: str, g: Tuple[str, ...], states_: Sequence[State]) -> Any:
    """Twin: mmred_hf.py:154 body, verbatim (str for person/room, int for number)."""
    rooms = _room_names(states_)
    if qtype == "first_app":
        return _char_room(states_[0], g[0])
    if qtype == "final_app":
        return _char_room(states_[-1], g[0])
    if qtype in ("char_on_char_first_app", "char_on_char_final_app"):
        a, b, room = g
        t = _app_step(states_, b, room, final=qtype.endswith("final_app"))
        return None if t is None else _char_room(states_[t], a)
    if qtype == "char_at_frame":
        return _char_room(states_[int(g[1]) - 1], g[0])
    if qtype in ("first_at_room", "last_at_room"):
        room = g[0]
        hits = [t for t, st in enumerate(states_) if _rooms(st).get(room, [])]
        if not hits:
            return NOBODY
        t = hits[-1] if qtype == "last_at_room" else hits[0]
        return _single_occupant(states_[t], room)
    if qtype in ("room_on_char_first_app", "room_on_char_final_app"):
        room_0, char, room_1 = g
        t = _app_step(states_, char, room_1, final=qtype.endswith("final_app"))
        return None if t is None else _single_occupant(states_[t], room_0)
    if qtype == "room_at_frame":
        return _single_occupant(states_[int(g[1]) - 1], g[0])
    if qtype == "char_on_char_at_frame":
        char, k = g[0], int(g[1])
        room = _char_room(states_[k - 1], char)
        others = [c for c in _rooms(states_[k - 1]).get(room, []) if c != char]
        if len(others) == 0:
            return NOBODY
        return others[0] if len(others) == 1 else None
    if qtype in ("n_room_on_char_first_app", "n_room_on_char_final_app"):
        room_0, char, room_1 = g
        t = _app_step(states_, char, room_1, final=qtype.endswith("final_app"))
        return None if t is None else len(_rooms(states_[t]).get(room_0, []))
    if qtype == "n_char_at_frame":
        char, k = g[0], int(g[1])
        room = _char_room(states_[k - 1], char)
        return len(_rooms(states_[k - 1]).get(room, [])) - 1
    if qtype == "n_empty":
        k = int(g[0])
        return sum(1 for occ in _rooms(states_[k - 1]).values() if not occ)
    if qtype == "room_empty":
        counts = {r: sum(1 for st in states_ if not _rooms(st).get(r, [])) for r in rooms}
        return _argcmp(counts, want_max=(g[0] == "more"))
    if qtype == "where_spend":
        char, cmp_word = g
        counts = {r: 0 for r in rooms}
        for st in states_:
            counts[_char_room(st, char)] += 1
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "crowded_room":
        n_crowd = int(g[0])
        counts = {r: sum(1 for st in states_ if len(_rooms(st).get(r, [])) >= n_crowd)
                  for r in rooms}
        return _argcmp(counts, want_max=True)
    if qtype == "who_spend":
        cmp_word, room = g
        chars = sorted({c for st in states_ for occ in _rooms(st).values() for c in occ})
        counts = {c: sum(1 for st in states_ if c in _rooms(st).get(room, [])) for c in chars}
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "spend_alone":
        cmp_word = g[0]
        chars = sorted({c for st in states_ for occ in _rooms(st).values() for c in occ})
        counts = {c: 0 for c in chars}
        for st in states_:
            for occ in _rooms(st).values():
                if len(occ) == 1:
                    counts[occ[0]] += 1
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "spend_together":
        char, cmp_word = g
        chars = sorted({c for st in states_ for occ in _rooms(st).values() for c in occ})
        counts = {c: 0 for c in chars if c != char}
        for st in states_:
            room = _char_room(st, char)
            for c in _rooms(st).get(room, []):
                if c != char:
                    counts[c] += 1
        return _argcmp(counts, want_max=(cmp_word == "most"))
    if qtype == "steps_in_room":
        char, room = g
        return sum(1 for st in states_ if char in _rooms(st).get(room, []))
    if qtype == "rooms_visited":
        return len({_char_room(st, g[0]) for st in states_})
    if qtype == "crowd_count":
        n_crowd = int(g[0])
        return sum(1 for st in states_ if any(len(occ) >= n_crowd for occ in _rooms(st).values()))
    raise ValueError(f"unknown qtype: {qtype}")


def recompute_answer(qtype: str, question: str, states_: Sequence[State]) -> Optional[str]:
    """Re-derive the published answer from the states alone, for all 24 qtypes. Dispatch by
    qtype, never by question-regex sniffing (several templates are prefix-ambiguous).
    Twin: mmred_hf.py:154 (ported function by function; the parity test
    tests/test_mmred.py::test_gold_parity demands 100 % on seq 8 + 16 test).
    Returns the answer as a string (numbers as their integer string), or None when a generator
    uniqueness guarantee does not hold."""
    v = _recompute(qtype, parse_question(qtype, question), states_)
    return None if v is None else str(v)


def evidence_frames(qtype: str, question: str, states_: Sequence[State]) -> Optional[Set[int]]:
    """Frame indices where the question's LOCAL predicate holds (the gate's labels), or None
    when the qtype has no content predicate. Twin: mmred_hf.py:466 (9 qtypes); extended to 24.
    Every character is in exactly one room at every step (upstream vgen), which makes some
    "first/final" questions positional:
      positional  char_at_frame / room_at_frame / char_on_char_at_frame / n_char_at_frame /
                  n_empty:                                  {step − 1}
                  first_app ("first appear" = step 1):      {0}
                  final_app ("at the final step"):          {N − 1}
      trigger     char_on_char_{first,final}_app(X, Y, R), room_on_char_*(R0, Y, R),
                  n_room_on_char_*(R0, Y, R):               every t with Y in R
                  first_at_room / last_at_room (R):         every t with R occupied
      dense       steps_in_room(X, R):                      every t with X in R
                  where_spend / rooms_visited / spend_together / spend_alone (X):
                                                            every t (X is always present)
                  who_spend(R), room_empty, crowded_room(m), crowd_count(m):
                                                            every t where the room condition
                                                            holds (occupied / empty / ≥ m)
    The readout does first/last/count/argmax over the kept frames. tests/test_mmred.py pins
    each family on synthetic states. An empty set is a valid label (nothing is evidence).
    """
    g = parse_question(qtype, question)
    N = len(states_)
    every = set(range(N))
    # positional
    if qtype in ("char_at_frame", "room_at_frame", "char_on_char_at_frame", "n_char_at_frame"):
        return {int(g[1]) - 1}
    if qtype == "n_empty":
        return {int(g[0]) - 1}
    if qtype == "first_app":
        return {0}
    if qtype == "final_app":
        return {N - 1}
    # trigger: every frame where the trigger character is in the trigger room
    if qtype in ("char_on_char_first_app", "char_on_char_final_app"):
        _, y, room = g
        return {t for t, st in enumerate(states_) if y in _rooms(st).get(room, [])}
    if qtype in ("room_on_char_first_app", "room_on_char_final_app",
                 "n_room_on_char_first_app", "n_room_on_char_final_app"):
        _, y, room = g
        return {t for t, st in enumerate(states_) if y in _rooms(st).get(room, [])}
    if qtype in ("first_at_room", "last_at_room"):
        room = g[0]
        return {t for t, st in enumerate(states_) if _rooms(st).get(room, [])}
    # dense
    if qtype == "steps_in_room":
        char, room = g
        return {t for t, st in enumerate(states_) if char in _rooms(st).get(room, [])}
    if qtype in ("where_spend", "rooms_visited", "spend_together", "spend_alone"):
        return every
    if qtype == "who_spend":
        room = g[1]
        return {t for t, st in enumerate(states_) if _rooms(st).get(room, [])}
    if qtype == "room_empty":
        return {t for t, st in enumerate(states_) if any(not occ for occ in _rooms(st).values())}
    if qtype in ("crowded_room", "crowd_count"):
        m = int(g[0])
        return {t for t, st in enumerate(states_) if any(len(occ) >= m for occ in _rooms(st).values())}
    return None


# ------------------------------------------------------------------ rendering + edits

def render_sequence(sequence: Sequence[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """The official renderer (mmred.vgen.visualization.render_sequence_from_json): 512 px
    PNGs `frame_%04d.png` under out_dir, "Step k" printed from each step_id. Forces the Agg
    backend and a node-local MPLCONFIGDIR before matplotlib is imported (the shared-NFS
    font-cache lock storm of 2026-08-01). `sequence` = [{"step_id", "rooms"}], as in the JSON."""
    import os
    import tempfile

    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mpl_", dir=os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from mmred.vgen.visualization import render_sequence_from_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    render_sequence_from_json(list(sequence), out, as_gif=False)
    return sorted(out.glob("frame_*.png"))


def with_char_moved(state: State, char: str, room: str) -> State:
    """A new state with `char` removed from wherever it is and appended to `room` (the
    probe's one-frame flip). All other occupants keep their order."""
    rooms = {r: [c for c in occ if c != char] for r, occ in _rooms(state).items()}
    rooms.setdefault(room, [])
    rooms[room] = rooms[room] + [char]
    out = dict(state)
    out["rooms"] = rooms
    return out


def as_sequence(states_: Sequence[State]) -> List[Dict[str, Any]]:
    """States -> the renderer's / JSON's sequence form (1-based step_id)."""
    return [{"step_id": t + 1, "rooms": {r: list(c) for r, c in _rooms(st).items()}} for t, st in enumerate(states_)]
