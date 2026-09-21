"""Per-frame fact labels for the arm-B GIN readout (shared by probe_suite,
dump_states, and the head-fitting pipeline). Labels mirror the gold-scan slot logic."""
from __future__ import annotations

from gnnformer.mmred_hf import CHAR_ORDER, ROOM_ORDER, _char_room, _match, _rooms


def frame_labels(qtype: str, q0: str, states):
    """-> (label per frame, label-space name). Labels are the arm-B head targets."""
    g = _match(qtype, q0)
    if qtype == "steps_in_room":
        c, r = g
        return [int(c in _rooms(st).get(r, [])) for st in states], "gate2"
    if qtype == "crowd_count":
        m = int(g[0])
        return [int(any(len(o) >= m for o in _rooms(st).values())) for st in states], "gate2"
    if qtype in ("char_at_frame", "first_app", "final_app", "where_spend",
                 "rooms_visited"):
        c = g[0]
        return [ROOM_ORDER.index(_char_room(st, c)) for st in states], "room6"
    if qtype in ("room_at_frame", "first_at_room", "last_at_room", "who_spend"):
        r = g[1] if qtype == "who_spend" else g[0]
        out = []
        for st in states:
            occ = _rooms(st).get(r, [])
            out.append(6 if len(occ) > 1 else (5 if not occ else CHAR_ORDER.index(occ[0])))
        return out, "occ7"
    if qtype == "n_char_at_frame":
        c = g[0]
        return [min(len(_rooms(st).get(_char_room(st, c), [])) - 1, 4) for st in states], "cnt5"
    if qtype == "n_empty":
        return [sum(1 for o in _rooms(st).values() if not o) for st in states], "cnt7"
    if qtype == "room_empty":  # per-frame: is Kitchen empty (fixed-room binary)
        return [int(not _rooms(st).get("Kitchen", [])) for st in states], "gate2"
    if qtype in ("char_on_char_first_app", "char_on_char_final_app"):
        # trigger gate: is B in R1 this frame (the conditional's needle bit)
        _a, b, r1 = g
        return [int(b in _rooms(st).get(r1, [])) for st in states], "gate2"
    if qtype in ("room_on_char_first_app", "room_on_char_final_app"):
        r0, _c, _r1 = g
        out = []
        for st in states:
            occ = _rooms(st).get(r0, [])
            out.append(6 if len(occ) > 1 else (5 if not occ else CHAR_ORDER.index(occ[0])))
        return out, "occ7"
    if qtype in ("n_room_on_char_first_app", "n_room_on_char_final_app"):
        r0 = g[0]
        return [min(len(_rooms(st).get(r0, [])), 5) for st in states], "cnt6"
    if qtype == "char_on_char_at_frame":
        c = g[0]
        out = []
        for st in states:
            others = [x for x in _rooms(st).get(_char_room(st, c), []) if x != c]
            out.append(6 if len(others) > 1 else
                       (5 if not others else CHAR_ORDER.index(others[0])))
        return out, "occ7"
    if qtype == "spend_together":
        c = g[0]
        other = next(x for x in CHAR_ORDER if x != c)  # fixed-partner binary
        return [int(other in _rooms(st).get(_char_room(st, c), [])) for st in states], "gate2"
    if qtype == "spend_alone":  # fixed-char binary: is Daniel alone this frame
        c = CHAR_ORDER[0]
        return [int(len(_rooms(st).get(_char_room(st, c), [])) == 1) for st in states], "gate2"
    if qtype == "crowded_room":
        m = int(g[0])
        out = []
        for st in states:
            cr = [r for r in ROOM_ORDER if len(_rooms(st).get(r, [])) >= m]
            out.append(6 if len(cr) != 1 else ROOM_ORDER.index(cr[0]))
        return out, "room7"
    raise ValueError(qtype)
