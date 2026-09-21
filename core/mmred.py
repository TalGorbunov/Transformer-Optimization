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
    (the --qtypes flag every script takes). Twin: mmred_hf.py:30."""
    raise NotImplementedError


def states(row: Row) -> List[State]:
    """Upstream sequence -> per-frame state dicts (step order asserted). Twin: mmred_hf.py:36."""
    raise NotImplementedError


def frames(row: Row, data_root: Path) -> List[Image.Image]:
    """The official renders for this row, native resolution, RGB. Twin: mmred_hf.py:41."""
    raise NotImplementedError


def stratified_order(rows: Sequence[Row], seed: int) -> List[Row]:
    """A shuffle that interleaves answer classes. The HF rows arrive sorted so that any head
    slice is all-zeros (the K0 trap, RESULTS.md 2026-08-09/10); every loader must go through
    this. Pinned by tests/test_mmred.py::test_stratified_order."""
    raise NotImplementedError


# ------------------------------------------------------------------ gold + evidence

def recompute_answer(qtype: str, question: str, states_: Sequence[State]) -> Optional[str]:
    """Re-derive the published answer from the states alone, for all 24 qtypes. Dispatch by
    qtype, never by question-regex sniffing (several templates are prefix-ambiguous).
    Twin: mmred_hf.py:154 (port it function by function; the parity test
    tests/test_mmred.py::test_gold_parity demands 100 % on seq 8 + 16 test)."""
    raise NotImplementedError


def evidence_frames(qtype: str, question: str, states_: Sequence[State]) -> Optional[Set[int]]:
    """Frame indices where the question's LOCAL predicate holds (the gate's labels), or None
    when the qtype has no content predicate. Twin: mmred_hf.py:466 (9 qtypes); extend to 24.
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
    each family on synthetic states.
    """
    raise NotImplementedError


def parse_question(qtype: str, question: str) -> Tuple[str, ...]:
    """The (char / room / step) arguments of a question for its qtype, via the upstream
    templates. Twin: mmred_hf.py:97 (_P) + :145 (_match)."""
    raise NotImplementedError
