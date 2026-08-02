"""Syntax-only grammar constraints for MMReD-HF scan decoding.

build_scan_regex(qtype, question, N) -> one anchored pattern describing every
syntactically legal scan for that sample: fixed skeleton (slot labels, parens,
anchor, END) is literal; content positions (payload words, tallies, counters,
the answer value) are alternations/\\d+ — the grammar NEVER computes semantic
values, it only enforces the format (formats.md). The gold scan of every sample
must fullmatch its pattern (enforced by tests/test_mmred_hf_adapter.py).

make_token_selector(pattern, tok) -> selector(lg, toks) for the decode loop:
greedy argmax restricted to tokens that keep the text a prefix of the language
(regex partial matching), scanning the top-K logits; FAIL-OPEN to plain argmax
if none of the top-K are legal (never worse than unconstrained).
"""
from __future__ import annotations

from typing import List, Optional

import regex as _rx

from .mmred_hf import CHAR_ABBR, CHAR_ORDER, ROOM_ABBR, ROOM_ORDER, _match

NUM = r"\d{1,3}"
_R = "|".join(r.lower() for r in ROOM_ORDER)
_C = "|".join(c.lower() for c in CHAR_ORDER)
_ROOMS_PLUS = rf"(?:{_R})(?:\+(?:{_R}))*"
_CHARS_PLUS = rf"(?:{_C})(?:\+(?:{_C}))*"
_OCC = rf"(?:-|{_CHARS_PLUS})"


def _counter_block(keys) -> str:
    abbr = ROOM_ABBR if keys[0] in ROOM_ABBR else CHAR_ABBR
    return "".join(rf" {_rx.escape(abbr[k])}{NUM}" for k in keys)


def build_scan_regex(qtype: str, question: str, n_frames: int) -> Optional[str]:
    """Anchored pattern for the full ' scan: ... END' text, or None (unknown qtype)."""
    try:
        g = _match(qtype, question)
    except Exception:
        return None

    anchor, val = "answer", None
    if qtype == "steps_in_room":
        room_l = _rx.escape(g[1].lower())
        slot = rf"(?:-|{room_l}\({NUM}\))"
        anchor, val = "total", NUM
    elif qtype == "crowd_count":
        slot = rf"(?:-|{_ROOMS_PLUS}\({NUM}\))"
        anchor, val = "total", NUM
    elif qtype == "rooms_visited":
        slot = rf"(?:{_R})\({NUM}\)"
        anchor, val = "total", NUM
    elif qtype in ("first_app", "final_app", "char_at_frame"):
        slot = rf"(?:{_R})"
        val = rf"(?:{_R})"
    elif qtype in ("first_at_room", "last_at_room", "room_at_frame",
                   "char_on_char_at_frame"):
        slot = _OCC
        val = rf"(?:nobody|{_C})"
    elif qtype in ("n_char_at_frame", "n_empty"):
        slot = NUM
        val = NUM
    elif qtype in ("char_on_char_first_app", "char_on_char_final_app"):
        slot = rf"(?:{_R})\*?"
        val = rf"(?:{_R})"
    elif qtype in ("room_on_char_first_app", "room_on_char_final_app"):
        slot = rf"{_OCC}\*?"
        val = rf"(?:nobody|{_C})"
    elif qtype in ("n_room_on_char_first_app", "n_room_on_char_final_app"):
        slot = rf"{NUM}\*?"
        val = NUM
    elif qtype == "where_spend":
        keys = ROOM_ORDER
        slot = rf"(?:-|{_R}){_counter_block(keys)}"
        anchor = "max" if g[1] == "most" else "min"
        val = rf"(?:{_R})\({NUM}\)"
    elif qtype == "who_spend":
        keys = CHAR_ORDER
        slot = rf"(?:-|{_CHARS_PLUS}){_counter_block(keys)}"
        anchor = "max" if g[0] == "most" else "min"
        val = rf"(?:{_C})\({NUM}\)"
    elif qtype == "spend_alone":
        keys = CHAR_ORDER
        slot = rf"(?:-|{_CHARS_PLUS}){_counter_block(keys)}"
        anchor = "max" if g[0] == "most" else "min"
        val = rf"(?:{_C})\({NUM}\)"
    elif qtype == "spend_together":
        char = g[0]
        keys = [c for c in CHAR_ORDER if c != char]
        alt = "|".join(c.lower() for c in keys)
        slot = rf"(?:-|(?:{alt})(?:\+(?:{alt}))*){_counter_block(keys)}"
        anchor = "max" if g[1] == "most" else "min"
        val = rf"(?:{alt})\({NUM}\)"
    elif qtype == "room_empty":
        keys = ROOM_ORDER
        slot = rf"(?:-|{_ROOMS_PLUS}){_counter_block(keys)}"
        anchor = "max" if g[0] == "more" else "min"
        val = rf"(?:{_R})\({NUM}\)"
    elif qtype == "crowded_room":
        keys = ROOM_ORDER
        slot = rf"(?:-|{_ROOMS_PLUS}){_counter_block(keys)}"
        anchor = "max"
        val = rf"(?:{_R})\({NUM}\)"
    else:
        return None

    body = " scan:" + "".join(rf" f{t}:{slot}" for t in range(1, n_frames + 1))
    return body + rf" \| {anchor}: {val} END"


def make_token_selector(pattern: str, tok, top_k: int = 256):
    """-> selector(lg, toks) choosing the highest-logit LEGAL token id.
    Legal = decoded(toks+[t]) is still a prefix of the pattern's language
    (partial fullmatch), or completes it (then EOS becomes legal too).
    Fail-open: if none of the top_k are legal, return plain argmax."""
    rx = _rx.compile(pattern)
    eos = tok.eos_token_id

    def selector(lg, toks: List[int]) -> int:
        txt = tok.decode(toks) if toks else ""
        if rx.fullmatch(txt):  # grammar complete -> only EOS
            return eos
        order = lg.topk(min(top_k, lg.shape[-1])).indices.tolist()
        for t in order:
            if t == eos:
                continue  # EOS only after completion
            m = rx.fullmatch(txt + tok.decode([t]), partial=True)
            if m or rx.fullmatch(txt + tok.decode([t])):
                return t
        return int(lg.argmax())  # fail-open

    return selector
