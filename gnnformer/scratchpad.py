"""Scratchpad readout: gold target builders + the eval answer parser.

The readout-expressivity fix: instead of a single answer digit, the model decodes a
structured scratchpad and the final count is a read-off. Format sweep verdict
(RESULTS.md [2026-07-24]): WINNER = 'caption' (per-frame slots with attribute words,
inline tally, END terminator); 'poslist' is the l12v2 control; 'chunked' refuted.

Every builder's output must round-trip through the tokenizer and parse back to gold —
pinned by tests/test_scratchpad.py (port of the legacy phase-0 sanity gate).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

SCRATCHPAD_FORMATS = ("poslist", "scan", "caption", "chunked")
CHUNK_FRAMES = 16


def build_target(task: str, evid: Set[int], aux, gold: int) -> str:
    """Plain verdict-scratchpad: evidence frame list (1-indexed) then '-> count'."""
    if task == "rooms":
        return f" rooms {', '.join(aux)} -> {gold}" if aux else " none -> 0"
    if evid:
        return f" frames {', '.join(str(i + 1) for i in sorted(evid))} -> {gold}"
    return " none -> 0"


def build_target_tally(task: str, evid: Set[int], aux, gold: int) -> str:
    """Running-tally variant: every verdict carries its running count —
    'frames 2 (1), 5 (2) -> 2' — the answer is a read-off of the last tally."""
    if task == "rooms":
        if not aux:
            return " none -> 0"
        return " rooms " + ", ".join(f"{r} ({k})" for k, r in enumerate(aux, 1)) + f" -> {gold}"
    if evid:
        return (
            " frames "
            + ", ".join(f"{i + 1} ({k})" for k, i in enumerate(sorted(evid), 1))
            + f" -> {gold}"
        )
    return " none -> 0"


def build_target_scan(
    task: str, evid: Set[int], gold: int, NF: int, labels: Optional[Dict[int, str]], caption: bool
) -> str:
    """Arms B/C: full scan — EVERY frame gets a slot in frame order, evidence slots
    increment the tally inline, explicit END terminator. B='yes' verdicts, C=attribute
    captions. rooms carries room words in both arms (tally increments on FIRST visit);
    'which' reads out the frame number in the total slot."""
    labels = labels or {}
    parts: List[str] = []
    k = 0
    seen: Set[str] = set()
    for t in range(NF):
        if t in evid:
            w = labels.get(t, "yes") if (caption or task == "rooms") else "yes"
            if task == "rooms":
                if w not in seen:
                    seen.add(w)
                    k += 1
                    parts.append(f"f{t + 1}:{w}({k})")
                else:
                    parts.append(f"f{t + 1}:{w}")
            else:
                k += 1
                parts.append(f"f{t + 1}:{w}({k})")
        else:
            parts.append(f"f{t + 1}:-")
    return " scan: " + " ".join(parts) + f" | total: {gold} END"


def build_target_chunked(
    task: str, evid: Set[int], gold: int, NF: int, labels: Optional[Dict[int, str]]
) -> str:
    """Arm D (refuted in the sweep, kept for reproducibility): CHUNK_FRAMES-frame blocks,
    positive-list per block (global 1-indexed), '| sub k' per block, sum expression."""
    labels = labels or {}
    parts: List[str] = []
    subs: List[int] = []
    seen: Set[str] = set()
    for c0 in range(0, NF, CHUNK_FRAMES):
        idx = [t for t in sorted(evid) if c0 <= t < c0 + CHUNK_FRAMES]
        if task == "rooms":
            items = []
            for t in idx:
                w = labels.get(t)
                if w and w not in seen:
                    seen.add(w)
                    items.append(w)
        else:
            items = [str(t + 1) for t in idx]
        subs.append(len(items))
        parts.append(
            f"c{len(subs)}: " + (", ".join(items) if items else "none") + f" | sub {len(items)}"
        )
    if task == "which":
        tail = f"total: {gold} END"
    else:
        tail = "total: " + "+".join(str(s) for s in subs) + f" = {gold} END"
    return " " + " ".join(parts) + " " + tail


def build_target_fmt(
    fmt: str,
    task: str,
    evid: Set[int],
    aux,
    gold: int,
    NF: int,
    labels: Optional[Dict[int, str]] = None,
) -> str:
    if fmt == "poslist":
        return build_target_tally(task, evid, aux, gold)
    if fmt in ("scan", "caption"):
        return build_target_scan(task, evid, gold, NF, labels, caption=(fmt == "caption"))
    if fmt == "chunked":
        return build_target_chunked(task, evid, gold, NF, labels)
    raise ValueError(f"unknown scratchpad format {fmt!r}")


def parse_scratchpad_answer(text: str, fmt: str) -> Optional[int]:
    """Eval parser. poslist: integer after the LAST '->'. scan/caption/chunked: anchor on
    the LAST 'total:', take the LAST integer before 'END' (or end-of-text)."""
    if fmt == "poslist":
        mm = re.findall(r"->\s*(\d+)", text)
        return int(mm[-1]) if mm else None
    seg = text.rsplit("total:", 1)
    if len(seg) < 2:
        return None
    mm = re.findall(r"(\d+)", seg[1].split("END")[0])
    return int(mm[-1]) if mm else None


def couple_offsets(token_texts: Sequence[str], NF: int) -> List[Tuple[int, int]]:
    """E-G position-coupling stream rule (REFUTED as a mechanism — kept so the E-G ckpts
    stay decodable): token k -> (anchor carrier 1-indexed, offset within anchor span).
    Deterministic from token strings alone; drives teacher-forced AND greedy positions."""
    out: List[Tuple[int, int]] = []
    anchor, off, seg, tail = 1, 0, "", False
    for t in token_texts:
        out.append((anchor, off))
        off += 1
        seg += t
        if "->" in seg:
            tail = True
        if not tail and "(" not in seg:
            mm = re.search(r"(\d+)", seg)
            if mm:
                m = int(mm.group(1))
                if 1 <= m <= NF and m != anchor:
                    anchor, off = m, 0
        if "," in t:
            seg = ""
    return out
