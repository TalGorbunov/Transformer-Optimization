"""REDUX task registry — THE one label module (prompt template, gold derivation,
answer parsing) for {count, exists, majority}. Every REDUX script (datagen checks,
exam builder, behavioral evals, trainer mixture, sensitivity probes) imports from
here so behavioral and probe cells cannot drift apart.

Conventions:
- count keeps gnnformer.data.build_count_prompt BYTE-IDENTICAL (PROMPT-CRITICAL).
- exists/majority reuse the count scaffold's opener line, so the fenced trainer's
  parse_layout ("You will be shown" needle) works unchanged for all three tasks.
- Gold is ALWAYS derived from states and (for count) asserted against the stored
  answer; mismatches are skip+count, never silently used.
- Answer parsing: count = first integer; yes/no = first alphabetic word,
  case-insensitive, must be exactly yes/no.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import build_count_prompt, parse_target_character_room  # noqa: E402

TASKS = ("count", "exists", "majority")
INTEGER_RE = re.compile(r"[+-]?\d+")
WORD_RE = re.compile(r"[A-Za-z]+")


# ------------------------------------------------------------------ target lookup

def target_of(sample_dir: Path, q0: str) -> Optional[Tuple[str, str]]:
    """(character, room) for this sample: metadata.json target_* if present
    (redux/balanced pools), else parsed from the question (park pools)."""
    meta_p = Path(sample_dir) / "metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            c, r = meta.get("target_character"), meta.get("target_room")
            if c and r:
                return str(c), str(r)
        except Exception:
            pass
    return parse_target_character_room(q0)


def evidence_count(states: Sequence[Dict[str, Any]], char: str, room: str) -> int:
    k = 0
    for st in states:
        occ = (st.get("rooms", {}) or {}).get(room, []) or []
        if char in occ:
            k += 1
    return k


# ------------------------------------------------------------------ prompts

def build_prompt(task: str, char: str, room: str, num_frames: int, q0: str = "") -> str:
    if task == "count":
        # byte-identical scaffold: q0 is the pool's own question text
        return build_count_prompt(q0, num_frames)
    if task == "exists":
        return (
            f"You will be shown {num_frames} frames describing steps in a house.\n"
            "Respond with a single word: yes or no.\n"
            f"Question: Was {char} ever in the {room}?\n"
            "Answer: "
        )
    if task == "majority":
        return (
            f"You will be shown {num_frames} frames describing steps in a house.\n"
            "Respond with a single word: yes or no.\n"
            f"Question: Was {char} in the {room} in more than half of the "
            f"{num_frames} frames?\n"
            "Answer: "
        )
    raise ValueError(f"unknown task {task}")


def replica_text(task: str, char: str, room: str, num_frames: int, q0: str = "") -> str:
    """The per-block replica question (fenced trainer): the bare question sentence."""
    if task == "count":
        return q0
    if task == "exists":
        return f"Was {char} ever in the {room}?"
    if task == "majority":
        return (f"Was {char} in the {room} in more than half of the "
                f"{num_frames} frames?")
    raise ValueError(f"unknown task {task}")


# ------------------------------------------------------------------ gold + parsing

def derive_gold(task: str, states: Sequence[Dict[str, Any]], char: str, room: str,
                num_frames: int) -> str:
    """Gold answer STRING for the task, derived from states."""
    k = evidence_count(states, char, room)
    if task == "count":
        return str(k)
    if task == "exists":
        return "yes" if k >= 1 else "no"
    if task == "majority":
        return "yes" if k > num_frames / 2 else "no"
    raise ValueError(f"unknown task {task}")


def parse_answer(task: str, text: str) -> Optional[str]:
    """Normalized predicted answer string, or None on parse failure."""
    if task == "count":
        m = INTEGER_RE.search(text)
        return m.group(0).lstrip("+") if m else None
    m = WORD_RE.search(text)
    if not m:
        return None
    w = m.group(0).lower()
    return w if w in ("yes", "no") else None


def task_of_dirs_file(path: str) -> str:
    """Infer the task from an exam dirs-file name (exam_{task}_N*.txt); count default."""
    name = Path(path).name
    for t in TASKS:
        if f"_{t}_" in name or name.startswith(t):
            return t
    return "count"
