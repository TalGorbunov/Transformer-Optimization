"""MMRED sample loading and task parsing (park renders, natural-image cells, dirs-files).

Sample-directory conventions:
  park render:   <dir>/qa.txt (question:/answer: + per-frame state dicts) + 000.png..NNN.png
  natural cell:  <dir>/meta.json (question/answer/concept/frames[].is_evidence) + frame_XX.jpg

Task templates (the question decides the task): steps / cooc / union / which / rooms.
Evidence labels derived here are legitimately available at TRAINING time only; eval scores
greedy-decoded answers against the gold count.
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

_STEPS_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)", flags=re.IGNORECASE
)


# ------------------------------------------------------------------ sample loading

def load_mmred_sample(sample_dir: Path):
    """-> (sample_id, frames[PIL], question, states[list of dicts], answer_text)."""
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    lines = (sample_dir / "qa.txt").read_text(encoding="utf-8").splitlines()
    q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "question:"), -1)
    a_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "answer:"), -1)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        raise RuntimeError(f"Bad qa.txt format: {sample_dir / 'qa.txt'}")
    states, question = [], None
    for ln in lines[q_idx + 1 : a_idx]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
            continue
        question = s
        break
    if question is None:
        raise RuntimeError(f"No question line in {sample_dir}")
    answer = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if answer is None:
        raise RuntimeError(f"No answer in {sample_dir}")
    frames = [Image.open(sample_dir / f"{i:03d}.png").convert("RGB") for i in range(len(states))]
    return sample_dir.name, frames, question, states, answer


def load_natural_sample(sample_dir: Path):
    """Natural-image cell -> (frames, question, gold, evidence_frame_set, concept_word)."""
    meta = json.loads((sample_dir / "meta.json").read_text())
    frames = [
        Image.open(sample_dir / f"frame_{i:02d}.jpg").convert("RGB")
        for i in range(int(meta["n_frames"]))
    ]
    evid = {i for i, f in enumerate(meta["frames"]) if f["is_evidence"]}
    return frames, meta["question"], int(meta["answer"]), evid, meta["concept"]


def iter_sample_dirs(data_root: Path) -> List[Path]:
    return [p for p in sorted(data_root.iterdir()) if p.is_dir() and (p / "qa.txt").exists()]


def iter_sample_dirs_shuffled(data_root: Path, seed: int) -> List[Path]:
    """Stratified deterministic shuffle: MMRED dir names sort by class tag, so any
    LIMIT < full would truncate the gold prior. Round-robin over per-class shuffles
    keeps EVERY prefix class-balanced (the 2026-07-18 E1 full-prior fix)."""
    dirs = iter_sample_dirs(data_root)
    rng = random.Random(seed)
    groups: Dict[Optional[str], List[Path]] = {}
    for d in dirs:
        m = re.search(r"_(?:K|e)(\d+)_", d.name)
        groups.setdefault(m.group(1) if m else None, []).append(d)
    if len(groups) <= 1:
        out = list(dirs)
        rng.shuffle(out)
        return out
    keys = sorted(groups, key=lambda k: (k is None, int(k) if k is not None else -1))
    for k in keys:
        rng.shuffle(groups[k])
    out: List[Path] = []
    for i in range(max(len(g) for g in groups.values())):
        for k in keys:
            if i < len(groups[k]):
                out.append(groups[k][i])
    return out


def read_dirs_file(path: Path) -> List[Path]:
    """A dirs-file pins an exact eval set (one sample-dir path per line)."""
    return [Path(ln.strip()) for ln in path.read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------------- state parsing

def rooms_to_room2chars(rooms: Any) -> Dict[str, List[str]]:
    if not isinstance(rooms, dict):
        return {}
    out: Dict[str, List[str]] = {}
    if any(isinstance(v, list) for v in rooms.values()):
        for room_name, chars in rooms.items():
            if not isinstance(room_name, str):
                continue
            norm = room_name[:1].upper() + room_name[1:].lower() if room_name else room_name
            out.setdefault(norm, [])
            if isinstance(chars, list):
                out[norm].extend(str(c) for c in chars)
    else:
        for char_name, room_name in rooms.items():
            if not isinstance(room_name, str):
                continue
            norm = room_name[:1].upper() + room_name[1:].lower()
            out.setdefault(norm, []).append(str(char_name))
    return {room: sorted(set(chars)) for room, chars in out.items()}


def parse_target_character_room(question: str) -> Optional[Tuple[str, str]]:
    m = _STEPS_RE.search(question)
    if not m:
        return None
    character, room = m.group(1).strip(), m.group(2).strip()
    return character, (room[:1].upper() + room[1:].lower() if room else room)


def collect_evidence_frame_indices(question: str, states: Sequence[Dict[str, Any]]) -> List[int]:
    parsed = parse_target_character_room(question)
    if parsed is None:
        return []
    character, room = parsed
    return [
        t
        for t, st in enumerate(states)
        if character in rooms_to_room2chars(st.get("rooms", {}) if isinstance(st, dict) else {}).get(room, [])
    ]


# -------------------------------------------------------------------- task parsing

def parse_task_labels(q0: str, states: Sequence[Dict[str, Any]], gold: int):
    """The question decides the task. -> (task, evidence_frame_set, aux) or None on a
    sanity-check failure (derived evidence must match the gold answer). aux carries the
    visited-room names for the rooms task."""
    mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
    if mm:
        nA, nB = mm.group(1), mm.group(2)
        evid: Set[int] = set()
        for t, st in enumerate(states):
            for occ in (st.get("rooms", {}) or {}).values():
                if nA in occ and nB in occ:
                    evid.add(t)
                    break
        return ("cooc", evid, None) if len(evid) == gold else None
    mm = re.search(r"How many frames was (\w+) in the (\w+) or the (\w+)", q0)
    if mm:
        c, r1, r2 = mm.groups()
        evid = {
            t
            for t, st in enumerate(states)
            if c in ((st.get("rooms", {}) or {}).get(r1, []) or [])
            or c in ((st.get("rooms", {}) or {}).get(r2, []) or [])
        }
        return ("union", evid, None) if len(evid) == gold else None
    mm = re.search(r"In which frame number \(1-\d+\) was (\w+) in the (\w+)", q0)
    if mm:
        c, r = mm.group(1), mm.group(2)
        evid = {t for t, st in enumerate(states) if c in ((st.get("rooms", {}) or {}).get(r, []) or [])}
        return ("which", evid, None) if (len(evid) == 1 and gold - 1 in evid) else None
    mm = re.search(r"How many distinct rooms did (\w+) visit", q0)
    if mm:
        name = mm.group(1)
        rooms_v: Set[str] = set()
        evid = set()
        for t, st in enumerate(states):
            for rname, occ in (st.get("rooms", {}) or {}).items():
                if name in occ:
                    rooms_v.add(rname)
                    evid.add(t)
        return ("rooms", evid, sorted(rooms_v)) if len(rooms_v) == gold else None
    mm = re.search(r"In how many of the \d+ frames does .+ appear", q0)
    if mm and states and isinstance(states[0], dict) and "natural" in states[0]:
        evid = {t for t, st in enumerate(states) if (st.get("natural", {}) or {}).get("evidence")}
        return ("steps", evid, None) if len(evid) == gold else None
    evid = set(collect_evidence_frame_indices(q0, states))
    if not evid and gold != 0:
        return None
    return ("steps", evid, None)


def frame_attr_labels(task: str, q0: str, states: Sequence[Dict[str, Any]], evid: Set[int]) -> Dict[int, str]:
    """Per-evidence-frame attribute word (room / concept) for scan/caption targets.
    Returns {} on parse failure — builders then fall back to 'yes' so no sample is ever
    skipped (train/eval SPLITS must stay byte-identical across format arms)."""
    out: Dict[int, str] = {}

    def room_of(t, pred):
        st = states[t] if t < len(states) else {}
        for rname, occ in ((st.get("rooms", {}) or {}) if isinstance(st, dict) else {}).items():
            if pred(rname, occ or []):
                return rname
        return None

    if task == "cooc":
        mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
        if not mm:
            return {}
        nA, nB = mm.group(1), mm.group(2)
        for t in evid:
            r = room_of(t, lambda rn, oc: nA in oc and nB in oc)
            if r:
                out[t] = r
    elif task == "union":
        mm = re.search(r"How many frames was (\w+) in the (\w+) or the (\w+)", q0)
        if not mm:
            return {}
        c, r1, r2 = mm.group(1), mm.group(2), mm.group(3)
        for t in evid:
            st = states[t] if t < len(states) else {}
            rooms = (st.get("rooms", {}) or {}) if isinstance(st, dict) else {}
            out[t] = r1 if c in (rooms.get(r1, []) or []) else r2
    elif task == "which":
        mm = re.search(r"In which frame number \(1-\d+\) was (\w+) in the (\w+)", q0)
        if not mm:
            return {}
        for t in evid:
            out[t] = mm.group(2)
    elif task == "rooms":
        mm = re.search(r"How many distinct rooms did (\w+) visit", q0)
        if not mm:
            return {}
        name = mm.group(1)
        for t in evid:
            r = room_of(t, lambda rn, oc: name in oc)
            if r:
                out[t] = r
    else:  # steps: the queried room (constant over evid); natural cells use the concept
        if states and isinstance(states[0], dict) and "natural" in states[0]:
            con = (states[0].get("natural", {}) or {}).get("concept")
            return {t: str(con) for t in evid} if con else {}
        pr = parse_target_character_room(q0)
        if not pr:
            return {}
        for t in evid:
            out[t] = pr[1]
    return out


# ------------------------------------------------------------------ prompt builders

def build_count_prompt(question: str, num_frames: int) -> str:
    """The canonical MMRED counting prompt (PROMPT-CRITICAL: results depend on this
    exact wording; ported byte-identical from legacy patching_core.build_prompt)."""
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). "
        "Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_prompt_inputs(processor: Any, frames: Sequence[Any], prompt: str):
    """[frames..., prompt] chat-templated tensor inputs (images first, then the text)."""
    messages = [{
        "role": "user",
        "content": ([{"type": "image", "image": im} for im in frames]
                    + [{"type": "text", "text": prompt}]),
    }]
    return dict(processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"))


def probe_evidence(task: str, q0: str, states, gold: int, rooms: Sequence[str]):
    """Supply-probe labels: -> (evidence_frame_set, locus_word) or None.
    steps: locus = the queried room word; cooc: locus = the SECOND character name."""
    if task == "cooc":
        mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
        if not mm:
            return None
        nA, nB = mm.group(1), mm.group(2)
        evid: Set[int] = set()
        for t, st in enumerate(states):
            for occ in (st.get("rooms", {}) or {}).values():
                if nA in occ and nB in occ:
                    evid.add(t)
                    break
        if len(evid) != gold:
            return None
        return evid, nB
    evid = set(collect_evidence_frame_indices(q0, states))
    if not evid and gold != 0:
        return None
    room = next((r for r in rooms if r.lower() in q0.lower()), None)
    if room is None:
        return None
    return evid, room
