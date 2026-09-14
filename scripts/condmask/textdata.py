"""condmask text data: loading, frame rendering, and the span-recording prompt builder.

Framewise-token guarantee: the prompt is built SEGMENT BY SEGMENT — each segment is
tokenized separately (add_special_tokens=False) and the ids are concatenated, so the
per-frame token spans are exact BY CONSTRUCTION and the model sees exactly this
segmentation. No needle matching anywhere. Every frame segment starts with "\\n" so BPE
never merges content across block boundaries.

Prompt style: RAW text (no chat template), fixed once — the tail ends with the
PROMPT-CRITICAL canonical wording of scripts/gating/probe_text_triple.py:80-89 /
gnnformer.data.build_count_prompt ("Question: {q}\\nAnswer: "). The question ALSO leads
the prompt (prefix) so blockwise heads can compute per-frame relevance; blockwise frame
tokens see own frame + prefix only.
"""
from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------------- loading

def load_text_sample(sample_dir: Path) -> Optional[Tuple[str, List[str], str, int]]:
    """Text MMReD dir -> (sid, state_lines, question, gold). None if malformed.
    Copy of scripts/gating/probe_text_triple.py:load_text_sample (same qa.txt grammar
    as gnnformer.data.load_mmred_sample, minus PNG loading)."""
    f = sample_dir / "qa.txt"
    if not f.exists():
        return None
    lines = f.read_text(encoding="utf-8").splitlines()
    q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "question:"), -1)
    a_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "answer:"), -1)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None
    states, question = [], None
    for ln in lines[q_idx + 1 : a_idx]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            ast.literal_eval(s)  # validate; rendering re-parses below
            states.append(s)
            continue
        question = s
        break
    if question is None or not states:
        return None
    ans = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if ans is None or not ans.isdigit():
        return None
    return sample_dir.name, states, question, int(ans)


def render_frame_line(state_line: str) -> str:
    """State-dict repr -> one natural-language line, deterministic.
    {'step_id': 3, 'rooms': {'Kitchen': ['Sandra'], 'Garden': ['John','Mary'], ...}}
    -> "Frame 3: Sandra is in the Kitchen. John and Mary are in the Garden."
    Rooms in dict order (the generator's room order); characters as stored (sorted by
    the generator). Empty rooms omitted."""
    st = ast.literal_eval(state_line)
    parts = []
    for room, chars in st["rooms"].items():
        if not chars:
            continue
        if len(chars) == 1:
            parts.append(f"{chars[0]} is in the {room}.")
        else:
            names = ", ".join(chars[:-1]) + f" and {chars[-1]}"
            parts.append(f"{names} are in the {room}.")
    return f"Frame {st['step_id']}: " + " ".join(parts)


# ------------------------------------------------------------- span-recording prompt

def build_segmented_prompt(tok: Any, states: List[str], question: str,
                           gold: int, sid: str) -> Optional[Dict[str, Any]]:
    """-> {ids, prefix_end, blocks[(a,b)], fin, seq, gold, sid, n_frames, text} or None.

    Segments (each tokenized separately, ids concatenated — spans exact by construction):
      prefix : "You will be shown {N} frames describing steps in a house.
                Question: {question}"
      block_i: "\\nFrame k: ..." (one rendered frame line, leading \\n = block boundary)
      tail   : "\\nQuestion: {question}\\nAnswer: "  (canonical, PROMPT-CRITICAL ending)
    """
    n = len(states)
    prefix_txt = (f"You will be shown {n} frames describing steps in a house.\n"
                  f"Question: {question}")
    frame_txts = ["\n" + render_frame_line(s) for s in states]
    tail_txt = f"\nQuestion: {question}\nAnswer: "

    ids: List[int] = list(tok(prefix_txt, add_special_tokens=False).input_ids)
    prefix_end = len(ids)
    blocks: List[Tuple[int, int]] = []
    for ft in frame_txts:
        seg = tok(ft, add_special_tokens=False).input_ids
        if not seg:
            return None
        blocks.append((len(ids), len(ids) + len(seg)))
        ids.extend(seg)
    fin = len(ids)
    tail_ids = tok(tail_txt, add_special_tokens=False).input_ids
    ids.extend(tail_ids)
    # framewise-token integrity, asserted at build time (repo lesson: never trust spans)
    if blocks[0][0] != prefix_end or blocks[-1][1] != fin:
        return None
    for (a, b), ft in zip(blocks, frame_txts):
        if b <= a:
            return None
        if tok.decode(ids[a:b]) != ft:
            return None
    return {"ids": ids, "prefix_end": prefix_end, "blocks": blocks, "fin": fin,
            "seq": len(ids), "gold": gold, "sid": sid, "n_frames": n,
            "text": prefix_txt + "".join(frame_txts) + tail_txt}


def annotate_prompt(rec: Dict[str, Any], tok: Any) -> str:
    """Human-eyeball dump: the prompt with block boundaries marked (|Bk|...)."""
    ids = rec["ids"]
    out = [tok.decode(ids[: rec["prefix_end"]])]
    for i, (a, b) in enumerate(rec["blocks"]):
        out.append(f"|B{i}|{tok.decode(ids[a:b])}")
    out.append(f"|TAIL|{tok.decode(ids[rec['fin']:])}")
    return "".join(out)


# ------------------------------------------------------------------------- datasets

def prep_root(tok: Any, root: Path, limit: int, seed: int = 0,
              gold_max: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load up to `limit` samples. Dir names embed the gold count (…_K{K}_…), so the
    dir list is SHUFFLED (seeded) BEFORE limiting — the repo's gold-sorted-slice trap."""
    dirs = [p for p in sorted(Path(root).iterdir())
            if p.is_dir() and (p / "qa.txt").exists()]
    random.Random(seed).shuffle(dirs)
    out: List[Dict[str, Any]] = []
    for sd in dirs:
        if len(out) >= limit:
            break
        loaded = load_text_sample(sd)
        if loaded is None:
            continue
        sid, states, question, gold = loaded
        if gold_max is not None and gold > gold_max:
            continue
        rec = build_segmented_prompt(tok, states, question, gold, sid)
        if rec is not None:
            out.append(rec)
    return out


def majority_baseline(golds: List[int]) -> Tuple[int, float]:
    """(majority class, its frequency) — every accuracy must be read against this."""
    counts: Dict[int, int] = {}
    for g in golds:
        counts[g] = counts.get(g, 0) + 1
    cls = max(counts, key=lambda k: counts[k])
    return cls, counts[cls] / max(len(golds), 1)
