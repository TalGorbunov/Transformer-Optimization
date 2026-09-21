#!/usr/bin/env python3
"""TREEFOLD shared plumbing — sampling, prompts, generation, JSON, fidelity.

Anchors imported READ-ONLY from the recagg campaign (never edited; their logged
numbers are law):
  ask_compile_execute.py -> SEEN_TYPES, norm(), run_program() sandbox, type_of()
  armB_execute.py        -> rebuild_task_dirs() (the exact armC v3 task order)
  caption_frames.py      -> sample_dirs() (armC stride + Arm B subsetting), NAMES
  load_hf_sample.py      -> parse_qa(), ROOMS_HF

House rules honored here: strided sampling everywhere (K0-sorted trap), every
generated text saved for audit, all parse failures surfaced as separate
counters (never silently wrong), one frozen model (7B nf4 via gnnformer.runtime)
for every stage — text stages go through the VL processor.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "scripts/ninv"), str(_REPO / "scripts/recagg")):
    if p not in sys.path:
        sys.path.insert(0, p)

from load_hf_sample import ROOMS_HF, parse_qa                     # noqa: E402
from ask_compile_execute import (SEEN_TYPES, norm, run_program,   # noqa: E402
                                 type_of)
from armB_execute import rebuild_task_dirs                        # noqa: E402
from caption_frames import NAMES, sample_dirs                     # noqa: E402

__all__ = ["ROOMS_HF", "NAMES", "SEEN_TYPES", "norm", "run_program", "type_of",
           "rebuild_task_dirs", "sample_dirs", "parse_qa", "COUNT_LIKE",
           "V3_PROGRAMS", "build_tasks", "load_v3_program_map", "extract_json",
           "render_state", "ask_prompt", "leaf_text_prompt", "leaf_vlm_prompt",
           "merge_prompt", "answer_prompt", "Gen", "match_val",
           "choose_count_field", "parse_ns"]

# Brief §T1: the answer-equals-note (count-like) types for the fidelity hooks.
COUNT_LIKE = ("steps_in_room", "crowd_count", "n_empty", "spend_alone",
              "spend_together", "rooms_visited", "n_room_on_char_first_app",
              "n_room_on_char_final_app", "n_char_at_frame")

V3_PROGRAMS = (_REPO / "outputs/recagg/armC_compile/20260817_193946_qwen14b_v3"
               / "programs.json")


def parse_ns(s: str) -> List[int]:
    """'16,32,64,128' / '16-32' / '16 32' all work (sbatch comma-split dodge)."""
    return [int(x) for x in s.replace(",", " ").replace("-", " ").split()]


def build_tasks(per_type: int = 8, ns=(16, 32, 64, 128), limit: int = 0):
    """Task dicts in deterministic order (sample_dirs = the armC stride; per_type=4
    yields the Arm B subset). limit>0 takes a STRIDE across the list, never a head."""
    tasks = []
    for n in ns:
        root = _REPO / f"data/mmred_hf/dirs/seq_len_{n}_test"
        for d in sample_dirs(root, per_type):
            q, states, a = parse_qa(d / "qa.txt")
            tasks.append({"N": n, "type": type_of(d.name), "dir": d.name,
                          "path": str(d), "q": q, "gold": str(a).strip(),
                          "n_frames": len(states)})
    if limit and limit < len(tasks):
        step = max(1, len(tasks) // limit)
        tasks = tasks[::step][:limit]
    return tasks


def load_v3_program_map(path: Path = V3_PROGRAMS, per_type: int = 8
                        ) -> Dict[str, str]:
    """dir name -> canonical v3 program code (armB_execute's index-zip pattern)."""
    recs = json.load(open(path))
    tasks = rebuild_task_dirs(per_type)
    assert len(tasks) == len(recs), "programs.json / task-order mismatch"
    return {d.name: r["code"] for (_, _, d), r in zip(tasks, recs)}


def extract_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """First balanced {...} block, strict json.loads. No repair, no retry."""
    i = text.find("{")
    if i < 0:
        return None, "no-brace"
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1]), None
                except Exception as e:                      # noqa: BLE001
                    return None, f"json:{e}"[:80]
    return None, "unbalanced"


def render_state(rooms: Dict[str, list], step: int, n_steps: int) -> str:
    """One frame's room occupancy as text (all 6 rooms listed — empty-room
    questions need the absences to be explicit)."""
    parts = [f"{r}: {', '.join(rooms.get(r, [])) or '(empty)'}" for r in ROOMS_HF]
    return f"Step {step} of {n_steps} — " + "; ".join(parts)


# --------------------------------------------------------------------- prompts
# Conventions = task definition (armC v3 parity), not a hint.
CONVENTIONS = """Setting (fixed, applies to every question):
- Characters: Mary, Michael, John, Daniel, Sandra. Rooms: Kitchen, Bathroom, Garden, Office, Bedroom, Hallway.
- The sequence is a series of frames; frame k shows step k (steps are 1-based).
- A character "appears" in a room at every step they are in that room; their FIRST appearance is the earliest such step and their FINAL appearance the latest such step.
- "Together" means being in the same room at the same step; "alone" means being the only person in that room at that step.
- If no one or no room matches, the answer is "Nobody"."""

ASK_HEADER = """You design a plan for answering a question about a sequence of frames WITHOUT seeing any frame. The plan is executed by workers: each worker sees either ONE frame, or only the notes of adjacent segments. Design a NOTE format that summarizes any contiguous segment of steps with just enough information, so that the notes of adjacent segments can be combined into the note of the union segment, and the note covering all steps answers the question."""

ASK_FORMAT = """Reply with ONLY one JSON object, exactly this shape:
{"note": {"<field>": "<type> - <meaning for a segment of steps>", ...},
 "leaf": "<how to fill the note from ONE frame (a segment of a single step)>",
 "merge": "<how to fill the note for LEFT+RIGHT adjacent segments, using ONLY left.note and right.note>",
 "answer": "<how to read the final answer from the note covering all steps>",
 "combinable": true}
The "leaf", "merge" and "answer" values must be short instructions WRITTEN IN WORDS that a worker can follow — never a JSON template or placeholder expressions. They may refer to note fields by name (e.g. to left.<field> and right.<field> in the merge).
Set "combinable": false (other fields empty) only if no such note format can answer this question."""

# Few-shot control: armC's 4 SEEN exemplar questions, rewritten as notes.
FEWSHOT = """Examples:

Question: How many steps did John spend in the Bedroom?
{"note": {"count": "int - number of steps in this segment where John is in the Bedroom"},
 "leaf": "count = 1 if John is in the Bedroom in this frame, else 0.",
 "merge": "count = left.count + right.count.",
 "answer": "The answer is count.",
 "combinable": true}

Question: How many different rooms did Sandra visit?
{"note": {"rooms": "list of strings - the distinct rooms Sandra appears in within this segment"},
 "leaf": "rooms = [the room Sandra is in in this frame] (or [] if she is in no room).",
 "merge": "rooms = the union of left.rooms and right.rooms, without duplicates.",
 "answer": "The answer is the number of rooms in the list.",
 "combinable": true}

Question: In which room did Michael spend the most time?
{"note": {"counts": "object mapping room name to int - steps Michael spent in each room in this segment"},
 "leaf": "counts = {the room Michael is in: 1} (or {} if he is in no room).",
 "merge": "counts = per-room sum of left.counts and right.counts.",
 "answer": "The answer is the room with the largest count in counts.",
 "combinable": true}

Question: In which room was Daniel at step 11?
{"note": {"room": "string or null - the room Daniel is in at step 11, or null if step 11 is not inside this segment"},
 "leaf": "room = the room Daniel is in if this frame is step 11, otherwise null.",
 "merge": "room = left.room if left.room is not null, else right.room.",
 "answer": "The answer is room.",
 "combinable": true}
"""


def ask_prompt(question: str, mode: str) -> str:
    parts = [ASK_HEADER, "", CONVENTIONS, "", ASK_FORMAT, ""]
    if mode == "few":
        parts += [FEWSHOT, ""]
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def leaf_text_prompt(ask: dict, step: int, n_steps: int, rendered: str) -> str:
    return "\n".join([
        "You are filling a note about ONE frame (a single step) of a sequence,"
        " as part of a bigger plan.", "", CONVENTIONS, "",
        "Note fields (fill every field):",
        json.dumps(ask["note"]),
        f"Leaf instructions: {ask['leaf']}", "",
        f"This frame is step {step} (of steps 1..{n_steps}).",
        f"Frame contents — {rendered}", "",
        f"Reply with ONLY the JSON note for this single step"
        f" (a segment containing just step {step}).",
    ])


def leaf_vlm_prompt(ask: dict, question: str, step: int, n_steps: int) -> str:
    """Question-conditioned VLM leaf (brief §2 MAP: frame + question + leaf rule
    + schema). The image is attached by the caller."""
    return "\n".join([
        "You are filling a note about ONE frame (a single step) of a sequence,"
        " as part of a plan to answer a question.", "", CONVENTIONS, "",
        f"Question (for context): {question}",
        "Note fields (fill every field):",
        json.dumps(ask["note"]),
        f"Leaf instructions: {ask['leaf']}", "",
        f"The attached image is step {step} (of steps 1..{n_steps}). Room names"
        " are written on the floor plan.", "",
        f"Reply with ONLY the JSON note for this single step"
        f" (a segment containing just step {step}).",
    ])


def minimal_leaf_prompt(ask: dict, step: int, n_steps: int, rendered: str) -> str:
    """Prompt-repair variant (ablation 2026-08-25: conventions+plan framing induce
    a conjunction yes-bias — 0.500 → 0.984 leaf acc when stripped)."""
    return "\n".join([
        f"Frame contents — {rendered}",
        f"This frame is step {step} (of steps 1..{n_steps}).",
        f"Instructions: {ask['leaf']}",
        "Note fields: " + json.dumps(ask["note"]),
        "Reply with ONLY the JSON note for this frame.",
    ])


def minimal_answer_prompt(ask: dict, question: str, n_steps: int,
                          root_note: dict) -> str:
    """Answer twin of minimal_leaf_prompt (read-off failed under the same clutter).
    NO Nobody clause: v1 carried one and the 7B pattern-matched it onto numeric
    answers (root {"count": 0..12} -> "Nobody"; 2026-08-25 read-off bug)."""
    return "\n".join([
        f"Question: {question}",
        f"A note summarizing all steps 1-{n_steps}:",
        json.dumps(root_note),
        f"How to read the answer: {ask['answer']}",
        "Reply with ONLY the final answer, nothing else.",
    ])


def merge_prompt(ask: dict, children: List[Tuple[int, int, dict]]) -> str:
    """children = [(start, end, note), ...] adjacent, in order."""
    a, z = children[0][0], children[-1][1]
    lines = [
        "You combine the notes of adjacent segments of steps into ONE note for"
        " the combined segment, as part of a bigger plan.", "",
        "Note fields (fill every field):",
        json.dumps(ask["note"]),
        f"Merge instructions (combines two adjacent segments; apply it"
        f" left-to-right across all segments listed below): {ask['merge']}", ""]
    for s, e, note in children:
        span = f"step {s}" if s == e else f"steps {s}-{e}"
        lines.append(f"Note for {span}:")
        lines.append(json.dumps(note))
    lines += ["", f"Reply with ONLY the JSON note for steps {a}-{z}."]
    return "\n".join(lines)


def answer_prompt(ask: dict, question: str, n_steps: int, root_note: dict) -> str:
    return "\n".join([
        CONVENTIONS, "",
        f"Question: {question}",
        f"A note summarizing all steps 1-{n_steps}:",
        json.dumps(root_note),
        f"How to read the answer: {ask['answer']}", "",
        "Reply with ONLY the final answer — a single number or name (or a"
        " comma-separated list), nothing else.",
    ])


# ------------------------------------------------------------------ generation
class Gen:
    """Batched greedy generation with the ONE campaign model (7B nf4, house
    runtime). Text-only stages go through the VL processor; padding is LEFT
    (prompts vary in length — the caption_frames right-pad was safe only
    because its prompts were identical)."""

    def __init__(self, batch: int = 48):
        from gnnformer.runtime import load_runtime
        t0 = time.time()
        self.rt = load_runtime()          # 7B is the hardcoded campaign model
        self.batch = batch
        tok = self.rt.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"[gen] loaded {self.rt.model_name} in {time.time()-t0:.0f}s",
              flush=True)

    def _wrap(self, text: str, with_image: bool = False) -> str:
        content = ([{"type": "image"}] if with_image else []) + [
            {"type": "text", "text": text}]
        return self.rt.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)

    def text(self, prompts: List[str], max_new: int, batch: int = 0,
             tag: str = "", log_every: int = 20) -> List[str]:
        import torch
        bs = batch or self.batch
        tok = self.rt.processor.tokenizer
        outs: List[str] = []
        t0 = time.time()
        for i in range(0, len(prompts), bs):
            chunk = [self._wrap(p) for p in prompts[i:i + bs]]
            enc = self.rt.processor(text=chunk, return_tensors="pt",
                                    padding=True).to(self.rt.device)
            with torch.no_grad():
                g = self.rt.model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id)
            for k in range(len(chunk)):
                outs.append(tok.decode(g[k, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip())
            if log_every and (i // bs) % log_every == 0:
                print(f"  [{tag}] {len(outs)}/{len(prompts)} "
                      f"{time.time()-t0:.0f}s", flush=True)
        return outs

    def vlm(self, images: list, prompts: List[str], max_new: int,
            batch: int = 0) -> List[str]:
        """images[i] goes with prompts[i]."""
        import torch
        bs = batch or self.batch
        tok = self.rt.processor.tokenizer
        outs: List[str] = []
        for i in range(0, len(prompts), bs):
            imgs = images[i:i + bs]
            chunk = [self._wrap(p, with_image=True)
                     for p in prompts[i:i + bs]]
            enc = self.rt.processor(text=chunk, images=imgs,
                                    return_tensors="pt",
                                    padding=True).to(self.rt.device)
            with torch.no_grad():
                g = self.rt.model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id)
            for k in range(len(chunk)):
                outs.append(tok.decode(g[k, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip())
        return outs


# -------------------------------------------------------------------- fidelity
def match_val(v: Any, gold: Any) -> bool:
    """Note-field value vs program output. A list/dict field matches an int
    gold by its length (notes often carry the SET whose size is the answer)."""
    if v is None or gold is None:
        return False
    if isinstance(v, (list, tuple, set, dict)) and isinstance(gold, int):
        return len(v) == gold
    try:
        return norm(v) == norm(gold)
    except Exception:                                       # noqa: BLE001
        return False


def choose_count_field(leaf_notes: List[Optional[dict]],
                       leaf_golds: List[Optional[Any]]
                       ) -> Tuple[Optional[str], float]:
    """Pick the note field that best reproduces the program's single-frame
    answers (max agreement over frames with defined gold + parsed note).
    Returns (field, agreement); (None, 0.0) if nothing reaches 0.5 — reported
    as schema-not-restriction-shaped, an instrument-coverage miss, not an EM
    miss. Instrument only: EM never touches this."""
    fields: Dict[str, List[bool]] = {}
    for note, gold in zip(leaf_notes, leaf_golds):
        if not isinstance(note, dict) or gold is None:
            continue
        for f, v in note.items():
            fields.setdefault(f, []).append(match_val(v, gold))
    best, best_acc = None, 0.0
    for f, hits in fields.items():
        acc = sum(hits) / len(hits)
        if acc > best_acc:
            best, best_acc = f, acc
    return (best, best_acc) if best_acc >= 0.5 else (None, best_acc)
