"""The paper's prompt, verbatim, and the message layouts the method uses.

Twin (for the prompt): data/mmred_hf/upstream_repo/scripts/openai_server_inference.py —
SYSTEM_PROMPT (line 68) and process_row (line 199: images then question, or question then
images with --prefix_question). Twin (for the parser): .../scripts/utils/parse_answers.py.
tests/test_prompt.py pins SYSTEM_PROMPT byte-identical to the upstream file and parse_answer
against a fixture of raw model outputs scored by the upstream parser.

Layouts (build_messages):
    paper           [system] [frame_1 … frame_N] [question]           — their default
    question-first  [system] [question] [frame_1 … frame_N]           — their --prefix_question;
                                                                        THE method's layout
    replica         [system] [question] [frame_i, question]×N          — the S1–S11 layout
                                                                        (upper bound / fallback)
The fence never changes the words: with layout != paper the only change is where they sit.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .constants import CHARS, NOBODY, ROOMS

SYSTEM_PROMPT = (
    "You are an assistant that analyzes sequences of human agents moving in an environment.\n"
    "If room contains a [\"?\"], it's masked and you should infer information from surrounding elements of sequence.\n"
    "Format your response as a following json:\n"
    "{ \"answer\": <value> }\n"
    "\n"
    "Where <value> is:\n"
    "- A **single room name** (e.g., \"Kitchen\") for location answers.\n"
    "- A **number** (e.g., \"3\") for counting answers.\n"
    "- A **single person name** (e.g., \"Michael\") for people answers or \"Nobody\" if no person satisfies given conditions."
)

LAYOUTS = ("paper", "question-first", "replica")


def build_messages(frames: Sequence[Any], question: str, layout: str = "question-first",
                   answer: Optional[str] = None) -> List[Dict[str, Any]]:
    """Chat messages for the processor's apply_chat_template. With `answer` given (training),
    an assistant turn `{ "answer": <answer> }` is appended — the paper's output format is the
    training target too. Pinned by tests/test_prompt.py::test_layouts."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    q = {"type": "text", "text": question}
    imgs = [{"type": "image", "image": f} for f in frames]
    if layout == "paper":                       # upstream process_row default: images, then question
        content = imgs + [q]
    elif layout == "question-first":            # upstream --prefix_question
        content = [q] + imgs
    else:                                       # replica: question, then (frame, question) per frame
        content = [q]
        for im in imgs:
            content += [im, dict(q)]
    # Every turn's content is a typed list: the HF processor (4.57) iterates message content
    # as items and breaks on a bare string. The chat template renders text items verbatim.
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                                  {"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer_target(answer)}]})
    return msgs


def answer_target(answer: str) -> str:
    """The exact assistant text the read is trained to emit: '{ "answer": "<answer>" }'.
    The value is always quoted, as in the SYSTEM_PROMPT's own examples ("3", "Kitchen")."""
    return '{ "answer": "' + str(answer) + '" }'


# Upstream parse_answers.py builds these from mmred.const at import time; people include NOBODY.
_ROOM_NAMES = list(ROOMS)
_PEOPLE_NAMES = list(CHARS) + [NOBODY]
_ANSWER_RE = re.compile("(" + "|".join(_ROOM_NAMES + _PEOPLE_NAMES + [r"\d"]) + ")", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"[<>'\"`,.]*(\d+)")              # upstream numeric_pattern
_CANON = {n.lower(): n for n in _ROOM_NAMES + _PEOPLE_NAMES}


def _strip_until_first_brace(text: str) -> str:
    """Upstream strip_until_first_brace: only acts when a </think> block precedes the answer."""
    think_end = text.find("</think>")
    if think_end == -1:
        return text
    text = text[think_end:]
    text = text.replace("</answer>", "").replace("<answer>", "").strip()
    brace = text.find("{")
    return text if brace == -1 else text[brace:]


def parse_answer(text: str) -> Optional[str]:
    """Port of upstream parse_answers.py (strip_until_first_brace + parse_predicted_answer):
    JSON (repaired) first, then the room/person/number regex; numbers are returned as their
    integer string. None if nothing parses (upstream returns the string "None")."""
    from json_repair import repair_json
    import json

    if text is None:
        return None
    text = _strip_until_first_brace(str(text))
    value: Any = None
    try:
        parsed = json.loads(repair_json(text))
        assert isinstance(parsed, dict)
        value = parsed.get("answer", "None")
        if "no" in str(value).lower():          # upstream: any "no" → Nobody (covers "Nobody", "None")
            value = NOBODY
        if isinstance(value, list):             # upstream: one name → that name, several → Nobody
            value = value[0] if len(value) == 1 else NOBODY
    except Exception:
        m = _NUMERIC_RE.search(text)
        if m:
            value = int(m.group(1))
        else:
            m = _ANSWER_RE.search(text)
            value = m.group(0) if m else "None"
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    out = str(value).strip().strip('"\'')
    if out.lower() in ("", "none"):
        return None
    return _CANON.get(out.lower(), out)


def exact_match(pred: Optional[str], gold: str) -> bool:
    """Case-insensitive string equality after parse_answer normalisation (numbers compared as
    ints). Two upstream validate_and_compare rules are folded in: on a numeric gold a "Nobody"
    prediction counts as 0, and a prediction with digits is compared by its first integer."""
    if pred is None:
        return False
    g = str(gold).strip().strip('"\'')
    p = str(pred).strip().strip('"\'')
    if g.lstrip("-").isdigit():
        if p.lower() == NOBODY.lower():
            p = "0"
        m = re.search(r"\d+", p)
        return m is not None and int(m.group(0)) == int(g) and p.lstrip("-").isdigit()
    return p.lower() == g.lower()
