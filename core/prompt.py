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
    raise NotImplementedError


def answer_target(answer: str) -> str:
    """The exact assistant text the read is trained to emit: '{ "answer": "<answer>" }'."""
    raise NotImplementedError


def parse_answer(text: str) -> Optional[str]:
    """Port of upstream parse_answers.py: JSON (repaired) first, then the room/person/number
    regex; numbers are returned as their integer string. None if nothing parses."""
    raise NotImplementedError


def exact_match(pred: Optional[str], gold: str) -> bool:
    """Case-insensitive string equality after parse_answer normalisation (numbers compared as ints)."""
    raise NotImplementedError
