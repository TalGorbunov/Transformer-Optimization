"""Multiple-choice protocol pieces, ported verbatim from the benchmarks' own evaluation code so
the comparability rows are byte-faithful:
  HERBench   github.com/DanBenAmi/HERBench evaluation/model_wrappers/base_vlm.py
             (_format_prompt, extract_answer_choice)                — fetched 2026-09-23
  MINERVA    lmms-eval lmms_eval/tasks/minerva/{utils.py,_default_template_yaml}
             (minerva_doc_to_text with the default post_prompt, _extract_choice_letter)
One deliberate difference: lmms-eval replaces an unparseable MINERVA answer by a RANDOM letter;
here it is a parse failure (None, scored wrong) so the number is reproducible.
Pinned by tests/test_data_videoqa.py.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

OPTIONS = ("A", "B", "C", "D", "E")

HERBENCH_SUFFIX = ("\n\nPlease respond with only the correct answer letter (A, B, C, D, or E) "
                   "without any explanations or additional text.")
MINERVA_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def lettered(choices: Sequence[str]) -> List[str]:
    """Choices as 'A. …' strings (unchanged when they already carry a letter)."""
    out = []
    for i, c in enumerate(choices):
        c = str(c)
        out.append(c if c.strip()[:1] in "ABCDE" and c.strip()[1:2] in (".", ")") else f"{OPTIONS[i]}. {c}")
    return out


def herbench_prompt(question: str, choices: Sequence[str]) -> str:
    """base_vlm.py::_format_prompt with lettered candidates (the HF rows' `choices`)."""
    candidates = list(choices)
    prompt = f"{question}\n\n"
    if candidates and candidates[0].strip()[0] in "ABCDE":
        prompt += "\n".join(candidates)
    else:
        for i, candidate in enumerate(candidates):
            prompt += f"{chr(65 + i)}. {candidate}\n"
    prompt += HERBENCH_SUFFIX
    return prompt


def herbench_extract(response: str) -> Optional[str]:
    """base_vlm.py::extract_answer_choice; their 'ERROR' is None here."""
    response = str(response).strip()
    if "ERROR" in response.upper() or not response:
        return None
    patterns = [
        r"^([A-E])[\s\.\,\)]",
        r"^\(([A-E])\)",
        r"^([A-E])$",
        r"[Aa]nswer:\s*([A-E])\b",
        r"[Cc]hoice:\s*([A-E])\b",
        r"\b([A-E])\b[\.\,]",
    ]
    for pattern in patterns:
        m = re.search(pattern, response)
        if m:
            return m.group(1).upper()
    upper = response.upper()
    for m in re.finditer(r"([A-E])", upper):
        s, e = m.start(1), m.end(1)
        before = upper[s - 1] if s > 0 else " "
        after = upper[e] if e < len(upper) else " "
        if not before.isalpha() and not after.isalpha():
            return m.group(1)
    return None


def minerva_prompt(question: str, choices: Sequence[str]) -> str:
    """utils.py::minerva_doc_to_text with the yaml defaults (pre_prompt "", post_prompt below).
    `choices` are the bare answer_choice_0..4 strings."""
    choice_text = "\n".join(f"{OPTIONS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n{choice_text}{MINERVA_SUFFIX}"


def minerva_extract(text: str) -> Optional[str]:
    """utils.py::_extract_choice_letter; '' is None here (no random fallback)."""
    cleaned = str(text).strip()
    explicit = re.compile(r"(?:answer|option)\s*(?:is|:)\s*([ABCDE])", re.IGNORECASE)
    ms = list(explicit.finditer(cleaned))
    if ms:
        return ms[-1].group(1).upper()
    for prefix in ("The best answer is", "The correct answer is", "The answer is", "Answer:", "Option:",
                   "Therefore, the final answer is:"):
        cleaned = cleaned.replace(prefix, "")
    ms = list(re.finditer(r"[ABCDE]", cleaned))
    return ms[-1].group(0) if ms else None


def letter_match(pred: Optional[str], gold: str) -> bool:
    return pred is not None and str(pred).strip().upper() == str(gold).strip().upper()


def has_letter(text: str) -> bool:
    """Early-stop rule for cache-free fenced decoding on MCQ: a standalone A–E has appeared."""
    return re.search(r"(?<![A-Za-z])[A-E](?![A-Za-z])", text) is not None
