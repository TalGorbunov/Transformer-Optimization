"""ONE-TIME PORT CHECKS: deviations from the official protocol; never used for a reported row.

The logged anchors the port must reproduce were produced under legacy settings, so a faithful
evaluator (native 512 px, SYSTEM_PROMPT verbatim, bare question, upstream parser) cannot hit
them. This module is the only code that knows those settings. Each is a verbatim port with its
legacy line cited; evaluate.py applies a preset via `--port-check {arm-a,p2}` and stamps
"PORT CHECK (deviation)" on line 1 of report.txt.

  arm-a  outputs/mmred_hf/frozen/grid_seq8_test  0.533 (640/1200)  and  frozen/seq_len_8_test_127776
         final_app 0.765 / steps_in_room 0.559 / where_spend 0.353 (34 per qtype, JSON order):
         legacy/v1/scripts/mmred_hf/eval_frozen.py — paper layout, frames PIL-resized to 392,
         user text "Question: {q}", model.generate(max_new_tokens=32), the regex ladder below.
  p2     outputs/loramech/p2_fenced_hf/20260827_193851_fenced  0.900 / 0.660 / 0.300 @ 8/16/32
         (steps_in_room test, 50 rows each): legacy/v1/scripts/loramech/train_sft_fenced.py —
         no system turn, [frame_i, question]×N then the integer count prompt, blocks
         [vs_i, vs_{i+1}) closed at the "You will be shown" needle, cache-free greedy, 4 tokens,
         stop at newline, first integer.

Resize: never a resize of ours. The processor's own `max_pixels` knob is set to 392*392 + 1:
Qwen's smart_resize floors 392*392 to 364 px (tests/test_port_check.py pins 364 vs 392).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.constants import CHARS, NOBODY, ROOMS
from core.fence import Span, find_subseq

MAX_PIXELS_392 = 392 * 392 + 1      # 153664 -> (364, 364); 153665 -> (392, 392) = 196 tokens
LEGACY_DECODE_ARM_A = 32            # eval_frozen.py --decode default
LEGACY_MAX_NEW_P2 = 4               # train_sft_fenced.py MAX_NEW
INTEGER_RE = re.compile(r"[+-]?\d+")  # redux/tasks.py INTEGER_RE


def set_max_pixels(processor: Any, max_pixels: int) -> None:
    """The processor's own resize bound (transformers 4.57 reads size["longest_edge"];
    older versions read max_pixels — set both)."""
    ip = processor.image_processor
    ip.max_pixels = int(max_pixels)
    if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
        ip.size["longest_edge"] = int(max_pixels)


def legacy_count_prompt(question: str, num_frames: int) -> str:
    """legacy/v1/gnnformer/data.py:257 build_count_prompt, byte-identical (PROMPT-CRITICAL)."""
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). "
        "Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def legacy_p2_messages(frames: Sequence[Any], question: str) -> List[Dict[str, Any]]:
    """train_sft_fenced.py:78 build_fenced_messages: [frame_0, q, ..., frame_{N-1}, q, count_prompt],
    no system turn."""
    content: List[Dict[str, Any]] = []
    for im in frames:
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": question})
    content.append({"type": "text", "text": legacy_count_prompt(question, len(frames))})
    return [{"role": "user", "content": content}]


def legacy_p2_blocks(ids: List[int], tokenizer: Any, n_frames: int, vs_id: int
                     ) -> Optional[Tuple[List[Span], int]]:
    """train_sft_fenced.py:109 parse_layout: blocks [vs_i, vs_{i+1}), the last one closed at
    the unique "You will be shown" needle (the count prompt's opener)."""
    vstarts = [p for p, t in enumerate(ids) if t == vs_id]
    if len(vstarts) != n_frames:
        return None
    fin_start = None
    for pre in ("", " ", "\n"):
        needle = tokenizer(pre + "You will be shown", add_special_tokens=False).input_ids
        occ = find_subseq(ids, needle)
        if len(occ) == 1:
            fin_start = occ[0]
            break
    if fin_start is None or fin_start <= vstarts[-1] + 1:
        return None
    n = len(vstarts)
    blocks = [(vstarts[i], vstarts[i + 1] if i + 1 < n else fin_start) for i in range(n)]
    return blocks, fin_start


_LEGACY_VOCAB = list(ROOMS) + list(CHARS) + [NOBODY]     # eval_frozen.py ROOMS + CHARS (incl. Nobody)


def legacy_regex_ladder(text: str) -> Tuple[Optional[str], bool]:
    """eval_frozen.py:53 parse_answer -> (value or None, parse_ok). JSON field, then
    'Answer: X', then a bare closed-vocab word, then the first integer."""
    m = re.search(r'"answer"\s*:\s*"?([A-Za-z0-9]+)"?', text)
    if m:
        return m.group(1), True
    m = re.search(r"[Aa]nswer\s*[:=]\s*\"?([A-Za-z0-9]+)", text)
    if m:
        return m.group(1), True
    for w in _LEGACY_VOCAB:
        if re.search(rf"\b{w}\b", text):
            return w, False
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return m.group(1), False
    return None, False


def legacy_norm(v: Any) -> str:
    """eval_frozen.py:70 norm."""
    s = str(v).strip().strip('"').strip("'").lower()
    return str(int(s)) if s.isdigit() else s


def first_integer(text: str) -> Optional[str]:
    """redux/tasks.py parse_answer('count', ·): the first integer, '+' stripped."""
    m = INTEGER_RE.search(text)
    return m.group(0).lstrip("+") if m else None


@dataclass(frozen=True)
class PortCheck:
    name: str
    max_pixels: int
    system_prompt: bool         # arm-a keeps SYSTEM_PROMPT; p2 had no system turn
    question_prefix: str        # arm-a: "Question: "
    layout: str                 # "paper" | "legacy-replica"
    fence: bool
    decode: str                 # "generate" | "manual"
    max_new: int
    parser: str                 # "legacy-ladder" | "first-integer"
    anchor: str


PORT_CHECKS: Dict[str, PortCheck] = {
    "arm-a": PortCheck("arm-a", MAX_PIXELS_392, True, "Question: ", "paper", False, "generate",
                       LEGACY_DECODE_ARM_A, "legacy-ladder",
                       "grid_seq8_test 640/1200 = 0.533; 127776 final_app 26/34, steps 19/34, where_spend 12/34"),
    "p2": PortCheck("p2", MAX_PIXELS_392, False, "", "legacy-replica", True, "manual",
                    LEGACY_MAX_NEW_P2, "first-integer",
                    "p2_fenced_hf longn_eval.csv 45/50, 33/50, 15/50 @ seq 8/16/32 (steps_in_room)"),
}
