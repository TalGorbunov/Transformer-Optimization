"""QGATE shared: task x layout message builder (sparse originals stay untouched).

build_task_messages_layout == the sparse builder for layout="replica" (byte-identical,
delegated) and extends {qfirst-once, qlast-once} to exists/majority: the non-count
final prompt is the redux task template (same "You will be shown" opener line, so
parse_layout's needle works for every combination).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "scripts" / "sparse"), str(_REPO / "scripts" / "redux")):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_sft_gated import build_fenced_messages, build_task_messages  # noqa: E402
from tasks import build_prompt as task_prompt  # noqa: E402
from tasks import replica_text  # noqa: E402


NEUTRAL_REPLICA = "What is shown in this frame?"


def build_task_messages_layout(frames, task, char, room, q0, answer=None,
                               nfree=False, layout="replica"):
    if layout == "replica-neutral":
        # D1: replica POSITIONS kept, question content removed (verdict-reader test)
        content = []
        for im in frames:
            content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": NEUTRAL_REPLICA})
        from tasks import build_prompt as _tp
        from train_sft_gated import build_count_prompt, build_nfree_prompt
        if task == "count":
            final = (build_nfree_prompt(q0) if nfree
                     else build_count_prompt(q0, len(frames)))
        else:
            final = _tp(task, char, room, len(frames))
        content.append({"type": "text", "text": final})
        msgs = [{"role": "user", "content": content}]
        if answer is not None:
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": str(answer)}]})
        return msgs
    if task == "count":
        return build_fenced_messages(frames, q0, answer=answer, nfree=nfree,
                                     layout=layout)
    if layout == "replica":
        return build_task_messages(frames, task, char, room, q0, answer=answer,
                                   nfree=nfree)
    nf = len(frames)
    content = []
    if layout == "qfirst-once":
        content.append({"type": "text", "text": replica_text(task, char, room, nf)})
    for im in frames:
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": task_prompt(task, char, room, nf)})
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": str(answer)}]})
    return msgs
