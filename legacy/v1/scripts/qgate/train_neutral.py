#!/usr/bin/env python3
"""QGATE Option 1 — train the gated fenced-SFT on the REPLICA-NEUTRAL layout.

Thin launcher: imports scripts/sparse/train_sft_gated.py (untouched anchor) and
rebinds its message builders to qgate_common.build_task_messages_layout with
layout='replica-neutral' (generic filler per block; the cacheable-summary-slot
design). Everything else — oracle gate, masks, posreset, decode, evals — is the
sparse trainer verbatim. CLI = the sparse trainer's CLI (do NOT pass --layout;
it is forced here).
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "scripts" / "sparse"), str(_REPO / "scripts" / "qgate")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_sft_gated as T  # noqa: E402
from qgate_common import build_task_messages_layout  # noqa: E402

if "--layout" in sys.argv:
    raise SystemExit("--layout is forced to replica-neutral by this launcher")

T.build_task_messages = partial(build_task_messages_layout, layout="replica-neutral")
T.build_fenced_messages = lambda frames, question, answer=None, declare_n=None, \
    nfree=False, layout=None: build_task_messages_layout(
        frames, "count", "", "", question, answer=answer, nfree=nfree,
        layout="replica-neutral")

if __name__ == "__main__":
    raise SystemExit(T.main())
