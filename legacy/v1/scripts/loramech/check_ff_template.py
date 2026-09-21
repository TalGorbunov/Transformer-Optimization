#!/usr/bin/env python3
"""LORAMECH parity gate (CPU, no model weights): the trainer's --frames-first template
must be TOKEN-IDENTICAL to gnnformer.data.build_prompt_inputs(processor, frames,
build_count_prompt(q, N)) — the frozen-baseline / ARMOR-A plain-arm prompt. Run before
any GPU submission. Checks N in {2, 8} at FRAME_RESIZE with synthetic frames.

Usage: python scripts/loramech/check_ff_template.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gnnformer.constants import FRAME_RESIZE  # noqa: E402
from gnnformer.data import build_count_prompt, build_prompt_inputs  # noqa: E402
from train_sft_baseline import build_messages_ff  # noqa: E402


def main() -> int:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    q0 = "How many steps does Alice take in the Kitchen?"
    for nf in (2, 8):
        frames = [Image.new("RGB", (FRAME_RESIZE, FRAME_RESIZE), (i * 20, 80, 120))
                  for i in range(nf)]
        ref = build_prompt_inputs(processor, frames, build_count_prompt(q0, nf))
        got = dict(processor.apply_chat_template(
            build_messages_ff(frames, q0), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"))
        a, b = ref["input_ids"][0].tolist(), got["input_ids"][0].tolist()
        if a != b:
            print(f"MISMATCH at N={nf}: ref {len(a)} toks vs ff {len(b)} toks")
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"  first diff at pos {i}: {x} vs {y}")
                    break
            return 1
        print(f"N={nf}: OK ({len(a)} tokens identical)")
    print("PARITY OK — --frames-first == build_prompt_inputs layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
