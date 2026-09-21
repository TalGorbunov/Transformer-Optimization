"""CPU tests for core.model (token-level only; loading the backbone needs a GPU) and the
real-tokenizer layout check that ties model, prompt and fence together.

Run: python tests/test_model.py   (needs the Qwen2.5-VL processor in the HF cache; skips otherwise)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.constants import MODEL_ID
from core.fence import frame_blocks, slot_positions
from core.model import image_token_groups, special_ids
from core.prompt import build_messages

NATIVE_TOKENS = 324   # 512 px render -> smart_resize 504 px -> (36/2)^2 merged patches


def _processor():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    except Exception as exc:
        print(f"  [skip] processor unavailable: {exc}")
        return None


def test_image_token_groups_synthetic():
    ids = torch.tensor([1, 9, 9, 9, 2, 9, 9, 3, 9])
    assert image_token_groups(ids, image_pad_id=9) == [[1, 2, 3], [5, 6], [8]]
    assert image_token_groups(torch.tensor([1, 2]), image_pad_id=9) == []


def test_special_ids():
    p = _processor()
    if p is None:
        return
    ids = special_ids(p)
    assert ids == {"vision_start": 151652, "vision_end": 151653, "image_pad": 151655}


def test_layout_with_real_tokenizer():
    """question-first layout, 3 native 512 px frames: 3 blocks of 324 image tokens, one slot
    per block at its last position, the question before every block."""
    p = _processor()
    if p is None:
        return
    from PIL import Image
    frames = [Image.new("RGB", (512, 512)) for _ in range(3)]
    msgs = build_messages(frames, "How many steps did Daniel spend in the Kitchen?", layout="question-first")
    enc = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    ids = enc["input_ids"][0]
    sid = special_ids(p)
    groups = image_token_groups(ids, sid["image_pad"])
    assert [len(g) for g in groups] == [NATIVE_TOKENS] * 3
    vs = (ids == sid["vision_start"]).nonzero().flatten().tolist()
    slots = slot_positions(ids, sid["vision_end"])
    blocks = frame_blocks(vs, slots)
    assert len(blocks) == 3 and all(b - a == NATIVE_TOKENS + 2 for a, b in blocks)
    assert all(s == b - 1 for s, (a, b) in zip(slots, blocks))
    q_ids = p.tokenizer("How many steps did Daniel spend in the Kitchen?", add_special_tokens=False).input_ids
    text = ids.tolist()
    first_q = next(i for i in range(len(text)) if text[i:i + len(q_ids)] == q_ids)
    assert first_q < blocks[0][0], "the question precedes every block (visible to all of them)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
