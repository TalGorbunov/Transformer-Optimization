"""CPU tests for the backbone seam (core/backbones): the Qwen spec reproduces the pre-seam
code (special ids, blocks == core.fence.layout_blocks, slot = last token of its block, 324
tokens per native frame on the real tokenizer — skips loudly without the cached processor),
the pad_run block rule on a synthetic stream, 1-D base positions, and the position reset on
1-D shapes.

Run: python tests/test_backbones.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, DEFAULT_BACKBONE, get_backbone
from core.backbones import base as bb
from core.backbones.base import BackboneSpec, Delimiters, SpecialIds
from core.constants import MODEL_ID
from core.fence import frame_blocks, layout_blocks, reset_positions, slot_positions
from core.model import special_ids as legacy_special_ids
from core.prompt import build_messages

VS_ID, VE_ID, PAD_ID, IM_END = 151652, 151653, 151655, 151645
QWEN_SID = SpecialIds(vision_start=VS_ID, vision_end=VE_ID, image_pad=PAD_ID, turn_end=IM_END)


def _processor(model_id=MODEL_ID, use_fast=False):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_id, use_fast=use_fast)
    except Exception as exc:
        print(f"  [skip] processor unavailable for {model_id}: {str(exc)[:120]}")
        return None


def _ids(prefix=4, nf=3, img=4, tail=3):
    ids = [1] * prefix
    for _ in range(nf):
        ids += [VS_ID] + [PAD_ID] * img + [VE_ID]
    return torch.tensor(ids + [2] * tail)


def test_registry_qwen_spec():
    spec = get_backbone(DEFAULT_BACKBONE)
    assert spec.name == "qwen2.5-vl-7b" and spec.model_id == MODEL_ID and spec in BACKBONES.values()
    assert spec.block_rule == "pair" and spec.rope == "mrope" and spec.image_attention == "causal"
    assert spec.delimiters == Delimiters("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|im_end|>")
    assert spec.tokens_per_frame_512 == 324 and spec.passthrough_keys == ("pixel_values", "image_grid_thw")
    assert spec.probe_layers == (12, 20, 24) and "q_proj" in spec.lora_targets and "down_proj" in spec.lora_targets
    try:
        get_backbone("nope")
        assert False
    except KeyError:
        pass


def test_resolution_modes():
    """Every spec names a lo and a hi mode; set_resolution applies them through the processor's own
    attributes (checked on the cached processors; skips without them) and refuses unknown modes."""
    for spec in BACKBONES.values():
        assert set(spec.res_modes) == {"lo", "hi"}, spec.name
    qwen = get_backbone(DEFAULT_BACKBONE)
    assert qwen.res_modes["lo"] == {"max_pixels": 151_200} and qwen.res_modes["hi"] == {"max_pixels": 1_003_520}
    assert bb.set_resolution(qwen, None, None) == {} and bb.set_resolution(qwen, None, "") == {}
    for spec in BACKBONES.values():
        proc = _processor(spec.model_id, use_fast=spec.use_fast_processor) if spec.use_fast_processor is not None else _processor(spec.model_id, use_fast=True)
        if proc is None:
            continue
        for mode in ("lo", "hi"):
            kw = bb.set_resolution(spec, proc, mode)
            assert kw == dict(spec.res_modes[mode])
            ip = proc.image_processor
            for k, v in kw.items():
                if k == "max_pixels":
                    assert ip.max_pixels == v and (not isinstance(getattr(ip, "size", None), dict) or "longest_edge" not in ip.size or ip.size["longest_edge"] == v)
                else:
                    assert getattr(ip, k) == v, (spec.name, k)
        try:
            bb.set_resolution(spec, proc, "ultra")
            assert False
        except ValueError:
            pass
        # the mode must reach the image processor THROUGH encode (the chat-template path): a 16:9 frame gets
        # more image tokens under hi than under lo on the tile / crop backbones, the same count on Qwen
        # (its max_pixels caps pixels, not tiles: 1920x1080 -> 180 tokens at lo, ~1,200 at hi)
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((1080, 1920, 3), np.uint8))
        sid = bb.special_ids(spec, proc.tokenizer)
        counts = {}
        for mode in ("lo", "hi"):
            bb.set_resolution(spec, proc, mode)
            ids = bb.encode(spec, proc, build_messages([img], "q", layout="paper", system_prompt=None))["input_ids"][0].tolist()
            counts[mode] = ids.count(sid.image_pad)
        assert counts["hi"] > counts["lo"] >= 1, (spec.name, counts)
        if spec.name == "gemma-3-12b":
            assert counts == {"lo": 256, "hi": 768}, counts            # original + 2 pan-and-scan crops on 16:9
        if spec.name == "internvl3.5-8b":
            assert counts["lo"] == 256 and counts["hi"] >= 2304, counts  # one tile vs dynamic tiling (9 tiles on 16:9)
        if spec.name == "qwen2.5-vl-7b":
            assert counts["lo"] <= 200 and counts["hi"] >= 1100, counts
        print(f"  {spec.name}: lo {spec.res_modes['lo']} hi {spec.res_modes['hi']} -> image tokens on 1920x1080: {counts}")


def test_blocks_pair_equals_layout_blocks():
    spec = get_backbone(DEFAULT_BACKBONE)
    ids = _ids()
    for layout in ("paper", "question-first"):
        assert bb.blocks(spec, ids, QWEN_SID, layout) == layout_blocks(ids, layout, QWEN_SID.legacy(), im_end_id=IM_END)
    blocks, fin = bb.blocks(spec, ids, QWEN_SID, "question-first")
    assert blocks == [(4, 10), (10, 16), (16, 22)] and fin == 22
    assert bb.slots(blocks) == slot_positions(ids, VE_ID) == [9, 15, 21]
    rep = [1] * 4
    for _ in range(3):
        rep += [VS_ID] + [PAD_ID] * 4 + [VE_ID] + [7, 7]
    rep = torch.tensor(rep + [IM_END] + [2] * 3)
    assert bb.blocks(spec, rep, QWEN_SID, "replica") == layout_blocks(rep, "replica", QWEN_SID.legacy(), im_end_id=IM_END)


def test_blocks_pad_run():
    """A model with no start token: block = placeholder run (+ inner tokens) + the end token."""
    spec = BackboneSpec(name="t", model_id="t", family="t", block_rule="pad_run", rope="1d",
                        delimiters=Delimiters(start=None, end="<E>", pad="<P>", turn_end="<T>", inner=("<B>",)))
    sid = SpecialIds(vision_start=None, vision_end=90, image_pad=91, turn_end=92, inner=(93,))
    ids = torch.tensor([1, 1, 91, 91, 93, 91, 91, 90, 3, 91, 91, 93, 91, 91, 90, 2, 2])
    blocks, fin = bb.blocks(spec, ids, sid, "question-first")
    assert blocks == [(2, 8), (9, 15)] and fin == 15
    assert bb.slots(blocks) == [7, 14] and all(int(ids[s]) == 90 for s in bb.slots(blocks))
    no_end = replace(spec, delimiters=replace(spec.delimiters, end=None))
    sid_no_end = SpecialIds(vision_start=None, vision_end=None, image_pad=91, turn_end=92, inner=(93,))
    assert bb.blocks(no_end, ids, sid_no_end, "paper")[0] == [(2, 7), (9, 14)]
    for bad in (torch.tensor([1, 91, 91, 3]), torch.tensor([1, 2, 3])):   # run without end / no run
        try:
            bb.blocks(spec, bad, sid, "paper")
            assert False, bad
        except ValueError:
            pass
    try:
        bb.blocks(spec, ids, sid, "replica")
        assert False, "replica needs paired delimiters"
    except ValueError:
        pass


def test_base_positions_1d_and_reset():
    spec = BackboneSpec(name="t", model_id="t", family="t", rope="1d",
                        delimiters=Delimiters(start="<S>", end="<E>", pad="<P>", turn_end="<T>"))
    ids = _ids()
    pos = bb.base_positions(spec, None, {"input_ids": ids.view(1, -1)})
    assert pos.shape == (1, len(ids)) and torch.equal(pos[0], torch.arange(len(ids)))
    blocks = frame_blocks([4, 10, 16], [9, 15, 21])
    r1 = reset_positions(pos, blocks, 22)
    r3 = reset_positions(pos.view(1, 1, -1).repeat(3, 1, 1), blocks, 22)
    assert torch.equal(r1[0], r3[0, 0]) and torch.equal(r3[1, 0], r3[0, 0])
    assert torch.equal(r1[0, 10:16], r1[0, 4:10]) and int(r1[0, 22]) == int(r1[0, 4:10].max()) + 1
    assert torch.equal(pos[0], torch.arange(len(ids))), "input not mutated"
    try:
        bb.base_positions(replace(spec, rope="weird"), None, {"input_ids": ids.view(1, -1)})
        assert False
    except ValueError:
        pass


def test_special_ids_real_tokenizer():
    p = _processor()
    if p is None:
        return
    spec = get_backbone(DEFAULT_BACKBONE)
    sid = bb.special_ids(spec, p.tokenizer)
    assert sid == QWEN_SID, sid
    assert sid.legacy() == legacy_special_ids(p) == {"vision_start": VS_ID, "vision_end": VE_ID, "image_pad": PAD_ID}
    try:
        bb.special_ids(replace(spec, delimiters=replace(spec.delimiters, end="<|no_such_token|>")), p.tokenizer)
        assert False, "missing token must raise"
    except ValueError:
        pass


def test_layout_with_real_tokenizer():
    """3 native 512 px frames under question-first: 3 blocks of tokens_per_frame_512 + 2 markers,
    slot = last token of each block, question before block 0 — through the spec."""
    p = _processor()
    if p is None:
        return
    from PIL import Image
    spec = get_backbone(DEFAULT_BACKBONE)
    sid = bb.special_ids(spec, p.tokenizer)
    frames = [Image.new("RGB", (512, 512)) for _ in range(3)]
    q = "How many steps did Daniel spend in the Kitchen?"
    enc = p.apply_chat_template(build_messages(frames, q, layout="question-first"), add_generation_prompt=True,
                                tokenize=True, return_dict=True, return_tensors="pt")
    ids = enc["input_ids"][0]
    blocks, fin = bb.blocks(spec, ids, sid, "question-first")
    assert len(blocks) == 3 and all(b - a == spec.tokens_per_frame_512 + 2 for a, b in blocks)
    assert bb.slots(blocks) == slot_positions(ids, sid.vision_end) and fin == blocks[-1][1]
    assert all(int(ids[s]) == sid.vision_end for s in bb.slots(blocks))
    q_ids = p.tokenizer(q, add_special_tokens=False).input_ids
    text = ids.tolist()
    first_q = next(i for i in range(len(text)) if text[i:i + len(q_ids)] == q_ids)
    assert first_q < blocks[0][0]
    for k in spec.passthrough_keys:
        assert k in enc, k


def test_rope_index_lookup_through_wrappers():
    """get_rope_index lives on the inner text-vision model; the lookup must reach it through
    ForConditionalGeneration and a PEFT-style wrapper whose `.model` resolves to the OUTER model
    (the 2026-09-23 C2 failure)."""
    from core.model import get_rope_index_fn

    class Inner:
        def get_rope_index(self, *a, **k):
            return "inner"

    class Outer:                       # ForConditionalGeneration: .model = Inner
        def __init__(self):
            self.model = Inner()

    class Lora:                        # LoraModel: .model = Outer
        def __init__(self, outer):
            self.model = outer

    class Peft:                        # PeftModel: base_model = Lora, attribute forwarding
        def __init__(self, outer):
            self.base_model = Lora(outer)

        def get_base_model(self):
            return self.base_model.model

        def __getattr__(self, name):
            return getattr(self.base_model.model, name)

    outer = Outer()
    assert get_rope_index_fn(outer)() == "inner"
    assert get_rope_index_fn(Peft(outer))() == "inner"
    assert get_rope_index_fn(Inner())() == "inner"
    try:
        get_rope_index_fn(object())
        assert False
    except RuntimeError:
        pass


def test_registry_new_backbones():
    iv, gm = get_backbone("internvl3.5-8b"), get_backbone("gemma-3-12b")
    assert set(BACKBONES) == {"qwen2.5-vl-7b", "internvl3.5-8b", "gemma-3-12b"}
    assert iv.rope == "1d" and iv.image_attention == "causal" and iv.layer_types is None and iv.sep_tokens == 1
    assert "Qwen3" in iv.family, "the LM lineage is stated in the family string"
    assert gm.rope == "1d" and gm.image_attention == "bidirectional" and gm.layer_types == "config" and gm.sep_tokens == 1
    assert "token_type_ids" in gm.passthrough_keys and iv.passthrough_keys == ("pixel_values",)
    assert iv.tokens_per_frame_512 == gm.tokens_per_frame_512 == 256
    assert get_backbone(DEFAULT_BACKBONE).sep_tokens == 0
    assert gm.torch_dtype == "bfloat16" and iv.torch_dtype is None and get_backbone(DEFAULT_BACKBONE).torch_dtype is None
    assert iv.assistant_prefix.startswith("<think>") and gm.assistant_prefix == "" and get_backbone(DEFAULT_BACKBONE).assistant_prefix == ""


def test_blocks_with_separator_tokens():
    """A separator token after every end token belongs to its block (or it would read every earlier
    frame and be read by every later one); the slot stays the end token; a stray token between two
    frames is refused."""
    spec = BackboneSpec(name="t", model_id="t", family="t", rope="1d", sep_tokens=1,
                        delimiters=Delimiters(start="<S>", end="<E>", pad="<P>", turn_end="<T>"))
    sid = SpecialIds(vision_start=80, vision_end=81, image_pad=82, turn_end=83)
    ids = torch.tensor([1, 1, 80, 82, 82, 82, 81, 7, 80, 82, 82, 82, 81, 7, 80, 82, 82, 82, 81, 9, 83, 2])
    for layout in ("question-first", "paper"):
        blocks, fin = bb.blocks(spec, ids, sid, layout)
        assert blocks == [(2, 8), (8, 14), (14, 20)] and fin == 20
    assert bb.slots(blocks, spec) == [6, 12, 18] and all(int(ids[s]) == 81 for s in bb.slots(blocks, spec))
    assert bb.image_spans(blocks, spec) == [(3, 6), (9, 12), (15, 18)]
    covered = {c for a, b in blocks for c in range(a, b)}
    assert all(i in covered for i in range(2, 20)), "no token between the first start and the tail is outside a block"
    stray = torch.tensor([1, 80, 82, 81, 7, 5, 80, 82, 81, 7, 83])
    for bad, lay in ((stray, "paper"), (ids, "replica"), (torch.tensor([1, 80, 82, 81]), "paper")):
        try:
            bb.blocks(spec, bad, sid, lay)
            assert False, (bad, lay)
        except ValueError:
            pass


def test_fenced_setup_bidirectional_and_layer_types():
    class TC:
        sliding_window = 1024
        num_attention_heads, num_key_value_heads, hidden_size, head_dim = 16, 8, 3840, 256

    class M:
        class config:
            text_config = TC

    spec = get_backbone("gemma-3-12b")
    sid = SpecialIds(vision_start=80, vision_end=81, image_pad=82, turn_end=83)
    ids = torch.tensor([1, 1, 80, 82, 82, 82, 81, 7, 80, 82, 82, 82, 81, 9, 83, 2])
    blocks, fin = bb.blocks(spec, ids, sid, "question-first")
    mask, pos = bb.fenced_setup(spec, M, {"input_ids": ids.view(1, -1)}, blocks, fin, None)
    assert set(mask) == {"full_attention", "sliding_attention"} and mask["full_attention"] is mask["sliding_attention"]
    m = mask["full_attention"]
    for a, b in bb.image_spans(blocks, spec):
        assert torch.all(m[a:b, a:b] == 0)
    assert m[3, 9] != 0 and m[9, 3] != 0, "frames stay fenced"
    assert pos.shape == (1, len(ids)) and torch.equal(pos[0, 8:14], pos[0, 2:8])
    assert bb.attention_dims(spec, M) == {"n_heads": 16, "n_kv": 8, "head_dim": 256, "mrope_section": None}
    iv = get_backbone("internvl3.5-8b")
    mask_iv, _ = bb.fenced_setup(iv, M, {"input_ids": ids.view(1, -1)}, blocks, fin, [True, False])
    assert torch.is_tensor(mask_iv) and mask_iv[4, 5] != 0, "causal backbone: one mask, no bidirectional square"
    assert all(mask_iv[15, c] != 0 for c in range(8, 14)), "the gate hides the dropped block from the tail"


def _real_layout(name):
    spec = get_backbone(name)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoProcessor
        p = AutoProcessor.from_pretrained(spec.model_id, **({} if spec.use_fast_processor is None else {"use_fast": spec.use_fast_processor}))
    except Exception as exc:
        print(f"  [skip] processor unavailable for {spec.model_id}: {str(exc)[:160]}")
        return
    from PIL import Image
    sid = bb.special_ids(spec, p.tokenizer)
    frames = [Image.new("RGB", (512, 512)) for _ in range(3)]
    q = "How many steps did Daniel spend in the Kitchen?"
    for layout in ("question-first", "paper"):
        enc = bb.encode(spec, p, build_messages(frames, q, layout=layout))
        ids = enc["input_ids"][0]
        assert all(v.shape[1] == ids.shape[0] for k, v in enc.items() if k in ("attention_mask", "token_type_ids"))
        if spec.assistant_prefix:
            pre = p.tokenizer(spec.assistant_prefix, add_special_tokens=False).input_ids
            assert ids.tolist()[-len(pre):] == pre, "the assistant prefix closes the prompt"
        blocks, fin = bb.blocks(spec, ids, sid, layout)
        assert len(blocks) == 3 and all(b - a == spec.tokens_per_frame_512 + 2 + spec.sep_tokens for a, b in blocks), blocks
        assert all(b == a2 for (_, b), (a2, _) in zip(blocks, blocks[1:])), "contiguous: nothing between frames"
        assert all(int(ids[s]) == sid.vision_end for s in bb.slots(blocks, spec))
        assert all(bool((ids[a:b] == sid.image_pad).all()) for a, b in bb.image_spans(blocks, spec))
        q_ids = p.tokenizer(q, add_special_tokens=False).input_ids
        text = ids.tolist()
        first_q = next((i for i in range(len(text)) if text[i:i + len(q_ids)] == q_ids), None)
        if first_q is not None:
            assert (first_q < blocks[0][0]) == (layout == "question-first"), (layout, first_q, blocks[0])
        for k in spec.passthrough_keys:
            assert k in enc, k
        print(f"  {name} {layout}: seq {len(text)} blocks {blocks[0]}..{blocks[-1]} fin {fin}")


def test_layout_internvl_real_tokenizer():
    _real_layout("internvl3.5-8b")


def test_layout_gemma_real_tokenizer():
    _real_layout("gemma-3-12b")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
