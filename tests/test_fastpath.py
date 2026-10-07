"""CPU parity test for the batched fenced path (core/fastpath.py) against the dense-mask path
(core/fence.py) on a TINY randomly initialised Qwen2.5-VL (3 layers, hidden 64, float32) with the
real processor: same slot states, same next-token logits, same greedy tokens — without a gate,
with a gate (some blocks hidden), under both layouts, and with chunked block encoding.

The two paths are the same computation arranged differently; random weights make the check
sensitive to any wrong position, wrong column or wrong block order. Needs the Qwen2.5-VL config
and processor in the HF cache (no weights); skips loudly otherwise.

Run: python tests/test_fastpath.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import get_backbone
from core.backbones import base as bb
from core.constants import MODEL_ID
from core.fastpath import fenced_forward, language_model, lm_head, merged_embeds
from core.fence import FENCED_SDPA, FenceHooks, greedy_decode
from core.model import get_layers
from core.prompt import SYSTEM_PROMPT, build_messages

Q = "How many steps did Daniel spend in the Kitchen?"
CAP = 1          # decoder module whose slot state is compared
_CACHE = {}


def _tiny():
    if "m" in _CACHE:
        return _CACHE["m"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        cfg = AutoConfig.from_pretrained(MODEL_ID)
        proc = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    except Exception as exc:
        print(f"  [skip] Qwen2.5-VL config / processor unavailable: {str(exc)[:120]}")
        _CACHE["m"] = None
        return None
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 3
    tc.num_attention_heads, tc.num_key_value_heads = 4, 2
    rs = dict(tc.rope_scaling or {})
    rs["mrope_section"] = [2, 3, 3]                       # sums to head_dim / 2 = 8
    tc.rope_scaling = rs
    vc.depth, vc.hidden_size, vc.num_heads, vc.intermediate_size, vc.out_hidden_size = 2, 32, 2, 64, 64
    vc.fullatt_block_indexes = [1]
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    _CACHE["m"] = (model, proc)
    return _CACHE["m"]


def _tiny_internvl():
    """A tiny randomly initialised InternVL3.5 (Qwen3 text model, 1-D RoPE, separator token after
    every frame) with the real processor: 448 px tile -> 256 image tokens per frame."""
    if "iv" in _CACHE:
        return _CACHE["iv"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    spec = get_backbone("internvl3.5-8b")
    try:
        from transformers import AutoConfig, AutoProcessor, InternVLForConditionalGeneration

        cfg = AutoConfig.from_pretrained(spec.model_id)
        proc = AutoProcessor.from_pretrained(spec.model_id)
    except Exception as exc:
        print(f"  [skip] InternVL3.5 config / processor unavailable: {str(exc)[:120]}")
        _CACHE["iv"] = None
        return None
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 3
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
    vc.hidden_size, vc.num_hidden_layers, vc.num_attention_heads, vc.intermediate_size = 32, 2, 2, 64
    torch.manual_seed(0)
    model = InternVLForConditionalGeneration(cfg).float().eval()
    _CACHE["iv"] = (model, proc)
    return _CACHE["iv"]


def _check_internvl(layout, keep, n=4, chunk_tokens=16000):
    t = _tiny_internvl()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("internvl3.5-8b")
    frames = _frames(n, size=(448, 448))
    d_logits, d_slots, d_text, seq, blocks = _dense(model, proc, spec, frames, layout, keep)
    r = fenced_forward(spec, model, model, proc, frames, Q, layout=layout, system_prompt=SYSTEM_PROMPT, keep=keep,
                       capture_layers=[CAP], max_new=4, chunk_tokens=chunk_tokens)
    ds = float((r.slot_states[CAP] - d_slots).abs().max())
    dl = float((r.first_logits - d_logits).abs().max())
    assert ds < 1e-4, f"InternVL slot states differ by {ds}"
    assert dl < 1e-4 * max(1.0, float(d_logits.abs().max())), f"InternVL first-token logits differ by {dl}"
    assert r.text == d_text, (r.text, d_text)
    k = n if keep is None else sum(keep)
    assert r.tokens_block == spec.tokens_per_frame_512 + 2 + spec.sep_tokens and r.tokens_read == seq - (n - k) * r.tokens_block
    print(f"  internvl {layout:14s} keep={'all' if keep is None else keep}: slots {ds:.1e} logits {dl:.1e} "
          f"read {r.tokens_read}/{seq} tokens")


def test_parity_internvl_question_first_gated():
    _check_internvl("question-first", [True, False, False, True])


def test_parity_internvl_no_gate_chunked():
    _check_internvl("question-first", None, chunk_tokens=600)


def test_parity_internvl_paper_last_frame_hidden():
    _check_internvl("paper", [False, True, True, False])


def _frames(n, size=(112, 112), seed=1):
    from PIL import Image

    g = torch.Generator().manual_seed(seed)
    return [Image.fromarray((torch.rand(size[1], size[0], 3, generator=g) * 255).to(torch.uint8).numpy()) for _ in range(n)]


def _dense(model, proc, spec, frames, layout, keep, max_new=4):
    enc = bb.encode(spec, proc, build_messages(frames, Q, layout=layout, system_prompt=SYSTEM_PROMPT))
    ids = enc["input_ids"]
    blocks, fin = bb.blocks(spec, ids[0], bb.special_ids(spec, proc), layout)
    assert len(blocks) == len(frames)
    hooks = FenceHooks(get_layers(model), capture_layers=[CAP]).install()
    try:
        mask, pos = bb.fenced_setup(spec, model, enc, blocks, fin, keep)
        cur = {k: v for k, v in enc.items() if k != "attention_mask"}
        hooks.set_mask(mask, "cpu")
        with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
            logits = model(**cur, position_ids=pos, use_cache=False).logits[0, -1]
        slots = hooks.hidden[CAP][0][bb.slots(blocks, spec)].clone()
        hooks.clear_mask()
        with torch.inference_mode():
            text = greedy_decode(model, hooks, enc, lambda c: bb.fenced_setup(spec, model, c, blocks, fin, keep),
                                 tokenizer=proc.tokenizer, max_new=max_new, eos_id=int(proc.tokenizer.eos_token_id),
                                 passthrough_keys=spec.passthrough_keys)
    finally:
        hooks.remove()
    return logits, slots, text, int(ids.shape[1]), blocks


def _check(layout, keep, n=5, chunk_tokens=16000):
    t = _tiny()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("qwen2.5-vl-7b")
    frames = _frames(n)
    d_logits, d_slots, d_text, seq, blocks = _dense(model, proc, spec, frames, layout, keep)
    r = fenced_forward(spec, model, model, proc, frames, Q, layout=layout, system_prompt=SYSTEM_PROMPT, keep=keep,
                       capture_layers=[CAP], max_new=4, chunk_tokens=chunk_tokens)
    assert r.slot_states[CAP].shape == d_slots.shape, (r.slot_states[CAP].shape, d_slots.shape)
    ds = float((r.slot_states[CAP] - d_slots).abs().max())
    dl = float((r.first_logits - d_logits).abs().max())
    scale = float(d_logits.abs().max())
    assert ds < 1e-4, f"slot states differ by {ds}"
    assert dl < 1e-4 * max(1.0, scale), f"first-token logits differ by {dl} (scale {scale})"
    assert r.text == d_text, (r.text, d_text)
    k = n if keep is None else sum(keep)
    T = blocks[0][1] - blocks[0][0]
    assert r.tokens_block == T and r.tokens_prefix == blocks[0][0] and r.tokens_read == seq - (n - k) * T
    assert r.keep == ([True] * n if keep is None else list(keep))
    print(f"  {layout:14s} keep={'all' if keep is None else keep}: slots {ds:.1e} logits {dl:.1e} text {r.text!r} "
          f"read {r.tokens_read}/{seq} tokens")


def test_accessors_and_merged_embeds():
    t = _tiny()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("qwen2.5-vl-7b")
    assert language_model(model) is model.model.language_model and lm_head(model) is model.lm_head
    frames = _frames(2)
    enc = bb.encode(spec, proc, build_messages(frames, Q, layout="question-first"))
    e = merged_embeds(model, enc)
    ids = enc["input_ids"][0]
    assert e.shape == (1, ids.shape[0], 64)
    text = model.get_input_embeddings()(ids)
    pad = ids == bb.special_ids(spec, proc).image_pad
    assert torch.equal(e[0][~pad], text[~pad]), "text positions are the token embeddings"
    assert not torch.allclose(e[0][pad], text[pad]), "image positions carry the vision features"
    assert not any(h for h in language_model(model)._forward_pre_hooks.values()), "the capture hook is removed"


def test_parity_question_first_no_gate():
    _check("question-first", None)


def test_parity_question_first_gated():
    _check("question-first", [True, False, True, False, False])


def test_parity_paper_layout_gated():
    _check("paper", [False, True, True, False, True])


def test_known_gate_encodes_only_kept_frames():
    """Same answer whether the hidden blocks are encoded (capturing) or skipped (gate known)."""
    t = _tiny()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("qwen2.5-vl-7b")
    frames = _frames(5)
    keep = [False, True, False, True, False]
    import core.fastpath as fp
    calls = []
    orig = fp.block_embeds
    fp.block_embeds = lambda *a, **k: (calls.append(len(a[3])), orig(*a, **k))[1]
    try:
        skip = fenced_forward(spec, model, model, proc, frames, Q, keep=keep, max_new=3, system_prompt=SYSTEM_PROMPT)
        n_skip = sum(calls); calls.clear()
        full = fenced_forward(spec, model, model, proc, frames, Q, keep=keep, capture_layers=[CAP], max_new=3,
                              system_prompt=SYSTEM_PROMPT)
        n_full = sum(calls)
    finally:
        fp.block_embeds = orig
    assert (n_skip, n_full) == (2, 5), (n_skip, n_full)
    assert skip.text == full.text and torch.allclose(skip.first_logits, full.first_logits, atol=1e-5)
    assert skip.keep == full.keep == keep and skip.tokens_read == full.tokens_read
    try:
        fenced_forward(spec, model, model, proc, frames, Q, keep=[True] * 5, max_read_tokens=50, system_prompt=SYSTEM_PROMPT)
        assert False, "the read-length guard must refuse"
    except ValueError as exc:
        assert "read too long" in str(exc)


def test_parity_chunked_blocks():
    """Blocks encoded two at a time give the same result as all at once."""
    _check("question-first", [True, True, False, True, False], chunk_tokens=40)


def test_gate_callable_and_capture_only():
    t = _tiny()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("qwen2.5-vl-7b")
    frames = _frames(5)
    seen = []

    def gate(states, first):
        seen.append((first, states[CAP].shape[0]))
        return [(first + i) % 2 == 0 for i in range(states[CAP].shape[0])]

    r = fenced_forward(spec, model, model, proc, frames, Q, keep=gate, capture_layers=[CAP], max_new=2, chunk_tokens=40,
                       system_prompt=SYSTEM_PROMPT)
    assert r.keep == [True, False, True, False, True] and sum(n for _, n in seen) == 5 and len(seen) > 1
    ref = fenced_forward(spec, model, model, proc, frames, Q, keep=r.keep, max_new=2, system_prompt=SYSTEM_PROMPT)
    assert ref.text == r.text and torch.allclose(ref.first_logits, r.first_logits, atol=1e-5)
    cap = fenced_forward(spec, model, model, proc, frames, Q, capture_layers=[CAP], decode=False, system_prompt=SYSTEM_PROMPT)
    assert cap.text == "" and cap.first_logits is None and torch.allclose(cap.slot_states[CAP], r.slot_states[CAP], atol=1e-5)
    try:
        fenced_forward(get_backbone("gemma-3-12b"), model, model, proc, frames, Q)
        assert False, "Gemma needs the masked path"
    except NotImplementedError:
        pass
    from PIL import Image
    try:
        fenced_forward(spec, model, model, proc, [frames[0], Image.new("RGB", (168, 112))], Q)
        assert False, "frames of different sizes must be refused"
    except ValueError:
        pass


JUDGE = "Does this frame show what the question asks about? Answer yes or no."


def _dense_judge(model, proc, spec, frame, judge_text):
    """Dense reference for the per-block judge: a ONE-frame question-first prompt whose user turn
    is question, frame, instruction; the fence on one block is plain causal attention, so this is
    exactly what block t sees in the batched judge (prefix + itself + instruction + opener)."""
    msgs = build_messages([frame], Q, layout="question-first", system_prompt=SYSTEM_PROMPT)
    msgs[-1]["content"].append({"type": "text", "text": judge_text})
    enc = bb.encode(spec, proc, msgs)
    ids = enc["input_ids"]
    blocks, fin = bb.blocks(spec, ids[0], bb.special_ids(spec, proc), "question-first")
    assert len(blocks) == 1
    hooks = FenceHooks(get_layers(model), capture_layers=[CAP]).install()
    try:
        mask, pos = bb.fenced_setup(spec, model, enc, blocks, fin, None)
        cur = {k: v for k, v in enc.items() if k != "attention_mask"}
        hooks.set_mask(mask, "cpu")
        with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
            logits = model(**cur, position_ids=pos, use_cache=False).logits[0, -1]
        a, b = blocks[0]
        img = ids[0, a:b] == bb.special_ids(spec, proc).image_pad
        pooled = hooks.hidden[CAP][0][a:b][img].mean(dim=0).clone()
    finally:
        hooks.clear_mask(); hooks.remove()
    return logits, pooled


def test_judge_and_pooled_states_parity():
    """Per-block judge logits == the dense one-frame prompt (question, frame, instruction) for
    every block; pooled image-token states == the dense masked path's; batching (chunk of 2)
    changes nothing; the kept blocks' caches are unaffected by the judge tokens appended after them."""
    t = _tiny()
    if t is None:
        return
    model, proc = t
    spec = get_backbone("qwen2.5-vl-7b")
    frames = _frames(5)
    tok = proc.tokenizer
    jid = tok.encode(JUDGE, add_special_tokens=False)
    vocab = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No", "yes", "no")]
    keep = [True, False, True, False, False]
    r = fenced_forward(spec, model, model, proc, frames, Q, keep=keep, capture_layers=[CAP], pool_layers=[CAP],
                       judge_ids=jid, judge_vocab=vocab, max_new=3, chunk_tokens=40, system_prompt=SYSTEM_PROMPT)
    assert r.judge_logits.shape == (5, 4) and r.pool_states[CAP].shape == r.slot_states[CAP].shape
    for i, f in enumerate(frames):
        d_logits, d_pool = _dense_judge(model, proc, spec, f, JUDGE)
        dj = float((r.judge_logits[i] - d_logits[vocab]).abs().max())
        dp = float((r.pool_states[CAP][i] - d_pool).abs().max())
        assert dj < 1e-4 * max(1.0, float(d_logits.abs().max())), f"judge logits of block {i} differ by {dj}"
        assert dp < 1e-4, f"pooled state of block {i} differs by {dp}"
    ref = fenced_forward(spec, model, model, proc, frames, Q, keep=keep, max_new=3, system_prompt=SYSTEM_PROMPT)
    assert ref.text == r.text and torch.allclose(ref.first_logits, r.first_logits, atol=1e-5), "the judge must not touch the read"
    one = fenced_forward(spec, model, model, proc, frames, Q, keep=keep, judge_ids=jid, judge_vocab=vocab, decode=False,
                         chunk_tokens=16000, system_prompt=SYSTEM_PROMPT)
    assert torch.allclose(one.judge_logits, r.judge_logits, atol=1e-5), "chunking changes the judge"
    assert one.keep == keep and one.first_logits is None
    # an empty instruction is the read over prefix + that block alone
    e = fenced_forward(spec, model, model, proc, frames, Q, judge_ids=[], judge_vocab=vocab, decode=False, system_prompt=SYSTEM_PROMPT)
    single = fenced_forward(spec, model, model, proc, frames, Q, keep=[i == 3 for i in range(5)], max_new=1, system_prompt=SYSTEM_PROMPT)
    assert torch.allclose(e.judge_logits[3], single.first_logits[vocab], atol=1e-5)
    try:
        fenced_forward(spec, model, model, proc, frames, Q, judge_ids=jid, decode=False, system_prompt=SYSTEM_PROMPT)
        assert False, "judge_vocab is required"
    except ValueError:
        pass
    print(f"  judge: 5 blocks match the dense one-frame prompt; pooled states match; chunked == unchunked")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
