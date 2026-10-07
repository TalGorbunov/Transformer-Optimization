"""The backbone seam: a BackboneSpec (data: which tokens delimit an image, how blocks are
found, which positional scheme the LM uses, what the fenced forward must re-pass) plus the
generic functions every experiment calls through it. Layering: core.fence (pure mask /
position arithmetic, hooks, decode) < core.backbones (spec-aware token and position logic)
< experiments. docs/SCALEUP_2026-09-23.md §1.2.

Registered specs: core/backbones/qwen2_5_vl.py (the anchor); InternVL3.5 and Gemma 3 follow
in their own steps. The Qwen path is byte-identical to the pre-seam code: tests/test_backbones.py
pins blocks() against core.fence.layout_blocks and special_ids() against the legacy ids, and
the GPU refactor proof (C1 faithful N=8, D4 fenced_qfirst N=16) must reproduce its numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..fence import Span, build_block_mask, hide_cols_for, layout_blocks, reset_positions, sliding_window_mask
from ..model import ModelRuntime, get_rope_index_fn, text_config


@dataclass(frozen=True)
class Delimiters:
    """Token strings around one image in the LM sequence. `start`/`end` may be absent
    (models with only a placeholder run); `inner` are tokens that may occur INSIDE an image run
    (e.g. Mistral's [IMG_BREAK]); `turn_end` closes the user turn (the replica layout's last block)."""
    start: Optional[str]
    end: Optional[str]
    pad: str
    turn_end: str
    inner: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SpecialIds:
    vision_start: Optional[int]
    vision_end: Optional[int]
    image_pad: int
    turn_end: int
    inner: Tuple[int, ...] = ()

    def legacy(self) -> Dict[str, int]:
        """The three-key dict core.model.special_ids has always returned (tests/test_model.py)."""
        return {"vision_start": self.vision_start, "vision_end": self.vision_end, "image_pad": self.image_pad}


@dataclass(frozen=True)
class BackboneSpec:
    name: str                                   # registry key, e.g. "qwen2.5-vl-7b"
    model_id: str                               # HF id (or a local path via --model)
    family: str                                 # LM lineage as it must appear in tables
    delimiters: Delimiters
    block_rule: str = "pair"                    # "pair": [start_i, end_i + 1 + sep_tokens); "pad_run": placeholder run (+ end)
    sep_tokens: int = 0                         # separator tokens the template puts after each end token (InternVL "\n",
                                                # Gemma "\n\n\n\n"): they belong to the block, or they leak across the fence
    rope: str = "1d"                            # "mrope" (Qwen-style 3-D, get_rope_index) | "1d" (arange)
    image_attention: str = "causal"             # "causal" | "bidirectional" (inside each image)
    layer_types: Optional[str] = None           # None | "config" (per-layer full/sliding from the text config)
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    probe_layers: Tuple[int, ...] = ()          # gate_capture defaults, legacy hidden-state numbering
    tokens_per_frame_512: int = 0               # LM tokens one native 512 px frame becomes (pinned by a tokenizer test)
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)   # e.g. tiling OFF for tile encoders
    use_fast_processor: Optional[bool] = False  # None = the processor's own default (what backbone_probe saw)
    torch_dtype: Optional[str] = None           # dtype of the NON-quantized modules; None = the checkpoint's own. Gemma 3 needs
                                                # "bfloat16": its float16 default gives NaN logits (smoke 164458, 2026-09-28)
    pixel_knob: str = ""                        # "max_pixels": the processor bounds the pixels it shows the model (Qwen);
                                                # "" = the processor has one fixed input size (InternVL 448, Gemma 896)
    assistant_prefix: str = ""                  # text pre-filled after the generation prompt (e.g. an empty think block
                                                # for a thinking LM: the short-answer protocol leaves no room to think)
    passthrough_keys: Tuple[str, ...] = ("pixel_values",)   # non-text inputs a fenced re-forward must carry
    res_modes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
                                                # named resolution modes through the processor's OWN attributes (UNIT baseline):
                                                # "lo" = the single-tile budget a 512 px frame gets; "hi" = the model's own
                                                # high-resolution mechanism (Qwen max_pixels cap, InternVL dynamic tiling,
                                                # Gemma pan-and-scan). set_resolution() applies one; none = the defaults.


# ------------------------------------------------------------------------------ loading

def load_runtime(spec: BackboneSpec, *, attn_implementation: str = "sdpa", use_4bit: bool = True,
                 device_map: Any = "cuda") -> ModelRuntime:
    """Frozen backbone from its spec: 4-bit nf4 + double quant, bf16 compute, sdpa (4-D masks);
    "eager" is refused. The model is put in eval(); LoRA adapters are attached by the experiments."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if attn_implementation == "eager":
        raise ValueError("eager attention is forbidden in this codebase (masks + speed).")
    pk = dict(spec.processor_kwargs)
    if spec.use_fast_processor is not None:
        pk["use_fast"] = spec.use_fast_processor
    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True, **pk)
    kwargs: Dict[str, Any] = {"attn_implementation": attn_implementation, "trust_remote_code": True,
                              "device_map": device_map}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    if spec.torch_dtype is not None:
        kwargs["torch_dtype"] = getattr(torch, spec.torch_dtype)
    model = AutoModelForImageTextToText.from_pretrained(spec.model_id, **kwargs)
    model.eval()
    return ModelRuntime(model_name=str(spec.model_id), processor=processor, model=model, spec=spec)


def set_max_pixels(spec: BackboneSpec, processor: Any, max_pixels: int) -> None:
    """Bound the pixels per frame the MODEL sees, through the processor's own knob (0 = leave the
    processor's default). This is the only resolution control on the model path: frames are stored
    at the video's own resolution and never resized by our code. transformers 4.57 reads
    size["longest_edge"], older versions max_pixels — both are set."""
    if not max_pixels:
        return
    if spec.pixel_knob != "max_pixels":
        raise ValueError(f"{spec.name}: the processor has a fixed input size; --max-pixels does not apply")
    ip = processor.image_processor
    ip.max_pixels = int(max_pixels)
    if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
        ip.size["longest_edge"] = int(max_pixels)


def set_resolution(spec: BackboneSpec, processor: Any, mode: Optional[str]) -> Dict[str, Any]:
    """Put the processor into one of the spec's named resolution modes (spec.res_modes: "lo" |
    "hi") through its own attributes; None / "" leaves the defaults. "max_pixels" goes through
    set_max_pixels (both knobs of transformers 4.57); every other key must already be an attribute
    of the image processor (the modes are pinned by tests/test_backbones.py on the cached
    processors). Returns the kwargs applied, for config.json. NOTE: Gemma pan-and-scan and InternVL
    tiling turn one frame into several image blocks — fine on the unfenced arms, not for the fence."""
    if not mode:
        return {}
    if mode not in spec.res_modes:
        raise ValueError(f"{spec.name}: no resolution mode {mode!r}; known: {sorted(spec.res_modes)}")
    kw = dict(spec.res_modes[mode])
    ip = processor.image_processor
    call_kw = {}
    for k, v in kw.items():
        if k == "max_pixels":
            set_max_pixels(spec, processor, int(v))
        elif not hasattr(ip, k):
            raise ValueError(f"{spec.name}: image processor {type(ip).__name__} has no attribute {k!r}")
        else:
            setattr(ip, k, v)
            call_kw[k] = v
    # The attribute alone is NOT enough: the processor's chat-template path merges its own class-level
    # defaults (Gemma3 do_pan_and_scan=False, InternVL crop_to_patches=True) over the image processor's
    # attributes, so the kwargs must also travel with every call (2026-10-07: Gemma's hi mode silently
    # stayed one tile; InternVL's "lo" silently tiled). encode() forwards them.
    processor._res_call_kwargs = call_kw
    return kw


def special_ids(spec: BackboneSpec, tokenizer: Any) -> SpecialIds:
    """Token ids of the spec's delimiters, checked against the tokenizer (unk / missing -> error)."""
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    unk = getattr(tok, "unk_token_id", None)

    def one(name: str, t: Optional[str]) -> Optional[int]:
        if t is None:
            return None
        i = tok.convert_tokens_to_ids(t)
        if i is None or int(i) < 0 or (unk is not None and int(i) == unk):
            raise ValueError(f"special token {name}={t!r} not in this tokenizer")
        return int(i)

    d = spec.delimiters
    return SpecialIds(vision_start=one("start", d.start), vision_end=one("end", d.end),
                      image_pad=one("pad", d.pad), turn_end=one("turn_end", d.turn_end),
                      inner=tuple(one(f"inner[{k}]", t) for k, t in enumerate(d.inner)))


def encode(spec: BackboneSpec, processor: Any, messages: Sequence[Dict[str, Any]], device: Any = None,
           *, add_generation_prompt: bool = True) -> Dict[str, Any]:
    """Chat messages -> model inputs through the processor's own template, plus the spec's
    assistant_prefix tokens appended after the generation prompt (input_ids, attention_mask and
    token_type_ids grow together). Every experiment encodes through this, so all arms of a
    backbone share one prompt protocol."""
    call_kw = dict(getattr(processor, "_res_call_kwargs", {}) or {})       # the active resolution mode (set_resolution)
    enc = dict(processor.apply_chat_template(list(messages), add_generation_prompt=add_generation_prompt, tokenize=True,
                                             return_dict=True, return_tensors="pt", **call_kw))
    if spec.assistant_prefix and add_generation_prompt:
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        extra = torch.tensor([tok(spec.assistant_prefix, add_special_tokens=False).input_ids], dtype=enc["input_ids"].dtype)
        enc["input_ids"] = torch.cat([enc["input_ids"], extra], dim=1)
        if "attention_mask" in enc:
            enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(extra)], dim=1)
        if "token_type_ids" in enc:
            enc["token_type_ids"] = torch.cat([enc["token_type_ids"], torch.zeros_like(extra)], dim=1)
    if device is not None:
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    return enc


# ------------------------------------------------------------------------------- blocks

def blocks(spec: BackboneSpec, input_ids_1d: torch.Tensor, sid: SpecialIds, layout: str
           ) -> Tuple[List[Span], int]:
    """(blocks, fin_start) for one tokenised prompt. "pair" = core.fence.layout_blocks (the
    tested rule: image span + its two markers; replica extends to the next start). "pad_run" =
    each maximal run of placeholder (+ inner) tokens, plus the end token when the spec has one;
    replica is not defined for pad_run. The slot of a block is its last position in both rules."""
    if spec.block_rule == "pair":
        if sid.vision_start is None or sid.vision_end is None:
            raise ValueError(f"{spec.name}: the pair rule needs start and end delimiters")
        if spec.sep_tokens == 0:
            return layout_blocks(input_ids_1d, layout, sid.legacy(), im_end_id=sid.turn_end)
        if layout == "replica":
            raise ValueError(f"{spec.name}: the replica layout is not defined with separator tokens")
        vs = (input_ids_1d == sid.vision_start).nonzero().flatten().tolist()
        ve = (input_ids_1d == sid.vision_end).nonzero().flatten().tolist()
        if not vs or len(vs) != len(ve):
            raise ValueError(f"vision markers mismatch: {len(vs)} starts, {len(ve)} ends")
        out = [(a, e + 1 + spec.sep_tokens) for a, e in zip(vs, ve)]
        if out[-1][1] > int(input_ids_1d.shape[0]):
            raise ValueError("the last block's separator runs past the prompt")
        for (_, b), (a2, _) in zip(out, out[1:]):
            if b != a2:                                  # a token between two frames would sit outside every block
                raise ValueError(f"{spec.name}: frames are not contiguous ({b} != {a2}); check sep_tokens")
        return out, out[-1][1]
    if spec.block_rule == "pad_run":
        if layout == "replica":
            raise ValueError(f"{spec.name}: the replica layout needs paired delimiters")
        ids = input_ids_1d.tolist()
        run = {sid.image_pad, *sid.inner}
        out: List[Span] = []
        i, L = 0, len(ids)
        while i < L:
            if ids[i] not in run:
                i += 1
                continue
            j = i
            while j < L and ids[j] in run:
                j += 1
            if sid.vision_end is not None:
                if j >= L or ids[j] != sid.vision_end:
                    raise ValueError(f"{spec.name}: image run at {i} has no end token")
                j += 1
            out.append((i, j))
            i = j
        if not out:
            raise ValueError("no image runs in the prompt")
        return out, out[-1][1]
    raise ValueError(f"unknown block_rule {spec.block_rule!r}")


def slots(blocks_: Sequence[Span], spec: Optional[BackboneSpec] = None) -> List[int]:
    """The per-frame slot = the frame's end token (Qwen <|vision_end|>, InternVL </img>, Gemma
    <end_of_image>): the last token of the block before its separator tokens. Where the gate reads."""
    off = spec.sep_tokens if spec is not None else 0
    return [b - 1 - off for _, b in blocks_]


def image_spans(blocks_: Sequence[Span], spec: BackboneSpec) -> List[Span]:
    """The image-token run inside each block: between the start token and the end token."""
    return [(a + 1, b - 1 - spec.sep_tokens) for a, b in blocks_]


# ---------------------------------------------------------------------------- positions

def base_positions(spec: BackboneSpec, model: Any, inputs: Dict[str, Any]) -> torch.Tensor:
    """The model's own position ids for this prompt, before the fence's per-block reset.
    mrope: get_rope_index -> (3, B, L). 1d: arange -> (B, L)."""
    ids = inputs["input_ids"]
    if spec.rope == "mrope":
        with torch.no_grad():
            pos, _ = get_rope_index_fn(model)(ids, image_grid_thw=inputs.get("image_grid_thw"),
                                              attention_mask=inputs.get("attention_mask"))
        return pos
    if spec.rope == "1d":
        B, L = int(ids.shape[0]), int(ids.shape[1])
        return torch.arange(L, device=ids.device).view(1, L).repeat(B, 1)
    raise ValueError(f"unknown rope {spec.rope!r}")


def fenced_setup(spec: BackboneSpec, model: Any, inputs: Dict[str, Any], blocks_: Sequence[Span],
                 fin_start: int, keep: Optional[Sequence[bool]]) -> Tuple[Any, torch.Tensor]:
    """(mask, position_ids) for one fenced forward: the block mask with the gate's hidden columns
    (keep=None or all-True = no gate) and the per-block position reset over the model's own
    positions. For a backbone with bidirectional image attention the image run of each block is
    opened as a full square; for one with local layers (layer_types="config") the mask is a dict
    {"full_attention": m, "sliding_attention": m_w}, the window taken in reset-position space
    (the same tensor twice when the window cuts nothing). Trainer and evaluator share this."""
    seq = int(inputs["input_ids"].shape[1])
    hide = hide_cols_for(blocks_, keep) if keep is not None else []
    spans = image_spans(blocks_, spec) if spec.image_attention == "bidirectional" else ()
    mask = build_block_mask(seq, blocks_, hide, bidir_spans=spans)
    pos = reset_positions(base_positions(spec, model, inputs), blocks_, fin_start)
    if spec.layer_types == "config":
        window = int(text_config(model).sliding_window)
        return {"full_attention": mask, "sliding_attention": sliding_window_mask(mask, pos.reshape(-1, seq)[0].cpu(), window)}, pos
    return mask, pos


# ----------------------------------------------------------------------------- geometry

def attention_dims(spec: BackboneSpec, model: Any) -> Dict[str, Any]:
    """Head geometry from the text config; mrope_section only exists for M-RoPE models."""
    tc = text_config(model)
    n_heads = int(tc.num_attention_heads)
    return {"n_heads": n_heads, "n_kv": int(tc.num_key_value_heads),
            "head_dim": int(getattr(tc, "head_dim", None) or tc.hidden_size // n_heads),
            "mrope_section": list(tc.rope_scaling["mrope_section"]) if spec.rope == "mrope" else None}
