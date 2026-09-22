"""Loading the frozen backbone and reaching into it.

Twin: legacy/v1/gnnformer/runtime.py (150 lines). Same API, minus the carrier-era
`attention_dims`/`dequantize_linear_weight` (only message recompute used them).

Contract (tests/test_model.py — CPU parts only; loading needs a GPU):
  * load_runtime() defaults to MODEL_ID, 4-bit nf4 + double quant, bf16 compute, sdpa.
    "eager" is refused (masks + speed). The model is put in eval() and never trained
    directly — LoRA adapters are attached by the experiments.
  * get_layers() returns the language-model decoder layer list across HF layout variants.
  * special_ids() returns the ids of VISION_START / VISION_END / IMAGE_PAD.
  * image_token_groups() returns, per image, the consecutive positions of its IMAGE_PAD
    tokens (used by the fence to delimit blocks).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from .constants import IMAGE_PAD, MODEL_ID, VISION_END, VISION_START


@dataclass(frozen=True)
class ModelRuntime:
    model_name: str
    processor: Any
    model: Any

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    @property
    def device(self) -> torch.device:
        return self.model.device


def load_runtime(model_name: str = MODEL_ID, *, attn_implementation: str = "sdpa",
                 use_4bit: bool = True, device_map: Any = "cuda") -> ModelRuntime:
    """Frozen Qwen2.5-VL runtime. Twin: runtime.py:47 (+ its build_4bit_quantization_config,
    inlined). sdpa is required for 4-D masks; "eager" is refused."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if attn_implementation == "eager":
        raise ValueError("eager attention is forbidden in this codebase (masks + speed).")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    kwargs: Dict[str, Any] = {
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model.eval()
    return ModelRuntime(model_name=str(model_name), processor=processor, model=model)


def get_layers(model: Any) -> Any:
    """The LM decoder layer list (model.model.language_model.layers on this HF version).
    Twin: runtime.py:70 (verbatim)."""
    for getter in (
        lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "layers", None),
        lambda m: getattr(getattr(m, "language_model", None), "layers", None),
        lambda m: getattr(getattr(m, "model", None), "layers", None),
    ):
        layers = getter(model)
        if layers is not None and len(layers) > 0:
            return layers
    raise RuntimeError("couldn't find transformer decoder layers on this model")


def text_config(model: Any) -> Any:
    """The text sub-config (hidden_size, num_hidden_layers, rope_scaling...). Twin: runtime.py:83."""
    return model.config.text_config if hasattr(model.config, "text_config") else model.config


def special_ids(processor: Any) -> Dict[str, int]:
    """{'vision_start': id, 'vision_end': id, 'image_pad': id} via the tokenizer.
    Pinned by tests/test_model.py::test_special_ids (CPU: tokenizer only)."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = {name: int(tok.convert_tokens_to_ids(t))
           for name, t in (("vision_start", VISION_START), ("vision_end", VISION_END), ("image_pad", IMAGE_PAD))}
    unk = getattr(tok, "unk_token_id", None)
    bad = [n for n, i in ids.items() if i is None or i < 0 or (unk is not None and i == unk)]
    if bad:
        raise ValueError(f"special tokens not in this tokenizer: {bad}")
    return ids


def image_token_groups(input_ids_1d: torch.Tensor, image_pad_id: int) -> List[List[int]]:
    """Consecutive runs of IMAGE_PAD positions, one list per image, in order.
    Twin: runtime.py:115 (there it re-derived the id from the processor each call)."""
    positions = (input_ids_1d == int(image_pad_id)).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return []
    groups: List[List[int]] = []
    current = [int(positions[0].item())]
    for pos in positions[1:]:
        p = int(pos.item())
        if p == current[-1] + 1:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    return groups


def get_rope_index_fn(model: Any) -> Any:
    """The model's multimodal RoPE position builder (its location differs across HF versions).
    Twin: runtime.py:110."""
    fn = getattr(model, "get_rope_index", None)
    if fn is None:
        inner = getattr(model, "model", None)
        fn = getattr(inner, "get_rope_index", None)
    if fn is None:
        raise RuntimeError("get_rope_index not found on this model")
    return fn


def move_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
