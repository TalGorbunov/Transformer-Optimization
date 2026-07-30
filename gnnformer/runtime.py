"""Model loading and access helpers for Qwen2.5-VL (frozen, 4-bit nf4, sdpa).

Replaces two legacy paths: models/model.py's ModelRuntime (whose default was the
32B — a recurring footgun) and the `gri.configure_runtime()/_model()/_processor()`
bootstrap that 40+ legacy scripts imported from a retired patching script.
Here the default is the thesis model (7B) and loading is a plain function call:

    rt = load_runtime()                      # 7B, 4-bit, sdpa
    layers = get_layers(rt.model)
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from .constants import MODEL_7B


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


def build_4bit_quantization_config(compute_dtype: torch.dtype = torch.bfloat16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )


def load_runtime(
    model_name: str = MODEL_7B,
    *,
    attn_implementation: str = "sdpa",
    use_4bit: bool = True,
    device_map: Any = "cuda",
) -> ModelRuntime:
    """Load a frozen Qwen2.5-VL runtime. sdpa is required for 4D-mask injection paths."""
    if attn_implementation == "eager":
        raise ValueError("eager attention is forbidden in this codebase (masks + speed).")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    kwargs: Dict[str, Any] = {
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if use_4bit:
        kwargs["quantization_config"] = build_4bit_quantization_config()
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model.eval()
    return ModelRuntime(model_name=str(model_name), processor=processor, model=model)


def get_layers(model: Any) -> Any:
    """The language-model decoder layer list, across HF layout variants."""
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
    return model.config.text_config if hasattr(model.config, "text_config") else model.config


def attention_dims(model: Any) -> Dict[str, Any]:
    """n_heads / n_kv / head_dim / mrope_section for message recompute."""
    cfg = text_config(model)
    n_heads = int(cfg.num_attention_heads)
    return {
        "n_heads": n_heads,
        "n_kv": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads)),
        "mrope_section": cfg.rope_scaling["mrope_section"],
        "hidden_size": int(cfg.hidden_size),
    }


def dequantize_linear_weight(linear: Any) -> torch.Tensor:
    """Dequantize a bnb Linear4bit weight to float32 (e.g. o_proj for message recompute)."""
    w = linear.weight
    if hasattr(w, "quant_state") and w.quant_state is not None:
        import bitsandbytes.functional as bnbF

        return bnbF.dequantize_4bit(w.data, w.quant_state).float()
    return w.data.float()


def get_rope_index_fn(model: Any) -> Any:
    """The model's multimodal RoPE position builder (layout differs across HF versions)."""
    return getattr(model, "get_rope_index", None) or model.model.get_rope_index


def image_token_groups(
    input_ids_1d: torch.Tensor,
    expected_num_frames: int,
    *,
    processor: Any,
) -> List[List[int]]:
    """Consecutive image-token position groups, one per frame (first N groups)."""
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    positions = (input_ids_1d == int(image_token_id)).nonzero(as_tuple=True)[0]
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
    return groups[:expected_num_frames]


def move_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
