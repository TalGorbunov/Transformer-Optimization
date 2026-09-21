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
    """Frozen Qwen2.5-VL runtime. Twin: runtime.py:47. sdpa is required for 4-D masks."""
    raise NotImplementedError


def get_layers(model: Any) -> Any:
    """The LM decoder layer list (model.model.language_model.layers on this HF version).
    Twin: runtime.py:70."""
    raise NotImplementedError


def text_config(model: Any) -> Any:
    """The text sub-config (hidden_size, num_hidden_layers, rope_scaling...). Twin: runtime.py:83."""
    raise NotImplementedError


def special_ids(processor: Any) -> Dict[str, int]:
    """{'vision_start': id, 'vision_end': id, 'image_pad': id} via the tokenizer.
    Pinned by tests/test_model.py::test_special_ids (CPU: tokenizer only)."""
    raise NotImplementedError


def image_token_groups(input_ids_1d: torch.Tensor, image_pad_id: int) -> List[List[int]]:
    """Consecutive runs of IMAGE_PAD positions, one list per image, in order.
    Twin: runtime.py:115 (there it re-derived the id from the processor each call)."""
    raise NotImplementedError


def get_rope_index_fn(model: Any) -> Any:
    """The model's multimodal RoPE position builder (its location differs across HF versions).
    Twin: runtime.py:110."""
    raise NotImplementedError


def move_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
