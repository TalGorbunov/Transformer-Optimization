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

from .constants import MODEL_ID


@dataclass(frozen=True)
class ModelRuntime:
    model_name: str
    processor: Any
    model: Any
    spec: Any = None            # the core.backbones.BackboneSpec this runtime was loaded from

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    @property
    def device(self) -> torch.device:
        return self.model.device


def load_runtime(model_name: str = MODEL_ID, *, attn_implementation: str = "sdpa",
                 use_4bit: bool = True, device_map: Any = "cuda") -> ModelRuntime:
    """Frozen Qwen2.5-VL runtime — kept for the scripts that still address the model by name.
    Since the backbone seam this is core.backbones.base.load_runtime on the Qwen spec (a
    different `model_name` = the same spec with that HF id / path). Twin: runtime.py:47."""
    from dataclasses import replace

    from .backbones.base import load_runtime as _load_runtime
    from .backbones.qwen2_5_vl import QWEN2_5_VL_7B

    spec = QWEN2_5_VL_7B if str(model_name) == MODEL_ID else replace(QWEN2_5_VL_7B, model_id=str(model_name))
    return _load_runtime(spec, attn_implementation=attn_implementation, use_4bit=use_4bit, device_map=device_map)


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
    """{'vision_start': id, 'vision_end': id, 'image_pad': id} via the tokenizer — the Qwen
    spec's ids in the legacy three-key shape (core.backbones.base.special_ids also carries
    turn_end). Pinned by tests/test_model.py::test_special_ids (CPU: tokenizer only)."""
    from .backbones.base import special_ids as _special_ids
    from .backbones.qwen2_5_vl import QWEN2_5_VL_7B

    return _special_ids(QWEN2_5_VL_7B, processor).legacy()


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
    """The model's multimodal RoPE position builder (its location differs across HF versions
    and wrappers): walks PEFT's get_base_model() and the `.model` nesting (ForConditionalGeneration
    -> inner model) until an object owns get_rope_index. Twin: runtime.py:110 (two levels only —
    the 2026-09-23 C2 failure: a PeftModel resolves `.model` to the OUTER model, one level short)."""
    obj = model
    for _ in range(6):
        fn = getattr(obj, "get_rope_index", None)
        if callable(fn):
            return fn
        base = getattr(obj, "get_base_model", None)       # PeftModel -> the wrapped model
        nxt = base() if callable(base) else None
        if nxt is None or nxt is obj:
            nxt = getattr(obj, "model", None)              # ForConditionalGeneration -> inner model
        if nxt is None or nxt is obj:
            break
        obj = nxt
    raise RuntimeError("get_rope_index not found on this model")


def move_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
