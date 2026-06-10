import gc
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
VALID_ATTENTION_IMPLEMENTATIONS = ("sdpa", "flash_attention_2")
_FLASH_ATTENTION_2_MODULE_NAMES = ("flash_attn", "flash_attn_2_cuda")
_LOGGED_ATTENTION_BACKEND_EVENTS: set[Tuple[str, str, str, str]] = set()
_DEFAULT_RUNTIME: Optional["ModelRuntime"] = None


@dataclass(frozen=True)
class ModelRuntime:
    model_name: str
    processor: Any
    model: Any


def build_4bit_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )


def load_model_runtime(
    model_name: str,
    *,
    device_map: Any = "cuda",
    device: Optional[str] = None,
    use_4bit: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = True,
    use_fast_processor: bool = False,
) -> ModelRuntime:
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_processor,
    )

    model_kwargs: Dict[str, Any] = {
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if use_4bit:
        quantization_config = build_4bit_quantization_config()
        if torch_dtype is not None:
            quantization_config.bnb_4bit_compute_dtype = torch_dtype
        model_kwargs["quantization_config"] = quantization_config
    elif torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    if device_map is None and device is not None:
        model.to(device)
    model.eval()
    configure_attention_backend(attn_implementation, model_obj=model)
    return ModelRuntime(model_name=str(model_name), processor=processor, model=model)


def get_default_runtime() -> ModelRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = load_model_runtime(DEFAULT_MODEL_ID)
    return _DEFAULT_RUNTIME


def move_inputs_to_model_device(
    inputs: Dict[str, torch.Tensor],
    *,
    model_obj: Any,
) -> Dict[str, torch.Tensor]:
    device = next(model_obj.parameters()).device
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def get_special_token_ids(*, processor: Any) -> set[int]:
    token_ids: set[int] = set()
    tokenizer = processor.tokenizer
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "sep_token_id", "cls_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            token_ids.add(int(token_id))

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(tokenizer, "image_token_id", None)
    if image_token_id is not None:
        token_ids.add(int(image_token_id))
    return token_ids


def longest_common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    limit = min(int(a.numel()), int(b.numel()))
    idx = 0
    while idx < limit and int(a[idx].item()) == int(b[idx].item()):
        idx += 1
    return idx


def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    last_start = len(haystack) - len(needle) + 1
    for start in range(last_start):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def find_answer_token_index(
    prompt_input_ids: torch.Tensor,
    full_input_ids: torch.Tensor,
    answer_text: str,
    attention_mask: Optional[torch.Tensor],
    *,
    processor: Any,
) -> int:
    prefix_len = longest_common_prefix_len(prompt_input_ids, full_input_ids)
    answer_ids = processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    suffix = [int(token) for token in full_input_ids[prefix_len:].tolist()]
    relative_start = find_subsequence(suffix, [int(token) for token in answer_ids])
    if relative_start is not None and answer_ids:
        return prefix_len + relative_start

    active_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(full_input_ids.numel())
    special_token_ids = get_special_token_ids(processor=processor)
    idx = min(prefix_len, max(0, active_len - 1))
    while idx < active_len and int(full_input_ids[idx].item()) in special_token_ids:
        idx += 1
    if idx >= active_len:
        idx = max(0, active_len - 1)
        while idx > 0 and int(full_input_ids[idx].item()) in special_token_ids:
            idx -= 1
    return idx


def build_inputs_for_answer_token(
    frames: List[Any],
    prompt_text: str,
    answer_text: str,
    *,
    processor: Any,
) -> Tuple[Dict[str, torch.Tensor], int]:
    user_content = (
        [{"type": "image", "image": image} for image in frames]
        + [{"type": "text", "text": prompt_text}]
    )
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": answer_text}],
    }]

    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_inputs = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    answer_token_index = find_answer_token_index(
        prompt_input_ids=prompt_inputs["input_ids"][0],
        full_input_ids=full_inputs["input_ids"][0],
        answer_text=answer_text,
        attention_mask=full_inputs.get("attention_mask"),
        processor=processor,
    )
    return dict(full_inputs), answer_token_index


def image_token_groups(
    input_ids_1d: torch.Tensor,
    expected_num_frames: int,
    *,
    processor: Any,
) -> List[List[int]]:
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        return []

    positions = (input_ids_1d == int(image_token_id)).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return []

    groups: List[List[int]] = []
    current_group = [int(positions[0].item())]
    for pos in positions[1:]:
        pos_int = int(pos.item())
        if pos_int == current_group[-1] + 1:
            current_group.append(pos_int)
        else:
            groups.append(current_group)
            current_group = [pos_int]
    groups.append(current_group)
    return groups[:expected_num_frames]


def get_layers(model_obj: Any) -> Any:
    candidates = [
        lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "layers", None),
        lambda m: getattr(getattr(m, "language_model", None), "layers", None),
        lambda m: getattr(getattr(m, "model", None), "layers", None),
        lambda m: getattr(getattr(m, "text_model", None), "layers", None),
    ]

    for getter in candidates:
        layers = getter(model_obj)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers

    raise RuntimeError("Couldn't automatically find transformer layers.")


def compute_frame_scores_from_single_layer_attn(
    attn: torch.Tensor,
    frame_to_tokens: Dict[int, List[int]],
    answer_token_index: int,
) -> Dict[int, float]:
    attention = attn[0].detach()
    if answer_token_index < 0 or answer_token_index >= int(attention.shape[-2]):
        raise RuntimeError(
            f"answer_token_index={answer_token_index} is outside attention shape {tuple(attention.shape)}"
        )

    frame_scores: Dict[int, float] = {}
    for frame_idx in sorted(frame_to_tokens.keys()):
        token_positions = frame_to_tokens[frame_idx]
        if not token_positions:
            frame_scores[frame_idx] = 0.0
            continue
        token_attention = attention[:, answer_token_index, token_positions].sum(dim=-1)
        frame_scores[frame_idx] = float(token_attention.mean().item())
    return frame_scores


def set_attr_if_exists(obj: Any, attr: str, value: Any) -> None:
    if obj is None or not hasattr(obj, attr):
        return
    try:
        setattr(obj, attr, value)
    except Exception:
        return


def _attention_configs(model_obj: Any) -> List[Tuple[str, Any]]:
    candidates = [
        ("model.config", getattr(model_obj, "config", None)),
        ("model.config.text_config", getattr(getattr(model_obj, "config", None), "text_config", None)),
        ("model.config.vision_config", getattr(getattr(model_obj, "config", None), "vision_config", None)),
        ("model.model.config", getattr(getattr(model_obj, "model", None), "config", None)),
        (
            "model.model.language_model.config",
            getattr(getattr(getattr(model_obj, "model", None), "language_model", None), "config", None),
        ),
        (
            "model.model.visual.config",
            getattr(getattr(getattr(model_obj, "model", None), "visual", None), "config", None),
        ),
        ("model.language_model.config", getattr(getattr(model_obj, "language_model", None), "config", None)),
        ("model.visual.config", getattr(getattr(model_obj, "visual", None), "config", None)),
    ]
    unique: List[Tuple[str, Any]] = []
    seen_ids: set[int] = set()
    for name, config in candidates:
        if config is None or id(config) in seen_ids:
            continue
        seen_ids.add(id(config))
        unique.append((name, config))
    return unique


def attention_implementation_details(model_obj: Any) -> Dict[str, str]:
    details: Dict[str, str] = {}
    for name, config in _attention_configs(model_obj):
        implementation = getattr(config, "_attn_implementation", None)
        if implementation is None:
            implementation = getattr(config, "attn_implementation", None)
        details[name] = "<unset>" if implementation is None else str(implementation)
    return details


def get_active_attention_implementation(model_obj: Any) -> str:
    details = attention_implementation_details(model_obj)
    configured = {
        name: implementation
        for name, implementation in details.items()
        if implementation not in {"", "<unset>"}
    }
    if not configured:
        raise RuntimeError("Unable to determine an active attention implementation from the loaded model configs.")
    unique_values = sorted(set(configured.values()))
    if len(unique_values) != 1:
        raise RuntimeError(
            "Inconsistent attention implementations detected across model configs: "
            f"{configured}"
        )
    return unique_values[0]


def assert_non_eager_attention_backend(model_obj: Any) -> str:
    active_implementation = get_active_attention_implementation(model_obj)
    if active_implementation == "eager":
        raise RuntimeError("Eager attention is forbidden in this codebase.")
    if active_implementation not in VALID_ATTENTION_IMPLEMENTATIONS:
        raise RuntimeError(
            "Unsupported attention implementation detected. "
            f"Expected one of {VALID_ATTENTION_IMPLEMENTATIONS}, found {active_implementation!r}."
        )
    return active_implementation


def assert_sdpa_attention_backend(model_obj: Any) -> str:
    active_implementation = assert_non_eager_attention_backend(model_obj)
    if active_implementation != "sdpa":
        raise RuntimeError(
            "This path requires SDPA attention. "
            f"Found active attention implementation {active_implementation!r}."
        )
    return active_implementation


def flash_attention_2_available() -> bool:
    return any(importlib.util.find_spec(module_name) is not None for module_name in _FLASH_ATTENTION_2_MODULE_NAMES)


def preferred_attention_backend(requires_abp_mask: bool) -> str:
    return "sdpa" if requires_abp_mask else "flash_attention_2"


def resolve_attention_backend_choice(
    *,
    requires_abp_mask: bool,
    allow_sdpa_fallback: bool = True,
) -> Tuple[str, str, Optional[str]]:
    preferred_backend = preferred_attention_backend(requires_abp_mask=requires_abp_mask)
    if preferred_backend == "flash_attention_2" and not flash_attention_2_available():
        if not allow_sdpa_fallback:
            raise RuntimeError(
                "flash_attention_2 was requested for a non-ABP path, but it is unavailable in this environment."
            )
        return preferred_backend, "sdpa", "flash_attention_2 is unavailable in this environment"
    return preferred_backend, preferred_backend, None


def attention_backend_policy_summary(model_obj: Any) -> Dict[str, Any]:
    preferred_unmasked, actual_unmasked, unmasked_reason = resolve_attention_backend_choice(
        requires_abp_mask=False,
        allow_sdpa_fallback=True,
    )
    preferred_masked, actual_masked, masked_reason = resolve_attention_backend_choice(
        requires_abp_mask=True,
        allow_sdpa_fallback=True,
    )
    return {
        "startup_active_backend": assert_non_eager_attention_backend(model_obj),
        "flash_attention_2_available": bool(flash_attention_2_available()),
        "unmasked_paths": {
            "preferred_backend": preferred_unmasked,
            "actual_backend": actual_unmasked,
            "fallback_reason": unmasked_reason,
        },
        "abp_masked_paths": {
            "preferred_backend": preferred_masked,
            "actual_backend": actual_masked,
            "fallback_reason": masked_reason,
        },
        "eager_forbidden": True,
    }


def configure_attention_backend(target_backend: str, *, model_obj: Any) -> str:
    if target_backend == "eager":
        raise RuntimeError("Eager attention is forbidden in this codebase.")
    if target_backend not in VALID_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported attention backend {target_backend!r}. "
            f"Expected one of {VALID_ATTENTION_IMPLEMENTATIONS}."
        )
    active_implementation = assert_non_eager_attention_backend(model_obj)
    if active_implementation == target_backend:
        for _, config in _attention_configs(model_obj):
            set_attr_if_exists(config, "output_attentions", False)
        return active_implementation

    if hasattr(model_obj, "set_attn_implementation"):
        model_obj.set_attn_implementation(target_backend)

    for _, config in _attention_configs(model_obj):
        set_attr_if_exists(config, "_attn_implementation", target_backend)
        set_attr_if_exists(config, "attn_implementation", target_backend)
        set_attr_if_exists(config, "output_attentions", False)

    active_implementation = assert_non_eager_attention_backend(model_obj)
    if active_implementation != target_backend:
        raise RuntimeError(
            f"Failed to activate attention backend {target_backend!r}; active backend is {active_implementation!r}."
        )
    return active_implementation


def configure_sdpa_attention_backend(*, model_obj: Any) -> str:
    return configure_attention_backend("sdpa", model_obj=model_obj)


def _log_attention_backend_selection(
    *,
    path_name: str,
    preferred_backend: str,
    active_backend: str,
    requires_abp_mask: bool,
    fallback_reason: Optional[str],
) -> None:
    event_key = (
        str(path_name),
        str(preferred_backend),
        str(active_backend),
        str(fallback_reason or ""),
    )
    if event_key in _LOGGED_ATTENTION_BACKEND_EVENTS:
        return
    _LOGGED_ATTENTION_BACKEND_EVENTS.add(event_key)
    suffix = "" if not fallback_reason else f" fallback_reason={fallback_reason}"
    print(
        f"[backend] path={path_name} requires_abp_mask={bool(requires_abp_mask)} "
        f"preferred={preferred_backend} active={active_backend} eager_forbidden=True{suffix}"
    )


def prepare_attention_backend_for_forward(
    *,
    path_name: str,
    requires_abp_mask: bool,
    output_attentions: bool = False,
    allow_sdpa_fallback: bool = True,
    model_obj: Any,
) -> str:
    if output_attentions:
        raise RuntimeError(
            "output_attentions=True is not supported in the non-eager configuration. "
            "This codebase never falls back to eager attention."
        )
    preferred_backend, target_backend, fallback_reason = resolve_attention_backend_choice(
        requires_abp_mask=requires_abp_mask,
        allow_sdpa_fallback=allow_sdpa_fallback,
    )
    try:
        active_backend = configure_attention_backend(target_backend, model_obj=model_obj)
    except Exception as exc:
        if target_backend != "flash_attention_2" or not allow_sdpa_fallback:
            raise
        fallback_reason = (
            f"{fallback_reason}; flash_attention_2 activation failed: {exc}"
            if fallback_reason
            else f"flash_attention_2 activation failed: {exc}"
        )
        active_backend = configure_attention_backend("sdpa", model_obj=model_obj)
    if active_backend == "eager":
        raise RuntimeError("Eager attention is forbidden in this codebase.")
    _log_attention_backend_selection(
        path_name=path_name,
        preferred_backend=preferred_backend,
        active_backend=active_backend,
        requires_abp_mask=requires_abp_mask,
        fallback_reason=fallback_reason,
    )
    return active_backend


def ensure_sdpa_runtime_ready(*, model_obj: Any, output_attentions: bool = False) -> None:
    prepare_attention_backend_for_forward(
        path_name="sdpa_required_path",
        requires_abp_mask=True,
        output_attentions=output_attentions,
        allow_sdpa_fallback=False,
        model_obj=model_obj,
    )


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def compute_per_layer_frame_scores(
    model_inputs: Dict[str, Any],
    frame_to_tokens: Dict[int, List[int]],
    answer_token_index: int,
    chunk_layers: int,
    *,
    model_obj: Any,
) -> Dict[int, Dict[int, float]]:
    del model_inputs, frame_to_tokens, answer_token_index, chunk_layers
    ensure_sdpa_runtime_ready(model_obj=model_obj, output_attentions=True)
    raise RuntimeError(
        "Per-layer attention capture is not available in the SDPA-only configuration. "
        "This codebase no longer switches back to eager attention in order to read `output_attentions`."
    )
