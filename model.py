import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def get_special_token_ids() -> set:
    token_ids = set()
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
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def find_answer_token_index(
    prompt_input_ids: torch.Tensor,
    full_input_ids: torch.Tensor,
    answer_text: str,
    attention_mask: Optional[torch.Tensor],
) -> int:
    prefix_len = longest_common_prefix_len(prompt_input_ids, full_input_ids)
    answer_ids = processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    suffix = [int(token) for token in full_input_ids[prefix_len:].tolist()]
    relative_start = find_subsequence(suffix, [int(token) for token in answer_ids])
    if relative_start is not None and answer_ids:
        return prefix_len + relative_start

    active_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(full_input_ids.numel())
    special_token_ids = get_special_token_ids()
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
) -> Tuple[Dict[str, torch.Tensor], int]:
    user_content = (
        [{"type": "image", "image": image} for image in frames] +
        [{"type": "text", "text": prompt_text}]
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
    )
    return dict(full_inputs), answer_token_index


def image_token_groups(input_ids_1d: torch.Tensor, expected_num_frames: int) -> List[List[int]]:
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


def get_layers(model_obj: Optional[Any] = None) -> Any:
    model_obj = model if model_obj is None else model_obj
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


def force_eager_attention_backend() -> None:
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ]
    for config in configs:
        set_attr_if_exists(config, "_attn_implementation", "eager")
        set_attr_if_exists(config, "attn_implementation", "eager")
        set_attr_if_exists(config, "output_attentions", True)


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
) -> Dict[int, Dict[int, float]]:
    if chunk_layers <= 0:
        raise ValueError("chunk_layers must be positive.")

    layers = get_layers()
    per_layer_scores: Dict[int, Dict[int, float]] = {}

    for chunk_start in range(0, len(layers), chunk_layers):
        chunk_end = min(len(layers), chunk_start + chunk_layers)
        original_forwards: Dict[int, Any] = {}
        outputs = None

        for layer_idx in range(chunk_start, chunk_end):
            layer = layers[layer_idx]
            original_forwards[layer_idx] = layer.forward

            def make_wrapped_forward(current_layer_idx: int, original_forward: Any):
                def wrapped_forward(*args, **kwargs):
                    kwargs["output_attentions"] = True
                    out = original_forward(*args, **kwargs)
                    if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
                        raise RuntimeError(f"Layer {current_layer_idx} did not return attention.")

                    per_layer_scores[current_layer_idx] = compute_frame_scores_from_single_layer_attn(
                        out[1],
                        frame_to_tokens,
                        answer_token_index,
                    )
                    return (out[0],) + tuple(out[2:])

                return wrapped_forward

            layer.forward = make_wrapped_forward(layer_idx, original_forwards[layer_idx])

        try:
            outputs = model(**model_inputs, output_attentions=False, use_cache=False, return_dict=True)
            for layer_idx in range(chunk_start, chunk_end):
                if layer_idx not in per_layer_scores:
                    raise RuntimeError(f"Missing attention capture for layer {layer_idx}.")
        finally:
            for layer_idx, original_forward in original_forwards.items():
                layers[layer_idx].forward = original_forward
            if outputs is not None:
                del outputs
            release_torch_memory()

    return {layer_idx: per_layer_scores[layer_idx] for layer_idx in sorted(per_layer_scores)}
