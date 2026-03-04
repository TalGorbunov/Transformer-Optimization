import argparse
import ast
import gc
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import model, processor
from utils import iter_sample_dirs, load_mmred_sample, plot_attention_importance_summary

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_inputs_for_answer_token(
    frames: List[Any],
    question: str,
    answer_text: str,
) -> Tuple[Dict[str, torch.Tensor], int]:
    user_content = (
        [{"type": "image", "image": image} for image in frames] +
        [{"type": "text", "text": build_prompt(question, num_frames=len(frames))}]
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


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    match = _STEPS_IN_ROOM_RE.search(question_text)
    if not match:
        return None
    character = match.group(1).strip()
    room = match.group(2).strip()
    room = room[:1].upper() + room[1:].lower() if room else room
    return character, room


def rooms_to_room2chars(rooms: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(rooms, dict):
        return {}

    if any(isinstance(value, list) for value in rooms.values()):
        room_to_chars: Dict[str, List[str]] = {}
        for room_name, chars in rooms.items():
            if not isinstance(room_name, str):
                continue
            normalized_room = room_name[:1].upper() + room_name[1:].lower() if room_name else room_name
            room_to_chars.setdefault(normalized_room, [])
            if isinstance(chars, list):
                room_to_chars[normalized_room].extend(str(char) for char in chars)
        return {room: sorted(set(chars)) for room, chars in room_to_chars.items()}

    room_to_chars: Dict[str, List[str]] = {}
    for char_name, room_name in rooms.items():
        if not isinstance(room_name, str):
            continue
        normalized_room = room_name[:1].upper() + room_name[1:].lower()
        room_to_chars.setdefault(normalized_room, []).append(str(char_name))
    return {room: sorted(set(chars)) for room, chars in room_to_chars.items()}


def get_evidence_frame_indices(question: str, states: List[Dict[str, Any]]) -> Optional[List[int]]:
    parsed = parse_target_character_room(question)
    if parsed is None:
        return None

    character, room = parsed
    evidence_frames: List[int] = []
    for frame_idx, state in enumerate(states):
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        if character in room_to_chars.get(room, []):
            evidence_frames.append(frame_idx)
    return evidence_frames


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


def get_language_model_layers() -> Any:
    candidates = [
        lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "layers", None),
        lambda m: getattr(getattr(m, "language_model", None), "layers", None),
        lambda m: getattr(getattr(m, "model", None), "layers", None),
    ]
    for getter in candidates:
        layers = getter(model)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise RuntimeError("Could not find language-model decoder layers.")


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

    layers = get_language_model_layers()
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


def parse_sample_metrics_file(sample_metrics_path: Path) -> Dict[str, Dict[int, List[float]]]:
    if not sample_metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {sample_metrics_path}")

    layer_re = re.compile(r"^layer=(\d+)\s+r=(\[[^\]]*\])\s+p=(\[[^\]]*\])\s+H_norm=")
    metrics_by_sample: Dict[str, Dict[int, List[float]]] = {}
    current_sample_id: Optional[str] = None

    for raw_line in sample_metrics_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("sample_id="):
            current_sample_id = line.split("=", 1)[1].strip()
            metrics_by_sample.setdefault(current_sample_id, {})
            continue
        if current_sample_id is None:
            continue

        match = layer_re.match(line)
        if not match:
            continue
        layer_idx = int(match.group(1))
        r_values = ast.literal_eval(match.group(2))
        metrics_by_sample[current_sample_id][layer_idx] = [float(value) for value in r_values]

    return metrics_by_sample


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def append_metric(
    target: Dict[int, List[float]],
    layer_idx: int,
    value: Optional[float],
) -> None:
    if value is None:
        return
    target.setdefault(layer_idx, []).append(value)


def process_sample(
    sample_dir: Path,
    importances_by_sample: Dict[str, Dict[int, List[float]]],
    chunk_layers: int,
) -> Optional[Dict[str, Dict[int, float]]]:
    sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
    evidence_frames = get_evidence_frame_indices(question, states)
    if not evidence_frames:
        print(f"[skip] sample={sample_id} no evidence frames parsed from question")
        return None

    sample_importances = importances_by_sample.get(sample_id)
    if sample_importances is None:
        print(f"[skip] sample={sample_id} missing sample_metrics entry")
        return None

    inputs, answer_token_index = build_inputs_for_answer_token(frames, question, str(answer).strip())
    frame_groups = image_token_groups(inputs["input_ids"][0], expected_num_frames=len(frames))
    if len(frame_groups) != len(frames):
        print(
            f"[skip] sample={sample_id} image-token groups mismatch "
            f"(frames={len(frames)} groups={len(frame_groups)})"
        )
        return None

    frame_to_tokens = {frame_idx: frame_groups[frame_idx] for frame_idx in range(len(frame_groups))}
    per_layer_scores = compute_per_layer_frame_scores(
        move_inputs_to_model_device(inputs),
        frame_to_tokens,
        answer_token_index,
        chunk_layers,
    )

    importance_values_by_layer: Dict[int, float] = {}
    attention_values_by_layer: Dict[int, float] = {}

    num_frames = len(frames)

    for layer_idx, frame_scores in per_layer_scores.items():
        r_values = sample_importances.get(layer_idx)
        if r_values is None:
            continue

        evidence_importance_by_frame: Dict[int, float]
        if len(r_values) == len(evidence_frames):
            evidence_importance_by_frame = {
                frame_idx: float(r_values[idx])
                for idx, frame_idx in enumerate(evidence_frames)
            }
        elif len(r_values) == num_frames:
            evidence_importance_by_frame = {
                frame_idx: float(r_values[frame_idx])
                for frame_idx in evidence_frames
                if 0 <= frame_idx < len(r_values)
            }
        else:
            continue

        valid_evidence_frames = [
            frame_idx for frame_idx in evidence_frames
            if frame_idx in frame_scores and frame_idx in evidence_importance_by_frame
        ]
        if not valid_evidence_frames:
            continue

        importance_values_by_layer[layer_idx] = mean(
            [evidence_importance_by_frame[frame_idx] for frame_idx in valid_evidence_frames]
        )

        total_attention = sum(float(score) for score in frame_scores.values())
        if total_attention <= 0.0:
            continue
        evidence_attention = sum(float(frame_scores[frame_idx]) for frame_idx in valid_evidence_frames)
        attention_values_by_layer[layer_idx] = evidence_attention / total_attention

    if not importance_values_by_layer or not attention_values_by_layer:
        print(f"[skip] sample={sample_id} no valid per-layer comparisons")
        return None

    compared_layers = sorted(set(importance_values_by_layer) & set(attention_values_by_layer))
    if not compared_layers:
        print(f"[skip] sample={sample_id} no overlapping valid layers")
        return None

    importance_values_by_layer = {layer_idx: importance_values_by_layer[layer_idx] for layer_idx in compared_layers}
    attention_values_by_layer = {layer_idx: attention_values_by_layer[layer_idx] for layer_idx in compared_layers}
    print(
        f"[ok] sample={sample_id} evidence_frames={evidence_frames} "
        f"answer_token_index={answer_token_index} layers={len(compared_layers)}"
    )
    return {
        "importance": importance_values_by_layer,
        "attention": attention_values_by_layer,
    }


def aggregate_samples(
    sample_results: List[Dict[str, Dict[int, float]]],
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    importance_by_layer: Dict[int, List[float]] = {}
    attention_by_layer: Dict[int, List[float]] = {}

    for result in sample_results:
        for layer_idx, value in result["importance"].items():
            append_metric(importance_by_layer, layer_idx, value)
        for layer_idx, value in result["attention"].items():
            append_metric(attention_by_layer, layer_idx, value)

    return importance_by_layer, attention_by_layer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-layer evidence attention ratios from the first answer token and "
            "compare them against per-layer evidence importances from sample_metrics.txt."
        )
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to sample_metrics.txt.")
    parser.add_argument("--limit", type=int, default=-1, help="Max number of compared samples; -1 means all.")
    parser.add_argument(
        "--attention_chunk_layers",
        type=int,
        default=1,
        help="Number of layers to capture per forward pass.",
    )
    parser.add_argument(
        "--plot_out_dir",
        type=str,
        default=None,
        help="Output directory for the plots. Defaults to the metrics file directory.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    importances_by_sample = parse_sample_metrics_file(metrics_path)
    sample_dirs = iter_sample_dirs(Path(args.data_root))
    sample_dirs = [sample_dir for sample_dir in sample_dirs if sample_dir.name in importances_by_sample]

    if not sample_dirs:
        raise RuntimeError(f"No matching samples found under {args.data_root}")

    force_eager_attention_backend()

    sample_results: List[Dict[str, Dict[int, float]]] = []
    for sample_dir in sample_dirs:
        result = process_sample(
            sample_dir=sample_dir,
            importances_by_sample=importances_by_sample,
            chunk_layers=args.attention_chunk_layers,
        )
        if result is None:
            continue
        sample_results.append(result)
        if args.limit >= 0 and len(sample_results) >= args.limit:
            break

    if not sample_results:
        raise RuntimeError("No valid samples were processed.")

    importance_by_layer, attention_by_layer = aggregate_samples(sample_results)
    plot_out_dir = Path(args.plot_out_dir) if args.plot_out_dir else metrics_path.parent
    plot_out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_attention_importance_summary(
        importance_by_layer,
        attention_by_layer,
        plot_out_dir / "attention_vs_importance_summary.png",
    )
    if plot_path is None:
        raise RuntimeError("No valid layers available for plotting.")
    print(f"[plot] wrote {plot_path}")


if __name__ == "__main__":
    main()
