import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import (
    build_inputs_for_answer_token,
    compute_per_layer_frame_scores,
    force_eager_attention_backend,
    image_token_groups,
    move_inputs_to_model_device,
)
from utils import (
    append_metric,
    iter_sample_dirs,
    load_mmred_sample,
    mean,
    parse_sample_metrics_file,
    plot_attention_importance_summary,
)

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

    prompt_text = build_prompt(question, num_frames=len(frames))
    inputs, answer_token_index = build_inputs_for_answer_token(frames, prompt_text, str(answer).strip())
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
