import argparse
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from nnsight import LanguageModel

from model import model as base_model, processor, get_layers, image_token_groups
from utils import iter_sample_dirs, load_mmred_sample

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)


def first_token_id_of_answer(answer_text: str) -> int:
    ids = processor.tokenizer.encode(str(answer_text).strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer text tokenized to empty: {answer_text!r}")
    return int(ids[0])


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_inputs(frames: Sequence[Any], question: str) -> Dict[str, torch.Tensor]:
    prompt = build_prompt(question, num_frames=len(frames))
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in frames] +
            [{"type": "text", "text": prompt}]
        ),
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(inputs)


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(base_model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def repeat_inputs_for_batch(
    inputs: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    if batch_size <= 1:
        return inputs

    repeated: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            repeated[key] = value
            continue

        if value.dim() == 0:
            repeated[key] = value.repeat(batch_size)
            continue

        # Text tensors are batch-major for a single sample and should be repeated on dim 0.
        if int(value.shape[0]) == 1:
            repeated[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))
            continue

        # Qwen-style multimodal tensors are not batch-major at batch=1; they are stacked over
        # all image/video items. For repeated identical samples, concatenate them batch_size times.
        if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
            repeated[key] = torch.cat([value] * batch_size, dim=0)
            continue

        raise ValueError(
            f"Cannot batch-repeat input {key!r} with shape={tuple(value.shape)}"
        )

    return repeated


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    match = _STEPS_IN_ROOM_RE.search(question_text)
    if not match:
        return None

    character = match.group(1).strip()
    room = match.group(2).strip()
    normalized_room = room[:1].upper() + room[1:].lower() if room else room
    return character, normalized_room


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


def collect_evidence_frame_indices(question: str, states: List[Dict[str, Any]]) -> List[int]:
    parsed = parse_target_character_room(question)
    if parsed is None:
        return []

    character, room = parsed
    frame_indices: List[int] = []
    for frame_idx, state in enumerate(states):
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        if character in room_to_chars.get(room, []):
            frame_indices.append(frame_idx)

    return frame_indices


def _materialize_saved(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def _to_hidden_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _to_hidden_tensor(x[0])
    raise TypeError(f"Unsupported layer output type for corruption: {type(x)}")


def run_clean_last_logits(lm: LanguageModel, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        with lm.trace(inputs):
            saved_logits = lm.output.logits[:, -1, :].save()
    return _materialize_saved(saved_logits)[0]


def run_layer_frame_corrupted_last_logits(
    lm: LanguageModel,
    layers: Any,
    inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    token_positions: List[int],
) -> torch.Tensor:
    if not token_positions:
        return run_clean_last_logits(lm, inputs)

    with torch.no_grad():
        with lm.trace(inputs):
            try:
                layer_out = _to_hidden_tensor(layers[layer_idx].output)
                layer_out[:, token_positions, :] = 0
            except Exception:
                # Some architectures expose output as tuple-like (hidden, ...)
                layer_out = _to_hidden_tensor(layers[layer_idx].output[0])
                layer_out[:, token_positions, :] = 0

            saved_logits = lm.output.logits[:, -1, :].save()
    return _materialize_saved(saved_logits)[0]


def run_layer_multi_frame_corrupted_last_logits(
    lm: LanguageModel,
    layers: Any,
    batched_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    token_positions_by_batch: List[List[int]],
) -> torch.Tensor:
    with torch.no_grad():
        with lm.trace(batched_inputs):
            try:
                layer_out = _to_hidden_tensor(layers[layer_idx].output)
                for batch_idx, token_positions in enumerate(token_positions_by_batch):
                    if token_positions:
                        layer_out[batch_idx, token_positions, :] = 0
            except Exception:
                # Some architectures expose output as tuple-like (hidden, ...)
                layer_out = _to_hidden_tensor(layers[layer_idx].output[0])
                for batch_idx, token_positions in enumerate(token_positions_by_batch):
                    if token_positions:
                        layer_out[batch_idx, token_positions, :] = 0

            saved_logits = lm.output.logits[:, -1, :].save()
    return _materialize_saved(saved_logits)


def pick_best_competitor_token_id(last_logits_1d: torch.Tensor, a_star_id: int) -> int:
    if a_star_id < 0 or a_star_id >= int(last_logits_1d.numel()):
        raise ValueError(f"a* token id out of range: {a_star_id}")

    competitor_logits = last_logits_1d.detach().clone()
    competitor_logits[a_star_id] = torch.finfo(competitor_logits.dtype).min
    a_hat_id = int(torch.argmax(competitor_logits).item())
    if a_hat_id == a_star_id:
        raise RuntimeError("Failed to select competitor token a^ distinct from a*.")
    return a_hat_id


def compute_ld(last_logits_1d: torch.Tensor, a_star_id: int, a_hat_id: int) -> float:
    return float((last_logits_1d[a_star_id] - last_logits_1d[a_hat_id]).item())


def normalize_to_probabilities(values: List[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0.0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def entropy_from_probabilities(probs: List[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in probs if p > 0.0)


def normalize_entropy(entropy: float, num_evidence_frames: int) -> float:
    if num_evidence_frames <= 1:
        return 0.0
    return float(entropy) / math.log(float(num_evidence_frames))


def format_corrupted_ld_table(
    evidence_frames: List[int],
    layer_rows: List[Tuple[int, List[float]]],
) -> str:
    headers = ["layer"] + [f"frame_{frame_idx}" for frame_idx in evidence_frames]

    table_rows: List[List[str]] = []
    for layer_idx, corrupted_lds in layer_rows:
        table_rows.append([str(layer_idx)] + [f"{float(ld):.4f}" for ld in corrupted_lds])

    col_widths = [
        max(
            len(headers[col_idx]),
            *(len(row[col_idx]) for row in table_rows),
        )
        for col_idx in range(len(headers))
    ]

    def _fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(col_widths[col_idx]) for col_idx, cell in enumerate(row)
        ) + " |"

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [separator, _fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in table_rows)
    lines.append(separator)
    return "\n".join(lines)


def format_layer_invalidity_table(
    sampled_counts: List[int],
    invalid_counts: List[int],
) -> str:
    headers = ["layer", "sampled", "invalid", "invalid_pct"]
    rows: List[List[str]] = []

    for layer_idx, (sampled, invalid) in enumerate(zip(sampled_counts, invalid_counts)):
        invalid_pct = (100.0 * float(invalid) / float(sampled)) if sampled > 0 else 0.0
        rows.append([str(layer_idx), str(sampled), str(invalid), f"{invalid_pct:.2f}%"])

    col_widths = [
        max(
            len(headers[col_idx]),
            *(len(row[col_idx]) for row in rows),
        )
        for col_idx in range(len(headers))
    ]

    def _fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(col_widths[col_idx]) for col_idx, cell in enumerate(row)
        ) + " |"

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [separator, _fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def plot_layer_invalidity_rates(
    sampled_counts: List[int],
    invalid_counts: List[int],
    output_dir: Path,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping invalidity plot: matplotlib is not available ({exc})")
        return None

    layers = list(range(len(sampled_counts)))
    invalid_rates = [
        (100.0 * float(invalid_counts[idx]) / float(sampled_counts[idx]))
        if sampled_counts[idx] > 0 else 0.0
        for idx in layers
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    ax.bar(layers, invalid_rates, color="#ff7f0e", width=0.8)
    ax.set_title("Layer Invalidity Rate (All Importances Zero)", fontsize=12, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Invalid rate (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    if layers:
        tick_step = max(1, math.ceil(len(layers) / 32))
        xticks = layers[::tick_step]
        if layers[-1] not in xticks:
            xticks.append(layers[-1])
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "layer_invalidity_rate.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_entropy_report(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    def _fmt_float_list(vals: List[float]) -> str:
        return "[" + ", ".join(f"{v:.8f}" for v in vals) + "]"

    lines: List[str] = []
    for sample in sample_metrics:
        lines.append(f"sample_id={sample['sample_id']}")
        lines.append(
            f"clean_ld={sample['clean_ld']:.8f} "
            f"a_star_id={sample['a_star_id']} a_hat_id={sample['a_hat_id']} "
            f"a_star_token={sample['a_star_token']!r} a_hat_token={sample['a_hat_token']!r}"
        )
        lines.append(f"evidence_frames={sample['evidence_frames']}")
        for layer_metrics in sample["layer_metrics"]["layers"]:
            lines.append(
                f"layer={layer_metrics['layer']} "
                f"r={_fmt_float_list(layer_metrics['r'])} "
                f"p={_fmt_float_list(layer_metrics['p'])} "
                f"H_norm={layer_metrics['entropy']:.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def plot_entropy_means_medians(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    seq_len_label: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot: matplotlib is not available ({exc})")
        return None

    entropy_by_layer: Dict[int, List[float]] = {}
    for sample in sample_metrics:
        for layer_metrics in sample["layer_metrics"]["layers"]:
            layer_idx = int(layer_metrics["layer"])
            entropy_by_layer.setdefault(layer_idx, []).append(float(layer_metrics["entropy"]))

    if not entropy_by_layer:
        return None

    rng = random.Random(seed)
    layers = sorted(entropy_by_layer.keys())
    mean_values: List[float] = []
    median_values: List[float] = []
    mean_lo_values: List[float] = []
    mean_hi_values: List[float] = []
    median_lo_values: List[float] = []
    median_hi_values: List[float] = []

    for layer_idx in layers:
        values = entropy_by_layer[layer_idx]
        n = len(values)
        sorted_values = sorted(values)

        mean_value = sum(values) / n
        if n % 2 == 1:
            median_value = sorted_values[n // 2]
        else:
            median_value = 0.5 * (sorted_values[n // 2 - 1] + sorted_values[n // 2])

        if n <= 1:
            mean_lo = mean_hi = mean_value
            median_lo = median_hi = median_value
        else:
            boot_means: List[float] = []
            boot_medians: List[float] = []
            for _ in range(n_bootstrap):
                sample = [values[rng.randrange(n)] for _ in range(n)]
                sample_sorted = sorted(sample)
                boot_means.append(sum(sample) / n)
                if n % 2 == 1:
                    boot_medians.append(sample_sorted[n // 2])
                else:
                    boot_medians.append(0.5 * (sample_sorted[n // 2 - 1] + sample_sorted[n // 2]))

            boot_means.sort()
            boot_medians.sort()
            lo_idx = int(0.025 * (n_bootstrap - 1))
            hi_idx = int(0.975 * (n_bootstrap - 1))
            mean_lo = boot_means[lo_idx]
            mean_hi = boot_means[hi_idx]
            median_lo = boot_medians[lo_idx]
            median_hi = boot_medians[hi_idx]

        mean_values.append(mean_value)
        median_values.append(median_value)
        mean_lo_values.append(mean_lo)
        mean_hi_values.append(mean_hi)
        median_lo_values.append(median_lo)
        median_hi_values.append(median_hi)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, mean_values, color="#1f77b4", linewidth=2.2, label="Mean entropy")
    ax.fill_between(
        layers,
        mean_lo_values,
        mean_hi_values,
        color="#1f77b4",
        alpha=0.2,
        label="Mean 95% CI",
    )
    ax.plot(layers, median_values, color="#d62728", linewidth=2.2, label="Median entropy")
    ax.fill_between(
        layers,
        median_lo_values,
        median_hi_values,
        color="#d62728",
        alpha=0.2,
        label="Median 95% CI",
    )

    title = "Entropy by Layer"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)

    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "entropy_summary.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def resolve_lambda_threshold(args: argparse.Namespace) -> float:
    lambda_arg = args.lambda_threshold
    min_clean_ld = args.min_clean_ld

    if lambda_arg is None and min_clean_ld is None:
        return 1.0
    if lambda_arg is None:
        return float(min_clean_ld)
    if min_clean_ld is None:
        return float(lambda_arg)

    if abs(float(lambda_arg) - float(min_clean_ld)) > 1e-12:
        raise ValueError(
            "Conflicting thresholds: --lambda and --min_clean_ld differ. "
            "Please pass only one or keep them equal."
        )
    return float(lambda_arg)


def main() -> None:
    start_time = time.perf_counter()

    ap = argparse.ArgumentParser(
        description=(
            "Compute per-layer evidence importances and entropies. "
            "For each sample: choose a* from GT answer, choose a^ as the best non-a* clean competitor, "
            "skip if LD_clean < lambda, then zero evidence-frame tokens at each layer and compute "
            "importance=max(LD_clean-LD_corrupted,0)."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs")

    ap.add_argument("--lambda", dest="lambda_threshold", type=float, default=None)
    ap.add_argument(
        "--min_clean_ld",
        type=float,
        default=None,
        help="Alias for --lambda (kept for backward compatibility).",
    )

    args = ap.parse_args()

    lambda_threshold = resolve_lambda_threshold(args)

    data_root = Path(args.data_root)
    output_dir = Path(args.output)

    seq_len_match = re.search(r"(seq_len_\d+)", str(data_root))
    seq_len_label = seq_len_match.group(1) if seq_len_match else None

    lm = LanguageModel(base_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    num_layers = len(layers)
    layer_sampled_counts = [0 for _ in range(num_layers)]
    layer_invalid_counts = [0 for _ in range(num_layers)]

    sample_dirs = iter_sample_dirs(data_root)
    sample_metrics: List[Dict[str, Any]] = []

    processed_samples = 0
    target_processed_samples = max(int(args.limit), 0)

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= target_processed_samples:
            break

        try:
            sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} skipped: load failure ({exc})")
            continue

        evidence_frames = collect_evidence_frame_indices(question, states)
        if len(evidence_frames) < 2:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} "
                f"skipped: evidence frames={len(evidence_frames)} < 2"
            )
            continue

        inputs = move_inputs_to_model_device(build_inputs(frames, question))

        try:
            clean_logits = run_clean_last_logits(lm, inputs)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: clean forward failed ({exc})")
            continue

        try:
            a_star_id = first_token_id_of_answer(answer)
            a_hat_id = pick_best_competitor_token_id(clean_logits, a_star_id)
            clean_ld = compute_ld(clean_logits, a_star_id, a_hat_id)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: token/LD setup failed ({exc})")
            continue

        if clean_ld < lambda_threshold:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                f"LD_clean={clean_ld:.4f} < lambda={lambda_threshold:.4f}"
            )
            continue

        frame_groups = image_token_groups(inputs["input_ids"][0], expected_num_frames=len(frames))
        if not frame_groups:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} "
                f"skipped: no image token groups found in tokenized input"
            )
            continue

        valid_evidence_frames: List[int] = []
        for frame_idx in evidence_frames:
            if frame_idx >= len(frame_groups):
                continue
            if not frame_groups[frame_idx]:
                continue
            valid_evidence_frames.append(frame_idx)

        if not valid_evidence_frames:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} "
                f"skipped: evidence frames exist but none map to image token spans"
            )
            continue

        print(
            f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} "
            f"LD_clean={clean_ld:.4f} lambda={lambda_threshold:.4f} "
            f"a*={a_star_id} a^={a_hat_id} evidence_frames={valid_evidence_frames}"
        )

        evidence_token_positions = [frame_groups[frame_idx] for frame_idx in valid_evidence_frames]
        try:
            batched_inputs = repeat_inputs_for_batch(inputs, batch_size=len(valid_evidence_frames))
        except Exception as exc:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} "
                f"skipped: failed to build batched inputs ({exc})"
            )
            continue

        per_layer_metrics: List[Dict[str, Any]] = []
        all_layer_corrupted_ld_rows: List[Tuple[int, List[float]]] = []
        skipped_zero_importance_layers = 0
        for layer_idx in range(len(layers)):
            layer_sampled_counts[layer_idx] += 1
            layer_corrupted_lds: List[float] = []
            layer_importances: List[float] = []

            try:
                corrupted_logits_batch = run_layer_multi_frame_corrupted_last_logits(
                    lm=lm,
                    layers=layers,
                    batched_inputs=batched_inputs,
                    layer_idx=layer_idx,
                    token_positions_by_batch=evidence_token_positions,
                )
            except Exception as exc:
                print(
                    f"  layer={layer_idx} failed batched corruption forward ({exc}); "
                    "using importance=0 for all evidence frames"
                )
                layer_corrupted_lds.extend([clean_ld] * len(valid_evidence_frames))
                layer_importances.extend([0.0] * len(valid_evidence_frames))
            else:
                for batch_idx in range(len(valid_evidence_frames)):
                    corrupted_ld = compute_ld(corrupted_logits_batch[batch_idx], a_star_id, a_hat_id)
                    importance = max(clean_ld - corrupted_ld, 0.0)
                    layer_corrupted_lds.append(corrupted_ld)
                    layer_importances.append(importance)

            all_layer_corrupted_ld_rows.append((layer_idx, list(layer_corrupted_lds)))

            if sum(layer_importances) <= 0.0:
                layer_invalid_counts[layer_idx] += 1
                skipped_zero_importance_layers += 1
                continue

            layer_probabilities = normalize_to_probabilities(layer_importances)
            layer_entropy = normalize_entropy(
                entropy_from_probabilities(layer_probabilities),
                num_evidence_frames=len(valid_evidence_frames),
            )

            per_layer_metrics.append({
                "layer": layer_idx,
                "evidence_frames": list(valid_evidence_frames),
                "corrupted_ld": layer_corrupted_lds,
                "r": layer_importances,
                "p": layer_probabilities,
                "entropy": layer_entropy,
            })

        if all_layer_corrupted_ld_rows:
            print("  Corrupted LD table (rows=layers, columns=evidence frames):")
            print(format_corrupted_ld_table(valid_evidence_frames, all_layer_corrupted_ld_rows))

        if not per_layer_metrics:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_id} skipped: "
                "all layers had zero total importance"
            )
            continue

        if skipped_zero_importance_layers > 0:
            print(
                f"  skipped zero-importance layers: {skipped_zero_importance_layers}"
            )

        sample_metrics.append({
            "sample_id": sample_id,
            "answer": answer,
            "clean_ld": clean_ld,
            "a_star_id": a_star_id,
            "a_hat_id": a_hat_id,
            "a_star_token": processor.tokenizer.decode([a_star_id], skip_special_tokens=True).strip(),
            "a_hat_token": processor.tokenizer.decode([a_hat_id], skip_special_tokens=True).strip(),
            "evidence_frames": list(valid_evidence_frames),
            "layer_metrics": {
                "layers": per_layer_metrics,
            },
        })
        processed_samples += 1

    output_path = write_entropy_report(sample_metrics, output_dir)
    print(f"Wrote sample metrics to: {output_path}")
    print(
        f"Processed {processed_samples} samples "
        f"(target limit={target_processed_samples}, lambda={lambda_threshold:.4f})."
    )

    plot_path = plot_entropy_means_medians(sample_metrics, output_dir, seq_len_label=seq_len_label)
    if plot_path is not None:
        print(f"Wrote entropy plot to: {plot_path}")
    else:
        print("Skipped entropy plot: no layer metrics available.")

    print("Layer invalidity summary (invalid = all importances are 0, entropy undefined):")
    print(format_layer_invalidity_table(layer_sampled_counts, layer_invalid_counts))
    invalidity_plot_path = plot_layer_invalidity_rates(
        layer_sampled_counts,
        layer_invalid_counts,
        output_dir,
    )
    if invalidity_plot_path is not None:
        print(f"Wrote layer invalidity plot to: {invalidity_plot_path}")
    else:
        print("Skipped layer invalidity plot: no matplotlib available.")

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_h = int(elapsed_seconds // 3600)
    elapsed_m = int((elapsed_seconds % 3600) // 60)
    elapsed_s = elapsed_seconds % 60.0
    print(
        f"Total runtime: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:05.2f} "
        f"({elapsed_seconds:.2f}s)"
    )


if __name__ == "__main__":
    main()
