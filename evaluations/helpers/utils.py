import ast
import math
import random
import re
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)

def describe(x, name="x", max_list=8):
    """Print structure + (if tensor) shape/dtype/device."""
    print(f"\n=== {name} ===")
    if x is None:
        print("None")
        return

    # torch tensor
    if isinstance(x, torch.Tensor):
        print("Tensor")
        print(" shape:", tuple(x.shape))
        print(" dtype:", x.dtype)
        print(" device:", x.device)
        return

    # tuple/list
    if isinstance(x, (tuple, list)):
        print(type(x).__name__, "len=", len(x))
        for i, xi in enumerate(x[:max_list]):
            describe(xi, name=f"{name}[{i}]", max_list=max_list)
        if len(x) > max_list:
            print(f"... ({len(x)-max_list} more)")
        return

    # dict
    if isinstance(x, dict):
        print("dict keys:", list(x.keys())[:max_list])
        for k in list(x.keys())[:max_list]:
            describe(x[k], name=f"{name}['{k}']", max_list=max_list)
        if len(x) > max_list:
            print(f"... ({len(x)-max_list} more keys)")
        return

    # fallback
    print("type:", type(x))
    s = str(x)
    print(s[:500] + ("..." if len(s) > 500 else ""))


def load_mmred_sample(sample_dir: Path):
    """
    Returns:
      (sample_id, frames_list[PIL.Image], question_text, states_list[dict], answer_text)

    Expected qa.txt format (like your example):
      qid: ...
      qtype: ...
      ...
      question:
      { ... }        <-- num_of_frames lines of python dicts (states)
      ...
      How many steps did John spend in the Garden?   <-- the NL question line
      answer:
      2
    """
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    sample_id = sample_dir.name

    qa_path = sample_dir / "qa.txt"
    lines = qa_path.read_text(encoding="utf-8").splitlines()

    # find block markers
    q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "question:"), -1)
    a_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "answer:"), -1)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        raise RuntimeError(f"Bad qa.txt format: {qa_path}")

    states = []
    question_text = None

    for ln in lines[q_idx + 1 : a_idx]:
        s = ln.strip()
        if not s:
            continue

        # state lines
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
            continue

        # THIS is the NL question (first non-dict line)
        question_text = s
        break

    if question_text is None:
        raise RuntimeError(f"Could not find NL question line in {qa_path}")

    # answer is first non-empty line after answer:
    answer_text = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if answer_text is None:
        raise RuntimeError(f"Could not find answer in {qa_path}")

    # frames: infer count from parsed states instead of using a global constant.
    frame_paths = [sample_dir / f"{i:03d}.png" for i in range(len(states))]
    missing = [p for p in frame_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame(s) for sample {sample_id}: {missing[0]}")
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    return sample_id, frames, question_text, states, answer_text

def print_top_k(logits, tokenizer, k=5):
    topk = torch.topk(logits, k=k)

    top_ids = topk.indices.tolist()

    probs = torch.softmax(logits, dim=-1)
    print(f"\nTop-{k} probs:")
    for rank, tok_id in enumerate(top_ids, start=1):
        print(f"{rank:>2}. id={tok_id:<6} p={probs[tok_id].item():.4f} token={tokenizer.decode([tok_id])!r}")


def iter_sample_dirs(data_root: Path) -> List[Path]:
    """
    Finds sample directories under data_root (directories that contain qa.txt).
    """
    out: List[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "qa.txt").exists():
            out.append(p)
    return out


def parse_layer_selection(raw: Optional[str], num_layers: int) -> List[int]:
    if raw is None:
        return list(range(num_layers))

    selected: set[int] = set()
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("--layers must not be empty when provided")

    for part in parts:
        if ":" not in part:
            try:
                selected.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid layer index in --layers: {part!r}") from exc
            continue

        fields = part.split(":")
        if len(fields) not in {2, 3}:
            raise ValueError(f"Invalid range in --layers: {part!r}. Expected start:end or start:end:step.")
        try:
            start = int(fields[0])
            end = int(fields[1])
            step = int(fields[2]) if len(fields) == 3 else 1
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --layers: {part!r}") from exc
        if step <= 0:
            raise ValueError(f"--layers step must be positive: {part!r}")
        if end <= start:
            raise ValueError(f"--layers range end must be greater than start: {part!r}")
        for layer_idx in range(start, end, step):
            selected.add(int(layer_idx))

    selected_layers = sorted(selected)
    invalid = [layer_idx for layer_idx in selected_layers if layer_idx < 0 or layer_idx >= num_layers]
    if invalid:
        raise ValueError(
            f"--layers contains out-of-bounds layers: {invalid}. Valid range is [0, {num_layers - 1}]."
        )
    return selected_layers


def resolve_seq_len_label(path_like: Any) -> Optional[str]:
    match = re.search(r"(seq_len_\d+)", str(path_like))
    return match.group(1) if match else None


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    match = _STEPS_IN_ROOM_RE.search(question_text)
    if not match:
        return None
    character = match.group(1).strip()
    room = match.group(2).strip()
    normalized_room = room[:1].upper() + room[1:].lower() if room else room
    return character, normalized_room


def parse_target_character_room_with_spans(
    question_text: str,
) -> Optional[Tuple[str, str, Tuple[int, int], Tuple[int, int]]]:
    match = _STEPS_IN_ROOM_RE.search(question_text)
    if not match:
        return None
    character = match.group(1).strip()
    room = match.group(2).strip()
    normalized_room = room[:1].upper() + room[1:].lower() if room else room
    return character, normalized_room, match.span(1), match.span(2)


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


def extract_characters_from_states(states: List[Dict[str, Any]]) -> List[str]:
    chars: set[str] = set()
    for state in states:
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        for room_chars in room_to_chars.values():
            chars.update(str(char) for char in room_chars)
    return sorted(chars)


def extract_rooms_from_states(states: List[Dict[str, Any]]) -> List[str]:
    rooms: set[str] = set()
    for state in states:
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        rooms.update(str(room) for room in room_to_chars.keys())
    return sorted(rooms)


def count_steps_for_character_room(states: List[Dict[str, Any]], character: str, room: str) -> int:
    target_room = room[:1].upper() + room[1:].lower() if room else room
    count = 0
    for state in states:
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        room_to_chars = rooms_to_room2chars(step_rooms)
        if character in room_to_chars.get(target_room, []):
            count += 1
    return count


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


def infer_corrupted_data_root(clean_data_root: Path) -> Path:
    clean_parts = list(clean_data_root.parts)
    for clean_name in ("mmred_images", "mmred"):
        if clean_name in clean_parts:
            idx = clean_parts.index(clean_name)
            new_parts = clean_parts[:]
            new_parts[idx] = "mmred_corrupted"
            return Path(*new_parts)
    raise ValueError("Could not infer corrupted root from --data_root. Please pass --corrupted_root explicitly.")


def resolve_corrupted_sample_dir(corrupted_data_root: Path, sample_id: str, frame_idx: int) -> Path:
    return corrupted_data_root / sample_id / f"corrupted_frame_{frame_idx}"


def format_centered_indices(n: int, cell_width: int = 9) -> str:
    return " ".join(str(i).center(cell_width) for i in range(n))


def format_centered_values(vals: List[float], cell_width: int = 9, precision: int = 4) -> str:
    return " ".join(f"{v:.{precision}f}".center(cell_width) for v in vals)


def format_corrupted_score_table(group_names: List[str], layer_rows: List[Tuple[int, List[float]]]) -> str:
    if not group_names or not layer_rows:
        return "<none>"
    cell_width = 12
    header = "layer".ljust(7) + " ".join(name[:cell_width].center(cell_width) for name in group_names)
    rows = [header]
    for layer_idx, row in layer_rows:
        values = " ".join(f"{value:.4f}".center(cell_width) for value in row)
        rows.append(f"{str(layer_idx).ljust(7)}{values}")
    return "\n".join(rows)


def write_metrics_json(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.json"
    output_path.write_text(json.dumps(sample_metrics, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_group_summary(group_names: List[str], sample_metrics: List[Dict[str, Any]]) -> None:
    counts_by_group = {group: 0 for group in group_names}
    for sample in sample_metrics:
        for group in sample["active_groups"]:
            if group in counts_by_group:
                counts_by_group[group] += 1
    print("Active-group coverage:")
    for group in group_names:
        print(f"  {group}: {counts_by_group.get(group, 0)} sample(s)")


def load_clean_score_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to parse clean-score cache at {path}: {exc}. Starting with empty cache.")
        return {}
    if not isinstance(payload, dict):
        print(f"[WARN] Invalid clean-score cache format at {path}. Expected JSON object; starting empty.")
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for sample_id, value in payload.items():
        if isinstance(sample_id, str) and isinstance(value, dict):
            cache[sample_id] = value
    return cache


def save_clean_score_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {sample_id: cache[sample_id] for sample_id in sorted(cache.keys())}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_or_compute_clean_answer_metrics(
    cache: Dict[str, Dict[str, Any]],
    sample_id: str,
    num_frames: int,
    answer_text: str,
    score_fn: Callable[[], Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    cache_entry = cache.get(sample_id)
    normalized_answer = str(answer_text).strip()
    if cache_entry is not None:
        cached_num_frames = int(cache_entry.get("num_frames", -1))
        cached_answer = str(cache_entry.get("answer_text", ""))
        if cached_num_frames == num_frames and cached_answer == normalized_answer:
            return {
                "clean_answer_score": float(cache_entry.get("clean_answer_score", float("-inf"))),
                "clean_correct_prob": float(cache_entry.get("clean_correct_prob", 0.0)),
                "clean_top1_correct": bool(cache_entry.get("clean_top1_correct", False)),
                "best_answer_text": str(cache_entry.get("best_answer_text", "")),
            }, False

    candidate_scores = score_fn()
    metrics = {
        "clean_answer_score": float(candidate_scores["scores_by_answer"].get(normalized_answer, float("-inf"))),
        "clean_correct_prob": float(candidate_scores["probs_by_answer"].get(normalized_answer, 0.0)),
        "best_answer_text": str(candidate_scores["best_answer_text"]),
    }
    metrics["clean_top1_correct"] = (metrics["best_answer_text"] == normalized_answer)
    cache[sample_id] = {
        "num_frames": num_frames,
        "answer_text": normalized_answer,
        **metrics,
    }
    return metrics, True


def persist_clean_score_cache(
    path: Path,
    cache: Dict[str, Dict[str, Any]],
    cache_updates: int,
) -> str:
    if cache_updates > 0:
        save_clean_score_cache(path, cache)
        return f"Updated clean-score cache at {path} ({cache_updates} new/changed entries)."
    if not path.exists():
        save_clean_score_cache(path, cache)
        return f"Wrote empty clean-score cache to: {path}"
    return f"No clean-score cache updates. Reused existing cache at: {path}"


def build_group_patch_batches(
    groups_payload: List[Dict[str, Any]],
    batch_size: int,
    clean_inputs: Dict[str, Any],
    answer_token_ids: List[int],
    repeat_inputs_for_batch: Callable[[Dict[str, Any], int], Dict[str, Any]],
    concatenate_inputs_for_batch: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    append_answer_tokens_for_scoring: Callable[[Dict[str, Any], List[int]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not groups_payload:
        return []

    chunk_size = min(batch_size, len(groups_payload))
    group_chunks = [
        groups_payload[start:start + chunk_size]
        for start in range(0, len(groups_payload), chunk_size)
    ]

    chunk_data: List[Dict[str, Any]] = []
    for group_chunk in group_chunks:
        chunk_len = len(group_chunk)
        repeated_clean_inputs = repeat_inputs_for_batch(clean_inputs, batch_size=chunk_len)
        clean_scoring_inputs = append_answer_tokens_for_scoring(repeated_clean_inputs, answer_token_ids)
        control_inputs_batch = concatenate_inputs_for_batch(
            [group_entry["control_inputs"] for group_entry in group_chunk]
        )
        control_scoring_inputs = append_answer_tokens_for_scoring(control_inputs_batch, answer_token_ids)
        chunk_data.append({
            "groups": group_chunk,
            "clean_scoring_inputs": clean_scoring_inputs,
            "control_scoring_inputs": control_scoring_inputs,
        })
    return chunk_data


def run_group_patch_layer_sweep(
    selected_layers: List[int],
    groups_payload: List[Dict[str, Any]],
    chunk_data: List[Dict[str, Any]],
    clean_answer_score: float,
    lm: Any,
    layers: Any,
    prompt_len: int,
    answer_token_ids: List[int],
    run_layer_patch: Callable[..., Any],
    normalize_to_probabilities: Callable[[List[float]], List[float]],
    entropy_from_probabilities: Callable[[List[float]], float],
    normalize_entropy: Callable[[float, int], float],
    logger: Callable[[str], None] = print,
    include_signed_delta: bool = True,
    normalize_by_token_count: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, List[float]]]]:
    per_layer_metrics: List[Dict[str, Any]] = []
    all_layer_corrupted_rows: List[Tuple[int, List[float]]] = []
    layer_group_order = [group["name"] for group in groups_payload]

    for layer_idx in selected_layers:
        per_group_corrupted_score: Dict[str, float] = {}
        per_group_signed_delta: Dict[str, float] = {}
        per_group_importance: Dict[str, float] = {}
        per_group_normalized_importance: Dict[str, float] = {}

        for chunk_idx, packed in enumerate(chunk_data, start=1):
            group_chunk = packed["groups"]
            clean_positions_by_batch = [group["clean_positions"] for group in group_chunk]
            control_positions_by_batch = [group["control_positions"] for group in group_chunk]
            try:
                corrupted_scores = run_layer_patch(
                    lm=lm,
                    layers=layers,
                    clean_batched_scoring_inputs=packed["clean_scoring_inputs"],
                    control_batched_scoring_inputs=packed["control_scoring_inputs"],
                    layer_idx=layer_idx,
                    clean_token_positions_by_batch=clean_positions_by_batch,
                    control_token_positions_by_batch=control_positions_by_batch,
                    prompt_len=prompt_len,
                    answer_token_ids=answer_token_ids,
                )
            except Exception as exc:
                logger(
                    f"  layer={layer_idx} failed batched corruption forward "
                    f"(chunk {chunk_idx}/{len(chunk_data)}, {exc}); using clean score for this chunk"
                )
                for group in group_chunk:
                    group_name = group["name"]
                    per_group_corrupted_score[group_name] = clean_answer_score
                    per_group_signed_delta[group_name] = 0.0
                    per_group_importance[group_name] = 0.0
                    if normalize_by_token_count:
                        per_group_normalized_importance[group_name] = 0.0
                continue

            for batch_idx, group in enumerate(group_chunk):
                group_name = group["name"]
                corrupted_score = float(corrupted_scores[batch_idx].item())
                signed_delta = float(clean_answer_score - corrupted_score)
                importance = max(signed_delta, 0.0)
                per_group_corrupted_score[group_name] = corrupted_score
                per_group_signed_delta[group_name] = signed_delta
                per_group_importance[group_name] = importance
                if normalize_by_token_count:
                    token_count = max(1, len(group["clean_positions"]))
                    per_group_normalized_importance[group_name] = importance / float(token_count)

        corrupted_score_row = [per_group_corrupted_score.get(group_name, clean_answer_score) for group_name in layer_group_order]
        signed_delta_row = [per_group_signed_delta.get(group_name, 0.0) for group_name in layer_group_order]
        importance_row = [per_group_importance.get(group_name, 0.0) for group_name in layer_group_order]
        all_layer_corrupted_rows.append((layer_idx, list(corrupted_score_row)))

        total_importance = float(sum(importance_row))
        if total_importance > 0.0:
            probs = normalize_to_probabilities(importance_row)
            entropy_value = normalize_entropy(
                entropy_from_probabilities(probs),
                len(layer_group_order),
            )
        else:
            probs = [0.0 for _ in importance_row]
            entropy_value = None

        layer_metrics: Dict[str, Any] = {
            "layer": layer_idx,
            "groups": list(layer_group_order),
            "corrupted_score": corrupted_score_row,
            "r": importance_row,
            "p": probs,
            "entropy": entropy_value,
            "total_importance": total_importance,
        }
        if include_signed_delta:
            layer_metrics["signed_delta"] = signed_delta_row
        if normalize_by_token_count:
            layer_metrics["r_normalized"] = [
                per_group_normalized_importance.get(group_name, 0.0) for group_name in layer_group_order
            ]
        per_layer_metrics.append(layer_metrics)

    return per_layer_metrics, all_layer_corrupted_rows


def write_sample_metrics(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    def _fmt_float_list(vals: List[float]) -> str:
        return "[" + ", ".join(f"{v:.8f}" for v in vals) + "]"

    lines: List[str] = []
    for sm in sample_metrics:
        lines.append(f"sample_id={sm['sample_id']}")
        for lmtr in sm["layer_metrics"]["layers"]:
            lines.append(
                f"layer={lmtr['layer']} "
                f"r={_fmt_float_list(lmtr['r'])} "
                f"p={_fmt_float_list(lmtr['p'])} "
                f"H_norm={lmtr['entropy']:.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


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


def plot_entropy_summary(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 0,
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot mean/median normalized entropy H(l)/N_evidence across layers
    with 95% bootstrap CIs.
    """
    entropy_by_layer: Dict[int, List[float]] = {}
    for sm in sample_metrics:
        for lmtr in sm["layer_metrics"]["layers"]:
            l = int(lmtr["layer"])
            entropy_by_layer.setdefault(l, []).append(float(lmtr["entropy"]))

    if not entropy_by_layer:
        return None

    rng = random.Random(seed)
    layers = sorted(entropy_by_layer.keys())

    means: List[float] = []
    medians: List[float] = []
    mean_lo: List[float] = []
    mean_hi: List[float] = []
    med_lo: List[float] = []
    med_hi: List[float] = []

    for l in layers:
        vals = entropy_by_layer[l]
        n = len(vals)
        sorted_vals = sorted(vals)

        mean = sum(vals) / n
        median = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])

        boot_mean: List[float] = []
        boot_median: List[float] = []
        for _ in range(n_bootstrap):
            sample = [vals[rng.randrange(n)] for _ in range(n)]
            s_sorted = sorted(sample)
            b_mean = sum(sample) / n
            b_median = s_sorted[n // 2] if n % 2 == 1 else 0.5 * (s_sorted[n // 2 - 1] + s_sorted[n // 2])
            boot_mean.append(b_mean)
            boot_median.append(b_median)

        boot_mean.sort()
        boot_median.sort()
        lo_idx = int(0.025 * (n_bootstrap - 1))
        hi_idx = int(0.975 * (n_bootstrap - 1))

        means.append(mean)
        medians.append(median)
        mean_lo.append(boot_mean[lo_idx])
        mean_hi.append(boot_mean[hi_idx])
        med_lo.append(boot_median[lo_idx])
        med_hi.append(boot_median[hi_idx])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, means, color="#1f77b4", linewidth=2.2, label="Mean H(l)/N")
    ax.fill_between(layers, mean_lo, mean_hi, color="#1f77b4", alpha=0.2, label="Mean 95% CI")
    ax.plot(layers, medians, color="#d62728", linewidth=2.2, label="Median H(l)/N")
    ax.fill_between(layers, med_lo, med_hi, color="#d62728", alpha=0.2, label="Median 95% CI")

    title = "Normalized Entropy by Layer (H/N)"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer l", fontsize=11)
    ax.set_ylabel("H(l)/N", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    # Avoid label collisions on wide/deep models by showing a spaced subset of layers.
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


def _bootstrap_center_and_ci(
    values: List[float],
    reducer,
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    n = len(values)
    center = reducer(values)
    if n <= 1:
        return center, center, center

    boot_values: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_values.append(reducer(sample))

    boot_values.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    return center, boot_values[lo_idx], boot_values[hi_idx]


def _plot_attention_importance_subplot(
    ax: Any,
    summary: Dict[str, List[float]],
    stat_name: str,
) -> None:
    layers = summary["layers"]
    if not layers:
        raise RuntimeError(f"No data available for {stat_name} subplot.")

    ax.plot(layers, summary["importance"], color="#1f77b4", linewidth=2.2, label=f"{stat_name} importance")
    ax.fill_between(
        layers,
        summary["importance_lo"],
        summary["importance_hi"],
        color="#1f77b4",
        alpha=0.18,
        label=f"{stat_name} importance 95% CI",
    )
    ax.plot(layers, summary["attention"], color="#d62728", linewidth=2.2, label=f"{stat_name} attention ratio")
    ax.fill_between(
        layers,
        summary["attention_lo"],
        summary["attention_hi"],
        color="#d62728",
        alpha=0.18,
        label=f"{stat_name} attention 95% CI",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title(f"Per-layer {stat_name.lower()}: importance vs attention-share")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)


def plot_attention_importance_summary(
    importance_by_layer: Dict[int, List[float]],
    attention_by_layer: Dict[int, List[float]],
    output_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    candidate_layers = sorted(set(importance_by_layer) & set(attention_by_layer))
    if not candidate_layers:
        return None

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals)

    def _median(vals: List[float]) -> float:
        ordered = sorted(vals)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])

    def _build_summary(reducer, rng_seed: int) -> Dict[str, List[float]]:
        rng = random.Random(rng_seed)
        summary: Dict[str, List[float]] = {
            "layers": [],
            "importance": [],
            "importance_lo": [],
            "importance_hi": [],
            "attention": [],
            "attention_lo": [],
            "attention_hi": [],
        }
        for layer_idx in candidate_layers:
            importance_values = importance_by_layer.get(layer_idx, [])
            attention_values = attention_by_layer.get(layer_idx, [])
            if not importance_values or not attention_values:
                continue

            imp_center, imp_lo, imp_hi = _bootstrap_center_and_ci(
                importance_values,
                reducer,
                n_bootstrap,
                rng,
            )
            att_center, att_lo, att_hi = _bootstrap_center_and_ci(
                attention_values,
                reducer,
                n_bootstrap,
                rng,
            )
            summary["layers"].append(layer_idx)
            summary["importance"].append(imp_center)
            summary["importance_lo"].append(imp_lo)
            summary["importance_hi"].append(imp_hi)
            summary["attention"].append(att_center)
            summary["attention_lo"].append(att_lo)
            summary["attention_hi"].append(att_hi)
        return summary

    mean_summary = _build_summary(_mean, seed)
    median_summary = _build_summary(_median, seed + 1)
    if not mean_summary["layers"] or not median_summary["layers"]:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=140, sharey=False)
    _plot_attention_importance_subplot(axes[0], mean_summary, "Mean")
    _plot_attention_importance_subplot(axes[1], median_summary, "Median")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
