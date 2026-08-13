import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)


def load_mmred_sample(sample_dir: Path):
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    sample_id = sample_dir.name

    qa_path = sample_dir / "qa.txt"
    lines = qa_path.read_text(encoding="utf-8").splitlines()

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
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
            continue
        question_text = s
        break

    if question_text is None:
        raise RuntimeError(f"Could not find NL question line in {qa_path}")

    answer_text = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if answer_text is None:
        raise RuntimeError(f"Could not find answer in {qa_path}")

    frame_paths = [sample_dir / f"{i:03d}.png" for i in range(len(states))]
    missing = [p for p in frame_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame(s) for sample {sample_id}: {missing[0]}")
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    return sample_id, frames, question_text, states, answer_text


def iter_sample_dirs(data_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "qa.txt").exists():
            out.append(p)
    return out


def iter_sample_dirs_shuffled(data_root: Path, seed: int) -> List[Path]:
    """Stratified deterministic shuffle of iter_sample_dirs (2026-07-18, E1 full-prior fix).

    MMRED dirs are name-sorted by class tag (seq8_e<g>_* / *_K<g>_*), so any LIMIT < full
    yields a truncated gold prior. This groups dirs by the class tag in the name, shuffles
    within groups (seeded), then round-robins across groups so EVERY prefix is class-balanced.
    Falls back to a plain seeded shuffle when no tag is found.
    """
    import random

    dirs = iter_sample_dirs(data_root)
    rng = random.Random(seed)
    groups: Dict[Optional[str], List[Path]] = {}
    for d in dirs:
        m = re.search(r"_(?:K|e)(\d+)_", d.name)
        groups.setdefault(m.group(1) if m else None, []).append(d)
    if len(groups) <= 1:
        out = list(dirs)
        rng.shuffle(out)
        return out
    keys = sorted(groups, key=lambda k: (k is None, int(k) if k is not None else -1))
    for k in keys:
        rng.shuffle(groups[k])
    out = []
    for i in range(max(len(g) for g in groups.values())):
        for k in keys:
            if i < len(groups[k]):
                out.append(groups[k][i])
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


def format_runtime(elapsed_seconds: float) -> str:
    elapsed_seconds = float(elapsed_seconds)
    elapsed_h = int(elapsed_seconds // 3600)
    elapsed_m = int((elapsed_seconds % 3600) // 60)
    elapsed_s = elapsed_seconds % 60.0
    return f"Total runtime: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:05.2f} ({elapsed_seconds:.2f}s)"


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


def build_composite_corrupted_frames(
    sample_id: str,
    clean_frames: List[Any],
    evidence_frame_indices: List[int],
    corrupted_data_root: Path,
) -> Tuple[Optional[List[Any]], Dict[str, str]]:
    corrupted_frames = list(clean_frames)
    issues: Dict[str, str] = {}
    for frame_idx in evidence_frame_indices:
        corrupted_sample_dir = resolve_corrupted_sample_dir(corrupted_data_root, sample_id, int(frame_idx))
        if not corrupted_sample_dir.is_dir():
            issues[f"frame_{frame_idx}"] = "missing_corrupted_sample_dir"
            return None, issues
        try:
            _, corrupted_sample_frames, _, _, _ = load_mmred_sample(corrupted_sample_dir)
        except Exception as exc:
            issues[f"frame_{frame_idx}"] = f"load_failure({exc})"
            return None, issues
        if len(corrupted_sample_frames) != len(clean_frames):
            issues[f"frame_{frame_idx}"] = (
                f"frame_count_mismatch(clean={len(clean_frames)},corrupted={len(corrupted_sample_frames)})"
            )
            return None, issues
        corrupted_frames[int(frame_idx)] = corrupted_sample_frames[int(frame_idx)]
    return corrupted_frames, issues
