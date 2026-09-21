#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# RECOVERED SOURCE (2026-07-29, serious-refactor phase 0).
# Original path: evaluations/scripts/generate_mmred_park_dataset.py (deleted; survived
# only as evaluations/scripts/__pycache__/generate_mmred_park_dataset.cpython-39.pyc).
# This is the generator that produced data/mmred_images_park (+ data/mmred_park
# metadata), i.e. THE main thesis dataset (seed=0, samples_per_evidence_count=100,
# seq_lens 1-8, created 2026-05-05T18:35 UTC per data/mmred_images_park/metadata.json).
#
# Recovery route: dangling (unreachable) git blob a180fcb83fd56b8fe9294bca9781940fb55d301c,
# found by content-grepping every object in .git — the file was `git add`ed once but
# never committed on any ref.
# Verification: compiled with .venv Python 3.9.21 and structurally compared against the
# orphan pyc: co_code, co_consts, co_names, co_varnames, co_flags AND the line-number
# table (co_lnotab) are IDENTICAL for the module and every nested function
# => byte-exact source recovery, not a decompilation.
# Corroboration: the same source appears in Codex transcripts
# ~/.codex/sessions/2026/05/05/rollout-2026-05-05T18-25-43-*.jsonl, and
# data/mmred_images_park/metadata.json matches root_metadata() field-for-field
# (tokenization_sanity ran with model_id Qwen/Qwen2.5-VL-32B-Instruct).
#
# Everything below this comment block is byte-identical to the recovered blob (only
# this header was inserted, which shifts line numbers vs. the original pyc).
# NOTE: the sys.path setup below takes parents[2] as the repo root; datasets/mmred/
# sits at the same depth as the original evaluations/scripts/, so the script works
# unmodified from this location.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset
from transformers import AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MMRED_RENDER_DIR = _REPO_ROOT / "datasets" / "mmred"
if str(_MMRED_RENDER_DIR) not in sys.path:
    sys.path.insert(0, str(_MMRED_RENDER_DIR))

import render_mmred  # noqa: E402
from models.model import DEFAULT_MODEL_ID  # noqa: E402


DEFAULT_ROOMS: Tuple[str, ...] = ("Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park")
ROOM_LAYOUT: Tuple[Tuple[str, str, str], Tuple[str, str, str]] = (
    ("Kitchen", "Bathroom", "Garden"),
    ("Office", "Bedroom", "Park"),
)
DEFAULT_SEQ_LENS: Tuple[int, ...] = tuple(range(1, 9))
DEFAULT_SAMPLES_PER_EVIDENCE_COUNT = 100
DEFAULT_SPLIT = "all_uniform"
CHARACTERS: Tuple[str, ...] = tuple(render_mmred.CHAR_COLORS.keys())
HARD_DISTRACTOR_TYPES: Tuple[str, ...] = (
    "target_character_wrong_room",
    "wrong_character_target_room",
    "mixed_non_evidence",
    "easy_non_evidence",
)


@dataclass(frozen=True)
class ParkSample:
    sample_id: str
    seq_len: int
    split: str
    evidence_count: int
    target_character: str
    target_room: str
    question: str
    answer: int
    states: Tuple[Dict[str, Any], ...]
    evidence_frame_indices: Tuple[int, ...]
    non_evidence_frame_indices: Tuple[int, ...]
    distractor_types_by_frame: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Park replacement version of the rendered MMReD-style evidence-count "
            "dataset. Park replaces Hallway in the existing six-room visual layout."
        )
    )
    parser.add_argument("--model-id", "--model_id", dest="model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-image-root",
        "--output_image_root",
        dest="output_image_root",
        type=Path,
        default=Path("data/mmred_images_park"),
    )
    parser.add_argument(
        "--output-metadata-root",
        "--output_metadata_root",
        dest="output_metadata_root",
        type=Path,
        default=Path("data/mmred_park"),
    )
    parser.add_argument(
        "--seq-lens",
        "--seq_lens",
        dest="seq_lens",
        nargs="+",
        default=[str(value) for value in DEFAULT_SEQ_LENS],
        help="Sequence lengths. Accepts spaces or comma-separated values.",
    )
    parser.add_argument(
        "--samples-per-evidence-count",
        "--samples_per_evidence_count",
        dest="samples_per_evidence_count",
        type=int,
        default=DEFAULT_SAMPLES_PER_EVIDENCE_COUNT,
    )
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--rooms",
        nargs="+",
        default=list(DEFAULT_ROOMS),
        help="Room names in renderer order. Must be Kitchen Bathroom Garden Office Bedroom Park.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output roots. By default existing Park outputs are protected.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Small debug dataset: seq_len 1 2 and at most 2 samples per evidence-count bucket.",
    )
    args = parser.parse_args()

    if int(args.samples_per_evidence_count) <= 0:
        raise ValueError("--samples-per-evidence-count must be positive")
    if args.dry_run:
        args.seq_lens = ["1", "2"]
        args.samples_per_evidence_count = min(int(args.samples_per_evidence_count), 2)

    rooms = [str(room) for room in args.rooms]
    if rooms != list(DEFAULT_ROOMS):
        raise ValueError(
            "--rooms must be exactly: Kitchen Bathroom Garden Office Bedroom Park. "
            f"Got: {rooms!r}"
        )
    if "Hallway" in rooms:
        raise ValueError("Hallway must not appear in --rooms; use Park.")
    return args


def parse_positive_int_values(raw_values: Sequence[Any], *, arg_name: str) -> List[int]:
    values: List[int] = []
    for raw_value in raw_values:
        for part in str(raw_value).replace(",", " ").split():
            if not part:
                continue
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {arg_name}: {part!r}") from exc
            if value <= 0:
                raise ValueError(f"{arg_name} values must be positive, got {value}")
            values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must not be empty")
    return sorted(dict.fromkeys(values))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_qa(path: Path, *, sample: ParkSample) -> None:
    state_lines = [repr({"step_id": state["step_id"], "rooms": state["rooms"]}) for state in sample.states]
    question_block = "\n".join(state_lines + [sample.question])
    text = (
        f"qid: {sample.sample_id}\n"
        "qtype: steps_in_room\n"
        "atype: integer\n"
        f"seq_len: {int(sample.seq_len)}\n"
        "question:\n"
        f"{question_block}\n"
        "answer:\n"
        f"{int(sample.answer)}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_output_root(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output root already exists: {path}. Pass --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def empty_occupancy(rooms: Sequence[str]) -> Dict[str, List[str]]:
    return {str(room): [] for room in rooms}


def normalize_occupancy(room_to_chars: Dict[str, List[str]], rooms: Sequence[str]) -> Dict[str, List[str]]:
    normalized = empty_occupancy(rooms)
    for room, chars in room_to_chars.items():
        room_name = str(room)
        normalized.setdefault(room_name, [])
        normalized[room_name].extend(str(character) for character in chars)
    return {str(room): sorted(dict.fromkeys(normalized.get(str(room), []))) for room in rooms}


def invert_occupancy(room_to_chars: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for room, chars in room_to_chars.items():
        for character in chars:
            out[str(character)] = str(room)
    return out


def has_target_evidence(
    room_to_chars: Dict[str, List[str]],
    *,
    target_character: str,
    target_room: str,
) -> bool:
    return str(target_character) in room_to_chars.get(str(target_room), [])


def assert_evidence_status(
    room_to_chars: Dict[str, List[str]],
    *,
    target_character: str,
    target_room: str,
    is_evidence: bool,
) -> None:
    target_in_room = has_target_evidence(
        room_to_chars,
        target_character=target_character,
        target_room=target_room,
    )
    if bool(is_evidence) and not target_in_room:
        raise AssertionError("Evidence frame must contain target_character in target_room")
    if not bool(is_evidence) and target_in_room:
        raise AssertionError("Non-evidence frame must not contain target_character in target_room")


def random_wrong_room(rooms: Sequence[str], target_room: str, rng: random.Random) -> str:
    choices = [str(room) for room in rooms if str(room) != str(target_room)]
    if not choices:
        raise ValueError("Need at least one non-target room")
    return str(rng.choice(choices))


def random_room_excluding(rooms: Sequence[str], excluded: Sequence[str], rng: random.Random) -> str:
    excluded_set = {str(room) for room in excluded}
    choices = [str(room) for room in rooms if str(room) not in excluded_set]
    if not choices:
        choices = [str(room) for room in rooms]
    return str(rng.choice(choices))


def generate_evidence_occupancy(
    *,
    target_character: str,
    target_room: str,
    rooms: Sequence[str],
    rng: random.Random,
) -> Dict[str, List[str]]:
    room_to_chars = empty_occupancy(rooms)
    room_to_chars[str(target_room)].append(str(target_character))
    for character in CHARACTERS:
        if str(character) == str(target_character):
            continue
        room_to_chars[str(rng.choice(list(rooms)))].append(str(character))
    normalized = normalize_occupancy(room_to_chars, rooms)
    assert_evidence_status(
        normalized,
        target_character=target_character,
        target_room=target_room,
        is_evidence=True,
    )
    return normalized


def generate_non_evidence_occupancy(
    *,
    distractor_type: str,
    target_character: str,
    target_room: str,
    rooms: Sequence[str],
    rng: random.Random,
) -> Dict[str, List[str]]:
    other_characters = [str(character) for character in CHARACTERS if str(character) != str(target_character)]
    if not other_characters:
        raise ValueError("Need at least one non-target character")

    room_to_chars = empty_occupancy(rooms)
    if distractor_type == "target_character_wrong_room":
        room_to_chars[random_wrong_room(rooms, target_room, rng)].append(str(target_character))
        for character in other_characters:
            room = random_room_excluding(rooms, [target_room], rng)
            room_to_chars[room].append(character)
    elif distractor_type == "wrong_character_target_room":
        decoy = str(rng.choice(other_characters))
        room_to_chars[str(target_room)].append(decoy)
        for character in other_characters:
            if character == decoy:
                continue
            room_to_chars[str(rng.choice(list(rooms)))].append(character)
    elif distractor_type == "mixed_non_evidence":
        room_to_chars[random_wrong_room(rooms, target_room, rng)].append(str(target_character))
        decoy = str(rng.choice(other_characters))
        room_to_chars[str(target_room)].append(decoy)
        for character in other_characters:
            if character == decoy:
                continue
            room_to_chars[str(rng.choice(list(rooms)))].append(character)
    elif distractor_type == "easy_non_evidence":
        room_to_chars[random_wrong_room(rooms, target_room, rng)].append(str(target_character))
        for character in other_characters:
            room = random_room_excluding(rooms, [target_room], rng)
            room_to_chars[room].append(character)
    else:
        raise ValueError(f"Unknown distractor_type={distractor_type!r}")

    normalized = normalize_occupancy(room_to_chars, rooms)
    assert_evidence_status(
        normalized,
        target_character=target_character,
        target_room=target_room,
        is_evidence=False,
    )
    return normalized


def next_balanced_target_room(rooms: Sequence[str], room_counter: Counter[str], rng: random.Random) -> str:
    min_count = min(room_counter[str(room)] for room in rooms)
    least_used = [str(room) for room in rooms if room_counter[str(room)] == min_count]
    selected = str(rng.choice(least_used))
    room_counter[selected] += 1
    return selected


def next_balanced_character(character_counter: Counter[str], rng: random.Random) -> str:
    min_count = min(character_counter[str(character)] for character in CHARACTERS)
    least_used = [str(character) for character in CHARACTERS if character_counter[str(character)] == min_count]
    selected = str(rng.choice(least_used))
    character_counter[selected] += 1
    return selected


def choose_distractor_type(sample_index: int, non_evidence_index: int, rng: random.Random) -> str:
    offset = rng.randrange(len(HARD_DISTRACTOR_TYPES))
    return HARD_DISTRACTOR_TYPES[
        (int(sample_index) + int(non_evidence_index) + int(offset)) % len(HARD_DISTRACTOR_TYPES)
    ]


def make_state(
    *,
    step_id: int,
    frame_index: int,
    rooms_for_frame: Dict[str, List[str]],
    is_evidence: bool,
    distractor_type: str,
    target_character: str,
    target_room: str,
    rooms: Sequence[str],
) -> Dict[str, Any]:
    normalized = normalize_occupancy(rooms_for_frame, rooms)
    assert_evidence_status(
        normalized,
        target_character=target_character,
        target_room=target_room,
        is_evidence=is_evidence,
    )
    return {
        "step_id": int(step_id),
        "frame_index": int(frame_index),
        "rooms": normalized,
        "character_rooms": invert_occupancy(normalized),
        "is_evidence": bool(is_evidence),
        "distractor_type": str(distractor_type),
        "target_character": str(target_character),
        "target_room": str(target_room),
    }


def build_sample(
    *,
    seq_len: int,
    split: str,
    evidence_count: int,
    sample_index: int,
    global_index: int,
    target_character: str,
    target_room: str,
    rooms: Sequence[str],
    rng: random.Random,
) -> ParkSample:
    records: List[Dict[str, Any]] = []
    for _ in range(int(evidence_count)):
        records.append(
            {
                "is_evidence": True,
                "distractor_type": "evidence",
                "rooms": generate_evidence_occupancy(
                    target_character=target_character,
                    target_room=target_room,
                    rooms=rooms,
                    rng=rng,
                ),
            }
        )

    for non_evidence_index in range(int(seq_len) - int(evidence_count)):
        distractor_type = choose_distractor_type(sample_index, non_evidence_index, rng)
        records.append(
            {
                "is_evidence": False,
                "distractor_type": distractor_type,
                "rooms": generate_non_evidence_occupancy(
                    distractor_type=distractor_type,
                    target_character=target_character,
                    target_room=target_room,
                    rooms=rooms,
                    rng=rng,
                ),
            }
        )

    rng.shuffle(records)
    states: List[Dict[str, Any]] = []
    evidence_frame_indices: List[int] = []
    non_evidence_frame_indices: List[int] = []
    distractor_types_by_frame: List[str] = []
    for frame_index, record in enumerate(records):
        is_evidence = bool(record["is_evidence"])
        if is_evidence:
            evidence_frame_indices.append(int(frame_index))
        else:
            non_evidence_frame_indices.append(int(frame_index))
        distractor_type = str(record["distractor_type"])
        distractor_types_by_frame.append(distractor_type)
        states.append(
            make_state(
                step_id=int(frame_index) + 1,
                frame_index=int(frame_index),
                rooms_for_frame=record["rooms"],
                is_evidence=is_evidence,
                distractor_type=distractor_type,
                target_character=target_character,
                target_room=target_room,
                rooms=rooms,
            )
        )

    observed_count = sum(
        1
        for state in states
        if has_target_evidence(
            state["rooms"],
            target_character=target_character,
            target_room=target_room,
        )
    )
    if observed_count != int(evidence_count):
        raise AssertionError(f"Expected {evidence_count} evidence frames, observed {observed_count}")

    sample_id = f"seq{int(seq_len)}_e{int(evidence_count)}_sample_{int(global_index):06d}"
    question = f"How many steps did {target_character} spend in the {target_room}?"
    return ParkSample(
        sample_id=sample_id,
        seq_len=int(seq_len),
        split=str(split),
        evidence_count=int(evidence_count),
        target_character=str(target_character),
        target_room=str(target_room),
        question=question,
        answer=int(evidence_count),
        states=tuple(states),
        evidence_frame_indices=tuple(evidence_frame_indices),
        non_evidence_frame_indices=tuple(non_evidence_frame_indices),
        distractor_types_by_frame=tuple(distractor_types_by_frame),
    )


def render_sample(sample: ParkSample, sample_dir: Path, rooms: Sequence[str]) -> List[str]:
    frame_paths: List[str] = []
    for frame_index, state in enumerate(sample.states):
        frame_name = f"frame_{int(frame_index):03d}.png"
        out_path = sample_dir / frame_name
        render_mmred.render_frame(state["rooms"], int(state["step_id"]), str(out_path))
        legacy_path = sample_dir / f"{int(frame_index):03d}.png"
        if legacy_path != out_path:
            shutil.copyfile(out_path, legacy_path)
        frame_paths.append(frame_name)
    return frame_paths


def sample_metadata(sample: ParkSample, *, frame_paths: Sequence[str], rooms: Sequence[str]) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "seq_len": int(sample.seq_len),
        "split": sample.split,
        "evidence_count": int(sample.evidence_count),
        "target_character": sample.target_character,
        "target_room": sample.target_room,
        "question": sample.question,
        "answer": int(sample.answer),
        "evidence_frame_indices": [int(value) for value in sample.evidence_frame_indices],
        "non_evidence_frame_indices": [int(value) for value in sample.non_evidence_frame_indices],
        "rooms": [str(room) for room in rooms],
        "frame_paths": [str(path) for path in frame_paths],
        "legacy_frame_paths": [f"{idx:03d}.png" for idx in range(int(sample.seq_len))],
        "distractor_types_by_frame": [str(value) for value in sample.distractor_types_by_frame],
        "target_character_actual_rooms_by_frame": [
            state.get("character_rooms", {}).get(sample.target_character) for state in sample.states
        ],
    }


def metadata_record(sample: ParkSample, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": "\n".join(
            [repr({"step_id": state["step_id"], "rooms": state["rooms"]}) for state in sample.states]
            + [sample.question]
        ),
        "seq_len": int(sample.seq_len),
        "answer": int(sample.answer),
        "qid": sample.sample_id,
        "qtype": "steps_in_room",
        "atype": "integer",
        "sample_id": sample.sample_id,
        "target_character": sample.target_character,
        "target_room": sample.target_room,
        "evidence_indices": [int(value) for value in sample.evidence_frame_indices],
        "evidence_frame_indices": [int(value) for value in sample.evidence_frame_indices],
        "non_evidence_frame_indices": [int(value) for value in sample.non_evidence_frame_indices],
        "evidence_count": int(sample.evidence_count),
        "rooms": list(metadata["rooms"]),
        "frame_paths": list(metadata["frame_paths"]),
        "distractor_types_by_frame": list(metadata["distractor_types_by_frame"]),
        "states": [dict(state) for state in sample.states],
    }


def save_dataset(records: Sequence[Dict[str, Any]], output_dir: Path, features: Optional[Any]) -> None:
    if records:
        dataset = Dataset.from_list(list(records))
    elif features is not None:
        dataset = Dataset.from_dict({name: [] for name in features}, features=features)
    else:
        dataset = Dataset.from_list([])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir), num_shards=1)


def load_tokenizer(model_id: str) -> Any:
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    return processor.tokenizer


def leading_space_token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(f" {text}", add_special_tokens=False)["input_ids"])


def tokenization_sanity(*, model_id: str, rooms: Sequence[str]) -> Dict[str, Any]:
    tokenizer = load_tokenizer(model_id)
    counts = {str(room): leading_space_token_count(tokenizer, str(room)) for room in rooms}
    print("Tokenization sanity with leading space:")
    for room in rooms:
        print(f'  " {room}": {int(counts[str(room)])} token(s)')
    bad_rooms = {room: count for room, count in counts.items() if int(count) != 1}
    if bad_rooms:
        print(
            "[WARN] One or more target rooms are not exactly one token with leading space: "
            + ", ".join(f"{room}={count}" for room, count in sorted(bad_rooms.items()))
        )
    return {
        "model_id": str(model_id),
        "room_token_counts_with_leading_space": counts,
        "bad_target_rooms": bad_rooms,
    }


def root_metadata(
    *,
    rooms: Sequence[str],
    seq_lens: Sequence[int],
    samples_per_evidence_count: int,
    split: str,
    seed: int,
    tokenization: Dict[str, Any],
    convention_notes: str,
    dry_run: bool,
) -> Dict[str, Any]:
    return {
        "rooms": [str(room) for room in rooms],
        "room_layout": [list(row) for row in ROOM_LAYOUT],
        "excluded_rooms": ["Hallway"],
        "replacement_room": "Park",
        "seq_lens": [int(value) for value in seq_lens],
        "samples_per_evidence_count": int(samples_per_evidence_count),
        "split": str(split),
        "seed": int(seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "characters": [str(character) for character in CHARACTERS],
        "distractor_types": [str(value) for value in HARD_DISTRACTOR_TYPES],
        "tokenization_sanity": tokenization,
        "convention": convention_notes,
    }


def generate_dataset(args: argparse.Namespace) -> None:
    image_root = Path(args.output_image_root)
    metadata_root = Path(args.output_metadata_root)
    rooms = [str(room) for room in args.rooms]
    seq_lens = parse_positive_int_values(args.seq_lens, arg_name="--seq-lens")
    split = str(args.split)
    samples_per_evidence_count = int(args.samples_per_evidence_count)
    rng = random.Random(int(args.seed))

    prepare_output_root(image_root, force=bool(args.force))
    prepare_output_root(metadata_root, force=bool(args.force))

    # Local renderer override: preserve the existing MMReD frame style while replacing Hallway.
    render_mmred.ROOMS = list(rooms)

    tokenization = tokenization_sanity(model_id=str(args.model_id), rooms=rooms)
    convention_notes = (
        "Image samples are duplicated under seq_len_k/all_uniform and "
        "seq_len_k/by_evidence_count/exact_j/all_uniform. Each sample also contains legacy "
        "000.png-style frame names so existing MMReD qa.txt loaders can read it."
    )
    root_meta = root_metadata(
        rooms=rooms,
        seq_lens=seq_lens,
        samples_per_evidence_count=samples_per_evidence_count,
        split=split,
        seed=int(args.seed),
        tokenization=tokenization,
        convention_notes=convention_notes,
        dry_run=bool(args.dry_run),
    )
    write_json(image_root / "metadata.json", root_meta)
    write_json(metadata_root / "metadata.json", root_meta)

    room_counter: Counter[str] = Counter({room: 0 for room in rooms})
    character_counter: Counter[str] = Counter({character: 0 for character in CHARACTERS})
    all_metadata_records_for_features: List[Dict[str, Any]] = []
    per_seq_summaries: List[Dict[str, Any]] = []
    total_buckets = sum(int(seq_len) + 1 for seq_len in seq_lens)
    bucket_index = 0

    for seq_len in seq_lens:
        records_by_evidence: Dict[int, List[Dict[str, Any]]] = {count: [] for count in range(int(seq_len) + 1)}
        all_uniform_records: List[Dict[str, Any]] = []
        seq_target_room_counts: Counter[str] = Counter()

        for evidence_count in range(int(seq_len) + 1):
            bucket_index += 1
            for sample_index in range(samples_per_evidence_count):
                global_index = (
                    sum((int(prev_seq) + 1) * samples_per_evidence_count for prev_seq in seq_lens if int(prev_seq) < int(seq_len))
                    + int(evidence_count) * samples_per_evidence_count
                    + int(sample_index)
                )
                target_room = next_balanced_target_room(rooms, room_counter, rng)
                target_character = next_balanced_character(character_counter, rng)
                seq_target_room_counts[target_room] += 1
                sample = build_sample(
                    seq_len=int(seq_len),
                    split=split,
                    evidence_count=int(evidence_count),
                    sample_index=int(sample_index),
                    global_index=int(global_index),
                    target_character=target_character,
                    target_room=target_room,
                    rooms=rooms,
                    rng=rng,
                )

                sample_rel = f"seq_len_{int(seq_len)}/{split}/{sample.sample_id}"
                sample_dir = image_root / sample_rel
                sample_dir.mkdir(parents=True, exist_ok=True)
                frame_paths = render_sample(sample, sample_dir, rooms)
                write_qa(sample_dir / "qa.txt", sample=sample)
                write_json(sample_dir / "states.json", [dict(state) for state in sample.states])
                metadata = sample_metadata(sample, frame_paths=frame_paths, rooms=rooms)
                write_json(sample_dir / "metadata.json", metadata)

                exact_rel = (
                    f"seq_len_{int(seq_len)}/by_evidence_count/"
                    f"exact_{int(evidence_count)}/{split}/{sample.sample_id}"
                )
                exact_dir = image_root / exact_rel
                exact_dir.mkdir(parents=True, exist_ok=True)
                for child in sample_dir.iterdir():
                    if child.is_file():
                        shutil.copyfile(child, exact_dir / child.name)

                record = metadata_record(sample, metadata)
                records_by_evidence[int(evidence_count)].append(record)
                all_uniform_records.append(record)
                all_metadata_records_for_features.append(record)

            print(
                f"[{bucket_index}/{total_buckets}] generated "
                f"seq_len={int(seq_len)} evidence_count={int(evidence_count)} "
                f"samples={samples_per_evidence_count}"
            )

        features = Dataset.from_list(all_metadata_records_for_features[:1]).features if all_metadata_records_for_features else None
        seq_metadata_dir = metadata_root / f"seq_len_{int(seq_len)}"
        for evidence_count in range(int(seq_len) + 1):
            exact_dir = seq_metadata_dir / "by_evidence_count" / f"exact_{int(evidence_count)}"
            save_dataset(records_by_evidence[int(evidence_count)], exact_dir, features)
        save_dataset(all_uniform_records, seq_metadata_dir / split, features)

        per_seq_summaries.append(
            {
                "seq_len": int(seq_len),
                "num_samples": len(all_uniform_records),
                "samples_per_evidence_count": samples_per_evidence_count,
                "evidence_histogram": {
                    str(count): len(records_by_evidence[int(count)]) for count in range(int(seq_len) + 1)
                },
                "target_room_counts": dict(sorted(seq_target_room_counts.items())),
            }
        )

    write_json(
        image_root / "generation_summary.json",
        {
            "summaries": per_seq_summaries,
            "global_target_room_counts": dict(sorted(room_counter.items())),
            "global_target_character_counts": dict(sorted(character_counter.items())),
        },
    )
    write_json(
        metadata_root / "generation_summary.json",
        {
            "summaries": per_seq_summaries,
            "global_target_room_counts": dict(sorted(room_counter.items())),
            "global_target_character_counts": dict(sorted(character_counter.items())),
        },
    )

    print(f"Wrote Park MMReD images to: {image_root.resolve()}")
    print(f"Wrote Park MMReD metadata to: {metadata_root.resolve()}")
    print("Global target-room counts: " + json.dumps(dict(sorted(room_counter.items())), sort_keys=True))


def main() -> None:
    args = parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
