#!/usr/bin/env python3
"""Generate and validate synthetic MMReD-style datasets in Hugging Face on-disk layout."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_SEQ_LENS: Tuple[int, ...] = (2, 4, 8, 16, 32)
DEFAULT_SAMPLES_PER_SEQ = 300
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)

DEFAULT_EVIDENCE_COUNTS: Dict[int, Tuple[int, ...]] = {
    2: (2,),
    4: (2, 3, 4),
    8: (2, 3, 4, 5, 6, 7, 8),
    16: (2, 4, 6, 8, 10, 12, 14, 16),
    32: (2, 4, 8, 12, 16, 20, 24, 28, 32),
}

PATTERNS: Tuple[str, ...] = (
    "clustered",
    "evenly_spread",
    "front_loaded",
    "back_loaded",
    "random",
)

DEFAULT_CHARACTERS: Tuple[str, ...] = (
    "Sandra",
    "Mary",
    "Michael",
    "John",
    "Daniel",
)

DEFAULT_ROOMS: Tuple[str, ...] = (
    "Kitchen",
    "Bathroom",
    "Garden",
    "Office",
    "Bedroom",
    "Hallway",
)


@dataclass(frozen=True)
class Frame:
    frame_index: int
    character: str
    room: str
    text: str
    is_evidence: bool


@dataclass(frozen=True)
class Sample:
    sample_id: str
    seq_len: int
    question: str
    answer: int
    target_character: str
    target_room: str
    evidence_indices: Tuple[int, ...]
    evidence_count: int
    positional_pattern: str
    frames: Tuple[Frame, ...]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seq_lens(value: str) -> List[int]:
    seq_lens = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seq_lens:
        raise ValueError("--seq-lens cannot be empty")
    return seq_lens


def parse_split_ratios(value: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratios must have exactly 3 comma-separated values (train,val,test)")
    if any(x <= 0 for x in parts):
        raise ValueError("split ratios must be positive")
    total = sum(parts)
    if total <= 0:
        raise ValueError("split ratios must sum to > 0")
    return (parts[0] / total, parts[1] / total, parts[2] / total)


def allocate_quota(values: Sequence[int], total: int) -> Dict[int, int]:
    """Balanced allocation with deterministic remainder distribution."""
    if total <= 0:
        raise ValueError("total must be > 0")
    if not values:
        raise ValueError("values cannot be empty")

    ordered = sorted(values)
    base = total // len(ordered)
    rem = total % len(ordered)
    quota = {v: base for v in ordered}
    for v in ordered[:rem]:
        quota[v] += 1
    return quota


def allocate_split_sizes(total: int, ratios: Tuple[float, float, float]) -> Dict[str, int]:
    names = ("train", "val", "test")
    raw = [total * r for r in ratios]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)

    frac_order = sorted(
        range(3),
        key=lambda i: (raw[i] - floors[i], ratios[i]),
        reverse=True,
    )
    for i in frac_order[:remainder]:
        floors[i] += 1

    return dict(zip(names, floors))


def pick_other(rng: random.Random, pool: Sequence[str], forbidden: str) -> str:
    choices = [x for x in pool if x != forbidden]
    if not choices:
        raise ValueError("pool must contain at least one value not equal to forbidden")
    return rng.choice(choices)


def generate_evidence_indices(
    rng: random.Random,
    seq_len: int,
    evidence_count: int,
    pattern: str,
) -> Tuple[int, ...]:
    all_indices = list(range(seq_len))
    if evidence_count < 2 or evidence_count > seq_len:
        raise ValueError("evidence_count out of range")
    if evidence_count == seq_len:
        return tuple(all_indices)

    if pattern == "clustered":
        start = rng.randint(0, seq_len - evidence_count)
        return tuple(range(start, start + evidence_count))

    if pattern == "evenly_spread":
        if evidence_count == 2:
            return (0, seq_len - 1)
        base = sorted({round(i * (seq_len - 1) / (evidence_count - 1)) for i in range(evidence_count)})
        while len(base) < evidence_count:
            candidate = rng.choice([i for i in all_indices if i not in base])
            base.append(candidate)
            base.sort()
        return tuple(base)

    if pattern == "front_loaded":
        front = list(range(max(evidence_count, seq_len // 2)))
        if len(front) < evidence_count:
            front = all_indices
        return tuple(sorted(rng.sample(front, evidence_count)))

    if pattern == "back_loaded":
        start = min(seq_len - evidence_count, seq_len // 2)
        back = list(range(start, seq_len))
        if len(back) < evidence_count:
            back = all_indices
        return tuple(sorted(rng.sample(back, evidence_count)))

    if pattern == "random":
        return tuple(sorted(rng.sample(all_indices, evidence_count)))

    raise ValueError(f"unknown pattern: {pattern}")


def build_frames(
    rng: random.Random,
    seq_len: int,
    target_character: str,
    target_room: str,
    evidence_indices: Sequence[int],
    characters: Sequence[str],
    rooms: Sequence[str],
) -> Tuple[Frame, ...]:
    evidence_set = set(evidence_indices)
    non_evidence_indices = [i for i in range(seq_len) if i not in evidence_set]

    distractor_types: List[str] = []
    base_types = [
        "same_character_wrong_room",
        "different_character_same_room",
        "different_character_different_room",
    ]
    if len(non_evidence_indices) >= 3:
        distractor_types.extend(base_types)
    while len(distractor_types) < len(non_evidence_indices):
        distractor_types.append(rng.choice(base_types))
    rng.shuffle(distractor_types)

    frames: List[Frame] = []
    type_pos = 0
    for idx in range(seq_len):
        if idx in evidence_set:
            character = target_character
            room = target_room
            is_evidence = True
        else:
            dtype = distractor_types[type_pos]
            type_pos += 1
            if dtype == "same_character_wrong_room":
                character = target_character
                room = pick_other(rng, rooms, target_room)
            elif dtype == "different_character_same_room":
                character = pick_other(rng, characters, target_character)
                room = target_room
            else:
                character = pick_other(rng, characters, target_character)
                room = pick_other(rng, rooms, target_room)
            is_evidence = False

        text = f"At this step, {character} is in the {room}."
        frames.append(
            Frame(
                frame_index=idx,
                character=character,
                room=room,
                text=text,
                is_evidence=is_evidence,
            )
        )

    return tuple(frames)


def pick_target_pair(
    rng: random.Random,
    characters: Sequence[str],
    rooms: Sequence[str],
    pair_counter: Counter[Tuple[str, str]],
) -> Tuple[str, str]:
    pairs = [(c, r) for c in characters for r in rooms]
    min_count = min(pair_counter[p] for p in pairs)
    least_used = [p for p in pairs if pair_counter[p] == min_count]
    pair = rng.choice(least_used)
    pair_counter[pair] += 1
    return pair


def build_mmred_question_text(
    rng: random.Random,
    sample_frames: Sequence[Frame],
    target_character: str,
    target_room: str,
    characters: Sequence[str],
    rooms: Sequence[str],
) -> str:
    state_lines: List[str] = []
    for step_idx, frame in enumerate(sample_frames, start=1):
        # Build a full per-step world state (all characters have a room).
        char_to_room: Dict[str, str] = {}

        # Keep target placement aligned with evidence logic.
        if frame.character == target_character:
            char_to_room[target_character] = frame.room
        else:
            char_to_room[target_character] = pick_other(rng, rooms, target_room)

        # Preserve the frame's visible fact for non-target distractor characters.
        if frame.character != target_character:
            char_to_room[frame.character] = frame.room

        # Assign all remaining characters to random rooms.
        for ch in characters:
            if ch in char_to_room:
                continue
            char_to_room[ch] = rng.choice(rooms)

        rooms_to_chars: Dict[str, List[str]] = {room: [] for room in rooms}
        for ch, rm in char_to_room.items():
            rooms_to_chars.setdefault(rm, []).append(ch)
        for rm in rooms_to_chars:
            rooms_to_chars[rm] = sorted(rooms_to_chars[rm])

        state = {
            "step_id": step_idx,
            "rooms": rooms_to_chars,
        }
        state_lines.append(repr(state))
    nl_question = f"How many steps did {target_character} spend in the {target_room}?"
    return "\n".join(state_lines + [nl_question])


def build_sample(
    rng: random.Random,
    seq_len: int,
    sample_index: int,
    evidence_count: int,
    pattern: str,
    characters: Sequence[str],
    rooms: Sequence[str],
    pair_counter: Counter[Tuple[str, str]],
) -> Sample:
    target_character, target_room = pick_target_pair(rng, characters, rooms, pair_counter)
    evidence_indices = generate_evidence_indices(rng, seq_len, evidence_count, pattern)
    frames = build_frames(
        rng=rng,
        seq_len=seq_len,
        target_character=target_character,
        target_room=target_room,
        evidence_indices=evidence_indices,
        characters=characters,
        rooms=rooms,
    )

    question = build_mmred_question_text(
        rng=rng,
        sample_frames=frames,
        target_character=target_character,
        target_room=target_room,
        characters=characters,
        rooms=rooms,
    )
    sample_id = f"seq{seq_len}_sample_{sample_index:04d}"

    return Sample(
        sample_id=sample_id,
        seq_len=seq_len,
        question=question,
        answer=evidence_count,
        target_character=target_character,
        target_room=target_room,
        evidence_indices=tuple(evidence_indices),
        evidence_count=evidence_count,
        positional_pattern=pattern,
        frames=frames,
    )


def sample_signature(sample: Sample) -> Tuple[Any, ...]:
    return (
        sample.seq_len,
        sample.target_character,
        sample.target_room,
        sample.positional_pattern,
        tuple((f.character, f.room) for f in sample.frames),
        tuple(f.text for f in sample.frames),
    )


def near_duplicate_signature(sample: Sample) -> Tuple[Any, ...]:
    tags: List[str] = []
    for frame in sample.frames:
        if frame.character == sample.target_character and frame.room == sample.target_room:
            tags.append("E")
        elif frame.character == sample.target_character and frame.room != sample.target_room:
            tags.append("SCWR")
        elif frame.character != sample.target_character and frame.room == sample.target_room:
            tags.append("DCSR")
        else:
            tags.append("DCDD")
    return (
        sample.seq_len,
        sample.target_character,
        sample.target_room,
        sample.evidence_indices,
        sample.positional_pattern,
        tuple(tags),
    )


def validate_sample(sample: Sample) -> List[str]:
    errors: List[str] = []

    question_lines = [ln.strip() for ln in sample.question.splitlines() if ln.strip()]
    if len(question_lines) < sample.seq_len + 1:
        errors.append("question missing state lines")
    else:
        state_lines = question_lines[: sample.seq_len]
        for i, line in enumerate(state_lines, start=1):
            if not line.startswith("{") or "'rooms'" not in line or f"'step_id': {i}" not in line:
                errors.append(f"malformed question state line at step {i}")
                break
        expected_tail = f"How many steps did {sample.target_character} spend in the {sample.target_room}?"
        if question_lines[-1] != expected_tail:
            errors.append("question tail mismatch")

    if sample.evidence_count < 2:
        errors.append("evidence_count < 2")

    if sample.answer != sample.evidence_count:
        errors.append("answer != evidence_count")

    if len(sample.frames) != sample.seq_len:
        errors.append("frames length mismatch")

    expected_indices = []
    for frame in sample.frames:
        if frame.frame_index < 0 or frame.frame_index >= sample.seq_len:
            errors.append(f"frame index out of range: {frame.frame_index}")
        is_ev = frame.character == sample.target_character and frame.room == sample.target_room
        if frame.is_evidence != is_ev:
            errors.append(f"is_evidence mismatch at frame {frame.frame_index}")
        if is_ev:
            expected_indices.append(frame.frame_index)

    expected_indices = sorted(expected_indices)
    if tuple(expected_indices) != tuple(sample.evidence_indices):
        errors.append("evidence_indices mismatch")

    if len(expected_indices) != sample.evidence_count:
        errors.append("derived evidence count mismatch")

    return errors


def sample_to_record(sample: Sample) -> Dict[str, Any]:
    return {
        # Mirror MMReD core columns.
        "question": sample.question,
        "seq_len": sample.seq_len,
        "answer": sample.answer,
        "qid": sample.sample_id,
        "qtype": "steps_in_room",
        "atype": "integer",
        # Required synthetic metadata.
        "sample_id": sample.sample_id,
        "target_character": sample.target_character,
        "target_room": sample.target_room,
        "evidence_indices": list(sample.evidence_indices),
        "evidence_count": sample.evidence_count,
        "positional_pattern": sample.positional_pattern,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "character": frame.character,
                "room": frame.room,
                "text": frame.text,
                "is_evidence": frame.is_evidence,
            }
            for frame in sample.frames
        ],
    }


def split_records(
    records: List[Dict[str, Any]],
    split_sizes: Dict[str, int],
) -> Dict[str, List[Dict[str, Any]]]:
    train_n = split_sizes["train"]
    val_n = split_sizes["val"]
    return {
        "train": records[:train_n],
        "val": records[train_n : train_n + val_n],
        "test": records[train_n + val_n :],
    }


def inspect_reference_schema(reference_root: Path) -> None:
    seq2 = reference_root / "seq_len_2"
    if not seq2.exists():
        print(f"Reference dataset not found at {seq2}; continuing without schema preview.")
        return

    try:
        ds = load_from_disk(str(seq2))
        split_name = next((name for name in ("train", "val", "test") if name in ds), None)
        if split_name is None:
            print(f"Reference {seq2} has no train/val/test splits; continuing.")
            return
        row = ds[split_name][0]
        print("Reference MMReD schema snapshot:")
        print(f"  split={split_name} columns={list(row.keys())}")
        print(f"  features={ds[split_name].features}")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to inspect reference schema from {seq2}: {exc}")


def generate_seq_len_dataset(
    seq_len: int,
    total_samples: int,
    split_ratios: Tuple[float, float, float],
    characters: Sequence[str],
    rooms: Sequence[str],
    seed: int,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    evidence_values = DEFAULT_EVIDENCE_COUNTS.get(seq_len)
    if evidence_values is None:
        raise ValueError(f"Unsupported seq_len={seq_len}; expected one of {sorted(DEFAULT_EVIDENCE_COUNTS)}")

    rng = random.Random(seed + seq_len * 1009)

    evidence_quota = allocate_quota(evidence_values, total_samples)
    pattern_quota = allocate_quota(PATTERNS, total_samples)

    evidence_pool: List[int] = []
    for count in sorted(evidence_quota):
        evidence_pool.extend([count] * evidence_quota[count])
    pattern_pool: List[str] = []
    for pattern in PATTERNS:
        pattern_pool.extend([pattern] * pattern_quota[pattern])
    rng.shuffle(evidence_pool)
    rng.shuffle(pattern_pool)

    pair_counter: Counter[Tuple[str, str]] = Counter()
    # seq_len=2 has very limited combinatorial diversity in this task setup.
    # Enforcing uniqueness there can make moderate sample targets infeasible.
    enforce_uniqueness = seq_len > 2
    if not enforce_uniqueness and total_samples > len(characters) * len(rooms) * len(PATTERNS):
        print(
            f"seq_len={seq_len}: uniqueness checks relaxed; requested {total_samples} samples "
            "exceed coarse unique-pattern capacity."
        )

    seen_exact: set[Tuple[Any, ...]] = set()
    seen_near: set[Tuple[Any, ...]] = set()

    samples: List[Sample] = []
    attempts = 0
    max_attempts = total_samples * 800

    while len(samples) < total_samples and attempts < max_attempts:
        idx = len(samples)
        attempts += 1
        sample = build_sample(
            rng=rng,
            seq_len=seq_len,
            sample_index=idx,
            evidence_count=evidence_pool[idx],
            pattern=pattern_pool[idx],
            characters=characters,
            rooms=rooms,
            pair_counter=pair_counter,
        )

        errors = validate_sample(sample)
        if errors:
            continue

        if enforce_uniqueness:
            sig = sample_signature(sample)
            near = near_duplicate_signature(sample)
            if sig in seen_exact or near in seen_near:
                continue
            seen_exact.add(sig)
            seen_near.add(near)
        samples.append(sample)

    if len(samples) != total_samples:
        raise RuntimeError(
            f"Could not generate enough unique samples for seq_len={seq_len}. "
            f"generated={len(samples)} target={total_samples} attempts={attempts}"
        )

    records = [sample_to_record(s) for s in samples]
    rng.shuffle(records)

    split_sizes = allocate_split_sizes(total_samples, split_ratios)
    split_records_map = split_records(records, split_sizes)

    dataset_dict = DatasetDict(
        {
            split: Dataset.from_list(rows)
            for split, rows in split_records_map.items()
        }
    )

    summary = {
        "seq_len": seq_len,
        "total": total_samples,
        "splits": split_sizes,
        "evidence_hist": dict(Counter(r["evidence_count"] for r in records)),
        "pattern_hist": dict(Counter(r["positional_pattern"] for r in records)),
    }
    return dataset_dict, summary


def validate_dataset_dir(
    output_dir: Path,
    seq_lens: Sequence[int],
    expected_total_per_seq: int,
    split_ratios: Tuple[float, float, float],
) -> List[str]:
    errors: List[str] = []
    expected_splits = allocate_split_sizes(expected_total_per_seq, split_ratios)

    for seq_len in seq_lens:
        seq_dir = output_dir / f"seq_len_{seq_len}"
        enforce_uniqueness = seq_len > 2
        if not seq_dir.exists():
            errors.append(f"missing directory: {seq_dir}")
            continue

        try:
            ds = load_from_disk(str(seq_dir))
        except Exception as exc:
            errors.append(f"failed to load {seq_dir}: {exc}")
            continue

        for split in ("train", "val", "test"):
            if split not in ds:
                errors.append(f"{seq_dir}: missing split '{split}'")

        if any(split not in ds for split in ("train", "val", "test")):
            continue

        seen_exact: set[Tuple[Any, ...]] = set()
        seen_near: set[Tuple[Any, ...]] = set()

        for split in ("train", "val", "test"):
            split_ds = ds[split]
            if len(split_ds) != expected_splits[split]:
                errors.append(
                    f"{seq_dir}:{split} count mismatch expected={expected_splits[split]} actual={len(split_ds)}"
                )

            for row_idx, row in enumerate(split_ds):
                row_id = f"{seq_dir.name}:{split}:{row_idx}"

                required = {
                    "sample_id",
                    "seq_len",
                    "question",
                    "answer",
                    "target_character",
                    "target_room",
                    "evidence_indices",
                    "evidence_count",
                    "positional_pattern",
                    "frames",
                }
                missing = [k for k in required if k not in row]
                if missing:
                    errors.append(f"{row_id} missing columns: {missing}")
                    continue

                if row["seq_len"] != seq_len:
                    errors.append(f"{row_id} seq_len mismatch")

                q_lines = [ln.strip() for ln in str(row["question"]).splitlines() if ln.strip()]
                expected_q = f"How many steps did {row['target_character']} spend in the {row['target_room']}?"
                if len(q_lines) < seq_len + 1 or q_lines[-1] != expected_q:
                    errors.append(f"{row_id} question format mismatch")

                frames = row["frames"]
                if not isinstance(frames, list) or len(frames) != seq_len:
                    errors.append(f"{row_id} malformed frames length")
                    continue

                derived_indices: List[int] = []
                per_frame_pairs: List[Tuple[str, str]] = []
                tags: List[str] = []
                for frame in frames:
                    if not isinstance(frame, dict):
                        errors.append(f"{row_id} malformed frame entry")
                        continue
                    fi = frame.get("frame_index")
                    ch = frame.get("character")
                    rm = frame.get("room")
                    if not isinstance(fi, int) or not isinstance(ch, str) or not isinstance(rm, str):
                        errors.append(f"{row_id} malformed frame fields")
                        continue
                    is_ev = ch == row["target_character"] and rm == row["target_room"]
                    if frame.get("is_evidence") != is_ev:
                        errors.append(f"{row_id} is_evidence mismatch at frame {fi}")
                    if is_ev:
                        derived_indices.append(fi)
                        tags.append("E")
                    elif ch == row["target_character"]:
                        tags.append("SCWR")
                    elif rm == row["target_room"]:
                        tags.append("DCSR")
                    else:
                        tags.append("DCDD")
                    per_frame_pairs.append((ch, rm))

                derived_indices = sorted(derived_indices)
                if derived_indices != sorted(row["evidence_indices"]):
                    errors.append(f"{row_id} bad evidence indices")

                if row["answer"] != len(derived_indices) or row["evidence_count"] != len(derived_indices):
                    errors.append(f"{row_id} incorrect answer/evidence_count")

                exact_sig = (
                    row["seq_len"],
                    row["target_character"],
                    row["target_room"],
                    row["positional_pattern"],
                    tuple(per_frame_pairs),
                    tuple(frame.get("text", "") for frame in frames if isinstance(frame, dict)),
                )
                near_sig = (
                    row["seq_len"],
                    row["target_character"],
                    row["target_room"],
                    tuple(sorted(row["evidence_indices"])),
                    row["positional_pattern"],
                    tuple(tags),
                )

                if enforce_uniqueness:
                    if exact_sig in seen_exact:
                        errors.append(f"{row_id} duplicate sample")
                    seen_exact.add(exact_sig)

                    if near_sig in seen_near:
                        errors.append(f"{row_id} near-duplicate sample")
                    seen_near.add(near_sig)

    return errors


def print_summary(summaries: Iterable[Dict[str, Any]]) -> None:
    print("Generation summary:")
    for summary in summaries:
        seq_len = summary["seq_len"]
        print(f"- seq_len={seq_len}: total={summary['total']}")
        print(
            f"  splits: train={summary['splits']['train']} val={summary['splits']['val']} test={summary['splits']['test']}"
        )
        ev_hist = ", ".join(
            f"{k}:{summary['evidence_hist'][k]}" for k in sorted(summary["evidence_hist"])
        )
        pt_hist = ", ".join(
            f"{k}:{summary['pattern_hist'][k]}" for k in sorted(summary["pattern_hist"])
        )
        print(f"  evidence-count histogram: {ev_hist}")
        print(f"  positional-pattern histogram: {pt_hist}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MMReD-style datasets in Hugging Face DatasetDict layout."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/mmred_generated"))
    parser.add_argument("--reference-root", type=Path, default=Path("data/mmred"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--samples-per-seq", type=int, default=DEFAULT_SAMPLES_PER_SEQ)
    parser.add_argument("--seq-lens", type=str, default=",".join(str(x) for x in DEFAULT_SEQ_LENS))
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--characters", type=str, default=",".join(DEFAULT_CHARACTERS))
    parser.add_argument("--rooms", type=str, default=",".join(DEFAULT_ROOMS))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_name_pool(name: str, values: Sequence[str]) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} requires at least 2 entries")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seq_lens = parse_seq_lens(args.seq_lens)
    split_ratios = parse_split_ratios(args.split_ratios)
    characters = parse_csv_list(args.characters)
    rooms = parse_csv_list(args.rooms)

    validate_name_pool("characters", characters)
    validate_name_pool("rooms", rooms)

    if args.validate_only:
        errors = validate_dataset_dir(
            output_dir=args.output_dir,
            seq_lens=seq_lens,
            expected_total_per_seq=args.samples_per_seq,
            split_ratios=split_ratios,
        )
        if errors:
            print(f"Validation failed: {len(errors)} issues")
            for err in errors[:100]:
                print(f"- {err}")
            raise SystemExit(1)
        print("Validation passed.")
        return

    inspect_reference_schema(args.reference_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    for seq_len in seq_lens:
        if seq_len not in DEFAULT_EVIDENCE_COUNTS:
            raise ValueError(f"Unsupported seq_len={seq_len}; expected {sorted(DEFAULT_EVIDENCE_COUNTS)}")

        ds_dict, summary = generate_seq_len_dataset(
            seq_len=seq_len,
            total_samples=args.samples_per_seq,
            split_ratios=split_ratios,
            characters=characters,
            rooms=rooms,
            seed=args.seed,
        )

        seq_dir = args.output_dir / f"seq_len_{seq_len}"
        if seq_dir.exists() and args.overwrite:
            shutil.rmtree(seq_dir)
        elif seq_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"{seq_dir} already exists. Pass --overwrite to replace existing generated data."
            )

        ds_dict.save_to_disk(str(seq_dir))
        summaries.append(summary)

    print_summary(summaries)


if __name__ == "__main__":
    main()
