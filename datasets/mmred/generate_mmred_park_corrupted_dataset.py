#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# RECOVERED SOURCE (2026-07-29, serious-refactor phase 0).
# Original path: evaluations/scripts/generate_mmred_park_corrupted_dataset.py (deleted;
# survived only as evaluations/scripts/__pycache__/generate_mmred_park_corrupted_dataset.cpython-39.pyc).
# Generator of the per-frame corrupted donors (data/mmred_corrupted_park) for the
# MMReD Park dataset.
#
# Recovery route: dangling (unreachable) git blob 28861a4fad59e305e71fc82bd9c2be7e79050663,
# found by content-grepping every object in .git — the file was `git add`ed once but
# never committed on any ref.
# Verification: compiled with .venv Python 3.9.21 and structurally compared against the
# orphan pyc: co_code, co_consts, co_names, co_varnames, co_flags AND the line-number
# table (co_lnotab) are IDENTICAL for the module and every nested function
# => byte-exact source recovery, not a decompilation.
# Corroboration: the same source appears in Codex transcript
# ~/.codex/sessions/2026/05/06/rollout-2026-05-06T22-04-25-*.jsonl (pyc mtime 2026-05-06 23:02).
#
# Everything below this comment block is byte-identical to the recovered blob (only
# this header was inserted, which shifts line numbers vs. the original pyc).
# NOTE: the sys.path setup below takes parents[2] as the repo root; datasets/mmred/
# sits at the same depth as the original evaluations/scripts/, so the script works
# unmodified from this location.
# ---------------------------------------------------------------------------
"""Generate per-frame corrupted donor samples for the MMReD Park dataset.

This mirrors the existing ``data/mmred_corrupted`` convention:

  corrupted_root/
    seq_len_k/
      all_uniform/
        sample_id/
          corrupted_frame_i/
            000.png ...
            qa.txt
      by_evidence_count/
        exact_j/
          all_uniform/
            sample_id/
              corrupted_frame_i/
                000.png ...
                qa.txt

For each clean rendered sample, one donor is created for each evidence frame.
The donor keeps the original question/answer but removes the target character
from the target room in the selected frame, matching the old corruption helper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MMRED_RENDER_DIR = _REPO_ROOT / "datasets" / "mmred"
if str(_MMRED_RENDER_DIR) not in sys.path:
    sys.path.insert(0, str(_MMRED_RENDER_DIR))

import render_mmred  # noqa: E402


DEFAULT_CLEAN_ROOT = Path("data/mmred_images_park")
DEFAULT_CORRUPTED_ROOT = Path("data/mmred_corrupted_park")
DEFAULT_SEQ_LENS = tuple(range(1, 9))
DEFAULT_SPLIT = "all_uniform"
DEFAULT_ROOMS = ("Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park")


def parse_int_values(raw_values: Sequence[Any], *, arg_name: str) -> List[int]:
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


def iter_sample_dirs(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(child for child in path.iterdir() if child.is_dir() and (child / "qa.txt").is_file())


def evidence_frames_for_sample(sample_dir: Path) -> Tuple[str, str, List[int]]:
    qa = render_mmred.read_sample_qa(str(sample_dir))
    states, _tail_lines = render_mmred.split_question_into_states_and_tail(qa["question"])
    character, room = render_mmred.parse_target_character_room(qa["question"])
    evidence_frames = [
        int(frame_idx)
        for frame_idx, state in enumerate(states)
        if render_mmred.char_in_room(state["rooms"], character, room)
    ]
    return str(character), str(room), evidence_frames


def dataset_rel_for_sample_dir(clean_root: Path, sample_dir: Path) -> Path:
    return sample_dir.parent.relative_to(clean_root)


def output_sample_dir(corrupted_root: Path, clean_root: Path, sample_dir: Path) -> Path:
    return corrupted_root / dataset_rel_for_sample_dir(clean_root, sample_dir) / sample_dir.name


def expected_corruption_dirs(corrupted_root: Path, clean_root: Path, sample_dir: Path, evidence_frames: Sequence[int]) -> List[Path]:
    sample_out_dir = output_sample_dir(corrupted_root, clean_root, sample_dir)
    return [sample_out_dir / f"corrupted_frame_{int(frame_idx)}" for frame_idx in evidence_frames]


def generate_corruptions_for_sample(
    *,
    clean_root: Path,
    corrupted_root: Path,
    sample_dir: Path,
    overwrite_existing_samples: bool,
) -> Dict[str, Any]:
    character, room, evidence_frames = evidence_frames_for_sample(sample_dir)
    dataset_rel = dataset_rel_for_sample_dir(clean_root, sample_dir)
    sample_out_dir = output_sample_dir(corrupted_root, clean_root, sample_dir)

    if overwrite_existing_samples and sample_out_dir.exists():
        shutil.rmtree(sample_out_dir)

    created = 0
    skipped_existing = 0
    for frame_idx in evidence_frames:
        out_dir = sample_out_dir / f"corrupted_frame_{int(frame_idx)}"
        if out_dir.is_dir() and (out_dir / "qa.txt").is_file() and not overwrite_existing_samples:
            skipped_existing += 1
            continue
        render_mmred.generate_corrupted_sample_from_rendered(
            sample_dir=sample_dir,
            corrupt_frame_idx=int(frame_idx),
            character=character,
            room=room,
            out_root=corrupted_root,
            dataset_rel_dir=dataset_rel,
        )
        created += 1

    return {
        "sample_id": sample_dir.name,
        "sample_dir": str(sample_dir),
        "output_sample_dir": str(sample_out_dir),
        "dataset_rel": str(dataset_rel),
        "target_character": character,
        "target_room": room,
        "evidence_frames": [int(value) for value in evidence_frames],
        "created_corruptions": int(created),
        "skipped_existing_corruptions": int(skipped_existing),
    }


def clean_bucket_dirs(*, clean_root: Path, seq_len: int, split: str, include_all_uniform: bool, include_exact: bool) -> List[Path]:
    seq_root = clean_root / f"seq_len_{int(seq_len)}"
    bucket_dirs: List[Path] = []
    if include_all_uniform:
        bucket_dirs.append(seq_root / str(split))
    if include_exact:
        exact_root = seq_root / "by_evidence_count"
        if exact_root.is_dir():
            bucket_dirs.extend(sorted(path / str(split) for path in exact_root.glob("exact_*") if path.is_dir()))
    return [path for path in bucket_dirs if path.is_dir()]


def generate_bucket(
    *,
    clean_root: Path,
    corrupted_root: Path,
    bucket_dir: Path,
    overwrite_existing_samples: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    sample_dirs = iter_sample_dirs(bucket_dir)
    if limit is not None:
        sample_dirs = sample_dirs[: int(limit)]

    created = 0
    skipped_existing = 0
    evidence_frame_total = 0
    processed_samples = 0
    for sample_idx, sample_dir in enumerate(sample_dirs, start=1):
        result = generate_corruptions_for_sample(
            clean_root=clean_root,
            corrupted_root=corrupted_root,
            sample_dir=sample_dir,
            overwrite_existing_samples=overwrite_existing_samples,
        )
        processed_samples += 1
        created += int(result["created_corruptions"])
        skipped_existing += int(result["skipped_existing_corruptions"])
        evidence_frame_total += len(result["evidence_frames"])
        if sample_idx == 1 or sample_idx % 100 == 0 or sample_idx == len(sample_dirs):
            print(
                f"  {bucket_dir.relative_to(clean_root)}: "
                f"sample {sample_idx}/{len(sample_dirs)} "
                f"created={created} skipped_existing={skipped_existing}"
            )

    return {
        "bucket_dir": str(bucket_dir),
        "relative_bucket_dir": str(bucket_dir.relative_to(clean_root)),
        "processed_samples": int(processed_samples),
        "candidate_samples": int(len(sample_dirs)),
        "evidence_frame_total": int(evidence_frame_total),
        "created_corruptions": int(created),
        "skipped_existing_corruptions": int(skipped_existing),
    }


def verify_sample_outputs(*, clean_root: Path, corrupted_root: Path, seq_lens: Sequence[int], split: str) -> Dict[str, Any]:
    missing: List[str] = []
    checked_samples = 0
    checked_corruptions = 0
    for seq_len in seq_lens:
        for bucket_dir in clean_bucket_dirs(
            clean_root=clean_root,
            seq_len=int(seq_len),
            split=split,
            include_all_uniform=True,
            include_exact=True,
        ):
            for sample_dir in iter_sample_dirs(bucket_dir):
                _character, _room, evidence_frames = evidence_frames_for_sample(sample_dir)
                checked_samples += 1
                for out_dir in expected_corruption_dirs(corrupted_root, clean_root, sample_dir, evidence_frames):
                    checked_corruptions += 1
                    if not (out_dir / "qa.txt").is_file():
                        missing.append(str(out_dir / "qa.txt"))
                        continue
                    for frame_idx in range(int(seq_len)):
                        if not (out_dir / f"{int(frame_idx):03d}.png").is_file():
                            missing.append(str(out_dir / f"{int(frame_idx):03d}.png"))
                            break
    return {
        "checked_samples": int(checked_samples),
        "checked_corruptions": int(checked_corruptions),
        "missing_count": int(len(missing)),
        "missing_preview": missing[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data/mmred_corrupted_park-style donors from the rendered Park dataset."
    )
    parser.add_argument("--clean-root", "--clean_root", dest="clean_root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument(
        "--output-root",
        "--output_root",
        "--corrupted-root",
        "--corrupted_root",
        dest="output_root",
        type=Path,
        default=DEFAULT_CORRUPTED_ROOT,
    )
    parser.add_argument(
        "--seq-lens",
        "--seq_lens",
        dest="seq_lens",
        nargs="+",
        default=[str(value) for value in DEFAULT_SEQ_LENS],
    )
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--limit-per-bucket", "--limit_per_bucket", dest="limit_per_bucket", type=int, default=None)
    parser.add_argument("--skip-all-uniform", "--skip_all_uniform", dest="include_all_uniform", action="store_false")
    parser.add_argument("--skip-exact-buckets", "--skip_exact_buckets", dest="include_exact", action="store_false")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the entire output root before generation.",
    )
    parser.add_argument(
        "--overwrite-existing-samples",
        "--overwrite_existing_samples",
        dest="overwrite_existing_samples",
        action="store_true",
        help="Regenerate sample output dirs that already exist without deleting the whole root.",
    )
    parser.add_argument("--no-verify", "--no_verify", dest="verify", action="store_false", default=True)
    args = parser.parse_args()

    if args.limit_per_bucket is not None and int(args.limit_per_bucket) <= 0:
        raise ValueError("--limit-per-bucket must be positive when provided")
    return args


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    clean_root = Path(args.clean_root)
    output_root = Path(args.output_root)
    seq_lens = parse_int_values(args.seq_lens, arg_name="--seq-lens")

    if not clean_root.is_dir():
        raise FileNotFoundError(f"Missing clean Park image root: {clean_root}")
    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Match the Park dataset renderer setup: same 2x3 layout, Park in the old Hallway slot.
    render_mmred.ROOMS = list(DEFAULT_ROOMS)

    print(f"clean_root={clean_root}")
    print(f"output_root={output_root}")
    print(f"seq_lens={seq_lens}")
    print(f"split={args.split}")
    print(f"rooms={render_mmred.ROOMS}")

    bucket_summaries: List[Dict[str, Any]] = []
    for seq_len in seq_lens:
        bucket_dirs = clean_bucket_dirs(
            clean_root=clean_root,
            seq_len=int(seq_len),
            split=str(args.split),
            include_all_uniform=bool(args.include_all_uniform),
            include_exact=bool(args.include_exact),
        )
        if not bucket_dirs:
            print(f"[warn] no clean bucket dirs found for seq_len={int(seq_len)}")
            continue
        for bucket_dir in bucket_dirs:
            print(f"Generating corrupted donors for bucket: {bucket_dir}")
            bucket_summaries.append(
                generate_bucket(
                    clean_root=clean_root,
                    corrupted_root=output_root,
                    bucket_dir=bucket_dir,
                    overwrite_existing_samples=bool(args.overwrite_existing_samples),
                    limit=args.limit_per_bucket,
                )
            )

    verification: Optional[Dict[str, Any]] = None
    if bool(args.verify):
        print("Verifying generated corrupted donor files...")
        verification = verify_sample_outputs(
            clean_root=clean_root,
            corrupted_root=output_root,
            seq_lens=seq_lens,
            split=str(args.split),
        )
        print(
            "verification: "
            f"checked_corruptions={verification['checked_corruptions']} "
            f"missing_count={verification['missing_count']}"
        )
        if int(verification["missing_count"]) > 0:
            raise RuntimeError(f"Generated corrupted dataset is incomplete: {verification['missing_preview']}")

    elapsed = time.perf_counter() - start_time
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clean_root": str(clean_root),
        "output_root": str(output_root),
        "seq_lens": [int(value) for value in seq_lens],
        "split": str(args.split),
        "rooms": list(DEFAULT_ROOMS),
        "generator": "evaluations/scripts/generate_mmred_park_corrupted_dataset.py",
        "based_on": "datasets/mmred/render_mmred.py corruption helpers",
        "force": bool(args.force),
        "overwrite_existing_samples": bool(args.overwrite_existing_samples),
        "limit_per_bucket": args.limit_per_bucket,
        "bucket_summaries": bucket_summaries,
        "totals": {
            "processed_samples": int(sum(row["processed_samples"] for row in bucket_summaries)),
            "evidence_frame_total": int(sum(row["evidence_frame_total"] for row in bucket_summaries)),
            "created_corruptions": int(sum(row["created_corruptions"] for row in bucket_summaries)),
            "skipped_existing_corruptions": int(sum(row["skipped_existing_corruptions"] for row in bucket_summaries)),
        },
        "verification": verification,
        "elapsed_seconds": float(elapsed),
    }
    write_json(output_root / "corruption_manifest.json", manifest)
    print(f"Wrote manifest: {output_root / 'corruption_manifest.json'}")
    print(
        "Done: "
        f"processed_samples={manifest['totals']['processed_samples']} "
        f"created_corruptions={manifest['totals']['created_corruptions']} "
        f"skipped_existing={manifest['totals']['skipped_existing_corruptions']} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
