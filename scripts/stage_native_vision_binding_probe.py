"""Stage 16 fixed-marginal MMReD Vision conjunction pairs on Slurm CPU.

This is an OOD input-counterfactual diagnostic, not a same-law efficacy test.
Each N32 pair changes K2 to K6 by four room swaps, preserving the character at
every position and both full marginals. No model or predictions are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Imported helpers have no heavy imports at module scope.
from scripts.stage_native_vision_v2_clean import (
    CHARS, DATA_BASE, PARK_ROOMS, V1_ROOT, describe, digest, exclusive_json,
    prepare_renderer, publish_sample, render_cached, stable_seed,
)
from scripts.stage_native_vision_pilot import read_candidate, source_states
from scripts.audit_native_vision_data import inspect

DEFAULT_ROOT = DATA_BASE / "v3_binding_probe"
OUTPUT_ROOT = REPO / "outputs/native_aggregation_vlm/v3/binding_data"
PAIR_COUNT = 16
SEMANTIC_KEYS = ("matches", "same_character_wrong_room",
                 "queried_room_other_character", "neither")
LAW = dict(
    name="fixed_marginal_conjunction_room_swaps",
    distribution_status="OOD diagnostic: fixed contingency counts and coupled low/high scenes",
    target="Uniform over 9 characters x 6 rooms",
    low_contingency_counts=[2, 10, 10, 10],
    high_contingency_counts=[6, 6, 6, 14],
    contingency_order=list(SEMANTIC_KEYS),
    marginal_C=12, marginal_R=12, n_frames=32,
    within_category_identities="Uniform among valid characters and rooms",
    initial_order="Uniform random shuffle of the 32 low-condition frames",
    intervention="Choose four C-only and four R-only positions without replacement; pair uniformly and swap their rooms",
    invariant="Every character identity at its original position, complete character histogram, complete room histogram, question and step labels",
    changed_frames=8, unchanged_frames=24,
    limitation="Associations outside the queried conjunction also change; not every alternative shortcut is excluded",
    primary_interpretation="Use both-members-correct and predicted count change versus gold change 4; a fixed marginal-only guess can score 50% example exact but zero both-pair exact",
)


def manifest_rows(manifest: dict) -> list[dict]:
    """Support the regular manifests and the existing nested causal manifest."""
    rows = [row for cell in manifest.get("splits", {}).values()
            for row in cell["samples"]]
    if not rows and "pairs" in manifest:
        rows = [pair[condition] for pair in manifest["pairs"]
                for condition in ("low", "high")]
    if not rows:
        raise ValueError("Exclusion manifest has no recognized samples")
    return rows


def exclusions() -> tuple[set[str], dict, dict]:
    paths = [V1_ROOT / f"{purpose}_manifest.json" for purpose in ("main", "profile")]
    paths += [DATA_BASE / "v2_clean" / f"{purpose}_manifest.json"
              for purpose in ("main", "count", "profile", "profile_count", "causal_probe")]
    used, ledger = set(), {}
    reference = None
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required exclusion manifest is absent or a symlink: {path}")
        manifest = json.loads(path.read_text())
        rows = manifest_rows(manifest)
        hashes = [row["content_sha256"] for row in rows]
        if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
            raise ValueError(f"Invalid content hashes in {path}")
        used.update(hashes)
        ledger[str(path.resolve())] = dict(sha256=digest(path), samples=len(rows),
                                                distinct_content=len(set(hashes)))
        if path == V1_ROOT / "main_manifest.json":
            reference = manifest
    if reference is None:
        raise AssertionError("Missing renderer parity references")
    return used, ledger, reference


def histogram(frames: list[tuple[str, str]]) -> dict:
    chars, rooms = Counter(c for c, _ in frames), Counter(r for _, r in frames)
    return dict(character_histogram={c: chars[c] for c in CHARS},
                room_histogram={r: rooms[r] for r in PARK_ROOMS})


def generate_pair(index: int, rng: random.Random, used: set[str], seed: int) -> list[dict]:
    pair_id = f"v3_binding_{index:04d}"
    for attempt in range(10000):
        character, room = rng.choice(CHARS), rng.choice(PARK_ROOMS)
        others = [c for c in CHARS if c != character]
        other_rooms = [r for r in PARK_ROOMS if r != room]
        low = ([(character, room)] * 2
               + [(character, rng.choice(other_rooms)) for _ in range(10)]
               + [(rng.choice(others), room) for _ in range(10)]
               + [(rng.choice(others), rng.choice(other_rooms)) for _ in range(10)])
        rng.shuffle(low)
        c_positions = [i for i, (c, r) in enumerate(low) if c == character and r != room]
        r_positions = [i for i, (c, r) in enumerate(low) if c != character and r == room]
        switches = list(zip(rng.sample(c_positions, 4), rng.sample(r_positions, 4)))
        high = list(low)
        for c_index, r_index in switches:
            c_left, r_left = low[c_index]
            c_right, r_right = low[r_index]
            high[c_index] = (c_left, r_right)
            high[r_index] = (c_right, r_left)
        changed = sorted(position for pair in switches for position in pair)
        records = []
        for condition, frames, gold in (("low", low, 2), ("high", high, 6)):
            states, question, content_sha, counts = describe(frames, character, room)
            records.append(dict(
                sid=f"{pair_id}_{condition}_N32_K{gold}", n_frames=32,
                split="test", gold=gold, seed=seed, pair_id=pair_id,
                condition=condition, anchor_id=pair_id, test_family="fixed_marginal_binding",
                axis="conjunction_fixed_marginals", target_character=character,
                target_room=room, marginal_C=12, marginal_R=12,
                question=question, states=states, frames=frames,
                content_sha256=content_sha, semantic_counts=counts,
                changed_positions=changed, changed_positions_indexing="zero-based",
                room_swap_pairs=[list(pair) for pair in switches],
                generation_attempt=attempt, **histogram(frames)))
        hashes = [row["content_sha256"] for row in records]
        if len(set(hashes)) == 2 and not any(value in used for value in hashes):
            used.update(hashes)
            return records
    raise RuntimeError(f"Could not find an unused pair for {pair_id}")


def independently_audit_row(row: dict, root: Path) -> dict:
    """Reparse published QA; rehash and decode every published image."""
    from PIL import Image

    path = Path(row["path"])
    expected = root / "mmred_vfiltered" / "seq_len_32" / "test" / row["sid"]
    if path != expected or path.is_symlink() or not path.is_relative_to(root):
        raise ValueError(f"Unexpected published sample path: {path}")
    parsed = read_candidate(path, "test", 32)
    for field in ("sid", "split", "n_frames", "gold", "qa_sha256", "content_sha256"):
        if row[field] != parsed[field]:
            raise AssertionError(f"Published QA disagrees with {field}: {path}")
    counts = inspect(row, "fixed_marginal_binding", "test_N32")
    if any(counts[key] != row["semantic_counts"][key] for key in SEMANTIC_KEYS):
        raise AssertionError(f"Independent semantic recount failed: {path}")
    for key in ("target_character", "target_room"):
        if counts[key] != row[key]:
            raise AssertionError(f"Question target differs from metadata: {path}")
    states = source_states(path)
    frames = []
    for state in states:
        occupancy = state["rooms"]
        if len(occupancy) != 1:
            raise AssertionError("Expected exactly one occupied room per frame")
        room, people = next(iter(occupancy.items()))
        if room not in PARK_ROOMS or len(people) != 1 or people[0] not in CHARS:
            raise AssertionError("Unknown identity or malformed occupancy")
        frames.append((people[0], room))
    hist = histogram(frames)
    if any(row[key] != value for key, value in hist.items()):
        raise AssertionError("Published full marginals disagree with independent recount")
    if hist["character_histogram"][row["target_character"]] != 12 or hist["room_histogram"][row["target_room"]] != 12:
        raise AssertionError("Queried marginals are not both exactly twelve")
    expected_counts = (2, 10, 10, 10) if row["condition"] == "low" else (6, 6, 6, 14)
    if tuple(counts[key] for key in SEMANTIC_KEYS) != expected_counts:
        raise AssertionError("Unexpected independently recounted contingency table")
    if len(row["image_files"]) != 32:
        raise AssertionError("Expected exactly thirty-two image records")
    pixel_hashes = []
    for index, image_record in enumerate(row["image_files"]):
        image_path = path / f"{index:03d}.png"
        if Path(image_record["path"]) != image_path or image_path.is_symlink():
            raise AssertionError("Unexpected image path")
        if image_path.stat().st_size != image_record["bytes"] or digest(image_path) != image_record["sha256"]:
            raise AssertionError(f"Published image checksum failed: {image_path}")
        with Image.open(image_path) as picture:
            picture.load()
            if list(picture.size) != image_record["dimensions"] or picture.mode != image_record["mode"]:
                raise AssertionError("Image shape/mode differs from manifest")
            rgb = picture.convert("RGB")
            payload = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
            pixel_hashes.append(hashlib.sha256(payload).hexdigest())
    return dict(frames=frames, pixel_sha256=pixel_hashes,
                content_sha256=parsed["content_sha256"], **hist)


def independently_audit_pairs(rows: list[dict], root: Path, excluded: set[str]) -> list[dict]:
    audited, seen = [], set()
    for index in range(PAIR_COUNT):
        low, high = rows[2 * index:2 * index + 2]
        if low["condition"] != "low" or high["condition"] != "high" or low["pair_id"] != high["pair_id"]:
            raise AssertionError("Pair order/condition mismatch")
        left, right = [independently_audit_row(row, root) for row in (low, high)]
        for checked in (left, right):
            sha = checked["content_sha256"]
            if sha in excluded or sha in seen:
                raise AssertionError("Duplicate content in independently parsed published data")
            seen.add(sha)
        for key in ("target_character", "target_room", "question", "changed_positions", "room_swap_pairs"):
            if low[key] != high[key]:
                raise AssertionError(f"Pair metadata disagrees: {key}")
        if left["character_histogram"] != right["character_histogram"] or left["room_histogram"] != right["room_histogram"]:
            raise AssertionError("Full marginals changed within a pair")
        if [c for c, _ in left["frames"]] != [c for c, _ in right["frames"]]:
            raise AssertionError("Character changed at a frame position")
        semantic_changes = [i for i, (a, b) in enumerate(zip(left["frames"], right["frames"])) if a != b]
        pixel_changes = [i for i, (a, b) in enumerate(zip(left["pixel_sha256"], right["pixel_sha256"])) if a != b]
        byte_changes = [i for i, (a, b) in enumerate(zip(low["image_files"], high["image_files"])) if a["sha256"] != b["sha256"]]
        changed = low["changed_positions"]
        if len(changed) != 8 or sorted(set(changed)) != changed or semantic_changes != changed or pixel_changes != changed or byte_changes != changed:
            raise AssertionError("The eight semantic/image interventions do not agree")
        if sorted(i for pair in low["room_swap_pairs"] for i in pair) != changed or len(low["room_swap_pairs"]) != 4:
            raise AssertionError("Invalid room-swap pairing")
        for c_index, r_index in low["room_swap_pairs"]:
            (c_left, r_left), (c_right, r_right) = left["frames"][c_index], left["frames"][r_index]
            if c_left != low["target_character"] or r_left == low["target_room"] or c_right == low["target_character"] or r_right != low["target_room"]:
                raise AssertionError("Swap sources are not C-only and R-only")
            if right["frames"][c_index] != (c_left, r_right) or right["frames"][r_index] != (c_right, r_left):
                raise AssertionError("The room assignments were not swapped exactly")
        audited.append(dict(pair_id=low["pair_id"], low_sid=low["sid"], high_sid=high["sid"],
                            gold_delta=4, changed_positions=changed, unchanged_frames=24,
                            character_identity_preserved_at_every_position=True,
                            complete_character_and_room_marginals_preserved=True,
                            independently_recounted_gold=[2, 6],
                            low_rgb_pixel_sha256=left["pixel_sha256"],
                            high_rgb_pixel_sha256=right["pixel_sha256"]))
    if len(seen) != 32:
        raise AssertionError("Expected thirty-two independently audited unique samples")
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Stage and audit the binding diagnostic only inside a Slurm CPU allocation")
    if args.dataset_root != DEFAULT_ROOT or args.dataset_root.resolve() != DEFAULT_ROOT:
        raise ValueError(f"This bounded diagnostic writes only to {DEFAULT_ROOT}")
    if not 1 <= args.workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Workers exceed the Slurm CPU allocation")
    if args.dataset_root.exists() or args.dataset_root.is_symlink():
        raise ValueError("Refusing to alter an existing diagnostic dataset root")
    started = time.monotonic()
    excluded, ledger, reference = exclusions()
    used = set(excluded)
    seed = stable_seed(args.seed, "v3_binding_fixed_marginals_K2_K6")
    rng = random.Random(seed)
    rows = [row for index in range(PAIR_COUNT) for row in generate_pair(index, rng, used, seed)]
    if len(used) != len(excluded) + 32:
        raise AssertionError("Generation did not produce exactly thirty-two fresh samples")
    args.dataset_root.mkdir(parents=True, exist_ok=False)
    renderer, renderer_provenance = prepare_renderer(reference["splits"]["train_N8"]["samples"])
    renderer_provenance = dict(renderer_provenance, method="Use the existing MMReD renderer after RGB parity checks on selected V1 reference frames")
    cache_root = args.dataset_root / "render_cache"
    cache_root.mkdir()
    specs = sorted({(c, r, i + 1) for row in rows for i, (c, r) in enumerate(row["frames"])})
    print(json.dumps(dict(pairs=16, samples=32, frames=1024, unique_renders=len(specs),
                          excluded_content=len(excluded))), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, cache_root, renderer), specs))
        published = list(pool.map(lambda row: publish_sample(row, args.dataset_root, cache), rows))
    for source, row in zip(rows, published):
        row.update(origin="generated_v3_fixed_marginal_binding", question=source["question"])
    checked = independently_audit_pairs(published, args.dataset_root, excluded)
    if any(digest(Path(path)) != item["sha256"] for path, item in ledger.items()):
        raise AssertionError("An existing exclusion manifest changed during staging")
    code_paths = ("scripts/stage_native_vision_binding_probe.py",
                  "slurm/native_aggregation_vision_binding_probe_stage.sbatch",
                  "scripts/stage_native_vision_v2_clean.py",
                  "scripts/stage_native_vision_pilot.py",
                  "scripts/audit_native_vision_data.py",
                  "datasets/mmred/render_mmred.py")
    code_hashes = {path: digest(REPO / path) for path in code_paths}
    audit = dict(schema_version=1, generator_code_sha256=code_hashes, data_seed=args.seed,
                 derived_seed=seed, pairs=16, samples=32, images=1024,
                 excluded_unique_content=len(excluded), exclusion_ledger=ledger,
                 existing_manifest_hashes_unchanged=True,
                 all_published_qa_and_image_hashes_verified=True,
                 all_gold_labels_independently_recounted=True,
                 all_new_content_independently_verified_disjoint=True,
                 all_pairs_exactly_eight_pixel_changes=True,
                 all_pairs_character_identity_preserved_at_every_position=True,
                 all_pairs_full_character_and_room_marginals_preserved=True,
                 renderer_provenance=renderer_provenance, pairs_audit=checked)
    exclusive_json(args.dataset_root / "audit.json", audit)
    pairs = [dict(pair_id=low["pair_id"], low_sid=low["sid"], high_sid=high["sid"],
                  target_character=low["target_character"], target_room=low["target_room"],
                  marginal_C=12, marginal_R=12, gold_delta=4,
                  changed_positions=low["changed_positions"],
                  room_swap_pairs=low["room_swap_pairs"])
             for low, high in zip(published[::2], published[1::2])]
    manifest = dict(schema_version=1, purpose="fixed_marginal_binding", dataset_root=str(args.dataset_root),
                    data_seed=args.seed, derived_seed=seed, generator_law=LAW,
                    splits={"test_N32": dict(samples=published, count=32, gold_histogram={2: 16, 6: 16})},
                    pairs=pairs, paired_test_unit="pair_id", changed_positions_indexing="zero-based",
                    generator_code_sha256=code_hashes, excluded_manifest_sha256={path: row["sha256"] for path, row in ledger.items()},
                    renderer_provenance=renderer_provenance, descriptive_only=True,
                    audit_file=str(args.dataset_root / "audit.json"),
                    audit_sha256=digest(args.dataset_root / "audit.json"))
    manifest_path = args.dataset_root / "manifest.json"
    exclusive_json(manifest_path, manifest)
    summary = dict(schema_version=1, slurm_job_id=os.environ["SLURM_JOB_ID"],
                   elapsed_seconds=time.monotonic() - started, manifest=str(manifest_path),
                   manifest_sha256=digest(manifest_path), audit=str(args.dataset_root / "audit.json"),
                   audit_sha256=digest(args.dataset_root / "audit.json"),
                   pairs=16, samples=32, images=1024, unique_renders=len(specs),
                   cached_png_bytes=sum(item["bytes"] for item in cache.values()),
                   logical_png_bytes=sum(image["bytes"] for row in published for image in row["image_files"]),
                   generator_law=LAW, generator_code_sha256=code_hashes)
    exclusive_json(OUTPUT_ROOT / "summary.json", summary)
    exclusive_json(OUTPUT_ROOT / "audit.json", audit)
    index = (
        "# Fixed-marginal conjunction diagnostic data\n\n"
        "Sixteen N32 pairs, K2 to K6, with both queried marginals fixed at twelve. "
        "Four room swaps preserve every character at its frame position and both full histograms; "
        "exactly eight images change and twenty-four remain identical.\n\n"
        "[Summary](summary.json) | [Independent audit](audit.json)\n\n"
        f"Frozen manifest: {manifest_path}\n\n"
        "This is an OOD input-counterfactual diagnostic. No model outcomes were read. "
        "Use both-members-correct and predicted count change versus the gold change of four; "
        "a fixed marginal-only guess can achieve 50% example exact and zero both-pair exact.\n"
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "INDEX.md").open("x") as stream:
        stream.write(index)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
