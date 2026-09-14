"""Generate a clean, paired MMReD Vision V2 dataset on Slurm CPU only.

All splits use one conditional law: K matches, and N-K IID distractors sampled
equally from same-person/wrong-room, other-person/same-room, and neither-match.
Paired tests extend anchors by interleaving fresh distractors. Counts 9..16 form
a separate test axis. Existing V1 data are read solely to exclude used content
and verify renderer parity. No source or existing destination file is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from scripts.stage_native_vision_pilot import (  # standard library only
    DATA_BASE, PARK_ROOMS, digest, exclusive_json, prepare_renderer,
)

CHARS = ["Sandra", "Mary", "Michael", "John", "Daniel", "Laura", "Peter", "Emma", "Noah"]
V1_ROOT = DATA_BASE / "native_aggregation_vision_pilot"
LAW = dict(name="v2_clean_conjunctive_distractors", target="uniform over 9 characters x 6 rooms",
           evidence="Exactly K copies of the queried character/room conjunction",
           distractor_type_probabilities={"same_character_wrong_room": 1 / 3,
                                          "queried_room_other_character": 1 / 3, "neither": 1 / 3},
           distractor_identity="Uniform among valid characters/rooms within the sampled type",
           order="Uniformly shuffle anchors; uniformly interleave fresh IID distractors on extension",
           finite_pool_exclusion="Reject full families colliding with any selected V1 or V2 sequence/question content",
           paired_note="Extensions preserve semantic frame order; displayed step indices are renumbered")


def stable_seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")


def distractor(rng: random.Random, character: str, room: str) -> tuple[str, str]:
    others, other_rooms = [c for c in CHARS if c != character], [r for r in PARK_ROOMS if r != room]
    kind = rng.randrange(3)
    if kind == 0:
        return character, rng.choice(other_rooms)
    if kind == 1:
        return rng.choice(others), room
    return rng.choice(others), rng.choice(other_rooms)


def interleave(rng: random.Random, old: list[tuple], extra: list[tuple]) -> tuple[list[tuple], list[int]]:
    """Uniform positions for the old subsequence; independent new distractors."""
    size = len(old) + len(extra)
    positions = sorted(rng.sample(range(size), len(old)))
    positions_set, old_iter, extra_iter = set(positions), iter(old), iter(extra)
    return [next(old_iter) if i in positions_set else next(extra_iter) for i in range(size)], positions


def describe(frames: list[tuple], character: str, room: str) -> tuple[list[dict], str, str, dict]:
    states = [dict(step_id=i + 1, rooms={r: [c]}) for i, (c, r) in enumerate(frames)]
    question = f"How many frames show {character} in the {room}?"
    normalized = json.dumps(dict(states=states, question=question), sort_keys=True, separators=(",", ":"))
    content_sha256 = hashlib.sha256(normalized.encode()).hexdigest()
    counts = Counter()
    for c, r in frames:
        kind = ("matches" if c == character and r == room else
                "same_character_wrong_room" if c == character else
                "queried_room_other_character" if r == room else "neither")
        counts[kind] += 1
    return states, question, content_sha256, {key: counts[key] for key in
        ("matches", "same_character_wrong_room", "queried_room_other_character", "neither")}


def make_family(rng: random.Random, lengths: tuple[int, ...], gold: int, label: str,
                split: str, axis: str, used: set[str], seed: int) -> list[dict]:
    for attempt in range(10000):
        character, room = rng.choice(CHARS), rng.choice(PARK_ROOMS)
        frames = [(character, room)] * gold + [distractor(rng, character, room) for _ in range(lengths[0] - gold)]
        rng.shuffle(frames)
        family, parent_positions, anchor_positions = [], [], list(range(lengths[0]))
        previous_n = None
        for n_frames in lengths:
            if previous_n is not None:
                frames, parent_positions = interleave(rng, frames, [distractor(rng, character, room) for _ in range(n_frames - previous_n)])
                anchor_positions = [parent_positions[index] for index in anchor_positions]
            states, question, content_hash, counts = describe(frames, character, room)
            if counts["matches"] != gold or len(frames) != n_frames:
                raise AssertionError("Generation changed the final count or sequence length")
            sid = f"v2c_{label}_N{n_frames}"
            family.append(dict(sid=sid, n_frames=n_frames, split=split, gold=gold, seed=seed,
                               axis=axis, anchor_id=label if len(lengths) > 1 else None,
                               pair_id=label if len(lengths) > 1 else None,
                               test_family=axis if split == "test" else None,
                               anchor_n_frames=lengths[0] if len(lengths) > 1 else None,
                               parent_n_frames=previous_n, parent_positions=parent_positions,
                               anchor_positions=anchor_positions if len(lengths) > 1 else [],
                               target_character=character, target_room=room, question=question,
                               states=states, frames=list(frames), content_sha256=content_hash,
                               semantic_counts=counts, generation_attempt=attempt))
            previous_n = n_frames
        hashes = [sample["content_sha256"] for sample in family]
        if len(set(hashes)) == len(hashes) and not any(value in used for value in hashes):
            used.update(hashes)
            return family
    raise RuntimeError(f"Could not find a fresh family for {label}; finite pool exhausted")


def render_cached(spec: tuple[str, str, int], cache_root: Path, renderer) -> tuple[tuple, dict]:
    from PIL import Image

    character, room, step = spec
    target = cache_root / f"{character}_{room}_{step:03d}.png"
    temporary = cache_root / f".{character}_{room}_{step:03d}_{os.environ['SLURM_JOB_ID']}.png"
    if temporary.exists():
        raise ValueError(f"Unexpected temporary file: {temporary}")
    try:
        renderer.render_frame({room: [character]}, step, str(temporary))
        sha = digest(temporary)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or digest(target) != sha:
                raise ValueError(f"Existing cached render differs: {target}")
        else:
            os.link(temporary, target)
        with Image.open(target) as image:
            dimensions, mode = list(image.size), image.mode
            image.verify()
        return spec, dict(path=str(target.resolve()), sha256=sha, bytes=target.stat().st_size,
                          dimensions=dimensions, mode=mode)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_sample(sample: dict, dataset_root: Path, cache: dict) -> dict:
    directory = dataset_root / "mmred_vfiltered" / f"seq_len_{sample['n_frames']}" / sample["split"] / sample["sid"]
    if not directory.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("Generated destination escapes dataset root")
    directory.mkdir(parents=True, exist_ok=True)
    qa = ("question:\n" + "\n".join(repr(state) for state in sample["states"])
          + f"\n{sample['question']}\nanswer:\n{sample['gold']}\n").encode()
    qa_path = directory / "qa.txt"
    if qa_path.exists():
        if qa_path.is_symlink() or qa_path.read_bytes() != qa:
            raise ValueError(f"Existing QA differs: {qa_path}")
    else:
        with qa_path.open("xb") as stream:
            stream.write(qa)
    qa_sha256 = hashlib.sha256(qa).hexdigest()
    if digest(qa_path) != qa_sha256:
        raise ValueError(f"QA publication checksum mismatch: {qa_path}")
    images = []
    for index, (character, room) in enumerate(sample["frames"]):
        reference = cache[character, room, index + 1]
        target = directory / f"{index:03d}.png"
        if target.exists() or target.is_symlink():
            if target.is_symlink() or digest(target) != reference["sha256"]:
                raise ValueError(f"Existing frame differs: {target}")
        else:
            os.link(reference["path"], target)
        # New links share the already hashed/validated cache inode; existing files
        # were individually hashed above. No output file is ever overwritten.
        images.append(dict(reference, path=str(target.resolve()), render_cache=reference["path"]))
    metadata = {key: value for key, value in sample.items() if key not in ("frames", "states", "question")}
    metadata.update(path=str(directory.resolve()), source_path=str(directory.resolve()),
                    origin="generated_v2_clean", qa_sha256=qa_sha256, image_files=images)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATA_BASE / "v2_clean")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Generate and audit V2 only inside a Slurm CPU allocation")
    if not args.dataset_root.resolve().is_relative_to(DATA_BASE.resolve()):
        raise ValueError(f"Dataset root must be under {DATA_BASE}")
    if not 1 <= args.workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Workers exceed the Slurm CPU allocation")
    started = time.monotonic()
    v1, v1_hashes, excluded = {}, {}, set()
    for purpose in ("main", "profile"):
        source = V1_ROOT / f"{purpose}_manifest.json"
        v1[purpose] = json.loads(source.read_text())
        v1_hashes[purpose] = digest(source)
        excluded.update(row["content_sha256"] for group in v1[purpose]["splits"].values() for row in group["samples"])
    used = set(excluded)
    groups = {purpose: defaultdict(list) for purpose in ("main", "count", "profile")}
    family_audit = []

    def generate(purpose: str, split: str, lengths: tuple[int, ...], golds, per_gold: int, axis: str) -> None:
        label = f"{purpose}_{split}_{axis}_N{lengths[0]}"
        seed = stable_seed(args.seed, label)
        rng = random.Random(seed)
        for gold in golds:
            for index in range(per_gold):
                family_label = f"{label}_K{gold}_{index:04d}"
                family = make_family(rng, lengths, gold, family_label, split, axis, used, seed)
                for sample in family:
                    groups[purpose][f"{split}_N{sample['n_frames']}"].append(sample)
                if len(family) > 1:
                    for parent, child in zip(family, family[1:]):
                        recovered = [child["frames"][position] for position in child["parent_positions"]]
                        if recovered != parent["frames"]:
                            raise AssertionError("Paired extension did not preserve the semantic subsequence")
                        inserted = [frame for i, frame in enumerate(child["frames"]) if i not in set(child["parent_positions"])]
                        _, _, _, added_counts = describe(inserted, child["target_character"], child["target_room"])
                        if added_counts["matches"]:
                            raise AssertionError("Paired extension added new evidence")
                        family_audit.append(dict(anchor_id=family_label, parent=parent["sid"], child=child["sid"],
                                                 added_semantic_counts=added_counts, preserved_subsequence=True))

    for split, per_gold in (("train", 10), ("dev", 4)):
        for n_frames in (8, 16):
            generate("main", split, (n_frames,), range(9), per_gold, "familiar")
    generate("main", "test", (16, 32, 64), range(9), 12, "length")
    generate("count", "test", (32, 64), range(9, 17), 8, "unseen_count")
    generate("profile", "train", (16,), (3, 6), 1, "software")
    generate("profile", "dev", (16,), (3, 6), 1, "software")
    generate("profile", "test", (8,), (3, 6), 1, "software")
    generate("profile", "test", (16, 32, 64), (3, 6), 1, "software")
    all_samples = [row for cells in groups.values() for rows in cells.values() for row in rows]
    if len(used) != len(excluded) + len(all_samples):
        raise AssertionError("Generated examples are not content-disjoint from V1 or each other")
    renderer, renderer_provenance = prepare_renderer(v1["main"]["splits"]["train_N8"]["samples"])
    specs = sorted({(character, room, i + 1) for row in all_samples for i, (character, room) in enumerate(row["frames"])})
    cache_root = args.dataset_root / "render_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(samples=len(all_samples), images=sum(row["n_frames"] for row in all_samples),
                          unique_renders=len(specs), excluded_v1_content=len(excluded))), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, cache_root, renderer), specs))
        manifests = {}
        for purpose, cells in groups.items():
            published = {}
            for key, rows in sorted(cells.items()):
                random.Random(stable_seed(args.seed, f"shuffle:{purpose}:{key}")).shuffle(rows)
                records = list(pool.map(lambda row: publish_sample(row, args.dataset_root, cache), rows))
                hist = dict(sorted(Counter(row["gold"] for row in records).items()))
                semantic_hist = {metric: dict(sorted(Counter(row["semantic_counts"][metric] for row in records).items()))
                                 for metric in ("matches", "same_character_wrong_room", "queried_room_other_character", "neither")}
                published[key] = dict(samples=records, count=len(records), gold_histogram=hist,
                                      semantic_count_histograms=semantic_hist)
                print(json.dumps(dict(purpose=purpose, group=key, samples=len(records), gold_histogram=hist)), flush=True)
            manifests[purpose] = dict(schema_version=1, purpose=purpose, dataset_root=str(args.dataset_root.resolve()),
                                     data_seed=args.seed, splits=published, generator_law=LAW,
                                     renderer_provenance=renderer_provenance,
                                     v1_manifest_sha256=v1_hashes, main_profile_content_disjoint=True,
                                     paired_test_unit="anchor_id; resample whole anchor families for comparisons across lengths",
                                     audit_file=str((args.dataset_root / "audit.json").resolve()))
    provenance = {name: digest(REPO / name) for name in
                  ("scripts/stage_native_vision_v2_clean.py", "scripts/stage_native_vision_pilot.py", "datasets/mmred/render_mmred.py")}
    audit = dict(schema_version=1, generator_law=LAW, generator_code_sha256=provenance,
                 data_seed=args.seed, samples=len(all_samples),
                 images=sum(row["n_frames"] for row in all_samples), unique_renders=len(specs),
                 excluded_v1_content=len(excluded), v1_manifest_sha256=v1_hashes,
                 all_gold_labels_recounted=True, all_paired_subsequences_verified=True,
                 all_selected_content_unique=True, renderer_provenance=renderer_provenance,
                 paired_extensions=family_audit,
                 storage="Immutable PNG cache hardlinked into sample directories; hashes identify exact image bytes")
    exclusive_json(args.dataset_root / "audit.json", audit)
    for purpose, manifest in manifests.items():
        exclusive_json(args.dataset_root / f"{purpose}_manifest.json", manifest)
    cost = dict(slurm_job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic() - started,
                cached_png_bytes=sum(row["bytes"] for row in cache.values()),
                logical_png_bytes=sum(cache[c, r, i + 1]["bytes"] for row in all_samples for i, (c, r) in enumerate(row["frames"])),
                samples=len(all_samples), images=audit["images"], unique_renders=len(specs),
                manifests={purpose: dict(path=str(args.dataset_root / f"{purpose}_manifest.json"),
                                         sha256=digest(args.dataset_root / f"{purpose}_manifest.json")) for purpose in manifests})
    exclusive_json(args.dataset_root / f"staging_cost_{os.environ['SLURM_JOB_ID']}.json", cost)
    print(json.dumps(cost, indent=2), flush=True)


if __name__ == "__main__":
    main()
