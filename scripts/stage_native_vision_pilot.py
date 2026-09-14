"""Stage a fixed, audited MMReD Vision pilot on a Slurm CPU allocation.

Only final gold counts are used for stratification. States are used solely for
the split-leakage audit, never copied into the manifest or model supervision.
Existing source and destination files are never modified. Reusing an existing
destination requires matching source SHA256. Manifest sample order is fixed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]
DATA_BASE = Path("/mnt/data/gabriele/gnn_transformer")
NS = (8, 16, 32, 64)
SPLITS = ("train", "dev", "test")
PARK_ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def exclusive_json(path: Path, value) -> None:
    """Permit an exact repeat; never replace an existing experiment manifest."""
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace existing manifest: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def read_candidate(path: Path, split: str, n_frames: int) -> dict:
    if path.is_symlink() or (path / "qa.txt").is_symlink():
        raise ValueError(f"Unexpected symlink: {path}")
    raw = (path / "qa.txt").read_bytes()
    lines = raw.decode().splitlines()
    q_start = next(i for i, line in enumerate(lines) if line.strip() == "question:")
    a_start = next(i for i, line in enumerate(lines) if line.strip() == "answer:")
    if a_start <= q_start:
        raise ValueError("Answer precedes question")
    block = [line.strip() for line in lines[q_start + 1:a_start] if line.strip()]
    states = [ast.literal_eval(line) for line in block
              if line.startswith("{") and line.endswith("}")]
    question = [line for line in block
                if not (line.startswith("{") and line.endswith("}"))]
    answer = next(line.strip() for line in lines[a_start + 1:] if line.strip())
    if len(states) != n_frames or len(question) != 1 or not answer.isdigit():
        raise ValueError("Invalid states/question/answer metadata")
    # Step order is significant; serialization/key order and question whitespace
    # are not. The answer is deliberately absent from this duplicate key.
    normalized = dict(states=states, question=" ".join(question[0].split()))
    content_hash = hashlib.sha256(json.dumps(
        normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(source_path=str(path), sid=path.name, split=split,
                n_frames=n_frames, gold=int(answer),
                qa_sha256=hashlib.sha256(raw).hexdigest(),
                content_sha256=content_hash)


def location(sample: dict) -> dict:
    return {key: sample[key] for key in ("split", "n_frames", "sid")}


def duplicate_groups(candidates: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for sample in candidates:
        groups[sample[field]].append(sample)
    return [dict(key=key, samples=[location(s) for s in rows])
            for key, rows in sorted(groups.items())
            if len({row["split"] for row in rows}) > 1]


def select(candidates: list[dict], count: int, seed: int,
           excluded_ids: set[tuple], used_content: set[str]) -> tuple[list[dict], list[dict]]:
    """Round-robin over gold strata, redistributing only exhausted strata."""
    buckets = defaultdict(deque)
    rows = sorted(candidates, key=lambda sample: sample["sid"])
    random.Random(seed).shuffle(rows)
    skipped = []
    for sample in rows:
        if (sample["split"], sample["n_frames"], sample["sid"]) in excluded_ids:
            skipped.append(dict(**location(sample), reason="prior_V0"))
        elif not 0 <= sample["gold"] <= 8:
            skipped.append(dict(**location(sample), reason="gold_outside_0_8"))
        else:
            buckets[sample["gold"]].append(sample)
    gold_order = list(range(9))
    # Shared stratum order gives identical remainder allocation across lengths
    # when all counts are available. Individual sample order uses the group seed.
    random.Random(20260910).shuffle(gold_order)
    chosen = []
    while len(chosen) < count:
        before = len(chosen)
        for gold in gold_order:
            while buckets[gold]:
                sample = buckets[gold].popleft()
                if sample["content_sha256"] in used_content:
                    skipped.append(dict(**location(sample), reason="duplicate_selected_content"))
                    continue
                used_content.add(sample["content_sha256"])
                chosen.append(sample)
                break
            if len(chosen) == count:
                break
        if len(chosen) == before:
            raise ValueError(f"Only {len(chosen)} nonduplicate eligible examples; need {count}")
    # Avoid presenting training/evaluation in a deterministic gold cycle.
    random.Random(seed + 1).shuffle(chosen)
    return chosen, skipped


def source_states(source: Path) -> list[dict]:
    lines = (source / "qa.txt").read_text().splitlines()
    return [ast.literal_eval(line.strip()) for line in lines
            if line.strip().startswith("{") and line.strip().endswith("}")]


def prepare_renderer(reference_samples: list[dict]) -> tuple[object, dict]:
    """Verify the existing renderer's RGB pixels before making new data."""
    from PIL import Image, __version__ as pillow_version

    renderer_path = REPO / "datasets/mmred/render_mmred.py"
    sys.path.insert(0, str(renderer_path.parent))
    import render_mmred

    render_mmred.ROOMS = list(PARK_ROOMS)
    fonts = []
    for size in (12, 16):
        font = render_mmred.load_font(size)
        path = getattr(font, "path", None)
        if isinstance(path, bytes):
            path = path.decode()
        font_path = Path(path).resolve() if path else None
        if font_path is None or not font_path.is_file():
            raise ValueError("Cannot record exact renderer font file")
        fonts.append(dict(size=size, path=str(font_path), sha256=digest(font_path)))
    parity = []
    with tempfile.TemporaryDirectory(prefix="native_vision_render_parity_") as scratch:
        for sample in reference_samples[:2]:
            source = Path(sample["source_path"])
            states = source_states(source)
            for index in (0, len(states) // 2, len(states) - 1):
                rendered = Path(scratch) / f"{sample['sid']}_{index:03d}.png"
                state = states[index]
                render_mmred.render_frame(state["rooms"], state["step_id"], str(rendered))
                existing = source / f"{index:03d}.png"
                with Image.open(existing) as original, Image.open(rendered) as new:
                    original_rgb, new_rgb = original.convert("RGB"), new.convert("RGB")
                    if original_rgb.size != new_rgb.size or original_rgb.tobytes() != new_rgb.tobytes():
                        raise ValueError(f"Renderer RGB pixel parity failed: {existing}")
                    pixels_hash = hashlib.sha256(new_rgb.tobytes()).hexdigest()
                parity.append(dict(source=str(existing), source_sha256=digest(existing),
                                   rendered_sha256=digest(rendered), rgb_pixels_sha256=pixels_hash,
                                   pixel_parity=True))
    if not parity:
        raise ValueError("Need existing N8 frames for renderer parity verification")
    provenance = dict(renderer_path=str(renderer_path), renderer_sha256=digest(renderer_path),
                      pillow_version=pillow_version, room_order=PARK_ROOMS, fonts=fonts,
                      parity_checks=parity,
                      method="Render original canonical N16 train/dev states; retain original QA bytes and final gold")
    return render_mmred, provenance


def stage_sample(sample: dict, destination_root: Path, seed: int, renderer=None) -> dict:
    from PIL import Image

    source = Path(sample["source_path"])
    destination = destination_root / f"seq_len_{sample['n_frames']}" / sample["split"] / sample["sid"]
    if not destination.resolve().is_relative_to(DATA_BASE.resolve()):
        raise ValueError(f"Destination outside requested data root: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    image_files = []
    copied_bytes = 0
    render_missing = sample.get("origin") == "canonical_text_rendered"
    states = source_states(source) if render_missing else None
    if render_missing and (renderer is None or len(states) != sample["n_frames"]):
        raise ValueError(f"Missing renderer or invalid canonical states: {source}")
    all_files = [source / "qa.txt"] + [source / f"{i:03d}.png" for i in range(sample["n_frames"])]
    for src in all_files:
        if render_missing and src.suffix == ".png":
            index = int(src.stem)
            target = destination / src.name
            temporary = target.with_name(f".{target.stem}.render-{os.environ['SLURM_JOB_ID']}.png")
            if temporary.exists():
                raise ValueError(f"Unexpected rendering temporary file: {temporary}")
            state = states[index]
            if set(state["rooms"]) - set(PARK_ROOMS):
                raise ValueError(f"Unknown room in canonical source: {source}")
            try:
                renderer.render_frame(state["rooms"], int(state["step_id"]), str(temporary))
                sha = digest(temporary)
                if target.exists() or target.is_symlink():
                    if target.is_symlink() or not target.is_file() or digest(target) != sha:
                        raise ValueError(f"Existing rendered destination differs: {target}")
                else:
                    os.link(temporary, target)
                    copied_bytes += target.stat().st_size
                if digest(target) != sha:
                    raise ValueError(f"Rendered destination checksum mismatch: {target}")
                with Image.open(target) as picture:
                    dimensions, mode = list(picture.size), picture.mode
                    picture.verify()
                image_files.append(dict(path=str(target.resolve()), bytes=target.stat().st_size,
                                        sha256=sha, dimensions=dimensions, mode=mode))
            finally:
                if temporary.exists():
                    temporary.unlink()
            continue
        if src.is_symlink() or not src.is_file():
            raise ValueError(f"Missing file or symlink: {src}")
        target = destination / src.name
        source_hash = digest(src)
        if src.name == "qa.txt" and source_hash != sample["qa_sha256"]:
            raise ValueError(f"Source changed since audit: {src}")
        if target.is_symlink():
            raise ValueError(f"Unexpected destination symlink: {target}")
        if target.exists():
            if not target.is_file() or digest(target) != source_hash:
                raise ValueError(f"Existing destination differs: {target}")
        else:
            temporary = target.with_name(f".{target.name}.stage-{os.environ['SLURM_JOB_ID']}.tmp")
            try:
                with src.open("rb") as reader, temporary.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                if digest(temporary) != source_hash:
                    raise ValueError(f"Copy checksum mismatch: {target}")
                try:
                    os.link(temporary, target)  # Atomic publish; never overwrite.
                    copied_bytes += target.stat().st_size
                except FileExistsError:
                    if target.is_symlink() or digest(target) != source_hash:
                        raise ValueError(f"Concurrent destination differs: {target}")
            finally:
                if temporary.exists():
                    temporary.unlink()
        if digest(target) != source_hash:
            raise ValueError(f"Destination checksum mismatch: {target}")
        if src.suffix == ".png":
            with Image.open(target) as picture:
                dimensions, mode = list(picture.size), picture.mode
                picture.verify()
            image_files.append(dict(path=str(target.resolve()), bytes=target.stat().st_size,
                                    sha256=source_hash, dimensions=dimensions, mode=mode))
    return dict(path=str(destination.resolve()), sid=sample["sid"],
                n_frames=sample["n_frames"], gold=sample["gold"], split=sample["split"],
                seed=seed, qa_sha256=sample["qa_sha256"],
                content_sha256=sample["content_sha256"], image_files=image_files,
                source_path=str(source), origin=sample.get("origin", "existing_vision"),
                copied_bytes=copied_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO / "data/mmred_vfiltered")
    parser.add_argument("--destination-root", type=Path, default=DATA_BASE / "mmred_vfiltered")
    parser.add_argument("--manifest-dir", type=Path, default=DATA_BASE / "native_aggregation_vision_pilot")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-missing-n16", action="store_true",
                        help="Render missing N16 train/dev from canonical text after existing-image pixel parity")
    parser.add_argument("--canonical-text-root", type=Path, default=REPO / "data/mmred_filtered_train")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    # All dataset enumeration, metadata reads, hashing, copies, and image checks
    # happen after this guard. Help/imports remain lightweight on the login node.
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit MMReD staging and auditing through Slurm CPU")
    for path in (args.destination_root, args.manifest_dir):
        if not path.resolve().is_relative_to(DATA_BASE.resolve()):
            raise ValueError(f"Destination must be under {DATA_BASE}: {path}")
    if not 1 <= args.workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Workers must fit the Slurm CPU allocation")
    started = time.monotonic()
    v0_path = REPO / "outputs/native_aggregation_vlm/staging_433137.json"
    v0 = json.loads(v0_path.read_text())
    excluded = {(row["split"], 8, row["sid"]) for row in v0["samples"]}
    candidates, populations, rejected = {}, {}, []
    for split in SPLITS:
        for n_frames in NS:
            key = f"{split}_N{n_frames}"
            root = args.source_root / f"seq_len_{n_frames}" / split
            origin = "existing_vision"
            if not root.is_dir() and args.render_missing_n16 and n_frames == 16 and split in ("train", "dev"):
                root = args.canonical_text_root / f"seq_len_{n_frames}" / split
                origin = "canonical_text_rendered"
            rows = []
            directories = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
            for path in directories:
                try:
                    sample = read_candidate(path, split, n_frames)
                    sample["origin"] = origin
                    rows.append(sample)
                except (ValueError, SyntaxError, StopIteration, UnicodeError, OSError) as error:
                    rejected.append(dict(path=str(path), reason=f"{type(error).__name__}: {error}"))
            candidates[key] = rows
            populations[key] = dict(source_root=str(root.resolve()), source_exists=root.is_dir(), origin=origin,
                                    directories=len(directories), valid_metadata=len(rows),
                                    gold_histogram=dict(sorted(Counter(row["gold"] for row in rows).items())),
                                    v0_excluded=sum((split, n_frames, row["sid"]) in excluded for row in rows))
            print(json.dumps(dict(population=key, **populations[key])), flush=True)
    all_candidates = [sample for group in candidates.values() for sample in group]
    audit = dict(schema_version=1, data_seed=args.seed,
                 source_root=str(args.source_root.resolve()),
                 populations=populations, rejected_metadata=rejected,
                 prior_v0_manifest=str(v0_path), prior_v0_manifest_sha256=digest(v0_path),
                 excluded_v0_ids=[dict(split=s, n_frames=n, sid=sid) for s, n, sid in sorted(excluded)],
                 cross_split_duplicate_sid=duplicate_groups(all_candidates, "sid"),
                 cross_split_duplicate_qa=duplicate_groups(all_candidates, "qa_sha256"),
                 cross_split_duplicate_content=duplicate_groups(all_candidates, "content_sha256"),
                 duplicate_key="JSON canonical full ordered states plus whitespace-normalized question; answer excluded")
    if args.audit_only:
        out = args.manifest_dir / f"audit_only_{os.environ['SLURM_JOB_ID']}.json"
        exclusive_json(out, audit)
        print(f"Audit only complete: {out}", flush=True)
        return
    selected, skipped, used = {}, {}, set()
    main_plan = [("train", n, 90) for n in (8, 16)] + [("dev", n, 36) for n in (8, 16)] + [("test", n, 100) for n in NS]
    # Profile examples are separate from *all* main samples. This keeps the main
    # test predictions sealed while checking all evaluation lengths in advance.
    profile_plan = [("train", 16, 2), ("dev", 16, 2)] + [("test", n, 2) for n in NS]
    for purpose, plan in (("main", main_plan), ("profile", profile_plan)):
        selected[purpose], skipped[purpose] = {}, {}
        for split, n_frames, count in plan:
            key = f"{split}_N{n_frames}"
            group_seed = args.seed + n_frames + 1000 * SPLITS.index(split) + (100000 if purpose == "profile" else 0)
            try:
                rows, rejected_rows = select(candidates[key], count, group_seed, excluded, used)
            except ValueError as error:
                raise ValueError(f"{purpose}/{key}: {error}") from error
            selected[purpose][key] = dict(rows=rows, seed=group_seed)
            skipped[purpose][key] = rejected_rows
    audit["selection_skips"] = skipped
    renderer, render_provenance = None, None
    if any(row.get("origin") == "canonical_text_rendered"
           for purpose in selected.values() for group in purpose.values() for row in group["rows"]):
        renderer, render_provenance = prepare_renderer(candidates["train_N8"])
        audit["render_provenance"] = render_provenance
    manifests, total_copied, total_bytes = {}, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for purpose in ("main", "profile"):
            groups = {}
            for key, group in selected[purpose].items():
                rows = list(executor.map(lambda row: stage_sample(row, args.destination_root, group["seed"], renderer), group["rows"]))
                total_copied += sum(row.pop("copied_bytes") for row in rows)
                total_bytes += sum(frame["bytes"] for row in rows for frame in row["image_files"])
                groups[key] = dict(samples=rows, count=len(rows), seed=group["seed"],
                                   gold_histogram=dict(sorted(Counter(row["gold"] for row in rows).items())))
                print(json.dumps(dict(staged=purpose + "/" + key, count=len(rows),
                                      gold_histogram=groups[key]["gold_histogram"])), flush=True)
            manifests[purpose] = dict(schema_version=1, purpose=purpose,
                                     dataset_root=str(DATA_BASE.resolve()),
                                     data_seed=args.seed, splits=groups,
                                     render_provenance=render_provenance,
                                     audit_file=str((args.manifest_dir / "audit.json").resolve()),
                                     selection="Gold 0..8 round-robin, seeded shuffle, V0 and selected-content duplicates excluded",
                                     profile_disjoint_from_main=True)
    audit["selected_content_unique"] = len(used)
    audit["selected_count"] = sum(len(group["rows"]) for purpose in selected.values() for group in purpose.values())
    if audit["selected_content_unique"] != audit["selected_count"]:
        raise AssertionError("Selected sample content is not unique")
    exclusive_json(args.manifest_dir / "audit.json", audit)
    for purpose, manifest in manifests.items():
        exclusive_json(args.manifest_dir / f"{purpose}_manifest.json", manifest)
    cost = dict(slurm_job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic() - started,
                selected_samples=audit["selected_count"], image_bytes=total_bytes, copied_bytes=total_copied,
                manifests={name: dict(path=str(args.manifest_dir / f"{name}_manifest.json"),
                                     sha256=digest(args.manifest_dir / f"{name}_manifest.json")) for name in manifests})
    exclusive_json(args.manifest_dir / f"staging_cost_{os.environ['SLURM_JOB_ID']}.json", cost)
    print(json.dumps(cost, indent=2), flush=True)


if __name__ == "__main__":
    main()
