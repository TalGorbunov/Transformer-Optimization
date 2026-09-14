"""Independently verify generated V2 labels, pairing, exclusions and image links."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_native_vision_data import inspect, summarize
from scripts.stage_native_vision_pilot import DATA_BASE, digest, exclusive_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATA_BASE / "v2_clean")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/native_aggregation_vlm/v2_clean/data_audit")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Audit V2 under Slurm CPU")
    started = time.monotonic()
    v1_content = set()
    for purpose in ("main", "profile"):
        manifest = json.loads((DATA_BASE / "native_aggregation_vision_pilot" / f"{purpose}_manifest.json").read_text())
        v1_content.update(row["content_sha256"] for group in manifest["splits"].values() for row in group["samples"])
    cells, samples, metadata, semantics, seen, caches = {}, [], {}, {}, set(), {}
    image_count, families = 0, defaultdict(dict)
    purposes = ["main", "count", "profile"]
    if (args.dataset_root / "profile_count_manifest.json").exists():
        purposes.append("profile_count")
    for purpose in purposes:
        manifest = json.loads((args.dataset_root / f"{purpose}_manifest.json").read_text())
        for key, group in manifest["splits"].items():
            rows = []
            for record in group["samples"]:
                checked = inspect(record, purpose, key)
                rows.append(checked)
                samples.append(checked)
                raw = (Path(record["path"]) / "qa.txt").read_text()
                states = [ast.literal_eval(line) for line in raw.splitlines() if line.startswith("{")]
                question = f"How many frames show {checked['target_character']} in the {checked['target_room']}?"
                serialized = json.dumps(dict(states=states, question=question), sort_keys=True, separators=(",", ":"))
                sha = hashlib.sha256(serialized.encode()).hexdigest()
                if sha != record["content_sha256"] or sha in v1_content or sha in seen:
                    raise AssertionError(f"Content exclusion/hash failed: {record['sid']}")
                seen.add(sha)
                frames = [(next(iter(state["rooms"].values()))[0], next(iter(state["rooms"]))) for state in states]
                metadata[record["sid"]], semantics[record["sid"]] = record, frames
                if record.get("pair_id"):
                    families[record["pair_id"]][record["n_frames"]] = record["sid"]
                for image in record["image_files"]:
                    path, cache = Path(image["path"]), Path(image["render_cache"])
                    if not path.samefile(cache) or path.stat().st_size != image["bytes"]:
                        raise AssertionError(f"Image hardlink/size mismatch: {path}")
                    if str(cache) not in caches:
                        actual = digest(cache)
                        if actual != image["sha256"]:
                            raise AssertionError(f"Cached image hash mismatch: {cache}")
                        caches[str(cache)] = actual
                    elif caches[str(cache)] != image["sha256"]:
                        raise AssertionError(f"Inconsistent cached image hash: {cache}")
                    image_count += 1
            summary = summarize(rows)
            per_gold = defaultdict(list)
            for row in rows:
                per_gold[row["gold"]].append(row)
            summary["by_gold"] = {gold: summarize(records) for gold, records in sorted(per_gold.items())}
            cells[f"{purpose}/{key}"] = summary
            print(json.dumps(dict(cell=f"{purpose}/{key}", count=len(rows),
                                  target_character_mean=summary["target_character_frames"]["mean"],
                                  same_character_wrong_room_mean=summary["same_character_wrong_room"]["mean"],
                                  queried_room_other_character_mean=summary["queried_room_other_character"]["mean"])), flush=True)
    extension_checks = 0
    for family_id, lengths in families.items():
        anchor_n = min(lengths)
        anchor = semantics[lengths[anchor_n]]
        for n_frames, sid in sorted(lengths.items()):
            record, frames = metadata[sid], semantics[sid]
            if [frames[i] for i in record["anchor_positions"]] != anchor:
                raise AssertionError(f"Anchor mapping failed: {sid}")
            if n_frames == anchor_n:
                continue
            parent_sid = lengths[record["parent_n_frames"]]
            positions = record["parent_positions"]
            if positions != sorted(set(positions)) or [frames[i] for i in positions] != semantics[parent_sid]:
                raise AssertionError(f"Parent subsequence failed: {sid}")
            target = (record["target_character"], record["target_room"])
            if any(frame == target for i, frame in enumerate(frames) if i not in set(positions)):
                raise AssertionError(f"Added evidence in extension: {sid}")
            extension_checks += 1
    result = dict(schema_version=1, slurm_job_id=os.environ["SLURM_JOB_ID"], samples=len(samples),
                  images=image_count, unique_render_hashes_verified=len(caches),
                  all_gold_recounts_passed=True, all_source_qa_hashes_passed=True,
                  all_content_exclusions_passed=True, all_image_links_and_hashes_passed=True,
                  paired_families=len(families), verified_extensions=extension_checks,
                  all_parent_and_anchor_mappings_passed=True, cells=cells,
                  elapsed_seconds=time.monotonic() - started)
    exclusive_json(args.output / f"audit_{os.environ['SLURM_JOB_ID']}.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
