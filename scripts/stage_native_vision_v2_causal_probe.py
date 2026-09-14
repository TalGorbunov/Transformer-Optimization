"""Create 16 matched K2/K6 image pairs for descriptive channel-interchange probes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.stage_native_vision_v2_clean import (
    DATA_BASE, LAW, V1_ROOT, describe, digest, exclusive_json, make_family,
    prepare_renderer, publish_sample, render_cached, stable_seed,
)
from scripts.audit_native_vision_data import inspect


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Generate and audit causal probe data under Slurm CPU")
    started = time.monotonic()
    root = DATA_BASE / "v2_clean"
    used, source_hashes = set(), {}
    for base, purposes in ((V1_ROOT, ("main", "profile")),
                           (root, ("main", "count", "profile", "profile_count"))):
        for purpose in purposes:
            source = base / f"{purpose}_manifest.json"
            manifest = json.loads(source.read_text())
            source_hashes[str(source)] = digest(source)
            used.update(row["content_sha256"] for cell in manifest["splits"].values() for row in cell["samples"])
    initial_excluded = set(used)
    seed = stable_seed(20260911, "causal_probe_K2_K6")
    rng = random.Random(seed)
    pairs = []
    for index in range(16):
        pair_id = f"causal_probe_{index:04d}"
        for attempt in range(10000):
            temporary_used = set(used)
            low = make_family(rng, (32,), 2, f"{pair_id}_K2", "probe", "causal_count", temporary_used, seed)[0]
            target = low["target_character"], low["target_room"]
            negative_positions = [i for i, frame in enumerate(low["frames"]) if frame != target]
            changed = sorted(rng.sample(negative_positions, 4))
            frames = list(low["frames"])
            for position in changed:
                frames[position] = target
            states, question, sha, counts = describe(frames, *target)
            if sha in temporary_used:
                continue
            high = dict(low, sid=f"v2c_{pair_id}_K6_N32", gold=6, states=states, frames=frames,
                        question=question, content_sha256=sha, semantic_counts=counts)
            for row in (low, high):
                row.update(pair_id=pair_id, anchor_id=None, test_family="causal_probe",
                           changed_positions=changed, pair_generation_attempt=attempt)
            if low["semantic_counts"]["matches"] != 2 or counts["matches"] != 6:
                raise AssertionError("Counterfactual count did not change from2to6")
            if [i for i, (left, right) in enumerate(zip(low["frames"], frames)) if left != right] != changed:
                raise AssertionError("Unexpected semantic frame difference")
            used.update((low["content_sha256"], high["content_sha256"]))
            pairs.append(dict(pair_id=pair_id, n_frames=32, changed_positions=changed,
                              question=question, target_character=target[0], target_room=target[1],
                              low=low, high=high))
            break
        else:
            raise RuntimeError(f"Could not generate disjoint pair {pair_id}")
    v1 = json.loads((V1_ROOT / "main_manifest.json").read_text())
    renderer, provenance = prepare_renderer(v1["splits"]["train_N8"]["samples"])
    rows = [pair[side] for pair in pairs for side in ("low", "high")]
    specs = sorted({(c, r, i + 1) for row in rows for i, (c, r) in enumerate(row["frames"])})
    with ThreadPoolExecutor(max_workers=min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, root / "render_cache", renderer), specs))
        published = list(pool.map(lambda row: publish_sample(row, root, cache), rows))
    published_by_sid = {row["sid"]: row for row in published}
    checked_pairs = []
    for pair in pairs:
        low, high = [published_by_sid[pair[side]["sid"]] for side in ("low", "high")]
        low["question"] = high["question"] = pair["question"]
        for row in (low, high):
            inspect(row, "causal_probe", "probe_N32")
        changed_pixels = [i for i, (left, right) in enumerate(zip(low["image_files"], high["image_files"]))
                          if left["sha256"] != right["sha256"]]
        if changed_pixels != pair["changed_positions"]:
            raise AssertionError(f"Pixel differences are not exactly the four interventions: {pair['pair_id']}")
        checked_pairs.append(dict(pair_id=pair["pair_id"], n_frames=32, low=low, high=high,
                                  question=pair["question"], target_character=pair["target_character"],
                                  target_room=pair["target_room"], changed_positions=pair["changed_positions"]))
    manifest = dict(schema_version=1, purpose="causal_probe", dataset_root=str(root), pairs=checked_pairs,
                    data_seed=20260911, derived_seed=seed, generator_law=LAW,
                    intervention="Uniformly choose four negative positions in the K2 sample and replace them by matching evidence",
                    descriptive_only=True, renderer_provenance=provenance,
                    excluded_manifest_sha256=source_hashes,
                    existing_main_count_profile_manifests_unchanged=True)
    exclusive_json(root / "causal_probe_manifest.json", manifest)
    if any(digest(Path(path)) != sha for path, sha in source_hashes.items()):
        raise AssertionError("An existing manifest changed during probe generation")
    audit = dict(slurm_job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic() - started,
                 pairs=16, samples=32, images=1024, low_gold=2, high_gold=6,
                 all_gold_recounts_verified=True, all_qa_hashes_verified=True,
                 all_exactly_four_semantic_and_pixel_changes_verified=True,
                 unchanged_frames_per_pair=28, all_content_exclusions_verified=len(used) == len(initial_excluded) + 32,
                 existing_manifest_hashes_unchanged=True,
                 manifest=str(root / "causal_probe_manifest.json"),
                 manifest_sha256=digest(root / "causal_probe_manifest.json"))
    out = REPO / "outputs/native_aggregation_vlm/v2_clean/causal_probe_data"
    exclusive_json(out / f"audit_{os.environ['SLURM_JOB_ID']}.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
