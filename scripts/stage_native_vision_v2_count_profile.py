"""Add four disjoint multi-digit software-profile samples without altering V2 tests."""
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
    DATA_BASE, LAW, V1_ROOT, digest, exclusive_json, make_family,
    prepare_renderer, publish_sample, render_cached, stable_seed,
)


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Generate count profiles on Slurm CPU")
    started = time.monotonic()
    root = DATA_BASE / "v2_clean"
    used, manifest_hashes = set(), {}
    for base, purposes in ((V1_ROOT, ("main", "profile")), (root, ("main", "count", "profile"))):
        for purpose in purposes:
            path = base / f"{purpose}_manifest.json"
            manifest = json.loads(path.read_text())
            manifest_hashes[str(path)] = digest(path)
            used.update(row["content_sha256"] for cell in manifest["splits"].values() for row in cell["samples"])
    v1 = json.loads((V1_ROOT / "main_manifest.json").read_text())
    renderer, provenance = prepare_renderer(v1["splits"]["train_N8"]["samples"])
    seed = stable_seed(20260911, "profile_count_multidigit")
    rng = random.Random(seed)
    rows = []
    for gold in (10, 16):
        rows.extend(make_family(rng, (32, 64), gold, f"profile_count_K{gold}", "test", "software_unseen_count", used, seed))
    specs = sorted({(c, r, i + 1) for row in rows for i, (c, r) in enumerate(row["frames"])})
    with ThreadPoolExecutor(max_workers=min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))) as pool:
        cache = dict(pool.map(lambda spec: render_cached(spec, root / "render_cache", renderer), specs))
        published = list(pool.map(lambda row: publish_sample(row, root, cache), rows))
    splits = {}
    for n_frames in (32, 64):
        selected = [row for row in published if row["n_frames"] == n_frames]
        rng.shuffle(selected)
        splits[f"test_N{n_frames}"] = dict(samples=selected, count=2, gold_histogram={10: 1, 16: 1})
    manifest = dict(schema_version=1, purpose="profile_count", dataset_root=str(root), data_seed=20260911,
                    splits=splits, generator_law=LAW, renderer_provenance=provenance,
                    existing_manifest_sha256=manifest_hashes, software_profile_only=True,
                    disjoint_from_all_main_count_profile_and_V1=True,
                    paired_test_unit="pair_id")
    exclusive_json(root / "profile_count_manifest.json", manifest)
    cost = dict(slurm_job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic() - started,
                samples=4, images=192, manifest=str(root / "profile_count_manifest.json"),
                manifest_sha256=digest(root / "profile_count_manifest.json"))
    exclusive_json(root / f"profile_count_cost_{os.environ['SLURM_JOB_ID']}.json", cost)
    print(json.dumps(cost, indent=2), flush=True)


if __name__ == "__main__":
    main()
