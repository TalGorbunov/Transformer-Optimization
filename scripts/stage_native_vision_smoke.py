"""Copy six V0 image samples on Slurm; preserve sources and verify existing files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.native_aggregation_vlm import sample_metadata


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit image staging through Slurm")
    destination_root = Path("/mnt/data/gabriele/gnn_transformer")
    plans = []
    for split in ("train", "dev", "test"):
        source_root = REPO / "data/mmred_vfiltered/seq_len_8" / split
        directories = sorted(p for p in source_root.iterdir() if p.is_dir() and (p / "qa.txt").exists())
        random.Random(1242).shuffle(directories)
        selected = 0
        for source in directories:
            metadata = sample_metadata(source, 8)
            if metadata["gold"] > 8:
                continue
            destination = destination_root / "mmred_vfiltered/seq_len_8" / split / source.name
            files = []
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Unexpected symlink: {path}")
                if not path.is_file():
                    continue
                target = destination / path.relative_to(source)
                if not target.resolve().is_relative_to(destination_root):
                    raise ValueError("Destination escapes requested data root")
                sha = digest(path)
                if target.exists() and (not target.is_file() or digest(target) != sha):
                    raise ValueError(f"Existing destination differs: {target}")
                files.append((path, target, sha))
            plans.append((split, metadata, files))
            selected += 1
            if selected == 2:
                break
        if selected != 2:
            raise ValueError(f"Need two valid samples for {split}")
    manifest = []
    for split, metadata, files in plans:
        copied = []
        for source, target, sha in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open("rb") as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
            if digest(target) != sha:
                raise ValueError(f"Copy verification failed: {target}")
            copied.append(dict(source=str(source), destination=str(target), sha256=sha))
        manifest.append(dict(split=split, sid=metadata["sid"], gold=metadata["gold"], files=copied))
    out = REPO / "outputs/native_aggregation_vlm" / f"staging_{os.environ['SLURM_JOB_ID']}.json"
    with out.open("x") as stream:
        json.dump(dict(purpose="V0 software smoke only", data_seed=1234,
                       slurm_job_id=os.environ["SLURM_JOB_ID"], samples=manifest), stream, indent=2)
    print(f"Verified {len(manifest)} samples; {out}", flush=True)


if __name__ == "__main__":
    main()
