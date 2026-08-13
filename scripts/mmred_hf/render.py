#!/usr/bin/env python
"""Cluster-safe render driver for the upstream MMReD renderer (.venv_mmred).

The upstream scripts/render_images.py uses ProcessPoolExecutor() with DEFAULT workers
= the node's full core count — on a shared node with an 8-CPU cgroup this fork-bombs
NFS (observed 2026-08-01: jobs 127752-127763, workers stuck in rpc_wait_bit_killable /
d_alloc_parallel, zero frames written in 10 min). This driver:
  - caps workers at SLURM_CPUS_PER_TASK (fallback 8),
  - points MPLCONFIGDIR at a node-local dir (no shared-NFS font-cache lock storms),
  - is resume-safe: skips qids whose dir already has exactly seq_len frames,
  - renders via mmred.vgen.visualization.render_sequence_from_json (same output).

Usage:
  .venv_mmred/bin/python scripts/mmred_hf/render.py \
      --input_path data/mmred_hf/json/seq_len_8_test.json \
      --output_dir data/mmred_hf/images/seq_len_8_test [--limit N] [--workers K]
"""
import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# node-local matplotlib config/cache BEFORE any matplotlib import (workers inherit env)
_mpl_dir = tempfile.mkdtemp(prefix="mpl_", dir=os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)
os.environ.setdefault("MPLBACKEND", "Agg")


def _render_one(args):
    sample, output_dir = args
    from mmred.vgen.visualization import render_sequence_from_json

    out = Path(output_dir) / sample["qid"]
    render_sequence_from_json(sample["sequence"], out, as_gif=False)
    return sample["qid"]


def _done(sample, output_dir: Path) -> bool:
    d = output_dir / sample["qid"]
    return d.is_dir() and len(list(d.glob("frame_*.png"))) == len(sample["sequence"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()

    dataset = json.loads(Path(args.input_path).read_text())
    if args.limit:
        dataset = dataset[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in dataset if not _done(s, output_dir)]
    print(f"{len(dataset)} samples, {len(dataset) - len(todo)} already rendered, "
          f"{len(todo)} to do, workers={args.workers}", flush=True)

    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for qid in ex.map(_render_one, ((s, args.output_dir) for s in todo),
                          chunksize=4):
            n += 1
            if n % 100 == 0:
                print(f"rendered {n}/{len(todo)}", flush=True)
    print(f"done: {n} rendered -> {output_dir}")


if __name__ == "__main__":
    main()
