#!/usr/bin/env python3
"""Materialize MMReD-HF samples as our qa.txt sample dirs (park-render convention).

Writes data/mmred_hf/dirs/<config>_<split>[_<qtype>]/<name>/qa.txt with the upstream
per-step state dicts (they carry "rooms" exactly like our park states, so ALL existing
gnnformer/data.py parsing works) + 000.png.. hardlinks to the rendered frames (same
filesystem; no data duplication). Every existing entrypoint (probe_supply, eval_carrier,
train_carrier_*, gate_tally) then runs on the original benchmark unchanged.

Dir naming: <qtype>_K<answer>_<qid> for numeric answers (the _K<digit>_ tag feeds
iter_sample_dirs_shuffled's stratified class-balanced shuffle); <qtype>_A<answer>_<qid>
otherwise. Notably steps_in_room questions match our steps-task regex verbatim.

Shared .venv or .venv_mmred (stdlib only). Usage:
  python scripts/mmred_hf/materialize_dirs.py --config seq_len_8 --split val \
      --qtypes steps_in_room
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--qtypes", nargs="+", default=None, help="default: all 24")
    ap.add_argument("--limit-per-qtype", type=int, default=None)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data/mmred_hf/dirs"))
    args = ap.parse_args()

    tag = f"{args.config}_{args.split}"
    rows = json.loads((REPO_ROOT / f"data/mmred_hf/json/{tag}.json").read_text())
    images_root = REPO_ROOT / f"data/mmred_hf/images/{tag}"
    suffix = f"_{args.qtypes[0]}" if args.qtypes and len(args.qtypes) == 1 else ""
    out_root = Path(args.out_root) / f"{tag}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = n_missing = 0
    per_qtype: dict = {}
    for row in rows:
        qt = row["qtype"]
        if args.qtypes and qt not in args.qtypes:
            continue
        if args.limit_per_qtype and per_qtype.get(qt, 0) >= args.limit_per_qtype:
            continue
        src = images_root / row["qid"]
        frames = sorted(src.glob("frame_*.png"))
        if len(frames) != row["seq_len"]:
            n_missing += 1
            continue
        ans = str(row["answer"])
        k = f"K{ans}" if ans.isdigit() else f"A{ans}"
        d = out_root / f"{qt}_{k}_{row['qid']}"
        d.mkdir(exist_ok=True)
        lines = ["question:"]
        lines += [repr({"rooms": s["rooms"]}) for s in row["sequence"]]
        lines += [row["question"], "answer:", ans]
        (d / "qa.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for i, f in enumerate(frames):
            dst = d / f"{i:03d}.png"
            if not dst.exists():
                os.link(f, dst)
        per_qtype[qt] = per_qtype.get(qt, 0) + 1
        n_done += 1

    print(f"materialized {n_done} dirs -> {out_root} "
          f"(missing renders: {n_missing}; per qtype: {sorted(per_qtype.items())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
