#!/usr/bin/env python3
"""RECAGG Arm B step 2 — execute the ARM C v3 PROGRAMS on VLM-caption records.

The controlled swap: identical questions, identical compiled programs (loaded
from the canonical Arm C v3 programs.json), identical sandbox — only `frames`
changes from oracle GT dicts to the frozen VLM's parsed per-frame captions
(caption_frames.py output). Any EM drop vs v3 on the same subset is therefore
attributable to perception alone; caption fidelity is reported alongside for
the factorization check (EM_B ~ EM_C x state-fidelity^facts-touched).

CPU-only. Usage:
  python scripts/recagg/armB_execute.py \
      --programs outputs/recagg/armC_compile/<v3>/programs.json \
      --captions outputs/recagg/armB_captions/<run>/captions.json \
      --output outputs/recagg/armB_execute/<ts>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/ninv"))
sys.path.insert(0, str(_REPO / "scripts/recagg"))

from ask_compile_execute import SEEN_TYPES, norm, run_program, type_of  # noqa: E402
from load_hf_sample import parse_qa  # noqa: E402


def rebuild_task_dirs(per_type: int = 8):
    """The exact Arm C task order (pool iteration + stride), as (N, type, dir)."""
    tasks = []
    for n in (16, 32, 64, 128):
        by_type = defaultdict(list)
        for d in sorted(Path(f"data/mmred_hf/dirs/seq_len_{n}_test").iterdir()):
            if d.is_dir() and (d / "qa.txt").is_file():
                by_type[type_of(d.name)].append(d)
        for t, dirs in sorted(by_type.items()):
            step = max(1, len(dirs) // per_type)
            tasks += [(n, t, d) for d in dirs[::step][:per_type]]
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    recs = json.load(open(args.programs))
    caps = json.load(open(args.captions))
    tasks = rebuild_task_dirs()
    assert len(tasks) == len(recs), "programs.json / task-order mismatch"

    rows = []
    for (n, t, d), r in zip(tasks, recs):
        cap = caps.get(d.name)
        if cap is None:
            continue                       # not in the captioned subset
        frames = [{"rooms": rooms} for rooms in cap["states"]]
        pred, err = run_program(r["code"], frames)
        em_b = int(err is None and pred is not None
                   and norm(pred) == norm(r["gold"]))
        rows.append({"N": n, "type": t, "dir": d.name, "gold": r["gold"],
                     "em_oracle": r["em"], "em_vlm": em_b,
                     "err": err, "char_acc": cap["char_acc"],
                     "frame_acc": cap["frame_acc"]})

    ns = sorted({r["N"] for r in rows})
    types = sorted({r["type"] for r in rows})
    lines = ["ARM B — v3 programs executed on VLM-caption records "
             "(same subset, same code; delta vs oracle = perception)",
             f"n={len(rows)} matched tasks; caption per-char placement "
             f"{np.mean([r['char_acc'] for r in rows]):.4f}, per-frame exact "
             f"{np.mean([r['frame_acc'] for r in rows]):.4f}", ""]
    hdr = (f"{'type':<26} " + " ".join(f"N={n}:vlm/orc" for n in ns))
    lines.append(hdr)
    for t in types:
        cells = []
        for n in ns:
            sel = [r for r in rows if r["type"] == t and r["N"] == n]
            if sel:
                cells.append(f"{np.mean([r['em_vlm'] for r in sel]):.2f}/"
                             f"{np.mean([r['em_oracle'] for r in sel]):.2f}")
            else:
                cells.append("  -  ")
        lines.append(f"{t:<26} " + "  ".join(cells)
                     + ("  SEEN" if t in SEEN_TYPES else ""))
    lines.append("")
    for label, fn in (("ALL", lambda t: True),
                      ("UNSEEN", lambda t: t not in SEEN_TYPES)):
        cells = []
        for n in ns:
            sel = [r for r in rows if fn(r["type"]) and r["N"] == n]
            cells.append(f"{np.mean([r['em_vlm'] for r in sel]):.2f}/"
                         f"{np.mean([r['em_oracle'] for r in sel]):.2f}")
        lines.append(f"{label:<26} " + "  ".join(cells))
    report = "\n".join(lines)
    print(report)
    (out / "report.txt").write_text(report + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
