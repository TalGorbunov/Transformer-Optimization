#!/usr/bin/env python3
"""Rebuild outputs/_by_date/ — a generated, untracked day-view of the live run dirs.

Scans outputs/ for timestamped run dirs (any dir whose name starts with
YYYYMMDD_HHMMSS) and symlinks each under outputs/_by_date/YYYY-MM-DD/ as
<parent-path-with-__>__<dirname>. The tree is wiped and rebuilt on every call
(symlinks only — the script refuses to delete anything that isn't a symlink).

Usage: python scripts/index_by_date.py [--root outputs] [--include-legacy]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

TS = re.compile(r"^(\d{8})_\d{6}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--include-legacy", action="store_true",
                    help="also index outputs_legacy/outputs")
    args = ap.parse_args()

    roots = [Path(args.root)]
    if args.include_legacy:
        roots.append(Path("outputs_legacy/outputs"))
    by_date = Path(args.root) / "_by_date"

    if by_date.exists():
        bad = [p for p in by_date.rglob("*") if not (p.is_symlink() or p.is_dir())]
        if bad:
            raise SystemExit(f"refusing to wipe {by_date}: non-symlink entries {bad[:3]}")
        shutil.rmtree(by_date)

    n = 0
    for root in roots:
        if not root.exists():
            continue
        for d in root.rglob("*"):
            if not d.is_dir() or d.is_symlink():
                continue
            if by_date in d.parents:
                continue
            m = TS.match(d.name)
            if not m:
                continue
            day = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
            rel = d.relative_to(root)
            link = by_date / day / "__".join(rel.parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            depth = len(link.parent.relative_to(Path(args.root)).parts) + 1
            target = Path(*[".."] * depth) / d if root == Path(args.root) else d.resolve()
            if not link.exists():
                link.symlink_to(target)
                n += 1
    print(f"linked {n} run dirs under {by_date}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
