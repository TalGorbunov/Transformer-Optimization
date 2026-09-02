#!/usr/bin/env python3
"""ARMOR Exp B: N=128 leaf capture — thin wrapper around scripts/ninv/probe_tree_ninv.py.

The ninv script's NS table stops at 64, so `--ns 128` silently captures nothing
(the 2026-08-22 21:13 21-second no-op jobs 136126/136127). The ninv script is
read-only mid-campaign; this wrapper registers the 128 row and delegates. All
other args pass through unchanged (use --root to pick the hf pool)."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/ninv"))

import probe_tree_ninv as ptn  # noqa: E402

ptn.NS.append((128, "data/mmred_longN_park/seq_len_128/all_uniform", 40))

if __name__ == "__main__":
    sys.exit(ptn.main())
