#!/usr/bin/env python3
"""REDUX C0 — exam dirs-files for {count, exists, majority} (CPU, no image loads).

count: byte-reuse of the LORAMECH exam_ff_N*.txt (copied under the redux name).
exists: 50% k=0; yes half stratified with k=1 >= 1/3 of yes, rest k in {2,4,8}.
        N=8 from park (150/cell); N>=16 from data/mmred_redux (100/cell — k=0
        stratum = 50 caps it; documented).
majority: stratified over signed d = k - N/2 in +/-{1,2,4,8,16,32} (clipped,
        d=0 excluded), balanced yes/no, 150/cell. N=8 from park; N>=16 from redux.

Gold is re-derived from states via tasks.py per dir and (count) asserted against the
stored answer; mismatches are skipped and counted. Draws are per-class seeded
shuffles (never head-slices). REPORT.txt: class dist + majority baseline per cell.
"""
from __future__ import annotations

import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "redux"))

from tasks import derive_gold, evidence_count, target_of  # noqa: E402

PARK8 = Path("data/mmred_images_park/seq_len_8/all_uniform")
REDUX = Path("data/mmred_redux")
FF_EXAMS = Path("outputs/loramech/examdirs")
OUT = Path("outputs/redux/examdirs")
SEED = 2


def light_load(sd: Path):
    """(question, states) from qa.txt without touching the PNGs."""
    lines = (sd / "qa.txt").read_text(encoding="utf-8").splitlines()
    qi = next(i for i, ln in enumerate(lines) if ln.strip() == "question:")
    ai = next(i for i, ln in enumerate(lines) if ln.strip() == "answer:")
    states, q0 = [], None
    for ln in lines[qi + 1: ai]:
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
        elif s:
            q0 = s
            break
    answer = next(ln.strip() for ln in lines[ai + 1:] if ln.strip())
    return q0, states, answer


def pool_by_k(root: Path):
    """{k: [sample_dir,...]} keyed by the dir-name K tag, seeded-shuffled per class."""
    groups = {}
    for d in sorted(root.iterdir()):
        m = re.search(r"_(?:K|e)(\d+)_", d.name)
        if m:
            groups.setdefault(int(m.group(1)), []).append(d)
    rng = random.Random(SEED)
    for k in groups:
        rng.shuffle(groups[k])
    return groups


def verified(sd: Path, n_frames: int, want_k: int, task: str):
    """gold string for task or None if the dir fails verification."""
    try:
        q0, states, answer = light_load(sd)
        tr = target_of(sd, q0)
        if tr is None or len(states) != n_frames:
            return None
        c, r = tr
        k = evidence_count(states, c, r)
        if k != want_k or str(k) != str(answer).strip():
            return None
        return derive_gold(task, states, c, r, n_frames)
    except Exception:
        return None


def take(groups, k, m, n_frames, task, used, bad):
    out = []
    for sd in groups.get(k, []):
        if len(out) >= m:
            break
        if sd in used:
            continue
        g = verified(sd, n_frames, k, task)
        if g is None:
            bad[0] += 1
            continue
        out.append((sd, g))
        used.add(sd)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []

    # ---- count: byte-reuse
    for n in (8, 16, 32, 64, 128):
        src = FF_EXAMS / f"exam_ff_N{n}.txt"
        (OUT / f"exam_count_N{n}.txt").write_text(src.read_text())
        report.append(f"exam_count_N{n}: {len(src.read_text().split())} dirs "
                      f"(byte-reuse of {src})")

    pools = {8: pool_by_k(PARK8)}
    for n in (16, 32, 64, 128):
        pools[n] = pool_by_k(REDUX / f"seq_len_{n}" / "all_uniform")

    # ---- exists
    for n in (8, 16, 32, 64, 128):
        cell, half = (150, 75) if n == 8 else (100, 50)
        used, bad = set(), [0]
        rows = take(pools[n], 0, half, n, "exists", used, bad)
        k1 = max(half // 3, 1)
        yes_plan = [(1, k1)] + [(k, (half - k1) // 3 + (1 if i < (half - k1) % 3 else 0))
                                for i, k in enumerate((2, 4, 8))]
        for k, m in yes_plan:
            rows += take(pools[n], k, m, n, "exists", used, bad)
        golds = Counter(g for _sd, g in rows)
        ks = Counter(int(re.search(r"_(?:K|e)(\d+)_", sd.name).group(1)) for sd, _g in rows)
        (OUT / f"exam_exists_N{n}.txt").write_text(
            "\n".join(str(sd) for sd, _g in rows) + "\n")
        maj = max(golds.values()) / max(1, len(rows))
        report.append(f"exam_exists_N{n}: {len(rows)} dirs (target {cell}, bad {bad[0]}) "
                      f"gold {dict(golds)} maj-baseline {maj:.3f} "
                      f"k-dist {dict(sorted(ks.items()))}")

    # ---- majority
    for n in (8, 16, 32, 64, 128):
        half_n = n // 2
        ds = [d for d in (1, 2, 4, 8, 16, 32) if d <= half_n]
        strata = [half_n + s * d for d in ds for s in (+1, -1) if 0 <= half_n + s * d <= n]
        per = max(150 // len(strata), 1)
        used, bad = set(), [0]
        rows = []
        for k in strata:
            rows += take(pools[n], k, per, n, "majority", used, bad)
        golds = Counter(g for _sd, g in rows)
        ks = Counter(int(re.search(r"_(?:K|e)(\d+)_", sd.name).group(1)) for sd, _g in rows)
        (OUT / f"exam_majority_N{n}.txt").write_text(
            "\n".join(str(sd) for sd, _g in rows) + "\n")
        maj = max(golds.values()) / max(1, len(rows))
        report.append(f"exam_majority_N{n}: {len(rows)} dirs (strata k={sorted(set(strata))}, "
                      f"{per}/stratum, bad {bad[0]}) gold {dict(golds)} "
                      f"maj-baseline {maj:.3f} k-dist {dict(sorted(ks.items()))}")

    (OUT / "REPORT.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
