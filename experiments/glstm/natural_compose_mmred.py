#!/usr/bin/env python3
"""P3a data prep (p0p2 campaign, 2026-07-24): compose MMRED-format natural-image counting
samples from the judge-gated per-frame pools of data/mmred_natural_v2 (dist_far/dist_near;
ident_* excluded — identical-evidence pathology cells).

Key discipline: a GLOBAL image-half split (seeded, consistent across cells and N) —
'train' samples draw ONLY from half A, 'eval' samples ONLY from half B, so L3 exam cells
are image-held-out, not just sample-held-out (the v2 pools reuse images heavily: 154
evidence images serve all samples). Distractors for concept c come only from frames judged
not-evidence RELATIVE TO c (i.e. frames of same-concept samples).

Each composed dir carries BOTH formats: qa.txt + NNN.png (MMRED loader / trainer / frozen
baseline) and meta.json + frame_XX.jpg (replica_carrier_probe --natural). Image files are
hardlinks to the v2 originals (no re-encode, no extra space).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPECS = [  # (N, split, golds, per_count)
    (8,  "train", list(range(9)), 25),
    (8,  "eval",  list(range(9)), 15),
    (16, "train", list(range(9)) + [12, 16], 20),
    (16, "eval",  list(range(9)) + [12, 16], 10),
]


def link(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=PROJECT_ROOT / "data/mmred_natural_v2")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/mmred_natural_mm")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cells = ("dist_far", "dist_near")
    # index: per cell, evidence/distractor image_id -> file path; concept per cell
    pools, concepts = {}, {}
    all_ids = set()
    for cell in cells:
        ev, di = {}, {}
        for sd in sorted((args.src / cell).iterdir()):
            if not (sd / "meta.json").exists():
                continue
            m = json.loads((sd / "meta.json").read_text())
            concepts[cell] = m["concept"]
            for i, fr in enumerate(m["frames"]):
                p = sd / f"frame_{i:02d}.jpg"
                (ev if fr["is_evidence"] else di)[fr["image_id"]] = p
        n_conflict = len(set(ev) & set(di))
        di = {i: p for i, p in di.items() if i not in ev}   # judge-conflicted ids -> evidence only
        if n_conflict:
            print(f"  [{cell}] {n_conflict} judge-conflicted image ids dropped from distractor pool")
        pools[cell] = {"ev": ev, "di": di}
        all_ids |= set(ev) | set(di)

    rng = random.Random(args.seed)
    ids = sorted(all_ids)
    rng.shuffle(ids)
    half_a = set(ids[: len(ids) // 2])          # train half; rest = eval half
    stats = []
    for cell in cells:
        concept = concepts[cell]
        for N, split, golds, per in SPECS:
            keep = (lambda i: i in half_a) if split == "train" else (lambda i: i not in half_a)
            ev = [(i, p) for i, p in sorted(pools[cell]["ev"].items()) if keep(i)]
            di = [(i, p) for i, p in sorted(pools[cell]["di"].items()) if keep(i)]
            root = args.out / f"seq_len_{N}" / f"{cell}_{split}"
            root.mkdir(parents=True, exist_ok=True)
            made = 0
            for g in golds:
                if g > len(ev) or (N - g) > len(di):
                    print(f"  SKIP {cell}/{split} N{N} K{g}: pool too small "
                          f"(ev {len(ev)}, di {len(di)})", flush=True)
                    continue
                for j in range(per):
                    pick = rng.sample(ev, g) + rng.sample(di, N - g)
                    rng.shuffle(pick)
                    sd = root / f"nat{N}_{cell}_{split}_K{g}_{j:04d}"
                    sd.mkdir(exist_ok=True)
                    flags = []
                    for t, (iid, p) in enumerate(pick):
                        link(p, sd / f"frame_{t:02d}.jpg")
                        link(p, sd / f"{t:03d}.png")
                        flags.append(iid in pools[cell]["ev"])
                    assert sum(flags) == g, (cell, split, N, g, sum(flags))
                    q = f"In how many of the {N} frames does a {concept} appear?"
                    states = "\n".join(
                        "{'natural': {'concept': '%s', 'evidence': %s}}" % (concept, f)
                        for f in flags)
                    (sd / "qa.txt").write_text(
                        f"qid: {sd.name}\nqtype: natural_count\natype: integer\n"
                        f"seq_len: {N}\nquestion:\n{states}\n{q}\nanswer:\n{g}\n",
                        encoding="utf-8")
                    (sd / "meta.json").write_text(json.dumps({
                        "question_id": sd.name, "cell": cell, "answer": g,
                        "n_frames": N, "concept": concept, "question": q,
                        "frames": [{"image_id": iid, "is_evidence": bool(fl)}
                                   for (iid, _), fl in zip(pick, flags)]}, indent=1),
                        encoding="utf-8")
                    made += 1
            stats.append(f"{cell}/{split} N{N}: {made} samples (ev-pool {len(ev)}, di-pool {len(di)})")
            print(stats[-1], flush=True)
    (args.out / "BUILD_INFO.txt").write_text(
        f"seed {args.seed}; global image-half split over {len(ids)} distinct images "
        f"(A/train {len(half_a)}, B/eval {len(ids) - len(half_a)}); src {args.src}\n"
        + "\n".join(stats) + "\n", encoding="utf-8")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
