#!/usr/bin/env python3
"""A4: build mmred_natural — NIAH-style counting over NATURAL images with per-frame GT
by construction, and two d′ dials:

  needle diversity : identical (ONE dog image repeated k times) vs distinct (k different dogs)
  distractor sim   : far (COCO no-animal images) vs near (COCO cat-but-no-dog images)

Cells: ident_far / dist_far / ident_near / dist_near. Task: "In how many of the 8 frames does
a dog appear?" — count k ∈ 0..8 uniform, per-frame GT from COCO instance annotations.

Sample dirs are HERBench-style (frame_XX.jpg + meta.json with per-frame is_evidence,
visible_count, question) so the existing instrument battery runs unchanged.
Output: data/mmred_natural/<cell>/<sid>/. Pilot mode: --per-cell 50 (validate the look-again
judge >=0.95 BEFORE the full build, per the plan gate).
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CELLS = ("ident_far", "dist_far", "ident_near", "dist_near")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="data/coco_val2017")
    ap.add_argument("--out", default="data/mmred_natural")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--per-cell", type=int, default=50, help="samples per cell (pilot 50; full ~375)")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="", help="judge_curation.json with bad_needles/bad_distractors "
                    "image ids to drop from the pools (pilot-gate curation)")
    args = ap.parse_args()

    from PIL import Image
    coco = Path(args.coco_root)
    d = json.loads((coco / "instances_val2017.json").read_text())
    cats = {c["id"]: (c["name"], c["supercategory"]) for c in d["categories"]}
    animal_ids = {i for i, (n, s) in cats.items() if s == "animal"}
    img_file = {im["id"]: im["file_name"] for im in d["images"]}
    img_cats = {}
    for a in d["annotations"]:
        img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
    DOG, CAT = 18, 17
    pool_dog = sorted(i for i, s in img_cats.items() if DOG in s)
    pool_near = sorted(i for i, s in img_cats.items() if CAT in s and DOG not in s)
    pool_far = sorted(i for i, s in img_cats.items() if not (s & animal_ids))
    if args.exclude:
        cur = json.loads(Path(args.exclude).read_text())
        bad_n, bad_d = set(cur["bad_needles"]), set(cur["bad_distractors"])
        pool_dog = [i for i in pool_dog if i not in bad_n]
        pool_near = [i for i in pool_near if i not in bad_d]
        pool_far = [i for i in pool_far if i not in bad_d]
        print(f"curation: dropped {len(bad_n)} needles, {len(bad_d)} distractors")
    print(f"pools: dog {len(pool_dog)}  near(cat) {len(pool_near)}  far(no-animal) {len(pool_far)}")

    rng = random.Random(args.seed)
    NF = int(args.n_frames)
    out_root = Path(args.out)

    def load_resized(img_id):
        p = coco / "val2017" / img_file[img_id]
        im = Image.open(p).convert("RGB")
        w, h = im.size
        sc = args.size / max(w, h)
        return im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)

    for cell in [c.strip() for c in args.cells.split(",")]:
        ident = cell.startswith("ident")
        near = cell.endswith("near")
        dpool = pool_near if near else pool_far
        made = 0
        for j in range(args.per_cell):
            k = rng.randint(0, NF)
            if ident:
                needle_ids = [rng.choice(pool_dog)] * k
            else:
                needle_ids = rng.sample(pool_dog, k)
            dist_ids = rng.sample(dpool, NF - k)
            frames = [(iid, True) for iid in needle_ids] + [(iid, False) for iid in dist_ids]
            rng.shuffle(frames)
            sid = f"nat_{cell}_K{k}_{j:04d}"
            sdir = out_root / cell / sid
            if (sdir / "meta.json").exists():
                made += 1
                continue
            sdir.mkdir(parents=True, exist_ok=True)
            fm = []
            for t, (iid, isev) in enumerate(frames):
                load_resized(iid).save(sdir / f"frame_{t:02d}.jpg", quality=90)
                fm.append({"image_id": int(iid), "is_evidence": bool(isev)})
            meta = {"question_id": sid, "cell": cell, "answer": k, "visible_count": k,
                    "n_frames": NF, "concept": "dog",
                    "question": f"In how many of the {NF} frames does a dog appear?",
                    "frames": fm, "needle_diversity": "identical" if ident else "distinct",
                    "distractor_sim": "near" if near else "far"}
            (sdir / "meta.json").write_text(json.dumps(meta, indent=1))
            made += 1
            if made % 25 == 0:
                print(f"  {cell}: {made}/{args.per_cell}", flush=True)
        print(f"{cell}: {made} samples", flush=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
