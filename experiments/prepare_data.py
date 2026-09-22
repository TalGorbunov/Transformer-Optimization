#!/usr/bin/env python3
"""HF `ef1e43ce/mmred` -> json/<config>_<split>.json -> official renders
images/<config>_<split>/<qid>/frame_%04d.png (Fr0do/mmred renderer, 512 px) -> images/<split>.tar.

Twin: legacy/v1/scripts/mmred_hf/prep.py (HF row -> JSON; the HF `question` column packs one
python-dict line per step plus the question as the last line) and render.py (ported verbatim:
upstream render_images.py uses ProcessPoolExecutor() with the node's FULL core count and
fork-bombed NFS on 2026-08-01, jobs 127752-127763). Both now run in the single .venv.

Anchor: `--verify-against data/mmred_hf --verify-n 50` on seq_len_8_val reproduces the 2026-08-01
tree: JSON bytes equal, every frame PIXEL-identical (PNG bytes may differ: encoder/metadata
chunks change with Pillow/matplotlib versions; 2026-09-21 smoke = 16/16 pixels, 0/16 md5: the only
difference is the PNG "Software" chunk, "Matplotlib version3.11.1" in the 2026-08-01 tree vs 3.11.2 now).
A pixel-level miss is a finding, reported with the max pixel delta.

headfit is not on HF: rows come from data/mmred_hf/headfit_raw.json filtered by seq_len, all
keys kept (that is how the existing seq_len_{32,64,128}_headfit.json were written).

Usage:
  python experiments/prepare_data.py --config seq_len_1 --split train            # prep+render+tar
  python experiments/prepare_data.py --config seq_len_8 --split val --verify-against data/mmred_hf \
      --verify-n 50 --out $TMPDIR/verify
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# node-local matplotlib config BEFORE any matplotlib import (workers inherit the env)
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_", dir=os.environ.get("TMPDIR", "/tmp")))
os.environ.setdefault("MPLBACKEND", "Agg")

from core.constants import HF_DATASET, SEQ_LENS  # noqa: E402
from core.mmred import QTYPES, recompute_answer, states  # noqa: E402

CONFIGS = [f"seq_len_{n}" for n in SEQ_LENS]
SPLITS = ("train", "val", "test", "headfit")
ROW_KEYS = ("qid", "seq_len", "qtype", "atype", "question", "answer", "sequence")


# ------------------------------------------------------------------ prep

def parse_question_field(q: str):
    """HF `question` column -> (sequence[list of {step_id, rooms}], question_text). prep.py:29."""
    seq, question = [], None
    for ln in q.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            d = ast.literal_eval(s)
            seq.append({"step_id": int(d["step_id"]), "rooms": {r: list(c) for r, c in d["rooms"].items()}})
        else:
            question = s
    if question is None or not seq:
        raise ValueError(f"unparseable question field: {q[:120]!r}")
    return seq, question


def prep_rows(config: str, split: str, data_root: Path, limit: int, qtypes) -> list:
    if split == "headfit":
        raw = json.loads((data_root / "headfit_raw.json").read_text(encoding="utf-8"))
        n = int(config.split("_")[-1])
        rows = [r for r in raw if int(r["seq_len"]) == n]
    else:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, config, cache_dir=str(data_root / "hf"))[split]
        rows = []
        for row in ds:
            seq, qtext = parse_question_field(row["question"])
            assert len(seq) == row["seq_len"], (row["qid"], len(seq), row["seq_len"])
            assert [s["step_id"] for s in seq] == list(range(1, row["seq_len"] + 1)), row["qid"]
            rows.append({"qid": row["qid"], "seq_len": row["seq_len"], "qtype": row["qtype"],
                         "atype": row["atype"], "question": qtext, "answer": row["answer"], "sequence": seq})
    if qtypes:
        rows = [r for r in rows if r["qtype"] in set(qtypes)]
    if limit:
        rows = rows[:limit]
    return rows


def check_gold(rows) -> int:
    bad = 0
    for r in rows:
        got = recompute_answer(r["qtype"], r["question"], states(r))
        if str(got) != str(r["answer"]):
            bad += 1
            if bad <= 5:
                print(f"  GOLD MISMATCH {r['qtype']} qid={r['qid']}: got {got!r} want {r['answer']!r}", flush=True)
    return bad


def dump_json(rows, path: Path) -> bytes:
    data = json.dumps(rows, ensure_ascii=False).encode("utf-8")      # no indent: byte parity with 2026-08-01
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


# ------------------------------------------------------------------ render (render.py, verbatim logic)

def _render_one(args):
    sample, output_dir = args
    from core.mmred import render_sequence

    render_sequence(sample["sequence"], Path(output_dir) / sample["qid"])
    return sample["qid"]


def _done(sample, output_dir: Path) -> bool:
    d = output_dir / sample["qid"]
    return d.is_dir() and len(list(d.glob("frame_*.png"))) == len(sample["sequence"])


def render_rows(rows, output_dir: Path, workers: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in rows if not _done(s, output_dir)]
    print(f"[render] {len(rows)} samples, {len(rows) - len(todo)} already rendered, {len(todo)} to do, "
          f"workers={workers}", flush=True)
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _qid in ex.map(_render_one, ((s, str(output_dir)) for s in todo), chunksize=4):
            n += 1
            if n % 100 == 0:
                print(f"[render] {n}/{len(todo)}", flush=True)
    return n


# ------------------------------------------------------------------ tar (tar_split.sbatch rule)

def tar_split(images_root: Path, split_dir: str) -> Path:
    import subprocess

    tar_path = images_root / f"{split_dir}.tar"
    n_tree = sum(1 for _ in (images_root / split_dir).rglob("*") if _.is_file())
    if tar_path.exists():
        n_tar = int(subprocess.run(["bash", "-c", f"tar tf '{tar_path}' | grep -vc '/$'"],
                                   capture_output=True, text=True).stdout.strip() or 0)
        if n_tar == n_tree:
            print(f"[tar] {tar_path} already complete ({n_tree} files)", flush=True)
            return tar_path
    partial = tar_path.with_suffix(".tar.partial")
    subprocess.run(["tar", "cf", str(partial), "-C", str(images_root), split_dir], check=True)
    partial.rename(tar_path)
    print(f"[tar] {tar_path} members={n_tree}", flush=True)
    return tar_path


# ------------------------------------------------------------------ verify

def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def verify(rows, json_bytes: bytes, canonical: Path, out_root: Path, split_dir: str, n: int, workers: int) -> int:
    from PIL import Image
    import numpy as np

    rc = 0
    canon_json = canonical / "json" / f"{split_dir}.json"
    same = canon_json.exists() and canon_json.read_bytes() == json_bytes
    print(f"[verify] JSON bytes {'EQUAL' if same else 'DIFFER'}: {canon_json}", flush=True)
    rc |= (not same)
    sub = rows[:n]
    render_rows(sub, out_root / "images" / split_dir, workers)
    n_frames = n_md5 = n_pix = 0
    worst = 0.0
    worst_frac = 0.0
    for r in sub:
        for i in range(len(r["sequence"])):
            name = f"frame_{i + 1:04d}.png"
            a = out_root / "images" / split_dir / r["qid"] / name
            b = canonical / "images" / split_dir / r["qid"] / name
            n_frames += 1
            if not b.exists():
                worst, worst_frac = 255.0, 1.0
                continue
            if _md5(a) == _md5(b):
                n_md5 += 1
                n_pix += 1
                continue
            x = np.asarray(Image.open(a).convert("RGBA")).astype(int)
            y = np.asarray(Image.open(b).convert("RGBA")).astype(int)
            if x.shape == y.shape:
                d = np.abs(x - y)
                if d.max() == 0:
                    n_pix += 1
                worst = max(worst, float(d.max()))
                worst_frac = max(worst_frac, float((d.max(axis=-1) > 0).mean()))
            else:
                worst, worst_frac = 255.0, 1.0
    import matplotlib, PIL, mmred
    # The pass criterion is PIXEL identity: PNG bytes also carry encoder/metadata chunks that
    # differ across Pillow/matplotlib versions (2026-09-21 smoke: 0/16 md5 but 16/16 pixels).
    print(f"[verify] frames pixel-identical {n_pix}/{n_frames} (PNG bytes md5-identical {n_md5}/{n_frames}); "
          f"max |delta| {worst:.0f}, max differing-pixel fraction {worst_frac:.4f}; "
          f"matplotlib {matplotlib.__version__}, Pillow {PIL.__version__}, mmred {getattr(mmred, '__version__', '?')}",
          flush=True)
    rc |= (n_pix != n_frames)
    return rc


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--out", type=Path, default=None, help="default: --data-root (the canonical tree)")
    ap.add_argument("--stage", choices=["prep", "render", "tar", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qtypes", nargs="+", default=None, choices=QTYPES)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--no-check-gold", action="store_true")
    ap.add_argument("--verify-against", type=Path, default=None, help="canonical root to compare with")
    ap.add_argument("--verify-n", type=int, default=50)
    args = ap.parse_args()

    out = args.out or args.data_root
    canonical_out = out.resolve() == args.data_root.resolve()
    if canonical_out and (args.limit or args.qtypes):
        raise SystemExit("--limit/--qtypes would write a partial split into the canonical tree; pass --out")
    split_dir = f"{args.config}_{args.split}"
    t0 = time.time()

    rows = prep_rows(args.config, args.split, args.data_root, args.limit, args.qtypes)
    print(f"[prep] {split_dir}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    if not args.no_check_gold:
        bad = check_gold(rows)
        if bad:
            raise SystemExit(f"[prep] {bad} gold mismatches; refusing to write")
        print(f"[prep] gold parity {len(rows)}/{len(rows)}", flush=True)

    if args.verify_against is not None:
        data = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        (out / "json").mkdir(parents=True, exist_ok=True)
        (out / "json" / f"{split_dir}.json").write_bytes(data)
        return verify(rows, data, args.verify_against, out, split_dir, args.verify_n, args.workers)

    if args.stage in ("prep", "all"):
        dump_json(rows, out / "json" / f"{split_dir}.json")
        print(f"[prep] wrote {out / 'json' / (split_dir + '.json')}", flush=True)
    if args.stage in ("render", "all"):
        n = render_rows(rows, out / "images" / split_dir, args.workers)
        print(f"[render] {n} rendered ({time.time() - t0:.0f}s)", flush=True)
    if args.stage in ("tar", "all"):
        tar_split(out / "images", split_dir)
    manifest = {"split": split_dir, "n_rows": len(rows), "n_frames": sum(len(r["sequence"]) for r in rows),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)}
    (out / "json" / f"{split_dir}.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[done] {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
