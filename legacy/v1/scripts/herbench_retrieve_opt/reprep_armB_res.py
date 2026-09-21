#!/usr/bin/env python3
"""Re-extract the armB_ev_fill16 frames at a higher resolution for the A1 arm (pure-resolution
single-frame anchor). Reads each armB sample's meta.json (frame times + is_evidence labels +
video_id), decodes those exact timestamps at --res from the source video, and writes an
identical dir tree with the meta.json copied verbatim (only its top-level res note updated).
Login-node PyAV, same decoder as extract_clips.

  PYTHONPATH=~/.local/pyav-py39 python scripts/herbench_retrieve_opt/reprep_armB_res.py \
    --armB-root data/herbench_ac/armB_ev_fill16 --video-root /scratch/.../herbench_videos \
    --res 672 --out data/herbench_ac/armB_ev_fill16_r672
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.herbench_retrieve_opt.extract_clips import decode_times, video_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--armB-root", default="data/herbench_ac/armB_ev_fill16")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    armB = Path(args.armB_root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vroot = Path(args.video_root)
    dirs = sorted(d for d in armB.iterdir() if (d / "meta.json").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    n_ok = n_miss = 0
    for i, sd in enumerate(dirs):
        m = json.loads((sd / "meta.json").read_text())
        vp = video_path(vroot, m["video_id"])
        if not vp.exists():
            n_miss += 1; continue
        times = [fr["time"] for fr in m["frames"]]
        frames = decode_times(vp, times, args.res)
        od = out / sd.name; od.mkdir(parents=True, exist_ok=True)
        ok = True
        for fr in m["frames"]:
            t = fr["time"]
            if t not in frames:
                ok = False; break
            frames[t].save(od / f"frame_{fr['idx']:02d}.jpg", quality=args.jpeg_quality)
        if not ok:
            n_miss += 1; continue
        m2 = dict(m); m2["res"] = args.res
        (od / "meta.json").write_text(json.dumps(m2, indent=1))
        n_ok += 1
        if (i + 1) % 30 == 0:
            print(f"[{i+1}/{len(dirs)}] ok={n_ok} miss={n_miss}", flush=True)
    print(f"done: ok={n_ok} miss={n_miss} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
