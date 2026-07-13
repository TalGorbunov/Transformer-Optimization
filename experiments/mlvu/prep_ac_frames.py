#!/usr/bin/env python3
"""Prep MLVU Action-Count questions into MMRED/HERBench-style frame samples.

Per question (1 video per question, 206 total):
  - one sequential decode collects (a) 2fps thumbnails for insertion detection and
    (b) --n-frames uniform frames saved at --size px (long side, JPEG q90).
  - insertion segments via duplicate detection (experiments/mlvu/detect_insertions logic,
    reusing the thumbs). GT is TRUSTED only when #segments == gold and gold >= 2
    (k=1 cannot self-match; some videos use distinct clips per insertion -> 0 segments).
  - meta.json: question/candidates/answer/action, per-frame {time, is_evidence|null},
    visible_count (null unless GT trusted), gt_segments, gt_source.

Idempotent: skips question dirs whose meta.json already exists (batched downloads).
Needs PyAV: run with PYTHONPATH=~/.local/pyav-py39.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.mlvu.detect_insertions import dhash_and_texture, hamming_matrix

ACT_RE = re.compile(r"['‘’\"]([^'‘’\"]{3,60})['‘’\"]")


def process_video(video_path: Path, n_frames: int, size: int, det_fps: float = 2.0):
    """One sequential decode -> (thumb hashes, thumb times, thumb texture,
    {target_time: PIL.Image}). Target times are n_frames uniform over the duration."""
    import av
    from PIL import Image
    with av.open(str(video_path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        dur = float(st.duration * st.time_base) if st.duration else None
        if dur is None:
            dur = float(c.duration / av.time_base)
        tgt = list(np.linspace(0.02 * dur, 0.98 * dur, n_frames))
        hashes, times, tex = [], [], []
        saved = {}
        next_thumb = 0.0
        ti = 0
        step = 1.0 / det_fps
        for fr in c.decode(st):
            t = fr.time
            if t is None:
                continue
            need_thumb = t >= next_thumb
            need_tgt = ti < len(tgt) and t >= tgt[ti]
            if not (need_thumb or need_tgt):
                continue
            img = fr.to_image()
            if need_thumb:
                h, tx = dhash_and_texture(img)
                hashes.append(h); times.append(t); tex.append(tx)
                next_thumb += step
            if need_tgt:
                w, hh = img.size
                sc = size / max(w, hh)
                saved[float(t)] = img.resize((max(1, round(w * sc)), max(1, round(hh * sc))),
                                             Image.LANCZOS)
                while ti < len(tgt) and t >= tgt[ti]:
                    ti += 1
    return np.stack(hashes), np.array(times), np.array(tex), saved, dur


def segments_from_thumbs(H, T, X, ham=6, min_gap=20.0, tex_min=4.0, run_gap=1.6, min_len=0.9):
    n = len(H)
    ok = X >= tex_min
    D = hamming_matrix(H)
    far = np.abs(T[:, None] - T[None, :]) >= min_gap
    match = (D <= ham) & far & ok[:, None] & ok[None, :]
    involved = match.any(1)
    segs = []
    i = 0
    while i < n:
        if not involved[i]:
            i += 1; continue
        j = i; last = i
        while j + 1 < n and (T[j + 1] - T[last]) <= run_gap:
            j += 1
            if involved[j]:
                last = j
        segs.append((float(T[i]), float(T[last])))
        i = last + 1
    return [(a, b) for a, b in segs if (b - a) >= min_len]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", required=True, help="dir containing count_*.mp4")
    ap.add_argument("--json", default="", help="path to 4_count.json (default: HF cache download)")
    ap.add_argument("--out", default="data/mlvu_ac")
    ap.add_argument("--n-frames", type=int, default=128)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--seg-pad", type=float, default=1.0, help="pad detected segments (s)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.json:
        qs = json.loads(Path(args.json).read_text())
    else:
        from huggingface_hub import hf_hub_download
        qs = json.loads(Path(hf_hub_download(
            "sy1998/MLVU", "MLVU/json/4_count.json", repo_type="dataset")).read_text())
    by_video = {q["video"]: q for q in qs}
    vroot = Path(args.videos_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    stats = {"prepped": 0, "skipped": 0, "gt_trusted": 0, "missing_video": 0}
    for vname, q in sorted(by_video.items()):
        qid = Path(vname).stem
        qdir = out_root / qid
        if (qdir / "meta.json").exists():
            stats["skipped"] += 1
            continue
        vpath = vroot / vname
        if not vpath.exists():
            stats["missing_video"] += 1
            continue
        if args.limit and done >= args.limit:
            break
        try:
            H, T, X, saved, dur = process_video(vpath, args.n_frames, args.size)
            segs = segments_from_thumbs(H, T, X)
        except Exception as exc:
            print(f"{qid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        gold = int(q["answer"])
        gt_trusted = (gold >= 2 and len(segs) == gold)
        padded = [(a - args.seg_pad, b + args.seg_pad) for a, b in segs]
        qdir.mkdir(parents=True, exist_ok=True)
        frames_meta = []
        for i, (t, img) in enumerate(sorted(saved.items())):
            img.save(qdir / f"frame_{i:03d}.jpg", quality=90)
            isev = (any(a <= t <= b for a, b in padded) if gt_trusted else None)
            frames_meta.append({"time": round(t, 2), "is_evidence": isev})
        m = ACT_RE.search(q["question"])
        meta = {"question_id": qid, "video": vname, "question": q["question"],
                "action": m.group(1) if m else None,
                "candidates": q["candidates"], "answer": gold, "duration": dur,
                "n_frames": len(frames_meta), "frames": frames_meta,
                "visible_count": (sum(1 for f in frames_meta if f["is_evidence"])
                                  if gt_trusted else None),
                "gt_segments": [[round(a, 2), round(b, 2)] for a, b in segs],
                "gt_source": "dup_detect" if gt_trusted else "none"}
        (qdir / "meta.json").write_text(json.dumps(meta, indent=1))
        stats["prepped"] += 1
        stats["gt_trusted"] += int(gt_trusted)
        done += 1
        print(f"{qid}: gold={gold} segs={len(segs)} trusted={gt_trusted} "
              f"frames={len(frames_meta)} dur={dur:.0f}s", flush=True)
    print("STATS", json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
