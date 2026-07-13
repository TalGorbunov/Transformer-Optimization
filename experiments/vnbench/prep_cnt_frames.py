#!/usr/bin/env python3
"""P3: prep VNBench counting items into frame samples with EXACT per-frame GT.

VNBench (VideoNIAH, arXiv 2406.09367) counting: 450 videos (44–60 s), needles inserted/edited
at known start times (json `needle_time`, one entry per occurrence; gt = len(needle_time),
range 1–16). Unlike MLVU, GT is exact — per-frame is_evidence = frame time within
[t, t+--needle-dur] of any needle start.

Output: data/vnbench_cnt/<qid>/frame_XXX.jpg (uniform --n-frames at --size px) + meta.json
(HERBench-compatible: question, answer, per-frame is_evidence, visible_count, options).
One dir per unique VIDEO (the 4 tries share videos; we keep try=0's question/options).
Needs PyAV (PYTHONPATH=~/.local/pyav-py39). Idempotent.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def extract_uniform(video_path: Path, n_frames: int, size: int):
    import av
    from PIL import Image
    with av.open(str(video_path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        dur = float(st.duration * st.time_base) if st.duration else float(c.duration / av.time_base)
        tgt = list(np.linspace(0.02 * dur, 0.98 * dur, n_frames))
        saved = {}
        ti = 0
        for fr in c.decode(st):
            t = fr.time
            if t is None or ti >= len(tgt):
                continue
            if t >= tgt[ti]:
                img = fr.to_image()
                w, h = img.size
                sc = size / max(w, h)
                saved[float(t)] = img.resize((max(1, round(w * sc)), max(1, round(h * sc))),
                                             Image.LANCZOS)
                while ti < len(tgt) and t >= tgt[ti]:
                    ti += 1
    return saved, dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", default="/home/tal.gorbunov/vnbench_transient/VNBench_new")
    ap.add_argument("--json", default="", help="VNBench-main-4try.json (default: HF cache)")
    ap.add_argument("--out", default="data/vnbench_cnt")
    ap.add_argument("--n-frames", type=int, default=128)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--needle-dur", type=float, default=1.5,
                    help="assumed needle duration (s) after each start time")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.json:
        items = json.loads(Path(args.json).read_text())
    else:
        from huggingface_hub import hf_hub_download
        items = json.loads(Path(hf_hub_download(
            "videoniah/VNBench", "VNBench-main-4try.json", repo_type="dataset")).read_text())
    cnt = [x for x in items if x["type"].startswith("cnt") and x.get("try", 0) == 0]
    vroot = Path(args.videos_root)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    stats = {"prepped": 0, "skipped": 0, "missing": 0}
    done = 0
    for it in sorted(cnt, key=lambda x: x["video"]):
        vname = Path(it["video"]).name
        qid = Path(vname).stem
        qdir = out_root / qid
        if (qdir / "meta.json").exists():
            stats["skipped"] += 1
            continue
        vpath = vroot / vname
        if not vpath.exists():
            stats["missing"] += 1
            continue
        if args.limit and done >= args.limit:
            break
        try:
            saved, dur = extract_uniform(vpath, args.n_frames, args.size)
        except Exception as exc:
            print(f"{qid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        needles = [(float(t), float(t) + args.needle_dur) for t in it["needle_time"]]
        qdir.mkdir(parents=True, exist_ok=True)
        fm = []
        for i, (t, img) in enumerate(sorted(saved.items())):
            img.save(qdir / f"frame_{i:03d}.jpg", quality=90)
            isev = any(a <= t <= b for a, b in needles)
            fm.append({"time": round(t, 2), "is_evidence": bool(isev)})
        meta = {"question_id": qid, "video": vname, "question": it["question"],
                "answer": int(it["gt"]), "options": it["options"],
                "gt_option": it.get("gt_option"), "duration": dur,
                "needle_time": it["needle_time"], "needle_dur": args.needle_dur,
                "n_frames": len(fm), "frames": fm,
                "visible_count": sum(1 for f in fm if f["is_evidence"]),
                "type": it["type"], "gt_source": "needle_time_exact"}
        (qdir / "meta.json").write_text(json.dumps(meta, indent=1))
        stats["prepped"] += 1
        done += 1
        if done % 25 == 0:
            print(f"  {done} prepped", flush=True)
    print("STATS", json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
