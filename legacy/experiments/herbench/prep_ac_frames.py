#!/usr/bin/env python3
"""Prep HERBench Action-Counting questions into MMRED-style frame samples.

Two arms per question (frames chronological in both):
  armA_evidence_only : one frame per required_timestamp -> N = true_count, all evidence.
                       Pure-aggregation behavioral test (count the shown occurrences).
  armB_ev_fill16     : true_count evidence frames + fillers to N=16 from the same video,
                       >= --margin s away from every occurrence of the queried pair.
                       Binary per-frame labels -> the d'/parity test. Rows with
                       true_count > --max-count-b are skipped for this arm.

Timestamps are "M:SS:ms" (minute field unbounded). Evidence frame taken at
t + --evidence-offset (default 0.3 s, mid-action). Frames resized so the long
side is --size px, saved as JPEG q90 with a meta.json per question
(question/choices/answer/per-frame labels+times) and a top-level manifest.

Needs PyAV (not in the shared .venv): PYTHONPATH=~/.local/pyav-py39 or similar.
CPU-only. ~28 videos x ~100 seeks, a few minutes total.
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def parse_ts(ts: str) -> float:
    m, s, ms = ts.split(":")
    return int(m) * 60 + int(s) + int(ms) / 1000.0


def far_from_all(t: float, ev_times: list, margin: float) -> bool:
    return all(abs(t - e) >= margin for e in ev_times)


def extract_frames(video_path: Path, times: list, size: int):
    """Decode the frame at each requested time (sorted seeks). Returns {t: PIL.Image}."""
    import av
    from PIL import Image
    out = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for t in sorted(times):
            container.seek(int(t / tb), stream=stream)
            for frame in container.decode(stream):
                if frame.time is None or frame.time >= t - (1 / 30):
                    img = frame.to_image()
                    w, h = img.size
                    sc = size / max(w, h)
                    if sc < 1:
                        img = img.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
                    out[t] = img
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True, help="dir containing HD_EPIC/Pxx/*.mp4")
    ap.add_argument("--out-root", default="data/herbench_ac")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--n-total-b", type=int, default=16, help="arm B total frames")
    ap.add_argument("--max-count-b", type=int, default=12)
    ap.add_argument("--margin", type=float, default=5.0, help="filler min distance (s) from any occurrence")
    ap.add_argument("--evidence-offset", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=2.0, help="keep frames this far (s) from video edges")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-videos", type=int, default=0, help="smoke: only first K videos")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("DanBenAmi/HERBench", "lite", split="test")
    ac = [r for r in ds if r["task_type"] == "Action Counting"]
    by_video = defaultdict(list)
    for r in ac:
        by_video[r["video_path"].removeprefix("videos/")].append(r)
    videos = sorted(by_video)
    if args.limit_videos:
        videos = videos[: args.limit_videos]

    rng = random.Random(args.seed)
    out_root = Path(args.out_root)
    manifest = {"armA_evidence_only": [], "armB_ev_fill16": []}
    skipped = []

    for vi, vp in enumerate(videos):
        vfile = Path(args.video_root) / vp
        if not vfile.exists():
            skipped.append((vp, "video file missing")); continue
        rows = by_video[vp]
        plans = []  # (row, armA_times, armB_times or None, ev_times)
        need_times = set()
        for r in rows:
            meta = json.loads(r["metadata_json"])
            dur = float(meta["duration"])
            ev = [parse_ts(t) for t in meta["required_timestamps"]]
            if any(t >= dur or t < 0 for t in ev):
                skipped.append((r["question_id"], "timestamp outside duration")); continue
            ev_f = [min(max(t + args.evidence_offset, args.edge), dur - args.edge) for t in ev]
            armB = None
            if len(ev) <= args.max_count_b:
                fillers, tries = [], 0
                while len(fillers) < args.n_total_b - len(ev) and tries < 4000:
                    tries += 1
                    t = rng.uniform(args.edge, dur - args.edge)
                    if far_from_all(t, ev, args.margin) and far_from_all(t, fillers, 1.0):
                        fillers.append(t)
                if len(fillers) == args.n_total_b - len(ev):
                    armB = sorted([(t, 1) for t in ev_f] + [(t, 0) for t in fillers])
                else:
                    skipped.append((r["question_id"], "could not place fillers"))
            plans.append((r, sorted(ev_f), armB, ev))
            need_times.update(ev_f)
            if armB:
                need_times.update(t for t, _ in armB)
        if not plans:
            continue
        print(f"[{vi+1}/{len(videos)}] {vp}: {len(plans)} questions, {len(need_times)} frames", flush=True)
        frames = extract_frames(vfile, list(need_times), args.size)

        for r, ev_f, armB, ev in plans:
            meta_src = json.loads(r["metadata_json"])
            base_meta = {
                "question_id": r["question_id"], "video_id": r["video_id"],
                "question": r["question"], "choices": r["choices"],
                "answer": r["answer"], "answer_text": r["answer_text"],
                "pair": " ".join(meta_src["pair"]),
                "true_count": len(ev), "occurrence_timestamps": ev,
                "source": "HERBench lite / Action Counting",
            }
            arms = {"armA_evidence_only": [(t, 1) for t in ev_f]}
            if armB:
                arms["armB_ev_fill16"] = armB
            for arm, tl in arms.items():
                d = out_root / arm / r["question_id"]
                d.mkdir(parents=True, exist_ok=True)
                labels = []
                for i, (t, is_ev) in enumerate(tl):
                    frames[t].save(d / f"frame_{i:02d}.jpg", quality=90)
                    labels.append({"idx": i, "time": round(t, 3), "is_evidence": is_ev})
                meta = dict(base_meta)
                meta["frames"] = labels
                meta["n_frames"] = len(labels)
                meta["visible_count"] = sum(l["is_evidence"] for l in labels)
                (d / "meta.json").write_text(json.dumps(meta, indent=1))
                manifest[arm].append(r["question_id"])

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(
        {"arms": {k: sorted(v) for k, v in manifest.items()},
         "counts": {k: len(v) for k, v in manifest.items()},
         "skipped": skipped, "args": vars(args)}, indent=1))
    print(f"\narmA: {len(manifest['armA_evidence_only'])}  armB: {len(manifest['armB_ev_fill16'])}"
          f"  skipped: {len(skipped)}")
    for s in skipped[:10]:
        print("  skip:", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
