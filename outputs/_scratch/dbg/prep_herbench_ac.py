"""Port (not an edit) of legacy/experiments/herbench/prep_ac_frames.py using the ffmpeg CLI (no PyAV).
HERBench Action-Counting questions -> MMRED-style frame samples with per-frame labels:
  ev_fill<N>: true_count evidence frames (at t + 0.3 s) + fillers sampled >= --margin s from every
  occurrence, N frames total, chronological. Questions with true_count > N-2 are skipped.
Layout: <out>/ev_fill<N>/<qid>/{000.jpg..}, meta.json (question, pair, count, per-frame times+labels).
--video-index i / --n-videos k : process the i-th shard of the video list (job array)."""
import argparse, json, random, subprocess, sys
from pathlib import Path


def parse_ts(ts):
    m, s, ms = ts.split(":"); return int(m) * 60 + int(s) + int(ms) / 1000.0


def grab(video, t, out, size):
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale='if(gt(iw,ih),{size},-2)':'if(gt(iw,ih),-2,{size})'", "-q:v", "3", str(out)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and Path(out).exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/herbench_ac/ac_questions.json"); ap.add_argument("--video-root", default="data/herbench_raw")
    ap.add_argument("--out", default="data/herbench_ac"); ap.add_argument("--ns", default="16+32"); ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--offset", type=float, default=0.3); ap.add_argument("--size", type=int, default=448); ap.add_argument("--per-video", type=int, default=6)
    ap.add_argument("--video-index", type=int, default=0); ap.add_argument("--n-videos", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); qs = json.load(open(a.questions)); vids = sorted({q["video"] for q in qs})
    mine = [v for i, v in enumerate(vids) if i % a.n_videos == a.video_index]; rng = random.Random(a.seed)
    done = 0; skipped = 0
    for v in mine:
        vp = Path(a.video_root) / v
        if not vp.exists(): print("missing video", vp, flush=True); continue
        qv = [q for q in qs if q["video"] == v]; rng.shuffle(qv); qv = qv[:a.per_video]
        for q in qv:
            ev = sorted(parse_ts(t) for t in q["timestamps"]); dur = q["duration"] or (max(ev) + 10)
            for N in [int(x) for x in a.ns.split("+")]:
                K = len(ev)
                if K > N - 2 or K == 0: skipped += 1; continue
                fill = []; tries = 0
                while len(fill) < N - K and tries < 5000:
                    tries += 1; t = rng.uniform(1.0, max(2.0, dur - 1.0))
                    if all(abs(t - e) >= a.margin for e in ev) and all(abs(t - f) >= 1.0 for f in fill): fill.append(t)
                if len(fill) < N - K: skipped += 1; continue
                frames = sorted([(e + a.offset, 1) for e in ev] + [(f, 0) for f in fill])
                d = Path(a.out) / f"ev_fill{N}" / q["qid"]; d.mkdir(parents=True, exist_ok=True); ok = True
                for i, (t, lab) in enumerate(frames):
                    if not grab(vp, t, d / f"{i:03d}.jpg", a.size): ok = False; break
                if not ok: skipped += 1; continue
                json.dump(dict(qid=q["qid"], video=v, pair=q["pair"], true_count=K, N=N, question=q["question"], answer=q["answer"],
                               frames=[dict(idx=i, t=t, label=lab) for i, (t, lab) in enumerate(frames)]), open(d / "meta.json", "w"))
                done += 1
        print(f"video {v} done (cum {done} samples, {skipped} skipped)", flush=True)
    print(f"-> {a.out}: {done} samples written, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
