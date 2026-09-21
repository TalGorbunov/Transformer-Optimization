#!/usr/bin/env python3
"""Extract +/-delta clip units for the HERBench retrieve-opt probe sweep (Phase 1).

A "unit" = one clip = `--frames-per-clip` frames evenly spanning [c-delta, c+delta]
around a center time c. Per (delta, res) arm we emit, for every armA question:

  POSITIVES: one clip per queried-pair occurrence t_i, centered at c = t_i +
             --evidence-offset. Dropped if the nearest SAME-PAIR occurrence is <= 2*delta
             away (clip-purity: window must not swallow a neighbour) or if the window
             runs past the --edge-safe video bounds.
  NEGATIVES: matched count (= #positives kept for that question), half "random-away"
             (center >= --margin s from every occurrence of the pair) and half "hard"
             (center just outside an occurrence window: |c - t_i| ~ 2*delta+buffer, but
             the whole window [c-delta,c+delta] contains NO occurrence timestamp).

Source videos are login-node-local (/scratch, volatile); this script is meant to run
nice'd on the login node (Tal-approved 2026-08-02) writing frames to shared data/.
occurrence_timestamps / pair / video_id come straight from the prepped armA meta.json;
durations/fps come from the video via PyAV. NO HF-dataset reload.

Each unit -> <out-root>/<arm>/<clip_id>/frame_00..NN.jpg + meta.json
  meta: clip_id, question_id, video_id, pair, verb, label(1/0), kind, delta, res,
        center_time, frame_times, n_frames
plus <out-root>/<arm>/manifest.json (counts, args, per-verb tallies, drops).

Run (login node):
  source .venv/bin/activate
  PYTHONPATH=~/.local/pyav-py39 nice -n 15 python \
    scripts/herbench_retrieve_opt/extract_clips.py \
    --armA-root data/herbench_ac/armA_evidence_only \
    --video-root /scratch/.../herbench_videos \
    --out-root data/herbench_retrieve_opt_clips \
    --delta 0.5 --res 448
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path


def far_from_all(t, times, margin):
    return all(abs(t - e) >= margin for e in times)


def video_path(video_root: Path, video_id: str) -> Path:
    return video_root / "HD_EPIC" / video_id.split("-")[0] / f"{video_id}.mp4"


def video_duration(vp: Path):
    import av
    with av.open(str(vp)) as c:
        s = c.streams.video[0]
        if s.duration:
            return float(s.duration * s.time_base)
        return c.duration / 1e6 if c.duration else None


def _resize(img, res: int):
    from PIL import Image
    w, h = img.size
    sc = res / max(w, h)
    return img.resize((round(w * sc), round(h * sc)), Image.LANCZOS) if sc != 1 else img


def decode_times(vp: Path, times, res: int, cluster_gap: float = 3.0):
    """Decode the frame at/after each requested time. Groups nearby times into clusters and
    does ONE seek per cluster, decoding forward — far cheaper than a seek per frame when the
    requested times come in tight bunches (our +/-delta clips span ~1 s). Same frame-selection
    rule as the legacy extractor (first frame with time >= t - 1/30). Returns {t: PIL.Image}."""
    import av
    times = sorted(times)
    if not times:
        return {}
    clusters, cur = [], [times[0]]
    for t in times[1:]:
        if t - cur[-1] <= cluster_gap:
            cur.append(t)
        else:
            clusters.append(cur); cur = [t]
    clusters.append(cur)

    out = {}
    with av.open(str(vp)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for cl in clusters:
            pending = list(cl)  # sorted requested times in this cluster
            container.seek(int((cl[0] - 0.2) / tb), stream=stream)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                while pending and frame.time >= pending[0] - (1 / 30):
                    out[pending[0]] = _resize(frame.to_image(), res)
                    pending.pop(0)
                if not pending:
                    break
    return out


def clip_frame_times(center, delta, n):
    """n frames evenly spanning [center-delta, center+delta] (n>=1; n=1 -> [center])."""
    if n <= 1:
        return [center]
    return [center - delta + 2 * delta * i / (n - 1) for i in range(n)]


def nearest_same_pair_gap(t, occ):
    others = [abs(t - o) for o in occ if o != t]
    return min(others) if others else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--armA-root", default="data/herbench_ac/armA_evidence_only")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out-root", default="data/herbench_retrieve_opt_clips")
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--frames-per-clip", type=int, default=5)
    ap.add_argument("--evidence-offset", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=5.0, help="random-away neg min dist from any occ")
    ap.add_argument("--hard-buffer", type=float, default=0.5, help="extra gap past 2*delta for hard negs")
    ap.add_argument("--edge", type=float, default=2.0, help="keep clip window this far from video bounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first K questions")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    armA = Path(args.armA_root)
    video_root = Path(args.video_root)
    delta, res, nfr, edge = args.delta, args.res, args.frames_per_clip, args.edge
    arm = f"d{delta:g}_r{res}"
    out_arm = Path(args.out_root) / arm
    out_arm.mkdir(parents=True, exist_ok=True)

    metas = []
    for d in sorted(armA.iterdir()):
        mp = d / "meta.json"
        if mp.exists():
            metas.append(json.loads(mp.read_text()))
    if args.limit:
        metas = metas[: args.limit]

    # group questions by video so we decode each video once
    by_video = defaultdict(list)
    for m in metas:
        by_video[m["video_id"]].append(m)

    manifest = {"arm": arm, "args": vars(args), "units": [],
                "counts": defaultdict(int), "per_verb": defaultdict(lambda: defaultdict(int)),
                "drops": defaultdict(int), "videos_missing": []}
    tot_pos = tot_neg = 0

    for vi, (vid, qs) in enumerate(sorted(by_video.items())):
        vp = video_path(video_root, vid)
        if not vp.exists():
            manifest["videos_missing"].append(vid)
            continue
        dur = video_duration(vp)
        if dur is None:
            manifest["videos_missing"].append(vid)
            continue

        # ---- plan all clips for this video, collect needed frame times ----
        plans = []  # (clip_id, verb, label, kind, center, [times])
        need = set()
        for m in qs:
            qid = m["question_id"]
            pair = m["pair"]
            verb = pair.split()[0]
            occ = sorted(m["occurrence_timestamps"])

            def window_ok(c):
                return (c - delta) >= edge and (c + delta) <= (dur - edge)

            def window_clean(c):  # no occurrence inside [c-delta, c+delta]
                return all(not (c - delta <= o <= c + delta) for o in occ)

            # positives
            kept_pos = []
            for k, t in enumerate(occ):
                c = t + args.evidence_offset
                if nearest_same_pair_gap(t, occ) <= 2 * delta:
                    manifest["drops"]["pos_unsafe_gap"] += 1
                    continue
                if not window_ok(c):
                    manifest["drops"]["pos_edge"] += 1
                    continue
                cid = f"{qid}_pos{k}"
                times = clip_frame_times(c, delta, nfr)
                plans.append((cid, verb, 1, "pos", c, times))
                need.update(times)
                kept_pos.append(t)

            npos = len(kept_pos)
            if npos == 0:
                continue
            n_hard = npos // 2
            n_rand = npos - n_hard

            # hard negatives: just outside an occurrence window
            hard_centers = []
            cand = []
            for t in occ:
                for sign in (+1, -1):
                    cand.append(t + sign * (2 * delta + args.hard_buffer))
            rng.shuffle(cand)
            for c in cand:
                if len(hard_centers) >= n_hard:
                    break
                if window_ok(c) and window_clean(c) and far_from_all(c, hard_centers, 2 * delta):
                    hard_centers.append(c)
            # random-away negatives
            rand_centers = []
            tries = 0
            lo, hi = edge + delta, dur - edge - delta
            while len(rand_centers) < n_rand and tries < 4000 and hi > lo:
                tries += 1
                c = rng.uniform(lo, hi)
                if (far_from_all(c, occ, args.margin)
                        and far_from_all(c, rand_centers, 2 * delta)
                        and far_from_all(c, hard_centers, 2 * delta)):
                    rand_centers.append(c)

            for k, c in enumerate(hard_centers):
                cid = f"{qid}_negH{k}"
                times = clip_frame_times(c, delta, nfr)
                plans.append((cid, verb, 0, "negH", c, times)); need.update(times)
            for k, c in enumerate(rand_centers):
                cid = f"{qid}_negR{k}"
                times = clip_frame_times(c, delta, nfr)
                plans.append((cid, verb, 0, "negR", c, times)); need.update(times)
            if len(hard_centers) + len(rand_centers) < npos:
                manifest["drops"]["neg_short"] += npos - (len(hard_centers) + len(rand_centers))

        if not plans:
            continue
        print(f"[{vi+1}/{len(by_video)}] {vid}: {len(plans)} clips, {len(need)} frames "
              f"(dur {dur:.0f}s)", flush=True)
        frames = decode_times(vp, sorted(need), res)

        for cid, verb, label, kind, c, times in plans:
            missing = [t for t in times if t not in frames]
            if missing:
                manifest["drops"]["frame_decode_fail"] += 1
                continue
            d = out_arm / cid
            d.mkdir(parents=True, exist_ok=True)
            ft = []
            for i, t in enumerate(times):
                frames[t].save(d / f"frame_{i:02d}.jpg", quality=args.jpeg_quality)
                ft.append(round(t, 3))
            qid = cid.rsplit("_", 1)[0]
            meta = {"clip_id": cid, "question_id": qid,
                    "video_id": vid,
                    "pair": next(m["pair"] for m in qs if m["question_id"] == qid),
                    "verb": verb, "label": label, "kind": kind, "delta": delta, "res": res,
                    "center_time": round(c, 3), "frame_times": ft, "n_frames": len(ft)}
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            manifest["units"].append(cid)
            manifest["counts"][kind] += 1
            manifest["per_verb"][verb][kind] += 1
            if label == 1:
                tot_pos += 1
            else:
                tot_neg += 1

    manifest["counts"] = dict(manifest["counts"])
    manifest["per_verb"] = {v: dict(c) for v, c in manifest["per_verb"].items()}
    manifest["drops"] = dict(manifest["drops"])
    manifest["total_pos"] = tot_pos
    manifest["total_neg"] = tot_neg
    (out_arm / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n[{arm}] pos={tot_pos} neg={tot_neg} units={len(manifest['units'])} "
          f"drops={manifest['drops']}")
    print("per-verb (top):", {v: c for v, c in sorted(
        manifest["per_verb"].items(), key=lambda kv: -sum(kv[1].values()))[:6]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
