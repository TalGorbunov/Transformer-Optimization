#!/usr/bin/env python3
"""Phase 0 inventory + gap analysis for the HERBench retrieve-opt campaign.

CPU-only. Reads the prepped armA/armB meta.json files under data/herbench_ac/ (which
already carry occurrence_timestamps in seconds, the queried pair, video_id and
true_count) and the source videos (durations/fps via PyAV) to produce the tables the
Phase-0 gate needs:

  * verb-class distribution + per-class sample counts (verb = first token of `pair`)
  * per-question intra-pair inter-occurrence gap distribution (per video and per verb)
    -> the constraint on the SAFE delta grid (a +/-delta clip must not swallow a
       neighbouring occurrence OF THE SAME PAIR)
  * video inventory: duration, fps, native resolution, reachability
  * disk estimate for the clip frames the Phase-1 sweep would extract

Nothing is extracted here. Output: a JSON report to --out (default stdout) plus a
human table to stderr. PyAV is optional (only for durations/fps/native-res); pass
--no-video to skip it and run purely off meta.json.

Run:
  source .venv/bin/activate
  PYTHONPATH=~/.local/pyav-py39 python scripts/herbench_retrieve_opt/phase0_inventory.py \
    --data-root data/herbench_ac \
    --video-root /scratch/tmp/.../herbench_videos \
    --out outputs/herbench_retrieve_opt/phase0/inventory.json
"""
from __future__ import annotations
import argparse, json, sys, statistics as st
from collections import defaultdict
from pathlib import Path


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_samples(arm_dir: Path):
    out = []
    for d in sorted(arm_dir.iterdir()):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        out.append(m)
    return out


def video_info(video_root: Path, video_id: str):
    """Return (path, duration_s, fps, w, h) or (path, None, ...) if unreachable."""
    import av
    # HD_EPIC/Pxx/<video_id>.mp4  ; participant = first token before '-'
    part = video_id.split("-")[0]
    vp = video_root / "HD_EPIC" / part / f"{video_id}.mp4"
    if not vp.exists():
        return str(vp), None, None, None, None
    try:
        with av.open(str(vp)) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else (
                c.duration / 1e6 if c.duration else None)
            fps = float(s.average_rate) if s.average_rate else None
            w, h = s.codec_context.width, s.codec_context.height
        return str(vp), dur, fps, w, h
    except Exception as e:  # noqa
        return str(vp), None, None, None, f"ERR:{e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/herbench_ac")
    ap.add_argument("--video-root", default=None,
                    help="dir containing HD_EPIC/Pxx/*.mp4 (for durations/fps)")
    ap.add_argument("--no-video", action="store_true", help="skip PyAV video probing")
    ap.add_argument("--delta-grid", default="0.5,1,2",
                    help="candidate +/-delta (s) to test for occurrence-swallow safety")
    ap.add_argument("--frames-per-clip", type=int, default=5,
                    help="frames sampled per +/-delta clip (disk estimate)")
    ap.add_argument("--kb-per-frame-448", type=float, default=38.0)
    ap.add_argument("--kb-per-frame-672", type=float, default=80.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    armA = load_samples(data_root / "armA_evidence_only")
    deltas = [float(x) for x in args.delta_grid.split(",")]

    # ---- verb / pair distribution (over armA questions) ----
    by_verb = defaultdict(list)      # verb -> [question_id]
    by_pair = defaultdict(list)
    counts_hist = defaultdict(int)   # true_count -> n questions
    for m in armA:
        pair = m["pair"]
        verb = pair.split()[0]
        by_verb[verb].append(m["question_id"])
        by_pair[pair].append(m["question_id"])
        counts_hist[m["true_count"]] += 1

    # ---- intra-pair inter-occurrence gaps (per question) ----
    # gap = consecutive-diff of a single question's occurrence_timestamps (same pair).
    all_gaps = []
    gaps_by_verb = defaultdict(list)
    gaps_by_video = defaultdict(list)
    multi_occ_questions = 0
    total_occurrences = 0
    for m in armA:
        ts = sorted(m["occurrence_timestamps"])
        total_occurrences += len(ts)
        if len(ts) >= 2:
            multi_occ_questions += 1
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            all_gaps.extend(gaps)
            gaps_by_verb[m["pair"].split()[0]].extend(gaps)
            gaps_by_video[m["video_id"]].extend(gaps)

    # For a +/-delta clip centred on an occurrence to be "clean", the nearest same-pair
    # neighbour must be > 2*delta away (else the clip window [t-delta, t+delta] would
    # touch the neighbour's). Count violations per delta.
    min_gap_per_occ = []  # for each occurrence that has a same-pair neighbour, the nearest gap
    for m in armA:
        ts = sorted(m["occurrence_timestamps"])
        for i, t in enumerate(ts):
            neigh = []
            if i > 0:
                neigh.append(t - ts[i - 1])
            if i < len(ts) - 1:
                neigh.append(ts[i + 1] - t)
            if neigh:
                min_gap_per_occ.append(min(neigh))
    delta_safety = {}
    for d in deltas:
        # occurrence clean if nearest neighbour gap > 2d (window half-width d each side)
        unsafe = sum(1 for g in min_gap_per_occ if g <= 2 * d)
        delta_safety[d] = {
            "n_occ_with_neighbour": len(min_gap_per_occ),
            "n_unsafe_2d": unsafe,
            "frac_unsafe": round(unsafe / len(min_gap_per_occ), 4) if min_gap_per_occ else 0.0,
        }

    # ---- video inventory ----
    videos = sorted({m["video_id"] for m in armA})
    vinfo = {}
    if not args.no_video and args.video_root:
        vroot = Path(args.video_root)
        for vid in videos:
            p, dur, fps, w, h = video_info(vroot, vid)
            vinfo[vid] = {"path": p, "dur_s": dur, "fps": fps, "w": w, "h": h,
                          "reachable": dur is not None}

    # ---- disk estimate ----
    n_pos = total_occurrences                      # one positive clip per occurrence
    n_neg = total_occurrences                      # matched negatives
    n_clips = n_pos + n_neg
    fr = args.frames_per_clip
    disk = {}
    for res, kb in (("448", args.kb_per_frame_448), ("672", args.kb_per_frame_672)):
        # per delta value we re-extract (frames differ); grid = clip arms
        per_delta_mb = n_clips * fr * kb / 1024.0
        disk[res] = {
            "clips": n_clips, "frames_per_clip": fr,
            "mb_per_delta": round(per_delta_mb, 1),
            "mb_full_grid": round(per_delta_mb * len(deltas), 1),
        }

    gap_stats = None
    if all_gaps:
        gap_stats = {
            "n_gaps": len(all_gaps),
            "min": round(min(all_gaps), 3), "p05": round(pct(all_gaps, 5), 3),
            "p10": round(pct(all_gaps, 10), 3), "p25": round(pct(all_gaps, 25), 3),
            "median": round(st.median(all_gaps), 3),
            "p75": round(pct(all_gaps, 75), 3), "max": round(max(all_gaps), 3),
        }

    report = {
        "data_root": str(data_root),
        "n_armA_questions": len(armA),
        "n_videos": len(videos),
        "total_occurrences": total_occurrences,
        "multi_occurrence_questions": multi_occ_questions,
        "true_count_hist": dict(sorted(counts_hist.items())),
        "verb_distribution": {v: len(q) for v, q in sorted(by_verb.items(),
                              key=lambda kv: -len(kv[1]))},
        "n_distinct_verbs": len(by_verb),
        "n_distinct_pairs": len(by_pair),
        "pair_distribution_top": dict(sorted(
            {p: len(q) for p, q in by_pair.items()}.items(),
            key=lambda kv: -kv[1])[:25]),
        "intra_pair_gap_stats_s": gap_stats,
        "gap_per_verb_median_s": {v: round(st.median(g), 3)
                                  for v, g in sorted(gaps_by_verb.items()) if g},
        "delta_safety": delta_safety,
        "video_inventory": vinfo,
        "n_videos_reachable": sum(1 for v in vinfo.values() if v.get("reachable")),
        "disk_estimate_mb": disk,
    }

    txt = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
        print("wrote", args.out, file=sys.stderr)
    else:
        print(txt)

    # human summary to stderr
    print("\n=== PHASE 0 SUMMARY ===", file=sys.stderr)
    print(f"armA questions: {len(armA)}  videos: {len(videos)}  "
          f"occurrences: {total_occurrences}  multi-occ Qs: {multi_occ_questions}",
          file=sys.stderr)
    print(f"verbs: {len(by_verb)}  pairs: {len(by_pair)}", file=sys.stderr)
    if gap_stats:
        print(f"intra-pair gaps (s): min={gap_stats['min']} p05={gap_stats['p05']} "
              f"p10={gap_stats['p10']} median={gap_stats['median']}", file=sys.stderr)
    for d, s in delta_safety.items():
        print(f"  delta={d}s -> {s['n_unsafe_2d']}/{s['n_occ_with_neighbour']} "
              f"occ unsafe ({s['frac_unsafe']*100:.1f}%)", file=sys.stderr)
    if vinfo:
        print(f"videos reachable: {report['n_videos_reachable']}/{len(videos)}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
