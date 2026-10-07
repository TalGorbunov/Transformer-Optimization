#!/usr/bin/env python3
"""Prepare a real-video MCQ benchmark (HERBench lite_v2, MINERVA) into the shared on-disk layout
that core/data/videoqa.py reads (its docstring is the layout spec). docs/SCALEUP_2026-09-23.md §2–3.

Stages (--stage, one or more, in this order):
  rows     hf/ annotations -> json/test.json. HERBench: lite_v2 rows, task codes, evidence intervals
           parsed per task (core.data.videoqa.parse_herbench_evidence). MINERVA: lmms minerva.json,
           `Listening` dropped (needs ASR), evidence = timestamps regex-parsed from the reasoning.
  videos   HERBench: sha256-check the four lite tar parts under hf/videos/, stream-extract the mp4s
           into videos/ (no joined copy), delete the parts, check every row's video exists.
           MINERVA: every video blob of the lmms Lance table -> videos/<video_id>.mp4 (NETWORK:
           run on the login node with HF_HUB_OFFLINE=0, nice'd; it is I/O, not compute).
  frames   per video (worker pool, SLURM_CPUS_PER_TASK): POOL frames on the official grid
           t_i = i*dur/POOL -> pool/<video_id>/frame_%04d.jpg + times.json; per question: one frame per
           evidence unit (points at t+0.3 s, ranges at the midpoint) -> evidence/<qid>/ + times.json.
           --frame-set native (default): the video's own resolution, as the benchmarks' own evaluation
           code feeds it (HERBench base_vlm.py decodes with OpenCV and passes the frame unresized);
           --frame-set 512: long side 512 px (the first extraction). JPEG q95, RGB. The resolution the
           MODEL sees is set per run by the processor's own max_pixels knob, never by resize code here.
           PyAV seek-to-keyframe + decode forward.
           Resumable (a video with a complete pool + evidence dirs is skipped).
  tar      pool.tar + evidence.tar for sbatch/lib/common.sh:stage_video.
  verify   counts, per-task coverage / k / skips for planted x uniform x N -> verify_report.txt.
  units    the UNIT baseline's evidence units (plan 2026-10-06): for the rows of --qids-file (default: every row
           with point / range evidence), per evidence interval a 9-frame grid at the video's own resolution
           (core.data.videoqa.unit_grid: points t+0.3 s +- 2 s at 0.5 s; ranges evenly over [a, b]) plus one
           hard negative (grid starting 2 s after the interval) and one easy negative (seeded, >= 5 s from every
           interval) -> units_native/<qid>/<uid>/frame_%02d.jpg + units.json (core.data.videoqa.plan_units).
           Resumable (a qid with units.json is skipped). One decode pass per video.
  verify_units  per task: rows / units / grid sizes / negatives present, frame files, hard-negative clearance,
           disk size -> verify_report_units.txt.
Usage:
  python experiments/prepare_video.py --dataset herbench --root data/herbench_v2 --stage rows videos
  sbatch -p l40s-shared --qos=4h_0g --time=04:00:00 --export=ALL,DATASET=herbench sbatch/prepare_video.sbatch
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.data import get_dataset  # noqa: E402
from core.data.mcq import OPTIONS, lettered  # noqa: E402
from core.data.videoqa import (  # noqa: E402
    EASY_MIN_DIST, HARD_GAP, NEG_MARGIN, UNIT_HALF, UNIT_STEP, UNITS_DIR, HERBENCH_CODE, MINERVA_CODE, N_FRAMES, POOL,
    evidence_times, frame_dirs, parse_herbench_evidence, parse_minerva_evidence, plan_units, pool_times,
)

JPEG_QUALITY = 95
LONG_SIDE = {"native": 0, "512": 512}      # frame set -> long side in px (0 = the video's own resolution, never resized)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ rows

def rows_herbench(root: Path) -> List[Dict[str, Any]]:
    src = root / "hf" / "data" / "herbench_annotations_lite_v2.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    out = []
    for r in raw:
        meta = r["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else dict(meta)
        code = HERBENCH_CODE[r["task_type"]]
        kind, ivs = parse_herbench_evidence(code, meta)
        out.append({
            "qid": str(r["question_id"]), "video_id": str(r["video_id"]), "video_path": str(r["video_path"]),
            "question": r["question"], "choices": lettered(r["choices"]), "answer": str(r["answer"]).strip().upper(),
            "answer_index": int(r["answer_index"]), "qtype": code, "task_name": r["task_type"],
            "source": meta.get("source_dataset"), "duration": meta.get("duration"),
            "evidence_kind": kind, "evidence": [[a, b] for a, b in ivs],
            "meta": {**meta, "answer_text": r.get("answer_text")},
        })
    return out


def rows_minerva(root: Path) -> List[Dict[str, Any]]:
    src = root / "hf" / "minerva.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    raw = raw if isinstance(raw, list) else raw["data"]
    out, dropped = [], Counter()
    for r in raw:
        qt = str(r["question_type"]).strip()
        if qt == "Listening":
            dropped["Listening"] += 1
            continue
        if qt not in MINERVA_CODE:
            dropped[f"unknown:{qt}"] += 1
            continue
        choices = [str(r[f"answer_choice_{i}"]).strip() for i in range(5)]
        kind, ivs = parse_minerva_evidence(r.get("reasoning", ""))
        out.append({
            "qid": str(r["key"]), "video_id": str(r["video_id"]), "video_path": f"videos/{r['video_id']}.mp4",
            "question": str(r["question"]).strip(), "choices": lettered(choices), "answer": OPTIONS[int(r["answer_id"])],
            "answer_index": int(r["answer_id"]), "qtype": MINERVA_CODE[qt], "task_name": qt,
            "source": f"{r.get('split')}/{r.get('category')}", "duration": None,
            "evidence_kind": kind, "evidence": [[a, b] for a, b in ivs],
            "meta": {"reasoning": r.get("reasoning", ""), "answer_text": r.get("answer"), "split": r.get("split"),
                     "category": r.get("category")},
        })
    log(f"minerva rows: kept {len(out)} dropped {dict(dropped)}")
    return out


def stage_rows(name: str, root: Path) -> None:
    rows = rows_herbench(root) if name == "herbench" else rows_minerva(root)
    (root / "json").mkdir(parents=True, exist_ok=True)
    out = root / "json" / "test.json"
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    kinds = Counter((r["qtype"], r["evidence_kind"]) for r in rows)
    log(f"wrote {out}: {len(rows)} rows, {len({r['video_id'] for r in rows})} videos; "
        f"answers {dict(Counter(r['answer'] for r in rows))}")
    for (q, k), n in sorted(kinds.items()):
        ks = [len(r["evidence"]) for r in rows if r["qtype"] == q and r["evidence_kind"] == k]
        log(f"  {q:24s} {k:7s} n={n:4d} k: mean {sum(ks) / max(1, len(ks)):.2f} max {max(ks) if ks else 0}")


# ------------------------------------------------------------------ videos

class _Concat(io.RawIOBase):
    """Sequential read over several files as one stream (the tar parts, no joined copy)."""

    def __init__(self, paths: Sequence[Path]):
        self._paths = list(paths)
        self._i = 0
        self._f = open(self._paths[0], "rb") if self._paths else None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # type: ignore[override]
        while self._f is not None:
            n = self._f.readinto(b)
            if n:
                return n
            self._f.close()
            self._i += 1
            self._f = open(self._paths[self._i], "rb") if self._i < len(self._paths) else None
        return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_videos_herbench(root: Path, keep_parts: bool) -> None:
    hfv = root / "hf" / "videos"
    parts = [hfv / f"videos.tar.part.{i:02d}" for i in range(4)]
    missing = [p for p in parts if not p.exists()]
    if missing:
        raise SystemExit(f"tar parts missing: {missing}")
    sums = {}
    for line in (hfv / "videos.tar.checksums.txt").read_text().splitlines():
        if line.strip():
            digest, fname = line.split()[:2]
            sums[Path(fname.lstrip("*")).name] = digest
    for p in parts:
        want = sums.get(p.name)
        got = _sha256(p)
        if want and got != want:
            raise SystemExit(f"sha256 MISMATCH {p.name}: {got} != {want}")
        log(f"sha256 ok {p.name}" if want else f"sha256 {p.name} = {got} (no reference)")
    vdir = root / "videos"
    vdir.mkdir(exist_ok=True)
    n_files, n_bytes = 0, 0
    try:
        with tarfile.open(fileobj=io.BufferedReader(_Concat(parts), buffer_size=1 << 24), mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                rel = m.name.lstrip("./")
                rel = rel[len("videos/"):] if rel.startswith("videos/") else rel
                dst = vdir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                with open(dst, "wb") as f:
                    for chunk in iter(lambda: src.read(1 << 24), b""):
                        f.write(chunk)
                n_files += 1
                n_bytes += m.size
                log(f"  extracted {rel} ({m.size / 1e6:.0f} MB)")
    except (tarfile.ReadError, EOFError) as exc:
        log(f"tar stream ended: {exc} (expected at the lite boundary; the last member may be a full-set video)")
    log(f"extracted {n_files} files, {n_bytes / 1e9:.1f} GB")
    rows = json.loads((root / "json" / "test.json").read_text())
    absent = sorted({r["video_path"] for r in rows if not (root / r["video_path"]).exists()})
    if absent:
        raise SystemExit(f"{len(absent)} row videos missing after extraction: {absent[:5]}")
    log(f"all {len({r['video_path'] for r in rows})} row videos present")
    if not keep_parts:
        for p in parts:
            p.unlink()
        log("tar parts deleted")


def stage_videos_minerva(root: Path, lance_uri: Optional[str] = None) -> None:
    """Blobs from the lmms Lance table. Over hf:// each take_blobs re-reads remote pages (~6 min per
    video on 2026-09-23); fetch the table once (`hf/lance/data/train.lance`, 8.85 GB) and read it
    locally: --lance-dir, default = that local copy when present, else hf://."""
    import lance

    vdir = root / "videos"
    vdir.mkdir(exist_ok=True)
    local = root / "hf" / "lance" / "data" / "train.lance"
    uri = lance_uri or (str(local) if local.exists() else "hf://datasets/lmms-lab-eval/minerva/data/train.lance")
    log(f"lance source: {uri}")
    ds = lance.dataset(uri)
    meta = ds.to_table(columns=["video_id", "video_ext", "video_size_bytes"]).to_pylist()
    rows = json.loads((root / "json" / "test.json").read_text())
    want = {r["video_id"] for r in rows}
    log(f"lance rows {len(meta)}, videos wanted by rows {len(want)}, missing from the mirror: "
        f"{len(want - {m['video_id'] for m in meta})}")
    for i, m in enumerate(meta):
        if m["video_id"] not in want:
            continue
        dst = vdir / f"{m['video_id']}.mp4"
        if dst.exists() and dst.stat().st_size == int(m["video_size_bytes"]):
            continue
        t = time.time()
        blob = ds.take_blobs("video_blob", indices=[i])[0]
        data = blob.read() if hasattr(blob, "read") else bytes(blob)
        if m["video_ext"] != "mp4":
            log(f"  NOTE {m['video_id']} ext={m['video_ext']} (saved as .mp4 container name)")
        dst.write_bytes(data)
        log(f"  {m['video_id']} {len(data) / 1e6:.1f} MB in {time.time() - t:.1f}s")
    absent = sorted(v for v in want if not (vdir / f"{v}.mp4").exists())
    log(f"videos present {len(want) - len(absent)}/{len(want)}; missing {absent[:10]}")


# ------------------------------------------------------------------ frames

def _open_video(path: Path):
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if container.duration:
        duration = float(container.duration) / av.time_base
    elif stream.duration is not None:
        duration = float(stream.duration * stream.time_base)
    else:
        raise RuntimeError(f"no duration for {path}")
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    return container, stream, duration, fps


def grab_frames(container, stream, targets: Sequence[float], fps: float, on_frame=None) -> Dict[float, Any]:
    """Decode one frame per target time (sorted ascending): seek to the keyframe at/before the
    target when the decoder is behind by more than a few seconds or ahead of it, then decode
    forward to the first frame at or after the target (EOF -> the last decoded frame). With
    `on_frame(t, image)` the frame is handed over immediately and NOT kept (memory: a 1408² video's
    260 targets would otherwise hold ~1.5 GB per worker — the 2026-09-23 OOM on 4h_0g)."""
    tb = float(stream.time_base)
    tol = 0.5 / fps
    out: Dict[float, Any] = {}
    gen = None
    last_t: Optional[float] = None
    keep = None
    for t in targets:
        if gen is None or last_t is None or t < last_t - tol or t - last_t > 8.0:
            container.seek(int(t / tb), stream=stream, backward=True, any_frame=False)
            gen = container.decode(stream)
            last_t = None
        got = None
        for frame in gen:
            ft = frame.time
            if ft is None:
                continue
            last_t = ft
            keep = frame
            if ft >= t - tol:
                got = frame
                break
        if got is None:                          # EOF before the target (target beyond the last frame)
            if keep is None:
                raise RuntimeError(f"no decodable frame for t={t:.2f}s")
            got = keep
            gen = None
        img = got.to_image()
        if on_frame is not None:
            on_frame(t, img)
        else:
            out[t] = img
    return out


def _save(img, path: Path, long_side: int) -> None:
    from PIL import Image

    im = img.convert("RGB")
    if long_side and max(im.size) > long_side:
        im.thumbnail((long_side, long_side), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=JPEG_QUALITY)


def process_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """One video: pool frames + every question's evidence frames (one decode pass, sorted targets)."""
    root, vid, vpath, pool_n = Path(job["root"]), job["video_id"], job["video_path"], int(job["pool"])
    pool_name, evid_name = frame_dirs(job["frame_set"])
    long_side = LONG_SIDE[job["frame_set"]]
    pdir = root / pool_name / vid
    try:
        container, stream, duration, fps = _open_video(root / vpath)
    except Exception as exc:  # noqa: BLE001
        return {"video_id": vid, "error": f"open: {exc}"}
    pool_t = pool_times(duration, pool_n)
    plan: Dict[float, List[Path]] = defaultdict(list)
    for i, t in enumerate(pool_t):
        plan[t].append(pdir / f"frame_{i + 1:04d}.jpg")
    evid_info = []
    for q in job["questions"]:
        ivs = [tuple(iv) for iv in q["evidence"]]
        kept = [iv for iv in ivs if iv[0] <= duration + 1.0]
        times = evidence_times(kept, duration)
        edir = root / evid_name / q["qid"]
        for j, t in enumerate(times):
            plan[t].append(edir / f"frame_{j + 1:04d}.jpg")
        evid_info.append((edir, {"qid": q["qid"], "video_id": vid, "duration": duration, "times": times,
                                 "units": [list(iv) for iv in kept], "dropped": len(ivs) - len(kept)}))
    def _on_frame(t: float, img) -> None:
        for p in plan[t]:
            _save(img, p, long_side)

    try:
        grab_frames(container, stream, sorted(plan), fps, on_frame=_on_frame)
    except Exception as exc:  # noqa: BLE001
        container.close()
        return {"video_id": vid, "error": f"decode: {exc}"}
    container.close()
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "times.json").write_text(json.dumps({"video_id": vid, "video_path": vpath, "duration": duration, "fps": fps,
                                                  "width": stream.width, "height": stream.height, "pool": pool_n,
                                                  "times": pool_t}))
    for edir, info in evid_info:
        edir.mkdir(parents=True, exist_ok=True)
        (edir / "times.json").write_text(json.dumps(info))
    return {"video_id": vid, "duration": duration, "n_targets": len(plan), "n_questions": len(evid_info),
            "size": [stream.width, stream.height]}


def _complete(root: Path, vid: str, qids: Sequence[str], pool_n: int, frame_set: str) -> bool:
    pool_name, evid_name = frame_dirs(frame_set)
    p = root / pool_name / vid
    if not (p / "times.json").exists() or len(list(p.glob("frame_*.jpg"))) != pool_n:
        return False
    return all((root / evid_name / q / "times.json").exists() for q in qids)


def stage_frames(root: Path, pool_n: int, workers: int, limit: int, max_failed: int = 0, frame_set: str = "native") -> None:
    rows = json.loads((root / "json" / "test.json").read_text())
    by_vid: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        v = by_vid.setdefault(r["video_id"], {"root": str(root), "video_id": r["video_id"], "video_path": r["video_path"],
                                               "pool": pool_n, "frame_set": frame_set, "questions": []})
        if r["evidence_kind"] not in ("all", "none"):
            v["questions"].append({"qid": r["qid"], "evidence": r["evidence"]})
    jobs = [j for j in by_vid.values()
            if not _complete(root, j["video_id"], [q["qid"] for q in j["questions"]], pool_n, frame_set)]
    if limit:
        jobs = jobs[:limit]
    log(f"frames: {len(by_vid)} videos, {len(jobs)} to do, workers={workers}, pool={pool_n}, frame set {frame_set!r} "
        f"(long side {LONG_SIDE[frame_set] or 'native'})")
    t0 = time.time()
    errors = []
    with Pool(processes=max(1, workers)) as pool:
        for k, res in enumerate(pool.imap_unordered(process_video, jobs)):
            if "error" in res:
                errors.append(res)
                log(f"  ERROR {res['video_id']}: {res['error']}")
            else:
                log(f"  {k + 1}/{len(jobs)} {res['video_id']} dur={res['duration']:.0f}s {res['size']} "
                    f"targets={res['n_targets']} q={res['n_questions']} ({time.time() - t0:.0f}s)")
    failed_path = root / "json" / f"failed_videos_{frame_set}.json"
    known = json.loads(failed_path.read_text()) if failed_path.exists() else {}
    known.update({e["video_id"]: e["error"] for e in errors})
    known = {v: e for v, e in known.items() if not (root / frame_dirs(frame_set)[0] / v / "times.json").exists()}
    failed_path.write_text(json.dumps(known, indent=1))
    if known:
        n_rows = sum(1 for r in rows if r["video_id"] in known)
        log(f"{len(known)} videos have no frames ({n_rows} rows, skipped by the loader as no_pool): {sorted(known)[:10]}")
    if len(known) > max_failed:
        raise SystemExit(f"{len(known)} videos failed (> --max-failed {max_failed}): {sorted(known)[:10]}")


# ------------------------------------------------------------------ evidence units (UNIT baseline)

def process_units(job: Dict[str, Any]) -> Dict[str, Any]:
    """One video: every unit grid (pos + hard + easy) of its listed questions, one decode pass."""
    root, vid, vpath, seed = Path(job["root"]), job["video_id"], job["video_path"], int(job["seed"])
    try:
        container, stream, duration, fps = _open_video(root / vpath)
    except Exception as exc:  # noqa: BLE001
        return {"video_id": vid, "error": f"open: {exc}"}
    plan: Dict[float, List[Path]] = defaultdict(list)
    infos: List[Tuple[str, Dict[str, Any]]] = []
    for q in job["questions"]:
        ivs = [tuple(iv) for iv in q["evidence"] if iv[0] <= duration + 1.0]
        qdir = root / UNITS_DIR / q["qid"]
        if not ivs:
            infos.append((q["qid"], {"qid": q["qid"], "video_id": vid, "duration": duration, "k": 0, "units": [],
                                     "dropped": {}, "dropped_beyond_duration": len(q["evidence"])}))
            continue
        info = plan_units(q["qid"], ivs, duration, seed)
        info.update({"video_id": vid, "fps": fps, "size": [stream.width, stream.height],
                     "dropped_beyond_duration": len(q["evidence"]) - len(ivs)})
        for u in info["units"]:
            for i, t in enumerate(u["times"]):
                plan[float(t)].append(qdir / u["uid"] / f"frame_{i + 1:02d}.jpg")
        infos.append((q["qid"], info))

    def _on_frame(t: float, img) -> None:
        for p in plan[t]:
            _save(img, p, 0)                       # native resolution, never resized

    try:
        grab_frames(container, stream, sorted(plan), fps, on_frame=_on_frame)
    except Exception as exc:  # noqa: BLE001
        container.close()
        return {"video_id": vid, "error": f"decode: {exc}"}
    container.close()
    for qid, info in infos:
        qdir = root / UNITS_DIR / qid
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "units.json").write_text(json.dumps(info))
    return {"video_id": vid, "duration": duration, "n_targets": len(plan), "n_questions": len(infos),
            "size": [stream.width, stream.height]}


def stage_units(root: Path, qids_file: Optional[Path], workers: int, limit: int, seed: int) -> None:
    rows = json.loads((root / "json" / "test.json").read_text())
    if qids_file is not None:
        want = [ln.strip() for ln in qids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        by = {r["qid"]: r for r in rows}
        missing = [q for q in want if q not in by]
        if missing:
            raise SystemExit(f"{len(missing)} qids of {qids_file} not in json/test.json: {missing[:5]}")
        rows = [by[q] for q in want]
    rows = [r for r in rows if r["evidence_kind"] not in ("all", "none")]
    todo = [r for r in rows if not (root / UNITS_DIR / r["qid"] / "units.json").exists()]
    by_vid: Dict[str, Dict[str, Any]] = {}
    for r in todo:
        v = by_vid.setdefault(r["video_id"], {"root": str(root), "video_id": r["video_id"], "video_path": r["video_path"],
                                               "seed": seed, "questions": []})
        v["questions"].append({"qid": r["qid"], "evidence": r["evidence"]})
    jobs = list(by_vid.values())
    if limit:
        jobs = jobs[:limit]
    log(f"units: {len(rows)} rows with evidence ({len(rows) - len(todo)} done), {len(jobs)} videos to do, "
        f"workers={workers}, seed={seed} -> {root / UNITS_DIR}")
    t0 = time.time()
    errors = []
    with Pool(processes=max(1, workers)) as pool:
        for k, res in enumerate(pool.imap_unordered(process_units, jobs)):
            if "error" in res:
                errors.append(res)
                log(f"  ERROR {res['video_id']}: {res['error']}")
            else:
                log(f"  {k + 1}/{len(jobs)} {res['video_id']} dur={res['duration']:.0f}s {res['size']} "
                    f"targets={res['n_targets']} q={res['n_questions']} ({time.time() - t0:.0f}s)")
    if errors:
        raise SystemExit(f"{len(errors)} videos failed: {[e['video_id'] for e in errors][:10]}")


def stage_verify_units(name: str, root: Path) -> None:
    from PIL import Image

    spec = get_dataset(name)
    rows = {r["qid"]: r for r in spec.rows(root)}
    infos = sorted((root / UNITS_DIR).glob("*/units.json"))
    per_task: Dict[str, Counter] = defaultdict(Counter)
    grid_sizes: Dict[str, Counter] = defaultdict(Counter)
    bad_files, bad_hard, bad_easy, sizes = [], [], [], Counter()
    n_bytes = 0
    half = UNIT_HALF * UNIT_STEP
    for p in infos:
        info = json.loads(p.read_text())
        qid = info["qid"]
        task = rows[qid]["qtype"] if qid in rows else "?"
        c = per_task[task]
        c["rows"] += 1
        c["k"] += info.get("k", 0)
        ivs = [tuple(u["interval"]) for u in info["units"] if u["kind"] == "pos"]
        for u in info["units"]:
            c[u["kind"]] += 1
            if u["kind"] == "pos":
                grid_sizes[task][len(u["times"])] += 1
            files = [p.parent / u["uid"] / f"frame_{i + 1:02d}.jpg" for i in range(len(u["times"]))]
            miss = [f for f in files if not f.exists()]
            if miss:
                bad_files.append(str(miss[0]))
            else:
                n_bytes += sum(f.stat().st_size for f in files)
                if c["rows"] <= 2 and u["kind"] == "pos":
                    sizes[Image.open(files[0]).size] += 1
            if u["kind"] == "hard":
                cc = float(u["centre"])
                if not all(cc + half <= a - NEG_MARGIN or cc - half >= b + NEG_MARGIN for a, b in ivs):
                    bad_hard.append(u["uid"] + "@" + qid)
            if u["kind"] == "easy":
                cc = float(u["centre"])
                if not all(cc + half <= a - EASY_MIN_DIST or cc - half >= b + EASY_MIN_DIST for a, b in ivs):
                    bad_easy.append(u["uid"] + "@" + qid)
        for kind, n in info.get("dropped", {}).items():
            c[f"dropped_{kind}"] += n
    lines = [f"{name} @ {root / UNITS_DIR}: {len(infos)} questions, {n_bytes / 1e9:.2f} GB, frame sizes {dict(sizes)}",
             f"missing frame files: {len(bad_files)} {bad_files[:3]}; hard negatives inside an interval+margin: {len(bad_hard)} "
             f"{bad_hard[:3]}; easy negatives closer than {EASY_MIN_DIST} s: {len(bad_easy)} {bad_easy[:3]}"]
    for task in sorted(per_task):
        c = per_task[task]
        lines.append(f"  {task:6s} rows {c['rows']:4d} k/row {c['k'] / max(1, c['rows']):.2f} pos {c['pos']:4d} hard {c['hard']:4d} "
                     f"easy {c['easy']:4d} (dropped hard {c['dropped_hard']} easy {c['dropped_easy']}) "
                     f"grid frames {dict(sorted(grid_sizes[task].items()))}")
    report = "\n".join(lines)
    (root / "verify_report_units.txt").write_text(report + "\n")
    print(report)
    if bad_files or bad_hard or bad_easy:
        raise SystemExit("verify_units: problems found (see above)")


# ------------------------------------------------------------------ tar + verify

def stage_tar(root: Path, frame_set: str = "native") -> None:
    for name in frame_dirs(frame_set):
        out = root / f"{name}.tar"
        subprocess.run(["tar", "-cf", str(out), "-C", str(root), name], check=True)
        log(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


def stage_verify(name: str, root: Path, frame_set: str = "native") -> None:
    from PIL import Image

    spec = get_dataset(name)
    spec.frame_set = frame_set
    pool_name, evid_name = frame_dirs(frame_set)
    rows = spec.rows(root)
    lines = [f"{name} @ {root}  rows={len(rows)} videos={len({r['video_id'] for r in rows})}  "
             f"answers={dict(Counter(r['answer'] for r in rows))}"]
    pools = list((root / pool_name).glob("*/times.json"))
    bad_pool = [p.parent.name for p in pools if len(list(p.parent.glob("frame_*.jpg"))) != spec.pool]
    ev_dirs = list((root / evid_name).glob("*/times.json"))
    sizes = Counter(Image.open(p.parent / "frame_0001.jpg").size for p in pools)
    lines.append(f"frame set {frame_set!r}: pool dirs {len(pools)} (incomplete: {bad_pool[:5]}), evidence dirs {len(ev_dirs)}; "
                 f"frame sizes {dict(sizes.most_common(6))}")
    for protocol in spec.protocols:
        for n in N_FRAMES:
            samples = spec.load(root, spec.split_name(protocol, n, "test"))
            rep = spec.last_load
            by_q: Dict[str, List[Any]] = defaultdict(list)
            for s in samples:
                by_q[s.qtype].append(s)
            cells = []
            for q in spec.qtypes:
                ss = by_q.get(q, [])
                if not ss:
                    cells.append(f"{q}:0")
                    continue
                cov = sum(1 for s in ss if s.evidence) / len(ss)
                k = [len(s.evidence) for s in ss if s.evidence is not None]
                cells.append(f"{q}:{len(ss)} cov {cov:.2f} k {sum(k) / max(1, len(k)):.1f}")
            lines.append(f"{protocol:8s} N={n:3d} samples={rep['samples']:5d} skipped={rep['skipped']}")
            lines.append("    " + " | ".join(cells))
            missing = [p for s in samples[:20] for p in s.frame_paths if not p.exists()]
            if missing:
                lines.append(f"    MISSING frame files in the first 20 samples: {missing[:3]}")
    report = "\n".join(lines)
    (root / f"verify_report_{frame_set}.txt").write_text(report + "\n")
    print(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["herbench", "minerva"])
    ap.add_argument("--root", type=Path, default=None, help="default: the dataset's data/ symlink")
    ap.add_argument("--stage", nargs="+", required=True, choices=["rows", "videos", "frames", "tar", "verify", "units", "verify_units"])
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--frame-set", default="native", choices=sorted(LONG_SIDE),
                    help="native = the video's own resolution (what the benchmarks feed the model); 512 = long side 512 px")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--limit", type=int, default=0, help="frames: only this many videos (smoke)")
    ap.add_argument("--max-failed", type=int, default=0, help="frames: undecodable videos tolerated (listed in json/failed_videos.json)")
    ap.add_argument("--keep-parts", action="store_true", help="videos (herbench): keep the tar parts")
    ap.add_argument("--lance-dir", default=None, help="videos (minerva): local train.lance dir (default: hf/lance copy, else hf://)")
    ap.add_argument("--qids-file", type=Path, default=None, help="units: only these rows (one qid per line; default: every row with evidence)")
    ap.add_argument("--seed", type=int, default=0, help="units: seed of the easy negatives")
    args = ap.parse_args()
    spec = get_dataset(args.dataset)
    root = (args.root or Path(spec.default_root)).resolve()
    log(f"{args.dataset} root={root} stages={args.stage}")
    for st in args.stage:
        if st == "rows":
            stage_rows(args.dataset, root)
        elif st == "videos":
            stage_videos_herbench(root, args.keep_parts) if args.dataset == "herbench" else stage_videos_minerva(root, args.lance_dir)
        elif st == "frames":
            stage_frames(root, args.pool, args.workers, args.limit, args.max_failed, args.frame_set)
        elif st == "tar":
            stage_tar(root, args.frame_set)
        elif st == "verify":
            stage_verify(args.dataset, root, args.frame_set)
        elif st == "units":
            stage_units(root, args.qids_file, args.workers, args.limit, args.seed)
        elif st == "verify_units":
            stage_verify_units(args.dataset, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
