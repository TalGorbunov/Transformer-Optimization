"""Real-video MCQ benchmarks as DatasetSpecs (HERBench lite_v2, MINERVA): the shared on-disk
layout, the evidence-timestamp parsers, the two frame protocols that realise a frame count N,
and the row -> Sample composition. docs/SCALEUP_2026-09-23.md §2, §3, §3.3.

On-disk layout, written by experiments/prepare_video.py under data/<dataset> (-> /rg):
    hf/                          raw downloads (annotations; tar parts are deleted after extraction)
    videos/<video_path>          the mp4s (kept)
    json/test.json               rows (schema below); the only split these benchmarks have
    pool/<video_id>/frame_%04d.jpg + times.json      POOL frames per video, t_i = i * dur / POOL
    evidence/<qid>/frame_%04d.jpg + times.json       one frame per evidence unit of the question
    pool.tar, evidence.tar       for sbatch/lib/common.sh:stage_video
Row schema (json/test.json): qid, video_id, question, choices (5 lettered strings), answer (letter),
    answer_index, qtype (short code), task_name, source, duration (annotation), evidence_kind
    ("points" | "ranges" | "mixed" | "all" | "none"), evidence [[t0, t1], ...] seconds, meta (raw).

Protocols (split_name = "<protocol>_N<k>_test"):
    planted_N<k>   the question's evidence frames (one per unit: points at t + 0.3 s, ranges at
                   their midpoint) + N - k fillers from the pool, every filler >= margin from every
                   evidence time (margin = clip(0.5 * dur / N, 1 s, 5 s)), fillers chosen by a uniform
                   stride over the candidates (deterministic), everything sorted by time. Exact
                   labels. Rows with k > N/2, kind "all"/"none", or too few candidates are skipped
                   (counted in spec.last_load). Rows of a video that could not be decoded (no pool;
                   json/failed_videos.json) are skipped under both protocols as "no_pool".
    isolated_N<k>  planted, with the fillers kept >= margin away from every evidence INTERVAL, not only
                   from the planted frame's time. For point evidence the two protocols are the same
                   frames; for interval evidence (a person visible for 50 s, one frame planted at the
                   midpoint) `planted` lets fillers fall inside the interval — frames that show the
                   evidence but are labelled 0 (found 2026-09-29 on the HERBench person tasks). Use
                   `isolated` for every labelled-frame measurement on interval evidence.
    uniform_N<k>   the benchmarks' own sampling: pool frames i = j * POOL / N (the official
                   int(i * step) grid, nested across N). Labels: a frame is evidence iff it lies
                   inside a range (± delta) or within ± delta of a point, delta = min(2 s, 0.5 * dur / N).
                   Coverage (>= 1 evidence frame) is reported; "all" rows label every frame.
    evidence_<unit> the UNIT baseline (plan 2026-10-06): ONLY the question's evidence units, no
                   fillers, in temporal order; split_name("evidence", <unit>, "test"). A unit is
                   the stored 9-frame grid of one evidence interval (units_native/<qid>/, written by
                   experiments/prepare_video.py --stage units) read at one of UNIT_OFFSETS:
                   frame (the planted frame alone) | clip3_d1 | clip5_d1 | clip5_d2 | clip9_d2.
                   Frames per prompt are capped at MAX_UNIT_FRAMES (k * m): a row that exceeds it
                   falls back along UNIT_FALLBACK and records unit_eff / m_eff. Every frame is
                   evidence. units() gives the per-unit records (pos / hard / easy) for the
                   per-unit detection regime.
Pinned by tests/test_data_videoqa.py.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from .base import DatasetSpec, Sample, evidence_strata
from .mcq import (
    has_letter, herbench_extract, herbench_prompt, letter_match, lettered, minerva_extract, minerva_prompt,
)

Interval = Tuple[float, float]
POOL = 256
N_FRAMES = (8, 16, 32, 64, 128, 256)


def frame_dirs(frame_set: str) -> Tuple[str, str]:
    """(pool dir, evidence dir) of a stored frame set under the dataset root. "512" is the first
    extraction (long side 512 px, dirs `pool` / `evidence`); every other set is `pool_<set>` /
    `evidence_<set>`, e.g. `pool_native`."""
    return ("pool", "evidence") if frame_set == "512" else (f"pool_{frame_set}", f"evidence_{frame_set}")
POINT_OFFSET = 0.3          # legacy prep_ac_frames.py --evidence-offset: mid-action
MARGIN_MIN, MARGIN_MAX = 1.0, 5.0
LABEL_DELTA_MAX = 2.0

# ---- evidence units (the UNIT baseline, plan 2026-10-06): one stored 9-frame grid per evidence interval
UNIT_STEP = 0.5                 # s between the stored frames of a point unit (and of every negative)
UNIT_HALF = 4                   # grid = centre + j * step, j in [-UNIT_HALF, UNIT_HALF] -> 9 frames
UNIT_OFFSETS: Dict[str, Tuple[int, ...]] = {
    "frame": (0,),                          # today's unit: the one planted frame
    "clip3_d1": (-2, 0, 2),                 # 3 frames, +-1 s on a point (quarter points of a range)
    "clip5_d1": (-2, -1, 0, 1, 2),          # 5 frames, +-1 s, 0.5 s step
    "clip5_d2": (-4, -2, 0, 2, 4),          # 5 frames, +-2 s, 1 s step
    "clip9_d2": tuple(range(-UNIT_HALF, UNIT_HALF + 1)),   # the whole stored grid
}
UNIT_FALLBACK: Dict[str, Optional[str]] = {"clip9_d2": "clip5_d2", "clip5_d2": "clip3_d1", "clip5_d1": "clip3_d1",
                                           "clip3_d1": "frame", "frame": None}
MAX_UNIT_FRAMES = 48            # frames per evidence-only prompt (k * m); above it the row falls back along UNIT_FALLBACK.
                                # 48 keeps clip9_d2 on the k = 5 HD-EPIC rows (45 frames ~ 55 K tokens at 1 MP, one unfenced
                                # forward); AC rows with k = 8 fall back to clip5_d2 (40).
HARD_GAP = 2.0                  # s between the padded evidence interval and the hard negative's grid
EASY_MIN_DIST = 5.0             # s: an easy negative's grid stays this far from every evidence interval
NEG_MARGIN = 1.0                # s: evidence intervals are padded by this before a negative is tested against them
RANGE_TOL = 0.1                 # s: a range unit's grid may overhang [a, b] by this (annotations are 0.1 s coarse)
UNITS_DIR = "units_native"      # <root>/units_native/<qid>/<uid>/frame_%02d.jpg + units.json


# ------------------------------------------------------------------ timestamps

def parse_hms(s: str) -> float:
    """'M:SS:mmm' (HD-EPIC, colon before the milliseconds) | 'MM:SS.s' | 'HH:MM:SS.mmm' | 'MM:SS'
    | 'H:MM:SS' -> seconds."""
    parts = str(s).strip().split(":")
    if len(parts) == 3 and len(parts[2]) == 3 and "." not in parts[2] and parts[2].isdigit():
        m, sec, ms = parts
        return int(m) * 60 + int(sec) + int(ms) / 1000.0
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    raise ValueError(f"unparseable timestamp {s!r}")


def parse_range(s: str) -> Interval:
    """'MM:SS-MM:SS' | 'MM:SS.s-MM:SS.s' | 'HH:MM:SS.mmm - HH:MM:SS.mmm' -> (t0, t1), ordered."""
    a, b = re.split(r"\s*-\s*", str(s).strip(), maxsplit=1)
    t0, t1 = parse_hms(a), parse_hms(b)
    return (min(t0, t1), max(t0, t1))


# HERBench task codes (the paper's) <-> the annotation's task_type strings.
HERBENCH_TASKS: Dict[str, str] = {
    "TSO": "Temporal Shot Ordering",
    "MPDR": "Multi-Person Duration Reasoning",
    "ASII": "Action Sequence Integrity Identification",
    "AGBI": "Appearance-Grounded Behavior & Interactions",
    "AGAR": "Appearance-Grounded Attribute Recognition",
    "AGLT": "Appearance-Grounded Localization & Trajectory",
    "FAM": "False Action Memory",
    "SVA": "Scene Verification & Arrangement",
    "FOM": "False Object Memory",
    "MEGL": "Multi-Entities Grounding and Localization",
    "AC": "Action Counting",
    "RLPC": "Region Localized People Counting",
}
HERBENCH_CODE = {v: k for k, v in HERBENCH_TASKS.items()}


def parse_herbench_evidence(code: str, meta: Dict[str, Any]) -> Tuple[str, List[Interval]]:
    """(kind, intervals) from a lite_v2 row's metadata (schema verified 2026-09-23):
    AC/ASII/FAM/FOM  required_timestamps: ['M:SS:mmm', ...]           -> points
    AGAR/AGBI/AGLT   required_timestamps: 'MM:SS-MM:SS'               -> one range
    MEGL/MPDR        required_timestamps: ['HH:MM:SS.mmm - HH:MM:SS.mmm', ...] -> ranges
    SVA/TSO          timestamps: ['MM:SS.s-MM:SS.s', ...]             -> ranges (SVA: distorted shots included)
    RLPC             required_timestamps: 'All'                       -> all"""
    if code in ("AC", "ASII", "FAM", "FOM"):
        ts = meta["required_timestamps"]
        return "points", [(parse_hms(t), parse_hms(t)) for t in ts]
    if code in ("AGAR", "AGBI", "AGLT"):
        return "ranges", [parse_range(meta["required_timestamps"])]
    if code in ("MEGL", "MPDR"):
        return "ranges", [parse_range(t) for t in meta["required_timestamps"]]
    if code in ("SVA", "TSO"):
        return "ranges", [parse_range(t) for t in meta["timestamps"]]
    if code == "RLPC":
        assert str(meta["required_timestamps"]).lower() == "all", meta
        return "all", []
    raise ValueError(f"unknown HERBench task code {code!r}")


_TS = r"(?<![\d:])(?:\d{1,2}:)?\d{1,2}:\d{2}(?![\d:])"
_RANGE = re.compile(rf"(?:from|between)?\s*({_TS})\s*(?:-|–|—|to|and|through)\s*({_TS})", re.IGNORECASE)
_POINT = re.compile(_TS)


def parse_minerva_evidence(reasoning: str, duration: Optional[float] = None) -> Tuple[str, List[Interval]]:
    """Timestamps inside a MINERVA reasoning trace: 'from 00:04 to 00:08' / '00:09-00:33' ->
    ranges, lone 'at 01:08' -> points; ordered by time, de-duplicated; > duration + 1 s dropped."""
    text = str(reasoning)
    ivs: List[Interval] = []
    spans: List[Tuple[int, int]] = []
    for m in _RANGE.finditer(text):
        a, b = parse_hms(m.group(1)), parse_hms(m.group(2))
        if b < a:                       # "x and y" that are not a range
            ivs += [(a, a), (b, b)]
        else:
            ivs.append((a, b))
        spans.append(m.span())
    for m in _POINT.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        t = parse_hms(m.group(0))
        ivs.append((t, t))
    if duration is not None:
        ivs = [iv for iv in ivs if iv[0] <= duration + 1.0]
    ivs = sorted(set(ivs))
    if not ivs:
        return "none", []
    kinds = {"points" if a == b else "ranges" for a, b in ivs}
    return ("mixed" if len(kinds) == 2 else kinds.pop()), ivs


# ------------------------------------------------------------------ frame protocols

def pool_times(duration: float, n: int = POOL) -> List[float]:
    """The benchmarks' own grid: t_i = i * dur / n (uniform.py: int(i * step)); nested across N."""
    return [i * float(duration) / n for i in range(n)]


def uniform_indices(n_frames: int, pool: int = POOL) -> List[int]:
    if pool % n_frames:
        raise ValueError(f"N={n_frames} must divide the pool size {pool}")
    step = pool // n_frames
    return [j * step for j in range(n_frames)]


def evidence_times(intervals: Sequence[Interval], duration: float) -> List[float]:
    """One frame time per evidence unit: points at t + 0.3 s, ranges at their midpoint; clamped."""
    out = []
    for a, b in intervals:
        t = a + POINT_OFFSET if a == b else 0.5 * (a + b)
        out.append(min(max(t, 0.0), max(0.0, float(duration) - 0.05)))
    return out


def uniform_labels(times: Sequence[float], intervals: Sequence[Interval], n_frames: int, duration: float
                   ) -> FrozenSet[int]:
    delta = min(LABEL_DELTA_MAX, 0.5 * float(duration) / n_frames)
    ev = set()
    for i, t in enumerate(times):
        for a, b in intervals:
            if a - delta <= t <= b + delta:
                ev.add(i)
                break
    return frozenset(ev)


def planted_margin(duration: float, n_frames: int) -> float:
    return min(MARGIN_MAX, max(MARGIN_MIN, 0.5 * float(duration) / n_frames))


def plant(pool_t: Sequence[float], evid_t: Sequence[float], n_frames: int, margin: float,
          intervals: Sequence[Interval] = ()) -> Optional[Tuple[List[int], List[Tuple[float, str, int]]]]:
    """Choose N - k pool fillers >= margin from every evidence time (and, when `intervals` is
    given, outside [a - margin, b + margin] of every evidence interval) by a uniform stride over
    the candidates; return (chosen pool indices, merged [(time, 'pool'|'evid', idx)] sorted by
    time), or None when the row is not plantable at this N."""
    k = len(evid_t)
    if k == 0 or k > n_frames // 2:
        return None
    cand = [i for i, t in enumerate(pool_t) if all(abs(t - e) >= margin for e in evid_t)
            and not any(a - margin < t < b + margin for a, b in intervals)]
    need = n_frames - k
    if len(cand) < need:
        return None
    chosen = [cand[int((j + 0.5) * len(cand) / need)] for j in range(need)]
    assert len(set(chosen)) == need, "stride picked a duplicate"
    merged = sorted([(pool_t[i], "pool", i) for i in chosen] + [(t, "evid", j) for j, t in enumerate(evid_t)])
    return chosen, merged


# ------------------------------------------------------------------ evidence units

def unit_grid(interval: Interval, duration: float) -> Tuple[float, float, List[Tuple[int, float]]]:
    """The stored grid of one evidence unit: (centre, step, [(offset j, time)]). A point t gets
    centre t + 0.3 s (the planted frame's own time), step 0.5 s, j in -4..4 (9 frames), clamped to
    the video. A range [a, b] gets centre = midpoint and step = max(0.5 s, (b - a) / 8) so j = +-4
    are the ends; offsets whose time leaves [a - 0.1, b + 0.1] are dropped (a 1 s shot keeps
    j = -1, 0, 1). j = 0 is always present, so unit "frame" is exactly the planted frame."""
    a, b = float(interval[0]), float(interval[1])
    end = max(0.0, float(duration) - 0.05)
    if a == b:
        centre = min(max(a + POINT_OFFSET, 0.0), end)
        step = UNIT_STEP
        lo, hi = 0.0, end
    else:
        centre = min(max(0.5 * (a + b), 0.0), end)
        step = max(UNIT_STEP, (b - a) / (2 * UNIT_HALF))
        lo, hi = max(0.0, a - RANGE_TOL), min(end, b + RANGE_TOL)
    grid = [(j, centre + j * step) for j in range(-UNIT_HALF, UNIT_HALF + 1)
            if lo - 1e-9 <= centre + j * step <= hi + 1e-9]
    assert any(j == 0 for j, _ in grid), (interval, duration)
    return centre, step, grid


def _clear(centre: float, intervals: Sequence[Interval], duration: float, dist: float) -> bool:
    """A point-style grid centred here (+- UNIT_HALF * UNIT_STEP) lies inside the video and at least
    `dist` from every evidence interval."""
    half = UNIT_HALF * UNIT_STEP
    if centre - half < 0.0 or centre + half > float(duration):
        return False
    return all(centre + half <= a - dist or centre - half >= b + dist for a, b in intervals)


def negative_centres(unit_iv: Interval, intervals: Sequence[Interval], duration: float, seed: str
                     ) -> Dict[str, Optional[float]]:
    """Centres of the two negative grids of one evidence unit (point-style grids, 0.5 s step,
    +-2 s). `hard`: the grid starts HARD_GAP after the unit's interval (centre = b + 2 + HARD_GAP),
    or ends HARD_GAP before it; None when neither side fits in the video and clear of every
    evidence interval of the question padded by NEG_MARGIN. `easy`: a seeded uniform centre whose
    grid stays >= EASY_MIN_DIST from every interval; None when 500 draws find none (an interval
    that covers the clip)."""
    a, b = float(unit_iv[0]), float(unit_iv[1])
    half = UNIT_HALF * UNIT_STEP
    out: Dict[str, Optional[float]] = {"hard": None, "easy": None}
    for c in (b + half + HARD_GAP, a - half - HARD_GAP):
        if _clear(c, intervals, duration, NEG_MARGIN):
            out["hard"] = c
            break
    rng = random.Random(seed)
    lo, hi = half, float(duration) - half
    if hi > lo:
        for _ in range(500):
            c = rng.uniform(lo, hi)
            if _clear(c, intervals, duration, EASY_MIN_DIST):
                out["easy"] = round(c, 3)
                break
    return out


def plan_units(qid: str, intervals: Sequence[Interval], duration: float, seed: int = 0) -> Dict[str, Any]:
    """What experiments/prepare_video.py --stage units decodes for one question (pure; no I/O):
    per evidence unit (sorted by time) its stored grid and, when they exist, its hard and easy
    negative grids. The schema of units_native/<qid>/units.json."""
    ivs = sorted((float(a), float(b)) for a, b in intervals)
    units: List[Dict[str, Any]] = []
    dropped = {"hard": 0, "easy": 0}
    for i, iv in enumerate(ivs):
        centre, step, grid = unit_grid(iv, duration)
        units.append({"uid": f"u{i:02d}_pos", "unit": i, "kind": "pos", "interval": list(iv), "centre": centre,
                      "step": step, "offsets": [j for j, _ in grid], "times": [round(t, 3) for _, t in grid]})
        for kind, c in negative_centres(iv, ivs, duration, f"{seed}:{qid}:{i}").items():
            if c is None:
                dropped[kind] += 1
                continue
            units.append({"uid": f"u{i:02d}_{kind}", "unit": i, "kind": kind, "interval": None, "centre": c,
                          "step": UNIT_STEP, "offsets": list(range(-UNIT_HALF, UNIT_HALF + 1)),
                          "times": [round(c + j * UNIT_STEP, 3) for j in range(-UNIT_HALF, UNIT_HALF + 1)]})
    return {"qid": qid, "duration": float(duration), "k": len(ivs), "units": units, "dropped": dropped}


def select_offsets(entry: Dict[str, Any], unit: str) -> List[Tuple[int, float, Path]]:
    """The frames of one stored unit read at `unit`: [(offset, time, relative path)] in time order."""
    want = set(UNIT_OFFSETS[unit])
    return [(j, t, Path(entry["uid"]) / f"frame_{i + 1:02d}.jpg")
            for i, (j, t) in enumerate(zip(entry["offsets"], entry["times"])) if j in want]


# ------------------------------------------------------------------ the spec

class VideoMCQ(DatasetSpec):
    """Shared implementation; subclasses fix name/root/qtypes/prompt/parser."""
    protocols = ("uniform", "isolated", "planted", "evidence")   # uniform = the benchmarks' own sampling = the PRIMARY protocol
    max_unit_frames = MAX_UNIT_FRAMES
    has_train = False
    supports_counterfactual = False
    system_prompt = None            # the official wrappers send no system turn
    pool = POOL
    task_names: Dict[str, str] = {}
    frame_set = "native"            # which stored frames to read: "native" (the video's own resolution, what the
                                    # benchmarks' evaluation code feeds the model) | "512" (long side 512, the
                                    # 2026-09-28 wave; kept so those run dirs stay reproducible)

    def __init__(self) -> None:
        self.last_load: Dict[str, Any] = {}

    def frame_dirs(self) -> Tuple[str, str]:
        return frame_dirs(self.frame_set)

    def split_name(self, protocol: str, n_frames: Union[int, str], split: str) -> str:
        """`<protocol>_N<k>_<split>`; for the evidence protocol the second argument is the UNIT
        name (frame | clip3_d1 | ...): `evidence_<unit>_<split>`."""
        if protocol not in self.protocols:
            raise ValueError(f"{self.name}: protocol must be one of {self.protocols}, got {protocol!r}")
        if split != "test":
            raise ValueError(f"{self.name} has only a test split")
        if protocol == "evidence":
            if str(n_frames) not in UNIT_OFFSETS:
                raise ValueError(f"{self.name}: evidence unit must be one of {sorted(UNIT_OFFSETS)}, got {n_frames!r}")
            return f"evidence_{n_frames}_{split}"
        if int(n_frames) not in N_FRAMES:
            raise ValueError(f"{self.name}: N must be one of {N_FRAMES}, got {n_frames}")
        return f"{protocol}_N{int(n_frames)}_{split}"

    @staticmethod
    def parse_split_name(split_name: str) -> Tuple[str, Union[int, str], str]:
        m = re.fullmatch(r"(planted|isolated|uniform)_N(\d+)_(\w+)", split_name)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
        m = re.fullmatch(r"evidence_(frame|clip\d+_d\d+)_([a-z]+)", split_name)
        if m and m.group(1) in UNIT_OFFSETS:
            return "evidence", m.group(1), m.group(2)
        raise ValueError(f"bad video split name {split_name!r} (want <protocol>_N<k>_test or evidence_<unit>_test)")

    # ---- rows -> samples
    def rows(self, root: Path, split: str = "test") -> List[Dict[str, Any]]:
        return json.loads((Path(root) / "json" / f"{split}.json").read_text(encoding="utf-8"))

    def load(self, root: Path, split_name: str, qtypes: Optional[Sequence[str]] = None) -> List[Sample]:
        protocol, n_frames, split = self.parse_split_name(split_name)
        root = Path(root)
        rows = self.rows(root, split)
        if qtypes is not None:
            unknown = set(qtypes) - set(self.qtypes)
            if unknown:
                raise ValueError(f"unknown qtypes: {sorted(unknown)}")
            rows = [r for r in rows if r["qtype"] in set(qtypes)]
        out: List[Sample] = []
        skipped: Dict[str, int] = {}
        pools: Dict[str, Dict[str, Any]] = {}
        pool_name, _ = self.frame_dirs()
        if not (root / pool_name).is_dir():
            raise FileNotFoundError(f"{root / pool_name} missing: frame set {self.frame_set!r} was not extracted "
                                    f"(experiments/prepare_video.py --stage frames --frame-set {self.frame_set})")
        for r in rows:
            if protocol == "evidence":
                s, why = self.compose_evidence(r, root, str(n_frames))
            else:
                vid = r["video_id"]
                if vid not in pools:
                    tj = root / pool_name / vid / "times.json"
                    pools[vid] = json.loads(tj.read_text()) if tj.exists() else None
                if pools[vid] is None:                       # undecodable video (json/failed_videos.json)
                    skipped["no_pool"] = skipped.get("no_pool", 0) + 1
                    continue
                s, why = self.compose(r, root, pools[vid], protocol, int(n_frames))
            if s is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            out.append(s)
        self.last_load = {"split": split_name, "rows": len(rows), "samples": len(out), "skipped": skipped}
        return out

    # ---- evidence units (the UNIT baseline)
    @staticmethod
    def units_info(root: Path, qid: str) -> Optional[Dict[str, Any]]:
        p = Path(root) / UNITS_DIR / qid / "units.json"
        return json.loads(p.read_text()) if p.exists() else None

    def compose_evidence(self, row: Dict[str, Any], root: Path, unit: str) -> Tuple[Optional[Sample], str]:
        """The question's evidence units only, read at `unit`, every frame in time order. Falls back
        along UNIT_FALLBACK until k * m <= max_unit_frames; records unit_eff, k, m_eff, unit_of_frame."""
        if row["evidence_kind"] in ("all", "none"):
            return None, f"kind={row['evidence_kind']}"
        info = self.units_info(root, row["qid"])
        if info is None:
            return None, "no_units"
        pos = [u for u in info["units"] if u["kind"] == "pos"]
        if not pos:
            return None, "no_units"
        chosen: Optional[str] = unit
        while chosen is not None:
            sel = [select_offsets(u, chosen) for u in pos]
            total = sum(len(s) for s in sel)
            if total <= self.max_unit_frames:
                break
            chosen = UNIT_FALLBACK[chosen]
        if chosen is None:
            return None, "too_many_frames"
        udir = root / UNITS_DIR / row["qid"]
        frames = sorted((t, i, udir / rel) for i, s in enumerate(sel) for _, t, rel in s)
        paths = tuple(p for _, _, p in frames)
        extra = {"protocol": "evidence", "unit": unit, "unit_eff": chosen, "k": len(pos),
                 "m_eff": len(paths) / len(pos), "unit_of_frame": [i for _, i, _ in frames],
                 "times": [t for t, _, _ in frames], "n_frames": len(paths)}
        return self._sample(row, paths, frozenset(range(len(paths))), extra), ""

    def units(self, root: Path, sample: Sample, unit: str = "frame") -> List[Dict[str, Any]]:
        """Per-unit records for the per-unit detection regime: every stored unit of the question
        (pos = an evidence unit, hard / easy = its negatives), each read at `unit` (a negative's
        grid is point-style, so it yields the same number of frames as a point unit)."""
        if unit not in UNIT_OFFSETS:
            raise ValueError(f"unit must be one of {sorted(UNIT_OFFSETS)}, got {unit!r}")
        info = self.units_info(root, sample.qid)
        if info is None:
            return []
        udir = Path(root) / UNITS_DIR / sample.qid
        out = []
        for u in info["units"]:
            sel = select_offsets(u, unit)
            out.append({"uid": u["uid"], "unit": u["unit"], "kind": u["kind"], "centre": u["centre"],
                        "interval": u.get("interval"), "m": len(sel), "k": info["k"],
                        "frame_paths": tuple(udir / rel for _, _, rel in sel), "times": [t for _, t, _ in sel]})
        return out

    def compose(self, row: Dict[str, Any], root: Path, pool_info: Dict[str, Any], protocol: str, n_frames: int
                ) -> Tuple[Optional[Sample], str]:
        pool_t: List[float] = pool_info["times"]
        duration = float(pool_info["duration"])
        intervals: List[Interval] = [tuple(iv) for iv in row["evidence"]]
        kind = row["evidence_kind"]
        pool_name, evid_name = self.frame_dirs()
        pdir = root / pool_name / row["video_id"]
        if protocol == "uniform":
            idx = uniform_indices(n_frames, len(pool_t))
            times = [pool_t[i] for i in idx]
            paths = tuple(pdir / f"frame_{i + 1:04d}.jpg" for i in idx)
            if kind == "all":
                ev: Optional[FrozenSet[int]] = frozenset(range(n_frames))
            elif kind == "none":
                ev = None
            else:
                ev = uniform_labels(times, intervals, n_frames, duration)
            return self._sample(row, paths, ev, {"protocol": protocol, "n_frames": n_frames, "times": times}), ""
        # planted
        if kind in ("all", "none"):
            return None, f"kind={kind}"
        edir = root / evid_name / row["qid"]
        einfo = json.loads((edir / "times.json").read_text())
        evid_t: List[float] = einfo["times"]
        res = plant(pool_t, evid_t, n_frames, planted_margin(duration, n_frames),
                    intervals=intervals if protocol == "isolated" else ())
        if res is None:
            return None, ("k>N/2" if len(evid_t) > n_frames // 2 else "too few fillers")
        _, merged = res
        paths = tuple((pdir / f"frame_{i + 1:04d}.jpg") if src == "pool" else (edir / f"frame_{i + 1:04d}.jpg")
                      for _, src, i in merged)
        ev = frozenset(k for k, (_, src, _) in enumerate(merged) if src == "evid")
        return self._sample(row, paths, ev, {"protocol": protocol, "n_frames": n_frames,
                                             "times": [t for t, _, _ in merged]}), ""

    def _sample(self, row: Dict[str, Any], paths: Tuple[Path, ...], ev: Optional[FrozenSet[int]],
                extra: Dict[str, Any]) -> Sample:
        return Sample(qid=str(row["qid"]), question=row["question"], answer=str(row["answer"]), qtype=row["qtype"],
                      frame_paths=paths, evidence=ev, group=str(row["video_id"]), meta={**row, **extra})

    # ---- protocol
    def answer_target(self, sample: Sample) -> str:
        return sample.answer

    def match(self, pred: Optional[str], sample: Sample) -> bool:
        return letter_match(pred, sample.answer)

    def stop_decoding(self, text: str) -> bool:
        return has_letter(text)

    def report_extras(self, samples: Sequence[Sample], preds: Sequence[Optional[str]]
                      ) -> Dict[str, Tuple[float, int]]:
        n = len(samples)
        if not n:
            return {}
        cov = sum(1 for s in samples if s.evidence)
        k = [len(s.evidence) for s in samples if s.evidence is not None]
        letters: Dict[str, int] = {}
        for s in samples:
            letters[s.answer] = letters.get(s.answer, 0) + 1
        top = max(letters.values())
        out = {"coverage(>=1 evidence frame)": (cov / n, n),
               "mean_k_evidence": (sum(k) / len(k) if k else float("nan"), len(k)),
               "letter_prior_baseline": (top / n, n)}
        out.update(evidence_strata(samples, preds, self.match))
        return out


class HERBench(VideoMCQ):
    name = "herbench"
    default_root = "data/herbench_v2"
    qtypes = tuple(HERBENCH_TASKS)
    numeric_qtypes = ()
    task_names = HERBENCH_TASKS

    def question_text(self, sample: Sample) -> str:
        return herbench_prompt(sample.question, lettered(sample.meta["choices"]))

    def bare_question(self, sample: Sample) -> str:
        return sample.question + "\n" + "\n".join(lettered(sample.meta["choices"]))

    def parse(self, raw: str) -> Optional[str]:
        return herbench_extract(raw)


MINERVA_SKILLS: Dict[str, str] = {
    "counting": "Counting", "object_recognition": "Object Recognition", "event_occurrence": "Event Occurence",
    "temporal_reasoning": "Temporal Reasoning", "reading": "Reading", "spatial_perception": "Spatial Perception",
    "numerical_reasoning": "Numerical Reasoning", "cause_and_effect": "Cause and Effect",
    "situational_awareness": "Situational Awareness", "counterfactual": "Counterfactual",
    "state_changes": "State Changes", "goal_reasoning": "Goal Reasoning",
}   # "Listening" (needs the ASR transcript) is dropped at prep time
MINERVA_CODE = {v: k for k, v in MINERVA_SKILLS.items()}


class Minerva(VideoMCQ):
    name = "minerva"
    default_root = "data/minerva"
    qtypes = tuple(MINERVA_SKILLS)
    numeric_qtypes = ()
    task_names = MINERVA_SKILLS

    def question_text(self, sample: Sample) -> str:
        bare = [re.sub(r"^[A-E]\.\s*", "", c, count=1) for c in sample.meta["choices"]]
        return minerva_prompt(sample.question, bare)

    def bare_question(self, sample: Sample) -> str:
        return sample.question + "\n" + "\n".join(lettered(sample.meta["choices"]))

    def parse(self, raw: str) -> Optional[str]:
        return minerva_extract(raw)
