"""CPU tests for the video-benchmark seam (core/data/videoqa.py, core/data/mcq.py): timestamp
parsers on every HERBench format and on MINERVA traces, the nested pool grid, planted / uniform
composition and labels, byte-equality of the prompts with the benchmarks' own formatters, the
letter extractors, and VideoMCQ.load on a synthetic tree.

Run: python tests/test_data_videoqa.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.data import DATASETS, get_dataset
from core.data.mcq import (
    HERBENCH_SUFFIX, MINERVA_SUFFIX, has_letter, herbench_extract, herbench_prompt, lettered, minerva_extract,
    minerva_prompt,
)
from core.data.videoqa import (
    EASY_MIN_DIST, HARD_GAP, HERBENCH_TASKS, MAX_UNIT_FRAMES, MINERVA_SKILLS, N_FRAMES, NEG_MARGIN, POOL, UNIT_FALLBACK,
    UNIT_HALF, UNIT_OFFSETS, UNIT_STEP, UNITS_DIR, HERBench, Minerva, evidence_times, frame_dirs, negative_centres,
    parse_herbench_evidence, parse_hms, parse_minerva_evidence, parse_range, plan_units, plant, planted_margin, pool_times,
    select_offsets, uniform_indices, uniform_labels, unit_grid,
)

# metadata examples copied from the real lite_v2 rows (2026-09-23)
META = {
    "AC": {"required_timestamps": ["1:29:590", "2:30:510"], "pair": ["pick up", "ginger"], "true_count": 2},
    "ASII": {"required_timestamps": ["0:13:870", "0:17:440", "0:21:310", "0:22:610", "0:24:610"]},
    "FAM": {"required_timestamps": ["0:10:000", "0:12:700", "0:14:880", "0:19:880"]},
    "FOM": {"required_timestamps": ["0:10:000", "0:11:170", "0:14:880", "0:19:880"]},
    "AGAR": {"required_timestamps": "00:00-00:04"},
    "AGBI": {"required_timestamps": "00:05-00:08"},
    "AGLT": {"required_timestamps": "00:07-00:59"},
    "MEGL": {"required_timestamps": ["00:00:43.110 - 00:00:59.960", "00:00:01.368 - 00:00:07.941"]},
    "MPDR": {"required_timestamps": ["00:00:05.506 - 00:00:11.378", "00:00:27.127 - 00:00:30.197"]},
    "SVA": {"timestamps": ["01:19.5-01:20.4", "00:52.7-00:53.5"], "Real/Distorted": ["Real", "Distorted"]},
    "TSO": {"timestamps": ["00:42.8-00:43.8", "01:05.3-01:06.2"]},
    "RLPC": {"required_timestamps": "All"},
}


def test_registry():
    assert {"herbench", "minerva"} <= set(DATASETS)
    h, m = get_dataset("herbench"), get_dataset("minerva")
    assert isinstance(h, HERBench) and isinstance(m, Minerva)
    assert h.protocols == ("uniform", "isolated", "planted", "evidence") and h.split_name("isolated", 32, "test") == "isolated_N32_test"
    assert h.split_name("evidence", "clip5_d1", "test") == "evidence_clip5_d1_test" and h.parse_split_name("evidence_frame_test") == ("evidence", "frame", "test")
    for bad in (("evidence", 8, "test"), ("evidence", "clip7", "test")):
        try:
            h.split_name(*bad)
            assert False, bad
        except ValueError:
            pass
    assert h.frame_set == "native" and h.frame_dirs() == ("pool_native", "evidence_native") and frame_dirs("512") == ("pool", "evidence")
    assert h.split_name("uniform", 256, "test") == "uniform_N256_test" and not h.has_train and not h.supports_counterfactual
    assert h.system_prompt is None and set(h.qtypes) == set(HERBENCH_TASKS) and len(h.qtypes) == 12
    assert set(m.qtypes) == set(MINERVA_SKILLS) and "listening" not in m.qtypes
    assert h.split_name("planted", 8, "test") == "planted_N8_test"
    assert h.parse_split_name("uniform_N128_test") == ("uniform", 128, "test")
    for bad in (("mmred", 8, "test"), ("planted", 7, "test"), ("planted", 8, "train")):
        try:
            h.split_name(*bad)
            assert False, bad
        except ValueError:
            pass


def test_parse_hms_and_range():
    assert parse_hms("1:29:590") == 89.59 and parse_hms("0:10:000") == 10.0     # HD-EPIC M:SS:mmm
    assert parse_hms("00:00:43.110") == 43.11                                  # HH:MM:SS.mmm
    assert parse_hms("01:19.5") == 79.5 and parse_hms("00:07") == 7.0           # MM:SS.s / MM:SS
    assert parse_hms("1:02:10") == 3730.0                                       # H:MM:SS (2-digit seconds)
    assert parse_range("00:07-00:59") == (7.0, 59.0)
    assert parse_range("00:00:43.110 - 00:00:59.960") == (43.11, 59.96)
    assert parse_range("01:20.4-01:19.5") == (79.5, 80.4), "ordered"


def test_parse_herbench_evidence_every_task():
    for code, meta in META.items():
        kind, ivs = parse_herbench_evidence(code, meta)
        if code in ("AC", "ASII", "FAM", "FOM"):
            assert kind == "points" and all(a == b for a, b in ivs) and len(ivs) == len(meta["required_timestamps"])
        elif code == "RLPC":
            assert kind == "all" and ivs == []
        else:
            assert kind == "ranges" and all(a < b for a, b in ivs)
    assert parse_herbench_evidence("AC", META["AC"])[1] == [(89.59, 89.59), (150.51, 150.51)]
    assert parse_herbench_evidence("AGLT", META["AGLT"])[1] == [(7.0, 59.0)]
    assert parse_herbench_evidence("SVA", META["SVA"])[1] == [(79.5, 80.4), (52.7, 53.5)]
    assert set(HERBENCH_TASKS) == set(META)


def test_parse_minerva_evidence():
    k, iv = parse_minerva_evidence("I found 1 at 00:07, another at 00:20, and another at 00:33.")
    assert k == "points" and iv == [(7.0, 7.0), (20.0, 20.0), (33.0, 33.0)]
    k, iv = parse_minerva_evidence("from 00:04 to 00:08 the player runs; 00:09-00:33 shows the goal")
    assert k == "ranges" and iv == [(4.0, 8.0), (9.0, 33.0)]
    k, iv = parse_minerva_evidence("between 1:02:10 and 1:02:40 nothing happens, then at 1:03:05 it does")
    assert k == "mixed" and iv == [(3730.0, 3760.0), (3785.0, 3785.0)]
    k, iv = parse_minerva_evidence("at 00:20 and 00:07 (not a range: descending)")
    assert k == "points" and iv == [(7.0, 7.0), (20.0, 20.0)]
    assert parse_minerva_evidence("no timestamps here, score 24 to 13") == ("none", [])
    k, iv = parse_minerva_evidence("at 00:07 and at 00:07 again, then 59:59", duration=100.0)
    assert iv == [(7.0, 7.0)], "de-duplicated and beyond-duration dropped"
    assert parse_minerva_evidence("ratio 3:1 and 12:30:00 PM")[1] == [(45000.0, 45000.0)], "3:1 is not a timestamp; 12:30:00 reads as H:MM:SS"
    assert parse_minerva_evidence("ratio 3:1 and 12:30:00 PM", duration=600.0) == ("none", []), "a clock time is dropped by the duration filter"


def test_pool_grid_nested():
    dur = 123.4
    pt = pool_times(dur, POOL)
    assert len(pt) == POOL and pt[0] == 0.0 and pt[-1] < dur
    for n in N_FRAMES:
        idx = uniform_indices(n, POOL)
        assert len(idx) == n and idx[0] == 0 and all(b - a == POOL // n for a, b in zip(idx, idx[1:]))
        assert all(math.isclose(pt[i], j * dur / n, rel_tol=1e-9) for j, i in enumerate(idx)), "the official grid, nested"
    try:
        uniform_indices(7, POOL)
        assert False
    except ValueError:
        pass


def test_evidence_times_and_uniform_labels():
    assert evidence_times([(10.0, 10.0), (20.0, 30.0)], 100.0) == [10.3, 25.0]
    assert evidence_times([(99.99, 99.99)], 100.0) == [99.95], "clamped inside the video"
    times = pool_times(64.0, 64)                                   # 1 s apart
    ev = uniform_labels(times, [(10.0, 10.0), (40.0, 42.0)], 64, 64.0)
    assert ev == frozenset({10, 40, 41, 42}), ev                   # delta = min(2, 0.5) = 0.5 s
    ev8 = uniform_labels(pool_times(64.0, 8), [(10.0, 10.0)], 8, 64.0)
    assert ev8 == frozenset({1}), "8 frames 8 s apart: t=8 is within delta=2 s of the point at 10"


def test_plant():
    dur = 60.0
    pool_t = pool_times(dur, POOL)
    evid = [10.3, 25.0, 40.3]
    for n in (8, 16, 32, 64, 128):
        res = plant(pool_t, evid, n, planted_margin(dur, n))
        assert res is not None, n
        chosen, merged = res
        assert len(merged) == n and len(chosen) == n - len(evid)
        assert [t for t, _, _ in merged] == sorted(t for t, _, _ in merged), "temporal order"
        assert all(abs(pool_t[i] - e) >= planted_margin(dur, n) for i in chosen for e in evid), "fillers keep the margin"
        assert res == plant(pool_t, evid, n, planted_margin(dur, n)), "deterministic"
        assert [i for _, s, i in merged if s == "evid"] == [0, 1, 2]
    assert plant(pool_t, evid, 4, planted_margin(dur, 4)) is None, "k > N/2"
    assert plant(pool_t, [], 8, 1.0) is None
    assert plant(pool_times(4.0, POOL), [1.0, 2.0, 3.0], 128, 1.0) is None, "too few fillers on a 4 s video"
    assert planted_margin(3000.0, 8) == 5.0 and planted_margin(60.0, 128) == 1.0 and planted_margin(60.0, 8) == 3.75
    # isolated: fillers stay outside the whole evidence interval; identical to planted for point evidence
    iv = [(20.0, 40.0)]
    et = evidence_times(iv, dur)
    loose, strict = plant(pool_t, et, 16, 2.0), plant(pool_t, et, 16, 2.0, intervals=iv)
    assert any(20.0 < pool_t[i] < 40.0 for i in loose[0]), "planted lets fillers fall inside the interval"
    assert all(not (18.0 < pool_t[i] < 42.0) for i in strict[0]) and len(strict[1]) == 16
    pts = [(10.0, 10.0), (25.0, 25.0)]
    assert plant(pool_t, evidence_times(pts, dur), 16, 2.0) == plant(pool_t, evidence_times(pts, dur), 16, 2.0, intervals=[(a - 0, b + 0) for a, b in pts][:0] or ())
    assert plant(pool_t, evidence_times([(1.0, 59.0)], dur), 16, 2.0, intervals=[(1.0, 59.0)]) is None, "an interval covering the clip leaves no fillers"


def test_unit_grid():
    # a point: centre t + 0.3, 0.5 s step, 9 frames; unit "frame" is exactly the planted frame
    c, s, g = unit_grid((10.0, 10.0), 100.0)
    assert (c, s) == (10.3, UNIT_STEP) and [j for j, _ in g] == list(range(-4, 5))
    assert [round(t, 2) for _, t in g] == [8.3, 8.8, 9.3, 9.8, 10.3, 10.8, 11.3, 11.8, 12.3]
    assert dict(g)[0] == evidence_times([(10.0, 10.0)], 100.0)[0]
    # a point near the start: offsets outside the video are dropped, j = 0 stays
    _, _, g0 = unit_grid((0.5, 0.5), 100.0)
    assert [j for j, _ in g0] == [-1, 0, 1, 2, 3, 4] and min(t for _, t in g0) >= 0.0
    # a range: centre = midpoint, step = (b - a) / 8, ends at j = +-4
    c, s, g = unit_grid((20.0, 40.0), 100.0)
    assert (c, s) == (30.0, 2.5) and [round(t, 2) for _, t in g] == [20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]
    # a 1 s shot: step 0.5, three frames {a, mid, b}; a 2 s range: five; >= 4 s: nine
    assert [j for j, _ in unit_grid((42.8, 43.8), 100.0)[2]] == [-1, 0, 1]
    assert [j for j, _ in unit_grid((10.0, 12.0), 100.0)[2]] == [-2, -1, 0, 1, 2]
    assert len(unit_grid((10.0, 14.0), 100.0)[2]) == 9
    assert dict(unit_grid((42.8, 43.8), 100.0)[2])[0] == evidence_times([(42.8, 43.8)], 100.0)[0]
    for name, offs in UNIT_OFFSETS.items():
        assert 0 in offs and offs == tuple(sorted(offs)) and set(offs) <= set(range(-UNIT_HALF, UNIT_HALF + 1)), name
    assert UNIT_OFFSETS["frame"] == (0,) and len(UNIT_OFFSETS["clip9_d2"]) == 9
    # fallback chain ends at "frame"
    for u in UNIT_OFFSETS:
        seen, cur = [], u
        while cur is not None:
            seen.append(cur); cur = UNIT_FALLBACK[cur]
        assert seen[-1] == "frame" and len(seen) == len(set(seen)), u


def test_negative_centres_and_plan_units():
    half = UNIT_HALF * UNIT_STEP
    dur = 120.0
    ivs = [(10.0, 10.0), (50.0, 60.0)]
    n = negative_centres((10.0, 10.0), ivs, dur, "s")
    assert n["hard"] == 10.0 + half + HARD_GAP, n                  # the grid starts HARD_GAP after the point
    e = n["easy"]
    assert e is not None and all(e + half <= a - EASY_MIN_DIST or e - half >= b + EASY_MIN_DIST for a, b in ivs)
    assert negative_centres((10.0, 10.0), ivs, dur, "s") == n, "seeded"
    assert negative_centres((10.0, 10.0), ivs, dur, "t")["easy"] != e, "seed changes the easy draw"
    # hard goes BEFORE the interval when the video ends right after it; None when neither side is clear
    assert negative_centres((110.0, 118.0), [(110.0, 118.0)], dur, "s")["hard"] == 110.0 - half - HARD_GAP
    assert negative_centres((3.0, 3.0), [(3.0, 3.0), (7.5, 7.5)], 12.0, "s")["hard"] is None
    # an occurrence 5 s later blocks the after-side hard negative (its grid would enter the padded interval)
    assert negative_centres((10.0, 10.0), [(10.0, 10.0), (15.0, 15.0)], dur, "s")["hard"] == 10.0 - half - HARD_GAP
    # a range covering the clip leaves no easy negative
    assert negative_centres((1.0, 59.0), [(1.0, 59.0)], 60.0, "s") == {"hard": None, "easy": None}
    info = plan_units("AC_1", [(40.0, 40.0), (10.0, 10.0)], dur, seed=0)
    kinds = [(u["unit"], u["kind"]) for u in info["units"]]
    assert kinds == [(0, "pos"), (0, "hard"), (0, "easy"), (1, "pos"), (1, "hard"), (1, "easy")], "sorted by time, negatives follow their unit"
    assert info["k"] == 2 and info["units"][0]["interval"] == [10.0, 10.0] and info["units"][0]["times"][4] == 10.3
    assert all(len(u["times"]) == 9 and u["offsets"] == list(range(-4, 5)) for u in info["units"])
    assert info["dropped"] == {"hard": 0, "easy": 0}
    assert plan_units("AC_1", [(40.0, 40.0), (10.0, 10.0)], dur, seed=0) == info, "deterministic"
    assert [(j, str(p)) for j, _, p in select_offsets(info["units"][0], "clip3_d1")] == [(-2, "u00_pos/frame_03.jpg"), (0, "u00_pos/frame_05.jpg"), (2, "u00_pos/frame_07.jpg")]
    assert [j for j, _, _ in select_offsets(info["units"][0], "frame")] == [0]


def _ref_herbench_prompt(question, candidates):
    """base_vlm.py::_format_prompt, copied verbatim (the reference for byte-equality)."""
    prompt = f"{question}\n\n"
    if candidates and candidates[0].strip()[0] in 'ABCDE':
        prompt += "\n".join(candidates)
    else:
        for i, candidate in enumerate(candidates):
            letter = chr(65 + i)
            prompt += f"{letter}. {candidate}\n"
    prompt += "\n\nPlease respond with only the correct answer letter (A, B, C, D, or E) without any explanations or additional text."
    return prompt


def _ref_minerva_prompt(question, choices):
    """lmms-eval minerva_doc_to_text with the yaml defaults, copied verbatim."""
    OPTIONS = ["A", "B", "C", "D", "E"]
    choice_text = "\n".join(f"{OPTIONS[idx]}. {choice}" for idx, choice in enumerate(choices))
    return f"{''}{question}\n{choice_text}{chr(10) + 'Answer with the option' + chr(39) + 's letter from the given choices directly.'}"


def test_prompts_byte_equal_official():
    ch = ["A. 3", "B. 5", "C. 6", "D. 2", "E. 4"]
    q = "How many times does the action-object pair 'pick up ginger' occur in this video?"
    assert herbench_prompt(q, ch) == _ref_herbench_prompt(q, ch)
    assert herbench_prompt(q, ["3", "5", "6", "2", "4"]) == _ref_herbench_prompt(q, ["3", "5", "6", "2", "4"])
    assert lettered(["3", "5"]) == ["A. 3", "B. 5"] and lettered(ch) == ch
    bare = ["red", "blue", "green", "black", "white"]
    assert minerva_prompt("What colour?", bare) == _ref_minerva_prompt("What colour?", bare)
    assert minerva_prompt("q", bare).endswith(MINERVA_SUFFIX) and herbench_prompt(q, ch).endswith(HERBENCH_SUFFIX)


def test_extractors():
    for raw, want in (("C", "C"), ("(B)", "B"), ("D.", "D"), ("Answer: E", "E"), ("The answer is B.", "B"),
                      ("b", "B"), ("ERROR", None), ("", None), ("AVAILABLE", None), ("I pick C, because", "C")):
        assert herbench_extract(raw) == want, (raw, herbench_extract(raw))
    for raw, want in (("C", "C"), ("The answer is B", "B"), ("Option: D", "D"), ("answer is E", "E"),
                      ("B. A red hoodie", "A"), ("", None), ("nothing", None)):
        assert minerva_extract(raw) == want, (raw, minerva_extract(raw))
    assert has_letter("C") and has_letter("Answer: B.") and not has_letter("The") and not has_letter("")


def _synthetic_tree(root: Path):
    dur = 60.0
    vid = "cam1_segment_6_300s_360s"
    rows = [
        {"qid": "AC_1", "video_id": vid, "video_path": f"videos/WildTrack/{vid}.mp4",
         "question": "How many?", "choices": ["A. 3", "B. 5", "C. 6", "D. 2", "E. 4"], "answer": "D", "answer_index": 3,
         "qtype": "AC", "task_name": "Action Counting", "source": "WildTrack", "duration": dur,
         "evidence_kind": "points", "evidence": [[10.0, 10.0], [25.0, 25.0], [40.0, 40.0]], "meta": {}},
        {"qid": "RLPC_1", "video_id": vid, "video_path": f"videos/WildTrack/{vid}.mp4",
         "question": "How many people?", "choices": ["A. 1-10", "B. 11-20", "C. 21-30", "D. 31-40", "E. 41+"],
         "answer": "A", "answer_index": 0, "qtype": "RLPC", "task_name": "Region Localized People Counting",
         "source": "WildTrack", "duration": dur, "evidence_kind": "all", "evidence": [], "meta": {}},
    ]
    (root / "json").mkdir(parents=True)
    (root / "json" / "test.json").write_text(json.dumps(rows))
    (root / "pool_native" / vid).mkdir(parents=True)
    (root / "pool_native" / vid / "times.json").write_text(json.dumps({"video_id": vid, "duration": dur, "times": pool_times(dur, POOL)}))
    (root / "evidence_native" / "AC_1").mkdir(parents=True)
    (root / "evidence_native" / "AC_1" / "times.json").write_text(json.dumps({"qid": "AC_1", "times": evidence_times([(10.0, 10.0), (25.0, 25.0), (40.0, 40.0)], dur)}))
    return rows, vid, dur


def test_load_synthetic_tree():
    spec = get_dataset("herbench")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, vid, dur = _synthetic_tree(root)
        s = spec.load(root, "planted_N8_test")
        assert [x.qid for x in s] == ["AC_1"] and spec.last_load["skipped"] == {"kind=all": 1}
        a = s[0]
        assert a.n_frames == 8 and a.group == vid and a.answer == "D" and a.qtype == "AC"
        assert a.evidence == frozenset(i for i, p in enumerate(a.frame_paths) if p.parent.name == "AC_1") and len(a.evidence) == 3
        assert a.meta["times"] == sorted(a.meta["times"]) and a.meta["protocol"] == "planted"
        assert sum(1 for p in a.frame_paths if p.parent.parent.name == "pool_native") == 5
        assert spec.question_text(a) == _ref_herbench_prompt("How many?", rows[0]["choices"])
        assert spec.answer_target(a) == "D" and spec.match("D", a) and not spec.match("A", a) and spec.match(spec.parse("D."), a)
        u = spec.load(root, "uniform_N8_test")
        assert [x.qid for x in u] == ["AC_1", "RLPC_1"] and spec.last_load["skipped"] == {}
        ua, ur = u
        assert all(p.parent.name == vid for p in ua.frame_paths) and [p.name for p in ua.frame_paths] == [f"frame_{1 + 32 * j:04d}.jpg" for j in range(8)]
        assert ua.evidence == uniform_labels([j * dur / 8 for j in range(8)], [(10.0, 10.0), (25.0, 25.0), (40.0, 40.0)], 8, dur)
        assert ur.evidence == frozenset(range(8))
        assert spec.load(root, "planted_N8_test", qtypes=["RLPC"]) == [] and spec.last_load["skipped"] == {"kind=all": 1}
        ex = spec.report_extras(u, ["D", "A"])
        assert ex["coverage(>=1 evidence frame)"][1] == 2 and ex["letter_prior_baseline"] == (0.5, 2)
        rows.append({**rows[0], "qid": "AC_2", "video_id": "broken_video"})
        (root / "json" / "test.json").write_text(json.dumps(rows))
        assert [x.qid for x in spec.load(root, "uniform_N8_test")] == ["AC_1", "RLPC_1"] and spec.last_load["skipped"] == {"no_pool": 1}
        try:
            spec.load(root, "planted_N8_test", qtypes=["nope"])
            assert False
        except ValueError:
            pass


def test_evidence_protocol_and_units():
    """The UNIT baseline on the synthetic tree: units.json written by plan_units (no jpgs needed for
    the loader), evidence_<unit> splits, the frame cap fallback, and the per-unit records."""
    spec = get_dataset("herbench")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows, vid, dur = _synthetic_tree(root)
        rows.append({**rows[0], "qid": "AC_8", "evidence": [[float(5 + 7 * i), float(5 + 7 * i)] for i in range(8)]})   # k = 8
        (root / "json" / "test.json").write_text(json.dumps(rows))
        (root / "evidence_native" / "AC_8").mkdir(parents=True)
        (root / "evidence_native" / "AC_8" / "times.json").write_text(json.dumps({"qid": "AC_8", "times": evidence_times([tuple(iv) for iv in rows[-1]["evidence"]], dur)}))
        for r in rows:
            if r["evidence_kind"] in ("all", "none"):
                continue
            qdir = root / UNITS_DIR / r["qid"]
            qdir.mkdir(parents=True)
            (qdir / "units.json").write_text(json.dumps(plan_units(r["qid"], [tuple(iv) for iv in r["evidence"]], dur)))
        s = spec.load(root, "evidence_frame_test")
        assert [x.qid for x in s] == ["AC_1", "AC_8"] and spec.last_load["skipped"] == {"kind=all": 1}
        a = s[0]
        assert a.n_frames == 3 and a.evidence == frozenset({0, 1, 2}) and a.meta["k"] == 3 and a.meta["m_eff"] == 1.0
        assert a.meta["protocol"] == "evidence" and a.meta["unit"] == a.meta["unit_eff"] == "frame"
        assert [p.parent.name for p in a.frame_paths] == ["u00_pos", "u01_pos", "u02_pos"] and all(p.name == "frame_05.jpg" for p in a.frame_paths)
        assert a.meta["times"] == [10.3, 25.3, 40.3] and a.meta["unit_of_frame"] == [0, 1, 2]
        c = spec.load(root, "evidence_clip9_d2_test")
        a9, k8 = c
        assert a9.n_frames == 27 and a9.meta["m_eff"] == 9.0 and a9.meta["times"] == sorted(a9.meta["times"])
        assert a9.meta["unit_of_frame"] == [0] * 9 + [1] * 9 + [2] * 9 and a9.evidence == frozenset(range(27))
        assert MAX_UNIT_FRAMES == 48
        assert k8.meta["k"] == 8 and k8.n_frames <= MAX_UNIT_FRAMES and k8.meta["unit_eff"] == "clip5_d2" and k8.n_frames == 40, "k=8: 9 (72 > 48) -> 5 frames per unit (40)"
        assert k8.meta["unit"] == "clip9_d2" and k8.meta["m_eff"] == 5.0
        c5 = spec.load(root, "evidence_clip5_d1_test")[1]
        assert c5.meta["unit_eff"] == "clip5_d1" and c5.n_frames == 40, "40 <= 48: no fallback"
        ex = spec.report_extras(c, ["D", "D"])
        assert ex["k=2-4"] == (1.0, 1) and ex["k>=5"] == (1.0, 1) and ex["unit_fallback_rows"] == (0.5, 2) and ex["mean_m_eff"][1] == 2
        # per-unit records for regime D, read at a clip: 3 pos + 3 hard + 3 easy, each with 5 frames
        u = spec.units(root, a, "clip5_d1")
        assert [x["kind"] for x in u] == ["pos", "hard", "easy"] * 3 and all(x["m"] == 5 and len(x["frame_paths"]) == 5 for x in u)
        assert u[0]["interval"] == [10.0, 10.0] and u[1]["interval"] is None and u[0]["k"] == 3
        assert u[1]["centre"] == 10.0 + UNIT_HALF * UNIT_STEP + HARD_GAP and u[1]["frame_paths"][0].parent.name == "u00_hard"
        half = UNIT_HALF * UNIT_STEP
        for x in u:
            if x["kind"] == "easy":
                assert all(x["centre"] + half <= t - EASY_MIN_DIST or x["centre"] - half >= t + EASY_MIN_DIST for t in (10.0, 25.0, 40.0))
            if x["kind"] == "hard":
                assert all(x["centre"] + half <= t - NEG_MARGIN or x["centre"] - half >= t + NEG_MARGIN for t in (10.0, 25.0, 40.0))
        assert spec.units(root, a, "frame")[0]["frame_paths"][0].name == "frame_05.jpg"
        assert spec.units(root, spec.load(root, "uniform_N8_test")[1], "frame") == [], "RLPC has no units"
        assert spec.bare_question(a) == "How many?\n" + "\n".join(rows[0]["choices"])
        # evidence_only (the generic one-frame reduction) agrees with the stored unit "frame" on times
        pl = spec.load(root, "planted_N8_test")[0]
        eo = spec.evidence_only(pl)
        assert eo.n_frames == 3 and eo.meta["k"] == 3 and eo.evidence == frozenset({0, 1, 2}) and eo.meta["unit"] == "frame"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
