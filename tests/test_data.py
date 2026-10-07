"""CPU tests for the dataset seam (core/data): the Sample/DatasetSpec contract on the MMReD
adapter — same rows and order as core.mmred.load_split, evidence == evidence_frames, gold
parity through verify() on the real seq_len_8 test split (skips if the data is not mounted),
parse/match == core.prompt, stratified_order parity, and the shared ARMS table.

Run: python tests/test_data.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.constants import ROOMS, SEQ_LENS
from core.data import DATASETS, DEFAULT_DATASET, get_dataset, stratified_order
from core.data.base import DatasetSpec, Sample
from core.data.mmred import MMReD
from core.mmred import QTYPES, evidence_frames, load_split, states
from core.mmred import stratified_order as mmred_order
from core.prompt import ARMS, LAYOUTS, SYSTEM_PROMPT, answer_target, exact_match, parse_answer

DATA_ROOT = _REPO / "data" / "mmred_hf"


def _tempdir_rows(root: Path):
    (root / "json").mkdir()
    rows = []
    for i, (qt, q, ans) in enumerate([
        ("steps_in_room", "How many steps did Daniel spend in the Kitchen?", "1"),
        ("n_empty", "How many rooms were empty at step 2?", "5"),
        ("char_at_frame", "In which room was Daniel at step 1?", "Kitchen"),
    ]):
        seq = [{"step_id": 1, "rooms": {r: (["Daniel"] if r == "Kitchen" else []) for r in ROOMS}},
               {"step_id": 2, "rooms": {r: (["Daniel"] if r == "Garden" else []) for r in ROOMS}}]
        rows.append({"qid": f"{i:07d}", "seq_len": 2, "qtype": qt, "atype": "x", "question": q, "answer": ans, "sequence": seq})
    (root / "json" / "seq_len_2_test.json").write_text(json.dumps(rows))
    return rows


def test_registry():
    spec = get_dataset("mmred")
    assert isinstance(spec, MMReD) and isinstance(spec, DatasetSpec)
    assert DEFAULT_DATASET == "mmred" and set(DATASETS) >= {"mmred"}
    assert list(spec.qtypes) == QTYPES and spec.protocols == ("mmred",)
    assert spec.has_train and spec.supports_counterfactual
    assert spec.system_prompt == SYSTEM_PROMPT and spec.default_root == "data/mmred_hf"
    try:
        get_dataset("nope")
        assert False, "unknown dataset must raise"
    except KeyError:
        pass


def test_split_name():
    spec = get_dataset("mmred")
    assert spec.split_name("mmred", 8, "test") == "seq_len_8_test"
    assert [spec.split_name("mmred", n, "train") for n in SEQ_LENS] == [f"seq_len_{n}_train" for n in SEQ_LENS]
    for bad in ((("planted", 8, "test")), (("mmred", 7, "test"))):
        try:
            spec.split_name(*bad)
            assert False, bad
        except ValueError:
            pass


def test_samples_mirror_rows():
    """Samples carry the row's fields, the official frame paths, evidence_frames and group=qid."""
    spec = get_dataset("mmred")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows = _tempdir_rows(root)
        samples = spec.load(root, "seq_len_2_test")
        assert [s.qid for s in samples] == [r["qid"] for r in load_split("seq_len_2", "test", root)]
        for s, r in zip(samples, rows):
            assert isinstance(s, Sample) and s.qtype == r["qtype"] and s.question == r["question"]
            assert s.answer == r["answer"] and s.group == s.qid and s.n_frames == 2
            assert s.frame_paths == tuple(root / "images" / "seq_len_2_test" / s.qid / f"frame_{k:04d}.png" for k in (1, 2))
            ev = evidence_frames(r["qtype"], r["question"], states(r))
            assert s.evidence == (None if ev is None else frozenset(ev))
            assert spec.verify(s), (s.qtype, s.answer)
            assert spec.question_text(s) == r["question"] and spec.answer_target(s) == answer_target(r["answer"])
        assert samples[0].evidence == frozenset({0}) and samples[1].evidence == frozenset({1})
        assert [s.qtype for s in spec.load(root, "seq_len_2_test", qtypes=["n_empty"])] == ["n_empty"]
        # a wrong published answer is caught by verify(), not silently accepted
        bad = Sample(**{**samples[0].__dict__, "answer": "7"})
        assert not spec.verify(bad)


def test_parse_match_delegate():
    spec = get_dataset("mmred")
    s = Sample(qid="x", question="q", answer="Kitchen", qtype="char_at_frame", frame_paths=(), evidence=None, group="x")
    for raw in ('{ "answer": "Kitchen" }', "kitchen", '{"answer": "Garden"}', "no idea"):
        assert spec.parse(raw) == parse_answer(raw)
        assert spec.match(spec.parse(raw), s) == exact_match(parse_answer(raw), "Kitchen")
    assert spec.stop_decoding('{ "answer": "3" }') and not spec.stop_decoding('{ "answer": "3')
    n = Sample(qid="y", question="q", answer="3", qtype="steps_in_room", frame_paths=(), evidence=None, group="y")
    assert spec.match("03", n) and not spec.match(None, n)
    extras = spec.report_extras([n, s], ["3", "Kitchen"])
    assert extras["numeric<=16"] == (1.0, 1) and extras["numeric>16"][1] == 0


def test_stratified_order_parity():
    """The generic Sample order equals core.mmred.stratified_order on the same rows and seed."""
    rows = [{"qid": str(i), "answer": "0"} for i in range(50)] + [{"qid": str(50 + i), "answer": "1"} for i in range(10)] \
        + [{"qid": str(60 + i), "answer": "2"} for i in range(5)]
    samples = [Sample(qid=r["qid"], question="q", answer=r["answer"], qtype="t", frame_paths=(), evidence=None, group=r["qid"])
               for r in rows]
    for seed in (0, 1, 7):
        assert [s.qid for s in stratified_order(samples, seed)] == [r["qid"] for r in mmred_order(rows, seed)]
    head = [s.answer for s in stratified_order(samples, 0)[:10]]
    assert len(set(head)) >= 2, head


def test_real_split_gold_parity():
    """verify() holds on every row of the real seq_len_8 test split; order == load_split; frames exist."""
    if not (DATA_ROOT / "json" / "seq_len_8_test.json").exists():
        print("  [skip] data/mmred_hf not mounted")
        return
    spec = get_dataset("mmred")
    samples = spec.load(DATA_ROOT, "seq_len_8_test")
    rows = load_split("seq_len_8", "test", DATA_ROOT)
    assert len(samples) == len(rows) == 1200
    assert [s.qid for s in samples] == [r["qid"] for r in rows]
    bad = [s.qid for s in samples if not spec.verify(s)]
    assert not bad, bad[:5]
    assert all(s.n_frames == 8 for s in samples)
    if samples[0].frame_paths[0].exists():
        assert all(p.exists() for s in samples[:3] for p in s.frame_paths)
    else:
        print("  [skip] renders not mounted; frame paths not checked")


def test_evidence_only_and_units():
    """The UNIT baseline's generic reductions on MMReD: evidence_only keeps the evidence frames in
    order (meta k / m_eff / unit), units() labels every frame pos / neg, bare_question is the question."""
    spec = get_dataset("mmred")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _tempdir_rows(root)
        s0, s1, s2 = spec.load(root, "seq_len_2_test")
        e = spec.evidence_only(s0)
        assert e.frame_paths == (s0.frame_paths[0],) and e.evidence == frozenset({0}) and e.meta["k"] == 1
        assert e.meta["protocol"] == "evidence" and e.meta["unit"] == e.meta["unit_eff"] == "frame" and e.meta["m_eff"] == 1.0
        assert e.qid == s0.qid and e.answer == s0.answer and spec.verify(e)
        assert spec.evidence_only(s1).frame_paths == (s1.frame_paths[1],)
        none = Sample(**{**s0.__dict__, "evidence": frozenset()})
        assert spec.evidence_only(none) is None and spec.evidence_only(Sample(**{**s0.__dict__, "evidence": None})) is None
        u = spec.units(root, s0, "frame")
        assert [(x["uid"], x["kind"], x["m"], x["k"]) for x in u] == [("f00", "pos", 1, 1), ("f01", "neg", 1, 1)]
        assert u[1]["frame_paths"] == (s0.frame_paths[1],)
        try:
            spec.units(root, s0, "clip3_d1")
            assert False
        except ValueError:
            pass
        assert spec.bare_question(s0) == s0.question
        ex = spec.report_extras([e, spec.evidence_only(s1)], ["1", "5"])
        assert ex["k=1"] == (1.0, 2) and ex["mean_m_eff"] == (1.0, 2) and ex["unit_fallback_rows"] == (0.0, 2)
        assert "k=1" not in spec.report_extras([s0, s1], ["1", "5"]), "no strata without k"


def test_arms():
    assert set(ARMS) == {"plain", "qfirst", "fenced_qlast", "fenced_qfirst", "gated"}
    for a in ARMS.values():
        assert a.layout in LAYOUTS and a.gate in ("none", "oracle") and (a.gate == "none" or a.fence)
    assert (ARMS["fenced_qfirst"].layout, ARMS["fenced_qfirst"].fence) == ("question-first", True)
    assert (ARMS["plain"].layout, ARMS["plain"].fence) == ("paper", False)
    assert ARMS["gated"].gate == "oracle"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
