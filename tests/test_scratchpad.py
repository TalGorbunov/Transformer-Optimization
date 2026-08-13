"""CPU tests for scratchpad target builders + the eval parser (port of the legacy
format-sweep phase-0 sanity gate). Synthetic samples cover every task x format cell;
structural checks pin the inline-tally / chunk-subtotal algebra; a tokenizer round-trip
runs when the Qwen tokenizer is available locally (skipped otherwise, never failed).

Run: .venv/bin/python tests/test_scratchpad.py   (or pytest tests/)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.scratchpad import (
    CHUNK_FRAMES,
    SCRATCHPAD_FORMATS,
    build_target_fmt,
    couple_offsets,
    parse_scratchpad_answer,
)

# (task, evid 0-indexed, aux, gold, NF, labels)
CASES = [
    ("steps", {1, 4, 7}, None, 3, 8, {1: "Kitchen", 4: "Kitchen", 7: "Kitchen"}),
    ("steps", set(), None, 0, 8, {}),
    ("cooc", {0, 5}, None, 2, 8, {0: "Garden", 5: "Office"}),
    ("union", {2, 3}, None, 2, 8, {2: "Park", 3: "Garden"}),
    ("which", {4}, None, 5, 8, {4: "Bedroom"}),
    ("rooms", {0, 2, 5}, ["Garden", "Kitchen"], 2, 8, {0: "Garden", 2: "Kitchen", 5: "Garden"}),
    ("steps", {3, 17, 30, 44}, None, 4, 48, {t: "Office" for t in (3, 17, 30, 44)}),
]


def check_scan(tgt, task, gold, NF, caption):
    assert tgt.endswith(" END"), tgt
    body, tail = tgt.split(" | total: ")
    slots = body.replace(" scan: ", "").split(" ")
    assert len(slots) == NF, (len(slots), NF)
    for i, s in enumerate(slots):
        assert s.startswith(f"f{i+1}:"), s
    tallies = [int(m) for m in re.findall(r"\((\d+)\)", body)]
    assert tallies == list(range(1, len(tallies) + 1)), tallies
    total = int(tail.split(" END")[0])
    if task != "which":
        assert total == (tallies[-1] if tallies else 0) == gold, (total, tallies, gold)
    else:
        assert total == gold
    if caption and task != "which":
        assert "yes" not in body, "caption arm fell back to 'yes': " + tgt


def check_chunked(tgt, task, gold, NF):
    assert tgt.endswith(" END"), tgt
    n_chunks = (NF + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    subs = [int(m) for m in re.findall(r"\| sub (\d+)", tgt)]
    assert len(subs) == n_chunks, (subs, n_chunks)
    tail = tgt.rsplit("total: ", 1)[1]
    if task != "which":
        expr = tail.split(" = ")[0]
        assert [int(x) for x in expr.split("+")] == subs, (expr, subs)
        assert sum(subs) == gold == int(tail.split(" = ")[1].split(" END")[0])
    else:
        assert int(tail.split(" END")[0]) == gold


def test_build_and_parse_all_formats():
    for task, evid, aux, gold, NF, labels in CASES:
        for fmt in SCRATCHPAD_FORMATS:
            tgt = build_target_fmt(
                fmt, task, evid, aux, gold, NF=NF, labels=(labels if fmt != "poslist" else None)
            )
            back = parse_scratchpad_answer(tgt, fmt)
            assert back == gold, f"parse-back FAIL [{fmt}] {task} gold={gold} got={back}: {tgt!r}"
            if fmt in ("scan", "caption"):
                check_scan(tgt, task, gold, NF, caption=(fmt == "caption"))
            elif fmt == "chunked":
                check_chunked(tgt, task, gold, NF)


def test_caption_falls_back_to_yes_without_labels():
    tgt = build_target_fmt("caption", "steps", {1, 3}, None, 2, NF=8, labels={})
    assert "yes(1)" in tgt and parse_scratchpad_answer(tgt, "caption") == 2


def test_parser_edge_cases():
    assert parse_scratchpad_answer("garbage", "poslist") is None
    assert parse_scratchpad_answer("garbage", "caption") is None
    assert parse_scratchpad_answer(" frames 2 (1), 9 (2) -> 2", "poslist") == 2
    # END never decoded: still parses the last integer after 'total:'
    assert parse_scratchpad_answer(" scan: f1:- f2:yes(1) | total: 1", "scan") == 1


def test_couple_offsets_stream_rule():
    toks = [" frames", " 2", " (", "1", ")", ",", " 5", " (", "2", ")", " ->", " 2"]
    out = couple_offsets(toks, NF=8)
    assert out[0] == (1, 0)  # starts on carrier 1
    anchors = [a for a, _ in out]
    assert 2 in anchors and 5 in anchors  # jumps to the named frames
    tail_anchor = out[-1][0]
    assert anchors[anchors.index(5):].count(tail_anchor) >= 1  # '->' freezes the anchor
    assert out == couple_offsets(toks, NF=8)  # deterministic


def test_tokenizer_round_trip():
    try:
        from transformers import AutoTokenizer

        from gnnformer.constants import MODEL_7B

        tok = AutoTokenizer.from_pretrained(MODEL_7B)
    except Exception as exc:
        print(f"  [skip] tokenizer unavailable: {exc}")
        return
    for task, evid, aux, gold, NF, labels in CASES:
        for fmt in SCRATCHPAD_FORMATS:
            tgt = build_target_fmt(
                fmt, task, evid, aux, gold, NF=NF, labels=(labels if fmt != "poslist" else None)
            )
            ids = tok(tgt, add_special_tokens=False).input_ids
            assert tok.decode(ids) == tgt, f"round-trip FAIL [{fmt}]: {tgt!r}"


def test_legacy_parity():
    sys.path.insert(0, str(_REPO / "legacy"))
    try:
        from experiments.glstm.carrier_layer_lora import (  # type: ignore
            build_target_fmt as l_fmt,
            parse_scratchpad_answer as l_parse,
        )
    except Exception as exc:
        print(f"  [skip] legacy import unavailable: {exc}")
        return
    for task, evid, aux, gold, NF, labels in CASES:
        for fmt in SCRATCHPAD_FORMATS:
            lab = labels if fmt != "poslist" else None
            ours = build_target_fmt(fmt, task, evid, aux, gold, NF=NF, labels=lab)
            theirs = l_fmt(fmt, task, evid, aux, gold, NF=NF, labels=lab)
            assert ours == theirs, f"target drift [{fmt}/{task}]: {ours!r} != {theirs!r}"
            assert parse_scratchpad_answer(ours, fmt) == l_parse(ours, fmt)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
