#!/usr/bin/env python3
"""The read-side pieces of experiments/evaluate.py added 2026-10-11/12 (plan cells 8 and 4): the letter readout
(forced choice among the option letters at the first answer token), the two-pass sample (the gate's kept frames, in
time order, read unfenced) and the original-positions read's token map (variant 2: the kept prompt's tokens at the
positions they had in the full N-frame prompt). CPU, no model: a stub tokenizer stands in for the real one."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from core.data.base import Sample  # noqa: E402
from experiments.evaluate import (  # noqa: E402
    OPTION_LETTERS, LetterReadout, ReadGeometry, answer_of, caption_blocks, full_prompt_inputs, kept_indices,
    kept_sample, letter_readout, merge_kept_runs, option_letters, original_index, positions_with_tail, with_timestamps,
)

def _exits(fn, *args, **kwargs) -> bool:
    """True when fn raises SystemExit (the scripts' fail-early convention)."""
    try:
        fn(*args, **kwargs)
    except SystemExit:
        return True
    return False


VOCAB = {"A": 10, " A": 11, "B": 20, " B": 21, "C": 30, "D": 40, " D": 41, "E": 50}   # "C" has one spelling only


class StubTokenizer:
    """tokenizer(text, add_special_tokens=False).input_ids: one id per VOCAB entry, two ids for anything else."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[VOCAB[text]] if text in VOCAB else [1, 2])


def _sample(n_frames: int = 5, choices: int = 4) -> Sample:
    return Sample(qid="q", question="?", answer="B", qtype="t",
                  frame_paths=tuple(Path(f"f{t}.jpg") for t in range(n_frames)), evidence=frozenset({1, 3}), group="g",
                  meta={"choices": [f"{L}. x" for L in OPTION_LETTERS[:choices]], "k": 2})


def test_letter_readout_collects_single_token_spellings():
    readout = letter_readout(StubTokenizer())
    assert readout.ids_by_letter == {"A": [10, 11], "B": [20, 21], "C": [30], "D": [40, 41], "E": [50]}
    assert _exits(letter_readout, StubTokenizer(), letters="AZ")        # "Z" is not a single token


def test_letter_prediction_takes_the_best_spelling_and_only_the_offered_letters():
    readout = LetterReadout({"A": [10, 11], "B": [20, 21], "C": [30], "D": [40, 41], "E": [50]})
    logits = torch.full((60,), -10.0)
    logits[11] = 2.0                                     # " A" beats "A" (-10): a letter scores its best spelling
    logits[20] = 1.5
    logits[50] = 9.0                                                      # "E" is not offered on a four-choice question
    letter, margin = readout.predict(logits, "ABCD")
    assert letter == "A"
    assert abs(margin - 0.5) < 1e-6
    assert readout.predict(logits, "ABCDE")[0] == "E"


def test_option_letters_follow_the_choices():
    assert option_letters(_sample(choices=4)) == "ABCD"
    assert option_letters(_sample(choices=5)) == "ABCDE"
    assert _exits(option_letters, Sample("q", "?", "3", "t", (), None, "g", meta={}))   # MMReD: no lettered choices


def test_answer_of_dispatches_on_the_readout():
    spec = SimpleNamespace(parse=lambda raw: raw.strip()[:1] or None)
    assert answer_of(None, spec, _sample(), "B. two", None) == ("B", "")
    readout = LetterReadout({"A": [10], "B": [20], "C": [30], "D": [40]})
    logits = torch.zeros(60)
    logits[30] = 3.0
    assert answer_of(readout, spec, _sample(), "", logits) == ("C", "3.000")


def test_kept_sample_is_the_kept_frames_in_time_order():
    sample = _sample(n_frames=5)
    keep = [False, True, False, True, True]
    read = kept_sample(sample, keep)
    assert read.frame_paths == (Path("f1.jpg"), Path("f3.jpg"), Path("f4.jpg"))
    assert read.evidence is None and read.meta["protocol"] == "two-pass" and read.meta["kept"] == [1, 3, 4]
    assert read.meta["scan_frames"] == 5                                   # the N the gate scanned, for variant 2
    timed = kept_sample(replace(sample, meta={**sample.meta, "times": [0.0, 1.5, 3.0, 4.5, 6.0]}), keep)
    assert timed.meta["times_read"] == [1.5, 4.5, 6.0]                     # the kept frames' times, for the captions
    assert read.meta["k"] == 2 and read.qid == sample.qid                 # the row's identity and labels travel with it
    assert kept_indices(keep) == "1 3 4" and kept_indices([False] * 3) == ""
    assert kept_sample(sample, [False] * 5).frame_paths == ()            # an empty gate leaves a text-only read


def test_merge_kept_runs_keeps_one_frame_per_needle_second():
    times = [round(i * 40 / 256, 3) for i in range(256)]              # the official grid of a 40 s video, 6.4 fps
    needle = [5.0 <= t < 6.0 for t in times]                           # a 1 s needle starting at second 5: 6-7 frames
    merged = merge_kept_runs(needle, times)
    assert sum(merged) == 1 and abs(times[merged.index(True)] - 5.5) <= 40 / 256 / 2
    two = [5.0 <= t < 7.0 for t in times]                              # two needles back to back (seconds 5 and 6)
    assert sum(merge_kept_runs(two, times)) == 2
    assert merge_kept_runs([False] * 256, times) == [False] * 256
    planted_times = [1.5, 3.2, 7.8, 24.5, 27.5, 30.1]                  # needle frames at t + 0.5, fillers >= 1 s away
    planted_keep = [True, False, False, True, True, False]
    assert merge_kept_runs(planted_keep, planted_times) == planted_keep


def _raises(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_original_index_maps_kept_blocks_into_the_full_prompt():
    # kept prompt: prefix 3 tokens | block (frame 1) 4 tokens | block (frame 3) 4 tokens | tail 2 tokens; scan N = 5
    blocks = [(3, 7), (7, 11)]
    index = original_index(blocks, fin_start=11, kept=[1, 3], n_frames=5, seq_len=13)
    assert index == [0, 1, 2] + [7, 8, 9, 10] + [15, 16, 17, 18] + [23, 24]   # full prompt: 3 + 5 * 4 + 2 = 25 tokens
    assert original_index([(3, 7)], 7, [0], 1, 9) == list(range(9))            # one frame kept of one: identity
    assert _raises(original_index, blocks, 11, [1], 5, 13)                   # two blocks for one kept frame
    assert _raises(original_index, [(3, 7), (8, 12)], 12, [1, 3], 5, 14)     # a gap between the blocks
    assert _raises(original_index, [(3, 7), (7, 10)], 10, [1, 3], 5, 12)     # unequal blocks
    assert _raises(original_index, blocks, 12, [1, 3], 5, 13)                # the tail does not start at the last block
    packed = original_index(blocks, 11, [1, 3], 5, 13, frames="packed")      # blocks 0 and 1, the question after all 5
    assert packed == [0, 1, 2] + [3, 4, 5, 6] + [7, 8, 9, 10] + [23, 24]
    adjacent = original_index(blocks, 11, [1, 3], 4, 13)                     # n_frames = kept[-1] + 1: the question right after block 3
    assert adjacent == [0, 1, 2] + [7, 8, 9, 10] + [15, 16, 17, 18] + [19, 20]


def test_caption_blocks_and_timestamp_captions():
    assert caption_blocks([(5, 9), (9, 13)], 2) == [(3, 9), (7, 13)]
    assert _raises(caption_blocks, [(1, 5)], 2)                              # no room for a caption before block 0
    turn = {"role": "user", "content": [{"type": "image", "image": "a"}, {"type": "image", "image": "b"}, {"type": "text", "text": "q"}]}
    out = with_timestamps(turn, [1.5, 24.5])
    assert [item["type"] for item in out["content"]] == ["text", "image", "text", "image", "text"]
    assert out["content"][0]["text"] == "<0001.5 seconds>" and out["content"][2]["text"] == "<0024.5 seconds>"
    assert len(out["content"][0]["text"]) == len(out["content"][2]["text"])  # fixed width
    assert ReadGeometry().tag == "" and ReadGeometry().is_plain
    assert ReadGeometry("reset", "fenced", "timestamps").tag == "_reset_fenced_ts"
    assert ReadGeometry("packed", "fenced").tag == "_fenced" and not ReadGeometry("packed", "fenced").is_plain


def test_full_prompt_inputs_repeats_the_first_block():
    ids = torch.tensor([[1, 2, 3, 9, 8, 8, 7, 9, 8, 8, 7, 4, 5]])
    enc = {"input_ids": ids, "image_grid_thw": torch.tensor([[1, 4, 6], [1, 4, 6]])}
    full = full_prompt_inputs(enc, block=(3, 7), fin_start=11, n_frames=5)
    assert full["input_ids"].tolist() == [[1, 2, 3] + [9, 8, 8, 7] * 5 + [4, 5]]
    assert full["attention_mask"].shape == full["input_ids"].shape and int(full["attention_mask"].sum()) == 25
    assert full["image_grid_thw"].tolist() == [[1, 4, 6]] * 5
    assert "image_grid_thw" not in full_prompt_inputs({"input_ids": ids}, (3, 7), 11, 5)   # 1-D backbones carry no grid


def test_positions_with_tail_continue_after_the_last_position():
    pos = torch.tensor([[[0, 1, 5], [0, 1, 2], [0, 1, 3]]])                  # M-RoPE (3, 1, L): axes may differ
    assert positions_with_tail(pos, 5).tolist() == [[[0, 1, 5, 6, 7], [0, 1, 2, 3, 4], [0, 1, 3, 4, 5]]]
    assert torch.equal(positions_with_tail(pos, 3), pos)
    assert positions_with_tail(torch.tensor([[0, 1, 2]]), 4).tolist() == [[0, 1, 2, 3]]   # 1-D (1, L)


if __name__ == "__main__":
    test_letter_readout_collects_single_token_spellings()
    test_letter_prediction_takes_the_best_spelling_and_only_the_offered_letters()
    test_option_letters_follow_the_choices()
    test_answer_of_dispatches_on_the_readout()
    test_kept_sample_is_the_kept_frames_in_time_order()
    test_original_index_maps_kept_blocks_into_the_full_prompt()
    test_full_prompt_inputs_repeats_the_first_block()
    test_positions_with_tail_continue_after_the_last_position()
    test_merge_kept_runs_keeps_one_frame_per_needle_second()
    test_caption_blocks_and_timestamp_captions()
    print("ALL OK")
