#!/usr/bin/env python3
"""FORMAT sweep Phase-0 sanity (2026-07-22, CPU/tokenizer only — no GPU, no model).

For ~20 samples per arm drawn across all task roots of the l12v2 mixture:
build gold text -> tokenizer round-trip -> parse back with the eval parser ->
recovered answer == gold; verify inline tally / chunk-subtotal correctness;
log token cost per (fmt, N) to fix the eval decode budgets in the PREREG.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import re
from evaluations.helpers.utils import iter_sample_dirs_shuffled, load_mmred_sample
from experiments.glstm.carrier_layer_lora import (parse_task_labels, frame_attr_labels,
                                                  build_target_fmt, parse_scratchpad_answer,
                                                  SCRATCHPAD_FORMATS, CHUNK_FRAMES)

ROOTS = [  # one root per task family + the long-N roots (l12v2 mixture members)
    "data/mmred_images_park/seq_len_8/all_uniform",
    "data/mmred_cooc_balanced/seq_len_8/all_uniform",
    "data/mmred_rooms_balanced/seq_len_8/all_uniform",
    "data/mmred_niah_which/seq_len_8/all_uniform",
    "data/mmred_union_or/seq_len_8/all_uniform",
    "data/mmred_longN_park/seq_len_16/all_uniform",
    "data/mmred_longN_park/seq_len_32/all_uniform",
    "data/mmred_longN_park/seq_len_64/all_uniform",
    "data/mmred_longN_park2/seq_len_48/all_uniform",
]
PER_ROOT = 3   # ~27 samples total, >=20 checked per arm


def check_scan(tgt, task, gold, NF, caption):
    """Structural checks for scan/caption: NF slots in frame order; tallies increment
    1..K; count-task total == last tally; END terminator present."""
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
    """Chunk count == ceil(NF/16); 'sub k' matches items per chunk; sum expr == gold."""
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


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    n_ok = defaultdict(int)
    tok_cost = defaultdict(list)   # (fmt, NF) -> token counts
    n_samples = 0
    for root in ROOTS:
        dirs = iter_sample_dirs_shuffled(Path(root), 0)[:PER_ROOT]
        for sd in dirs:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            parsed = parse_task_labels(q0, states, gold)
            if parsed is None:
                print(f"  [skip] {sd} (parse_task_labels None)")
                continue
            task, evid, aux = parsed
            NF = len(states)
            n_samples += 1
            labels = frame_attr_labels(task, q0, states, evid)
            assert set(labels) == set(evid) or task not in (
                "rooms", "steps", "cooc", "union", "which"), (task, evid, labels)
            for fmt in SCRATCHPAD_FORMATS:
                tgt = build_target_fmt(fmt, task, evid, aux, gold, NF=NF,
                                       labels=(labels if fmt != "poslist" else None))
                ids = tok(tgt, add_special_tokens=False).input_ids
                rt = tok.decode(ids)
                assert rt == tgt, f"round-trip FAIL [{fmt}] {tgt!r} -> {rt!r}"
                back = parse_scratchpad_answer(tgt, fmt)
                assert back == gold, f"parse-back FAIL [{fmt}] {task} gold={gold} " \
                                     f"parsed={back} tgt={tgt!r}"
                if fmt in ("scan", "caption"):
                    check_scan(tgt, task, gold, NF, caption=(fmt == "caption"))
                elif fmt == "chunked":
                    check_chunked(tgt, task, gold, NF)
                n_ok[fmt] += 1
                tok_cost[(fmt, NF)].append(len(ids))
                if n_ok[fmt] <= 1 or (NF >= 32 and len(tok_cost[(fmt, NF)]) == 1):
                    print(f"[example fmt={fmt} task={task} N={NF} gold={gold} "
                          f"tok={len(ids)}] {tgt[:140]!r}{'...' if len(tgt)>140 else ''}")
    print(f"\nsamples={n_samples}  per-arm checks OK: "
          + " ".join(f"{f}:{n_ok[f]}" for f in SCRATCHPAD_FORMATS))
    print("\ntoken cost per (fmt, N)  [min/mean/max]:")
    for (fmt, NF) in sorted(tok_cost, key=lambda k: (k[0], k[1])):
        c = tok_cost[(fmt, NF)]
        print(f"  {fmt:8s} N={NF:3d}  {min(c)}/{sum(c)/len(c):.0f}/{max(c)}")
    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
