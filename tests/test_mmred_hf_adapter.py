"""CPU parity test for the MMReD-HF adapter (gnnformer/mmred_hf.py).

Anchors: on the published seq_len_8 + seq_len_16 test splits (50/qtype each),
`recompute_answer` must reproduce the published gold answer for 100% of samples,
for ALL 24 qtypes (>=50 samples per qtype as required by the campaign brief; 100 here).
Also checks: states shape, probe-evidence consistency, and (if a render exists)
frame loading for a handful of samples.

Needs data/mmred_hf/json/ prepped (scripts/mmred_hf/prep.py). Pure CPU, seconds.
"""
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gnnformer.mmred_hf import (  # noqa: E402
    DC_QTYPES,
    NIAH_QTYPES,
    load_index,
    load_mmred_hf_sample,
    probe_evidence_mmred,
    recompute_answer,
    row_states,
)

JSON_DIR = REPO_ROOT / "data/mmred_hf/json"
ROOMS = {"Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"}


def main():
    assert set(NIAH_QTYPES) | set(DC_QTYPES) == set(NIAH_QTYPES + DC_QTYPES)
    assert len(NIAH_QTYPES) == 15 and len(DC_QTYPES) == 9

    rows = []
    for name in ("seq_len_8_test.json", "seq_len_16_test.json"):
        path = JSON_DIR / name
        assert path.exists(), f"missing {path} — run scripts/mmred_hf/prep.py first"
        rows += load_index(path)

    per_qtype = defaultdict(int)
    mismatches = []
    for row in rows:
        states = row_states(row)
        assert len(states) == row["seq_len"]
        assert all(set(st["rooms"]) == ROOMS for st in states), row["qid"]
        got = recompute_answer(row["qtype"], row["question"], states)
        if str(got) != str(row["answer"]):
            mismatches.append((row["qtype"], row["qid"], got, row["answer"]))
        per_qtype[row["qtype"]] += 1

    assert len(per_qtype) == 24, f"expected 24 qtypes, saw {len(per_qtype)}"
    low = {q: n for q, n in per_qtype.items() if n < 50}
    assert not low, f"qtypes below 50 samples: {low}"
    assert not mismatches, (
        f"{len(mismatches)}/{len(rows)} answer mismatches; first 10: {mismatches[:10]}"
    )
    print(f"answer parity: {len(rows)} samples, 24 qtypes, 0 mismatches")

    # probe evidence consistency (where defined)
    n_evid = 0
    for row in rows:
        states = row_states(row)
        pe = probe_evidence_mmred(row["qtype"], row["question"], states)
        if pe is None:
            continue
        evid, locus = pe
        assert isinstance(locus, str) and locus
        assert all(0 <= t < row["seq_len"] for t in evid), row["qid"]
        if row["qtype"] == "steps_in_room":
            assert len(evid) == int(row["answer"]), (row["qid"], evid, row["answer"])
        else:
            assert len(evid) == 1, (row["qid"], row["qtype"], evid)
        n_evid += 1
    print(f"probe evidence: {n_evid} samples consistent")

    # frame loading (only if a render exists yet)
    for images_root in (REPO_ROOT / "data/mmred_hf/images/seq_len_8_test",
                        REPO_ROOT / "data/mmred_hf/images_probe/run1"):
        if images_root.is_dir():
            checked = 0
            for row in load_index(JSON_DIR / "seq_len_8_test.json"):
                if not (images_root / row["qid"]).is_dir():
                    continue
                qid, frames, q, states, ans = load_mmred_hf_sample(row, images_root)
                assert len(frames) == row["seq_len"] and frames[0].size[0] > 0
                checked += 1
                if checked >= 5:
                    break
            print(f"frame loading: {checked} samples from {images_root.name} OK")
            break
    else:
        print("frame loading: SKIPPED (no render dir yet)")

    print("test_mmred_hf_adapter: ALL PASS")


if __name__ == "__main__":
    main()
