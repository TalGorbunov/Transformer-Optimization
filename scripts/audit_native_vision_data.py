"""Post-launch, read-only QA semantics audit of the fixed V1 vision manifests.

Run on Slurm CPU. This diagnostic neither reads model predictions nor changes
samples, training, checkpoint selection, or the registered model comparisons.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = Path("/mnt/data/gabriele/gnn_transformer/native_aggregation_vision_pilot")
METRICS = ("matches", "target_character_frames", "same_character_wrong_room",
           "queried_room_other_character", "other_character_frames", "neither")


def summarize(rows: list[dict]) -> dict:
    result = dict(count=len(rows), gold_histogram=dict(sorted(Counter(r["gold"] for r in rows).items())),
                  target_characters=dict(sorted(Counter(r["target_character"] for r in rows).items())),
                  target_rooms=dict(sorted(Counter(r["target_room"] for r in rows).items())),
                  templates=dict(Counter(r["question_template"] for r in rows)),
                  origins=dict(Counter(r["origin"] for r in rows)))
    for metric in METRICS:
        values = [r[metric] for r in rows]
        result[metric] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values),
                              stdev=statistics.pstdev(values), histogram=dict(sorted(Counter(values).items())))
    result["target_character_above_8"] = sum(r["target_character_frames"] > 8 for r in rows)
    result["people_per_frame"] = dict(sorted(sum((Counter(r["people_per_frame"]) for r in rows), Counter()).items()))
    return result


def inspect(row: dict, purpose: str, group: str) -> dict:
    source = Path(row["source_path"]) / "qa.txt"
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["qa_sha256"]:
        raise ValueError(f"Source QA differs from staged manifest: {source}")
    lines = payload.decode().splitlines()
    q_start = next(i for i, line in enumerate(lines) if line.strip() == "question:")
    a_start = next(i for i, line in enumerate(lines) if line.strip() == "answer:")
    block = [line.strip() for line in lines[q_start + 1:a_start] if line.strip()]
    states = [ast.literal_eval(line) for line in block if line.startswith("{") and line.endswith("}")]
    questions = [line for line in block if not (line.startswith("{") and line.endswith("}"))]
    if len(states) != row["n_frames"] or len(questions) != 1:
        raise ValueError(f"Malformed states/question: {source}")
    match = re.fullmatch(r"How many frames show ([A-Za-z]+) in the ([A-Za-z]+)\?", questions[0])
    if match is None:
        raise ValueError(f"Unexpected question template: {source}")
    character, room = match.groups()
    counts = Counter()
    people_per_frame = []
    for index, state in enumerate(states):
        if state["step_id"] != index + 1:
            raise ValueError(f"Noncanonical step index: {source}")
        occupancy = state["rooms"]
        people = [person for occupants in occupancy.values() for person in occupants]
        people_per_frame.append(len(people))
        target = character in people
        conjunction = character in occupancy.get(room, [])
        room_other = any(person != character for person in occupancy.get(room, []))
        counts["matches"] += conjunction
        counts["target_character_frames"] += target
        counts["same_character_wrong_room"] += target and not conjunction
        counts["queried_room_other_character"] += room_other
        counts["other_character_frames"] += not target
        counts["neither"] += not target and not room_other
    source_gold = int(next(line.strip() for line in lines[a_start + 1:] if line.strip()))
    if source_gold != row["gold"] or counts["matches"] != source_gold:
        raise ValueError(f"Gold mismatch: {source}, manifest={row['gold']}, source={source_gold}, recount={counts['matches']}")
    if set(people_per_frame) != {1}:
        raise ValueError(f"Expected one character per frame: {source}")
    if counts["matches"] + counts["same_character_wrong_room"] + counts["queried_room_other_character"] + counts["neither"] != row["n_frames"]:
        raise AssertionError(f"Distractor categories do not partition frames: {source}")
    return dict(purpose=purpose, group=group, sid=row["sid"], n_frames=row["n_frames"],
                gold=row["gold"], target_character=character, target_room=room,
                question_template="How many frames show <CHARACTER> in the <ROOM>?",
                origin=row["origin"], source_qa=str(source), qa_sha256=row["qa_sha256"],
                people_per_frame=people_per_frame, **counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/native_aggregation_vlm/v1/data_audit")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run the metadata diagnostic inside a Slurm CPU allocation")
    started = time.monotonic()
    rows, cells, hashes = [], {}, {}
    for purpose in ("main", "profile"):
        path = args.manifest_root / f"{purpose}_manifest.json"
        payload = path.read_bytes()
        hashes[purpose] = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        for group, selection in manifest["splits"].items():
            group_rows = [inspect(row, purpose, group) for row in selection["samples"]]
            rows.extend(group_rows)
            by_gold = defaultdict(list)
            for row in group_rows:
                by_gold[row["gold"]].append(row)
            summary = summarize(group_rows)
            summary["by_gold"] = {gold: summarize(subset) for gold, subset in sorted(by_gold.items())}
            cells[f"{purpose}/{group}"] = summary
            print(json.dumps(dict(cell=f"{purpose}/{group}", count=summary["count"],
                                  target_character_mean=summary["target_character_frames"]["mean"],
                                  target_character_range=[summary["target_character_frames"]["minimum"], summary["target_character_frames"]["maximum"]],
                                  target_character_above_8=summary["target_character_above_8"],
                                  same_character_wrong_room_mean=summary["same_character_wrong_room"]["mean"],
                                  queried_room_other_character_mean=summary["queried_room_other_character"]["mean"])), flush=True)
    comparisons = {}
    for n_frames in (8, 16):
        train, dev, test = [cells[f"main/{split}_N{n_frames}"] for split in ("train", "dev", "test")]
        contrasts = {}
        for metric in METRICS:
            contrasts[metric] = dict(train_minus_test=train[metric]["mean"] - test[metric]["mean"],
                                     dev_minus_test=dev[metric]["mean"] - test[metric]["mean"],
                                     train_minus_test_gold_uniform=statistics.mean(
                                         train["by_gold"][gold][metric]["mean"] - test["by_gold"][gold][metric]["mean"] for gold in range(9)),
                                     dev_minus_test_gold_uniform=statistics.mean(
                                         dev["by_gold"][gold][metric]["mean"] - test["by_gold"][gold][metric]["mean"] for gold in range(9)))
        comparisons[f"N{n_frames}_same_length_split_shift"] = contrasts
    result = dict(schema_version=1, slurm_job_id=os.environ["SLURM_JOB_ID"],
                  purpose="Post-launch data-semantic diagnostic; no model outcomes read or protocol changes",
                  manifest_sha256=hashes, source_qa_sha256_all_verified=True,
                  gold_recount_all_verified=True, samples=len(rows), cells=cells,
                  same_length_split_comparisons=comparisons, records=rows,
                  elapsed_seconds=time.monotonic() - started)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"audit_{os.environ['SLURM_JOB_ID']}.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Verified all {len(rows)} QA hashes, count labels, and single-character frames; {destination}", flush=True)


if __name__ == "__main__":
    main()
