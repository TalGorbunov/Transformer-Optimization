"""Read-only post-hoc formatting diagnostics for native aggregation P0.

Usage: python scripts/analyze_native_aggregation.py RUN_DIR [RUN_DIR ...]

Run under CPU Slurm alongside other report work. Uses only saved JSON and the
standard library; loads no models and changes no files. JSON goes to stdout.
Registered whole-integer exact remains primary. Leading-integer accuracy is an
explicitly post-hoc diagnostic, NOT a replacement success criterion. Text after
the leading integer is ignored only for that diagnostic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import mean


def leading_integer(text):
    # A token boundary excludes incidental prefixes such as "4th" or "4abc".
    match = re.match(r"\s*([0-9]+)(?=\s|$|[.,:;!?])", text)
    return int(match.group(1)) if match else None


def summarize(rows):
    n = len(rows)
    leading = [leading_integer(row["output_text"]) for row in rows]
    strict = [bool(row["exact"]) for row in rows]
    parsed = [row["prediction"] is not None for row in rows]
    correct_leading = [value == row["gold"] for value, row in zip(leading, rows)]
    formatting_only = sum(good and not valid for good, valid in zip(correct_leading, parsed))
    return dict(
        n=n, strict_correct=sum(strict), strict_exact=mean(strict),
        strict_parsed=sum(parsed), strict_parse_rate=mean(parsed),
        leading_parsed=sum(value is not None for value in leading),
        leading_correct=sum(correct_leading), leading_exact_posthoc=mean(correct_leading),
        unparsed_correct_leading=formatting_only,
        unparsed_incorrect_leading=sum(not valid and value is not None and not good
                                      for valid, value, good in zip(parsed, leading, correct_leading)),
        unparsed_no_leading_integer=sum(not valid and value is None
                                       for valid, value in zip(parsed, leading)),
        strict_parsed_wrong=sum(valid and not good for valid, good in zip(parsed, strict)),
        gold_first_token_nll=mean(row["gold_first_token_nll"] for row in rows),
        mean_seconds=mean(row["seconds"] for row in rows),
        mean_generated_tokens=mean(row["generated_tokens"] for row in rows),
    )


def paired(left, right):
    def keyed(rows):
        result = {(row["n_frames"], row["sid"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate sample keys in one comparison group")
        return result

    lhs, rhs = keyed(left), keyed(right)
    if lhs.keys() != rhs.keys():
        raise ValueError("Paired comparison requires identical sample sets; no silent intersection")
    a, b = [], []
    for key in sorted(lhs):
        if lhs[key]["gold"] != rhs[key]["gold"]:
            raise ValueError(f"Gold differs for paired sample {key}")
        a.append(lhs[key])
        b.append(rhs[key])
    result = dict(n=len(a), direction="right minus left")
    for label, outcome in (
        ("strict", lambda row: bool(row["exact"])),
        ("leading_posthoc", lambda row: leading_integer(row["output_text"]) == row["gold"]),
    ):
        flags = [(outcome(x), outcome(y)) for x, y in zip(a, b)]
        result[label] = dict(
            exact_difference=mean(int(y) - int(x) for x, y in flags),
            right_wins=sum(y and not x for x, y in flags),
            left_wins=sum(x and not y for x, y in flags),
            both_correct=sum(x and y for x, y in flags),
            both_wrong=sum(not x and not y for x, y in flags),
        )
    for field in ("gold_first_token_nll", "seconds"):
        result[field + "_difference"] = mean(y[field] - x[field] for x, y in zip(a, b))
    result["parse_rate_difference"] = mean(
        int(y["prediction"] is not None) - int(x["prediction"] is not None) for x, y in zip(a, b))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    report = dict(
        status="Post-hoc diagnostic; registered primary metrics remain unchanged",
        notes=[
            "Leading integer ignores subsequent text; it does not verify the final answer or EOS.",
            "Unparsed-correct-leading marks formatting failure conditional on a correct leading answer.",
            "Leading correctness and first-token NLL separate some stopping effects from answer selection.",
            "Latency includes original startup/warm-up effects and varies with generated token count.",
            "Paired differences are descriptive; no uncertainty/significance claim is made.",
        ], runs=[], comparisons=[])
    group_sets = []
    for path in args.runs:
        summary = json.loads((path / "summary.json").read_text())
        rows = json.loads((path / "predictions.json").read_text())
        groups = defaultdict(list)
        for row in rows:
            if row["tag"] != "test":
                raise ValueError("Expected test-only predictions")
            mode, n = row["mode"], row["n_frames"]
            groups[(mode, str(n))].append(row)
            groups[(mode, "in_range_N8_32" if n in (8, 32) else "out_of_range_N64_128"
                    if n in (64, 128) else "other_lengths")].append(row)
        report["runs"].append(dict(
            path=str(path), run_id=summary["run_id"], arm=summary["arm"],
            parameters=summary["parameters"],
            metrics=[dict(mode=mode, lengths=lengths, **summarize(group))
                     for (mode, lengths), group in sorted(groups.items())]))
        group_sets.append((summary["run_id"], groups))
    for (left_id, left), (right_id, right) in combinations(group_sets, 2):
        for mode, lengths in sorted(left.keys() & right.keys()):
            report["comparisons"].append(dict(
                left=left_id, right=right_id, mode=mode, lengths=lengths,
                **paired(left[(mode, lengths)], right[(mode, lengths)])))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
