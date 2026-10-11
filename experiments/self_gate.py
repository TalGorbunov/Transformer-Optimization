#!/usr/bin/env python3
"""Can the frozen model gate itself? Label-free, within-question rules on the D regime's own scores (CPU).

Reads units.csv of a unit_judge run: one row per unit (the k positives plus the stored hard and easy
negatives of each question), `score` = log P(yes) - log P(no) at the first answer token. Every rule is
applied per question and uses no labels:
  yes          keep units with score > 0 (the model's literal "yes" = one fixed global threshold)
  max-d        keep every unit whose score is within d of the question's best unit (log-odds)
  top-k        keep the k best units, k = the true needle count (NOT label-free: the ranking ceiling)
Per task and pooled: recall (positives kept), rejection (negatives dropped) and the exact-set rate
(every positive kept AND every negative dropped). The negatives are the stored ~2 per positive, not
the N - k haystack frames of a real read, so rejection is measured on the hardest few only.
  python experiments/self_gate.py outputs/scaleup/vnbench/qwen2.5-vl-7b/s1u_unit/D_frame_lo/<run> [--deltas 1 2 4] [--out X.md]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

DEFAULT_DELTAS = (1.0, 2.0, 4.0)  # log-odds windows below the question's best unit for the max-d rules
YES_THRESHOLD = 0.0  # log P(yes) - log P(no) > 0 is the model's literal "yes"
POOLED = "pooled"  # the extra table column that tallies every task together
REQUIRED_COLUMNS = ("qid", "qtype", "kind", "score")
MISSING_SCORES = ("", "nan")  # unit_judge writes these when the judge pass produced no score for a unit
CELL_LEGEND = ("cell = recall / rejection / exact-set rate; recall = positives kept, "
               "rejection = stored negatives dropped, "
               "exact = every positive kept and every negative dropped (per question).")


@dataclass(frozen=True)
class Unit:
    score: float  # log P(yes) - log P(no) at the first answer token
    is_positive: bool  # kind == "pos"


Rule = Callable[[List[Unit]], List[bool]]  # the units of one question -> which to keep


@dataclass
class Tally:
    """The counts behind one table cell, summed over the questions of one task (or all of them)."""
    kept_positives: int = 0
    positives: int = 0
    dropped_negatives: int = 0
    negatives: int = 0
    exact_questions: int = 0
    questions: int = 0

    def add_question(self, units: List[Unit], keep: List[bool]) -> None:
        positives = sum(unit.is_positive for unit in units)
        kept_positives = sum(kept and unit.is_positive for kept, unit in zip(keep, units))
        negatives = len(units) - positives
        dropped_negatives = sum((not kept) and (not unit.is_positive) for kept, unit in zip(keep, units))
        self.kept_positives += kept_positives
        self.positives += positives
        self.dropped_negatives += dropped_negatives
        self.negatives += negatives
        self.exact_questions += int(kept_positives == positives and dropped_negatives == negatives)
        self.questions += 1

    def cell(self) -> str:
        recall = self.kept_positives / self.positives
        rejection = self.dropped_negatives / self.negatives
        exact_rate = self.exact_questions / self.questions
        return f"{recall:.2f} / {rejection:.2f} / {exact_rate:.2f}"


def rule_yes(units: List[Unit]) -> List[bool]:
    return [unit.score > YES_THRESHOLD for unit in units]


def make_rule_max_delta(delta: float) -> Rule:
    def rule_max_delta(units: List[Unit]) -> List[bool]:
        best = max(unit.score for unit in units)
        return [unit.score >= best - delta for unit in units]

    return rule_max_delta


def rule_top_k(units: List[Unit]) -> List[bool]:
    """Keep the k best-scoring units with k = the question's true positive count: the ranking ceiling, not a gate."""
    k = sum(unit.is_positive for unit in units)
    by_score_descending = sorted(range(len(units)), key=lambda i: -units[i].score)  # stable: ties keep file order
    kept = set(by_score_descending[:k])
    return [i in kept for i in range(len(units))]


def rules(deltas: Sequence[float]) -> Dict[str, Rule]:
    out: Dict[str, Rule] = {"yes": rule_yes}
    for delta in deltas:
        out[f"max-{delta:g}"] = make_rule_max_delta(delta)
    out["top-k (oracle k)"] = rule_top_k
    return out


def load_units(run_dir: Path) -> Tuple[Dict[str, List[Unit]], Dict[str, str]]:
    """units.csv of a unit_judge run -> (units per qid, qtype per qid). Rows without a score are skipped."""
    path = run_dir / "units.csv"
    if not path.is_file():
        raise SystemExit(f"{path}: not found (expected a unit_judge run dir)")
    units_by_qid: Dict[str, List[Unit]] = defaultdict(list)
    qtype_by_qid: Dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        for row in reader:
            if row["score"] in MISSING_SCORES:
                continue
            units_by_qid[row["qid"]].append(Unit(score=float(row["score"]), is_positive=row["kind"] == "pos"))
            qtype_by_qid[row["qid"]] = row["qtype"]
    return dict(units_by_qid), qtype_by_qid


def tally_rule(rule: Rule, units_by_qid: Dict[str, List[Unit]], qtype_by_qid: Dict[str, str]) -> Dict[str, Tally]:
    """Apply one rule to every question; one Tally per qtype plus one under POOLED."""
    tallies: Dict[str, Tally] = defaultdict(Tally)
    for qid, units in units_by_qid.items():
        keep = rule(units)
        for key in (qtype_by_qid[qid], POOLED):
            tallies[key].add_question(units, keep)
    return tallies


def format_table(run_dir: Path, units_by_qid: Dict[str, List[Unit]], qtype_by_qid: Dict[str, str],
                 rule_set: Dict[str, Rule]) -> str:
    columns = sorted(set(qtype_by_qid.values())) + [POOLED]
    n_units = sum(len(units) for units in units_by_qid.values())
    lines = [
        f"# self-gate on `{run_dir}` ({len(units_by_qid)} questions, {n_units} units)",
        "",
        "| rule | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name, rule in rule_set.items():
        tallies = tally_rule(rule, units_by_qid, qtype_by_qid)
        lines.append(f"| {name} | " + " | ".join(tallies[key].cell() for key in columns) + " |")
    lines += ["", CELL_LEGEND]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="unit_judge run dir (has units.csv)")
    parser.add_argument("--deltas", nargs="+", type=float, default=list(DEFAULT_DELTAS),
                        help="max-d windows (log-odds)")
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this markdown file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    units_by_qid, qtype_by_qid = load_units(args.run)
    table = format_table(args.run, units_by_qid, qtype_by_qid, rules(args.deltas))
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
