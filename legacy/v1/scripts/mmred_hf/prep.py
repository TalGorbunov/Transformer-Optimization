#!/usr/bin/env python
"""Convert the HF `ef1e43ce/mmred` dataset into upstream-renderer JSON files.

Runs under `.venv_mmred` (python 3.11, has `datasets` + upstream `mmred`), NOT the
shared `.venv`. The HF rows pack the step context into the `question` column (one
python-dict line per step, actual question as the last line); this script splits
that back out into the upstream `Sample`-style JSON that
`data/mmred_hf/upstream_repo/scripts/render_images.py` consumes.

Output: data/mmred_hf/json/<config>_<split>.json — a list of
  {qid, seq_len, qtype, atype, question, answer, sequence:[{step_id, rooms}]}

Usage (from repo root):
  .venv_mmred/bin/python scripts/mmred_hf_prep.py --config seq_len_8 --split test
  .venv_mmred/bin/python scripts/mmred_hf_prep.py --config seq_len_8 --split test \
      --limit 20 --out data/mmred_hf/json_probe
"""
import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HF_CACHE = REPO_ROOT / "data/mmred_hf/hf"


def parse_question_field(q: str):
    """HF `question` column -> (sequence[list of {step_id, rooms}], question_text)."""
    seq, question = [], None
    for ln in q.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            d = ast.literal_eval(s)
            seq.append({"step_id": int(d["step_id"]),
                        "rooms": {r: list(c) for r, c in d["rooms"].items()}})
        else:
            question = s
    if question is None or not seq:
        raise ValueError(f"unparseable question field: {q[:120]!r}")
    return seq, question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qtypes", nargs="+", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/mmred_hf/json"))
    args = ap.parse_args()

    ds = load_dataset("ef1e43ce/mmred", args.config, cache_dir=str(HF_CACHE))[args.split]
    out = []
    for row in ds:
        if args.qtypes and row["qtype"] not in args.qtypes:
            continue
        seq, qtext = parse_question_field(row["question"])
        assert len(seq) == row["seq_len"], (row["qid"], len(seq), row["seq_len"])
        assert [s["step_id"] for s in seq] == list(range(1, row["seq_len"] + 1)), row["qid"]
        out.append({"qid": row["qid"], "seq_len": row["seq_len"], "qtype": row["qtype"],
                    "atype": row["atype"], "question": qtext, "answer": row["answer"],
                    "sequence": seq})
        if args.limit and len(out) >= args.limit:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.config}_{args.split}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {len(out)} samples -> {out_path}")


if __name__ == "__main__":
    main()
