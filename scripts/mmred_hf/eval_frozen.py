#!/usr/bin/env python3
"""Arm A: frozen Qwen2.5-VL on the ORIGINAL MMReD benchmark (their renders, their protocol).

Mirrors the paper's image-modality setup (upstream openai_server_inference.py):
system prompt asking for JSON {"answer": <value>} with the closed answer vocabulary
(room name / number / person name or Nobody), frames as images, question last.
Greedy decode; parse the JSON (fallbacks: Answer: X, bare value); exact match vs gold.
Reports per-qtype accuracy, per-atype accuracy, and parse-fail rate.

Runs in the SHARED .venv (GPU). Usage:
  python scripts/mmred_hf/eval_frozen.py \
      --json data/mmred_hf/json/seq_len_8_test.json \
      --images data/mmred_hf/images/seq_len_8_test \
      --qtypes steps_in_room final_app where_spend --limit-per-qtype 34 \
      --output outputs/mmred_hf/frozen_anchor
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.mmred_hf import load_index, load_mmred_hf_sample  # noqa: E402
from gnnformer.runtime import load_runtime, move_to_device  # noqa: E402

# upstream SYSTEM_PROMPT, verbatim (incl. the masked-room clause, unused in our data,
# for protocol fidelity)
SYSTEM_PROMPT = """You are an assistant that analyzes sequences of human agents moving in an environment.
If room contains a ["?"], it's masked and you should infer information from surrounding elements of sequence.
Format your response as a following json:
{ "answer": <value> }

Where <value> is:
- A **single room name** (e.g., "Kitchen") for location answers.
- A **number** (e.g., "3") for counting answers.
- A **single person name** (e.g., "Michael") for people answers or "Nobody" if no person satisfies given conditions."""

ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
CHARS = ["Sandra", "Mary", "John", "Daniel", "Michael", "Nobody"]


def parse_answer(text: str):
    """-> (value or None, parse_ok). JSON first, then Answer: X, then bare value."""
    m = re.search(r'"answer"\s*:\s*"?([A-Za-z0-9]+)"?', text)
    if m:
        return m.group(1), True
    m = re.search(r"[Aa]nswer\s*[:=]\s*\"?([A-Za-z0-9]+)", text)
    if m:
        return m.group(1), True
    for w in ROOMS + CHARS:  # bare closed-vocab word
        if re.search(rf"\b{w}\b", text):
            return w, False
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return m.group(1), False
    return None, False


def norm(v) -> str:
    s = str(v).strip().strip('"').strip("'").lower()
    return str(int(s)) if s.isdigit() else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--qtypes", nargs="+", default=None, help="default: all 24")
    ap.add_argument("--limit-per-qtype", type=int, default=50)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/mmred_hf/frozen")
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor = rt.model, rt.processor

    rows = load_index(Path(args.json))
    images_root = Path(args.images)
    by_qtype: dict = defaultdict(list)
    for r in rows:
        if args.qtypes and r["qtype"] not in args.qtypes:
            continue
        if len(by_qtype[r["qtype"]]) < args.limit_per_qtype:
            by_qtype[r["qtype"]].append(r)

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    acc = defaultdict(lambda: [0, 0])   # qtype -> [hits, n]
    acc_atype = defaultdict(lambda: [0, 0])
    n_parse_fail = n_skip = 0
    records = []
    t0 = time.time()
    for qtype, qrows in sorted(by_qtype.items()):
        for row in qrows:
            try:
                qid, frames, q0, states, gold = load_mmred_hf_sample(row, images_root)
            except FileNotFoundError:
                n_skip += 1
                continue
            if args.resize > 0:
                frames = [f.resize((args.resize, args.resize)) for f in frames]
            content = [{"type": "image", "image": f} for f in frames]
            content.append({"type": "text", "text": f"Question: {q0}"})
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, rt.device)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.decode,
                                     do_sample=False,
                                     pad_token_id=rt.tokenizer.eos_token_id)
            text = rt.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            pred, ok = parse_answer(text)
            n_parse_fail += not ok
            hit = pred is not None and norm(pred) == norm(gold)
            acc[qtype][0] += hit
            acc[qtype][1] += 1
            acc_atype[row["atype"]][0] += hit
            acc_atype[row["atype"]][1] += 1
            records.append({"qid": qid, "qtype": qtype, "atype": row["atype"],
                            "gold": gold, "pred": pred, "parse_ok": int(ok),
                            "hit": int(hit), "raw": text[:120]})
        h, n = acc[qtype]
        print(f"[{qtype}] acc {h}/{n} = {h/max(n,1):.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    n_all = sum(n for _, n in acc.values())
    h_all = sum(h for h, _ in acc.values())
    lines = [f"MMRED-HF FROZEN (arm A), json={args.json}, n={n_all}, skip={n_skip}, "
             f"parse_fail={n_parse_fail} ({n_parse_fail/max(n_all,1):.3f})",
             f"OVERALL acc {h_all}/{n_all} = {h_all/max(n_all,1):.3f}"]
    lines += [f"  {qt}: {h}/{n} = {h/max(n,1):.3f}" for qt, (h, n) in sorted(acc.items())]
    lines += [f"  atype {at}: {h}/{n} = {h/max(n,1):.3f}"
              for at, (h, n) in sorted(acc_atype.items())]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    with open(out / "per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    print(report)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
