#!/usr/bin/env python3
"""Text-addition sanity: which pass-2 template makes the model add two digits?

TP_pred in repeater4c scored 0.213 despite sum2=0.780 (conditional ~0.27!?) —
template suspect. Sweep templates x force-prefixes on all digit pairs (a,b) with
a+b<=8, 4 repeats each, EM on first emitted digit (greedy 3 steps).
Usage: python scripts/superquery/probe_addsanity.py --output outputs/superquery/addsanity
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.runtime import dequantize_linear_weight, load_runtime, move_to_device  # noqa: E402

TEMPLATES = {
    "tp_v1": ("In the first half of a video, a person appears in the room {a} times. "
              "In the second half, {b} times. How many times in total?"),
    "plain": "What is {a} + {b}?",
    "counts": "Two partial counts are {a} and {b}. What is the total count?",
    "story": ("John was in the kitchen {a} times in the morning and {b} times in "
              "the evening. How many times was John in the kitchen in total?"),
}
FORCES = {"paren": "Answer: ( ", "plain": "Answer: ", "none": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    dev = model.device
    W_U = dequantize_linear_weight(model.lm_head).float().to(dev)
    final_norm = model.model.language_model.norm
    dig_ids = torch.tensor([tok(f"{k}", add_special_tokens=False).input_ids[0]
                            for k in range(10)], device=dev)
    dig_set = set(dig_ids.tolist())
    pairs = [(a, b) for a in range(5) for b in range(5) if a + b <= 8]

    rows = []
    for tname, tmpl in TEMPLATES.items():
        for fname, ftxt in FORCES.items():
            em = n = 0
            ctr = Counter()
            for a, b in pairs:
                for _rep in range(4):
                    q = tmpl.format(a=a, b=b)
                    it = processor.apply_chat_template(
                        [{"role": "user", "content": [{"type": "text", "text": q}]}],
                        add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt")
                    it = move_to_device(it, dev)
                    if ftxt:
                        f2 = torch.tensor([tok(ftxt, add_special_tokens=False).input_ids],
                                          device=dev)
                        it["input_ids"] = torch.cat([it["input_ids"], f2], 1)
                        it["attention_mask"] = torch.cat(
                            [it["attention_mask"], torch.ones_like(f2)], 1)
                    first_digit = None
                    with torch.no_grad():
                        for _s in range(4):
                            h = model(**it, output_hidden_states=True
                                      ).hidden_states[-1][0, -1]
                            lg = (final_norm(h.unsqueeze(0)).float() @ W_U.T)[0]
                            t = int(lg.argmax().item())
                            if t in dig_set:
                                first_digit = t
                                break
                            nt = torch.tensor([[t]], device=dev)
                            it["input_ids"] = torch.cat([it["input_ids"], nt], 1)
                            it["attention_mask"] = torch.cat(
                                [it["attention_mask"], torch.ones_like(nt)], 1)
                    em += int(first_digit == int(dig_ids[a + b].item()))
                    n += 1
                    ctr[tok.decode([first_digit]) if first_digit else "none"] += 1
            rows.append([tname, fname, em / n, n])
            print(f"[add {tname}/{fname}] EM {em/n:.3f} (n={n}) "
                  f"tops: {ctr.most_common(4)}", flush=True)
    with open(out / "addsanity.csv", "w", newline="") as f:
        csv.writer(f).writerows([["template", "force", "em", "n"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
