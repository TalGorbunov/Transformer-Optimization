#!/usr/bin/env python3
"""Text-counting capacity curve: can the model SUM N pasted clean symbols by itself?

Pure text, full causal, native emission — the upper bound for 'paste the codes and
let the model aggregate at the last token' (Tal's question). Prompt: N space-
separated bits, k of them 1 (k uniform in 0..min(9,N) so the answer is always one
digit token), forced assistant prefix "Answer: (" -> EM on the emitted token.

Rows per N in {4,8,16,32,64,128}: emit-EM, restricted-EM, MAE, top1-digit-frac.
Usage: python scripts/superquery/probe_textcount.py --output outputs/superquery/textcount
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

NS = (4, 8, 16, 32, 64, 128)
N_PER = 120


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
    force_ids = tok("Answer: ( ", add_special_tokens=False).input_ids

    rng = np.random.default_rng(0)
    rows = []
    for N in NS:
        em = em_r = digf = 0
        mae = []
        ctr = Counter()
        for i in range(N_PER):
            k = int(rng.integers(0, min(9, N) + 1))
            bits = np.zeros(N, dtype=int)
            bits[rng.choice(N, size=k, replace=False)] = 1
            q = ("Here is a list of bits: " + " ".join(map(str, bits.tolist())) +
                 ". How many of the bits are 1?")
            it = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": q}]}],
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            it = move_to_device(it, dev)
            force = torch.tensor([force_ids], device=dev)
            it["input_ids"] = torch.cat([it["input_ids"], force], 1)
            it["attention_mask"] = torch.cat([it["attention_mask"],
                                              torch.ones_like(force)], 1)
            with torch.no_grad():
                # greedy decode up to 3 steps; score the FIRST emitted digit
                top1 = None
                first_digit = None
                for _step in range(3):
                    h = model(**it, output_hidden_states=True).hidden_states[-1][0, -1]
                    lg = (final_norm(h.unsqueeze(0)).float() @ W_U.T)[0]
                    t = int(lg.argmax().item())
                    if top1 is None:
                        top1 = t
                        pr = int(lg[dig_ids].argmax().item())
                    if t in dig_set:
                        first_digit = t
                        pr = int(lg[dig_ids].argmax().item())
                        break
                    nt = torch.tensor([[t]], device=dev)
                    it["input_ids"] = torch.cat([it["input_ids"], nt], 1)
                    it["attention_mask"] = torch.cat(
                        [it["attention_mask"], torch.ones_like(nt)], 1)
            em += int(first_digit == int(dig_ids[k].item()))
            em_r += int(pr == k)
            mae.append(abs(pr - k))
            digf += int(top1 in dig_set)
            ctr[tok.decode([top1])] += 1
        rows.append([N, em / N_PER, em_r / N_PER, float(np.mean(mae)), digf / N_PER])
        print(f"[textcount N={N}] EM {em/N_PER:.3f} restricted {em_r/N_PER:.3f} "
              f"MAE {np.mean(mae):.2f} top1-digit {digf/N_PER:.3f} "
              f"tops: {ctr.most_common(4)}", flush=True)
    with open(out / "textcount.csv", "w", newline="") as f:
        csv.writer(f).writerows([["N", "em", "em_restricted", "mae",
                                  "top1_digit_frac"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
