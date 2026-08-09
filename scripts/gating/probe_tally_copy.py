#!/usr/bin/env python3
"""Is teacher-forced COUNT accuracy measuring counting, or copying?

The caption scratchpad carries a RUNNING TALLY, and the final answer is always a verbatim
copy of the last tally value already present in the transcript:

    scan: f1:- f2:Kitchen(1) f3:Kitchen(2) ... f7:Kitchen(5) f8:- | total: 5 END
                                        ^^^                            ^ same number

Under teacher forcing the model is FED that transcript and only predicts ` G END`, so a
pure copier scores 1.000 without doing any aggregation. That would make `tf_acc` — the
metric the whole gating campaign leans on, and the trainer's headline `acc` — close to
uninformative, and would explain both its saturation and the surprising 0.965 the LoRA
control scored on the held-out N=128 root.

The decisive test: shift every running tally by a constant DELTA while leaving the gold
`total:` untouched, then teacher-force that corrupted transcript and read the answer slot.

    predicts gold          -> the model is COUNTING (it ignored the corrupted tally)
    predicts gold + DELTA  -> the model is COPYING  (it followed the tally)

Usage:
  python scripts/gating/probe_tally_copy.py --ckpt <carrier_layer_best.pt> \
      --roots data/mmred_images_park/seq_len_8/all_uniform \
              data/mmred_longN_park/seq_len_128/all_uniform \
      --limit 40 --delta 3 --output outputs/gating/p35_tallycopy/<stamp>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.data import (
    frame_attr_labels,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
)
from gnnformer.engine import CarrierEngine
from gnnformer.gating import attach_gate
from gnnformer.runtime import attention_dims, get_layers, load_runtime
from gnnformer.scratchpad import build_target_fmt

_TALLY = re.compile(r"\((\d+)\)")


def shift_tally(target: str, delta: int) -> str:
    """Add `delta` to every running-tally value, leaving the final `total: G` alone."""
    head, sep, tail = target.rpartition("| total:")
    return _TALLY.sub(lambda m: f"({int(m.group(1)) + delta})", head) + sep + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--delta", type=int, default=3)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    layers, tok, dev = get_layers(rt.model), rt.tokenizer, rt.device
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = load_carrier_layer_ckpt(Path(args.ckpt))
    e_c = torch.as_tensor(ck.e_c).to(dev)
    lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha, device=dev,
                       state=ck.lora_state)
    gate = None
    gs = ck.extra.get("gate")
    if gs is not None:
        d = attention_dims(rt.model)
        gate = attach_gate(layers, gs["layer_ids"], gs["variant"], hidden=d["hidden_size"],
                           n_heads=d["n_heads"], n_kv=d["n_kv"], head_dim=d["head_dim"],
                           device=dev, state=gs)
    eng = CarrierEngine(rt, l_open=ck.l_open, e_c=e_c)
    print(f"[ckpt] {args.ckpt} L*={ck.l_open} fmt={ck.scratchpad_format} "
          f"gate={gs['variant'] if gs else 'none'}", flush=True)

    def answer_of(rec, tgt_str: str):
        """Teacher-force `tgt_str`, return the integer the model predicts in the answer slot."""
        ids = tok(tgt_str, add_special_tokens=False).input_ids + [tok.eos_token_id]
        d = eng.build_training_cache(rec, ids)
        ak = len(tok(" " + tgt_str.rsplit(": ", 1)[-1], add_special_tokens=False).input_ids) + 1
        with torch.no_grad():
            hs = eng.top_hidden(d)
            lg = eng.head(hs[0, d["seq"] - 1 : d["seq"] + d["e"] - 1])
        txt = tok.decode(lg.argmax(-1).tolist()[-ak:-1]).strip()
        m = re.match(r"-?\d+", txt)
        return (int(m.group(0)) if m else None), txt

    rows: List[Dict[str, Any]] = []
    for root in args.roots:
        n = cnt_gold = cnt_copy = clean_ok = 0
        t0 = time.time()
        for sd in iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs):
            if n >= args.limit:
                break
            try:
                _s, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
            except Exception:
                continue
            pa = parse_task_labels(q0, states, gold)
            if pa is None:
                continue
            task, evid, aux = pa
            tgt = build_target_fmt("caption", task, evid, aux, gold, NF=len(frames),
                                   labels=frame_attr_labels(task, q0, states, evid))
            if not _TALLY.search(tgt):
                continue  # gold==0 has no tally at all -> nothing to corrupt
            rec = eng.prepare_sample(frames, q0, gold=gold, task=task, resize=args.resize)
            if rec is None:
                continue
            clean, _ = answer_of(rec, tgt)
            shifted, stxt = answer_of(rec, shift_tally(tgt, args.delta))
            clean_ok += int(clean == gold)
            cnt_gold += int(shifted == gold)
            cnt_copy += int(shifted == gold + args.delta)
            n += 1
            if n <= 3:
                print(f"  [{root}] gold={gold} clean_pred={clean} "
                      f"shifted_pred={shifted!r} (copy would be {gold+args.delta})", flush=True)
            if n % 10 == 0:
                print(f"  [{root}] {n}/{args.limit} ({time.time()-t0:.0f}s)", flush=True)
        if n == 0:
            continue
        rows.append({"root": root, "n": n, "clean_acc": clean_ok / n,
                     "shifted_pred_gold": cnt_gold / n, "shifted_pred_copy": cnt_copy / n,
                     "delta": args.delta})
        print(f"[root] {root}: n={n} clean {clean_ok/n:.3f} | after shifting the tally by "
              f"+{args.delta}: predicts GOLD {cnt_gold/n:.3f}, predicts COPY {cnt_copy/n:.3f}",
              flush=True)

    lora.remove()
    if gate is not None:
        gate.remove()
    (out / "tally_copy.json").write_text(json.dumps({"ckpt": args.ckpt, "rows": rows}, indent=2))
    lines = [f"=== TALLY-COPY PROBE (ckpt={args.ckpt}, delta=+{args.delta}) ===",
             "A COUNTER ignores the corrupted tally and still predicts gold.",
             "A COPIER follows the tally and predicts gold+delta.", "",
             f"{'root':<52} {'n':>4} {'clean':>7} {'->gold':>7} {'->copy':>7}"]
    for r in rows:
        lines.append(f"{r['root']:<52} {r['n']:>4} {r['clean_acc']:>7.3f} "
                     f"{r['shifted_pred_gold']:>7.3f} {r['shifted_pred_copy']:>7.3f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
