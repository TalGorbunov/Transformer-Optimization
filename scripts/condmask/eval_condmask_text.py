#!/usr/bin/env python3
"""condmask length-transfer evaluator: EM per N under hard input-conditioned gates.

Per root (one per N): shuffled slice (repo law), deterministic gates (logit > 0,
K = MASK_MIN), FREE greedy decode (recompute per step, stop at first non-digit,
max 4 tokens — N=128 golds are 3 digits), reported against the per-N majority
baseline. Also logs the per-sample hard gate patterns (mean P(block), cross-sample
disagreement) so conditioning behavior at transfer lengths is documented.

Control rows without a checkpoint: --regime full | blockwise.

Run:
  HF_HOME=$HOME/.cache/huggingface python scripts/condmask/eval_condmask_text.py \\
      --ckpt outputs/condmask/<run>/condmask_best.pt \\
      --roots data/mmred_text_cond/seq_len_8/test=400,... --output outputs/condmask/eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gnnformer.carriers import attach_lora
from gnnformer.constants import MASK_MIN
from gatenet import GateNet, pool_embeddings
from modeling import (greedy_digit_decode, load_text_model, model_dims, pad_batch,
                      single_mask_builder)
from textdata import majority_baseline, prep_root

DEFAULT_ROOTS = ",".join(
    f"data/mmred_text_cond/seq_len_{n}/test=400" for n in (8, 16, 32, 64, 128))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--regime", default="", choices=["", "full", "blockwise"],
                    help="control row without a ckpt (frozen model, fixed mask)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--roots", default=DEFAULT_ROOTS,
                    help="comma list root[=LIMIT]")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="outputs/condmask/eval")
    args = ap.parse_args()
    assert bool(args.ckpt) != bool(args.regime), "exactly one of --ckpt / --regime"

    # timestamp alone collides when sbatch starts several evals in the same second
    # (measured 2026-08-18: 12 jobs -> 6 surviving dirs, results overwritten)
    import os
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
    model_name = ck["config"]["model"] if ck else args.model
    tok, model = load_text_model(model_name, device=args.device)
    L, H, D = model_dims(model)
    dtype = next(model.parameters()).dtype

    gatenet, lora = None, None
    tag = args.regime or "ckpt"
    if ck is not None:
        tag = f"arm{ck['arm']}"
        if ck.get("lora") is not None:
            lora = attach_lora(model.model.layers, 0, rank=ck["rank"],
                               alpha=ck["alpha"], device=args.device,
                               state=ck["lora"])
            for p in lora.parameters():
                p.requires_grad_(False)
        elif ck["config"].get("readout_ckpt"):
            # E/F fallback ckpts: the readout LoRA was frozen (not saved) — re-attach
            # it from the recorded arm-A checkpoint path.
            rck = torch.load(ck["config"]["readout_ckpt"], map_location="cpu")
            lora = attach_lora(model.model.layers, 0, rank=rck["rank"],
                               alpha=rck["alpha"], device=args.device,
                               state=rck["lora"])
            for p in lora.parameters():
                p.requires_grad_(False)
            print(f"frozen readout re-attached from {ck['config']['readout_ckpt']}")
        if ck.get("gatenet") is not None:
            c = ck["config"]
            gatenet = GateNet(D, L, H, hidden=c.get("hidden", 256),
                              init=("blockwise" if "blockwise" in str(c) else "random"),
                              init_bias=c.get("init_bias", 1.0),
                              seed=c.get("seed", 0),
                              norm_input=c.get("gate_norm_input", False)
                              ).to(args.device)
            gatenet.load_state_dict(ck["gatenet"])
            gatenet.eval()
    print(f"eval {tag}: model {model_name} ({L}x{H}), gates "
          f"{'ON' if gatenet else 'off'}, lora {'ON' if lora else 'off'}")

    embed = model.model.embed_tokens
    rows = []
    csv = ["root,tag,n,em,mae,class_acc_le9,majority,p_block_mean,gate_disagree"]
    for spec in [s for s in args.roots.replace("+", ",").split(",") if s]:
        root, _, lim = spec.partition("=")
        limit = int(lim) if lim else 400
        recs = prep_root(tok, Path(root), limit, seed=args.seed)
        if not recs:
            print(f"[skip] {root}: no samples")
            continue
        maj_cls, maj = majority_baseline([r["gold"] for r in recs])
        hits = mae = cls_hit = cls_n = 0
        patterns = []
        t0 = time.time()
        for r in recs:
            if gatenet is not None:
                with torch.no_grad():
                    ids, valid, _, _ = pad_batch([r], tok.pad_token_id or 0,
                                                 args.device)
                    g = (gatenet(pool_embeddings(embed(ids), valid))[0] > 0).float()
                patterns.append(g.cpu())
            else:
                # fixed-mask value: from the ckpt's ARM for ckpt evals (A=full 0.0,
                # B=blockwise 1.0 — evaluating B under full was a measured bug,
                # 2026-08-17), from --regime for no-ckpt control rows.
                if ck is not None:
                    gv = 1.0 if ck["arm"] == "B" else 0.0
                else:
                    gv = 1.0 if args.regime == "blockwise" else 0.0
                g = torch.full((L, 1), gv, device=args.device)
            builder = single_mask_builder(g, MASK_MIN, dtype, args.device)
            pred, first_digit, _ = greedy_digit_decode(
                model, tok, r, builder, max_new=args.max_new, device=args.device)
            hits += int(pred == r["gold"])
            mae += abs((pred if pred is not None else 0) - r["gold"])
            if r["gold"] <= 9:
                cls_hit += int(first_digit == r["gold"])
                cls_n += 1
        n = len(recs)
        p_block = float(torch.stack(patterns).mean()) if patterns else (
            1.0 if args.regime == "blockwise" else 0.0)
        disagree = 0.0
        if patterns:
            frac = torch.stack(patterns).mean(0)
            disagree = float(((frac > 0) & (frac < 1)).float().mean())
        em, mm = hits / n, mae / n
        ca = cls_hit / max(cls_n, 1)
        print(f"[{root}] n={n} EM {em:.3f} (majority {maj:.3f}, class {maj_cls}) "
              f"MAE {mm:.2f} class-acc(<=9) {ca:.3f} P(block) {p_block:.3f} "
              f"disagree {disagree:.3f} ({time.time() - t0:.0f}s)")
        csv.append(f"{root},{tag},{n},{em:.4f},{mm:.3f},{ca:.4f},{maj:.4f},"
                   f"{p_block:.4f},{disagree:.4f}")
        rows.append({"root": root, "em": em, "mae": mm, "class_acc": ca,
                     "majority": maj, "p_block": p_block, "disagree": disagree})
        (out / "results.csv").write_text("\n".join(csv) + "\n")
        (out / "report.txt").write_text(
            f"condmask eval tag={tag} ckpt={args.ckpt or '-'} regime="
            f"{args.regime or '-'}\n" + json.dumps(rows, indent=2) + "\n")
    if lora is not None:
        lora.remove()
    print(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
