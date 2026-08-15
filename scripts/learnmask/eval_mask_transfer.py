#!/usr/bin/env python3
"""LEARNMASK zero-shot length transfer: assemble a trained relation×layer gate table
at arbitrary N and score EMITTED answers on the benchmark's own test splits.

Gate logits attach to (relation, Δ-bucket, layer) — no positions — so the SAME table
assembles at any sequence length by cell classification. This is each arm's transfer
step (campaign decision 2026-08-12: extrapolation folds into every arm; the S0
free-table is layout-bound and deliberately has no transfer path).

Regimes: hand (fence + R4/R7 open >= l_open) | init (pure fence) | nofence (plain
causal) | gates (a trained gates_*.pt via --gates). Scoring per sample: gold <= 9 ->
one-forward 10-way class read; gold > 9 -> greedy digit-sequence decode. Masks are
assembled per layer ON DEVICE from the int16 cell map (hard_mask_lut) — at N=64 one
fp32 mask is ~1.8 GB, so nothing ever holds 28 of them.

Usage:
  python scripts/learnmask/eval_mask_transfer.py \
      --gates outputs/learnmask/<run>/gates_best.pt \
      --dirs data/mmred_hf/dirs/seq_len_32_test=100,data/mmred_hf/dirs/seq_len_64_headfit=60 \
      --output outputs/learnmask/transfer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.constants import L_OPEN
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample
from gnnformer.engine import CarrierEngine
from gnnformer.learnmask import (
    N_CH,
    MaskGates,
    arm_learn_mask,
    fence_open_table,
    gated_greedy_digits,
    gated_stack_logits,
    hand_open_table,
    hard_mask_lut,
    prepare_sample_replicas,
    readers_of,
    relation_cell_map,
)
from gnnformer.mmred_hf import qtype_from_dirname
from gnnformer.runtime import get_layers, load_runtime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True,
                    help="comma-separated MMReD-HF dirs roots, each may carry =LIMIT")
    ap.add_argument("--limit", type=int, default=100, help="default per-root cap")
    ap.add_argument("--gates", default="", help="gates_*.pt for the 'gates' regime")
    ap.add_argument("--regimes", default="hand,init,nofence",
                    help="comma list of hand|init|nofence|gates ('gates' auto-added "
                         "when --gates is set)")
    ap.add_argument("--scaffold", choices=("replica", "carrier"), default="replica")
    ap.add_argument("--carrier-ckpt", default="checkpoints/carrier_layer_digit_p7a_lora_best.pt",
                    help="carrier scaffold only (frozen e_c+LoRA)")
    ap.add_argument("--readout-ckpt", default="",
                    help="replica scaffold: readout_*.pt from --train lora — LoRA "
                         "attached FROZEN (adds the s2open regime = its native mask)")
    ap.add_argument("--qtype-filter", default="steps_in_room",
                    help="only dirs whose qtype matches (empty = all counting qtypes)")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=6)
    ap.add_argument("--l-open", type=int, default=L_OPEN)
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/learnmask/transfer")
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    layers = get_layers(rt.model)
    tok = rt.tokenizer
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    lora = None
    if args.scaffold == "carrier":
        ck = load_carrier_layer_ckpt(Path(args.carrier_ckpt))
        e_c = ck.e_c.float().to(rt.device)
        lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha,
                           device=rt.device, state=ck.lora_state)
        for p in lora.parameters():
            p.requires_grad_(False)
    else:
        e_c = None
        if args.readout_ckpt:
            rck = torch.load(args.readout_ckpt, map_location="cpu")
            lora = attach_lora(layers, int(rck.get("l_open", 0)),
                               rank=int(rck["rank"]), alpha=float(rck["alpha"]),
                               device=rt.device, state=rck["lora"])
            for p in lora.parameters():
                p.requires_grad_(False)
            print(f"[readout] frozen LoRA from {args.readout_ckpt} "
                  f"(regime={rck.get('fixed_regime')} acc={rck.get('acc')})", flush=True)
    eng = CarrierEngine(rt, l_open=args.l_open, e_c=e_c)
    digit_ids = eng.digit_ids

    # '+' also separates: comma lists cannot ride sbatch --export values
    regimes = [r.strip() for r in args.regimes.replace("+", ",").split(",") if r.strip()]
    known = {"hand", "init", "nofence", "s2open", "gates"}
    bad = [r for r in regimes if r not in known]
    if bad:
        raise SystemExit(f"unknown regimes {bad} (known: {sorted(known)})")
    gates = None
    if args.gates:
        st = torch.load(args.gates, map_location="cpu")
        gates = MaskGates.from_state(st).to(rt.device)
        if "gates" not in regimes:
            regimes.append("gates")
        print(f"[gates] {args.gates} arm={st['arm']} est={st['estimator']} "
              f"epoch={st.get('epoch')} train_acc={st.get('acc')}", flush=True)

    def table(regime: str) -> torch.Tensor:
        if regime == "hand":
            return hand_open_table(eng.n_layers, args.l_open)
        if regime == "init":
            return hand_open_table(eng.n_layers, eng.n_layers + 1)  # fence everywhere
        if regime == "nofence":
            return torch.ones(N_CH, eng.n_layers, dtype=torch.bool)
        if regime == "s2open":  # everything the S2 sweep can reach, open
            t = fence_open_table(eng.n_layers)
            t[arm_learn_mask("s2")] = True
            return t
        return gates.hard_open_table()

    lines = [f"=== LEARNMASK TRANSFER (scaffold={args.scaffold} l_open={args.l_open} "
             f"gates={args.gates or '-'} regimes={regimes}) ==="]
    results = ["root,regime,n,em,mae,class_acc_le9"]
    for spec in args.dirs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        lim = args.limit
        if "=" in spec:
            spec, lim = spec.rsplit("=", 1)
            lim = int(lim)
        recs = []
        n_skip = 0
        t0 = time.time()
        for sd in iter_sample_dirs_shuffled(Path(spec), args.shuffle_dirs):
            if len(recs) >= lim:
                break
            qtype = qtype_from_dirname(sd.name)
            if qtype is None or (args.qtype_filter and qtype != args.qtype_filter):
                continue
            try:
                _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
            except Exception:
                n_skip += 1
                continue
            gold_s = str(a0).strip()
            if not gold_s.isdigit():
                n_skip += 1
                continue
            if args.scaffold == "replica":
                rec = prepare_sample_replicas(eng, frames, q0, gold=int(gold_s),
                                              task=qtype, resize=args.resize)
            else:
                rec = eng.prepare_sample(frames, q0, gold=int(gold_s), task=qtype,
                                         resize=args.resize)
            if rec is None:
                n_skip += 1
                continue
            recs.append({"emb": rec["emb"].cpu(), "pos": rec["pos"].cpu(),
                         "blocks": rec["blocks"], "fin": rec["fin"],
                         "seq": rec["seq"], "gold": int(gold_s),
                         "cpos": rec.get("cpos", []),
                         **({"readers": rec["readers"]} if "readers" in rec else {}),
                         "sd": str(sd)})
        dist = Counter(r["gold"] for r in recs)
        ln = (f"[root] {spec}: n={len(recs)} skip={n_skip} ({time.time()-t0:.0f}s) "
              f"dist={dict(sorted(dist.items()))}")
        print(ln, flush=True)
        lines.append(ln)

        for regime in regimes:
            tb = table(regime)
            em = n_cls = cls_ok = 0
            mae = 0.0
            per_gold: dict = {}
            with torch.no_grad():
                for d in recs:
                    e_max = 0 if d["gold"] <= 9 else args.decode_tokens
                    cm = relation_cell_map(d["seq"], d["blocks"], readers_of(d),
                                           d["fin"], e=e_max).to(eng.dev)

                    def mfn_full(li: int) -> torch.Tensor:
                        return hard_mask_lut(cm, tb[:, li])

                    if d["gold"] <= 9:
                        lg = gated_stack_logits(
                            eng, d, [], lambda li, S: mfn_full(li))[-1]
                        pred = int(lg[digit_ids].argmax())   # class read (diagnostic)
                        cls_ok += pred == d["gold"]
                        n_cls += 1
                        # EM = UNRESTRICTED emission (restricted argmax would credit
                        # samples whose true argmax is a non-digit token)
                        ok = tok.decode([int(lg.argmax())]).strip() == str(d["gold"])
                    else:
                        parsed, _fd, _txt = gated_greedy_digits(
                            eng, d, mfn_full, max_tokens=args.decode_tokens)
                        pred = parsed if parsed is not None else -1
                        ok = pred == d["gold"]
                    em += ok
                    mae += abs(pred - d["gold"])
                    pg = per_gold.setdefault(d["gold"], [0, 0])
                    pg[1] += 1
                    pg[0] += ok
            n = len(recs)
            pgs = " ".join(f"{g}:{c}/{m}" for g, (c, m) in sorted(per_gold.items()))
            ln = (f"[{Path(spec).name} | {regime}] em {em/n:.3f} mae {mae/n:.2f} "
                  f"class_acc(<=9) {cls_ok/max(n_cls,1):.3f} (n_cls={n_cls}) [{pgs}]")
            print(ln, flush=True)
            lines.append(ln)
            results.append(f"{Path(spec).name},{regime},{n},{em/n:.4f},{mae/n:.3f},"
                           f"{cls_ok/max(n_cls,1):.4f}")
            (out / "report.txt").write_text("\n".join(lines) + "\n")
            (out / "results.csv").write_text("\n".join(results) + "\n")
    print("wrote", out)
    if lora is not None:
        lora.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
