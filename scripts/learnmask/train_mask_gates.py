#!/usr/bin/env python3
"""LEARNMASK gate trainer: relation×layer mask gates from the fence init, DIRECT
answer-class CE, everything else frozen (outputs/learnmask/CAMPAIGN_BRIEF.md).

Scaffold (Tal, 2026-08-12: replicas first):
  --scaffold replica (default): per-frame QUESTION REPLICAS, no leading question —
      the A3 one-forward supply construction with ZERO trained components (no carrier
      token, no e_c, no LoRA). The gate logits are the only parameters in the entire
      system, so whatever topology training finds is untainted by components that were
      co-trained under the hand fence.
  --scaffold carrier: the deployed carrier stack (frozen e_c + LoRA from
      --carrier-ckpt, default = gating P7a digit LoRA control). Confirmation arm;
      note its LoRA was TRAINED under the hand fence at 392 (baked-in rediscovery
      bias + distribution shift — interpret against the replica line).

Metric policy (Tal, 2026-08-12): NO scratchpad, NO teacher-forced transcript metrics.
  --target class (default): loss/metric at the answer position (= last prompt row,
      no appended rows): CE over the 10 digit-token logits vs the gold class; class
      acc (0-9-restricted argmax); unrestricted-argmax EM as the mass-drift canary.
  --target digit: multi-digit generalization (CE on the digit sequence + greedy
      emitted EM) for large-N splits where counts exceed 9.

ep0 rows (before any training):
  handfence : fence + R4/R7 open at layers >= L_OPEN (the hand design)
  init      : fence at every layer (readers isolated; no aggregation)
  nofence   : everything open (plain causal — frozen model, no intervention)
Startup parity: handfence-as-gates vs CarrierEngine.forward_logits with the span
lo/hi injected via d["lo"]/d["hi"], same-shape forwards, last row re-headed 1-D
(kernel-exact; see jobs 131330/131332/131337 in outputs/learnmask/STATE.md).

Smoke:
  python scripts/learnmask/train_mask_gates.py --arm s1 --limit 120 --epochs 4 \
      --output outputs/learnmask/s1_replica_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.constants import L_OPEN, MASK_MIN
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample
from gnnformer.engine import CarrierEngine
from gnnformer.learnmask import (
    ARM_RELATIONS,
    ESTIMATORS,
    N_CH,
    FreeTableGates,
    MaskGates,
    fence_open,
    freetable_tf_logits,
    gated_greedy_digits,
    gated_stack_logits,
    gated_tf_logits,
    hand_open,
    hand_open_table,
    handfence_tf_logits,
    hard_mask,
    hard_masks_by_layer,
    layout_key,
    make_masks_spans,
    prepare_sample_replicas,
    readers_of,
    relation_cell_map,
)
from gnnformer.mmred_hf import qtype_from_dirname
from gnnformer.runtime import get_layers, load_runtime

DIGIT_CKPT = "checkpoints/carrier_layer_digit_p7a_lora_best.pt"


def class_dist_line(tag: str, golds) -> str:
    c = Counter(golds)
    tot = sum(c.values())
    top = " ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))
    maj = max(c.values()) / tot if tot else 0.0
    return f"[class-dist] {tag} n={tot} majority {maj:.2f} [{top}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_hf/dirs/seq_len_8_train_steps_in_room",
                    help="MMReD-HF materialized dirs; comma-separated roots, each may "
                         "carry a per-root cap as path=LIMIT (mixed-length training)")
    ap.add_argument("--eval-root", default="",
                    help="separate eval dirs root(s), comma-separated; "
                         "empty -> --train-frac split of --data_root")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--eval-limit", type=int, default=60, help="cap on --eval-root dirs")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--scaffold", choices=("replica", "carrier"), default="replica")
    ap.add_argument("--train", choices=("gates", "lora"), default="gates",
                    help="lora: train a FENCE-AGNOSTIC digit readout instead of gates "
                         "— fresh LoRA on ALL layers, mask PINNED at --fixed-regime "
                         "(the 2026-08-13 existence-test consequence: no mask makes "
                         "the frozen model emit counts; a readout must exist first)")
    ap.add_argument("--fixed-regime", choices=("nofence", "init", "hand"),
                    default="nofence",
                    help="--train lora: the pinned mask regime (nofence = the readout "
                         "never sees ANY fence — the clean scaffold for gate arms)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--readout-ckpt", default="",
                    help="gates mode, replica scaffold: readout_*.pt from a --train "
                         "lora run — LoRA attached FROZEN (the trained answer path)")
    ap.add_argument("--target", choices=("class", "digit"), default="class")
    ap.add_argument("--arm", choices=tuple(sorted(ARM_RELATIONS)) + ("s0",), default="s1",
                    help="s0 = per-cell free table (diagnostic; class target, one "
                         "fixed layout, never a headline)")
    ap.add_argument("--estimator", choices=ESTIMATORS, default="st-gumbel")
    ap.add_argument("--init-logit", type=float, default=2.0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--lam-open", type=float, default=1e-2,
                    help="deviation penalty on p_open of fence-OFF learnable gates")
    ap.add_argument("--lam-close", type=float, default=0.0,
                    help="S3: penalty on 1-p_open of fence-ON learnable gates")
    ap.add_argument("--lr-gate", type=float, default=3e-2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--resize", type=int, default=512, help="campaign spec: 512, not 392")
    ap.add_argument("--decode-tokens", type=int, default=6, help="digit-mode decode cap")
    ap.add_argument("--carrier-ckpt", default=DIGIT_CKPT,
                    help="carrier scaffold only: carrier-LAYER ckpt (e_c+LoRA, frozen)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--parity-samples", type=int, default=2,
                    help="startup check: handfence-as-gates vs engine.forward_logits")
    ap.add_argument("--ep0-regimes", default="",
                    help="extra ep0 reference rows, comma list of supply|handsupply "
                         "(supply = tail reads replicas NOT frames — the "
                         "readout-through-the-supply topology)")
    ap.add_argument("--tail-style", choices=("canonical", "plain"), default="canonical",
                    help="replica tail: canonical = data.build_count_prompt (the "
                         "PROMPT-CRITICAL wording that makes the frozen model emit "
                         "a digit); plain = bare final question (em 0.000 — ablation)")
    ap.add_argument("--answer-hint", default="",
                    help="replica scaffold: user-side sentence appended after the "
                         "final question (e.g. 'Answer with the number only.')")
    ap.add_argument("--answer-prime", default="",
                    help="replica scaffold: assistant-side prime appended after the "
                         "generation prompt (e.g. 'Answer:') — the read position "
                         "follows it")
    ap.add_argument("--qtype-filter", default="steps_in_room",
                    help="only dirs of this qtype (mixed test splits carry all 24; "
                         "empty = any numeric-answer qtype)")
    ap.add_argument("--shuffle-dirs", type=int, default=0, metavar="SEED",
                    help="stratified round-robin shuffle seed (K0-trap mitigation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/learnmask/train")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train == "lora" and (args.scaffold != "replica" or args.readout_ckpt):
        raise SystemExit("--train lora is replica-scaffold only and starts fresh "
                         "(no --readout-ckpt)")
    rt = load_runtime(args.model) if args.model else load_runtime()
    layers = get_layers(rt.model)
    tok = rt.tokenizer
    dev = rt.device
    tag = (f"lorareadout_{args.fixed_regime}" if args.train == "lora"
           else f"{args.arm}_{args.estimator}")
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S")
                               + f"_{args.scaffold}_{tag}_{args.target}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    lora = None
    if args.scaffold == "carrier":
        ck = load_carrier_layer_ckpt(Path(args.carrier_ckpt))
        if ck.pos_couple:
            raise SystemExit("pos_couple ckpts are not supported by the gated forward")
        e_c = ck.e_c.float().to(dev)
        lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha, device=dev,
                           state=ck.lora_state)
        for p in lora.parameters():
            p.requires_grad_(False)  # frozen at the canonical weights
        l_open_ref = ck.l_open
        print(f"[init] carrier scaffold: frozen e_c+LoRA from {args.carrier_ckpt} "
              f"(l_open={ck.l_open} r={ck.rank} acc={ck.acc})", flush=True)
    else:
        e_c = None
        l_open_ref = L_OPEN
        if args.readout_ckpt:
            rck = torch.load(args.readout_ckpt, map_location="cpu")
            lora = attach_lora(layers, int(rck.get("l_open", 0)),
                               rank=int(rck["rank"]), alpha=float(rck["alpha"]),
                               device=dev, state=rck["lora"])
            for p in lora.parameters():
                p.requires_grad_(False)
            print(f"[init] replica scaffold + FROZEN readout LoRA from "
                  f"{args.readout_ckpt} (fixed_regime={rck.get('fixed_regime')} "
                  f"acc={rck.get('acc')})", flush=True)
        else:
            print(f"[init] replica scaffold: ZERO trained components (no e_c, no "
                  f"LoRA); hand-design reference opens R4/R7 at layers >= "
                  f"{l_open_ref}", flush=True)
    eng = CarrierEngine(rt, l_open=l_open_ref, e_c=e_c)
    digit_ids = eng.digit_ids
    if args.arm == "s0" and args.target != "class":
        raise SystemExit("--arm s0 is class-target only (fixed layout, e=0)")

    # ---- prep: prompt geometry + embedded prompt per sample (emb cached on CPU) ----
    def prep_root(root: str, lim: int, train_side: bool):
        recs, n_skip = [], 0
        for sd in iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs):
            if len(recs) >= lim:
                break
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            except Exception:
                n_skip += 1
                continue
            qtype = qtype_from_dirname(sd.name)
            gold_s = str(a0).strip()
            if qtype is None or not gold_s.isdigit() or (
                    args.qtype_filter and qtype != args.qtype_filter):
                n_skip += 1
                continue
            gold = int(gold_s)
            if (train_side or args.arm == "s0") and args.target == "class" and gold > 9:
                n_skip += 1  # class CE is a 10-way single-token read; topology does
                continue     # not care about answer magnitude (eval handles >9;
                             # s0's fixed-layout masks cannot extend to decode rows)
            if args.scaffold == "replica":
                rec = prepare_sample_replicas(eng, frames, q0, gold=gold, task=qtype,
                                              resize=args.resize,
                                              tail_style=args.tail_style,
                                              answer_hint=args.answer_hint,
                                              answer_prime=args.answer_prime)
            else:
                rec = eng.prepare_sample(frames, q0, gold=gold, task=qtype,
                                         resize=args.resize)
            if rec is None:
                n_skip += 1
                continue
            tgt_ids = (tok(gold_s, add_special_tokens=False).input_ids
                       + [tok.eos_token_id]) if args.target == "digit" else []
            d = {"emb": rec["emb"].cpu(), "pos": rec["pos"].cpu(),
                 "blocks": rec["blocks"], "fin": rec["fin"], "seq": rec["seq"],
                 "gold": gold, "task": qtype, "tgt": tgt_ids, "sd": str(sd),
                 "cpos": rec.get("cpos", [])}
            if "readers" in rec:
                d["readers"] = rec["readers"]
            if not recs:
                print(f"[target-debug] qtype={qtype} gold={gold} target={args.target} "
                      f"scaffold={args.scaffold} seq={d['seq']} "
                      f"readers[0]={readers_of(d)[0]}", flush=True)
            recs.append(d)
        return recs, n_skip

    def prep_roots(spec: str, default_lim: int, train_side: bool):
        recs, n_skip = [], 0
        for root in spec.split(","):
            root = root.strip()
            if not root:
                continue
            lim = default_lim
            if "=" in root:
                root, lim = root.rsplit("=", 1)
                lim = int(lim)
            r, s = prep_root(root, lim, train_side)
            print(f"  [prep] {root}: {len(r)} samples (skip {s})", flush=True)
            recs.extend(r)
            n_skip += s
        return recs, n_skip

    t0 = time.time()
    data, n_skip = prep_roots(args.data_root, args.limit, train_side=True)
    if args.eval_root:
        tr = data
        ev, ev_skip = prep_roots(args.eval_root, args.eval_limit, train_side=False)
        n_skip += ev_skip
    else:
        order = np.random.default_rng(args.seed).permutation(len(data))
        n_tr = int(len(data) * args.train_frac)
        tr = [data[i] for i in order[:n_tr]]
        ev = [data[i] for i in order[n_tr:]]
    if not tr or not ev:
        raise SystemExit(f"empty split: train={len(tr)} eval={len(ev)} skip={n_skip}")
    print(f"prep done: train={len(tr)} eval={len(ev)} skip={n_skip} "
          f"({time.time()-t0:.0f}s)", flush=True)
    lines = [f"=== LEARNMASK GATE TRAINER (scaffold={args.scaffold} arm={args.arm} "
             f"est={args.estimator} target={args.target} init_logit={args.init_logit} "
             f"tau {args.tau0}->{args.tau1} lam_open={args.lam_open} "
             f"lam_close={args.lam_close} lr={args.lr_gate} resize={args.resize} "
             f"root={args.data_root} n_tr={len(tr)} n_ev={len(ev)}) ==="]
    for tag, split in (("train", tr), ("eval", ev)):
        ln = class_dist_line(tag, [d["gold"] for d in split])
        print(ln, flush=True)
        lines.append(ln)
    (out / "train_dirs.txt").write_text("\n".join(d["sd"] for d in tr) + "\n")
    (out / "eval_dirs.txt").write_text("\n".join(d["sd"] for d in ev) + "\n")

    # ---- gates (s0 needs ONE layout: keep the dominant one, report the drop) ----
    if args.arm == "s0":
        keys = Counter(layout_key(d) for d in tr)
        dom, _n = keys.most_common(1)[0]
        n_tr0, n_ev0 = len(tr), len(ev)
        tr = [d for d in tr if layout_key(d) == dom]
        ev = [d for d in ev if layout_key(d) == dom]
        if len(tr) < n_tr0 or len(ev) < n_ev0:
            ln = (f"[s0-layout] {len(keys)} distinct layouts (question token length "
                  f"varies); keeping dominant: train {len(tr)}/{n_tr0} "
                  f"eval {len(ev)}/{n_ev0}")
            print(ln, flush=True)
            lines.append(ln)
            for tag, split in (("train/s0", tr), ("eval/s0", ev)):
                ln = class_dist_line(tag, [d["gold"] for d in split])
                print(ln, flush=True)
                lines.append(ln)
            (out / "train_dirs.txt").write_text("\n".join(d["sd"] for d in tr) + "\n")
            (out / "eval_dirs.txt").write_text("\n".join(d["sd"] for d in ev) + "\n")
        if not tr or not ev:
            raise SystemExit(f"s0: dominant layout leaves train={len(tr)} "
                             f"eval={len(ev)} — pick roots with a shared layout")
        d00 = tr[0]
        cm00 = relation_cell_map(d00["seq"], d00["blocks"], readers_of(d00), d00["fin"])
        gates = FreeTableGates(cm00, eng.n_layers, estimator=args.estimator,
                               init_logit=args.init_logit).to(dev)
    else:
        gates = MaskGates(eng.n_layers, arm=args.arm, estimator=args.estimator,
                          init_logit=args.init_logit).to(dev)
    if args.train == "lora":
        lora = attach_lora(layers, 0, rank=args.rank, alpha=args.alpha, device=dev)
        opt = torch.optim.Adam([{"params": lora.parameters(), "lr": args.lr_lora}])
        print(f"[params] READOUT LoRA {lora.num_parameters()} (all {eng.n_layers} "
              f"layers, r={args.rank}, mask PINNED at {args.fixed_regime}); gates "
              f"unused; backbone frozen", flush=True)
    else:
        opt = torch.optim.Adam([{"params": [gates.logits], "lr": args.lr_gate}])
        print(f"[params] gates {gates.logits.numel()} ({gates.n_learn} ch/cells x "
              f"{eng.n_layers} layers, arm {args.arm}, est {args.estimator}, target "
              f"{args.target}, scaffold {args.scaffold}); everything else frozen",
              flush=True)

    # ---- CPU mask parity on the REAL layout (span construction vs gate assembly) ----
    d0 = tr[0]
    lo0, hi0 = make_masks_spans(d0["seq"], d0["blocks"], readers_of(d0), d0["fin"])
    cm0 = relation_cell_map(d0["seq"], d0["blocks"], readers_of(d0), d0["fin"])
    if not (torch.equal(hard_mask(cm0, fence_open()), lo0)
            and torch.equal(hard_mask(cm0, hand_open(l_open_ref, l_open_ref)), hi0)):
        raise SystemExit("MASK PARITY FAILURE on the real layout (vs make_masks_spans)")
    ln = "[mask-parity] real-layout lo/hi bit-for-bit OK"
    print(ln, flush=True)
    lines.append(ln)

    # ---- startup parity: same-shape forwards, last row re-headed 1-D; the engine
    # ---- gets the SAME span lo/hi via d["lo"]/d["hi"] (its lazy path is carrier-only)
    for d in tr[: args.parity_samples]:
        lo_s, hi_s = make_masks_spans(d["seq"], d["blocks"], readers_of(d), d["fin"])
        d2 = {**d, "lo": lo_s, "hi": hi_s}
        with torch.no_grad():
            lg_rows, hh = handfence_tf_logits(eng, d, [], l_open=l_open_ref,
                                              return_h=True)
            lg_1d = eng.head(hh[0, -1])
            lg_e = eng.forward_logits(d2, False, extra=())
        diff = float((lg_1d - lg_e).abs().max())
        info = float((lg_rows[-1] - lg_e).abs().max())
        agree = int(lg_rows[-1].argmax()) == int(lg_e.argmax())
        ln = (f"[parity] gates vs engine max|Δlogit| {diff:.2e} (1-D head) | "
              f"{info:.2e} (row-GEMM, info) argmax_agree={agree}")
        print(ln, flush=True)
        lines.append(ln)
        if diff > 1e-3:
            raise SystemExit(f"PARITY FAILURE ({diff:.2e}): gated forward drifted "
                             f"from the anchored engine — do not train on this")

    # ---- eval: DIRECT answer metrics under a hard mask regime ----
    def open_table(regime: str) -> torch.Tensor:
        from gnnformer.learnmask import _CH0, fence_open_table
        if regime == "hand":
            return hand_open_table(eng.n_layers, l_open_ref)
        if regime == "init":
            return fence_open_table(eng.n_layers)
        if regime == "nofence":
            return torch.ones(N_CH, eng.n_layers, dtype=torch.bool)
        if regime == "supply":  # force the readout THROUGH the supply positions:
            t = fence_open_table(eng.n_layers)   # tail reads replicas, NOT frames
            t[_CH0["R7"], :] = True
            t[_CH0["R6"], :] = False
            return t
        if regime == "handsupply":  # hand design (R4/R7 >= l_open) + frames hidden
            t = hand_open_table(eng.n_layers, l_open_ref)  # from the tail
            t[_CH0["R6"], :] = False
            return t
        return gates.hard_open_table()  # "gates" (== fence init before training)

    def hard_layer_masks(d, regime: str, e: int):
        if regime == "gates" and args.arm == "s0":
            gt = gates.gate_table(mode="hard")
            return [gates.layer_mask(gt[:, li], MASK_MIN) for li in range(eng.n_layers)]
        cm = relation_cell_map(d["seq"], d["blocks"], readers_of(d), d["fin"], e=e)
        return [m.to(eng.dev) for m in hard_masks_by_layer(cm, open_table(regime))]

    def answer_row_logits(d, masks):
        return gated_stack_logits(eng, d, [], lambda li, S: masks[li])[-1]

    def evaluate(regime: str):
        """-> (class acc/EM, class CE (over gold<=9), EM, MAE, per-gold str).
        Per-sample rule: gold <= 9 under --target class -> one-forward 10-way class
        read; otherwise greedy digit-sequence decode (counts beyond 9 are two digit
        tokens — no single-token class exists). EM/MAE cover ALL samples."""
        n = len(ev)
        acc = em = n_cls = 0
        ce_tot = mae = 0.0
        per_gold: dict = {}
        with torch.no_grad():
            for d in ev:
                if args.target == "class" and d["gold"] <= 9:
                    masks = hard_layer_masks(d, regime, 0)
                    lg = answer_row_logits(d, masks)
                    cls = lg[digit_ids]
                    pred = int(cls.argmax())
                    ce_tot += float(F.cross_entropy(
                        cls.view(1, -1), torch.tensor([d["gold"]], device=cls.device)))
                    n_cls += 1
                    em_ok = tok.decode([int(lg.argmax())]).strip() == str(d["gold"])
                    ok = pred == d["gold"]
                else:
                    masks = hard_layer_masks(d, regime, args.decode_tokens)
                    parsed, _first_d, _txt = gated_greedy_digits(
                        eng, d, lambda li: masks[li], max_tokens=args.decode_tokens)
                    pred = parsed if parsed is not None else -1
                    ok = em_ok = pred == d["gold"]
                acc += ok
                em += em_ok
                mae += abs(pred - d["gold"])
                pg = per_gold.setdefault(d["gold"], [0, 0])
                pg[1] += 1
                pg[0] += ok
        pgs = " ".join(f"{g}:{c}/{m}" for g, (c, m) in sorted(per_gold.items()))
        return acc / n, ce_tot / max(n_cls, 1), em / n, mae / n, pgs

    ep0 = [("handfence", "hand"), ("init", "gates"), ("nofence", "nofence")]
    if args.ep0_regimes:
        # '+' also separates: comma lists cannot ride sbatch --export values
        extra = [r.strip() for r in args.ep0_regimes.replace("+", ",").split(",")]
        known = {"supply", "handsupply"}
        bad = [r for r in extra if r and r not in known]
        if bad:
            raise SystemExit(f"unknown --ep0-regimes {bad} (known: {sorted(known)})")
        ep0 += [(r, r) for r in extra if r]
    for tag, regime in ep0:
        acc, ce, em, mae, pgs = evaluate(regime)
        ln = (f"[ep 0 {tag}] class_acc {acc:.3f} class_ce {ce:.4f} em {em:.3f} "
              f"mae {mae:.2f} [{pgs}]")
        print(ln, flush=True)
        lines.append(ln)
        (out / "report.txt").write_text("\n".join(lines) + "\n")

    # ---- train: direct answer CE + deviation penalty ----
    rng = np.random.default_rng(args.seed)
    total_steps = max(args.epochs * len(tr), 1)
    gstep = 0
    best = ((-1.0, float("-inf")), 0)
    acc = em = mae = 0.0
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(tr))
        tot = tot_ce = 0.0
        te = time.time()
        for step, i in enumerate(idx):
            d = tr[int(i)]
            tau = args.tau0 * (args.tau1 / args.tau0) ** (gstep / max(total_steps - 1, 1))
            gstep += 1
            if args.train == "lora":
                masks = hard_layer_masks(d, args.fixed_regime, len(d["tgt"]))
                lg = gated_stack_logits(eng, d, d["tgt"], lambda li, S: masks[li],
                                        grad_ckpt=args.grad_ckpt)
            elif args.arm == "s0":
                lg = freetable_tf_logits(eng, d, gates, tau=tau, mode="train",
                                         grad_ckpt=args.grad_ckpt)
            else:
                lg = gated_tf_logits(eng, d, d["tgt"], gates, tau=tau, mode="train",
                                     grad_ckpt=args.grad_ckpt)
            if args.target == "class":
                cls = lg[-1][digit_ids]
                ce = F.cross_entropy(cls.view(1, -1),
                                     torch.tensor([d["gold"]], device=cls.device))
            else:
                ce = F.cross_entropy(lg[:-1],
                                     torch.tensor(d["tgt"], device=lg.device))
            if args.train == "lora":
                loss = ce
            else:
                dev_open, dev_close = gates.deviation()
                loss = ce + args.lam_open * dev_open + args.lam_close * dev_close
            (loss / args.accum).backward()
            tot += float(loss.detach())
            tot_ce += float(ce.detach())
            if (step + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        eval_regime = args.fixed_regime if args.train == "lora" else "gates"
        acc, ce_ev, em, mae, pgs = evaluate(eval_regime)
        ln = (f"[ep {ep}] loss {tot/len(tr):.4f} ce {tot_ce/len(tr):.4f} tau {tau:.2f} "
              f"| HARD class_acc {acc:.3f} class_ce {ce_ev:.4f} em {em:.3f} "
              f"mae {mae:.2f} [{pgs}] ({time.time()-te:.0f}s/ep)")
        print(ln, flush=True)
        lines.append(ln)
        if args.train == "lora":
            st = {"lora": lora.state(), "rank": args.rank, "alpha": args.alpha,
                  "l_open": 0, "fixed_regime": args.fixed_regime, "epoch": ep,
                  "acc": acc, "class_ce": ce_ev, "em": em, "scaffold": args.scaffold,
                  "resize": args.resize, "target": args.target,
                  "tail_style": args.tail_style}
            torch.save(st, out / "readout_last.pt")
            if (acc, -ce_ev) > best[0]:
                best = ((acc, -ce_ev), ep)
                torch.save(st, out / "readout_best.pt")
        else:
            gline = gates.stats_line()  # gates that never moved => the arm is VOID
            print(f"[ep {ep}] {gline}", flush=True)
            lines.append(f"ep{ep} {gline}")
            st = {**gates.state(), "epoch": ep, "acc": acc, "class_ce": ce_ev,
                  "em": em, "scaffold": args.scaffold, "l_open_ref": l_open_ref,
                  "carrier_ckpt": (args.carrier_ckpt if args.scaffold == "carrier"
                                   else None),
                  "readout_ckpt": args.readout_ckpt or None,
                  "resize": args.resize, "target": args.target}
            torch.save(st, out / "gates_last.pt")
            np.savetxt(out / f"heatmap_ep{ep}.csv",
                       gates.full_p_open().cpu().numpy(), delimiter=",", fmt="%.4f")
            if (acc, -ce_ev) > best[0]:
                best = ((acc, -ce_ev), ep)
                torch.save(st, out / "gates_best.pt")
        # rewrite the report every epoch: a walltime kill must not lose the summary
        (out / "report.txt").write_text("\n".join(lines) + "\n")
    lines.append(f"BEST class_acc {best[0][0]:.3f} (class_ce {-best[0][1]:.4f}) "
                 f"@ ep {best[1]}")
    lines.append(f"LAST class_acc {acc:.3f} em {em:.3f} mae {mae:.2f} @ ep {args.epochs}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))
    print("wrote", out)
    if lora is not None:
        lora.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
