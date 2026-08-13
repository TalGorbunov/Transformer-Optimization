#!/usr/bin/env python3
"""S0 — the free-table mask oracle: can ANY attention mask make a frozen VLM count?

THE QUESTION
    Our hand-designed attention fence provably delivers per-frame information to
    known positions inside the network (linear-probe d' 6.34), yet the frozen model
    answers counting questions at majority-class level under every hand-picked
    topology. Is that because no mask can help — or because we picked the wrong
    masks? S0 answers by brute force: give EVERY individually toggleable attention
    edge its own learnable open/closed probability and let gradient descent search
    the space directly. If a helpful mask exists over these edges, this is the
    least-constrained learner that could find it; if even this finds nothing, the
    bottleneck is not attention routing.

THE SETUP (all components frozen except the mask)
    Model    frozen Qwen2.5-VL-7B, 4-bit nf4 (gnnformer.runtime).
    Prompt   [frame_i + question-replica] x N, then the canonical count prompt
             ("Respond with a single integer ... Answer: ") — the PROMPT-CRITICAL
             wording without which the frozen model emits no digit at all.
    Edges    every causal (query,key) cell in four families, per layer:
             replica->earlier replica, replica->earlier frame content,
             answer-tail->frame content, answer-tail->replicas.
             (239,232 cells x 28 layers = 6.7M gates on the seq-8 layout; anchors —
             prefix, within-frame causal attention — stay fixed; NOTHING above the
             causal diagonal is ever learnable.)
    Gate     independent Bernoulli per (cell, layer): P(open) = sigmoid(logit),
             initialized at the fence (open ~0.88 / closed ~0.12).
    Sampling ST-Gumbel (Jang et al. '17): the forward pass uses a HARD 0/1 sample,
             gradients flow through the soft relaxation, temperature annealed.
    Loss     answer-class CE: cross-entropy over the 10 digit-token logits at the
             answer position (one forward, nothing appended, nothing copyable).
             NO deviation penalty — the search is unconstrained on purpose.
    Eval     the deploy-semantics mask (logit > 0 -> open, else -infinity), i.e.
             the mask you would actually ship, plus baseline rows: the hand fence,
             the fence init, and no-fence (plain causal attention).

THE VERDICT CRITERIA
    A positive result = eval accuracy meaningfully above the majority class under
    the thresholded mask. Diagnostics that separate "search failed" from "nothing
    to find": the flip counter (did ANY gate cross its decision boundary), the
    logit-drift stats (did gradients reach and move the gates), and per-family
    P(open) means (which directions the objective pulls).

WHAT WE OBSERVED (2026-08-13, jobs 131404 / 132431; outputs/learnmask/exist_s0*)
    Train CE fell (sampled-mask calibration) while the thresholded mask NEVER
    changed: 0 flips out of 6,698,496 after unconstrained training; 96.5% of logits
    moved (mean |dlogit| 0.28, max 1.64 — gradients were real), drifting weakly
    TOWARD the hand design (aggregation edges +, tail->frames -) but ~100x too
    faintly to cross any boundary; eval stayed bit-identical to the fence init at
    majority-class accuracy. Together with the probe result (the information IS
    linearly present), this locates the wall in the frozen READOUT, not in the
    attention topology — no mask can teach a frozen model to say what it cannot
    compute into its answer position.

Run (single GPU, ~15 min at the defaults):
    python scripts/learnmask/s0_free_table_exp.py --limit 60 --epochs 4
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import MASK_MIN
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample
from gnnformer.engine import CarrierEngine
from gnnformer.learnmask import (
    CHANNELS,               # the relation vocabulary (names + fence init values)
    FENCE_ON,               # [n_channels] bool: open/closed in the hand fence
    SOFT_FORBID,            # -30: train-time "closed" (keeps d/dlogit sane in bf16)
    arm_learn_mask,         # which relation channels the S2 scope covers
    gated_stack_logits,     # full-stack forward with caller-supplied per-layer masks
    hand_open_table,        # the deployed hand design, as hard gate values
    hard_masks_by_layer,    # hard masks for a (channel x layer) open/closed table
    layout_key,             # token-layout identity (S0 needs ONE shared layout)
    mask_parts,             # cell map -> (frozen base mask, gather index)
    prepare_sample_replicas,
    relation_cell_map,      # classify every attention cell into a relation
)
from gnnformer.mmred_hf import qtype_from_dirname
from gnnformer.runtime import load_runtime


# --------------------------------------------------------------------- free table

def build_free_table(cell_map: torch.Tensor, n_layers: int, init_logit: float = 2.0):
    """One learnable logit per (cell, layer) over the S2-scope cells.

    Returns (logits [n_cells, n_layers], flat_idx [n_cells] into the flattened
    mask, base [S,S] frozen structure, fence_on [n_cells] init direction).
    Init: +init_logit where the fence opens the cell, -init_logit where it closes
    it — so thresholding at 0 reproduces the fence exactly (verified bit-for-bit
    against the anchored mask construction by tests/test_learnmask.py)."""
    learn_ch = arm_learn_mask("s2")
    cm = cell_map.long()
    learnable = (cm >= 0) & learn_ch[cm.clamp(min=0)]
    flat_idx = learnable.view(-1).nonzero(as_tuple=True)[0]
    fence_on = FENCE_ON[cm.view(-1)[flat_idx]]
    base, _ = mask_parts(cell_map, learn_ch, FENCE_ON.clone())
    sign = torch.where(fence_on, 1.0, -1.0)
    logits = torch.nn.Parameter(
        (init_logit * sign).view(-1, 1).repeat(1, n_layers))
    return logits, flat_idx, base, fence_on


def sample_gates(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """ST-Gumbel: hard 0/1 forward, gradient through the soft relaxation."""
    u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / tau)
    return (soft > 0.5).float() - soft.detach() + soft   # exactly 0/1 forward


def layer_mask(base, flat_idx, g_col, K):
    """Additive attention mask for one layer: frozen structure + (1-g)*K scattered
    into the learnable cells (g=1 -> open/0, g=0 -> closed/K). Differentiable."""
    S = base.shape[0]
    m = base.clone().view(-1)
    m = m.index_put((flat_idx,), (1.0 - g_col) * K, accumulate=True)
    return m.view(S, S)


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_hf/dirs/seq_len_8_train_steps_in_room")
    ap.add_argument("--eval-root", default="data/mmred_hf/dirs/seq_len_8_val_steps_in_room")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--eval-limit", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    rt = load_runtime()                      # frozen 4-bit 7B
    eng = CarrierEngine(rt, l_open=12, e_c=None)   # geometry helpers only; no e_c
    digit_ids = eng.digit_ids                # token ids of '0'..'9'
    n_layers = eng.n_layers

    # ---- data: replica prompts, one shared token layout -------------------------
    def prep(root, lim):
        out = []
        for sd in iter_sample_dirs_shuffled(Path(root), 0):   # class-balanced order
            if len(out) >= lim:
                break
            qtype = qtype_from_dirname(sd.name)
            try:
                _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
            except Exception:
                continue
            if qtype != "steps_in_room" or not str(a0).strip().isdigit():
                continue
            gold = int(str(a0).strip())
            if gold > 9:
                continue                     # class CE is a 10-way single-token read
            rec = prepare_sample_replicas(eng, frames, q0, gold=gold, task=qtype)
            if rec is not None:
                rec["gold"] = gold
                out.append(rec)
        return out

    t0 = time.time()
    tr, ev = prep(args.data_root, args.limit), prep(args.eval_root, args.eval_limit)
    # a per-cell table only makes sense on ONE token layout -> keep the dominant one
    dom = Counter(layout_key(d) for d in tr).most_common(1)[0][0]
    tr = [d for d in tr if layout_key(d) == dom]
    ev = [d for d in ev if layout_key(d) == dom]
    print(f"data: train {len(tr)} eval {len(ev)} (dominant layout, {time.time()-t0:.0f}s)")
    print(f"eval golds: {dict(sorted(Counter(d['gold'] for d in ev).items()))}")

    # ---- the free table ----------------------------------------------------------
    d0 = tr[0]
    cm = relation_cell_map(d0["seq"], d0["blocks"], d0["readers"], d0["fin"])
    logits, flat_idx, base, fence_on = build_free_table(cm, n_layers)
    dev = eng.dev
    logits = torch.nn.Parameter(logits.detach().to(dev))  # leaf on the GPU
    flat_idx, base = flat_idx.to(dev), base.to(dev)
    init = logits.detach().clone()
    print(f"free table: {flat_idx.numel():,} cells x {n_layers} layers "
          f"= {logits.numel():,} gates (causal lower triangle only)")

    # ---- eval: thresholded (deploy-semantics) mask + reference regimes ----------
    def answer_logits(d, masks):
        with torch.no_grad():
            return gated_stack_logits(eng, d, [], lambda li, S: masks[li])[-1]

    def evaluate(masks) -> tuple:
        hits, ce = 0, 0.0
        for d in ev:
            lg = answer_logits(d, masks)
            cls = lg[digit_ids]
            hits += int(cls.argmax()) == d["gold"]
            ce += float(F.cross_entropy(cls.view(1, -1),
                                        torch.tensor([d["gold"]], device=cls.device)))
        return hits / len(ev), ce / len(ev)

    def hard_masks():
        g = (logits.detach() > 0).float()
        return [layer_mask(base, flat_idx, g[:, li], MASK_MIN) for li in range(n_layers)]

    for name, table in (("hand fence", hand_open_table(n_layers, 12)),
                        ("no fence", torch.ones_like(hand_open_table(n_layers, 12)))):
        masks = [m.to(dev) for m in hard_masks_by_layer(cm, table)]
        acc, ce = evaluate(masks)
        print(f"[baseline] {name:10s} acc {acc:.3f} ce {ce:.4f}")
    acc, ce = evaluate(hard_masks())
    print(f"[baseline] fence init  acc {acc:.3f} ce {ce:.4f}   <- the starting mask")

    # ---- train: unconstrained search over 6.7M gates -----------------------------
    opt = torch.optim.Adam([logits], lr=args.lr)
    step_total = max(args.epochs * len(tr), 1)
    gstep = 0
    for ep in range(1, args.epochs + 1):
        te, ce_sum = time.time(), 0.0
        for i in torch.randperm(len(tr)).tolist():
            d = tr[i]
            tau = args.tau0 * (args.tau1 / args.tau0) ** (gstep / max(step_total - 1, 1))
            gstep += 1
            g = sample_gates(logits, tau)                       # one hard sample
            lg = gated_stack_logits(
                eng, d, [],
                lambda li, S: layer_mask(base, flat_idx, g[:, li], SOFT_FORBID))[-1]
            cls = lg[digit_ids]
            loss = F.cross_entropy(cls.view(1, -1),
                                   torch.tensor([d["gold"]], device=cls.device))
            (loss / 8).backward()
            ce_sum += float(loss.detach())
            if gstep % 8 == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        delta = logits.detach() - init
        flips = int(((logits.detach() > 0) != (init > 0)).sum())
        acc, ce = evaluate(hard_masks())
        print(f"[ep {ep}] train-ce {ce_sum/len(tr):.4f} tau {tau:.2f} | "
              f"HARD acc {acc:.3f} ce {ce:.4f} | flips {flips}/{logits.numel():,} "
              f"max|dlogit| {delta.abs().max():.2f} ({time.time()-te:.0f}s)")

    # ---- verdict ------------------------------------------------------------------
    p = torch.sigmoid(logits.detach())
    print("\nper-family mean P(open) after training (init: open 0.88 / closed 0.12):")
    ch = cm.view(-1).long()[flat_idx.cpu()]
    for c in sorted(set(ch.tolist())):
        sel = (ch == c).to(dev)
        print(f"  {CHANNELS[c].name:12s} {p[sel].mean():.3f}")
    print("\nA flip count of 0 with moving logits + flat HARD accuracy = the search "
          "is real,\nand there is nothing in this space for it to find.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
