#!/usr/bin/env python3
"""Co-Former trainer: two-pass select-and-repack on filtered-counting.

Arms:
  A2     keep-all control, LoRA trained at N=8                  (baseline)
  A2mix  keep-all control, LoRA trained at N={8,32}             (fair baseline for LS/LA)
  OR     oracle selector (keep relevant frames), repack, N=8    (ceiling / geometry arm)
  LS     learned selector, SPLIT-LENGTH: LoRA steps on N=8 batches (selector hard,
         detached), selector steps on N=32 batches (ST-Gumbel keep -> mask grad;
         positions follow the hard decisions) — selection gets gradient exactly where
         it pays (incentive fix; cf. relevance-gap 0.000 in every in-length-only run)
  LA     LS + selector warm-started 1 epoch with BCE on derivable keep labels
         (learned retrieval; oracle bounds it)

Eval: hard keep + repack, EM at N=8/16/32/64 (gold<=8 everywhere) + selection quality
(keep-rate relevant vs distractor). Machinery validation: keep-all == full attention
with ORIGINAL positions (asserted); oracle-repack on a full-attention-trained LoRA
reproduces the deletion probe (~1.0 @N=64).

Smoke:
  HF_HOME=$HOME/.cache/huggingface python scripts/condmask/train_coformer.py --arm LS \\
      --limit 64 --long-limit 64 --dev-limit 24 --epochs 1 --output outputs/_scratch/cof
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gnnformer.carriers import attach_lora
from gnnformer.constants import MASK_MIN
from coformer import (PairSelector, Selector, SoftGate, answer_row_frame_mass,
                      answer_row_token_mass, oracle_keep, repack_mask,
                      repack_positions, shared_positions, soft_repack_positions,
                      soft_rope, token_mask, token_repack, log_gate_mask)
from gatenet import st_gumbel, tau_schedule
from masks import SOFT_FORBID, assemble_batch_masks, batch_mask_parts
from modeling import (assert_lora_targets, forward_answer_logits, load_text_model,
                      model_dims, pad_batch)
from relgate import (clause_token_labels, early_hidden, frame_feats,
                     prep_root_crowded, prep_root_rel, rel_mask_parts)
from textdata import majority_baseline

ARMS = ("A2", "A2mix", "OR", "LS", "LA", "LAF", "SP", "AS", "AST", "ASTH", "AS2", "AS3", "AS3T", "OTOK", "AS4", "AS4M", "AS5")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--short-root", default="data/mmred_filtered/seq_len_8/train")
    ap.add_argument("--long-root", default="data/mmred_filtered/seq_len_32/train")
    ap.add_argument("--dev-root", default="data/mmred_filtered/seq_len_8/dev")
    ap.add_argument("--long-dev-root", default="data/mmred_filtered/seq_len_32/dev")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--long-limit", type=int, default=3600)
    ap.add_argument("--dev-limit", type=int, default=270)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr-sel", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--clip-sel", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--tau-hold", type=float, default=0.3)
    ap.add_argument("--feat-layer", type=int, default=2)
    ap.add_argument("--warm-labels", choices=("oracle", "pseudo"), default="oracle",
                    help="pseudo = LABEL-FREE: warm the selector on the frozen "
                         "model's own zero-shot yes/no relevance judgments "
                         "(LLM-as-judge retrieval), no gold labels anywhere")
    ap.add_argument("--warm-extra-root", default="",
                    help="extra warm-BCE data (e.g. an N=128 slice: frame-NUMBER "
                         "tokens must cover the deployment range — measured leak: "
                         "keep(dist) 0.13@N=64 -> 0.83@N=128 when warmed at N<=32)")
    ap.add_argument("--sel-arch", choices=("tf", "pairwise"), default="tf",
                    help="pairwise = per-frame (q x f) MLP, position-free features "
                         "recommended (--feat-layer 0): N-invariant by construction")
    ap.add_argument("--as-layer", type=int, default=16,
                    help="AS arm: selection depth (attention-relevance AUC peaks at "
                         "L16 — measured 2026-08-19)")
    ap.add_argument("--lr-gate", type=float, default=3e-2,
                    help="AS4: lr for the two SoftGate scalars")
    ap.add_argument("--gate-a0", type=float, default=-1.0,
                    help="AS4 gate init threshold. NEGATIVE = gates open (keep-all) "
                         "at step 0 — measured 2026-08-24: at init the L16 answer-row "
                         "mass is UNIFORM (~0.16*1/n, rel==junk; the retrieval head "
                         "is CREATED by training, not found), so a0=0.5 starts "
                         "all-closed: tail blind, CE pinned at majority, "
                         "self-reinforcing closed-gate optimum (probe 398003)")
    ap.add_argument("--gate-tau0", type=float, default=0.25,
                    help="AS4: initial (learnable) gate temperature")
    ap.add_argument("--sel-freeze-after-warm", action="store_true",
                    help="AS5: distill-and-stop — the selector trains ONLY in the "
                         "warm stage (measured: ST-CE fine-tune drifts back to "
                         "keep-all; both vertices fit train CE, residual tilt "
                         "points open)")
    ap.add_argument("--warm-sel", action="store_true",
                    help="AS5: warm the selector with BCE on the model's own "
                         "zero-shot judgments (label-free) BEFORE the ST-vertex "
                         "CE fine-tune — a separated init has no cold-start "
                         "cliff (Addendum 9: crossings don't recover)")
    ap.add_argument("--st-vertex", action="store_true",
                    help="AS5: straight-through AT THE VERTEX — forward always the "
                         "exact hard repack (every visited configuration is "
                         "decodable; the soft interior is OOD, Addendum 9 vi), "
                         "backward through the soft branch linearized there")
    ap.add_argument("--as-topk", type=int, default=0,
                    help="attn regime: keep the top-k frames by mass instead of "
                         "thresholding (N-independent rule for bounded-evidence "
                         "tasks; the alpha threshold's optimum drifts with N: "
                         "0.5@128 -> 2.0@1024, measured 2026-08-26)")
    ap.add_argument("--gate-lam", type=float, default=0.1,
                    help="AS5: sparsity prior on gates (Co-GNN action cost). CE "
                         "defends evidence through differentiable positions; lam "
                         "pays for closing junk; no relevance labels anywhere")
    ap.add_argument("--init-ckpt", default="",
                    help="warm-start the LoRA from this ckpt (gate NOT loaded)")
    ap.add_argument("--freeze-readout", action="store_true",
                    help="freeze LoRA on layers >= as_layer: the readout cannot "
                         "absorb the long length, so CE's ONLY route down is "
                         "selection (measured 2026-08-24: N=128 in-stream -> "
                         "dev128 .988 under keep-all, gates never close — CE "
                         "always prefers absorption when available)")
    ap.add_argument("--as-alpha", type=float, default=0.5,
                    help="AS arm: keep frames with mass >= alpha * uniform share; "
                         "recalibrated on dev after training (recall >= 0.99)")
    ap.add_argument("--crowded", action="store_true",
                    help="load data with per-clause spans (dense-counting testbed)")
    ap.add_argument("--keep-heads", action="store_true",
                    help="OTOK/AS3T: also keep each kept frame's 'Frame k:' header — "
                         "distinct anchors make repeated evidence clauses countable "
                         "(oracle without heads plateaus at 0.55: identical-item "
                         "counting wall)")
    ap.add_argument("--aux-labels", choices=("evid", "entity"), default="evid",
                    help="AS3T token-aux teacher: evidence clauses (C AND R — "
                         "N-invariant kept set; selection = detection) or any "
                         "C-clause (recall-safe superset, kept set O(N))")
    ap.add_argument("--ast-alpha", type=float, default=0.25,
                    help="AST: token kept if mass >= ast-alpha * uniform-over-frame-"
                         "region share (recall-tilted: keeping junk is cheap)")
    ap.add_argument("--head-cuts", action="store_true",
                    help="AS3T: additionally cut, per head, columns below "
                         "asth-beta * uniform share (mask-only refinement on top "
                         "of the global token repack; positions are per-token, so "
                         "head granularity can never repack)")
    ap.add_argument("--asth-beta", type=float, default=0.1,
                    help="ASTH: per-head extra column cut below beta * uniform share")
    ap.add_argument("--as-lam", type=float, default=1.0,
                    help="AS arm: weight of the attention-supervision aux loss "
                         "(pseudo-label teacher, training-time only)")
    ap.add_argument("--keep-margin", type=float, default=2.0,
                    help="hard-decision logit shift toward KEEP: dropping a relevant "
                         "frame is fatal to counting ((1-p)^N), keeping an extra "
                         "distractor is nearly free — tilt the operating point to "
                         "recall (LAF/eval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="outputs/coformer")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--lora-ckpt", default="",
                    help="eval-only: external LoRA (e.g. relmask A2) for the "
                         "oracle-repack machinery validation")
    ap.add_argument("--eval-regime", default="",
                    choices=("", "keepall", "oracle", "pseudo", "attn"))
    ap.add_argument("--eval-roots", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok, model = load_text_model(args.model, device=args.device)
    L, H, D = model_dims(model)
    dtype = next(model.parameters()).dtype
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    _sel_cls = PairSelector if args.sel_arch == "pairwise" else Selector
    selector = _sel_cls(D, seed=args.seed).to(args.device) \
        if args.arm in ("LS", "LA", "LAF", "AS5") or args.eval_only else None
    gate = SoftGate(args.gate_a0, args.gate_tau0).to(args.device)

    def keeps_for(recs, ids, parts, tau, mode):
        """-> list of [F] keep tensors + (logits or None)."""
        if args.eval_only and args.eval_regime == "keepall":
            return [torch.ones(len(r["blocks"]), device=args.device) for r in recs], None
        if args.eval_only and args.eval_regime == "oracle":
            return [oracle_keep(r, args.device) for r in recs], None
        if args.eval_only and args.eval_regime == "pseudo":
            # TRAINING-FREE selection: the frozen model's own zero-shot yes/no
            # judgment per frame decides keep/drop at eval time
            return pseudo_labels(recs), None
        if args.eval_only and args.eval_regime == "attn":
            # TRAINING-FREE selection from NATIVE attention: run the frozen
            # model (native full attention, native positions) to as_layer,
            # threshold the answer row's summed per-frame mass (user proposal;
            # risk: this signal is exactly what dilutes with N)
            from modeling import model_parts as _mp, sdpa_kernel as _sk,                 SDPA_BACKENDS as _SB
            embed_, layers_, _, _, rotary_ = _mp(model)
            S_ = ids.shape[1]
            bases_ = torch.stack([p_[0] for p_ in parts]).unsqueeze(1).to(
                next(model.parameters()).dtype)
            h_ = embed_(ids)
            pos_ = torch.arange(S_, device=args.device).unsqueeze(0).expand(
                len(recs), -1)
            c_, s_ = rotary_(h_, pos_)
            pe_ = (c_.to(h_.dtype), s_.to(h_.dtype))
            with torch.no_grad():
                for li in range(args.as_layer):
                    with _sk(_SB):
                        o_ = layers_[li](h_, attention_mask=bases_,
                                         position_ids=pos_,
                                         position_embeddings=pe_)
                    h_ = o_[0] if isinstance(o_, tuple) else o_
                ans_ = torch.tensor([r["seq"] - 1 for r in recs],
                                    device=args.device)
                mass_ = answer_row_frame_mass(
                    layers_[args.as_layer], h_, pe_,
                    [r["blocks"] for r in recs], S_, row=ans_)
            ks_ = []
            for b, r in enumerate(recs):
                nfr = len(r["blocks"])
                if args.as_topk > 0:
                    kb = torch.zeros(nfr, device=args.device)
                    kb[mass_[b, :nfr].topk(min(args.as_topk, nfr)).indices] = 1.0
                else:
                    kb = (mass_[b, :nfr] * nfr >= args.as_alpha).float()
                    if kb.sum() == 0:
                        kb = torch.ones_like(kb)
                ks_.append(kb)
            return ks_, None
        if args.arm in ("A2", "A2mix"):
            return [torch.ones(len(r["blocks"]), device=args.device) for r in recs], None
        if args.arm == "OR":
            return [oracle_keep(r, args.device) for r in recs], None
        bases = torch.stack([p_[0] for p_ in parts])
        feats = early_hidden(model, ids, bases, args.feat_layer)
        q, f, fv = frame_feats(feats, recs, ids.shape[1])
        lg = selector(q, f, fv)                              # [B, F]
        ks = st_gumbel(lg + (args.keep_margin if mode == "hard" else 0.0), tau, mode)
        return [ks[b, : len(recs[b]["blocks"])] for b in range(len(recs))], lg

    _free_g = {"v": None}
    _osq = {}

    def as_forward(recs, K, want_aux=False, soft=False):
        """AS: ONE forward — native full attention to layer as_layer, the model's own
        answer-row attention mass selects frames, remaining layers run with dropped
        frames cut and kept frames position-compacted. -> (logits [B,V], mass [B,F])."""
        from modeling import model_parts, sdpa_kernel, SDPA_BACKENDS
        embed_, layers_, norm_, head_, rotary_ = model_parts(model)
        ids, valid, ans, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
        parts = [rel_mask_parts(r, S, args.device) for r in recs]
        bases = torch.stack([p_[0] for p_ in parts]).unsqueeze(1).to(dtype)
        B = len(recs)
        h = embed_(ids)
        pos = torch.arange(S, device=args.device).unsqueeze(0).expand(B, -1)
        cos, sin = rotary_(h, pos)
        pe = (cos.to(h.dtype), sin.to(h.dtype))
        blocks_list = [r["blocks"] for r in recs]
        if args.arm in ("AS3", "AS3T", "OTOK", "AS4", "AS4M", "AS5"):
            # FULLY N-invariant selection: pre-segment isolates every frame (shared
            # as-if-first positions) AND blinds the tail to frames — the selection
            # QUERY is purely question-derived (AS2's failure: invariant keys but a
            # query built from an N-diluted read; recursion principle, third site)
            pre_mask = torch.stack([repack_mask(parts[b][0], parts[b][1],
                                                parts[b][2],
                                                torch.zeros(len(recs[b]["blocks"]),
                                                            device=args.device),
                                                K, dtype)
                                    for b in range(B)])
            pre_pos = torch.stack([shared_positions(r, S, args.device) for r in recs])
            c_, s_ = rotary_(h, pre_pos)
            pre_pe = (c_.to(h.dtype), s_.to(h.dtype))
        elif args.arm == "AS2":
            # length-invariant pre-segment: frames isolated + as-if-first positions,
            # so the SELECTING attention at as_layer sees every frame at identical
            # in-distribution geometry (kills the recursion-principle decay AND the
            # RoPE distance confound on selection mass)
            from masks import batch_mask_parts as _bmp, assemble_batch_masks as _abm
            b_, t_ = _bmp(recs, S, args.device)
            pre_mask = _abm(b_, t_, torch.ones(B, 1, device=args.device), K, dtype)
            pre_pos = torch.stack([shared_positions(r, S, args.device) for r in recs])
            c_, s_ = rotary_(h, pre_pos)
            pre_pe = (c_.to(h.dtype), s_.to(h.dtype))
        else:
            pre_mask, pre_pos, pre_pe = bases, pos, pe
        for li in range(args.as_layer):
            with sdpa_kernel(SDPA_BACKENDS):
                out = layers_[li](h, attention_mask=pre_mask, position_ids=pre_pos,
                                  position_embeddings=pre_pe)
            h = out[0] if isinstance(out, tuple) else out
        if os.environ.get("OSQ") and not soft:
            h = h.detach().requires_grad_(True)
            _osq["h"] = h
        sel_pe = pre_pe if args.arm in ("AS2", "AS3", "AS3T", "OTOK", "AS4", "AS4M", "AS5") else pe
        mass = answer_row_frame_mass(layers_[args.as_layer], h, sel_pe, blocks_list, S)
        keeps, masks_l, pos2_l = [], [], []
        if args.arm in ("AST", "ASTH", "AS3T", "OTOK"):
            tmass, hmass = answer_row_token_mass(layers_[args.as_layer], h, sel_pe)
            if args.arm == "OTOK":
                for b, r in enumerate(recs):
                    keep_t = torch.zeros(S, dtype=torch.bool, device=args.device)
                    keep_t[: r["prefix_end"]] = True
                    keep_t[r["fin"]: r["seq"]] = True
                    lab = clause_token_labels(r, S, args.device, "evid").bool()
                    if args.keep_heads:
                        for fi, cl in enumerate(r["clauses"]):
                            if any(e for (_, _, _, e) in cl):
                                a0 = r["blocks"][fi][0]
                                lab[a0: cl[0][0]] = True   # the "Frame k:" header
                    keep_t += lab
                    masks_l.append(token_mask(keep_t, K, dtype, parts[b][0]))
                    pos2_l.append(token_repack(keep_t))
                    keeps.append(torch.zeros(len(r["blocks"]), device=args.device))
                masks2 = torch.stack(masks_l)
                pos2 = torch.stack(pos2_l)
                aux_t = None
            else:
             for b, r in enumerate(recs):
                fr0, fr1 = r["blocks"][0][0], r["blocks"][-1][1]
                nft = fr1 - fr0
                keep_t = torch.ones(S, dtype=torch.bool, device=args.device)
                thr = args.ast_alpha / nft
                keep_t[fr0:fr1] = tmass[b, fr0:fr1].detach() >= thr
                keep_t[r["seq"]:] = False                     # padding stays dropped
                keep_t[r["fin"]: r["seq"]] = True             # tail always kept
                if args.keep_heads and "clauses" in r:
                    # distinct anchors: keep a frame's header iff any of its clause
                    # tokens survive (identical-item counting wall: oracle 0.55 flat
                    # without headers -> 0.88-0.98 with; measured 2026-08-24)
                    for fi, cl in enumerate(r["clauses"]):
                        a0, b0 = r["blocks"][fi]
                        if keep_t[cl[0][0]: b0].any():
                            keep_t[a0: cl[0][0]] = True
                hw = None
                if args.arm == "ASTH" or (args.arm == "AS3T" and args.head_cuts):
                    hw = (hmass[b, :, :].detach() < args.asth_beta / nft).float()
                    hw[:, :fr0] = 0.0                          # prefix/tail untouched
                    hw[:, fr1:] = 0.0
                masks_l.append(token_mask(keep_t, K, dtype, parts[b][0], hw))
                pos2_l.append(token_repack(keep_t))
                nfr = len(r["blocks"])
                kb = torch.zeros(nfr, device=args.device)
                for i, (a, e) in enumerate(r["blocks"]):
                    kb[i] = keep_t[a:e].float().mean()
                keeps.append(kb)
            masks2 = torch.stack(masks_l) if args.arm == "ASTH" else                 torch.stack(masks_l)
            pos2 = torch.stack(pos2_l)
        elif args.arm == "AS5":
            # Co-GNN proper: gates from an EXTERNAL N-invariant action head that
            # never touches the answer path (all LoRA frozen) — the only module
            # CE can train is the selector, so absorption is impossible
            # (measured escape routes: readout ep-v3, body ep-p2).
            if _free_g["v"] is not None:      # fixed-point dynamics probe:
                slg = _free_g["v"]             # free per-(sample,frame) logits
            else:
                feats = early_hidden(model, ids,
                                     torch.stack([p_[0] for p_ in parts]),
                                     args.feat_layer)
                qf, ff, fv = frame_feats(feats, recs, S)
                slg = selector(qf, ff, fv)                   # [B, F]
            for b, r in enumerate(recs):
                nfr = len(r["blocks"])
                if soft:
                    g = torch.sigmoid(slg[b, :nfr])
                    if args.st_vertex:
                        gh = (g > 0.5).float()
                        g = gh + g - g.detach()   # forward hard, backward soft
                        masks_l.append(repack_mask(parts[b][0], parts[b][1],
                                                   parts[b][2], g, K, dtype))
                    else:
                        masks_l.append(log_gate_mask(parts[b][0], parts[b][1],
                                                     parts[b][2], g, K, dtype))
                    pos2_l.append(soft_repack_positions(r, g, S, args.device))
                    keeps.append(g)              # live — ce_of adds lam * mean(g)
                else:
                    kb = (slg[b, :nfr] > 0).float()
                    if kb.sum() == 0:
                        kb = torch.ones_like(kb)
                    masks_l.append(repack_mask(parts[b][0], parts[b][1],
                                               parts[b][2], kb, K, dtype))
                    pos2_l.append(repack_positions(r, kb > 0.5, S,
                                                   args.device).float())
                    keeps.append(kb)
            masks2 = torch.stack(masks_l)
            pos2 = torch.stack(pos2_l)
        elif args.arm in ("AS4", "AS4M"):
            # DIFFERENTIABLE select-and-repack: no labels, no aux, no threshold —
            # CE alone. Gates from UNDETACHED mass (the gradient also trains the
            # selecting attention); positions = cumsum of gates (AS4) so the
            # positional payoff is finally differentiable. AS4M = same gates but
            # positions from detached hard keeps — the blindness ablation.
            for b, r in enumerate(recs):
                nfr = len(r["blocks"])
                mn = mass[b, :nfr] * nfr
                if soft:
                    g = gate(mn)
                    if os.environ.get("AS4_MASK") == "hard":   # bisect probe
                        kbm = (mn.detach() >= gate.a.detach()).float()
                        if kbm.sum() == 0:
                            kbm = torch.ones_like(kbm)
                        masks_l.append(repack_mask(parts[b][0], parts[b][1],
                                                   parts[b][2], kbm, K, dtype))
                    else:
                        masks_l.append(log_gate_mask(parts[b][0], parts[b][1],
                                                     parts[b][2], g, K, dtype))
                    if args.arm == "AS4":
                        pos2_l.append(soft_repack_positions(r, g, S, args.device))
                    else:
                        kb = (mn.detach() >= gate.a.detach()).float()
                        if kb.sum() == 0:
                            kb = torch.ones_like(kb)
                        pos2_l.append(repack_positions(r, kb > 0.5, S,
                                                       args.device).float())
                    keeps.append(g.detach())
                else:                                  # deployment: exact hard repack
                    kb = (mn >= gate.a).float().detach()
                    if kb.sum() == 0:
                        kb = torch.ones_like(kb)
                    masks_l.append(repack_mask(parts[b][0], parts[b][1],
                                               parts[b][2], kb, K, dtype))
                    pos2_l.append(repack_positions(r, kb > 0.5, S,
                                                   args.device).float())
                    keeps.append(kb)
            masks2 = torch.stack(masks_l)
            pos2 = torch.stack(pos2_l)
        else:
            for b, r in enumerate(recs):
                nfr = len(r["blocks"])
                kb = (mass[b, :nfr].detach() >= args.as_alpha / nfr).float()
                if kb.sum() == 0:
                    kb = torch.ones_like(kb)          # never drop everything
                keeps.append(kb)
                masks_l.append(repack_mask(parts[b][0], parts[b][1], parts[b][2],
                                           kb, K, dtype))
                pos2_l.append(repack_positions(r, kb > 0.5, S, args.device))
            masks2 = torch.stack(masks_l)
            pos2 = torch.stack(pos2_l)
        if args.arm in ("AS4", "AS4M", "AS5"):
            pe2 = soft_rope(rotary_, pos2, h.dtype)     # grad flows through pos2
            pos2 = pos2.detach().round().clamp(min=0).long()
        else:
            cos2, sin2 = rotary_(h, pos2)
            pe2 = (cos2.to(h.dtype), sin2.to(h.dtype))
        for li in range(args.as_layer, len(layers_)):
            with sdpa_kernel(SDPA_BACKENDS):
                out = layers_[li](h, attention_mask=masks2, position_ids=pos2,
                                  position_embeddings=pe2)
            h = out[0] if isinstance(out, tuple) else out
        rows = norm_(h[torch.arange(B, device=args.device), ans])
        sel_sig = tmass if args.arm == "AS3T" else mass
        if os.environ.get("OSQ") and not soft:
            _osq.update(masks=masks2, pe2=pe2, layer=layers_[args.as_layer])
        return head_(rows).float(), sel_sig, keeps

    def batch_logits(recs, K, tau=1.0, mode="hard"):
        if args.arm in ("AS", "AS2", "AS3", "AS3T", "OTOK", "AST", "ASTH", "AS4", "AS4M", "AS5"):
            lg, mass, keeps = as_forward(recs, K, soft=(mode != "hard"))
            return lg, keeps, mass
        ids, valid, ans, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
        if args.arm == "SP":   # shared-positions single-pass: blockwise mask + as-if-
            bases, templates = batch_mask_parts(recs, S, args.device)   # first frames
            g1 = torch.ones(len(recs), 1, device=args.device)
            masks_sp = assemble_batch_masks(bases, templates, g1, K, dtype)
            pos = torch.stack([shared_positions(r, S, args.device) for r in recs])
            out = forward_answer_logits(model, ids, ans, lambda li: masks_sp,
                                        pos_ids=pos)
            return out, None, None
        parts = [rel_mask_parts(r, S, args.device) for r in recs]
        keeps, lg = keeps_for(recs, ids, parts, tau, mode)
        masks = torch.stack([repack_mask(parts[b][0], parts[b][1], parts[b][2],
                                         keeps[b], K, dtype)
                             for b in range(len(recs))])
        pos = torch.stack([repack_positions(recs[b], keeps[b].detach() > 0.5, S,
                                            args.device)
                           for b in range(len(recs))])
        out = forward_answer_logits(model, ids, ans, lambda li: masks, pos_ids=pos)
        return out, keeps, lg

    @torch.no_grad()
    def greedy_em(recs, batch=16, max_new=2):
        hits = n = 0
        for i in range(0, len(recs), batch):
            chunk = [dict(r) for r in recs[i : i + batch]]
            done = [False] * len(chunk)
            texts = [""] * len(chunk)
            for _ in range(max_new):
                lg, _, _ = batch_logits(chunk, MASK_MIN, mode="hard")
                for b, t in enumerate(lg.argmax(-1).tolist()):
                    if done[b]:
                        continue
                    s = tok.decode([t]).strip()
                    if s.isdigit():
                        texts[b] += s
                        chunk[b]["ids"] = chunk[b]["ids"] + [t]
                        chunk[b]["seq"] += 1
                    else:
                        done[b] = True
                if all(done):
                    break
            for b, r in enumerate(recs[i : i + batch]):
                hits += int(texts[b].isdigit() and int(texts[b]) == r["gold"])
                n += 1
        return hits / max(n, 1)

    @torch.no_grad()
    def selection_stats(recs):
        if selector is None and not (args.eval_only and args.eval_regime):
            return None
        ids, valid, _, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
        parts = [rel_mask_parts(r, S, args.device) for r in recs]
        keeps, _ = keeps_for(recs, ids, parts, 1.0, "hard")
        kr = kd = nr = nd = 0.0
        for b, r in enumerate(recs):
            for i, c in enumerate(r["frame_chars"]):
                v = float(keeps[b][i])
                if c == r["q_char"]:
                    kr += v; nr += 1
                else:
                    kd += v; nd += 1
        return kr / max(nr, 1), kd / max(nd, 1)

    _pl_cache = {}

    @torch.no_grad()
    def pseudo_labels(recs):
        """[B, F]: 1[logit(' yes') > logit(' no')] for 'Is this frame relevant to the
        question?' — the frozen model as its own relevance judge (LoRA is zero-init
        during warm = exactly frozen). No gold labels used."""
        yes = tok(" yes", add_special_tokens=False).input_ids[0]
        no = tok(" no", add_special_tokens=False).input_ids[0]
        out = []
        for r in recs:
            if r["sid"] in _pl_cache:
                out.append(_pl_cache[r["sid"]])
                continue
            minis = []
            q = tok.decode(r["ids"][: r["prefix_end"]]).split("Question:")[-1].strip()
            for a, b in r["blocks"]:
                line = tok.decode(r["ids"][a:b]).strip()
                t = (f"Question: {q}\nFrame: {line}\nIs this frame relevant to the "
                     f"question? Answer yes or no:")
                minis.append({"ids": tok(t, add_special_tokens=False).input_ids,
                              "blocks": [], "fin": 0, "prefix_end": 0})
            for m_ in minis:
                m_["seq"] = len(m_["ids"])
            lab = []
            for j in range(0, len(minis), 64):
                chunk = minis[j : j + 64]
                cids, _, cans, CS = pad_batch(chunk, tok.pad_token_id or 0,
                                              args.device)
                from masks import layout_mask_parts
                cb = torch.stack([layout_mask_parts(c, CS, args.device)[0]
                                  for c in chunk]).unsqueeze(1).to(dtype)
                lg = forward_answer_logits(model, cids, cans, lambda li: cb)
                lab += (lg[:, yes] > lg[:, no]).float().tolist()
            _pl_cache[r["sid"]] = torch.tensor(lab, device=args.device)
            out.append(_pl_cache[r["sid"]])
        return out

    # ------------------------------------------------------------------ eval-only
    if args.eval_only:
        assert args.eval_roots
        if args.ckpt:
            _cli_alpha, _cli_layer = args.as_alpha, args.as_layer
            ck = torch.load(args.ckpt, map_location="cpu")
            args.arm = ck["arm"]
            args.keep_margin = ck["config"].get("keep_margin", 0.0)
            args.feat_layer = ck["config"].get("feat_layer", 2)
            args.as_layer = ck["config"].get("as_layer", 16)
            args.as_alpha = ck["config"].get("as_alpha", 0.5)
            args.ast_alpha = ck["config"].get("ast_alpha", 0.25)
            if args.eval_regime == "attn":   # external-selection eval: the CLI
                args.as_alpha = _cli_alpha   # threshold/layer must win over the
                args.as_layer = _cli_layer   # ckpt's trained-arm values
            # (as_topk is never stored in old ckpts; CLI value survives)
            args.crowded = ck["config"].get("crowded", False)
            args.aux_labels = ck["config"].get("aux_labels", "evid")
            args.keep_heads = ck["config"].get("keep_heads", False)
            if ck.get("gate") is not None:
                gate.load_state_dict(ck["gate"])
            args.asth_beta = ck["config"].get("asth_beta", 0.1)
            args.head_cuts = ck["config"].get("head_cuts", False)
            if ck["config"].get("sel_arch") == "pairwise" and selector is not None:
                selector = PairSelector(D, seed=args.seed).to(args.device)
            lora = attach_lora(model.model.layers, 0, rank=ck["rank"],
                               alpha=ck["alpha"], device=args.device, state=ck["lora"])
            if ck.get("selector") is not None:
                selector.load_state_dict(ck["selector"])
            else:
                selector = None
        elif args.lora_ckpt:
            rk = torch.load(args.lora_ckpt, map_location="cpu")
            lora = attach_lora(model.model.layers, 0, rank=rk["rank"],
                               alpha=rk["alpha"], device=args.device, state=rk["lora"])
            selector = None
            assert args.eval_regime, "external-LoRA eval needs --eval-regime"
        else:                       # fully FROZEN eval (zero-init LoRA = no-op)
            lora = attach_lora(model.model.layers, 0, rank=args.rank,
                               alpha=args.alpha, device=args.device)
            selector = None
            assert args.eval_regime, "frozen eval needs --eval-regime"
        for p in lora.parameters():
            p.requires_grad_(False)
        out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
        out.mkdir(parents=True, exist_ok=True)
        if os.environ.get("OSQ"):
            for spec in [x for x in args.eval_roots.replace("+", ",").split(",")
                         if x]:
                root, _, lim = spec.partition("=")
                recs = (prep_root_crowded if args.crowded else prep_root_rel)(
                    tok, Path(root), 12, seed=args.seed)
                lg, _, keeps = as_forward(recs, MASK_MIN)
                gold = torch.tensor(
                    [digit_ids[r["gold"]] for r in recs], device=args.device)
                lg[torch.arange(len(recs), device=args.device), gold].sum(
                    ).backward()
                g = _osq["h"].grad
                # influence share of evidence frames at the as_layer boundary
                sh = []
                for b, r in enumerate(recs):
                    tot = rl = 0.0
                    for i, (a0, b0) in enumerate(r["blocks"]):
                        v = float(g[b, a0:b0].norm())
                        tot += v
                        if r["frame_chars"][i] == r["q_char"]:
                            rl += v
                    sh.append(rl / max(tot, 1e-9))
                # mask-aware answer-row fan-in on the POST-repack geometry
                lay, pe2, masks = _osq["layer"], _osq["pe2"], _osq["masks"]
                hn = lay.input_layernorm(_osq["h"].detach())
                at = lay.self_attn
                hd = pe2[0].shape[-1]
                nh = at.q_proj.weight.shape[0] // hd
                B2, S2, _ = hn.shape
                q = at.q_proj(hn).view(B2, S2, nh, hd).transpose(1, 2)
                k = at.k_proj(hn).view(B2, S2, -1, hd).transpose(1, 2)

                def _rope(x):
                    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
                    return (x * pe2[0].unsqueeze(1)
                            + torch.cat([-x2, x1], -1) * pe2[1].unsqueeze(1))

                q, k = _rope(q), _rope(k)
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                ans_ = torch.tensor([r["seq"] - 1 for r in recs],
                                    device=args.device)
                qa = q[torch.arange(B2, device=args.device), :, ans_]
                sc = (qa.unsqueeze(2) * k).sum(-1).float() / hd ** 0.5
                sc = sc + masks[torch.arange(B2, device=args.device), 0,
                                ans_, :].float().unsqueeze(1)
                w = torch.softmax(sc, dim=-1)
                keff, kcnt = [], []
                for b, r in enumerate(recs):
                    fc = [t for (a0, b0) in r["blocks"] for t in range(a0, b0)]
                    wm = w[b].mean(0)[fc]
                    a = wm / max(float(wm.sum()), 1e-9)
                    keff.append(float(1.0 / max(float((a * a).sum()), 1e-9)))
                    kcnt.append(float(keeps[b].sum()))
                nfr = sum(len(r["blocks"]) for r in recs) / len(recs)
                print(json.dumps({
                    "root": root, "arm": args.arm, "frames": nfr,
                    "kept": sum(kcnt) / len(kcnt),
                    "fanin": sum(keff) / len(keff),
                    "infl_share": sum(sh) / len(sh)}), flush=True)
                _osq.clear()
            lora.remove()
            return 0
        rows = []
        for spec in [s for s in args.eval_roots.replace("+", ",").split(",") if s]:
            root, _, lim = spec.partition("=")
            recs = (prep_root_crowded if args.crowded else prep_root_rel)(
                tok, Path(root), int(lim) if lim else 400, seed=args.seed)
            em = greedy_em(recs)
            st = selection_stats(recs[:64])
            _, maj = majority_baseline([r["gold"] for r in recs])
            rows.append({"root": root, "n": len(recs), "em": em, "majority": maj,
                         "arm": args.arm, "regime": args.eval_regime,
                         "keep_rel": st[0] if st else None,
                         "keep_dist": st[1] if st else None})
            print(f"[{root}] n={len(recs)} EM {em:.3f} (maj {maj:.3f})"
                  + (f" keep(rel) {st[0]:.3f} keep(dist) {st[1]:.3f}" if st else ""))
            (out / "report.txt").write_text(json.dumps(rows, indent=2) + "\n")
        lora.remove()
        return 0

    # ------------------------------------------------------------------ training
    run = f"arm{args.arm}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.output) / run
    out.mkdir(parents=True, exist_ok=True)
    tele = open(out / "telemetry.jsonl", "a")
    def prep_multi(spec, limit, seed):
        recs = []
        # '+' separates roots (comma dies in sbatch --export); each may carry =LIMIT
        _prep = prep_root_crowded if args.crowded else prep_root_rel
        for part in [x for x in spec.replace("+", ",").split(",") if x.strip()]:
            root, _, lim = part.partition("=")
            recs += _prep(tok, Path(root), int(lim) if lim else limit, seed=seed)
        random.Random(seed + 5).shuffle(recs)
        return recs

    tr_s = prep_multi(args.short_root, args.limit, args.seed)
    tr_l = prep_multi(args.long_root, args.long_limit, args.seed) \
        if args.arm in ("A2mix", "LS", "LA", "LAF", "AS", "AS2", "AS3", "AS3T", "OTOK", "AST", "ASTH", "AS4", "AS4M", "AS5") else []
    ev_s = prep_multi(args.dev_root, args.dev_limit, 1234)
    ev_l = prep_multi(args.long_dev_root, args.dev_limit, 1234)
    print(f"data: short {len(tr_s)} long {len(tr_l)} dev {len(ev_s)}/{len(ev_l)}")

    assert_lora_targets(model)
    if args.init_ckpt:
        _ick = torch.load(args.init_ckpt, map_location="cpu")
        lora = attach_lora(model.model.layers, 0, rank=_ick["rank"],
                           alpha=_ick["alpha"], device=args.device,
                           state=_ick["lora"])
        print(f"[init] LoRA from {args.init_ckpt}", flush=True)
    else:
        lora = attach_lora(model.model.layers, 0, rank=args.rank, alpha=args.alpha,
                           device=args.device)
    if args.freeze_readout:
        nfroz = 0
        for (li, _nm), (A, B) in lora.params.items():
            if li >= args.as_layer:
                A.requires_grad_(False); B.requires_grad_(False); nfroz += 2
        print(f"[freeze] readout LoRA frozen: {nfroz} tensors "
              f"(layers >= {args.as_layer})", flush=True)
    opt_lora = torch.optim.Adam([p for p in lora.parameters()
                                 if p.requires_grad] or lora.parameters(),
                                lr=args.lr_lora)
    opt_sel = (torch.optim.Adam(selector.parameters(), lr=args.lr_sel)
               if selector is not None else None)
    opt_gate = (torch.optim.Adam(gate.parameters(), lr=args.lr_gate)
                if args.arm in ("AS4", "AS4M") else None)

    # keep-all parity: repacked positions must be EXACTLY arange
    ids0, _, _, S0 = pad_batch(tr_s[:1], tok.pad_token_id or 0, args.device)
    pos0 = repack_positions(tr_s[0], torch.ones(len(tr_s[0]["blocks"])), S0,
                            args.device)
    assert torch.equal(pos0[: tr_s[0]["seq"]],
                       torch.arange(tr_s[0]["seq"], device=args.device)), \
        "PARITY: keep-all positions != arange"

    def ce_of(recs, K, tau, mode):
        tgt = torch.tensor([digit_ids[r["gold"]] for r in recs], device=args.device)
        lg, keeps, sel_lg = batch_logits(recs, K, tau=tau, mode=mode)
        loss = F.cross_entropy(lg, tgt)
        if args.arm == "AS5" and args.gate_lam > 0 and mode != "hard":
            loss = loss + args.gate_lam * torch.cat(keeps).mean()
        if args.arm == "AS3T" and args.as_lam > 0:
            tmass = sel_lg                        # [B, S] token mass
            aux = 0.0
            for b, r in enumerate(recs):
                S_ = tmass.shape[1]
                lab = clause_token_labels(r, S_, args.device, args.aux_labels)
                fr0, fr1 = r["blocks"][0][0], r["blocks"][-1][1]
                nft = fr1 - fr0
                rl = torch.log(tmass[b, fr0:fr1].clamp(min=1e-8) * nft)
                aux = aux + F.binary_cross_entropy_with_logits(
                    rl, lab[fr0:fr1], pos_weight=torch.tensor(6.0,
                                                              device=args.device))
            loss = loss + args.as_lam * aux / len(recs)
        elif args.arm in ("AS", "AS2", "AS3", "AST", "ASTH") and args.as_lam > 0:
            mass = sel_lg                        # [B, F_max]
            pls = pseudo_labels(recs)
            aux = 0.0
            for b, r in enumerate(recs):
                nfr = len(r["blocks"])
                rel_logit = torch.log(mass[b, :nfr].clamp(min=1e-8) * nfr)
                aux = aux + F.binary_cross_entropy_with_logits(
                    rel_logit, pls[b], pos_weight=torch.tensor(4.0,
                                                               device=args.device))
            loss = loss + args.as_lam * aux / len(recs)
        return loss, sel_lg


    # LA warm start: class-balanced BCE on derivable keep labels (features only).
    # pos_weight ~4 (only ~1/5 of frames are relevant): without it the bias races to
    # the label prior and drops EVERYTHING before separation develops (measured
    # 2026-08-18: mean logit +1 -> -1.9 in 4 steps, std 0.03). Warm until the
    # keep(rel)-keep(dist) gap > 0.8 or 3 epochs.
    if args.arm in ("LA", "LAF") or (args.arm == "AS5" and args.warm_sel):
      warm_data = list(tr_l)
      warm_dev = list(ev_l[:64])
      if args.warm_extra_root:
          warm_data += prep_root_rel(tok, Path(args.warm_extra_root), 900,
                                     seed=args.seed)
          random.Random(args.seed).shuffle(warm_data)
          warm_dev += prep_root_rel(
              tok, Path(args.warm_extra_root.replace("/warm", "/warmdev")), 64,
              seed=1234)
      for wep in range(1, 4):
        t0 = time.time()
        for i in range(0, len(warm_data), args.batch):
            recs = warm_data[i : i + args.batch]
            ids, valid, _, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
            parts = [rel_mask_parts(r, S, args.device) for r in recs]
            bases = torch.stack([p_[0] for p_ in parts])
            with torch.no_grad():
                feats = early_hidden(model, ids, bases, args.feat_layer)
            q, f, fv = frame_feats(feats, recs, S)
            lg = selector(q, f, fv)
            if args.warm_labels == "pseudo":
                pls = pseudo_labels(recs)
                lbl = torch.stack([torch.cat([pl, torch.zeros(
                    f.shape[1] - len(pl), device=args.device)]) for pl in pls])
                if i == 0 and wep == 1:  # diagnostic: pseudo-vs-oracle agreement
                    agr = sum(float((pl == oracle_keep(r, args.device)).float().mean())
                              for pl, r in zip(pls, recs)) / len(recs)
                    print(f"[pseudo] label agreement with oracle: {agr:.3f}",
                          flush=True)
            else:
                lbl = torch.stack([torch.cat([oracle_keep(r, args.device),
                                              torch.zeros(f.shape[1] - len(r["blocks"]),
                                                          device=args.device)])
                                   for r in recs])
            loss = F.binary_cross_entropy_with_logits(
                lg[fv], lbl[fv], pos_weight=torch.tensor(4.0, device=args.device))
            opt_sel.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), args.clip_sel)
            opt_sel.step()
        st = selection_stats(warm_dev)
        print(f"[warm ep{wep}] ({time.time()-t0:.0f}s) keep(rel) {st[0]:.3f} "
              f"keep(dist) {st[1]:.3f} gap {st[0]-st[1]:+.3f}", flush=True)
        if st[0] - st[1] > 0.8:
            break
      # calibrate the recall margin on dev: smallest shift with keep(rel) >= 0.99
      # (dropping a relevant frame is fatal — (1-p)^N; keeping distractors is cheap)
      for m in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
          args.keep_margin = m
          st = selection_stats(warm_dev)
          if st[0] >= 0.99:
              break
      print(f"[calib] keep_margin {args.keep_margin} -> keep(rel) {st[0]:.3f} "
            f"keep(dist) {st[1]:.3f}", flush=True)

    if os.environ.get("AS5_PROBE"):
        sl = tr_s[:8]
        loss, _ = ce_of(sl, SOFT_FORBID, 1.0, "train")
        gl0, args.gate_lam = args.gate_lam, 0.0
        ce_only, _ = ce_of(sl, SOFT_FORBID, 1.0, "train")
        args.gate_lam = gl0
        print(f"[p5] loss {float(loss):.4f} ce {float(ce_only):.4f} "
              f"lam-term {float(loss - ce_only):.4f}", flush=True)
        loss.backward()
        for n_, p_ in selector.named_parameters():
            g_ = p_.grad
            print(f"[p5] grad {n_}: "
                  f"{'None' if g_ is None else f'{float(g_.norm()):.3e}'}",
                  flush=True)
        opt_sel.zero_grad()
        for _step in range(30):
            loss, _ = ce_of(sl, SOFT_FORBID, 1.0, "train")
            opt_sel.zero_grad(); loss.backward(); opt_sel.step()
        with torch.no_grad():
            _, ks, _ = batch_logits(sl, SOFT_FORBID, mode="train")
            km = float(torch.cat(ks).mean())
        print(f"[p5] after 30 steps: mean soft keep {km:.4f} (init ~0.73)",
              flush=True)
        # FREE-GATE fixed-point test: one logit per (sample, frame), NO shared
        # selector — does CE + lam separate evidence from junk when parameters
        # are disentangled? (probe of mechanism vi)
        if os.environ.get("AS5_ST"):
            args.st_vertex = True
        fb = tr_s[:8] if os.environ.get("AS5_ST") else tr_l[:4]
        Fm = max(len(r["blocks"]) for r in fb)
        th = torch.full((len(fb), Fm), 1.0, device=args.device,
                        requires_grad=True)
        opt_f = torch.optim.Adam([th], lr=3e-2)
        tgt_f = torch.tensor([digit_ids[r["gold"]] for r in fb],
                             device=args.device)
        for lam_f in ((0.3, 1.0, 3.0) if os.environ.get("AS5_ST")
                      else (3.0, 6.0)):
            with torch.no_grad():
                th.fill_(1.0)
            for it in range(301):
                _free_g["v"] = th
                lg, _, ks = as_forward(fb, SOFT_FORBID, soft=True)
                ce_f = F.cross_entropy(lg, tgt_f)
                loss_f = ce_f + lam_f * torch.cat(
                    [torch.sigmoid(th[b, :len(r["blocks"])])
                     for b, r in enumerate(fb)]).mean()
                opt_f.zero_grad(); loss_f.backward(); opt_f.step()
                if it % 100 == 0:
                    ev, jk = [], []
                    with torch.no_grad():
                        for b, r in enumerate(fb):
                            gg = torch.sigmoid(th[b, :len(r["blocks"])])
                            for i, c in enumerate(r["frame_chars"]):
                                (ev if c == r["q_char"] else jk).append(
                                    float(gg[i]))
                    print(f"[p5free] lam {lam_f} it {it}: ce {float(ce_f):.3f} "
                          f"g(rel) {sum(ev)/len(ev):.3f} "
                          f"g(junk) {sum(jk)/len(jk):.3f}", flush=True)
            _free_g["v"] = None
        # DUAL-ASCENT vertex test: minimize mean(g) s.t. CE <= eps, K=-6 so
        # dropped frames keep a live reopening gradient (K=-30 ratchet: e^-30)
        if os.environ.get("AS5_DUAL"):
            fb2 = tr_s[:8]
            Fm2 = max(len(r["blocks"]) for r in fb2)
            th2 = torch.full((len(fb2), Fm2), 1.0, device=args.device,
                             requires_grad=True)
            opt2 = torch.optim.Adam([th2], lr=3e-2)
            tgt2 = torch.tensor([digit_ids[r["gold"]] for r in fb2],
                                device=args.device)
            mu, eps, eta = 0.0, 0.05, 0.1
            for it in range(601):
                _free_g["v"] = th2
                lg, _, _ = as_forward(fb2, -6.0, soft=True)
                ce2 = F.cross_entropy(lg, tgt2)
                mg = torch.cat([torch.sigmoid(th2[b, :len(r["blocks"])])
                                for b, r in enumerate(fb2)]).mean()
                (mu * ce2 + mg).backward()
                opt2.step(); opt2.zero_grad()
                mu = min(max(0.0, mu + eta * (float(ce2) - eps)), 100.0)
                if it % 100 == 0:
                    ev, jk = [], []
                    with torch.no_grad():
                        for b, r in enumerate(fb2):
                            gg = torch.sigmoid(th2[b, :len(r["blocks"])])
                            for i, c in enumerate(r["frame_chars"]):
                                (ev if c == r["q_char"] else jk).append(
                                    float(gg[i]))
                    print(f"[p5dual] it {it}: ce {float(ce2):.3f} mu {mu:.2f} "
                          f"g(rel) {sum(ev)/len(ev):.3f} "
                          f"g(junk) {sum(jk)/len(jk):.3f}", flush=True)
            _free_g["v"] = None
            return 0

        # per-frame CE force on the gates: |dCE/dg| for evidence vs junk frames
        # (rel labels used ONLY as probe instrumentation). If the bands separate,
        # a lam inside the band makes selection the fixed point of CE + lam.
        for tag, batch_ in (("N=8", tr_s[:8]), ("N=128", tr_l[:4])):
            tgt = torch.tensor([digit_ids[r["gold"]] for r in batch_],
                               device=args.device)
            lg, ks, _ = batch_logits(batch_, SOFT_FORBID, mode="train")
            ce = F.cross_entropy(lg, tgt)
            gr = torch.autograd.grad(ce, list(ks), allow_unused=True)
            ev, jk = [], []
            for b, r in enumerate(batch_):
                if gr[b] is None:
                    continue
                for i, c in enumerate(r["frame_chars"]):
                    (ev if c == r["q_char"] else jk).append(
                        abs(float(gr[b][i])))
            import statistics as st_
            for nm2, v2 in (("evid", ev), ("junk", jk)):
                if v2:
                    print(f"[p5] {tag} |dCE/dg| {nm2}: median "
                          f"{st_.median(v2):.3e} p90 "
                          f"{sorted(v2)[int(.9 * len(v2))]:.3e} "
                          f"max {max(v2):.3e} n={len(v2)}", flush=True)
        return 0

    if os.environ.get("AS4_PROBE"):
        sl = tr_s[:16]
        arm0, al0 = args.arm, args.as_lam
        args.as_lam = 0.0
        for nm, arm2, md2, envm in [
                ("AS3-hard (ref)", "AS3", "hard", ""),
                ("AS4-hard", "AS4", "hard", ""),
                ("AS4-soft FULL", "AS4", "train", ""),
                ("AS4M-soft (hard pos)", "AS4M", "train", ""),
                ("AS4-soft (hard mask)", "AS4", "train", "hard")]:
            args.arm = arm2
            os.environ["AS4_MASK"] = envm
            with torch.no_grad():
                loss, _ = ce_of(sl, SOFT_FORBID, 1.0, md2)
            print(f"[probe] {nm:22s} ce {float(loss):.4f}", flush=True)
        os.environ["AS4_MASK"] = ""
        args.arm = "AS4"
        with torch.no_grad():
            _, mass, keeps = as_forward(sl[:8], SOFT_FORBID, soft=True)
        for b in range(4):
            nfr = len(sl[b]["blocks"])
            mn = mass[b, :nfr] * nfr
            rel = [i for i, c in enumerate(sl[b]["frame_chars"])
                   if c == sl[b]["q_char"]]
            mr = float(mn[rel].mean()) if rel else -1
            print(f"[probe] mass*n s{b}: rel {mr:.2f} all-mean "
                  f"{float(mn.mean()):.2f} max {float(mn.max()):.2f} "
                  f"g {[round(float(x), 2) for x in gate(mn)[:6]]}", flush=True)
        loss, _ = ce_of(sl[:8], SOFT_FORBID, 1.0, "train")
        loss.backward()
        gn = sum(float(p_.grad.norm()) for p_ in lora.parameters()
                 if p_.grad is not None)
        print(f"[probe] soft backward: lora gradnorm {gn:.3e} "
              f"gate.a grad {float(gate.a.grad or 0):.3e}", flush=True)
        args.arm, args.as_lam = arm0, al0
        return 0

    best, best_ep, bad, gstep = -1.0, -1, 0, 0
    total = ((len(tr_s) + len(tr_l)) // args.batch + 1) * args.epochs
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        random.Random(ep).shuffle(tr_s)
        random.Random(ep + 99).shuffle(tr_l)
        ce_sum, nb, li = 0.0, 0, 0
        for i in range(0, len(tr_s), args.batch):
            tau = tau_schedule(gstep, total, args.tau0, args.tau1, args.tau_hold)
            # LoRA step on a SHORT batch (selector hard + detached)
            with torch.no_grad():
                pass
            md = "train" if args.arm in ("AS4", "AS4M", "AS5") else "hard"
            loss, _ = ce_of(tr_s[i : i + args.batch], SOFT_FORBID, tau, md)
            opt_lora.zero_grad()
            if opt_gate:
                opt_gate.zero_grad()
            if args.arm == "AS5":
                opt_sel.zero_grad()
            loss.backward()
            opt_lora.step()
            if opt_gate:
                opt_gate.step()
            if args.arm == "AS5" and not args.sel_freeze_after_warm:
                torch.nn.utils.clip_grad_norm_(selector.parameters(),
                                               args.clip_sel)
                opt_sel.step()
            ce_sum += float(loss.detach()); nb += 1; gstep += 1
            # selector (or A2mix LoRA) step on a LONG batch
            if tr_l:
                recs_l = tr_l[li : li + args.batch] or tr_l[: args.batch]
                li = (li + args.batch) % max(len(tr_l) - args.batch, 1)
                if args.arm in ("A2mix", "LAF", "AS", "AS2", "AS3", "AS3T", "OTOK", "AST", "ASTH", "AS4", "AS4M"):
                    # LAF: selector FROZEN post-warm (ST cannot see position-repack
                    # benefits — measured: LS stuck at keep-all, LA collapsed);
                    # long batches still train the LoRA under the fixed selector.
                    # AS4/AS4M: soft mode; the gate steps here too.
                    loss_l, _ = ce_of(recs_l, SOFT_FORBID, tau, md)
                    opt_lora.zero_grad()
                    if opt_gate:
                        opt_gate.zero_grad()
                    loss_l.backward()
                    opt_lora.step()
                    if opt_gate:
                        opt_gate.step()
                else:
                    loss_l, _ = ce_of(recs_l, SOFT_FORBID, tau, "train")
                    opt_sel.zero_grad(); opt_lora.zero_grad()
                    loss_l.backward()
                    if not args.sel_freeze_after_warm:
                        torch.nn.utils.clip_grad_norm_(selector.parameters(),
                                                       args.clip_sel)
                        opt_sel.step()      # LoRA grads from long batches discarded
                gstep += 1
        em_s = greedy_em(ev_s)
        em_l = greedy_em(ev_l)
        st = selection_stats(ev_l[:64])
        line = (f"[ep {ep}] ce {ce_sum / max(nb, 1):.4f} | dev8 {em_s:.3f} dev32 "
                f"{em_l:.3f} | {time.time() - t0:.0f}s")
        rec_t = {"ep": ep, "dev8": em_s, "dev32": em_l}
        if st:
            line += f" | keep(rel) {st[0]:.3f} keep(dist) {st[1]:.3f} gap {st[0]-st[1]:+.3f}"
            rec_t.update({"keep_rel": st[0], "keep_dist": st[1]})
        if args.arm in ("AS4", "AS4M", "AS5"):
            with torch.no_grad():
                _, ks, _ = batch_logits(ev_l[:48], MASK_MIN, mode="hard")
            kr = kd = nr = nd = 0.0
            for b, r in enumerate(ev_l[:48]):
                for i2, c in enumerate(r["frame_chars"]):
                    v = float(ks[b][i2])
                    if c == r["q_char"]:
                        kr += v; nr += 1
                    else:
                        kd += v; nd += 1
            ga, gt = float(gate.a), float(gate.log_tau.exp())
            line += (f" | a {ga:.3f} tau {gt:.3f} keep(rel) {kr/max(nr,1):.3f} "
                     f"keep(dist) {kd/max(nd,1):.3f}")
            rec_t.update({"gate_a": ga, "gate_tau": gt,
                          "keep_rel": kr / max(nr, 1), "keep_dist": kd / max(nd, 1)})
        print(line, flush=True)
        tele.write(json.dumps(rec_t) + "\n"); tele.flush()
        sel_metric = em_l  # in-length for the selector's objective
        ck = {"arm": args.arm, "config": vars(args), "epoch": ep, "dev8": em_s,
              "dev32": em_l, "rank": args.rank, "alpha": args.alpha,
              "lora": lora.state(),
              "gate": gate.state_dict(),
              "selector": selector.state_dict() if selector is not None else None}
        torch.save(ck, out / "coformer_last.pt")
        if sel_metric > best:
            best, best_ep, bad = sel_metric, ep, 0
            torch.save(ck, out / "coformer_best.pt")
        else:
            bad += 1
        (out / "report.txt").write_text(
            f"{run}\nep {ep}: dev8 {em_s:.3f} dev32 {em_l:.3f} best32 {best:.3f} "
            f"@ep{best_ep}\n" + (f"keep(rel) {st[0]:.3f} keep(dist) {st[1]:.3f}\n"
                                 if st else ""))
        if bad >= args.patience and ep >= 4:
            break
    lora.remove()
    tele.close()
    print(f"DONE best dev32 {best:.3f} @ep{best_ep} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
