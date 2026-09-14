#!/usr/bin/env python3
"""Relational-mask trainer (condmask follow-up): per-(layer, head, FRAME) gates on the
filtered-counting task, where relevance is question-dependent and no static mask can be
optimal. Gold <= 8 at every N by construction (longer N = more distractors only), so
length transfer isolates distractor filtering from answer-range extrapolation.

Arms (LoRA r8 on all layers co-trained in EVERY arm):
  A2  full attention                     (control)
  B2  static: every frame isolated, tail reads all (old blockwise semantics)
  Gg  GLOBAL learned gates per (layer, head)       (granularity ablation, old semantics)
  G   RELATIONAL learned gates per (layer, head, frame), question x frame conditioned
  O   ORACLE: isolate exactly the distractor frames (upper bound for G)

Money metrics: (1) EM vs N per arm; (2) for G: P(cut | distractor frame) vs
P(cut | relevant frame) — the gap is direct evidence the gate learned relational
relevance; (3) O - A2 gap = the headroom relational masking could ever buy.

Smoke:
  HF_HOME=$HOME/.cache/huggingface python scripts/condmask/train_relmask.py --arm G \\
      --limit 64 --dev-limit 32 --epochs 1 --output outputs/_scratch/relmask_smoke
"""
from __future__ import annotations

import argparse
import json
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
from gatenet import GateNet, pool_embeddings, st_gumbel, tau_schedule
from masks import SOFT_FORBID, assemble_batch_masks, batch_mask_parts
from modeling import (assert_lora_targets, forward_answer_logits, load_text_model,
                      model_dims, pad_batch)
from relgate import (DynGate, RelGateNet, RelGateTF, assemble_dyn_masks,
                     assemble_rel_masks, early_hidden, frame_feats, oracle_gates,
                     prep_root_rel, rel_mask_parts)
from textdata import majority_baseline

ARMS = ("A2", "B2", "Gg", "G", "Gx", "Dyn", "O")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-root", default="data/mmred_filtered/seq_len_8/train")
    ap.add_argument("--dev-root", default="data/mmred_filtered/seq_len_8/dev")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--dev-limit", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr-gate", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--clip-gates", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--tau-hold", type=float, default=0.3)
    ap.add_argument("--init-bias", type=float, default=1.0)
    ap.add_argument("--gate-init", choices=("random", "blockwise", "open"),
                    default="random")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gate-feat-grad", action="store_true",
                    help="backprop THROUGH the feature pass (trained-through emitter "
                         "— the expressiveness ceiling of the mask interface)")
    ap.add_argument("--gate-feat-layer", type=int, default=2,
                    help="conditioning features = frozen backbone hidden state after "
                         "this many full-attention layers (0 = raw embeddings)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="outputs/relmask")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--eval-roots", default="",
                    help="'+'-separated root[=LIMIT] (commas die in sbatch --export)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok, model = load_text_model(args.model, device=args.device)
    L, H, D = model_dims(model)
    dtype = next(model.parameters()).dtype
    embed = model.model.embed_tokens
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    gate = None
    if args.arm == "G":
        gate = RelGateNet(D, L, H, hidden=args.hidden, init=args.gate_init,
                          init_bias=args.init_bias, seed=args.seed).to(args.device)
    elif args.arm == "Gx":
        gate = RelGateTF(D, L, H, init=args.gate_init, init_bias=args.init_bias,
                         seed=args.seed).to(args.device)
    elif args.arm == "Dyn":
        gate = DynGate(D, L, H, init=("open" if args.gate_init == "random"
                                      else args.gate_init),
                       init_bias=args.init_bias, seed=args.seed).to(args.device)
    elif args.arm == "Gg":
        gate = GateNet(D, L, H, hidden=args.hidden,
                       init=("blockwise" if args.gate_init == "blockwise" else "random"),
                       init_bias=args.init_bias, seed=args.seed,
                       norm_input=True).to(args.device)

    def batch_masks_fn(recs, ids, valid, S, K, tau, mode):
        """-> (layer_masks callable, aux dict with gate probs for telemetry)."""
        aux = {}
        if args.arm in ("A2", "B2", "Gg"):
            bases, templates = batch_mask_parts(recs, S, args.device)
            if args.arm == "A2":
                g = torch.zeros(len(recs), 1, device=args.device)
                return (lambda li: assemble_batch_masks(bases, templates, g, K, dtype)), aux
            if args.arm == "B2":
                g = torch.ones(len(recs), 1, device=args.device)
                return (lambda li: assemble_batch_masks(bases, templates, g, K, dtype)), aux
            lg = gate(pool_embeddings(embed(ids), valid))          # [B, L, H]
            g = st_gumbel(lg, tau, mode)
            aux["glogits"] = lg
            return (lambda li: assemble_batch_masks(bases, templates, g[:, li], K,
                                                    dtype)), aux
        if args.arm == "Dyn":
            parts = [rel_mask_parts(r, S, args.device) for r in recs]
            bases_c = torch.stack([p_[0] for p_ in parts])
            regions = torch.stack([p_[1] for p_ in parts])
            in_frame = torch.stack([p_[2] >= 0 for p_ in parts])
            def fn(li, h=None):
                lg = gate(h, li)                       # [B, S, H] from h^{l-1}
                gs = st_gumbel(lg, tau, mode)
                if mode == "train":
                    aux.setdefault("dyn_p", []).append(torch.sigmoid(lg.detach()))
                return assemble_dyn_masks(bases_c, regions, in_frame, gs, K, dtype)
            return fn, aux
        # per-frame arms: G / Gx (learned) / O (oracle)
        parts = [rel_mask_parts(r, S, args.device) for r in recs]
        if args.arm == "O":
            gates_f = [oracle_gates(r, L, H, args.device) for r in recs]  # [F,L,H]
        else:
            bases_c = torch.stack([p_[0] for p_ in parts])
            feats = early_hidden(model, ids, bases_c, args.gate_feat_layer,
                                 grad=args.gate_feat_grad)
            q, f, fv = frame_feats(feats, recs, S, detach=not args.gate_feat_grad)
            lg = gate(q, f, fv) if args.arm == "Gx" else gate(q, f)  # [B, F, L, H]
            gs = st_gumbel(lg, tau, mode)
            gates_f = [gs[b, : len(recs[b]["blocks"])] for b in range(len(recs))]
            aux["rel_logits"] = lg
        def fn(li):
            return torch.stack([
                assemble_rel_masks(parts[b][0], parts[b][1], parts[b][2],
                                   gates_f[b][:, li], K, dtype)
                for b in range(len(recs))])
        return fn, aux

    def batch_logits(recs, K, tau=1.0, mode="hard"):
        ids, valid, ans, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
        fn, aux = batch_masks_fn(recs, ids, valid, S, K, tau, mode)
        return forward_answer_logits(model, ids, ans, fn), aux

    @torch.no_grad()
    def greedy_em(recs, batch=16, max_new=2):
        hits = n = 0
        rel_stats = []
        for i in range(0, len(recs), batch):
            chunk = [dict(r) for r in recs[i : i + batch]]
            done = [False] * len(chunk)
            texts = [""] * len(chunk)
            for _ in range(max_new):
                lg, aux = batch_logits(chunk, MASK_MIN, mode="hard")
                if "rel_logits" in aux and not rel_stats:
                    pass
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
    def rel_gap(recs):
        """P(cut|distractor) vs P(cut|relevant) under hard gates — arm G only."""
        if args.arm == "Dyn":
            ids, valid, ans, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
            fn, aux = batch_masks_fn(recs, ids, valid, S, MASK_MIN, 1.0, "train")
            forward_answer_logits(model, ids, ans, fn)
            ps = torch.stack(aux["dyn_p"])            # [L, B, S, H]
            pd = pr = nd = nr = 0.0
            for b, r in enumerate(recs):
                for i, (a, e) in enumerate(r["blocks"]):
                    m = float(ps[:, b, a:e].mean())
                    if r["frame_chars"][i] != r["q_char"]:
                        pd += m; nd += 1
                    else:
                        pr += m; nr += 1
            return pd / max(nd, 1), pr / max(nr, 1)
        if args.arm not in ("G", "Gx"):
            return None
        ids, valid, _, S = pad_batch(recs, tok.pad_token_id or 0, args.device)
        parts = [rel_mask_parts(r, S, args.device) for r in recs]
        feats = early_hidden(model, ids, torch.stack([p_[0] for p_ in parts]),
                             args.gate_feat_layer)
        q, f, fv = frame_feats(feats, recs, S)
        p = torch.sigmoid(gate(q, f, fv) if args.arm == "Gx" else gate(q, f))
        pd, pr, nd, nr = 0.0, 0.0, 0, 0
        for b, r in enumerate(recs):
            for i, c in enumerate(r["frame_chars"]):
                m = float(p[b, i].mean())
                if c != r["q_char"]:
                    pd += m; nd += 1
                else:
                    pr += m; nr += 1
        return pd / max(nd, 1), pr / max(nr, 1)

    # ---------------------------------------------------------------- eval-only
    if args.eval_only:
        assert args.ckpt and args.eval_roots
        ck = torch.load(args.ckpt, map_location="cpu")
        # the ckpt, not the CLI, decides the arm (and hence the gate class/regime) —
        # evaluating a learned-gate ckpt under a CLI arm crashed/mis-masked (2026-08-18)
        args.arm = ck["arm"]
        gate = None
        if args.arm == "G":
            gate = RelGateNet(D, L, H, hidden=args.hidden, init=args.gate_init,
                              init_bias=args.init_bias, seed=args.seed).to(args.device)
        elif args.arm == "Gx":
            gate = RelGateTF(D, L, H, init=args.gate_init, init_bias=args.init_bias,
                             seed=args.seed).to(args.device)
        elif args.arm == "Dyn":
            gate = DynGate(D, L, H, init="open", init_bias=args.init_bias,
                           seed=args.seed).to(args.device)
        args.gate_feat_layer = ck["config"].get("gate_feat_layer", 2)
        args.gate_feat_grad = False
        lora = attach_lora(model.model.layers, 0, rank=ck["rank"], alpha=ck["alpha"],
                           device=args.device, state=ck["lora"])
        for p in lora.parameters():
            p.requires_grad_(False)
        if ck.get("gate") is not None:
            gate.load_state_dict(ck["gate"])
        import os
        out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for spec in [s for s in args.eval_roots.replace("+", ",").split(",") if s]:
            root, _, lim = spec.partition("=")
            recs = prep_root_rel(tok, Path(root), int(lim) if lim else 400,
                                 seed=args.seed)
            em = greedy_em(recs)
            _, maj = majority_baseline([r["gold"] for r in recs])
            gap = rel_gap(recs[:64])
            row = {"root": root, "n": len(recs), "em": em, "majority": maj,
                   "arm": ck["arm"],
                   "p_cut_dist": gap[0] if gap else None,
                   "p_cut_rel": gap[1] if gap else None}
            rows.append(row)
            print(f"[{root}] n={len(recs)} EM {em:.3f} (maj {maj:.3f})"
                  + (f" cut(dist) {gap[0]:.3f} cut(rel) {gap[1]:.3f}" if gap else ""))
            (out / "report.txt").write_text(json.dumps(rows, indent=2) + "\n")
        lora.remove()
        return 0

    # ---------------------------------------------------------------- training
    run = f"arm{args.arm}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.output) / run
    out.mkdir(parents=True, exist_ok=True)
    tele = open(out / "telemetry.jsonl", "a")
    tr = prep_root_rel(tok, Path(args.data_root), args.limit, seed=args.seed)
    ev = prep_root_rel(tok, Path(args.dev_root), args.dev_limit, seed=1234)
    assert tr and ev, "empty data"
    _, maj = majority_baseline([r["gold"] for r in ev])
    print(f"data: train {len(tr)} dev {len(ev)} (dev majority {maj:.3f}); "
          f"{L}x{H} heads; arm {args.arm}")
    fit_probe = tr[:256]

    assert_lora_targets(model)
    lora = attach_lora(model.model.layers, 0, rank=args.rank, alpha=args.alpha,
                       device=args.device)
    groups = [{"params": list(lora.parameters()), "lr": args.lr_lora}]
    if gate is not None:
        groups.insert(0, {"params": list(gate.parameters()), "lr": args.lr_gate})
    opt = torch.optim.Adam(groups)
    steps_per_ep = (len(tr) + args.batch - 1) // args.batch
    total = steps_per_ep * args.epochs
    warm = max(int(total * args.warmup_frac), 1)
    lams = ([lambda s: 1.0] if gate is not None else []) + \
           [lambda s: min(1.0, (s + 1) / warm)]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lams)

    def train_loss(recs, tau):
        tgt = torch.tensor([digit_ids[r["gold"]] for r in recs], device=args.device)
        lg, aux = batch_logits(recs, SOFT_FORBID, tau=tau, mode="train")
        return F.cross_entropy(lg, tgt)

    # startup: canary + oracle sanity + ep0 rows
    l0 = train_loss(tr[: args.batch], args.tau0)
    l0.backward()
    if gate is not None:
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in gate.parameters()
                            if p.grad is not None))
        assert torch.isfinite(gn) and float(gn) > 0, "CANARY: gate got no gradient"
        print(f"[canary] gate grad_norm {float(gn):.3e}")
    opt.zero_grad()
    r0 = tr[0]
    og = oracle_gates(r0, L, H, args.device)[:, 0, 0]
    assert all((r0["frame_chars"][i] != r0["q_char"]) == bool(og[i])
               for i in range(len(og))), "oracle/char parse mismatch"
    ep0 = {}
    for nm, arm_probe in (("full", "A2"), ("static-iso", "B2"), ("oracle", "O")):
        saved = args.arm
        args.arm = arm_probe
        ep0[nm] = greedy_em(ev)
        args.arm = saved
        print(f"[ep0 {nm:10s}] dev EM {ep0[nm]:.3f} (maj {maj:.3f})")

    best, best_ep, bad, gstep = -1.0, -1, 0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = list(range(len(tr)))
        random.Random(args.seed * 100 + ep).shuffle(order)
        ce = 0.0
        for i in range(0, len(order), args.batch):
            recs = [tr[j] for j in order[i : i + args.batch]]
            tau = tau_schedule(gstep, total, args.tau0, args.tau1, args.tau_hold)
            loss = train_loss(recs, tau)
            opt.zero_grad()
            loss.backward()
            if gate is not None and args.clip_gates > 0:
                torch.nn.utils.clip_grad_norm_(gate.parameters(), args.clip_gates)
            opt.step()
            sched.step()
            gstep += 1
            ce += float(loss.detach())
        with torch.no_grad():
            tgt = torch.tensor([digit_ids[r["gold"]] for r in fit_probe],
                               device=args.device)
            fit = 0.0
            for i in range(0, len(fit_probe), 32):
                lg, _ = batch_logits(fit_probe[i : i + 32], MASK_MIN, mode="hard")
                fit += float((lg.argmax(-1) == tgt[i : i + 32]).float().sum())
            fit /= len(fit_probe)
        dev_em = greedy_em(ev)
        gap = rel_gap(ev[:64])
        line = (f"[ep {ep}] ce {ce / max(len(order) // args.batch, 1):.4f} | fit "
                f"{fit:.3f} | dev EM {dev_em:.3f} | {time.time() - t0:.0f}s")
        rec_t = {"ep": ep, "fit": fit, "dev_em": dev_em}
        if gap:
            line += f" | cut(dist) {gap[0]:.3f} cut(rel) {gap[1]:.3f} gap {gap[0]-gap[1]:+.3f}"
            rec_t.update({"p_cut_dist": gap[0], "p_cut_rel": gap[1]})
        print(line, flush=True)
        tele.write(json.dumps(rec_t) + "\n")
        tele.flush()
        ck = {"arm": args.arm, "config": vars(args), "epoch": ep, "dev_em": dev_em,
              "fit": fit, "ep0": ep0, "rank": args.rank, "alpha": args.alpha,
              "lora": lora.state(),
              "gate": gate.state_dict() if gate is not None else None}
        torch.save(ck, out / "relmask_last.pt")
        if dev_em > best:
            best, best_ep, bad = dev_em, ep, 0
            torch.save(ck, out / "relmask_best.pt")
        else:
            bad += 1
        (out / "report.txt").write_text(
            f"{run}\nep0 {json.dumps(ep0)} maj {maj:.3f}\nep {ep}: fit {fit:.3f} "
            f"dev {dev_em:.3f}" + (f" gap {gap[0]-gap[1]:+.3f}\n" if gap else "\n")
            + f"best {best:.3f} @ep{best_ep}\n")
        if bad >= args.patience and ep >= 5:
            break
    lora.remove()
    tele.close()
    print(f"DONE best dev EM {best:.3f} @ep{best_ep} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
