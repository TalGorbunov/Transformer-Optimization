#!/usr/bin/env python3
"""condmask text-pilot trainer — input-conditioned per-(layer, head) attention masks.

Claim under test: training with a LEARNABLE attention mask at N=8 improves exact-match
at longer N. Gates g[l,h] ∈ {0,1} (1 = blockwise: frame tokens attend own frame +
question prefix; 0 = full causal; tail rows always causal), decided per sample by
GateNet(mean-pooled input embeddings), optimized with hard ST-Gumbel, trained with CE
on the answer digit token at the answer position (nothing copyable in context).

Arms (--arm):
  A  LoRA + fixed full attention           (control)
  B  LoRA + fixed blockwise                (static-mask control)
  C  LoRA + learned gates, blockwise init  (main)
  D  LoRA + learned gates, random init     (ablation 1)
  E  gates only, frozen model, blockwise init
  F  gates only, frozen model, random init

Anti-S0 protocol (ASSESSMENT.md Appendix A): barrier |init_bias|=1.0; thousands of
optimizer steps (travel >= 100x barrier); startup gradient canary; ep0 parity row
(gates-init(blockwise) EM must EQUAL fixed-blockwise EM exactly); per-epoch flip
telemetry on a fixed probe batch; underpowered-abort at 30% of steps.

Training-fit guarantee: train CE + train acc on a 256-sample train probe every epoch;
a run ending below train acc 0.95 is flagged UNDERFIT (arms A-D).

Run (smoke):
  HF_HOME=$HOME/.cache/huggingface python scripts/condmask/train_condmask_text.py \\
      --arm C --model Qwen/Qwen2.5-0.5B-Instruct --limit 64 --epochs 1 \\
      --output outputs/_scratch/condmask_smoke
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
from textdata import annotate_prompt, majority_baseline, prep_root

ARMS = {
    "A": dict(lora=True, gates=False, fixed="full"),
    "B": dict(lora=True, gates=False, fixed="blockwise"),
    "C": dict(lora=True, gates=True, gate_init="blockwise"),
    "D": dict(lora=True, gates=True, gate_init="random"),
    "E": dict(lora=False, gates=True, gate_init="blockwise"),
    "F": dict(lora=False, gates=True, gate_init="random"),
}


def make_masks_fn(bases, templates, g, K, dtype):
    """g: float (fixed regime, 0.0 full / 1.0 blockwise) or [B, L, H] tensor."""
    if isinstance(g, float):
        gv = torch.full((bases.shape[0], 1), g, device=bases.device)
        m = assemble_batch_masks(bases, templates, gv, K, dtype)
        return lambda li: m
    return lambda li: assemble_batch_masks(bases, templates, g[:, li], K, dtype)


def gold_token_ids(tok):
    ids = []
    for d in range(10):
        t = tok(str(d), add_special_tokens=False).input_ids
        assert len(t) == 1, f"digit {d} is not a single token"
        ids.append(t[0])
    return ids


class Runner:
    """Shared forward plumbing for train/dev/probe passes."""

    def __init__(self, model, tok, gatenet, dims, dtype, device, math_only=False):
        self.model, self.tok, self.gatenet = model, tok, gatenet
        self.L, self.H, self.D = dims
        self.dtype, self.device = dtype, device
        self.embed = model.model.embed_tokens
        self.digit_ids = gold_token_ids(tok)
        self.math_only = math_only  # gates-only arms: LSE-alignment workaround

    def gates_for(self, ids, valid, tau, mode):
        with torch.no_grad():
            emb = self.embed(ids)
        pooled = pool_embeddings(emb, valid)
        logits = self.gatenet(pooled)
        return st_gumbel(logits, tau, mode), logits

    def batch_logits(self, recs, g_or_fixed, K, tau=1.0, mode="hard"):
        """-> (answer logits [B, V] fp32, gate logits or None). g_or_fixed: 'gates'
        (use GateNet) or a float fixed regime value."""
        ids, valid, ans, S = pad_batch(recs, self.tok.pad_token_id or 0, self.device)
        bases, templates = batch_mask_parts(recs, S, self.device)
        if g_or_fixed == "gates":
            g, glog = self.gates_for(ids, valid, tau, mode)
        else:
            g, glog = float(g_or_fixed), None
        fn = make_masks_fn(bases, templates, g, K, self.dtype)
        return forward_answer_logits(self.model, ids, ans, fn,
                                     math_only=self.math_only), glog

    @torch.no_grad()
    def greedy_em(self, recs, g_or_fixed, batch=16, max_new=2):
        """Batched free greedy decode EM (+MAE) under hard gates, K=MASK_MIN."""
        hits, mae, n = 0, 0.0, 0
        for i in range(0, len(recs), batch):
            chunk = [dict(r) for r in recs[i : i + batch]]
            done = [False] * len(chunk)
            texts = [""] * len(chunk)
            for _ in range(max_new):
                lg, _ = self.batch_logits(chunk, g_or_fixed, MASK_MIN, mode="hard")
                nxt = lg.argmax(-1).tolist()
                for b, t in enumerate(nxt):
                    if done[b]:
                        continue
                    s = self.tok.decode([t])
                    if s.strip().isdigit() and (s == s.strip() or not texts[b]):
                        texts[b] += s.strip()
                        chunk[b]["ids"] = chunk[b]["ids"] + [t]
                        chunk[b]["seq"] += 1
                    else:
                        done[b] = True
                if all(done):
                    break
            for b, r in enumerate(recs[i : i + batch]):
                pred = int(texts[b]) if texts[b].isdigit() else None
                hits += int(pred == r["gold"])
                mae += abs((pred if pred is not None else 0) - r["gold"])
                n += 1
        return hits / max(n, 1), mae / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-root", default="data/mmred_text_cond/seq_len_8/train")
    ap.add_argument("--dev-root", default="data/mmred_text_cond/seq_len_8/dev")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--dev-limit", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr-gate", type=float, default=3e-2,
                    help="v1 grid value; 3e-2 was calibrated for DIRECT logit params "
                         "(S0-style) — through the GateNet MLP it saturates gates "
                         "within ~1 epoch (measured: flips freeze, cross-input-var 0). "
                         "v2 recipe: 1e-3 with --clip-gates 1.0 --gate-norm-input")
    ap.add_argument("--clip-gates", type=float, default=0.0,
                    help="grad-norm clip on GateNet params (0 = off)")
    ap.add_argument("--gate-norm-input", action="store_true",
                    help="LayerNorm on the pooled GateNet input")
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--tau-hold", type=float, default=0.3)
    ap.add_argument("--init-bias", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lam-open", type=float, default=0.0,
                    help="OPTIONAL stretch: penalty on P(open); default OFF "
                         "(unconstrained CE is the registered main setting)")
    ap.add_argument("--readout-ckpt", default="",
                    help="pre-specified E/F fallback (preflight FAILED 2026-08-17: "
                         "frozen 1.5B <= majority+2pp): attach this ckpt's LoRA "
                         "FROZEN as the readout; only GateNet trains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="outputs/condmask")
    args = ap.parse_args()
    cfg = ARMS[args.arm]
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run_name = f"arm{args.arm}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.output) / run_name
    out.mkdir(parents=True, exist_ok=True)
    tele = open(out / "telemetry.jsonl", "a")

    tok, model = load_text_model(args.model, device=args.device)
    dims = model_dims(model)
    L, H, D = dims
    dtype = next(model.parameters()).dtype
    print(f"model {args.model}: {L} layers x {H} heads, d={D}, dtype={dtype}")

    # ---- data (shuffled BEFORE limiting — gold-sorted-dir trap) --------------------
    # Roots are comma-separated, each optionally root=LIMIT (length-mixture training,
    # e.g. seq_len_2/train=1800,seq_len_4/train=1800,seq_len_8/train=1800). Records
    # are shuffled ACROSS roots afterwards so batches, the fit probe, and the gate
    # probe all mix lengths.
    def prep_spec(spec: str, default_lim: int, seed: int, gold_max):
        recs = []
        # '+' also separates: comma-lists cannot ride sbatch --export values
        for part in [s for s in spec.replace("+", ",").split(",") if s.strip()]:
            root, _, lim = part.partition("=")
            r = prep_root(tok, Path(root), int(lim) if lim else default_lim,
                          seed=seed, gold_max=gold_max)
            print(f"  [prep] {root}: {len(r)} samples")
            recs.extend(r)
        return recs

    tr = prep_spec(args.data_root, args.limit, args.seed, gold_max=9)
    ev = prep_spec(args.dev_root, args.dev_limit, 1234, gold_max=None)
    random.Random(args.seed + 7).shuffle(tr)
    random.Random(4321).shuffle(ev)   # fixed: the gate probe (ev[:64]) must be stable
    maj_cls, maj = majority_baseline([r["gold"] for r in ev])
    print(f"data: train {len(tr)}  dev {len(ev)}  (dev majority: class {maj_cls} "
          f"@ {maj:.3f})")
    assert tr and ev, "empty data"
    fit_probe = tr[:256]                       # training-fit criterion
    gate_probe = ev[:64]                       # flip telemetry, fixed all run

    # ---- trainables ---------------------------------------------------------------
    gatenet = None
    if cfg["gates"]:
        gatenet = GateNet(D, L, H, hidden=args.hidden, init=cfg["gate_init"],
                          init_bias=args.init_bias, seed=args.seed,
                          norm_input=args.gate_norm_input).to(args.device)
    lora = None
    if cfg["lora"]:
        assert cfg["lora"] and not args.readout_ckpt, \
            "--readout-ckpt is for the gates-only arms (E/F)"
        assert_lora_targets(model)
        lora = attach_lora(model.model.layers, 0, rank=args.rank, alpha=args.alpha,
                           device=args.device)
        print(f"LoRA attached: all {L} layers, rank {args.rank}, "
              f"{lora.num_parameters():,} params")
    elif args.readout_ckpt:
        rck = torch.load(args.readout_ckpt, map_location="cpu")
        lora = attach_lora(model.model.layers, 0, rank=rck["rank"],
                           alpha=rck["alpha"], device=args.device,
                           state=rck["lora"])
        for p in lora.parameters():
            p.requires_grad_(False)
        print(f"FROZEN readout LoRA attached from {args.readout_ckpt} "
              f"(arm {rck.get('arm')}, dev EM {rck.get('dev_em')})")
        lora = None  # frozen: excluded from the optimizer; hooks stay installed
    groups = []
    if gatenet is not None:
        groups.append({"params": list(gatenet.parameters()), "lr": args.lr_gate})
    if lora is not None:
        groups.append({"params": list(lora.parameters()), "lr": args.lr_lora})
    assert groups, "arm has no trainables"
    opt = torch.optim.Adam(groups)
    steps_per_ep = (len(tr) + args.batch - 1) // args.batch
    total_steps = steps_per_ep * args.epochs
    warmup = max(int(total_steps * args.warmup_frac), 1)
    lambdas = []
    if gatenet is not None:
        lambdas.append(lambda s: 1.0)
    if lora is not None:
        lambdas.append(lambda s: min(1.0, (s + 1) / warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    travel_target = 100.0 * args.init_bias
    print(f"steps/epoch {steps_per_ep}, total {total_steps}; gate travel ceiling "
          f"~{total_steps * args.lr_gate:.0f} vs target {travel_target:.0f} "
          f"(barrier {args.init_bias})")

    run = Runner(model, tok, gatenet, dims, dtype, args.device,
                 math_only=not cfg["lora"])
    digit_of = {g: run.digit_ids[g] for g in range(10)}

    # ---- startup: prompt audit + gradient canary + ep0 reference rows --------------
    (out / "report.txt").write_text("")  # created; rewritten each epoch
    prompt_dump = annotate_prompt(tr[0], tok)
    print(f"[audit] sample 0 prompt with block boundaries -> report.txt")

    def train_loss(recs, tau):
        tgt = torch.tensor([digit_of[r["gold"]] for r in recs], device=args.device)
        src = "gates" if cfg["gates"] else (1.0 if cfg["fixed"] == "blockwise" else 0.0)
        lg, glog = run.batch_logits(recs, src, SOFT_FORBID, tau=tau, mode="train")
        loss = F.cross_entropy(lg, tgt)
        if args.lam_open > 0 and glog is not None:
            loss = loss + args.lam_open * (1.0 - torch.sigmoid(glog)).mean()
        acc = float((lg.argmax(-1) == tgt).float().mean())
        return loss, acc, glog

    loss0, _, _ = train_loss(tr[: args.batch], tau=args.tau0)
    loss0.backward()
    if gatenet is not None:
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in gatenet.parameters()
                            if p.grad is not None))
        assert torch.isfinite(gn) and float(gn) > 0, "CANARY: GateNet got no gradient"
        print(f"[canary] gatenet grad_norm {float(gn):.3e} — gradient path LIVE")
    if lora is not None:
        ln = torch.sqrt(sum((p.grad ** 2).sum() for p in lora.parameters()
                            if p.grad is not None))
        assert torch.isfinite(ln) and float(ln) > 0, "CANARY: LoRA got no gradient"
        print(f"[canary] lora grad_norm {float(ln):.3e}")
    opt.zero_grad()

    ep0 = {}
    for name, gv in (("full", 0.0), ("blockwise", 1.0)):
        em, mae = run.greedy_em(ev, gv)
        ep0[name] = em
        print(f"[ep0 {name:9s}] dev EM {em:.3f} MAE {mae:.2f} (majority {maj:.3f})")
    if cfg["gates"]:
        em, mae = run.greedy_em(ev, "gates")
        ep0["gates-init"] = em
        print(f"[ep0 gates-init] dev EM {em:.3f} MAE {mae:.2f}")
        if cfg["gate_init"] == "blockwise":
            assert abs(em - ep0["blockwise"]) < 1e-9, (
                "PARITY FAILURE: blockwise-init gates != fixed blockwise")

    # ---- training -------------------------------------------------------------
    best_em, best_ep, bad = -1.0, -1, 0
    gstep = 0
    grad_norm_sum, grad_norm_n = 0.0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = list(range(len(tr)))
        random.Random(args.seed * 1000 + ep).shuffle(order)
        ce_sum, n_b = 0.0, 0
        for i in range(0, len(order), args.batch):
            recs = [tr[j] for j in order[i : i + args.batch]]
            tau = tau_schedule(gstep, total_steps, args.tau0, args.tau1, args.tau_hold)
            loss, _, _ = train_loss(recs, tau)
            opt.zero_grad()
            loss.backward()
            if gatenet is not None:
                if args.clip_gates > 0:
                    gn = torch.nn.utils.clip_grad_norm_(gatenet.parameters(),
                                                        args.clip_gates)
                else:
                    gn = torch.sqrt(sum((p.grad ** 2).sum()
                                        for p in gatenet.parameters()
                                        if p.grad is not None))
                grad_norm_sum += float(gn)
                grad_norm_n += 1
            opt.step()
            sched.step()
            gstep += 1
            ce_sum += float(loss.detach())
            n_b += 1
            # underpowered-abort (gate arms): 30% in, logits must be moving
            if (gatenet is not None and gstep == max(int(total_steps * 0.3), 1)):
                with torch.no_grad():
                    ids, valid, _, _ = pad_batch(gate_probe,
                                                 tok.pad_token_id or 0, args.device)
                    lg_p = gatenet(pool_embeddings(run.embed(ids), valid))
                    dmax = float((lg_p - gatenet.init_logits).abs().max())
                    fl = int(((lg_p > 0) != (gatenet.init_logits > 0)).any(0).sum())
                if dmax < 0.1 and fl == 0:
                    (out / "report.txt").write_text("UNDERPOWERED at 30% of steps: "
                                                    f"max|dlogit| {dmax:.4f}, 0 flips\n")
                    print("UNDERPOWERED — aborting (see report)")
                    return 3

        # ---- per-epoch metrics ------------------------------------------------
        with torch.no_grad():
            fit_acc, fit_ce = 0.0, 0.0
            src = "gates" if cfg["gates"] else (
                1.0 if cfg["fixed"] == "blockwise" else 0.0)
            for i in range(0, len(fit_probe), 32):
                recs = fit_probe[i : i + 32]
                tgt = torch.tensor([digit_of[r["gold"]] for r in recs],
                                   device=args.device)
                lg, _ = run.batch_logits(recs, src, MASK_MIN, mode="hard")
                fit_acc += float((lg.argmax(-1) == tgt).float().sum())
                fit_ce += float(F.cross_entropy(lg, tgt, reduction="sum"))
            fit_acc /= len(fit_probe)
            fit_ce /= len(fit_probe)
        dev_em, dev_mae = run.greedy_em(
            ev, "gates" if cfg["gates"] else (1.0 if cfg["fixed"] == "blockwise"
                                              else 0.0))
        line = (f"[ep {ep}] train-ce {ce_sum / max(n_b, 1):.4f} | fit-probe acc "
                f"{fit_acc:.3f} ce {fit_ce:.4f} | dev EM {dev_em:.3f} MAE "
                f"{dev_mae:.2f} | {time.time() - t0:.0f}s")
        rec_t = {"ep": ep, "train_ce": ce_sum / max(n_b, 1), "fit_acc": fit_acc,
                 "fit_ce": fit_ce, "dev_em": dev_em, "dev_mae": dev_mae}
        if gatenet is not None:
            with torch.no_grad():
                ids, valid, _, _ = pad_batch(gate_probe, tok.pad_token_id or 0,
                                             args.device)
                lg_p = gatenet(pool_embeddings(run.embed(ids), valid))
                hard = (lg_p > 0)
                init_hard = (gatenet.init_logits > 0)
                d = (lg_p - gatenet.init_logits).abs()
                p_block = torch.sigmoid(lg_p).mean(0)
                frac = hard.float().mean(0)
                cross_var = float(((frac > 0) & (frac < 1)).float().mean())
                flips_mv = int(((frac > 0.5) != init_hard).sum())
                flips_any = int((hard != init_hard).any(0).sum())
            travel = gstep * args.lr_gate
            gline = (f"      gates: flips(majority) {flips_mv}/{L * H} "
                     f"flips(any-input) {flips_any} | mean|dlog| {float(d.mean()):.3f} "
                     f"max {float(d.max()):.3f} | P(block) {float(p_block.mean()):.3f} "
                     f"| cross-input-var {cross_var:.3f} | grad-norm "
                     f"{grad_norm_sum / max(grad_norm_n, 1):.2e} | travel "
                     f"{travel:.0f}/{travel_target:.0f}")
            print(line + "\n" + gline)
            rec_t.update({"flips_majority": flips_mv, "flips_any": flips_any,
                          "dlogit_mean": float(d.mean()), "dlogit_max": float(d.max()),
                          "p_block_mean": float(p_block.mean()),
                          "cross_input_var": cross_var, "travel": travel})
            with open(out / f"heatmap_ep{ep}.csv", "w") as f:
                f.write("layer," + ",".join(f"h{h}" for h in range(H)) + "\n")
                for li in range(L):
                    f.write(f"{li}," + ",".join(f"{float(p_block[li, h]):.4f}"
                                                for h in range(H)) + "\n")
        else:
            print(line)
        tele.write(json.dumps(rec_t) + "\n")
        tele.flush()

        # ---- ckpt + report (rewritten every epoch — walltime-kill safe) --------
        ck = {"arm": args.arm, "config": vars(args), "epoch": ep, "dev_em": dev_em,
              "fit_acc": fit_acc, "ep0": ep0, "n_layers": L, "n_heads": H,
              "d_model": D,
              "gatenet": gatenet.state_dict() if gatenet is not None else None,
              "lora": lora.state() if lora is not None else None,
              "rank": args.rank, "alpha": args.alpha}
        torch.save(ck, out / "condmask_last.pt")
        if dev_em > best_em:
            best_em, best_ep, bad = dev_em, ep, 0
            torch.save(ck, out / "condmask_best.pt")
        else:
            bad += 1
        underfit = (cfg["lora"] and fit_acc < 0.95)
        (out / "report.txt").write_text(
            f"condmask {run_name}\narm {args.arm} model {args.model} seed {args.seed}\n"
            f"ep0: {json.dumps(ep0)}  dev majority {maj:.3f}\n"
            f"epoch {ep}: fit acc {fit_acc:.3f} (criterion >=0.95"
            f"{' — UNDERFIT' if underfit else ''}), dev EM {dev_em:.3f}\n"
            f"best dev EM {best_em:.3f} @ep{best_ep}\n\n=== sample 0 prompt ===\n"
            f"{prompt_dump}\n")
        if bad >= args.patience and ep >= 6:
            print(f"early stop at ep {ep} (best {best_em:.3f} @ep{best_ep})")
            break

    if lora is not None:
        lora.remove()
    tele.close()
    print(f"DONE best dev EM {best_em:.3f} @ep{best_ep} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
