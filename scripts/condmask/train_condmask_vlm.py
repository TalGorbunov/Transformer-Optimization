#!/usr/bin/env python3
"""condmask VISION variant: input-conditioned per-(layer, head) block-vs-full gates on
the frozen 4-bit Qwen2.5-VL-7B, co-trained with a LoRA readout (arms A-F as in
train_condmask_text.py). Runs UNCONDITIONALLY on the text pilot's outcome (user req).

Differences from the anchored learnmask trainer (deliberate, condmask spec):
  - NATIVE M-RoPE positions (prepare_sample_replicas(posreset=False)) — the S0 audit
    showed block-aliased positions cripple exactly the cross-block edges under study.
  - Gates are PER (layer, head) binary block-vs-full (HeadGates), input-conditioned on
    the mean-pooled prompt embedding — not relation×layer tables.
  - g=1 closes cross-block relations {R2,R4,R5}; tail rows (R6/R7) always open.
  - Mask dtype fp32 (4-bit VLM law); K=-30 train / MASK_MIN eval.

Data: park-style rendered dirs (data/mmred_park_cond, balanced generator WITH render).

Smoke:
  HF_HOME=$HOME/.cache/huggingface python scripts/condmask/train_condmask_vlm.py \\
      --arm C --limit 24 --dev-limit 12 --epochs 1 --output outputs/_scratch/cmv_smoke
Transfer eval:
  ... --eval-only --ckpt <run>/condmask_vlm_best.pt \\
      --eval-roots "data/mmred_park_cond/seq_len_16/test=100,..."
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
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample
from gnnformer.engine import CarrierEngine
from gnnformer.learnmask import (HeadGates, SOFT_FORBID, gated_stack_logits,
                                 head_layer_mask_fn, head_mask_parts,
                                 prepare_sample_replicas, readers_of,
                                 relation_cell_map)
from gnnformer.runtime import attention_dims, get_layers, load_runtime
from gatenet import st_gumbel, tau_schedule

ARMS = {
    "A": dict(lora=True, gates=False, fixed="full"),
    "B": dict(lora=True, gates=False, fixed="blockwise"),
    "C": dict(lora=True, gates=True, gate_init="blockwise"),
    "D": dict(lora=True, gates=True, gate_init="random"),
    "E": dict(lora=False, gates=True, gate_init="blockwise"),
    "F": dict(lora=False, gates=True, gate_init="random"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="C")
    ap.add_argument("--data-root", default="data/mmred_park_cond/seq_len_8/train")
    ap.add_argument("--dev-root", default="data/mmred_park_cond/seq_len_8/dev")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--dev-limit", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr-gate", type=float, default=1e-3,
                    help="calm default: 3e-2 (the S0-style direct-logit value) blew "
                         "gate logits to |Δ|>2000 through the MLP within one smoke "
                         "epoch (2026-08-17)")
    ap.add_argument("--clip-gates", type=float, default=1.0,
                    help="grad-norm clip on HeadGates params (0 = off)")
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.5)
    ap.add_argument("--tau-hold", type=float, default=0.3)
    ap.add_argument("--init-bias", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--resize", type=int, default=392,
                    help="392 = 196 tok/frame (synthetic park renders are clean; "
                         "512 was the HF-margin choice)")
    ap.add_argument("--readout-ckpt", default="",
                    help="E/F fallback: frozen LoRA readout from an arm-A vlm run")
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/condmask_vlm")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default="", help="--eval-only: trained ckpt")
    ap.add_argument("--eval-roots", default="",
                    help="--eval-only: comma list root[=LIMIT]")
    ap.add_argument("--decode-tokens", type=int, default=4)
    args = ap.parse_args()
    cfg = ARMS[args.arm]
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rt = load_runtime()
    layers = get_layers(rt.model)
    eng = CarrierEngine(rt, l_open=12, e_c=None)  # geometry only; no e_c anywhere
    dims = attention_dims(rt.model)
    L, H, D = eng.n_layers, dims["n_heads"], dims["hidden_size"]
    dev = eng.dev
    digit_ids = eng.digit_ids
    tok = rt.tokenizer

    def prep(root: str, lim: int, gold_max=9):
        recs = []
        for sd in iter_sample_dirs_shuffled(Path(root), args.seed):
            if len(recs) >= lim:
                break
            try:
                _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
            except Exception:
                continue
            g = str(a0).strip()
            if not g.isdigit() or (gold_max is not None and int(g) > gold_max):
                continue
            rec = prepare_sample_replicas(eng, frames, q0, gold=int(g),
                                          task="steps_in_room", resize=args.resize,
                                          posreset=False)   # NATIVE positions
            if rec is not None:
                rec["emb"] = rec["emb"].cpu()
                rec["pos"] = rec["pos"].cpu()
                recs.append(rec)
        return recs

    gates = None
    if cfg["gates"] or args.eval_only:
        gates = HeadGates(D, L, H, hidden=args.hidden,
                          init=cfg.get("gate_init", "blockwise"),
                          init_bias=args.init_bias, seed=args.seed).to(dev)

    def pooled_of(d):
        return d["emb"].float().mean(0).to(dev)

    def sample_forward(d, g_lh, K, e_ids=()):
        """One sample under per-head gates g_lh [L,H] (or a float fixed regime)."""
        e = len(e_ids)
        cm = relation_cell_map(d["seq"], d["blocks"], readers_of(d), d["fin"],
                               e=e).to(dev)
        base, cross = head_mask_parts(cm)
        if isinstance(g_lh, float):
            g_lh = torch.full((L, H), g_lh, device=dev)
        fn = head_layer_mask_fn(base, cross, g_lh, K)
        d2 = {**d, "emb": d["emb"].to(dev), "pos": d["pos"].to(dev)}
        return gated_stack_logits(eng, d2, list(e_ids), fn,
                                  grad_ckpt=args.grad_ckpt and not args.eval_only)

    def gates_of(d, tau, mode):
        lg = gates(pooled_of(d))[0]                      # [L, H]
        return st_gumbel(lg, tau, mode), lg

    @torch.no_grad()
    def greedy_em(d, g_or_fixed):
        """Free greedy digit decode, recompute per step, per-head hard masks."""
        g_lh = (gates_of(d, 1.0, "hard")[0] if g_or_fixed == "gates"
                else float(g_or_fixed))
        toks, first = [], -1
        for step in range(args.decode_tokens):
            lg = sample_forward(d, g_lh, MASK_MIN, e_ids=toks)[-1]
            if step == 0:
                first = int(torch.stack([lg[t] for t in digit_ids]).argmax())
            t = int(lg.argmax())
            if not tok.decode([t]).strip().isdigit():
                break
            toks.append(t)
        txt = tok.decode(toks).strip()
        return (int(txt) if txt.isdigit() else None), first

    # ---------------------------------------------------------------- eval-only mode
    if args.eval_only:
        assert args.ckpt and args.eval_roots
        ck = torch.load(args.ckpt, map_location="cpu")
        lora = None
        lst = ck.get("lora") or (torch.load(ck["config"]["readout_ckpt"],
                                            map_location="cpu")["lora"]
                                 if ck["config"].get("readout_ckpt") else None)
        if lst is not None:
            lora = attach_lora(layers, 0, rank=ck.get("rank", 8),
                               alpha=ck.get("alpha", 16.0), device=dev, state=lst)
            for p in lora.parameters():
                p.requires_grad_(False)
        if ck.get("gatenet") is not None:
            gates = HeadGates.from_state(ck["gatenet"]).to(dev)
        out = Path(args.output) / ("eval_" + time.strftime("%Y%m%d_%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        # '+' also separates: comma-lists cannot ride sbatch --export values
        for spec in [s for s in args.eval_roots.replace("+", ",").split(",") if s]:
            root, _, lim = spec.partition("=")
            recs = prep(root, int(lim) if lim else 100, gold_max=None)
            hits = n = 0
            golds = {}
            for d in recs:
                pred, _ = greedy_em(d, "gates" if ck.get("gatenet") else
                                    (1.0 if ck.get("fixed") == "blockwise" else 0.0))
                hits += int(pred == d["gold"])
                golds[d["gold"]] = golds.get(d["gold"], 0) + 1
                n += 1
            maj = max(golds.values()) / max(n, 1)
            print(f"[{root}] n={n} EM {hits / max(n,1):.3f} (majority {maj:.3f})")
            rows.append({"root": root, "n": n, "em": hits / max(n, 1),
                         "majority": maj})
            (out / "report.txt").write_text(json.dumps(rows, indent=2) + "\n")
        return 0

    # ---------------------------------------------------------------- training mode
    run_name = f"arm{args.arm}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.output) / run_name
    out.mkdir(parents=True, exist_ok=True)
    tr = prep(args.data_root, args.limit)
    ev = prep(args.dev_root, args.dev_limit, gold_max=None)
    assert tr and ev, "empty data"
    fit_probe = tr[:128]
    print(f"data: train {len(tr)} dev {len(ev)}; {L}x{H} gates, resize {args.resize}, "
          f"seq[0] {tr[0]['seq']}")

    lora = None
    if cfg["lora"]:
        lora = attach_lora(layers, 0, rank=args.rank, alpha=args.alpha, device=dev)
        print(f"LoRA: all {L} layers, {lora.num_parameters():,} params")
    elif args.readout_ckpt:
        rck = torch.load(args.readout_ckpt, map_location="cpu")
        ro = attach_lora(layers, 0, rank=rck["rank"], alpha=rck["alpha"], device=dev,
                         state=rck["lora"])
        for p in ro.parameters():
            p.requires_grad_(False)
        print(f"FROZEN readout from {args.readout_ckpt}")
    groups = []
    if gates is not None:
        groups.append({"params": list(gates.parameters()), "lr": args.lr_gate})
    if lora is not None:
        groups.append({"params": list(lora.parameters()), "lr": args.lr_lora})
    assert groups, "no trainables"
    opt = torch.optim.Adam(groups)
    total_steps = ((len(tr) + args.accum - 1) // args.accum) * args.epochs
    warmup = max(int(total_steps * args.warmup_frac), 1)
    lams = ([lambda s: 1.0] if gates is not None else []) + \
           ([lambda s: min(1.0, (s + 1) / warmup)] if lora is not None else [])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lams)

    def loss_of(d, tau):
        if cfg["gates"]:
            g, _ = gates_of(d, tau, "train")
        else:
            g = 1.0 if cfg["fixed"] == "blockwise" else 0.0
        lg = sample_forward(d, g, SOFT_FORBID)[-1]
        cls = lg[digit_ids]
        return F.cross_entropy(cls.view(1, -1),
                               torch.tensor([d["gold"]], device=cls.device))

    # startup: gradient canary + ep0 rows (+ blockwise-init parity)
    l0 = loss_of(tr[0], args.tau0)
    l0.backward()
    if gates is not None:
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in gates.parameters()
                            if p.grad is not None))
        assert torch.isfinite(gn) and float(gn) > 0, "CANARY: no GateNet gradient"
        print(f"[canary] headgates grad_norm {float(gn):.3e}")
    opt.zero_grad()
    ep0 = {}
    for name, gv in (("full", 0.0), ("blockwise", 1.0)):
        hits = sum(int(greedy_em(d, gv)[0] == d["gold"]) for d in ev)
        ep0[name] = hits / len(ev)
        print(f"[ep0 {name:9s}] dev EM {ep0[name]:.3f}")
    if cfg["gates"]:
        hits = sum(int(greedy_em(d, "gates")[0] == d["gold"]) for d in ev)
        ep0["gates-init"] = hits / len(ev)
        print(f"[ep0 gates-init] dev EM {ep0['gates-init']:.3f}")
        if cfg["gate_init"] == "blockwise":
            assert abs(ep0["gates-init"] - ep0["blockwise"]) < 1e-9, "PARITY FAILURE"

    best_em, best_ep, bad, gstep = -1.0, -1, 0, 0
    probe_pool = torch.stack([pooled_of(d) for d in ev[:48]])
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = list(range(len(tr)))
        random.Random(args.seed * 100 + ep).shuffle(order)
        ce_sum = 0.0
        for i, j in enumerate(order):
            tau = tau_schedule(gstep, total_steps, args.tau0, args.tau1, args.tau_hold)
            loss = loss_of(tr[j], tau)
            (loss / args.accum).backward()
            ce_sum += float(loss.detach())
            if (i + 1) % args.accum == 0:
                if gates is not None and args.clip_gates > 0:
                    torch.nn.utils.clip_grad_norm_(gates.parameters(), args.clip_gates)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
        opt.step(); sched.step(); opt.zero_grad(); gstep += 1
        with torch.no_grad():
            fit = sum(int(int(sample_forward(
                d, gates_of(d, 1.0, "hard")[0] if cfg["gates"] else
                (1.0 if cfg["fixed"] == "blockwise" else 0.0),
                MASK_MIN)[-1][digit_ids].argmax()) == d["gold"])
                for d in fit_probe) / len(fit_probe)
        dev_em = sum(int(greedy_em(d, "gates" if cfg["gates"] else
                                   (1.0 if cfg["fixed"] == "blockwise" else 0.0)
                                   )[0] == d["gold"]) for d in ev) / len(ev)
        line = (f"[ep {ep}] ce {ce_sum / len(tr):.4f} | fit {fit:.3f} | dev EM "
                f"{dev_em:.3f} | {time.time() - t0:.0f}s")
        if gates is not None:
            with torch.no_grad():
                lgp = gates(probe_pool)
                dmax = float((lgp - gates.init_logits).abs().max())
                fl = gates.flips(probe_pool)
                pb = float(torch.sigmoid(lgp).mean())
            line += f" | flips {fl}/{L * H} max|dlog| {dmax:.2f} P(block) {pb:.3f}"
        print(line, flush=True)
        ck = {"arm": args.arm, "config": vars(args), "epoch": ep, "dev_em": dev_em,
              "fit": fit, "ep0": ep0, "rank": args.rank, "alpha": args.alpha,
              "fixed": cfg.get("fixed"),
              "gatenet": gates.state() if gates is not None else None,
              "lora": lora.state() if lora is not None else None}
        torch.save(ck, out / "condmask_vlm_last.pt")
        if dev_em > best_em:
            best_em, best_ep, bad = dev_em, ep, 0
            torch.save(ck, out / "condmask_vlm_best.pt")
        else:
            bad += 1
        (out / "report.txt").write_text(
            f"{run_name}\nep0 {json.dumps(ep0)}\nep {ep}: fit {fit:.3f} dev EM "
            f"{dev_em:.3f}\nbest {best_em:.3f} @ep{best_ep}\n")
        if bad >= args.patience and ep >= 3:
            break
    if lora is not None:
        lora.remove()
    print(f"DONE best dev EM {best_em:.3f} @ep{best_ep} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
