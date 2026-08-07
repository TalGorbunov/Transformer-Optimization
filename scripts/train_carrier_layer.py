#!/usr/bin/env python3
"""THE production carrier trainer (cached lo-phase): frozen distilled e_c + LoRA on
layers >= L_OPEN, teacher-forced scratchpad targets, TF-count + tf-exact model selection.

Layers 0..L_OPEN-1 are static per sample (e_c frozen), so they run ONCE at prep and the
hi phase trains on cached h_{L_OPEN} (~3x cheaper steps). Produced every headline in-model
readout ckpt: l12v2, the format-sweep winner (caption), the LOTO writer, the P1.1 seeds.

Anchor recipe (RESULTS.md [2026-07-22] l12v2 / [2026-07-24] FORMAT SWEEP):
  --running-tally --jitter-gap 16 --grad-ckpt --l-open 12 --limit 900 --epochs 5
  --scratchpad-format caption --carrier-ckpt checkpoints/carrier_token_room_k1_best.pt
  + the 15/16-root mixture -> BEST ~0.999 / tf-exact ~0.99, N=32 exam 0.987.

Usage:
  python scripts/train_carrier_layer.py --carrier-ckpt checkpoints/carrier_token_room_k1_best.pt \
      --data_root <comma roots, each optionally path=LIMIT> --scratchpad-format caption \
      --running-tally --jitter-gap 16 --grad-ckpt --l-open 12 --epochs 5 --output outputs/carrier/train

Gated-attention ablation (2026-08-07, outputs/gating/): --gate/--gate-layers/--gate-lr/
--gate-b0/--gate-only attach a gnnformer.gating adapter. ALL DEFAULT OFF and every gate
code path is skipped at --gate none, so the anchor recipe above is bit-identical to before
the edit — arm 1 of that campaign's P3 is the regression test for exactly this.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_distilled_carrier, save_carrier_layer_ckpt
from gnnformer.data import (
    frame_attr_labels,
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
)
from gnnformer.engine import CarrierEngine
from gnnformer.gating import GATE_VARIANTS, attach_gate
from gnnformer.mmred_hf import (
    build_scan_mmred,
    build_target_v4,
    build_target_v5,
    qtype_from_dirname,
)
from gnnformer.runtime import attention_dims, get_layers, load_runtime
from gnnformer.scratchpad import (
    SCRATCHPAD_FORMATS,
    build_target,
    build_target_fmt,
    build_target_tally,
    couple_offsets,
)


def verdict_weights(tgt_str: str, tgt_ids: list, tok, w_hi: float) -> list:
    """Per-token CE weights: w_hi on tokens overlapping an informative char span
    (the slot after each 'fN:', the value after 'total:/answer:'), 1.0 elsewhere.
    Char spans -> token spans via cumulative per-token decode lengths (targets are
    pure ASCII; prep asserts decode roundtrip)."""
    import re
    spans = [m.span(1) for m in re.finditer(r"f\d+:(\S+)", tgt_str)]
    m = re.search(r"counts: (.+?) \|", tgt_str)
    if m:
        spans.append(m.span(1))
    m = re.search(r"(?:total|answer|max|min): (.+?) END", tgt_str)
    if m:
        spans.append(m.span(1))
    w, pos = [], 0
    for t in tgt_ids:
        piece = tok.decode([t])
        a, b = pos, pos + len(piece)
        w.append(w_hi if any(a < e and s < b for s, e in spans) else 1.0)
        pos = b
    return w


def parse_gate_layers(spec: str, l_open: int, n_layers: int) -> list:
    """'open' -> l_open..end | 'ge:L' -> L..end | 'A-B' -> inclusive range | 'L' -> one.

    Hard-rejects anything below l_open: layers 0..l_open-1 run ONCE at prep and their
    output is cached, so a gate placed there would be frozen at its identity init and the
    arm would silently be a no-op."""
    s = spec.strip()
    if s in ("open", ""):
        out = list(range(l_open, n_layers))
    elif s.startswith("ge:"):
        out = list(range(int(s[3:]), n_layers))
    elif "-" in s:
        a, b = s.split("-", 1)
        out = list(range(int(a), int(b) + 1))
    else:
        out = [int(s)]
    if not out:
        raise SystemExit(f"--gate-layers {spec!r} selected no layers")
    if min(out) < l_open:
        raise SystemExit(f"--gate-layers {spec!r} -> {out}: layers below --l-open "
                         f"({l_open}) sit in the cached lo phase and would never train")
    if max(out) >= n_layers:
        raise SystemExit(f"--gate-layers {spec!r} -> {out}: model has {n_layers} layers")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform",
                    help="comma-separated roots; each may carry a per-root cap as path=LIMIT")
    ap.add_argument("--limit", type=int, default=900, help="default PER-root cap")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--l-open", type=int, default=12)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--carrier-ckpt", required=True, help="distilled carrier_best.pt (e_c FROZEN)")
    ap.add_argument("--task", choices=("park", "mmred_hf"), default="park",
                    help="mmred_hf: MMReD-HF materialized dirs — qtype from the dir-name "
                         "prefix, gold scans via gnnformer.mmred_hf.build_scan_mmred "
                         "(formats.md; word golds allowed); park defaults untouched")
    ap.add_argument("--verdict-weight", type=float, default=1.0, metavar="W",
                    help="mmred_hf scratchpad: CE weight on the informative target "
                         "tokens (per-frame verdict slots + the final value) vs 1.0 "
                         "on copyable boilerplate (' scan:', 'fN:', '| total:'). "
                         "Counters the all-x/all– verdict collapse in free decode")
    ap.add_argument("--mmred-target", choices=("caption", "v4", "v5"), default="caption",
                    help="v4: verdict scans for aggregation qtypes + DIRECT answers "
                         "for content qtypes (build_target_v4; 2026-08-03 rescope)")
    ap.add_argument("--scratchpad", action="store_true")
    ap.add_argument("--running-tally", action="store_true", help="implies --scratchpad")
    ap.add_argument("--pos-couple", action="store_true", help="E-G (refuted); implies --running-tally")
    ap.add_argument("--scratchpad-format", default="poslist", choices=SCRATCHPAD_FORMATS)
    ap.add_argument("--jitter-gap", type=int, default=0, metavar="G",
                    help="train-only hi-phase carrier position jitter ~U{1..G}")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-qfirst", action="store_true", help="C2 ablation")
    ap.add_argument("--no-posreset", action="store_true", help="C2 ablation")
    ap.add_argument("--truncate-at", type=int, default=None, metavar="L",
                    help="TRUNC E4 deploy-matched training; must equal --l-open")
    ap.add_argument("--shuffle-dirs", type=int, default=0, metavar="SEED")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="pin the train/eval dir split while --seed varies init/jitter/shuffle")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/carrier/train")
    # --- gated attention (arXiv:2505.06708) ablation; OFF by default, and every code
    # --- path below is skipped when --gate none, so behaviour is bit-identical unused.
    ap.add_argument("--gate", default="none", choices=("none",) + GATE_VARIANTS,
                    help="attach a gated-attention adapter (gnnformer/gating.py). "
                         "g1_* gate after SDPA (query-side, post-sum); g2_literal gates "
                         "v_proj (message-side, pre-sum)")
    ap.add_argument("--gate-layers", default="open", metavar="SPEC",
                    help="'open' = L_OPEN..end (default) | 'ge:L' | 'A-B' | 'L'. "
                         "Every gated layer must be >= --l-open: layers below it are "
                         "cached ONCE at prep, so a gate there would never see training")
    ap.add_argument("--gate-lr", type=float, default=3e-4,
                    help="gate LR (different geometry from LoRA's 1e-4)")
    ap.add_argument("--gate-b0", type=float, default=2.0,
                    help="identity-init bias: g = sigmoid(xW+b)/sigmoid(b). Do NOT use "
                         "6.0 (sigma'(6)=0.0025, the gate cannot learn in our budget)")
    ap.add_argument("--gate-only", action="store_true",
                    help="train the gate alone: LoRA stays attached at its B=0 init "
                         "(bit-identical to absent) so the ckpt schema is unchanged, but "
                         "is excluded from the optimizer")
    args = ap.parse_args()
    if args.pos_couple:
        args.running_tally = True
    if args.scratchpad_format != "poslist":
        args.scratchpad = True
        if args.pos_couple:
            raise SystemExit("--pos-couple is poslist-only")
    if args.running_tally:
        args.scratchpad = True
    if args.truncate_at is not None and args.truncate_at != args.l_open:
        raise SystemExit(f"--truncate-at must equal --l-open, got {args.truncate_at} vs {args.l_open}")
    if args.task == "mmred_hf" and args.scratchpad_format not in ("scan", "caption"):
        raise SystemExit("--task mmred_hf requires --scratchpad-format scan|caption")
    if args.gate == "none" and args.gate_only:
        raise SystemExit("--gate-only needs --gate <variant>")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rt = load_runtime(args.model) if args.model else load_runtime()
    layers = get_layers(rt.model)
    tok = rt.tokenizer
    dev = rt.device
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_L{args.l_open}_r{args.rank}")
    out.mkdir(parents=True, exist_ok=True)

    ckc = load_distilled_carrier(Path(args.carrier_ckpt))
    e_c = ckc["e_c"].float().to(dev)  # FROZEN by design
    _dp = ckc.get("dprime") if ckc.get("dprime") is not None else ckc.get("probe", 0.0)
    print(f"[init] frozen e_c from {args.carrier_ckpt} (distill metric {_dp:.2f})", flush=True)
    lora = attach_lora(layers, args.l_open, rank=args.rank, alpha=args.alpha, device=dev)
    groups = ([] if args.gate_only else [{"params": lora.parameters(), "lr": args.lr_lora}])
    if args.gate_only:
        for p in lora.parameters():
            p.requires_grad_(False)  # B=0 stays 0 -> the LoRA hook contributes exactly 0
    gate = None
    if args.gate != "none":
        gl = parse_gate_layers(args.gate_layers, args.l_open, len(layers))
        dims = attention_dims(rt.model)
        gate = attach_gate(layers, gl, args.gate, hidden=dims["hidden"],
                           n_heads=dims["n_heads"], n_kv=dims["n_kv"],
                           head_dim=dims["head_dim"], device=dev, b0=args.gate_b0)
        groups.append({"params": gate.parameters(), "lr": args.gate_lr})
        print(f"[gate] {args.gate} on layers {gl} b0={args.gate_b0} lr={args.gate_lr} "
              f"params {gate.num_parameters()} "
              f"(lora {'FROZEN at B=0' if args.gate_only else 'trained'})", flush=True)
    opt = torch.optim.Adam(groups)
    print(f"[params] lora {lora.num_parameters()}"
          f"{' (frozen)' if args.gate_only else ''} "
          f"gate {gate.num_parameters() if gate else 0} (e_c frozen)", flush=True)
    eng = CarrierEngine(rt, l_open=args.l_open, e_c=e_c, pos_couple=args.pos_couple)
    digit_ids = eng.digit_ids

    # ---- prep: lo phase once per sample, cache h_{L_OPEN} ----
    data = []
    n_done = n_skip = 0
    cache_gb = 0.0
    t0 = time.time()
    for root in args.data_root.split(","):
        root = root.strip()
        if not root:
            continue
        lim = args.limit
        if "=" in root:
            root, lim = root.rsplit("=", 1)
            lim = int(lim)
        it = (iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs)
              if args.shuffle_dirs is not None else iter_sample_dirs(Path(root)))
        n_root = 0
        for sd in it:
            if n_root >= lim:
                break
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            except Exception:
                n_skip += 1
                continue
            tgt_ids: list = []
            anch = None
            if args.task == "mmred_hf":
                qtype = qtype_from_dirname(sd.name)
                _builder = {"v4": build_target_v4, "v5": build_target_v5,
                            "caption": build_scan_mmred}[args.mmred_target]
                tgt_str = (_builder(qtype, q0, states, str(a0).strip())
                           if qtype else None)
                if tgt_str is None:
                    n_skip += 1
                    continue
                gold, task = str(a0).strip(), qtype
            else:
                try:
                    gold = int(str(a0).strip())
                except Exception:
                    n_skip += 1
                    continue
                parsed = parse_task_labels(q0, states, gold)
                if parsed is None or (gold > 9 and not args.scratchpad):
                    n_skip += 1
                    continue
                task, evid, aux = parsed
                if args.scratchpad:
                    if args.scratchpad_format != "poslist":
                        labels = (frame_attr_labels(task, q0, states, evid)
                                  if (args.scratchpad_format in ("caption", "chunked") or task == "rooms")
                                  else None)
                        tgt_str = build_target_fmt(args.scratchpad_format, task, evid, aux,
                                                   gold, NF=len(frames), labels=labels)
                    else:
                        tgt_str = (build_target_tally if args.running_tally else build_target)(
                            task, evid, aux, gold)
            if args.scratchpad:
                tgt_ids = tok(tgt_str, add_special_tokens=False).input_ids + [tok.eos_token_id]
                if args.pos_couple and task != "rooms":
                    anch = couple_offsets([tok.decode([t]) for t in tgt_ids], len(frames))
                if n_done == 0:
                    rtxt = tok.decode(tgt_ids[:-1])
                    print(f"[target-debug] task={task} gold={gold} target={tgt_str!r} "
                          f"tokens={len(tgt_ids)} roundtrip_ok={rtxt == tgt_str}", flush=True)
            rec = eng.prepare_sample(frames, q0, gold=gold, task=task, resize=args.resize,
                                     qfirst=not args.no_qfirst, posreset=not args.no_posreset)
            if rec is None:
                n_skip += 1
                continue
            d = eng.build_training_cache(rec, tgt_ids, anch=anch,
                                         truncate=args.truncate_at is not None)
            # readout slice at the tail of tgt_ids (count tokens + EOS)
            if args.scratchpad_format == "poslist":
                ans_sfx = f" {gold}"
            elif args.task == "mmred_hf":
                ans_sfx = " " + tgt_str.rsplit(": ", 1)[-1]  # value END after the anchor
            else:
                ans_sfx = f" {gold} END"
            d["ans_k"] = (len(tok(ans_sfx, add_special_tokens=False).input_ids) + 1
                          if tgt_ids else 0)
            if args.task == "mmred_hf" and args.scratchpad and args.verdict_weight != 1.0:
                d["w"] = verdict_weights(tgt_str, tgt_ids, tok, args.verdict_weight)
            d["sd"] = str(sd)
            data.append(d)
            cache_gb += (d["seq"] + d["e"]) * d["h"].shape[-1] * 2 / 1e9
            n_done += 1
            n_root += 1
            if n_done % 100 == 0:
                print(f"  prep {n_done} (skip {n_skip}) {time.time()-t0:.0f}s "
                      f"cache {cache_gb:.1f}GB", flush=True)
    print(f"prep done: n={n_done} skip={n_skip} cache {cache_gb:.1f}GB "
          f"tasks {Counter(d['task'] for d in data)} "
          f"golds {Counter(d['gold'] for d in data)}", flush=True)

    split_rng = np.random.default_rng(args.seed if args.split_seed is None else args.split_seed)
    rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(n_done)
    n_tr = int(n_done * args.train_frac)
    tr_idx, ev_idx = order[:n_tr], order[n_tr:]
    (out / "eval_dirs.txt").write_text("\n".join(data[i]["sd"] for i in ev_idx) + "\n")
    (out / "train_dirs.txt").write_text("\n".join(data[i]["sd"] for i in tr_idx) + "\n")

    def evaluate():
        """scratchpad: teacher-forced COUNT-token acc (headline) + tf-exact; digit: emitted acc."""
        hits = exact = 0
        mae, mae_n = 0.0, 0
        per_task: dict = {}
        with torch.no_grad():
            for i in ev_idx:
                d = data[i]
                g = d["gold"]
                if args.scratchpad:
                    hs = eng.top_hidden(d)
                    seqp, e, ak = d["seq"], d["e"], d["ans_k"]
                    lg = eng.head(hs[0, seqp - 1 : seqp + e - 1])
                    preds = lg.argmax(-1).tolist()
                    tf_all = preds == d["tgt"]
                    cnt_ok = preds[-ak:-1] == d["tgt"][-ak:-1]
                    exact += tf_all
                    hits += cnt_ok
                    ptxt = tok.decode(preds[-ak:-1]).strip()
                    if ptxt.endswith("END"):
                        ptxt = ptxt[:-3].strip()
                    if ptxt.isdigit() and str(g).isdigit():
                        mae += abs(int(ptxt) - int(g))
                        mae_n += 1
                    ok = cnt_ok
                else:
                    lg = eng.head(eng.top_hidden(d)[0, -1])
                    dg = int(np.argmax([float(lg[t2]) for t2 in digit_ids]))
                    ok = dg == g
                    hits += ok
                    mae += abs(dg - g)
                    mae_n += 1
                pt = per_task.setdefault(d["task"], [0, 0])
                pt[1] += 1
                pt[0] += ok
        pts = " ".join(f"{t}:{c}/{n2}" for t, (c, n2) in sorted(per_task.items()))
        if args.scratchpad:
            pts += f"  tf-exact {exact/len(ev_idx):.3f}"
        return hits / len(ev_idx), mae / max(mae_n, 1), pts, exact / len(ev_idx)

    acc0, mae0, pt0, ex0 = evaluate()
    print(f"[ep 0] acc {acc0:.3f} MAE {mae0:.2f}  [{pt0}]", flush=True)
    lines = [f"=== CARRIER TRAINER (L_open={args.l_open}, r={args.rank}, frozen e_c, "
             f"n={n_done}, train={len(tr_idx)}, scratchpad={args.scratchpad}, "
             f"fmt={args.scratchpad_format}, jitter={args.jitter_gap}, "
             f"noqfirst={args.no_qfirst}, noposreset={args.no_posreset}, "
             f"trunc={args.truncate_at}, roots={args.data_root}, "
             f"gate={args.gate}@{args.gate_layers} b0={args.gate_b0} lr={args.gate_lr} "
             f"gate_only={args.gate_only}) ===",
             f"ep0 acc {acc0:.3f} mae {mae0:.2f} [{pt0}]"]
    # model selection: (TF-count acc, tf-exact) lexicographic — TF-count saturates early
    # and acc-only selection picked weaker-transcript ckpts (2026-07-21 lesson)
    best = ((acc0, ex0), 0)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        tot = 0.0
        te = time.time()
        if gate is not None:
            gate.reset_stats()
        for step, i in enumerate(tr_idx):
            d = data[i]
            if args.scratchpad:
                hs = eng.top_hidden(d, jitter_gap=args.jitter_gap, grad_ckpt=args.grad_ckpt)
                lg = eng.head(hs[0, d["seq"] - 1 : d["seq"] + d["e"] - 1])
                tgt_t = torch.tensor(d["tgt"], device=dev)
                if "w" in d:
                    ce = F.cross_entropy(lg, tgt_t, reduction="none")
                    wt = torch.tensor(d["w"], device=dev, dtype=ce.dtype)
                    loss = (ce * wt).sum() / wt.sum()
                else:
                    loss = F.cross_entropy(lg, tgt_t)
            else:
                lg = eng.head(eng.top_hidden(d, jitter_gap=args.jitter_gap,
                                             grad_ckpt=args.grad_ckpt)[0, -1])
                loss = F.cross_entropy(lg.unsqueeze(0),
                                       torch.tensor([digit_ids[d["gold"]]], device=dev))
            (loss / 8).backward()
            tot += float(loss)
            if (step + 1) % 8 == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        gline = ""
        if gate is not None:
            # MANDATORY instrumentation: a gate still sitting at ~1.0 learned nothing and
            # that arm is VOID, not a null result. Read BEFORE evaluate() so the number
            # reflects the training forwards only.
            gline = gate.stats_line()
            print(f"[ep {ep}] {gline}", flush=True)
        acc, mae, pts, ex = evaluate()
        print(f"[ep {ep}] loss {tot/len(tr_idx):.4f} acc {acc:.3f} MAE {mae:.2f} [{pts}] "
              f"({time.time()-te:.0f}s/ep)", flush=True)
        lines.append(f"ep{ep} loss {tot/len(tr_idx):.4f} acc {acc:.3f} mae {mae:.2f} [{pts}]")
        if gline:
            lines.append(f"ep{ep} {gline}")
        if (acc, ex) > best[0]:
            best = ((acc, ex), ep)
            save_carrier_layer_ckpt(
                out / "carrier_layer_best.pt", e_c=e_c, lora=lora, epoch=ep, acc=acc,
                scratchpad=args.scratchpad, scratchpad_format=args.scratchpad_format,
                running_tally=args.running_tally, pos_couple=args.pos_couple,
                jitter_gap=args.jitter_gap, truncate_at=args.truncate_at,
                **({"gate": gate.state(), "gate_only": args.gate_only} if gate else {}))
    lines.append(f"BEST acc {best[0][0]:.3f} (tf-exact {best[0][1]:.3f}) @ ep {best[1]} "
                 f"(scaffold 0.998; frozen 0.219)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))
    print("wrote", out)
    lora.remove()
    if gate is not None:
        gate.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
