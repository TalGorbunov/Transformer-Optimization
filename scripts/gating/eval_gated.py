#!/usr/bin/env python3
"""P3.5 of the gating campaign: the capacity-vs-interference discriminator.

Evaluates a carrier-layer checkpoint — with its gate re-attached if it has one — across
sample roots, one root per report row. Two sweeps, both eval-only, on data that already
exists:

  distractor axis   hold the evidence distribution, vary N
                    data/mmred_images_park/seq_len_{2..8}/all_uniform
  capacity axis     hold N=8, vary the evidence count
                    data/mmred_images_park/seq_len_8/by_evidence_count/exact_{0..8}

Interpretation (write the verdict in STATE):
  gain on the distractor axis only -> bottleneck is INTERFERENCE, gating is the right
  family, G2 should beat G1;  flat on both / gain only on the capacity axis -> CAPACITY
  wall, gating cannot help, and the honest-null chapter has a mechanism attached.

Metrics. `tf` (default) is one forward per sample: teacher-forced COUNT-token accuracy
(the trainer's headline `acc`) + tf-exact over the whole transcript — directly comparable
to the P3 report lines. `decode` additionally runs the greedy scratchpad exam
(`acc_raw`), which is what Figure 1 plots; it costs ~decode_tokens forwards per sample,
so keep --limit small for it.

Nothing here touches gnnformer core: it composes CarrierEngine, gnnformer.gating and the
same target builders the trainer uses.

Usage:
  python scripts/gating/eval_gated.py --ckpt <run>/carrier_layer_best.pt \
      --roots-file slurm/lib/roots_distractor.txt --limit 100 \
      --output outputs/gating/p35_discriminator/<stamp>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.data import (
    frame_attr_labels,
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
    read_dirs_file,
)
from gnnformer.engine import CarrierEngine
from gnnformer.gating import attach_gate
from gnnformer.runtime import attention_dims, get_layers, load_runtime
from gnnformer.scratchpad import build_target, build_target_fmt, build_target_tally


def build_caption_target(fmt: str, running_tally: bool, task: str, q0: str, states, evid,
                         aux, gold: int, n_frames: int) -> str:
    """Byte-identical to scripts/train_carrier_layer.py's park/scratchpad target build."""
    if fmt != "poslist":
        labels = (frame_attr_labels(task, q0, states, evid)
                  if (fmt in ("caption", "chunked") or task == "rooms") else None)
        return build_target_fmt(fmt, task, evid, aux, gold, NF=n_frames, labels=labels)
    return (build_target_tally if running_tally else build_target)(task, evid, aux, gold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="carrier_layer_best.pt (gate state, if any, rides in it)")
    ap.add_argument("--roots-file", default=None, help="one sample root per line (no comma lists)")
    ap.add_argument("--data_root", default=None, help="comma-separated roots (alternative)")
    ap.add_argument("--dirs-file", default=None, help="pin the exact sample dirs (single cell)")
    ap.add_argument("--limit", type=int, default=100, help="per-root cap")
    ap.add_argument("--per-gold", type=int, default=0, metavar="K",
                    help="cap samples per GOLD value within each root (0 = off). This is "
                         "what makes the distractor axis clean: seq_len_N/all_uniform is "
                         "uniform over golds 0..N, so sweeping N alone also grows the "
                         "evidence count. Capping per gold gives a balanced (N, evidence) "
                         "grid from one sweep — read columns for distractors, rows for "
                         "capacity")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--shuffle-dirs", type=int, default=0, metavar="SEED")
    ap.add_argument("--mode", choices=("tf", "decode", "both"), default="tf")
    ap.add_argument("--decode-tokens", type=int, default=320)
    ap.add_argument("--fast-decode", action="store_true",
                    help="cached incremental decode (16-311x). Safe with a gate: "
                         "decode_fast re-runs every layer over [cache || appended], so "
                         "the hooks fire and the gate — a pointwise function of X — is "
                         "applied identically to cached and appended rows. --exactness-n "
                         "verifies that against the plain decode")
    ap.add_argument("--exactness-n", type=int, default=0, metavar="K",
                    help="check decode_fast against decode_scratchpad on the first K "
                         "samples of each root and report token identity")
    ap.add_argument("--no-gate", action="store_true", help="ablate the ckpt's gate at eval time")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    roots: List[str] = []
    if args.roots_file:
        roots = [ln.strip() for ln in Path(args.roots_file).read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    elif args.data_root:
        roots = [r.strip() for r in args.data_root.split(",") if r.strip()]
    elif args.dirs_file:
        roots = [args.dirs_file]
    else:
        raise SystemExit("give --roots-file, --data_root or --dirs-file")

    rt = load_runtime(args.model) if args.model else load_runtime()
    layers = get_layers(rt.model)
    tok = rt.tokenizer
    dev = rt.device
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = load_carrier_layer_ckpt(Path(args.ckpt))
    e_c = torch.as_tensor(ck.e_c).to(dev)
    lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha, device=dev,
                       state=ck.lora_state)
    gate = None
    gstate = ck.extra.get("gate")
    if gstate is not None and not args.no_gate:
        dims = attention_dims(rt.model)
        gate = attach_gate(layers, gstate["layer_ids"], gstate["variant"],
                           hidden=dims["hidden_size"], n_heads=dims["n_heads"],
                           n_kv=dims["n_kv"], head_dim=dims["head_dim"],
                           device=dev, state=gstate)
        print(f"[gate] {gstate['variant']} on {gstate['layer_ids']} b0={gstate.get('b0')} "
              f"params {gate.num_parameters()}", flush=True)
    else:
        print(f"[gate] none ({'ablated by --no-gate' if gstate is not None else 'ckpt has no gate'})",
              flush=True)
    eng = CarrierEngine(rt, l_open=ck.l_open, e_c=e_c)
    fmt = ck.scratchpad_format
    running_tally = bool(ck.extra.get("running_tally", True))
    print(f"[ckpt] {args.ckpt}: L*={ck.l_open} rank={ck.rank} fmt={fmt} "
          f"scratchpad={ck.scratchpad} ep={ck.epoch} acc={ck.acc}", flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for root in roots:
        p = Path(root)
        dirs = (read_dirs_file(p) if p.is_file()
                else (iter_sample_dirs_shuffled(p, args.shuffle_dirs)
                      if args.shuffle_dirs is not None else iter_sample_dirs(p)))
        if args.per_gold > 0:
            kept, seen = [], {}
            for sd in dirs:
                try:
                    g = int(json.loads((Path(sd) / "metadata.json").read_text())["answer"])
                except Exception:
                    continue
                if seen.get(g, 0) < args.per_gold:
                    seen[g] = seen.get(g, 0) + 1
                    kept.append(sd)
            dirs = kept
            print(f"[{root}] per-gold {args.per_gold} -> {len(dirs)} dirs "
                  f"({dict(sorted(seen.items()))})", flush=True)
        dirs = dirs[: args.limit]
        n = n_skip = 0
        tf_hits = tf_exact = dec_hits = dec_parse_fail = 0
        mae = 0.0
        per_gold: Dict[int, List[int]] = {}
        exact_checks: List[Any] = []
        if gate is not None:
            gate.reset_stats()
        for sd in dirs:
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(Path(sd))
                gold = int(str(a0).strip())
            except Exception:
                n_skip += 1
                continue
            parsed = parse_task_labels(q0, states, gold)
            if parsed is None:
                n_skip += 1
                continue
            task, evid, aux = parsed
            rec = eng.prepare_sample(frames, q0, gold=gold, task=task, resize=args.resize,
                                     with_masks=(args.mode != "tf"))
            if rec is None:
                n_skip += 1
                continue
            ok_tf = ok_ex = ok_dec = None
            if args.mode in ("tf", "both"):
                tgt_str = build_caption_target(fmt, running_tally, task, q0, states, evid,
                                               aux, gold, len(frames))
                tgt_ids = tok(tgt_str, add_special_tokens=False).input_ids + [tok.eos_token_id]
                d = eng.build_training_cache(rec, tgt_ids)
                ans_sfx = f" {gold}" if fmt == "poslist" else f" {gold} END"
                ak = len(tok(ans_sfx, add_special_tokens=False).input_ids) + 1
                with torch.no_grad():
                    hs = eng.top_hidden(d)
                    lg = eng.head(hs[0, d["seq"] - 1 : d["seq"] + d["e"] - 1])
                preds = lg.argmax(-1).tolist()
                ok_tf = preds[-ak:-1] == d["tgt"][-ak:-1]
                ok_ex = preds == d["tgt"]
                tf_hits += int(ok_tf)
                tf_exact += int(ok_ex)
                del d, hs, lg
            if args.mode in ("decode", "both"):
                with torch.no_grad():
                    if args.fast_decode:
                        val, _txt, dt, _pf = eng.decode_fast(
                            rec, decode_tokens=args.decode_tokens, fmt=fmt)
                        if n < args.exactness_n:
                            vS, _tS, dS = eng.decode_scratchpad(
                                rec, decode_tokens=args.decode_tokens, fmt=fmt)
                            exact_checks.append((dt == dS, val == vS))
                    else:
                        val, _txt, dt = eng.decode_scratchpad(
                            rec, decode_tokens=args.decode_tokens, fmt=fmt)
                dec_parse_fail += int(val is None)
                ok_dec = val == gold
                dec_hits += int(ok_dec)
                if val is not None:
                    mae += abs(int(val) - gold)
            # [n, count-token ok, tf-exact ok, decoded ok] — the count metric saturates
            # at N<=8, so the grid must carry the harder ones too
            pg = per_gold.setdefault(gold, [0, 0, 0, 0])
            pg[0] += 1
            pg[1] += int(bool(ok_tf))
            pg[2] += int(bool(ok_ex))
            pg[3] += int(bool(ok_dec))
            n += 1
            if n % 25 == 0:
                print(f"  [{root}] {n}/{len(dirs)} ({time.time()-t0:.0f}s)", flush=True)
        if n == 0:
            print(f"[warn] no usable samples in {root}", flush=True)
            continue
        row = {"root": root, "n": n, "skip": n_skip,
               "tf_acc": tf_hits / n if args.mode != "decode" else float("nan"),
               "tf_exact": tf_exact / n if args.mode != "decode" else float("nan"),
               "dec_acc": dec_hits / n if args.mode != "tf" else float("nan"),
               "parse_fail": dec_parse_fail / n if args.mode != "tf" else float("nan"),
               "mae": mae / max(dec_hits or n, 1) if args.mode != "tf" else float("nan"),
               "gold_mean": float(np.mean([g for g, v in per_gold.items() for _ in range(v[1])])),
               "gate_mean": (float(np.mean(list(gate.mean_scores().values())))
                             if gate is not None and gate.mean_scores() else float("nan")),
               "fast_exact_ok": (f"{sum(1 for a, _ in exact_checks if a)}/{len(exact_checks)}"
                                 if exact_checks else ""),
               "per_gold": json.dumps({str(g): v for g, v in sorted(per_gold.items())})}
        rows.append(row)
        if exact_checks:
            bad = [i for i, (tok_id, _) in enumerate(exact_checks) if not tok_id]
            print(f"[exactness] {root}: decode_fast token-identical to decode_scratchpad "
                  f"on {len(exact_checks)-len(bad)}/{len(exact_checks)}"
                  + (f" MISMATCH at {bad}" if bad else ""), flush=True)
        print(f"[root] {root}: n={n} tf_acc {row['tf_acc']:.3f} tf_exact {row['tf_exact']:.3f} "
              f"dec {row['dec_acc']:.3f} gate_mean {row['gate_mean']:.4f}", flush=True)

    lora.remove()
    if gate is not None:
        gate.remove()

    with (out / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"=== GATED EVAL (ckpt={args.ckpt}, mode={args.mode}, limit={args.limit}, "
             f"gate={'none' if gate is None else gate.variant}) ===",
             f"{'root':<62} {'n':>4} {'tf_acc':>7} {'tf_exact':>9} {'dec_acc':>8} "
             f"{'pfail':>6} {'gold_mu':>8} {'gate_mu':>8}"]
    for r in rows:
        lines.append(f"{r['root']:<62} {r['n']:>4} {r['tf_acc']:>7.3f} {r['tf_exact']:>9.3f} "
                     f"{r['dec_acc']:>8.3f} {r['parse_fail']:>6.3f} {r['gold_mean']:>8.2f} "
                     f"{r['gate_mean']:>8.4f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    (out / "ABOUT.md").write_text(
        "# P3.5 — capacity vs interference discriminator\n\n"
        f"Checkpoint: `{args.ckpt}` (gate re-attached from the ckpt's `gate` entry"
        f"{'; ABLATED by --no-gate' if args.no_gate else ''}).\n"
        f"Mode `{args.mode}`: tf = teacher-forced COUNT-token accuracy + tf-exact (the\n"
        "trainer's headline metric, one forward per sample); decode = greedy scratchpad\n"
        "exam (acc_raw). One row per sample root.\n\n"
        "Distractor axis = `mmred_images_park/seq_len_{2..8}/all_uniform` (evidence\n"
        "distribution fixed by the generator, N grows). Capacity axis =\n"
        "`mmred_images_park/seq_len_8/by_evidence_count/exact_{0..8}` (N fixed at 8,\n"
        "evidence count = the gold answer varies). Artifacts: report.txt, cells.csv.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
