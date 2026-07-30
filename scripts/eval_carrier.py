#!/usr/bin/env python3
"""The carrier exam: greedy decode of a trained carrier-layer checkpoint on a pinned
sample set, plus the truncation/efficiency instruments.

Anchors this script must reproduce (RESULTS.md): caption winner ckpt on arm-A's
eval_dirs_N32all.txt -> acc_raw 0.987, parse_fail 0.000 ([2026-07-24] FORMAT SWEEP /
P1.1 seed0); l12v2 ckpt on the same dirs -> 0.953 ([2026-07-22] l12v2).

Modes:
  default            greedy scratchpad (or digit) decode, full lo/hi forward
  --drop-frame-kv    decoded rows lose frame columns (mask-only)
  --truncate-at L    physically drop frame rows entering layer L (implies drop-kv)
  --fast-decode      cached incremental decode (exact vs the mask semantics, 16-311x)
  --exactness-check  E1: decode baseline AND flagged arms, report token identity/timing
  --chunked-prefill  E5: per-block short forwards below L (+ --verify-chunked)
  --dump-carrier-states L1,L2  carrier-row hidden-state cache (gate_tally-compatible)

Usage:
  python scripts/eval_carrier.py --ckpt checkpoints/carrier_layer_fmt_caption_best.pt \
      --dirs-file <run>/eval_dirs_N32all.txt --limit 150 --decode-tokens 320
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.data import (
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
    read_dirs_file,
)
from gnnformer.engine import CarrierEngine
from gnnformer.metrics import format_gold_histogram
from gnnformer.runtime import get_layers, load_runtime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="carrier_layer_best.pt")
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform",
                    help="comma-separated roots (ignored when --dirs-file is given)")
    ap.add_argument("--dirs-file", default=None, help="file pinning the exact sample dirs")
    ap.add_argument("--limit", type=int, default=300, help="per root / dirs-file cap")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--decode-tokens", type=int, default=3,
                    help="greedy decode cap (scratchpad ckpts auto-raise to >=48)")
    ap.add_argument("--scratchpad-format", default=None,
                    help="override the ckpt's stored format (poslist/scan/caption/chunked)")
    ap.add_argument("--alien-task", action="store_true",
                    help="accept non-MMRED questions (decode-only scoring; e.g. MLVU-AC)")
    ap.add_argument("--no-reset-positions", action="store_true",
                    help="eval-time posreset ablation — a TRAIN/TEST MISMATCH by design; "
                         "report it as such, not as an architectural verdict")
    ap.add_argument("--dump-decodes", type=int, default=3)
    ap.add_argument("--drop-frame-kv", action="store_true")
    ap.add_argument("--truncate-at", type=int, default=None, metavar="L")
    ap.add_argument("--fast-decode", action="store_true")
    ap.add_argument("--exactness-check", action="store_true")
    ap.add_argument("--chunked-prefill", action="store_true")
    ap.add_argument("--verify-chunked", action="store_true")
    ap.add_argument("--dump-carrier-states", default=None, metavar="LAYERS")
    ap.add_argument("--model", default=None, help="override model id (default: 7B)")
    ap.add_argument("--output", default="outputs/carrier/exam")
    args = ap.parse_args()

    trunc_flags = (args.drop_frame_kv or args.truncate_at is not None or args.fast_decode
                   or args.exactness_check or args.dump_carrier_states is not None)
    dump_layers = ([int(x) for x in args.dump_carrier_states.split(",")]
                   if args.dump_carrier_states else [])
    if args.fast_decode and not (args.drop_frame_kv or args.truncate_at is not None):
        ap.error("--fast-decode requires --drop-frame-kv or --truncate-at")
    if args.exactness_check and not (args.drop_frame_kv or args.truncate_at is not None):
        ap.error("--exactness-check requires --drop-frame-kv or --truncate-at")
    if args.chunked_prefill and (args.truncate_at is None or not args.fast_decode):
        ap.error("--chunked-prefill requires --truncate-at and --fast-decode")
    if args.verify_chunked and not args.chunked_prefill:
        ap.error("--verify-chunked requires --chunked-prefill")

    ck = load_carrier_layer_ckpt(Path(args.ckpt))
    sfmt = args.scratchpad_format or ck.scratchpad_format
    if ck.pos_couple and trunc_flags:
        raise SystemExit("truncation flags are not supported with pos-coupled ckpts")
    if args.chunked_prefill and args.truncate_at != ck.l_open:
        raise SystemExit(f"--chunked-prefill is only exact when --truncate-at == L_open "
                         f"({ck.l_open}), got {args.truncate_at}")
    decode_tokens = max(args.decode_tokens, 48) if ck.scratchpad else args.decode_tokens
    print(f"[ckpt] {args.ckpt} (ep {ck.epoch}, acc {ck.acc}, L_open={ck.l_open}, "
          f"rank={ck.rank}, alpha={ck.alpha}, scratchpad={ck.scratchpad}, fmt={sfmt}, "
          f"pos_couple={ck.pos_couple})", flush=True)

    rt = load_runtime(args.model) if args.model else load_runtime()
    layers = get_layers(rt.model)
    e_c = ck.e_c.float().to(rt.device)
    lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha,
                       device=rt.device, state=ck.lora_state)
    eng = CarrierEngine(rt, l_open=ck.l_open, e_c=e_c, pos_couple=ck.pos_couple)
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S")
                               + f"_L{ck.l_open}_r{ck.rank}_{sfmt}")
    out.mkdir(parents=True, exist_ok=True)

    # ---- sample stream ----
    if args.dirs_file:
        sample_iter = [("dirsfile", sd) for sd in read_dirs_file(Path(args.dirs_file))]
        print(f"[dirs-file] {len(sample_iter)} dirs from {args.dirs_file}", flush=True)
        limits = {"dirsfile": args.limit}
    else:
        roots = [r.strip() for r in args.data_root.split(",") if r.strip()]
        sample_iter = []
        for root in roots:
            rd = (iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs)
                  if args.shuffle_dirs is not None else iter_sample_dirs(Path(root)))
            sample_iter.extend((root, sd) for sd in rd)
        limits = {r: args.limit for r in roots}

    # ---- explicit stats (no mode-dependent column meanings) ----
    n = 0
    acc_raw = 0          # parsed answer == gold
    parse_fail = 0       # scratchpad decode produced no parseable count
    acc_restricted = 0   # digit mode only: 0-9-restricted first-token argmax == gold
    mae_sum, mae_n = 0.0, 0
    dec_toks = 0
    per_count, per_task = {}, {}
    golds = []
    exact = None
    chunkv = None
    fastt = None
    dump = None
    n_skip = 0
    done_per_root = {k: 0 for k in limits}
    t0 = time.time()

    for root, sd in sample_iter:
        if done_per_root[root] >= limits[root]:
            continue
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            n_skip += 1
            continue
        pt = parse_task_labels(q0, states, gold)
        if pt is None and args.alien_task:
            pt = ("steps", set(), None)
        if pt is None:
            n_skip += 1
            continue
        task, evid, _aux = pt
        rec = eng.prepare_sample(frames, q0, gold=gold, task=task, resize=args.resize,
                                 posreset=not args.no_reset_positions,
                                 with_masks=True, with_trunc_cols=trunc_flags)
        if rec is None:
            n_skip += 1
            continue
        if n == 0 and trunc_flags:
            ks, fs = set(rec["keep"]), set(rec["fcols"])
            assert not (ks & fs) and len(ks) + len(fs) == rec["seq"], "keep/frame split"
            print(f"[trunc-debug] seq={rec['seq']} keep={len(ks)} frame_cols={len(fs)} "
                  f"trunc_at={args.truncate_at} dropkv={args.drop_frame_kv} "
                  f"fast={args.fast_decode}", flush=True)

        if dump_layers:
            with torch.no_grad():
                cD = eng.prefill_capture(rec, args.truncate_at)
            a0d = rec["blocks"][0][0]
            NF = len(rec["cpos"])
            if dump is None:
                dump = {"rep": {L: [] for L in dump_layers}, "labels": [], "gold": [], "sd": []}
            for L in dump_layers:
                dump["rep"][L].append(cD[0][L][a0d:a0d + NF].float().cpu().numpy().astype(np.float16))
            dump["labels"].append(np.array([1 if t in evid else 0 for t in range(NF)], dtype=np.int64))
            dump["gold"].append(gold)
            dump["sd"].append(str(sd))
            del cD

        with torch.no_grad():
            if ck.scratchpad:
                if args.exactness_check:
                    cuda = torch.cuda.is_available()

                    def _arm(fn):
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        ta = time.time()
                        r = fn()
                        if cuda:
                            torch.cuda.synchronize()
                        pk = (torch.cuda.max_memory_allocated() / 2**30) if cuda else 0.0
                        return r, time.time() - ta, pk

                    (vB, xB, kB), tB, pB = _arm(lambda: eng.decode_scratchpad(
                        rec, decode_tokens=decode_tokens, fmt=sfmt))
                    (vM, _xM, kM), tM, _ = _arm(lambda: eng.decode_scratchpad(
                        rec, decode_tokens=decode_tokens, fmt=sfmt,
                        dropkv=True, trunc=args.truncate_at))
                    if exact is None:
                        exact = {"n": 0, "ident": 0, "ans": 0, "fident": 0, "fans": 0,
                                 "nf": 0, "tB": 0.0, "tM": 0.0, "tF": 0.0, "pB": 0.0, "pF": 0.0}
                    exact["n"] += 1
                    exact["ident"] += int(kB == kM)
                    exact["ans"] += int(vB == vM)
                    exact["tB"] += tB
                    exact["tM"] += tM
                    exact["pB"] = max(exact["pB"], pB)
                    line = (f"  [exact] gold={gold} N={len(rec['cpos'])} base({vB},{len(kB)}t,"
                            f"{tB:.1f}s) mask({vM},{len(kM)}t,{tM:.1f}s) ident={kB == kM}")
                    if args.fast_decode:
                        (vF, _xF, kF, pf_s), tF, pF = _arm(lambda: eng.decode_fast(
                            rec, decode_tokens=decode_tokens, fmt=sfmt, trunc=args.truncate_at))
                        exact["nf"] += 1
                        exact["fident"] += int(kF == kM)
                        exact["fans"] += int(vF == vM)
                        exact["tF"] += tF
                        exact["pF"] = max(exact["pF"], pF)
                        line += f" fast({vF},{len(kF)}t,{tF:.1f}s,pf{pf_s:.1f}s) fast==mask={kF == kM}"
                    print(line, flush=True)
                    val, txt, ndec = vB, xB, len(kB)  # headline scores the BASELINE arm
                elif args.verify_chunked:
                    cA = eng.prefill_capture(rec, args.truncate_at)
                    cB = eng.prefill_chunked(rec, args.truncate_at)
                    hA = cA[0][args.truncate_at].float()
                    hB = cB[0][args.truncate_at].float()
                    a0v = rec["blocks"][0][0]
                    NFv = len(rec["cpos"])
                    d_q = float((hA[:a0v] - hB[:a0v]).abs().max())
                    d_c = float((hA[a0v:a0v + NFv] - hB[a0v:a0v + NFv]).abs().max())
                    d_t = float((hA[a0v + NFv:] - hB[a0v + NFv:]).abs().max())
                    del cA, cB
                    vD, _xD, kD, _ = eng.decode_fast(rec, decode_tokens=decode_tokens,
                                                     fmt=sfmt, trunc=args.truncate_at)
                    val, txt, dtoks, _pf = eng.decode_fast(rec, decode_tokens=decode_tokens,
                                                           fmt=sfmt, trunc=args.truncate_at,
                                                           chunked=True)
                    if chunkv is None:
                        chunkv = {"n": 0, "ident": 0, "ans": 0, "dq": 0.0, "dc": 0.0, "dt": 0.0}
                    chunkv["n"] += 1
                    chunkv["ident"] += int(kD == dtoks)
                    chunkv["ans"] += int(vD == val)
                    chunkv["dq"] = max(chunkv["dq"], d_q)
                    chunkv["dc"] = max(chunkv["dc"], d_c)
                    chunkv["dt"] = max(chunkv["dt"], d_t)
                    print(f"  [chunkverify] gold={gold} dq={d_q:.4f} dc={d_c:.4f} dt={d_t:.4f} "
                          f"tok-ident={kD == dtoks}", flush=True)
                    ndec = len(dtoks)
                elif args.fast_decode:
                    cuda = torch.cuda.is_available()
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    tf0 = time.time()
                    val, txt, dtoks, pf_s = eng.decode_fast(
                        rec, decode_tokens=decode_tokens, fmt=sfmt,
                        trunc=args.truncate_at, chunked=args.chunked_prefill)
                    if cuda:
                        torch.cuda.synchronize()
                    if fastt is None:
                        fastt = {"pf": 0.0, "dec": 0.0, "tok": 0, "vram": 0.0}
                    fastt["pf"] += pf_s
                    fastt["dec"] += time.time() - tf0 - pf_s
                    fastt["tok"] += len(dtoks)
                    if cuda:
                        fastt["vram"] = max(fastt["vram"], torch.cuda.max_memory_allocated() / 2**30)
                    ndec = len(dtoks)
                else:
                    val, txt, dtoks = eng.decode_scratchpad(
                        rec, decode_tokens=decode_tokens, fmt=sfmt,
                        dropkv=(args.drop_frame_kv or args.truncate_at is not None),
                        trunc=args.truncate_at)
                    ndec = len(dtoks)
                dec_toks += ndec
                if n < args.dump_decodes:
                    print(f"  [decode-sample] gold={gold} parsed={val} text={txt!r}", flush=True)
                parse_fail += int(val is None)
            else:
                val, fd = eng.decode_answer(
                    rec, decode_tokens=decode_tokens,
                    dropkv=(args.drop_frame_kv or args.truncate_at is not None) if trunc_flags else False,
                    trunc=args.truncate_at)
                acc_restricted += int(fd == gold)
                if val is None and fd is not None:
                    mae_sum += abs(fd - gold)
                    mae_n += 1

        n += 1
        golds.append(gold)
        acc_raw += int(val == gold)
        if val is not None:
            mae_sum += abs(val - gold)
            mae_n += 1
        pg = per_count.setdefault(gold, [0, 0])
        pg[0] += int(val == gold)
        pg[1] += 1
        tt = per_task.setdefault(task, [0, 0])
        tt[0] += int(val == gold)
        tt[1] += 1
        done_per_root[root] += 1
        if n % 25 == 0:
            print(f"  eval {n} (skip {n_skip}) {time.time()-t0:.0f}s", flush=True)

    nn = max(n, 1)
    lines = [
        f"=== CARRIER EXAM (ckpt={args.ckpt}, n={n}, decode<={decode_tokens}, "
        f"scratchpad={ck.scratchpad}, fmt={sfmt}, posreset={not args.no_reset_positions}, "
        f"data={args.dirs_file or args.data_root}) ===",
        "[gold-hist] " + format_gold_histogram(golds),
        f"acc_raw {acc_raw/nn:.3f}   "
        + (f"parse_fail {parse_fail/nn:.3f}" if ck.scratchpad
           else f"acc_digit_restricted {acc_restricted/nn:.3f}")
        + f"   MAE {mae_sum/max(mae_n,1):.2f} (over {mae_n} parsed)",
        "per-count acc/n: " + " ".join(f"g{g}:{a}/{t}" for g, (a, t) in sorted(per_count.items())),
        "per-task acc: " + " ".join(f"{t}:{a}/{c}" for t, (a, c) in sorted(per_task.items())),
    ]
    if ck.scratchpad:
        lines.append(f"mean decode tokens {dec_toks/nn:.1f} (cap {decode_tokens})")
    if trunc_flags:
        lines.append(f"[trunc-flags] dropkv={args.drop_frame_kv} truncate_at={args.truncate_at} "
                     f"fast={args.fast_decode} exact={args.exactness_check}")
    if exact:
        lines.append(f"[exactness] mask-only identical {exact['ident']}/{exact['n']}, "
                     f"answer-equal {exact['ans']}/{exact['n']}"
                     + (f"; fast==mask {exact['fident']}/{exact['nf']}, fast answer-equal "
                        f"{exact['fans']}/{exact['nf']}" if exact["nf"] else ""))
        lines.append(f"[timing] decode s/sample: base {exact['tB']/exact['n']:.1f} "
                     f"mask {exact['tM']/exact['n']:.1f}"
                     + (f" fast {exact['tF']/exact['nf']:.1f} -> speedup base/fast "
                        f"{(exact['tB']/exact['n'])/max(exact['tF']/exact['nf'],1e-9):.1f}x"
                        if exact["nf"] else ""))
        lines.append(f"[vram] peak GiB: base {exact['pB']:.1f} fast {exact['pF']:.1f}")
    if chunkv:
        lines.append(f"[chunkverify] tok-ident {chunkv['ident']}/{chunkv['n']}, answer-equal "
                     f"{chunkv['ans']}/{chunkv['n']}; max|dh_L*| question {chunkv['dq']:.4f} "
                     f"carriers {chunkv['dc']:.4f} tail {chunkv['dt']:.4f}")
    if fastt:
        lines.append(f"[fast-timing] prefill {fastt['pf']/nn:.2f} s/sample, decode "
                     f"{fastt['dec']/nn:.2f} s/sample, {fastt['tok']/max(fastt['dec'],1e-9):.1f} "
                     f"tok/s, peak VRAM {fastt['vram']:.1f} GiB (chunked={args.chunked_prefill})")
    if dump:
        torch.save({"rep": {L: np.stack(v) for L, v in dump["rep"].items()},
                    "labels": np.stack(dump["labels"]),
                    "gold": np.array(dump["gold"], dtype=np.int64),
                    "sd": dump["sd"], "ckpt": args.ckpt, "truncate_at": args.truncate_at},
                   out / "carrier_states_cache.pt")
        lines.append(f"[dump] carrier_states_cache.pt: n={len(dump['gold'])} "
                     f"layers={sorted(dump['rep'])} (gate_tally-compatible)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    lora.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
