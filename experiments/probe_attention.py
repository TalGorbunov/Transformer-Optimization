#!/usr/bin/env python3
"""The attention photograph on official MMReD rows — DIAG cells D2 / D3 / D8
(docs/DIAGNOSTICS_2026-09-22.md §2): at the answer row, how much attention mass lands on the
evidence frames, the non-evidence frames, and everything else (prompt + sink), per head x layer,
as N (and k) vary; plus the per-head evidence-vs-non-evidence AUC (the "headscan" that found
L24h20 on park data).

sdpa exposes no weights, so each row is recomputed from q_proj / k_proj captures exactly as the
model computes it (experiments/_diag_common.row_attention: rotary q.k * module.scaling + the
injected mask row -> softmax; per-row sum == 1 asserted). The PEFT adapter, if any, is loaded
BEFORE the q/k hooks so trained projections are photographed. The answer row = the last token of
the teacher-forced prefix '{ "answer": "' (the position whose next token is the answer), the same
locus as probe_hahn's `final`.

Arms: plain (paper layout, images then question, plain causal), qfirst (question-first, plain
causal), fenced (question-first + fence + posreset), gated (fenced + oracle gate on the evidence
frames). Knobs --attn-sharpen / --sharpen-from-layer / --attn-logn-sref act through module.scaling,
so photograph and forward see the same temperature by construction. Evidence labels come from
core.mmred.evidence_frames (never from the model); qtypes whose rule returns None are skipped.

Twin: legacy/v1/scripts/sparse/probe_attn_photo.py (+ selfgate/probe_headscan.py). Legacy
layer numbers there were decoder module indices too (--layers 12,16,20,24,27).
Outputs (run dir): mass.csv, block_mass.csv, samples.csv (first-token margin/accuracy), auc.json,
report.txt, config.json.
Usage:
  python experiments/probe_attention.py --config seq_len_32 --qtypes steps_in_room --arm qfirst \
      --k-list 2 4 8 --limit 30 --output outputs/diag/photo/steps_in_room/qfirst/N32
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.fence import FENCED_SDPA, FenceHooks, build_block_mask, hide_cols_for, layout_blocks, reset_positions  # noqa: E402
from core.mmred import NUMERIC_QTYPES, QTYPES, evidence_frames, frames, load_split, recompute_answer, states, stratified_order  # noqa: E402
from core.model import get_layers, get_rope_index_fn, load_runtime, special_ids  # noqa: E402
from experiments._diag_common import (ANSWER_PREFIX, QKCapture, answer_vocab, attention_dims, block_masses, cond_tag,  # noqa: E402
                                      encode_prompt, gold_index, margin_of, rank_auc, row_attention, set_logn, set_sharpen)

ARMS = ("plain", "qfirst", "fenced", "gated")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=["steps_in_room"], choices=QTYPES)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--k-list", nargs="*", type=int, default=None,
                    help="count qtypes: gold strata to fill with --limit samples each (default: first --limit rows of the class-interleaved order)")
    ap.add_argument("--limit", type=int, default=30, help="samples per stratum (or total when no --k-list)")
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 16, 20, 24, 27], help="decoder module indices (0-based)")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--attn-sharpen", type=float, default=0.0)
    ap.add_argument("--sharpen-from-layer", type=int, default=12)
    ap.add_argument("--attn-logn-sref", type=int, default=0)
    ap.add_argument("--max-seq-tokens", type=int, default=60000)
    ap.add_argument("--no-block-csv", action="store_true", help="skip block_mass.csv (large at N=128)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    out.mkdir(parents=True, exist_ok=True)
    cond = cond_tag(args.attn_sharpen, args.sharpen_from_layer, args.attn_logn_sref)
    (out / "config.json").write_text(json.dumps({**vars(args), "cond": cond}, indent=1, default=str))
    layout = "paper" if args.arm == "plain" else "question-first"
    fenced = args.arm in ("fenced", "gated")

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    sid = special_ids(processor)
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    prefix_ids = tok(ANSWER_PREFIX, add_special_tokens=False).input_ids
    dims = attention_dims(model)
    base_scaling = set_sharpen(layers, args.attn_sharpen, args.sharpen_from_layer)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        print(f"[model] adapter {args.adapter}", flush=True)
    hooks = FenceHooks(layers).install()
    qk = QKCapture(layers, args.layers).install()        # post-wrap: LoRA'd projections
    nH = dims["n_heads"]

    mass_f = open(out / "mass.csv", "w", newline="")
    mass_w = csv.writer(mass_f)
    mass_w.writerow(["qid", "qtype", "N", "k", "layer", "head", "evid_mass", "nonevid_mass", "other_mass"])
    blk_f = open(out / "block_mass.csv", "w", newline="")
    blk_w = csv.writer(blk_f)
    blk_w.writerow(["qid", "qtype", "N", "k", "layer", "head", "block", "is_evid", "mass"])
    smp_f = open(out / "samples.csv", "w", newline="")
    smp_w = csv.writer(smp_f)
    smp_w.writerow(["qid", "qtype", "N", "k", "gold", "seq_tokens", "margin", "pred_label", "first_token_correct"])
    auc_pos: Dict[tuple, List[float]] = defaultdict(list)
    auc_neg: Dict[tuple, List[float]] = defaultdict(list)
    totals = defaultdict(int)
    t0 = time.time()

    for qtype in args.qtypes:
        rows = stratified_order(load_split(args.config, args.split, args.data_root, [qtype]), args.seed)
        labels, vocab_ids = answer_vocab(qtype, tok)
        need: Optional[Dict[int, int]] = None
        if args.k_list and qtype in NUMERIC_QTYPES:
            need = {k: args.limit for k in args.k_list}
        n_done = 0
        for row in rows:
            if need is None and n_done >= args.limit:
                break
            if need is not None and all(v <= 0 for v in need.values()):
                break
            question, gold = row["question"], str(row["answer"])
            st = states(row)
            if recompute_answer(qtype, question, st) != gold:
                totals["gold_mismatch"] += 1
                continue
            if need is not None:
                g = int(gold)
                if need.get(g, 0) <= 0:
                    continue
            evid = evidence_frames(qtype, question, st)
            if evid is None:
                totals["no_evidence_rule"] += 1
                continue
            fr = frames(row, args.data_root)
            NF = len(fr)
            k = len(evid)
            enc = encode_prompt(processor, fr, question, layout, prefix_ids, rt.device)
            ids = enc["input_ids"][0].cpu()
            seq = int(ids.shape[0])
            if seq > args.max_seq_tokens:
                totals["too_long"] += 1
                continue
            try:
                blocks, fin = layout_blocks(ids, layout, sid, im_end_id=im_end_id)
            except ValueError:
                totals["layout_skip"] += 1
                continue
            if len(blocks) != NF:
                totals["layout_skip"] += 1
                continue
            mask = pos = None
            if fenced:
                keep = [t in evid for t in range(NF)] if args.arm == "gated" else None
                hide = hide_cols_for(blocks, keep) if keep is not None else []
                mask = build_block_mask(seq, blocks, hide)
                with torch.inference_mode():
                    base_pos, _ = rope_fn(enc["input_ids"], image_grid_thw=enc.get("image_grid_thw"),
                                          attention_mask=enc.get("attention_mask"))
                pos = reset_positions(base_pos, blocks, fin)
            cur = {kk: v for kk, v in enc.items() if kk != "attention_mask"} if fenced else dict(enc)
            qk.clear()
            set_logn(layers, seq, args.attn_logn_sref, base_scaling)
            if mask is not None:
                hooks.set_mask(mask, rt.device)
            try:
                with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
                    outp = model(**cur, position_ids=pos.to(rt.device) if pos is not None else None, use_cache=False)
            finally:
                hooks.clear_mask()
            row_i = seq - 1
            lg = outp.logits[0, -1].float().cpu()
            del outp
            vl = np.array([float(lg[t]) for t in vocab_ids])
            gi = gold_index(qtype, gold, labels)
            if gi is not None:
                m, pred = margin_of(vl, gi)
                smp_w.writerow([row["qid"], qtype, NF, k, gold, seq, f"{m:.4f}", labels[pred], int(pred == gi)])
            else:
                smp_w.writerow([row["qid"], qtype, NF, k, gold, seq, "", labels[int(np.argmax(vl))], ""])
            mask_row = mask[row_i] if mask is not None else None
            with torch.inference_mode():
                for L in args.layers:
                    w = row_attention(qk, layers, L, row_i, mask_row, dims)      # [H, T]
                    bm = block_masses(w, blocks).cpu().numpy()                    # [H, NF]
                    is_ev = np.array([t in evid for t in range(NF)])
                    ev = bm[:, is_ev].sum(1)
                    ne = bm[:, ~is_ev].sum(1)
                    for h in range(nH):
                        mass_w.writerow([row["qid"], qtype, NF, k, L, h, f"{ev[h]:.6f}", f"{ne[h]:.6f}", f"{1.0 - ev[h] - ne[h]:.6f}"])
                        if not args.no_block_csv:
                            for t in range(NF):
                                blk_w.writerow([row["qid"], qtype, NF, k, L, h, t, int(is_ev[t]), f"{bm[h, t]:.6f}"])
                        if is_ev.any() and (~is_ev).any():
                            auc_pos[(L, h)].extend(bm[h, is_ev].tolist())
                            auc_neg[(L, h)].extend(bm[h, ~is_ev].tolist())
            qk.clear()
            if need is not None:
                need[int(gold)] -= 1
            n_done += 1
            totals["samples"] += 1
            mass_f.flush(); blk_f.flush(); smp_f.flush()
            if n_done % 10 == 0:
                print(f"  {qtype}: {n_done} samples ({time.time() - t0:.0f}s)" + (f" need {need}" if need else ""), flush=True)
        totals[f"samples_{qtype}"] = n_done

    qk.remove(); hooks.remove()
    mass_f.close(); blk_f.close(); smp_f.close()
    auc = {f"L{L}_h{h}": rank_auc(np.array(auc_pos[(L, h)]), np.array(auc_neg[(L, h)])) for (L, h) in sorted(auc_pos)}
    (out / "auc.json").write_text(json.dumps(auc, indent=1))
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    with open(out / "mass.csv") as fh:
        for r in csv.DictReader(fh):
            a = agg[(r["qtype"], int(r["k"]), int(r["layer"]))]
            a[0] += float(r["evid_mass"]); a[1] += float(r["nonevid_mass"]); a[2] += float(r["other_mass"]); a[3] += 1
    lines = [f"=== ATTN PHOTO arm={args.arm} {args.config}_{args.split} qtypes={args.qtypes} cond={cond} "
             f"adapter={args.adapter or 'none'} layers={args.layers} samples={totals['samples']} "
             f"(skips: {dict((k_, v) for k_, v in totals.items() if k_.startswith(('too', 'no_', 'lay', 'gold')))}) ==="]
    for (qt, k, L) in sorted(agg):
        a = agg[(qt, k, L)]
        lines.append(f"  {qt:16s} k={k:<3d} L{L:<3d} evid {a[0]/a[3]:.4f}  nonevid {a[1]/a[3]:.4f}  other {a[2]/a[3]:.4f}  (head-mean, n_rows={a[3]})")
    best = sorted(auc.items(), key=lambda kv: -kv[1])[:8]
    lines.append("  headscan top: " + ", ".join(f"{k_} {v:.4f}" for k_, v in best))
    acc_rows = [r for r in csv.DictReader(open(out / "samples.csv")) if r["first_token_correct"] != ""]
    if acc_rows:
        by = defaultdict(list)
        for r in acc_rows:
            by[r["qtype"]].append((int(r["first_token_correct"]), float(r["margin"])))
        for qt, v in sorted(by.items()):
            lines.append(f"  first-token {qt:16s} acc {np.mean([x[0] for x in v]):.3f}  median margin {np.median([x[1] for x in v]):+.3f}  n={len(v)}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
