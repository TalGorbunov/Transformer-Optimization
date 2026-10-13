#!/usr/bin/env python3
"""The attention photograph — DIAG cells D2 / D3 / D8 on official MMReD rows
(docs/DIAGNOSTICS_2026-09-22.md §2) and, since 2026-10-11, the share measurement on the video
datasets (VNBench planted rows; plan `docs/VNBENCH_2026-10-09.md`, "what next" item 2): at the
answer row, how much attention mass lands on the evidence frames, the non-evidence frames, and
everything else (prompt + sink), per head x layer, as N (and k) vary; plus the per-head
evidence-vs-non-evidence AUC (the "headscan" that found L24h20 on park data).

sdpa exposes no weights, so each row is recomputed from q_proj / k_proj captures exactly as the
model computes it (experiments/_diag_common.row_attention: rotary q.k * module.scaling + the
injected mask row -> softmax; per-row sum == 1 asserted; Qwen2.5-VL's M-RoPE, so Qwen only). The
PEFT adapter, if any, is loaded BEFORE the q/k hooks so trained projections are photographed.
The answer row = the position whose next token is the answer's first token: on MMReD the last
token of the teacher-forced prefix '{ "answer": "' (probe_hahn's `final` locus, the paper's system turn:
the D2/D3 protocol); on the MCQ video datasets the prompt's last token, where the letter comes
(their official prompt, core.data). Rows: --config (MMReD) or --protocol/--n-frames (video).

Arms: plain (paper layout, images then question, plain causal), qfirst (question-first, plain
causal), fenced (question-first + fence + posreset), gated (fenced + oracle gate on the evidence
frames). Knobs --attn-sharpen / --sharpen-from-layer / --attn-logn-sref act through module.scaling,
so photograph and forward see the same temperature by construction. Evidence labels come from the
dataset (MMReD: core.mmred.evidence_frames; video: the protocol's labels), never from the model;
rows without labels are skipped. --k-list fills --limit rows per stratum: the gold count on MMReD
count qtypes, the needle count k on video rows.

Twin: legacy/v1/scripts/sparse/probe_attn_photo.py (+ selfgate/probe_headscan.py). Legacy
layer numbers there were decoder module indices too (--layers 12,16,20,24,27).
Outputs (run dir): mass.csv, block_mass.csv, samples.csv (first-token margin/accuracy), auc.json,
report.txt, config.json.
Usage:
  python experiments/probe_attention.py --config seq_len_32 --qtypes steps_in_room --arm qfirst \\
      --k-list 2 4 8 --limit 30 --output outputs/diag/photo/steps_in_room/qfirst/N32
  python experiments/probe_attention.py --dataset vnbench --protocol planted --n-frames 64 --arm plain \\
      --max-pixels 151200 --limit 20 --output outputs/scaleup/vnbench/qwen2.5-vl-7b/s5_photo/planted_N64/plain
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, DEFAULT_BACKBONE, get_backbone  # noqa: E402
from core.backbones import base as bb  # noqa: E402
from core.data import DATASETS, get_dataset, stratified_order  # noqa: E402
from core.data.base import DatasetSpec  # noqa: E402
from core.fence import FENCED_SDPA, FenceHooks, build_block_mask, hide_cols_for, reset_positions  # noqa: E402
from core.mmred import NUMERIC_QTYPES, evidence_frames, frames, load_split, recompute_answer, states  # noqa: E402
from core.mmred import stratified_order as mmred_stratified_order  # noqa: E402
from core.model import get_layers, get_rope_index_fn  # noqa: E402
from core.prompt import build_messages  # noqa: E402
from experiments._diag_common import (ANSWER_PREFIX, QKCapture, answer_vocab, attention_dims, block_masses, cond_tag,  # noqa: E402
                                      gold_index, margin_of, rank_auc, row_attention, set_logn, set_sharpen)
from experiments.evaluate import letter_readout, option_letters  # noqa: E402

ARMS = ("plain", "qfirst", "fenced", "gated")
MMRED_DEFAULT_QTYPES = ("steps_in_room",)   # the D2/D3 default cell


@dataclass(frozen=True)
class PhotoItem:
    """One row to photograph: its frames and prompt, the evidence labels, and the first-token answer vocabulary
    (labels with their token ids; gold_index None = no margin row, e.g. MMReD counts >= 10)."""
    qid: str
    qtype: str
    frames: List[Any]
    question: str
    evidence: FrozenSet[int]
    gold: str
    labels: List[str]
    vocab_ids: List[int]
    gold_index: Optional[int]
    system_prompt: Optional[str]
    prefix_ids: List[int]

    @property
    def k(self) -> int:
        return len(self.evidence)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rows = ap.add_argument_group("rows")
    rows.add_argument("--dataset", default="mmred", choices=sorted(DATASETS))
    rows.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    rows.add_argument("--data-root", type=Path, default=None, help="default: the dataset's data/ symlink")
    rows.add_argument("--config", default=None, help="MMReD config (seq_len_<N>); video takes --protocol/--n-frames")
    rows.add_argument("--protocol", default=None, help="video: frame protocol (planted | uniform | isolated)")
    rows.add_argument("--n-frames", type=int, default=None, help="video: N")
    rows.add_argument("--split", default="test")
    rows.add_argument("--qtypes", nargs="+", default=None, help="default: steps_in_room on MMReD, every task on video")
    rows.add_argument("--k-list", nargs="*", type=int, default=None,
                      help="strata filled with --limit rows each (MMReD count qtypes: the gold; video: needle count)")
    rows.add_argument("--limit", type=int, default=30, help="rows per stratum (or per qtype when no --k-list)")
    rows.add_argument("--seed", type=int, default=0)
    photo = ap.add_argument_group("photograph")
    photo.add_argument("--arm", choices=ARMS, required=True)
    photo.add_argument("--layers", nargs="+", type=int, default=[12, 16, 20, 24, 27],
                       help="decoder module indices (0-based)")
    photo.add_argument("--adapter", type=Path, default=None)
    photo.add_argument("--attn-sharpen", type=float, default=0.0)
    photo.add_argument("--sharpen-from-layer", type=int, default=12)
    photo.add_argument("--attn-logn-sref", type=int, default=0)
    photo.add_argument("--max-pixels", type=int, default=0, help="processor's pixel bound per frame (0 = its default)")
    photo.add_argument("--max-seq-tokens", type=int, default=60000)
    photo.add_argument("--no-block-csv", action="store_true", help="skip block_mass.csv (large at N=128)")
    photo.add_argument("--model", default=None, help="HF id / path overriding the backbone spec's model_id")
    photo.add_argument("--output", type=Path, required=True)
    return ap


def mmred_items(args: argparse.Namespace, spec: DatasetSpec, data_root: Path, tokenizer: Any,
                totals: Dict[str, int]) -> Iterator[PhotoItem]:
    """The D2/D3 rows: per qtype, the class-interleaved order, gold recomputed from the states (mismatches
    skipped), --k-list = gold strata on count qtypes, the paper's system turn, the teacher-forced JSON prefix."""
    prefix_ids = list(tokenizer(ANSWER_PREFIX, add_special_tokens=False).input_ids)
    for qtype in args.qtypes:
        rows = mmred_stratified_order(load_split(args.config, args.split, data_root, [qtype]), args.seed)
        labels, vocab_ids = answer_vocab(qtype, tokenizer)
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
            if need is not None and need.get(int(gold), 0) <= 0:
                continue
            evid = evidence_frames(qtype, question, st)
            if evid is None:
                totals["no_evidence_rule"] += 1
                continue
            yield PhotoItem(row["qid"], qtype, frames(row, data_root), question, frozenset(evid), gold, labels,
                            vocab_ids, gold_index(qtype, gold, labels), spec.system_prompt, prefix_ids)
            if need is not None:
                need[int(gold)] -= 1
            n_done += 1
        totals[f"samples_{qtype}"] = n_done


def video_items(args: argparse.Namespace, spec: DatasetSpec, data_root: Path, tokenizer: Any,
                totals: Dict[str, int]) -> Iterator[PhotoItem]:
    """The video rows: per qtype, the spec's stratified order, the protocol's evidence labels (rows without
    them skipped), --k-list = needle-count strata; the official MCQ prompt, the letters as answer vocabulary."""
    split_name = spec.split_name(args.protocol or spec.protocols[0], args.n_frames, args.split)
    samples = spec.load(data_root, split_name, args.qtypes)
    ids_by_letter = letter_readout(tokenizer).ids_by_letter
    for qtype in args.qtypes:
        rows = stratified_order([s for s in samples if s.qtype == qtype], args.seed, key=spec.stratify_key)
        need: Optional[Dict[int, int]] = {k: args.limit for k in args.k_list} if args.k_list else None
        n_done = 0
        for sample in rows:
            if need is None and n_done >= args.limit:
                break
            if need is not None and all(v <= 0 for v in need.values()):
                break
            if sample.evidence is None:
                totals["no_evidence_label"] += 1
                continue
            if need is not None and need.get(len(sample.evidence), 0) <= 0:
                continue
            letters = option_letters(sample)
            vocab_ids = [ids_by_letter[letter][0] for letter in letters]       # one id per label: the bare letter
            gold_at = letters.index(sample.answer) if sample.answer in letters else None
            yield PhotoItem(sample.qid, qtype, spec.frames(sample), spec.question_text(sample), sample.evidence,
                            sample.answer, list(letters), vocab_ids, gold_at, spec.system_prompt, [])
            if need is not None:
                need[len(sample.evidence)] -= 1
            n_done += 1
        totals[f"samples_{qtype}"] = n_done


def encode_item(item: PhotoItem, bspec: Any, processor: Any, layout: str, device: Any) -> Dict[str, Any]:
    """The item's prompt through the backbone's template (core.prompt.build_messages + core.backbones.encode), plus
    the teacher-forced prefix when the item has one; the last position's next token is the answer's first."""
    messages = build_messages(item.frames, item.question, layout=layout, system_prompt=item.system_prompt)
    enc = bb.encode(bspec, processor, messages, device)
    if item.prefix_ids:
        prefix = torch.tensor([list(item.prefix_ids)], dtype=enc["input_ids"].dtype, device=enc["input_ids"].device)
        enc["input_ids"] = torch.cat([enc["input_ids"], prefix], dim=1)
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return enc


def main() -> int:
    args = build_parser().parse_args()
    spec = get_dataset(args.dataset)
    bspec = get_backbone(args.backbone)
    if args.model:
        bspec = replace(bspec, model_id=str(args.model))
    if bspec.family != "qwen2.5-vl":
        raise SystemExit("the photograph recomputes Qwen2.5-VL's M-RoPE attention rows; "
                         "other backbones need their own row_attention")
    if spec.name == "mmred" and args.config is None:
        raise SystemExit("MMReD rows need --config seq_len_<N>")
    if spec.name != "mmred" and args.n_frames is None:
        raise SystemExit(f"{spec.name} rows need --n-frames [--protocol]")
    args.qtypes = args.qtypes or (list(MMRED_DEFAULT_QTYPES) if spec.name == "mmred" else list(spec.qtypes))
    unknown = sorted(set(args.qtypes) - set(spec.qtypes))
    if unknown:
        raise SystemExit(f"unknown qtypes for {spec.name}: {unknown}")
    data_root = args.data_root or Path(spec.default_root)
    rows_name = f"{args.config}_{args.split}" if spec.name == "mmred" else \
        spec.split_name(args.protocol or spec.protocols[0], args.n_frames, args.split)
    out = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    out.mkdir(parents=True, exist_ok=True)
    cond = cond_tag(args.attn_sharpen, args.sharpen_from_layer, args.attn_logn_sref)
    (out / "config.json").write_text(json.dumps({**vars(args), "cond": cond, "rows": rows_name, "backbone": bspec.name,
                                                 "model_id": bspec.model_id}, indent=1, default=str))
    layout = "paper" if args.arm == "plain" else "question-first"
    fenced = args.arm in ("fenced", "gated")

    rt = bb.load_runtime(bspec)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    sid = bb.special_ids(bspec, tok)
    bb.set_max_pixels(bspec, processor, args.max_pixels)
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
    totals: Dict[str, int] = defaultdict(int)
    t0 = time.time()

    if spec.name == "mmred":
        items = mmred_items(args, spec, data_root, tok, totals)
    else:
        items = video_items(args, spec, data_root, tok, totals)
    for item in items:
        NF = len(item.frames)
        k = item.k
        enc = encode_item(item, bspec, processor, layout, rt.device)
        ids = enc["input_ids"][0].cpu()
        seq = int(ids.shape[0])
        if seq > args.max_seq_tokens:
            totals["too_long"] += 1
            continue
        try:
            blocks, fin = bb.blocks(bspec, ids, sid, layout)
        except ValueError:
            totals["layout_skip"] += 1
            continue
        if len(blocks) != NF:
            totals["layout_skip"] += 1
            continue
        mask = pos = None
        if fenced:
            keep = [t in item.evidence for t in range(NF)] if args.arm == "gated" else None
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
            with torch.inference_mode(), (sdpa_kernel(FENCED_SDPA) if mask is not None else contextlib.nullcontext()):
                outp = model(**cur, position_ids=pos.to(rt.device) if pos is not None else None, use_cache=False)
        finally:
            hooks.clear_mask()
        row_i = seq - 1
        lg = outp.logits[0, -1].float().cpu()
        del outp
        vl = np.array([float(lg[t]) for t in item.vocab_ids])
        if item.gold_index is not None:
            m, pred = margin_of(vl, item.gold_index)
            smp_w.writerow([item.qid, item.qtype, NF, k, item.gold, seq, f"{m:.4f}", item.labels[pred],
                            int(pred == item.gold_index)])
        else:
            smp_w.writerow([item.qid, item.qtype, NF, k, item.gold, seq, "", item.labels[int(np.argmax(vl))], ""])
        mask_row = mask[row_i] if mask is not None else None
        with torch.inference_mode():
            for L in args.layers:
                w = row_attention(qk, layers, L, row_i, mask_row, dims)      # [H, T]
                bm = block_masses(w, blocks).cpu().numpy()                    # [H, NF]
                is_ev = np.array([t in item.evidence for t in range(NF)])
                ev = bm[:, is_ev].sum(1)
                ne = bm[:, ~is_ev].sum(1)
                for h in range(nH):
                    mass_w.writerow([item.qid, item.qtype, NF, k, L, h, f"{ev[h]:.6f}", f"{ne[h]:.6f}",
                                     f"{1.0 - ev[h] - ne[h]:.6f}"])
                    if not args.no_block_csv:
                        for t in range(NF):
                            blk_w.writerow([item.qid, item.qtype, NF, k, L, h, t, int(is_ev[t]), f"{bm[h, t]:.6f}"])
                    if is_ev.any() and (~is_ev).any():
                        auc_pos[(L, h)].extend(bm[h, is_ev].tolist())
                        auc_neg[(L, h)].extend(bm[h, ~is_ev].tolist())
        qk.clear()
        totals["samples"] += 1
        mass_f.flush()
        blk_f.flush()
        smp_f.flush()
        if totals["samples"] % 10 == 0:
            print(f"  {item.qtype}: {totals['samples']} samples ({time.time() - t0:.0f}s)", flush=True)

    qk.remove()
    hooks.remove()
    mass_f.close()
    blk_f.close()
    smp_f.close()
    auc = {f"L{L}_h{h}": rank_auc(np.array(auc_pos[(L, h)]), np.array(auc_neg[(L, h)])) for (L, h) in sorted(auc_pos)}
    (out / "auc.json").write_text(json.dumps(auc, indent=1))
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    with open(out / "mass.csv") as fh:
        for r in csv.DictReader(fh):
            a = agg[(r["qtype"], int(r["k"]), int(r["layer"]))]
            a[0] += float(r["evid_mass"]); a[1] += float(r["nonevid_mass"]); a[2] += float(r["other_mass"]); a[3] += 1
    lines = [f"=== ATTN PHOTO arm={args.arm} {spec.name}/{rows_name} qtypes={args.qtypes} cond={cond} "
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
