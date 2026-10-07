#!/usr/bin/env python3
"""D4 / SCALEUP S2: capture each frame's slot state (the last token of its block: Qwen's
<|vision_end|>) under one (layout x fence) arm so a CPU fit (experiments/gate_fit.py) can say
whether "this block holds an evidence frame" is linearly readable at the slot, per task and per
N — the per-block evidence classification that runs BEFORE the method on every dataset and
backbone (docs/SCALEUP_2026-09-23.md §5 S2; DIAG plan §2 D4 on MMReD).

Arms (core.prompt.ARMS, one forward each): plain = paper layout (images then question), no
fence; qfirst = question-first, no fence; fenced_qlast = paper layout + fence + posreset
(question-BLIND blocks); fenced_qfirst = question-first + fence + posreset (THE method's block:
the question is in the shared prefix). The gated arm is not a capture arm.
Labels = the Sample's evidence set (MMReD: core.mmred.evidence_frames; video: the planted /
uniform protocol labels, core.data.videoqa); rows whose evidence is unknown are skipped.
Layer numbers follow the legacy convention (hidden_states[L] = output of decoder module L-1);
the default is the backbone spec's probe_layers.
Output (run dir): capture.npz with X[n_frames_total, n_layers, hidden] float16, plus per-frame
arrays qid, group (held-out key: qid on MMReD, video_id on video), qtype, N, frame, is_evid,
gold; meta.json. One arm per invocation. On the batched path two more readouts of the same
blocks: --pool stores P (mean of the block's image-token states, same layers) and --judge TEXT
stores J (the first-token logits of a per-block yes/no judge: prefix + block + TEXT + assistant
opener, JUDGE_WORDS order) — the readout comparison of SCALEUP stage 2b (slot vs pooled vs judge).
Usage:
  python experiments/gate_capture.py --config seq_len_8 --split train --arm fenced_qfirst --limit 20 --output outputs/diag/gate/N8/fenced_qfirst
  python experiments/gate_capture.py --dataset herbench --protocol planted --n-frames 16 --arm fenced_qfirst --limit 50 --output ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, DEFAULT_BACKBONE, get_backbone  # noqa: E402
from core.backbones import base as bb  # noqa: E402
from core.data import DATASETS, DEFAULT_DATASET, get_dataset, stratified_order  # noqa: E402
from core.fastpath import fenced_forward  # noqa: E402
from core.fence import FENCED_SDPA, FenceHooks  # noqa: E402
from core.model import get_layers  # noqa: E402
from core.prompt import ARMS, build_messages  # noqa: E402

CAPTURE_ARMS = tuple(a for a in ARMS if ARMS[a].gate == "none")
JUDGE_WORDS = ("Yes", "No", "yes", "no", " Yes", " No")     # first answer token of the per-block judge; stored in this order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    ap.add_argument("--data-root", type=Path, default=None, help="default: the dataset's data/ symlink")
    ap.add_argument("--config", default=None, help="MMReD config (seq_len_<N>); or use --n-frames")
    ap.add_argument("--protocol", default=None, help="frame protocol for --n-frames (default: the dataset's first)")
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=None, help="default: all of the dataset's task types")
    ap.add_argument("--arm", choices=CAPTURE_ARMS, required=True)
    ap.add_argument("--limit", type=int, default=50, help="rows per qtype (class-interleaved order)")
    ap.add_argument("--hs-layers", nargs="+", type=int, default=None, help="legacy convention (1-based tuple index); default: the backbone's probe_layers")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-seq-tokens", type=int, default=60000)
    ap.add_argument("--fast", action="store_true", help="fenced arms through the batched path (core.fastpath)")
    ap.add_argument("--chunk-tokens", type=int, default=16000)
    ap.add_argument("--pool", action="store_true", help="--fast: also store the mean of each block's image-token states (P, same layers)")
    ap.add_argument("--judge", default=None, help="--fast: per-block yes/no judge instruction appended after the frame; stores J [frames, |JUDGE_VOCAB|]")
    ap.add_argument("--frame-set", default=None, help="video datasets: stored frames to read (native | 512)")
    ap.add_argument("--max-pixels", type=int, default=0, help="processor's own bound on pixels per frame (0 = its default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    spec = get_dataset(args.dataset)
    if args.frame_set is not None:
        if not hasattr(spec, "frame_set"):
            raise SystemExit(f"{spec.name} has one frame set; --frame-set does not apply")
        spec.frame_set = args.frame_set
    bspec = get_backbone(args.backbone)
    if args.model:
        bspec = replace(bspec, model_id=str(args.model))
    data_root = args.data_root or Path(spec.default_root)
    if args.qtypes:
        unknown = sorted(set(args.qtypes) - set(spec.qtypes))
        if unknown:
            raise SystemExit(f"unknown qtypes for {spec.name}: {unknown}")
    if args.config is not None:
        if spec.name != "mmred" or args.n_frames is not None:
            raise SystemExit("--config is MMReD-only and excludes --n-frames")
        split_name = f"{args.config}_{args.split}"
    elif args.n_frames is not None:
        split_name = spec.split_name(args.protocol or spec.protocols[0], args.n_frames, args.split)
    else:
        raise SystemExit("one of --config (MMReD) or --n-frames [--protocol] is required")
    arm = ARMS[args.arm]
    layout, fenced = arm.layout, arm.fence
    hs_layers = args.hs_layers or list(bspec.probe_layers)
    if not hs_layers:
        raise SystemExit(f"{bspec.name} has no probe_layers; pass --hs-layers")
    out = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "split_name": split_name, "hs_layers": hs_layers,
                                                  "model_id": bspec.model_id}, indent=1, default=str))

    rt = bb.load_runtime(bspec)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    base_model = model                                    # positions from the unwrapped model
    layers = get_layers(model)
    sid = bb.special_ids(bspec, tok)
    bb.set_max_pixels(bspec, processor, args.max_pixels)
    if args.fast and not ARMS[args.arm].fence:
        raise SystemExit("--fast is the batched FENCED path; choose a fenced arm")
    if (args.pool or args.judge) and not args.fast:
        raise SystemExit("--pool / --judge run on the batched path (--fast)")
    judge_ids = judge_vocab = None
    if args.judge:
        judge_ids = tok.encode(args.judge, add_special_tokens=False)
        judge_vocab = []
        for w in JUDGE_WORDS:
            ids_w = tok.encode(w, add_special_tokens=False)
            if len(ids_w) != 1:
                raise SystemExit(f"judge word {w!r} is not one token for {bspec.name}: {ids_w}")
            judge_vocab.append(ids_w[0])
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
    capture_mods = [L - 1 for L in hs_layers]
    hooks = FenceHooks(layers, capture_layers=capture_mods)
    if not args.fast:
        hooks.install()

    X, qids, groups, qts, Ns, fidx, lab, golds = [], [], [], [], [], [], [], []
    Pm, J = [], []                                          # pooled states / judge logits (--pool / --judge)
    totals = defaultdict(int)
    t0 = time.time()
    all_samples = spec.load(data_root, split_name, args.qtypes)
    if getattr(spec, "last_load", None):
        totals["load_skipped"] = sum(spec.last_load.get("skipped", {}).values())
    for qtype in (args.qtypes or spec.qtypes):
        rows = stratified_order([s for s in all_samples if s.qtype == qtype], args.seed, key=spec.stratify_key)[: args.limit]
        n_done = 0
        for s in rows:
            if not spec.verify(s):
                totals["gold_mismatch"] += 1
                continue
            if s.evidence is None:
                totals["no_evidence_label"] += 1
                continue
            if args.fast:
                try:
                    r = fenced_forward(bspec, model, base_model, processor, list(s.frame_paths), spec.question_text(s),
                                       layout=layout, system_prompt=spec.system_prompt, capture_layers=capture_mods,
                                       pool_layers=capture_mods if args.pool else (), judge_ids=judge_ids,
                                       judge_vocab=judge_vocab or (), decode=False, chunk_tokens=args.chunk_tokens)
                except ValueError as exc:
                    totals["layout_skip"] += 1
                    print(f"  [skip] {s.qid}: {exc}", flush=True)
                    continue
                NF = s.n_frames
                hs_fast = [r.slot_states[m].numpy().astype(np.float16) for m in capture_mods]      # each [NF, hidden]
                if args.pool:
                    ps_fast = [r.pool_states[m].numpy().astype(np.float16) for m in capture_mods]
                    Pm.extend(np.stack([p[t] for p in ps_fast]) for t in range(NF))
                if args.judge:
                    J.extend(r.judge_logits.numpy().astype(np.float32))
                for t in range(NF):
                    X.append(np.stack([h[t] for h in hs_fast]))
                    qids.append(s.qid); groups.append(s.group); qts.append(qtype); Ns.append(NF); fidx.append(t)
                    lab.append(int(t in s.evidence)); golds.append(s.answer)
                totals["tokens_per_frame"] = r.tokens_block
                n_done += 1
                totals["rows"] += 1
                if totals["rows"] % 25 == 0:
                    print(f"  {totals['rows']} rows, {len(X)} frames, {r.tokens_block} tok/frame ({time.time() - t0:.0f}s)", flush=True)
                continue
            fr = spec.frames(s)
            NF = len(fr)
            msgs = build_messages(fr, spec.question_text(s), layout=layout, system_prompt=spec.system_prompt)
            enc = bb.encode(bspec, processor, msgs, rt.device)
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
            slots = bb.slots(blocks, bspec)
            mask = pos = None
            cur = dict(enc)
            if fenced:
                mask, pos = bb.fenced_setup(bspec, base_model, enc, blocks, fin, None)
                cur.pop("attention_mask", None)
            hooks.hidden.clear()
            if mask is not None:
                hooks.set_mask(mask, rt.device)
            try:
                with torch.inference_mode(), (sdpa_kernel(FENCED_SDPA) if mask is not None else contextlib.nullcontext()):
                    model(**cur, position_ids=pos.to(rt.device) if pos is not None else None, use_cache=False)
            finally:
                hooks.clear_mask()
            hs = [hooks.hidden[m][0] for m in capture_mods]              # each [seq, hidden]
            for t in range(NF):
                X.append(np.stack([h[slots[t]].float().cpu().numpy().astype(np.float16) for h in hs]))
                qids.append(s.qid); groups.append(s.group); qts.append(qtype); Ns.append(NF); fidx.append(t)
                lab.append(int(t in s.evidence)); golds.append(s.answer)
            n_done += 1
            totals["rows"] += 1
            if totals["rows"] % 25 == 0:
                print(f"  {totals['rows']} rows, {len(X)} frames ({time.time() - t0:.0f}s)", flush=True)
        totals[f"rows_{qtype}"] = n_done
    hooks.remove()
    extra = {}
    if args.pool:
        extra["P"] = np.stack(Pm) if Pm else np.zeros((0, len(capture_mods), 1), np.float16)
    if args.judge:
        extra["J"] = np.stack(J) if J else np.zeros((0, len(judge_vocab)), np.float32)
        extra["judge_vocab"] = np.array(judge_vocab)
    np.savez_compressed(out / "capture.npz", X=np.stack(X) if X else np.zeros((0, len(capture_mods), 1), np.float16),
                        qid=np.array(qids), group=np.array(groups), qtype=np.array(qts), N=np.array(Ns),
                        frame=np.array(fidx), is_evid=np.array(lab), gold=np.array(golds), layers=np.array(hs_layers), **extra)
    meta = {"arm": args.arm, "layout": layout, "fenced": fenced, "dataset": spec.name, "backbone": bspec.name,
            "split_name": split_name, "config": args.config, "protocol": args.protocol, "n_frames": args.n_frames,
            "split": args.split, "hs_layers": hs_layers, "frame_set": getattr(spec, "frame_set", None),
            "max_pixels": args.max_pixels, "path": "fast" if args.fast else "dense", "frames": len(X), "pos_rate": float(np.mean(lab)) if lab else None,
            "pool": bool(args.pool), "judge": args.judge, "judge_words": list(JUDGE_WORDS) if args.judge else None,
            **totals}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
