#!/usr/bin/env python3
"""UNIT baseline, regime D — per-unit detection by the model's OWN yes/no (plan 2026-10-06).

One dense, unfenced prompt per unit: [the unit's frames] + the bare question (+ options) + the
judge line, the benchmark's layout (frames first), no system turn by default (the judge prompt
is ours, not the benchmark's protocol). One forward; the score is the first answer token's
log P(yes) - log P(no) (logsumexp over the spelling variants), P(yes) is the summed softmax mass
on the yes spellings, and the arg-max token is kept as a sanity column (is the model answering
yes / no at all?). Model-generic: any backbone that generates; no hidden states, no probe.

Units come from the dataset spec (core.data.base.DatasetSpec.units): on video datasets every
stored unit of the question (units_native/<qid>/, experiments/prepare_video.py --stage units) —
pos = an evidence unit, hard = the grid 2 s after it, easy = a random grid >= 5 s from every
evidence interval — each read at --unit (frame | clip3_d1 | clip5_d1 | clip5_d2 | clip9_d2);
on MMReD every frame of the row (pos = evidence_frames, neg = the rest), unit = frame.
Position-free by construction: a unit is shown alone.

Outputs (run dir): config.json, units.csv (one row per unit: qid, qtype, uid, kind, k, m, score,
p_yes, top1, tokens), metrics.csv + report.txt (per task and pooled: AUROC, recall / precision at
95 % and 99 % specificity, yes-rate and mean P(yes) per kind; negatives = hard, easy and all).
Usage:
  python experiments/unit_judge.py --dataset herbench --n-frames 8 --qids-file sbatch/lib/splits/qids_herbench_native582.txt \\
      --unit clip5_d1 --res hi --frozen --output outputs/scaleup/herbench/qwen2.5-vl-7b/s1u_unit/D_clip5_d1_hi
  python experiments/unit_judge.py --dataset mmred --config seq_len_8 --per-qtype 50 --frozen --output ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, DEFAULT_BACKBONE, get_backbone  # noqa: E402
from core.backbones import base as bb  # noqa: E402
from core.data import DATASETS, DEFAULT_DATASET, get_dataset, stratified_order  # noqa: E402
from core.data.videoqa import UNIT_OFFSETS  # noqa: E402
from core.prompt import build_messages  # noqa: E402
from experiments.gate_capture import JUDGE_WORDS  # noqa: E402
from experiments.gate_fit import METRICS, _scores  # noqa: E402

JUDGE_TEXT = "Does this {what} contain evidence needed to answer the question above? Answer yes or no."
# "frame" for a one-frame unit, "clip" otherwise — the stage-2b wording (2026-09-30), unchanged.


def judge_text(m: int, template: str = JUDGE_TEXT) -> str:
    return template.format(what="frame" if m == 1 else "clip")


def judge_vocab(tokenizer: Any, backbone: str) -> Dict[str, List[int]]:
    """Token ids of the yes / no spellings (each must be ONE token for this tokenizer)."""
    yes, no = [], []
    for w in JUDGE_WORDS:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"judge word {w!r} is not one token for {backbone}: {ids}")
        (yes if w.strip().lower() == "yes" else no).append(ids[0])
    return {"yes": yes, "no": no}


@torch.inference_mode()
def score_unit(bspec: bb.BackboneSpec, model: Any, processor: Any, tokenizer: Any, frames: Sequence[Image.Image],
               text: str, vocab: Dict[str, List[int]], device: Any, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """One unit: frames then text (the benchmark's own order), one forward, first-token stats."""
    msgs = build_messages(list(frames), text, layout="paper", system_prompt=system_prompt)
    enc = bb.encode(bspec, processor, msgs, device)
    out = model(**enc, use_cache=False)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    lse_yes = torch.logsumexp(lp[vocab["yes"]], dim=0)
    lse_no = torch.logsumexp(lp[vocab["no"]], dim=0)
    top = int(lp.argmax())
    try:
        top1 = tokenizer.decode([top])
    except (TypeError, IndexError):           # an id outside the tokenizer's table (tiny test models pad the vocab)
        top1 = f"<id {top}>"
    return {"score": float(lse_yes - lse_no), "p_yes": float(lse_yes.exp()), "p_no": float(lse_no.exp()),
            "top1": top1, "tokens": int(enc["input_ids"].shape[1])}


def summarise(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per task (and pooled) x negative kind: the gate_fit operating-point metrics on the model's
    own score, plus the yes-rate (score > 0) and mean P(yes) of each kind."""
    rows = []
    tasks = sorted({r["qtype"] for r in records}) + ["all"]
    for task in tasks:
        sel = [r for r in records if task == "all" or r["qtype"] == task]
        pos = [r for r in sel if r["kind"] == "pos"]
        kinds = {"hard": [r for r in sel if r["kind"] == "hard"], "easy": [r for r in sel if r["kind"] == "easy"],
                 "neg": [r for r in sel if r["kind"] == "neg"]}
        kinds["all"] = kinds["hard"] + kinds["easy"] + kinds["neg"]
        for neg_name, neg in kinds.items():
            if not neg or (neg_name == "neg" and not neg) or not pos:
                continue
            y = np.array([1] * len(pos) + [0] * len(neg))
            s = np.array([r["score"] for r in pos] + [r["score"] for r in neg], dtype=np.float64)
            m = _scores(y, s)
            m.update({"task": task, "neg": neg_name, "n_pos": len(pos), "n_neg": len(neg),
                      "yes_rate_pos": float(np.mean([r["score"] > 0 for r in pos])),
                      "yes_rate_neg": float(np.mean([r["score"] > 0 for r in neg])),
                      "p_yes_pos": float(np.mean([r["p_yes"] for r in pos])),
                      "p_yes_neg": float(np.mean([r["p_yes"] for r in neg]))})
            rows.append(m)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    ap.add_argument("--data-root", type=Path, default=None, help="default: the dataset's data/ symlink")
    ap.add_argument("--config", default=None, help="MMReD config (seq_len_<N>)")
    ap.add_argument("--protocol", default="uniform", help="video datasets: the split whose rows are judged (frames unused)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=None)
    ap.add_argument("--qids-file", type=Path, default=None, help="one qid per line; pins the rows and their order")
    ap.add_argument("--limit", type=int, default=0, help="rows after stratified_order")
    ap.add_argument("--per-qtype", type=int, default=0, help="rows per qtype after a per-qtype stratified_order")
    ap.add_argument("--unit", choices=sorted(UNIT_OFFSETS), default="frame")
    ap.add_argument("--neg", choices=("both", "hard", "easy"), default="both", help="video datasets: which negatives to score")
    ap.add_argument("--judge", default=None, help="judge line; default: the stage-2b wording with frame / clip")
    ap.add_argument("--system-prompt", choices=("none", "dataset"), default="none",
                    help="none (default): no system turn — the judge prompt is ours; dataset: the benchmark's system prompt")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter", type=Path, default=None)
    grp.add_argument("--frozen", action="store_true")
    ap.add_argument("--res", choices=("lo", "hi"), default=None, help="the backbone's named resolution mode")
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--frame-set", default=None)
    ap.add_argument("--max-seq-tokens", type=int, default=24000, help="skip a unit prompt longer than this")
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
    if args.res and args.max_pixels:
        raise SystemExit("give --res or --max-pixels, not both")
    data_root = args.data_root or Path(spec.default_root)
    if args.qtypes:
        unknown = sorted(set(args.qtypes) - set(spec.qtypes))
        if unknown:
            raise SystemExit(f"unknown qtypes for {spec.name}: {unknown}")
    if args.config is not None:
        if spec.name != "mmred":
            raise SystemExit("--config is MMReD's config name")
        split_name = f"{args.config}_{args.split}"
    elif spec.name == "mmred":
        raise SystemExit("MMReD needs --config seq_len_<N>")
    else:
        split_name = spec.split_name(args.protocol, args.n_frames, args.split)
    if spec.name == "mmred" and args.unit != "frame":
        raise SystemExit("MMReD has one unit (a frame); --unit frame")

    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_D-{args.unit}{'_res' + args.res if args.res else ''}" \
                            f"{'_mp' + str(args.max_pixels) if args.max_pixels else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "dataset": spec.name, "backbone": bspec.name, "model_id": bspec.model_id, "split_name": split_name,
           "data_root": str(data_root), "frame_set": getattr(spec, "frame_set", None), "judge_template": args.judge or JUDGE_TEXT,
           "judge_words": list(JUDGE_WORDS), "regime": f"D-{args.unit}"}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))

    # ---- rows (the same selection rules as evaluate.py)
    samples = spec.load(data_root, split_name, args.qtypes)
    if args.qids_file is not None:
        want = [ln.strip() for ln in args.qids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        by = {s.qid: s for s in samples}
        missing = [q for q in want if q not in by]
        if missing:
            print(f"[rows] {len(missing)} qids of {args.qids_file} not in {split_name} (skipped)", flush=True)
        samples = [by[q] for q in want if q in by]
    elif args.per_qtype:
        keep_rows = []
        for qt in (args.qtypes or spec.qtypes):
            keep_rows += stratified_order([s for s in samples if s.qtype == qt], args.seed, key=spec.stratify_key)[: args.per_qtype]
        samples = keep_rows
    else:
        samples = stratified_order(samples, args.seed, key=spec.stratify_key)
        if args.limit:
            samples = samples[: args.limit]
    print(f"[rows] {len(samples)} from {spec.name}/{split_name} unit={args.unit} neg={args.neg} backbone={bspec.name}", flush=True)

    # ---- model
    rt = bb.load_runtime(bspec)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    bb.set_max_pixels(bspec, processor, args.max_pixels)
    res_kwargs = bb.set_resolution(bspec, processor, args.res)
    if res_kwargs:
        cfg["res_kwargs"] = res_kwargs
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
    vocab = judge_vocab(tok, bspec.name)
    system_prompt = spec.system_prompt if args.system_prompt == "dataset" else None

    # ---- loop
    records: List[Dict[str, Any]] = []
    n_units = n_too_long = n_no_units = n_rows = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        units = spec.units(data_root, s, args.unit)
        if args.neg != "both":
            units = [u for u in units if u["kind"] in ("pos", args.neg)]
        if not units:
            n_no_units += 1
            continue
        n_rows += 1
        q = spec.bare_question(s)
        for u in units:
            frames = [Image.open(p).convert("RGB") for p in u["frame_paths"]]
            text = q + "\n\n" + (args.judge or judge_text(len(frames)))
            try:
                r = score_unit(bspec, model, processor, tok, frames, text, vocab, rt.device, system_prompt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_too_long += 1
                print(f"  [skip] {s.qid}/{u['uid']}: OOM", flush=True)
                continue
            if r["tokens"] > args.max_seq_tokens:
                n_too_long += 1
                continue
            n_units += 1
            records.append({"qid": s.qid, "qtype": s.qtype, "group": s.group, "uid": u["uid"], "unit": u["unit"], "kind": u["kind"],
                            "k": u["k"], "m": len(frames), "centre": u["centre"], "gold": s.answer, **r})
        if (i + 1) % 25 == 0:
            yes = sum(1 for r in records if r["kind"] == "pos" and r["score"] > 0)
            npos = sum(1 for r in records if r["kind"] == "pos")
            print(f"  {i + 1}/{len(samples)} rows, {n_units} units, yes on {yes}/{npos} evidence units, "
                  f"{records[-1]['tokens']} tok ({time.time() - t0:.0f}s)", flush=True)

    # ---- outputs
    with open(run_dir / "units.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["qid"])
        w.writeheader()
        w.writerows(records)
    metrics = summarise(records)
    cols = ["task", "neg", "n_pos", "n_neg", "yes_rate_pos", "yes_rate_neg", "p_yes_pos", "p_yes_neg"] + list(METRICS)
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics)
    lines = [f"UNIT baseline D: dataset={spec.name} split={split_name} unit={args.unit} res={args.res or 'default'} "
             f"backbone={bspec.name} rows={n_rows} (no units {n_no_units}) units={n_units} too_long={n_too_long} "
             f"judge={(args.judge or JUDGE_TEXT)!r} system_prompt={args.system_prompt}"]
    for m in metrics:
        if math.isnan(m.get("auc", float("nan"))):
            continue
        lines.append(f"  {m['task']:8s} vs {m['neg']:4s} n={m['n_pos']}+{m['n_neg']:<4d} AUC {m['auc']:.3f} "
                     f"rec@spec95 {m['recall@spec95']:.3f} prec@spec95 {m['precision@spec95']:.3f} "
                     f"rec@spec99 {m['recall@spec99']:.3f} | yes-rate pos {m['yes_rate_pos']:.2f} neg {m['yes_rate_neg']:.2f} "
                     f"P(yes) pos {m['p_yes_pos']:.2f} neg {m['p_yes_neg']:.2f}")
    report = "\n".join(lines)
    (run_dir / "report.txt").write_text(report + "\n")
    print(report)
    print("wrote", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
