#!/usr/bin/env python3
"""Evaluate the frozen model or a LoRA adapter on one split of one dataset: the dataset's
official prompt and parser, native-resolution frames, exact match / MCQ accuracy per task.

Dataset and backbone are registry names (core.data.DATASETS, core.backbones.BACKBONES);
the defaults reproduce the pre-seam behaviour: official MMReD on Qwen2.5-VL-7B.

Faithful protocol on MMReD (every run without --port-check): core.prompt.SYSTEM_PROMPT as the
system turn; the user turn = the frames and the bare question (layout paper = frames then
question, question-first = their --prefix_question order = THE method's layout, replica = the
S1-S11 layout); greedy decode; core.prompt.parse_answer / exact_match; per-qtype EM with
bootstrap CIs; numeric qtypes also split by answer <= 16 / > 16 (the spec's report_extras).

--fence puts every frame in its own attention block with a per-block position reset
(core.fence via core.backbones); --gate oracle additionally hides every non-evidence block from
the tail (the Sample's evidence labels; NEVER used for a reported model-gate row). --arm names a
(layout, fence, gate) triple from core.prompt.ARMS instead of the three flags.

UNIT baseline (plan 2026-10-06; unfenced, the model's own answer): --no-frames = regime T (the
question alone); --unit <frame|clip3_d1|clip5_d1|clip5_d2|clip9_d2> = regime E (ONLY the evidence
units, no fillers, the official layout; video datasets read units_native/, MMReD keeps the
evidence frames of the config split; accuracy also by k); --res lo|hi = the backbone's own named
resolution mode (spec.res_modes). Run dirs carry _T / _E-<unit> and _res<mode>.

Anchors (ONE-TIME port checks, experiments/_port_check.py; MMReD + Qwen only; never a reported row):
  --port-check arm-a  --frozen  seq_len_8 test, all 24 qtypes  -> 0.533 (640/1200), grid_seq8_test
                      --qtypes final_app steps_in_room where_spend --limit-per-qtype 34
                                                                 -> 0.765 / 0.559 / 0.353 (127776)
  --port-check p2     --adapter checkpoints/sft_fenced_hf_adapter --qids-file sbatch/lib/splits/qids_p2_seq8_test.txt
                      (and seq16 / seq32)                        -> 45/50, 33/50, 15/50
Refactor proof (docs/SCALEUP_2026-09-23.md §8): faithful seq_len_8 test on the defaults = 0.515.

Outputs (run dir): config.json, eval.csv (one row per sample), summary.csv, report.txt.
Usage:
  python experiments/evaluate.py --config seq_len_8 --split test --frozen --limit 20 --output outputs/_scratch/eval
  python experiments/evaluate.py --dataset mmred --n-frames 8 --split test --frozen --arm fenced_qfirst ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, DEFAULT_BACKBONE, get_backbone  # noqa: E402
from core.backbones import base as bb  # noqa: E402
from core.data import DATASETS, DEFAULT_DATASET, get_dataset, stratified_order  # noqa: E402
from core.data.videoqa import UNIT_OFFSETS  # noqa: E402
from core.fastpath import fenced_forward  # noqa: E402
from core.fence import FenceHooks, greedy_decode  # noqa: E402
from core.metrics import bootstrap_ci  # noqa: E402
from core.model import get_layers  # noqa: E402
from core.prompt import ARMS, LAYOUTS, build_messages  # noqa: E402
from experiments._diag_common import cond_tag, set_logn, set_sharpen  # noqa: E402
from experiments._port_check import (  # noqa: E402
    PORT_CHECKS, first_integer, legacy_norm, legacy_p2_blocks, legacy_p2_messages, legacy_regex_ladder, set_max_pixels,
)


def _stop_newline(text: str) -> bool:
    return "\n" in text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    ap.add_argument("--data-root", type=Path, default=None, help="default: the dataset's data/ symlink")
    ap.add_argument("--config", default=None, help="MMReD config (seq_len_<N>); or use --n-frames")
    ap.add_argument("--protocol", default=None, help="frame protocol for --n-frames (default: the dataset's first)")
    ap.add_argument("--n-frames", type=int, default=None, help="N; with --protocol names the split via the dataset spec")
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=None, help="default: all of the dataset's task types")
    ap.add_argument("--qids-file", type=Path, default=None, help="one qid per line; pins the rows and their order")
    ap.add_argument("--limit", type=int, default=0, help="rows after stratified_order (class-interleaved)")
    ap.add_argument("--limit-per-qtype", type=int, default=0, help="rows per qtype in JSON order (legacy semantics)")
    ap.add_argument("--per-qtype", type=int, default=0, help="rows per qtype after a per-qtype stratified_order (the S1-S3 row budget)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter", type=Path, default=None)
    grp.add_argument("--frozen", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS), default=None, help="(layout, fence, gate) triple; replaces the three flags")
    ap.add_argument("--layout", choices=LAYOUTS, default=None, help="default: adapter's train_config.json, else paper")
    ap.add_argument("--fence", action="store_true", help="default: adapter's train_config.json")
    ap.add_argument("--gate", choices=["none", "oracle"], default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-seq-tokens", type=int, default=24000, help="refuse the dense fenced path above this")
    ap.add_argument("--fast", action="store_true", help="fenced arms through the batched path (core.fastpath): no mask, "
                    "per-frame encoding, the read sees prefix + kept blocks only")
    ap.add_argument("--chunk-tokens", type=int, default=16000, help="--fast: tokens per batch of blocks")
    ap.add_argument("--frame-set", default=None, help="video datasets: stored frames to read (native | 512); default = the spec's")
    ap.add_argument("--max-pixels", type=int, default=0, help="processor's own bound on pixels per frame (0 = its default)")
    ap.add_argument("--res", choices=("lo", "hi"), default=None, help="UNIT baseline: the backbone's named resolution mode (replaces --max-pixels)")
    ap.add_argument("--unit", choices=sorted(UNIT_OFFSETS), default=None,
                    help="UNIT baseline, regime E: only the evidence units, read at this unit (video: names the split; MMReD: 'frame' on --config)")
    ap.add_argument("--no-frames", action="store_true", help="UNIT baseline, regime T: the question alone, no image")
    ap.add_argument("--attn-sharpen", type=float, default=0.0, help="DIAG: tau on decoder modules >= --sharpen-from-layer (0 = off)")
    ap.add_argument("--sharpen-from-layer", type=int, default=12)
    ap.add_argument("--attn-logn-sref", type=int, default=0, help="DIAG: log-N logit scaling reference length in tokens (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="HF id / path overriding the backbone spec's model_id")
    ap.add_argument("--port-check", choices=sorted(PORT_CHECKS), default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # ---- dataset + backbone
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
            raise SystemExit(f"unknown qtypes for {spec.name}: {unknown}; known: {list(spec.qtypes)}")
    if args.no_frames and (args.unit is not None or args.fast or args.fence or args.gate not in (None, "none")):
        raise SystemExit("--no-frames is the unfenced text-only regime; no --unit/--fast/--fence/--gate")
    if args.unit is not None and (args.fast or args.fence or args.gate not in (None, "none") or args.port_check):
        raise SystemExit("--unit is the unfenced evidence-only regime; no --fast/--fence/--gate/--port-check")
    if args.res and args.max_pixels:
        raise SystemExit("give --res or --max-pixels, not both")
    if args.unit is not None and "evidence" in spec.protocols:
        if args.config is not None or args.n_frames is not None or args.protocol is not None:
            raise SystemExit("--unit on a video dataset names the split itself; drop --config/--protocol/--n-frames")
        split_name = spec.split_name("evidence", args.unit, args.split)
    elif args.config is not None:
        if spec.name != "mmred":
            raise SystemExit("--config is MMReD's config name; other datasets take --protocol/--n-frames")
        if args.n_frames is not None:
            raise SystemExit("give --config or --n-frames, not both")
        split_name = f"{args.config}_{args.split}"
    elif args.n_frames is not None:
        split_name = spec.split_name(args.protocol or spec.protocols[0], args.n_frames, args.split)
    else:
        raise SystemExit("one of --config (MMReD) or --n-frames [--protocol] is required")

    # ---- resolve the eval contract
    adapter_cfg: Dict[str, Any] = {}
    if args.adapter is not None:
        p = args.adapter / "train_config.json"
        if p.exists():
            adapter_cfg = json.loads(p.read_text())
    if args.arm is not None:
        if args.layout is not None or args.fence or args.gate is not None:
            raise SystemExit("--arm replaces --layout/--fence/--gate; do not combine")
        arm = ARMS[args.arm]
        layout, fence, gate = arm.layout, arm.fence, arm.gate
    else:
        layout = args.layout or adapter_cfg.get("layout") or "paper"
        fence = bool(args.fence or adapter_cfg.get("fence", False))
        gate = args.gate or "none"
    if args.attn_logn_sref == 0 and adapter_cfg.get("attn_logn_sref"):
        args.attn_logn_sref = int(adapter_cfg["attn_logn_sref"])      # the log-N training prior is part of the adapter contract
    pc = PORT_CHECKS[args.port_check] if args.port_check else None
    if pc is not None:
        if spec.name != "mmred" or bspec.family != "qwen2.5-vl":
            raise SystemExit("--port-check reproduces legacy MMReD/Qwen2.5-VL numbers only")
        if gate != "none" or args.layout is not None or args.arm is not None:
            raise SystemExit("--port-check fixes layout/gate; do not combine with --layout/--gate/--arm")
        if pc.name == "p2" and args.adapter is None:
            raise SystemExit("--port-check p2 needs --adapter checkpoints/sft_fenced_hf_adapter")
        if pc.name == "arm-a" and args.adapter is not None:
            raise SystemExit("--port-check arm-a is the frozen model")
        layout, fence = pc.layout, pc.fence
    if gate == "oracle" and not fence:
        raise SystemExit("--gate oracle requires --fence (the gate hides blocks)")
    if adapter_cfg and args.layout is None and args.arm is None and pc is None and adapter_cfg.get("layout") != layout:
        print(f"[warn] adapter trained with layout={adapter_cfg.get('layout')} fence={adapter_cfg.get('fence')}", flush=True)

    cond = cond_tag(args.attn_sharpen, args.sharpen_from_layer, args.attn_logn_sref)
    if args.fast and (not fence or pc is not None):
        raise SystemExit("--fast is the batched FENCED path; it needs a fenced arm and no port check")
    regime = "T" if args.no_frames else (f"E-{args.unit}" if args.unit else "")
    run_dir = args.output / (f"{time.strftime('%Y%m%d_%H%M%S')}_{'pc-' + pc.name if pc else 'faithful'}"
                             f"{'' if cond == 'base' else '_' + cond}{'_fast' if args.fast else ''}"
                             f"{'_mp' + str(args.max_pixels) if args.max_pixels else ''}"
                             f"{'_' + regime if regime else ''}{'_res' + args.res if args.res else ''}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "frame_set": getattr(spec, "frame_set", None), "dataset": spec.name, "backbone": bspec.name, "model_id": bspec.model_id, "split_name": split_name,
           "data_root": str(data_root), "layout": layout, "fence": fence, "gate": gate, "regime": regime or "full",
           "port_check": pc.name if pc else None, "cond": cond, "adapter_train_config": adapter_cfg}

    # ---- rows
    samples = spec.load(data_root, split_name, args.qtypes)
    if args.qids_file is not None:
        want = [ln.strip() for ln in args.qids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        by = {s.qid: s for s in samples}
        missing = [q for q in want if q not in by]
        if missing and (pc is not None or args.unit is None):
            raise SystemExit(f"{len(missing)} qids not in {split_name}: {missing[:5]}")
        if missing:                                       # evidence protocol: rows without units (RLPC "all", none) drop out
            print(f"[rows] {len(missing)} qids of {args.qids_file} have no evidence units in {split_name} (skipped)", flush=True)
            cfg["qids_without_units"] = len(missing)
        samples = [by[q] for q in want if q in by]
    elif args.limit_per_qtype:
        seen: Dict[str, int] = {}
        keep_rows = []
        for s in samples:                                # file order: the legacy eval_frozen semantics
            if seen.get(s.qtype, 0) < args.limit_per_qtype:
                keep_rows.append(s)
                seen[s.qtype] = seen.get(s.qtype, 0) + 1
        samples = keep_rows
    elif args.per_qtype:
        keep_rows = []
        for qt in (args.qtypes or spec.qtypes):
            keep_rows += stratified_order([s for s in samples if s.qtype == qt], args.seed, key=spec.stratify_key)[: args.per_qtype]
        samples = keep_rows
    else:
        samples = stratified_order(samples, args.seed, key=spec.stratify_key)
        if args.limit:
            samples = samples[: args.limit]
    n_no_evidence = 0
    if args.unit is not None and "evidence" not in spec.protocols:     # MMReD: the config split reduced to its evidence frames
        if args.unit != "frame":
            raise SystemExit(f"{spec.name} has one unit (a frame); --unit frame")
        reduced = [spec.evidence_only(s) for s in samples]
        n_no_evidence = sum(1 for r in reduced if r is None)
        samples = [r for r in reduced if r is not None]
    if args.no_frames:
        samples = [replace(s, frame_paths=(), evidence=None, meta={**s.meta, "protocol": "text"}) for s in samples]
    cfg["rows_without_evidence"] = n_no_evidence
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"[rows] {len(samples)} from {spec.name}/{split_name} backbone={bspec.name} layout={layout} fence={fence} "
          f"gate={gate} regime={regime or 'full'} port_check={pc.name if pc else None} load={getattr(spec, 'last_load', {}) or 'n/a'}", flush=True)

    # ---- model
    rt = bb.load_runtime(bspec)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    base_model = model                                    # positions (get_rope_index) come from the unwrapped model
    layers = get_layers(model)                            # structural refs BEFORE the PEFT wrap
    base_scaling = set_sharpen(layers, args.attn_sharpen, args.sharpen_from_layer)   # DIAG knobs (eval-only)
    sid = bb.special_ids(bspec, tok)
    eos_id = int(tok.eos_token_id)
    if pc is not None:
        set_max_pixels(processor, pc.max_pixels)
    bb.set_max_pixels(bspec, processor, args.max_pixels)
    res_kwargs = bb.set_resolution(bspec, processor, args.res)
    if res_kwargs:
        cfg["res_kwargs"] = res_kwargs
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
        print(f"[model] resolution mode {args.res}: {res_kwargs}", flush=True)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        print(f"[model] adapter {args.adapter}", flush=True)
    hooks = FenceHooks(layers).install() if (fence and not args.fast) else None
    max_new = pc.max_new if pc else args.max_new_tokens
    stop_fn = _stop_newline if (pc and pc.name == "p2") else spec.stop_decoding

    # ---- loop
    records: List[Dict[str, Any]] = []
    preds: List[Optional[str]] = []
    used: List[Any] = []
    n_gold_mismatch = n_gate_skip = n_layout_skip = n_parse_fail = n_too_long = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        if not spec.verify(s):
            n_gold_mismatch += 1
        if args.fast:
            n = s.n_frames
            keep = None
            if gate == "oracle":
                if s.evidence is None:
                    n_gate_skip += 1
                    continue
                keep = [t in s.evidence for t in range(n)]
            try:
                r = fenced_forward(bspec, model, base_model, processor, list(s.frame_paths), spec.question_text(s),
                                   layout=layout, system_prompt=spec.system_prompt, keep=keep, max_new=max_new,
                                   stop_fn=stop_fn, chunk_tokens=args.chunk_tokens, max_read_tokens=args.max_seq_tokens)
            except ValueError as exc:
                if "read too long" in str(exc):
                    n_too_long += 1
                    continue
                n_layout_skip += 1
                print(f"  [skip] {s.qid}: {exc}", flush=True)
                continue
            raw, n_tokens = r.text, r.tokens_read
            pred = spec.parse(raw)
            ok = pred is not None
            correct = spec.match(pred, s)
            n_parse_fail += not ok
            used.append(s)
            preds.append(pred)
            records.append({"qid": s.qid, "qtype": s.qtype, "seq_len": s.n_frames, "atype": s.meta.get("atype", ""),
                            "gold": s.answer, "raw": raw[:160].replace("\n", "\\n"), "pred": "" if pred is None else pred,
                            "correct": int(correct), "parse_ok": int(ok), "n_evid": "" if keep is None else sum(keep),
                            "tokens": n_tokens, "tokens_per_frame": r.tokens_block,
                            "k": s.meta.get("k", ""), "unit_eff": s.meta.get("unit_eff", "")})
            if (i + 1) % 25 == 0:
                acc = sum(x["correct"] for x in records) / max(1, len(records))
                print(f"  {i + 1}/{len(samples)} acc={acc:.3f} read {n_tokens} tok, {r.tokens_block}/frame ({time.time() - t0:.0f}s)", flush=True)
            continue
        fr = spec.frames(s)
        n = len(fr)
        if pc is not None and pc.name == "p2":
            msgs = legacy_p2_messages(fr, s.question)
        else:
            qtext = (pc.question_prefix if pc else "") + spec.question_text(s)
            msgs = build_messages(fr, qtext, layout=layout, system_prompt=spec.system_prompt)
        enc = bb.encode(bspec, processor, msgs, rt.device)
        ids = enc["input_ids"]
        if fence and int(ids.shape[1]) > args.max_seq_tokens:
            n_too_long += 1
            continue
        keep = None
        if gate == "oracle":
            if s.evidence is None:
                n_gate_skip += 1
                continue
            keep = [t in s.evidence for t in range(n)]
        raw = ""
        set_logn(layers, int(ids.shape[1]), args.attn_logn_sref, base_scaling)   # per prompt length
        with torch.inference_mode():
            if not fence:
                try:
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=eos_id)
                except torch.cuda.OutOfMemoryError:          # the longest unfenced prompts (evidence-only clips, ~55 K tokens)
                    torch.cuda.empty_cache()
                    n_too_long += 1
                    print(f"  [skip] {s.qid}: CUDA OOM at {int(ids.shape[1])} tokens ({n} frames)", flush=True)
                    continue
                raw = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            else:
                if pc is not None and pc.name == "p2":
                    parsed = legacy_p2_blocks(ids[0].tolist(), tok, n, sid.vision_start)
                    if parsed is None:
                        n_layout_skip += 1
                        continue
                    blocks, fin = parsed
                else:
                    try:
                        blocks, fin = bb.blocks(bspec, ids[0].cpu(), sid, layout)
                    except ValueError:
                        n_layout_skip += 1
                        continue
                    if len(blocks) != n:
                        n_layout_skip += 1
                        continue
                setup = lambda cur: bb.fenced_setup(bspec, base_model, cur, blocks, fin, keep)  # noqa: E731
                raw = greedy_decode(model, hooks, enc, setup, tokenizer=tok, max_new=max_new, eos_id=eos_id,
                                    stop_fn=stop_fn, passthrough_keys=bspec.passthrough_keys)
        # ---- parse + score
        if pc is None:
            pred = spec.parse(raw)
            ok = pred is not None
            correct = spec.match(pred, s)
        elif pc.parser == "legacy-ladder":
            pred, ok = legacy_regex_ladder(raw)
            correct = pred is not None and legacy_norm(pred) == legacy_norm(s.answer)
        else:
            pred = first_integer(raw)
            ok = pred is not None
            correct = pred is not None and int(pred) == int(s.answer)
        n_parse_fail += not ok
        used.append(s)
        preds.append(pred)
        records.append({"qid": s.qid, "qtype": s.qtype, "seq_len": s.n_frames, "atype": s.meta.get("atype", ""),
                        "gold": s.answer, "raw": raw[:160].replace("\n", "\\n"), "pred": "" if pred is None else pred,
                        "correct": int(correct), "parse_ok": int(ok),
                        "n_evid": "" if keep is None else sum(keep),
                        "tokens": int(ids.shape[1]), "tokens_per_frame": "",
                        "k": s.meta.get("k", ""), "unit_eff": s.meta.get("unit_eff", "")})
        if (i + 1) % 25 == 0:
            acc = sum(r["correct"] for r in records) / max(1, len(records))
            print(f"  {i + 1}/{len(samples)} acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if hooks is not None:
        hooks.remove()

    # ---- summaries (from the recorded flags: the dataset's own rule, port checks included)
    lines = []
    toks = [r["tokens"] for r in records if r.get("tokens") != ""]
    head = f"{'PORT CHECK (deviation: ' + pc.name + ') ' if pc else 'FAITHFUL '}dataset={spec.name} split={split_name} " \
           f"regime={regime or 'full'} res={args.res or 'default'} " \
           f"frames={getattr(spec, 'frame_set', 'native')} max_pixels={args.max_pixels or 'default'} path={'fast' if args.fast else 'dense'} " \
           f"tokens_mean={sum(toks) / max(1, len(toks)):.0f} " \
           f"backbone={bspec.name} layout={layout} fence={fence} gate={gate} adapter={args.adapter or 'none'} n={len(records)} " \
           f"parse_fail={n_parse_fail} ({n_parse_fail / max(1, len(records)):.3f}) gold_mismatch={n_gold_mismatch} " \
           f"gate_skip={n_gate_skip} layout_skip={n_layout_skip} too_long={n_too_long} no_evidence={n_no_evidence}"
    lines.append(head)
    if pc:
        lines.append(f"  anchor: {pc.anchor}")
    summary = [("group", "n", "correct", "acc", "ci_lo", "ci_hi")]
    if records:
        flags = [bool(r["correct"]) for r in records]
        lo, hi = bootstrap_ci(flags)
        lines.append(f"OVERALL acc {sum(flags)}/{len(flags)} = {sum(flags) / len(flags):.3f}  [{lo:.3f}, {hi:.3f}]")
        summary.append(("all", len(flags), sum(flags), f"{sum(flags) / len(flags):.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        for q in sorted({r["qtype"] for r in records}):
            fl = [bool(r["correct"]) for r in records if r["qtype"] == q]
            lo, hi = bootstrap_ci(fl)
            lines.append(f"  {q}: {sum(fl)}/{len(fl)} = {sum(fl) / len(fl):.3f}")
            summary.append((q, len(fl), sum(fl), f"{sum(fl) / len(fl):.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        for at in sorted({r["atype"] for r in records if r["atype"] != ""}):
            fl = [bool(r["correct"]) for r in records if r["atype"] == at]
            lines.append(f"  atype {at}: {sum(fl)}/{len(fl)} = {sum(fl) / len(fl):.3f}")
            summary.append((f"atype_{at}", len(fl), sum(fl), f"{sum(fl) / len(fl):.4f}", "", ""))
        if pc is None:
            for k, (acc, nk) in spec.report_extras(used, preds).items():
                lines.append(f"  {k}: n={nk} acc={acc:.3f}" if nk else f"  {k}: n=0")
                summary.append((k, nk, "", f"{acc:.4f}" if nk else "", "", ""))
    report = "\n".join(lines)
    (run_dir / "report.txt").write_text(report + "\n")
    with open(run_dir / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["qid"])
        w.writeheader()
        w.writerows(records)
    with open(run_dir / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(summary)
    print(report)
    print("wrote", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
