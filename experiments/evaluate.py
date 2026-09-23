#!/usr/bin/env python3
"""Evaluate the frozen model or a LoRA adapter on one official MMReD split: exact match per
question type, the paper's prompt verbatim, native 512 px frames, upstream parsing rules.

Faithful protocol (every run without --port-check): core.prompt.SYSTEM_PROMPT as the system
turn; the user turn = the frames and the bare question (layout paper = frames then question,
question-first = their --prefix_question order = THE method's layout, replica = the S1-S11
layout); greedy decode; core.prompt.parse_answer / exact_match; per-qtype EM with bootstrap CIs;
numeric qtypes also split by answer <= 16 / > 16.

--fence puts every frame in its own attention block with a per-block position reset
(core.fence); --gate oracle additionally hides every non-evidence block from the tail
(core.mmred.evidence_frames = the gate's labels; NEVER used for a reported model-gate row).

Anchors (ONE-TIME port checks, experiments/_port_check.py; never a reported row):
  --port-check arm-a  --frozen  seq_len_8 test, all 24 qtypes  -> 0.533 (640/1200), grid_seq8_test
                      --qtypes final_app steps_in_room where_spend --limit-per-qtype 34
                                                                 -> 0.765 / 0.559 / 0.353 (127776)
  --port-check p2     --adapter checkpoints/sft_fenced_hf_adapter --qids-file sbatch/lib/splits/qids_p2_seq8_test.txt
                      (and seq16 / seq32)                        -> 45/50, 33/50, 15/50

Outputs (run dir): config.json, eval.csv (one row per sample), summary.csv, report.txt.
Usage:
  python experiments/evaluate.py --config seq_len_8 --split test --frozen --limit 20 --output outputs/_scratch/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.fence import FENCED_SDPA, FenceHooks, fenced_setup, greedy_decode, layout_blocks  # noqa: E402
from experiments._diag_common import cond_tag, set_logn, set_sharpen  # noqa: E402
from core.metrics import bootstrap_ci, em_table, split_by_answer  # noqa: E402
from core.mmred import NUMERIC_QTYPES, QTYPES, evidence_frames, frames, load_split, recompute_answer, states, stratified_order  # noqa: E402
from core.model import get_layers, get_rope_index_fn, load_runtime, move_to_device, special_ids  # noqa: E402
from core.prompt import LAYOUTS, build_messages, exact_match, parse_answer  # noqa: E402
from experiments._port_check import (  # noqa: E402
    PORT_CHECKS, first_integer, legacy_norm, legacy_p2_blocks, legacy_p2_messages, legacy_regex_ladder, set_max_pixels,
)


def _stop_json(text: str) -> bool:
    return "answer" in text and "}" in text.split("answer", 1)[1]


def _stop_newline(text: str) -> bool:
    return "\n" in text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=None, choices=QTYPES, help="default: all 24")
    ap.add_argument("--qids-file", type=Path, default=None, help="one qid per line; pins the rows and their order")
    ap.add_argument("--limit", type=int, default=0, help="rows after stratified_order (class-interleaved)")
    ap.add_argument("--limit-per-qtype", type=int, default=0, help="rows per qtype in JSON order (legacy semantics)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter", type=Path, default=None)
    grp.add_argument("--frozen", action="store_true")
    ap.add_argument("--layout", choices=LAYOUTS, default=None, help="default: adapter's train_config.json, else paper")
    ap.add_argument("--fence", action="store_true", help="default: adapter's train_config.json")
    ap.add_argument("--gate", choices=["none", "oracle"], default="none")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-seq-tokens", type=int, default=24000, help="refuse the dense fenced path above this")
    ap.add_argument("--attn-sharpen", type=float, default=0.0, help="DIAG: tau on decoder modules >= --sharpen-from-layer (0 = off)")
    ap.add_argument("--sharpen-from-layer", type=int, default=12)
    ap.add_argument("--attn-logn-sref", type=int, default=0, help="DIAG: log-N logit scaling reference length in tokens (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--port-check", choices=sorted(PORT_CHECKS), default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # ---- resolve the eval contract
    adapter_cfg: Dict[str, Any] = {}
    if args.adapter is not None:
        p = args.adapter / "train_config.json"
        if p.exists():
            adapter_cfg = json.loads(p.read_text())
    layout = args.layout or adapter_cfg.get("layout") or "paper"
    fence = bool(args.fence or adapter_cfg.get("fence", False))
    if args.attn_logn_sref == 0 and adapter_cfg.get("attn_logn_sref"):
        args.attn_logn_sref = int(adapter_cfg["attn_logn_sref"])      # the log-N training prior is part of the adapter contract
    gate = args.gate
    pc = PORT_CHECKS[args.port_check] if args.port_check else None
    if pc is not None:
        if gate != "none" or args.layout is not None:
            raise SystemExit("--port-check fixes layout/gate; do not combine with --layout/--gate")
        if pc.name == "p2" and args.adapter is None:
            raise SystemExit("--port-check p2 needs --adapter checkpoints/sft_fenced_hf_adapter")
        if pc.name == "arm-a" and args.adapter is not None:
            raise SystemExit("--port-check arm-a is the frozen model")
        layout, fence = pc.layout, pc.fence
    if gate == "oracle" and not fence:
        raise SystemExit("--gate oracle requires --fence (the gate hides blocks)")
    if adapter_cfg and args.layout is None and pc is None and adapter_cfg.get("layout") != layout:
        print(f"[warn] adapter trained with layout={adapter_cfg.get('layout')} fence={adapter_cfg.get('fence')}", flush=True)

    cond = cond_tag(args.attn_sharpen, args.sharpen_from_layer, args.attn_logn_sref)
    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{'pc-' + pc.name if pc else 'faithful'}{'' if cond == 'base' else '_' + cond}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "layout": layout, "fence": fence, "gate": gate, "port_check": pc.name if pc else None, "cond": cond,
           "adapter_train_config": adapter_cfg}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))

    # ---- rows
    rows = load_split(args.config, args.split, args.data_root, args.qtypes)
    if args.qids_file is not None:
        want = [ln.strip() for ln in args.qids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        by = {r["qid"]: r for r in rows}
        missing = [q for q in want if q not in by]
        if missing:
            raise SystemExit(f"{len(missing)} qids not in {args.config}_{args.split}: {missing[:5]}")
        rows = [by[q] for q in want]
    elif args.limit_per_qtype:
        seen: Dict[str, int] = {}
        keep_rows = []
        for r in rows:                                   # JSON order: the legacy eval_frozen semantics
            if seen.get(r["qtype"], 0) < args.limit_per_qtype:
                keep_rows.append(r)
                seen[r["qtype"]] = seen.get(r["qtype"], 0) + 1
        rows = keep_rows
    else:
        rows = stratified_order(rows, args.seed)
        if args.limit:
            rows = rows[: args.limit]
    print(f"[rows] {len(rows)} from {args.config}_{args.split} layout={layout} fence={fence} gate={gate} "
          f"port_check={pc.name if pc else None}", flush=True)

    # ---- model
    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)                            # structural refs BEFORE the PEFT wrap
    base_scaling = set_sharpen(layers, args.attn_sharpen, args.sharpen_from_layer)   # DIAG knobs (eval-only)
    rope_fn = get_rope_index_fn(model)
    sid = special_ids(processor)
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    eos_id = int(tok.eos_token_id)
    if pc is not None:
        set_max_pixels(processor, pc.max_pixels)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        print(f"[model] adapter {args.adapter}", flush=True)
    hooks = FenceHooks(layers).install() if fence else None
    max_new = pc.max_new if pc else args.max_new_tokens
    stop_fn = _stop_newline if (pc and pc.name == "p2") else _stop_json

    # ---- loop
    records: List[Dict[str, Any]] = []
    preds: List[Optional[str]] = []
    used_rows: List[Dict[str, Any]] = []
    n_gold_mismatch = n_gate_skip = n_layout_skip = n_parse_fail = n_too_long = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        qtype, question, gold = row["qtype"], row["question"], str(row["answer"])
        st = states(row)
        if recompute_answer(qtype, question, st) != gold:
            n_gold_mismatch += 1
        fr = frames(row, args.data_root)
        n = len(fr)
        if pc is not None and pc.name == "p2":
            msgs = legacy_p2_messages(fr, question)
        else:
            qtext = (pc.question_prefix if pc else "") + question
            msgs = build_messages(fr, qtext, layout=layout)
        enc = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt")
        enc = move_to_device(dict(enc), rt.device)
        ids = enc["input_ids"]
        if fence and int(ids.shape[1]) > args.max_seq_tokens:
            n_too_long += 1
            continue
        keep = None
        if gate == "oracle":
            ev = evidence_frames(qtype, question, st)
            if ev is None:
                n_gate_skip += 1
                continue
            keep = [t in ev for t in range(n)]
        raw = ""
        set_logn(layers, int(ids.shape[1]), args.attn_logn_sref, base_scaling)   # per prompt length
        with torch.inference_mode():
            if not fence:
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=eos_id)
                raw = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            else:
                ids_list = ids[0].tolist()
                if pc is not None and pc.name == "p2":
                    parsed = legacy_p2_blocks(ids_list, tok, n, sid["vision_start"])
                    if parsed is None:
                        n_layout_skip += 1
                        continue
                    blocks, fin = parsed
                else:
                    try:
                        blocks, fin = layout_blocks(ids[0].cpu(), layout, sid, im_end_id=im_end_id)
                    except ValueError:
                        n_layout_skip += 1
                        continue
                    if len(blocks) != n:
                        n_layout_skip += 1
                        continue
                setup = lambda cur: fenced_setup(cur, blocks, fin, keep, rope_fn)  # noqa: E731
                raw = greedy_decode(model, hooks, enc, setup, tokenizer=tok, max_new=max_new, eos_id=eos_id,
                                    stop_fn=stop_fn)
        # ---- parse + score
        if pc is None:
            pred = parse_answer(raw)
            ok = pred is not None
            correct = exact_match(pred, gold)
        elif pc.parser == "legacy-ladder":
            pred, ok = legacy_regex_ladder(raw)
            correct = pred is not None and legacy_norm(pred) == legacy_norm(gold)
        else:
            pred = first_integer(raw)
            ok = pred is not None
            correct = pred is not None and int(pred) == int(gold)
        n_parse_fail += not ok
        used_rows.append(row)
        preds.append(pred)
        records.append({"qid": row["qid"], "qtype": qtype, "seq_len": row["seq_len"], "atype": row["atype"],
                        "gold": gold, "raw": raw[:160].replace("\n", "\\n"), "pred": "" if pred is None else pred,
                        "correct": int(correct), "parse_ok": int(ok),
                        "n_evid": "" if keep is None else sum(keep)})
        if (i + 1) % 25 == 0:
            acc = sum(r["correct"] for r in records) / max(1, len(records))
            print(f"  {i + 1}/{len(rows)} acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if hooks is not None:
        hooks.remove()

    # ---- summaries
    def _flags(rs):
        return [bool(r["correct"]) for r in rs]

    lines = []
    head = f"{'PORT CHECK (deviation: ' + pc.name + ') ' if pc else 'FAITHFUL '}config={args.config} split={args.split} " \
           f"layout={layout} fence={fence} gate={gate} adapter={args.adapter or 'none'} n={len(records)} " \
           f"parse_fail={n_parse_fail} ({n_parse_fail / max(1, len(records)):.3f}) gold_mismatch={n_gold_mismatch} " \
           f"gate_skip={n_gate_skip} layout_skip={n_layout_skip} too_long={n_too_long}"
    lines.append(head)
    if pc:
        lines.append(f"  anchor: {pc.anchor}")
    summary = [("group", "n", "correct", "acc", "ci_lo", "ci_hi")]
    if records:
        flags = _flags(records)
        lo, hi = bootstrap_ci(flags)
        lines.append(f"OVERALL acc {sum(flags)}/{len(flags)} = {sum(flags) / len(flags):.3f}  [{lo:.3f}, {hi:.3f}]")
        summary.append(("all", len(flags), sum(flags), f"{sum(flags) / len(flags):.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        if pc is None:
            table = em_table(used_rows, preds)
        else:   # port checks score with the legacy rule, so tabulate the recorded flags
            table = {}
            for r in records:
                table.setdefault(r["qtype"], []).append(bool(r["correct"]))
            table = {q: (sum(v) / len(v), len(v)) for q, v in table.items()}
        for q in sorted(k for k in table if k != "all"):
            acc, nq = table[q]
            fl = [bool(r["correct"]) for r in records if r["qtype"] == q]
            lo, hi = bootstrap_ci(fl)
            lines.append(f"  {q}: {sum(fl)}/{nq} = {acc:.3f}")
            summary.append((q, nq, sum(fl), f"{acc:.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        for at in sorted({r["atype"] for r in records}):
            fl = [bool(r["correct"]) for r in records if r["atype"] == at]
            lines.append(f"  atype {at}: {sum(fl)}/{len(fl)} = {sum(fl) / len(fl):.3f}")
            summary.append((f"atype_{at}", len(fl), sum(fl), f"{sum(fl) / len(fl):.4f}", "", ""))
        if pc is None and any(r["qtype"] in NUMERIC_QTYPES for r in used_rows):
            sp = split_by_answer(used_rows, preds, 16)
            for k, (acc, nk) in sp.items():
                lines.append(f"  numeric answer {k}: n={nk} acc={acc:.3f}" if nk else f"  numeric answer {k}: n=0")
                summary.append((f"numeric{k}", nk, "", f"{acc:.4f}" if nk else "", "", ""))
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
