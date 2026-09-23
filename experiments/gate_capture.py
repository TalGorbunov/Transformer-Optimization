#!/usr/bin/env python3
"""D4: capture each frame's <|vision_end|> slot state under the four (layout x fence) arms so a
CPU fit (experiments/gate_fit.py) can say whether the per-frame fact is linearly readable at the
slot, per question type and per N — the "selection requires conditioning" 2x2 and the feasibility
table for the task-agnostic gate (plan §2 D4; rewrite plan "first experiment of the new repo").

Arms (one forward each): plain = paper layout (images then question), no fence; qfirst =
question-first, no fence; fenced_qlast = paper layout + fence + posreset (question-BLIND blocks);
fenced_qfirst = question-first + fence + posreset (THE method's block: the question is in the
shared prefix). Labels = core.mmred.evidence_frames (the local-predicate rule; qtypes whose rule
returns None are skipped). Layer numbers follow the legacy convention (hidden_states[L] = output
of decoder module L-1).
Output (run dir): capture.npz with X[n_frames_total, n_layers, hidden] float16, plus per-frame
arrays qid, qtype, N, frame, is_evid, gold; meta.json. One arm per invocation.
Usage:
  python experiments/gate_capture.py --config seq_len_8 --split train --arm fenced_qfirst --limit 20 --output outputs/diag/gate/N8/fenced_qfirst
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.fence import FENCED_SDPA, FenceHooks, build_block_mask, layout_blocks, reset_positions, slot_positions  # noqa: E402
from core.mmred import QTYPES, evidence_frames, frames, load_split, recompute_answer, states, stratified_order  # noqa: E402
from core.model import get_layers, get_rope_index_fn, load_runtime, move_to_device, special_ids  # noqa: E402
from core.prompt import build_messages  # noqa: E402

ARMS = {"plain": ("paper", False), "qfirst": ("question-first", False),
        "fenced_qlast": ("paper", True), "fenced_qfirst": ("question-first", True)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=None, choices=QTYPES, help="default: all 24")
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--limit", type=int, default=50, help="rows per qtype (class-interleaved order)")
    ap.add_argument("--hs-layers", nargs="+", type=int, default=[12, 20, 24], help="legacy convention (1-based tuple index)")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-seq-tokens", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    layout, fenced = ARMS[args.arm]
    out = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    sid = special_ids(processor)
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
    capture_mods = [L - 1 for L in args.hs_layers]
    hooks = FenceHooks(layers, capture_layers=capture_mods).install()

    X, qids, qts, Ns, fidx, lab, golds = [], [], [], [], [], [], []
    totals = defaultdict(int)
    t0 = time.time()
    for qtype in (args.qtypes or QTYPES):
        rows = stratified_order(load_split(args.config, args.split, args.data_root, [qtype]), args.seed)[: args.limit]
        n_done = 0
        for row in rows:
            question, gold = row["question"], str(row["answer"])
            st = states(row)
            if recompute_answer(qtype, question, st) != gold:
                totals["gold_mismatch"] += 1
                continue
            evid = evidence_frames(qtype, question, st)
            if evid is None:
                totals[f"no_rule_{qtype}"] += 1
                break                                   # positional rule absent for this qtype: skip it entirely
            fr = frames(row, args.data_root)
            NF = len(fr)
            enc = dict(processor.apply_chat_template(build_messages(fr, question, layout=layout), add_generation_prompt=True,
                                                     tokenize=True, return_dict=True, return_tensors="pt"))
            enc = move_to_device(enc, rt.device)
            ids = enc["input_ids"][0].cpu()
            seq = int(ids.shape[0])
            if seq > args.max_seq_tokens:
                totals["too_long"] += 1
                continue
            slots = slot_positions(ids, sid["vision_end"])
            if len(slots) != NF:
                totals["layout_skip"] += 1
                continue
            mask = pos = None
            cur = dict(enc)
            if fenced:
                blocks, fin = layout_blocks(ids, layout, sid, im_end_id=im_end_id)
                if len(blocks) != NF:
                    totals["layout_skip"] += 1
                    continue
                mask = build_block_mask(seq, blocks, hide_cols=[])
                with torch.inference_mode():
                    base_pos, _ = rope_fn(enc["input_ids"], image_grid_thw=enc.get("image_grid_thw"),
                                          attention_mask=enc.get("attention_mask"))
                pos = reset_positions(base_pos, blocks, fin)
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
                qids.append(str(row["qid"])); qts.append(qtype); Ns.append(NF); fidx.append(t)
                lab.append(int(t in evid)); golds.append(gold)
            n_done += 1
            totals["rows"] += 1
            if totals["rows"] % 25 == 0:
                print(f"  {totals['rows']} rows, {len(X)} frames ({time.time() - t0:.0f}s)", flush=True)
        totals[f"rows_{qtype}"] = n_done
    hooks.remove()
    np.savez_compressed(out / "capture.npz", X=np.stack(X) if X else np.zeros((0, len(capture_mods), 1), np.float16),
                        qid=np.array(qids), qtype=np.array(qts), N=np.array(Ns), frame=np.array(fidx),
                        is_evid=np.array(lab), gold=np.array(golds), layers=np.array(args.hs_layers))
    meta = {"arm": args.arm, "layout": layout, "fenced": fenced, "config": args.config, "split": args.split,
            "hs_layers": args.hs_layers, "frames": len(X), **totals}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
