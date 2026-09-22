#!/usr/bin/env python3
"""Fenced LoRA SFT on official MMReD rows: the paper's prompt verbatim, native 512 px frames,
the answer target `{ "answer": "<v>" }`, every frame in its own attention block with a per-block
position reset (core.fence), optional oracle evidence gate.

Recipe = P2 (legacy/v1/scripts/loramech/train_sft_fenced.py, run
outputs/loramech/p2_fenced_hf/20260827_193851_fenced, job 137799: 0.900/0.660/0.300 @ 8/16/32
at 392 px under the OLD count prompt): prepare_model_for_kbit_training; LoRA r=8 alpha=32
dropout 0.05 on q/k/v/o/gate/up/down; AdamW lr 2e-4; grad-accum 8; clip 1.0; answer-token loss;
seed 0. The mask is HELD THROUGH BACKWARD (legacy/v1/scripts/sparse/train_sft_gated.py:545-552,
the 139063 diagnosis): gradient checkpointing re-runs layer forwards inside backward, so a mask
cleared before backward makes the recompute run unfenced -> corrupted gradients. The P2 trainer's
try/finally clear is the bug; it is not ported.

Deliberate differences from P2 (this is the NEW baseline row, not a parity check): official
val split instead of an 80/20 carve of train; all train rows of each config; 5 epochs
(Tal, 2026-09-01); the faithful prompt; train.py never touches test rows (evaluate.py does).

Reserved, refused for now: --gate model, --gate-npz, --gate-bonus, --virtual-n.
Usage:
  python experiments/train.py --configs seq_len_8 seq_len_16 --qtypes steps_in_room --limit 40 --epochs 1 \
      --val-limit 20 --output outputs/_scratch/train
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.constants import TRAIN_SEQ_LENS  # noqa: E402
from core.fence import FENCED_SDPA, FenceHooks, fenced_setup, greedy_decode, layout_blocks  # noqa: E402
from core.mmred import QTYPES, evidence_frames, frames, load_split, recompute_answer, states, stratified_order  # noqa: E402
from core.model import get_layers, get_rope_index_fn, load_runtime, move_to_device, special_ids  # noqa: E402
from core.prompt import build_messages, exact_match, parse_answer  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--configs", nargs="+", default=["seq_len_8", "seq_len_16"])
    ap.add_argument("--qtypes", nargs="+", default=None, choices=QTYPES, help="default: all 24")
    ap.add_argument("--limit", type=int, default=0, help="train rows per config after stratified_order")
    ap.add_argument("--val-limit", type=int, default=100, help="val rows total after stratified_order")
    ap.add_argument("--layout", choices=["question-first", "replica"], default="question-first")
    ap.add_argument("--gate", choices=["none", "oracle", "model"], default="none")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0, help="0 = run every epoch")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gate-npz", default=None)
    ap.add_argument("--gate-bonus", type=float, default=0.0)
    ap.add_argument("--virtual-n", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for flag, val, off in (("--gate model", args.gate, "model"), ("--gate-npz", args.gate_npz, None),
                           ("--gate-bonus", args.gate_bonus, 0.0), ("--virtual-n", args.virtual_n, None)):
        if (val == off) if flag == "--gate model" else (val not in (None, 0.0)):
            raise SystemExit(f"reserved: {flag} is not implemented in v2 yet")
    for c in args.configs:
        n = int(c.split("_")[-1])
        if n not in TRAIN_SEQ_LENS:
            raise SystemExit(f"{c}: headline training data is seq_len <= 16 only (CLAUDE.md §7)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.layout}_{args.gate}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m: str) -> None:
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    # ---- data
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for c in args.configs:
        tr = stratified_order(load_split(c, "train", args.data_root, args.qtypes), args.seed)
        train_rows += tr[: args.limit] if args.limit else tr
        val_rows += load_split(c, "val", args.data_root, args.qtypes)
    val_rows = stratified_order(val_rows, args.seed)[: args.val_limit] if args.val_limit else stratified_order(val_rows, args.seed)
    bad = sum(recompute_answer(r["qtype"], r["question"], states(r)) != str(r["answer"]) for r in train_rows + val_rows)
    if bad:
        raise SystemExit(f"{bad} rows whose recomputed gold differs from the published answer")
    majority = Counter(str(r["answer"]) for r in val_rows).most_common(1)[0][1] / max(1, len(val_rows))
    emit(f"[data] train={len(train_rows)} val={len(val_rows)} configs={args.configs} qtypes={args.qtypes or 'all'} "
         f"val majority baseline={majority:.3f}")
    (run_dir / "train_rows.txt").write_text("\n".join(f"{r['split_dir']}/{r['qid']}" for r in train_rows) + "\n")
    (run_dir / "val_rows.txt").write_text("\n".join(f"{r['split_dir']}/{r['qid']}" for r in val_rows) + "\n")

    # ---- model (structural refs BEFORE the PEFT wrap)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    sid = special_ids(processor)
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    eos_id = int(tok.eos_token_id)
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    hooks = FenceHooks(layers).install()
    train_config = {"layout": args.layout, "fence": True, "gate": args.gate, "configs": args.configs,
                    "qtypes": args.qtypes, "prompt": "paper-verbatim", "resize": "native-512", "core": "2.0.0"}

    def keep_for(row, n):
        if args.gate == "none":
            return None
        ev = evidence_frames(row["qtype"], row["question"], states(row))
        if ev is None:
            raise ValueError("no evidence label")
        return [t in ev for t in range(n)]

    def train_loss(row):
        fr = frames(row, args.data_root)
        keep = keep_for(row, len(fr))
        full = processor.apply_chat_template(build_messages(fr, row["question"], args.layout, answer=str(row["answer"])),
                                             add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = processor.apply_chat_template(build_messages(fr, row["question"], args.layout),
                                               add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        full = move_to_device(dict(full), rt.device)
        blocks, fin = layout_blocks(full["input_ids"][0].cpu(), args.layout, sid, im_end_id=im_end_id)
        if len(blocks) != len(fr):
            raise ValueError("layout parse failed")
        mask, pos = fenced_setup(full, blocks, fin, keep, rope_fn)
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :bnd] = -100
        full.pop("attention_mask", None)
        hooks.set_mask(mask, rt.device)                 # stays set through backward (see docstring)
        with sdpa_kernel(FENCED_SDPA):
            return model(**full, position_ids=pos.to(rt.device), labels=labels).loss

    @torch.inference_mode()
    def predict(row) -> Optional[str]:
        fr = frames(row, args.data_root)
        keep = keep_for(row, len(fr))
        enc = processor.apply_chat_template(build_messages(fr, row["question"], args.layout), add_generation_prompt=True,
                                            tokenize=True, return_dict=True, return_tensors="pt")
        enc = move_to_device(dict(enc), rt.device)
        blocks, fin = layout_blocks(enc["input_ids"][0].cpu(), args.layout, sid, im_end_id=im_end_id)
        if len(blocks) != len(fr):
            raise ValueError("layout parse failed")
        setup = lambda cur: fenced_setup(cur, blocks, fin, keep, rope_fn)  # noqa: E731
        raw = greedy_decode(model, hooks, enc, setup, tokenizer=tok, max_new=args.max_new_tokens, eos_id=eos_id,
                            stop_fn=lambda t: "answer" in t and "}" in t.split("answer", 1)[1])
        return parse_answer(raw)

    def evaluate(rows) -> tuple:
        model.eval()
        n = ok = 0
        for r in rows:
            try:
                pred = predict(r)
            except Exception as exc:
                emit(f"  val skip {r['qid']}: {exc}")
                continue
            n += 1
            ok += int(exact_match(pred, str(r["answer"])))
        return ok / max(1, n), n

    # ---- train
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    best_val, best_state, best_epoch = -1.0, None, -1
    vrows = ["epoch,train_loss,val_acc,n_val"]
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(train_rows))
    n_skip = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(order)
        opt.zero_grad()
        run_loss, seen = 0.0, 0
        for step, i in enumerate(order):
            row = train_rows[int(i)]
            try:
                loss = train_loss(row)
                (loss / args.grad_accum).backward()
            except Exception as exc:
                emit(f"  train skip {row['qid']}: {exc}")
                n_skip += 1
                hooks.clear_mask()
                torch.cuda.empty_cache()
                continue
            hooks.clear_mask()                          # after backward, never before
            run_loss += float(loss.detach())
            seen += 1
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
        vacc, vn = evaluate(val_rows)
        vrows.append(f"{epoch},{run_loss / max(1, seen):.4f},{vacc:.4f},{vn}")
        emit(f"epoch {epoch}: train_loss={run_loss / max(1, seen):.3f} val_acc={vacc:.3f} n_val={vn} "
             f"skips={n_skip} ({time.time() - t0:.0f}s)")
        if vacc > best_val:
            best_val, best_epoch = vacc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k.lower()})
            model.save_pretrained(str(run_dir / "adapter"))
            (run_dir / "adapter" / "train_config.json").write_text(json.dumps({**train_config, "best_epoch": epoch,
                                                                             "val_acc": vacc}, indent=1))
            emit(f"  adapter saved @ep{epoch} -> {run_dir / 'adapter'}")
        if args.patience and (epoch - best_epoch) >= args.patience:
            emit(f"early stop (best={best_epoch})")
            break
    (run_dir / "val_by_epoch.csv").write_text("\n".join(vrows) + "\n")
    if best_state is not None:
        model.load_state_dict({k: v.to(rt.device) for k, v in best_state.items()}, strict=False)
    hooks.remove()
    summary = f"best_epoch,val_acc,n_val,n_train,majority_baseline\n{best_epoch},{best_val:.4f},{len(val_rows)},{len(train_rows)},{majority:.4f}\n"
    (run_dir / "summary.csv").write_text(summary)
    report = (f"FENCED SFT layout={args.layout} gate={args.gate} configs={args.configs} qtypes={args.qtypes or 'all'}\n"
              f"train={len(train_rows)} val={len(val_rows)} epochs={args.epochs} best_epoch={best_epoch} "
              f"val_acc={best_val:.3f} (majority {majority:.3f}) skips={n_skip}\n"
              f"adapter: {run_dir / 'adapter'}  (eval contract: layout={args.layout}, fence, gate={args.gate})\n")
    (run_dir / "report.txt").write_text(report)
    emit(report)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
