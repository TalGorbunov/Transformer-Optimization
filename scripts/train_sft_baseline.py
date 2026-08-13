#!/usr/bin/env python3
"""Plain-LoRA SFT baseline: fine-tune the model itself (PEFT LoRA, q/k/v/o+MLP) and let
native softmax attention learn to aggregate — the control column against the carrier
method. Train: supervise the answer tokens (LM loss, prompt masked). Eval: generate and
parse the integer.

Anchors (RESULTS.md): P4.1 in-length SFT -> N=32 0.967 / N=64 0.787 ([2026-07-25] P4);
P1.3 ladder cells via --eval-only-adapter (0.480/0.350/0.220). The PROMPT TEMPLATE is
kept byte-identical to the legacy runs, so saved adapters (checkpoints/sft_*_adapter)
restore and reproduce; --eval-dirs-file pins the comparable cells.

vs legacy/experiments/glstm/lora_sft_baseline.py: the dead frame-axis-era arms
(frame-sup, isolation, decompose, count-holdout, CoT) are gone, and data comes from
MMRED roots + a seeded split (the carrier trainer's conventions) instead of the retired
frame-axis split machinery. Reproducing the exact historic ckpts = run the legacy script.

Ops (the FLASH/GQA lesson, jobs 125260/125263): batch=1 no padding -> drop the
attention_mask so transformers takes the is_causal path where FLASH is eligible; with a
4-D mask MATH materializes 45.6 GiB @N=64.

Usage:
  python scripts/train_sft_baseline.py --data_root <roots, path=LIMIT ok> --epochs 5 \
      --eval-dirs-file <arm-A dirs> --output outputs/carrier/sft
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import (
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
    read_dirs_file,
)
from gnnformer.metrics import format_gold_histogram
from gnnformer.runtime import load_runtime, move_to_device

# FLASH first (supports GQA+causal at O(seq) memory), then EFFICIENT, then MATH.
SFT_SDPA = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
INTEGER_RE = re.compile(r"[+-]?\d+")


def build_messages(frames, question):
    """BYTE-IDENTICAL to the legacy SFT runs — do not change (saved adapters depend on it)."""
    preamble = (f"Question: {question}\nThe following are the {len(frames)} frames "
                f"showing rooms in a house:")
    return [{"role": "user", "content": [{"type": "text", "text": preamble}]
             + [{"type": "image", "image": im} for im in frames]}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform",
                    help="comma-separated roots; each may carry a per-root cap as path=LIMIT")
    ap.add_argument("--limit", type=int, default=900, help="default PER-root cap")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-cap", type=int, default=60)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--target", choices=["mlp", "attn", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--eval-longn", default="", help="comma-separated roots for generate-and-parse eval")
    ap.add_argument("--eval-longn-limit", type=int, default=100)
    ap.add_argument("--eval-dirs-file", action="append", default=[],
                    help="dirs-file for generate-and-parse eval (repeatable; the comparable cells)")
    ap.add_argument("--eval-only-adapter", type=Path, default=None,
                    help="restore a saved LoRA adapter dir and skip training")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/carrier/sft"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_lora"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    if args.eval_only_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.eval_only_adapter), is_trainable=False)
        emit(f"eval-only: adapter loaded from {args.eval_only_adapter}")
        args.epochs = 0
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        tmods = {"mlp": ["gate_proj", "up_proj", "down_proj"],
                 "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "both": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]}[args.target]
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM", target_modules=tmods)
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    # ---- data: MMRED roots, validity-gated, seeded split ----
    samples = []  # (sd, gold)
    n_skip = 0
    for root in args.data_root.split(","):
        root = root.strip()
        if not root:
            continue
        lim = args.limit
        if "=" in root:
            root, lim = root.rsplit("=", 1)
            lim = int(lim)
        n_root = 0
        for sd in iter_sample_dirs_shuffled(Path(root), 0):
            if n_root >= lim:
                break
            try:
                _sid, _frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
            except Exception:
                n_skip += 1
                continue
            if parse_task_labels(q0, states, gold) is None:
                n_skip += 1
                continue
            samples.append((sd, gold))
            n_root += 1
    emit(f"[data] {len(samples)} samples (skip {n_skip}); gold-hist "
         + format_gold_histogram(g for _sd, g in samples))
    split_rng = np.random.default_rng(args.seed if args.split_seed is None else args.split_seed)
    order = split_rng.permutation(len(samples))
    n_tr = int(len(samples) * args.train_frac)
    n_va = int(len(samples) * args.val_frac)
    tr_idx, va_idx, te_idx = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    (run_dir / "train_dirs.txt").write_text("\n".join(str(samples[i][0]) for i in tr_idx) + "\n")
    (run_dir / "eval_dirs.txt").write_text("\n".join(str(samples[i][0]) for i in te_idx) + "\n")
    emit(f"[split] train={len(tr_idx)} val={len(va_idx)} test_iid={len(te_idx)}")

    def load_frames(sd):
        _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        return frames, q0, int(str(a0).strip())

    def train_loss(frames, question, gold):
        msgs = build_messages(frames, question)
        full = processor.apply_chat_template(
            msgs + [{"role": "assistant", "content": [{"type": "text", "text": str(gold)}]}],
            add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        full = move_to_device(dict(full), rt.device)
        full.pop("attention_mask", None)  # FLASH-eligible is_causal path (the GQA fix)
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :bnd] = -100
        with sdpa_kernel(SFT_SDPA):
            return model(**full, labels=labels).loss

    @torch.inference_mode()
    def predict(frames, question):
        inp = processor.apply_chat_template(build_messages(frames, question),
                                            add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt")
        inp = move_to_device(dict(inp), rt.device)
        with sdpa_kernel(SFT_SDPA):
            g = model.generate(**inp, max_new_tokens=5, do_sample=False)
        txt = tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = INTEGER_RE.search(txt)
        return int(m.group(0)) if m else None

    def evaluate(idx_list, cap=None):
        model.eval()
        use = idx_list if cap is None else idx_list[:cap]
        n = ok = 0
        for i in use:
            sd, _g = samples[i]
            try:
                frames, q0, gold = load_frames(sd)
            except Exception:
                continue
            pred = predict(frames, q0)
            n += 1
            ok += int(pred == gold)
        return ok / max(1, n), n

    # ---- train ----
    if args.epochs:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        best_val, best_state, best_epoch = -1.0, None, -1
        vrows = ["epoch,val_acc,n"]
        rng = np.random.default_rng(args.seed)
        for epoch in range(args.epochs):
            model.train()
            rng.shuffle(tr_idx)
            opt.zero_grad()
            run_loss, seen = 0.0, 0
            for step, i in enumerate(tr_idx):
                sd, _g = samples[i]
                try:
                    frames, q0, gold = load_frames(sd)
                    loss = train_loss(frames, q0, gold)
                except Exception as exc:
                    emit(f"  train skip {Path(str(sd)).name}: {exc}")
                    torch.cuda.empty_cache()  # OOM debris cascades into later samples
                    continue
                (loss / args.grad_accum).backward()
                run_loss += float(loss.detach())
                seen += 1
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    opt.zero_grad()
            vacc, vn = evaluate(list(va_idx), cap=args.val_cap)
            vrows.append(f"{epoch},{vacc:.4f},{vn}")
            emit(f"epoch {epoch}: train_loss={run_loss/max(1,seen):.3f} val_acc={vacc:.3f}")
            if vacc > best_val:
                best_val, best_epoch = vacc, epoch
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()
                                            if "lora" in k.lower()})
                model.save_pretrained(str(run_dir / "adapter"))  # persist EVERY new best
                emit(f"  adapter saved @ep{epoch} -> {run_dir / 'adapter'}")
            if args.patience and (epoch - best_epoch) >= args.patience:
                emit(f"early stop (best={best_epoch})")
                break
        (run_dir / "val_by_epoch.csv").write_text("\n".join(vrows) + "\n")
        emit(f"best_epoch={best_epoch} val_acc={best_val:.3f}")
        if best_state is not None:
            # LoRA params only, strict=False: a full 4-bit state_dict carries bnb
            # quant-metadata keys that load_state_dict rejects
            model.load_state_dict({k: v.to(rt.device) for k, v in best_state.items()}, strict=False)

    acc, n = evaluate(list(te_idx))
    emit(f"TEST_IID: acc={acc:.4f} n={n}")
    (run_dir / "summary.csv").write_text(f"split,n,accuracy\ntest_iid,{n},{acc:.4f}\n")

    if args.eval_longn or args.eval_dirs_file:
        lrows = ["source,n,accuracy,parse_fail,mae"]
        sources = [(r.strip(), None) for r in str(args.eval_longn).split(",") if r.strip()]
        sources += [(f, read_dirs_file(Path(f))) for f in args.eval_dirs_file]
        for src, dirlist in sources:
            dirs = dirlist if dirlist is not None else iter_sample_dirs_shuffled(Path(src), 0)
            hits = pf = nn = mn = 0
            mae = 0.0
            per: dict = {}
            for sd in dirs[: args.eval_longn_limit]:
                try:
                    frames, q0, gold = load_frames(sd)
                except Exception:
                    continue
                pred = predict(frames, q0)
                nn += 1
                hits += int(pred == gold)
                if pred is None:
                    pf += 1
                else:
                    mae += abs(pred - gold)
                    mn += 1
                pg = per.setdefault(gold, [0, 0])
                pg[1] += 1
                pg[0] += int(pred == gold)
            pc = " ".join(f"g{g}:{c}/{t2}" for g, (c, t2) in sorted(per.items()))
            emit(f"LONGN {src}: n={nn} acc={hits/max(nn,1):.4f} parse_fail={pf/max(nn,1):.3f} "
                 f"mae={mae/max(mn,1):.2f}  {pc}")
            lrows.append(f"{src},{nn},{hits/max(nn,1):.4f},{pf/max(nn,1):.3f},{mae/max(mn,1):.2f}")
        (run_dir / "longn_eval.csv").write_text("\n".join(lrows) + "\n")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
