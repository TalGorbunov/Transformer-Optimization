#!/usr/bin/env python3
"""Plain-LoRA SFT baseline (the "softmax" baseline) for the minimal-crowding counting tasks.

Contrast to the frame-axis adapter: instead of an explicit sum/soft-OR aggregation module on frozen
reps, here we LoRA-fine-tune the model itself and let its NATIVE softmax attention learn to aggregate.
Tests whether the structured aggregator is necessary, or whether fine-tuning the count-blind softmax
suffices once extraction is clean (minimal crowding). No aggregation module -- LoRA only.

Same data/splits/frames as frame_axis_aggregator_adapter (one task, minimal-crowding dataset, image).
Train: supervise the count-answer token (LM loss, prompt masked). Eval: generate, parse integer.
Writes summary.csv + predictions.csv in the SAME format as the adapter for apples-to-apples comparison.
"""
from __future__ import annotations
import argparse, copy, json, random, re, sys, time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_text_frames_acc as tf
import experiments.glstm.frame_axis_aggregator_adapter as fa  # declare_splits, make_example


def parse_args():
    p = argparse.ArgumentParser(description="Plain-LoRA SFT baseline on a minimal-crowding counting task.")
    p.add_argument("--task", required=True, choices=["steps_in_room", "rooms_visited", "co_occupancy"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--train-seq-lens", default="8")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-cap", type=int, default=60)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--holdout-counts", default="",
                   help="comma-separated gold counts held OUT of training -> test_ood (count-extrapolation OOD).")
    p.add_argument("--cot", action="store_true",
                   help="chain-of-thought baseline: step-by-step prompt + parse final 'Answer: N' (use with --epochs 0).")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "agg_min" / "lora_run")
    return p.parse_args()


def build_messages(frames, question, cot=False):
    preamble = f"Question: {question}\nThe following are the {len(frames)} frames showing rooms in a house:"
    if cot:
        preamble += ("\nThink step by step: go frame by frame and state whether the condition holds in "
                     "each, then on the final line write exactly 'Answer: <number>'.")
    return [{"role": "user", "content": [{"type": "text", "text": preamble}]
             + [{"type": "image", "image": im} for im in frames]}]


def main() -> int:
    args = parse_args()
    print(f"[lora] main reached: task={args.task} data={args.data_root}", flush=True)
    if args.smoke:
        args.epochs = 2
    shared_rng = random.Random(int(args.seed))
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_lora"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")
    def emit(m): print(m, flush=True); log.write(m + "\n"); log.flush()

    emit(f"loading model {args.model_name} (4bit={args.load_in_4bit}) ...")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    emit("model loaded; setting up LoRA")
    if bool(args.load_in_4bit):
        model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    tok = processor.tokenizer

    train_seq = [int(x) for x in str(args.train_seq_lens).replace(",", " ").split()]
    holdout = [int(x) for x in str(args.holdout_counts).replace(",", " ").split()] if args.holdout_counts else []
    if holdout:
        splits = fa.declare_splits_count_holdout(args.data_root, args.split, train_seq[0], args.task,
                                                 set(holdout), args.val_frac, args.test_frac, args.split_seed)
    else:
        splits = fa.declare_splits(args.data_root, args.split, train_seq, [], args.val_frac, args.test_frac,
                                   0, None, args.split_seed)
    emit(f"task={args.task} LoRA r={args.lora_r} holdout={holdout} | splits: train={len(splits['train'])} "
         f"val={len(splits['val'])} test_iid={len(splits['test_iid'])} test_ood={len(splits['test_ood'])}")
    (run_dir / "splits.json").write_text(json.dumps({k: [d for d, _ in v] for k, v in splits.items()}), encoding="utf-8")

    def train_loss(frames, question, gold):
        msgs = build_messages(frames, question)
        full = processor.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": str(gold)}]}],
                                             add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        full = base.move_inputs_to_device(dict(full), device)
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone(); labels[:, :bnd] = -100
        return model(**full, labels=labels).loss

    @torch.inference_mode()
    def predict(frames, question):
        msgs = build_messages(frames, question, cot=args.cot)
        inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inp = base.move_inputs_to_device(dict(inp), device)
        out = model.generate(**inp, max_new_tokens=(256 if args.cot else 5), do_sample=False)
        txt = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if args.cot:  # prefer the 'Answer: N' line; else fall back to the last integer in the reasoning
            mm = re.search(r"[Aa]nswer\s*:?\s*(-?\d+)", txt)
            if mm:
                return int(mm.group(1))
            ints = tf.INTEGER_RE.findall(txt)
            return int(ints[-1]) if ints else None
        m = tf.INTEGER_RE.search(txt)
        return int(m.group(0)) if m else None

    def evaluate(items, cap=None, records=None, split_name="test_iid"):
        model.eval()
        use = items if cap is None else items[:cap]
        n = ok = 0; gs = ps = 0.0
        for (dstr, sl) in use:
            ex = fa.make_example(Path(dstr), args.task, shared_rng, eval_mode=True)
            if ex is None:
                continue
            frames, question, gold, nf, _ = ex
            pred = predict(frames, question)
            n += 1; ok += int(pred is not None and pred == gold); gs += gold; ps += (pred or 0)
            if records is not None and pred is not None:
                records.append((args.task, split_name, int(sl), int(gold), int(pred)))
        return (ok / max(1, n), (ps - gs) / max(1, n), gs / max(1, n), ps / max(1, n), n)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best_val, best_state, best_epoch = -1.0, None, -1
    vrows = ["epoch,val_acc,val_bias,n"]
    for epoch in range(args.epochs):
        model.train()
        order = list(splits["train"]); shared_rng.shuffle(order)
        opt.zero_grad(); run_loss = 0.0; seen = 0; step = 0
        for i, (dstr, sl) in enumerate(order):
            ex = fa.make_example(Path(dstr), args.task, shared_rng, eval_mode=False)
            if ex is None:
                continue
            frames, question, gold, nf, _ = ex
            try:
                loss = train_loss(frames, question, gold)
            except Exception as exc:
                emit(f"  train skip {Path(dstr).name}: {exc}"); continue
            (loss / args.grad_accum).backward()
            run_loss += float(loss.detach()); seen += 1; step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
        vacc, vbias, _, _, vn = evaluate(splits["val"], cap=args.val_cap)
        vrows.append(f"{epoch},{vacc:.4f},{vbias:+.3f},{vn}")
        emit(f"epoch {epoch}: train_loss={run_loss/max(1,seen):.3f} val_acc={vacc:.3f} val_bias={vbias:+.3f}")
        if vacc > best_val:
            best_val, best_epoch = vacc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k.lower()})
        if args.patience and (epoch - best_epoch) >= args.patience:
            emit(f"early stop: no val improvement in {args.patience} epochs (best={best_epoch})"); break
    (run_dir / "val_by_epoch.csv").write_text("\n".join(vrows) + "\n", encoding="utf-8")
    emit(f"best_epoch={best_epoch} val_acc={best_val:.3f}")

    if best_state is not None:
        # load ONLY the LoRA params (strict=False) -- a full state_dict() on a 4-bit model carries
        # bitsandbytes quant-metadata keys (.absmax/.quant_map/...) that load_state_dict rejects.
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    records = []
    summary = ["split,task,readout,n,accuracy,mean_gold,mean_pred,bias"]
    acc, bias, mg, mp, n = evaluate(splits["test_iid"], records=records, split_name="test_iid")
    summary.append(f"test_iid,{args.task},lm,{n},{acc:.4f},{mg:.3f},{mp:.3f},{bias:+.3f}")
    emit(f"TEST_IID {args.task}: acc={acc:.4f} bias={bias:+.3f} n={n}  (best_epoch={best_epoch})")
    if splits["test_ood"]:
        oacc, obias, omg, omp, on = evaluate(splits["test_ood"], records=records, split_name="test_ood")
        summary.append(f"test_ood,{args.task},lm,{on},{oacc:.4f},{omg:.3f},{omp:.3f},{obias:+.3f}")
        emit(f"TEST_OOD {args.task}: acc={oacc:.4f} bias={obias:+.3f} n={on}")
    (run_dir / "summary.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (run_dir / "predictions.csv").write_text(
        "\n".join(["task,split,seq_len,gold,pred"] + [f"{t},{s},{sl},{g},{p}" for (t, s, sl, g, p) in records]) + "\n",
        encoding="utf-8")
    log.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
