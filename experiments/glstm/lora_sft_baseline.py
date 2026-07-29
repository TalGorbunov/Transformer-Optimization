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
from torch.nn.attention import sdpa_kernel, SDPBackend
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Long-N generate: mask=None -> transformers sets enable_gqa=True (28 q-heads vs 4 kv) ->
# the mem-efficient kernel is INELIGIBLE and MATH materializes 17GB (job 125260 OOM).
# FLASH supports GQA + causal: peak 8.3 GiB at seq 12.7k (smoke 125263).
EFF_SDPA = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_text_frames_acc as tf
import experiments.glstm.frame_axis_aggregator_adapter as fa
from experiments.glstm.cache_minimal_frame_reps import frame_labels  # declare_splits, make_example


def parse_args():
    p = argparse.ArgumentParser(description="Plain-LoRA SFT baseline on a minimal-crowding counting task.")
    p.add_argument("--task", required=True, choices=["steps_in_room","rooms_visited","co_occupancy","distinct_visitors","distinct_companions"])
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
    p.add_argument("--target", choices=["mlp", "attn", "both"], default="both",
                   help="LoRA module isolation: mlp=gate/up/down (per-frame nonlinearity), attn=q/k/v/o (linear re-route), both")
    p.add_argument("--lora-layers", default="",
                   help="restrict LoRA to a layer band, e.g. '8-18' (empty = all layers)")
    p.add_argument("--decompose", action="store_true",
                   help="after training, run the aggregation decomposition with adapter OFF vs ON (paired before/after)")
    p.add_argument("--decomp-layer", type=int, default=19)
    p.add_argument("--frame-sup-weight", type=float, default=0.0,
                   help="V2: aux BCE on a per-frame head over L_decomp reps vs per-frame evidence labels "
                        "(shapes the reps to be evidence-separable -> the per-frame gate). 0=off.")
    p.add_argument("--frame-isolation", action="store_true",
                   help="apply block-diagonal frame-isolation attention mask during all forwards "
                        "(clean per-frame extraction in one forward; the in-model structural solution)")
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
    p.add_argument("--eval-longn", default="",
                   help="E-B (2026-07-20): comma-separated longN data roots — after test_iid, "
                        "generate-and-parse eval on each (stratified-shuffled prefix)")
    p.add_argument("--eval-longn-limit", type=int, default=100)
    p.add_argument("--eval-dirs-file", action="append", default=[],
                   help="P4.1 (2026-07-24): file of sample-dir paths — generate-and-parse "
                        "eval on its first --eval-longn-limit dirs (arm-A dirs-file exams; "
                        "repeatable)")
    p.add_argument("--eval-only-adapter", type=Path, default=None,
                   help="P1.3 (2026-07-23): load a saved LoRA adapter dir and skip training; "
                        "runs test_iid sanity + --eval-longn only")
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
    if args.eval_only_adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.eval_only_adapter), is_trainable=False)
        emit(f"eval-only: adapter loaded from {args.eval_only_adapter}")
    else:
        if bool(args.load_in_4bit):
            model = prepare_model_for_kbit_training(model)
        tmods = {"mlp": ["gate_proj", "up_proj", "down_proj"],
                 "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "both": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}[args.target]
        ltt = None
        if args.lora_layers:
            a, b = (int(x) for x in args.lora_layers.split("-"))
            ltt = list(range(a, b + 1))
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=tmods, layers_to_transform=ltt)
        emit(f"LoRA target={args.target} modules={tmods} layers={args.lora_layers or 'all'}")
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

    # ---- V2: per-frame supervision (aux BCE head over L_decomp frame reps) ----
    import torch.nn as nn, torch.nn.functional as F
    from models.model import get_layers, image_token_groups as _itg
    fr_head, st_fr = None, {"spans": None, "reps": None}
    if args.frame_sup_weight > 0:
        hidden = int(model.config.text_config.hidden_size) if hasattr(model.config, "text_config") else int(model.config.hidden_size)
        fr_head = nn.Linear(hidden, 1).to(device).to(torch.float32)
        base_m = model.get_base_model() if hasattr(model, "get_base_model") else model

        def fr_pre_hook(module, hargs, hkwargs):
            def cap(hs):
                if st_fr["spans"] is not None and hs.shape[1] > max(max(s) for s in st_fr["spans"]):
                    st_fr["reps"] = torch.stack([hs[0, idx, :].float().mean(0) for idx in st_fr["spans"]], 0)  # [N,H] keep grad
                return hs
            if len(hargs) >= 1:
                return (cap(hargs[0]),) + tuple(hargs[1:]), hkwargs
            hkwargs = dict(hkwargs); hkwargs["hidden_states"] = cap(hkwargs["hidden_states"]); return hargs, hkwargs
        get_layers(base_m)[args.frame_sup_layer if hasattr(args, "frame_sup_layer") else args.decomp_layer].register_forward_pre_hook(fr_pre_hook, with_kwargs=True)
        emit(f"per-frame supervision ON: weight={args.frame_sup_weight} @L{args.decomp_layer}")

    # ---- frame-isolation mask (the in-model structural solution): block-diagonal over frames ----
    iso_on = bool(args.frame_isolation)
    if iso_on:
        from experiments.glstm.frame_isolation_diagnostic import (custom_attention as _ISO_ATTN,
                                                                   STATE as _ISO_STATE, build_iso as _ISO_BUILD)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
        _AAF["sdpa"] = _ISO_ATTN
        emit("frame-isolation mask ON (each frame attends only to itself + the question)")

    def set_iso(input_ids, n_frames):
        if not iso_on:
            return
        sp = _itg(input_ids[0].detach().cpu(), n_frames, processor=processor)
        if len(sp) == n_frames:
            _ISO_STATE["iso"] = _ISO_BUILD(int(input_ids.shape[1]), sp); _ISO_STATE["active"] = True
        else:
            _ISO_STATE["active"] = False

    def clear_iso():
        if iso_on:
            _ISO_STATE["active"] = False; _ISO_STATE["iso"] = None

    def train_loss(frames, question, gold, pf=None):
        msgs = build_messages(frames, question)
        full = processor.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": str(gold)}]}],
                                             add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        full = base.move_inputs_to_device(dict(full), device)
        # P4.1 (2026-07-24): batch=1, no padding -> drop the all-ones mask so transformers
        # takes the mask-None/is_causal path where FLASH is eligible (with a 4-D mask the
        # fused kernels never engage and MATH asks 45.6 GiB @N=64 — OOMs even a 140GB h200
        # in the ckpt-recompute backward). get_rope_index treats mask None as all-ones.
        full.pop("attention_mask", None)
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone(); labels[:, :bnd] = -100
        if fr_head is not None and pf is not None:
            sp = _itg(full["input_ids"][0].detach().cpu(), len(frames), processor=processor)
            st_fr["spans"] = sp if len(sp) == len(frames) else None
            st_fr["reps"] = None
        set_iso(full["input_ids"], len(frames))
        with sdpa_kernel(EFF_SDPA):     # P4.1: long-N training forward needs FLASH too
            loss = model(**full, labels=labels).loss
        clear_iso()
        if fr_head is not None and st_fr["reps"] is not None and pf is not None and st_fr["reps"].shape[0] == len(pf):
            lg = fr_head(st_fr["reps"]).squeeze(-1)
            loss = loss + args.frame_sup_weight * F.binary_cross_entropy_with_logits(
                lg, torch.tensor(pf, device=device, dtype=lg.dtype))
        st_fr["spans"] = None
        return loss

    @torch.inference_mode()
    def predict(frames, question):
        msgs = build_messages(frames, question, cot=args.cot)
        inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inp = base.move_inputs_to_device(dict(inp), device)
        set_iso(inp["input_ids"], len(frames))
        with sdpa_kernel(EFF_SDPA):
            out = model.generate(**inp, max_new_tokens=(256 if args.cot else 5), do_sample=False)
        clear_iso()
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

    if args.eval_only_adapter is not None:
        args.epochs = 0
        _trainable = []
        opt = None
    else:
        _trainable = [p for p in model.parameters() if p.requires_grad] + (list(fr_head.parameters()) if fr_head is not None else [])
        opt = torch.optim.AdamW(_trainable, lr=args.lr)
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
            frames, question, gold, nf, states = ex
            pf = None
            if fr_head is not None:
                try:
                    meta = json.loads((Path(dstr) / "metadata.json").read_text(encoding="utf-8"))
                    fl = frame_labels(args.task, states, meta)
                    pf = [float(x) for x in fl] if fl is not None else None
                except Exception:
                    pf = None
            try:
                loss = train_loss(frames, question, gold, pf)
            except Exception as exc:
                emit(f"  train skip {Path(dstr).name}: {exc}")
                torch.cuda.empty_cache()    # P4.1: OOM debris cascades into every later sample
                continue
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
    if args.eval_only_adapter is None:
        model.save_pretrained(str(run_dir / "adapter"))    # E-B fix: persist the LoRA adapter
        emit(f"adapter saved -> {run_dir / 'adapter'}")
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

    if args.eval_longn or args.eval_dirs_file:  # E-B/P4.1: long-N generate-and-parse eval
        from evaluations.helpers.utils import load_mmred_sample, iter_sample_dirs_shuffled
        lrows = ["root,n,accuracy,parse_fail,mae"]
        sources = [(r.strip(), None) for r in str(args.eval_longn).split(",") if r.strip()]
        sources += [(f, [Path(l.strip()) for l in open(f) if l.strip()])
                    for f in args.eval_dirs_file]
        for root, dirlist in sources:
            hits = pf = nn = mn = 0; mae = 0.0; per = {}
            dirs = (dirlist if dirlist is not None
                    else iter_sample_dirs_shuffled(Path(root), 0))
            for sd in dirs[: int(args.eval_longn_limit)]:
                try:
                    _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
                    gold = int(str(a0).strip())
                except Exception:
                    continue
                frames = [f.resize((392, 392)) for f in frames]
                pred = predict(frames, q0)
                nn += 1; hits += int(pred == gold)
                if pred is None:
                    pf += 1
                else:
                    mae += abs(pred - gold); mn += 1
                per.setdefault(gold, [0, 0]); per[gold][1] += 1; per[gold][0] += int(pred == gold)
            pc = " ".join(f"g{g}:{c}/{t2}" for g, (c, t2) in sorted(per.items()))
            emit(f"LONGN {root}: n={nn} acc={hits/max(nn,1):.4f} parse_fail={pf/max(nn,1):.3f} "
                 f"mae={mae/max(mn,1):.2f}  {pc}")
            lrows.append(f"{root},{nn},{hits/max(nn,1):.4f},{pf/max(nn,1):.3f},{mae/max(mn,1):.2f}")
        (run_dir / "longn_eval.csv").write_text("\n".join(lrows) + "\n", encoding="utf-8")

    # ---- paired before/after decomposition (does the fix work THROUGH the mechanism?) ----
    if args.decompose:
        from models.model import get_layers
        import experiments.glstm.decompose_reps as dr
        base_m = model.get_base_model() if hasattr(model, "get_base_model") else model
        tgt = get_layers(base_m)[args.decomp_layer]
        items = splits["test_iid"]
        model.eval()
        emit("decomposition: extracting reps with adapter ON ...")
        ca = dr.extract_reps(model, processor, tgt, items, args.decomp_layer, args.task, device, build_messages)
        emit("decomposition: extracting reps with adapter OFF (frozen base) ...")
        with model.disable_adapter():
            cb = dr.extract_reps(model, processor, tgt, items, args.decomp_layer, args.task, device, build_messages)
        mb, ma = dr.decompose(cb), dr.decompose(ca)
        (run_dir / "decomp_before.json").write_text(json.dumps(mb, indent=2))
        (run_dir / "decomp_after.json").write_text(json.dumps(ma, indent=2))
        emit(f"DECOMP @L{args.decomp_layer}  (before=adapter OFF | after=adapter ON):")
        for k in ("delta_over_mu", "perframe_snr", "corr_Sall_g", "S_all_linear_acc", "S_all_R2",
                  "sigmoid_then_sum_acc", "last_tok_acc"):
            emit(f"  {k:22s}: {mb.get(k)} -> {ma.get(k)}")
    log.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
