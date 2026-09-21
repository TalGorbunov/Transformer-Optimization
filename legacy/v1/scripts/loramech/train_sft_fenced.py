#!/usr/bin/env python3
"""LORAMECH P1 — fenced-SFT: plain LoRA trained under the architectural fence.

Layout (Tal's spec, 2026-08-27): N independent [frame_i + question-replica_i] blocks
behind the canonical block-diagonal fence with per-block M-RoPE posreset, then ONE
final counting prompt (build_count_prompt, ends "Answer: ") that sees all blocks.
NO Q-first: within a block the replica FOLLOWS its frame, so the per-frame
question-conditioned verdict is computed in the replica positions; the fence makes
the blocks independent (the ARMOR-A flat-supply channel); only the tally read
remains at the answer positions. This is the causal test of the P0 explanation:
conditioning restored architecturally — does learnability come back?

Recipe mirrors scripts/train_sft_baseline.py (r=8 alpha=32 q/k/v/o+MLP, lr 2e-4,
accum 8, 5 ep, answer-token LM loss, seeded split, --exclude-dirs-file discipline).
Fencing primitives imported READ-ONLY from gnnformer.fencing (the one canonical
mask: build_block_mask with hide_cols=[] — replicas VISIBLE to the tail, they carry
the verdicts). 4-D mask => FLASH ineligible: sdpa EFFICIENT/MATH only, so train at
N<=16 on 48 GB. Decode = manual greedy re-forward (no KV cache; mask+posreset are
rebuilt per step) — exact, cache-free, ~4 tokens.

Usage:
  python scripts/loramech/train_sft_fenced.py --data_root <roots> --epochs 5 \
      --exclude-dirs-file <exam.txt> --eval-dirs-file <exam.txt> --output outputs/loramech/p1
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import (  # noqa: E402
    build_count_prompt,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
    read_dirs_file,
)
from gnnformer.fencing import (  # noqa: E402
    FenceHooks,
    build_block_mask,
    find_subseq,
    frame_blocks,
    reset_positions,
)
from gnnformer.metrics import format_gold_histogram  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import (  # noqa: E402  (REDUX label module — count path stays byte-identical)
    build_prompt as task_prompt,
    derive_gold,
    evidence_count,
    parse_answer,
    replica_text,
    target_of,
    task_of_dirs_file,
)

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]  # 4-D mask: no FLASH
INTEGER_RE = re.compile(r"[+-]?\d+")
MAX_NEW = 4


def build_fenced_messages(frames, question, answer=None):
    """[frame_0, q, frame_1, q, ..., frame_{N-1}, q, count_prompt] (+ assistant answer)."""
    content = []
    for im in frames:
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": question})
    content.append({"type": "text", "text": build_count_prompt(question, len(frames))})
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
    return msgs


def build_task_messages(frames, task, char, room, q0, answer=None):
    """REDUX task-aware fenced layout. count == build_fenced_messages byte-identical;
    exists/majority swap the replica sentence + final prompt (same opener line, so
    parse_layout is unchanged)."""
    if task == "count":
        return build_fenced_messages(frames, q0, answer=answer)
    nf = len(frames)
    content = []
    for im in frames:
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": replica_text(task, char, room, nf)})
    content.append({"type": "text", "text": task_prompt(task, char, room, nf)})
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
    return msgs


def parse_layout(ids, tok, q0, nf, vs_id):
    """-> (blocks, fin_start) or None.

    Blocks need only the N vision starts + where the final count prompt begins.
    The question text is NOT matched (its N+1 copies tokenize differently by context —
    the smoke-137728 lesson); build_count_prompt opens with the unique fragment
    "You will be shown", which q0 ("How many steps...") can never contain.
    """
    vstarts = [p for p, t in enumerate(ids) if t == vs_id]
    if len(vstarts) != nf:
        return None
    fin_start = None
    for pre in ("", " ", "\n"):
        needle = tok(pre + "You will be shown", add_special_tokens=False).input_ids
        occ = find_subseq(ids, needle)
        if len(occ) == 1:
            fin_start = occ[0]
            break
    # each block must hold its frame AND a non-empty replica before the final prompt
    if fin_start is None or fin_start <= vstarts[-1] + 1:
        return None
    return frame_blocks(vstarts, fin_start), fin_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-cap", type=int, default=60)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-dirs-file", action="append", default=[])
    ap.add_argument("--eval-longn-limit", type=int, default=150)
    ap.add_argument("--exclude-dirs-file", action="append", default=[])
    ap.add_argument("--eval-only-adapter", type=Path, default=None)
    ap.add_argument("--tasks", default="count",
                    help="REDUX mixture: comma-set of {count,exists,majority}. Training "
                         "samples get tasks round-robin (balanced); eval dirs-files "
                         "infer their task from the filename. Default count = the "
                         "P1b behavior byte-identical.")
    ap.add_argument("--attn-logn-sref", type=int, default=0,
                    help="L5 (Chiang–Cholak): >0 enables log-N attention-logit scaling "
                         "— scaling = head_dim^-0.5 * ln(S)/ln(S_ref) on the LM decoder "
                         "attention modules, applied at DECODE and DURING TRAINING "
                         "(P3: the readout calibrates to the compensated geometry); "
                         "S_ref = this token count (training-window seq len). An "
                         "adapter trained with this flag must be evaluated with it.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/loramech/p1_fenced"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_fenced"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    # L5 machinery — attn modules captured PRE-wrap (the 137800 PeftModel lesson)
    attn_mods = [ly.self_attn for ly in layers]
    base_scaling = float(attn_mods[0].scaling)

    def set_logn_scaling(seq_len):
        import math

        s = base_scaling * math.log(max(seq_len, 2)) / math.log(args.attn_logn_sref)
        for m_ in attn_mods:
            m_.scaling = s
    if args.eval_only_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.eval_only_adapter), is_trainable=False)
        emit(f"eval-only: adapter loaded from {args.eval_only_adapter}")
        args.epochs = 0
    elif args.epochs == 0:
        emit("frozen eval: no adapter, no LoRA (REDUX C1 p1fence_frozen mode)")
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    emit("[template] FENCED frames-first: N x [frame+q] blocks + posreset + count prompt")

    # ---- data (baseline trainer's conventions: validity gate, exclusion, seeded split)
    excluded = set()
    for f in args.exclude_dirs_file:
        excluded.update(str(Path(p).resolve()) for p in read_dirs_file(Path(f)))
    if excluded:
        emit(f"[exclude] {len(excluded)} dirs across {len(args.exclude_dirs_file)} file(s)")
    samples, n_skip, n_excl = [], 0, 0
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
            if excluded and str(Path(sd).resolve()) in excluded:
                n_excl += 1
                continue
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
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    assert all(t in ("count", "exists", "majority") for t in task_list), task_list
    # round-robin task assignment over the collection order = balanced & deterministic
    samples = [(sd, gold, task_list[i % len(task_list)])
               for i, (sd, gold) in enumerate(samples)]
    emit(f"[tasks] {task_list}; per-task "
         + str({t: sum(1 for s in samples if s[2] == t) for t in task_list}))
    emit(f"[data] {len(samples)} samples (skip {n_skip}, excluded {n_excl}); gold-hist "
         + format_gold_histogram(g for _sd, g, _t in samples))
    split_rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(len(samples))
    n_tr = int(len(samples) * args.train_frac)
    n_va = int(len(samples) * args.val_frac)
    tr_idx, va_idx, te_idx = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    (run_dir / "train_dirs.txt").write_text("\n".join(str(samples[i][0]) for i in tr_idx) + "\n")
    (run_dir / "eval_dirs.txt").write_text("\n".join(str(samples[i][0]) for i in te_idx) + "\n")
    emit(f"[split] train={len(tr_idx)} val={len(va_idx)} test_iid={len(te_idx)}")

    hooks = FenceHooks(layers).install()
    n_layout_skip = 0

    def load_frames(sd):
        """-> (frames, q0, char, room, states). Raises on missing target."""
        _sid, frames, q0, states, _a0 = load_mmred_sample(sd)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        tr = target_of(Path(str(sd)), q0)
        if tr is None:
            raise ValueError("no target character/room")
        return frames, q0, tr[0], tr[1], states

    def fenced_setup(inp, q0, nf):
        """-> (mask_2d, pos) for this exact token sequence, or None on layout-parse miss."""
        ids = inp["input_ids"][0].tolist()
        parsed = parse_layout(ids, tok, q0, nf, vs_id)
        if parsed is None:
            return None
        blocks, fin_start = parsed
        mask = build_block_mask(len(ids), blocks, hide_cols=[])
        with torch.no_grad():
            base_pos, _ = rope_fn(inp["input_ids"], image_grid_thw=inp.get("image_grid_thw"),
                                  attention_mask=inp.get("attention_mask"))
        return mask, reset_positions(base_pos, blocks, fin_start)

    def train_loss(frames, task, char, room, q0, gold):
        full = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0, answer=gold),
            add_generation_prompt=False, tokenize=True, return_dict=True,
            return_tensors="pt")
        prompt = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0),
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        full = move_to_device(dict(full), rt.device)
        setup = fenced_setup(full, q0, len(frames))
        if setup is None:
            raise ValueError("layout parse failed")
        mask, pos = setup
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :bnd] = -100
        full.pop("attention_mask", None)
        if args.attn_logn_sref > 0:  # P3: scaling active in the TRAINING forward too
            set_logn_scaling(int(full["input_ids"].shape[1]))
        hooks.set_mask(mask, rt.device)
        try:
            with sdpa_kernel(FENCED_SDPA):
                return model(**full, position_ids=pos.to(rt.device), labels=labels).loss
        finally:
            hooks.clear_mask()

    @torch.inference_mode()
    def predict(frames, task, char, room, q0):
        inp = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0),
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        inp = move_to_device(dict(inp), rt.device)
        ids = inp["input_ids"]
        grid = inp.get("image_grid_thw")
        out_toks = []
        if args.attn_logn_sref > 0:
            set_logn_scaling(int(ids.shape[1]))
        for _ in range(MAX_NEW):
            cur = {"input_ids": ids, "image_grid_thw": grid,
                   "pixel_values": inp.get("pixel_values"),
                   "attention_mask": torch.ones_like(ids)}
            setup = fenced_setup(cur, q0, len(frames))
            if setup is None:
                return None
            mask, pos = setup
            cur.pop("attention_mask", None)
            hooks.set_mask(mask, rt.device)
            try:
                with sdpa_kernel(FENCED_SDPA):
                    logits = model(**cur, position_ids=pos.to(rt.device)).logits
            finally:
                hooks.clear_mask()
            nxt = int(logits[0, -1].argmax())
            if nxt == tok.eos_token_id:
                break
            out_toks.append(nxt)
            txt = tok.decode(out_toks)
            if "\n" in txt:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return parse_answer(task, tok.decode(out_toks, skip_special_tokens=True))

    def evaluate(idx_list, cap=None):
        model.eval()
        use = idx_list if cap is None else idx_list[:cap]
        n = ok = 0
        for i in use:
            sd, _g, task = samples[i]
            try:
                frames, q0, char, room, states = load_frames(sd)
                gold = derive_gold(task, states, char, room, len(frames))
                pred = predict(frames, task, char, room, q0)
            except Exception:
                continue
            n += 1
            ok += int(pred == gold)
        return ok / max(1, n), n

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
                sd, _g, task = samples[i]
                try:
                    frames, q0, char, room, states = load_frames(sd)
                    gold = derive_gold(task, states, char, room, len(frames))
                    loss = train_loss(frames, task, char, room, q0, gold)
                except Exception as exc:
                    emit(f"  train skip {Path(str(sd)).name}: {exc}")
                    n_layout_skip += 1
                    torch.cuda.empty_cache()
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
            emit(f"epoch {epoch}: train_loss={run_loss/max(1,seen):.3f} val_acc={vacc:.3f} "
                 f"(skips so far {n_layout_skip})")
            if vacc > best_val:
                best_val, best_epoch = vacc, epoch
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()
                                            if "lora" in k.lower()})
                model.save_pretrained(str(run_dir / "adapter"))
                emit(f"  adapter saved @ep{epoch} -> {run_dir / 'adapter'}")
            if args.patience and (epoch - best_epoch) >= args.patience:
                emit(f"early stop (best={best_epoch})")
                break
        (run_dir / "val_by_epoch.csv").write_text("\n".join(vrows) + "\n")
        emit(f"best_epoch={best_epoch} val_acc={best_val:.3f}")
        if best_state is not None:
            model.load_state_dict({k: v.to(rt.device) for k, v in best_state.items()},
                                  strict=False)

    acc, n = evaluate(list(te_idx))
    emit(f"TEST_IID: acc={acc:.4f} n={n}")
    (run_dir / "summary.csv").write_text(f"split,n,accuracy\ntest_iid,{n},{acc:.4f}\n")

    if args.eval_dirs_file:
        lrows = ["source,task,n,accuracy,parse_fail,mae"]
        prows = ["source,sample_dir,task,n_frames,k,gold,pred"]
        for f in args.eval_dirs_file:
            e_task = task_of_dirs_file(f)
            dirs = read_dirs_file(Path(f))
            hits = pf = nn = mn = 0
            mae = 0.0
            per: dict = {}
            per_k: dict = {}
            for sd in dirs[: args.eval_longn_limit]:
                try:
                    frames, q0, char, room, states = load_frames(sd)
                    k = evidence_count(states, char, room)
                    gold = derive_gold(e_task, states, char, room, len(frames))
                    pred = predict(frames, e_task, char, room, q0)
                except Exception:
                    continue
                nn += 1
                hits += int(pred == gold)
                if pred is None:
                    pf += 1
                elif e_task == "count":
                    mae += abs(int(pred) - int(gold))
                    mn += 1
                pg = per.setdefault(gold, [0, 0])
                pg[1] += 1
                pg[0] += int(pred == gold)
                kk = per_k.setdefault(k, [0, 0])
                kk[1] += 1
                kk[0] += int(pred == gold)
                prows.append(f"{f},{sd},{e_task},{len(frames)},{k},{gold},"
                             f"{'' if pred is None else pred}")
            pc = " ".join(f"g{g}:{c}/{t2}" for g, (c, t2) in sorted(per.items()))
            pk = " ".join(f"k{k_}:{c}/{t2}" for k_, (c, t2) in sorted(per_k.items()))
            emit(f"LONGN {f} task={e_task}: n={nn} acc={hits/max(nn,1):.4f} "
                 f"parse_fail={pf/max(nn,1):.3f} mae={mae/max(mn,1):.2f}\n"
                 f"  per-gold {pc}\n  per-k {pk}")
            lrows.append(f"{f},{e_task},{nn},{hits/max(nn,1):.4f},{pf/max(nn,1):.3f},"
                         f"{mae/max(mn,1):.2f}")
        (run_dir / "longn_eval.csv").write_text("\n".join(lrows) + "\n")
        (run_dir / "longn_predictions.csv").write_text("\n".join(prows) + "\n")
    hooks.remove()
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
