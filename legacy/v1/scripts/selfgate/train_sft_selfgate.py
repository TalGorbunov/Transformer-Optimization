#!/usr/bin/env python3
"""SELFGATE G1 — learned discrete gate (CoGNN-style), trained by the ANSWER LOSS ONLY.

Mechanism (brief §2 G1): gate head = linear probe (+bias) on each block's replica-slot
state at layer L_g (default 12, the S2 capture slot) -> logit per block ->
straight-through Gumbel-sigmoid bit (hard forward, soft backward; temperature annealed
--tau-start -> --tau-end over epochs) -> the bits build a DIFFERENTIABLE column penalty
on the tail rows (read segment), applied at decoder layers >= L_g via the two-hook
split. Gate head + LoRA train jointly on the answer loss. NO sparsity penalty, NO
evidence supervision: `oracle_evid` is computed ONLY in eval paths for gate-bit
accounting (H-SAFETY rule 2).

Two-forward training step:
  pass A (no_grad, plain fence): capture L_g replica-slot states -> gate logits
  (head input detached by construction; the head's gradient arrives via the mask).
  pass B (grad, layers < L_g plain fence / layers >= L_g gated mask): answer loss.
  The pass-B mask tensor CARRIES GRAD to the gate head and is HELD THROUGH BACKWARD
  (the sparse 139064 lesson; gradient checkpointing re-injects the same tensor).

Mask equivalence: with bits == oracle bits the differentiable tail-row penalty must
reproduce build_block_mask(hide_cols=oracle) exactly on every row the read uses —
asserted in scripts/selfgate/diag_grad_selfgate.py (run BEFORE any full training).

Builders/parse imported from scripts/sparse/train_sft_gated.py (single source; the
original stays the anchor). Eval contract of the produced adapter: learned gate +
(--nfree-prompt if trained with it).

Usage (G1a):
  python scripts/selfgate/train_sft_selfgate.py --nfree-prompt --epochs 10 \
      --exclude-dirs-file ... --eval-dirs-file ... --output outputs/selfgate/g1a
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SPARSE = _REPO / "scripts" / "sparse"
if str(_SPARSE) not in sys.path:
    sys.path.insert(0, str(_SPARSE))
_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))

from train_sft_gated import (  # noqa: E402  (anchors; imported, never copied)
    build_task_messages,
    parse_layout,
)
from tasks import (  # noqa: E402
    derive_gold,
    evidence_count,
    parse_answer,
    target_of,
    task_of_dirs_file,
)

from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
    read_dirs_file,
    rooms_to_room2chars,
)
from gnnformer.fencing import (  # noqa: E402
    FenceHooks,
    build_block_mask,
    frame_blocks,
    locate_word_token,
    reset_positions,
)
from gnnformer.metrics import format_gold_histogram  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
# Pass B (the grad pass) MUST run MATH: EFFICIENT errors on attn-mask-grad backward
# ("LSE is not correctly aligned", micro-diag 148505); MATH verified live (1.5e4).
GRAD_SDPA = [SDPBackend.MATH]
MAX_NEW = 4


def set_grad_mask(hooks, mask_2d):
    """Inject a GRAD-CARRYING mask: the exact 4-D view goes into the holder directly
    (the micro-diag-148505 verified pattern; set_mask's view/to chain was the config
    under which gate grads read zero — not re-litigated, the working pattern is
    adopted verbatim)."""
    hooks._holder["mask"] = mask_2d.view(1, 1, *mask_2d.shape)


class GateHead(torch.nn.Module):
    """One linear logit per block from its replica-slot state (fp32)."""

    def __init__(self, dim, init_bias=0.0):
        super().__init__()
        self.lin = torch.nn.Linear(dim, 1)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.constant_(self.lin.bias, init_bias)

    def forward(self, x):  # x: [B, dim] fp32
        return self.lin(x).squeeze(-1)  # [B]


def st_gumbel_bits(logits, tau, rng_noise=None, hard_eval=False):
    """Straight-through Gumbel-sigmoid. hard_eval: deterministic sigmoid>0.5, no grad."""
    if hard_eval:
        return (torch.sigmoid(logits) > 0.5).float()
    if rng_noise is None:
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        rng_noise = torch.log(u) - torch.log1p(-u)  # logistic noise
    soft = torch.sigmoid((logits + rng_noise) / tau)
    hard = (soft > 0.5).float()
    return hard + (soft - soft.detach())


def gated_tail_mask(base_mask, blocks, fin_start, gate, mode="hard"):
    """Gate the read: base fence + per-block column penalty on TAIL ROWS ONLY
    (rows >= fin_start — the read segment; other cross-block edges are already
    fenced, own-block rows stay open). Functional build (no in-place on grad path).

    mode="hard" (EVAL): gate = hard 0/1 bits; penalty (1-bit)*MASK_MIN — the exact
      oracle-mask semantics (equals build_block_mask(hide_cols) on reachable rows).
    mode="soft" (TRAIN): gate = soft probs in (0,1); penalty log(clamp(p, 1e-6)) —
      MULTIPLICATIVE soft gating (attention mass scaled by p). This is the diag-148538
      amendment: a hard-closed block has softmax weight EXACTLY 0, and mask-entry
      gradients are proportional to that weight, so hard-closed gates can never
      receive a reopen signal (bits.grad measured exactly 0). The soft-log penalty
      keeps ~1e-6 mass on closed blocks so both gradient directions stay alive;
      sigmoid saturation anneals it toward hard."""
    seq = base_mask.shape[-1]
    segs, cursor = [], 0
    for t, (a, b) in enumerate(blocks):
        a, b = int(a), int(b)
        if a > cursor:
            segs.append(torch.ones(a - cursor, device=gate.device))
        segs.append(gate[t].expand(b - a))
        cursor = b
    if cursor < seq:
        segs.append(torch.ones(seq - cursor, device=gate.device))
    cols = torch.cat(segs)                           # [S], grad -> gate
    if mode == "soft":
        pen = torch.log(cols.clamp(min=1e-6))
    else:
        pen = (1.0 - cols) * MASK_MIN
    base = base_mask.to(gate.device)
    return torch.cat([base[:fin_start],
                      base[fin_start:] + pen.unsqueeze(0)], dim=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root",
                    default="data/mmred_images_park/seq_len_8/all_uniform,"
                            "data/mmred_longN_park/seq_len_16/all_uniform")
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-cap", type=int, default=60)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gate-lr", type=float, default=1e-3)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--gate-layer", type=int, default=12, help="L_g: head input + mask split")
    ap.add_argument("--tau-start", type=float, default=5.0)
    ap.add_argument("--tau-end", type=float, default=0.5)
    ap.add_argument("--gate-init-bias", type=float, default=0.0,
                    help="G1b retry knob: +2 = open init")
    ap.add_argument("--l1-bits", type=float, default=0.0,
                    help="G1b knob: lambda * mean(bits) penalty (0 = G1a, none)")
    ap.add_argument("--tasks", default="count")
    ap.add_argument("--nfree-prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-dirs-file", action="append", default=[])
    ap.add_argument("--eval-longn-limit", type=int, default=150)
    ap.add_argument("--exclude-dirs-file", action="append", default=[])
    ap.add_argument("--eval-only-adapter", type=Path, default=None,
                    help="adapter dir containing gate_head.pt; skip training")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/selfgate/g1a"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.nfree_prompt and args.tasks != "count":
        raise SystemExit("--nfree-prompt is count-only")

    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_selfgate"
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
    hidden = int(model.config.hidden_size)
    Lg = args.gate_layer

    gate_head = GateHead(hidden, init_bias=args.gate_init_bias).to(rt.device).float()
    if args.eval_only_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.eval_only_adapter),
                                          is_trainable=False)
        gh_state = torch.load(Path(args.eval_only_adapter) / "gate_head.pt",
                              map_location=rt.device)
        gate_head.load_state_dict(gh_state)
        gate_head.eval()
        emit(f"eval-only: adapter+gate_head loaded from {args.eval_only_adapter}")
        args.epochs = 0
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        n_layers = len(layers)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          layers_to_transform=list(range(Lg, n_layers)),
                          layers_pattern="layers")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        # AMENDMENT 2 (jobs 148543/148554/148563, all logged): GC (both flavors) is
        # incompatible with a hook-injected grad-carrying mask, and no-GC with
        # all-layer LoRA OOMs at N=16 even on 96 GB. Scope: LoRA on decoder layers
        # >= L_g ONLY (the method's own L*=12 design; vision untouched); layers
        # below run grad-free -> no stored graph there (~58 GB @N=16). Bonus: the
        # gate head reads a STATIONARY L12 distribution (frozen verdict-former).
        model.gradient_checkpointing_disable()
        model.disable_input_require_grads()
        emit(f"[selfgate] GC OFF + input-grads OFF + LoRA restricted to layers "
             f">= {Lg} (of {n_layers}) — amendment 2, see STATE 2026-09-14")
    emit(f"[selfgate] L_g={Lg} tau {args.tau_start}->{args.tau_end} "
         f"gate-lr {args.gate_lr} init-bias {args.gate_init_bias} l1 {args.l1_bits} "
         f"nfree={args.nfree_prompt} — ANSWER LOSS ONLY (oracle_evid never in loss)")

    # ---- data (gated trainer's conventions)
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
    samples = [(sd, gold, task_list[i % len(task_list)])
               for i, (sd, gold) in enumerate(samples)]
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

    # two-hook split at L_g: below = plain fence, above = gated (possibly grad) mask
    hooks_lo = FenceHooks(layers[:Lg]).install()
    hooks_hi = FenceHooks(layers[Lg:]).install()

    def clear_masks():
        hooks_lo.clear_mask()
        hooks_hi.clear_mask()

    def load_frames(sd):
        _sid, frames, q0, states, _a0 = load_mmred_sample(sd)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        tr = target_of(Path(str(sd)), q0)
        if tr is None:
            raise ValueError("no target character/room")
        return frames, q0, tr[0], tr[1], states

    def oracle_evid_eval(states, char, room):
        """EVAL-ONLY gate-bit reference. Never enters any loss."""
        return {t for t, st in enumerate(states)
                if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}

    def layout_of(inp, q0, nf):
        ids = inp["input_ids"][0].tolist()
        parsed = parse_layout(ids, tok, q0, nf, vs_id)
        if parsed is None:
            return None
        blocks, fin_start = parsed
        with torch.no_grad():
            pos, _ = rope_fn(inp["input_ids"], image_grid_thw=inp.get("image_grid_thw"),
                             attention_mask=inp.get("attention_mask"))
        return ids, blocks, fin_start, reset_positions(pos, blocks, fin_start)

    def replica_states(inp, ids, blocks, fin_start, pos, room):
        """Pass A: plain-fence forward (no grad), L_g replica-slot states [T, H]."""
        loci = [locate_word_token(ids, tok, room, b) for b in blocks]
        if any(p is None for p in loci):
            return None
        base = build_block_mask(len(ids), blocks, hide_cols=[])
        cur = {k: v for k, v in inp.items() if k != "attention_mask"}
        hooks_lo.set_mask(base, rt.device)
        hooks_hi.set_mask(base, rt.device)
        try:
            with torch.no_grad(), sdpa_kernel(FENCED_SDPA):
                hs = model(**cur, position_ids=pos.to(rt.device),
                           output_hidden_states=True, use_cache=False
                           ).hidden_states[Lg][0]
        finally:
            clear_masks()
        return hs[torch.tensor(loci)].float().detach()

    def train_loss(frames, task, char, room, q0, gold, tau):
        full = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0, answer=gold,
                                nfree=args.nfree_prompt),
            add_generation_prompt=False, tokenize=True, return_dict=True,
            return_tensors="pt")
        prompt = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0, nfree=args.nfree_prompt),
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        full = move_to_device(dict(full), rt.device)
        lay = layout_of(full, q0, len(frames))
        if lay is None:
            raise ValueError("layout parse failed")
        ids, blocks, fin_start, pos = lay
        st = replica_states(full, ids, blocks, fin_start, pos, room)
        if st is None:
            raise ValueError("replica locus miss")
        logits = gate_head(st)
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        soft = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / tau)
        bits = (soft > 0.5).float()  # reported only; the TRAIN mask is soft
        base = build_block_mask(len(ids), blocks, hide_cols=[])
        gmask = gated_tail_mask(base, blocks, fin_start, soft, mode="soft")
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :bnd] = -100
        full.pop("attention_mask", None)
        # masks HELD THROUGH BACKWARD (sparse 139064); caller clears after step
        hooks_lo.set_mask(base, rt.device)
        set_grad_mask(hooks_hi, gmask)
        with sdpa_kernel(GRAD_SDPA):
            loss = model(**full, position_ids=pos.to(rt.device), labels=labels).loss
        if args.l1_bits > 0:
            loss = loss + args.l1_bits * soft.mean()
        return loss, bits.detach()

    @torch.inference_mode()
    def infer_bits(inp, ids, blocks, fin_start, pos, room):
        st = replica_states(inp, ids, blocks, fin_start, pos, room)
        if st is None:
            return None
        return st_gumbel_bits(gate_head(st), 1.0, hard_eval=True)

    def predict(frames, task, char, room, q0):
        """-> (answer or None, hard bits or None). Learned gate, deterministic."""
        inp = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0, nfree=args.nfree_prompt),
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        inp = move_to_device(dict(inp), rt.device)
        lay = layout_of(inp, q0, len(frames))
        if lay is None:
            return None, None
        ids, blocks, fin_start, pos = lay
        bits = infer_bits(inp, ids, blocks, fin_start, pos, room)
        if bits is None:
            return None, None
        ids_t = inp["input_ids"]
        grid = inp.get("image_grid_thw")
        out_toks = []
        with torch.inference_mode():
            for _ in range(MAX_NEW):
                cur_ids = ids_t[0].tolist()
                parsed = parse_layout(cur_ids, tok, q0, len(frames), vs_id)
                if parsed is None:
                    return None, bits
                blocks2, fin2 = parsed
                base2 = build_block_mask(len(cur_ids), blocks2, hide_cols=[])
                gmask2 = gated_tail_mask(base2, blocks2, fin2, bits)
                with torch.no_grad():
                    pos2, _ = rope_fn(ids_t, image_grid_thw=grid,
                                      attention_mask=torch.ones_like(ids_t))
                pos2 = reset_positions(pos2, blocks2, fin2)
                cur = {"input_ids": ids_t, "image_grid_thw": grid,
                       "pixel_values": inp.get("pixel_values")}
                hooks_lo.set_mask(base2, rt.device)
                hooks_hi.set_mask(gmask2.detach(), rt.device)
                try:
                    with sdpa_kernel(FENCED_SDPA):
                        lg = model(**cur, position_ids=pos2.to(rt.device)).logits
                finally:
                    clear_masks()
                nxt = int(lg[0, -1].argmax())
                if nxt == tok.eos_token_id:
                    break
                out_toks.append(nxt)
                if "\n" in tok.decode(out_toks):
                    break
                ids_t = torch.cat([ids_t, torch.tensor([[nxt]], device=ids_t.device)], 1)
        return parse_answer(task, tok.decode(out_toks, skip_special_tokens=True)), bits

    def evaluate(idx_list, cap=None):
        model.eval(); gate_head.eval()
        use = idx_list if cap is None else idx_list[:cap]
        n = ok = 0
        for i in use:
            sd, _g, task = samples[i]
            try:
                frames, q0, char, room, states = load_frames(sd)
                gold = derive_gold(task, states, char, room, len(frames))
                pred, _bits = predict(frames, task, char, room, q0)
            except Exception:
                continue
            n += 1
            ok += int(pred == gold)
        return ok / max(1, n), n

    if args.epochs:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": params, "lr": args.lr},
                                 {"params": gate_head.parameters(), "lr": args.gate_lr}])
        best_val, best_state, best_epoch = -1.0, None, -1
        vrows = ["epoch,tau,val_acc,n,mean_bit"]
        rng = np.random.default_rng(args.seed)
        for epoch in range(args.epochs):
            tau = args.tau_start * (args.tau_end / args.tau_start) ** (
                epoch / max(1, args.epochs - 1))
            model.train(); gate_head.train()
            rng.shuffle(tr_idx)
            opt.zero_grad()
            run_loss, seen, bit_sum, bit_n = 0.0, 0, 0.0, 0
            for step, i in enumerate(tr_idx):
                sd, _g, task = samples[i]
                try:
                    frames, q0, char, room, states = load_frames(sd)
                    gold = derive_gold(task, states, char, room, len(frames))
                    loss, bits = train_loss(frames, task, char, room, q0, gold, tau)
                except Exception as exc:
                    emit(f"  train skip {Path(str(sd)).name}: {exc}")
                    clear_masks()
                    torch.cuda.empty_cache()
                    continue
                (loss / args.grad_accum).backward()
                clear_masks()  # AFTER backward (the 139064 pattern)
                run_loss += float(loss.detach())
                seen += 1
                bit_sum += float(bits.mean()); bit_n += 1
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        params + list(gate_head.parameters()), 1.0)
                    opt.step()
                    opt.zero_grad()
            vacc, vn = evaluate(list(va_idx), cap=args.val_cap)
            mb = bit_sum / max(1, bit_n)
            vrows.append(f"{epoch},{tau:.3f},{vacc:.4f},{vn},{mb:.4f}")
            emit(f"epoch {epoch}: tau={tau:.2f} train_loss={run_loss/max(1,seen):.3f} "
                 f"val_acc={vacc:.3f} mean_bit={mb:.3f}")
            if vacc > best_val:
                best_val, best_epoch = vacc, epoch
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()
                                            if "lora" in k.lower()})
                model.save_pretrained(str(run_dir / "adapter"))
                torch.save(gate_head.state_dict(), run_dir / "adapter" / "gate_head.pt")
                emit(f"  adapter+gate saved @ep{epoch}")
            if args.patience and (epoch - best_epoch) >= args.patience:
                emit(f"early stop (best={best_epoch})")
                break
        (run_dir / "val_by_epoch.csv").write_text("\n".join(vrows) + "\n")
        emit(f"best_epoch={best_epoch} val_acc={best_val:.3f}")
        if best_state is not None:
            model.load_state_dict({k: v.to(rt.device) for k, v in best_state.items()},
                                  strict=False)
            gate_head.load_state_dict(torch.load(run_dir / "adapter" / "gate_head.pt",
                                                 map_location=rt.device))

    acc, n = evaluate(list(te_idx))
    emit(f"TEST_IID: acc={acc:.4f} n={n}")
    (run_dir / "summary.csv").write_text(f"split,n,accuracy\ntest_iid,{n},{acc:.4f}\n")

    if args.eval_dirs_file:
        lrows = ["source,task,n,accuracy,parse_fail,gate_bit_acc,gate_fn,gate_fp"]
        prows = ["source,sample_dir,task,n_frames,k,gold,pred,bits,gate_fn,gate_fp"]
        for f in args.eval_dirs_file:
            e_task = task_of_dirs_file(f)
            dirs = read_dirs_file(Path(f))
            hits = pf = nn = 0
            bit_ok = bit_tot = fn_tot = fp_tot = 0
            per: dict = {}
            for sd in dirs[: args.eval_longn_limit]:
                try:
                    frames, q0, char, room, states = load_frames(sd)
                    k = evidence_count(states, char, room)
                    gold = derive_gold(e_task, states, char, room, len(frames))
                    pred, bits = predict(frames, e_task, char, room, q0)
                except Exception:
                    continue
                nn += 1
                hits += int(pred == gold)
                pf += int(pred is None)
                evid = oracle_evid_eval(states, char, room)  # EVAL-ONLY reference
                g_fn = g_fp = 0
                bstr = ""
                if bits is not None:
                    bl = [int(b) for b in bits.tolist()]
                    bstr = "".join(str(b) for b in bl)
                    for t, b in enumerate(bl):
                        want = int(t in evid)
                        bit_ok += int(b == want)
                        bit_tot += 1
                        g_fn += int(want == 1 and b == 0)
                        g_fp += int(want == 0 and b == 1)
                    fn_tot += g_fn
                    fp_tot += g_fp
                pg = per.setdefault(k, [0, 0])
                pg[1] += 1
                pg[0] += int(pred == gold)
                prows.append(f"{f},{sd},{e_task},{len(frames)},{k},{gold},"
                             f"{'' if pred is None else pred},{bstr},{g_fn},{g_fp}")
            pk = " ".join(f"k{k_}:{c}/{t2}" for k_, (c, t2) in sorted(per.items()))
            emit(f"LONGN {f} task={e_task}: n={nn} acc={hits/max(nn,1):.4f} "
                 f"pf={pf/max(nn,1):.3f} gate_bit_acc={bit_ok/max(bit_tot,1):.4f} "
                 f"fn={fn_tot} fp={fp_tot}\n  per-k {pk}")
            lrows.append(f"{f},{e_task},{nn},{hits/max(nn,1):.4f},{pf/max(nn,1):.3f},"
                         f"{bit_ok/max(bit_tot,1):.4f},{fn_tot},{fp_tot}")
        (run_dir / "longn_eval.csv").write_text("\n".join(lrows) + "\n")
        (run_dir / "longn_predictions.csv").write_text("\n".join(prows) + "\n")
    hooks_lo.remove()
    hooks_hi.remove()
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
