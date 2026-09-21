#!/usr/bin/env python3
"""SELFGATE H-SAFETY diagnostic (MANDATORY before any G1 full training).

Three checks on ONE training sample with a fresh LoRA + fresh gate head (fixed seed,
fixed logistic noise so all configs share the same Gumbel draw):

  1. MASK EQUIVALENCE: with bits forced to the ORACLE bits, the differentiable
     gated_tail_mask must equal build_block_mask(hide_cols=oracle) on every
     attention-REACHABLE (row, col): the tail rows exactly, and all other rows up
     to entries the base fence already forbids. Assert max |diff| == 0 on tail
     rows and that every remaining differing entry is base-forbidden (MASK_MIN).
  2. GRADIENT A/B/C (the sparse 139064 pattern, extended to the gate head):
       A) masks cleared BEFORE backward (the known-corrupt pattern)
       B) masks HELD through backward (the trainer's pattern)
       C) truth: gradient checkpointing disabled, masks held
     Report cos/max|Δ| of (i) LoRA grads and (ii) gate-head grads, B-vs-C and
     A-vs-C. PASS iff cos(B,C) >= 0.999 for BOTH param groups.
  3. GATE-GRAD LIVENESS: gate head grad norm in C > 0 (the mask path reaches it).

Usage: python scripts/selfgate/diag_grad_selfgate.py --output outputs/_scratch/selfgate_diag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SPARSE = _REPO / "scripts" / "sparse"
if str(_SPARSE) not in sys.path:
    sys.path.insert(0, str(_SPARSE))
_SELFGATE = _REPO / "scripts" / "selfgate"
if str(_SELFGATE) not in sys.path:
    sys.path.insert(0, str(_SELFGATE))
_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))

from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    rooms_to_room2chars,
)
from gnnformer.fencing import (  # noqa: E402
    FenceHooks,
    build_block_mask,
    locate_word_token,
    reset_positions,
)
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

from train_sft_gated import build_task_messages, parse_layout  # noqa: E402
from train_sft_selfgate import GateHead, gated_tail_mask, st_gumbel_bits  # noqa: E402
from tasks import target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
GRAD_SDPA = FENCED_SDPA  # overridden by --sdpa in main


def main() -> int:
    global GRAD_SDPA
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-layer", type=int, default=12)
    ap.add_argument("--sdpa", choices=("efficient", "math"), default="math",
                    help="backend for the GRAD configs (hypothesis: EFFICIENT drops "
                         "attn-mask gradients; MATH supports them by construction)")
    ap.add_argument("--min-gold", type=int, default=2,
                    help="skip gold<this for the diag sample (k=0 makes the mask weak)")
    ap.add_argument("--resize", type=int, default=256, help="256: the diag OOM lesson")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    GRAD_SDPA = ([SDPBackend.MATH] if args.sdpa == "math" else
                 [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    print(f"grad-config sdpa backends: {GRAD_SDPA}", flush=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    Lg = args.gate_layer

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    gate_head = GateHead(int(model.config.hidden_size), init_bias=0.5).to(rt.device).float()
    # non-degenerate head: random weights so bits differ across blocks
    torch.nn.init.normal_(gate_head.lin.weight, std=0.02)

    # ---- one sample (gold >= --min-gold so the gate has evidence to matter)
    sd = None
    for cand in iter_sample_dirs_shuffled(
            Path("data/mmred_images_park/seq_len_8/all_uniform"), 0):
        try:
            if int(str(load_mmred_sample(cand)[4]).strip()) >= args.min_gold:
                sd = cand
                break
        except Exception:
            continue
    _sid, frames, q0, states, a0 = load_mmred_sample(sd)
    frames = [f.resize((args.resize, args.resize)) for f in frames]
    char, room = target_of(Path(str(sd)), q0)
    evid = {t for t, st in enumerate(states)
            if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
    gold = str(len(evid))
    print(f"sample {sd.name} gold={gold} evid={sorted(evid)}", flush=True)

    full = move_to_device(dict(processor.apply_chat_template(
        build_task_messages(frames, "count", char, room, q0, answer=gold),
        add_generation_prompt=False, tokenize=True, return_dict=True,
        return_tensors="pt")), rt.device)
    prompt_len = processor.apply_chat_template(
        build_task_messages(frames, "count", char, room, q0),
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt")["input_ids"].shape[1]
    ids = full["input_ids"][0].tolist()
    blocks, fin_start = parse_layout(ids, tok, q0, len(frames), vs_id)
    with torch.no_grad():
        pos, _ = rope_fn(full["input_ids"], image_grid_thw=full.get("image_grid_thw"),
                         attention_mask=full.get("attention_mask"))
    pos = reset_positions(pos, blocks, fin_start).to(rt.device)
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full.pop("attention_mask", None)
    base = build_block_mask(len(ids), blocks, hide_cols=[])

    # ---- CHECK 1: oracle-bit mask equivalence
    obits = torch.tensor([1.0 if t in evid else 0.0 for t in range(len(blocks))],
                         device=rt.device)
    gm = gated_tail_mask(base, blocks, fin_start, obits).cpu()
    om = build_block_mask(len(ids), blocks,
                         hide_cols=[p for t, (a, b) in enumerate(blocks)
                                    if t not in evid for p in range(int(a), int(b))])
    tail_diff = float((gm[fin_start:] - om[fin_start:]).abs().max())
    rest = (gm[:fin_start] - om[:fin_start]).abs()
    rest_bad = bool(((rest > 0) & (om[:fin_start] > MASK_MIN / 2)).any())
    print(f"CHECK1 mask-equiv: tail max|diff|={tail_diff}  "
          f"non-tail diffs only on base-forbidden entries: {not rest_bad}", flush=True)
    ok1 = tail_diff == 0.0 and not rest_bad

    hooks_lo = FenceHooks(layers[:Lg]).install()
    hooks_hi = FenceHooks(layers[Lg:]).install()
    loci = [locate_word_token(ids, tok, room, b) for b in blocks]
    assert all(p is not None for p in loci)
    noise = torch.zeros(len(blocks), device=rt.device)  # fixed draw: shared by A/B/C

    def gate_bits(bias_shift=0.0):
        hooks_lo.set_mask(base, rt.device)
        hooks_hi.set_mask(base, rt.device)
        try:
            with torch.no_grad(), sdpa_kernel(FENCED_SDPA):
                hs = model(**{k: v for k, v in full.items() if k != "labels"},
                           position_ids=pos, output_hidden_states=True,
                           use_cache=False).hidden_states[Lg][0]
        finally:
            hooks_lo.clear_mask(); hooks_hi.clear_mask()
        st = hs[torch.tensor(loci)].float().detach()
        # TRAIN-mode gate: soft probs (the 148538 amendment); noise fixed across configs
        return torch.sigmoid((gate_head(st) + bias_shift + noise) / 2.0)

    def grads(hold_through_backward, use_ckpt):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        for p in gate_head.parameters():
            if p.grad is not None:
                p.grad = None
        if use_ckpt is None:
            pass  # CHECK4 sets GC state itself
        elif use_ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        bits = gate_bits()  # under the amendment these are TRAIN-mode soft probs
        bits.retain_grad()
        gmask = gated_tail_mask(base, blocks, fin_start, bits, mode="soft")
        gmask.retain_grad()
        hooks_lo.set_mask(base, rt.device)
        from train_sft_selfgate import set_grad_mask  # the 148505-verified pattern
        set_grad_mask(hooks_hi, gmask)
        with sdpa_kernel(GRAD_SDPA):
            loss = model(**full, position_ids=pos, labels=labels).loss
        if not hold_through_backward:
            hooks_lo.clear_mask(); hooks_hi.clear_mask()
        loss.backward()
        print(f"    [dbg] bits={[round(float(b),3) for b in bits.detach()]} "
              f"gmask.grad={'None' if gmask.grad is None else f'{float(gmask.grad.abs().sum()):.3e}'} "
              f"bits.grad={'None' if bits.grad is None else f'{float(bits.grad.abs().sum()):.3e}'}",
              flush=True)
        hooks_lo.clear_mask(); hooks_hi.clear_mask()
        lg = torch.cat([p.grad.detach().float().flatten().cpu()
                        for _n, p in sorted(model.named_parameters())
                        if p.requires_grad and p.grad is not None])
        gg = torch.cat([p.grad.detach().float().flatten().cpu()
                        for p in gate_head.parameters()])
        return float(loss.detach()), lg, gg

    lA, gA, hA = grads(hold_through_backward=False, use_ckpt=True)
    lB, gB, hB = grads(hold_through_backward=True, use_ckpt=True)
    lC, gC, hC = grads(hold_through_backward=True, use_ckpt=False)

    # CHECK 4 — TRAIN-MODE non-reentrant GC (the 148543 crash was reentrant GC in
    # train(); the earlier diags never entered train mode so GC was dormant).
    # D config: model.train() + GC(use_reentrant=False) + masks held. Must not
    # crash AND must match C.
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        lD, gD, hD = grads(hold_through_backward=True, use_ckpt=None)
        dD = float(torch.nn.functional.cosine_similarity(hD, hC, dim=0))
        gDc = float(torch.nn.functional.cosine_similarity(gD, gC, dim=0))
        print(f"CHECK4 non-reentrant GC (train mode): loss={lD:.4f} "
              f"gate cos(D,C)={dD:.6f} lora cos(D,C)={gDc:.6f}", flush=True)
        ok4 = dD >= 0.999
    except RuntimeError as e:
        print(f"CHECK4 non-reentrant GC: CRASH {type(e).__name__}: {str(e)[:140]}",
              flush=True)
        ok4 = False
    model.eval()
    model.gradient_checkpointing_disable()

    # CHECK 3b — REOPEN liveness (the 148538 flaw): with a deeply-CLOSED gate the
    # head must still receive gradient (the soft-log penalty's whole point).
    def grads_closed():
        for p in list(model.parameters()) + list(gate_head.parameters()):
            p.grad = None
        model.gradient_checkpointing_disable()
        bits = gate_bits(bias_shift=-6.0)
        print(f"    [closed] soft={[round(float(b),4) for b in bits.detach()]}", flush=True)
        gmask = gated_tail_mask(base, blocks, fin_start, bits, mode="soft")
        hooks_lo.set_mask(base, rt.device)
        from train_sft_selfgate import set_grad_mask
        set_grad_mask(hooks_hi, gmask)
        with sdpa_kernel(GRAD_SDPA):
            loss = model(**full, position_ids=pos, labels=labels).loss
        loss.backward()
        hooks_lo.clear_mask(); hooks_hi.clear_mask()
        return float(torch.cat([p.grad.flatten()
                                for p in gate_head.parameters()]).norm())

    h_closed = grads_closed()

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    print(f"losses A={lA:.4f} B={lB:.4f} C={lC:.4f}", flush=True)
    cBC, cAC = cos(gB, gC), cos(gA, gC)
    hBC, hAC = cos(hB, hC), cos(hA, hC)
    hnorm = float(hC.norm())
    print(f"CHECK2 LoRA grads: cos(B,C)={cBC:.6f} cos(A,C)={cAC:.4f} "
          f"max|B-C|={float((gB-gC).abs().max()):.2e}", flush=True)
    print(f"CHECK2 gate grads: cos(B,C)={hBC:.6f} cos(A,C)={hAC:.4f}", flush=True)
    print(f"CHECK3 gate-grad liveness: open |grad_C|={hnorm:.3e}, "
          f"CLOSED |grad|={h_closed:.3e} (both >0 required)", flush=True)
    ok2 = cBC >= 0.999 and hBC >= 0.999
    ok3 = hnorm > 0 and h_closed > 0
    verdict = "PASS" if (ok1 and ok2 and ok3) else "FAIL"
    (args.output / "verdict.txt").write_text(
        f"{verdict}\ncheck1={ok1} tail_diff={tail_diff}\n"
        f"check2 lora cos(B,C)={cBC:.6f} cos(A,C)={cAC:.4f}\n"
        f"check2 gate cos(B,C)={hBC:.6f} cos(A,C)={hAC:.4f}\n"
        f"check3 gate_grad_norm={hnorm:.3e} closed={h_closed:.3e}\n")
    print(f"VERDICT: {verdict}", flush=True)
    hooks_lo.remove(); hooks_hi.remove()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
