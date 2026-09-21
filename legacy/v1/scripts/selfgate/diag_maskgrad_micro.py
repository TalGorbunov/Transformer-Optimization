#!/usr/bin/env python3
"""SELFGATE micro-diag: WHERE does the attn-mask gradient die?

Stage 1 (torch-only): F.scaled_dot_product_attention with attn_mask.requires_grad,
per backend (MATH/EFFICIENT) x dtype (fp32/bf16): is mask.grad non-None/non-zero?

Stage 2 (model): one fenced forward through the full model with the gate-built mask;
a .register_hook on the injected (post-view) tensor reports whether backward ever
reaches it, and the loss's dependence on the mask is verified by finite difference
(loss with one evidence block hidden vs open — must differ).

Usage: python scripts/selfgate/diag_maskgrad_micro.py --output outputs/_scratch/selfgate_micro
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "scripts" / "sparse"), str(_REPO / "scripts" / "selfgate"),
          str(_REPO / "scripts" / "redux")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, rooms_to_room2chars  # noqa: E402
from gnnformer.fencing import FenceHooks, build_block_mask, locate_word_token, reset_positions  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402
from train_sft_gated import build_task_messages, parse_layout  # noqa: E402
from train_sft_selfgate import GateHead, gated_tail_mask, st_gumbel_bits  # noqa: E402
from tasks import target_of  # noqa: E402


def stage1():
    print("=== STAGE 1: raw sdpa attn-mask grad ===", flush=True)
    dev = "cuda"
    for backend, bname in ((SDPBackend.MATH, "MATH"),
                           (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT")):
        for dt in (torch.float32, torch.bfloat16):
            q = torch.randn(1, 2, 6, 8, device=dev, dtype=dt)
            k = torch.randn(1, 2, 6, 8, device=dev, dtype=dt)
            v = torch.randn(1, 2, 6, 8, device=dev, dtype=dt)
            m = torch.zeros(1, 1, 6, 6, device=dev, dtype=dt, requires_grad=True)
            try:
                with sdpa_kernel([backend]):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
                out.float().sum().backward()
                g = m.grad
                print(f"  {bname} {dt}: grad "
                      f"{'None' if g is None else f'norm={float(g.float().norm()):.3e}'}",
                      flush=True)
            except Exception as e:
                print(f"  {bname} {dt}: EXC {type(e).__name__}: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-layer", type=int, default=12)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    stage1()

    print("=== STAGE 2: model-path mask grad ===", flush=True)
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
        target_modules=["q_proj", "v_proj"]))
    model.gradient_checkpointing_disable()
    gate_head = GateHead(int(model.config.hidden_size), init_bias=0.0).to(rt.device).float()
    torch.nn.init.normal_(gate_head.lin.weight, std=0.02)

    sd = None
    for cand in iter_sample_dirs_shuffled(
            Path("data/mmred_images_park/seq_len_8/all_uniform"), 0):
        if int(str(load_mmred_sample(cand)[4]).strip()) >= 2:
            sd = cand
            break
    _sid, frames, q0, states, a0 = load_mmred_sample(sd)
    frames = [f.resize((args.resize, args.resize)) for f in frames]
    char, room = target_of(Path(str(sd)), q0)
    evid = {t for t, st in enumerate(states)
            if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
    gold = str(len(evid))
    full = move_to_device(dict(processor.apply_chat_template(
        build_task_messages(frames, "count", char, room, q0, answer=gold),
        add_generation_prompt=False, tokenize=True, return_dict=True,
        return_tensors="pt")), rt.device)
    plen = processor.apply_chat_template(
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
    labels[:, :plen] = -100
    full.pop("attention_mask", None)
    base = build_block_mask(len(ids), blocks, hide_cols=[])
    hooks_lo = FenceHooks(layers[:Lg]).install()
    hooks_hi = FenceHooks(layers[Lg:]).install()
    loci = [locate_word_token(ids, tok, room, b) for b in blocks]

    # finite-difference: does the LOSS depend on hiding one evidence block? (value path)
    def loss_with_bits(bits_vec):
        gm = gated_tail_mask(base, blocks, fin_start, bits_vec)
        hooks_lo.set_mask(base, rt.device)
        hooks_hi.set_mask(gm.detach(), rt.device)
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
                return float(model(**full, position_ids=pos, labels=labels).loss)
        finally:
            hooks_lo.clear_mask(); hooks_hi.clear_mask()

    ones = torch.ones(len(blocks), device=rt.device)
    hide1 = ones.clone()
    hide1[sorted(evid)[0]] = 0.0
    l_open, l_hide = loss_with_bits(ones), loss_with_bits(hide1)
    print(f"finite-diff: loss(all open)={l_open:.4f} loss(hide 1 evid)={l_hide:.4f} "
          f"delta={l_hide-l_open:+.4f} (must be != 0)", flush=True)

    # grad path with a hook on the INJECTED tensor
    hooks_lo.set_mask(base, rt.device)
    st_states = None
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        hooks_hi.set_mask(base, rt.device)
        hs = model(**{k: v for k, v in full.items()}, position_ids=pos,
                   output_hidden_states=True, use_cache=False).hidden_states[Lg][0]
        st_states = hs[torch.tensor(loci)].float().detach()
        hooks_hi.clear_mask()
    bits = st_gumbel_bits(gate_head(st_states), tau=2.0,
                          rng_noise=torch.zeros(len(blocks), device=rt.device))
    print(f"bits: {[round(float(b),3) for b in bits]}, grad_fn={bits.grad_fn is not None}",
          flush=True)
    gm = gated_tail_mask(base, blocks, fin_start, bits)
    print(f"gmask grad_fn={gm.grad_fn is not None}", flush=True)
    reached = {"flag": False}
    gm4 = gm.view(1, 1, *gm.shape)

    def mark(_g):
        reached["flag"] = True

    gm4.register_hook(mark)
    # inject the EXACT 4-D tensor (bypass set_mask's view/to to keep our hook)
    hooks_hi._holder["mask"] = gm4
    with sdpa_kernel([SDPBackend.MATH]):
        loss = model(**full, position_ids=pos, labels=labels).loss
    loss.backward()
    hooks_lo.clear_mask(); hooks_hi.clear_mask()
    gnorm = float(torch.cat([p.grad.flatten() for p in gate_head.parameters()]).norm())
    print(f"backward reached injected mask tensor: {reached['flag']}", flush=True)
    print(f"gate head grad norm: {gnorm:.3e}", flush=True)

    # ---- STAGE 3: does an A-style pass (GC ON, clear-before-backward) BEFORE a
    # C-style pass poison the later pass? (the diag3-vs-micro ordering difference)
    print("=== STAGE 3: GC-ordering pollution test ===", flush=True)

    def one_pass(tag, use_ckpt, hold):
        for p in list(model.parameters()) + list(gate_head.parameters()):
            p.grad = None
        if use_ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        st2 = None
        hooks_lo.set_mask(base, rt.device)
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            hooks_hi.set_mask(base, rt.device)
            hs2 = model(**full, position_ids=pos, output_hidden_states=True,
                        use_cache=False).hidden_states[args.gate_layer][0]
            st2 = hs2[torch.tensor(loci)].float().detach()
            hooks_hi.clear_mask()
        b2 = st_gumbel_bits(gate_head(st2), tau=2.0,
                            rng_noise=torch.zeros(len(blocks), device=rt.device))
        gm2 = gated_tail_mask(base, blocks, fin_start, b2)
        hooks_lo.set_mask(base, rt.device)
        hooks_hi._holder["mask"] = gm2.view(1, 1, *gm2.shape)
        with sdpa_kernel([SDPBackend.MATH]):
            l2 = model(**full, position_ids=pos, labels=labels).loss
        if not hold:
            hooks_lo.clear_mask(); hooks_hi.clear_mask()
        l2.backward()
        hooks_lo.clear_mask(); hooks_hi.clear_mask()
        gn = float(torch.cat([p.grad.flatten() for p in gate_head.parameters()]).norm())
        print(f"  {tag}: loss={float(l2):.4f} gate_grad={gn:.3e}", flush=True)
        return gn

    gn_c1 = one_pass("C-style fresh   (ckpt off, hold)", False, True)
    gn_a = one_pass("A-style          (ckpt ON, clear)", True, False)
    gn_c2 = one_pass("C-style AFTER A (ckpt off, hold)", False, True)
    (args.output / "verdict.txt").write_text(
        f"finite_diff_delta={l_hide-l_open:+.4f}\nmask_hook_reached={reached['flag']}\n"
        f"gate_grad_norm={gnorm:.3e}\n"
        f"stage3 C_fresh={gn_c1:.3e} A={gn_a:.3e} C_after_A={gn_c2:.3e}\n")
    hooks_lo.remove(); hooks_hi.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
