#!/usr/bin/env python3
"""SPARSE diagnostic — is the backward recompute (gradient checkpointing) seeing
the fence mask?

One gated training sample, three gradient computations on a fresh LoRA:
  A) current pattern: mask cleared BEFORE backward (suspect: recompute ungated)
  B) mask HELD through backward (candidate fix)
  C) ground truth: gradient checkpointing DISABLED, mask held
Report max|g|/cos similarity of LoRA grads B-vs-C and A-vs-C. If B==C and A!=C,
the recompute corruption is proven and the fix is validated.

Usage: python scripts/sparse/diag_grad_mask.py --output outputs/_scratch/sparse_smoke/diag_grad
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

from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, rooms_to_room2chars  # noqa: E402
from gnnformer.fencing import FenceHooks, build_block_mask, reset_positions  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

_LORAMECH = _REPO / "scripts" / "loramech"
if str(_LORAMECH) not in sys.path:
    sys.path.insert(0, str(_LORAMECH))
from train_sft_fenced import build_fenced_messages, parse_layout  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    hooks = FenceHooks(layers).install()

    # one gated sample with k>0 and hidden blocks
    sample = None
    for sd in iter_sample_dirs_shuffled(Path("data/mmred_images_park/seq_len_8/all_uniform"), 0):
        _sid, frames, q0, states, a0 = load_mmred_sample(sd)
        gold = int(str(a0).strip())
        if not (2 <= gold <= 5):
            continue
        tr = target_of(Path(str(sd)), q0)
        if tr is None:
            continue
        char, room = tr
        evid = {t for t, st in enumerate(states)
                if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
        if len(evid) != gold:
            continue
        frames = [f.resize((256, 256)) for f in frames]
        sample = (frames, q0, gold, evid)
        break
    frames, q0, gold, evid = sample
    print(f"sample gold={gold} evid={sorted(evid)}", flush=True)

    full = processor.apply_chat_template(
        build_fenced_messages(frames, q0, answer=gold), add_generation_prompt=False,
        tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template(
        build_fenced_messages(frames, q0), add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    full = move_to_device(dict(full), rt.device)
    ids = full["input_ids"][0].tolist()
    blocks, fin_start = parse_layout(ids, tok, q0, len(frames), vs_id)
    hide = [p for t, (a, b) in enumerate(blocks) if t not in evid
            for p in range(int(a), int(b))]
    mask = build_block_mask(len(ids), blocks, hide_cols=hide)
    bnd = int(prompt["input_ids"].shape[1])
    labels = full["input_ids"].clone()
    labels[:, :bnd] = -100
    with torch.no_grad():
        pos, _ = rope_fn(full["input_ids"], image_grid_thw=full.get("image_grid_thw"),
                         attention_mask=full.get("attention_mask"))
    pos = reset_positions(pos, blocks, fin_start)
    full.pop("attention_mask", None)

    names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]

    def grads(hold_mask, ckpt):
        model.zero_grad(set_to_none=True)
        if ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.train()
        hooks.set_mask(mask, rt.device)
        with sdpa_kernel(FENCED_SDPA):
            loss = model(**full, position_ids=pos.to(rt.device), labels=labels).loss
        if not hold_mask:
            hooks.clear_mask()
        loss.backward()
        hooks.clear_mask()
        g = {n: p.grad.detach().float().cpu().clone()
             for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        return float(loss.detach()), g

    lA, gA = grads(hold_mask=False, ckpt=True)   # current (suspect)
    print(f"A(current,ckpt) loss={lA:.4f}", flush=True)
    torch.cuda.empty_cache()
    lB, gB = grads(hold_mask=True, ckpt=True)    # fix candidate
    print(f"B(fix,ckpt) loss={lB:.4f}", flush=True)
    torch.cuda.empty_cache()
    lC, gC = grads(hold_mask=True, ckpt=False)   # ground truth
    print(f"C(truth,nockpt) loss={lC:.4f}", flush=True)
    torch.cuda.empty_cache()

    def cmp(g1, g2, tag):
        num = den1 = den2 = 0.0
        mx = 0.0
        for n in g1:
            if n not in g2:
                continue
            a, b = g1[n].flatten(), g2[n].flatten()
            num += float((a * b).sum())
            den1 += float((a * a).sum())
            den2 += float((b * b).sum())
            mx = max(mx, float((a - b).abs().max()))
        cos = num / max((den1 ** 0.5) * (den2 ** 0.5), 1e-12)
        print(f"{tag}: cos={cos:.6f} max|diff|={mx:.3e} "
              f"(norms {den1**0.5:.4e} vs {den2**0.5:.4e})", flush=True)
        return cos

    print(f"losses A={lA:.4f} B={lB:.4f} C={lC:.4f}", flush=True)
    cAB = cmp(gA, gB, "A(current)-vs-B(fix)")
    cAC = cmp(gA, gC, "A(current)-vs-C(truth)")
    cBC = cmp(gB, gC, "B(fix)-vs-C(truth)")
    verdict = ("CONFIRMED: recompute drops the mask; holding it through backward fixes it"
               if cBC > 0.99 and cAC < 0.9 else
               f"INCONCLUSIVE: cos A-C {cAC:.4f}, B-C {cBC:.4f}")
    print("VERDICT:", verdict)
    (args.output / "report.txt").write_text(
        f"losses A={lA:.4f} B={lB:.4f} C={lC:.4f}\ncosAC={cAC:.6f} cosBC={cBC:.6f}\n{verdict}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
