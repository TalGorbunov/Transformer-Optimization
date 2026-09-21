#!/usr/bin/env python3
"""SPARSE S8 step-0 — the equivalence smoke (HARD GATE for S8 training).

Claim to prove: a gated N-frame forward (A) is computationally equivalent to the
synthetic evidence-only twin (B): only the evidence frames as blocks + a prompt
declaring the ORIGINAL N. Under build_block_mask, hidden blocks contribute zero
to visible rows; under reset_positions every block carries block-0's positions
and the tail starts at blk0_max+1 — both N-independent. So A's visible-row
computation should equal B's up to bf16 reduction-order (different seq lengths
=> different tiling; bit-identity not expected).

PASS (pre-registered): ALL greedy decodes identical A vs B, AND median relative
L2 diff of the answer-position hidden state at L20 < 0.02 (L16/L28 reported).
FAIL => S8 must not train; the divergence channel is itself the S7-(ii) answer.

Usage:
  python scripts/sparse/diag_equiv.py --peft-adapter checkpoints/sft_fenced_gated_adapter \
      --limit 10 --output outputs/_scratch/sparse_smoke/equiv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    rooms_to_room2chars,
)
from gnnformer.fencing import FenceHooks, build_block_mask, reset_positions  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

_SPARSE = _REPO / "scripts" / "sparse"
if str(_SPARSE) not in sys.path:
    sys.path.insert(0, str(_SPARSE))
from train_sft_gated import build_fenced_messages, parse_layout  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import target_of  # noqa: E402

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
HS_LAYERS = (16, 20, 28)
MAX_NEW = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="samples per N")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--peft-adapter", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    if args.peft_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.peft_adapter), is_trainable=False)
        model.eval()
    hooks = FenceHooks(layers).install()

    def forward_and_decode(frames, q0, hide_evid, declare_n):
        """One fenced config: -> (answer-pos hidden per layer, greedy text)."""
        msgs = build_fenced_messages(frames, q0, declare_n=declare_n)
        inp = move_to_device(dict(processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")), rt.device)
        ids = inp["input_ids"]
        grid = inp.get("image_grid_thw")

        def setup(cur_ids):
            l = cur_ids[0].tolist()
            parsed = parse_layout(l, tok, q0, len(frames), vs_id)
            if parsed is None:
                raise ValueError("layout parse failed")
            blocks, fin = parsed
            hide = []
            if hide_evid is not None:
                for t, (a, b) in enumerate(blocks):
                    if t not in hide_evid:
                        hide.extend(range(int(a), int(b)))
            mask = build_block_mask(len(l), blocks, hide_cols=hide)
            with torch.inference_mode():
                pos, _ = rope_fn(cur_ids, image_grid_thw=grid,
                                 attention_mask=torch.ones_like(cur_ids))
            return mask, reset_positions(pos, blocks, fin)

        # capture pass
        mask, pos = setup(ids)
        hooks.set_mask(mask, rt.device)
        try:
            with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
                o = model(input_ids=ids, pixel_values=inp.get("pixel_values"),
                          image_grid_thw=grid, position_ids=pos.to(rt.device),
                          output_hidden_states=True, use_cache=False)
        finally:
            hooks.clear_mask()
        caps = {L: o.hidden_states[L][0, -1].float().cpu() for L in HS_LAYERS}
        del o
        # greedy decode (cache-free)
        toks = []
        for _ in range(MAX_NEW):
            mask, pos = setup(ids)
            hooks.set_mask(mask, rt.device)
            try:
                with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
                    lg = model(input_ids=ids, pixel_values=inp.get("pixel_values"),
                               image_grid_thw=grid,
                               position_ids=pos.to(rt.device)).logits
            finally:
                hooks.clear_mask()
            nxt = int(lg[0, -1].argmax())
            if nxt == tok.eos_token_id:
                break
            toks.append(nxt)
            if "\n" in tok.decode(toks):
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return caps, tok.decode(toks, skip_special_tokens=True).strip()

    rows = ["N,sample,k,decode_A,decode_B,decode_eq," +
            ",".join(f"reldiff_L{L}" for L in HS_LAYERS)]
    rel_by_layer = {L: [] for L in HS_LAYERS}
    n_eq = n_tot = 0
    for N, root in ((16, "data/mmred_longN_park/seq_len_16/all_uniform"),
                    (32, "data/mmred_longN_park/seq_len_32/all_uniform")):
        done = 0
        for sd in iter_sample_dirs_shuffled(Path(root), 3):
            if done >= args.limit:
                break
            try:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
                if not (1 <= gold <= 16):
                    continue
                tr = target_of(Path(str(sd)), q0)
                char, room = tr
                evid = sorted(t for t, st in enumerate(states)
                              if char in rooms_to_room2chars(st.get("rooms", {}))
                              .get(room, []))
                if len(evid) != gold:
                    continue
                frames = [f.resize((args.resize, args.resize)) for f in frames]
                capsA, decA = forward_and_decode(frames, q0, set(evid), None)
                capsB, decB = forward_and_decode([frames[t] for t in evid], q0,
                                                 None, N)
            except Exception as e:
                print(f"  [skip] {sd.name}: {e}", flush=True)
                continue
            rel = {}
            for L in HS_LAYERS:
                d = float(torch.linalg.norm(capsA[L] - capsB[L]))
                n_ = float(torch.linalg.norm(capsA[L]))
                rel[L] = d / max(n_, 1e-9)
                rel_by_layer[L].append(rel[L])
            eq = decA == decB
            n_eq += int(eq)
            n_tot += 1
            rows.append(f"{N},{sd.name},{gold},{decA},{decB},{int(eq)},"
                        + ",".join(f"{rel[L]:.5f}" for L in HS_LAYERS))
            print(rows[-1], flush=True)
            done += 1
    med = {L: float(np.median(rel_by_layer[L])) for L in HS_LAYERS if rel_by_layer[L]}
    verdict = ("PASS" if n_tot and n_eq == n_tot and med.get(20, 1) < 0.02
               else "FAIL")
    summary = (f"decodes identical {n_eq}/{n_tot}; median reldiff "
               + " ".join(f"L{L}={med.get(L, float('nan')):.5f}" for L in HS_LAYERS)
               + f"\nVERDICT: {verdict} (pass = all decodes equal AND median L20 < 0.02)")
    (out / "equiv.csv").write_text("\n".join(rows) + "\n")
    (out / "report.txt").write_text(summary + "\n")
    print(summary)
    hooks.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
