#!/usr/bin/env python3
"""P1 of the gating campaign: is there a sink worth filtering in OUR model?

Qwen2.5-VL-7B frozen (4-bit nf4, sdpa), MMReD park N=8, two layouts:

  plain     the canonical counting prompt, plain causal mask, base M-RoPE positions
  deployed  THE method layout — carrier token per frame, block fence + per-block
            position reset, frozen distilled e_c injected at the carrier rows

Per layer it reports
  (a) F-Attn  fraction of attention mass on token 0, head-averaged over all query rows
  (b) M-Act   max |hidden activation| at the layer input
  (c) the one that matters — of the READ rows' attention mass, how much goes to the
      sink vs to frame tokens vs to carriers vs to the question. Read rows are the
      last prompt row (plain) and the carrier rows + tail rows (deployed).

Gate: if the read rows lose a meaningful share of mass to a sink, the interference
story has a mechanism and P3 is well-motivated; if not, P3 is a falsification.

Attention is recomputed from the captured q/k projections (same recipe as
scripts/presentation_diagnostics/probe_attention_map.py) because sdpa returns no weights.

Usage:
  python scripts/gating/probe_sink_7b.py --ckpt checkpoints/carrier_layer_fmt_caption_best.pt \
      --limit 8 --output outputs/gating/p1_sink_7b/<stamp>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.constants import MASK_MIN, ROOMS
from gnnformer.data import (
    build_count_prompt,
    build_prompt_inputs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    probe_evidence,
)
from gnnformer.engine import SDPA_BACKENDS, CarrierEngine
from gnnformer.runtime import (
    attention_dims,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)


def head_avg_attention(qr, kr, madd, n_heads: int, hd: int) -> torch.Tensor:
    """Head-averaged softmax attention [seq, seq] from rotated q/k and an additive mask."""
    seq = qr.shape[1]
    A = torch.zeros(seq, seq, device=qr.device, dtype=torch.float32)
    for h in range(n_heads):
        A += torch.softmax(qr[h] @ kr[h].T / (hd ** 0.5) + madd, -1)
    return A / n_heads


def mass(A: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> float:
    """Mean over `rows` of the attention mass they place on `cols`."""
    if rows.numel() == 0 or cols.numel() == 0:
        return 0.0
    return float(A[rows][:, cols].sum(1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/carrier_layer_fmt_caption_best.pt",
                    help="carrier_layer ckpt: supplies e_c, L*, and (unless --no-lora) LoRA")
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["plain", "deployed"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    n_layers = len(layers)
    dims = attention_dims(model)
    n_heads, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = load_carrier_layer_ckpt(Path(args.ckpt))
    e_c = torch.as_tensor(ck.e_c).to(dev)
    lora = None
    if not args.no_lora:
        lora = attach_lora(layers, ck.l_open, rank=ck.rank, alpha=ck.alpha, device=dev,
                           state=ck.lora_state)
    eng = CarrierEngine(rt, l_open=ck.l_open, e_c=e_c)
    print(f"[ckpt] {args.ckpt}: L*={ck.l_open} rank={ck.rank} "
          f"lora={'off' if args.no_lora else 'on'} | heads {n_heads} kv {n_kv} hd {hd} "
          f"layers {n_layers}", flush=True)

    # accumulators: arm -> key -> [n_layers]
    acc: Dict[str, Dict[str, np.ndarray]] = {}
    n_done: Dict[str, int] = {a: 0 for a in args.arms}

    def bump(arm: str, key: str, li: int, v: float) -> None:
        acc.setdefault(arm, {}).setdefault(key, np.zeros(n_layers))[li] += v

    t0 = time.time()
    n_seen = 0
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs):
        if n_seen >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            if probe_evidence(args.task, q0, states, gold, ROOMS) is None:
                continue
        except Exception:
            continue
        NF = len(frames)
        ok = False

        # ------------------------------------------------------------------ deployed
        if "deployed" in args.arms:
            d = eng.prepare_sample(frames, q0, gold=gold, task=args.task, resize=args.resize,
                                   with_masks=True)
            if d is not None:
                seq, cpos, blocks, fin = d["seq"], d["cpos"], d["blocks"], d["fin"]
                col_sink = torch.tensor([0], device=dev)
                col_car = torch.tensor(cpos, device=dev)
                fcols: List[int] = []
                for (a, b), c in zip(blocks, cpos):
                    fcols.extend(i for i in range(a, b) if i != c)
                col_frames = torch.tensor(fcols, device=dev)
                col_pre = torch.tensor([i for i in range(1, blocks[0][0])], device=dev)
                col_tail = torch.tensor(list(range(fin, seq)), device=dev)
                row_car = torch.tensor(cpos, device=dev)
                row_tail = torch.tensor([seq - 1], device=dev)
                with torch.no_grad():
                    emb = d["emb"].to(dev).unsqueeze(0).clone()
                    emb[0, row_car] = e_c.to(torch.bfloat16)
                    # masks in the hidden dtype: sdpa rejects an fp32 bias against a bf16
                    # query (the fused kernels' check), and the diagnostic only needs
                    # MASK_MIN to zero the forbidden entries. Same as
                    # scripts/presentation_diagnostics/probe_attention_map.py.
                    lo = d["lo"].to(dev).to(emb.dtype).view(1, 1, seq, seq)
                    hi = d["hi"].to(dev).to(emb.dtype).view(1, 1, seq, seq)
                    cos_, sin_ = text_model.rotary_emb(emb, d["pos"].to(dev))
                    pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                    h = emb
                    for li in range(n_layers):
                        mask4 = lo if li < eng.l_open else hi
                        ln = layers[li].input_layernorm(h)
                        at = layers[li].self_attn
                        q = at.q_proj(ln).view(1, seq, n_heads, hd).transpose(1, 2)
                        k = at.k_proj(ln).view(1, seq, n_kv, hd).transpose(1, 2)
                        qr, kr = apply_multimodal_rotary_pos_emb(
                            q.float(), k.float(), pe[0].float(), pe[1].float(),
                            dims["mrope_section"])
                        A = head_avg_attention(qr[0], repeat_kv(kr, n_heads // n_kv)[0],
                                               mask4[0, 0].float(), n_heads, hd)
                        bump("deployed", "f_attn", li, float(A[1:, 0].mean()))
                        bump("deployed", "m_act", li, float(h.abs().max()))
                        bump("deployed", "car_sink", li, mass(A, row_car, col_sink))
                        bump("deployed", "car_frames", li, mass(A, row_car, col_frames))
                        bump("deployed", "car_pre", li, mass(A, row_car, col_pre))
                        bump("deployed", "car_car", li, mass(A, row_car, col_car))
                        bump("deployed", "tail_sink", li, mass(A, row_tail, col_sink))
                        bump("deployed", "tail_frames", li, mass(A, row_tail, col_frames))
                        bump("deployed", "tail_pre", li, mass(A, row_tail, col_pre))
                        bump("deployed", "tail_car", li, mass(A, row_tail, col_car))
                        bump("deployed", "tail_tail", li, mass(A, row_tail, col_tail))
                        del A, qr, kr, q, k, ln
                        with sdpa_kernel(SDPA_BACKENDS):
                            h = layers[li](h, attention_mask=mask4, position_embeddings=pe)[0]
                n_done["deployed"] += 1
                ok = True

        # --------------------------------------------------------------------- plain
        if "plain" in args.arms:
            prompt = build_count_prompt(q0, NF)
            fr = [f.resize((args.resize, args.resize)) for f in frames] if args.resize > 0 else frames
            inputs = move_to_device(build_prompt_inputs(processor, fr, prompt), dev)
            ids = inputs["input_ids"][0]
            seq = int(ids.shape[0])
            fg = image_token_groups(ids.cpu(), expected_num_frames=NF, processor=processor)
            if len(fg) == NF:
                fcols = [int(p) for g in fg for p in g]
                col_frames = torch.tensor(fcols, device=dev)
                col_sink = torch.tensor([0], device=dev)
                fmin = min(fcols)
                col_pre = torch.tensor(list(range(1, fmin)), device=dev)
                col_tail = torch.tensor(list(range(max(fcols) + 1, seq)), device=dev)
                row_last = torch.tensor([seq - 1], device=dev)
                with torch.no_grad():
                    pos, _ = rope_fn(inputs["input_ids"],
                                     image_grid_thw=inputs.get("image_grid_thw"),
                                     attention_mask=inputs.get("attention_mask"))
                    emb = text_model.embed_tokens(inputs["input_ids"])
                    img = model.model.get_image_features(inputs["pixel_values"],
                                                         inputs["image_grid_thw"])
                    img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
                    emb = emb.clone()
                    emb[0, ids == model.config.image_token_id] = img.to(emb.dtype)
                    causal = torch.zeros(seq, seq, dtype=torch.float32, device=dev)
                    causal.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool,
                                                              device=dev), 1), MASK_MIN)
                    mask4 = causal.to(emb.dtype).view(1, 1, seq, seq)
                    cos_, sin_ = text_model.rotary_emb(emb, pos.to(dev))
                    pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                    h = emb
                    for li in range(n_layers):
                        ln = layers[li].input_layernorm(h)
                        at = layers[li].self_attn
                        q = at.q_proj(ln).view(1, seq, n_heads, hd).transpose(1, 2)
                        k = at.k_proj(ln).view(1, seq, n_kv, hd).transpose(1, 2)
                        qr, kr = apply_multimodal_rotary_pos_emb(
                            q.float(), k.float(), pe[0].float(), pe[1].float(),
                            dims["mrope_section"])
                        A = head_avg_attention(qr[0], repeat_kv(kr, n_heads // n_kv)[0],
                                               causal, n_heads, hd)
                        bump("plain", "f_attn", li, float(A[1:, 0].mean()))
                        bump("plain", "m_act", li, float(h.abs().max()))
                        bump("plain", "read_sink", li, mass(A, row_last, col_sink))
                        bump("plain", "read_frames", li, mass(A, row_last, col_frames))
                        bump("plain", "read_pre", li, mass(A, row_last, col_pre))
                        bump("plain", "read_tail", li, mass(A, row_last, col_tail))
                        del A, qr, kr, q, k, ln
                        with sdpa_kernel(SDPA_BACKENDS):
                            h = layers[li](h, attention_mask=mask4, position_embeddings=pe)[0]
                n_done["plain"] += 1
                ok = True

        if ok:
            n_seen += 1
            print(f"  {n_seen}/{args.limit} ({time.time()-t0:.0f}s)", flush=True)

    if lora is not None:
        lora.remove()

    res = {arm: {k: (v / max(n_done[arm], 1)).tolist() for k, v in d.items()}
           for arm, d in acc.items()}
    (out / "sink_stats.json").write_text(json.dumps(
        {"n": n_done, "l_open": ck.l_open, "ckpt": args.ckpt, "data_root": args.data_root,
         "no_lora": args.no_lora, "stats": res}, indent=2))

    lines = [f"=== P1 SINK DIAGNOSTIC (7B frozen, L*={ck.l_open}, "
             f"lora={'off' if args.no_lora else 'on'}, n={n_done}) ===", ""]
    if "plain" in res:
        r = res["plain"]
        lines += ["--- plain (causal, canonical count prompt); read row = last prompt token",
                  f"{'L':>3} {'F-Attn':>8} {'M-Act':>9} | {'read>sink':>9} {'read>frames':>11} "
                  f"{'read>pre':>9} {'read>tail':>9}"]
        for li in range(n_layers):
            lines.append(f"{li:>3} {r['f_attn'][li]:>8.4f} {r['m_act'][li]:>9.1f} | "
                         f"{r['read_sink'][li]:>9.4f} {r['read_frames'][li]:>11.4f} "
                         f"{r['read_pre'][li]:>9.4f} {r['read_tail'][li]:>9.4f}")
        lines.append("")
    if "deployed" in res:
        r = res["deployed"]
        lines += ["--- deployed (fence + posreset + carriers); read rows = carriers / final tail",
                  f"{'L':>3} {'F-Attn':>8} {'M-Act':>9} | {'car>sink':>8} {'car>frames':>10} "
                  f"{'car>pre':>8} | {'tail>sink':>9} {'tail>car':>8} {'tail>frames':>10} "
                  f"{'tail>pre':>8} {'tail>tail':>9}"]
        for li in range(n_layers):
            lines.append(f"{li:>3} {r['f_attn'][li]:>8.4f} {r['m_act'][li]:>9.1f} | "
                         f"{r['car_sink'][li]:>8.4f} {r['car_frames'][li]:>10.4f} "
                         f"{r['car_pre'][li]:>8.4f} | {r['tail_sink'][li]:>9.4f} "
                         f"{r['tail_car'][li]:>8.4f} {r['tail_frames'][li]:>10.4f} "
                         f"{r['tail_pre'][li]:>8.4f} {r['tail_tail'][li]:>9.4f}")
        lines.append("")
        lo_span = slice(0, ck.l_open)
        hi_span = slice(ck.l_open, n_layers)
        lines.append(f"SUMMARY  carrier rows below L*: sink {np.mean(r['car_sink'][lo_span]):.3f} "
                     f"vs own-frame {np.mean(r['car_frames'][lo_span]):.3f}")
        lines.append(f"SUMMARY  tail row at/above L*: sink {np.mean(r['tail_sink'][hi_span]):.3f} "
                     f"vs carriers {np.mean(r['tail_car'][hi_span]):.3f}")
    if "plain" in res:
        rp = res["plain"]
        lines.append(f"SUMMARY  plain read row: sink {np.mean(rp['read_sink']):.3f} "
                     f"vs frames {np.mean(rp['read_frames']):.3f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: sink share of the read rows, per layer, both arms
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    xs = np.arange(n_layers)
    if "plain" in res:
        axes[0].plot(xs, res["plain"]["f_attn"], label="plain", lw=2)
    if "deployed" in res:
        axes[0].plot(xs, res["deployed"]["f_attn"], label="deployed", lw=2)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("F-Attn (mass on token 0)")
    axes[0].set_title("attention sink, all query rows")
    axes[0].legend()
    if "plain" in res:
        axes[1].plot(xs, res["plain"]["read_sink"], label="plain: last row > sink", lw=2)
        axes[1].plot(xs, res["plain"]["read_frames"], label="plain: last row > frames", lw=2, ls="--")
    if "deployed" in res:
        axes[1].plot(xs, res["deployed"]["car_sink"], label="deployed: carriers > sink", lw=2)
        axes[1].plot(xs, res["deployed"]["car_frames"], label="deployed: carriers > frames",
                     lw=2, ls="--")
        axes[1].plot(xs, res["deployed"]["tail_sink"], label="deployed: tail > sink", lw=2)
        axes[1].axvline(ck.l_open, color="k", ls=":", lw=1)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("attention mass")
    axes[1].set_title("what the READ rows spend their mass on")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"P1 sink diagnostic — Qwen2.5-VL-7B frozen, MMReD park N=8 (n={n_done})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sink_7b.{ext}", dpi=300)

    (out / "ABOUT.md").write_text(
        "# P1 — is there a sink worth filtering in Qwen2.5-VL-7B?\n\n"
        "Frozen 7B (4-bit nf4). Two layouts on MMReD park N=8: `plain` (canonical count\n"
        "prompt, plain causal, base M-RoPE) and `deployed` (carrier per frame, block fence\n"
        f"+ per-block position reset, frozen e_c and LoRA from {args.ckpt}, L*={ck.l_open}).\n"
        "Attention is recomputed from the captured q/k projections (sdpa returns no weights),\n"
        "head-averaged. Reported per layer: F-Attn (mass on token 0 over all query rows),\n"
        "M-Act (max |hidden| at the layer input), and the read rows' mass split across\n"
        "sink / frame tokens / carriers / question prefix / tail.\n\n"
        f"Data: {args.data_root}, n={n_done}. Artifacts: report.txt, sink_stats.json,\n"
        "sink_7b.png/pdf.\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
