#!/usr/bin/env python3
"""E6: extract the model's NATIVE readout axis at the carrier — the gradient of the answer-digit
logit margin w.r.t. the carrier token's residual state entering layer L+1 — and save it so CPU
analysis can compare it (cos) against delta-hat and the whitened LDA axis, and measure rho along it.

For each sample: one forward WITH grad enabled, hidden states captured at layers[L+1] input via a
pre-hook with retain_grad; loss = gold-digit logit − logsumexp(other digit logits) at the last
position; backward; take grad at the carrier position. Axes are averaged after sign-alignment.
Frozen 4-bit model — gradients flow to activations only. Task: steps (count).
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--layers", default="16,20", help="message layers; grad taken at entry of L+1")
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--text-frames", action="store_true",
                    help="A1 text-MMRED: frames as text blocks (same builder as the carrier probe)")
    ap.add_argument("--task", choices=["count", "text_cwe"], default="count",
                    help="text_cwe = synthetic word-count rung (implies --text-frames)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--resize", type=int, default=0,
                    help="resize frames to <resize>px before the processor (0 = native)")
    ap.add_argument("--output", default="outputs/frame_axis/probes/native_axis")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    n_rg = sum(1 for p in model.parameters() if p.requires_grad)
    model.requires_grad_(False)   # any grad-enabled float param builds the graph from layer 0
    print(f"[freeze] {n_rg} params had requires_grad=True -> all frozen")
    layers = get_layers(model)
    tok = processor.tokenizer
    cand = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(9)
            if len(tok.encode(str(d), add_special_tokens=False)) == 1}

    captured = {}

    # memory fix (2026-07-11 OOM at N>=32): build the autograd graph ONLY from the first probed
    # layer upward — a pre-hook on layers[min(Ls)+1] detaches the incoming hidden state and
    # re-enables grad (leaf), so layers 0..min(Ls) run graph-free. Halves activation memory.
    GRAD_FROM = min(Ls) + 1

    def graph_start_hook(_m, hargs, hkwargs):
        hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
        if not captured.get("_dbg"):
            captured["_dbg"] = True
            print(f"[graph-start] hook fired: hs={'None' if hs is None else tuple(hs.shape)} "
                  f"req_grad={getattr(hs,'requires_grad',None)} in_kwargs={'hidden_states' in hkwargs} "
                  f"n_args={len(hargs)} grad_enabled={torch.is_grad_enabled()}", flush=True)
        if hs is not None and not hs.requires_grad and torch.is_grad_enabled():
            hs2 = hs.detach().requires_grad_(True)
            captured["_start"] = hs2
            if "hidden_states" in hkwargs:
                hkwargs["hidden_states"] = hs2
            else:
                hargs = (hs2,) + tuple(hargs[1:])
        return hargs, hkwargs
    layers[GRAD_FROM].register_forward_pre_hook(graph_start_hook, with_kwargs=True)

    def mk_pre(L):
        def pre_hook(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is not None and hs.requires_grad:
                if not hs.is_leaf:
                    hs.retain_grad()
                captured[L] = hs
            return hargs, hkwargs
        return pre_hook
    for L in Ls:
        layers[L + 1].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    from evaluations.scripts.patch_importence.probe_frame_to_carrier_message import (
        build_text_inputs_and_groups, gen_cwe_sample)
    if args.task == "text_cwe":
        all_dirs = list(range(int(args.limit) * 3))
    else:
        all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    grads = {L: [] for L in Ls}
    golds = []
    n = 0
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            if args.task == "text_cwe":
                sid, blocks, q0, gold, _ft = gen_cwe_sample(int(sd), int(args.n_frames))
                if gold not in cand:
                    continue
                t_inputs, _fg = build_text_inputs_and_groups(None, q0, processor,
                                                             hi=int(args.n_frames), blocks=blocks)
                inputs = tgi.move_inputs_to_model_device(t_inputs)
            else:
                sid, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
                if gold not in cand:
                    continue
                if int(args.resize) > 0 and frames:
                    frames = [f.resize((int(args.resize), int(args.resize))) for f in frames]
                if args.text_frames:
                    t_inputs, _fg = build_text_inputs_and_groups(states, q0, processor,
                                                                 hi=len(states))
                    inputs = tgi.move_inputs_to_model_device(t_inputs)
                else:
                    inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
            seq = int(inputs["input_ids"].shape[1])
            pos = seq - 1 - args.offset
            captured.clear()
            # graph starts at GRAD_FROM via the pre-hook; layers below run graph-free.
            # last-position logits only (num_logits_to_keep) when the HF version supports it.
            try:
                outp = model(**inputs, use_cache=False, num_logits_to_keep=1)
            except TypeError:
                outp = model(**inputs, use_cache=False)
            logits = outp.logits[0, -1].float()
            others = torch.stack([logits[cand[d]] for d in cand if d != gold])
            loss = logits[cand[gold]] - torch.logsumexp(others, 0)
            model.zero_grad(set_to_none=True)
            loss.backward()
            del outp, logits, loss
            ok = True
            for L in Ls:
                hs = captured.get(L)
                if hs is None or hs.grad is None:
                    ok = False; break
                g = hs.grad[0, pos].detach().float().cpu().numpy()
                if not np.isfinite(g).all() or np.linalg.norm(g) < 1e-9:
                    ok = False; break
                grads[L].append(g / np.linalg.norm(g))
            if ok:
                golds.append(gold); n += 1
                if n % 10 == 0:
                    print(f"  {n}/{args.limit}", flush=True)
            if n % 10 == 0:
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{sd}: {type(exc).__name__}: {exc}")
            continue

    res = {}
    for L in Ls:
        G = np.stack(grads[L])                     # [n, H] unit grads
        ref = G[0]
        signs = np.sign(G @ ref); signs[signs == 0] = 1
        Ga = G * signs[:, None]
        axis = Ga.mean(0); axis /= np.linalg.norm(axis)
        coh = float(np.mean(np.abs(Ga @ axis)))    # alignment coherence across samples
        res[L] = {"axis": axis, "coherence": coh, "n": len(G)}
        print(f"L{L}: native axis from {len(G)} grads, coherence={coh:.3f}")
    torch.save({"axes": res, "golds": golds, "offset": args.offset,
                "config": vars(args)}, out / "native_axes.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
