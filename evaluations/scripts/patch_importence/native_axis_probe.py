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
    ap.add_argument("--output", default="outputs/frame_axis/probes/native_axis")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    tok = processor.tokenizer
    cand = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(9)
            if len(tok.encode(str(d), add_special_tokens=False)) == 1}

    captured = {}

    def mk_pre(L):
        def pre_hook(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is not None and hs.requires_grad:
                hs.retain_grad()
                captured[L] = hs
            return hargs, hkwargs
        return pre_hook
    for L in Ls:
        layers[L + 1].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    grads = {L: [] for L in Ls}
    golds = []
    n = 0
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            if gold not in cand:
                continue
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
            seq = int(inputs["input_ids"].shape[1])
            pos = seq - 1 - args.offset
            captured.clear()
            # embeddings need grad so hidden states carry requires_grad through the stack
            emb = model.get_input_embeddings()
            ids = inputs.pop("input_ids")
            iem = emb(ids).detach().requires_grad_(True)
            outp = model(inputs_embeds=iem, **{k: v for k, v in inputs.items()}, use_cache=False)
            logits = outp.logits[0, -1].float()
            others = torch.stack([logits[cand[d]] for d in cand if d != gold])
            loss = logits[cand[gold]] - torch.logsumexp(others, 0)
            model.zero_grad(set_to_none=True)
            loss.backward()
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
            iem.grad = None
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
