#!/usr/bin/env python3
"""Track B phase 4a: InternVL2.5-8B NATIVE reading axis (E6 port).

Gradient of the gold-digit logit margin w.r.t. the carrier-token state entering layer L+1, sign-aligned
mean over samples. Grad flow is enabled by replacing the hidden states entering LM layer 0 with a
detached requires-grad leaf (the frozen 4-bit weights otherwise leave activations grad-free; the
InternVL forward splices vision embeds in-place, so the leaf must be created AFTER the splice).
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from experiments.internvl.baseline_eval import build_transform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--layers", default="16,20")
    ap.add_argument("--offset", type=int, default=13, help="carrier offset-from-end (room token)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/native_axis")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(args.model_name, quantization_config=bnb, trust_remote_code=True,
                                      use_flash_attn=False, low_cpu_mem_usage=True, device_map={"": 0}).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    lm_layers = model.language_model.model.layers
    tfm = build_transform()
    vdt = next(p_.dtype for p_ in model.vision_model.parameters() if p_.is_floating_point())
    import copy as _copy
    cand = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(9)}

    captured = {}
    def leaf_pre(_m, hargs, hkwargs):
        hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
        if hs is None or hs.requires_grad:
            return hargs, hkwargs
        hs = hs.detach().requires_grad_(True)
        if hargs:
            return (hs,) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = hs
        return hargs, hkwargs
    # graph starts at the FIRST probed layer (eager-attention backward is ~0.8GB/layer at S~2.4k;
    # layers below min(Ls) contribute nothing to the carrier gradients we read)
    lm_layers[min(Ls)].register_forward_pre_hook(leaf_pre, with_kwargs=True)

    def mk_pre(L):
        def pre(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is not None and hs.requires_grad:
                hs.retain_grad()
                captured[L] = hs
            return hargs, hkwargs
        return pre
    for L in Ls:
        lm_layers[L + 1].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    def build(frames, question):
        pv = torch.cat([tfm(f).unsqueeze(0) for f in frames]).to(vdt).cuda()
        tpl = _copy.deepcopy(model.conv_template)
        prefix = "".join(f"Frame-{i+1}: <image>\n" for i in range(len(frames)))
        tpl.append_message(tpl.roles[0], prefix + question + "\nAnswer with a single number.")
        tpl.append_message(tpl.roles[1], None)
        prompt = tpl.get_prompt()
        for _ in frames:
            prompt = prompt.replace("<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>", 1)
        enc = tok(prompt, return_tensors="pt")
        return pv, enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    grads = {L: [] for L in Ls}
    golds = []
    n = 0; fails = 0
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            if gold not in cand:
                continue
            pv, ids, am = build(frames, q0)
            S = int(ids.shape[1]); pos = S - 1 - args.offset
            fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
            captured.clear()
            outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
            logits = outp.logits[0, -1].float()
            others = torch.stack([logits[cand[d]] for d in cand if d != gold])
            loss = logits[cand[gold]] - torch.logsumexp(others, 0)
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.empty_cache()
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
        except Exception as exc:
            fails += 1
            print(f"{sd}: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    res = {}
    for L in Ls:
        G = np.stack(grads[L])
        signs = np.sign(G @ G[0]); signs[signs == 0] = 1
        Ga = G * signs[:, None]
        axis = Ga.mean(0); axis /= np.linalg.norm(axis)
        coh = float(np.mean(np.abs(Ga @ axis)))
        res[L] = {"axis": axis, "coherence": coh, "n": len(G)}
        print(f"L{L}: native axis from {len(G)} grads, coherence={coh:.3f}")
    torch.save({"axes": res, "golds": golds, "offset": args.offset, "config": vars(args)},
               out / "native_axes.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
