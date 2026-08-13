#!/usr/bin/env python3
"""Track B phase 4b: E5 causal verification on InternVL2.5-8B.

delta_L from a carrier_map messages_cache (auto-picks the peak offset; uses all cached layers with
d' >= 0.8 * peak at that offset). Arms: base | multi:lam (dose (lam-1)*g*delta_L at every chosen
layer) | scrub (project dhat out continuously) | scrubrand (random-axis control). Per arm: emitted
digit + final-layer last-token rep (repaired-readout CPU analysis).
Registered predictions (cross-family E5): scrub -> undercount collapse, scrubrand inert, dose moves
decodability/repaired-read but not emission.
"""
from __future__ import annotations
import argparse, glob, random, sys, time
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
    ap.add_argument("--cache", default="", help="messages_cache.pt from carrier_map (default: latest carrier_map_ext)")
    ap.add_argument("--lams", default="4,16")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/dose_scrub")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    cache = args.cache or sorted(glob.glob("outputs/frame_axis/internvl/carrier_map_ext/*/messages_cache.pt"))[-1]
    mc = torch.load(cache, map_location="cpu", weights_only=False)
    dmat, lab = mc["dmat"], mc["labels"]
    peak = max(((d, L, o) for L, row in dmat.items() for o, d in enumerate(row) if d == d))
    off = peak[2]
    use_L = [L for L, row in dmat.items() if row[off] >= 0.8 * peak[0]]
    print(f"cache {cache}: peak d'={peak[0]:.2f} @L{peak[1]} off{off}; dose/scrub layers {use_L}")
    delta, dhat, rnd = {}, {}, {}
    rng = np.random.RandomState(0)
    for L in use_L:
        M = mc["msgs"][L][off].astype(np.float32)
        fl_ = lab.reshape(-1)
        X = M.reshape(-1, M.shape[-1])
        d = X[fl_ == 1].mean(0) - X[fl_ == 0].mean(0)
        delta[L] = d
        dhat[L] = d / (np.linalg.norm(d) + 1e-12)
        r = rng.randn(d.shape[0]).astype(np.float32)
        rnd[L] = r / np.linalg.norm(r)

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
    cand = [(d_, tok.encode(str(d_), add_special_tokens=False)[0]) for d_ in range(9)]
    cand_vals = [d_ for d_, _ in cand]
    cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)
    dev = next(lm_layers[0].parameters()).device
    delta_t = {L: torch.tensor(delta[L], device=dev, dtype=torch.float32) for L in use_L}
    dhat_t = {L: torch.tensor(dhat[L], device=dev, dtype=torch.float32) for L in use_L}
    rnd_t = {L: torch.tensor(rnd[L], device=dev, dtype=torch.float32) for L in use_L}

    st = {"mode": "base", "prm": 1.0, "pos": None, "gold": 0, "final": None}

    def mk_pre(L):
        def pre(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is None or st["mode"] == "base" or st["pos"] is None or hs.shape[1] <= st["pos"]:
                return hargs, hkwargs
            hs = hs.clone()
            h = hs[0, st["pos"], :].float()
            if st["mode"] == "multi":
                h = h + (st["prm"] - 1.0) * st["gold"] * delta_t[L]
            elif st["mode"] == "scrub":
                u = dhat_t[L]; h = h - (h @ u) * u
            elif st["mode"] == "scrubrand":
                u = rnd_t[L]; h = h - (h @ u) * u
            hs[0, st["pos"], :] = h.to(hs.dtype)
            if hargs:
                return (hs,) + tuple(hargs[1:]), hkwargs
            hkwargs = dict(hkwargs); hkwargs["hidden_states"] = hs
            return hargs, hkwargs
        return pre
    for L in use_L:
        lm_layers[L + 1].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    def final_hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        st["final"] = h[0, -1, :].float().cpu().clone()
    lm_layers[-1].register_forward_hook(final_hook)

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

    arms = [("base", 1.0)] + [("multi", float(x)) for x in args.lams.replace(",", " ").split() if x] + \
           [("scrub", 0.0), ("scrubrand", 0.0)]
    nm = lambda a: f"{a[0]}:{a[1]:g}" if a[0] == "multi" else a[0]
    res = {a: {"pred": [], "gold": []} for a in arms}
    reps_final = {a: [] for a in arms}
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    n = 0; fails = 0
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pv, ids, am = build(frames, q0)
            S = int(ids.shape[1])
            st["pos"] = S - 1 - off; st["gold"] = gold
            fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
            for a in arms:
                st["mode"], st["prm"] = a
                st["final"] = None
                with torch.no_grad():
                    outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
                lg = outp.logits[0, -1].float().cpu()
                pred = int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])
                res[a]["pred"].append(pred); res[a]["gold"].append(gold)
                reps_final[a].append(st["final"].half())
            st["mode"] = "base"
            n += 1
            if n % 20 == 0:
                accs = {nm(a): float(np.mean(np.array(res[a]["pred"]) == np.array(res[a]["gold"]))) for a in arms}
                print(f"  {n}: " + "  ".join(f"{k}={v:.3f}" for k, v in accs.items()), flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    lines = [f"=== InternVL2.5-8B E5 dose/scrub (off{off}, layers {use_L}, n={n}) ==="]
    for a in arms:
        p = np.array(res[a]["pred"]); g = np.array(res[a]["gold"])
        by = " ".join(f"g{gv}:{np.mean(p[g == gv] == gv):.2f}" for gv in sorted(set(g.tolist())))
        lines.append(f"  {nm(a):>10} acc={np.mean(p==g):.3f}  MAE={np.mean(np.abs(p-g)):.2f}   [{by}]")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"res": {nm(a): v for a, v in res.items()},
                "reps_final": {nm(a): torch.stack(v) for a, v in reps_final.items() if v},
                "off": off, "layers": use_L, "config": vars(args)}, out / "doses.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
