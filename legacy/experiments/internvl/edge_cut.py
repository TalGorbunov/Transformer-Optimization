#!/usr/bin/env python3
"""Does InternVL's LAST token read the CARRIER for its answer content? Edge-cut necessity test.

Cut specific last-token attention edges with an additive mask (eager attention accepts 4D masks) and
measure what happens to the count content AT the last token (per-arm final decode; emission is
prior-locked so decodability is the informative endpoint).

Arms:
  base        no cuts
  cutcar_win  last -/-> carrier, layers in the measured window (8..19)
  cutcar_post last -/-> carrier, post-window (20..27)  [control: should be inert]
  cutq_win    last -/-> ALL other question tokens, window layers [alternative-route probe]
  cutimg_win  last -/-> all image tokens, window layers [direct-from-images route probe]
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

MIN = -65504.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--offset", type=int, default=13)
    ap.add_argument("--win-layers", default="8-19")
    ap.add_argument("--post-layers", default="20-27")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/edge_cut")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    off = int(args.offset)
    def rng_(spec):
        a, b = spec.split("-"); return list(range(int(a), int(b) + 1))
    WIN, POST = rng_(args.win_layers), rng_(args.post_layers)

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(args.model_name, quantization_config=bnb, trust_remote_code=True,
                                      use_flash_attn=False, low_cpu_mem_usage=True, device_map={"": 0}).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    IMG_CTX = model.img_context_token_id
    lm_layers = model.language_model.model.layers
    tfm = build_transform()
    vdt = next(p_.dtype for p_ in model.vision_model.parameters() if p_.is_floating_point())
    import copy as _copy
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)]
    cand_vals = [d for d, _ in cand]
    cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    st = {"masks": {}, "final": None}
    def mk_pre(L):
        def pre(_m, hargs, hkwargs):
            mk = st["masks"].get(L)
            if mk is None:
                return hargs, hkwargs
            hs = hargs[0] if hargs else hkwargs.get("hidden_states")
            if hs is not None and mk.dtype != hs.dtype:
                mk = mk.to(hs.dtype); st["masks"][L] = mk
            if len(hargs) >= 2:
                hargs = (hargs[0], mk) + tuple(hargs[2:])
            else:
                hkwargs = dict(hkwargs); hkwargs["attention_mask"] = mk
            return hargs, hkwargs
        return pre
    for L in range(len(lm_layers)):
        lm_layers[L].register_forward_pre_hook(mk_pre(L), with_kwargs=True)
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

    arms = ["base", "cutcar_win", "cutcar_post", "cutq_win", "cutimg_win"]
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
            S = int(ids.shape[1]); dev = next(model.parameters()).device
            car = S - 1 - off; last = S - 1
            idrow = ids[0]
            imgpos = torch.nonzero(idrow == IMG_CTX).flatten()
            qspan = [p for p in range(int(imgpos.max().item()) + 1, S) if p not in (car, last)]
            def causal():
                m = torch.zeros(S, S, dtype=torch.float32, device=dev)
                m.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool, device=dev), 1), MIN)
                return m
            def plan(keys, layers):
                m = causal()
                m[last, torch.tensor(keys, dtype=torch.long, device=dev)] = MIN
                m4 = m.view(1, 1, S, S)
                return {L: m4 for L in layers}
            plans = {"base": {},
                     "cutcar_win": plan([car], WIN),
                     "cutcar_post": plan([car], POST),
                     "cutq_win": plan(qspan, WIN),
                     "cutimg_win": plan(imgpos.tolist(), WIN)}
            fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
            for a in arms:
                st["masks"] = plans[a]; st["final"] = None
                with torch.no_grad():
                    outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
                lg = outp.logits[0, -1].float().cpu()
                res[a]["pred"].append(int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())]))
                res[a]["gold"].append(gold)
                reps_final[a].append(st["final"].half())
            st["masks"] = {}
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{args.limit}", flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    gold_a = np.array(res["base"]["gold"])
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.RandomState(0); idx = rng.permutation(n); ntr = int(0.6 * n); te = idx[ntr:]
    lines = [f"=== InternVL EDGE-CUT necessity (n={n}; window {WIN[0]}-{WIN[-1]}; post {POST[0]}-{POST[-1]}) ===",
             f"{'arm':>12} | final decode (per-arm head) | emitted"]
    for a in arms:
        X = torch.stack(reps_final[a]).float().numpy()
        clf = RidgeClassifier(alpha=10.0).fit(X[idx[:ntr]], gold_a[idx[:ntr]])
        dec = float(np.mean(clf.predict(X[te]) == gold_a[te]))
        emi = float(np.mean(np.array(res[a]["pred"]) == gold_a))
        lines.append(f"{a:>12} |          {dec:.3f}          | {emi:.3f}")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"res": res, "reps_final": {a: torch.stack(v) for a, v in reps_final.items()},
                "config": vars(args)}, out / "edges.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
