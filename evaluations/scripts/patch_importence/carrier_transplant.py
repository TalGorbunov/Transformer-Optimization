#!/usr/bin/env python3
"""Causal #2: REAL-STATE carrier transplant. No synthetic directions anywhere.

Pair samples that share the exact question text but differ in gold count. Run the donor, capture its
carrier-token hidden state entering layers L+1 (L in --multi-layers). Run the recipient with that state
transplanted (or alpha-blended) at the same positions. If the carrier state IS the count, the emitted /
repaired-read answer follows the DONOR's gold; interpolation should step monotonically.

Arms: base | tx:a for alpha in --alphas (1.0 = full replace) | txsame (same-gold donor, control:
transplant should be ~harmless).
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
from collections import defaultdict
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
    ap.add_argument("--multi-layers", default="14,16,18,20")
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--min-dg", type=int, default=2, help="min |gold_donor - gold_recipient|")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/probes/carrier_transplant")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    mls = [int(x) for x in args.multi_layers.replace(",", " ").split()]
    alphas = [float(x) for x in args.alphas.replace(",", " ").split()]

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    tok = processor.tokenizer
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)
            if len(tok.encode(str(d), add_special_tokens=False)) == 1]
    cand_vals = [d for d, _ in cand]; cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    st = {"mode": "idle", "pos": None, "alpha": 0.0, "donor": {}, "cap": {}, "final": None}

    def mk_pre(L):
        def pre(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is None or st["pos"] is None or hs.shape[1] <= st["pos"]:
                return hargs, hkwargs
            if st["mode"] == "capture":
                st["cap"][L] = hs[0, st["pos"], :].detach().float().clone()
                return hargs, hkwargs
            if st["mode"] == "tx" and L in st["donor"]:
                hs = hs.clone()
                a = st["alpha"]
                d = st["donor"][L].to(device=hs.device)
                h = hs[0, st["pos"], :].float()
                hs[0, st["pos"], :] = ((1 - a) * h + a * d).to(hs.dtype)
                if hargs:
                    return (hs,) + tuple(hargs[1:]), hkwargs
                hkwargs = dict(hkwargs); hkwargs["hidden_states"] = hs
            return hargs, hkwargs
        return pre
    for L in mls:
        layers[L + 1].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    def final_hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        st["final"] = h[0, -1, :].float().cpu().clone()
    layers[-1].register_forward_hook(final_hook)

    # index samples by exact question text
    print("indexing samples by question ...")
    by_q = defaultdict(list)
    for sd in iter_sample_dirs(Path(args.data_root)):
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            g = int(str(a0).strip())
        except Exception:
            continue
        by_q[q0].append((sd, g))
    rng = random.Random(args.sample_seed)
    pairs, same_pairs = [], []
    for q, lst in by_q.items():
        rng.shuffle(lst)
        for i in range(0, len(lst) - 1, 2):
            (sd1, g1), (sd2, g2) = lst[i], lst[i + 1]
            if abs(g1 - g2) >= args.min_dg:
                pairs.append((sd1, g1, sd2, g2, q))
            elif g1 == g2:
                same_pairs.append((sd1, g1, sd2, g2, q))
    rng.shuffle(pairs); rng.shuffle(same_pairs)
    pairs = pairs[: args.pairs]; same_pairs = same_pairs[: max(20, args.pairs // 4)]
    print(f"{len(by_q)} distinct questions; {len(pairs)} diff-gold pairs (|dg|>={args.min_dg}); "
          f"{len(same_pairs)} same-gold control pairs")

    def fwd(inputs):
        st["final"] = None
        with torch.no_grad():
            o = model(**inputs, use_cache=False)
        lg = o.logits[0, -1].float().cpu()
        return int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])

    arm_names = ["base"] + [f"tx:{a:g}" for a in alphas]
    res = {a: defaultdict(list) for a in arm_names + ["txsame"]}
    reps_final = {a: [] for a in arm_names + ["txsame"]}
    fails = 0
    for kind, plist in (("diff", pairs), ("same", same_pairs)):
        for (sd_r, g_r, sd_d, g_d, q) in plist:
            try:
                _, fr_r, q_r, _, _ = load_mmred_sample(sd_r)
                _, fr_d, q_d, _, _ = load_mmred_sample(sd_d)
                in_r = tgi.move_inputs_to_model_device(tgi.build_inputs(fr_r, q_r))
                in_d = tgi.move_inputs_to_model_device(tgi.build_inputs(fr_d, q_d))
                s_r = int(in_r["input_ids"].shape[1]); s_d = int(in_d["input_ids"].shape[1])
                if s_r != s_d:
                    continue
                # donor capture
                st["pos"] = s_d - 1 - args.offset
                st["mode"], st["cap"] = "capture", {}
                fwd(in_d)
                st["donor"] = {L: v for L, v in st["cap"].items()}
                if kind == "diff":
                    st["mode"] = "idle"
                    p = fwd(in_r)
                    res["base"]["pred"].append(p); res["base"]["g_r"].append(g_r); res["base"]["g_d"].append(g_d)
                    reps_final["base"].append(st["final"].half())
                    for a in alphas:
                        st["mode"], st["alpha"] = "tx", a
                        p = fwd(in_r)
                        nm = f"tx:{a:g}"
                        res[nm]["pred"].append(p); res[nm]["g_r"].append(g_r); res[nm]["g_d"].append(g_d)
                        reps_final[nm].append(st["final"].half())
                else:
                    st["mode"], st["alpha"] = "tx", 1.0
                    p = fwd(in_r)
                    res["txsame"]["pred"].append(p); res["txsame"]["g_r"].append(g_r); res["txsame"]["g_d"].append(g_d)
                    reps_final["txsame"].append(st["final"].half())
                st["mode"] = "idle"
            except Exception as exc:
                fails += 1
                print(f"pair failed: {type(exc).__name__}: {exc}")
                if fails >= 25 and not res["base"]["pred"]:
                    raise RuntimeError("25 consecutive failures with 0 successes — aborting")
                continue
            nb = len(res["base"]["pred"])
            if nb and nb % 20 == 0:
                for nm in ("base", "tx:1"):
                    p = np.array(res[nm]["pred"]); gr = np.array(res[nm]["g_r"]); gd = np.array(res[nm]["g_d"])
                    if len(p):
                        print(f"  n={nb} {nm}: acc_vs_recipient={np.mean(p==gr):.3f} acc_vs_DONOR={np.mean(p==gd):.3f}", flush=True)

    lines = [f"=== CARRIER TRANSPLANT (offset {args.offset}, layers {mls}, |dg|>={args.min_dg}) ==="]
    for nm in arm_names + ["txsame"]:
        p = np.array(res[nm]["pred"]); gr = np.array(res[nm]["g_r"]); gd = np.array(res[nm]["g_d"])
        if not len(p):
            continue
        mae_r = float(np.mean(np.abs(p - gr))); mae_d = float(np.mean(np.abs(p - gd)))
        lines.append(f"  {nm:>8} n={len(p)}  acc_vs_recip={np.mean(p==gr):.3f} acc_vs_donor={np.mean(p==gd):.3f}  "
                     f"MAE_recip={mae_r:.2f} MAE_donor={mae_d:.2f}  mean_pred={p.mean():.2f}")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"res": {k: dict(v) for k, v in res.items()},
                "reps_final": {k: torch.stack(v) for k, v in reps_final.items() if v},
                "config": vars(args)}, out / "transplant.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
