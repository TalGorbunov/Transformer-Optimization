#!/usr/bin/env python3
"""Solution-3 probe: is there a LINEAR count-direction at L_READ that EXTRAPOLATES to unseen counts?

Fits a linear count direction `w` on residuals (input to decoder layer L_READ, at the answer position)
for the FIT counts only, then asks two questions:
  (READ)  predict held-out higher counts -> does the linear readout keep tracking, or saturate?
  (WRITE) steer the L_READ residual along `w` to FORCE an (extrapolated) count and check whether the
          frozen LM actually EMITS that number.
If both extrapolate, Solution 3 (inject c*d, let the frozen LM verbalize via its own numeracy) is
viable. A per-count codebook cannot extrapolate by construction (no vector for unseen counts), so the
read-side linear test is itself the codebook contrast.

Frozen 7B, reuses frame_axis_aggregator_adapter for model loading / input building / the L_READ site.
"""
from __future__ import annotations
import argparse, random, re, sys, time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch

import experiments.glstm.frame_axis_aggregator_adapter as fa
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base

INT_RE = re.compile(r"-?\d+")


def parse_args():
    p = argparse.ArgumentParser(description="Linear count-direction extrapolation probe (Solution 3).")
    p.add_argument("--task", default="steps_in_room")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--fit-counts", default="0,1,2,3,4")
    p.add_argument("--test-counts", default="5,6,7,8")
    p.add_argument("--n-per-count", type=int, default=40)
    p.add_argument("--n-steer", type=int, default=24)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--scale-sweep", default="",
                   help="if set (e.g. '0,0.5,1,2,4,8'): inject ±a*rms*unit(w) at cur_pos and report the "
                        "dose-response of the emitted number -- a magnitude-calibrated causal-control test.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "count_direction_extrap")
    return p.parse_args()


def main():
    args = parse_args()
    fitC = [int(x) for x in str(args.fit_counts).replace(",", " ").split()]
    testC = [int(x) for x in str(args.test_counts).replace(",", " ").split()]
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    run = args.output / (time.strftime("%Y%m%d_%H%M%S") + f"_{args.task}")
    run.mkdir(parents=True, exist_ok=True)

    def emit(m):
        print(m, flush=True)

    emit(f"loading {args.model_name} (4bit={args.load_in_4bit}) ...")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    for p_ in model.parameters():
        p_.requires_grad_(False)
    layers = fa.get_layers(model); target = layers[int(args.read_layer)]

    # hook: read -> capture L_READ residual at cur_pos (prefill only); write -> steer it to a target count
    st = {"mode": "read", "cur_pos": -1, "captured": None, "target": None,
          "w": None, "b": 0.0, "wn": None, "vec": None}

    def edit(hs):
        cp = st["cur_pos"]
        if hs.shape[1] <= cp:   # decode step (KV cache) -> nothing to edit
            return hs
        if st["mode"] == "read":
            st["captured"] = hs[0, cp, :].detach().float().cpu().clone()
            return hs
        if st["mode"] == "writevec" and st["vec"] is not None:
            hs = hs.clone()
            hs[0, cp, :] = hs[0, cp, :] + st["vec"].to(hs.dtype)
            return hs
        if st["mode"] == "write" and st["w"] is not None:
            h = hs[0, cp, :].float()
            cur = float(st["w"] @ h) + st["b"]
            delta = (float(st["target"]) - cur)
            hs = hs.clone()
            hs[0, cp, :] = hs[0, cp, :] + (delta * st["wn"]).to(hs.dtype)
        return hs

    def pre_hook(module, hargs, hkwargs):
        if len(hargs) >= 1:
            return (edit(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = edit(hkwargs["hidden_states"])
        return hargs, hkwargs
    target.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # ---- gather residuals by gold count (READ pass) ----
    splits = fa.declare_splits(args.data_root, args.split, [args.seq_len], [], 0.0, 0.0, 0, None, args.split_seed)
    pool = list(splits["train"]); random.Random(0).shuffle(pool)
    reps = defaultdict(list)
    cap = args.n_per_count
    rng = random.Random(0)
    st["mode"] = "read"
    for dstr, sl in pool:
        if all(len(reps[c]) >= cap for c in fitC + testC):
            break
        ex = fa.make_example(Path(dstr), args.task, rng, eval_mode=True)
        if ex is None:
            continue
        frames, question, gold, nf, states = ex
        if gold not in fitC + testC or len(reps[gold]) >= cap:
            continue
        inputs = fa.build_inputs(processor, frames, question, device)
        st["cur_pos"] = int(inputs["input_ids"].shape[1]) - 1
        st["captured"] = None
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if st["captured"] is not None:
            reps[gold].append(st["captured"].numpy())
    emit("captured reps per count: " + ", ".join(f"{c}:{len(reps[c])}" for c in sorted(reps)))

    # ---- fit ridge regression count ~ w.x + b on FIT counts ----
    Xf, yf = [], []
    for c in fitC:
        for v in reps[c]:
            Xf.append(v); yf.append(c)
    Xf = np.asarray(Xf, np.float64); yf = np.asarray(yf, np.float64)
    mu = Xf.mean(0); Xc = Xf - mu
    A = Xc.T @ Xc + args.ridge * Xc.shape[0] * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ (yf - yf.mean()))
    b = float(yf.mean() - w @ mu)
    pred = lambda v: float(w @ v) + b

    # ---- READ test: does the linear readout extrapolate to held-out counts? ----
    read_rows = ["count,split,n,mean_pred,acc_round"]
    emit(f"=== READ: linear count-direction fit on {fitC} (codebook could NOT do this) ===")
    for c in sorted(set(fitC + testC)):
        ps = [pred(v) for v in reps[c]]
        if not ps:
            continue
        acc = float(np.mean([round(p) == c for p in ps]))
        tag = "FIT" if c in fitC else "TEST"
        emit(f"  [{tag}] count {c}: mean_pred={np.mean(ps):6.2f}  acc_round={acc:.2f}  n={len(ps)}")
        read_rows.append(f"{c},{tag},{len(ps)},{np.mean(ps):.3f},{acc:.3f}")
    (run / "read_extrapolation.csv").write_text("\n".join(read_rows) + "\n")

    # ---- WRITE/steer test: force the residual to a target count, does the LM emit it? ----
    st["w"] = torch.tensor(w, dtype=torch.float32, device=device)
    st["b"] = b
    st["wn"] = torch.tensor(w / (w @ w), dtype=torch.float32, device=device)  # +1 in readout per unit move
    steer = []
    for dstr, sl in pool:
        ex = fa.make_example(Path(dstr), args.task, rng, eval_mode=True)
        if ex is not None:
            steer.append(ex)
        if len(steer) >= args.n_steer:
            break

    # ---- magnitude-calibrated dose-response: inject a*rms*unit(w), does emitted number track a? ----
    scales = [float(x) for x in str(args.scale_sweep).replace(",", " ").split()] if args.scale_sweep else []
    if scales:
        u = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.float32, device=device)
        sweep_rows = ["scale_signed,n,emit_mean,emit_min,emit_max"]
        emit(f"=== DOSE-RESPONSE: inject a*rms*unit(w) at L{args.read_layer} (does emit track a?) ===")
        signed = sorted(set([-s for s in scales] + scales))
        for a in signed:
            ems = []
            for frames, question, gold, nf, states in steer:
                inputs = fa.build_inputs(processor, frames, question, device)
                cp = int(inputs["input_ids"].shape[1]) - 1
                st["cur_pos"] = cp; st["mode"] = "read"; st["captured"] = None
                with torch.inference_mode():
                    model(**inputs, use_cache=False)
                rms = float(np.sqrt(np.mean(st["captured"].numpy() ** 2))) if st["captured"] is not None else 1.0
                st["vec"] = (a * rms) * u; st["mode"] = "writevec"
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
                st["mode"] = "read"; st["vec"] = None
                m = INT_RE.search(processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
                if m:
                    ems.append(int(m.group(0)))
            if ems:
                emit(f"  a={a:+.2f}: emit_mean={np.mean(ems):6.2f}  [{min(ems)}..{max(ems)}]  n={len(ems)}")
                sweep_rows.append(f"{a:.3f},{len(ems)},{np.mean(ems):.3f},{min(ems)},{max(ems)}")
        (run / "dose_response.csv").write_text("\n".join(sweep_rows) + "\n")
        emit(f"wrote {run}")
        return 0
    write_rows = ["target,split,n,emit_mean,acc"]
    emit(f"=== WRITE: steer L{args.read_layer} residual to a FORCED count, read emitted number ===")
    for ct in sorted(set(fitC + testC)):
        ems = []
        for frames, question, gold, nf, states in steer:
            inputs = fa.build_inputs(processor, frames, question, device)
            st["cur_pos"] = int(inputs["input_ids"].shape[1]) - 1
            st["mode"] = "write"; st["target"] = ct
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            st["mode"] = "read"; st["target"] = None
            gen = out[0, inputs["input_ids"].shape[1]:]
            m = INT_RE.search(processor.tokenizer.decode(gen, skip_special_tokens=True))
            if m:
                ems.append(int(m.group(0)))
        if ems:
            acc = float(np.mean([e == ct for e in ems]))
            tag = "FIT" if ct in fitC else "TEST"
            emit(f"  [{tag}] target {ct}: emit_mean={np.mean(ems):6.2f}  acc={acc:.2f}  n={len(ems)}")
            write_rows.append(f"{ct},{tag},{len(ems)},{np.mean(ems):.3f},{acc:.3f}")
    (run / "write_steer.csv").write_text("\n".join(write_rows) + "\n")
    emit(f"wrote {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
