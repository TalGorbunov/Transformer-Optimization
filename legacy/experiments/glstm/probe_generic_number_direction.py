#!/usr/bin/env python3
"""Refined Solution-3 probe: fit the number direction from the LM's GENERIC numeracy (not the
collapsed MMRED task reps), then test whether it extrapolates.

Source = simple arithmetic prompts ("What is a plus b?") where the model is about to emit a number.
We read the residual at L_READ at the answer position, fit a linear direction on numbers {fit}, then:
  (READ)  predict held-out higher numbers -> is the GENERIC number axis linear past the fit range?
  (STEER) force the readout to a target number / sweep magnitude -> does the frozen LM EMIT it,
          including numbers never used to fit the direction?
If this generic axis extrapolates where the task-count axis (probe_count_direction_extrapolation)
compressed, then: compute the count externally (extensive sum) + inject along THIS axis = a working,
extrapolating "improve-the-model" injection.
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
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]


def parse_args():
    p = argparse.ArgumentParser(description="Generic number-direction extrapolation probe (refined Solution 3).")
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--fit-nums", default="0,1,2,3,4")
    p.add_argument("--test-nums", default="5,6,7,8")
    p.add_argument("--n-per-num", type=int, default=40)
    p.add_argument("--n-steer", type=int, default=20)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--scale-sweep", default="0,0.5,1,2,4,8,16")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "generic_number_direction")
    return p.parse_args()


def prompt_for(processor, a, b, device):
    msgs = [{"role": "user", "content": [{"type": "text",
             "text": f"What is {a} plus {b}? Reply with only the number."}]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
    return base.move_inputs_to_device(dict(inp), device)


def pairs_for(k, rng, n):
    """sample n (a,b) with a+b==k, a,b in 0..8."""
    opts = [(a, k - a) for a in range(0, k + 1) if 0 <= k - a <= 8]
    return [opts[rng.randrange(len(opts))] for _ in range(n)] if opts else []


def main():
    args = parse_args()
    fitN = [int(x) for x in str(args.fit_nums).replace(",", " ").split()]
    testN = [int(x) for x in str(args.test_nums).replace(",", " ").split()]
    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    run = args.output / (time.strftime("%Y%m%d_%H%M%S") + f"_L{args.read_layer}")
    run.mkdir(parents=True, exist_ok=True)
    emit = lambda m: print(m, flush=True)

    emit(f"loading {args.model_name} ...")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    for p_ in model.parameters():
        p_.requires_grad_(False)
    target = fa.get_layers(model)[int(args.read_layer)]

    st = {"mode": "read", "cur_pos": -1, "captured": None, "target": None,
          "w": None, "b": 0.0, "wn": None, "vec": None}

    def edit(hs):
        cp = st["cur_pos"]
        if hs.shape[1] <= cp:
            return hs
        if st["mode"] == "read":
            st["captured"] = hs[0, cp, :].detach().float().cpu().clone(); return hs
        if st["mode"] == "writevec" and st["vec"] is not None:
            hs = hs.clone(); hs[0, cp, :] = hs[0, cp, :] + st["vec"].to(hs.dtype); return hs
        if st["mode"] == "write" and st["w"] is not None:
            h = hs[0, cp, :].float(); delta = float(st["target"]) - (float(st["w"] @ h) + st["b"])
            hs = hs.clone(); hs[0, cp, :] = hs[0, cp, :] + (delta * st["wn"]).to(hs.dtype)
        return hs

    def pre_hook(module, hargs, hkwargs):
        if len(hargs) >= 1:
            return (edit(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = edit(hkwargs["hidden_states"]); return hargs, hkwargs
    target.register_forward_pre_hook(pre_hook, with_kwargs=True)

    rng = random.Random(0)
    # ---- READ pass: residuals for "about to emit K" across arithmetic prompts ----
    reps = defaultdict(list)
    st["mode"] = "read"
    for k in fitN + testN:
        for (a, b) in pairs_for(k, rng, args.n_per_num):
            inp = prompt_for(processor, a, b, device)
            st["cur_pos"] = int(inp["input_ids"].shape[1]) - 1; st["captured"] = None
            with torch.inference_mode():
                model(**inp, use_cache=False)
            if st["captured"] is not None:
                reps[k].append(st["captured"].numpy())
    emit("captured reps per number: " + ", ".join(f"{k}:{len(reps[k])}" for k in sorted(reps)))

    Xf, yf = [], []
    for k in fitN:
        for v in reps[k]:
            Xf.append(v); yf.append(k)
    Xf = np.asarray(Xf, np.float64); yf = np.asarray(yf, np.float64)
    mu = Xf.mean(0); Xc = Xf - mu
    A = Xc.T @ Xc + args.ridge * Xc.shape[0] * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ (yf - yf.mean())); b = float(yf.mean() - w @ mu)
    pred = lambda v: float(w @ v) + b

    read_rows = ["num,split,n,mean_pred,acc_round"]
    emit(f"=== READ: generic number direction fit on {fitN} ===")
    for k in sorted(set(fitN + testN)):
        ps = [pred(v) for v in reps[k]]
        if not ps:
            continue
        acc = float(np.mean([round(p) == k for p in ps])); tag = "FIT" if k in fitN else "TEST"
        emit(f"  [{tag}] number {k}: mean_pred={np.mean(ps):6.2f}  acc_round={acc:.2f}  n={len(ps)}")
        read_rows.append(f"{k},{tag},{len(ps)},{np.mean(ps):.3f},{acc:.3f}")
    (run / "read_extrapolation.csv").write_text("\n".join(read_rows) + "\n")

    # steer prompts: arithmetic prompts (varied true answers) we will OVERRIDE
    steer = [pairs_for(rng.randrange(0, 9), rng, 1)[0] for _ in range(args.n_steer)]

    # ---- STEER (target): force readout to c, does the LM emit c (incl. extrapolated)? ----
    st["w"] = torch.tensor(w, dtype=torch.float32, device=device); st["b"] = b
    st["wn"] = torch.tensor(w / (w @ w), dtype=torch.float32, device=device)
    write_rows = ["target,split,n,emit_mean,acc"]
    emit(f"=== STEER(target): force readout to c at L{args.read_layer}, read emitted number ===")
    for ct in sorted(set(fitN + testN)):
        ems = []
        for (a, b) in steer:
            inp = prompt_for(processor, a, b, device)
            st["cur_pos"] = int(inp["input_ids"].shape[1]) - 1; st["mode"] = "write"; st["target"] = ct
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=4, do_sample=False)
            st["mode"] = "read"; st["target"] = None
            m = INT_RE.search(processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
            if m:
                ems.append(int(m.group(0)))
        if ems:
            acc = float(np.mean([e == ct for e in ems])); tag = "FIT" if ct in fitN else "TEST"
            emit(f"  [{tag}] target {ct}: emit_mean={np.mean(ems):6.2f}  acc={acc:.2f}  n={len(ems)}")
            write_rows.append(f"{ct},{tag},{len(ems)},{np.mean(ems):.3f},{acc:.3f}")
    (run / "steer_target.csv").write_text("\n".join(write_rows) + "\n")

    # ---- DOSE-RESPONSE: inject a*rms*unit(w), does emitted number track a? ----
    scales = [float(x) for x in str(args.scale_sweep).replace(",", " ").split()] if args.scale_sweep else []
    if scales:
        u = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.float32, device=device)
        dose_rows = ["scale_signed,n,emit_mean,emit_min,emit_max"]
        emit(f"=== DOSE-RESPONSE: inject a*rms*unit(w) at L{args.read_layer} ===")
        for a_ in sorted(set([-s for s in scales] + scales)):
            ems = []
            for (a, b) in steer:
                inp = prompt_for(processor, a, b, device)
                cp = int(inp["input_ids"].shape[1]) - 1
                st["cur_pos"] = cp; st["mode"] = "read"; st["captured"] = None
                with torch.inference_mode():
                    model(**inp, use_cache=False)
                rms = float(np.sqrt(np.mean(st["captured"].numpy() ** 2))) if st["captured"] is not None else 1.0
                st["vec"] = (a_ * rms) * u; st["mode"] = "writevec"
                with torch.inference_mode():
                    out = model.generate(**inp, max_new_tokens=4, do_sample=False)
                st["mode"] = "read"; st["vec"] = None
                m = INT_RE.search(processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
                if m:
                    ems.append(int(m.group(0)))
            if ems:
                emit(f"  a={a_:+.2f}: emit_mean={np.mean(ems):6.2f}  [{min(ems)}..{max(ems)}]  n={len(ems)}")
                dose_rows.append(f"{a_:.3f},{len(ems)},{np.mean(ems):.3f},{min(ems)},{max(ems)}")
        (run / "dose_response.csv").write_text("\n".join(dose_rows) + "\n")
    emit(f"wrote {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
