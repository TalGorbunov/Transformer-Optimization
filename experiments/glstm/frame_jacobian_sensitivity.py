#!/usr/bin/env python3
"""Jacobian-sensitivity CONTROL: is the counting failure classical over-squashing (info can't reach the
answer) or our aggregation-SNR failure (info reaches but the sum is unreadable)? We compute the gradient
of the gold-count logit w.r.t. each frame's input representation (residual entering layer 0) and report
the per-frame sensitivity. If sensitivities are HEALTHY and roughly UNIFORM across frames (low CV), then
propagation is fine -> NOT Jacobian-over-squashing -> the wall is the aggregation SNR, not connectivity.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

import experiments.glstm.frame_axis_aggregator_adapter as fa
from experiments.glstm.cache_minimal_frame_reps import frame_labels
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import image_token_groups, get_layers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="steps_in_room")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--grad-layer", type=int, default=0, help="residual entering this layer = 'frame input rep'")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "jacobian")
    return p.parse_args()


def main():
    args = parse_args()
    device = base.resolve_device("cuda"); dtype = base.resolve_dtype("bfloat16", device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    print("loading model ...", flush=True)
    model, processor = base.load_model_and_processor("Qwen/Qwen2.5-VL-7B-Instruct", device, dtype, True)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    model.eval()
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    target = get_layers(model)[args.grad_layer]

    st = {"hs": None, "spans": None}
    def pre_hook(module, hargs, hkwargs):
        def cap(hs):
            if st["spans"] is not None:
                hs = hs.clone(); hs.requires_grad_(True); hs.retain_grad(); st["hs"] = hs
            return hs
        if len(hargs) >= 1:
            return (cap(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = cap(hkwargs["hidden_states"]); return hargs, hkwargs
    target.register_forward_pre_hook(pre_hook, with_kwargs=True)

    splits = fa.declare_splits(args.data_root, args.split, [args.seq_len], [], 0.0, 0.0, 0, None, 12345)
    dirs = [Path(d) for d, _ in splits["train"]][:args.limit]
    import random
    rng = random.Random(0)
    per_frame_sens = []   # [N] normalized sensitivity per example
    ev_sens, nv_sens = [], []
    cvs = []
    n_done = 0
    for d in dirs:
        ex = fa.make_example(d, args.task, rng, eval_mode=True)
        if ex is None:
            continue
        frames, question, gold, nf, states = ex
        meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
        fl = frame_labels(args.task, states, meta)
        if fl is None or len(fl) != len(frames):
            continue
        inputs = fa.build_inputs(processor, frames, question, device)
        ids = inputs["input_ids"]
        spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
        if len(spans) != len(frames):
            continue
        gold_id = int(processor.tokenizer(str(int(gold)), add_special_tokens=False).input_ids[0])
        st["spans"] = spans; st["hs"] = None
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(**inputs, use_cache=False)
            logit = out.logits[0, -1, gold_id]
            logit.backward()
        if st["hs"] is None or st["hs"].grad is None:
            continue
        g = st["hs"].grad[0].float()   # [S, H]
        sens = np.array([float(g[torch.tensor(sp, device=g.device)].norm(dim=-1).mean()) for sp in spans])
        if sens.sum() <= 0:
            continue
        norm = sens / sens.sum()
        per_frame_sens.append(norm)
        cvs.append(float(sens.std() / (sens.mean() + 1e-9)))
        lab = np.asarray([int(x) for x in fl])
        ev_sens.append(float(sens[lab == 1].mean()) if (lab == 1).any() else np.nan)
        nv_sens.append(float(sens[lab == 0].mean()) if (lab == 0).any() else np.nan)
        n_done += 1
        del out, logit, g
        torch.cuda.empty_cache()
        if n_done % 10 == 0:
            print(f"  {n_done} examples", flush=True)

    P = np.stack(per_frame_sens)  # [n, N]
    report = [
        f"n={n_done}  grad_layer=L{args.grad_layer}",
        f"per-frame sensitivity (normalized, mean over examples): {np.round(P.mean(0), 3).tolist()}",
        f"  -> uniform would be {1.0/P.shape[1]:.3f} each; min={P.mean(0).min():.3f} max={P.mean(0).max():.3f}",
        f"coefficient of variation across frames (mean over ex): {np.nanmean(cvs):.3f}  (low => uniform => no squashing)",
        f"evidence vs non-evidence mean sensitivity: {np.nanmean(ev_sens):.4f} vs {np.nanmean(nv_sens):.4f} "
        f"(ratio {np.nanmean(ev_sens)/(np.nanmean(nv_sens)+1e-9):.2f})",
        f"min per-frame sensitivity / max: {P.min():.4f} / {P.max():.4f}  (all >0 => every frame influences the answer)",
    ]
    txt = "\n".join(report)
    print("\n" + txt, flush=True)
    (run_dir / "report.txt").write_text(txt + "\n")
    np.save(run_dir / "per_frame_sens.npy", P)
    print(f"\nwrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
