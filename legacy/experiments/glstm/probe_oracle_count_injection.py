#!/usr/bin/env python3
"""#3 oracle-count injection: write a CLEAN oracle count into the last-token residual at L19 (along the
model's own decodable count direction v_count, fit from the cache) and read the native digit output.
Isolates 'can the readout verbalize a perfect count' (readout) from 'can we compute it' (aggregation).
Sweeps injection strength; reports emitted-count vs oracle accuracy. Frames-first (deployed) layout."""
from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import experiments.glstm.frame_axis_aggregator_adapter as fa
from experiments.glstm.cache_frames_first_reps import build_inputs_frames_first
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import image_token_groups
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def fit_count_dir(cache_path, seq_len, layer):
    import torch as T
    cache = T.load(cache_path, map_location="cpu")
    X, y = [], []
    for v in cache.values():
        if int(v.get("seq_len", -1)) != seq_len:
            continue
        # layersweep cache has query_by_layer[L]; minimal cache has query_rep (single layer)
        q = v["query_by_layer"][layer] if "query_by_layer" in v else v["query_rep"]
        X.append(q.float().numpy()); y.append(int(v["gold"]))
    X = np.stack(X); y = np.asarray(y)
    sc = StandardScaler().fit(X)
    r = Ridge(alpha=1.0).fit(sc.transform(X), y)
    w = r.coef_ / sc.scale_                       # direction in raw space
    vhat = w / (np.linalg.norm(w) + 1e-9)
    proj = X @ vhat
    a = np.polyfit(y, proj, 1)                    # proj ~ a[0]*g + a[1]
    return vhat.astype(np.float32), float(a[0]), float(a[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcount-cache", required=True)
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    ap.add_argument("--task", default="steps_in_room")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--read-layer", type=int, default=19)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--alphas", default="0,1,2,4")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.replace(",", " ").split()]
    device = base.resolve_device("cuda"); dtype = base.resolve_dtype("bfloat16", device)
    vhat_np, slope, intercept = fit_count_dir(args.vcount_cache, args.seq_len, args.read_layer)
    print(f"fit count-dir: slope(proj/count)={slope:.3f} intercept={intercept:.3f}", flush=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, True)
    for p in model.parameters():
        p.requires_grad_(False)
    vhat = torch.tensor(vhat_np, device=device, dtype=torch.float32)
    digit_ids = [processor.tokenizer(str(d), add_special_tokens=False).input_ids[0] for d in range(args.seq_len + 1)]

    st = {"pos": -1, "g": 0, "alpha": 0.0}
    def pre_hook(module, hargs, hkwargs):
        def edit(hs):
            if st["alpha"] == 0.0 or hs.shape[1] <= st["pos"]:
                return hs
            x = hs[0, st["pos"], :].float()
            cur = float(x @ vhat); tgt = slope * st["g"] + intercept
            hs = hs.clone()
            hs[0, st["pos"], :] = (x + st["alpha"] * (tgt - cur) * vhat).to(hs.dtype)
            return hs
        if len(hargs) >= 1:
            return (edit(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = edit(hkwargs["hidden_states"]); return hargs, hkwargs
    fa.get_layers(model)[args.read_layer].register_forward_pre_hook(pre_hook, with_kwargs=True)

    splits = fa.declare_splits(args.data_root, "all_uniform", [args.seq_len], [], 0.0, 0.0, 0, None, 12345)
    dirs = [Path(d) for d, _ in splits["train"]][:args.limit]
    import random
    rng = random.Random(0)
    res = {a: [0, 0] for a in alphas}  # alpha -> [correct, n]
    for d in dirs:
        ex = fa.make_example(d, args.task, rng, eval_mode=True)
        if ex is None:
            continue
        frames, question, gold, nf, states = ex
        inputs = build_inputs_frames_first(processor, frames, question, device)
        ids = inputs["input_ids"]
        if len(image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)) != len(frames):
            continue
        st["pos"] = int(ids.shape[1]) - 1; st["g"] = int(gold)
        for a in alphas:
            st["alpha"] = a
            with torch.inference_mode():
                out = model(**inputs, use_cache=False)
            dl = out.logits[0, -1, digit_ids]
            pred = int(torch.argmax(dl))
            res[a][0] += int(pred == gold); res[a][1] += 1
    print("\nalpha  emitted-count acc vs oracle  (n)")
    for a in alphas:
        c, n = res[a]
        print(f"{a:>5}  {c/max(1,n):.3f}  (n={n})")
    print("\nreading: alpha=0 is the frozen model; if acc rises with alpha, the count direction is "
          "CAUSAL/verbalizable (readout can use a clean count); if flat, decodable!=causal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
