#!/usr/bin/env python3
"""Capture the LAST-TOKEN residual state at late layers for steps, seq_len 1..8 (park, 392px),
so a ridge probe can decode the count. Model loaded once; loops seq lengths internally.
Output: outputs/ladder/image_smallN/last_token/N{s}/cache.pt  {states:{L:[n,H]}, gold:[n]}.
"""
import argparse, random, sys
from pathlib import Path
import numpy as np, torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))
from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--layers", default="20,24,27")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-root", default="outputs/ladder/image_smallN/last_token")
    ap.add_argument("--sample-seed", type=int, default=1)  # MUST shuffle: iter_sample_dirs is count-grouped
    args = ap.parse_args()
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.split(",")]
    cap = {}
    for L in Ls:
        layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0].detach()))

    for s in [int(x) for x in args.seq_lens.split(",")]:
        droot = Path(f"data/mmred_images_park/seq_len_{s}/all_uniform")
        dirs = list(iter_sample_dirs(droot))
        random.Random(args.sample_seed).shuffle(dirs)   # de-group counts (matches behavior_vs_n)
        states = {L: [] for L in Ls}; gold = []
        n = 0
        for sd in dirs:
            if n >= args.limit: break
            try:
                sid, frames, q0, st, a0 = load_mmred_sample(sd)
                g = int(str(a0).strip())
                if args.resize > 0:
                    frames = [f.resize((args.resize, args.resize)) for f in frames]
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
                with torch.no_grad():
                    model(**inputs, use_cache=False)
                for L in Ls:
                    states[L].append(cap[L][0, -1].float().cpu().numpy())  # last token
                gold.append(g); n += 1
            except Exception as e:
                print(f"  skip {sd}: {type(e).__name__}", flush=True); continue
        out = Path(args.out_root) / f"N{s}"; out.mkdir(parents=True, exist_ok=True)
        torch.save({"states": {L: np.stack(states[L]) for L in Ls}, "gold": np.array(gold), "layers": Ls},
                   out / "cache.pt")
        print(f"seq {s}: saved {n} -> {out/'cache.pt'}", flush=True)

if __name__ == "__main__":
    main()
