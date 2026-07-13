#!/usr/bin/env python3
"""Per-frame look-again judge on prepped MLVU-AC samples (per-frame evidence labels for the
d'/parity instruments — the released jsons carry no insertion GT and dup-detect only covers
~3%). Judges an N-frame uniform subsample of the stored 128 per question:
"Is the action 'X' being performed in this frame? yes/no" -> P(yes). Writes lookagain_N{N}.json
per sample dir (frame index -> prob)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mlvu_ac")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from evaluations.helpers import patching_core as tgi
    from evaluations.scripts.patch_importence import group_restoration_importance as gri
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    yes_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("yes", "Yes")]
    no_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("no", "No")]

    dirs = sorted(d for d in Path(args.data_root).iterdir() if (d / "meta.json").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    NF = int(args.n_frames)
    for i, sd in enumerate(dirs):
        outp = sd / f"lookagain_N{NF}.json"
        if outp.exists():
            continue
        meta = json.loads((sd / "meta.json").read_text())
        action = meta.get("action") or "the queried action"
        idx = np.linspace(0, meta["n_frames"] - 1, NF).round().astype(int)
        out = {}
        for t in idx:
            img = Image.open(sd / f"frame_{t:03d}.jpg").convert("RGB")
            prompt = (f"Look at this single frame from a video.\n"
                      f"Is the action '{action}' being performed in this frame? "
                      f"Answer yes or no.\nAnswer: ")
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([img], prompt))
            with torch.no_grad():
                logits = model(**inputs, use_cache=False).logits[0, -1].float()
            py = torch.logsumexp(logits[yes_ids], 0)
            pn = torch.logsumexp(logits[no_ids], 0)
            out[int(t)] = round(torch.sigmoid(py - pn).item(), 4)
        outp.write_text(json.dumps(out, indent=1))
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(dirs)}]", flush=True)
    print("done:", len(dirs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
