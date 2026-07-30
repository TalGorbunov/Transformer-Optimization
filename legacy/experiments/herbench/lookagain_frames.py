#!/usr/bin/env python3
"""Per-frame look-again VQA on prepped HERBench AC samples: single-frame forward,
"Is the person performing the action 'X' in this frame? yes/no" -> P(yes) from the
yes/no token logits. Saves lookagain.json per sample dir. Used as an INDEPENDENT
judge to curate a cleanly-binary frame subset (never curate on the probe axis).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

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
    for i, sd in enumerate(dirs):
        meta = json.loads((sd / "meta.json").read_text())
        out = {}
        for fp in sorted(sd.glob("frame_*.jpg")):
            img = Image.open(fp).convert("RGB")
            prompt = (f"Look at this single frame from an egocentric kitchen video.\n"
                      f"Is the person performing the action '{meta['pair']}' in this frame? "
                      f"Answer yes or no.\nAnswer: ")
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([img], prompt))
            with torch.no_grad():
                logits = model(**inputs, use_cache=False).logits[0, -1].float()
            py = torch.logsumexp(logits[yes_ids], 0)
            pn = torch.logsumexp(logits[no_ids], 0)
            out[fp.name] = round(torch.sigmoid(py - pn).item(), 4)
        (sd / "lookagain.json").write_text(json.dumps(out, indent=1))
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(dirs)}]", flush=True)
    print("done:", len(dirs), "samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
