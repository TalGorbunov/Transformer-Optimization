#!/usr/bin/env python3
"""P1d closure: capture a DEPLOYABLE clean query — q_c from ONE gray-image forward per sample
(same question, same padded length as the joint context; no real frame seen) — for recompute
against the existing joint k/v captures. If d' ~= the grid's clean-q x joint-kv (3.1-3.5), the
2-forward gate option is validated (economics note only; the >=5 wiring bar stays unmet)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers

OFFSET = 9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="existing qkv_2x2 capture dir (for the sample list)")
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--pad-to-frames", type=int, default=8)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()
    cap = Path(args.capture)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    want = {r["sid"] for r in blob["samples"]}
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    qkv = {L: {} for L in Ls}
    for L in Ls:
        def mk(L):
            def hook(_m, _i, o):
                qkv[L]["q"] = o.detach()[0]
            return hook
        layers[L].self_attn.q_proj.register_forward_hook(mk(L))

    from evaluations.helpers.patching_core import build_prompt as _bp
    from PIL import Image
    base_fill = ("The weather report for today mentions mild temperatures, light winds "
                 "and clear skies across the region. ")
    ntok = len(processor.tokenizer.encode(base_fill, add_special_tokens=False))
    filler = base_fill * max(1, round((args.pad_to_frames - 1) * 196 / ntok))
    gray = Image.new("RGB", (392, 392), (128, 128, 128))

    out = {}
    droot = Path(blob["config"]["data_root"])
    for sd in iter_sample_dirs(droot):
        if sd.name not in want:
            continue
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        messages = [{"role": "user", "content": [
            {"type": "text", "text": filler},
            {"type": "image", "image": gray},
            {"type": "text", "text": _bp(q0, num_frames=1)}]}]
        inputs = tgi.move_inputs_to_model_device(dict(processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")))
        s_carrier = int(inputs["input_ids"].shape[1]) - 1 - OFFSET
        with torch.no_grad():
            model(**inputs, use_cache=False)
        out[sid] = {L: qkv[L]["q"][s_carrier].detach().half().cpu() for L in Ls}
        if len(out) % 25 == 0:
            print(f"  {len(out)}/{len(want)}", flush=True)
    torch.save(out, cap / "frameless_q.pt")
    print(f"saved {len(out)} frameless queries -> {cap/'frameless_q.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
