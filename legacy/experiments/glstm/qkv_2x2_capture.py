#!/usr/bin/env python3
"""P1 (2026-07-12): capture for the QUERY/ENCODING 2×2 — the decisive joint-context-tax probe.

msg_f = m_f · o_proj(Σ_j ŵ_j v_j): m_f is excluded (d′ is per-frame-scale-invariant; renorm
refuted), leaving routing ŵ_j (set by the carrier query q_c) × values v_j (frame encoding).
This captures, per sample, PRE-ROTARY q_c / k_j / v_j at L14/L16 from THREE forwards:
  joint : all 8 frames + question (contaminated q AND kv)
  mp    : each frame alone + question (clean-everything anchor)
  pad   : each frame alone, text-padded to 8-frame-equivalent context (the CLEAN arm —
          position/context-length matched so the ~1.9 long-context term can't masquerade)
plus the JOINT forward's rotary cos/sin slices at the carrier + frame positions (recompute
applies ONE consistent geometry to every cell) and the dequantized o_proj matrices (CPU-side
message reconstruction). Analysis: qkv_2x2_analysis.py.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups

OFFSET = 9   # room token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--pad-to-frames", type=int, default=8)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]
    NF = int(args.n_frames)

    # dequantize o_proj once (CPU reconstruction needs dense mats)
    import bitsandbytes.functional as bnbF
    oproj_dense = {}
    for L in Ls:
        w = layers[L].self_attn.o_proj.weight
        oproj_dense[L] = bnbF.dequantize_4bit(w.data, w.quant_state).float().cpu()
    torch.save(oproj_dense, out / "oproj_dense.pt")
    print("o_proj dequantized:", {L: tuple(m.shape) for L, m in oproj_dense.items()}, flush=True)

    qkv: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in Ls}
    posemb = {}
    for L in Ls:
        for nm in ("q_proj", "k_proj", "v_proj"):
            def mk(L, nm):
                def hook(_m, _i, o):
                    qkv[L][nm] = o.detach()[0]
                return hook
            getattr(layers[L].self_attn, nm).register_forward_hook(mk(L, nm))
    def mk_pe(_m, args_, kwargs_):
        pe = kwargs_.get("position_embeddings")
        if pe is None and len(args_) >= 1 and isinstance(args_[-1], tuple):
            pe = args_[-1]
        if pe is not None:
            posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
    layers[Ls[0]].self_attn.register_forward_pre_hook(mk_pe, with_kwargs=True)

    from evaluations.helpers.patching_core import build_prompt as _bp
    base_fill = ("The weather report for today mentions mild temperatures, light winds "
                 "and clear skies across the region. ")
    ntok = len(processor.tokenizer.encode(base_fill, add_special_tokens=False))
    reps = max(1, round((int(args.pad_to_frames) - 1) * 196 / ntok))
    filler = base_fill * reps

    def grab(arm_store, carrier_pos, fpos_list):
        """From the freshly-run forward's hooks: q at carrier, per-frame k/v."""
        for L in Ls:
            st = arm_store.setdefault(L, {"q": {}, "k": {}, "v": {}})
            st["q"]["all"] = qkv[L]["q_proj"][carrier_pos].detach().half().cpu()   # [3584]
            for fi, fpos in enumerate(fpos_list):
                idx = torch.tensor(fpos, dtype=torch.long)
                st["k"][fi] = qkv[L]["k_proj"][idx].detach().half().cpu()          # [196, 512]
                st["v"][fi] = qkv[L]["v_proj"][idx].detach().half().cpu()

    dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(dirs)
    samples = []
    n = 0
    for sd in dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            if not evid or len(frames) < NF:
                continue
            if int(args.resize) > 0:
                frames = [f.resize((int(args.resize), int(args.resize))) for f in frames]
        except Exception:
            continue
        try:
            rec = {"sid": sid, "gold": gold,
                   "labels": [1 if t in evid else 0 for t in range(NF)], "arms": {}}
            # ---- joint forward ----
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
            ids = inputs["input_ids"][0].detach().cpu()
            seq = int(ids.shape[0])
            fg = [[int(p) for p in g] for g in image_token_groups(
                ids, expected_num_frames=NF, processor=processor)]
            carrier = seq - 1 - OFFSET
            with torch.no_grad():
                model(**inputs, use_cache=False)
            joint_store = {}
            grab(joint_store, carrier, fg)
            # rotary slices from the JOINT forward: [3(or1), 1, seq, hd] -> positions
            pos_idx = [carrier] + [p for g in fg for p in g]
            cos = posemb["cos"]; sin = posemb["sin"]
            rec["rope_cos"] = cos[..., pos_idx, :].detach().half().cpu()
            rec["rope_sin"] = sin[..., pos_idx, :].detach().half().cpu()
            rec["joint_carrier"] = carrier
            rec["joint_fg_sizes"] = [len(g) for g in fg]
            rec["arms"]["joint"] = joint_store
            # ---- mp + pad forwards (per frame) ----
            rec["arm_rope"] = {"mp": {}, "pad": {}}
            for arm in ("mp", "pad"):
                store = {}
                for t in range(NF):
                    if arm == "mp":
                        s_inputs = tgi.build_inputs([frames[t]], q0)
                    else:
                        messages = [{"role": "user", "content": [
                            {"type": "text", "text": filler},
                            {"type": "image", "image": frames[t]},
                            {"type": "text", "text": _bp(q0, num_frames=1)}]}]
                        s_inputs = dict(processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=True,
                            return_dict=True, return_tensors="pt"))
                    s_inputs = tgi.move_inputs_to_model_device(s_inputs)
                    s_ids = s_inputs["input_ids"][0].detach().cpu()
                    s_fg = [[int(p) for p in g] for g in image_token_groups(
                        s_ids, expected_num_frames=1, processor=processor)]
                    s_carrier = int(s_ids.shape[0]) - 1 - OFFSET
                    with torch.no_grad():
                        model(**s_inputs, use_cache=False)
                    # own-geometry rope for this arm/frame (carrier slice + frame-token slice)
                    cs = posemb["cos"]; sn = posemb["sin"]; fidx = s_fg[0]
                    rec["arm_rope"][arm][t] = {
                        "cos_c": cs[..., s_carrier:s_carrier + 1, :].detach().half().cpu(),
                        "sin_c": sn[..., s_carrier:s_carrier + 1, :].detach().half().cpu(),
                        "cos_f": cs[..., fidx, :].detach().half().cpu(),
                        "sin_f": sn[..., fidx, :].detach().half().cpu()}
                    # store this frame's single-forward q and its own k/v under frame index t
                    for L in Ls:
                        store.setdefault(L, {}).setdefault("q", {})[t] = \
                            qkv[L]["q_proj"][s_carrier].detach().half().cpu()
                        idx = torch.tensor(s_fg[0], dtype=torch.long)
                        store[L].setdefault("k", {})[t] = \
                            qkv[L]["k_proj"][idx].detach().half().cpu()
                        store[L].setdefault("v", {})[t] = \
                            qkv[L]["v_proj"][idx].detach().half().cpu()
                rec["arms"][arm] = store
        except Exception as exc:
            print(f"{sid} failed: {type(exc).__name__}: {exc}", flush=True)
            fail = globals().get("_f", 0) + 1; globals()["_f"] = fail
            if fail >= 10 and n == 0:
                raise
            continue
        samples.append(rec)
        n += 1
        if n % 10 == 0:
            print(f"  {n}/{args.limit}", flush=True)
        if n % 50 == 0:
            torch.save({"samples": samples, "layers": Ls, "offset": OFFSET,
                        "config": vars(args)}, out / "qkv_capture.pt")
    torch.save({"samples": samples, "layers": Ls, "offset": OFFSET,
                "config": vars(args)}, out / "qkv_capture.pt")
    print(f"saved {len(samples)} samples -> {out/'qkv_capture.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
