#!/usr/bin/env python3
"""CPU probe of a candidate backbone's token layout — the facts a core.backbones spec is written
from: text-config geometry, processor / image-processor knobs, delimiter token ids, and how a
3-frame 512 px prompt tokenises under the question-first layout (tokens per frame, the tokens
around each image run, which non-text inputs the processor returns). Needs only the config /
tokenizer / processor files in the HF cache, not the weights.
Usage:  python experiments/backbone_probe.py --model-id OpenGVLab/InternVL3_5-8B-HF --out outputs/scaleup/_backbones
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CANDIDATES = ["<img>", "</img>", "<IMG_CONTEXT>", "<|im_end|>", "<|im_start|>", "<start_of_image>", "<end_of_image>",
              "<image_soft_token>", "<end_of_turn>", "<start_of_turn>", "<|vision_start|>", "<|vision_end|>",
              "<|image_pad|>", "[IMG]", "[IMG_END]", "[IMG_BREAK]", "<image>"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--set-image-processor", nargs="*", default=[], help="k=v attributes set on the image processor (e.g. crop_to_patches=False)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    rep = {"model_id": args.model_id}
    cfg = AutoConfig.from_pretrained(args.model_id)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    rep["architectures"] = cfg.architectures
    rep["text"] = {k: getattr(tc, k, None) for k in ("model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                                                      "hidden_size", "head_dim", "sliding_window", "rope_scaling", "max_position_embeddings")}
    lt = getattr(tc, "layer_types", None)
    rep["layer_types"] = dict(Counter(lt)) if lt else None
    rep["layer_types_head"] = list(lt[:12]) if lt else None
    for k in ("image_token_id", "image_seq_length", "mm_tokens_per_image", "boi_token_index", "eoi_token_index", "downsample_ratio"):
        if getattr(cfg, k, None) is not None:
            rep[k] = getattr(cfg, k)
    p = AutoProcessor.from_pretrained(args.model_id)
    tok, ip = p.tokenizer, p.image_processor
    for kv in args.set_image_processor:
        k, v = kv.split("=", 1)
        setattr(ip, k, json.loads(v.lower()) if v.lower() in ("true", "false") or v.lstrip("-").isdigit() else v)
    rep["processor"] = type(p).__name__
    rep["image_processor"] = type(ip).__name__
    rep["image_processor_attrs"] = {k: str(getattr(ip, k)) for k in ("crop_to_patches", "min_patches", "max_patches", "size", "do_pan_and_scan",
                                                                     "do_resize", "max_pixels", "min_pixels", "patch_size") if hasattr(ip, k)}
    rep["processor_attrs"] = {k: str(getattr(p, k)) for k in ("image_seq_length", "start_image_token", "end_image_token", "image_token",
                                                              "boi_token", "eoi_token", "full_image_sequence") if hasattr(p, k)}
    unk = tok.unk_token_id
    rep["token_ids"] = {t: tok.convert_tokens_to_ids(t) for t in CANDIDATES
                        if tok.convert_tokens_to_ids(t) is not None and tok.convert_tokens_to_ids(t) != unk}
    frames = [Image.new("RGB", tuple(args.size)) for _ in range(args.frames)]
    q = "How many steps did Daniel spend in the Kitchen?"
    msgs = [{"role": "user", "content": [{"type": "text", "text": q}] + [{"type": "image", "image": f} for f in frames]}]
    enc = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    rep["enc_keys"] = {k: (list(v.shape), str(v.dtype)) for k, v in enc.items() if hasattr(v, "shape")}
    rep["seq_len"] = len(ids)
    top_id, top_n = Counter(ids).most_common(1)[0]
    rep["placeholder"] = {"token": tok.convert_ids_to_tokens(top_id), "id": top_id, "count": top_n, "per_frame": top_n / args.frames}
    runs, i = [], 0
    while i < len(ids):
        if ids[i] == top_id:
            j = i
            while j < len(ids) and ids[j] == top_id:
                j += 1
            runs.append({"start": i, "end": j, "len": j - i,
                         "before": tok.convert_ids_to_tokens(ids[max(0, i - 4):i]), "after": tok.convert_ids_to_tokens(ids[j:j + 4])})
            i = j
        else:
            i += 1
    rep["runs"] = runs
    rep["positions"] = {t: [k for k, x in enumerate(ids) if x == i_] for t, i_ in rep["token_ids"].items() if i_ in ids}
    rep["head_text"] = tok.decode(ids[: runs[0]["start"] + 1]) if runs else tok.decode(ids[:80])
    rep["tail_text"] = tok.decode(ids[runs[-1]["end"] - 1:]) if runs else ""
    q_ids = tok(q, add_special_tokens=False).input_ids
    rep["question_start"] = next((k for k in range(len(ids)) if ids[k:k + len(q_ids)] == q_ids), None)
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.model_id.replace("/", "__") + ("" if not args.set_image_processor else "__" + "_".join(args.set_image_processor))
    (args.out / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
