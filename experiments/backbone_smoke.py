#!/usr/bin/env python3
"""GPU smoke for a backbone's LOADING configuration: the same short MMReD prompt (2 official frames,
question-first) under several load variants, reporting for each whether the next-token logits are
finite, the first generated token ids and the decoded text. Written for the Gemma-3 4-bit question
(empty outputs, 2026-09-28) and kept for every new backbone's day-1 check.
Variants:  spec       the spec's load (4-bit nf4, no explicit dtype)
           bf16       + torch_dtype=bfloat16 for the non-quantized modules
           skipvis    bf16 + vision tower and projector kept out of the quantization
           full       no quantization, bfloat16
Usage:  python experiments/backbone_smoke.py --backbone gemma-3-12b --variants spec bf16 skipvis full --output outputs/scaleup/_backbones/smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, get_backbone  # noqa: E402
from core.backbones import base as bb  # noqa: E402
from core.data import get_dataset  # noqa: E402
from core.prompt import build_messages  # noqa: E402

SKIP = ["vision_tower", "multi_modal_projector", "vision_model", "visual"]


def load(spec, variant):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    pk = dict(spec.processor_kwargs)
    if spec.use_fast_processor is not None:
        pk["use_fast"] = spec.use_fast_processor
    processor = AutoProcessor.from_pretrained(spec.model_id, **pk)
    kw = {"attn_implementation": "sdpa", "device_map": "cuda"}
    if variant != "full":
        q = dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        if variant == "skipvis":
            q["llm_int8_skip_modules"] = SKIP
        kw["quantization_config"] = BitsAndBytesConfig(**q)
    if variant in ("bf16", "skipvis", "full"):
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(spec.model_id, **kw)
    model.eval()
    return processor, model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--variants", nargs="+", default=["spec", "bf16", "skipvis", "full"])
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    spec = get_backbone(args.backbone)
    ds = get_dataset("mmred")
    samples = ds.load(Path(ds.default_root), "seq_len_2_test")[: args.rows * 40: 40]
    report = {"backbone": spec.name, "variants": {}}
    for v in args.variants:
        t0 = time.time()
        try:
            processor, model = load(spec, v)
        except Exception as exc:  # noqa: BLE001
            report["variants"][v] = {"load_error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(v, report["variants"][v], flush=True)
            continue
        tok = processor.tokenizer
        dtypes = {}
        for n, p in model.named_parameters():
            key = n.split(".")[0] + "." + n.split(".")[1] if "." in n else n
            dtypes.setdefault(key, set()).add(str(p.dtype))
        rows = []
        for s in samples:
            msgs = build_messages(ds.frames(s), ds.question_text(s), layout="question-first", system_prompt=ds.system_prompt)
            enc = bb.encode(spec, processor, msgs, model.device)
            with torch.inference_mode():
                out = model(**enc, use_cache=False)
                lg = out.logits[0, -1].float()
                gen = model.generate(**enc, max_new_tokens=24, do_sample=False)
            new = gen[0, enc["input_ids"].shape[1]:].tolist()
            rows.append({"qid": s.qid, "gold": s.answer, "seq": int(enc["input_ids"].shape[1]),
                         "logits_finite": bool(torch.isfinite(lg).all()), "logits_nan": int(torch.isnan(lg).sum()),
                         "top5": [(tok.convert_ids_to_tokens(int(i)), round(float(x), 2)) for x, i in zip(*torch.topk(torch.nan_to_num(lg, nan=-1e9), 5))],
                         "new_ids": new[:24], "text": tok.decode(new, skip_special_tokens=True)[:80],
                         "text_raw": tok.decode(new, skip_special_tokens=False)[:80]})
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        report["variants"][v] = {"load_s": round(time.time() - t0), "gpu_gb": round(mem, 1),
                                 "param_dtypes": {k: sorted(d) for k, d in dtypes.items()}, "rows": rows}
        print(v, json.dumps(report["variants"][v], default=str)[:1500], flush=True)
        del model, processor
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{spec.name}.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
