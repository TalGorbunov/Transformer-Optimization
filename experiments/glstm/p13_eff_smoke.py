#!/usr/bin/env python3
"""P1.3 debug smoke (2026-07-23): why does the EFFICIENT sdpa backend not engage inside
model.generate at N=64 (job 125260 fell back to MATH -> 17GB OOM), while the manual-layer
eval path (carrier_layer_lora) runs efficient at seq 23k?

Loads the SFT setup, one N=64 sample, and tries generate under each backend list,
printing what happens. Efficient-only makes PyTorch print per-backend rejection reasons.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.helpers.utils import load_mmred_sample, iter_sample_dirs_shuffled


def main() -> int:
    warnings.simplefilter("always")
    device = base.resolve_device("cuda")
    dtype = base.resolve_dtype("bfloat16", device)
    model, processor = base.load_model_and_processor("Qwen/Qwen2.5-VL-7B-Instruct", device, dtype, True)
    model.eval()
    sd = iter_sample_dirs_shuffled(Path("data/mmred_longN_park/seq_len_64/all_uniform"), 0)[0]
    _sid, frames, q0, _states, a0 = load_mmred_sample(sd)
    frames = [f.resize((392, 392)) for f in frames]
    msgs = [{"role": "user", "content":
             [{"type": "text", "text": f"Question: {q0}\nThe following are the {len(frames)} frames showing rooms in a house:"}]
             + [{"type": "image", "image": im} for im in frames]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
    inp = base.move_inputs_to_device(dict(inp), device)
    print(f"[smoke] sample={_sid} gold={a0} seq={int(inp['input_ids'].shape[1])}", flush=True)

    for name, backends in [("EFFICIENT-only", [SDPBackend.EFFICIENT_ATTENTION]),
                           ("FLASH-only", [SDPBackend.FLASH_ATTENTION]),
                           ("EFF+MATH (job 125260 config)", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel(backends), torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=5, do_sample=False)
            txt = processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[smoke] {name}: OK -> {txt!r} (peak {peak:.1f} GiB)", flush=True)
        except Exception as exc:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[smoke] {name}: {type(exc).__name__}: {exc} (peak {peak:.1f} GiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
