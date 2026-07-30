#!/usr/bin/env python3
"""Exp 1 Step 1 smoke: each candidate loads + generates 10 tokens (runs in .venv_arch)."""
from __future__ import annotations
import argparse, json, time, traceback
import torch

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "softmax"),
    ("Qwen/Qwen2.5-14B-Instruct", "softmax"),
    ("tiiuae/Falcon-H1-7B-Instruct", "hybrid-mamba2"),
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "hybrid-mamba2"),
    ("NX-AI/xLSTM-7b", "mlstm"),
    ("google/recurrentgemma-9b-it", "griffin"),
]
PROMPT = ("You are given 2 frames describing steps in a house, as text.\n"
          "Frame 1:\n  Kitchen: John\n  Garden: (empty)\nFrame 2:\n  Kitchen: (empty)\n  Garden: John\n\n"
          "Respond with a single integer from 0 to 2 (0 is allowed). Output only the integer.\n"
          "Question: How many steps did John spend in the Kitchen?\nAnswer: ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma substring filter")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    print(f"torch {torch.__version__} cuda {torch.version.cuda} avail {torch.cuda.is_available()} "
          f"| transformers {transformers.__version__}", flush=True)
    results = {}
    for name, klass in MODELS:
        if args.only and not any(s in name for s in args.only.split(",")):
            continue
        t0 = time.time()
        print(f"=== {name} ({klass}) ===", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(name)
            kw = dict(dtype=torch.bfloat16, device_map="auto")
            if args.load_in_4bit:
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(name, **kw)
            model.eval()
            n_par = sum(p.numel() for p in model.parameters())
            if tok.chat_template is not None:
                ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                              add_generation_prompt=True, return_tensors="pt")
            else:
                ids = tok(PROMPT, return_tensors="pt").input_ids
            if not torch.is_tensor(ids):        # transformers v5 returns BatchEncoding
                ids = ids["input_ids"]
            ids = ids.to(model.device)
            t1 = time.time()
            with torch.inference_mode():
                out = model.generate(ids, do_sample=False, max_new_tokens=10,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            txt = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
            dt = time.time() - t1
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            print(f"  OK params={n_par/1e9:.1f}B load={t1-t0:.0f}s gen10={dt:.1f}s "
                  f"gpu_peak={mem:.1f}G out={txt!r}", flush=True)
            results[name] = {"ok": True, "params_b": round(n_par / 1e9, 1),
                             "load_s": round(t1 - t0), "gen10_s": round(dt, 1),
                             "gpu_peak_gb": round(mem, 1), "output": txt}
            del model
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            traceback.print_exc()
            results[name] = {"ok": False, "error": str(e)[:300]}
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    n_ok = sum(1 for r in results.values() if r["ok"])
    print(f"SMOKE DONE: {n_ok}/{len(results)} ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
