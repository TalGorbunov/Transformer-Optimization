#!/usr/bin/env python3
"""E-D no-harm benchmarks (2026-07-19): MME + POPE, base model vs LoRA-hooks-active.

Both arms run on IDENTICAL samples in one process (model loaded once; hooks registered for
arm 2 via the frozen_baseline_eval --lora-ckpt pattern, alpha=16). Yes/No readout = single
forward, argmax over {Yes, No} first-token logits — no generation, no parsing ambiguity.
Images resized to max side 640 (identical for both arms; the DELTA is the metric).
Band: |delta| <= 2 pts per benchmark => no-harm GO.
"""
from __future__ import annotations
import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers


def cap_size(img, m=640):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > m:
        s = m / max(w, h)
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", default=None, help="carrier_layer_best.pt for arm 2")
    ap.add_argument("--peft-adapter", default=None,
                    help="P4.3 (2026-07-24): PEFT adapter dir (plain-LoRA SFT baseline) — "
                         "arm 2 loads it via PeftModel instead of carrier LoRA hooks; also "
                         "greedy-dumps up to 20 failing items (emission anatomy)")
    ap.add_argument("--mme-n", type=int, default=500)
    ap.add_argument("--pope-n", type=int, default=500)
    ap.add_argument("--output", default="outputs/ladder/image_longN/noharm_bench")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    items = []          # (bench, category, image, question, gold_yes)
    mme = load_dataset("lmms-lab/MME")["test"].shuffle(seed=0)
    n = 0
    for r in mme:
        if n >= args.mme_n:
            break
        a = str(r["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        items.append(("MME", r["category"], cap_size(r["image"]), r["question"], a == "yes"))
        n += 1
    pope = load_dataset("lmms-lab/POPE")["test"].shuffle(seed=0)
    n = 0
    for r in pope:
        if n >= args.pope_n:
            break
        a = str(r["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        items.append(("POPE", r.get("category", "all"), cap_size(r["image"]),
                      r["question"], a == "yes"))
        n += 1
    print(f"[data] {sum(1 for i in items if i[0]=='MME')} MME + "
          f"{sum(1 for i in items if i[0]=='POPE')} POPE items", flush=True)

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    dev = model.device
    yes_id = tok("Yes", add_special_tokens=False).input_ids[0]
    no_id = tok("No", add_special_tokens=False).input_ids[0]

    def run_pass(tag):
        preds = []
        t0 = time.time()
        for i, (bench, cat, img, q, gy) in enumerate(items):
            content = [{"type": "image", "image": img}, {"type": "text", "text": q}]
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                lg = model(**inputs).logits[0, -1].float()
            preds.append(bool(lg[yes_id] > lg[no_id]))
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        return preds

    def score(preds):
        res = {}
        for bench in ("MME", "POPE"):
            idx = [i for i, it in enumerate(items) if it[0] == bench]
            g = np.array([items[i][4] for i in idx]); p = np.array([preds[i] for i in idx])
            acc = float((g == p).mean())
            tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            per = {}
            for i in idx:
                per.setdefault(items[i][1], [0, 0])
                per[items[i][1]][1] += 1; per[items[i][1]][0] += int(preds[i] == items[i][4])
            res[bench] = {"acc": acc, "f1": f1,
                          "per": {c: a / t for c, (a, t) in sorted(per.items())}}
        return res

    if not args.lora_ckpt and not args.peft_adapter:
        ap.error("need --lora-ckpt or --peft-adapter")
    base = run_pass("base")
    if args.peft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft_adapter, is_trainable=False)
        print(f"[arm2] PEFT adapter loaded: {args.peft_adapter}", flush=True)
    else:
        layers_ = get_layers(model)
        ckl = torch.load(args.lora_ckpt, map_location="cpu")
        scale = 16.0 / int(ckl["rank"])
        for key, (A, B) in ckl["lora"].items():
            li, nm = key.split("."); li = int(li)
            A = A.float().to(dev); B = B.float().to(dev)
            mod = getattr(layers_[li].self_attn, nm)

            def mk(A=A, B=B):
                def hook(_m, inp, o):
                    return o + (scale * (inp[0].float() @ A.T) @ B.T).to(o.dtype)
                return hook
            mod.register_forward_hook(mk())
        print(f"[arm2] {len(ckl['lora'])} LoRA hooks active (ckpt {args.lora_ckpt}, "
              f"trained acc {ckl.get('acc'):.3f})", flush=True)
    lora = run_pass("lora")

    if args.peft_adapter:                   # emission anatomy on failures (P4.3)
        fails = [i for i, (p, it) in enumerate(zip(lora, items)) if p != it[4]][:20]
        for i in fails:
            bench, cat, img, q, gy = items[i]
            content = [{"type": "image", "image": img}, {"type": "text", "text": q}]
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                g = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            txt = tok.decode(g[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  [fail-dump] {bench} gold={'Yes' if gy else 'No'} emits={txt!r} q={q[:60]!r}",
                  flush=True)

    rb, rl = score(base), score(lora)
    lines = [f"=== NO-HARM BENCH (n_items={len(items)}, "
             f"ckpt={args.peft_adapter or args.lora_ckpt}) ==="]
    for bench in ("MME", "POPE"):
        b, l = rb[bench], rl[bench]
        lines.append(f"{bench}: base acc {b['acc']:.3f} f1 {b['f1']:.3f} | "
                     f"lora acc {l['acc']:.3f} f1 {l['f1']:.3f} | "
                     f"delta acc {100*(l['acc']-b['acc']):+.1f} pts")
        for c in b["per"]:
            lines.append(f"  {bench}/{c}: base {b['per'][c]:.3f} lora {l['per'][c]:.3f} "
                         f"delta {100*(l['per'][c]-b['per'][c]):+.1f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
