#!/usr/bin/env python3
"""No-harm benchmarks: MME + POPE, base model vs adapter-active, identical samples in
one process. Yes/No readout = single forward, argmax over {Yes, No} first-token logits.
Band: |delta| <= 2 pts per benchmark => no-harm GO.

Anchors (RESULTS.md): carrier LoRA MME -0.2 / POPE -1.4 ([2026-07-19] E-D);
SFT adapter MME -0.6 / POPE +1.2 ([2026-07-24] P4.3).

Arm 2 is either a carrier ckpt (--lora-ckpt, alpha read from the ckpt) or a PEFT
adapter dir (--peft-adapter, + up-to-20 failure dumps for emission anatomy).

Usage:
  python scripts/bench_noharm.py --lora-ckpt checkpoints/carrier_layer_fmt_caption_best.pt \
      --output outputs/carrier/noharm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.carriers import attach_lora, load_carrier_layer_ckpt
from gnnformer.runtime import get_layers, load_runtime, move_to_device


def cap_size(img, m=640):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > m:
        s = m / max(w, h)
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora-ckpt", default=None, help="carrier_layer_best.pt for arm 2")
    ap.add_argument("--peft-adapter", default=None, help="PEFT adapter dir (SFT baseline)")
    ap.add_argument("--mme-n", type=int, default=500)
    ap.add_argument("--pope-n", type=int, default=500)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/carrier/noharm")
    args = ap.parse_args()
    if not args.lora_ckpt and not args.peft_adapter:
        ap.error("need --lora-ckpt or --peft-adapter")
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    items = []  # (bench, category, image, question, gold_yes)
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
        items.append(("POPE", r.get("category", "all"), cap_size(r["image"]), r["question"], a == "yes"))
        n += 1
    print(f"[data] {sum(1 for i in items if i[0]=='MME')} MME + "
          f"{sum(1 for i in items if i[0]=='POPE')} POPE items", flush=True)

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    yes_id = tok("Yes", add_special_tokens=False).input_ids[0]
    no_id = tok("No", add_special_tokens=False).input_ids[0]

    def run_pass(tag, m_obj):
        preds = []
        t0 = time.time()
        for i, (_bench, _cat, img, q, _gy) in enumerate(items):
            content = [{"type": "image", "image": img}, {"type": "text", "text": q}]
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, rt.device)
            with torch.no_grad():
                lg = m_obj(**inputs).logits[0, -1].float()
            preds.append(bool(lg[yes_id] > lg[no_id]))
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        return preds

    def score(preds):
        res = {}
        for bench in ("MME", "POPE"):
            idx = [i for i, it in enumerate(items) if it[0] == bench]
            g = np.array([items[i][4] for i in idx])
            p = np.array([preds[i] for i in idx])
            acc = float((g == p).mean())
            tp = int((p & g).sum())
            fp = int((p & ~g).sum())
            fn = int((~p & g).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            per: dict = {}
            for i in idx:
                per.setdefault(items[i][1], [0, 0])
                per[items[i][1]][1] += 1
                per[items[i][1]][0] += int(preds[i] == items[i][4])
            res[bench] = {"acc": acc, "f1": f1, "per": {c: a / t for c, (a, t) in sorted(per.items())}}
        return res

    base = run_pass("base", model)
    arm2_model = model
    if args.peft_adapter:
        from peft import PeftModel

        arm2_model = PeftModel.from_pretrained(model, args.peft_adapter, is_trainable=False)
        print(f"[arm2] PEFT adapter loaded: {args.peft_adapter}", flush=True)
    else:
        ck = load_carrier_layer_ckpt(Path(args.lora_ckpt))
        attach_lora(get_layers(model), ck.l_open, rank=ck.rank, alpha=ck.alpha,
                    device=rt.device, state=ck.lora_state)
        print(f"[arm2] LoRA hooks active (ckpt {args.lora_ckpt}, trained acc {ck.acc}, "
              f"alpha {ck.alpha})", flush=True)
    lora_preds = run_pass("lora", arm2_model)

    if args.peft_adapter:  # emission anatomy on failures
        fails = [i for i, (p, it) in enumerate(zip(lora_preds, items)) if p != it[4]][:20]
        for i in fails:
            bench, _cat, img, q, gy = items[i]
            content = [{"type": "image", "image": img}, {"type": "text", "text": q}]
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, rt.device)
            with torch.no_grad():
                g = arm2_model.generate(**inputs, max_new_tokens=8, do_sample=False)
            txt = tok.decode(g[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  [fail-dump] {bench} gold={'Yes' if gy else 'No'} emits={txt!r} "
                  f"q={q[:60]!r}", flush=True)

    rb, rl = score(base), score(lora_preds)
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
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
