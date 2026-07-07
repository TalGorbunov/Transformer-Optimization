#!/usr/bin/env python3
"""Track B phase 1: InternVL2.5-8B behavioral baseline on image-MMRED (steps + rooms).

Foreign VLM family (InternLM2 LM, dynamic-tile ViT, linear MLP adapter). Zero-shot via model.chat();
integer parsed from the response. Reports acc / MAE / per-gold rows (does the fraction-reader
signature replicate on a different family?). Preprocessing follows the official model-card snippet
(448px tiles, max_num=1 -> 256 visual tokens per frame).
"""
from __future__ import annotations
import argparse, random, re, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.helpers import utils as eval_utils

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--tasks", default="count,rooms_visited")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/baseline")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("loading model ...", flush=True)
    model = AutoModel.from_pretrained(
        args.model_name, quantization_config=bnb, trust_remote_code=True,
        use_flash_attn=False, low_cpu_mem_usage=True, device_map={"": 0}).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    tfm = build_transform()
    import copy as _copy
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)]
    cand_vals = [d for d, _ in cand]
    cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    def ask_digit(pv, npl, question):
        """One forward pass; argmax over digit tokens at the last position (matches the Qwen protocol).
        Avoids .generate(), which the remote code lacks under transformers>=4.50 (GenerationMixin split)."""
        tpl = _copy.deepcopy(model.conv_template)
        tpl.append_message(tpl.roles[0], question)
        tpl.append_message(tpl.roles[1], None)
        prompt = tpl.get_prompt()
        for np_ in npl:
            blk = "<img>" + "<IMG_CONTEXT>" * (model.num_image_token * np_) + "</img>"
            prompt = prompt.replace("<image>", blk, 1)
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].cuda(); am = enc["attention_mask"].cuda()
        fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
        with torch.no_grad():
            outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
        lg = outp.logits[0, -1].float().cpu()
        return int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])
    vdt = next(p_.dtype for p_ in model.vision_model.parameters() if p_.is_floating_point())
    print(f"loaded; num_image_token={model.num_image_token}; vision dtype={vdt}", flush=True)

    from evaluations.scripts.patch_importence.probe_frame_to_carrier_message import char_room_at as cra

    results = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        all_dirs = list(iter_sample_dirs(Path(args.data_root)))
        random.Random(args.sample_seed).shuffle(all_dirs)
        preds, golds = [], []
        n = 0; fails = 0
        for sd in all_dirs:
            if n >= args.limit:
                break
            try:
                sid, frames, q0, states, a0 = load_mmred_sample(sd)
                if task == "count":
                    gold = int(str(a0).strip())
                    question = q0
                elif task == "rooms_visited":
                    chars = sorted(eval_utils.extract_characters_from_states(states))
                    if len(chars) < 2:
                        continue
                    present = lambda c: [t_ for t_ in range(len(states)) if cra(states, t_, c)]
                    char = max(chars, key=lambda c: (len(present(c)), c))
                    pres = present(char)
                    if len(pres) < 2:
                        continue
                    gold = len({cra(states, t_, char) for t_ in pres})
                    question = f"How many distinct rooms did {char} visit?"
                else:
                    raise ValueError(task)
                pv = torch.cat([tfm(f).unsqueeze(0) for f in frames]).to(vdt).cuda()
                npl = [1] * len(frames)
                prefix = "".join(f"Frame-{i+1}: <image>\n" for i in range(len(frames)))
                q = prefix + question + "\nAnswer with a single number."
                preds.append(ask_digit(pv, npl, q))
                golds.append(gold)
                n += 1
                if n % 25 == 0:
                    p_, g_ = np.array(preds), np.array(golds)
                    print(f"  [{task}] {n}: acc={np.mean(p_==g_):.3f} MAE={np.mean(np.abs(np.clip(p_,0,None)-g_)):.2f}", flush=True)
            except Exception as exc:
                fails += 1
                print(f"{sd} failed: {type(exc).__name__}: {exc}")
                if fails >= 25 and n == 0:
                    raise RuntimeError("25 consecutive failures with 0 successes — aborting")
                continue
        p, g = np.array(preds), np.array(golds)
        acc = float(np.mean(p == g)); mae = float(np.mean(np.abs(np.clip(p, 0, None) - g)))
        maj = Counter(g.tolist()).most_common(1)[0][1] / max(1, len(g))
        by = {gv: float(np.mean(p[g == gv] == gv)) for gv in sorted(set(g.tolist()))}
        # emitted-vs-gold slope (fraction-reader fingerprint)
        slope = float(np.polyfit(g, np.clip(p, 0, None), 1)[0]) if len(set(g.tolist())) > 1 else float("nan")
        results[task] = dict(n=len(g), acc=acc, mae=mae, majority_floor=maj, slope=slope, by_gold=by,
                             preds=p.tolist(), golds=g.tolist())
        print(f"[{task}] n={len(g)} acc={acc:.3f} MAE={mae:.2f} majority={maj:.3f} slope={slope:.2f}")
        print("   by-gold: " + " ".join(f"g{k}:{v:.2f}" for k, v in by.items()))

    lines = [f"=== InternVL2.5-8B baseline (image MMRED, n<={args.limit}/task) ==="]
    for task, r in results.items():
        lines.append(f"  {task:>14}: acc={r['acc']:.3f} MAE={r['mae']:.2f} majority={r['majority_floor']:.3f} "
                     f"slope={r['slope']:.2f} n={r['n']}")
        lines.append("      by-gold: " + " ".join(f"g{k}:{v:.2f}" for k, v in r["by_gold"].items()))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    torch.save({"results": results, "config": vars(args)}, out / "baseline.pt")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
