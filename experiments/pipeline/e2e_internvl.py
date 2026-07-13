#!/usr/bin/env python3
"""P4a: cross-family spot-check — verify-then-tally on InternVL2.5-8B (steps, N=32).

Design note: the Qwen pipeline's gate/shortlist stage is skipped (no InternVL mass-cache;
"one number is enough" per the charter) — every frame is verified with InternVL's OWN yes/no
look-again read, the tally is rendered as the predicate-matched fact, and InternVL verbalizes
(manual greedy decode; remote code lacks .generate under this transformers version).
Registered: works — its multipass supply 6.5 clears the crush line. Baseline: InternVL frozen
joint (same samples, digit read is insufficient for N=32 gold, so greedy decode there too).
"""
from __future__ import annotations
import argparse, copy, json, random, re, sys, time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from experiments.internvl.baseline_eval import build_transform

INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_longN_park/seq_len_32/all_uniform")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(args.model_name, quantization_config=bnb,
                                      trust_remote_code=True, use_flash_attn=False,
                                      low_cpu_mem_usage=True, device_map={"": 0}).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    tfm = build_transform()
    vdt = next(p.dtype for p in model.vision_model.parameters() if p.is_floating_point())
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    eos = tok.eos_token_id

    def make_inputs(frames, question):
        # InternVL forward REQUIRES pixel_values: text-only prompts get a gray dummy image
        # with an explicit ignore instruction.
        if not frames:
            from PIL import Image as _I
            frames = [_I.new("RGB", (448, 448), (128, 128, 128))]
            question = "<image>\nIgnore the image above; answer from the note only.\n" + question
        tpl = copy.deepcopy(model.conv_template)
        tpl.append_message(tpl.roles[0], question)
        tpl.append_message(tpl.roles[1], None)
        prompt = tpl.get_prompt()
        for _ in range(len(frames)):
            blk = "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
            prompt = prompt.replace("<image>", blk, 1)
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].cuda(); am = enc["attention_mask"].cuda()
        if frames:
            pv = torch.stack([tfm(f) for f in frames]).to(vdt).cuda()
            fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
            return dict(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
        return dict(input_ids=ids, attention_mask=am)

    def last_logits(inp):
        with torch.no_grad():
            return model(**inp).logits[0, -1].float().cpu()

    def greedy_int(inp, max_new=4):
        ids = inp["input_ids"]; am = inp["attention_mask"]
        text = ""
        for _ in range(max_new):
            lg = last_logits({**inp, "input_ids": ids, "attention_mask": am})
            nxt = int(lg.argmax())
            if nxt == eos:
                break
            text += tok.decode([nxt])
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
            am = torch.cat([am, torch.ones(1, 1, dtype=am.dtype, device=am.device)], 1)
        m = INT_RE.search(text)
        return int(m.group(0)) if m else None

    dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(dirs)
    rows = []
    n = 0
    for sd in dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            m = re.search(r"did (\w+) spend in the (\w+)", q0)
            C, R = m.group(1), m.group(2)
        except Exception:
            continue
        try:
            votes = 0
            for fr in frames:
                inp = make_inputs([fr], f"<image>\nIs {C} in the {R} in this frame? "
                                        f"Answer yes or no.\nAnswer:")
                lg = last_logits(inp)
                votes += int(float(lg[yes_id]) > float(lg[no_id]))
            fact = f"Note: {C} spent exactly {votes} steps in the {R}."
            rend = greedy_int(make_inputs([], f"{fact}\nRespond with a single integer. "
                                              f"Output only the integer.\nQuestion: {q0}\nAnswer:"))
            joint_q = ("\n".join(["<image>"] * len(frames))
                       + f"\nRespond with a single integer from 0 to {len(frames)} "
                         f"(0 is allowed). Output only the integer.\nQuestion: {q0}\nAnswer:")
            frozen = greedy_int(make_inputs(frames, joint_q))
        except Exception as exc:
            print(f"{sid} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({"sid": sid, "gold": gold, "tally": votes, "rendered": rend,
                     "frozen": frozen})
        n += 1
        if n % 10 == 0:
            t = np.array([r["tally"] for r in rows]); g = np.array([r["gold"] for r in rows])
            print(f"  {n}: tally-exact {float((t == g).mean()):.3f}", flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))

    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    g = np.array([r["gold"] for r in rows])
    t = np.array([r["tally"] for r in rows])
    rd = np.array([r["rendered"] if r["rendered"] is not None else -99 for r in rows])
    fz = np.array([r["frozen"] if r["frozen"] is not None else -99 for r in rows])
    lines = [f"=== InternVL2.5-8B verify-then-tally (steps N=32, n={len(rows)}) ===",
             f"  tally exact = {float((t == g).mean()):.3f}  MAE {float(np.abs(t - g).mean()):.2f}",
             f"  rendered exact = {float((rd == g).mean()):.3f}",
             f"  frozen joint exact = {float((fz == g).mean()):.3f}"]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
