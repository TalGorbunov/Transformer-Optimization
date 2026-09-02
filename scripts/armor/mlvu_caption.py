#!/usr/bin/env python3
"""ARMOR Exp C step 1 — per-frame question-conditioned records on MLVU-AC.

The external-benchmark anchor for the composed frozen system (VLM per-frame
perception -> compiled program -> exact execution). Frozen Qwen2.5-VL 7B, one
independent call per frame (multipass = measured-equivalent of the fenced
forward, A3 parity). 32 frames per question, uniformly subsampled from the
stored 128 by round(linspace) — BYTE-IDENTICAL to the [2026-07-11c] frozen
baseline protocol and to the lookagain_N32.json judge keys (verified).

Per frame the VLM answers whether the queried action is visible, plus a short
description (audit + program fodder). Records: {"time", "desc", "vis"}.
Fidelity proxy reported against lookagain_N32.json judge scores (>0.5 = judge
evidence) where present.

Usage: see slurm/armor_mlvu_caption.sbatch. Smoke: --limit 3.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

CAP_PROMPT = (
    "This is one frame sampled from a longer video. In one short sentence, "
    "describe what is happening in this frame. Then state whether the frame "
    "shows the action '{action}'. Use exactly this format:\n"
    "Desc: <one sentence>\nAction: yes/no")

_ACT_RE = re.compile(r"action:\s*(yes|no)", re.IGNORECASE)
_DESC_RE = re.compile(r"desc:\s*(.+)", re.IGNORECASE)


def parse_cap(text: str):
    m = _ACT_RE.search(text)
    vis = (m.group(1).lower() == "yes") if m else False
    d = _DESC_RE.search(text)
    desc = d.group(1).strip() if d else text.strip().split("\n")[0][:160]
    return vis, desc, m is None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/mlvu_ac")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image

    from gnnformer.runtime import load_runtime

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor = rt.model, rt.processor
    t0 = time.time()

    dirs = sorted(d for d in Path(args.data_root).iterdir()
                  if (d / "meta.json").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"{len(dirs)} questions x {args.n_frames} frames", flush=True)

    results = {}
    agree, agree_n, pfails = 0, 0, 0
    for qi, sd in enumerate(dirs):
        meta = json.loads((sd / "meta.json").read_text())
        n_stored = int(meta["n_frames"])
        idx = np.linspace(0, n_stored - 1, args.n_frames).round().astype(int)
        frames = [Image.open(sd / f"frame_{i:03d}.jpg").convert("RGB")
                  .resize((args.resize, args.resize)) for i in idx]
        times = [float(meta["frames"][i]["time"]) for i in idx]
        prompt = CAP_PROMPT.format(action=meta["action"])
        caps = []
        for b0 in range(0, len(frames), args.batch):
            imgs = frames[b0:b0 + args.batch]
            msgs = [[{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
                for _ in imgs]
            texts = [processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True) for m in msgs]
            enc = processor(text=texts, images=imgs, return_tensors="pt",
                            padding=True).to(rt.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False)
            for k in range(len(imgs)):
                caps.append(processor.tokenizer.decode(
                    gen[k, enc["input_ids"].shape[1]:],
                    skip_special_tokens=True).strip())
        records = []
        for t_, c in zip(times, caps):
            vis, desc, pf = parse_cap(c)
            pfails += int(pf)
            records.append({"time": t_, "desc": desc, "vis": vis})
        # judge-agreement proxy (lookagain N32, same indices by construction)
        la_path = sd / "lookagain_N32.json"
        if la_path.exists():
            la = json.loads(la_path.read_text())
            la_bits = [float(v) > 0.5 for v in la.values()]
            if len(la_bits) == len(records):
                agree += sum(int(r["vis"] == b)
                             for r, b in zip(records, la_bits))
                agree_n += len(records)
        results[sd.name] = {"action": meta["action"],
                            "question": meta["question"],
                            "gold": int(meta["answer"]),
                            "candidates": meta["candidates"],
                            "records": records, "raw": caps}
        if (qi + 1) % 10 == 0:
            ag = agree / max(agree_n, 1)
            print(f"  {qi + 1}/{len(dirs)}  judge-agree {ag:.3f}  "
                  f"parse-fails {pfails}  {time.time() - t0:.0f}s", flush=True)

    summary = {"n_questions": len(results),
               "judge_agreement": agree / max(agree_n, 1),
               "parse_fails": pfails, "args": vars(args)}
    (out / "records.json").write_text(json.dumps(results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE {len(results)} questions  judge-agree "
          f"{summary['judge_agreement']:.4f}  parse-fails {pfails}  "
          f"wall {time.time() - t0:.0f}s")
    print(f"wrote {out}/records.json, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
