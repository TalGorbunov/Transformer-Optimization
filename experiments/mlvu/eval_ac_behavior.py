#!/usr/bin/env python3
"""A3 (MLVU-AC) behavioral eval on prepped frames: N frames uniformly subsampled from the
stored 128, two protocols per question:
  mcq  : standard MLVU multiple-choice (4 candidates), letter answer
  open : "Respond with a single integer" + generation reader (our ladder protocol)
Reports accuracy overall + by gold count, per protocol, at each requested N.
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.scripts.patch_importence import group_restoration_importance as gri

INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mlvu_ac")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--protocols", default="mcq,open")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    from PIL import Image

    dirs = sorted(d for d in Path(args.data_root).iterdir() if (d / "meta.json").exists())
    rng = random.Random(args.sample_seed)
    rng.shuffle(dirs)
    if args.limit:
        dirs = dirs[: args.limit]
    protos = [p.strip() for p in args.protocols.split(",")]
    NF = int(args.n_frames)
    rows = []
    for i, sd in enumerate(dirs):
        meta = json.loads((sd / "meta.json").read_text())
        gold = int(meta["answer"])
        n_stored = meta["n_frames"]
        idx = np.linspace(0, n_stored - 1, NF).round().astype(int)
        frames = []
        for t in idx:
            img = Image.open(sd / f"frame_{t:03d}.jpg").convert("RGB")
            if args.resize:
                w, h = img.size
                sc = args.resize / max(w, h)
                img = img.resize((max(1, round(w * sc)), max(1, round(h * sc))))
            frames.append(img)
        action = meta.get("action") or "the action"
        for proto in protos:
            if proto == "mcq":
                letters = ["A", "B", "C", "D"]
                cands = meta["candidates"]
                opts = "\n".join(f"{L}. {c}" for L, c in zip(letters, cands))
                prompt = (f"You will be shown {NF} frames sampled uniformly from a long video.\n"
                          f"Question: {meta['question']}\n{opts}\n"
                          f"Answer with the letter of the correct option only.\nAnswer: ")
                target = letters[cands.index(meta.get("answer") if isinstance(meta.get("answer"), str)
                                             else str(gold))] if str(gold) in cands else None
            else:
                if meta.get("action"):
                    open_q = (f"In how many separate instances does the '{action}' action "
                              f"scene occur in the video?")
                else:       # VNBench and other ports: meta carries its own question
                    open_q = meta["question"]
                prompt = (f"You will be shown {NF} frames sampled uniformly from a long video.\n"
                          f"Respond with a single integer. Output only the integer.\n"
                          f"Question: {open_q}\nAnswer: ")
                target = gold
            try:
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt(frames, prompt))
                plen = int(inputs["input_ids"].shape[1])
                with torch.no_grad():
                    gen = model.generate(**inputs, do_sample=False, max_new_tokens=6,
                                         pad_token_id=pad)
                dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0].strip()
            except Exception as exc:
                print(f"{sd.name}/{proto} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            if proto == "mcq":
                m = re.search(r"[ABCD]", dec)
                pred_letter = m.group(0) if m else None
                correct = int(pred_letter == target) if target else None
                rows.append({"qid": sd.name, "proto": proto, "gold": gold, "pred": pred_letter,
                             "target": target, "raw": dec[:24], "correct": correct})
            else:
                m = INT_RE.search(dec)
                pred = int(m.group(0)) if m else None
                rows.append({"qid": sd.name, "proto": proto, "gold": gold, "pred": pred,
                             "raw": dec[:24], "correct": int(pred == gold)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(dirs)}", flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))

    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    lines = [f"=== MLVU-AC behavior (N={NF} frames @{args.resize}px, n={len(dirs)} questions) ==="]
    summary = {}
    for proto in protos:
        rr = [r for r in rows if r["proto"] == proto and r["correct"] is not None]
        acc = float(np.mean([r["correct"] for r in rr])) if rr else float("nan")
        per = defaultdict(list)
        for r in rr:
            per[r["gold"]].append(r["correct"])
        by = {g: round(float(np.mean(v)), 3) for g, v in sorted(per.items())}
        lines.append(f"  {proto:<5s} n={len(rr):<4d} acc={acc:.3f}  by-gold {by}")
        summary[proto] = {"n": len(rr), "acc": acc, "by_gold": by}
        if proto == "open":
            preds = [r["pred"] for r in rr if r["pred"] is not None]
            golds = [r["gold"] for r in rr if r["pred"] is not None]
            if preds:
                mae = float(np.mean(np.abs(np.array(preds) - np.array(golds))))
                lines.append(f"        MAE {mae:.2f}  mean-pred {np.mean(preds):.2f} "
                             f"(mean gold {np.mean(golds):.2f})")
                summary[proto]["mae"] = mae
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
