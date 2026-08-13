#!/usr/bin/env python3
"""Frozen-model own-answer eval on prepped HERBench AC frame samples (both arms).

Generation-based (handles multi-token numbers, unlike the probe's digit argmax).
Metrics: exact-match vs visible_count, MAE, by-count table; plus HERBench-style
nearest-choice MCQ accuracy for arm A (where visible_count == true_count).
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="e.g. data/herbench_ac/armA_evidence_only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from evaluations.helpers import patching_core as tgi
    from evaluations.scripts.patch_importence import group_restoration_importance as gri
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()

    dirs = sorted(d for d in Path(args.data_root).iterdir() if (d / "meta.json").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    rows = []
    for i, sd in enumerate(dirs):
        meta = json.loads((sd / "meta.json").read_text())
        frames = [Image.open(p).convert("RGB") for p in sorted(sd.glob("frame_*.jpg"))]
        n = len(frames)
        prompt = (f"You will be shown {n} frames sampled from an egocentric kitchen video.\n"
                  f"Respond with a single integer from 0 to {n} (0 is allowed). Output only the integer.\n"
                  f"Question: In how many of these {n} frames is the person performing "
                  f"the action '{meta['pair']}'?\nAnswer: ")
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt(frames, prompt))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False,
                                 pad_token_id=processor.tokenizer.eos_token_id)
        text = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        m = re.search(r"-?\d+", text)
        pred = int(m.group()) if m else None
        gold = int(meta["visible_count"])
        choice_vals = [int(re.search(r"\d+", c).group()) for c in meta["choices"]]
        mcq_ok = None
        if pred is not None and gold == int(meta["true_count"]):
            # nearest-choice mapping, HERBench-comparable (arm A only, all evidence shown)
            mcq_ok = int(min(choice_vals, key=lambda v: (abs(v - pred), v)) == int(meta["answer_text"]))
        rows.append({"qid": meta["question_id"], "gold": gold, "pred": pred,
                     "raw": text.strip(), "true_count": int(meta["true_count"]),
                     "n_frames": n, "mcq_ok": mcq_ok})
        if (i + 1) % 20 == 0:
            done = [r for r in rows if r["pred"] is not None]
            acc = sum(r["pred"] == r["gold"] for r in done) / max(1, len(done))
            print(f"[{i+1}/{len(dirs)}] running exact-match {acc:.3f}", flush=True)

    ok = [r for r in rows if r["pred"] is not None]
    em = sum(r["pred"] == r["gold"] for r in ok) / max(1, len(ok))
    mae = sum(abs(r["pred"] - r["gold"]) for r in ok) / max(1, len(ok))
    bias = sum(r["pred"] - r["gold"] for r in ok) / max(1, len(ok))
    mcq = [r for r in rows if r["mcq_ok"] is not None]
    by_gold = defaultdict(list)
    for r in ok:
        by_gold[r["gold"]].append(r)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_root": args.data_root, "n": len(rows), "n_parsed": len(ok),
        "exact_match": round(em, 4), "mae": round(mae, 3), "bias": round(bias, 3),
        "mcq_nearest_choice_acc": round(sum(r["mcq_ok"] for r in mcq) / len(mcq), 4) if mcq else None,
        "n_mcq": len(mcq),
        "by_gold": {g: {"n": len(v),
                        "acc": round(sum(r["pred"] == r["gold"] for r in v) / len(v), 3),
                        "mean_pred": round(sum(r["pred"] for r in v) / len(v), 2)}
                    for g, v in sorted(by_gold.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
