#!/usr/bin/env python3
"""Plain-model per-unit yes/no accuracy for the HERBench retrieve-opt sweep.

The frozen model's OWN judgment (no probe): feed each unit's frame(s) + "Is the person
performing the action '<pair>'? Answer yes or no.", read the yes-vs-no logit margin at the
last position (look-again readout, legacy lookagain_frames.py generalised to clips). Per unit
this gives a score; label = occurrence(1)/not(0). Reports, per verb and pooled:
  * AUROC              — threshold-free (primary; robust to the model's yes/no bias)
  * acc@0              — accuracy thresholding the margin at 0
  * bal_acc / yes_rate — balanced acc and the model's yes-rate (bias diagnostic)
Units are balanced pos/neg -> chance AUROC/acc = 0.50.

Same unit loaders as probe_clips (clips dirs, or armB frames for the δ0 single-frame anchor);
--input-mode {clips,armB}. Eval-only, one forward per unit. Usage:
  python scripts/herbench_retrieve_opt/eval_plain_acc.py --input-mode clips \
    --data-root data/herbench_retrieve_opt_clips/d1_r448 --arm-tag d1_r448 \
    --output outputs/herbench_retrieve_opt/plain_acc/d1_r448
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.runtime import load_runtime, move_to_device
from scripts.herbench_retrieve_opt.probe_clips import load_units_clips, load_units_armB


def metrics(scores, labels):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    out = {"n": int(len(y)), "npos": int((y == 1).sum())}
    if len(set(y.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y, s))
    else:
        out["auroc"] = float("nan")
    pred0 = (s > 0).astype(int)
    out["acc0"] = float((pred0 == y).mean())
    out["yes_rate"] = float((pred0 == 1).mean())
    # balanced acc at threshold 0
    tpr = float(((pred0 == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    tnr = float(((pred0 == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    out["bal_acc"] = 0.5 * (tpr + tnr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--input-mode", choices=("clips", "armB"), required=True)
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--resize", type=int, default=0, help="0 = keep extracted res")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    rt = load_runtime(args.model) if args.model else load_runtime()
    print(f"[timing] load_runtime {time.time()-t0:.1f}s", flush=True)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    yes_ids = sorted({tok.encode(t, add_special_tokens=False)[0] for t in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({tok.encode(t, add_special_tokens=False)[0] for t in ("no", "No", " no", " No")})
    yes_t = torch.tensor(yes_ids); no_t = torch.tensor(no_ids)

    by_q = (load_units_clips if args.input_mode == "clips" else load_units_armB)(Path(args.data_root))
    qids = sorted(by_q)
    if args.limit:
        qids = qids[: args.limit]

    rows = []  # (verb, label, margin)
    n = 0
    for qi, qid in enumerate(qids):
        for u in by_q[qid]:
            frs = [Image.open(p).convert("RGB") for p in u["frame_paths"]]
            if args.resize > 0:
                frs = [f.resize((args.resize, args.resize)) for f in frs]
            unit_word = "clip" if len(frs) > 1 else "frame"
            prompt = (f"Look at this {unit_word} from an egocentric kitchen video. "
                      f"Is the person performing the action '{u['pair']}'? Answer yes or no.")
            content = [{"type": "image", "image": f} for f in frs]
            content.append({"type": "text", "text": prompt})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, model.device)
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[0, -1].float().cpu()
            margin = float(torch.logsumexp(logits[yes_t], 0) - torch.logsumexp(logits[no_t], 0))
            rows.append((u["verb"], int(u["label"]), margin))
            n += 1
        if (qi + 1) % 25 == 0:
            print(f"  {qi+1}/{len(qids)} q, {n} units", flush=True)

    verbs = defaultdict(lambda: ([], []))
    allS, allY = [], []
    for verb, lab, mrg in rows:
        verbs[verb][0].append(mrg); verbs[verb][1].append(lab)
        allS.append(mrg); allY.append(lab)

    report = {"arm": args.arm_tag, "data_root": args.data_root, "input_mode": args.input_mode,
              "n_units": n, "pooled": metrics(allS, allY), "per_verb": {}}
    for verb, (S, Y) in verbs.items():
        if sum(Y) >= 20 and (len(Y) - sum(Y)) >= 20:
            report["per_verb"][verb] = metrics(S, Y)

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = [f"=== PLAIN yes/no ACC ({args.arm_tag}, n={n}) — AUROC / acc@0 / bal / yes_rate ==="]
    p = report["pooled"]
    lines.append(f"POOLED: auroc={p['auroc']:.3f} acc0={p['acc0']:.3f} bal={p['bal_acc']:.3f} "
                 f"yes_rate={p['yes_rate']:.3f} (n={p['n']}, pos={p['npos']})")
    for verb, m in sorted(report["per_verb"].items(), key=lambda kv: -kv[1]["auroc"]):
        lines.append(f"  {verb:8s}: auroc={m['auroc']:.3f} acc0={m['acc0']:.3f} "
                     f"bal={m['bal_acc']:.3f} yes_rate={m['yes_rate']:.3f} "
                     f"(n={m['n']}, pos={m['npos']})")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    np.savez(out / "scores.npz", scores=np.array(allS), labels=np.array(allY),
             verbs=np.array([r[0] for r in rows]))
    print("\n".join(lines)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
