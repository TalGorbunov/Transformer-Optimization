#!/usr/bin/env python3
"""Readout-ceiling test: how much count is RECOVERABLE from the last-token rep, vs what the model emits.

Combines two Garcia-style validations in one frozen pass:
  - MODEL acc        : the frozen model's own constrained prediction (argmax of digit logits over 0..N).
  - LINEAR readout   : logistic head on the last-token rep -> exact count (= "digit-row repair" proxy;
                       if LINEAR > MODEL, the model under-reads a count its own head doesn't align to).
  - MLP readout      : 1-hidden-layer head on the same rep -> is the ceiling LINEAR-limited or is there
                       non-linear headroom? (if MLP >> LINEAR, the count is present but not linearly clean).
  - reg_r2           : continuous count R^2 (axis quality).

If MODEL ~ LINEAR ~ MLP and all modest -> consolidation is genuinely noisy (the wall is aggregation
quality, not readout). If LINEAR/MLP >> MODEL -> readout/misalignment is leaving accuracy on the table.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import numpy as np
import torch
from nnsight import LanguageModel

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.eval_mmred_text_frames_acc import frames_as_text
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers


def two_heads(x, y, seeds):
    """Return (linear_acc, mlp_acc, reg_r2) as held-out means over seeds."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split
    yi = [int(v) for v in y]
    lin, mlp, r2 = [], [], []
    for s in seeds:
        strat = yi if min(Counter(yi).values()) >= 2 else None
        xtr, xte, ytr, yte = train_test_split(x, yi, test_size=0.35, random_state=s, stratify=strat)
        lc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s)).fit(xtr, ytr)
        mc = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=800, random_state=s)).fit(xtr, ytr)
        rc = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(xtr, [float(v) for v in ytr])
        lin.append(accuracy_score(yte, lc.predict(xte)))
        mlp.append(accuracy_score(yte, mc.predict(xte)))
        r2.append(r2_score([float(v) for v in yte], rc.predict(xte)))
    return float(np.mean(lin)), float(np.mean(mlp)), float(np.mean(r2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--layers", default="19,22,27")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-count", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="cond")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    proc = gri._processor()
    lm = LanguageModel(gri._model(), tokenizer=proc.tokenizer)
    layers = get_layers(lm.model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    cand_tids = [tgi.token_ids_of_answer(str(k), processor=proc)[0] for k in range(args.max_count + 1)]

    feats: Dict[int, List[np.ndarray]] = defaultdict(list)
    gold: List[int] = []
    model_pred: List[int] = []
    import random
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(0).shuffle(all_dirs)
    n = 0
    for sd in all_dirs:
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            g = int(str(a0).strip())
        except Exception:
            continue
        try:
            if args.text:
                prompt = (f"{frames_as_text(states)}\n"
                          f"Respond with a single integer from 0 to {len(states)}. Output only the integer.\n"
                          f"Question: {q0}\nAnswer: ")
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([], prompt, processor=proc))
            else:
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0, processor=proc))
            last_idx = int(inputs["input_ids"].shape[1]) - 1
            saved, saved_logits = {}, None
            with torch.no_grad():
                with lm.trace(inputs):
                    for L in probe_layers:
                        saved[L] = tgi._to_hidden_tensor(layers[L].output).save()
                    saved_logits = lm.output.logits[0, last_idx, :].save()
            for L in probe_layers:
                feats[L].append(tgi._materialize_saved(saved[L])[0][last_idx, :].float().cpu().numpy().astype(np.float32))
            lg = tgi._materialize_saved(saved_logits).float().cpu().numpy()
            mp = int(np.argmax([lg[t] for t in cand_tids]))   # model's constrained digit prediction
            model_pred.append(mp)
        except Exception as exc:
            print(f"{sid} fail: {type(exc).__name__}: {exc}", flush=True)
            continue
        gold.append(g)
        n += 1
        if n % 25 == 0:
            print(f"  {n}: counts {sorted(Counter(gold).items())}", flush=True)

    y = np.array(gold)
    model_acc = float(np.mean(np.array(model_pred) == y)) if gold else float("nan")
    lines = [f"=== READOUT CEILING  tag={args.tag} text={args.text} n={len(gold)} ===",
             f"gold dist: {sorted(Counter(gold).items())}",
             f"MODEL constrained acc (frozen digit logits): {model_acc:.3f}",
             f"  LINEAR>MODEL => under-read (readout fix helps); MLP>>LINEAR => nonlinear headroom; all~equal+low => noisy consolidation\n",
             f"{'layer':>5} {'reg_r2':>8} {'lin_acc':>8} {'mlp_acc':>8} {'lin-model':>10} {'mlp-lin':>8}"]
    rows = ["tag,text,layer,n,model_acc,reg_r2,lin_acc,mlp_acc"]
    for L in probe_layers:
        if len(feats[L]) < 12 or len(set(gold)) < 2:
            continue
        x = np.stack(feats[L])
        lin, mlp, r2 = two_heads(x, y, seeds)
        lines.append(f"{L:>5} {r2:>8.3f} {lin:>8.3f} {mlp:>8.3f} {lin-model_acc:>+10.3f} {mlp-lin:>+8.3f}")
        rows.append(f"{args.tag},{int(args.text)},{L},{len(gold)},{model_acc:.4f},{r2:.4f},{lin:.4f},{mlp:.4f}")
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
