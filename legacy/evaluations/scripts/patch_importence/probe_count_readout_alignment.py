#!/usr/bin/env python3
"""Absent-vs-misaligned probe: is the gold COUNT present at the last/readout token, and if so, is its
linear direction ALIGNED with the count-token unembedding rows?

Motivated by Garcia 2605.03258 ("The Right Answer, the Wrong Direction"): in TEXT LLMs the count is
linearly decodable at the last token (R^2>0.99) but the count direction is ~orthogonal to the digit
output-head rows (|cos|<=0.032) -> a READOUT/geometry bottleneck, not absence. Our earlier MMRED probe
found last-token count CLASSIFICATION ~= majority and concluded "count absent / never consolidated".
This script re-tests with (a) REGRESSION R^2 (continuous presence) and (b) UNEMBEDDING-ALIGNMENT cosine,
on IMAGE and TEXT, uncrowded vs crowded -> tells us whether to BUILD an aggregator or ALIGN a readout.

Per layer we report, at the last prompt token:
  - reg_r2   : Ridge regression R^2 predicting the integer gold count   (>~0.9 => count IS present)
  - cls_acc  : 9-way classification accuracy vs majority baseline       (our original "absent" metric)
  - align    : max_k |cos(count_direction, unembed_row[token('k')])|     (low => present-but-misaligned)
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


def reg_r2(x, y, seeds):
    """Mean test R^2 of a Ridge regression predicting the integer count from last-token reps."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    r2s = []
    for s in seeds:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=s)
        clf = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        clf.fit(xtr, ytr)
        r2s.append(r2_score(yte, clf.predict(xte)))
    return float(np.mean(r2s)), float(np.std(r2s))


def cls_acc(x, y, seeds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    accs, base = [], []
    yi = [int(v) for v in y]
    for s in seeds:
        strat = yi if min(Counter(yi).values()) >= 2 else None
        xtr, xte, ytr, yte = train_test_split(x, yi, test_size=0.35, random_state=s, stratify=strat)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
        clf.fit(xtr, ytr)
        accs.append(accuracy_score(yte, clf.predict(xte)))
        mc = Counter(ytr).most_common(1)[0][0]
        base.append(accuracy_score(yte, [mc] * len(yte)))
    return float(np.mean(accs)), float(np.mean(base))


def count_direction_alignment(x, y, unembed_rows):
    """Fit Ridge on RAW reps -> count axis w [hidden]; report max & mean |cos(w, unembed_row[k])|."""
    from sklearn.linear_model import Ridge
    w = Ridge(alpha=10.0, fit_intercept=True).fit(x, y).coef_.astype(np.float64)  # [hidden]
    wn = w / (np.linalg.norm(w) + 1e-8)
    cos = []
    for r in unembed_rows:                       # each [hidden]
        rn = r / (np.linalg.norm(r) + 1e-8)
        cos.append(abs(float(wn @ rn)))
    return float(np.max(cos)), float(np.mean(cos))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_agg/steps_1char/seq_len_8/all_uniform")
    ap.add_argument("--text", action="store_true", help="feed frames as TEXT instead of images")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--layers", default="4,8,12,16,19,22,25,27")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-count", type=int, default=8)
    ap.add_argument("--output", default="outputs/frame_axis/probes/count_readout_alignment")
    ap.add_argument("--tag", default="image_uncrowded")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    proc = gri._processor()
    lm = LanguageModel(gri._model(), tokenizer=proc.tokenizer)
    layers = get_layers(lm.model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # count-token unembedding rows (the readout directions for "0".."max_count")
    # read from the RAW HF model (not the nnsight envoy) to avoid wrapper/quantization surprises
    W = gri._model().get_output_embeddings().weight.detach().float().cpu().numpy()  # [vocab, hidden]
    unembed_rows = []
    for k in range(args.max_count + 1):
        tid = tgi.token_ids_of_answer(str(k), processor=proc)[0]
        unembed_rows.append(W[tid].astype(np.float64))

    last_feats: Dict[int, List[np.ndarray]] = defaultdict(list)
    gold: List[int] = []
    # sample dirs are named/sorted by count (K0,K1,...); SHUFFLE so a --limit spans the full count range
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
            saved = {}
            with torch.no_grad():
                with lm.trace(inputs):
                    for L in probe_layers:
                        saved[L] = tgi._to_hidden_tensor(layers[L].output).save()
            for L in probe_layers:
                h = tgi._materialize_saved(saved[L])[0][last_idx, :].float().cpu().numpy().astype(np.float32)
                last_feats[L].append(h)
        except Exception as exc:
            print(f"{sid} capture failed: {type(exc).__name__}: {exc}")
            continue
        gold.append(g)
        n += 1
        if n % 25 == 0:
            print(f"  scanned {n}: counts so far {sorted(Counter(gold).items())}", flush=True)

    y = np.array(gold, dtype=np.float64)
    lines = [f"=== COUNT READOUT-ALIGNMENT PROBE  tag={args.tag} text={args.text} n={len(gold)} ===",
             f"gold-count dist: {sorted(Counter(gold).items())}",
             f"interpretation: high reg_r2 + LOW align => count PRESENT but MISALIGNED (readout fix);"
             f" low reg_r2 => count ABSENT (build aggregator)",
             f"\n{'layer':>5} {'reg_r2':>10} {'cls_acc':>9} {'majority':>9} {'align_max':>10} {'align_mean':>11}"]
    rows = ["tag,text,layer,n,reg_r2,reg_r2_std,cls_acc,majority,align_max,align_mean"]
    for L in probe_layers:
        if len(last_feats[L]) < 12 or len(set(gold)) < 2:
            continue
        x = np.stack(last_feats[L])
        r2, r2s = reg_r2(x, y, seeds)
        acc, maj = cls_acc(x, y, seeds)
        amax, amean = count_direction_alignment(x, y, unembed_rows)
        lines.append(f"{L:>5} {r2:>10.3f} {acc:>9.3f} {maj:>9.3f} {amax:>10.4f} {amean:>11.4f}")
        rows.append(f"{args.tag},{int(args.text)},{L},{len(gold)},{r2:.4f},{r2s:.4f},{acc:.4f},{maj:.4f},{amax:.4f},{amean:.4f}")

    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
