#!/usr/bin/env python3
"""Candidate-solution #1 premise test: is distinct-room count more readable from POOLED FRAME-TOKEN
hidden states than from the frame->carrier attention messages?

The message probe (probe_rooms_visited_messages.py) showed distinct-room count is only weakly readable
from the carrier attention-message channel (peak ~0.30 at L16). Hypothesis: the information is richer in
the frame TOKEN residual states themselves (the corruption ablation shows frames remain causally
restorable). If a linear probe reads distinct-room count much better from pooled frame-token states than
from messages (~0.30), then a frame-token-pooling adapter is worth building. If similar, it's a limiting
result (the info isn't linearly accessible from token states either).

For N rooms_visited samples: capture residual hidden states at the frame token positions at several
layers, pool (mean/sum) over frame tokens, and Ridge-probe distinct-room count (multi-seed). No training.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from evaluations.scripts.patch_importence import token_group_corruption_new_tasks as tgc
from models.model import get_layers, image_token_groups


def probe_multiseed(x: np.ndarray, y: np.ndarray, seeds: List[int]) -> Dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    accs, maes, baccs = [], [], []
    for s in seeds:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=s,
                                              stratify=y if len(set(y.tolist())) > 1 else None)
        clf = Ridge(alpha=1.0, random_state=s).fit(xtr, ytr)
        pf = clf.predict(xte); pr = np.rint(pf).astype(int)
        accs.append(accuracy_score(yte, pr)); maes.append(mean_absolute_error(yte, pf))
        bp = int(round(float(np.mean(ytr)))); baccs.append(accuracy_score(yte, np.full_like(yte, bp)))
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "mae": float(np.mean(maes)), "blind": float(np.mean(baccs)), "n": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rooms_visited", choices=["rooms_visited", "co_occupancy"])
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=90)
    ap.add_argument("--layers", default="4,6,8,12,16,20")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--output", default="outputs/probe_frame_token_states")
    args = ap.parse_args()

    from models.model import DEFAULT_MODEL_ID
    gri.configure_runtime(args.model_name or DEFAULT_MODEL_ID)
    lm = LanguageModel(gri._model(), tokenizer=gri._processor().tokenizer)
    layers = get_layers(lm.model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # feats[(pool,layer)] = list of vectors ; ys aligned
    feats: Dict[tuple, List[np.ndarray]] = {}
    ys: List[int] = []
    n = 0
    for i, sd in enumerate(iter_sample_dirs(Path(args.data_root)), 1):
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        spec = tgc.task_spec(args.task, states, len(frames), qa_question=q0, qa_answer=a0)
        if spec is None:
            continue
        question, gold, _ev = spec
        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
            fg = image_token_groups(inputs["input_ids"][0].detach().cpu(),
                                    expected_num_frames=len(frames), processor=gri._processor())
        except Exception:
            continue
        frame_pos = [int(p) for g in fg for p in g]
        if not frame_pos:
            continue
        saved: Dict[int, Any] = {}
        try:
            with torch.no_grad():
                with lm.trace(inputs):
                    for L in probe_layers:
                        saved[L] = tgi._to_hidden_tensor(layers[L].output).save()
            pooled = {}
            for L in probe_layers:
                h = tgi._materialize_saved(saved[L])  # [1, seq, dim]
                fs = h[0, frame_pos, :].float()
                pooled[("mean", L)] = fs.mean(0).cpu().numpy().astype(np.float32)
                pooled[("sum", L)] = fs.sum(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            import traceback
            if i <= 3:
                traceback.print_exc()
            print(f"[{i}] {sid} capture failed: {type(exc).__name__}: {exc}")
            continue
        for k, v in pooled.items():
            feats.setdefault(k, []).append(v)
        ys.append(int(gold)); n += 1
        if n % 20 == 0:
            print(f"  captured {n} samples")

    y = np.asarray(ys, dtype=int)
    lines = [f"=== frame-token-state probe: {args.task}  n={len(y)}  (compare to message probe ~0.30) ==="]
    rows = ["pool,layer,n,acc,acc_std,mae,blind,lift"]
    for pool in ("mean", "sum"):
        for L in probe_layers:
            key = (pool, L)
            if key not in feats:
                continue
            x = np.stack(feats[key])
            if len(y) < 10 or len(set(y.tolist())) < 2:
                continue
            r = probe_multiseed(x, y, seeds)
            lift = r["acc"] - r["blind"]
            lines.append(f"  {pool:<5} L{L:<2d} n={r['n']:<4d} acc={r['acc']:.3f}±{r['acc_std']:.3f} "
                         f"mae={r['mae']:.2f} (blind {r['blind']:.3f}) lift={lift:+.3f}")
            rows.append(f"{pool},{L},{r['n']},{r['acc']:.4f},{r['acc_std']:.4f},{r['mae']:.4f},{r['blind']:.4f},{lift:.4f}")
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "probe_report.txt").write_text(report, encoding="utf-8")
    (out / "probe_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/probe_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
