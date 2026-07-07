#!/usr/bin/env python3
"""Consolidation ladder (steps_in_room): WHERE is the count precision lost?

One joint forward pass per sample. Capture, at layer L, each frame's mean-pooled frame-token rep
(+ per-frame evidence label "is C in R this frame") and the last-token rep (+ gold count). Then:

  B. decode-then-count : train a per-frame evidence probe on the JOINT-pass frame reps, predict each
       frame, SUM -> count. (= "the per-frame info is in the joint pass; sum it externally")
  C. last-token count  : train a count probe on the last-token rep -> count. (= the model's own
       consolidation site)

Compare to A = per-frame SEPARATE-pass verification (~0.93, measured elsewhere = extraction ceiling).

Reading:
  A->B gap  = extraction degradation under joint interference (compounded per-frame noise)
  B->C gap  = consolidation loss writing the sum into the carrier/last token (over-squashing)
Whichever gap is larger is where the precision is lost.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter
from typing import List, Dict
import numpy as np
import torch
from nnsight import LanguageModel

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.eval_mmred_text_frames_acc import room_of
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups


def build_qfirst_inputs(frames, question, proc):
    """QUESTION-FIRST ordering: question precedes images so each frame's vision tokens are
    query-conditioned (matches the per_frame_vision_reps probe that gets ~0.94); answer cue after
    images so the last token still attends to everything (valid count read-out site)."""
    n = len(frames)
    pre = f"Question: {question}\nHere are the {n} frames showing rooms in a house:"
    post = f"\nRespond with a single integer from 0 to {n}. Output only the integer.\nAnswer: "
    messages = [{"role": "user", "content":
                 [{"type": "text", "text": pre}]
                 + [{"type": "image", "image": im} for im in frames]
                 + [{"type": "text", "text": post}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")
    return dict(inputs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_steps_balanced/seq_len_8/all_uniform")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", default="outputs/frame_axis/probes/consolidation_ladder")
    ap.add_argument("--tag", default="steps_5char")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    proc = gri._processor()
    lm = LanguageModel(gri._model(), tokenizer=proc.tokenizer)
    layers = get_layers(lm.model)
    L = int(args.layer)
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    samples: List[Dict] = []   # per sample: frame_reps [N,H], frame_labels [N], last_rep [H], gold
    import random
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(0).shuffle(all_dirs)
    n = 0
    for sd in all_dirs:
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            meta = json.loads((sd / "metadata.json").read_text())
            C, R = meta.get("target_character"), meta.get("target_room")
            if not C or not R:
                continue
            labels = [int(room_of(st, C) == R) for st in states]
        except Exception:
            continue
        try:
            inputs = tgi.move_inputs_to_model_device(build_qfirst_inputs(frames, q0, proc))
            fg = image_token_groups(inputs["input_ids"][0].detach().cpu(),
                                    expected_num_frames=len(frames), processor=proc)
            last_idx = int(inputs["input_ids"].shape[1]) - 1
            with torch.no_grad():
                with lm.trace(inputs):
                    saved = tgi._to_hidden_tensor(layers[L].output).save()
            h = tgi._materialize_saved(saved)[0].float().cpu().numpy()   # [seq, H]
            if len(fg) != len(states):
                continue
            frame_reps = np.stack([h[[int(p) for p in fg[t]], :].mean(0) for t in range(len(states))])  # [N,H]
            samples.append(dict(fr=frame_reps.astype(np.float32), fl=np.array(labels),
                                last=h[last_idx, :].astype(np.float32), gold=gold))
        except Exception as exc:
            print(f"{sid} fail: {type(exc).__name__}: {exc}", flush=True)
            continue
        n += 1
        if n % 25 == 0:
            print(f"  {n}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    idx = np.arange(len(samples))
    pf_auc, b_hard, b_soft, c_acc, c_maj = [], [], [], [], []
    for s in seeds:
        rng = np.random.RandomState(s); perm = rng.permutation(idx)
        cut = int(0.65 * len(perm)); tr, te = perm[:cut], perm[cut:]
        # ---- per-frame evidence probe (trained on JOINT-pass frame reps of train samples) ----
        Xtr = np.concatenate([samples[i]["fr"] for i in tr]); ytr = np.concatenate([samples[i]["fl"] for i in tr])
        if len(set(ytr.tolist())) < 2:
            continue
        pf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s)).fit(Xtr, ytr)
        # frame-level AUC on test frames
        Xte = np.concatenate([samples[i]["fr"] for i in te]); yte = np.concatenate([samples[i]["fl"] for i in te])
        from evaluations.scripts.probe_evidence_selection_linear import auc_score
        pf_auc.append(float(auc_score(torch.tensor(pf.predict_proba(Xte)[:, 1]), torch.tensor(yte))))
        # ---- B: decode-then-count per TEST sample ----
        bh, bs = [], []
        for i in te:
            p = pf.predict_proba(samples[i]["fr"])[:, 1]
            bh.append(int(int((p > 0.5).sum()) == samples[i]["gold"]))
            bs.append(int(int(round(float(p.sum()))) == samples[i]["gold"]))
        b_hard.append(np.mean(bh)); b_soft.append(np.mean(bs))
        # ---- C: last-token count probe ----
        Ltr = np.stack([samples[i]["last"] for i in tr]); gtr = np.array([samples[i]["gold"] for i in tr])
        Lte = np.stack([samples[i]["last"] for i in te]); gte = np.array([samples[i]["gold"] for i in te])
        cc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s)).fit(Ltr, gtr)
        c_acc.append(float((cc.predict(Lte) == gte).mean()))
        mc = Counter(gtr.tolist()).most_common(1)[0][0]; c_maj.append(float((gte == mc).mean()))

    def m(x): return float(np.mean(x)) if x else float("nan")
    lines = [f"=== CONSOLIDATION LADDER  tag={args.tag}  L={L}  n={len(samples)} ===",
             f"per-frame evidence probe AUC (joint pass): {m(pf_auc):.3f}",
             "",
             f"A. per-frame SEPARATE passes (extraction ceiling, measured elsewhere) : ~0.93",
             f"B. decode-then-count from JOINT-pass frame tokens : hard={m(b_hard):.3f}  soft={m(b_soft):.3f}",
             f"C. last-token consolidated count                  : {m(c_acc):.3f}  (majority {m(c_maj):.3f})",
             "",
             f"A->B gap (joint interference on extraction) ~ {0.93 - max(m(b_hard), m(b_soft)):+.3f}",
             f"B->C gap (consolidation into carrier)       ~ {m(c_acc) - max(m(b_hard), m(b_soft)):+.3f}",
             "  larger gap = where count precision is lost (A->B: compounded extraction; B->C: over-squashing)"]
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
