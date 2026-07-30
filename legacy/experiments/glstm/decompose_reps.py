#!/usr/bin/env python3
"""Reusable: extract per-frame L_read reps from a (possibly LoRA'd) model, and compute the aggregation
decomposition metrics. Used inside lora_sft_baseline.py to measure the decomposition BEFORE vs AFTER the
fix (adapter disabled vs enabled) on the same held-out examples -> does the fix work *through* the
mechanism (delta grows, S_all/sigmoid-then-sum decodability rises)?"""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

import experiments.glstm.frame_axis_aggregator_adapter as fa
from experiments.glstm.cache_minimal_frame_reps import frame_labels
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import image_token_groups
from experiments.glstm.probe_message_sum_decodability import fit_ridge, agg_sum


def extract_reps(model, processor, target_layer, items, read_layer, task, device, build_messages,
                 cap: int = 300) -> Dict[str, dict]:
    st = {"spans": None, "cur_pos": -1, "frame_reps": None, "query_rep": None}

    def edit(hs):
        if st["spans"] is None or hs.shape[1] <= st["cur_pos"]:
            return hs
        st["frame_reps"] = torch.stack([hs[0, idx, :].float().mean(0).cpu() for idx in st["spans"]], 0)
        st["query_rep"] = hs[0, st["cur_pos"], :].float().cpu().clone()
        return hs

    def pre_hook(module, hargs, hkwargs):
        if len(hargs) >= 1:
            return (edit(hargs[0]),) + tuple(hargs[1:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["hidden_states"] = edit(hkwargs["hidden_states"]); return hargs, hkwargs
    handle = target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    cache: Dict[str, dict] = {}
    rng = random.Random(0)
    for (dstr, sl) in items[:cap]:
        d = Path(dstr)
        ex = fa.make_example(d, task, rng, eval_mode=True)
        if ex is None:
            continue
        frames, question, gold, nf, states = ex
        try:
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        msgs = build_messages(frames, question)
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = base.move_inputs_to_device(dict(inputs), device)
        ids = inputs["input_ids"]
        spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
        if len(spans) != len(frames):
            continue
        st["spans"] = spans; st["cur_pos"] = int(ids.shape[1]) - 1; st["frame_reps"] = None
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if st["frame_reps"] is None:
            continue
        cache[d.name] = {"reps": st["frame_reps"].half(), "query_rep": st["query_rep"].half(),
                         "gold": int(gold), "frame_labels": frame_labels(task, states, meta), "seq_len": int(nf)}
    handle.remove()
    return cache


def decompose(cache: Dict[str, dict], seq_len: int = 8, seeds=(0, 1, 2)) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    reps_l, gold_l = [], []
    for v in cache.values():
        if int(v.get("seq_len", -1)) != seq_len or v.get("frame_labels") is None:
            continue
        fl = v["frame_labels"]
        reps = v["reps"].float().numpy()
        try:
            lab = np.asarray([int(x) for x in fl])
        except (TypeError, ValueError):
            continue
        if lab.shape[0] != reps.shape[0] or not set(lab.tolist()) <= {0, 1}:
            continue
        reps_l.append((reps, lab, int(v["gold"]), v["query_rep"].float().numpy()))
        gold_l.append(int(v["gold"]))
    if len(reps_l) < 40:
        return {"n": len(reps_l), "error": "too few"}
    ev = np.stack([r[i] for r, lab, g, q in reps_l for i in range(r.shape[0]) if lab[i] == 1])
    nv = np.stack([r[i] for r, lab, g, q in reps_l for i in range(r.shape[0]) if lab[i] == 0])
    mu_all = np.concatenate([ev, nv]).mean(0); delta = (ev.mean(0) - nv.mean(0)) / 2.0
    dhat = delta / (np.linalg.norm(delta) + 1e-9)
    sig = 0.5 * ((ev @ dhat).std() + (nv @ dhat).std())
    snr = abs((ev.mean(0) - nv.mean(0)) @ dhat) / (sig + 1e-9)
    gold = np.asarray([g for _, _, g, _ in reps_l])
    S_all = np.stack([agg_sum(r) for r, _, _, _ in reps_l])
    S_evid = np.stack([agg_sum(r, np.where(lab == 1)[0]) for r, lab, _, _ in reps_l])
    Xq = np.stack([q for _, _, _, q in reps_l])
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    # sigmoid-then-sum (per-frame logistic -> soft count)
    idx = np.arange(len(reps_l)); tr, te = train_test_split(idx, test_size=0.35, random_state=0)
    N = reps_l[0][0].shape[0]
    Xtr = np.stack([reps_l[i][0][f] for i in tr for f in range(N)])
    ytr = np.asarray([reps_l[i][1][f] for i in tr for f in range(N)])
    sts_acc = float("nan")
    try:
        sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
        soft = [int(round(clf.predict_proba(sc.transform(reps_l[i][0]))[:, 1].sum())) for i in te]
        sts_acc = float(accuracy_score([reps_l[i][2] for i in te], soft))
    except Exception:
        pass
    return {
        "n": len(reps_l),
        "delta_over_mu": float(np.linalg.norm(delta) / np.linalg.norm(mu_all)),
        "perframe_snr": float(snr),
        "corr_Sevid_g": corr(np.linalg.norm(S_evid, axis=1), gold),
        "corr_Sall_g": corr(np.linalg.norm(S_all, axis=1), gold),
        "S_all_linear_acc": fit_ridge(S_all, gold, list(seeds))["acc"],
        "S_all_R2": fit_ridge(S_all, gold, list(seeds))["r2"],
        "last_tok_acc": fit_ridge(Xq, gold, list(seeds))["acc"],
        "sigmoid_then_sum_acc": sts_acc,
    }
