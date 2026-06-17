#!/usr/bin/env python3
"""Per-frame MESSAGE (frame tokens -> question/carrier tokens) probes for the 2 set-cardinality tasks.

The actual message routed from frame f's tokens into a carrier position c at layer L is
    msg_{f->c}^L = o_proj_L( concat_h ( sum_{j in frame_f} A^h[c,j] * v^h[j] ) )
(o_proj is linear so the per-frame split is exact). Carrier = ALL question tokens (the span after the
image block through the last token) averaged, since we don't know the carriers for the new tasks.

Two experiments, each per layer (default 16,18,19), linear AND MLP, with PCA (p>>n guard) + shuffle ctrl:
  (A) PER-FRAME evidence from msg_{f->carrier}  (room / same-room)  -> does the evidence reach the carrier?
  (B) COUNT from the CONCATENATION of all frames' messages [msg_1|...|msg_8] -> can the count be
      recovered if the probe sees every frame? Contrast vs SUM_f msg_f (imposes sum-aggregation) and
      vs the model's own answer. concat>>sum & concat>>model  =>  AGGREGATION bottleneck.
Eager attention (output_attentions). 7B. --limit 150 default (small but enough for the concat dim).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)


def char_room_at(states, t, char) -> Optional[str]:
    for room, occ in eval_utils.rooms_to_room2chars(states[t].get("rooms", {})).items():
        if char in occ:
            return room
    return None


def pick_pair(states, chars):
    best, best_gold = None, -1
    for c1, c2 in combinations(sorted(chars), 2):
        g = sum(1 for t in range(len(states))
                if char_room_at(states, t, c1) is not None
                and char_room_at(states, t, c1) == char_room_at(states, t, c2))
        if g > best_gold:
            best, best_gold = (c1, c2), g
    return best, best_gold


def make_clf(spec, n_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    kind = spec.get("kind", "logistic"); pca = spec.get("pca", 128); arch = spec.get("arch", (256,))
    # early_stopping=False on purpose: its internal stratified val-split breaks on
    # small/imbalanced multiclass folds (test_size<n_classes) and runs np.isnan on
    # string preds. Training-loss stopping via n_iter_no_change is fine for a probe.
    head = (MLPClassifier(hidden_layer_sizes=tuple(arch), max_iter=900, random_state=0,
                          early_stopping=False, n_iter_no_change=20, tol=1e-4)
            if kind == "mlp" else LogisticRegression(max_iter=3000, random_state=0))
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=max(2, min(int(pca), n_train - 1)), random_state=0))
    steps.append(head)
    return make_pipeline(*steps)


# Probe-optimization sweep configs (capacity / PCA dim)
PROBE_SPECS = [
    {"name": "logistic_pca256", "kind": "logistic", "pca": 256},
    {"name": "logistic_nopca",  "kind": "logistic", "pca": None},
    {"name": "mlp256_pca256",   "kind": "mlp", "arch": (256,), "pca": 256},
    {"name": "mlp512x256_pca256", "kind": "mlp", "arch": (512, 256), "pca": 256},
]


def probe(x, y, seeds, spec, binary=False, shuffle=False):
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    accs, aucs, base = [], [], []
    # integer-encode labels: MLPClassifier(early_stopping=True) runs np.isnan on
    # y_pred, which fails for string labels (room names). Codes are metric-neutral.
    classes = sorted(set(y))
    code = {c: i for i, c in enumerate(classes)}
    y = np.array([code[v] for v in y])
    for s in seeds:
        yy = y.copy()
        if shuffle:
            rng = np.random.RandomState(s); rng.shuffle(yy)
        strat = yy if len(set(yy)) > 1 and min(Counter(yy).values()) >= 2 else None
        xtr, xte, ytr, yte = train_test_split(x, yy, test_size=0.35, random_state=s, stratify=strat)
        clf = make_clf(spec, len(xtr)); clf.fit(xtr, ytr)
        pred = clf.predict(xte)
        accs.append(accuracy_score(yte, pred))
        mc = Counter(ytr).most_common(1)[0][0]
        base.append(accuracy_score(yte, [mc] * len(yte)))
        if binary and len(set(ytr)) == 2:
            try:
                aucs.append(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))
            except Exception:
                pass
    return (float(np.mean(accs)), float(np.std(accs)), float(np.mean(base)),
            float(np.mean(aucs)) if aucs else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rooms_visited", "co_occupancy", "count"], required=True)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--layers", default="16,18,19")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--output", default="outputs/probe_frame_to_carrier_message")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)  # SDPA (codebase forbids eager/output_attentions)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope_section = (getattr(cfg, "rope_scaling", None) or {}).get("mrope_section", None)
    attn_scale = head_dim ** -0.5
    NF = int(args.n_frames)
    out = Path(args.output) / args.task; out.mkdir(parents=True, exist_ok=True)
    # candidate single-token ids for digits 0..8 (proper count accuracy, not raw vocab argmax)
    tok = processor.tokenizer
    cand_ids, cand_vals = [], []
    for d in range(0, 9):
        enc = tok.encode(str(d), add_special_tokens=False)
        if len(enc) == 1:
            cand_ids.append(int(enc[0])); cand_vals.append(d)
    cand_ids_t = torch.tensor(cand_ids, dtype=torch.long)

    # per-layer accumulators
    pf_feat: Dict[int, List[np.ndarray]] = {L: [] for L in probe_layers}   # per-frame evidence feature
    pf_lab: List[str] = []
    pf_sample: List[int] = []  # which sample each per-frame example belongs to (for decode-then-count)
    concat_feat: Dict[int, List[np.ndarray]] = {L: [] for L in probe_layers}  # concat of NF frame msgs
    sum_feat: Dict[int, List[np.ndarray]] = {L: [] for L in probe_layers}     # sum of frame msgs
    cnt_lab: List[int] = []
    model_correct: List[int] = []
    n = 0
    for sd in iter_sample_dirs(Path(args.data_root)):
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        chars = sorted(eval_utils.extract_characters_from_states(states))
        if len(chars) < 2:
            continue
        if args.task == "count":
            # original counting task: question/answer from qa, evidence = matching frames
            try:
                gold = int(str(a0).strip())
            except Exception:
                continue
            evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            if not evid:
                continue
            question = q0
            frame_targets = {t: ("evid" if t in evid else "noev") for t in range(len(frames))}
        elif args.task == "rooms_visited":
            present = lambda c: [t for t in range(len(states)) if char_room_at(states, t, c)]
            char = max(chars, key=lambda c: (len(present(c)), c))
            pres = present(char)
            if len(pres) < 2:
                continue
            gold = len({char_room_at(states, t, char) for t in pres})
            question = f"How many distinct rooms did {char} visit?"
            frame_targets = {t: char_room_at(states, t, char) for t in pres}
        else:
            (c1, c2), gold = pick_pair(states, chars)
            if gold < 1:
                continue
            question = f"In how many of the {len(frames)} frames were {c1} and {c2} in the same room?"
            frame_targets = {}
            for t in range(len(states)):
                r1, r2 = char_room_at(states, t, c1), char_room_at(states, t, c2)
                if r1 is not None and r2 is not None:
                    frame_targets[t] = "same" if r1 == r2 else "diff"

        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
            ids = inputs["input_ids"][0].detach().cpu()
            fg = image_token_groups(ids, expected_num_frames=len(frames), processor=processor)
            last_img = max(int(p) for grp in fg for p in grp)
            seq = int(ids.shape[0])
            carrier = list(range(last_img + 1, seq))  # all question tokens (span after images)
            if not carrier or len(fg) < NF:
                continue
            # SANCTIONED capture: model stays SDPA. Grab pre-rotary q/k/v proj outputs + the
            # position embeddings (cos,sin), then recompute the carrier-row softmax offline.
            qkv: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in probe_layers}
            posemb: Dict[str, torch.Tensor] = {}
            handles = []
            for L in probe_layers:
                for nm in ("q_proj", "k_proj", "v_proj"):
                    def mk(L, nm):
                        def hook(_m, _i, o):
                            qkv[L][nm] = o.detach()[0]
                        return hook
                    handles.append(getattr(layers[L].self_attn, nm).register_forward_hook(mk(L, nm)))
                def mk_pe(_m, args_, kwargs_):
                    pe = kwargs_.get("position_embeddings", None)
                    if pe is None and len(args_) >= 1 and isinstance(args_[-1], tuple):
                        pe = args_[-1]
                    if pe is not None and "cos" not in posemb:
                        posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
                handles.append(layers[probe_layers[0]].self_attn.register_forward_pre_hook(mk_pe, with_kwargs=True))
            with torch.no_grad():
                outp = model(**inputs, use_cache=False)
            for h in handles:
                h.remove()
            # proper count prediction: argmax over candidate digit tokens only
            last_logits = outp.logits[0, -1].float().cpu()
            pred = int(cand_vals[int(torch.argmax(last_logits[cand_ids_t]).item())])
            if "cos" not in posemb:
                raise RuntimeError("position_embeddings not captured")
        except Exception as exc:
            print(f"{sid} capture failed: {type(exc).__name__}: {exc}")
            continue

        carrier_t = torch.tensor(carrier, dtype=torch.long)
        # recompute attention weights for carrier rows from captured post-rotary q,k (identical math
        # to the model's SDPA, just materialized offline for the analysis), per layer.
        attnA: Dict[int, torch.Tensor] = {}
        vrep: Dict[int, torch.Tensor] = {}
        cos = posemb["cos"]; sin = posemb["sin"]
        for L in probe_layers:
            q = qkv[L]["q_proj"].view(1, seq, n_heads, head_dim).transpose(1, 2)   # [1,H,S,hd]
            k = qkv[L]["k_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
            v = qkv[L]["v_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
            if mrope_section is not None:
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
            k = repeat_kv(k, n_heads // n_kv); v = repeat_kv(v, n_heads // n_kv)    # [1,H,S,hd]
            qf = q[0].float().cpu(); kf = k[0].float().cpu()
            scores = torch.einsum("hcd,hkd->hck", qf[:, carrier_t], kf) * attn_scale   # [H,|C|,S]
            key_idx = torch.arange(seq); allow = key_idx[None, :] <= carrier_t[:, None]  # causal
            scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
            attnA[L] = torch.softmax(scores, dim=-1)                                # [H,|C|,S]
            vrep[L] = v[0].float().cpu()                                            # [H,S,hd]

        def msg_layer(pos_list, L):
            pos = torch.tensor([p for p in pos_list if p < seq], dtype=torch.long)
            if pos.numel() == 0:
                return np.zeros(int(cfg.hidden_size), dtype=np.float32)
            Asel = attnA[L][:, :, pos]                          # [H,|C|,|f|]
            vsel = vrep[L][:, pos, :]                           # [H,|f|,hd]
            ctx = torch.einsum("hcj,hjd->hd", Asel, vsel) / max(1, len(carrier))   # [H,hd]
            # o_proj is 4-bit (bnb) -> apply as a module (dequantizes internally), don't touch .weight
            oproj = layers[L].self_attn.o_proj
            dev = next(oproj.parameters()).device
            with torch.no_grad():
                out = oproj(ctx.reshape(1, -1).to(device=dev, dtype=torch.bfloat16))
            return out[0].float().cpu().numpy().astype(np.float32)

        hidden = int(cfg.hidden_size)
        per_layer_frames = {L: [] for L in probe_layers}
        for t in range(NF):
            posl = [int(p) for p in fg[t]]
            for L in probe_layers:
                m = msg_layer(posl, L)
                per_layer_frames[L].append(m)
                if t in frame_targets and frame_targets[t] is not None:
                    pf_feat[L].append(m)
            if t in frame_targets and frame_targets[t] is not None:
                pf_lab.append(str(frame_targets[t]))
                pf_sample.append(len(cnt_lab))  # this sample's index (cnt_lab appended just below)
        for L in probe_layers:
            stk = np.stack(per_layer_frames[L])           # [NF, hidden]
            concat_feat[L].append(stk.reshape(-1))         # [NF*hidden]
            sum_feat[L].append(stk.sum(0))                 # [hidden]
        cnt_lab.append(int(gold))
        model_correct.append(int(pred == gold))
        n += 1
        if n % 10 == 0:
            print(f"  scanned {n}: {len(pf_lab)} frame-msg, {len(cnt_lab)} count examples")

    lines = [f"=== FRAME->CARRIER MESSAGE PROBES ({args.task}, 7B) n_samples={len(cnt_lab)} "
             f"layers={probe_layers} carriers=all_question_tokens ==="]
    if model_correct:
        lines.append(f"model own-answer accuracy (candidate-digit argmax): {np.mean(model_correct):.3f}")
    rows = ["experiment,layer,clf,n,acc,acc_std,majority,lift,auroc,shuffle_acc"]

    ym = np.array(pf_lab); binary = len(set(ym.tolist())) == 2
    # multi-layer concat feature (per-frame example), aligned with pf_lab/pf_sample
    aligned = all(len(pf_feat[L]) == len(pf_lab) for L in probe_layers)
    pf_feats = {f"L{L}": pf_feat[L] for L in probe_layers}
    if aligned and len(probe_layers) > 1:
        pf_feats["concat"] = [np.concatenate([pf_feat[L][i] for L in probe_layers]) for i in range(len(pf_lab))]

    lines.append(f"\n(A) PER-FRAME evidence — PROBE SWEEP (feature x capacity)  classes={sorted(set(ym.tolist()))} n={len(ym)}")
    bestA = ("", -1.0)
    for fname, feats in pf_feats.items():
        if len(feats) < 12 or len(set(ym.tolist())) < 2:
            continue
        xm = np.stack(feats)
        for spec in PROBE_SPECS:
            acc, std, base, auc = probe(xm, ym, seeds, spec, binary=binary)
            score = auc if (binary and auc == auc) else acc
            lines.append(f"  {fname:<7s} {spec['name']:<17s}: acc={acc:.3f}±{std:.3f} (maj {base:.3f})"
                         + (f" AUROC={auc:.3f}" if binary else ""))
            rows.append(f"per_frame_msg,{fname},{spec['name']},{len(ym)},{acc:.4f},{std:.4f},{base:.4f},{acc-base:.4f},{auc:.4f},nan")
            if score > bestA[1]:
                bestA = (f"{fname}/{spec['name']}: acc={acc:.3f}" + (f" AUROC={auc:.3f}" if binary else ""), score)
    lines.append(f"  >> BEST per-frame: {bestA[0]}")

    ys = np.array(cnt_lab)
    _bspec = ({"name": "logistic", "kind": "logistic", "pca": 256}, {"name": "mlp512x256", "kind": "mlp", "arch": (512, 256), "pca": 256})
    lines.append(f"\n(B) COUNT from concat/sum of {NF} msgs (probe-capacity limited)  classes={sorted(set(ys.tolist()))} n={len(ys)}")
    for L in probe_layers:
        if len(concat_feat[L]) < 12 or len(set(ys.tolist())) < 2:
            continue
        for nm, feats in (("count_concat", concat_feat[L]), ("count_sum", sum_feat[L])):
            xs = np.stack(feats)
            for spec in _bspec:
                acc, std, base, _ = probe(xs, ys, seeds, spec)
                lines.append(f"  L{L} {nm:<13s} {spec['name']:<11s}: acc={acc:.3f}±{std:.3f} (maj {base:.3f})")
                rows.append(f"{nm},{L},{spec['name']},{len(ys)},{acc:.4f},{std:.4f},{base:.4f},{acc-base:.4f},nan,nan")

    # (C) DECODE-THEN-COUNT — PROBE SWEEP: best per-frame clf -> aggregate -> count vs model.
    from sklearn.model_selection import GroupShuffleSplit
    samp_arr = np.array(pf_sample)
    # integer-encode per-frame labels for the MLP (np.isnan on string preds fails);
    # map the positive label through the same code so the count semantics survive.
    _classesC = sorted(set(ym.tolist()))
    _codeC = {c: i for i, c in enumerate(_classesC)}
    ym_c = np.array([_codeC[v] for v in ym.tolist()])
    pos_lab = {"co_occupancy": "same", "count": "evid"}.get(args.task)
    pos_code = _codeC.get(pos_lab)
    lines.append(f"\n(C) DECODE-THEN-COUNT — PROBE SWEEP (per-frame clf -> aggregate -> count) vs model")
    bestC = ("", -1.0)
    for fname, feats in pf_feats.items():
        if len(feats) < 20 or len(set(ym.tolist())) < 2:
            continue
        X = np.stack(feats)
        for spec in _bspec:
            accs, majs, modelaccs = [], [], []
            for s in seeds:
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=s).split(X, ym_c, groups=samp_arr))
                clf = make_clf(spec, len(tr)); clf.fit(X[tr], ym_c[tr])
                predf = clf.predict(X[te])
                per: Dict[int, list] = {}
                for idx, p in zip(te, predf):
                    per.setdefault(int(samp_arr[idx]), []).append(p)
                tsx = sorted(per)
                pc = {si: (len(set(per[si])) if args.task == "rooms_visited" else sum(1 for p in per[si] if p == pos_code)) for si in tsx}
                accs.append(np.mean([pc[si] == cnt_lab[si] for si in tsx]))
                modelaccs.append(np.mean([model_correct[si] for si in tsx]))
                majs.append(Counter(cnt_lab[si] for si in tsx).most_common(1)[0][1] / max(1, len(tsx)))
            dca = float(np.mean(accs))
            lines.append(f"  {fname:<7s} {spec['name']:<11s}: decode_then_count={dca:.3f}±{np.std(accs):.3f}  model={np.mean(modelaccs):.3f}  maj={np.mean(majs):.3f}")
            rows.append(f"decode_then_count,{fname}/{spec['name']},dtc,{len(ym)},{dca:.4f},{np.std(accs):.4f},{np.mean(majs):.4f},{dca-np.mean(modelaccs):.4f},nan,{np.mean(modelaccs):.4f}")
            if dca > bestC[1]:
                bestC = (f"{fname}/{spec['name']} = {dca:.3f} (model {np.mean(modelaccs):.3f}, maj {np.mean(majs):.3f})", dca)
    lines.append(f"  >> BEST decode-then-count: {bestC[0]}")

    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
