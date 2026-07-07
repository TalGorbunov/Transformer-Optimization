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
import argparse, random, sys
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
    ap.add_argument("--task", choices=["rooms_visited", "co_occupancy", "count", "first_occurrence", "herbench_ac"], required=True)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--layers", default="16,18,19")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--output", default="outputs/probe_frame_to_carrier_message")
    ap.add_argument("--carrier", choices=["all_question", "last", "per_token"], default="all_question",
                    help="which carrier position(s) receive the message: mean over all question tokens, "
                         "just the last/answer token (deployed readout), or per_token = compute the message "
                         "into EACH question token separately and sweep SNR by offset-from-end (no pooling)")
    ap.add_argument("--max-offset", type=int, default=11, help="per_token: how many question tokens back to sweep")
    ap.add_argument("--decode-offsets", default="", help="per_token: comma offsets to store per-frame messages "
                    "and run count-decode (F/G), e.g. 9,12,13,14 (room + char region)")
    ap.add_argument("--sample-seed", type=int, default=0, help="shuffle seed for a representative count draw")
    ap.add_argument("--fence-cross-frame", action="store_true",
                    help="single-pass multipass emulation: block visual tokens from attending to OTHER "
                         "frames' visual tokens (4D additive mask) in layers 0..fence-upto-1; read layers "
                         "stay full-attention so the message definition is unchanged")
    ap.add_argument("--fence-upto", type=int, default=0,
                    help="fence layers [0, K); 0 = min(probe layers)")
    ap.add_argument("--save-messages", action="store_true",
                    help="per_token+decode-offsets: torch.save the per-frame carrier messages "
                         "(messages_cache.pt) so d'-parity / decomposition analyses run on CPU later")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)  # SDPA (codebase forbids eager/output_attentions)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    fence_holder = {"mask": None}
    if args.fence_cross_frame:
        fence_K = int(args.fence_upto) or min(probe_layers)
        assert fence_K <= min(probe_layers), "fence must end below the first read layer"

        def _fence_pre(_m, hargs, hkwargs):
            if fence_holder["mask"] is not None:
                hs = hargs[0] if hargs else hkwargs.get("hidden_states")
                mk = fence_holder["mask"]
                if hs is not None and mk.dtype != hs.dtype:   # match runtime compute dtype exactly
                    mk = mk.to(hs.dtype); fence_holder["mask"] = mk
                if len(hargs) >= 2:                      # attention_mask passed positionally
                    hargs = (hargs[0], mk) + tuple(hargs[2:])
                else:
                    hkwargs["attention_mask"] = mk
            return hargs, hkwargs
        for _L in range(fence_K):
            layers[_L].register_forward_pre_hook(_fence_pre, with_kwargs=True)
        print(f"[fence] cross-frame visual attention BLOCKED in layers 0..{fence_K-1} (mask dtype set at runtime)")
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
    MAXOFF = int(args.max_offset)  # sweep the last MAXOFF+1 question tokens (offset 0 = last/answer token)
    DEC_OFF = [int(x) for x in str(args.decode_offsets).replace(",", " ").split()] if args.decode_offsets else []
    pt: Dict[int, Dict[int, Dict[str, list]]] = {L: {} for L in probe_layers}  # pt[L][offset]={x:[],y:[]}
    pt_tok: Dict[int, Counter] = {}                                            # pt_tok[offset]=Counter(text)
    dec: Dict[int, Dict[int, list]] = {L: {o: [] for o in DEC_OFF} for L in probe_layers}  # dec[L][o]=[ [NF,H] ]
    dec_gold: List[int] = []
    dec_labels: List[np.ndarray] = []  # per-sample [NF] binary positive-class labels (task-mapped)
    dec_labels_raw: List[list] = []    # per-sample [NF] raw string labels (room names / same-diff / evid)
    n = 0
    if args.task == "herbench_ac":
        # HERBench Action-Counting samples prepped by experiments/herbench/prep_ac_frames.py:
        # <dir>/frame_XX.jpg + meta.json (per-frame is_evidence labels; gold = visible_count)
        all_dirs = sorted(d for d in Path(args.data_root).iterdir() if (d / "meta.json").exists())
    else:
        all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)  # iter order is count-grouped; shuffle for a representative draw
    for sd in all_dirs:
        if n >= int(args.limit):
            break
        if args.task == "herbench_ac":
            import json as _json
            from PIL import Image as _Image
            try:
                hb = _json.loads((sd / "meta.json").read_text())
                fpaths = sorted(sd.glob("frame_*.jpg"))
                frames = [_Image.open(p).convert("RGB") for p in fpaths]
            except Exception:
                continue
            if len(frames) != len(hb["frames"]):
                continue
            sid = hb["question_id"]
            gold = int(hb["visible_count"])
            pair = hb.get("pair") or " ".join(hb.get("pair_words", []))
            question = (f"In how many of these {len(frames)} frames is the person "
                        f"performing the action '{pair}'?")
            frame_targets = {i: ("evid" if fr["is_evidence"] else "noev")
                             for i, fr in enumerate(hb["frames"])}
            states = None
        else:
            try:
                sid, frames, q0, states, a0 = load_mmred_sample(sd)
            except Exception:
                continue
            chars = sorted(eval_utils.extract_characters_from_states(states))
            if len(chars) < 2:
                continue
        if args.task == "herbench_ac":
            pass
        elif args.task == "first_occurrence":
            import re
            evid = sorted(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            if not evid:
                continue
            gold = int(evid[0]) + 1                     # 1-based first frame index
            m_ = re.search(r"did (\w+) spend in the (\w+)", q0)
            if not m_:
                continue
            Cn, Rn = m_.group(1), m_.group(2)
            question = (f"In which frame, numbered 1 to {len(frames)}, was {Cn} in the {Rn} "
                        f"for the first time?")
            frame_targets = {t: ("evid" if t in set(evid) else "noev") for t in range(len(frames))}
        elif args.task == "count":
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
            if args.task == "herbench_ac":
                _prompt = (f"You will be shown {len(frames)} frames sampled from an egocentric kitchen video.\n"
                           f"Respond with a single integer from 0 to {len(frames)} (0 is allowed). "
                           f"Output only the integer.\nQuestion: {question}\nAnswer: ")
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt(frames, _prompt))
            else:
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
            ids = inputs["input_ids"][0].detach().cpu()
            fg = image_token_groups(ids, expected_num_frames=len(frames), processor=processor)
            last_img = max(int(p) for grp in fg for p in grp)
            seq = int(ids.shape[0])
            q_span = list(range(last_img + 1, seq))  # all question tokens (span after images)
            carrier = [seq - 1] if args.carrier == "last" else q_span
            if not q_span or len(fg) < NF:
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
            if args.fence_cross_frame:
                MIN = -65504.0  # fp16-representable; safe under cast to bf16/fp16 in the hook
                fm = torch.zeros(seq, seq, dtype=torch.float32)
                fm.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
                allv = sorted({int(p) for grp in fg for p in grp})
                allv_t = torch.tensor(allv, dtype=torch.long)
                for grp in fg:
                    rows = torch.tensor(sorted(int(p) for p in grp), dtype=torch.long)
                    own = set(int(p) for p in grp)
                    forb = torch.tensor([p for p in allv if p not in own], dtype=torch.long)
                    if forb.numel():
                        fm[rows.unsqueeze(1), forb.unsqueeze(0)] = MIN
                fmask = fm.view(1, 1, seq, seq).to(next(model.parameters()).device)
                if n == 0:  # self-check: fence must change the read-layer inputs
                    with torch.no_grad():
                        model(**inputs, use_cache=False)
                    v_ref = qkv[probe_layers[0]]["v_proj"].clone()
                    fence_holder["mask"] = fmask
                    with torch.no_grad():
                        model(**inputs, use_cache=False)
                    dd = (qkv[probe_layers[0]]["v_proj"] - v_ref).abs().max().item()
                    print(f"[fence] self-check: max |Δv@L{probe_layers[0]}| = {dd:.4f}")
                    assert dd > 1e-3, "fence mask had NO effect — kwarg not applied?"
                fence_holder["mask"] = fmask
            with torch.no_grad():
                outp = model(**inputs, use_cache=False)
            fence_holder["mask"] = None
            for h in handles:
                h.remove()
            # proper count prediction: argmax over candidate digit tokens only
            last_logits = outp.logits[0, -1].float().cpu()
            pred = int(cand_vals[int(torch.argmax(last_logits[cand_ids_t]).item())])
            if "cos" not in posemb:
                raise RuntimeError("position_embeddings not captured")
        except Exception as exc:
            print(f"{sid} capture failed: {type(exc).__name__}: {exc}")
            fail_count = globals().get("_fail_count", 0) + 1
            globals()["_fail_count"] = fail_count
            if fail_count >= 25 and n == 0:
                raise RuntimeError(f"{fail_count} consecutive capture failures with 0 successes — aborting")
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

        def msg_per_carrier(pos_list, L):
            # per-frame message into EACH carrier token separately (no cross-token pooling):
            # msg_{f->c} = o_proj(concat_h sum_{j in f} A[c,j] v_j), one row per carrier token.
            pos = torch.tensor([p for p in pos_list if p < seq], dtype=torch.long)
            if pos.numel() == 0:
                return np.zeros((len(carrier), int(cfg.hidden_size)), dtype=np.float32)
            Asel = attnA[L][:, :, pos]                            # [H,|C|,|f|]
            vsel = vrep[L][:, pos, :]                             # [H,|f|,hd]
            ctx = torch.einsum("hcj,hjd->hcd", Asel, vsel)        # [H,|C|,hd] per carrier, no division
            ctx = ctx.permute(1, 0, 2).reshape(len(carrier), -1)  # [|C|, H*hd]
            oproj = layers[L].self_attn.o_proj
            dev = next(oproj.parameters()).device
            with torch.no_grad():
                out = oproj(ctx.to(device=dev, dtype=torch.bfloat16))
            return out.float().cpu().numpy().astype(np.float32)   # [|C|, hidden]

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
        if args.carrier == "per_token":
            off_to_ci = {(len(carrier) - 1) - ci: ci for ci in range(len(carrier))}  # offset-from-end -> carrier idx
            per_dec = {L: {o: np.zeros((NF, int(cfg.hidden_size)), dtype=np.float16) for o in DEC_OFF}
                       for L in probe_layers}
            for t in range(NF):
                posl = [int(p) for p in fg[t]]
                lab = frame_targets.get(t)
                for L in probe_layers:
                    mm = msg_per_carrier(posl, L)          # [|C|, hidden]
                    if lab is not None:                    # SNR needs the evidence label (all frames labeled for count)
                        for off, ci in off_to_ci.items():
                            if off <= MAXOFF:
                                d = pt[L].setdefault(off, {"x": [], "y": []})
                                d["x"].append(mm[ci]); d["y"].append(str(lab))
                    for o in DEC_OFF:                       # decode storage: ALL frames (need every frame to sum)
                        ci = off_to_ci.get(o)
                        if ci is not None:
                            per_dec[L][o][t] = mm[ci].astype(np.float16)
            for off, ci in off_to_ci.items():
                if off <= MAXOFF:
                    txt = tok.decode([int(ids[carrier[ci]])]).strip()
                    pt_tok.setdefault(off, Counter())[txt] += 1
            if DEC_OFF:
                for L in probe_layers:
                    for o in DEC_OFF:
                        dec[L][o].append(per_dec[L][o])
                dec_gold.append(int(gold))
                _pos = {"co_occupancy": "same", "count": "evid", "first_occurrence": "evid", "herbench_ac": "evid"}.get(args.task)
                dec_labels.append(np.array([1 if (str(frame_targets.get(t)) == _pos) else 0 for t in range(NF)],
                                           dtype=np.int64))
                dec_labels_raw.append([str(frame_targets.get(t)) for t in range(NF)])
        cnt_lab.append(int(gold))
        model_correct.append(int(pred == gold))
        n += 1
        if n % 10 == 0:
            print(f"  scanned {n}: {len(pf_lab)} frame-msg, {len(cnt_lab)} count examples")

    lines = [f"=== FRAME->CARRIER MESSAGE PROBES ({args.task}, 7B) n_samples={len(cnt_lab)} "
             f"layers={probe_layers} carriers={args.carrier} ==="]
    if getattr(args, "save_messages", False) and DEC_OFF and dec_gold:
        cache_obj = {"msgs": {L: {o: np.stack(dec[L][o]) for o in DEC_OFF} for L in probe_layers},
                     "gold": np.array(dec_gold, dtype=np.int64),
                     "labels": np.stack(dec_labels),
                     "labels_raw": dec_labels_raw,
                     "model_correct": np.array(model_correct, dtype=np.int64),
                     "layers": probe_layers, "offsets": DEC_OFF, "task": args.task,
                     "data_root": str(args.data_root), "sample_seed": int(args.sample_seed),
                     "carrier": args.carrier, "n_frames": NF}
        torch.save(cache_obj, out / "messages_cache.pt")
        print(f"saved per-frame carrier messages -> {out/'messages_cache.pt'} "
              f"({len(dec_gold)} samples x {NF} frames, layers {probe_layers}, offsets {DEC_OFF})")

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
    pos_lab = {"co_occupancy": "same", "count": "evid", "first_occurrence": "evid", "herbench_ac": "evid"}.get(args.task)
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

    # (D) MESSAGE DECOMPOSITION  m_k = mu + s_k*delta + eps  on the frame->carrier messages
    # (deployed, query-conditioned analog of the frame-token sweep; per-layer d' and d'/sqrt(N)).
    import math
    pos_lab_d = {"co_occupancy": "same", "count": "evid", "first_occurrence": "evid", "herbench_ac": "evid"}.get(args.task)
    if pos_lab_d is not None and binary and len(ym) >= 12:
        lines.append(f"\n(D) MESSAGE DECOMPOSITION  m=mu+s*delta+eps  pos='{pos_lab_d}'  "
                     f"carrier={args.carrier}  N={NF}")
        lines.append(f"  {'layer':>5} {'|mu|':>9} {'|delta|':>8} {'sigma':>9} "
                     f"{'|d|/|mu|':>9} {'nE':>5} {'nN':>5} {'SNR':>7} {'SNR/sqrtN':>10}")
        for L in probe_layers:
            X = np.stack(pf_feat[L]); msk = ym == pos_lab_d
            ev, nv = X[msk], X[~msk]
            if len(ev) < 2 or len(nv) < 2:
                continue
            mu_all = X.mean(0); delta = (ev.mean(0) - nv.mean(0)) / 2.0
            dhat = delta / (np.linalg.norm(delta) + 1e-9)
            sig = 0.5 * ((ev @ dhat).std() + (nv @ dhat).std())
            snr = abs((ev.mean(0) - nv.mean(0)) @ dhat) / (sig + 1e-9)
            ratio = np.linalg.norm(delta) / (np.linalg.norm(mu_all) + 1e-9)
            lines.append(f"  {L:>5} {np.linalg.norm(mu_all):>9.2f} {np.linalg.norm(delta):>8.3f} "
                         f"{sig:>9.3f} {ratio:>9.4f} {len(ev):>5} {len(nv):>5} {snr:>7.3f} "
                         f"{snr/math.sqrt(NF):>10.3f}")
            rows.append(f"msg_decomp,{L},snr,{len(ym)},{snr:.4f},nan,{ratio:.4f},"
                        f"{snr/math.sqrt(NF):.4f},nan,nan")

    # (E) PER-CARRIER-TOKEN SNR SWEEP: per-frame message SNR into each question token (offset from end),
    # per layer -> identifies WHICH question token is the evidence carrier (no cross-token pooling).
    pos_lab_e = {"co_occupancy": "same", "count": "evid", "first_occurrence": "evid", "herbench_ac": "evid"}.get(args.task)

    def _dprime_naive(ev, nv):
        delta = (ev.mean(0) - nv.mean(0)) / 2.0
        dh = delta / (np.linalg.norm(delta) + 1e-9)
        sig = 0.5 * ((ev @ dh).std() + (nv @ dh).std())
        return abs((ev.mean(0) - nv.mean(0)) @ dh) / (sig + 1e-9)

    def _token_snr(x, y):
        """binary tasks: d' of pos-vs-rest; multiclass (rooms): mean one-vs-rest d' over classes."""
        if pos_lab_e is not None:
            ev, nv = x[y == pos_lab_e], x[y != pos_lab_e]
            if len(ev) < 5 or len(nv) < 5:
                return None
            return _dprime_naive(ev, nv)
        vals = []
        for c in sorted(set(y.tolist())):
            ev, nv = x[y == c], x[y != c]
            if len(ev) >= 5 and len(nv) >= 5:
                vals.append(_dprime_naive(ev, nv))
        return float(np.mean(vals)) if vals else None

    if args.carrier == "per_token":
        offs = sorted({o for L in probe_layers for o in pt[L]})
        lines.append(f"\n(E) PER-CARRIER-TOKEN SNR SWEEP  pos='{pos_lab_e or 'mean one-vs-rest (multiclass)'}'"
                     f"  [offset 0 = last/answer token]")
        lines.append("  carrier token by offset-from-end (most common decoded):")
        for o in offs:
            lines.append(f"    off -{o:<2}: {pt_tok.get(o, Counter()).most_common(3)}")
        lines.append("  per-frame message SNR  [rows=layer, cols=offset-from-end]:")
        lines.append("  layer " + " ".join(f"-{o:>5}" for o in offs))
        for L in probe_layers:
            cells = []
            for o in offs:
                d = pt[L].get(o)
                if d is None or len(d["y"]) < 20:
                    cells.append("     .")
                    continue
                snr = _token_snr(np.stack(d["x"]), np.array(d["y"]))
                if snr is None:
                    cells.append("     .")
                    continue
                cells.append(f"{snr:>6.3f}")
                rows.append(f"per_carrier_snr,L{L}_off{o},snr,{len(d['y'])},{snr:.4f},nan,nan,nan,nan,nan")
            lines.append(f"  {L:>5} " + " ".join(cells))

    # (F)/(G) COUNT DECODE AT IDENTIFIED CARRIERS (shared carrier ranking).
    if args.carrier == "per_token" and DEC_OFF and len(dec_gold) >= 40:
        ys = np.array(dec_gold)
        maj = Counter(ys.tolist()).most_common(1)[0][1] / len(ys)

        def peak_snr(o):
            best = 0.0
            for L in probe_layers:
                d = pt[L].get(o)
                if not d or len(d["y"]) < 20:
                    continue
                snr = _token_snr(np.stack(d["x"]), np.array(d["y"]))
                if snr is not None:
                    best = max(best, snr)
            return best

        tok_of = lambda o: (pt_tok.get(o, Counter()).most_common(1) or [("?", 0)])[0][0]
        ranked = sorted(DEC_OFF, key=peak_snr, reverse=True)
        top2 = ranked[:2]

        # (F) DIRECT COUNT DECODE: sum vs concat over frames, per carrier + top-2 combined.
        #   sum=model's aggregation; concat=oracle sees all frames; concat>>sum => aggregation bottleneck.
        lines.append(f"\n(F) DIRECT COUNT DECODE  model={np.mean(model_correct):.3f}  majority={maj:.3f}  n={len(ys)}")
        lines.append("    carriers ranked by peak SNR: "
                     + ", ".join(f"-{o}(snr {peak_snr(o):.2f},'{tok_of(o)}')" for o in ranked))
        spec = {"name": "mlp512x256", "kind": "mlp", "arch": (512, 256), "pca": 256}
        for L in probe_layers:
            feats = {}
            for o in DEC_OFF:
                st = dec[L][o]
                feats[f"off{o}({tok_of(o)})_sum"] = np.stack([s.astype(np.float32).sum(0) for s in st])
                feats[f"off{o}({tok_of(o)})_concat"] = np.stack([s.astype(np.float32).reshape(-1) for s in st])
            if len(top2) >= 2:
                a, b = top2
                combo = [np.concatenate([dec[L][a][i].astype(np.float32), dec[L][b][i].astype(np.float32)], axis=1)
                         for i in range(len(dec_gold))]
                feats["top2(room+char)_sum"] = np.stack([c.sum(0) for c in combo])
                feats["top2(room+char)_concat"] = np.stack([c.reshape(-1) for c in combo])
            for nm, X in feats.items():
                acc, std, base, _ = probe(X, ys, seeds, spec)
                lines.append(f"  L{L} {nm:<26s}: acc={acc:.3f}±{std:.3f} (maj {base:.3f})")
                rows.append(f"count_decode,{nm},L{L},{len(ys)},{acc:.4f},{std:.4f},{base:.4f},"
                            f"{acc-float(np.mean(model_correct)):.4f},nan,nan")

        # (G) DECODE-THEN-COUNT: per-frame evidence clf (well-powered) -> sum predictions -> count.
        #   dtc>>model => aggregation-limited; dtc~=model => extraction/SNR-limited.
        from sklearn.model_selection import GroupShuffleSplit
        csets = [([ranked[0]], tok_of(ranked[0]))]
        if len(ranked) >= 2:
            csets.append((ranked[:2], f"{tok_of(ranked[0])}+{tok_of(ranked[1])}"))
        clf_spec = {"name": "logistic", "kind": "logistic", "pca": 128}
        lines.append(f"\n(G) DECODE-THEN-COUNT (per-frame clf -> sum -> count)   model_acc={np.mean(model_correct):.3f}")
        lines.append(f"  {'layer':>5} {'carriers':>14} {'dtc_acc':>8} {'dtc_MAE':>8} {'model_acc':>10}")
        for L in probe_layers:
            for cset, cname in csets:
                X = []; yf = []; grp = []
                for i in range(len(dec_gold)):
                    for t in range(NF):
                        X.append(np.concatenate([dec[L][o][i][t].astype(np.float32) for o in cset]))
                        yf.append(int(dec_labels[i][t])); grp.append(i)
                X = np.stack(X); yf = np.array(yf); grp = np.array(grp)
                if len(set(yf.tolist())) < 2:
                    lines.append(f"  {L:>5} {cname:>14}   (skipped: single-class binary labels for this task)")
                    continue
                accs = []; maes = []; m_accs = []
                for s in seeds:
                    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=s).split(X, yf, grp))
                    clf = make_clf(clf_spec, len(tr)); clf.fit(X[tr], yf[tr])
                    pf = clf.predict(X[te])
                    per = {}
                    for idx, p in zip(te, pf):
                        per.setdefault(int(grp[idx]), []).append(int(p))
                    sids = sorted(per)
                    pred = {si: sum(per[si]) for si in sids}
                    accs.append(np.mean([pred[si] == int(ys[si]) for si in sids]))
                    maes.append(np.mean([abs(pred[si] - int(ys[si])) for si in sids]))
                    m_accs.append(np.mean([model_correct[si] for si in sids]))
                lines.append(f"  {L:>5} {cname:>14} {np.mean(accs):>8.3f} {np.mean(maes):>8.3f} {np.mean(m_accs):>10.3f}")
                rows.append(f"decode_then_count,{cname},L{L},{len(ys)},{np.mean(accs):.4f},{np.std(accs):.4f},"
                            f"{np.mean(maes):.4f},{np.mean(accs)-np.mean(m_accs):.4f},nan,nan")

    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
