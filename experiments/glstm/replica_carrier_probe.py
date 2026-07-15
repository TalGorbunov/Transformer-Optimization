#!/usr/bin/env python3
"""REPLICA CARRIERS (2026-07-14): per-frame question replicas with masked attention — a
one-forward architectural fix for the query half of the joint-context tax.

Mechanism: after each frame's image tokens, insert a REPLICA of the question. A custom 4D
attention mask (all layers) makes each replica (a) attend ONLY {prefix text, its own frame,
itself} — so its query is computed from an uncontaminated residual — and (b) INVISIBLE to every
other token — so the original joint computation (frame encoding, final question, model answer)
is bit-for-bit undisturbed except for the replicas' own rows.

Per sample we then read, at L14/L16, the per-frame message into (1) each replica's room token
(the new mechanism) and (2) the final question's room token at off −9 (the in-run JOINT anchor).
Held-out d′ on both. Pre-registered prediction (from the 2×2): replica read ≈ clean-query ×
joint-values ≈ 3.3–4.0 (eval scale) vs joint anchor ≈ 2.0; GO if ≥3.

Steps task, mmred park images, 392px, N frames per sample. Forward-only (no generation).
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from experiments.glstm.dprime_vs_n import dprime_pair
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)

MIN = -65504.0


def find_subseq(hay: List[int], needle: List[int]) -> List[int]:
    out, n = [], len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            out.append(i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--output", default="outputs/ladder/image_longN/replica_carrier")
    ap.add_argument("--no-mask", action="store_true",
                    help="UNMASKED arm: plain interleaved prompt, replicas fully visible/contaminated "
                         "(the prompt-engineering control; per-copy d' traces the contamination ladder)")
    args = ap.parse_args()

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.split(",")]
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads, n_kv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope = cfg.rope_scaling["mrope_section"]
    tok = processor.tokenizer
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    import bitsandbytes.functional as bnbF
    oproj = {}
    for L in Ls:
        w = layers[L].self_attn.o_proj.weight
        oproj[L] = bnbF.dequantize_4bit(w.data, w.quant_state).float()

    # ---- hooks: mask injection (ALL layers) + qkv/pos capture (probe layers) ----
    holder = {"mask": None}

    def mask_pre(_m, hargs, hkwargs):
        mk = holder["mask"]
        if mk is None:
            return hargs, hkwargs
        hs = hargs[0] if hargs else hkwargs.get("hidden_states")
        if hs is not None and mk.dtype != hs.dtype:
            mk = mk.to(hs.dtype); holder["mask"] = mk
        if len(hargs) >= 2:
            return (hargs[0], mk) + tuple(hargs[2:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["attention_mask"] = mk
        return hargs, hkwargs

    qkv: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in Ls}
    posemb = {}
    handles = [ly.register_forward_pre_hook(mask_pre, with_kwargs=True) for ly in layers]

    def mk_qkv(L, nm):
        def hook(_m, _i, o):
            qkv[L][nm] = o.detach()
        return hook
    for L in Ls:
        for nm in ("q_proj", "k_proj", "v_proj"):
            handles.append(getattr(layers[L].self_attn, nm).register_forward_hook(mk_qkv(L, nm)))

    def pe_hook(_m, a_, k_):
        pe = k_.get("position_embeddings")
        if pe is None and len(a_) >= 1 and isinstance(a_[-1], tuple):
            pe = a_[-1]
        if pe is not None:
            posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
    handles.append(layers[Ls[0]].self_attn.register_forward_pre_hook(pe_hook, with_kwargs=True))

    ROOMS = ("Park", "Garden", "Bathroom", "Kitchen", "Office", "Bedroom")
    feats_rep = {L: [] for L in Ls}; feats_anc = {L: [] for L in Ls}
    labels_all, gold_all = [], []
    n_done = n_skip = 0
    dirs = iter_sample_dirs(Path(args.data_root))

    for sd in dirs:
        if n_done >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            n_skip += 1; continue
        evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
        if not evid and gold != 0:
            n_skip += 1; continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        room = next((r for r in ROOMS if r.lower() in q0.lower()), None)
        if room is None:
            n_skip += 1; continue

        content = []
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        content.append({"type": "text", "text": q0})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        ids = inputs["input_ids"][0].tolist(); seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF, processor=processor)
        if len(fg) != NF:
            n_skip += 1; continue

        # locate the N+1 question occurrences (replicas + final)
        spans = None
        for pre in ("", " ", "\n"):
            needle = tok(pre + q0, add_special_tokens=False).input_ids
            occ = find_subseq(ids, needle)
            if len(occ) == NF + 1:
                spans = [(o, o + len(needle)) for o in occ]; break
        if spans is None:
            n_skip += 1; continue
        rep_spans, fin_span = spans[:NF], spans[NF]

        # room token inside each replica: last token in span whose decode contains the room word prefix
        def room_pos(a, b):
            for p in range(b - 1, a - 1, -1):
                if room[:4].lower() in tok.decode([ids[p]]).strip().lower():
                    return p
            return None
        rep_c = [room_pos(a, b) for a, b in rep_spans]
        if any(c is None for c in rep_c):
            n_skip += 1; continue
        anc_c = seq - 1 - 9                                     # off -9 convention

        # ---- 4D mask ----
        m = torch.zeros(seq, seq, dtype=torch.float32)
        m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
        rep_tok = torch.zeros(seq, dtype=torch.bool)
        for a, b in rep_spans:
            rep_tok[a:b] = True
        # replicas invisible to everyone else
        nonrep_rows = (~rep_tok).nonzero(as_tuple=True)[0]
        m[nonrep_rows.unsqueeze(1), rep_tok.nonzero(as_tuple=True)[0].unsqueeze(0)] = MIN
        # replica_i rows: forbid other frames' visual tokens and other replicas
        vis_by_frame = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
        all_vis = torch.cat(vis_by_frame)
        for i, (a, b) in enumerate(rep_spans):
            rows = torch.arange(a, b)
            own_vis = set(vis_by_frame[i].tolist())
            forb_vis = torch.tensor([int(p) for p in all_vis.tolist() if p not in own_vis], dtype=torch.long)
            if forb_vis.numel():
                m[rows.unsqueeze(1), forb_vis.unsqueeze(0)] = MIN
            for j, (a2, b2) in enumerate(rep_spans):
                if j != i:
                    m[rows.unsqueeze(1), torch.arange(a2, b2).unsqueeze(0)] = MIN
            # own span stays causal; prefix/suffix text stays visible (causal)
        if args.no_mask:
            m = torch.zeros(seq, seq, dtype=torch.float32)
            m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
            holder["mask"] = None          # plain causal forward; m used only for recompute rows
        else:
            holder["mask"] = m.view(1, 1, seq, seq).to(model.device)

        with torch.inference_mode():
            model(**inputs)
        holder["mask"] = None

        cos, sin = posemb["cos"], posemb["sin"]
        for L in Ls:
            q = qkv[L]["q_proj"].view(1, seq, n_heads, hd).transpose(1, 2)
            k = qkv[L]["k_proj"].view(1, seq, n_kv, hd).transpose(1, 2)
            v = qkv[L]["v_proj"].view(1, seq, n_kv, hd).transpose(1, 2)
            qr, kr = apply_multimodal_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float(), mrope)
            kr = repeat_kv(kr, n_heads // n_kv)[0]; vv = repeat_kv(v, n_heads // n_kv)[0].float()
            qr = qr[0]
            W = oproj[L].to(qr.device)

            def msg_at(c, allowed_mask_row, frame_idx):
                lg = torch.einsum("hd,htd->ht", qr[:, c], kr) / (hd ** 0.5)
                lg = lg + allowed_mask_row                     # additive mask row (0 / MIN)
                wgt = torch.softmax(lg, -1)
                fidx = vis_by_frame[frame_idx].to(qr.device)
                ctx = torch.einsum("ht,htd->hd", wgt[:, fidx], vv[:, fidx]).reshape(-1)
                return (W @ ctx).float().cpu().numpy()

            mrow = holder  # noqa
            full_mask = m.to(qr.device)
            rep_msgs, anc_msgs = [], []
            for i in range(NF):
                rep_msgs.append(msg_at(rep_c[i], full_mask[rep_c[i]], i))
                anc_msgs.append(msg_at(anc_c, full_mask[anc_c], i))
            feats_rep[L].append(np.stack(rep_msgs)); feats_anc[L].append(np.stack(anc_msgs))

        labels_all.append([1 if t in evid else 0 for t in range(NF)])
        gold_all.append(gold)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done} samples (skip {n_skip})", flush=True)

    for h in handles:
        h.remove()
    y = np.array(labels_all); gold_arr = np.array(gold_all)
    lines = [f"=== REPLICA CARRIERS (n={n_done}, skip={n_skip}, data={args.data_root}) ==="]
    cache = {"labels": y, "gold": gold_arr, "rep": {}, "anc": {}}
    for L in Ls:
        Xr = np.stack(feats_rep[L]); Xa = np.stack(feats_anc[L])
        cache["rep"][L], cache["anc"][L] = Xr, Xa
        dr, sr, ar = dprime_pair(Xr, y)
        da, sa, aa = dprime_pair(Xa, y)
        lines.append(f"L{L}: REPLICA read d'={dr:.2f}±{sr:.2f} (auc {ar:.2f})   "
                     f"JOINT anchor d'={da:.2f}±{sa:.2f} (auc {aa:.2f})   ratio {dr/max(da,1e-9):.2f}x")
        per = []
        for i in range(Xr.shape[1]):
            try:
                di, _, _ = dprime_pair(Xr[:, [i], :], y[:, [i]])
                per.append(f"{di:.2f}")
            except Exception:
                per.append("--")
        lines.append(f"L{L}: per-copy d' (index 0..{Xr.shape[1]-1}): " + " ".join(per))
    torch.save(cache, out / "messages_cache.pt")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
