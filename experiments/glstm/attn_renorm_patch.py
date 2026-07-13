#!/usr/bin/env python3
"""P1f: WITHIN-FRAME ATTENTION RENORMALIZATION during a JOINT pass — the mass-competition test.

Hypothesis (registered, RESULTS [2026-07-11d] §2): the joint-context tax (carrier d′ ~2 joint vs
7–8 multipass, N-invariant) is caused by frames COMPETING for the carrier's softmax mass. This
script runs a normal joint forward but, in layers --patch-layers (default 14–17), recomputes the
QUESTION-TOKEN rows' attention with per-frame renormalization — every frame block receives an
equal share M/N of the row's total visual mass M (within-frame relative weights and all text
columns untouched) — and overwrites those rows' outputs so downstream layers see the patched
stream. Carrier messages at --layers are computed with the SAME renormalized weights.

  patched d′ >> joint  →  mass competition is the tax (recovery toward multipass)
  patched ≈ joint      →  the tax is upstream in-context frame ENCODING; hypothesis refuted
                          at the carrier hop.

Output: messages_cache.pt (schema-compatible; arm='renorm') + report. Joint/multipass anchors
for the same data/seed already exist (B1 caches).
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--layers", default="14,16", help="read layers (messages)")
    ap.add_argument("--patch-layers", default="14,15,16,17")
    ap.add_argument("--decode-offsets", default="9,13")
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    READ = [int(x) for x in args.layers.replace(",", " ").split()]
    PATCH = [int(x) for x in args.patch_layers.replace(",", " ").split()]
    DEC_OFF = [int(x) for x in args.decode_offsets.replace(",", " ").split()]
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope_section = (getattr(cfg, "rope_scaling", None) or {}).get("mrope_section", None)
    attn_scale = head_dim ** -0.5
    NF = int(args.n_frames)
    hidden = int(cfg.hidden_size)
    out = Path(args.output) / "count"; out.mkdir(parents=True, exist_ok=True)
    tok = processor.tokenizer
    cand_ids, cand_vals = [], []
    for d in range(0, 9):
        e = tok.encode(str(d), add_special_tokens=False)
        if len(e) == 1:
            cand_ids.append(int(e[0])); cand_vals.append(d)
    cand_ids_t = torch.tensor(cand_ids, dtype=torch.long)

    # ---- shared per-sample state ----
    ps = {"fg": None, "qspan": None, "posemb": None, "selfcheck": None}
    qkv_read: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in READ}

    def renorm_rows(A, fg_t):
        """A [H, |Q|, S] float; renormalize each frame block to an equal share of the row's
        total visual mass. Returns A' (text columns untouched)."""
        masses = []
        for pos in fg_t:
            masses.append(A[:, :, pos].sum(-1, keepdim=True))       # [H,|Q|,1]
        M = torch.stack(masses, 0).sum(0)                            # [H,|Q|,1]
        Ap = A.clone()
        share = M / len(fg_t)
        for pos, m_f in zip(fg_t, masses):
            scale = share / m_f.clamp_min(1e-9)
            Ap[:, :, pos] = A[:, :, pos] * scale
        return Ap

    def make_patched_forward(L, attn):
        orig = attn.forward

        def patched(hidden_states, *a, **kw):
            outp = orig(hidden_states, *a, **kw)
            if ps["fg"] is None:
                return outp
            pe = kw.get("position_embeddings")
            if pe is None:
                return outp
            hs = hidden_states[0]                                    # [S, hid]
            S = hs.shape[0]
            q = attn.q_proj(hidden_states).view(1, S, n_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(1, S, n_kv, head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(1, S, n_kv, head_dim).transpose(1, 2)
            cos, sin = pe
            if mrope_section is not None:
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
            k = repeat_kv(k, n_heads // n_kv); v = repeat_kv(v, n_heads // n_kv)
            Q = ps["qspan_t"]
            qf = q[0, :, Q].float()                                  # [H,|Q|,hd]
            kf = k[0].float()                                        # [H,S,hd]
            scores = torch.einsum("hqd,hkd->hqk", qf, kf) * attn_scale
            allow = torch.arange(S, device=scores.device)[None, :] <= Q[:, None]
            scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
            A = torch.softmax(scores, dim=-1)
            Ap = renorm_rows(A, ps["fg_t"])
            ctx = torch.einsum("hqk,hkd->hqd", Ap, v[0].float())     # [H,|Q|,hd]
            ctx = ctx.permute(1, 0, 2).reshape(len(Q), -1)
            rows = attn.o_proj(ctx.to(dtype=hidden_states.dtype))    # [|Q|, hid]
            if ps["selfcheck"] is None:
                d0 = (outp[0][0, Q] - rows).abs().max().item()
                ps["selfcheck"] = d0
            outp[0][0, Q] = rows
            return outp

        return patched

    originals = {}
    for L in PATCH:
        attn = layers[L].self_attn
        originals[L] = attn.forward
        attn.forward = make_patched_forward(L, attn)
    for L in READ:
        for nm in ("q_proj", "k_proj", "v_proj"):
            def mk(L, nm):
                def hook(_m, _i, o):
                    qkv_read[L][nm] = o.detach()[0]
                return hook
            getattr(layers[L].self_attn, nm).register_forward_hook(mk(L, nm))
    def mk_pe(_m, args_, kwargs_):
        pe = kwargs_.get("position_embeddings")
        if pe is None and len(args_) >= 1 and isinstance(args_[-1], tuple):
            pe = args_[-1]
        if pe is not None:
            ps["posemb"] = (pe[0].detach(), pe[1].detach())
    layers[READ[0]].self_attn.register_forward_pre_hook(mk_pe, with_kwargs=True)

    dec = {L: {o: [] for o in DEC_OFF} for L in READ}
    dec_mass = {L: {o: [] for o in DEC_OFF} for L in READ}
    dec_gold: List[int] = []
    dec_labels: List[np.ndarray] = []
    model_correct: List[int] = []
    n = 0
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            continue
        if len(frames) < NF:
            continue
        evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
        if not evid:
            continue
        try:
            if int(args.resize) > 0:
                frames = [f.resize((int(args.resize), int(args.resize))) for f in frames]
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
            ids = inputs["input_ids"][0].detach().cpu()
            fg = image_token_groups(ids, expected_num_frames=NF, processor=processor)
            seq = int(ids.shape[0])
            last_img = max(int(p) for g in fg for p in g)
            qspan = list(range(last_img + 1, seq))
            dev = next(model.parameters()).device
            ps["fg"] = fg
            ps["fg_t"] = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long, device=dev)
                          for g in fg]
            ps["qspan_t"] = torch.tensor(qspan, dtype=torch.long, device=dev)
            with torch.no_grad():
                outp = model(**inputs, use_cache=False)
            ps["fg"] = None
            if n == 0:
                print(f"[renorm] self-check max|Δ question-row output| = {ps['selfcheck']:.4f} "
                      f"(must be > 0)", flush=True)
                assert ps["selfcheck"] and ps["selfcheck"] > 1e-3, "patch had no effect"
            # messages at read layers, WITH the renormalized weights
            carrier_t = torch.tensor(qspan, dtype=torch.long)
            off_to_ci = {(len(qspan) - 1) - ci: ci for ci in range(len(qspan))}
            cosn, sinn = ps["posemb"]
            per_dec = {L: {o: np.zeros((NF, hidden), dtype=np.float16) for o in DEC_OFF}
                       for L in READ}
            per_mass = {L: {o: np.zeros(NF, dtype=np.float32) for o in DEC_OFF} for L in READ}
            for L in READ:
                q = qkv_read[L]["q_proj"].view(1, seq, n_heads, head_dim).transpose(1, 2)
                k = qkv_read[L]["k_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
                v = qkv_read[L]["v_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
                if mrope_section is not None:
                    q, k = apply_multimodal_rotary_pos_emb(q, k, cosn, sinn, mrope_section)
                k = repeat_kv(k, n_heads // n_kv); v = repeat_kv(v, n_heads // n_kv)
                qf = q[0].float().cpu(); kf = k[0].float().cpu()
                scores = torch.einsum("hcd,hkd->hck", qf[:, carrier_t], kf) * attn_scale
                allow = torch.arange(seq)[None, :] <= carrier_t[:, None]
                scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
                A = torch.softmax(scores, dim=-1)
                fg_cpu = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
                Ap = renorm_rows(A, fg_cpu)
                vf = v[0].float().cpu()
                oproj = layers[L].self_attn.o_proj
                odev = next(oproj.parameters()).device
                for t, pos in enumerate(fg_cpu):
                    Asel = Ap[:, :, pos]; vsel = vf[:, pos, :]
                    mass = Asel.sum(-1).mean(0).numpy()
                    ctx = torch.einsum("hcj,hjd->hcd", Asel, vsel)
                    ctx = ctx.permute(1, 0, 2).reshape(len(qspan), -1)
                    with torch.no_grad():
                        mm = oproj(ctx.to(device=odev, dtype=torch.bfloat16)).float().cpu().numpy()
                    for o in DEC_OFF:
                        ci = off_to_ci.get(o)
                        if ci is not None:
                            per_dec[L][o][t] = mm[ci].astype(np.float16)
                            per_mass[L][o][t] = mass[ci]
            last_logits = outp.logits[0, -1].float().cpu()
            pred = int(cand_vals[int(torch.argmax(last_logits[cand_ids_t]).item())])
        except Exception as exc:
            ps["fg"] = None
            print(f"{sid} failed: {type(exc).__name__}: {exc}", flush=True)
            fail = globals().get("_f", 0) + 1; globals()["_f"] = fail
            if fail >= 10 and n == 0:
                raise
            continue
        for L in READ:
            for o in DEC_OFF:
                dec[L][o].append(per_dec[L][o])
                dec_mass[L][o].append(per_mass[L][o])
        dec_gold.append(gold)
        dec_labels.append(np.array([1 if t in evid else 0 for t in range(NF)], dtype=np.int64))
        model_correct.append(int(pred == gold))
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{args.limit}  model acc {np.mean(model_correct):.3f}", flush=True)

    for L in PATCH:
        layers[L].self_attn.forward = originals[L]
    cache_obj = {"msgs": {L: {o: np.stack(dec[L][o]) for o in DEC_OFF} for L in READ},
                 "mass": {L: {o: np.stack(dec_mass[L][o]) for o in DEC_OFF} for L in READ},
                 "gold": np.array(dec_gold, dtype=np.int64),
                 "labels": np.stack(dec_labels),
                 "labels_raw": [["evid" if x else "noev" for x in row] for row in dec_labels],
                 "model_correct": np.array(model_correct, dtype=np.int64),
                 "layers": READ, "offsets": DEC_OFF, "task": "count",
                 "data_root": str(args.data_root), "sample_seed": int(args.sample_seed),
                 "carrier": "per_token_renorm", "n_frames": NF,
                 "patch_layers": PATCH, "mode": "attn_renorm", "resize": int(args.resize)}
    torch.save(cache_obj, out / "messages_cache.pt")
    rep = (f"=== ATTN RENORM PATCH (joint pass, per-frame equal mass @L{PATCH}) ===\n"
           f"n={len(dec_gold)} NF={NF} read layers={READ} offsets={DEC_OFF}\n"
           f"self-check max|Δ question rows| (sample 0): {ps['selfcheck']:.4f}\n"
           f"model own-answer acc under patch: {float(np.mean(model_correct)):.3f}\n"
           f"cache -> {out/'messages_cache.pt'}\n")
    (out / "report.txt").write_text(rep)
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
