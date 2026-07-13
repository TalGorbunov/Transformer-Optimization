#!/usr/bin/env python3
"""P1d closure analysis: frameless-q (deployable clean query) x joint-kv d' vs the grid cells."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from experiments.glstm.dprime_vs_n import dprime_pair
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

N_HEADS, N_KV, HD = 28, 4, 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cap = Path(args.capture)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    fq = torch.load(cap / "frameless_q.pt", map_location="cpu", weights_only=False)
    oproj = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    Ls = blob["layers"]; NF = int(blob["config"]["n_frames"])
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope = tc.rope_scaling["mrope_section"]

    feats = {L: [] for L in Ls}
    labels = []
    used = 0
    for rec in blob["samples"]:
        if rec["sid"] not in fq:
            continue
        cos, sin = rec["rope_cos"], rec["rope_sin"]
        cos_c, sin_c = cos[..., 0:1, :].float(), sin[..., 0:1, :].float()
        sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
        labels.extend(rec["labels"])
        for L in Ls:
            q = fq[rec["sid"]][L].float().view(1, 1, N_HEADS, HD).transpose(1, 2)
            qr, _ = apply_multimodal_rotary_pos_emb(q, q, cos_c, sin_c, mrope)
            qv = qr[0, :, 0]
            for t in range(NF):
                a, b = int(offs[t]), int(offs[t]) + sizes[t]
                k = rec["arms"]["joint"][L]["k"][t].float().view(1, -1, N_KV, HD).transpose(1, 2)
                _, kr = apply_multimodal_rotary_pos_emb(
                    k, k, cos[..., a:b, :].float(), sin[..., a:b, :].float(), mrope)
                kf = kr[0].repeat_interleave(N_HEADS // N_KV, dim=0)
                v = rec["arms"]["joint"][L]["v"][t].float().view(-1, N_KV, HD)\
                    .transpose(0, 1).repeat_interleave(N_HEADS // N_KV, dim=0)
                lg = torch.einsum("hd,htd->ht", qv, kf) / np.sqrt(HD)
                w = torch.softmax(lg, -1)
                ctx = torch.einsum("ht,htd->hd", w, v).reshape(-1)
                feats[L].append((oproj[L] @ ctx).numpy().astype(np.float32))
        used += 1
    y = np.array(labels).reshape(used, NF)
    lines = [f"=== FRAMELESS-Q x JOINT-KV (deployable clean-query read; n={used}) ==="]
    for L in Ls:
        X = np.stack(feats[L]).reshape(used, NF, -1)
        dw, ds, da = dprime_pair(X, y)
        lines.append(f"  L{L}: d'_w {dw:.2f}±{ds:.2f} (auc {da:.2f})   "
                     f"[grid refs: pad-q x joint-kv 3.14, joint x joint 1.71 @L16]")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
