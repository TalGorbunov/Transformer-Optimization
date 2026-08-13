#!/usr/bin/env python3
"""Query-generalization test (2026-07-12): does a per-frame clean query read OTHER frames well?

Decides whether a deployable 1-forward query fix exists. Reuses the qkv_2x2 capture
(joint/mp/pad q,k,v per frame). For each reader frame t, read its JOINT-encoded k/v with:
  joint q      : the joint carrier query (baseline, ~1.7)
  own mp q     : frame t's own single-frame query (ceiling for a query fix, ~3.1)
  donor mp q   : a DIFFERENT frame's single-frame query (t+1, and a fixed donor=0)
If donor ≈ own → queries generalize → transfer/probe fix works. If donor ≈ joint → frame-specific.

Joint rotary geometry throughout (relative comparison; same convention as qkv_2x2_analysis).
"""
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
    cap = Path(args.capture); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    oproj = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    samples = blob["samples"]; Ls = blob["layers"]; NF = int(blob["config"]["n_frames"])
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope = tc.rope_scaling["mrope_section"]
    print(f"{len(samples)} samples, layers {Ls}", flush=True)

    def rot_q(qf, cc, sc):
        q = qf.float().view(1, 1, N_HEADS, HD).transpose(1, 2)
        qr, _ = apply_multimodal_rotary_pos_emb(q, q, cc.float(), sc.float(), mrope)
        return qr[0, :, 0]

    def rot_k(kt, cf, sf):
        k = kt.float().view(1, -1, N_KV, HD).transpose(1, 2)
        _, kr = apply_multimodal_rotary_pos_emb(k, k, cf.float(), sf.float(), mrope)
        return kr[0].repeat_interleave(N_HEADS // N_KV, dim=0)

    def msg(q, k, v, L):
        lg = torch.einsum("hd,htd->ht", q, k) / np.sqrt(HD)
        ctx = torch.einsum("ht,htd->hd", torch.softmax(lg, -1), v).reshape(-1)
        return (oproj[L] @ ctx).numpy().astype(np.float32)

    CONDS = ("joint_q", "own_q", "donor_next", "donor0")
    feats = {L: {c: [] for c in CONDS} for L in Ls}
    labels = []
    for si, rec in enumerate(samples):
        cos = rec["rope_cos"]; sin = rec["rope_sin"]
        cc = cos[..., 0:1, :]; sc = sin[..., 0:1, :]
        sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
        labels.extend(rec["labels"])
        for L in Ls:
            A = rec["arms"]
            qj = rot_q(A["joint"][L]["q"]["all"], cc, sc)
            q_own = {t: rot_q(A["mp"][L]["q"][t], cc, sc) for t in range(NF)}
            for t in range(NF):
                a, b = int(offs[t]), int(offs[t]) + sizes[t]
                cf = cos[..., a:b, :]; sf = sin[..., a:b, :]
                kj = rot_k(A["joint"][L]["k"][t], cf, sf)
                vj = (A["joint"][L]["v"][t].float().view(-1, N_KV, HD).transpose(0, 1)
                      .repeat_interleave(N_HEADS // N_KV, dim=0))
                feats[L]["joint_q"].append(msg(qj, kj, vj, L))
                feats[L]["own_q"].append(msg(q_own[t], kj, vj, L))
                feats[L]["donor_next"].append(msg(q_own[(t + 1) % NF], kj, vj, L))
                feats[L]["donor0"].append(msg(q_own[0 if t != 0 else 1], kj, vj, L))
        if (si + 1) % 25 == 0:
            print(f"  {si+1}/{len(samples)}", flush=True)

    y = np.array(labels).reshape(len(samples), NF)
    lines = [f"=== QUERY GENERALIZATION (n={len(samples)}) — does one frame's query read others? ==="]
    rows = ["layer,cond,dprime_w,std,auc"]
    for L in Ls:
        lines.append(f"--- L{L} ---")
        vals = {}
        for c in CONDS:
            X = np.stack(feats[L][c]).reshape(len(samples), NF, -1)
            dw, ds, da = dprime_pair(X, y); vals[c] = dw
            lines.append(f"  {c:<12} d′={dw:.2f}±{ds:.2f} (auc {da:.2f})")
            rows.append(f"{L},{c},{dw:.4f},{ds:.4f},{da:.4f}")
        gen = np.mean([vals["donor_next"], vals["donor0"]])
        span = vals["own_q"] - vals["joint_q"]
        frac = (gen - vals["joint_q"]) / span if abs(span) > 1e-6 else float("nan")
        verdict = ("GENERALIZES → deployable query fix" if frac > 0.5 else
                   "FRAME-SPECIFIC → no single-query fix" if frac < 0.2 else "PARTIAL")
        lines.append(f"  >> donor(mean)={gen:.2f} vs own {vals['own_q']:.2f} vs joint "
                     f"{vals['joint_q']:.2f}: transfer recovers {frac:.0%} of the query gap -> {verdict}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(lines)); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
