#!/usr/bin/env python3
"""P1 analysis: the QUERY/ENCODING 2×2 (CPU) — reconstruct msg_f under every (q-arm × kv-arm)
combination with ONE consistent rotary geometry (the joint forward's), within-frame softmax
(m_f excluded by construction), o_proj applied from the saved dense matrices; then held-out
shrinkage-LDA d′ per cell (3 sample-disjoint seeds).

HARD GATE (charter): pad×pad must land in [5.3, 7.2] and joint×joint ≈ 2.0 (±0.5) before any
off-diagonal is interpreted.

Registered: joint-q×pad-kv ≪ pad×pad ⇒ QUERY contamination dominant; pad-q×joint-kv ≪ pad×pad
⇒ ENCODING dominant; both intermediate ⇒ split (report fractions of the pad×pad − joint×joint gap).
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
    ap.add_argument("--capture", required=True, help="dir with qkv_capture.pt + oproj_dense.pt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cap = Path(args.capture)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    oproj = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    samples = blob["samples"]; Ls = blob["layers"]
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope_section = tc.rope_scaling["mrope_section"]
    NF = int(blob["config"]["n_frames"])
    print(f"{len(samples)} samples, layers {Ls}, mrope {mrope_section}", flush=True)

    def rot_q(q_flat, cos_c, sin_c):
        q = q_flat.float().view(1, 1, N_HEADS, HD).transpose(1, 2)      # [1,28,1,128]
        qr, _ = apply_multimodal_rotary_pos_emb(q, q, cos_c.float(), sin_c.float(), mrope_section)
        return qr[0, :, 0]                                              # [28,128]

    def rot_k(k_tok, cos_f, sin_f):
        k = k_tok.float().view(1, -1, N_KV, HD).transpose(1, 2)         # [1,4,196,128]
        _, kr = apply_multimodal_rotary_pos_emb(k, k, cos_f.float(), sin_f.float(), mrope_section)
        return kr[0].repeat_interleave(N_HEADS // N_KV, dim=0)          # [28,196,128]

    ARMS = ("pad", "joint", "mp")
    feats = {L: {(qa, ka): [] for qa in ARMS for ka in ARMS} for L in Ls}
    # own-geometry diagonal anchors: each arm read under ITS OWN rope (mp/pad only; joint = grid)
    own = {L: {a: [] for a in ("mp", "pad")} for L in Ls}
    have_own = all("arm_rope" in r for r in samples)
    labels = []
    for si, rec in enumerate(samples):
        cos = rec["rope_cos"]; sin = rec["rope_sin"]                    # [..., 1+NF*196, 128]
        cos_c = cos[..., 0:1, :]; sin_c = sin[..., 0:1, :]
        sizes = rec["joint_fg_sizes"]
        offs = np.cumsum([1] + sizes)[:-1]
        labels.extend(rec["labels"])
        for L in Ls:
            arms = rec["arms"]
            q_rot = {}
            for qa in ARMS:
                if qa == "joint":
                    q_rot[qa] = {t: rot_q(arms["joint"][L]["q"]["all"], cos_c, sin_c)
                                 for t in range(NF)}
                else:
                    q_rot[qa] = {t: rot_q(arms[qa][L]["q"][t], cos_c, sin_c) for t in range(NF)}
            for t in range(NF):
                a, b = int(offs[t]), int(offs[t]) + sizes[t]
                cos_f = cos[..., a:b, :]; sin_f = sin[..., a:b, :]
                k_rot = {ka: rot_k(arms[ka][L]["k"][t], cos_f, sin_f) for ka in ARMS}
                v_rep = {ka: arms[ka][L]["v"][t].float().view(-1, N_KV, HD).transpose(0, 1)
                              .repeat_interleave(N_HEADS // N_KV, dim=0) for ka in ARMS}
                for qa in ARMS:
                    q = q_rot[qa][t]
                    for ka in ARMS:
                        logits = torch.einsum("hd,htd->ht", q, k_rot[ka]) / np.sqrt(HD)
                        w = torch.softmax(logits, dim=-1)
                        ctx = torch.einsum("ht,htd->hd", w, v_rep[ka]).reshape(-1)
                        msg = (oproj[L] @ ctx).numpy().astype(np.float32)
                        feats[L][(qa, ka)].append(msg)
                # own-geometry anchors: arm A's q AND k/v both rotated by arm A's own rope
                if have_own:
                    for a in ("mp", "pad"):
                        rp = rec["arm_rope"][a][t]
                        qa_ = rot_q(arms[a][L]["q"][t], rp["cos_c"], rp["sin_c"])
                        ka_ = rot_k(arms[a][L]["k"][t], rp["cos_f"], rp["sin_f"])
                        va_ = (arms[a][L]["v"][t].float().view(-1, N_KV, HD).transpose(0, 1)
                               .repeat_interleave(N_HEADS // N_KV, dim=0))
                        lg = torch.einsum("hd,htd->ht", qa_, ka_) / np.sqrt(HD)
                        cx = torch.einsum("ht,htd->hd", torch.softmax(lg, -1), va_).reshape(-1)
                        own[L][a].append((oproj[L] @ cx).numpy().astype(np.float32))
        if (si + 1) % 25 == 0:
            print(f"  {si+1}/{len(samples)}", flush=True)

    y = np.array(labels).reshape(len(samples), NF)
    lines = [f"=== QUERY/ENCODING 2x2 (n={len(samples)}, within-frame softmax, joint rope "
             f"geometry, o_proj applied) ==="]
    rows = ["layer,q_arm,kv_arm,dprime_w,dprime_std,dprime_auc"]
    grid = {}
    for L in Ls:
        lines.append(f"--- L{L} (rows=q-arm, cols=kv-arm) ---")
        hdr = "  q\\kv    " + "".join(f"{ka:>12}" for ka in ARMS)
        lines.append(hdr)
        for qa in ARMS:
            cells = []
            for ka in ARMS:
                X = np.stack(feats[L][(qa, ka)]).reshape(len(samples), NF, -1)
                dw, ds, da = dprime_pair(X, y)
                grid[(L, qa, ka)] = dw
                rows.append(f"{L},{qa},{ka},{dw:.4f},{ds:.4f},{da:.4f}")
                cells.append(f"{dw:>7.2f}±{ds:.2f}")
            lines.append(f"  {qa:<8}" + " ".join(f"{c:>11}" for c in cells))
        clean = grid[(L, "pad", "pad")]; joint = grid[(L, "joint", "joint")]
        gate = 5.3 <= clean <= 7.2 and abs(joint - 2.0) <= 0.5
        lines.append(f"  ANCHOR GATE (fixed joint geometry): pad×pad={clean:.2f} (band 5.3–7.2), "
                     f"joint×joint={joint:.2f} (2.0±0.5) -> {'PASS' if gate else 'FAIL'}")
        if have_own:
            om, _s, _a = dprime_pair(np.stack(own[L]["mp"]).reshape(len(samples), NF, -1), y)
            op, _s, _a = dprime_pair(np.stack(own[L]["pad"]).reshape(len(samples), NF, -1), y)
            og = 5.3 <= om <= 8.5
            lines.append(f"  OWN-GEOMETRY ANCHOR: mp×mp(own)={om:.2f} (vs direct 7.18 band 5.3–8.5), "
                         f"pad×pad(own)={op:.2f} (vs direct ~5.3) -> {'PASS' if og else 'FAIL'}")
            rows.append(f"{L},mp_own,mp_own,{om:.4f},nan,nan")
            rows.append(f"{L},pad_own,pad_own,{op:.4f},nan,nan")
        if gate:
            gap = clean - joint
            q_share = (clean - grid[(L, 'joint', 'pad')]) / gap
            e_share = (clean - grid[(L, 'pad', 'joint')]) / gap
            lines.append(f"  gap {gap:.2f}: QUERY-contamination share "
                         f"{q_share:.2f}, ENCODING share {e_share:.2f} "
                         f"(shares of pad×pad − joint×joint; may exceed 1/overlap)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
