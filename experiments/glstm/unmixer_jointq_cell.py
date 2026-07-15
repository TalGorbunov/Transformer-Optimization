#!/usr/bin/env python3
"""The missing repair-2x2 cell (2026-07-13): joint-QUERY x un-mixed-VALUES.

encoding_unmixer.py evaluated the value un-mixer only under the PERFECT (mp) query; the repair
table in the theory artifact therefore has one unmeasured cell. This retrains the same MLP
un-mixer (same seed/split/recipe) on the n=500 capture and evaluates FOUR cells at one layer:
  mp_q_joint_kv / mp_q_unmixed_kv    (cross-checks vs the logged 3.82 / 6.17)
  joint_q_joint_kv                    (cross-check vs the logged 2.09)
  joint_q_unmixed_kv                  (the new cell)
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
from experiments.glstm.encoding_unmixer import train_unmixer, N_HEADS, N_KV, HD
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()
    cap = Path(args.capture); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    oproj = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    S = blob["samples"]; NF = int(blob["config"]["n_frames"]); L = int(args.layer)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope = tc.rope_scaling["mrope_section"]

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(S)); n_tr = int(args.train_frac * len(S))
    tr, ev = idx[:n_tr], idx[n_tr:]
    print(f"{len(S)} samples -> unmix-train {len(tr)}, eval {len(ev)}; L{L}", flush=True)

    def rot_q(qf, cc, sc):
        q = torch.as_tensor(qf).float().view(1, 1, N_HEADS, HD).transpose(1, 2)
        qr, _ = apply_multimodal_rotary_pos_emb(q, q, cc.float(), sc.float(), mrope)
        return qr[0, :, 0]

    def rot_k(kt, cf, sf):
        k = torch.as_tensor(kt).float().view(1, -1, N_KV, HD).transpose(1, 2)
        _, kr = apply_multimodal_rotary_pos_emb(k, k, cf.float(), sf.float(), mrope)
        return kr[0].repeat_interleave(N_HEADS // N_KV, dim=0)

    def vrep(vt):
        return (torch.as_tensor(vt).float().view(-1, N_KV, HD).transpose(0, 1)
                .repeat_interleave(N_HEADS // N_KV, dim=0))

    def msg(q, k, v):
        lg = torch.einsum("hd,htd->ht", q, k) / np.sqrt(HD)
        ctx = torch.einsum("ht,htd->hd", torch.softmax(lg, -1), v).reshape(-1)
        return (oproj[L] @ ctx).numpy().astype(np.float32)

    Xk, Yk, Xv, Yv = [], [], [], []
    for si in tr:
        A = S[si]["arms"]
        for t in range(NF):
            Xk.append(np.asarray(A["joint"][L]["k"][t], dtype=np.float32))
            Yk.append(np.asarray(A["mp"][L]["k"][t], dtype=np.float32))
            Xv.append(np.asarray(A["joint"][L]["v"][t], dtype=np.float32))
            Yv.append(np.asarray(A["mp"][L]["v"][t], dtype=np.float32))
    Xk = np.concatenate(Xk); Yk = np.concatenate(Yk)
    Xv = np.concatenate(Xv); Yv = np.concatenate(Yv)
    print(f"L{L}: {len(Xk)} training tokens", flush=True)
    gk = train_unmixer(Xk, Yk, "mlp"); gv = train_unmixer(Xv, Yv, "mlp")

    cells = ("mp_q_joint_kv", "mp_q_unmixed_kv", "joint_q_joint_kv", "joint_q_unmixed_kv")
    feats = {c: [] for c in cells}
    labels = []
    for si in ev:
        rec = S[si]; cos = rec["rope_cos"]; sin = rec["rope_sin"]
        cc = cos[..., 0:1, :]; sc = sin[..., 0:1, :]
        sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
        A = rec["arms"]; labels.extend(rec["labels"])
        qj = rot_q(A["joint"][L]["q"]["all"], cc, sc)                   # the joint carrier query
        for t in range(NF):
            a, b = int(offs[t]), int(offs[t]) + sizes[t]
            cf = cos[..., a:b, :]; sf = sin[..., a:b, :]
            qm = rot_q(A["mp"][L]["q"][t], cc, sc)                      # perfect query
            jk = np.asarray(A["joint"][L]["k"][t], np.float32)
            jv = np.asarray(A["joint"][L]["v"][t], np.float32)
            uk, uv = gk(jk), gv(jv)
            feats["mp_q_joint_kv"].append(msg(qm, rot_k(jk, cf, sf), vrep(jv)))
            feats["mp_q_unmixed_kv"].append(msg(qm, rot_k(uk, cf, sf), vrep(uv)))
            feats["joint_q_joint_kv"].append(msg(qj, rot_k(jk, cf, sf), vrep(jv)))
            feats["joint_q_unmixed_kv"].append(msg(qj, rot_k(uk, cf, sf), vrep(uv)))
    y = np.array(labels).reshape(len(ev), NF)
    lines = [f"=== JOINT-Q x UNMIXED-KV CELL (train {len(tr)}, eval {len(ev)}, L{L}, mlp) ==="]
    rows = ["layer,cond,dprime_w,std,auc"]
    for c in cells:
        X = np.stack(feats[c]).reshape(len(ev), NF, -1)
        dw, ds, da = dprime_pair(X, y)
        lines.append(f"  {c:<20} d'={dw:.2f}+-{ds:.2f} (auc {da:.2f})")
        rows.append(f"{L},{c},{dw:.4f},{ds:.4f},{da:.4f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(lines)); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
