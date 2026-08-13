#!/usr/bin/env python3
"""Query un-mixer (2026-07-12): can the per-frame ideal query q^(f) be reconstructed from the
SINGLE joint forward's state? The decisive one-forward-fix test.

Value un-mixing already works ([2026-07-12h]). The remaining question: from one joint forward,
can a learned map produce frame f's ideal (mp) query from (the joint carrier query + frame f's
joint content)? If yes, one joint forward + query-adapter + value-adapter = full supply recovery,
no per-frame forwards.

Conditions (read frame f, joint rope geometry, held-out eval):
  joint_q  × joint_kv   deployed baseline (~1.7)
  unmix_q  × joint_kv   query fix only  -> ceiling = mp_q × joint_kv (~3.1)
  unmix_q  × unmix_kv   BOTH adapters (the deployable full fix) -> ceiling = mp_q × mp_kv (~4.4)
Query un-mixer input = [joint carrier q, mean-pooled frame-f joint k, mean-pooled frame-f joint v]
-> mp query for frame f (pre-rotary; ridge, data-limited at ~720 examples — recovery is a lower bd).
No label leak (trained on query reconstruction). CPU; reuses the qkv_2x2 capture.
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
from experiments.glstm.encoding_unmixer import train_unmixer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from sklearn.linear_model import Ridge

N_HEADS, N_KV, HD = 28, 4, 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()
    cap = Path(args.capture); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    oproj = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    S = blob["samples"]; Ls = blob["layers"]; NF = int(blob["config"]["n_frames"])
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope = tc.rope_scaling["mrope_section"]
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(S)); n_tr = int(args.train_frac * len(S))
    tr, ev = idx[:n_tr], idx[n_tr:]
    print(f"{len(S)} samples -> train {len(tr)}, eval {len(ev)}", flush=True)

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

    def msg(q, k, v, L):
        lg = torch.einsum("hd,htd->ht", q, k) / np.sqrt(HD)
        ctx = torch.einsum("ht,htd->hd", torch.softmax(lg, -1), v).reshape(-1)
        return (oproj[L] @ ctx).numpy().astype(np.float32)

    def qfeat(rec, L, t):
        A = rec["arms"]
        return np.concatenate([np.asarray(A["joint"][L]["q"]["all"], np.float32),
                               np.asarray(A["joint"][L]["k"][t], np.float32).mean(0),
                               np.asarray(A["joint"][L]["v"][t], np.float32).mean(0)])

    lines = [f"=== QUERY UN-MIXER (train {len(tr)}, eval {len(ev)} samples) ==="]
    rows = ["layer,cond,dprime_w,std,auc"]
    for L in Ls:
        # value un-mixers (lots of data, MLP) for the combined condition
        Xk, Yk, Xv, Yv = [], [], [], []
        for si in tr:
            A = S[si]["arms"]
            for t in range(NF):
                Xk.append(np.asarray(A["joint"][L]["k"][t], np.float32)); Yk.append(np.asarray(A["mp"][L]["k"][t], np.float32))
                Xv.append(np.asarray(A["joint"][L]["v"][t], np.float32)); Yv.append(np.asarray(A["mp"][L]["v"][t], np.float32))
        gk = train_unmixer(np.concatenate(Xk), np.concatenate(Yk), "mlp")
        gv = train_unmixer(np.concatenate(Xv), np.concatenate(Yv), "mlp")
        # query un-mixer (data-limited, ridge on the residual q_mp - q_joint)
        Xq, Yq = [], []
        for si in tr:
            for t in range(NF):
                Xq.append(qfeat(S[si], L, t))
                Yq.append(np.asarray(S[si]["arms"]["mp"][L]["q"][t], np.float32))
        Xq = np.stack(Xq); Yq = np.stack(Yq)
        rq = Ridge(alpha=500.0).fit(Xq, Yq)
        print(f"  L{L}: query un-mixer trained on {len(Xq)} examples "
              f"(train R2={rq.score(Xq, Yq):.3f})", flush=True)

        feats = {c: [] for c in ("joint_q_joint_kv", "unmix_q_joint_kv", "mp_q_joint_kv",
                                  "unmix_q_unmix_kv", "mp_q_mp_kv")}
        labels = []
        for si in ev:
            rec = S[si]; cos = rec["rope_cos"]; sin = rec["rope_sin"]
            cc = cos[..., 0:1, :]; sc = sin[..., 0:1, :]
            sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
            A = rec["arms"]; labels.extend(rec["labels"])
            qj = A["joint"][L]["q"]["all"]
            for t in range(NF):
                a, b = int(offs[t]), int(offs[t]) + sizes[t]
                cf = cos[..., a:b, :]; sf = sin[..., a:b, :]
                q_j = rot_q(qj, cc, sc)
                q_u = rot_q(rq.predict(qfeat(rec, L, t)[None])[0], cc, sc)
                q_m = rot_q(A["mp"][L]["q"][t], cc, sc)
                jk = np.asarray(A["joint"][L]["k"][t], np.float32); jv = np.asarray(A["joint"][L]["v"][t], np.float32)
                kj, vj = rot_k(jk, cf, sf), vrep(jv)
                ku, vu = rot_k(gk(jk), cf, sf), vrep(gv(jv))
                km, vm = rot_k(A["mp"][L]["k"][t], cf, sf), vrep(A["mp"][L]["v"][t])
                feats["joint_q_joint_kv"].append(msg(q_j, kj, vj, L))
                feats["unmix_q_joint_kv"].append(msg(q_u, kj, vj, L))
                feats["mp_q_joint_kv"].append(msg(q_m, kj, vj, L))
                feats["unmix_q_unmix_kv"].append(msg(q_u, ku, vu, L))
                feats["mp_q_mp_kv"].append(msg(q_m, km, vm, L))
        y = np.array(labels).reshape(len(ev), NF)
        d = {}
        lines.append(f"--- L{L} ---")
        for c in feats:
            X = np.stack(feats[c]).reshape(len(ev), NF, -1)
            dw, ds, da = dprime_pair(X, y); d[c] = dw
            lines.append(f"  {c:<20} d′={dw:.2f}±{ds:.2f} (auc {da:.2f})")
            rows.append(f"{L},{c},{dw:.4f},{ds:.4f},{da:.4f}")
        qfrac = (d["unmix_q_joint_kv"] - d["joint_q_joint_kv"]) / \
                max(d["mp_q_joint_kv"] - d["joint_q_joint_kv"], 1e-6)
        fullfrac = (d["unmix_q_unmix_kv"] - d["joint_q_joint_kv"]) / \
                   max(d["mp_q_mp_kv"] - d["joint_q_joint_kv"], 1e-6)
        vq = ("RECONSTRUCTABLE → one-forward query fix" if qfrac > 0.5 else
              "NOT reconstructable → query needs per-frame forwards" if qfrac < 0.2 else "PARTIAL")
        lines.append(f"  >> QUERY un-mix recovers {qfrac:.0%} of the query gap "
                     f"({d['joint_q_joint_kv']:.2f}→{d['unmix_q_joint_kv']:.2f}, ceil {d['mp_q_joint_kv']:.2f}) -> {vq}")
        lines.append(f"  >> BOTH adapters recover {fullfrac:.0%} of the full joint→clean gap "
                     f"({d['joint_q_joint_kv']:.2f}→{d['unmix_q_unmix_kv']:.2f}, ceil {d['mp_q_mp_kv']:.2f})")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(lines)); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
