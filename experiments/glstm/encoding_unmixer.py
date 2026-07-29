#!/usr/bin/env python3
"""Encoding un-mixer (2026-07-12): can a LEARNED map clean joint-encoded k/v toward multipass?

The 2×2 showed clean-q × joint-kv caps at ~3.1 vs clean-q × mp-kv ~4.8: the joint VALUES are
degraded. This tests whether that degradation is entangled-but-recoverable (a learned
joint→clean map recovers it) or destroyed (it can't). Trains g_k, g_v (per-token, pre-rotary,
joint→mp) on a disjoint split, applies to held-out frames, reconstructs the message with the
PERFECT (mp) query, and measures d′ vs the joint baseline and the mp ceiling.

No label leakage: the un-mixer is trained on RECONSTRUCTION (joint→mp reps), never on the
evidence label; d′ is measured on the eval split, which the un-mixer never saw.
Data-processing note: d′ can exceed the 3.1 pooled-message baseline because the un-mixer acts
on the 196 per-token reps BEFORE within-frame pooling (more information than the pooled message).
CPU only; reuses the qkv_2x2 capture.
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
from sklearn.linear_model import Ridge

N_HEADS, N_KV, HD = 28, 4, 128


def train_unmixer(Xtr, Ytr, kind, epochs=8):
    """Xtr,Ytr: [n_tok, 512] pre-rotary joint->mp. Returns (callable g, net-or-None)."""
    if kind == "ridge":
        r = Ridge(alpha=100.0).fit(Xtr, Ytr)
        return lambda X: r.predict(X).astype(np.float32), None
    import torch.nn as nn
    dev = "cpu"
    net = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32); Yt = torch.tensor(Ytr, dtype=torch.float32)
    n = len(Xt); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            loss.backward(); opt.step()
        print(f"    {kind} ep{ep} mse={loss.item():.4f}", flush=True)
    net.eval()
    return (lambda X: net(torch.tensor(X, dtype=torch.float32)).detach().numpy().astype(np.float32),
            net)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--kind", default="mlp", choices=["ridge", "mlp", "both"])
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--layers", default=None,
                    help="comma list; restrict to these layers (default: all in the capture)")
    ap.add_argument("--save-dir", default=None,
                    help="save trained MLP g_k/g_v state dicts here (unmixer_L<L>.pt)")
    args = ap.parse_args()
    cap = Path(args.capture); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    oproj = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    S = blob["samples"]; Ls = blob["layers"]; NF = int(blob["config"]["n_frames"])
    if args.layers:
        keep = [int(x) for x in args.layers.split(",")]
        Ls = [L for L in Ls if L in keep]
        assert Ls, f"none of {keep} in capture layers"
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope = tc.rope_scaling["mrope_section"]

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(S)); n_tr = int(args.train_frac * len(S))
    tr, ev = idx[:n_tr], idx[n_tr:]
    print(f"{len(S)} samples -> unmix-train {len(tr)}, eval {len(ev)}; layers {Ls}", flush=True)

    def rot_q(qf, cc, sc):
        q = qf.float().view(1, 1, N_HEADS, HD).transpose(1, 2)
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

    kinds = ["ridge", "mlp"] if args.kind == "both" else [args.kind]
    lines = [f"=== ENCODING UN-MIXER (train {len(tr)}, eval {len(ev)} samples) ==="]
    rows = ["layer,kind,cond,dprime_w,std,auc"]
    for L in Ls:
        # build per-token training pairs (joint -> mp) for k and v
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
        print(f"  L{L}: {len(Xk)} training tokens", flush=True)
        for kind in kinds:
            gk, gk_net = train_unmixer(Xk, Yk, kind)
            gv, gv_net = train_unmixer(Xv, Yv, kind)
            if args.save_dir and kind == "mlp":
                sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
                torch.save({"gk": gk_net.state_dict(), "gv": gv_net.state_dict(),
                            "layer": L, "kind": kind, "train_frac": args.train_frac,
                            "capture": str(cap), "arch": "512-1024-gelu-512"},
                           sd / f"unmixer_L{L}.pt")
                print(f"  saved g_k/g_v -> {sd / f'unmixer_L{L}.pt'}", flush=True)
            feats = {c: [] for c in ("joint_kv", "unmixed_kv", "mp_kv")}
            labels = []
            for si in ev:
                rec = S[si]; cos = rec["rope_cos"]; sin = rec["rope_sin"]
                cc = cos[..., 0:1, :]; sc = sin[..., 0:1, :]
                sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
                A = rec["arms"]; labels.extend(rec["labels"])
                for t in range(NF):
                    a, b = int(offs[t]), int(offs[t]) + sizes[t]
                    cf = cos[..., a:b, :]; sf = sin[..., a:b, :]
                    q = rot_q(A["mp"][L]["q"][t], cc, sc)            # PERFECT query
                    jk = np.asarray(A["joint"][L]["k"][t], np.float32)
                    jv = np.asarray(A["joint"][L]["v"][t], np.float32)
                    feats["joint_kv"].append(msg(q, rot_k(jk, cf, sf), vrep(jv), L))
                    feats["unmixed_kv"].append(msg(q, rot_k(gk(jk), cf, sf), vrep(gv(jv)), L))
                    feats["mp_kv"].append(msg(q, rot_k(A["mp"][L]["k"][t], cf, sf),
                                              vrep(A["mp"][L]["v"][t]), L))
            y = np.array(labels).reshape(len(ev), NF)
            lines.append(f"--- L{L} / {kind} ---")
            for c in ("joint_kv", "unmixed_kv", "mp_kv"):
                X = np.stack(feats[c]).reshape(len(ev), NF, -1)
                dw, ds, da = dprime_pair(X, y)
                lines.append(f"  {c:<12} d′={dw:.2f}±{ds:.2f} (auc {da:.2f})")
                rows.append(f"{L},{kind},{c},{dw:.4f},{ds:.4f},{da:.4f}")
            base = [r for r in rows if f"{L},{kind},joint_kv" in r][-1].split(",")[3]
            unm = [r for r in rows if f"{L},{kind},unmixed_kv" in r][-1].split(",")[3]
            cei = [r for r in rows if f"{L},{kind},mp_kv" in r][-1].split(",")[3]
            b, u, c = float(base), float(unm), float(cei)
            frac = (u - b) / (c - b) if abs(c - b) > 1e-6 else float("nan")
            verdict = ("RECOVERABLE (learned map cleans the encoding)" if frac > 0.5 else
                       "MOSTLY DESTROYED (un-mix recovers little)" if frac < 0.2 else "PARTIAL")
            lines.append(f"  >> un-mixer recovers {frac:.0%} of the joint→clean encoding gap "
                         f"({b:.2f}→{u:.2f}, ceiling {c:.2f}) -> {verdict}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(lines)); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
