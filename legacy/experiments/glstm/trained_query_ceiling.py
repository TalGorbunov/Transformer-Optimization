#!/usr/bin/env python3
"""Trained-query ceiling (Exp 2, 2026-07-13): what d' can a TRAINED shared query reach on
JOINT-encoded k/v?

The 2x2 says the joint carrier query reads d'~2.1-2.3 from joint k/v while a perfect
frame-specific (mp) query reads 4.47 (n=500 scale). This trains a single query q* in R^3584 --
shared across frames, i.e. exactly what a DETR/slot-style head could deploy without frame
identity at inference -- by differentiating through the SAME message path as the 2x2 analysis
(joint rope geometry, within-frame softmax, o_proj) with a logistic proxy loss
sigma(w . msg(q*, k_f, v_f) + b) on the per-frame evidence label. Train samples only
(RandomState(0) 60/40, the un-mixer split); d' via dprime_pair on the held-out eval split.

Note on the brief's arm (b) "q* per head": q* in R^3584 already parameterizes all 28 heads
independently (28 x 128); the binding constraint is shared-across-frames, so arm (b) == arm (a).
The init ablation (joint-mean / mp-mean / random) covers the local-minima question instead.

Anchors recomputed in-run (must reproduce or the run is invalid):
  eval-split scale (n=200): joint-q x joint-kv ~2.09, mp-q x joint-kv ~3.82, mp x mp ~6.33
  full-500 scale:           joint-q x joint-kv ~2.33, mp-q x joint-kv ~4.47, pad-q x joint-kv ~3.97
"""
from __future__ import annotations
import argparse, json, sys, time
from math import comb
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from experiments.glstm.dprime_vs_n import dprime_pair
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

N_HEADS, N_KV, HD = 28, 4, 128
GS = N_HEADS // N_KV
ND = NormalDist()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def merge_mrope(cs, mrope):
    """cs [3,1,T,128] -> merged [T,128] (the section-interleave HF applies inside rope)."""
    sec = list(mrope) * 2
    r = torch.cat([m[i % 3] for i, m in enumerate(cs.float().split(sec, dim=-1))], dim=-1)
    return r.reshape(-1, HD)


def gate_law_exact(dp, N, E):
    """P(miss-count == false-alarm-count), miss~Bin(E,p), fa~Bin(N-E,p), p=1-Phi(d'/2)."""
    p = 1.0 - ND.cdf(dp / 2.0)
    return sum(comb(E, j) * p ** j * (1 - p) ** (E - j)
               * comb(N - E, j) * p ** j * (1 - p) ** (N - E - j)
               for j in range(0, min(E, N - E) + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--inits", default="joint,mp,random")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    cap = Path(args.capture); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    blob = torch.load(cap / "qkv_capture.pt", map_location="cpu", weights_only=False)
    oproj_all = torch.load(cap / "oproj_dense.pt", map_location="cpu", weights_only=False)
    S = blob["samples"]; NF = int(blob["config"]["n_frames"]); L = int(args.layer)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(blob["config"]["model_name"])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    mrope = tc.rope_scaling["mrope_section"]
    oproj = oproj_all[L].float().to(dev)
    print(f"loaded blob in {time.time()-t0:.0f}s: {len(S)} samples, L{L}, mrope {mrope}", flush=True)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(S)); n_tr = int(args.train_frac * len(S))
    tr, ev = idx[:n_tr], idx[n_tr:]
    print(f"split: train {len(tr)}, eval {len(ev)} (RandomState(0), {args.train_frac})", flush=True)

    # ---- precompute per-sample merged rope + per-frame rotated K / repeated V on device ----
    # order: all samples in original index order; frame f of sample s at row s*NF+f
    n = len(S)
    cc_m = torch.zeros(n, 1, HD)
    sc_m = torch.zeros(n, 1, HD)
    sizes0 = None
    Kr = None; Vr = None
    Qj = torch.zeros(n, N_HEADS, HD)                       # joint carrier query, pre-rotary
    Qmp = torch.zeros(n, NF, N_HEADS, HD)                  # mp per-frame queries
    Qpad = torch.zeros(n, NF, N_HEADS, HD)                 # pad per-frame queries
    y_all = np.zeros((n, NF), dtype=np.int64)
    for si, rec in enumerate(S):
        cos = rec["rope_cos"]; sin = rec["rope_sin"]
        cc_m[si] = merge_mrope(cos[..., 0:1, :], mrope)
        sc_m[si] = merge_mrope(sin[..., 0:1, :], mrope)
        sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
        if sizes0 is None:
            sizes0 = list(sizes)
            T = sizes0[0]
            assert all(s == T for s in sizes0), f"variable frame sizes {sizes0}"
            Kr = torch.zeros(n, NF, N_HEADS, T, HD, dtype=torch.float16, device=dev)
            Vr = torch.zeros(n, NF, N_HEADS, T, HD, dtype=torch.float16, device=dev)
        assert list(rec["joint_fg_sizes"]) == sizes0, "frame sizes differ across samples"
        A = rec["arms"]
        Qj[si] = torch.as_tensor(np.asarray(A["joint"][L]["q"]["all"], np.float32)).view(N_HEADS, HD)
        y_all[si] = np.asarray(rec["labels"], dtype=np.int64)
        for t in range(NF):
            a, b = int(offs[t]), int(offs[t]) + sizes[t]
            cf = merge_mrope(cos[..., a:b, :], mrope)      # [T,128]
            sf = merge_mrope(sin[..., a:b, :], mrope)
            k = torch.as_tensor(np.asarray(A["joint"][L]["k"][t], np.float32)).view(-1, N_KV, HD).transpose(0, 1)
            kr = k * cf.unsqueeze(0) + rotate_half(k) * sf.unsqueeze(0)
            Kr[si, t] = kr.repeat_interleave(GS, dim=0).half().to(dev)
            v = torch.as_tensor(np.asarray(A["joint"][L]["v"][t], np.float32)).view(-1, N_KV, HD).transpose(0, 1)
            Vr[si, t] = v.repeat_interleave(GS, dim=0).half().to(dev)
            Qmp[si, t] = torch.as_tensor(np.asarray(A["mp"][L]["q"][t], np.float32)).view(N_HEADS, HD)
            Qpad[si, t] = torch.as_tensor(np.asarray(A["pad"][L]["q"][t], np.float32)).view(N_HEADS, HD)
        if (si + 1) % 100 == 0:
            print(f"  precompute {si+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    cc_m = cc_m.to(dev); sc_m = sc_m.to(dev)
    Qj = Qj.to(dev); Qmp = Qmp.to(dev); Qpad = Qpad.to(dev)

    def rot_q_batch(qh, sids):
        """qh [B?,28,128] or [28,128]; rotate by each sample's carrier rope -> [B,28,128]."""
        c = cc_m[sids]; s = sc_m[sids]                      # [B,1,128]
        return qh * c + rotate_half(qh) * s

    def msg_batch(q_rot, sids, fids):
        """q_rot [B,28,128] rotated; -> messages [B,3584] fp32 (grad flows through q_rot)."""
        K = Kr[sids, fids].float(); V = Vr[sids, fids].float()   # [B,28,T,128]
        lg = torch.einsum("bhd,bhtd->bht", q_rot, K) / np.sqrt(HD)
        ctx = torch.einsum("bht,bhtd->bhd", torch.softmax(lg, -1), V).reshape(len(sids), -1)
        return ctx @ oproj.T

    # ---- cross-check the vectorized path against the reference (2x2 analysis) path ----
    def msg_ref(q_flat, si, t):
        rec = S[si]; cos = rec["rope_cos"]; sin = rec["rope_sin"]
        sizes = rec["joint_fg_sizes"]; offs = np.cumsum([1] + sizes)[:-1]
        a, b = int(offs[t]), int(offs[t]) + sizes[t]
        q = torch.as_tensor(q_flat).float().view(1, 1, N_HEADS, HD).transpose(1, 2)
        qr, _ = apply_multimodal_rotary_pos_emb(q, q, cos[..., 0:1, :].float(), sin[..., 0:1, :].float(), mrope)
        qr = qr[0, :, 0]
        k = torch.as_tensor(np.asarray(rec["arms"]["joint"][L]["k"][t], np.float32)).view(1, -1, N_KV, HD).transpose(1, 2)
        _, kr = apply_multimodal_rotary_pos_emb(k, k, cos[..., a:b, :].float(), sin[..., a:b, :].float(), mrope)
        kr = kr[0].repeat_interleave(GS, dim=0)
        v = (torch.as_tensor(np.asarray(rec["arms"]["joint"][L]["v"][t], np.float32))
             .view(-1, N_KV, HD).transpose(0, 1).repeat_interleave(GS, dim=0))
        lg = torch.einsum("hd,htd->ht", qr, kr) / np.sqrt(HD)
        ctx = torch.einsum("ht,htd->hd", torch.softmax(lg, -1), v).reshape(-1)
        return (oproj_all[L].float() @ ctx).numpy()

    chk_s = int(ev[0])
    sids = torch.full((NF,), chk_s, dtype=torch.long)
    fids = torch.arange(NF)
    with torch.no_grad():
        qr = rot_q_batch(Qj[chk_s].unsqueeze(0).expand(NF, -1, -1), sids.to(dev))
        m_gpu = msg_batch(qr, sids.to(dev), fids.to(dev)).cpu().numpy()
    m_ref = np.stack([msg_ref(S[chk_s]["arms"]["joint"][L]["q"]["all"], chk_s, t) for t in range(NF)])
    rel = np.abs(m_gpu - m_ref).max() / (np.abs(m_ref).max() + 1e-9)
    print(f"cross-check vectorized-vs-reference msg: max rel diff {rel:.2e}", flush=True)
    assert rel < 5e-3, "vectorized message path deviates from reference"

    lines = [f"=== TRAINED-QUERY CEILING (L{L}, train {len(tr)}, eval {len(ev)}) ==="]
    rows = ["cond,split,dprime_w,std,auc"]

    def eval_dprime(name, build_q, split_idx, split_name):
        """build_q(sids [B], fids [B]) -> rotated q [B,28,128]; returns d' on split."""
        sid = torch.as_tensor(np.repeat(split_idx, NF), dtype=torch.long, device=dev)
        fid = torch.as_tensor(np.tile(np.arange(NF), len(split_idx)), dtype=torch.long, device=dev)
        msgs = []
        with torch.no_grad():
            for i in range(0, len(sid), 1024):
                q = build_q(sid[i:i + 1024], fid[i:i + 1024])
                msgs.append(msg_batch(q, sid[i:i + 1024], fid[i:i + 1024]).cpu().numpy())
        X = np.concatenate(msgs).reshape(len(split_idx), NF, -1).astype(np.float32)
        dw, ds, da = dprime_pair(X, y_all[split_idx])
        lines.append(f"  {name:<26} [{split_name}] d'={dw:.2f}+-{ds:.2f} (auc {da:.2f})")
        rows.append(f"{name},{split_name},{dw:.4f},{ds:.4f},{da:.4f}")
        print(lines[-1], flush=True)
        return dw

    # ---- anchors (both scales) ----
    all_idx = np.arange(n)
    anchors = {}
    for split_name, split_idx in (("eval", ev), ("full500", all_idx)):
        anchors[("joint_q_joint_kv", split_name)] = eval_dprime(
            "joint_q_joint_kv", lambda s, f: rot_q_batch(Qj[s], s), split_idx, split_name)
        anchors[("mp_q_joint_kv", split_name)] = eval_dprime(
            "mp_q_joint_kv", lambda s, f: rot_q_batch(Qmp[s, f], s), split_idx, split_name)
        anchors[("pad_q_joint_kv", split_name)] = eval_dprime(
            "pad_q_joint_kv", lambda s, f: rot_q_batch(Qpad[s, f], s), split_idx, split_name)
    gate_ok = (1.8 <= anchors[("joint_q_joint_kv", "eval")] <= 2.5
               and 3.4 <= anchors[("mp_q_joint_kv", "eval")] <= 4.2
               and 2.0 <= anchors[("joint_q_joint_kv", "full500")] <= 2.6
               and 4.1 <= anchors[("mp_q_joint_kv", "full500")] <= 4.8)
    lines.append(f"  ANCHOR GATE: joint-q eval/full {anchors[('joint_q_joint_kv','eval')]:.2f}/"
                 f"{anchors[('joint_q_joint_kv','full500')]:.2f} (want ~2.09/~2.33), "
                 f"mp-q eval/full {anchors[('mp_q_joint_kv','eval')]:.2f}/"
                 f"{anchors[('mp_q_joint_kv','full500')]:.2f} (want ~3.82/~4.47) -> "
                 f"{'PASS' if gate_ok else 'FAIL'}")
    print(lines[-1], flush=True)

    # ---- training arms: shared q*, three inits ----
    tr_dev = torch.as_tensor(tr, dtype=torch.long, device=dev)
    frame_sid = torch.as_tensor(np.repeat(tr, NF), dtype=torch.long, device=dev)
    frame_fid = torch.as_tensor(np.tile(np.arange(NF), len(tr)), dtype=torch.long, device=dev)
    frame_lab = torch.as_tensor(y_all[tr].reshape(-1).astype(np.float32), device=dev)
    n_fr = len(frame_sid)
    q_joint_mean = Qj[tr_dev].mean(0)                       # [28,128] pre-rotary
    q_mp_mean = Qmp[tr_dev].mean(dim=(0, 1))
    g = torch.Generator(device="cpu").manual_seed(0)
    q_rand = torch.randn(N_HEADS, HD, generator=g).to(dev)
    q_rand = q_rand * (q_joint_mean.norm() / q_rand.norm())
    init_q = {"joint": q_joint_mean, "mp": q_mp_mean, "random": q_rand}

    results = {}
    for init_name in [x.strip() for x in args.inits.split(",") if x.strip()]:
        print(f"--- training q* (init={init_name}) ---", flush=True)
        torch.manual_seed(0)
        q = init_q[init_name].clone().requires_grad_(True)
        w = torch.zeros(3584, device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([q, w, b], lr=args.lr, weight_decay=args.weight_decay)
        lossf = torch.nn.BCEWithLogitsLoss()
        hist = ["epoch,loss,train_auc,eval_dprime"]

        def quick_eval_dprime(q_now):
            """held-out d' of the current q* (diagnostic trajectory; NOT used for selection)."""
            sid = torch.as_tensor(np.repeat(ev, NF), dtype=torch.long, device=dev)
            fid = torch.as_tensor(np.tile(np.arange(NF), len(ev)), dtype=torch.long, device=dev)
            msgs = []
            with torch.no_grad():
                for i in range(0, len(sid), 1024):
                    qr = rot_q_batch(q_now.unsqueeze(0).expand(len(sid[i:i + 1024]), -1, -1),
                                     sid[i:i + 1024])
                    msgs.append(msg_batch(qr, sid[i:i + 1024], fid[i:i + 1024]).cpu().numpy())
            X = np.concatenate(msgs).reshape(len(ev), NF, -1).astype(np.float32)
            return dprime_pair(X, y_all[ev])[0]
        for ep in range(args.epochs):
            perm = torch.randperm(n_fr, device=dev)
            ep_loss = 0.0; nb = 0
            for i in range(0, n_fr, args.batch):
                bi = perm[i:i + args.batch]
                sid, fid, lab = frame_sid[bi], frame_fid[bi], frame_lab[bi]
                qr = rot_q_batch(q.unsqueeze(0).expand(len(bi), -1, -1), sid)
                logit = msg_batch(qr, sid, fid) @ w + b
                loss = lossf(logit.squeeze(-1), lab)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            if ep % 20 == 0 or ep == args.epochs - 1:
                with torch.no_grad():
                    qr = rot_q_batch(q.unsqueeze(0).expand(n_fr, -1, -1), frame_sid)
                    lg = []
                    for i in range(0, n_fr, 2048):
                        lg.append((msg_batch(qr[i:i + 2048], frame_sid[i:i + 2048],
                                             frame_fid[i:i + 2048]) @ w + b).cpu())
                    lg = torch.cat(lg).squeeze(-1).numpy()
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_all[tr].reshape(-1), lg)
                dep = quick_eval_dprime(q.detach())
                hist.append(f"{ep},{ep_loss/nb:.5f},{auc:.4f},{dep:.4f}")
                print(f"  ep{ep:>3} loss {ep_loss/nb:.4f} train-auc {auc:.3f} "
                      f"eval-d'={dep:.2f} ({time.time()-t0:.0f}s)", flush=True)
                torch.save({"q": q.detach().cpu(), "w": w.detach().cpu(), "b": b.detach().cpu(),
                            "epoch": ep, "init": init_name},
                           out / f"q_star_{init_name}.pt")
        (out / f"train_history_{init_name}.csv").write_text("\n".join(hist) + "\n")
        traj = [(int(h.split(",")[0]), float(h.split(",")[3])) for h in hist[1:]]
        best_ep, best_d = max(traj, key=lambda x: x[1])
        lines.append(f"  trained_q[{init_name}] eval-d' trajectory max={best_d:.2f} (ep {best_ep}) "
                     f"-- cherry-picked-on-eval UPPER bound, not the deployable number")
        qf = q.detach()
        dw = eval_dprime(f"trained_q[{init_name}]",
                         lambda s, f, _q=qf: rot_q_batch(_q.unsqueeze(0).expand(len(s), -1, -1), s),
                         ev, "eval")
        eval_dprime(f"trained_q[{init_name}]",
                    lambda s, f, _q=qf: rot_q_batch(_q.unsqueeze(0).expand(len(s), -1, -1), s),
                    all_idx, "full500")
        results[init_name] = dw

    best_init = max(results, key=results.get)
    best = results[best_init]
    band = ("GO (>=4): slot-head architecture promises gate-law exact "
            + "/".join(f"{gate_law_exact(best, N, N // 2):.2f}@N={N}" for N in (32, 64, 128))
            if best >= 4.0 else
            "REQUIREMENT-1 CONFIRMED (~2): learned shared queries cannot beat the joint query; "
            "per-frame addressing must be architectural" if best <= 2.6 else
            f"PARTIAL RECOVERY ({best:.2f} between the joint ~2.1 and mp 3.82/4.47 anchors)")
    lines.append(f"  VERDICT (eval-split, best init={best_init}): trained-q d'={best:.2f} -> {band}")
    lines.append(f"  (gate-law assumes E=N/2 evidence frames; per-frame p=1-Phi(d'/2))")
    print(lines[-2], flush=True)

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    (out / "run_config.json").write_text(json.dumps({
        "capture": str(cap), "layer": L, "train_frac": args.train_frac, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr, "inits": args.inits, "device": str(dev),
        "anchor_gate_pass": bool(gate_ok), "trained_dprime_eval": results,
    }, indent=2))
    print(f"wrote {out} ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
