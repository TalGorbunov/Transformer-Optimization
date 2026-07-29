#!/usr/bin/env python3
"""CoGNN-style broadcast gate (Exp C, oneforward brief 2026-07-15): can a per-token learned
ADDITIVE logit offset repair the joint query's within-frame routing?

The trained-query NO-GO showed a shared trained query cannot beat the joint query (~2.1) —
per-frame addressing needs frame identity. This probes the CONTENT side instead: keep the
(contaminated) joint carrier query, but add b_j = MLP(features of token j) to the within-frame
attention logits: message = o_proj(sum_j softmax(q_joint.k~_j/sqrt(d) + b_j) v~_j).
The gate sees only each token's own features (no frame/sample identity), so it is deployable.

Arms:
  content: features = [k_j, v_j] pre-rotary (1024-dim in)
  qcond:   features = [k_j, v_j, q_pad(frame)] (pad arm's clean per-frame query; the question
           is always available at inference, so still deployable)

Same capture / split (RandomState(0), 60/40) / logistic proxy loss / dprime_pair eval as
trained_query_ceiling.py. Anchors must reproduce (eval scale): joint-q x joint-kv ~2.09,
mp-q x joint-kv ~3.82. Pre-registered bands: floor 2.09, ceiling 3.82 (= clean routing on
joint values), GO >= 3.0, ~2.1 => routing not repairable from content.
Gate MLP last layer is zero-init, so epoch 0 == the joint anchor by construction.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from experiments.glstm.dprime_vs_n import dprime_pair

N_HEADS, N_KV, HD = 28, 4, 128
GS = N_HEADS // N_KV


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def merge_mrope(cs, mrope):
    """cs [3,1,T,128] -> merged [T,128] (the section-interleave HF applies inside rope)."""
    sec = list(mrope) * 2
    r = torch.cat([m[i % 3] for i, m in enumerate(cs.float().split(sec, dim=-1))], dim=-1)
    return r.reshape(-1, HD)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--arms", default="content,qcond")
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

    # ---- precompute: rotated K/V for the message path + PRE-ROTARY k/v for gate features ----
    n = len(S)
    cc_m = torch.zeros(n, 1, HD)
    sc_m = torch.zeros(n, 1, HD)
    sizes0 = None
    Kr = Vr = Kp = Vp = None
    Qj = torch.zeros(n, N_HEADS, HD)                       # joint carrier query, pre-rotary
    Qpad = torch.zeros(n, NF, N_HEADS * HD)                # pad per-frame queries (flat, gate feat)
    Qmp = torch.zeros(n, NF, N_HEADS, HD)                  # mp queries (anchor only)
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
            Kp = torch.zeros(n, NF, T, N_KV * HD, dtype=torch.float16, device=dev)
            Vp = torch.zeros(n, NF, T, N_KV * HD, dtype=torch.float16, device=dev)
        assert list(rec["joint_fg_sizes"]) == sizes0, "frame sizes differ across samples"
        A = rec["arms"]
        Qj[si] = torch.as_tensor(np.asarray(A["joint"][L]["q"]["all"], np.float32)).view(N_HEADS, HD)
        y_all[si] = np.asarray(rec["labels"], dtype=np.int64)
        for t in range(NF):
            a, b = int(offs[t]), int(offs[t]) + sizes[t]
            cf = merge_mrope(cos[..., a:b, :], mrope)      # [T,128]
            sf = merge_mrope(sin[..., a:b, :], mrope)
            k_raw = torch.as_tensor(np.asarray(A["joint"][L]["k"][t], np.float32))  # [T,512]
            v_raw = torch.as_tensor(np.asarray(A["joint"][L]["v"][t], np.float32))
            Kp[si, t] = k_raw.half().to(dev)
            Vp[si, t] = v_raw.half().to(dev)
            k = k_raw.view(-1, N_KV, HD).transpose(0, 1)
            kr = k * cf.unsqueeze(0) + rotate_half(k) * sf.unsqueeze(0)
            Kr[si, t] = kr.repeat_interleave(GS, dim=0).half().to(dev)
            v = v_raw.view(-1, N_KV, HD).transpose(0, 1)
            Vr[si, t] = v.repeat_interleave(GS, dim=0).half().to(dev)
            Qpad[si, t] = torch.as_tensor(np.asarray(A["pad"][L]["q"][t], np.float32)).view(-1)
            Qmp[si, t] = torch.as_tensor(np.asarray(A["mp"][L]["q"][t], np.float32)).view(N_HEADS, HD)
        if (si + 1) % 100 == 0:
            print(f"  precompute {si+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    cc_m = cc_m.to(dev); sc_m = sc_m.to(dev)
    Qj = Qj.to(dev); Qpad = Qpad.to(dev); Qmp = Qmp.to(dev)

    def rot_q_batch(qh, sids):
        c = cc_m[sids]; s = sc_m[sids]                      # [B,1,128]
        return qh * c + rotate_half(qh) * s

    def msg_batch(q_rot, sids, fids, gate=None):
        """q_rot [B,28,128]; gate [B,T] additive logit offsets or None -> messages [B,3584]."""
        K = Kr[sids, fids].float(); V = Vr[sids, fids].float()   # [B,28,T,128]
        lg = torch.einsum("bhd,bhtd->bht", q_rot, K) / np.sqrt(HD)
        if gate is not None:
            lg = lg + gate.unsqueeze(1)                          # broadcast over heads
        ctx = torch.einsum("bht,bhtd->bhd", torch.softmax(lg, -1), V).reshape(len(sids), -1)
        return ctx @ oproj.T

    lines = [f"=== BROADCAST GATE (L{L}, train {len(tr)}, eval {len(ev)}, hidden {args.hidden}) ==="]
    rows = ["cond,split,dprime_w,std,auc"]

    def eval_dprime(name, build_q, split_idx, split_name, gate_fn=None):
        sid = torch.as_tensor(np.repeat(split_idx, NF), dtype=torch.long, device=dev)
        fid = torch.as_tensor(np.tile(np.arange(NF), len(split_idx)), dtype=torch.long, device=dev)
        msgs = []
        with torch.no_grad():
            for i in range(0, len(sid), 1024):
                s_, f_ = sid[i:i + 1024], fid[i:i + 1024]
                q = build_q(s_, f_)
                g = gate_fn(s_, f_) if gate_fn is not None else None
                msgs.append(msg_batch(q, s_, f_, gate=g).cpu().numpy())
        X = np.concatenate(msgs).reshape(len(split_idx), NF, -1).astype(np.float32)
        dw, ds, da = dprime_pair(X, y_all[split_idx])
        lines.append(f"  {name:<28} [{split_name}] d'={dw:.2f}+-{ds:.2f} (auc {da:.2f})")
        rows.append(f"{name},{split_name},{dw:.4f},{ds:.4f},{da:.4f}")
        print(lines[-1], flush=True)
        return dw

    # ---- anchors (must reproduce before anything is trusted) ----
    all_idx = np.arange(n)
    anchors = {}
    for split_name, split_idx in (("eval", ev), ("full500", all_idx)):
        anchors[("joint", split_name)] = eval_dprime(
            "joint_q_joint_kv", lambda s, f: rot_q_batch(Qj[s], s), split_idx, split_name)
        anchors[("mp", split_name)] = eval_dprime(
            "mp_q_joint_kv", lambda s, f: rot_q_batch(Qmp[s, f], s), split_idx, split_name)
    gate_ok = (1.8 <= anchors[("joint", "eval")] <= 2.5
               and 3.4 <= anchors[("mp", "eval")] <= 4.2)
    lines.append(f"  ANCHOR GATE: joint-q eval {anchors[('joint','eval')]:.2f} (want ~2.09), "
                 f"mp-q eval {anchors[('mp','eval')]:.2f} (want ~3.82) -> "
                 f"{'PASS' if gate_ok else 'FAIL'}")
    print(lines[-1], flush=True)

    # ---- gate feature stats (train split) for input standardization ----
    tr_dev = torch.as_tensor(tr, dtype=torch.long, device=dev)
    with torch.no_grad():
        kf = Kp[tr_dev].float().reshape(-1, N_KV * HD)
        vf = Vp[tr_dev].float().reshape(-1, N_KV * HD)
        k_mu, k_sd = kf.mean(0), kf.std(0).clamp_min(1e-4)
        v_mu, v_sd = vf.mean(0), vf.std(0).clamp_min(1e-4)
        qp = Qpad[tr_dev].reshape(-1, N_HEADS * HD)
        q_mu, q_sd = qp.mean(0), qp.std(0).clamp_min(1e-4)
        del kf, vf, qp

    def gate_features(sids, fids, arm):
        k = (Kp[sids, fids].float() - k_mu) / k_sd          # [B,T,512]
        v = (Vp[sids, fids].float() - v_mu) / v_sd
        if arm == "content":
            return torch.cat([k, v], dim=-1)                # [B,T,1024]
        q = ((Qpad[sids, fids] - q_mu) / q_sd).unsqueeze(1).expand(-1, k.shape[1], -1)
        return torch.cat([k, v, q], dim=-1)                 # [B,T,1024+3584]

    IN_DIM = {"content": 2 * N_KV * HD, "qcond": 2 * N_KV * HD + N_HEADS * HD}

    frame_sid = torch.as_tensor(np.repeat(tr, NF), dtype=torch.long, device=dev)
    frame_fid = torch.as_tensor(np.tile(np.arange(NF), len(tr)), dtype=torch.long, device=dev)
    frame_lab = torch.as_tensor(y_all[tr].reshape(-1).astype(np.float32), device=dev)
    n_fr = len(frame_sid)

    results = {}
    for arm in [x.strip() for x in args.arms.split(",") if x.strip()]:
        print(f"--- training gate (arm={arm}, in_dim={IN_DIM[arm]}) ---", flush=True)
        torch.manual_seed(0)
        mlp = nn.Sequential(nn.Linear(IN_DIM[arm], args.hidden), nn.GELU(),
                            nn.Linear(args.hidden, 1)).to(dev)
        nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)   # start at joint anchor
        w = torch.zeros(3584, device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam(list(mlp.parameters()) + [w, b], lr=args.lr,
                               weight_decay=args.weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        hist = ["epoch,loss,eval_dprime"]

        def gate_fn(s_, f_, _arm=arm, _mlp=mlp):
            return _mlp(gate_features(s_, f_, _arm)).squeeze(-1)

        def quick_eval_dprime():
            sid = torch.as_tensor(np.repeat(ev, NF), dtype=torch.long, device=dev)
            fid = torch.as_tensor(np.tile(np.arange(NF), len(ev)), dtype=torch.long, device=dev)
            msgs = []
            with torch.no_grad():
                for i in range(0, len(sid), 1024):
                    s_, f_ = sid[i:i + 1024], fid[i:i + 1024]
                    qr = rot_q_batch(Qj[s_], s_)
                    msgs.append(msg_batch(qr, s_, f_, gate=gate_fn(s_, f_)).cpu().numpy())
            X = np.concatenate(msgs).reshape(len(ev), NF, -1).astype(np.float32)
            return dprime_pair(X, y_all[ev])[0]

        for ep in range(args.epochs):
            perm = torch.randperm(n_fr, device=dev)
            ep_loss = 0.0; nb = 0
            for i in range(0, n_fr, args.batch):
                bi = perm[i:i + args.batch]
                sid, fid, lab = frame_sid[bi], frame_fid[bi], frame_lab[bi]
                qr = rot_q_batch(Qj[sid], sid)
                logit = msg_batch(qr, sid, fid, gate=gate_fn(sid, fid)) @ w + b
                loss = lossf(logit.squeeze(-1), lab)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
            if ep % 20 == 0 or ep == args.epochs - 1:
                dep = quick_eval_dprime()
                hist.append(f"{ep},{ep_loss/nb:.5f},{dep:.4f}")
                print(f"  ep{ep:>3} loss {ep_loss/nb:.4f} eval-d'={dep:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                torch.save({"mlp": mlp.state_dict(), "w": w.detach().cpu(),
                            "b": b.detach().cpu(), "epoch": ep, "arm": arm,
                            "norm": {"k_mu": k_mu.cpu(), "k_sd": k_sd.cpu(),
                                     "v_mu": v_mu.cpu(), "v_sd": v_sd.cpu(),
                                     "q_mu": q_mu.cpu(), "q_sd": q_sd.cpu()}},
                           out / f"gate_{arm}.pt")
        (out / f"train_history_{arm}.csv").write_text("\n".join(hist) + "\n")
        traj = [(int(h.split(",")[0]), float(h.split(",")[2])) for h in hist[1:]]
        best_ep, best_d = max(traj, key=lambda x: x[1])
        lines.append(f"  gate[{arm}] eval-d' trajectory max={best_d:.2f} (ep {best_ep}) "
                     f"-- cherry-picked-on-eval UPPER bound, not the deployable number")
        dw = eval_dprime(f"gate[{arm}]",
                         lambda s, f: rot_q_batch(Qj[s], s), ev, "eval", gate_fn=gate_fn)
        eval_dprime(f"gate[{arm}]",
                    lambda s, f: rot_q_batch(Qj[s], s), all_idx, "full500", gate_fn=gate_fn)
        results[arm] = dw

    best_arm = max(results, key=results.get)
    best = results[best_arm]
    band = ("GO (>=3.0): a deployable trained routing-repair module exists"
            if best >= 3.0 else
            "NOT REPAIRABLE FROM CONTENT (~joint anchor): completes the addressing story "
            "from the third direction" if best <= 2.4 else
            f"PARTIAL ({best:.2f} between floor 2.09 and ceiling 3.82)")
    lines.append(f"  VERDICT (eval-split, best arm={best_arm}): gate d'={best:.2f} -> {band}")
    print(lines[-1], flush=True)

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "results.csv").write_text("\n".join(rows) + "\n")
    (out / "run_config.json").write_text(json.dumps({
        "capture": str(cap), "layer": L, "train_frac": args.train_frac, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr, "hidden": args.hidden, "arms": args.arms,
        "device": str(dev), "anchor_gate_pass": bool(gate_ok), "gate_dprime_eval": results,
    }, indent=2))
    print(f"wrote {out} ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
