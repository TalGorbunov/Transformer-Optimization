#!/usr/bin/env python3
"""ARMOR Experiment B — the Buitrago & Gu state-passing control (arXiv:2507.02782).

Closes RELATED_WORK Top Threat #7: "the recurrent drift rung may be under-trained,
not fundamental — cheap state-coverage interventions fix recurrent length gen."
This script reruns the EXACT recagg P2 extrapolation protocol (fit @{8,16},
zero-shot @{32,64,128}, canonical 20k-epoch recipe, PCA-128, aux BCE, subset-count
augmentation, order canary) with intervention arms on R2 (GRU) and R3 (SSM):

  base    the P2 head unchanged (replication line)
  sp      STATE PASSING, count-consistent port: with prob --p-sp the initial state
          is a DETACHED final state of another training sequence from the previous
          step of the same group (batch-permuted), and the count target becomes
          carried_count + subset_count — i.e. the virtual CONCATENATED stream is
          the sample, exactly "carry final state across concatenated training
          sequences". Borrows are rejected where the total would exceed the cls
          support (n_fit_max), so the LABEL RANGE IS UNCHANGED — the intervention
          delivers long-horizon STATE coverage without extending count supervision
          (that separation is the point of the control). Chains compound across
          steps, so training visits states of arbitrarily long virtual streams.
  noise   FITTED-NOISE initial state (paper variant): h0 ~ N(mu, sigma) with mu,
          sigma running per-dim stats of observed final states; target = current
          count (paper semantics — the model must be robust to a nonzero start).
          PRE-REGISTERED EXPECTATION: ill-posed for the pure integrator R3 (a
          borrowed-looking state offsets the integral and the current count is
          not recoverable from h_N alone) — a NEGATIVE effect on R3 is itself
          evidence for the identifiability leg of the mechanism triad. The GRU
          can learn to gate the initial state away.
  spnoise both.

R1 sum-probe runs as the invariant reference line (0.996/1.000 in P2).
New per-arm readout-range probe: max-correct-count (largest gold predicted
correctly, and largest gold with recall >= 0.5) per zero-shot cell — the Yehudai
range-cap prediction says trained heads stay capped near the fit-range counts
even if state dynamics stabilize.

Pre-registered bands (outputs/armor/CAMPAIGN_BRIEF.md): RESCUED = EM_reg >= 0.90
@N=64 zero-shot HF@512 for any intervention arm; NOT RESCUED = below.

recagg originals are imported, never edited (their anchors are law).

Usage (one domain per invocation; HF @512 is the primary cell):
  python scripts/armor/train_heads_sp.py \
      --fit  outputs/ninv/20260809_235142_hf8_leaf512/feats_N8.npz:8 \
             outputs/recagg/p1_captures/20260817_154347_hf16_512/feats_N16.npz:16 \
      --eval outputs/recagg/p1_captures/20260817_154347_hf32_512/feats_N32.npz:32 \
             outputs/recagg/p1_captures/20260817_154347_hf64_512/feats_N64.npz:64 \
             outputs/armor/p1_captures_128/<ts>_hf128_512/feats_N128.npz:128 \
      --label hf512_sp --output outputs/armor/b_statepass/<ts>_hf512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "scripts/ninv"))
sys.path.insert(0, str(_REPO / "scripts/recagg"))

from metrics_skew import class_report, cluster_bootstrap_ci  # noqa: E402
from train_heads import (  # noqa: E402
    GRUHead,
    SelectiveSSMHead,
    dist_line,
    fit_frame_probe,
    load_capture,
    n_params,
)

from sklearn.decomposition import PCA  # noqa: E402


# ------------------------------------------------- heads with initial-state input

class GRUHeadSP(GRUHead):
    """P2 GRU head + optional initial hidden state; returns final state."""

    def forward(self, x, h0=None):                 # x (B, N, H); h0 (1, B, hidden)
        z, hN = self.gru(self.proj(x), h0)
        cls, reg = self.out(z[:, -1])
        return cls, reg, self.aux(z).squeeze(-1), hN.detach()


class SelectiveSSMHeadSP(SelectiveSSMHead):
    """P2 minimal selective SSM + optional initial state; returns final state."""

    def forward(self, x, h0=None):                 # x (B, N, H); h0 (B, state)
        u = self.proj(x)
        A = -torch.exp(self.A_log)
        h = x.new_zeros(x.shape[0], self.A_log.shape[0]) if h0 is None else h0
        ys = []
        for t in range(u.shape[1]):
            ut = u[:, t]
            dt = torch.sigmoid(self.w_dt(ut))
            h = torch.exp(dt * A) * h + dt * self.b
            ys.append(self.w_c(ut) * h)
        aux = self.aux(torch.stack(ys, dim=1)).squeeze(-1)
        cls, reg = self.out(h)
        return cls, reg, aux, h.detach()


def make_h0_gru(state, mask):                       # state (B, hidden) -> (1, B, hidden)
    return (state * mask.unsqueeze(-1)).unsqueeze(0)


def make_h0_ssm(state, mask):
    return state * mask.unsqueeze(-1)


class RunningStats:
    """Per-dim running mean/std of observed final states (fitted noise)."""

    def __init__(self, momentum=0.01):
        self.mu = None
        self.var = None
        self.m = momentum

    def update(self, h):                            # h (B, D) detached
        mu_b, var_b = h.mean(0), h.var(0, unbiased=False)
        if self.mu is None:
            self.mu, self.var = mu_b, var_b
        else:
            self.mu = (1 - self.m) * self.mu + self.m * mu_b
            self.var = (1 - self.m) * self.var + self.m * var_b

    def sample(self, B, g):
        if self.mu is None:
            return None
        eps = torch.randn(B, self.mu.shape[0], generator=g)
        return self.mu + eps * self.var.clamp_min(1e-8).sqrt()


# ------------------------------------------------------------------ training loop

def train_sp(model, groups, epochs, aux_w, mode, n_max, p_sp=0.5, lr=1e-3,
             wd=1e-4, seed=0):
    """P2 train_multilen + interventions. groups: list of (Xtr, Ytr) tensors.
    mode in {none, sp, noise, spnoise}. Buffers/noise stats are PER GROUP (fit
    lengths differ); borrowed states are always detached (paper: stop-gradient)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = next(model.parameters()).device
    is_gru = isinstance(model, GRUHeadSP)
    make_h0 = make_h0_gru if is_gru else make_h0_ssm
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    buf = [None] * len(groups)                     # (final_states (B,D), counts (B,))
    stats = [RunningStats() for _ in groups]
    n_borrow = n_noise = 0
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for gi, (Xtr, Ytr) in enumerate(groups):
            B, N = Ytr.shape
            k = int(torch.randint(1, N + 1, (1,), generator=g))
            idx = torch.argsort(torch.rand(B, N, generator=g), dim=1)[:, :k].to(dev)
            Xb = Xtr[torch.arange(B, device=dev)[:, None], idx]
            Yb = Ytr[torch.arange(B, device=dev)[:, None], idx]
            Gb = Yb.sum(1)
            target = Gb
            h0 = None
            use_sp = mode in ("sp", "spnoise") and buf[gi] is not None \
                and float(torch.rand(1, generator=g)) < p_sp
            use_noise = mode in ("noise", "spnoise") and not use_sp \
                and stats[gi].mu is not None \
                and float(torch.rand(1, generator=g)) < p_sp
            if use_sp:
                Hp, Cp = buf[gi]
                pidx = torch.randperm(Hp.shape[0], generator=g)[:B]
                h_cand, c_cand = Hp[pidx].to(dev), Cp[pidx].to(dev)
                ok = ((c_cand + Gb) <= n_max).float()
                h0 = make_h0(h_cand, ok)
                target = Gb + (c_cand * ok.long())
                n_borrow += int(ok.sum())
            elif use_noise:
                h_cand = stats[gi].sample(B, g).to(dev)
                h0 = make_h0(h_cand, torch.ones(B, device=dev))
                n_noise += B
            logits, scalar, frame_logits, hN = model(Xb, h0)
            loss = loss + (F.cross_entropy(logits, target)
                           + 0.1 * F.mse_loss(scalar, target.float())
                           + aux_w * F.binary_cross_entropy_with_logits(
                               frame_logits, Yb.float()))
            hN_flat = hN[0] if is_gru else hN      # (B, D)
            buf[gi] = (hN_flat.cpu(), target.detach().cpu())
            stats[gi].update(hN_flat.cpu())
        loss.backward()
        opt.step()
    model.eval()
    return {"n_borrow": n_borrow, "n_noise": n_noise}


@torch.no_grad()
def predict(model, X, clamp_max):
    logits, scalar, _, _ = model(X.to(next(model.parameters()).device))
    return (logits.argmax(-1).cpu().numpy(),
            scalar.round().clamp(0, clamp_max).long().cpu().numpy())


def max_correct(pred, gold):
    """(largest gold predicted correctly at least once, largest gold with
    recall >= 0.5) — the readout-range probe."""
    hi_any = hi_maj = -1
    for c in np.unique(gold):
        rec = float((pred[gold == c] == c).mean())
        if rec > 0:
            hi_any = int(c)
        if rec >= 0.5:
            hi_maj = int(c)
    return hi_any, hi_maj


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True, metavar="NPZ:N")
    ap.add_argument("--eval", nargs="+", required=True, metavar="NPZ:N")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--label", default="sp")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--p-sp", type=float, default=0.5)
    ap.add_argument("--canary-perms", type=int, default=3)
    ap.add_argument("--modes", default="base,sp,noise,spnoise")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                       else args.device if args.device != "auto" else "cpu")
    print(f"device: {dev}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def parse(spec):
        p, n = spec.rsplit(":", 1)
        X, Y, G = load_capture(p, args.layer)
        assert X.shape[1] == int(n), f"{p}: N mismatch"
        return {"path": p, "n": int(n), "X": X, "Y": Y, "G": G}

    fits = [parse(s) for s in args.fit]
    evals = [parse(s) for s in args.eval]
    H = fits[0]["X"].shape[-1]
    n_fit_max = max(f["n"] for f in fits)
    for d in fits + evals:
        print(f"[{'fit' if d in fits else 'eval'}] N={d['n']} n={len(d['G'])} "
              f"{d['path']}\n    gold dist: {dist_line(d['G'], d['n'])}")

    modes = args.modes.split(",")
    arms = ["R1_sum_probe"] + [f"{h}_{m}" for h in ("R2_gru", "R3_ssm")
                               for m in modes]
    cells = [f"N{f['n']}_in" for f in fits] + [f"N{e['n']}_zs" for e in evals]
    pool = {c: {a: {"cls": [], "reg": []} for a in arms} for c in cells}
    true = {c: [] for c in cells}
    canary = {c: {a: [] for a in arms} for c in cells}
    p_frame_cell = {c: [] for c in cells}

    for s in range(args.seeds):
        print(f"\n--- seed {s} ---", flush=True)
        halves = []
        for f in fits:
            m = len(f["G"])
            idx = np.random.default_rng(s).permutation(m)
            tr, ev = idx[: m // 2], idx[m // 2:]
            halves.append((f, tr, ev))
            print(f"N={f['n']} TRAIN dist: {dist_line(f['G'][tr], f['n'])}")
            print(f"N={f['n']} EVAL  dist: {dist_line(f['G'][ev], f['n'])}")

        Ftr = np.concatenate([f["X"][tr].reshape(-1, H) for f, tr, _ in halves])
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        pca = PCA(n_components=min(args.pca, len(Ftr) - 1, H), whiten=True,
                  random_state=0).fit((Ftr - mu) / sd)
        tx = lambda X3: pca.transform(  # noqa: E731
            ((X3.reshape(-1, H) - mu) / sd)).reshape(*X3.shape[:2], -1).astype(np.float32)
        probe = fit_frame_probe(
            np.concatenate([f["X"][tr].reshape(1, -1, H) for f, tr, _ in halves], axis=1),
            np.concatenate([f["Y"][tr].reshape(1, -1) for f, tr, _ in halves], axis=1))
        groups = [(torch.from_numpy(tx(f["X"][tr])).to(dev),
                   torch.from_numpy(f["Y"][tr]).to(dev))
                  for f, tr, _ in halves]

        heads = {}
        for m_ in modes:
            heads[f"R2_gru_{m_}"] = GRUHeadSP(args.pca, n_fit_max, args.proj,
                                              args.hidden).to(dev)
            heads[f"R3_ssm_{m_}"] = SelectiveSSMHeadSP(args.pca, n_fit_max,
                                                       args.proj, args.hidden).to(dev)
        for name, model in heads.items():
            mode = name.split("_")[-1]
            if s == 0:
                print(f"{name}: {n_params(model):,} params", flush=True)
            info = train_sp(model, groups, args.epochs, args.aux_weight, mode,
                            n_fit_max, p_sp=args.p_sp, seed=s)
            if s == 0:
                print(f"  {name}: borrows={info['n_borrow']} noise={info['n_noise']}")
                torch.save({"state_dict": model.state_dict(), "scaler_mu": mu,
                            "scaler_sd": sd, "pca_components": pca.components_,
                            "pca_mean": pca.mean_, "n_max": n_fit_max,
                            "layer": args.layer, "mode": mode, "fit": args.fit},
                           out / f"{name}_seed0.pt")

        def eval_cell(cell, X, Y, G, N):
            bits = probe(X)
            p_frame_cell[cell].append(float((bits == Y).mean()))
            pool[cell]["R1_sum_probe"]["cls"].append(bits.sum(1))
            pool[cell]["R1_sum_probe"]["reg"].append(bits.sum(1))
            canary[cell]["R1_sum_probe"].append(0.0)
            Xt = torch.from_numpy(tx(X))
            for name, model in heads.items():
                pc, pr = predict(model, Xt, N)
                pool[cell][name]["cls"].append(pc)
                pool[cell][name]["reg"].append(pr)
                em0 = float((pr == G).mean())
                deltas = []
                for k in range(args.canary_perms):
                    prm = np.stack([np.random.default_rng(7000 + s * 91 + k * 13 + i)
                                    .permutation(N) for i in range(len(G))])
                    _, prk = predict(model, Xt[np.arange(len(G))[:, None], prm], N)
                    deltas.append(abs(float((prk == G).mean()) - em0))
                canary[cell][name].append(max(deltas))
            true[cell].append(G)

        for f, tr, ev in halves:
            eval_cell(f"N{f['n']}_in", f["X"][ev], f["Y"][ev], f["G"][ev], f["n"])
        for e in evals:
            eval_cell(f"N{e['n']}_zs", e["X"], e["Y"], e["G"], e["n"])
        print(f"  seed {s} done  {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------------- report
    lines = [f"ARMOR B state-passing control — {args.label}",
             f"fit: {args.fit}", f"eval: {args.eval}",
             f"layer={args.layer} seeds={args.seeds} epochs={args.epochs} "
             f"pca={args.pca} aux={args.aux_weight} p_sp={args.p_sp} "
             f"(canonical P2 recipe + interventions)",
             f"cls support capped at 0..{n_fit_max}; EM_reg (scalar-round, clamped "
             f"to eval N) carries the verdict. Band: RESCUED = EM_reg >= 0.90 "
             f"@N64_zs (HF@512).", ""]
    hdr = (f"{'cell':<9} {'arm':<16} {'EM_reg':>6} {'ci95':>15} {'EM_cls':>6} "
           f"{'major':>6} {'bal':>6} {'bound':>6} {'mc_any':>6} {'mc_50':>6} "
           f"{'canaryΔ':>8}")
    lines.append(hdr)
    results = {}
    for c in cells:
        tr_ = np.concatenate(true[c])
        n_cell = int(c[1:].split("_")[0])
        pf = float(np.mean(p_frame_cell[c]))
        bound = pf ** n_cell
        for a in arms:
            pr = np.concatenate(pool[c][a]["reg"])
            pc = np.concatenate(pool[c][a]["cls"])
            rep = class_report(pr, tr_, n_classes=n_cell + 1)
            lo, hi, _ = cluster_bootstrap_ci((pr == tr_)[:, None])
            em_cls = float((pc == tr_).mean())
            can = max(canary[c][a])
            mc_any, mc_50 = max_correct(pr, tr_)
            lines.append(f"{c:<9} {a:<16} {rep['raw']:>6.3f} [{lo:.3f},{hi:.3f}] "
                         f"{em_cls:>6.3f} {rep['majority']:>6.3f} "
                         f"{rep['balanced']:>6.3f} {bound:>6.3f} {mc_any:>6d} "
                         f"{mc_50:>6d} {can:>8.3f}")
            results.setdefault(c, {})[a] = {
                "em_reg": rep["raw"], "ci": [lo, hi], "em_cls": em_cls,
                "majority": rep["majority"], "balanced": rep["balanced"],
                "recall": rep["recall"], "support": rep["support"],
                "bound": bound, "p_frame": pf, "canary_max": can,
                "max_correct_any": mc_any, "max_correct_rec50": mc_50,
                "n": int(len(tr_))}
        lines.append("")
    lines.append("per-class recall (zero-shot cells):")
    for c in cells:
        if not c.endswith("_zs"):
            continue
        for a in arms:
            r = results[c][a]
            rec = " ".join(f"c{k}:{v:.2f}" for k, v in enumerate(r["recall"])
                           if r["support"][k] > 0)
            lines.append(f"  {c} {a}: {rec}")
    lines.append(f"wall {time.time() - t0:.0f}s")
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    print(f"wrote {out}/report.txt, results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
