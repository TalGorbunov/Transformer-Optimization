"""Bundle D (PREREG_AGG 'Added 2026-09-03', D1-D3): label-free attention-pattern statistics at the band
layers' tail rows -> K. Stats per (layer, head, rowset): k_eff over sentences, exp-entropy, peak counts at
tau in {2,4,8} x uniform, context share. Baselines: residual state, oracle needle share."""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from needles import story, PROMPT, REGIMES  # noqa: E402
from eval_babilong import build  # noqa: E402
from modeling import load_text_model  # noqa: E402
from osq import model_parts, _call  # noqa: E402
from needles_band import prep, tail_attention  # noqa: E402

STATS = ("keff", "expH", "peak2", "peak4", "peak8", "ctx")


@torch.no_grad()
def pattern_feats(model, rec, facts, dev, FL, T0=8):
    """-> feats [len(FL), H, 2, 6] (rowsets: mean of last T0 tail rows, answer row), oracle share [len(FL), H, 2],
    states {layer: answer-row state} for layers in FL and 'fin'."""
    embed, layers, norm, head, rotary = model_parts(model); h, p, pe = prep(model, rec["ids"], dev)
    S = rec["seq"]; fin = rec["fin"]; bl = rec["blocks"]; F = len(bl)
    blk = torch.zeros(S, dtype=torch.long, device=dev) - 1
    for j, (a_, b_) in enumerate(bl): blk[a_:b_] = j
    ctx = blk >= 0; fm = torch.zeros(F, dtype=torch.bool, device=dev); fm[facts] = True
    feats, oracle, states = {}, {}, {}
    for li, lay in enumerate(layers):
        if li in FL:
            states[li] = h[0, -1].detach().half().cpu()
            w, _, _ = tail_attention(lay, h, pe, fin, None)                    # [H,T,S] fp32
            H_, T = w.shape[0], w.shape[1]
            ws = torch.zeros(H_, T, F, device=dev).index_add_(2, blk[ctx], w[:, :, ctx])   # per-sentence weights
            cshare = ws.sum(-1)                                                 # [H,T]
            pn = ws / (cshare[..., None] + 1e-9)
            keff = 1.0 / ((pn ** 2).sum(-1) + 1e-9)
            expH = torch.exp(-(pn * torch.log(pn + 1e-12)).sum(-1))
            dead = cshare < 1e-3                                                 # head puts (almost) nothing on the context
            keff = torch.where(dead, torch.full_like(keff, float(F)), keff.clamp(max=float(F)))
            expH = torch.where(dead, torch.full_like(expH, float(F)), expH.clamp(max=float(F)))
            u = 1.0 / F
            pk = [(pn > tau * u).float().sum(-1) for tau in (2, 4, 8)]
            st = torch.stack([keff, expH, pk[0], pk[1], pk[2], cshare], -1)   # [H,T,6]
            f = torch.stack([st[:, -T0:].mean(1), st[:, -1]], 1)              # [H,2,6]
            feats[li] = f.cpu(); osh = pn[:, :, fm].sum(-1); oracle[li] = torch.stack([osh[:, -T0:].mean(1), osh[:, -1]], 1).cpu()
        h = _call(lay, h, p, pe)
    states["fin"] = norm(h[0, -1]).detach().half().cpu()
    return torch.stack([feats[li] for li in FL]), torch.stack([oracle[li] for li in FL]), states


def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev)
    nL = len(model_parts(model)[1]); b0, b1 = [int(x) for x in a.band.split("+")]
    FL = sorted(set(range(b0, b1 + 1)) | {a.layer, a.layer // 2, min(a.layer + 4, nL - 1)})
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), n_layers=nL, FL=FL), indent=1)); items = []; t0 = time.time()
    for reg in a.regimes.split("+"):
        for K in [int(k) for k in a.ks.split("+")]:
            D = REGIMES[reg](K)
            if D is None or D < 0: continue
            rng = random.Random(a.seed * 1000 + K)
            for i in range(a.n):
                sents, facts, q = story(K, D, rng); rec = build(tok, "qa1", q, sents, PROMPT)
                f, o, st = pattern_feats(model, rec, facts, dev, set(FL))
                items.append(dict(regime=reg, K=K, i=i, F=len(rec["blocks"]), feats=f, oracle=o, states=st))
            fk = torch.stack([it["feats"] for it in items[-a.n:]])         # [n, FL, H, 2, 6]
            li = FL.index(a.layer); print(f"[{time.time()-t0:.0f}s] {reg} K={K:2d} F={items[-1]['F']} keff(tail, head-mean) band {fk[:, :len(range(b0,b1+1)), :, 0, 0].mean():.1f} L {fk[:, li, :, 0, 0].mean():.1f} | max-head keff at band {fk[:, :len(range(b0,b1+1)), :, 0, 0].mean(0).max():.1f} | peak4 head-mean {fk[:, :, :, 0, 3].mean():.1f}", flush=True)
    torch.save(dict(items=items, FL=FL), odir / "pattern.pt"); print("->", odir); return analyze(odir)


def _ridge_fit_pred(Xtr, ytr, Xte, lam):
    import numpy as np
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym)); return Xte @ w + ym


def _cv(X, y, K, lam=1.0, folds=5):
    import numpy as np
    n = len(y); idx = np.arange(n); pred = np.zeros(n)
    for f in range(folds):
        te = idx % folds == f; pred[te] = _ridge_fit_pred(X[~te], y[~te], X[te], lam)
    Kh = np.rint(np.exp(pred)); e = np.abs(Kh - K); s = K <= 16
    return dict(exact=float((e == 0).mean()), exact16=float((e[s] == 0).mean()), within1=float((e <= 1).mean()))


def _extrap(X, y, K, lam=1.0):
    import numpy as np
    tr, te = K <= 16, K > 16
    if te.sum() == 0: return dict(rel=float("nan"), med48=float("nan"), exact=float("nan"))
    pred = np.exp(_ridge_fit_pred(X[tr], y[tr], X[te], lam)); Kt = K[te]; Kh = np.rint(pred)
    kmax = Kt.max(); return dict(rel=float((np.abs(pred - Kt) / Kt).mean()), med48=float(np.median(Kh[Kt == kmax])), exact=float((Kh == Kt).mean()), kmax=int(kmax))


def analyze(d):
    import numpy as np
    d = Path(d); cfg = json.load(open(d / "config.json")); L = cfg["layer"]; P = torch.load(d / "pattern.pt"); items, FL = P["items"], P["FL"]
    b0, b1 = [int(x) for x in cfg["band"].split("+")]; band_idx = [FL.index(l) for l in range(b0, b1 + 1)]; li_L = FL.index(L)
    out = [f"# Bundle D — {cfg['model']} L{L} band {b0}–{b1} FL={FL} ({d})"]
    for reg in sorted({it["regime"] for it in items}):
        R = [it for it in items if it["regime"] == reg]; K = np.array([it["K"] for it in R]); y = np.log(K)
        Fe = torch.stack([it["feats"] for it in R]).numpy()      # [n, FL, H, 2, 6]
        Or = torch.stack([it["oracle"] for it in R]).numpy()     # [n, FL, H, 2]
        n = len(R); sets = {
            "pattern: band layers, tail+answer, all stats": Fe[:, band_idx].reshape(n, -1),
            "pattern: band layers, k_eff only": Fe[:, band_idx, :, :, 0].reshape(n, -1),
            "pattern: band layers, peak counts only": Fe[:, band_idx, :, :, 2:5].reshape(n, -1),
            "pattern: all FL layers, all stats": Fe.reshape(n, -1),
            "pattern: probe layer L only": Fe[:, li_L].reshape(n, -1),
            "ORACLE needle share, band layers": Or[:, band_idx].reshape(n, -1),
            "state: answer row at L": torch.stack([it["states"][L] for it in R]).float().numpy(),
            "state: answer row final": torch.stack([it["states"]["fin"] for it in R]).float().numpy(),
        }
        out.append(f"\n## {reg}  (n={n}, K={sorted(set(K.tolist()))}, sentences F={sorted(set(it['F'] for it in R))[:3]}…)")
        out.append(f"{'feature set':50s} {'exact':>6s} {'ex≤16':>6s} {'within1':>8s} | fit≤16→test>16: {'exact':>6s} {'rel.err':>8s} {'med K̂@max':>10s}")
        for name, X in sets.items():
            lam = 10.0 if name.startswith("state") else 1.0; cv = _cv(X, y, K, lam); ex = _extrap(X, y, K, lam)
            out.append(f"{name:50s} {cv['exact']:6.2f} {cv['exact16']:6.2f} {cv['within1']:8.2f} | {'':17s}{ex['exact']:6.2f} {ex['rel']:8.2f} {ex['med48']:10.1f}")
        # D3: per (layer, head) linearity of k_eff (tail-row mean) vs K
        ks = sorted(set(K.tolist())); means = np.stack([Fe[K == k][:, :, :, 0, 0].mean(0) for k in ks])   # [nK, FL, H]
        x = np.array(ks, float); xm = x - x.mean(); slope = (xm[:, None, None] * (means - means.mean(0))).sum(0) / (xm ** 2).sum()
        pred = means.mean(0) + slope * xm[:, None, None]; r2 = 1 - ((means - pred) ** 2).sum(0) / (((means - means.mean(0)) ** 2).sum(0) + 1e-9)
        good = (slope >= .6) & (slope <= 1.1) & (r2 >= .95); best = np.unravel_index(np.argmax(r2 * ((slope > .4) & (slope < 1.3))), r2.shape)
        out.append(f"D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: {int(good.sum())}/{good.size}; best L{FL[best[0]]}H{best[1]} slope {slope[best]:.2f} R² {r2[best]:.3f}; k_eff by K: " + " ".join(f"{k}:{means[i, best[0], best[1]]:.1f}" for i, k in enumerate(ks)))
        # single-head calibrated estimator: fit a,b on K<=16 for the best head, test on K>16
        tr = K <= 16; xf = Fe[:, best[0], best[1], 0, 0]
        A = np.vstack([xf[tr], np.ones(tr.sum())]).T; ab = np.linalg.lstsq(A, K[tr], rcond=None)[0]; te = K > 16
        if te.sum(): Kh = ab[0] * xf[te] + ab[1]; out.append(f"D3 single-head estimator (fit K≤16 → K>16): rel.err {np.mean(np.abs(Kh - K[te]) / K[te]):.2f}, within-1 {np.mean(np.abs(np.rint(Kh) - K[te]) <= 1):.2f}, median K̂ at K={K.max()}: {np.median(Kh[K[te] == K.max()]):.1f}")
    txt = "\n".join(out); print(txt); (d / "ANALYSIS.md").write_text(txt + "\n"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--layer", type=int, default=27); ap.add_argument("--band", default="18+24")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--ks", default="1+2+3+4+6+8+12+16+24+32+48"); ap.add_argument("--regimes", default="HC+HF")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--output", default="outputs/needles_pattern"); ap.add_argument("--analyze", default="")
    a = ap.parse_args(); return analyze(Path(a.analyze)) if a.analyze else run(a)


if __name__ == "__main__":
    sys.exit(main())
