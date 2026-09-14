"""Stage 0 of the set-read plan (PREREG_AGG.md, 'Added 2026-09-03'): is the count in the softmax
DENOMINATOR? For every layer and head, record the last row's log-normalizer log Z, the needle /
junk log-sum-exps and the needle share, for (A) the real question, (B) a control question about a
(person, room) pair absent from the story (same context, same prefix), (C) the counterfactual
context with the needles removed. Also the read-node residual state at three depths (baseline
probe). --analyze: ridge probes (log K target, 5-fold CV, rounded) per feature set."""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, ".")
from needles import story, PROMPT, REGIMES, PEOPLE, ROOMS  # noqa: E402
from eval_babilong import build  # noqa: E402
from modeling import load_text_model  # noqa: E402
from osq import model_parts, rope, _call  # noqa: E402

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def control_question(sents, q):
    """A (person, room) pair that occurs in NO sentence of the story (true count 0)."""
    p0 = q.split("did ")[1].split(" go")[0]
    pairs = {(p, r) for s in sents for p in PEOPLE for r in ROOMS if s.startswith(p + " ") and s.endswith("the " + r + ".")}
    for p in PEOPLE:
        for r in ROOMS:
            if (p, r) not in pairs and p != p0:
                return f"How many times did {p} go to the {r}?"
    return None


@torch.no_grad()
def scan(model, ids, blocks, facts, dev, depths):
    """Per layer & head: logZ over all columns, logZ over context, lse over needle tokens, lse over
    junk context tokens, needle share (context-restricted), context share. Returns dict of [L,H]
    fp32 tensors + residual states at `depths` (input of that layer; 'fin' = final norm) [d] fp16."""
    embed, layers, norm, head, rotary = model_parts(model)
    x = torch.tensor([ids], device=dev); S = len(ids)
    p = torch.arange(S, device=dev)[None]
    h = embed(x); c, s = rotary(h, p); pe = (c.to(h.dtype), s.to(h.dtype))
    a0, b1 = blocks[0][0], blocks[-1][1]
    fm = torch.zeros(S, dtype=torch.bool, device=dev)
    for j in facts:
        fm[blocks[j][0]:blocks[j][1]] = True
    ctx = torch.zeros(S, dtype=torch.bool, device=dev); ctx[a0:b1] = True
    junk = ctx & ~fm
    out = {k: [] for k in ("logZ_all", "logZ_ctx", "lse_e", "lse_j", "mass", "ctx_share")}
    states = {}
    for li, lay in enumerate(layers):
        if li in depths:
            states[li] = h[0, -1].detach().half().cpu()
        hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]
        nh = at.q_proj.weight.shape[0] // hd
        q = rope(at.q_proj(hn[:, -1:]).view(1, 1, nh, hd).transpose(1, 2), pe[0][:, -1:], pe[1][:, -1:])
        k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe)
        k = k.repeat_interleave(nh // k.shape[1], dim=1)
        sc = ((q @ k.transpose(-1, -2))[0, :, 0] / hd ** 0.5).float()      # [H, S]
        neg = torch.finfo(torch.float32).min
        out["logZ_all"].append(torch.logsumexp(sc, -1))
        out["logZ_ctx"].append(torch.logsumexp(sc.masked_fill(~ctx, neg), -1))
        out["lse_e"].append(torch.logsumexp(sc.masked_fill(~fm, neg), -1) if fm.any() else torch.full((nh,), float("nan"), device=dev))
        out["lse_j"].append(torch.logsumexp(sc.masked_fill(~junk, neg), -1) if junk.any() else torch.full((nh,), float("nan"), device=dev))
        wb = torch.softmax(sc.masked_fill(~ctx, neg), -1)
        out["mass"].append(wb[:, fm].sum(-1) if fm.any() else torch.zeros(nh, device=dev))
        out["ctx_share"].append(torch.softmax(sc, -1)[:, ctx].sum(-1))
        h = _call(lay, h, p, pe)
    states["fin"] = norm(h[:, -1])[0].detach().half().cpu()
    return {k: torch.stack(v).cpu() for k, v in out.items()}, states


def run(a):
    dev = "cuda"
    tok, model = load_text_model(a.model, device=dev)
    nL = len(model_parts(model)[1])
    depths = {a.layer, min(a.layer + 6, nL - 1)}
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), n_layers=nL, depths=sorted(depths)), indent=1))
    rows, feats, t0 = [], [], time.time()
    for reg in a.regimes.split("+"):
        for K in [int(k) for k in a.ks.split("+")]:
            D = REGIMES[reg](K)
            if D is None or D < 0:
                continue
            rng = random.Random(a.seed * 1000 + K)
            for i in range(a.n):
                sents, facts, q = story(K, D, rng)
                rec = build(tok, "qa1", q, sents, PROMPT)
                fA, stA = scan(model, rec["ids"], rec["blocks"], facts, dev, depths)
                item = dict(regime=reg, K=K, D=D, i=i, A=fA, states=stA)
                qc = control_question(sents, q)
                if qc:
                    recB = build(tok, "qa1", qc, sents, PROMPT)
                    assert recB["blocks"] == rec["blocks"], "control prompt moved the context"
                    item["B"], _ = scan(model, recB["ids"], recB["blocks"], facts, dev, set())
                if D > 0:
                    sc_ = [s_ for j, s_ in enumerate(sents) if j not in set(facts)]
                    recC = build(tok, "qa1", q, sc_, PROMPT)
                    item["C"], _ = scan(model, recC["ids"], recC["blocks"], [], dev, set())
                feats.append(item)
                rows.append(dict(regime=reg, K=K, D=D, i=i, tokens=len(rec["ids"]), has_ctrl=bool(qc),
                                 logZ_ctx_L_mean=float(fA["logZ_ctx"][a.layer].mean()),
                                 mass_L_max=float(fA["mass"][a.layer].max())))
            print(f"[{time.time()-t0:.0f}s] {reg} K={K:2d} D={D:2d} n={a.n} logZ_ctx(L) head-mean {sum(r['logZ_ctx_L_mean'] for r in rows[-a.n:])/a.n:.2f} "
                  f"best-head mass {sum(r['mass_L_max'] for r in rows[-a.n:])/a.n:.2f}", flush=True)
    with open(odir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    torch.save(feats, odir / "features.pt")
    print("->", odir)
    return analyze(odir)


# ----------------------------------------------------------------------------- analysis (CPU)
def _ridge_cv(X, y, K, lam=1.0, folds=5, select=None):
    """Standardized ridge on log K, 5-fold CV by story index; returns exact, within-1, exact(K<=16), R2."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64); K = np.asarray(K)
    n = len(y); idx = np.arange(n); pred = np.zeros(n)
    for f in range(folds):
        te = idx % folds == f; tr = ~te
        Xtr, Xte = X[tr], X[te]
        if select is not None:
            cols = select(tr); Xtr, Xte = Xtr[:, cols], Xte[:, cols]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ym = y[tr].mean()
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (y[tr] - ym))
        pred[te] = Xte @ w + ym
    Khat = np.rint(np.exp(pred)); err = np.abs(Khat - K)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    small = K <= 16
    return dict(exact=float((err == 0).mean()), within1=float((err <= 1).mean()),
                exact16=float((err[small] == 0).mean()), r2=float(r2))


def _extrap(X, y, K, lam=1.0, select=None):
    """Length generalization at the probe level: fit on K <= 16, test on K > 16 (24, 32, 48, 64)."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64); K = np.asarray(K)
    tr, te = K <= 16, K > 16
    if te.sum() == 0:
        return dict(exact=float("nan"), within1=float("nan"), relerr=float("nan"), med64=float("nan"))
    Xtr, Xte = X[tr], X[te]
    if select is not None:
        cols = select(tr); Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = y[tr].mean()
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - ym))
    pred = np.exp(Xte @ w + ym); Khat = np.rint(pred); Kt = K[te]
    m64 = float(np.median(Khat[Kt == Kt.max()]))
    return dict(exact=float((Khat == Kt).mean()), within1=float((np.abs(Khat - Kt) <= 1).mean()),
                relerr=float((np.abs(pred - Kt) / Kt).mean()), med64=m64)


def analyze(d):
    import numpy as np
    d = Path(d); cfg = json.load(open(d / "config.json")); L = cfg["layer"]
    feats = torch.load(d / "features.pt")
    lines = [f"# log Z probe — {cfg['model']} L{L} n={cfg['n']} ({d})"]
    for reg in sorted({f["regime"] for f in feats}):
        F = [f for f in feats if f["regime"] == reg]
        K = np.array([f["K"] for f in F]); y = np.log(K)
        A = {k: torch.stack([f["A"][k] for f in F]).numpy() for k in F[0]["A"]}      # [n, L, H]
        H = A["logZ_ctx"].shape[2]
        sets = {"logZ_ctx L": A["logZ_ctx"][:, L, :], "logZ_all L": A["logZ_all"][:, L, :],
                "logZ_ctx all layers": A["logZ_ctx"].reshape(len(F), -1)}
        if all("B" in f for f in F):
            B = torch.stack([f["B"]["logZ_ctx"] for f in F]).numpy()
            sets["dlogZ ctrl L"] = A["logZ_ctx"][:, L, :] - B[:, L, :]
            sets["dlogZ ctrl all layers"] = (A["logZ_ctx"] - B).reshape(len(F), -1)
        if all("C" in f for f in F):
            C = torch.stack([f["C"]["logZ_ctx"] for f in F]).numpy()
            sets["dlogZ counterfactual L"] = A["logZ_ctx"][:, L, :] - C[:, L, :]
        # selective heads (chosen inside each training fold): mean needle share at K>=12 > .8
        massL = A["mass"][:, L, :]
        def sel(tr):
            m = massL[tr & (K >= 12)].mean(0) if (tr & (K >= 12)).any() else massL[tr].mean(0)
            cols = np.where(m > .8)[0]
            return cols if len(cols) >= 1 else np.argsort(-m)[:3]
        res = {name: _ridge_cv(X, y, K) for name, X in sets.items()}
        res["logZ_ctx L, selective heads"] = _ridge_cv(A["logZ_ctx"][:, L, :], y, K, select=sel)
        for name, key in (("state L", L), ("state L+6", min(L + 6, cfg["n_layers"] - 1)), ("state final", "fin")):
            X = torch.stack([f["states"][key] for f in F]).float().numpy()
            res[name] = _ridge_cv(X, y, K, lam=10.0)
        lines.append(f"\n## {reg}  (n={len(F)}, K={sorted(set(K.tolist()))})")
        lines.append(f"{'feature set':34s} {'exact':>6s} {'exact≤16':>9s} {'within1':>8s} {'R2':>6s} | {'fit K≤16 → test K>16:':>22s} {'exact':>6s} {'within1':>8s} {'rel.err':>8s} {'med K̂@64':>9s}")
        Xstate = {name: torch.stack([f["states"][key] for f in F]).float().numpy()
                  for name, key in (("state L", L), ("state L+6", min(L + 6, cfg["n_layers"] - 1)), ("state final", "fin"))}
        allX = dict(sets); allX["logZ_ctx L, selective heads"] = A["logZ_ctx"][:, L, :]; allX.update(Xstate)
        for name, r in res.items():
            X = allX[name]; lam = 10.0 if name.startswith("state") else 1.0
            ex = _extrap(X, y, K, lam=lam, select=(sel if name.endswith("selective heads") else None))
            lines.append(f"{name:34s} {r['exact']:6.2f} {r['exact16']:9.2f} {r['within1']:8.2f} {r['r2']:6.2f} | {'':>22s} {ex['exact']:6.2f} {ex['within1']:8.2f} {ex['relerr']:8.2f} {ex['med64']:9.1f}")
        # N3: slope of mean logZ_ctx vs log K for the 3 most selective heads (all data) and head-mean
        m12 = massL[K >= 12].mean(0) if (K >= 12).any() else massL.mean(0)
        top = np.argsort(-m12)[:3]
        for label, series in (("3 most selective heads", A["logZ_ctx"][:, L, top].mean(1)), ("head-mean", A["logZ_ctx"][:, L, :].mean(1))):
            ks = sorted(set(K.tolist())); means = [series[K == k].mean() for k in ks]
            slope = np.polyfit(np.log(ks), means, 1)[0]
            lines.append(f"N3 slope d(mean logZ_ctx)/d(log K), {label}: {slope:.2f}   " + " ".join(f"K{k}:{m:.2f}" for k, m in zip(ks, means)))
        lines.append(f"selective heads at L (share>.8 at K>=12): {int((m12 > .8).sum())}/{H}; best-head share by K: "
                     + " ".join(f"K{k}:{massL[K == k].max(1).mean():.2f}" for k in sorted(set(K.tolist()))))
    txt = "\n".join(lines); print(txt); (d / "ANALYSIS.md").write_text(txt + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--ks", default="+".join(map(str, KS)))
    ap.add_argument("--regimes", default="H0+HF+HC"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/needles_logz"); ap.add_argument("--analyze", default="")
    a = ap.parse_args()
    return analyze(Path(a.analyze)) if a.analyze else run(a)


if __name__ == "__main__":
    sys.exit(main())
