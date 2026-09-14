"""Bundle A (PREREG_AGG 'Added 2026-09-03', A1-A4): per-token running-count probes, individuation
variants, fp32 emission, score decomposition at the read layer. Same stories as needles.py."""
import argparse, json, math, os, random, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, ".")
from needles import story, PROMPT, REGIMES  # noqa: E402
from eval_babilong import build, decode  # noqa: E402
from modeling import load_text_model  # noqa: E402
from osq import model_parts, rope, _call  # noqa: E402

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def variant(sents, kind, rng):
    if kind == "base":
        return sents
    if kind == "number":
        return [f"{i + 1}. {s}" for i, s in enumerate(sents)]
    if kind == "time":
        t0 = rng.randint(6 * 60, 9 * 60); ts = sorted(rng.sample(range(t0, t0 + 6 * 60), len(sents)))
        return [f"At {t // 60:02d}:{t % 60:02d}, {s[0].lower() + s[1:]}" for t, s in zip(ts, sents)]
    raise ValueError(kind)


def emit(model, tok, rec, eos, dev):
    text = tok.decode(decode(model, rec["ids"], list(range(rec["seq"])), eos, dev, 4))
    m = re.search(r"\d+", text)
    return (int(m.group()) if m else None), text


@torch.no_grad()
def pass_states(model, rec, facts, L, dev, n_dis=8, rng=None):
    """Manual layer loop: states at depths {L//2, L, L+6, fin} for needle/distractor/tail/answer
    tokens; at L, per head: |q|, mean|k|, mean cos, mean score over needle and junk tokens."""
    embed, layers, norm, head, rotary = model_parts(model)
    nL = len(layers); depths = sorted({L // 2, L, min(L + 6, nL - 1)})
    ids = rec["ids"]; S = len(ids); blocks = rec["blocks"]
    x = torch.tensor([ids], device=dev); p = torch.arange(S, device=dev)[None]
    h = embed(x); c, s = rotary(h, p); pe = (c.to(h.dtype), s.to(h.dtype))
    fset = set(facts); dis = [j for j in range(len(blocks)) if j not in fset]
    dis_s = sorted(rng.sample(dis, min(n_dis, len(dis)))) if dis else []
    toks = [(blocks[j][1] - 1, "needle", j) for j in facts] + [(blocks[j][1] - 1, "dis", j) for j in dis_s] \
        + [(S - 8, "tail", -1), (S - 1, "answer", -1)]
    pos = torch.tensor([t[0] for t in toks], device=dev)
    states, dec = {}, {}
    fm = torch.zeros(S, dtype=torch.bool, device=dev)
    for j in facts: fm[blocks[j][0]:blocks[j][1]] = True
    ctx = torch.zeros(S, dtype=torch.bool, device=dev); ctx[blocks[0][0]:blocks[-1][1]] = True
    junk = ctx & ~fm
    for li, lay in enumerate(layers):
        if li in depths: states[li] = h[0, pos].detach().half().cpu()
        if li == L:
            hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]; nh = at.q_proj.weight.shape[0] // hd
            q = rope(at.q_proj(hn[:, -1:]).view(1, 1, nh, hd).transpose(1, 2), pe[0][:, -1:], pe[1][:, -1:])[0, :, 0].float()  # [H,hd]
            k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe); k = k.repeat_interleave(nh // k.shape[1], dim=1)[0].float()  # [H,S,hd]
            sc = torch.einsum("hd,hsd->hs", q, k) / hd ** 0.5
            qn = q.norm(dim=-1); kn = k.norm(dim=-1); cos = sc * hd ** 0.5 / (qn[:, None] * kn + 1e-6)
            def agg(mask): return dict(k=float(kn[:, mask].mean()), cos=float(cos[:, mask].mean()), s=float(sc[:, mask].mean())) if mask.any() else dict(k=float("nan"), cos=float("nan"), s=float("nan"))
            dec = dict(q=float(qn.mean()), needle=agg(fm), junk=agg(junk),
                       q_h=qn.cpu(), s_needle_h=(sc[:, fm].mean(-1).cpu() if fm.any() else None), cos_needle_h=(cos[:, fm].mean(-1).cpu() if fm.any() else None),
                       k_needle_h=(kn[:, fm].mean(-1).cpu() if fm.any() else None), s_junk_h=(sc[:, junk].mean(-1).cpu() if junk.any() else None))
        h = _call(lay, h, p, pe)
    states["fin"] = norm(h[0, pos]).detach().half().cpu()
    # labels: rank_so_far for each token (needles before or at it)
    labels = []
    for (t, kind, j) in toks:
        before = sum(1 for f in facts if f <= j) if kind in ("needle", "dis") else len(facts)
        labels.append(dict(pos=t, kind=kind, block=j, rank=before))
    return states, labels, dec


def run(a):
    dev = "cuda"
    tok, model = load_text_model(a.model, device=dev)
    eos = model.generation_config.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    odir.mkdir(parents=True, exist_ok=True); (odir / "config.json").write_text(json.dumps(vars(a), indent=1))
    items, t0 = [], time.time()
    for reg in a.regimes.split("+"):
        for K in [int(k) for k in a.ks.split("+")]:
            D = REGIMES[reg](K)
            if D is None or D < 0: continue
            rng = random.Random(a.seed * 1000 + K); rv = random.Random(a.seed * 7 + K)
            for i in range(a.n):
                sents, facts, q = story(K, D, rng)
                item = dict(regime=reg, K=K, D=D, i=i, emit={}, out={})
                for kind in a.variants.split("+"):
                    rec = build(tok, "qa1", q, variant(sents, kind, rv), PROMPT)
                    pred, text = emit(model, tok, rec, eos, dev); item["emit"][kind] = pred; item["out"][kind] = text[:40]
                    if kind == "base":
                        st, lab, dec = pass_states(model, rec, facts, a.layer, dev, rng=rv)
                        item.update(states=st, labels=lab, dec=dec)
                    else:
                        st, lab, _ = pass_states(model, rec, facts, a.layer, dev, n_dis=0, rng=rv)
                        item[f"answer_state_{kind}"] = {k: v[-1] for k, v in st.items()}
                items.append(item)
            def acc(kind): return sum(int(it["emit"].get(kind) == K) for it in items[-a.n:]) / a.n
            print(f"[{time.time()-t0:.0f}s] {reg} K={K:2d} " + " ".join(f"{k}:{acc(k):.2f}" for k in a.variants.split("+"))
                  + f" | |q| {sum(it['dec']['q'] for it in items[-a.n:])/a.n:.2f} s_needle {sum(it['dec']['needle']['s'] for it in items[-a.n:])/a.n:.2f} cos_needle {sum(it['dec']['needle']['cos'] for it in items[-a.n:])/a.n:.3f}", flush=True)
    torch.save(items, odir / "items.pt")
    if a.fp32:
        del model; torch.cuda.empty_cache()
        tok, model = load_text_model(a.model, device=dev, dtype=torch.float32)
        for reg in a.regimes.split("+"):
            for K in [int(k) for k in a.ks.split("+")]:
                D = REGIMES[reg](K)
                if D is None or D < 0: continue
                rng = random.Random(a.seed * 1000 + K); hits = 0
                for i in range(a.n):
                    sents, facts, q = story(K, D, rng); rec = build(tok, "qa1", q, sents, PROMPT)
                    pred, _ = emit(model, tok, rec, eos, dev)
                    it = next(x for x in items if x["regime"] == reg and x["K"] == K and x["i"] == i); it["emit"]["fp32"] = pred; hits += int(pred == K)
                print(f"[{time.time()-t0:.0f}s] fp32 {reg} K={K:2d} acc {hits/a.n:.2f}", flush=True)
        torch.save(items, odir / "items.pt")
    print("->", odir); return analyze(odir)


def _ridge(X, y, groups, lam=10.0, folds=5):
    import numpy as np
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64); g = np.asarray(groups); pred = np.zeros(len(y))
    for f in range(folds):
        te = g % folds == f; tr = ~te
        if te.sum() == 0 or tr.sum() < 5: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd; ym = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(X.shape[1]), Xtr.T @ (y[tr] - ym)); pred[te] = Xte @ w + ym
    return pred


def analyze(d):
    import numpy as np
    d = Path(d); cfg = json.load(open(d / "config.json")); L = cfg["layer"]; items = torch.load(d / "items.pt")
    out = [f"# Bundle A — {cfg['model']} L{L} n={cfg['n']} ({d})"]
    for reg in sorted({it["regime"] for it in items}):
        R = [it for it in items if it["regime"] == reg]; ks = sorted({it["K"] for it in R})
        out.append(f"\n## {reg}")
        # A2 / A3 emission
        kinds = [k for k in cfg["variants"].split("+")] + (["fp32"] if cfg["fp32"] else [])
        out.append("emission exact by K: " + " | ".join(f"{k}: " + " ".join(f"{K}:{np.mean([it['emit'].get(k) == K for it in R if it['K'] == K]):.2f}" for K in ks) for k in kinds))
        # A1 probes per depth
        depths = list(R[0]["states"].keys())
        for dep in depths:
            X, y, g, kind, isl = [], [], [], [], []
            for si, it in enumerate(R):
                st = it["states"][dep].float().numpy()
                lastn = max([l["block"] for l in it["labels"] if l["kind"] == "needle"], default=-1)
                for row, lab in zip(st, it["labels"]):
                    X.append(row); y.append(lab["rank"] if lab["kind"] in ("needle", "dis") else it["K"]); g.append(si); kind.append(lab["kind"]); isl.append(lab["kind"] == "needle" and lab["block"] == lastn)
            X, y, g, kind, isl = np.array(X), np.array(y), np.array(g), np.array(kind), np.array(isl)
            res = []
            for name, m in (("needle→rank", kind == "needle"), ("distractor→needles before", kind == "dis"), ("last needle→K", isl), ("tail→K", kind == "tail"), ("answer→K", kind == "answer")):
                if m.sum() < 20: continue
                pr = _ridge(X[m], np.log1p(y[m]), g[m]); yh = np.rint(np.expm1(pr)); yt = y[m]
                small = yt <= 16
                res.append(f"{name}: exact {np.mean(yh == yt):.2f} (≤16: {np.mean(yh[small] == yt[small]):.2f}) within1 {np.mean(np.abs(yh - yt) <= 1):.2f} n={m.sum()}")
            out.append(f"A1 depth {dep}: " + " | ".join(res))
        # A4 decomposition (head-mean) by K
        for key, lab in (("q", "|q|"), (("needle", "s"), "s_needle"), (("junk", "s"), "s_junk"), (("needle", "cos"), "cos_needle"), (("needle", "k"), "|k_needle|"), (("junk", "cos"), "cos_junk")):
            vals = []
            for K in ks:
                v = [it["dec"][key] if isinstance(key, str) else it["dec"][key[0]][key[1]] for it in R if it["K"] == K]
                vals.append(np.nanmean(v))
            out.append(f"A4 {lab:10s} by K: " + " ".join(f"{K}:{v:.2f}" for K, v in zip(ks, vals)))
    txt = "\n".join(out); print(txt); (d / "ANALYSIS.md").write_text(txt + "\n"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--ks", default="+".join(map(str, KS)))
    ap.add_argument("--regimes", default="H0+HC"); ap.add_argument("--variants", default="base+time+number")
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/needles_diag"); ap.add_argument("--analyze", default="")
    a = ap.parse_args(); return analyze(Path(a.analyze)) if a.analyze else run(a)


if __name__ == "__main__":
    sys.exit(main())
