"""Bundle E (PREREG_AGG 'Added 2026-09-03', E1-E5): competitor-type factorial, score structure by class,
test-time sharpening, share->number transfer function; --leakage: CPU correlation on the original runs."""
import argparse, json, os, random, re, sys, time, glob
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from needles import PROMPT, REGIMES, PEOPLE, ROOMS, VERBS, story, KS as KS0  # noqa: E402
from eval_babilong import build  # noqa: E402
from modeling import load_text_model  # noqa: E402
from osq import model_parts, rope, _call  # noqa: E402
from needles_band import prep, tail_replace  # noqa: E402

BAD = re.compile(r"\b(" + "|".join(PEOPLE + ROOMS + [v.split()[0] for v in VERBS]) + r")\b", re.I)


def load_pool(n_rows=400):
    import pandas as pd
    df = pd.read_parquet(sorted(glob.glob("data/babilong/1k/4k/qa1*.parquet"))[0])
    txt = " ".join(df["input"].astype(str).tolist()[:n_rows]); pool = []
    for s in re.split(r"(?<=[.!?])\s+", txt):
        s = " ".join(s.split())
        if 6 <= len(s.split()) <= 14 and s[0].isupper() and s.endswith(".") and not BAD.search(s) and not re.search(r"[\d\"“”_\[\]()]", s) and not s.endswith(("Mr.", "Mrs.", "Dr.")):
            pool.append(s)
    return pool


def story_typed(K, D, kind, rng, pool):
    p, r = rng.choice(PEOPLE), rng.choice(ROOMS); items = [(f"{p} {rng.choice(VERBS)} the {r}.", "full") for _ in range(K)]
    while len(items) < K + D:
        if kind == "HU": items.append((rng.choice(pool), "filler")); continue
        if kind == "HP": pp, rr = p, rng.choice([x for x in ROOMS if x != r])
        elif kind == "HR": pp, rr = rng.choice([x for x in PEOPLE if x != p]), r
        elif kind == "HN": pp, rr = rng.choice([x for x in PEOPLE if x != p]), rng.choice([x for x in ROOMS if x != r])
        else:
            pp, rr = rng.choice(PEOPLE), rng.choice(ROOMS)
            if (pp, rr) == (p, r): continue
        cls = "half_p" if pp == p else ("half_r" if rr == r else "no")
        items.append((f"{pp} {rng.choice(VERBS)} the {rr}.", cls))
    rng.shuffle(items)
    return [s for s, _ in items], [j for j, (_, c) in enumerate(items) if c == "full"], [c for _, c in items], f"How many times did {p} go to the {r}?"


def tail_scores(lay, h, pe, fin):
    hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]; nh = at.q_proj.weight.shape[0] // hd; S = h.shape[1]
    q = rope(at.q_proj(hn[:, fin:]).view(1, S - fin, nh, hd).transpose(1, 2), pe[0][:, fin:], pe[1][:, fin:])[0]
    k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe); g = nh // k.shape[1]; k = k.repeat_interleave(g, dim=1)[0]
    v = at.v_proj(hn).view(1, S, -1, hd).transpose(1, 2).repeat_interleave(g, dim=1)[0]
    sc = torch.einsum("htd,hsd->hts", q.float(), k.float()) / hd ** 0.5
    T = S - fin; causal = torch.arange(S, device=h.device)[None, :] > (fin + torch.arange(T, device=h.device))[:, None]
    return sc.masked_fill(causal[None], float("-inf")), v, at


@torch.no_grad()
def emit(model, tok, rec, facts, dev, mode, layers_set, beta=1.0, mult=1.0, max_new=3):
    """mode none | sharp (logits*beta on tail rows at layers_set) | scale (needle share x mult at layers_set)."""
    embed, layers, norm, head, rotary = model_parts(model); ids = list(rec["ids"]); bl = rec["blocks"]; fin = rec["fin"]; toks = []
    for _ in range(max_new):
        S = len(ids); h, p, pe = prep(model, ids, dev); fm = torch.zeros(S, dtype=torch.bool, device=dev)
        for j in facts: fm[bl[j][0]:bl[j][1]] = True
        for li, lay in enumerate(layers):
            h_out = _call(lay, h, p, pe)
            if mode != "none" and li in layers_set:
                sc, v, at = tail_scores(lay, h, pe, fin)
                if mode == "sharp": w_new = torch.softmax(beta * sc, -1)
                else:
                    w = torch.softmax(sc, -1); m = w[:, :, fm].sum(-1); t = (mult * m).clamp(max=0.98)
                    ok = (m > 1e-4) & (m < 0.999); sn = torch.where(ok, t / m, torch.ones_like(m)); sj = torch.where(ok, (1 - t) / (1 - m), torch.ones_like(m))
                    w_new = torch.where(fm[None, None], w * sn[..., None], w * sj[..., None])
                h_out = tail_replace(lay, h, h_out, fin, w_new, v, at)
            h = h_out
        t_id = int(head(norm(h[:, -1])).float()[0].argmax()); s_ = tok.decode([t_id])
        if not s_.strip().isdigit(): break
        toks.append(t_id); ids.append(t_id)
    txt = tok.decode(toks); mm = re.search(r"\d+", txt); return int(mm.group()) if mm else None


@torch.no_grad()
def class_scores(model, rec, classes, dev, band, T0=8):
    """Per band layer, per head: mean over last T0 tail rows of the mean sentence logit for each class."""
    embed, layers, norm, head, rotary = model_parts(model); h, p, pe = prep(model, rec["ids"], dev); bl = rec["blocks"]; fin = rec["fin"]
    out = {}
    for li, lay in enumerate(layers):
        if li in band:
            sc, _, _ = tail_scores(lay, h, pe, fin); sc = sc[:, -T0:]                                  # [H,T0,S]
            sent = torch.stack([sc[:, :, a_:b_].mean(-1) for a_, b_ in bl], -1).mean(1)             # [H,F]
            res = {}
            for c in ("full", "half_p", "half_r", "no", "filler"):
                idx = [j for j, cc in enumerate(classes) if cc == c]
                res[c] = sent[:, idx].mean(-1).cpu() if idx else None
            out[li] = res
        h = _call(lay, h, p, pe)
    return out


def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev); pool = load_pool()
    b0, b1 = [int(x) for x in a.band.split("+")]; band = set(range(b0, b1 + 1)); nL = len(model_parts(model)[1])
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), n_layers=nL, pool=len(pool)), indent=1)); t0 = time.time(); rows = []; scores = []
    ks = [int(k) for k in a.ks.split("+")]
    for kind in a.kinds.split("+"):
        for K in ks:
            D = 64 - K; rng = random.Random(a.seed * 1000 + K + hash(kind) % 1000)
            for i in range(a.n):
                sents, facts, classes, q = story_typed(K, D, kind, rng, pool); rec = build(tok, "qa1", q, sents, PROMPT)
                row = dict(kind=kind, K=K, i=i, n_half=sum(c in ("half_p", "half_r") for c in classes), pred={})
                row["pred"]["none"] = emit(model, tok, rec, facts, dev, "none", set())
                if kind in a.sharp_kinds.split("+"):
                    for beta in (1.5, 2.0, 3.0): row["pred"][f"sharp{beta}_band"] = emit(model, tok, rec, facts, dev, "sharp", band, beta=beta)
                    row["pred"]["sharp2.0_all"] = emit(model, tok, rec, facts, dev, "sharp", set(range(nL)), beta=2.0)
                if kind == "HC" and K in (4, 8):
                    for mult in (0.5, 2.0, 3.0, 4.0): row["pred"][f"scale{mult}"] = emit(model, tok, rec, facts, dev, "scale", set(range(nL)), mult=mult)
                if kind in ("HC", "HN", "HP", "HR", "HU") and K in (4, 8): scores.append(dict(kind=kind, K=K, cs=class_scores(model, rec, classes, dev, band)))
                rows.append(row)
            cell = [r_ for r_ in rows if r_["kind"] == kind and r_["K"] == K]; import statistics
            def med(key): v = [r_["pred"].get(key) for r_ in cell if r_["pred"].get(key) is not None]; return statistics.median(v) if v else float("nan")
            print(f"[{time.time()-t0:.0f}s] {kind} K={K:2d} exact none {sum(r_['pred']['none'] == K for r_ in cell)/len(cell):.2f} median {med('none'):.0f}" + ("" if kind not in a.sharp_kinds.split("+") else " | sharp1.5/2/3/2all exact " + " ".join(f"{sum(r_['pred'][k_] == K for r_ in cell)/len(cell):.2f}" for k_ in ("sharp1.5_band", "sharp2.0_band", "sharp3.0_band", "sharp2.0_all"))), flush=True)
    torch.save(dict(rows=rows, scores=scores), odir / "types.pt"); print("->", odir); return analyze(odir)


def analyze(d):
    import numpy as np, statistics
    d = Path(d); cfg = json.load(open(d / "config.json")); P = torch.load(d / "types.pt"); rows, scores = P["rows"], P["scores"]
    out = [f"# Bundle E — {cfg['model']} band {cfg['band']} n={cfg['n']} ({d})"]; ks = sorted({r["K"] for r in rows}); kinds = [k for k in cfg["kinds"].split("+")]
    def cell(kind, K, key="none"): return [r["pred"].get(key) for r in rows if r["kind"] == kind and r["K"] == K]
    def ex(v, K): return np.mean([x == K for x in v]); 
    def med(v): v = [x for x in v if x is not None]; return statistics.median(v) if v else float("nan")
    out.append("\nE1 exact by competitor type × K:"); out.append("kind  " + " ".join(f"K{K:>3d}" for K in ks))
    for kind in kinds: out.append(f"{kind:5s} " + " ".join(f"{ex(cell(kind, K), K):4.2f}" for K in ks))
    out.append("E1 median emitted by type × K:")
    for kind in kinds: out.append(f"{kind:5s} " + " ".join(f"{med(cell(kind, K)):4.0f}" for K in ks))
    out.append("\nE2 score structure at the band (head-mean logits; fraction = (class − no)/(full − no)):")
    for kind in ("HC", "HN", "HP", "HR", "HU"):
        S_ = [s for s in scores if s["kind"] == kind]
        if not S_: continue
        agg = {}
        for s in S_:
            for li, res in s["cs"].items():
                for c, v in res.items():
                    if v is not None: agg.setdefault(c, []).append(v)
        means = {c: torch.stack(v).mean().item() for c, v in agg.items()}
        base = means.get("no", means.get("filler")); full = means.get("full")
        frac = {c: (means[c] - base) / (full - base) if full is not None and base is not None and abs(full - base) > 1e-6 else float("nan") for c in means}
        out.append(f"{kind}: " + " ".join(f"{c}:{means[c]:.2f}(frac {frac[c]:.2f})" for c in means))
    out.append("\nE3 sharpening (exact by β × K), HC and HN:")
    for kind in [k for k in cfg["sharp_kinds"].split("+") if k in kinds]:
        for key in ("none", "sharp1.5_band", "sharp2.0_band", "sharp3.0_band", "sharp2.0_all"):
            out.append(f"{kind} {key:14s} " + " ".join(f"{ex(cell(kind, K, key), K):4.2f}" for K in ks) + "  | median " + " ".join(f"{med(cell(kind, K, key)):3.0f}" for K in ks))
    out.append("\nE4 transfer function (HC, needle share × m at all layers): median emitted (exact)")
    for K in (4, 8):
        out.append(f"K={K}: " + " ".join(f"m={m}:{med(cell('HC', K, key)):.0f}({ex(cell('HC', K, key), K):.2f})" for m, key in ((0.5, "scale0.5"), (1, "none"), (2, "scale2.0"), (3, "scale3.0"), (4, "scale4.0"))))
    txt = "\n".join(out); print(txt); (d / "ANALYSIS.md").write_text(txt + "\n"); return 0


def leakage():
    """E5 (CPU): regenerate the original needles stories (same seed/order), count half-matches, correlate with the emitted count."""
    import numpy as np
    from scipy.stats import spearmanr
    out = ["# E5 leakage — within-K Spearman(#half-match distractors, emitted count), original needles runs"]
    for m in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"):
        d = sorted(glob.glob(f"outputs/needles/{m}/*/"))[-1]; cfg = json.load(open(d + "config.json")); rows = [json.loads(l) for l in open(d + "results.jsonl") if l.strip()]
        pred = {(r["regime"], r["K"], r["i"]): r["pred"] for r in rows}; rng = random.Random(cfg["seed"]); ks = [int(k) for k in cfg["ks"].split("+")]
        recs = []
        for reg in cfg["regimes"].split("+"):
            for K in ks:
                D = REGIMES[reg](K)
                if D is None or D < 0: continue
                for i in range(cfg["n"]):
                    if reg == "DUP": sents, facts, q = story(K // 2, D, rng); sents = sents * 2
                    else: sents, facts, q = story(K, D, rng)
                    p, r_ = re.fullmatch(r"How many times did (\w+) go to the (\w+)\?", q).groups()
                    hp = sum(1 for j, s in enumerate(sents) if j not in set(facts) and s.startswith(p + " ")); hr = sum(1 for j, s in enumerate(sents) if j not in set(facts) and s.endswith("the " + r_ + ".") and not s.startswith(p + " "))
                    recs.append(dict(regime=reg, K=K, i=i, hp=hp, hr=hr, pred=pred.get((reg, K, i))))
        for reg in ("HF", "HC"):
            line = []
            for K in [k for k in ks if k <= 8]:
                sub = [x for x in recs if x["regime"] == reg and x["K"] == K and x["pred"] is not None]
                if len(sub) > 8:
                    rho = spearmanr([x["hp"] + x["hr"] for x in sub], [x["pred"] for x in sub]).correlation
                    rp = spearmanr([x["hp"] for x in sub], [x["pred"] for x in sub]).correlation; rr = spearmanr([x["hr"] for x in sub], [x["pred"] for x in sub]).correlation
                    line.append(f"K{K}:{rho:+.2f}(p{rp:+.2f},r{rr:+.2f})")
            vals = [float(l.split(":")[1].split("(")[0]) for l in line]
            out.append(f"{m[:10]} {reg}: " + " ".join(line) + f"  | mean {np.mean(vals):+.2f}")
    txt = "\n".join(out); print(txt); Path("outputs/_scratch/dbg/E5_leakage.md").write_text(txt + "\n"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--layer", type=int, default=27); ap.add_argument("--band", default="18+24")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--ks", default="1+2+4+6+8+12+16"); ap.add_argument("--kinds", default="HC+HN+HP+HR+HU"); ap.add_argument("--sharp-kinds", default="HC+HN")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--output", default="outputs/needles_types"); ap.add_argument("--analyze", default=""); ap.add_argument("--leakage", action="store_true")
    a = ap.parse_args()
    if a.leakage: return leakage()
    return analyze(Path(a.analyze)) if a.analyze else run(a)


if __name__ == "__main__":
    sys.exit(main())
