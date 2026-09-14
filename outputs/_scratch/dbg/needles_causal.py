"""Bundle B (PREREG_AGG 'Added 2026-09-03', B1-B3): minimal-pair activation patching, share clamp at
the read layer, per-head direct logit attribution. HC regime for B1/B3 (fixed length); B2 in H0+HC."""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, ".")
from needles import story, PROMPT, REGIMES, VERBS  # noqa: E402
from eval_babilong import build  # noqa: E402
from modeling import load_text_model  # noqa: E402
from osq import model_parts, rope, _call  # noqa: E402

GROUPS = ("flip", "earlier_needles", "later_needles", "distractors", "tail", "answer")


def seg_len(tok, s, first):
    return len(tok(("" if first else " ") + s, add_special_tokens=False).input_ids)


def minimal_pair(K, D, rng, tok):
    sents, facts, q = story(K, D, rng)
    p, r = re.fullmatch(r"How many times did (\w+) go to the (\w+)\?", q).groups()
    dis = [j for j in range(len(sents)) if j not in facts]; rng.shuffle(dis)
    for j in dis:
        n0 = seg_len(tok, sents[j], j == 0); verbs = list(VERBS); rng.shuffle(verbs)
        for v in verbs:
            s2 = f"{p} {v} the {r}."
            if seg_len(tok, s2, j == 0) == n0:
                s_new = list(sents); s_new[j] = s2
                return sents, s_new, facts, j, q
    return None


class Cap:
    """Captures o_proj inputs (concatenated head outputs) at the last row, per layer."""
    def __init__(self, layers):
        self.buf = {}; self.hs = [lay.self_attn.o_proj.register_forward_hook(self._mk(i)) for i, lay in enumerate(layers)]
    def _mk(self, i):
        def hook(mod, inp, out): self.buf[i] = inp[0][0, -1].detach().float()
        return hook
    def close(self):
        for h in self.hs: h.remove()


@torch.no_grad()
def forward_cache(model, ids, dev, cap=None):
    embed, layers, norm, head, rotary = model_parts(model)
    x = torch.tensor([ids], device=dev); S = len(ids); p = torch.arange(S, device=dev)[None]
    h = embed(x); c, s = rotary(h, p); pe = (c.to(h.dtype), s.to(h.dtype)); hs = []
    for lay in layers:
        hs.append(h); h = _call(lay, h, p, pe)
    fin = h[:, -1]; logits = head(norm(fin)).float()[0]
    heads = dict(cap.buf) if cap is not None else None
    return hs, logits, pe, p, fin[0].float(), heads


@torch.no_grad()
def run_from(model, h, li, pe, p):
    _, layers, norm, head, _ = model_parts(model)
    for lay in layers[li:]: h = _call(lay, h, p, pe)
    return head(norm(h[:, -1])).float()[0]


@torch.no_grad()
def emit_intervened(model, tok, rec, facts, L, dev, mode, ref, max_new=3):
    """Greedy emission with the answer row's attention at layer L modified: mode none|clamp|amp."""
    embed, layers, norm, head, rotary = model_parts(model)
    ids = list(rec["ids"]); blocks = rec["blocks"]; toks = []
    for _ in range(max_new):
        S = len(ids); x = torch.tensor([ids], device=dev); p = torch.arange(S, device=dev)[None]
        h = embed(x); c, s = rotary(h, p); pe = (c.to(h.dtype), s.to(h.dtype))
        fm = torch.zeros(S, dtype=torch.bool, device=dev)
        for j in facts: fm[blocks[j][0]:blocks[j][1]] = True
        for li, lay in enumerate(layers):
            if li != L or mode == "none":
                h = _call(lay, h, p, pe); continue
            hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]; nh = at.q_proj.weight.shape[0] // hd
            q = rope(at.q_proj(hn[:, -1:]).view(1, 1, nh, hd).transpose(1, 2), pe[0][:, -1:], pe[1][:, -1:])[0, :, 0]
            k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe); g = nh // k.shape[1]; k = k.repeat_interleave(g, dim=1)[0]
            v = at.v_proj(hn).view(1, S, -1, hd).transpose(1, 2).repeat_interleave(g, dim=1)[0]
            w = torch.softmax((torch.einsum("hd,hsd->hs", q.float(), k.float()) / hd ** 0.5), -1)   # [H,S]
            m = w[:, fm].sum(-1); t = ref.to(dev) if mode == "clamp" else (1.5 * m).clamp(max=0.98)
            ok = (m > 1e-3) & (m < 0.999) & (t > 1e-3) & (t < 0.999)
            wn = w.clone(); wn[ok] = torch.where(fm[None], w[ok] * (t[ok] / m[ok])[:, None], w[ok] * ((1 - t[ok]) / (1 - m[ok]))[:, None])
            a = torch.einsum("hs,hsd->hd", wn.to(v.dtype), v).reshape(1, -1); a = at.o_proj(a)
            last = h[:, -1] + a; last = last + lay.mlp(lay.post_attention_layernorm(last))
            h = _call(lay, h, p, pe); h[:, -1] = last
        t_id = int(head(norm(h[:, -1])).float()[0].argmax()); s_ = tok.decode([t_id])
        if not s_.strip().isdigit(): break
        toks.append(t_id); ids.append(t_id)
    txt = tok.decode(toks); mm = re.search(r"\d+", txt)
    return int(mm.group()) if mm else None


@torch.no_grad()
def share_at_L(model, rec, facts, L, dev):
    embed, layers, norm, head, rotary = model_parts(model)
    ids = rec["ids"]; S = len(ids); x = torch.tensor([ids], device=dev); p = torch.arange(S, device=dev)[None]
    h = embed(x); c, s = rotary(h, p); pe = (c.to(h.dtype), s.to(h.dtype))
    for li in range(L): h = _call(layers[li], h, p, pe)
    lay = layers[L]; hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]; nh = at.q_proj.weight.shape[0] // hd
    q = rope(at.q_proj(hn[:, -1:]).view(1, 1, nh, hd).transpose(1, 2), pe[0][:, -1:], pe[1][:, -1:])[0, :, 0]
    k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe); k = k.repeat_interleave(nh // k.shape[1], dim=1)[0]
    w = torch.softmax(torch.einsum("hd,hsd->hs", q.float(), k.float()) / hd ** 0.5, -1)
    fm = torch.zeros(S, dtype=torch.bool, device=dev)
    for j in facts: fm[rec["blocks"][j][0]:rec["blocks"][j][1]] = True
    return w[:, fm].sum(-1).cpu()


def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev)
    embed, layers, norm, head, rotary = model_parts(model); nL = len(layers); L = a.layer
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    odir.mkdir(parents=True, exist_ok=True); (odir / "config.json").write_text(json.dumps(dict(vars(a), n_layers=nL), indent=1))
    t0 = time.time(); pairs = []
    # ---------------- B1 + B3: minimal pairs in HC
    Wo = [lay.self_attn.o_proj.weight for lay in layers]; hd = model.config.hidden_size // model.config.num_attention_heads
    gamma = norm.weight.float(); WU = head.weight
    for K in [int(k) for k in a.ks.split("+")]:
        D = REGIMES["HC"](K); rng = random.Random(a.seed * 1000 + K); made = 0; tries = 0
        while made < a.n and tries < 4 * a.n:
            tries += 1; mp = minimal_pair(K, D, rng, tok)
            if mp is None: continue
            sA, sB, facts, jf, q = mp
            recA, recB = build(tok, "qa1", q, sA, PROMPT), build(tok, "qa1", q, sB, PROMPT)
            assert recA["seq"] == recB["seq"] and recA["blocks"] == recB["blocks"]
            capA = Cap(layers); hsA, lgA, pe, p, finA, headsA = forward_cache(model, recA["ids"], dev, capA); capA.close()
            capB = Cap(layers); hsB, lgB, _, _, finB, headsB = forward_cache(model, recB["ids"], dev, capB); capB.close()
            tA, tB = tok(str(K), add_special_tokens=False).input_ids[0], tok(str(K + 1), add_special_tokens=False).input_ids[0]
            dA, dB = float(lgA[tB] - lgA[tA]), float(lgB[tB] - lgB[tA])
            S = recA["seq"]; bl = recA["blocks"]; idx = {}
            idx["flip"] = list(range(*bl[jf]))
            idx["earlier_needles"] = [t for j in facts if j < jf for t in range(*bl[j])]
            idx["later_needles"] = [t for j in facts if j > jf for t in range(*bl[j])]
            idx["distractors"] = [t for j in range(len(bl)) if j not in set(facts) and j != jf for t in range(*bl[j])]
            idx["tail"] = list(range(recA["fin"], S - 1)); idx["answer"] = [S - 1]
            eff = {}
            for gname in GROUPS:
                ii = torch.tensor(idx[gname], device=dev) if idx[gname] else None; row = []
                for li in range(nL):
                    if ii is None: row.append(float("nan")); continue
                    hp = hsA[li].clone(); hp[:, ii] = hsB[li][:, ii]
                    lg = run_from(model, hp, li, pe, p); row.append((float(lg[tB] - lg[tA]) - dA) / (dB - dA) if abs(dB - dA) > 1e-3 else float("nan"))
                eff[gname] = row
            # B3: per-head DLA of the (K+1 - K) direction, change B - A
            u = (WU[tB] - WU[tA]).float(); rms = 0.5 * (finA.pow(2).mean().sqrt() + finB.pow(2).mean().sqrt()) + 1e-6
            dla = torch.zeros(nL, model.config.num_attention_heads)
            for li in range(nL):
                dlt = (headsB[li] - headsA[li]); W = Wo[li].float()
                for hh in range(dla.shape[1]):
                    contrib = W[:, hh * hd:(hh + 1) * hd] @ dlt[hh * hd:(hh + 1) * hd]
                    dla[li, hh] = float(((contrib * gamma / rms) @ u).item())
            pairs.append(dict(K=K, i=made, flip_block=jf, n_blocks=len(bl), dA=dA, dB=dB, predA=tok.decode([int(lgA.argmax())]), predB=tok.decode([int(lgB.argmax())]), eff=eff, dla=dla)); made += 1
            del hsA, hsB
        fl = [pr for pr in pairs if pr["K"] == K]
        def m_at(g, li): 
            import statistics; v = [pr["eff"][g][li] for pr in fl if pr["eff"][g][li] == pr["eff"][g][li]]; return statistics.mean(v) if v else float("nan")
        print(f"[{time.time()-t0:.0f}s] B1 K={K} pairs {len(fl)} dA {sum(p_['dA'] for p_ in fl)/len(fl):.2f} dB {sum(p_['dB'] for p_ in fl)/len(fl):.2f} | effect at L{L}: " + " ".join(f"{g}:{m_at(g, L):.2f}" for g in GROUPS) + f" | at L{nL-1}: " + " ".join(f"{g}:{m_at(g, nL-1):.2f}" for g in GROUPS), flush=True)
    torch.save(pairs, odir / "pairs.pt")
    # ---------------- B2: share clamp
    clamp = []
    for reg in a.regimes.split("+"):
        Kref = 4; D = REGIMES[reg](Kref); rng = random.Random(a.seed * 1000 + Kref); shares = []
        for i in range(a.n):
            sents, facts, q = story(Kref, D, rng); shares.append(share_at_L(model, build(tok, "qa1", q, sents, PROMPT), facts, L, dev))
        ref = torch.stack(shares).mean(0)
        for K in [int(k) for k in a.ks_clamp.split("+")]:
            D = REGIMES[reg](K)
            if D is None or D < 0: continue
            rng = random.Random(a.seed * 1000 + K); res = {m: [] for m in ("none", "clamp", "amp")}
            for i in range(a.n):
                sents, facts, q = story(K, D, rng); rec = build(tok, "qa1", q, sents, PROMPT)
                for mode in res: res[mode].append(emit_intervened(model, tok, rec, facts, L, dev, mode, ref))
            clamp.append(dict(regime=reg, K=K, ref=ref.tolist(), **res))
            import statistics
            def med(v): v = [x for x in v if x is not None]; return statistics.median(v) if v else float("nan")
            print(f"[{time.time()-t0:.0f}s] B2 {reg} K={K:2d} median emitted none {med(res['none']):.0f} clamp {med(res['clamp']):.0f} amp {med(res['amp']):.0f} | exact none {sum(x == K for x in res['none'])/a.n:.2f} clamp {sum(x == K for x in res['clamp'])/a.n:.2f} amp {sum(x == K for x in res['amp'])/a.n:.2f}", flush=True)
    torch.save(clamp, odir / "clamp.pt"); print("->", odir); return analyze(odir)


def analyze(d):
    import numpy as np, statistics
    d = Path(d); cfg = json.load(open(d / "config.json")); L = cfg["layer"]; nL = cfg["n_layers"]
    pairs = torch.load(d / "pairs.pt"); clamp = torch.load(d / "clamp.pt")
    out = [f"# Bundle B — {cfg['model']} L{L} ({d})", f"pairs: {len(pairs)}; digit log-odds dA/dB mean: {np.mean([p['dA'] for p in pairs]):.2f}/{np.mean([p['dB'] for p in pairs]):.2f}; predA==K: {np.mean([p['predA'].strip()==str(p['K']) for p in pairs]):.2f}, predB==K+1: {np.mean([p['predB'].strip()==str(p['K']+1) for p in pairs]):.2f}"]
    E = {g: np.array([p["eff"][g] for p in pairs], dtype=float) for g in GROUPS}       # [n, L]
    out.append("\nB1 mean patch effect (fraction of full K→K+1 log-odds change), by layer:")
    out.append("layer " + " ".join(f"{g[:9]:>9s}" for g in GROUPS))
    for li in range(nL):
        out.append(f"{li:5d} " + " ".join(f"{np.nanmean(E[g][:, li]):9.2f}" for g in GROUPS))
    for K in sorted({p["K"] for p in pairs}):
        sel = np.array([p["K"] == K for p in pairs])
        out.append(f"B1 K={K}: max over layers of mean effect: " + " ".join(f"{g}:{np.nanmax(np.nanmean(E[g][sel], 0)):.2f}@L{int(np.nanargmax(np.nanmean(E[g][sel], 0)))}" for g in GROUPS))
    D_ = torch.stack([p["dla"] for p in pairs]).mean(0).numpy(); flat = np.argsort(-np.abs(D_).ravel())[:8]
    out.append("\nB3 DLA of the (K+1−K) digit direction, top |heads| (layer,head:value): " + " ".join(f"L{i // D_.shape[1]}H{i % D_.shape[1]}:{D_.ravel()[i]:+.2f}" for i in flat))
    out.append("B3 per-layer sum of DLA: " + " ".join(f"L{li}:{D_[li].sum():+.2f}" for li in range(nL)))
    out.append("\nB2 share clamp — median emitted count (exact) by K:")
    for reg in sorted({c["regime"] for c in clamp}):
        for mode in ("none", "clamp", "amp"):
            row = []
            for c in [c for c in clamp if c["regime"] == reg]:
                v = [x for x in c[mode] if x is not None]; row.append(f"K{c['K']}:{statistics.median(v) if v else float('nan'):.0f}({np.mean([x == c['K'] for x in c[mode]]):.2f})")
            out.append(f"{reg} {mode:5s}: " + " ".join(row))
    txt = "\n".join(out); print(txt); (d / "ANALYSIS.md").write_text(txt + "\n"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--ks", default="2+4+6+8"); ap.add_argument("--ks-clamp", default="1+2+4+6+8+12+16+24+32")
    ap.add_argument("--regimes", default="H0+HC"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/needles_causal"); ap.add_argument("--analyze", default="")
    a = ap.parse_args(); return analyze(Path(a.analyze)) if a.analyze else run(a)


if __name__ == "__main__":
    sys.exit(main())
