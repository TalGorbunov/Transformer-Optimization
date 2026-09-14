"""Bundle C (PREREG_AGG 'Added 2026-09-03', C1-C3): tail-row needle share by (layer,row); share clamp
on the tail rows at the band layers / all layers; head-level path patching at the band. HC regime."""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, ".")
from needles import story, PROMPT, REGIMES, VERBS  # noqa: E402
from eval_babilong import build  # noqa: E402
from modeling import load_text_model  # noqa: E402
from osq import model_parts, rope, _call  # noqa: E402
sys.path.insert(0, "outputs/_scratch/dbg")
from needles_causal import minimal_pair, Cap, forward_cache  # noqa: E402


def prep(model, ids, dev):
    embed, layers, norm, head, rotary = model_parts(model)
    S = len(ids); x = torch.tensor([ids], device=dev); p = torch.arange(S, device=dev)[None]
    h = embed(x); c, s = rotary(h, p); return h, p, (c.to(h.dtype), s.to(h.dtype))


def tail_attention(lay, h, pe, fin, fm):
    """Attention of the tail rows [fin, S) at this layer: weights [H,T,S] (fp32), v [H,S,hd], hn."""
    hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]; nh = at.q_proj.weight.shape[0] // hd; S = h.shape[1]
    q = rope(at.q_proj(hn[:, fin:]).view(1, S - fin, nh, hd).transpose(1, 2), pe[0][:, fin:], pe[1][:, fin:])[0]        # [H,T,hd]
    k = rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe); g = nh // k.shape[1]; k = k.repeat_interleave(g, dim=1)[0]
    v = at.v_proj(hn).view(1, S, -1, hd).transpose(1, 2).repeat_interleave(g, dim=1)[0]
    sc = torch.einsum("htd,hsd->hts", q.float(), k.float()) / hd ** 0.5
    T = S - fin; causal = torch.arange(S, device=h.device)[None, :] > (fin + torch.arange(T, device=h.device))[:, None]
    w = torch.softmax(sc.masked_fill(causal[None], float("-inf")), -1)
    return w, v, at


def tail_replace(lay, h, h_out, fin, w_new, v, at):
    a = at.o_proj(torch.einsum("hts,hsd->thd", w_new.to(v.dtype), v).reshape(h.shape[1] - fin, -1))[None]
    mid = h[:, fin:] + a; h_out[:, fin:] = mid + lay.mlp(lay.post_attention_layernorm(mid)); return h_out


@torch.no_grad()
def share_map(model, rec, facts, dev, Tkeep):
    """C1: head-mean needle share of the last Tkeep tail rows at every layer -> [nL, Tkeep]; plus k_eff
    over needles (head-mean weights) for the answer row and for the best tail row, per layer."""
    embed, layers, norm, head, rotary = model_parts(model); h, p, pe = prep(model, rec["ids"], dev)
    S = rec["seq"]; fin = rec["fin"]; fm = torch.zeros(S, dtype=torch.bool, device=dev); bl = rec["blocks"]
    for j in facts: fm[bl[j][0]:bl[j][1]] = True
    shares, keff_ans = [], []
    for lay in layers:
        w, v, at = tail_attention(lay, h, pe, fin, fm)
        sh = w[:, :, fm].sum(-1).mean(0)                      # [T]
        shares.append(sh[-Tkeep:].cpu())
        wm = w.mean(0)[-1]; wn = torch.tensor([float(wm[bl[j][0]:bl[j][1]].sum()) for j in facts]); pn = wn / (wn.sum() + 1e-9)
        keff_ans.append(float(1 / (pn ** 2).sum()))
        h = _call(lay, h, p, pe)
    return torch.stack(shares), keff_ans


@torch.no_grad()
def emit_clamped(model, tok, rec, facts, dev, clamp_layers, ref, mode, max_new=3):
    """Greedy emission with the tail rows' needle share at `clamp_layers` reset to ref[li][h, row_from_end]."""
    embed, layers, norm, head, rotary = model_parts(model); ids = list(rec["ids"]); bl = rec["blocks"]; fin = rec["fin"]; toks = []
    for _ in range(max_new):
        S = len(ids); h, p, pe = prep(model, ids, dev); fm = torch.zeros(S, dtype=torch.bool, device=dev)
        for j in facts: fm[bl[j][0]:bl[j][1]] = True
        for li, lay in enumerate(layers):
            h_out = _call(lay, h, p, pe)
            if li in clamp_layers and mode != "none":
                w, v, at = tail_attention(lay, h, pe, fin, fm); T = S - fin
                m = w[:, :, fm].sum(-1)                                      # [H,T]
                if mode == "amp": t = (1.5 * m).clamp(max=0.98)
                else:
                    R = ref[li]; Tr = R.shape[1]; t = m.clone()
                    n = min(T, Tr); t[:, T - n:] = R[:, Tr - n:].to(dev)        # align from the end
                ok = (m > 1e-3) & (m < 0.999) & (t > 1e-3) & (t < 0.999)
                sn = torch.where(ok, t / m, torch.ones_like(m)); sj = torch.where(ok, (1 - t) / (1 - m), torch.ones_like(m))
                w_new = torch.where(fm[None, None], w * sn[..., None], w * sj[..., None])
                h_out = tail_replace(lay, h, h_out, fin, w_new, v, at)
            h = h_out
        t_id = int(head(norm(h[:, -1])).float()[0].argmax()); s_ = tok.decode([t_id])
        if not s_.strip().isdigit(): break
        toks.append(t_id); ids.append(t_id)
    txt = tok.decode(toks); mm = re.search(r"\d+", txt); return int(mm.group()) if mm else None


@torch.no_grad()
def ref_shares(model, rec, facts, dev, layers_needed, Tkeep):
    embed, layers, norm, head, rotary = model_parts(model); h, p, pe = prep(model, rec["ids"], dev)
    S = rec["seq"]; fin = rec["fin"]; fm = torch.zeros(S, dtype=torch.bool, device=dev)
    for j in facts: fm[rec["blocks"][j][0]:rec["blocks"][j][1]] = True
    out = {}
    for li, lay in enumerate(layers):
        if li in layers_needed:
            w, _, _ = tail_attention(lay, h, pe, fin, fm); out[li] = w[:, -Tkeep:, fm].sum(-1).cpu()   # [H,Tkeep]
        h = _call(lay, h, p, pe)
    return out


@torch.no_grad()
def head_patch_effect(model, recA, hsA_in, capA, capB, li, heads, tA, tB, dA, dB, dev):
    """Patch heads `heads` (list) at layer li, tail rows only, B's o_proj-input slices into A."""
    embed, layers, norm, head, rotary = model_parts(model); lay = layers[li]; S = recA["seq"]; fin = recA["fin"]
    h, p, pe = hsA_in[li], torch.arange(S, device=dev)[None], None
    _, _, pe = prep(model, recA["ids"], dev)
    h_out = _call(lay, h, p, pe); hd = model.config.hidden_size // model.config.num_attention_heads
    cat = capA[li][fin:].clone()
    for hh in heads: cat[:, hh * hd:(hh + 1) * hd] = capB[li][fin:, hh * hd:(hh + 1) * hd]
    a = lay.self_attn.o_proj(cat.to(h.dtype))[None]; mid = h[:, fin:] + a; h_out[:, fin:] = mid + lay.mlp(lay.post_attention_layernorm(mid))
    hh_ = h_out
    for l2 in layers[li + 1:]: hh_ = _call(l2, hh_, p, pe)
    lg = head(norm(hh_[:, -1])).float()[0]
    return (float(lg[tB] - lg[tA]) - dA) / (dB - dA) if abs(dB - dA) > 1e-3 else float("nan")


class CapAll:
    def __init__(self, layers, want):
        self.buf = {}; self.hs = [layers[i].self_attn.o_proj.register_forward_hook(self._mk(i)) for i in want]
    def _mk(self, i):
        def hook(mod, inp, out): self.buf[i] = inp[0][0].detach()
        return hook
    def close(self):
        for h in self.hs: h.remove()


def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev)
    embed, layers, norm, head, rotary = model_parts(model); nL = len(layers); L = a.layer
    b0, b1 = [int(x) for x in a.band.split("+")]; band = list(range(b0, b1 + 1)); allL = list(range(nL))
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), n_layers=nL, band=band), indent=1)); t0 = time.time()
    Tkeep = 12; ks = [int(k) for k in a.ks.split("+")]
    # ---- C1 share map + C2 reference (K=4)
    maps, refs = [], []
    rng4 = random.Random(a.seed * 1000 + 4)                     # reference shares: dedicated K=4 pass
    for i in range(max(a.n, 8)):
        sents, facts, q = story(4, REGIMES["HC"](4), rng4); refs.append(ref_shares(model, build(tok, "qa1", q, sents, PROMPT), facts, dev, set(allL), Tkeep))
    for K in ks:
        D = REGIMES["HC"](K); rng = random.Random(a.seed * 1000 + K)
        for i in range(a.n):
            sents, facts, q = story(K, D, rng); rec = build(tok, "qa1", q, sents, PROMPT)
            sm, keff = share_map(model, rec, facts, dev, Tkeep); maps.append(dict(K=K, i=i, share=sm, keff_ans=keff, T=rec["seq"] - rec["fin"]))
        print(f"[{time.time()-t0:.0f}s] C1 K={K:2d} tail-share head-mean: band {torch.stack([m['share'][band].mean() for m in maps if m['K']==K]).mean():.3f} at L {torch.stack([m['share'][L].mean() for m in maps if m['K']==K]).mean():.3f} answer row@L {torch.stack([m['share'][L, -1] for m in maps if m['K']==K]).mean():.3f} keff_ans@band {sum(sum(m['keff_ans'][b] for b in band)/len(band) for m in maps if m['K']==K)/a.n:.1f}", flush=True)
    ref = {li: torch.stack([r[li] for r in refs]).mean(0) for li in allL}
    torch.save(dict(maps=maps, ref=ref), odir / "sharemap.pt")
    # ---- C2 clamp
    clamp = []
    for K in ks:
        D = REGIMES["HC"](K); rng = random.Random(a.seed * 1000 + K); res = {m: [] for m in ("none", "band", "all", "amp_band")}
        for i in range(a.n):
            sents, facts, q = story(K, D, rng); rec = build(tok, "qa1", q, sents, PROMPT)
            res["none"].append(emit_clamped(model, tok, rec, facts, dev, set(), ref, "none"))
            res["band"].append(emit_clamped(model, tok, rec, facts, dev, set(band), ref, "clamp"))
            res["all"].append(emit_clamped(model, tok, rec, facts, dev, set(allL), ref, "clamp"))
            res["amp_band"].append(emit_clamped(model, tok, rec, facts, dev, set(band), ref, "amp"))
        clamp.append(dict(K=K, **res)); import statistics
        def med(v): v = [x for x in v if x is not None]; return statistics.median(v) if v else float("nan")
        print(f"[{time.time()-t0:.0f}s] C2 K={K:2d} median emitted none {med(res['none']):.0f} band {med(res['band']):.0f} all {med(res['all']):.0f} amp {med(res['amp_band']):.0f} | exact " + " ".join(f"{m}:{sum(x == K for x in res[m])/a.n:.2f}" for m in res), flush=True)
    torch.save(clamp, odir / "clamp.pt")
    # ---- C3 head patching at the band
    hd = model.config.hidden_size // model.config.num_attention_heads; nh = model.config.num_attention_heads
    eff_sum = torch.zeros(nL, nh); cnt = 0; pair_store = []
    for K in [int(k) for k in a.ks_pairs.split("+")]:
        D = REGIMES["HC"](K); rng = random.Random(a.seed * 1000 + K); made = 0; tries = 0
        while made < a.n_pairs and tries < 4 * a.n_pairs:
            tries += 1; mp = minimal_pair(K, D, rng, tok)
            if mp is None: continue
            sA, sB, facts, jf, q = mp; recA, recB = build(tok, "qa1", q, sA, PROMPT), build(tok, "qa1", q, sB, PROMPT)
            cA = CapAll(layers, band); hsA, lgA, pe, p, _, _ = forward_cache(model, recA["ids"], dev); cA.close()
            cB = CapAll(layers, band); hsB, lgB, _, _, _, _ = forward_cache(model, recB["ids"], dev); cB.close()
            tA, tB = tok(str(K), add_special_tokens=False).input_ids[0], tok(str(K + 1), add_special_tokens=False).input_ids[0]
            dA, dB = float(lgA[tB] - lgA[tA]), float(lgB[tB] - lgB[tA])
            if abs(dB - dA) < 0.05: continue
            E = torch.full((nL, nh), float("nan"))
            for li in band:
                for hh in range(nh): E[li, hh] = head_patch_effect(model, recA, hsA, cA.buf, cB.buf, li, [hh], tA, tB, dA, dB, dev)
            top = torch.topk(torch.nan_to_num(E.abs().flatten(), nan=0.0), 5).indices; joint = {}
            for li in band:
                hs_ = [int(t % nh) for t in top if int(t // nh) == li]
                if hs_: joint[li] = hs_
            # joint patch: apply the top-5 heads across their layers sequentially (approximate: patch each layer's set in one forward)
            # (single-layer joint if all top heads sit in one layer; otherwise report per-layer joint)
            jeff = {li: head_patch_effect(model, recA, hsA, cA.buf, cB.buf, li, hs_, tA, tB, dA, dB, dev) for li, hs_ in joint.items()}
            eff_sum += torch.nan_to_num(E, nan=0.0); cnt += 1; pair_store.append(dict(K=K, E=E, joint=jeff, dA=dA, dB=dB)); made += 1
            del hsA, hsB
        print(f"[{time.time()-t0:.0f}s] C3 K={K} pairs {made}", flush=True)
    torch.save(pair_store, odir / "heads.pt"); print("->", odir); return analyze(odir)


def analyze(d):
    import numpy as np, statistics
    d = Path(d); cfg = json.load(open(d / "config.json")); L = cfg["layer"]; band = cfg["band"]; nL = cfg["n_layers"]
    sm = torch.load(d / "sharemap.pt"); maps = sm["maps"]; clamp = torch.load(d / "clamp.pt"); heads = torch.load(d / "heads.pt")
    out = [f"# Bundle C — {cfg['model']} L{L} band {band[0]}–{band[-1]} ({d})"]
    ks = sorted({m["K"] for m in maps})
    out.append("\nC1 head-mean needle share of the ANSWER row by layer (rows: K): layers " + " ".join(f"{li:5d}" for li in range(nL)))
    for K in ks:
        S_ = torch.stack([m["share"] for m in maps if m["K"] == K]).mean(0)   # [nL, Tkeep]
        out.append(f"K{K:2d} " + " ".join(f"{float(S_[li, -1]):5.2f}" for li in range(nL)))
    out.append("C1 head-mean needle share of the TAIL rows (mean over last 12 rows) at band layers, by K: " + " ".join(f"K{K}:{float(torch.stack([m['share'][band].mean() for m in maps if m['K']==K]).mean()):.3f}" for K in ks))
    out.append("C1 same at the probe layer L: " + " ".join(f"K{K}:{float(torch.stack([m['share'][L].mean() for m in maps if m['K']==K]).mean()):.3f}" for K in ks))
    out.append("C1 k_eff over needles, answer row, mean over band layers: " + " ".join(f"K{K}:{np.mean([np.mean([m['keff_ans'][b] for b in band]) for m in maps if m['K']==K]):.1f}" for K in ks))
    out.append("\nC2 clamp — median emitted count (exact) by K:")
    for mode in ("none", "band", "all", "amp_band"):
        out.append(f"{mode:9s}: " + " ".join(f"K{c['K']}:{statistics.median([x for x in c[mode] if x is not None]) if any(x is not None for x in c[mode]) else float('nan'):.0f}({np.mean([x == c['K'] for x in c[mode]]):.2f})" for c in clamp))
    E = torch.stack([p["E"] for p in heads]).nanmean(0).numpy() if heads else None
    if E is not None:
        flat = np.argsort(-np.nan_to_num(np.abs(E)).ravel())[:10]; nh = E.shape[1]
        out.append("\nC3 head patch effect at the band (mean over pairs), top |heads|: " + " ".join(f"L{i // nh}H{i % nh}:{E.ravel()[i]:+.2f}" for i in flat))
        out.append("C3 per-layer sum of head effects: " + " ".join(f"L{li}:{np.nansum(E[li]):+.2f}" for li in band))
        js = {}
        for p in heads:
            for li, v in p["joint"].items(): js.setdefault(li, []).append(v)
        out.append("C3 joint patch of each pair's top-5 heads (per layer, mean effect): " + " ".join(f"L{li}:{np.nanmean(v):+.2f}(n={len(v)})" for li, v in sorted(js.items())))
    txt = "\n".join(out); print(txt); (d / "ANALYSIS.md").write_text(txt + "\n"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--layer", type=int, default=27); ap.add_argument("--band", default="18+24")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--ks", default="1+2+4+6+8+12+16+24+32"); ap.add_argument("--ks-pairs", default="2+4+6+8"); ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--output", default="outputs/needles_band"); ap.add_argument("--analyze", default="")
    a = ap.parse_args(); return analyze(Path(a.analyze)) if a.analyze else run(a)


if __name__ == "__main__":
    sys.exit(main())
