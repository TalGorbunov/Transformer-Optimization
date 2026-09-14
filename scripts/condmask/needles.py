"""Many-needle dose-response at the read node (PREREG_AGG.md).

bAbI-style stories: K NEEDLES "<person> <verb> the <room>." that all match the question
"How many times did <person> go to the <room>?", randomly interleaved with D DISTRACTORS
(other person/room pairs, incl. the same person elsewhere and others in the same room).
Regimes: H0 = no distractors, HF = 64 distractors, HC = 64 sentences in total (D = 64 - K),
H32 = 32 sentences in total, DUP = the H32 story with K/2 needles repeated twice (gold K).
Per story, at the answer row (the last token):
  emission   greedy decode -> first integer; ok, err = |pred - K|
  margin     log P(K) - max(log P(K-1), log P(K+1)); candidate tokens + eos, KV-cached
  attention  osq.py metrics at --layer + the per-needle mass vector: keff_needles = 1/sum p_i^2
  jacobian   per-needle grad norm of the gold logit at --layer's input: mean, sum (share = infl)
  states     read-node state after EVERY layer (fp16, states.pt) -> per-layer decodability of K
  smoothing  mean pairwise cosine between the K needle-block mean states, at --layer and per layer
gold token = first token of str(K) (Qwen splits digits: for K >= 10 the Jacobian/margin
gold is the leading digit; the margin uses full candidate strings).

  python scripts/condmask/needles.py --model Qwen/Qwen2.5-3B-Instruct --layer 27 --n 50
  python scripts/condmask/needles.py --analyze outputs/needles/Qwen2.5-3B-Instruct/<stamp>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from modeling import load_text_model  # noqa: E402
from osq import osq_probe  # noqa: E402
from eval_babilong import build, decode  # noqa: E402

PEOPLE = ["Mary", "John", "Sandra", "Daniel", "Fred", "Bill", "Julie", "Jeff"]
ROOMS = ["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"]
VERBS = ["went to", "moved to", "travelled to", "journeyed to", "went back to", "walked to"]
PROMPT = {
    "instruction": "I will give you context with the facts about movements of different persons "
                   "hidden in some random text and a question. You need to answer the question "
                   "based only on the information from the facts: count how many facts match.",
    "examples": "<example>\nMary went to the kitchen. John moved to the garden. Mary travelled to "
                "the kitchen. Sandra went to the office. How many times did Mary go to the kitchen?\n"
                "Answer: 2\n</example>\n\n<example>\nDaniel went back to the hallway. John journeyed "
                "to the bathroom. Daniel moved to the hallway. Daniel walked to the hallway. "
                "How many times did Daniel go to the hallway?\nAnswer: 3\n</example>",
    "post_prompt": "Your answer should contain only the number, written in digits. "
                   "Do not write anything else after that.",
}
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
# regime -> distractor count D(K); None = cell not run. H32 = 32 sentences total (K <= 16);
# DUP = the H32 story with K/2 needles repeated twice (gold K; even K <= 32): the GIN mean-vs-sum
# test -- a mean aggregator reads S||S exactly like S.
REGIMES = {"H0": lambda K: 0, "HF": lambda K: 64, "HC": lambda K: 64 - K,
           "H32": lambda K: 32 - K if K <= 16 else None,
           "DUP": lambda K: 32 - K // 2 if K % 2 == 0 and K <= 32 else None}


def story(K: int, D: int, rng: random.Random):
    p, r = rng.choice(PEOPLE), rng.choice(ROOMS)
    items = [(f"{p} {rng.choice(VERBS)} the {r}.", True) for _ in range(K)]
    while len(items) < K + D:
        pp, rr = rng.choice(PEOPLE), rng.choice(ROOMS)
        if (pp, rr) != (p, r):
            items.append((f"{pp} {rng.choice(VERBS)} the {rr}.", False))
    rng.shuffle(items)
    return ([s for s, _ in items], [j for j, (_, f) in enumerate(items) if f],
            f"How many times did {p} go to the {r}?")


@torch.no_grad()
def cand_logp(model, tok, ids: List[int], eos_id: int, cands: List[str], dev) -> List[float]:
    """log P(candidate string, then eos | prompt): one KV-cached pass per candidate, cache cropped back."""
    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True, logits_to_keep=1)
    past, n = out.past_key_values, len(ids)
    first = torch.log_softmax(out.logits[0, -1].float(), -1)
    res = []
    for c in cands:
        t = tok(c, add_special_tokens=False).input_ids + [eos_id]
        o = model(input_ids=torch.tensor([t[:-1]], device=dev),
                  position_ids=torch.arange(n, n + len(t) - 1, device=dev)[None],
                  past_key_values=past, use_cache=True)
        ls = torch.log_softmax(o.logits[0].float(), -1)
        res.append(float(first[t[0]]) + sum(float(ls[i, t[i + 1]]) for i in range(len(t) - 1)))
        past.crop(n)
    return res


def needle_cos(h: torch.Tensor, nb) -> float:
    """mean pairwise cosine between the needle-block mean states of h [S, d] (None if one needle)."""
    K = len(nb)
    if K < 2:
        return None
    hb = torch.nn.functional.normalize(torch.stack([h[a:b].float().mean(0) for a, b in nb]), dim=-1)
    return float((hb @ hb.T).sum() - K) / (K * (K - 1))


def needle_metrics(ex: dict, rec, facts: List[int]) -> Dict[str, float]:
    """From osq_probe's raw tensors: per-needle mass (head-mean, context-restricted softmax --
    sums to osq `mass`), its fan-in, per-needle Jacobian norm sums, needle-state cosine."""
    nb = [rec["blocks"][j] for j in facts]
    a0, b1 = rec["blocks"][0][0], rec["blocks"][-1][1]
    w = torch.softmax(ex["sc"][:, a0:b1].float(), -1).mean(0)
    wn = torch.tensor([float(w[a - a0:b - a0].sum()) for a, b in nb])
    p = wn / wn.sum()
    gs = torch.tensor([float(ex["gn"][a:b].sum()) for a, b in nb])
    # A11: nb is in story order, so [-1] is the LAST needle -- a running counter read from the last
    # item (Hasani et al. 2511.17699) gives it a share >> 1/K; a mean read gives ~1/K.
    return dict(mass_needles=float(wn.sum()), keff_needles=float(1 / (p ** 2).sum()),
                mass_needle_min=float(wn.min()), mass_needle_max=float(wn.max()),
                mass_last_share=float(p[-1]), jac_last_share=float(gs[-1] / gs.sum()),
                jac_needle_mean=float(gs.mean()), jac_needle_sum=float(gs.sum()),
                cos_needles=needle_cos(ex["h_mid"], nb))


@torch.no_grad()
def layer_scan(model, ids: List[int], rec, facts: List[int], dev):
    """One forward with hidden states: read-node state after every layer ([L+1, d] fp16; [0] =
    embeddings, [-1] = final-normed) and the needle cosine per layer -- where the count is
    decodable and where the needles smooth out."""
    nb = [rec["blocks"][j] for j in facts]
    hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
    return torch.stack([x[0, -1] for x in hs]).half().cpu(), [needle_cos(x[0], nb) for x in hs]


def run(args) -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_text_model(args.model, device=dev)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    ks = [int(k) for k in args.ks.split("+")]
    odir = Path(args.output) / args.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(args), indent=1))
    fout = open(odir / "results.jsonl", "w")
    st: Dict[str, list] = {"regime": [], "K": [], "i": [], "h_layers": []}
    rng, t0 = random.Random(args.seed), time.time()
    for reg in args.regimes.split("+"):
        for K in ks:
            D = REGIMES[reg](K)
            if D is None or D < 0:
                continue
            cell = []
            for i in range(args.n):
                if reg == "DUP":
                    sents, facts, q = story(K // 2, D, rng)
                    sents, facts = sents * 2, facts + [j + len(sents) for j in facts]
                else:
                    sents, facts, q = story(K, D, rng)
                rec = build(tok, "qa1", q, sents, PROMPT)
                ids, pos = rec["ids"], list(range(rec["seq"]))
                text = tok.decode(decode(model, ids, pos, eos, dev, args.max_new))
                m = re.search(r"\d+", text)
                pred = int(m.group()) if m else None
                gold = tok(str(K), add_special_tokens=False).input_ids[0]
                ex: dict = {}
                row = osq_probe(model, ids, pos, rec["blocks"], facts, args.layer, gold, dev,
                                jac=True, ckpt=False, extra=ex)
                lp = cand_logp(model, tok, ids, tok.eos_token_id, [str(K - 1), str(K), str(K + 1)], dev)
                hl, cl = layer_scan(model, ids, rec, facts, dev)
                row.update(needle_metrics(ex, rec, facts), regime=reg, K=K, D=D, i=i, out=text, pred=pred,
                           ok=pred == K, err=None if pred is None else abs(pred - K), tokens=len(ids),
                           lp_gold=lp[1], margin=lp[1] - max(lp[0], lp[2]), cos_layers=cl)
                fout.write(json.dumps(row) + "\n")
                cell.append(row)
                for key, val in (("regime", reg), ("K", K), ("i", i), ("h_layers", hl)):
                    st[key].append(val)
            mean = lambda k: sum(r[k] for r in cell if r[k] is not None) / max(1, sum(r[k] is not None for r in cell))
            print(f"[{time.time()-t0:.0f}s] {reg} K={K:2d} D={D:2d} acc {mean('ok'):.2f} err {mean('err'):.2f} "
                  f"mass {mean('mass'):.3f} keff_n {mean('keff_needles'):.1f} jac_n {mean('jac_needle_mean'):.3g} "
                  f"margin {mean('margin'):.2f} cos {mean('cos_needles') if K > 1 else float('nan'):.3f}", flush=True)
    fout.close()
    torch.save({k: (torch.stack(v) if k == "h_layers" else v) for k, v in st.items()}, odir / "states.pt")
    print("->", odir)
    return 0


# ------------------------------------------------------------------------------ analysis (CPU)

def _spearman(x, y) -> float:
    import numpy as np
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def _slope(x, y) -> float:
    import numpy as np
    return float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x) > 1 else float("nan")


def analyze(d: Path) -> int:
    """Per (regime, K) means; A2 reversed-dispersion fit; A4 slopes; A3/A6 Spearmans;
    A5 ridge decodability (probe_text_triple.ridge_round, half/half by story index)."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.DataFrame([json.loads(l) for l in open(d / "results.jsonl")])
    st = torch.load(d / "states.pt")
    layer = json.load(open(d / "config.json"))["layer"]
    st["h_mid"], st["h_fin"] = st["h_layers"][:, layer], st["h_layers"][:, -1]   # probe-layer input, final normed
    out = [f"# needles — {d}\n"]

    def ridge(X, y, tr, te):                              # same pipeline as probe_text_triple.ridge_round
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6))).fit(X[tr], y[tr])
        pr = mdl.predict(X[te])
        return np.rint(pr), 1 - ((y[te] - pr) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum()
    cols = ["ok", "err", "mass", "keff_sent", "keff_needles", "jac_needle_mean", "jac_needle_sum",
            "infl", "margin", "cos_needles", "mass_needle_min", "mass_needle_max",
            "mass_last_share", "jac_last_share"]
    out.append(df.groupby(["regime", "K"])[cols].mean().round(3).to_string() + "\n")
    for reg, sub in df.groupby("regime"):
        c = sub.groupby("K")[cols + ["n_tok", "n_fact_tok"]].mean()
        K = c.index.values.astype(float)
        lines = [f"## {reg}"]
        if reg == "DUP":                                   # A9: is the duplicated half counted?
            pr = pd.to_numeric(sub["pred"])
            g9 = sub.assign(ratio=pr / sub["K"], half=pr == sub["K"] // 2).groupby("K")[["ratio", "half"]].mean()
            lines.append("A9 duplication: mean pred/K " + " ".join(f"K{k}:{v:.2f}" for k, v in g9["ratio"].items())
                         + " | P(pred==K/2) " + " ".join(f"K{k}:{v:.2f}" for k, v in g9["half"].items()))
        if reg != "H0" and 1 in c.index:                   # A2: m = kF e^g / (kF e^g + J), g fitted at K=1
            m1, kF1, J1 = c.loc[1, "mass"], c.loc[1, "n_fact_tok"], c.loc[1, "n_tok"] - c.loc[1, "n_fact_tok"]
            g_ = math.log(m1 / (1 - m1) * J1 / kF1)
            pred = c["n_fact_tok"] * math.exp(g_) / (c["n_fact_tok"] * math.exp(g_) + c["n_tok"] - c["n_fact_tok"])
            mae = float((pred - c["mass"]).abs()[c.index > 1].mean())
            lines.append(f"A2 reversed dispersion: g(K=1)={g_:.2f}  MAE(pred mass, K>1)={mae:.3f}  "
                         + " ".join(f"K{int(k)}:{p:.2f}/{m:.2f}" for k, p, m in zip(c.index, pred, c["mass"])))
            res = c["mass"] * (1 - c["mass"]) / K
            lines.append(f"A3 Spearman(resolution m(1-m)/K, margin) = {_spearman(res.values, c['margin'].values):.2f}")
        lines.append("A2' keff_needles/K: " + " ".join(f"K{int(k)}:{v/k:.2f}" for k, v in zip(K, c["keff_needles"])))
        sel = K <= 32
        lines.append(f"A4 log-log slope per-needle Jacobian (K<=32) = {_slope(K[sel], c['jac_needle_mean'].values[sel]):.2f}; "
                     f"sum range x{c['jac_needle_sum'].values[sel].max()/c['jac_needle_sum'].values[sel].min():.2f}")
        lines.append(f"A7 cos_needles min over K>=4 = {c.loc[c.index >= 4, 'cos_needles'].min():.3f}")
        lines.append("A11 last-needle share x K (1 = uniform read; >>1 = running-counter read) mass/jac: "
                     + " ".join(f"K{int(k)}:{m*k:.1f}/{j*k:.1f}" for k, m, j in zip(K, c["mass_last_share"], c["jac_last_share"])))
        # A5 decodability: same pipeline as probe_text_triple.ridge_round; even stories train / odd test
        idx = [n for n, r in enumerate(st["regime"]) if r == reg]
        y = np.array([st["K"][n] for n in idx], dtype=float)
        ii = np.array([st["i"][n] for n in idx])
        tr, te = ii % 2 == 0, ii % 2 == 1
        yt = y[te]
        for name in ("h_mid", "h_fin"):
            pk, r2 = ridge(st[name][idx].float().numpy(), y, tr, te)
            per_k = {int(k): float((pk[yt == k] == k).mean()) for k in np.unique(yt)}
            lines.append(f"A5 ridge {name}: exact {(pk == yt).mean():.2f} within1 {(np.abs(pk - yt) <= 1).mean():.2f} "
                         f"R2 {r2:.2f} (chance {1/len(per_k):.2f}); per-K probe/emit: "
                         + " ".join(f"K{k}:{per_k[k]:.2f}/{c.loc[k, 'ok']:.2f}" for k in sorted(per_k)))
        # layer scans: where K is linearly decodable (exact / R2 per layer) and where the needles smooth out
        H = st["h_layers"][idx].float().numpy()
        prof = [ridge(H[:, l], y, tr, te) for l in range(H.shape[1])]
        lines.append("A5 layer scan exact: " + " ".join(f"{(pk == yt).mean():.2f}" for pk, _ in prof))
        lines.append("A5 layer scan R2:    " + " ".join(f"{r2:.2f}" for _, r2 in prof))
        cs = np.array([r for r, k in zip(sub["cos_layers"], sub["K"]) if k > 1], dtype=float)
        lines.append("A7 needle cosine per layer (mean over K>1): " + " ".join(f"{v:.2f}" for v in cs.mean(0)))
        # A6 resolution d'(K, K+) along the mean-difference direction, final state
        X = st["h_fin"][idx].float().numpy()
        ks_ = sorted(set(int(k) for k in y))
        dp = []
        for a, b in zip(ks_[:-1], ks_[1:]):
            A, B = X[y == a], X[y == b]
            u = B.mean(0) - A.mean(0)
            u /= np.linalg.norm(u) + 1e-9
            pa, pb = A @ u, B @ u
            dp.append((a, float((pb.mean() - pa.mean()) / math.sqrt((pa.var() + pb.var()) / 2 + 1e-9))))
        lines.append("A6 d'(K,K+): " + " ".join(f"K{a}:{v:.2f}" for a, v in dp)
                     + f"  slope {_slope(np.array([a for a, _ in dp], float), np.array([v for _, v in dp])):.2f}"
                     + f"  Spearman(d', acc) {_spearman([v for _, v in dp], [c.loc[a, 'ok'] for a, _ in dp]):.2f}")
        out.append("\n".join(lines) + "\n")
    (d / "ANALYSIS.md").write_text("\n".join(out))
    print("\n".join(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layer", type=int, default=27, help="probe layer (BABILong selection layer)")
    ap.add_argument("--n", type=int, default=50, help="stories per (regime, K)")
    ap.add_argument("--ks", default="+".join(map(str, KS)))
    ap.add_argument("--regimes", default="H0+HF+HC+H32+DUP")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/needles")
    ap.add_argument("--analyze", default="", help="run dir: analysis only (CPU)")
    args = ap.parse_args()
    return analyze(Path(args.analyze)) if args.analyze else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
