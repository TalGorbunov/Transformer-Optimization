"""Pre-registered over-squashing analysis (PREREG_OSQ.md P1-P7) over the BABILong run roots.

  python scripts/condmask/analyze_osq.py [--osq outputs/babilong_osq] [--out ANALYSIS.md]

Reads results.jsonl of the LATEST run per (root, model, task, length); every accuracy /
recall carries a bootstrap 95% CI (1000 resamples); per-example joins are on
(model, task, length, i) -- all roots read the same split in the same order. Each section
prints the pre-registered prediction, the measured numbers and a verdict
(supported / falsified / inconclusive) by the thresholds written before the runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

LENS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
MODELS = {"Qwen2.5-3B-Instruct": "Q3", "Qwen2.5-7B-Instruct": "Q7", "Llama-3.1-8B-Instruct": "LL"}
METR = ["mass", "keff_sent", "keff_tok", "gap", "pred_mass", "m_eff", "infl", "keff_jac", "ctx_share", "tokens"]
rng = np.random.default_rng(0)


def load(root: str) -> pd.DataFrame:
    """Latest run per cell -> long frame (one row per example x arm)."""
    rows: List[dict] = []
    for cell in sorted(Path(root).glob("*/*_*k")):
        runs = sorted(d for d in cell.iterdir() if (d / "results.jsonl").exists())
        if not runs:
            continue
        run = runs[-1]
        cfg = json.loads((run / "config.json").read_text())
        task, length = cell.name.rsplit("_", 1)
        for line in open(run / "results.jsonl"):
            r = json.loads(line)
            rows.append({"root": root, "model": cell.parent.name, "task": task, "length": length,
                         "i": r["i"], "arm": r["arm"], "ok": int(r["ok"]), "recall": r.get("recall"),
                         "yarn": cfg.get("yarn", 0.0), "rounds": cfg.get("sel_rounds", 1),
                         **{m: r.get(m) for m in METR}})
    df = pd.DataFrame(rows)
    if len(df):
        df["length"] = pd.Categorical(df["length"], LENS, ordered=True)
    return df


def ci(x) -> str:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]                       # recall is undefined on the few samples with no fact sentence
    if len(x) == 0:
        return "n/a"
    b = rng.choice(x, (1000, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(b, 2.5):.2f},{np.percentile(b, 97.5):.2f}]"


def cell_means(df: pd.DataFrame, arm: str, cols: List[str]) -> pd.DataFrame:
    d = df[df.arm == arm].groupby(["model", "task", "length"], observed=True)
    out = d[["ok", "recall"] + cols].mean(numeric_only=True)
    out["n"] = d.size()
    return out.reset_index()


def paired(a: pd.DataFrame, b: pd.DataFrame, arm_a: str, arm_b: str, col: str) -> pd.DataFrame:
    """Per-example join of two frames' arms on (model, task, length, i) -> diff b - a."""
    ka = a[a.arm == arm_a][["model", "task", "length", "i", col]].rename(columns={col: "a"})
    kb = b[b.arm == arm_b][["model", "task", "length", "i", col]].rename(columns={col: "b"})
    j = ka.merge(kb, on=["model", "task", "length", "i"])
    j["d"] = j.b - j.a
    return j


def logit_fit(x: np.ndarray, y: np.ndarray):
    """Newton logistic regression y ~ 1 + x -> (coef, p-value) (Wald); no statsmodels needed."""
    X = np.c_[np.ones_like(x), (x - x.mean()) / (x.std() + 1e-9)]
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ w))
    se = np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2))))
    z = w[1] / se[1]
    return w[1], 2 * stats.norm.sf(abs(z))


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def verdict(sup: List, fals: List = ()) -> str:
    """Pre-reg semantics: FALSIFIED iff a named falsifier fires; SUPPORTED iff every support check passes
    and none fires; otherwise mixed. None entries (cells not landed) are ignored."""
    sup = [v for v in sup if v is not None]
    fals = [v for v in fals if v is not None]
    tag = ("**FALSIFIED**" if any(fals) else "**SUPPORTED**" if sup and all(sup) else "inconclusive" if not sup else "mixed")
    return f"{tag} — supports {sum(sup)}/{len(sup)} pass; falsifiers fired {sum(fals)}/{len(fals)}"


class Report:
    def __init__(self):
        self.lines: List[str] = []

    def h(self, s):
        self.lines += ["", f"## {s}", ""]

    def p(self, s=""):
        self.lines.append(s)

    def table(self, df: pd.DataFrame, fmt="{:.3f}"):
        if df is None or len(df) == 0:
            self.lines.append("_no data_")
            return
        cols = list(df.columns)
        self.lines.append("| " + " | ".join(map(str, cols)) + " |")
        self.lines.append("|" + "---|" * len(cols))
        for _, r in df.iterrows():
            self.lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")


def p1(R: Report, osq, chunk, one):
    R.h("P1 — Tournament selection (recall at 128k vs chunked / one-forward)")
    R.p("Prediction: Llama recall qa1>=.90 qa2>=.82 qa3>=.62 qa5>=.85; repack acc qa1>=.85 qa2>=.50; "
        "Qwen tournament >= chunked-.03 on every task. Falsified if Llama within +-.05 of chunked on qa1/qa2.")
    rows, ver, fals = [], [], []
    for m in MODELS:
        for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
            o = osq[(osq.model == m) & (osq.task == t) & (osq.length == "128k") & (osq.arm == "attn")]
            if len(o) == 0:
                continue
            c = chunk[(chunk.model == m) & (chunk.task == t) & (chunk.length == "128k") & (chunk.arm == "attn")] if len(chunk) else chunk
            f = one[(one.model == m) & (one.task == t) & (one.length == "128k") & (one.arm == "attn")] if len(one) else one
            d = paired(chunk, osq, "attn", "attn", "recall") if len(chunk) else pd.DataFrame()
            d = d[(d.model == m) & (d.task == t) & (d.length == "128k")] if len(d) else d
            da = paired(chunk, osq, "attn", "attn", "ok") if len(chunk) else pd.DataFrame()
            da = da[(da.model == m) & (da.task == t) & (da.length == "128k")] if len(da) else da
            rows.append({"model": MODELS[m], "task": t, "tourn recall": ci(o.recall), "chunked recall": ci(c.recall) if len(c) else "n/a",
                         "one-forward recall": ci(f.recall) if len(f) else "n/a", "paired Δ(tourn-chunk)": ci(d.d) if len(d) else "n/a",
                         "tourn acc": ci(o.ok), "chunked acc": ci(c.ok) if len(c) else "n/a",
                         "paired Δacc": ci(da.d.astype(float)) if len(da) else "n/a"})
            if m == "Llama-3.1-8B-Instruct":
                thr = {"qa1": .90, "qa2": .82, "qa3": .62, "qa5": .85}.get(t)
                if thr:
                    ver.append(o.recall.mean() >= thr)
                if t in ("qa1", "qa2"):
                    ver.append(o.ok.mean() >= {"qa1": .85, "qa2": .50}[t])
                    if len(c):
                        fals.append(abs(o.recall.mean() - c.recall.mean()) <= .05)
            elif len(c):
                ver.append(o.recall.mean() >= c.recall.mean() - .03)
    R.table(pd.DataFrame(rows))
    R.p(verdict(ver, fals))


def p2(R: Report, osq):
    R.h("P2 — Dispersion law (pred_mass from the margin vs measured mass; margin premise)")
    R.p("Prediction: Spearman(pred_mass, mass) > .8 over full-arm cells; |pred-mass| <= .10 in >= 70% of cells; "
        "gap(128k)-gap(4k) < 1.0 every model/task. Falsified if gap grows >= 2.0 or Spearman < .5.")
    c = cell_means(osq, "full", ["mass", "pred_mass", "gap", "m_eff", "keff_sent"])
    c = c[c.length.isin(["4k", "8k", "16k", "32k", "64k", "128k"])].dropna(subset=["pred_mass"])
    if len(c) < 3:
        R.p("_no data_")
        return
    rho = stats.spearmanr(c.pred_mass, c.mass).correlation
    within = (abs(c.pred_mass - c.mass) <= .10).mean()
    g = c.pivot_table(index=["model", "task"], columns="length", values="gap", observed=True)
    dg = (g["128k"] - g["4k"]).dropna() if "128k" in g and "4k" in g else pd.Series(dtype=float)
    R.p(f"Spearman(pred_mass, mass) over {len(c)} cells = **{rho:.3f}**; |pred-mass| <= .10 in **{within:.0%}** of cells.")
    R.p(f"gap(128k) - gap(4k): " + ", ".join(f"{MODELS[m]}/{t} {v:+.2f}" for (m, t), v in dg.items()))
    ge = c.pivot_table(index=["model", "task"], columns="length", values="m_eff", observed=True)
    de = (ge["128k"] - ge["4k"]).dropna() if "128k" in ge and "4k" in ge else pd.Series(dtype=float)
    R.p(f"m_eff(128k) - m_eff(4k) (post-hoc; the margin that governs mass): " + ", ".join(f"{MODELS[m]}/{t} {v:+.2f}" for (m, t), v in de.items())
        + "  — holding mass would need +3.47 (log 32).")
    # post-hoc (not pre-registered): gap - m_eff = the Jensen penalty of heavy-tailed junk scores;
    # the mean-margin law over-predicts mass by exactly this many nats
    c["jensen"] = c.gap - c.m_eff
    R.p(f"Post-hoc: mean-margin gap exceeds the mass-implied margin m_eff by {c.jensen.mean():.2f} nats "
        f"(range {c.jensen.min():.2f}..{c.jensen.max():.2f}) -> junk scores are heavy-tailed, dilution is not uniform.")
    c["model"] = c.model.map(MODELS)
    R.table(c[["model", "task", "length", "n", "ok", "mass", "pred_mass", "gap", "m_eff", "jensen", "keff_sent"]])
    R.p(verdict([rho > .8, within >= .7, (dg.max() < 1.0) if len(dg) else None],
                [rho < .5, (dg.max() >= 2.0) if len(dg) else None]))


def p3(R: Report, osq, yarn):
    R.h("P3 — Fan-in growth and the factorial separation")
    R.p("Prediction: full keff_sent(128k)/keff_sent(4k) >= 4 and mass(128k) <= .5 mass(4k); attn keff_sent varies < 2x; "
        "norp keff_sent within 1.5x of attn while acc differs >= 12pp on qa1/qa2 at 128k; YaRN full trails oracle >= 15pp on qa1/qa2 at 128k.")
    rows, ver, fals = [], {}, {}
    for m in MODELS:
        for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
            d = osq[(osq.model == m) & (osq.task == t)]
            if len(d) == 0:
                continue
            cm = lambda arm, L, col: d[(d.arm == arm) & (d.length == L)][col].mean()
            r = {"model": MODELS[m], "task": t}
            for arm in ("full", "attn", "attn_norp"):
                r[f"{arm} keff 4k"] = cm(arm, "4k", "keff_sent")
                r[f"{arm} keff 128k"] = cm(arm, "128k", "keff_sent")
            r["full mass 4k"], r["full mass 128k"] = cm("full", "4k", "mass"), cm("full", "128k", "mass")
            r["acc attn 128k"], r["acc norp 128k"] = cm("attn", "128k", "ok"), cm("attn_norp", "128k", "ok")
            rows.append(r)
            has = lambda *ks: all(pd.notna(r[k]) for k in ks)          # a cell that has not landed yet -> None, not FAIL
            ver[f"{MODELS[m]}/{t} full fan-in x4"] = r["full keff 128k"] / r["full keff 4k"] >= 4 if has("full keff 4k", "full keff 128k") else None
            fals[f"{MODELS[m]}/{t} full fan-in flat (<1.5x)"] = r["full keff 128k"] / r["full keff 4k"] < 1.5 if has("full keff 4k", "full keff 128k") else None
            if has("full keff 4k", "full keff 128k"):
                a4, a128 = (d[(d.arm == "full") & (d.length == L)].keff_sent.dropna().values for L in ("4k", "128k"))
                bo = rng.choice(a128, (1000, len(a128))).mean(1) / rng.choice(a4, (1000, len(a4))).mean(1)
                r["full keff ratio 128k/4k"] = f"{a128.mean() / a4.mean():.2f} [{np.percentile(bo, 2.5):.2f},{np.percentile(bo, 97.5):.2f}]"
            ver[f"{MODELS[m]}/{t} full mass halves"] = r["full mass 128k"] <= .5 * r["full mass 4k"] if has("full mass 4k", "full mass 128k") else None
            at = d[d.arm == "attn"].groupby("length", observed=True).keff_sent.mean().dropna()
            ver[f"{MODELS[m]}/{t} attn keff <2x"] = (at.max() / at.min() < 2) if len(at) > 1 else None
            if has("attn keff 128k", "attn_norp keff 128k"):
                ratio = max(r["attn keff 128k"], r["attn_norp keff 128k"]) / min(r["attn keff 128k"], r["attn_norp keff 128k"])
                ver[f"{MODELS[m]}/{t} norp keff within 1.5x"] = ratio <= 1.5
                fals[f"{MODELS[m]}/{t} norp keff >2x attn"] = ratio > 2
                if t in ("qa1", "qa2"):
                    ver[f"{MODELS[m]}/{t} acc attn-norp >= 12pp"] = r["acc attn 128k"] - r["acc norp 128k"] >= .12
    R.table(pd.DataFrame(rows))
    if len(yarn):
        R.p("YaRN x4 (Qwen), 128k:")
        yr = []
        for m in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"):
            for t in ("qa1", "qa2"):
                d = yarn[(yarn.model == m) & (yarn.task == t) & (yarn.length == "128k")]
                if len(d) == 0:
                    continue
                gap = d[d.arm == "oracle"].ok.mean() - d[d.arm == "full"].ok.mean()
                yr.append({"model": MODELS[m], "task": t, "full acc": ci(d[d.arm == "full"].ok), "oracle acc": ci(d[d.arm == "oracle"].ok),
                           "oracle-full": gap, "full keff 64k": yarn[(yarn.model == m) & (yarn.task == t) & (yarn.length == "64k") & (yarn.arm == "full")].keff_sent.mean(),
                           "full keff 128k": d[d.arm == "full"].keff_sent.mean(), "full mass 128k": d[d.arm == "full"].mass.mean()})
                ver[f"YaRN {MODELS[m]}/{t} full trails oracle >= 15pp"] = gap >= .15
        R.table(pd.DataFrame(yr))
    R.p("Checks: " + "; ".join(f"{k}: {'ok' if v else ('FAIL' if v is False else '?')}" for k, v in ver.items()))
    fired = [k for k, v in fals.items() if v]
    R.p(("Falsifiers fired: " + ", ".join(fired)) if fired else "No falsifier fired.")
    R.p(verdict(list(ver.values()), list(fals.values())))


def p4(R: Report, osq):
    R.h("P4 — Sensitivity chain and the necessity invariant")
    R.p("Prediction: within each model Spearman(cell infl, cell mass) > .7 (full arm); per-example logistic acc ~ infl positive, p < .001; "
        "cell mass separates succeeding [acc(full) >= acc(oracle) - .10] from failing cells with AUC > .85; no succeeding cell has keff_sent > 3x its 4k value. "
        "Falsified if a succeeding cell has mass below the failing cells' median or AUC < .7.")
    c = cell_means(osq, "full", ["mass", "infl", "keff_sent"])
    c = c[c.length.isin(["4k", "8k", "16k", "32k", "64k", "128k"])]
    orc = cell_means(osq, "oracle", [])[["model", "task", "length", "ok"]].rename(columns={"ok": "ok_oracle"})
    c = c.merge(orc, on=["model", "task", "length"])
    if len(c) == 0:
        R.p("_no data_")
        return
    c["succeeds"] = c.ok >= c.ok_oracle - .10
    ver = []
    for m in MODELS:
        cm = c[c.model == m].dropna(subset=["infl"])
        ex = osq[(osq.model == m) & (osq.arm == "full")].dropna(subset=["infl"])
        if len(cm) < 3 or len(ex) < 20:
            continue
        rho = stats.spearmanr(cm.infl, cm.mass).correlation
        coef, pv = logit_fit(ex.infl.values.astype(float), ex.ok.values.astype(float))
        R.p(f"{MODELS[m]}: Spearman(infl, mass) over {len(cm)} cells = **{rho:.3f}**; logistic acc~infl: coef {coef:+.2f} (per SD), p = {pv:.2e}, n = {len(ex)}")
        ver += [rho > .7, coef > 0 and pv < .001]
        # post-hoc: the pooled logistic mixes tasks whose base rates differ; per-task strata are the honest disaggregation
        per_t = []
        for t, g in ex.groupby("task"):
            if len(g) >= 20 and 0 < g.ok.mean() < 1:
                cf, pp = logit_fit(g.infl.values.astype(float), g.ok.values.astype(float))
                per_t.append(f"{t} {cf:+.2f} (p={pp:.1e}, n={len(g)})")
        R.p(f"  post-hoc per-task logistic acc~infl for {MODELS[m]}: " + "; ".join(per_t))
        am = c[c.model == m]
        if am.succeeds.any() and (~am.succeeds).any():
            R.p(f"  post-hoc per-model necessity for {MODELS[m]}: AUC(mass) = {auc(am[am.succeeds].mass, am[~am.succeeds].mass):.3f} "
                f"({int(am.succeeds.sum())} succeeding / {int((~am.succeeds).sum())} failing cells)")
    a = auc(c[c.succeeds].mass, c[~c.succeeds].mass)
    med_fail = c[~c.succeeds].mass.median() if (~c.succeeds).any() else float("nan")
    low_succ = c[c.succeeds & (c.mass < med_fail)] if (~c.succeeds).any() else c.iloc[0:0]
    k4 = c[c.length == "4k"][["model", "task", "keff_sent"]].rename(columns={"keff_sent": "k4"})
    cc = c.merge(k4, on=["model", "task"])
    blow = cc[cc.succeeds & (cc.keff_sent > 3 * cc.k4)]
    R.p(f"Necessity: {int(c.succeeds.sum())} succeeding / {int((~c.succeeds).sum())} failing cells; AUC(mass) = **{a:.3f}**; "
        f"failing-cell median mass {med_fail:.3f}; succeeding cells below it: {len(low_succ)}; succeeding cells with keff_sent > 3x own 4k: {len(blow)}")
    for _, r in low_succ.iterrows():
        R.p(f"  - offending cell {MODELS[r.model]}/{r.task}/{r.length}: mass {r.mass:.3f} ({r.mass - med_fail:+.3f} vs median), acc(full) {r.ok:.2f} vs oracle {r.ok_oracle:.2f}, n={int(r.n)}")
    c["model"] = c.model.map(MODELS)
    R.table(c[["model", "task", "length", "n", "ok", "ok_oracle", "succeeds", "mass", "infl", "keff_sent"]])
    R.p(verdict(ver + [a > .85 if not np.isnan(a) else None, len(blow) == 0],
                [(a < .7) if not np.isnan(a) else None, len(low_succ) > 0 if (~c.succeeds).any() else None]))


def p5(R: Report, gap, gy):
    R.h("P5 — Position dose-response (oracle set, gap P between evidence and tail / after the prefix)")
    R.p("Prediction: Qwen native within .05 of P=0 for P <= 24k, <= .5x at P >= 40k (qa1/qa2); Qwen YaRN within .10 through 96k; "
        "Llama at 120k >= 10pp below P=0 on qa1/qa2; pgap effect smaller than gap at every P. Falsified if Llama is flat to 120k.")
    ver, fals = [], []
    for name, d in (("native", gap), ("YaRN x4", gy)):
        if len(d) == 0:
            continue
        d = d.copy()
        d["base"] = d.arm.str.partition("@")[0]
        d["P"] = pd.to_numeric(d.arm.str.partition("@")[2], errors="coerce")
        d = d.dropna(subset=["P"])
        for m in MODELS:
            for t in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
                s = d[(d.model == m) & (d.task == t)]
                if len(s) == 0:
                    continue
                piv = s.pivot_table(index="P", columns="base", values=["ok", "mass", "keff_sent", "ctx_share"], aggfunc="mean", observed=True)
                acc_g = piv["ok"]["oracle_gap"] if "oracle_gap" in piv["ok"] else None
                acc_p = piv["ok"]["oracle_pgap"] if "oracle_pgap" in piv["ok"] else None
                R.p(f"**{MODELS[m]} {t} ({name})** acc by P (gap / pgap): " + "  ".join(
                    f"{int(P)//1000}k: {acc_g.get(P, float('nan')):.2f}/{(acc_p.get(P, float('nan')) if acc_p is not None else float('nan')):.2f}" for P in acc_g.index))
                if "mass" in piv and "oracle_gap" in piv["mass"]:
                    R.p("  gap-arm read metrics by P (mass / keff_sent / ctx_share): " + "  ".join(
                        f"{int(P)//1000}k: {piv['mass']['oracle_gap'].get(P, float('nan')):.2f}/{piv['keff_sent']['oracle_gap'].get(P, float('nan')):.1f}/"
                        f"{piv['ctx_share']['oracle_gap'].get(P, float('nan')):.2f}" for P in acc_g.index))
                a0 = acc_g.get(0)
                if a0 is None or t not in ("qa1", "qa2"):
                    continue
                if m.startswith("Qwen") and name == "native":
                    ver.append(all(abs(acc_g[P] - a0) <= .05 for P in acc_g.index if 0 < P <= 24000))
                    ver.append(all(acc_g[P] <= .5 * a0 for P in acc_g.index if P >= 40000))
                if m.startswith("Qwen") and name == "YaRN x4":
                    ver.append(all(abs(acc_g[P] - a0) <= .10 for P in acc_g.index if P <= 96000))
                if m.startswith("Llama") and 120000 in acc_g.index:
                    ver.append(a0 - acc_g[120000] >= .10)
                    fals.append(abs(a0 - acc_g[120000]) < .03)          # "flat to 120k"
                if acc_p is not None:
                    ver.append(all((a0 - acc_p[P]) <= (a0 - acc_g[P]) + 1e-9 for P in acc_g.index if P in acc_p.index and P > 0))
    R.p(verdict(ver, fals))


def p6(R: Report, rounds, osq):
    R.h("P6 — Rounds = problem radius (two-round vs one-round selection, same tournament selector)")
    R.p("Prediction: qa2/qa3 recall >= one-round + .05; qa1 within +-.03; qa2 acc >= +5pp. Falsified if qa2/qa3 recall change < .02.")
    R.p("Design-stage outcome (2026-09-02, before any grid): the copy-row mechanism carries no hop-2 signal. Qwen-3B 4k k=16 n=10 "
        "(logs/dbg_sel-415422.out): one-round vs two-round recall qa1 1.00/.86, qa2 .71/.47, qa3 .24/.15; per-layer scan of the copy rows "
        "(logs/dbg_rscan-415456.out): best layer qa2 .54 (L33), qa3 .19 (L25), s2 top-8 precision never above chance (.20/.52). "
        "The ROUNDS grid was therefore not launched: P6 is UNTESTED, not falsified — the second hop may still be the limit, but this mechanism cannot read it.")
    if len(osq):
        R.p("Observational stand-in (not pre-registered): one-round selector recall by hop count (qa1 = 1 hop, qa2 = 2, qa3 = 3), attn arm:")
        rc = osq[(osq.arm == "attn") & osq.task.isin(["qa1", "qa2", "qa3"])].groupby(["model", "length", "task"], observed=True).recall.mean().unstack("task").reset_index()
        rc["model"] = rc.model.map(MODELS)
        R.table(rc)
    if len(rounds) == 0:
        return
    rows, ver, fals = [], [], []
    for m in MODELS:
        for t in ("qa1", "qa2", "qa3"):
            for L in ("64k", "128k"):
                dr = paired(osq, rounds, "attn", "attn", "recall")
                dr = dr[(dr.model == m) & (dr.task == t) & (dr.length == L)]
                da = paired(osq, rounds, "attn", "attn", "ok")
                da = da[(da.model == m) & (da.task == t) & (da.length == L)]
                if len(dr) == 0:
                    continue
                rows.append({"model": MODELS[m], "task": t, "length": L, "n": len(dr), "one-round recall": ci(dr.a), "two-round recall": ci(dr.b),
                             "Δ recall": ci(dr.d), "one-round acc": ci(da.a), "two-round acc": ci(da.b), "Δ acc": ci(da.d)})
                if t == "qa1":
                    ver.append(abs(dr.d.mean()) <= .03)
                else:
                    ver.append(dr.d.mean() >= .05)
                    fals.append(dr.d.mean() < .02)
                if t == "qa2":
                    ver.append(da.d.mean() >= .05)
    R.table(pd.DataFrame(rows))
    R.p(verdict(ver, fals))


def p7(R: Report, ks, osq):
    R.h("P7 — k-identity (accuracy vs sentence budget at 128k; repack is degree control)")
    R.p("Prediction: acc rises with k while recall limits, then falls once recall >= .9; on the falling branch acc(attn@k) is within 10pp of "
        "acc(full) at the length with matching token count; keff_sent of the repacked read grows with k. Falsified if acc is flat in k beyond recall saturation.")
    if len(ks) == 0:
        R.p("_no data_")
        return
    ks = ks.copy()
    ks["k"] = ks.arm.map(lambda a: 64 if a == "attn" else int(a.split("@k")[1]) if "@k" in a else np.nan)
    ks = ks.dropna(subset=["k"])
    fullc = cell_means(osq, "full", ["tokens"])
    ver, fals = [], []
    for m in MODELS:
        for t in ("qa1", "qa2", "qa3"):
            s = ks[(ks.model == m) & (ks.task == t)]
            if len(s) == 0:
                continue
            g = s.groupby("k")[["ok", "recall", "keff_sent", "mass", "tokens"]].mean().reset_index()
            fm = fullc[(fullc.model == m) & (fullc.task == t)]
            g["matched full acc"] = [fm.iloc[(fm.tokens - tk).abs().argsort().iloc[0]].ok if len(fm) else np.nan for tk in g.tokens]
            g["matched length"] = [str(fm.iloc[(fm.tokens - tk).abs().argsort().iloc[0]].length) if len(fm) else "" for tk in g.tokens]
            R.p(f"**{MODELS[m]} {t}**")
            R.table(g[["k", "ok", "recall", "mass", "keff_sent", "tokens", "matched length", "matched full acc"]])
            sat = g[g.recall >= .9]
            if len(sat) >= 3:
                ver.append(sat.ok.iloc[-1] < sat.ok.iloc[0] - .02)          # falls beyond saturation
                fals.append(abs(sat.ok.max() - sat.ok.min()) < .03)        # flat in k beyond saturation
                ver.append(all(abs(r.ok - r["matched full acc"]) <= .10 for _, r in sat.iloc[1:].iterrows() if not np.isnan(r["matched full acc"])))
            ver.append(bool(np.all(np.diff(g.keff_sent.values) > 0)) if g.keff_sent.notna().all() else None)
    R.p(verdict(ver, fals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osq", default="outputs/babilong_osq")
    ap.add_argument("--yarn", default="outputs/babilong_osq_yarn4")
    ap.add_argument("--chunk", default="outputs/babilong_chunk")
    ap.add_argument("--one", default="outputs/babilong")
    ap.add_argument("--gap", default="outputs/babilong_gap")
    ap.add_argument("--gapyarn", default="outputs/babilong_gap_yarn4")
    ap.add_argument("--rounds", default="outputs/babilong_rounds")
    ap.add_argument("--ksweep", default="outputs/babilong_ksweep")
    ap.add_argument("--out", default="")
    ap.add_argument("--figdir", default="", help="write paper figures (png) here")
    a = ap.parse_args()
    L = lambda r: load(r) if Path(r).exists() else pd.DataFrame(columns=["model", "task", "length", "i", "arm", "ok", "recall"] + METR)
    osq, yarn, chunk, one = L(a.osq), L(a.yarn), L(a.chunk), L(a.one)
    R = Report()
    R.p(f"# OSQ pre-registered analysis — {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    R.p(f"cells: osq {osq.groupby(['model','task','length'], observed=True).ngroups if len(osq) else 0}, yarn {yarn.groupby(['model','task','length'], observed=True).ngroups if len(yarn) else 0}")
    p1(R, osq, chunk, one)
    p2(R, osq)
    p3(R, osq, yarn)
    p4(R, osq)
    p5(R, L(a.gap), L(a.gapyarn))
    p6(R, L(a.rounds), osq)
    p7(R, L(a.ksweep), osq)
    R.h("Appendix — per-cell arm table (probe grid)")
    if len(osq):
        t = osq.groupby(["model", "task", "length", "arm"], observed=True)[["ok", "recall", "mass", "keff_sent", "gap", "infl", "ctx_share"]].mean().reset_index()
        t["model"] = t.model.map(MODELS)
        R.table(t)
    txt = "\n".join(R.lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    if a.figdir:
        figures(Path(a.figdir), osq, L(a.gap), L(a.gapyarn), L(a.ksweep))


def figures(out: Path, osq, gap, gy, ks):
    """Four paper figures: fan-in/mass vs N per arm, dispersion law, position dose-response, k-sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    arms = {"full": "C0", "attn": "C2", "attn_norp": "C1", "oracle": "C7"}
    if len(osq):
        fig, ax = plt.subplots(2, len(MODELS), figsize=(4.2 * len(MODELS), 6), sharex=True)
        for j, m in enumerate(MODELS):
            for arm, col in arms.items():
                d = osq[(osq.model == m) & (osq.arm == arm) & osq.task.isin(["qa1", "qa2", "qa3"])]
                if len(d) == 0:
                    continue
                g = d.groupby("length", observed=True)[["keff_sent", "mass"]].mean()
                x = [LENS.index(str(l)) for l in g.index]
                ax[0, j].plot(x, g.keff_sent, "o-", color=col, label=arm)
                ax[1, j].plot(x, g.mass, "o-", color=col, label=arm)
            ax[0, j].set_title(MODELS[m]); ax[0, j].set_yscale("log"); ax[0, j].set_ylabel("k_eff (sentences)")
            ax[1, j].set_ylabel("evidence mass"); ax[1, j].set_xticks(range(len(LENS))); ax[1, j].set_xticklabels(LENS)
        ax[0, 0].legend(); fig.suptitle("Read-node fan-in and evidence mass vs context length (qa1-3 mean)")
        fig.tight_layout(); fig.savefig(out / "fig_fanin_vs_N.png", dpi=160); plt.close(fig)
        c = cell_means(osq, "full", ["mass", "pred_mass", "m_eff", "gap"]).dropna(subset=["pred_mass"])
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.8))
        for m in MODELS:
            cm = c[c.model == m]
            ax[0].scatter(cm.pred_mass, cm.mass, label=MODELS[m], s=18)
            ax[1].scatter(cm.gap, cm.m_eff, label=MODELS[m], s=18)
        ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("pred_mass (mean-margin law)"); ax[0].set_ylabel("measured mass")
        lim = [0, max(c.gap.max(), 1) * 1.05] if len(c) else [0, 1]
        ax[1].plot(lim, lim, "k--", lw=.8); ax[1].set_xlabel("gap (mean margin, nats)"); ax[1].set_ylabel("m_eff (mass-implied margin)")
        ax[0].legend(); fig.suptitle("Dispersion law: the mean-margin prediction vs the heavy-tailed reality")
        fig.tight_layout(); fig.savefig(out / "fig_dispersion_law.png", dpi=160); plt.close(fig)
    for name, d in (("native", gap), ("yarn4", gy)):
        if len(d) == 0:
            continue
        d = d.copy(); d["base"] = d.arm.str.partition("@")[0]; d["P"] = pd.to_numeric(d.arm.str.partition("@")[2], errors="coerce")
        d = d.dropna(subset=["P"])
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for m in MODELS:
            for base, ls in (("oracle_gap", "-"), ("oracle_pgap", ":")):
                s = d[(d.model == m) & (d.base == base) & d.task.isin(["qa1", "qa2"])]
                if len(s) == 0:
                    continue
                g = s.groupby("P")[["ok", "mass", "keff_sent"]].mean()
                ax[0].plot(g.index / 1000, g.ok, "o" + ls, ms=3, label=f"{MODELS[m]} {base}")
                ax[1].plot(g.index / 1000, g.mass, "o" + ls, ms=3, label=f"{MODELS[m]} {base}")
        ax[0].set_xlabel("gap P (k tokens)"); ax[0].set_ylabel("accuracy (qa1-2, oracle set)")
        ax[1].set_xlabel("gap P (k tokens)"); ax[1].set_ylabel("evidence mass at the read")
        ax[0].legend(fontsize=7); fig.suptitle(f"Position dose-response, degree fixed ({name})")
        fig.tight_layout(); fig.savefig(out / f"fig_dose_response_{name}.png", dpi=160); plt.close(fig)
    if len(ks):
        ks = ks.copy(); ks["k"] = ks.arm.map(lambda a: 64 if a == "attn" else int(a.split("@k")[1]) if "@k" in a else np.nan)
        ks = ks.dropna(subset=["k"])
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        for m in MODELS:
            for t in ("qa1", "qa2", "qa3"):
                s = ks[(ks.model == m) & (ks.task == t)]
                if len(s) == 0:
                    continue
                g = s.groupby("k")[["ok", "recall", "keff_sent"]].mean()
                ax[0].plot(g.index, g.ok, "o-", ms=3, label=f"{MODELS[m]} {t}")
                ax[1].plot(g.index, g.recall, "o-", ms=3); ax[2].plot(g.index, g.keff_sent, "o-", ms=3)
        for a_, yl in zip(ax, ("accuracy", "fact recall", "k_eff (sentences)")):
            a_.set_xscale("log", base=2); a_.set_xlabel("k (sentences kept)"); a_.set_ylabel(yl)
        ax[2].set_yscale("log"); ax[0].legend(fontsize=7); fig.suptitle("k-identity at 128k: the dispersion law inside the prompt")
        fig.tight_layout(); fig.savefig(out / "fig_ksweep.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
