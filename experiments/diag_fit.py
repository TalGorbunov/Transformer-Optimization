#!/usr/bin/env python3
"""CPU fitting library + CLI for the DIAG campaign (docs/DIAGNOSTICS_2026-09-22.md, sections 1.2,
1.3 and 2). Reads the CSVs written by the GPU probes and fits the quantities the record argues from:

  alpha     single-frame flip response   median ||dh|| ~ N^-alpha              (E1/E2/E10/E11)
  margins   logit margins, accuracy, flip-changes-answer rate, margin noise floor (E4)
  sharelaw  attention share law   m = k e^s / (k e^s + (N-k) + C)               (E7/E8/E17) and the
            record's prediction alpha_pred = -dlog[m(k+1,N)-m(k,N)]/dlogN (E9: 0.72 at s=.30, C=8, k=4)
  headscan  per-head evidence AUC population across N                          (E23)
  gamma     decay in k at fixed N   median ||dh|| ~ (k + C)^-gamma              (E20)
  mechform  flip curve fitted to the mechanistic form G * [m(k+1,N) - m(k,N)]  (D5)
  summary   markdown table of the alpha cells (+ margins)

Input schemas (produced by the probes; this module only reads them):
  flip run dir:        pairs.csv  qid,qtype,n_frames,gold,flip_t,flip_kind,arm,layer,locus,dnorm,base_norm
                       controls.csv  qid,qtype,n_frames,kind,arm,layer,locus,dnorm,base_norm  (kind replay|ctrl|perm)
                       logits.csv  qid,qtype,n_frames,gold,arm,flip_kind,margin_base,margin_flip,pred_base,pred_flip,dlogit
  photograph run dir:  mass.csv  qid,qtype,N,k,layer,head,evid_mass,nonevid_mass,other_mass
                       block_mass.csv  qid,qtype,N,k,layer,head,block,is_evid,mass
                       auc.json  {"L{L}_h{h}": auc}
  config.json in both: arm, qtypes, adapter, attn_sharpen, sharpen_from_layer, attn_logn_sref, ...
Legacy runs (outputs/armor/hahn, outputs/fixedk, outputs/sparse/s10|s3|s11) use sample_id / sample for
qid, have no qtype or arm column in some files and sparse configs; defaults: qtype steps_in_room, tau
off, sref off, model frozen, arm from config or "plain".

Cell key everywhere: (qtype, arm, cond, model) [+ layer, locus for flip cells] where cond is derived
from config ("base", "tau1.5L12", "logn3398", "tau2L12_logn3398"; = the probes' cond_tag) and model is "frozen" or the adapter's
basename. Conventions: median over pairs; OLS in log-log space; bootstrap resamples pairs (or samples)
WITHIN each N (or k) cell; all randomness seeded; missing N values tolerated; no division by zero.
Every subcommand writes --out (JSON) and a flat CSV next to it (nested dicts stay JSON-only).

Usage:
  python experiments/diag_fit.py alpha    --runs 'outputs/diag/d1/*' --out outputs/diag/fits/alpha.json
  python experiments/diag_fit.py margins  --runs 'outputs/diag/d1/*' --out outputs/diag/fits/margins.json
  python experiments/diag_fit.py sharelaw --runs 'outputs/diag/d2/*' --out outputs/diag/fits/sharelaw.json
  python experiments/diag_fit.py headscan --runs 'outputs/diag/d2/*' --out outputs/diag/fits/headscan.json --head L24_h20
  python experiments/diag_fit.py gamma    --runs 'outputs/sparse/s11/gate_oracle_N64_k*' --out outputs/diag/fits/gamma.json [--C 2]
  python experiments/diag_fit.py mechform --alpha alpha.json --sharelaw sharelaw.json --k 4 --out mechform.json
  python experiments/diag_fit.py summary  --alpha alpha.json [--margins margins.json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

N_PRED = (8, 16, 32, 64, 128)       # N grid of the record's alpha_pred (memory: share-law-predicts-alpha)
K_PRED = (1, 2, 4, 8)
GAMMA_C_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
KEY = ("qtype", "arm", "cond", "model")
FLIP_KEY = KEY + ("layer", "locus")
ARM_RANK = {"plain": 0, "frozen": 0, "qfirst": 1, "repjoint": 1, "fenced": 2, "p1fence": 2, "trained": 3, "gated": 4}
EPS = 1e-12
Table = Dict[str, np.ndarray]


# --------------------------------------------------------------------------------------------- io
def read_table(path) -> Table:
    """CSV -> {column: array}. Numeric columns become float arrays, the rest object (str) arrays."""
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        cols = [[] for _ in header]
        for row in rd:
            if row:
                for c, v in zip(cols, row):
                    c.append(v)
    out = {}
    for name, c in zip(header, cols):
        try:
            out[name] = np.array(c, dtype=float)
        except ValueError:
            out[name] = np.array(c, dtype=object)
    return out


def load_config(run_dir) -> dict:
    p = Path(run_dir) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cond_of(cfg: dict) -> str:
    """Intervention tag: config's own `cond` (written by the probes via _diag_common.cond_tag) or the same
    derivation from attn_sharpen / sharpen_from_layer / attn_logn_sref: base | tau<t>[L<from>] | logn<sref>."""
    if cfg.get("cond"):
        return str(cfg["cond"])
    parts = []
    tau, sref = float(cfg.get("attn_sharpen") or 0), int(cfg.get("attn_logn_sref") or 0)
    if tau > 0 and tau != 1.0:
        parts.append(f"tau{tau:g}" + (f"L{int(cfg['sharpen_from_layer'])}" if cfg.get("sharpen_from_layer") else ""))
    if sref > 0:
        parts.append(f"logn{sref}")
    return "_".join(parts) or "base"


def model_of(cfg: dict) -> str:
    ad = cfg.get("adapter") or cfg.get("peft_adapter")
    return Path(str(ad)).name if ad else "frozen"


def qtype_of(cfg: dict) -> Optional[str]:
    q = cfg.get("qtypes") or cfg.get("qtype")
    if isinstance(q, str):
        q = [s for s in re.split(r"[,\s]+", q) if s]
    return q[0] if isinstance(q, (list, tuple)) and len(q) == 1 else None


def n_of(cfg: dict) -> Optional[int]:
    for key in ("n", "N", "n_frames", "seq_len"):
        if cfg.get(key) is not None:
            return int(cfg[key])
    m = re.search(r"seq_len_(\d+)", str(cfg.get("config", "")) + str(cfg.get("data_root", "")))
    return int(m.group(1)) if m else None


def annotate(t: Table, cfg: dict, run_dir: str) -> Table:
    """Add the normalised columns qid, qtype, N, arm, cond, model, run (legacy names/defaults)."""
    n = len(next(iter(t.values())))
    qid = t.get("qid", t.get("sample_id", t.get("sample")))
    t["qid"] = np.array([str(v) for v in qid], dtype=object) if qid is not None else np.array([str(i) for i in range(n)], dtype=object)
    if "qtype" not in t:
        t["qtype"] = np.full(n, qtype_of(cfg) or "steps_in_room", dtype=object)
    nn = t.get("n_frames", t.get("N"))
    if nn is None:
        if n_of(cfg) is None:
            raise ValueError(f"{run_dir}: no N column and no N in config")
        nn = np.full(n, n_of(cfg), dtype=float)
    t["N"] = np.asarray(nn, dtype=float)
    if "arm" not in t:
        t["arm"] = np.full(n, str(cfg.get("arm") or "plain"), dtype=object)
    t["cond"] = np.full(n, cond_of(cfg), dtype=object)
    t["model"] = np.full(n, model_of(cfg), dtype=object)
    t["run"] = np.full(n, str(run_dir), dtype=object)
    return t


def concat(tables: List[Table]) -> Optional[Table]:
    if not tables:
        return None
    names = sorted(set().union(*[t.keys() for t in tables]))
    out = {}
    for name in names:
        parts = [t[name] if name in t else np.full(len(next(iter(t.values()))), "", dtype=object) for t in tables]
        col = np.concatenate([np.asarray(p, dtype=object) for p in parts])
        try:
            col = np.array([float(v) if v != "" else np.nan for v in col], dtype=float)
        except (ValueError, TypeError):
            pass
        out[name] = col
    return out


def run_dirs(patterns: Sequence[str]) -> List[str]:
    """Run dirs matching the globs, FINISHED ones only: a probe writes report.txt and a capture
    meta.json as its last act, so a dir without either is a job still running (partial CSVs)."""
    out = set()
    for p in patterns:
        for d in glob.glob(p):
            dp = Path(d)
            if dp.is_dir() and ((dp / "report.txt").exists() or (dp / "meta.json").exists()):
                out.add(d)
    return sorted(out)


def load_runs(dirs: Sequence[str], name: str) -> Optional[Table]:
    """Concatenate <run>/<name>.csv over run dirs (missing files skipped), annotated per run's config."""
    tables = []
    for d in dirs:
        p = Path(d) / f"{name}.csv"
        if p.exists():
            tables.append(annotate(read_table(p), load_config(d), d))
    return concat(tables)


def _keyvals(arr: np.ndarray) -> list:
    if arr.dtype.kind == "f":
        return [int(v) if np.isfinite(v) and v == int(v) else float(v) for v in arr]
    return [str(v) for v in arr]


def groups(t: Table, cols: Sequence[str], idx: Optional[np.ndarray] = None) -> Dict[tuple, np.ndarray]:
    """Row indices grouped by the tuple of `cols` (sorted keys; ints for integral floats)."""
    idx = np.arange(len(t[cols[0]])) if idx is None else np.asarray(idx)
    keys = list(zip(*[_keyvals(t[c][idx]) for c in cols]))
    d = defaultdict(list)
    for i, k in zip(idx, keys):
        d[k].append(i)
    return {k: np.array(v) for k, v in sorted(d.items(), key=lambda kv: tuple((0, v) if isinstance(v, (int, float)) else (1, str(v)) for v in kv[0]))}


def _py(o):
    """numpy -> plain python for json (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def _flat_rows(records: List[dict]) -> List[dict]:
    """One CSV row per list index: scalars repeated, equal-length lists expanded, nested dicts dropped."""
    rows = []
    for rec in records:
        scal = {k: v for k, v in rec.items() if not isinstance(v, (list, dict))}
        lists = {k: v for k, v in rec.items() if isinstance(v, list) and v and not isinstance(v[0], (list, dict))}
        if "ci" in lists:
            scal["ci_lo"], scal["ci_hi"] = lists.pop("ci")
        L = len(lists["N"]) if "N" in lists else (len(lists["k"]) if "k" in lists else 0)
        lists = {k: v for k, v in lists.items() if len(v) == L}
        rows += [dict(scal, **{k: v[i] for k, v in lists.items()}) for i in range(L)] if L else [scal]
    return rows


def write_out(obj: dict, out: str, records: List[dict]) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_py(obj), indent=1))
    rows = _flat_rows(_py(records))
    if rows:
        cols = list(dict.fromkeys(c for r in rows for c in r))
        with open(out.with_suffix(".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


# ------------------------------------------------------------------------------------------ stats
def ols(x, Y):
    """OLS of Y[..., n] on x[n]: (slope, r2), vectorised over leading dims; nan when degenerate."""
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    xm = x - x.mean()
    sxx = float((xm ** 2).sum())
    if sxx <= 0 or len(x) < 2:
        return np.full(Y.shape[:-1], np.nan), np.full(Y.shape[:-1], np.nan)
    ym = Y - Y.mean(-1, keepdims=True)
    slope = (ym * xm).sum(-1) / sxx
    ss_res = ((ym - slope[..., None] * xm) ** 2).sum(-1)
    ss_tot = (ym ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    return slope, r2


def _log(v):
    return np.log(np.maximum(np.asarray(v, float), EPS))


def _boot_medians(groups_: List[np.ndarray], n_boot: int, rng) -> np.ndarray:
    """(n_boot, len(groups)) medians of within-group resamples."""
    out = np.empty((n_boot, len(groups_)))
    for j, g in enumerate(groups_):
        out[:, j] = np.median(g[rng.randint(0, len(g), size=(n_boot, len(g)))], axis=1)
    return out


def fit_power(by_x: Dict[float, np.ndarray], n_boot: int = 2000, seed: int = 0, min_pts: int = 3) -> Optional[dict]:
    """median(x) ~ x^-alpha by OLS in log-log; bootstrap CI resamples within each x. None if < min_pts x."""
    xs = sorted(x for x, g in by_x.items() if len(g) > 0)
    if len(xs) < min_pts:
        return None
    gs = [np.asarray(by_x[x], float) for x in xs]
    med = np.array([np.median(g) for g in gs])
    slope, r2 = ols(np.log(np.asarray(xs, float)), _log(med))
    boots = _boot_medians(gs, n_boot, np.random.RandomState(seed))
    bs, _ = ols(np.log(np.asarray(xs, float)), _log(boots))
    return {"N": xs, "median": med, "iqr_lo": [np.percentile(g, 25) for g in gs], "iqr_hi": [np.percentile(g, 75) for g in gs],
            "n": [len(g) for g in gs], "alpha": -float(slope), "r2": float(r2), "ci": list(np.percentile(-bs, [2.5, 97.5]))}


# ------------------------------------------------------------------------------------- share law
def law(X, s, C, B=1.0):
    """Evidence share m(N,k) = k e^s / (k e^s + (N-k)/B + C); B=1 ungated, B=inf gated (leaky gate = 1/B)."""
    N, k = np.asarray(X[0], float), np.asarray(X[1], float)
    comp = np.zeros_like(N) if np.isinf(B) else (N - k) / B
    return k * np.exp(s) / (k * np.exp(s) + comp + C)


LAWS = {"ungated": lambda X, s, C: law(X, s, C, 1.0), "gated": lambda X, s, C: law(X, s, C, np.inf)}


def alpha_pred(s: float, C: float, k: int, Ns=N_PRED, B: float = 1.0) -> float:
    """The record's prediction: -dlog[m(k+1,N) - m(k,N)]/dlogN over Ns (0.716 at s=.30, C=8, k=4, B=1)."""
    Ns = np.asarray(Ns, float)
    dm = law((Ns, np.full_like(Ns, k + 1)), s, C, B) - law((Ns, np.full_like(Ns, k)), s, C, B)
    if np.any(dm <= 0) or len(Ns) < 2:
        return float("nan")
    return -float(ols(np.log(Ns), np.log(dm))[0])


def fit_law(N, k, m, form: str = "ungated", p0=(0.3, 8.0)) -> dict:
    """curve_fit of one share-law form over (N,k) cells; bounds s in [-6,6], C in [0,5000]."""
    N, k, m = (np.asarray(v, float) for v in (N, k, m))
    nan = {"s": np.nan, "C": np.nan, "r2": np.nan}
    if len(m) < 3:
        return nan
    try:
        popt, _ = curve_fit(LAWS[form], (N, k), m, p0=list(p0), bounds=([-6, 0], [6, 5000]), maxfev=20000)
    except (RuntimeError, ValueError):
        return nan
    ss_res = float(((m - LAWS[form]((N, k), *popt)) ** 2).sum())
    ss_tot = float(((m - m.mean()) ** 2).sum())
    return {"s": float(popt[0]), "C": float(popt[1]), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def fit_law_boot(cells: Dict[tuple, np.ndarray], form: str, n_boot: int = 500, seed: int = 0) -> dict:
    """Point fit + bootstrap CI (resample SAMPLES within each (N,k) cell, refit). cells: (N,k) -> per-sample means.
    Returns s, C, r2, ci_s, ci_C, n_cells, C_eff (= C e^-s, the gated form's only identifiable combination)."""
    keys = sorted(cells)
    N = np.array([kk[0] for kk in keys], float)
    k = np.array([kk[1] for kk in keys], float)
    m = np.array([cells[kk].mean() for kk in keys])
    fit = fit_law(N, k, m, form)
    fit.update({"ci_s": [np.nan, np.nan], "ci_C": [np.nan, np.nan], "n_cells": len(keys), "C_eff": fit["C"] * np.exp(-fit["s"])})
    # gated form: m = k / (k + C e^-s) -> only C_eff = C e^-s is identifiable (the record's s'=.26, C'=46 is one point on that ridge)
    if not np.isfinite(fit["s"]):
        return fit
    rng = np.random.RandomState(seed)
    bs, bC = [], []
    for _ in range(n_boot):
        mb = np.array([cells[kk][rng.randint(0, len(cells[kk]), len(cells[kk]))].mean() for kk in keys])
        f = fit_law(N, k, mb, form, p0=(fit["s"], fit["C"]))
        if np.isfinite(f["s"]):
            bs.append(f["s"])
            bC.append(f["C"])
    if bs:
        fit["ci_s"] = list(np.percentile(bs, [2.5, 97.5]))
        fit["ci_C"] = list(np.percentile(bC, [2.5, 97.5]))
    return fit


# ------------------------------------------------------------------------------------- A. alpha
def _floors(controls: Optional[Table], key: tuple) -> dict:
    """Per control kind: pooled median/max and per-N median of dnorm for one flip cell."""
    if controls is None:
        return {}
    out = {}
    for (kind,), idx in groups(controls, ("kind",)).items():
        keep = np.ones(len(idx), bool)
        for c, v in zip(FLIP_KEY, key):
            keep &= np.array([kv == v for kv in _keyvals(controls[c][idx])])
        sel = idx[keep]
        if len(sel) == 0:
            continue
        d = controls["dnorm"][sel]
        per_n = {int(n): float(np.median(controls["dnorm"][ii])) for (n,), ii in groups(controls, ("N",), sel).items()}
        out[kind] = {"median": float(np.median(d)), "max": float(np.max(d)), "per_N": per_n}
    return out


def fit_alpha(pairs: Table, controls: Optional[Table] = None, n_boot: int = 2000, seed: int = 0, min_gold_pairs: int = 5) -> dict:
    """alpha per (qtype, arm, cond, model, layer, locus) from evid flips; floors; by_gold; by_position."""
    evid = pairs["flip_kind"] != "ctrl" if "flip_kind" in pairs else np.ones(len(pairs["N"]), bool)
    cells = []
    for key, idx in groups(pairs, FLIP_KEY, np.flatnonzero(evid)).items():
        by_n = {n: pairs["dnorm"][ii] for (n,), ii in groups(pairs, ("N",), idx).items()}
        fit = fit_power(by_n, n_boot, seed)
        if fit is None:
            continue
        cell = dict(zip(FLIP_KEY, key), **fit)
        cell["floors"] = _floors(controls, key)
        cell["by_gold"] = {}
        if "gold" in pairs:
            for (g,), gi in groups(pairs, ("gold",), idx).items():
                sub = {n: pairs["dnorm"][ii] for (n,), ii in groups(pairs, ("N",), gi).items() if len(ii) >= min_gold_pairs}
                f = fit_power(sub, n_boot, seed)
                if f is not None:
                    cell["by_gold"][str(g)] = f
        cell["by_position"] = {}
        if "flip_t" in pairs:
            pos = pairs["flip_t"][idx] / np.maximum(pairs["N"][idx], 1)
            terc = np.where(pos < 1 / 3, "first", np.where(pos < 2 / 3, "middle", "last"))
            for name in ("first", "middle", "last"):
                ti = idx[terc == name]
                f = fit_power({n: pairs["dnorm"][ii] for (n,), ii in groups(pairs, ("N",), ti).items()}, 200, seed) if len(ti) else None
                if f is not None:
                    cell["by_position"][name] = {k: f[k] for k in ("N", "median", "n", "alpha", "r2")}
        bg = {g: f["alpha"] for g, f in cell["by_gold"].items()}
        cell["pooled_vs_stratified"] = (f"pooled alpha {fit['alpha']:.2f} [{fit['ci'][0]:.2f}, {fit['ci'][1]:.2f}]; "
                                        + (f"by-gold alpha {min(bg.values()):.2f}..{max(bg.values()):.2f} over golds {sorted(bg)}"
                                           if bg else f"no gold stratum with >= {min_gold_pairs} pairs at >= 3 N"))
        cells.append(cell)
    return {"cells": cells}


# ----------------------------------------------------------------------------------- B. margins
_LABELS = {"digits": [str(d) for d in range(10)],
           "rooms": ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"],
           "people": ["Sandra", "Mary", "John", "Daniel", "Michael", "Nobody"]}
_ROOM_Q = {"first_app", "final_app", "char_on_char_first_app", "char_on_char_final_app", "char_at_frame",
           "room_empty", "where_spend", "crowded_room"}
_PEOPLE_Q = {"first_at_room", "last_at_room", "room_on_char_first_app", "room_on_char_final_app", "room_at_frame",
             "char_on_char_at_frame", "who_spend", "spend_alone", "spend_together"}


def _label_of(qtype, pred) -> str:
    """probe_hahn writes pred_* as an INDEX into the qtype's answer vocabulary (digits / rooms / people,
    in core.constants order); map it back to the label so it can be compared with the gold string."""
    try:
        i = int(float(pred))
    except (TypeError, ValueError):
        return str(pred)
    vocab = _LABELS["rooms"] if qtype in _ROOM_Q else _LABELS["people"] if qtype in _PEOPLE_Q else _LABELS["digits"]
    return vocab[i] if 0 <= i < len(vocab) else str(pred)


def _same(a, b, qtype=None) -> bool:
    if qtype is not None:
        a = _label_of(qtype, a)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def fit_margins(logits: Table) -> dict:
    """Per (qtype, arm, cond, model) per N: base margin median/IQR, accuracy, flip-changes-answer, noise floor."""
    kind = logits["flip_kind"] if "flip_kind" in logits else np.full(len(logits["N"]), "evid", dtype=object)
    cells = []
    for key, idx in groups(logits, KEY).items():
        cell = dict(zip(KEY, key), N=[], margin_median=[], iqr_lo=[], iqr_hi=[], accuracy=[], flip_changes_answer=[],
                    dmargin_evid=[], noise_floor=[], n=[])
        for (n,), ii in groups(logits, ("N",), idx).items():
            _, first = np.unique(logits["qid"][ii], return_index=True)      # one base row per pair
            base = ii[first]
            ev, ct = ii[kind[ii] != "ctrl"], ii[kind[ii] == "ctrl"]
            mb = logits["margin_base"][base]
            dm = np.abs(logits["margin_flip"] - logits["margin_base"])
            cell["N"].append(int(n))
            cell["margin_median"].append(float(np.median(mb)))
            cell["iqr_lo"].append(float(np.percentile(mb, 25)))
            cell["iqr_hi"].append(float(np.percentile(mb, 75)))
            cell["accuracy"].append(float(np.mean([_same(a, b, q) for a, b, q in zip(logits["pred_base"][base], logits["gold"][base], logits["qtype"][base])])) if "gold" in logits else np.nan)
            cell["flip_changes_answer"].append(float(np.mean([not _same(a, b) for a, b in zip(logits["pred_flip"][ev], logits["pred_base"][ev])])) if len(ev) else np.nan)
            cell["dmargin_evid"].append(float(np.median(dm[ev])) if len(ev) else np.nan)
            cell["noise_floor"].append(float(np.median(dm[ct])) if len(ct) else np.nan)
            cell["n"].append(int(len(base)))
        cross = [n for n, m in zip(cell["N"], cell["margin_median"]) if m <= 0]
        cell["n_cross_zero"] = cross[0] if cross else None
        cells.append(cell)
    return {"cells": cells}


# ---------------------------------------------------------------------------------- C. sharelaw
def perframe_from_blocks(blocks: Table, idx: np.ndarray) -> dict:
    """Per (N,k): median over (sample, head) of mean per-block mass on evidence / non-evidence blocks,
    their ratio, the within-evidence concentration max/mean; log-log slopes vs N (pooled and per k)."""
    N, k, ev, mass = (np.asarray(blocks[c][idx], float) for c in ("N", "k", "is_evid", "mass"))
    head = np.asarray(blocks["head"][idx], float) if "head" in blocks else np.zeros(len(idx))
    q = np.unique(np.asarray(blocks["qid"][idx], dtype=str), return_inverse=True)[1].ravel()
    uniq, g = np.unique(np.stack([N, k, q, head], 1), axis=0, return_inverse=True)
    g = g.ravel()
    G = len(uniq)
    cnt_e, cnt_n = np.bincount(g, ev, G), np.bincount(g, 1 - ev, G)
    mean_e = np.bincount(g, mass * ev, G) / np.maximum(cnt_e, 1)
    mean_n = np.bincount(g, mass * (1 - ev), G) / np.maximum(cnt_n, 1)
    max_e = np.zeros(G)
    np.maximum.at(max_e, g[ev > 0], mass[ev > 0])
    conc = np.where(mean_e > 0, max_e / np.maximum(mean_e, EPS), np.nan)
    out = {"N": [], "k": [], "evid": [], "nonevid": [], "ratio": [], "conc": [], "n": []}
    for (n, kk) in sorted({(int(r[0]), int(r[1])) for r in uniq}):
        sel = (uniq[:, 0] == n) & (uniq[:, 1] == kk)
        e, ne = np.median(mean_e[sel]), np.median(mean_n[sel])
        out["N"].append(n); out["k"].append(kk); out["evid"].append(float(e)); out["nonevid"].append(float(ne))
        out["ratio"].append(float(e / ne) if ne > 0 else np.nan)
        out["conc"].append(float(np.nanmedian(conc[sel])) if np.any(np.isfinite(conc[sel])) else np.nan)
        out["n"].append(int(sel.sum()))
    pooled_n = sorted(set(out["N"]))
    pe = [float(np.median(mean_e[uniq[:, 0] == n])) for n in pooled_n]
    pn = [float(np.median(mean_n[uniq[:, 0] == n])) for n in pooled_n]
    out["evid_slope"] = float(ols(np.log(pooled_n), _log(pe))[0]) if len(pooled_n) >= 2 else np.nan
    out["nonevid_slope"] = float(ols(np.log(pooled_n), _log(pn))[0]) if len(pooled_n) >= 2 else np.nan
    out["slopes_by_k"] = {}
    for kk in sorted(set(out["k"])):
        rows = [i for i, v in enumerate(out["k"]) if v == kk]
        if len(rows) >= 2:
            ln = np.log([out["N"][i] for i in rows])
            out["slopes_by_k"][str(kk)] = {"evid_slope": float(ols(ln, _log([out["evid"][i] for i in rows]))[0]),
                                           "nonevid_slope": float(ols(ln, _log([out["nonevid"][i] for i in rows]))[0])}
    return out


def fit_sharelaw(mass: Table, blocks: Optional[Table] = None, n_boot: int = 500, seed: int = 0) -> dict:
    """Per (qtype, arm, cond, model, layer): (N,k) cell masses (mean over heads then samples), both law
    forms fitted with bootstrap CIs, alpha_pred from the primary form, per-frame ladder from block_mass."""
    fits = []
    block_groups = groups(blocks, KEY + ("layer",)) if blocks is not None else {}
    for key, idx in groups(mass, KEY + ("layer",)).items():
        rec = dict(zip(KEY + ("layer",), key))
        cells, samples = [], {}
        for (n, k), ci in groups(mass, ("N", "k"), idx).items():
            per_sample = {c: np.array([mass[c][ii].mean() for (_,), ii in groups(mass, ("qid",), ci).items()])
                          for c in ("evid_mass", "nonevid_mass", "other_mass") if c in mass}
            e = per_sample["evid_mass"]
            samples[(n, k)] = e
            cell = {"N": int(n), "k": int(k), "evid": float(e.mean()), "evid_median": float(np.median(e)),
                    "nonevid": float(per_sample["nonevid_mass"].mean()) if "nonevid_mass" in per_sample else np.nan,
                    "other": float(per_sample["other_mass"].mean()) if "other_mass" in per_sample else np.nan, "n": int(len(e))}
            # frame-only decomposition (identifiable from mass.csv alone): the prompt+sink takes a FRACTION of the
            # mass (constant in N on the official prompt), the frames split the rest; the per-frame edge
            # e^s = (evid/k) / (nonevid/(N-k)) is read directly, no 2-parameter fit needed.
            if "nonevid_mass" in per_sample and 0 < int(k) < int(n):
                ne = per_sample["nonevid_mass"]
                edge = (e / int(k)) / (ne / (int(n) - int(k)))
                edge = edge[np.isfinite(edge) & (edge > 0)]
                cell["edge"] = float(np.mean(edge)) if edge.size else np.nan
                cell["edge_median"] = float(np.median(edge)) if edge.size else np.nan
                cell["s_frame"] = float(np.log(np.mean(edge))) if edge.size else np.nan
                cell["frame_share"] = float(np.mean(e / (e + ne))) if edge.size else np.nan
            cells.append(cell)
        primary = "gated" if "gated" in str(rec["arm"]) else "ungated"
        both = {form: fit_law_boot(samples, form, n_boot, seed) for form in ("ungated", "gated")}
        rec.update(form=primary, **both[primary])
        rec["fit_ungated"], rec["fit_gated"] = both["ungated"], both["gated"]
        r2s = {f: both[f]["r2"] for f in both if np.isfinite(both[f]["r2"])}
        rec["best_form"] = max(r2s, key=r2s.get) if r2s else None
        B = np.inf if primary == "gated" else 1.0
        rec["alpha_pred"] = {str(k): alpha_pred(rec["s"], rec["C"], k, B=B) for k in K_PRED} if np.isfinite(rec["s"]) else {}
        rec["cells"] = cells
        rec["perframe"] = perframe_from_blocks(blocks, block_groups[key]) if key in block_groups else {}
        # frame-law summary per N (pooled over k cells, weighted by n): edge, s_frame, sink fraction
        by_n = defaultdict(list)
        for c in cells:
            if np.isfinite(c.get("edge", np.nan)):
                by_n[c["N"]].append(c)
        fl = {"N": sorted(by_n), "edge": [], "s_frame": [], "sink": [], "frame_share": []}
        for n in fl["N"]:
            w = np.array([c["n"] for c in by_n[n]], float)
            fl["edge"].append(float(np.average([c["edge"] for c in by_n[n]], weights=w)))
            fl["s_frame"].append(float(np.log(fl["edge"][-1])))
            fl["sink"].append(float(np.average([c["other"] for c in by_n[n]], weights=w)))
            fl["frame_share"].append(float(np.average([c["frame_share"] for c in by_n[n]], weights=w)))
        if fl["N"]:
            fl["s_frame_mean"] = float(np.mean(fl["s_frame"]))
            fl["sink_mean"] = float(np.mean(fl["sink"]))
            fl["alpha_pred_frame"] = {str(k): alpha_pred(fl["s_frame_mean"], 0.0, k, B=1.0) for k in K_PRED}
        rec["frame_law"] = fl
        fits.append(rec)
    return {"fits": fits}


# ---------------------------------------------------------------------------------- D. headscan
def headscan(runs: List[dict], named: str = "L24_h20", thresh: float = 0.98) -> dict:
    """runs: [{qtype, arm, cond, model, N, auc: {"L{L}_h{h}": auc}}] -> per group: AUC per head per N,
    best head per layer per N, population of heads >= thresh per N, the named head's AUC per N."""
    grp = defaultdict(lambda: defaultdict(list))
    for r in runs:
        grp[(r["qtype"], r["arm"], r["cond"], r["model"])][int(r["N"])].append(r["auc"])
    out = []
    for key, by_n in sorted(grp.items()):
        rec = dict(zip(KEY, key), N=sorted(by_n), auc=defaultdict(dict), best_per_layer={}, population={}, named={"key": named, "auc": {}})
        for n in rec["N"]:
            merged = defaultdict(list)
            for d in by_n[n]:
                for h, v in d.items():
                    merged[h].append(float(v))
            aucs = {h: float(np.mean(v)) for h, v in merged.items()}
            for h, v in aucs.items():
                rec["auc"][h][str(n)] = v
            best = {}
            for h, v in aucs.items():
                m = re.match(r"L(\d+)_h(\d+)", h)
                L = m.group(1) if m else h.split("_")[0]
                if L not in best or v > best[L][1]:
                    best[L] = [h, v]
            rec["best_per_layer"][str(n)] = best
            pop = sorted([h for h, v in aucs.items() if v >= thresh], key=lambda h: -aucs[h])
            rec["population"][str(n)] = {"count": len(pop), "heads": pop, "n_heads": len(aucs)}
            if named in aucs:
                rec["named"]["auc"][str(n)] = aucs[named]
        rec["auc"] = dict(rec["auc"])
        out.append(rec)
    return {"groups": out, "thresh": thresh}


def load_headscan_runs(dirs: Sequence[str]) -> List[dict]:
    runs = []
    for d in dirs:
        p = Path(d) / "auc.json"
        if not p.exists():
            continue
        cfg = load_config(d)
        n, qt = n_of(cfg), qtype_of(cfg)
        if (n is None or qt is None) and (Path(d) / "mass.csv").exists():
            t = annotate(read_table(Path(d) / "mass.csv"), cfg, d)
            n, qt = n or int(np.median(t["N"])), qt or str(t["qtype"][0])
        runs.append({"qtype": qt or "steps_in_room", "arm": str(cfg.get("arm") or "plain"), "cond": cond_of(cfg),
                     "model": model_of(cfg), "N": n, "auc": json.loads(p.read_text())})
    return runs


# ------------------------------------------------------------------------------------- E. gamma
def fit_gamma(pairs: Table, n_boot: int = 2000, seed: int = 0, C_grid=GAMMA_C_GRID, C_fixed: Optional[float] = None) -> dict:
    """At fixed (qtype, arm, cond, model, layer, locus, N) with >= 3 base-gold values k: median dnorm per k,
    Delta ~ (k + C)^-gamma by OLS in log space over a grid of C (best R2), bootstrap CI (resample within k,
    C re-selected per iteration). C_fixed pins C (the record's S3/S11 gamma = 1.19 used a fixed offset)."""
    if C_fixed is not None:
        C_grid = (float(C_fixed),)
    evid = pairs["flip_kind"] != "ctrl" if "flip_kind" in pairs else np.ones(len(pairs["N"]), bool)
    cells = []
    for key, idx in groups(pairs, FLIP_KEY + ("N",), np.flatnonzero(evid)).items():
        by_k = {int(k): pairs["dnorm"][ii] for (k,), ii in groups(pairs, ("gold",), idx).items() if len(ii) > 0}
        ks = sorted(by_k)
        if len(ks) < 3:
            continue
        gs = [np.asarray(by_k[k], float) for k in ks]
        med = np.array([np.median(g) for g in gs])
        boots = _boot_medians(gs, n_boot, np.random.RandomState(seed))
        grid, best, boot_slopes, boot_r2 = {}, None, [], []
        for C in C_grid:
            if C == 0 and min(ks) <= 0:
                continue
            lx = np.log(np.asarray(ks, float) + C)
            slope, r2 = ols(lx, _log(med))
            grid[str(C)] = {"gamma": -float(slope), "r2": float(r2)}
            if best is None or r2 > best[2]:
                best = (C, -float(slope), float(r2))
            bs, br = ols(lx, _log(boots))
            boot_slopes.append(-bs)
            boot_r2.append(br)
        pick = np.argmax(np.nan_to_num(np.stack(boot_r2), nan=-np.inf), axis=0)
        gam_b = np.stack(boot_slopes)[pick, np.arange(n_boot)]
        cells.append(dict(zip(FLIP_KEY + ("N",), key), k=ks, median=med, iqr_lo=[np.percentile(g, 25) for g in gs],
                          iqr_hi=[np.percentile(g, 75) for g in gs], n=[len(g) for g in gs], gamma=best[1], C=best[0],
                          r2=best[2], ci=list(np.percentile(gam_b, [2.5, 97.5])), grid=grid))
    return {"cells": cells}


# ---------------------------------------------------------------------------------- F. mechform
def _dm(N, k, s, C):
    N = np.asarray(N, float)
    return law((N, np.full_like(N, k + 1)), s, C) - law((N, np.full_like(N, k)), s, C)


def fit_mechform_cell(N, median, s0: float, C0: float, k: int = 4) -> dict:
    """log median(N) ~ log G + log[m(k+1,N) - m(k,N)] (ungated law): (G,s,C) free from (s0,C0), and G only
    with s,C fixed. R2 in log space (comparable to alpha's r2); the fixed-G fit is closed form. With 5 N values
    the free fit is often degenerate (G and C trade off): free["dr2"] << 0.01 says so."""
    N, ly = np.asarray(N, float), _log(median)
    lf0 = _log(_dm(N, k, s0, C0))
    logG = float(np.mean(ly - lf0))
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = lambda pred: 1 - float(((ly - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    out = {"k": k, "s0": s0, "C0": C0, "fixed": {"G": float(np.exp(logG)), "r2": r2(logG + lf0)}, "free": None}
    if len(N) >= 4:
        f = lambda n, lg, s, C: lg + _log(_dm(n, k, s, C))
        try:
            popt, _ = curve_fit(f, N, ly, p0=[logG, s0, C0], bounds=([-50, -6, 0], [50, 6, 5000]), maxfev=20000)
            out["free"] = {"G": float(np.exp(popt[0])), "s": float(popt[1]), "C": float(popt[2]), "r2": r2(f(N, *popt)),
                           "ds": float(popt[1] - s0), "dC": float(popt[2] - C0), "dr2": r2(f(N, *popt)) - out["fixed"]["r2"],
                           "at_bound": bool(abs(popt[1]) > 5.99 or popt[2] < 1e-3 or popt[2] > 4999)}
            # dr2 ~ 0 means (G, s, C) are not separately identifiable on this curve (G-C trade-off): quote the fixed fit
        except (RuntimeError, ValueError):
            pass
    return out


def _arm_order(rec: dict):
    return (0 if rec.get("model") == "frozen" else 1, ARM_RANK.get(str(rec.get("arm")), 9), str(rec.get("arm")), str(rec.get("model")))


def fit_mechform(alpha: dict, sharelaw: Optional[dict] = None, k: int = 4, s: Optional[float] = None, C: Optional[float] = None) -> dict:
    """Every alpha cell fitted to the mechanistic form using the matching sharelaw (s,C) (same
    qtype/arm/cond/model/layer, else same qtype+layer, else same qtype, else first fit, else --s/--C)."""
    fits = [f for f in (sharelaw or {}).get("fits", []) if f.get("s") is not None]

    def pick(cell):
        for cols in (KEY + ("layer",), ("qtype", "layer"), ("qtype",), ()):
            m = [f for f in fits if all(str(f.get(c)) == str(cell.get(c)) for c in cols)]
            if m:
                return m[0]["s"], m[0]["C"], dict(zip(cols, [m[0].get(c) for c in cols]))
        return s, C, {"source": "--s/--C"}

    cells = []
    for cell in alpha["cells"]:
        s0, C0, src = pick(cell)
        if s0 is None or C0 is None:
            continue
        rec = {c: cell[c] for c in FLIP_KEY}
        rec.update(fit_mechform_cell(cell["N"], cell["median"], s0, C0, k), alpha=cell["alpha"], alpha_r2=cell["r2"], sharelaw_match=src)
        cells.append(rec)
    cells.sort(key=lambda c: (c["qtype"], c["cond"], str(c["layer"]), c["locus"]) + _arm_order(c))
    ladder, notes = [], []
    for c in cells:
        ladder.append({"qtype": c["qtype"], "cond": c["cond"], "layer": c["layer"], "locus": c["locus"], "arm": c["arm"], "model": c["model"],
                       "G_fixed": c["fixed"]["G"], "r2_fixed": c["fixed"]["r2"],
                       **({"G_free": c["free"]["G"], "s_free": c["free"]["s"], "C_free": c["free"]["C"], "r2_free": c["free"]["r2"],
                           "at_bound": c["free"]["at_bound"]} if c["free"] else {})})
    for a, b in zip(ladder, ladder[1:]):          # consecutive rungs of the same (qtype, cond, layer, locus) ladder
        if all(a[f] == b[f] for f in ("qtype", "cond", "layer", "locus")) and "G_free" in a and "G_free" in b:
            flag = " [free fit at bound: quote G_fixed]" if a["at_bound"] or b["at_bound"] else ""
            notes.append(f"{a['qtype']} {a['cond']} L{b['layer']} {b['locus']}: {a['arm']}/{a['model']} -> {b['arm']}/{b['model']}: "
                         f"G_fixed x{b['G_fixed'] / max(a['G_fixed'], EPS):.2f}; free G x{b['G_free'] / max(a['G_free'], EPS):.2f}, "
                         f"s {b['s_free'] - a['s_free']:+.2f}, C {b['C_free'] - a['C_free']:+.1f}{flag}")
    return {"k": k, "cells": cells, "ladder": ladder, "which_moves": notes}


# ---------------------------------------------------------------------------------- G. summary
def summary_table(alpha: dict, margins: Optional[dict] = None) -> str:
    lines = ["| qtype | arm | model | cond | layer | locus | alpha [CI] | r2 | floors ctrl/perm/replay (median) | n per N |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(alpha["cells"], key=lambda c: (_arm_order(c), str(c["layer"]), c["locus"])):
        fl = "/".join(f"{c['floors'][k]['median']:.2f}" if k in c["floors"] else "-" for k in ("ctrl", "perm", "replay"))
        npn = " ".join(f"{n}:{m}" for n, m in zip(c["N"], c["n"]))
        ci = c.get("ci") or [None, None]
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "[-]"
        lines.append(f"| {c['qtype']} | {c['arm']} | {c['model']} | {c['cond']} | {c['layer']} | {c['locus']} | "
                     f"{c['alpha']:.3f} {cis} | {c['r2']:.2f} | {fl} | {npn} |")
    if margins:
        lines += ["", "| qtype | arm | model | cond | N | margin median [IQR] | accuracy | flip changes answer | noise floor | n |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(margins["cells"], key=_arm_order):
            for i, n in enumerate(c["N"]):
                f = lambda v: "-" if v is None else f"{v:.2f}"
                lines.append(f"| {c['qtype']} | {c['arm']} | {c['model']} | {c['cond']} | {n} | {f(c['margin_median'][i])} "
                             f"[{f(c['iqr_lo'][i])}, {f(c['iqr_hi'][i])}] | {f(c['accuracy'][i])} | {f(c['flip_changes_answer'][i])} | "
                             f"{f(c['noise_floor'][i])} | {c['n'][i]} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- CLI
def _load_json(p):
    return json.loads(Path(p).read_text()) if p else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("alpha", "margins", "sharelaw", "headscan", "gamma"):
        sp = sub.add_parser(name)
        sp.add_argument("--runs", nargs="+", required=True, help="glob pattern(s) of run dirs")
        sp.add_argument("--out", required=True, help="JSON path (a flat CSV is written next to it)")
        sp.add_argument("--n-boot", type=int, default=None)
        sp.add_argument("--seed", type=int, default=0)
        if name == "headscan":
            sp.add_argument("--head", default="L24_h20", help="named head, legacy layer numbering (raw auc.json key)")
            sp.add_argument("--thresh", type=float, default=0.98)
        if name == "gamma":
            sp.add_argument("--C", type=float, default=None, help="pin the offset C instead of the grid search")
    sp = sub.add_parser("mechform")
    sp.add_argument("--alpha", required=True)
    sp.add_argument("--sharelaw", default=None)
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--s", type=float, default=None, help="fallback s when no sharelaw fit matches")
    sp.add_argument("--C", type=float, default=None)
    sp.add_argument("--out", required=True)
    sp = sub.add_parser("summary")
    sp.add_argument("--alpha", required=True)
    sp.add_argument("--margins", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "summary":
        print(summary_table(_load_json(a.alpha), _load_json(a.margins)))
        return 0
    if a.cmd == "mechform":
        res = fit_mechform(_load_json(a.alpha), _load_json(a.sharelaw), a.k, a.s, a.C)
        write_out(res, a.out, [{**{c: r[c] for c in FLIP_KEY}, "k": r["k"], "s0": r["s0"], "C0": r["C0"], "alpha": r["alpha"],
                                "G_fixed": r["fixed"]["G"], "r2_fixed": r["fixed"]["r2"],
                                **({f"{p}_free": r["free"][p] for p in ("G", "s", "C", "r2")} if r["free"] else {})} for r in res["cells"]])
        for line in res["which_moves"]:
            print(line)
        print(f"mechform: {len(res['cells'])} cells -> {a.out}")
        return 0

    dirs = run_dirs(a.runs)
    if not dirs:
        print(f"no run dirs match {a.runs}", file=sys.stderr)
        return 1
    if a.cmd == "alpha":
        pairs = load_runs(dirs, "pairs")
        if pairs is None:
            print("no pairs.csv found", file=sys.stderr)
            return 1
        res = fit_alpha(pairs, load_runs(dirs, "controls"), a.n_boot or 2000, a.seed)
        write_out(res, a.out, [{k: v for k, v in c.items() if k not in ("by_gold", "by_position", "floors", "pooled_vs_stratified")} for c in res["cells"]])
        for c in res["cells"]:
            print(f"{c['qtype']} {c['arm']} {c['model']} {c['cond']} L{c['layer']} {c['locus']}: alpha {c['alpha']:.3f} "
                  f"[{c['ci'][0]:.2f}, {c['ci'][1]:.2f}] r2 {c['r2']:.2f} N={c['N']} n={c['n']}")
    elif a.cmd == "margins":
        logits = load_runs(dirs, "logits")
        if logits is None:
            print("no logits.csv found", file=sys.stderr)
            return 1
        res = fit_margins(logits)
        write_out(res, a.out, res["cells"])
        for c in res["cells"]:
            print(f"{c['qtype']} {c['arm']} {c['model']} {c['cond']}: N={c['N']} margin={[round(m, 2) for m in c['margin_median']]} "
                  f"acc={[round(x, 2) for x in c['accuracy']]} cross0@{c['n_cross_zero']}")
    elif a.cmd == "sharelaw":
        mass = load_runs(dirs, "mass")
        if mass is None:
            print("no mass.csv found", file=sys.stderr)
            return 1
        res = fit_sharelaw(mass, load_runs(dirs, "block_mass"), a.n_boot or 500, a.seed)
        write_out(res, a.out, [{**{c: f[c] for c in KEY + ("layer", "form", "s", "C", "r2", "best_form")},
                                "N": [c["N"] for c in f["cells"]], "k": [c["k"] for c in f["cells"]], "evid": [c["evid"] for c in f["cells"]],
                                "nonevid": [c["nonevid"] for c in f["cells"]], "other": [c["other"] for c in f["cells"]], "n": [c["n"] for c in f["cells"]]}
                               for f in res["fits"]])
        for f in res["fits"]:
            ap4 = f["alpha_pred"].get("4")
            print(f"{f['qtype']} {f['arm']} {f['model']} {f['cond']} L{f['layer']} [{f['form']}]: s {f['s']:.3f} C {f['C']:.1f} r2 {f['r2']:.3f} "
                  f"ci_s {[round(float(v), 2) for v in f['ci_s']]} alpha_pred(k=4) {ap4 if ap4 is None else round(ap4, 3)} best_form {f['best_form']}")
    elif a.cmd == "headscan":
        res = headscan(load_headscan_runs(dirs), a.head, a.thresh)
        write_out(res, a.out, [{**{c: g[c] for c in KEY}, "N": g["N"], "named_auc": [g["named"]["auc"].get(str(n)) for n in g["N"]],
                                "population": [g["population"][str(n)]["count"] for n in g["N"]]} for g in res["groups"]])
        for g in res["groups"]:
            print(f"{g['qtype']} {g['arm']} {g['model']} {g['cond']}: N={g['N']} {a.head}={[round(float(v), 4) for v in g['named']['auc'].values()]} "
                  f"pop>={a.thresh}: {[g['population'][str(n)]['count'] for n in g['N']]}")
    elif a.cmd == "gamma":
        pairs = load_runs(dirs, "pairs")
        if pairs is None:
            print("no pairs.csv found", file=sys.stderr)
            return 1
        res = fit_gamma(pairs, a.n_boot or 2000, a.seed, C_fixed=a.C)
        write_out(res, a.out, [{k: v for k, v in c.items() if k != "grid"} for c in res["cells"]])
        for c in res["cells"]:
            print(f"{c['qtype']} {c['arm']} {c['model']} {c['cond']} L{c['layer']} {c['locus']} N={c['N']}: gamma {c['gamma']:.3f} "
                  f"[{c['ci'][0]:.2f}, {c['ci'][1]:.2f}] C {c['C']} r2 {c['r2']:.3f} k={c['k']}")
    print(f"{a.cmd}: wrote {a.out} (+ .csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
