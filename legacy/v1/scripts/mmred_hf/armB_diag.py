#!/usr/bin/env python3
"""Arm-B diagnostics on existing dumps (CPU):
1a  per-frame-index head accuracy (position-drift hypothesis: fit @<=16, eval @128)
1b  conditioned heads: per-room occ-of-R and per-R1 trigger heads vs pooled
2   position-robust heads: project out index-correlated directions, refit, re-eval

Usage: python scripts/mmred_hf/armB_diag.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/mmred_hf"))

from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.mmred_hf import CHAR_ORDER, ROOM_ORDER, _char_room, _match, _rooms  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

D = _REPO / "outputs/mmred_hf/armB_dumps"
DIRS = _REPO / "data/mmred_hf/dirs"
NOB, MULTI = 5, 6
_meta: dict = {}


def meta(rel):
    if rel not in _meta:
        _s, _f, q0, states, a0 = load_mmred_sample(DIRS / rel)
        _meta[rel] = (q0, states, a0)
    return _meta[rel]


def occ_code(occ):
    return MULTI if len(occ) > 1 else (NOB if not occ else CHAR_ORDER.index(occ[0]))


def collect(dump, qts, lab_fn, cond_fn=None, train=False):
    X, y, fi, cond = [], [], [], []
    for i in range(len(dump["y"])):
        qt = str(dump["qtype"][i])
        if qt not in qts:
            continue
        cfg = dump["_cfg"]
        rel = (f"{cfg}_train_{qt}/{dump['sid'][i]}" if train else
               f"{cfg}/{dump['sid'][i]}")
        q0, states, _a = meta(rel)
        g = _match(qt, q0)
        t = int(dump["fidx"][i])
        try:
            y.append(lab_fn(qt, g, states, t))
        except Exception:
            continue
        X.append(i)
        fi.append(t)
        cond.append(cond_fn(qt, g) if cond_fn else 0)
    return np.array(X), np.array(y), np.array(fi), np.array(cond)


def main():
    fit8 = dict(np.load(D / "seq_len_8_trainfit.npz", allow_pickle=False))
    fit16 = dict(np.load(D / "seq_len_16_trainfit.npz", allow_pickle=False))
    fit8["_cfg"], fit16["_cfg"] = "seq_len_8", "seq_len_16"
    tests = {}
    for n in (8, 16, 32, 64, 128):
        t = dict(np.load(D / f"seq_len_{n}_test.npz", allow_pickle=False))
        t["_cfg"] = f"seq_len_{n}_test"
        tests[n] = t

    def fitXY(qts, lab_fn, cond_fn=None):
        idx8, y8, f8, c8 = collect(fit8, qts, lab_fn, cond_fn, train=True)
        idx16, y16, f16, c16 = collect(fit16, qts, lab_fn, cond_fn, train=True)
        X = np.concatenate([fit8["X"][idx8], fit16["X"][idx16]]).astype(np.float32)
        return (X, np.concatenate([y8, y16]), np.concatenate([f8, f16]),
                np.concatenate([c8, c16]))

    # ---------- 1a: per-index accuracy for roomofc (the drift witness)
    roomofc = lambda qt, g, st, t: ROOM_ORDER.index(_char_room(st[t], g[0]))
    ROOM_QTS = ["char_at_frame", "first_app", "final_app", "where_spend", "rooms_visited"]
    X, y, fi, _ = fitXY(ROOM_QTS, roomofc)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    print("== 1a per-index accuracy (roomofc head, fit @8/16) ==")
    for n in (16, 32, 128):
        t = tests[n]
        idx, yt, ft, _ = collect(t, ROOM_QTS, roomofc)
        Xt = t["X"][idx].astype(np.float32)
        pred = clf.predict(Xt)
        bins = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
        row = []
        for a, b in bins:
            m = (ft >= a) & (ft < b)
            row.append(f"[{a:3d}-{b:3d}) {float((pred[m]==yt[m]).mean()):.3f} (n={int(m.sum())})"
                       if m.sum() else f"[{a:3d}-{b:3d}) --")
        print(f"  N={n:3d}: " + "  ".join(row))

    # ---------- 1b: pooled vs per-condition heads (occ-of-R)
    occofr = lambda qt, g, st, t: occ_code(_rooms(st[t]).get(g[1] if qt == "who_spend" else g[0], []))
    condR = lambda qt, g: ROOM_ORDER.index(g[1] if qt == "who_spend" else g[0])
    OCC_QTS = ["room_at_frame", "first_at_room", "last_at_room", "who_spend"]
    X, y, fi, cR = fitXY(OCC_QTS, occofr, condR)
    pooled = LogisticRegression(max_iter=2000).fit(X, y)
    perR = {}
    for r in range(6):
        m = cR == r
        if m.sum() > 100 and len(set(y[m])) > 1:
            perR[r] = LogisticRegression(max_iter=2000).fit(X[m], y[m])
    t = tests[8]
    idx, yt, ft, cRt = collect(t, OCC_QTS, occofr, condR)
    Xt = t["X"][idx].astype(np.float32)
    pa = float((pooled.predict(Xt) == yt).mean())
    hits = tot = 0
    for r, c in perR.items():
        m = cRt == r
        if m.sum():
            hits += int((c.predict(Xt[m]) == yt[m]).sum())
            tot += int(m.sum())
    print(f"\n== 1b occ-of-R @seq8 test: pooled {pa:.3f} vs per-room "
          f"{hits/max(tot,1):.3f} (n={tot}) ==")
    # triggers: pooled vs per-R1
    trig = lambda qt, g, st, t: int(g[1] in _rooms(st[t]).get(g[2], []))
    condR1 = lambda qt, g: ROOM_ORDER.index(g[2])
    TRIG_QTS = ["char_on_char_first_app", "char_on_char_final_app",
                "room_on_char_first_app", "room_on_char_final_app",
                "n_room_on_char_first_app", "n_room_on_char_final_app"]
    X, y, fi, cR = fitXY(TRIG_QTS, trig, condR1)
    pooled = LogisticRegression(max_iter=2000).fit(X, y)
    perR = {r: LogisticRegression(max_iter=2000).fit(X[cR == r], y[cR == r])
            for r in range(6) if (cR == r).sum() > 100 and len(set(y[cR == r])) > 1}
    idx, yt, ft, cRt = collect(tests[8], TRIG_QTS, trig, condR1)
    Xt = tests[8]["X"][idx].astype(np.float32)
    pa = float((pooled.predict(Xt) == yt).mean())
    hits = tot = 0
    for r, c in perR.items():
        m = cRt == r
        hits += int((c.predict(Xt[m]) == yt[m]).sum())
        tot += int(m.sum())
    print(f"== 1b trigger @seq8 test: pooled {pa:.3f} vs per-R1 "
          f"{hits/max(tot,1):.3f} (n={tot}) ==")

    # ---------- 2: position-robust roomofc (project out index-correlated dirs)
    X, y, fi, _ = fitXY(ROOM_QTS, roomofc)
    mu = X.mean(0)
    Zc = X - mu
    beta, *_ = np.linalg.lstsq(
        np.stack([fi - fi.mean(), np.ones_like(fi)], 1).astype(np.float32), Zc,
        rcond=None)
    b = beta[0]
    b /= np.linalg.norm(b) + 1e-9
    means = np.stack([Zc[fi == k].mean(0) for k in sorted(set(fi)) if (fi == k).sum() > 30])
    U = np.linalg.svd(means - means.mean(0), full_matrices=False)[2][:4]
    P = np.concatenate([b[None], U])
    Q, _ = np.linalg.qr(P.T)

    def strip(A):
        Ac = A - mu
        return Ac - (Ac @ Q) @ Q.T

    clf2 = LogisticRegression(max_iter=2000).fit(strip(X), y)
    print("\n== 2 position-robust roomofc (index-dirs projected out) ==")
    for n in (16, 32, 128):
        idx, yt, ft, _ = collect(tests[n], ROOM_QTS, roomofc)
        Xt = tests[n]["X"][idx].astype(np.float32)
        a_raw = float((clf.predict(Xt) == yt).mean())
        a_rob = float((clf2.predict(strip(Xt)) == yt).mean())
        print(f"  N={n:3d}: raw {a_raw:.3f} -> robust {a_rob:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
