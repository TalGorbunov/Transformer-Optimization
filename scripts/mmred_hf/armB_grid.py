#!/usr/bin/env python3
"""Arm B for ALL tasks: per-frame heads on dumped carrier states + symbolic reduction.

Pipeline (CPU): load fit dumps (seq8+16 trainfit npz) -> for every fact head, compute
labels from the sample dirs' gold states -> fit LogisticRegression and MLP(256)
(the GIN-psi variant) -> for each test dump, predict per-frame facts -> reduce per
sample with the task's symbolic reduction -> EM vs gold. Heads are per-frame, so
seq8/16-fitted heads apply unchanged at N=32..128 (train-short/eval-long).

Fact heads (all computed from (qtype-conditioned) carrier states):
  match(C,R) gate | room-of-C 6-way | occ-of-R {5 names,nobody,multi} | counts |
  per-entity binaries (room-empty(r), crowded(r), alone(c), with-partner(c), in-R(c))

Usage:
  python scripts/mmred_hf/armB_grid.py --fit outputs/mmred_hf/armB_dumps/seq_len_8_trainfit.npz \
      outputs/mmred_hf/armB_dumps/seq_len_16_trainfit.npz \
      --test outputs/mmred_hf/armB_dumps/seq_len_*_test.npz --head mlp
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/mmred_hf"))

from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.mmred_hf import (  # noqa: E402
    CHAR_ORDER, ROOM_ORDER, _char_room, _match, _rooms, qtype_from_dirname,
)

NOB, MULTI = 5, 6


# ---------------------------------------------------------------- fact definitions
# fact name -> (label_fn(qtype, g, states, t) -> int, [qtypes whose samples train it])

def _occ_code(occ):
    return MULTI if len(occ) > 1 else (NOB if not occ else CHAR_ORDER.index(occ[0]))


FACTS = {
    "match": (lambda qt, g, st, t: int(g[0] in _rooms(st[t]).get(g[1], [])),
              ["steps_in_room"]),
    "crowdany": (lambda qt, g, st, t: int(any(len(o) >= int(g[0]) for o in _rooms(st[t]).values())),
                 ["crowd_count"]),
    "roomofc": (lambda qt, g, st, t: ROOM_ORDER.index(_char_room(st[t], g[0])),
                ["char_at_frame", "first_app", "final_app", "where_spend",
                 "rooms_visited"]),
    "occofr": (lambda qt, g, st, t: _occ_code(_rooms(st[t]).get(
                   g[1] if qt == "who_spend" else g[0], [])),
               ["room_at_frame", "first_at_room", "last_at_room", "who_spend"]),
    "nwithc": (lambda qt, g, st, t: min(len(_rooms(st[t]).get(_char_room(st[t], g[0]), [])) - 1, 4),
               ["n_char_at_frame"]),
    "nempty": (lambda qt, g, st, t: sum(1 for o in _rooms(st[t]).values() if not o),
               ["n_empty"]),
    "trig": (lambda qt, g, st, t: int(g[1] in _rooms(st[t]).get(g[2], [])),
             ["char_on_char_first_app", "char_on_char_final_app",
              "room_on_char_first_app", "room_on_char_final_app",
              "n_room_on_char_first_app", "n_room_on_char_final_app"]),
    "payl_room": (lambda qt, g, st, t: ROOM_ORDER.index(_char_room(st[t], g[0])),
                  ["char_on_char_first_app", "char_on_char_final_app"]),
    "payl_occ": (lambda qt, g, st, t: _occ_code(_rooms(st[t]).get(g[0], [])),
                 ["room_on_char_first_app", "room_on_char_final_app"]),
    "payl_cnt": (lambda qt, g, st, t: min(len(_rooms(st[t]).get(g[0], [])), 5),
                 ["n_room_on_char_first_app", "n_room_on_char_final_app"]),
    "roommate": (lambda qt, g, st, t: _occ_code(
                     [x for x in _rooms(st[t]).get(_char_room(st[t], g[0]), [])
                      if x != g[0]]),
                 ["char_on_char_at_frame", "spend_together"]),
}
# per-entity binaries share one head each, entity one-hot NOT needed: we fit ONE head
# per entity index (6 room heads / 5 char heads), trained on the corresponding label.
for i, r in enumerate(ROOM_ORDER):
    FACTS[f"empty_{i}"] = (
        (lambda i_: lambda qt, g, st, t: int(not _rooms(st[t]).get(ROOM_ORDER[i_], [])))(i),
        ["room_empty"])
    FACTS[f"crowded_{i}"] = (
        (lambda i_: lambda qt, g, st, t: int(len(_rooms(st[t]).get(ROOM_ORDER[i_], [])) >= int(g[0])))(i),
        ["crowded_room"])
for i, c in enumerate(CHAR_ORDER):
    FACTS[f"alone_{i}"] = (
        (lambda i_: lambda qt, g, st, t: int(len(_rooms(st[t]).get(_char_room(st[t], CHAR_ORDER[i_]), [])) == 1))(i),
        ["spend_alone"])
    FACTS[f"inr_{i}"] = (
        (lambda i_: lambda qt, g, st, t: int(CHAR_ORDER[i_] in _rooms(st[t]).get(
            g[1] if qt == "who_spend" else g[0], [])))(i),
        ["who_spend"])
    FACTS[f"withc_{i}"] = (
        (lambda i_: lambda qt, g, st, t: int(CHAR_ORDER[i_] in _rooms(st[t]).get(_char_room(st[t], g[0]), [])
                                             and CHAR_ORDER[i_] != g[0]))(i),
        ["spend_together"])

# which facts each qtype needs at REDUCTION time
NEEDS = {
    "steps_in_room": ["match"], "crowd_count": ["crowdany"],
    "char_at_frame": ["roomofc"], "first_app": ["roomofc"], "final_app": ["roomofc"],
    "where_spend": ["roomofc"], "rooms_visited": ["roomofc"],
    "room_at_frame": ["occofr"], "first_at_room": ["occofr"], "last_at_room": ["occofr"],
    "who_spend": [f"inr_{i}" for i in range(5)],
    "n_char_at_frame": ["nwithc"],
    "n_empty": [f"empty_{i}" for i in range(6)],  # derive count from per-room bits
    #   (the direct nempty head caps ~0.7; the six bits fit at 1.000)
    "char_on_char_first_app": ["trig", "payl_room"],
    "char_on_char_final_app": ["trig", "payl_room"],
    "room_on_char_first_app": ["trig", "payl_occ"],
    "room_on_char_final_app": ["trig", "payl_occ"],
    "n_room_on_char_first_app": ["trig", "payl_cnt"],
    "n_room_on_char_final_app": ["trig", "payl_cnt"],
    "char_on_char_at_frame": ["roommate"],
    "spend_alone": [f"alone_{i}" for i in range(5)],
    "spend_together": [f"withc_{i}" for i in range(5)],
    "room_empty": [f"empty_{i}" for i in range(6)],
    "crowded_room": [f"crowded_{i}" for i in range(6)],
}


def reduce_answer(qt: str, g, preds: dict, N: int):
    """Symbolic reduction over PREDICTED per-frame facts -> answer string or None."""
    def first_last(mask, final):
        hits = [t for t in range(N) if mask[t]]
        return (hits[-1] if final else hits[0]) if hits else None

    if qt == "steps_in_room":
        return str(int(sum(preds["match"])))
    if qt == "crowd_count":
        return str(int(sum(preds["crowdany"])))
    if qt in ("char_at_frame", "first_app", "final_app"):
        k = (int(g[1]) - 1 if qt == "char_at_frame" else (0 if qt == "first_app" else N - 1))
        return ROOM_ORDER[preds["roomofc"][k]]
    if qt == "where_spend":
        cnt = np.bincount(preds["roomofc"], minlength=6)
        i = int(np.argmax(cnt)) if g[1] == "most" else int(np.argmin(cnt))
        return ROOM_ORDER[i]
    if qt == "rooms_visited":
        return str(len(set(preds["roomofc"])))
    if qt in ("room_at_frame", "first_at_room", "last_at_room"):
        occ = preds["occofr"]
        if qt == "room_at_frame":
            v = occ[int(g[1]) - 1]
        else:
            t = first_last([o != NOB for o in occ], qt == "last_at_room")
            v = occ[t] if t is not None else NOB
        return "Nobody" if v == NOB else (CHAR_ORDER[v] if v < NOB else None)
    if qt == "who_spend":
        cnt = [int(sum(preds[f"inr_{i}"])) for i in range(5)]
        i = int(np.argmax(cnt)) if g[0] == "most" else int(np.argmin(cnt))
        return CHAR_ORDER[i]
    if qt == "n_char_at_frame":
        return str(int(preds["nwithc"][int(g[1]) - 1]))
    if qt == "n_empty":
        k = int(g[0]) - 1
        return str(int(sum(int(preds[f"empty_{i}"][k]) for i in range(6))))
    if qt in ("char_on_char_first_app", "char_on_char_final_app",
              "room_on_char_first_app", "room_on_char_final_app",
              "n_room_on_char_first_app", "n_room_on_char_final_app"):
        t = first_last(preds["trig"], qt.endswith("final_app"))
        if t is None:
            return None
        if qt.startswith("char_on_char"):
            return ROOM_ORDER[preds["payl_room"][t]]
        if qt.startswith("n_room"):
            return str(int(preds["payl_cnt"][t]))
        v = preds["payl_occ"][t]
        return "Nobody" if v == NOB else (CHAR_ORDER[v] if v < NOB else None)
    if qt == "char_on_char_at_frame":
        v = preds["roommate"][int(g[1]) - 1]
        return "Nobody" if v == NOB else (CHAR_ORDER[v] if v < NOB else None)
    if qt == "spend_alone":
        cnt = [int(sum(preds[f"alone_{i}"])) for i in range(5)]
        i = int(np.argmax(cnt)) if g[0] == "most" else int(np.argmin(cnt))
        return CHAR_ORDER[i]
    if qt == "spend_together":
        c = g[0]
        cand = [i for i in range(5) if CHAR_ORDER[i] != c]
        cnt = {i: int(sum(preds[f"withc_{i}"])) for i in cand}
        pick = (max if g[1] == "most" else min)(cand, key=lambda i: cnt[i])
        return CHAR_ORDER[pick]
    if qt == "room_empty":
        cnt = [int(sum(preds[f"empty_{i}"])) for i in range(6)]
        i = int(np.argmax(cnt)) if g[0] == "more" else int(np.argmin(cnt))
        return ROOM_ORDER[i]
    if qt == "crowded_room":
        cnt = [int(sum(preds[f"crowded_{i}"])) for i in range(6)]
        return ROOM_ORDER[int(np.argmax(cnt))]
    return None


def load_dump(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def sample_meta(sid: str, root: Path, cache: dict):
    if sid not in cache:
        _s, _f, q0, states, a0 = load_mmred_sample(root / sid)
        cache[sid] = (q0, states, a0)
    return cache[sid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--head", choices=("linear", "mlp"), default="mlp")
    ap.add_argument("--output", default="outputs/mmred_hf/armB_grid")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    # ---- assemble fit features + labels per fact
    fitX, fit_meta = [], []
    for p in args.fit:
        d = load_dump(p)
        cfg = Path(p).stem.replace("_trainfit", "")
        root = _REPO / f"data/mmred_hf/dirs"  # per-qtype train roots
        fitX.append(d["X"])
        fit_meta += [(cfg, str(q), str(s), int(f)) for q, s, f in
                     zip(d["qtype"], d["sid"], d["fidx"])]
    fitX = np.concatenate(fitX).astype(np.float32)

    meta_cache: dict = {}
    heads = {}
    for fact, (lab_fn, src_qts) in FACTS.items():
        idx, ys = [], []
        for i, (cfg, qt, sid, fi) in enumerate(fit_meta):
            if qt not in src_qts:
                continue
            rel = (f"{cfg}/{sid}" if "headfit" in cfg else f"{cfg}_train_{qt}/{sid}")
            q0, states, _a = sample_meta(rel, _REPO / "data/mmred_hf/dirs", meta_cache)
            try:
                ys.append(lab_fn(qt, _match(qt, q0), states, fi))
                idx.append(i)
            except Exception:
                continue
        if len(idx) < 100 or len(set(ys)) < 2:
            print(f"[fit] {fact}: SKIP (n={len(idx)})", flush=True)
            continue
        Xf, yf = fitX[idx], np.array(ys)
        if args.head == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                early_stopping=True, random_state=0)
        else:
            clf = LogisticRegression(max_iter=2000)
        clf.fit(Xf, yf)
        heads[fact] = clf
        print(f"[fit] {fact}: n={len(idx)} train-acc {clf.score(Xf, yf):.3f}", flush=True)

    # ---- evaluate each test dump
    out = _REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    rows = []
    for p in sorted(sum([glob.glob(x) for x in args.test], [])):
        d = load_dump(p)
        cfg_split = Path(p).stem  # e.g. seq_len_32_test
        root = _REPO / f"data/mmred_hf/dirs/{cfg_split}"
        by_sid = defaultdict(list)
        for i, s in enumerate(d["sid"]):
            by_sid[str(s)].append(i)
        acc = defaultdict(lambda: [0, 0])
        mae = defaultdict(lambda: [0, 0, 0])
        pred_cache: dict = {}
        for sid, idxs in by_sid.items():
            qt = str(d["qtype"][idxs[0]])
            need = NEEDS.get(qt, [])
            if any(f not in heads for f in need):
                continue
            idxs = sorted(idxs, key=lambda i: int(d["fidx"][i]))
            Xs = d["X"][idxs].astype(np.float32)
            preds = {}
            for f in need:
                preds[f] = heads[f].predict(Xs)
            q0, states, gold = sample_meta(f"{cfg_split}/{sid}",
                                           _REPO / "data/mmred_hf/dirs", meta_cache)
            try:
                ans = reduce_answer(qt, _match(qt, q0), preds, len(idxs))
            except Exception:
                ans = None
            hit = int(ans is not None and str(ans) == str(gold))
            acc[qt][0] += hit
            acc[qt][1] += 1
            if ans is not None and str(ans).isdigit() and str(gold).isdigit():
                d_ = abs(int(ans) - int(gold))
                mae[qt][0] += d_
                mae[qt][1] += 1
                mae[qt][2] += int(d_ <= 1)
        tot = [sum(a for a, _ in acc.values()), sum(n for _, n in acc.values())]
        print(f"== {cfg_split}: overall {tot[0]}/{tot[1]} = {tot[0]/max(tot[1],1):.3f}")
        for qt, (a, n) in sorted(acc.items()):
            rows.append((cfg_split, qt, n, a / max(n, 1)))
            extra = ""
            if mae[qt][1]:
                extra = (f"  MAE {mae[qt][0]/mae[qt][1]:.2f}"
                         f"  ±1-acc {mae[qt][2]/mae[qt][1]:.3f}")
            print(f"   {qt:28s} {a:3d}/{n:3d} = {a/max(n,1):.3f}{extra}")
        rows.append((cfg_split, "_all", tot[1], tot[0] / max(tot[1], 1)))

    with open(out / f"armB_grid_{args.head}.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["config_split", "qtype", "n", "acc"])
        w.writerows(rows)
    print(f"wrote {out}/armB_grid_{args.head}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
