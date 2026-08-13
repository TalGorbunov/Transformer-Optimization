#!/usr/bin/env python3
"""Per-task diagnostic: measured task acc vs per-frame head acc vs predicted acc.

For each (qtype, N): per-frame accuracy of the task's primary head(s) on that test
dump, and a predicted task accuracy under an independence error model per reduction:
  select-k      pred = p_payload
  first/last    pred = p_gate^N * p_payload      (pessimistic: every gate call right)
  count-exact   pred = sum_k P(#false_pos = #false_neg)  (binomial cancellation)
  argmax        pred = simulation (10k draws with per-frame error rate)
Gap (measured - predicted) > 0 => errors correlate/cancel favorably; << 0 => extra
failure source beyond per-frame quality (e.g. head-composition, protocol).

Usage: python scripts/mmred_hf/armB_diag_v2.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/mmred_hf"))

from armB_grid import FACTS, NEEDS, load_dump, sample_meta  # noqa: E402
from gnnformer.mmred_hf import _match  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

D = _REPO / "outputs/mmred_hf/armB_dumps_v2"


def main() -> int:
    # ---- refit heads exactly as armB_grid does
    fitX, fit_meta = [], []
    for p in (D / "seq_len_8_trainfit.npz", D / "seq_len_16_trainfit.npz"):
        d = load_dump(p)
        cfg = Path(p).stem.replace("_trainfit", "")
        fitX.append(d["X"])
        fit_meta += [(cfg, str(q), str(s), int(f)) for q, s, f in
                     zip(d["qtype"], d["sid"], d["fidx"])]
    fitX = np.concatenate(fitX).astype(np.float32)
    cache: dict = {}
    heads = {}
    for fact, (lab_fn, src_qts) in FACTS.items():
        idx, ys = [], []
        for i, (cfg, qt, sid, fi) in enumerate(fit_meta):
            if qt not in src_qts:
                continue
            q0, states, _a = sample_meta(f"{cfg}_train_{qt}/{sid}",
                                         _REPO / "data/mmred_hf/dirs", cache)
            try:
                ys.append(lab_fn(qt, _match(qt, q0), states, fi))
                idx.append(i)
            except Exception:
                continue
        if len(idx) < 100 or len(set(ys)) < 2:
            continue
        heads[fact] = LogisticRegression(max_iter=2000).fit(fitX[idx], np.array(ys))

    # ---- measured task acc from the grid CSV
    meas = defaultdict(dict)
    with open(_REPO / "outputs/mmred_hf/armB_grid_v2/armB_grid_linear.csv") as f:
        for r in csv.DictReader(f):
            meas[r["qtype"]][int(r["config_split"].split("_")[2])] = float(r["acc"])

    rng = np.random.default_rng(0)
    print(f"{'qtype':26s} {'N':>4} {'task':>6} {'pframe':>7} {'pred':>6} {'gap':>6}")
    for N in (8, 16, 32):
        d = load_dump(D / f"seq_len_{N}_test.npz")
        by_q = defaultdict(list)
        for i in range(len(d["y"])):
            by_q[str(d["qtype"][i])].append(i)
        for qt in sorted(by_q):
            need = NEEDS.get(qt, [])
            if any(f not in heads for f in need):
                continue
            idxs = by_q[qt]
            # per-frame acc of each needed head on THIS qtype's frames
            pf = {}
            for fact in need:
                lab_fn = FACTS[fact][0]
                Xs, ys = [], []
                for i in idxs:
                    q0, states, _a = sample_meta(
                        f"seq_len_{N}_test/{d['sid'][i]}",
                        _REPO / "data/mmred_hf/dirs", cache)
                    try:
                        ys.append(lab_fn(qt, _match(qt, q0), states, int(d["fidx"][i])))
                        Xs.append(i)
                    except Exception:
                        continue
                if not Xs:
                    continue
                pred = heads[fact].predict(d["X"][Xs].astype(np.float32))
                pf[fact] = float((pred == np.array(ys)).mean())
            p_main = float(np.mean(list(pf.values())))
            # predicted task acc per reduction model
            if qt in ("steps_in_room", "crowd_count"):
                p = pf[need[0]]
                # binomial cancellation: exact count right if #FP == #FN
                e = 1 - p
                sims = (rng.random((20000, N)) < e)
                flips = rng.random((20000, N)) < 0.5  # FP vs FN direction
                delta = (sims * np.where(flips, 1, -1)).sum(1)
                pred_acc = float((delta == 0).mean())
            elif qt in ("char_at_frame", "room_at_frame", "n_char_at_frame", "n_empty",
                        "char_on_char_at_frame"):
                pred_acc = p_main
            elif qt in ("first_app", "final_app"):
                pred_acc = pf.get("roomofc", p_main)
            elif qt in ("first_at_room", "last_at_room"):
                p = pf[need[0]]
                pred_acc = p ** 2  # gate the selection + read the payload (same head)
            elif qt == "rooms_visited":
                p = pf[need[0]]
                pred_acc = p ** N * 1.5 if p ** N * 1.5 < 1 else 0.99  # rough
            elif qt.endswith("_app"):  # conditionals: trigger^N * payload (pessimistic)
                pg = pf.get("trig", 0.9)
                pp = float(np.mean([v for k, v in pf.items() if k != "trig"]))
                pred_acc = (pg ** N) * pp
            else:  # argmax families: simulate margin flips crudely as p_main^(N/4)
                pred_acc = p_main ** (N / 4)
            m = meas.get(qt, {}).get(N)
            if m is None:
                continue
            print(f"{qt:26s} {N:>4} {m:6.2f} {p_main:7.3f} {pred_acc:6.2f} "
                  f"{m - pred_acc:+6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
