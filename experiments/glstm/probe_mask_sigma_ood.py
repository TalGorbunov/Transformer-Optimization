#!/usr/bin/env python3
"""Canonical clean solution test: frozen model + frame-ISOLATION mask + fixed Sigma-sigma readout,
with COUNT-HOLDOUT OOD. Trains the per-frame gate on counts <=4 and tests the summed count on 5-8.
Does mask + Sigma-sigma (a) hit ~0.95 IID and (b) EXTRAPOLATE to unseen counts? Compares masked vs joint
reps (from frame_isolation_diagnostic's reps.pkl), and the gate (extensive) vs a linear-sum readout
(learned, should cap OOD)."""
from __future__ import annotations
import argparse, glob, os, pickle, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def count_holdout(reps_list, labels, gold, hold=(5, 6, 7, 8), seed=0):
    keep = [i for i, r in enumerate(reps_list) if r is not None]
    g = np.asarray([gold[i] for i in keep])
    reps = [reps_list[i] for i in keep]; lab = [labels[i] for i in keep]
    indist = [k for k in range(len(keep)) if g[k] not in hold]
    ood = [k for k in range(len(keep)) if g[k] in hold]
    tr, iid_te = train_test_split(indist, test_size=0.35, random_state=seed)
    N = reps[0].shape[0]
    # per-frame gate trained on counts <=4 frames only
    Xtr = np.stack([reps[k][f] for k in tr for f in range(N)])
    ytr = np.asarray([lab[k][f] for k in tr for f in range(N)])
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
    sctr = np.concatenate([clf.decision_function(sc.transform(reps[k])) for k in tr])
    smean, sstd = float(sctr.mean()), float(sctr.std() + 1e-9)

    def gate_count(k, kap):
        z = (clf.decision_function(sc.transform(reps[k])) - smean) / sstd
        return int(round((1.0 / (1.0 + np.exp(-kap * z))).sum())) if kap > 0 \
            else int(round(clf.predict_proba(sc.transform(reps[k]))[:, 1].sum()))

    # learned linear-sum readout (Ridge on S_all) — the foil that should cap OOD
    S = np.stack([reps[k].sum(0) for k in range(len(reps))])
    scS = StandardScaler().fit(S[tr]); rid = Ridge(alpha=1.0).fit(scS.transform(S[tr]), g[tr])

    def acc(ks, fn):
        return accuracy_score([g[k] for k in ks], [fn(k) for k in ks])
    out = {}
    for tag, kap in (("gate_soft", 0.0), ("gate_sharp_k20", 20.0)):
        out[f"{tag}_IID"] = acc(iid_te, lambda k, kp=kap: gate_count(k, kp))
        out[f"{tag}_OOD"] = acc(ood, lambda k, kp=kap: gate_count(k, kp))
    out["linear_sum_IID"] = acc(iid_te, lambda k: int(round(float(rid.predict(scS.transform(S[k:k+1]))[0]))))
    out["linear_sum_OOD"] = acc(ood, lambda k: int(round(float(rid.predict(scS.transform(S[k:k+1]))[0]))))
    out["n_iid"], out["n_ood"] = len(iid_te), len(ood)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default=None, help="reps.pkl from frame_isolation_diagnostic (default: latest)")
    args = ap.parse_args()
    path = args.reps or sorted(glob.glob(str(PROJECT_ROOT / "outputs/frame_axis/probes/frame_isolation/*/reps.pkl")),
                               key=os.path.getmtime)[-1]
    d = pickle.load(open(path, "rb"))
    print(f"reps: {path}  (n={sum(1 for r in d['masked'] if r is not None)})\n")
    print(f"{'cond':<8} {'gate_soft IID/OOD':>20} {'gate_sharp IID/OOD':>22} {'linear-sum IID/OOD':>22}")
    for cond in ("joint", "masked"):
        r = count_holdout(d[cond], d["labels"], d["gold"])
        print(f"{cond:<8} {r['gate_soft_IID']:>9.3f}/{r['gate_soft_OOD']:.3f}     "
              f"{r['gate_sharp_k20_IID']:>9.3f}/{r['gate_sharp_k20_OOD']:.3f}      "
              f"{r['linear_sum_IID']:>9.3f}/{r['linear_sum_OOD']:.3f}   (n_iid={r['n_iid']} n_ood={r['n_ood']})")
    print("\nreading: gate (extensive Sigma-sigma) should hold OOD; linear-sum (learned) should DROP OOD; "
          "masked >> joint (clean per-frame).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
