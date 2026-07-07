#!/usr/bin/env python3
"""BATCH-0 of the tally-register solution (RESULTS [2026-07-04]+; design: docs/dprime_explainer.html §10).

Applies the theory-derived solution to the three established tasks, on their DEPLOYED carrier-message
caches (frames-first, real attention routing):

  gate     per-frame decision at the carrier: binary LDA gate (steps: room@L16; cooc: char2@L16)
           or multiclass per-frame room classifier (rooms: char@L14)
  tally    integer accumulation OUTSIDE the model: k = sum(gates)  /  n_r per room
  reduce   task algebra: sum (steps, cooc) | support-size union (rooms)
  readout  the tally IS the answer (symbolic; no frozen-head involvement)

Reported per task, 3 seeds, held-out:
  - solution exact acc + MAE vs the model's own answer (from the cache runs)
  - count-OOD: gate trained ONLY on low-answer samples -> evaluated on high answers
    (the extrapolation claim: the gate is per-frame and answer-blind, so nothing caps)
  - distillation proxy: gate labels flipped at the MEASURED look-again error rate
    (steps 0.013, cooc 0.091 from eval_per_frame_verification [2026-06-20b]; rooms: not measured -> 0.05 flag)

CPU only. Outputs: report + CSV under outputs/frame_axis/probes/tally_solution/<ts>/.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

TASKS = {
    "steps": dict(cache="outputs/frame_axis/probes/carrier_message/count_msgcache/count/messages_cache.pt",
                  layer=16, off=9, kind="binary", flip=0.013, model_acc=0.207, ood_split=4),
    "cooc": dict(cache="outputs/frame_axis/probes/carrier_message/cooc_msgcache_big/co_occupancy/messages_cache.pt",
                 layer=16, off=13, kind="binary", flip=0.091, model_acc=0.155, ood_split=4),
    "rooms": dict(cache="outputs/frame_axis/probes/carrier_message/rooms_msgcache_big/rooms_visited/messages_cache.pt",
                  layer=14, off=10, kind="multiclass", flip=0.05, model_acc=0.087, ood_split=3),
}


def fit_gate_binary(Xtr, ytr, rng, max_n=4000):
    sub = rng.permutation(len(Xtr))[:max_n]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr[sub], ytr[sub].astype(int))
    return lda


def run_task(name, cfg, seeds, out_lines, rows):
    c = torch.load(cfg["cache"], map_location="cpu", weights_only=False)
    M = c["msgs"][cfg["layer"]][cfg["off"]].astype(np.float32)
    gold = np.asarray(c["gold"])
    n, NF, H = M.shape
    if cfg["kind"] == "binary":
        lab = c["labels"].astype(int)                                   # [n, NF]
    else:
        lab = np.array([[str(x) for x in row] for row in c["labels_raw"]])
    out_lines.append(f"\n== {name} (cache n={n}, L{cfg['layer']} off{cfg['off']}, {cfg['kind']}) "
                     f"model own-answer={cfg['model_acc']:.3f} ==")
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n); ntr = int(0.6 * n)
        tr, te = idx[:ntr], idx[ntr:]
        variants = {"gt": lab, }
        # distillation proxy: flip labels at the measured look-again error rate (train only)
        if cfg["kind"] == "binary":
            noisy = lab.copy()
            flips = rng.rand(*noisy.shape) < cfg["flip"]
            noisy = np.where(flips, 1 - noisy, noisy)
            variants["distill"] = noisy
        for vname, L in variants.items():
            Xtr = M[tr].reshape(-1, H)
            if cfg["kind"] == "binary":
                gate = fit_gate_binary(Xtr, L[tr].reshape(-1), rng)
                khat = gate.predict(M[te].reshape(-1, H)).reshape(len(te), NF).sum(1)
            else:
                sub = rng.permutation(len(Xtr))[:4000]
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                    Xtr[sub], L[tr].reshape(-1)[sub])
                pf = clf.predict(M[te].reshape(-1, H)).reshape(len(te), NF)
                khat = np.array([len(set(r.tolist()) - {"None"}) for r in pf])
            acc = float(np.mean(khat == gold[te])); mae = float(np.mean(np.abs(khat - gold[te])))
            out_lines.append(f"  seed{seed} [{vname:7s}] tally acc={acc:.3f}  MAE={mae:.2f}  "
                             f"(model {cfg['model_acc']:.3f}, lift {acc - cfg['model_acc']:+.3f})")
            rows.append(f"{name},{vname},{seed},iid,{acc:.4f},{mae:.4f}")
        # count-OOD: train gate ONLY on samples with gold <= ood_split, evaluate on gold > split
        lo = tr[gold[tr] <= cfg["ood_split"]]
        hi = te[gold[te] > cfg["ood_split"]]
        if len(lo) >= 40 and len(hi) >= 20:
            Xlo = M[lo].reshape(-1, H)
            if cfg["kind"] == "binary":
                gate = fit_gate_binary(Xlo, lab[lo].reshape(-1), rng)
                khat = gate.predict(M[hi].reshape(-1, H)).reshape(len(hi), NF).sum(1)
            else:
                sub = rng.permutation(len(Xlo))[:4000]
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                    Xlo[sub], lab[lo].reshape(-1)[sub])
                pf = clf.predict(M[hi].reshape(-1, H)).reshape(len(hi), NF)
                khat = np.array([len(set(r.tolist()) - {"None"}) for r in pf])
            acc = float(np.mean(khat == gold[hi])); mae = float(np.mean(np.abs(khat - gold[hi])))
            out_lines.append(f"  seed{seed} [OOD: train gold<={cfg['ood_split']} -> eval gold>{cfg['ood_split']}, "
                             f"n_eval={len(hi)}] tally acc={acc:.3f}  MAE={mae:.2f}")
            rows.append(f"{name},gt,{seed},ood,{acc:.4f},{mae:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "tally_solution")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = args.output / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["=== BATCH-0: tally-register solution on the three established tasks (deployed carriers) ==="]
    rows = ["task,variant,seed,split,acc,mae"]
    for name, cfg in TASKS.items():
        run_task(name, cfg, seeds, lines, rows)
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report)
    (out / "metrics.csv").write_text("\n".join(rows) + "\n")
    (out / "run_config.json").write_text(json.dumps({"tasks": TASKS, "seeds": seeds}, indent=2, default=str))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
