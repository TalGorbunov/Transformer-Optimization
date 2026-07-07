#!/usr/bin/env python3
"""Per-frame SNR diagnosis for co_occupancy (binary, like steps) and rooms_visited (categorical -> per-room
one-vs-rest average SNR). Reports per-frame SNR + the linear-sum count accuracy + frozen last-token count
accuracy, per cache. Run on frozen caches (crowded vs decrowded) to show the SNR bottleneck generalizes."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from experiments.glstm.probe_message_sum_decodability import fit_ridge, agg_sum


def load(path, seq_len):
    out = []
    for v in torch.load(path, map_location="cpu").values():
        if int(v.get("seq_len", -1)) != seq_len or v.get("frame_labels") is None:
            continue
        reps = v["reps"].float().numpy(); fl = v["frame_labels"]
        if len(fl) != reps.shape[0]:
            continue
        out.append((reps, list(fl), int(v["gold"]), v["query_rep"].float().numpy()))
    return out


def fisher_snr(ev, nv):
    if len(ev) < 5 or len(nv) < 5:
        return None
    ev, nv = np.stack(ev), np.stack(nv)
    d = ev.mean(0) - nv.mean(0); dh = d / (np.linalg.norm(d) + 1e-9)
    sig = 0.5 * ((ev @ dh).std() + (nv @ dh).std())
    return float(abs(d @ dh) / (sig + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True, help="LABEL:PATH")
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()
    print(f"{'label':<16} {'type':<6} {'per-frame SNR':>14} {'count acc(sum)':>15} {'last-tok acc':>13}")
    for spec in args.caches:
        label, _, path = spec.partition(":")
        exs = load(Path(path), args.seq_len)
        if len(exs) < 40:
            print(f"{label:<16} (only {len(exs)} ex — skip)"); continue
        # binary vs categorical
        flat = [x for r, fl, g, q in exs for x in fl]
        binary = set(str(x) for x in flat) <= {"0", "1", "0.0", "1.0"}
        if binary:
            ev = [r[i] for r, fl, g, q in exs for i in range(r.shape[0]) if int(float(fl[i])) == 1]
            nv = [r[i] for r, fl, g, q in exs for i in range(r.shape[0]) if int(float(fl[i])) == 0]
            snr = fisher_snr(ev, nv); ttype = "binary"
        else:
            rooms = [x for x in set(flat) if flat.count(x) >= 20]
            snrs = []
            for rm in rooms:
                ev = [r[i] for r, fl, g, q in exs for i in range(r.shape[0]) if fl[i] == rm]
                nv = [r[i] for r, fl, g, q in exs for i in range(r.shape[0]) if fl[i] != rm]
                s = fisher_snr(ev, nv)
                if s is not None:
                    snrs.append(s)
            snr = float(np.mean(snrs)) if snrs else float("nan"); ttype = f"rooms({len(snrs)})"
        gold = np.asarray([g for r, fl, g, q in exs])
        Sall = np.stack([agg_sum(r) for r, fl, g, q in exs])
        Xq = np.stack([q for r, fl, g, q in exs])
        acc_sum = fit_ridge(Sall, gold, [0, 1, 2])["acc"]
        acc_tok = fit_ridge(Xq, gold, [0, 1, 2])["acc"]
        print(f"{label:<16} {ttype:<6} {snr:>14.3f} {acc_sum:>15.3f} {acc_tok:>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
