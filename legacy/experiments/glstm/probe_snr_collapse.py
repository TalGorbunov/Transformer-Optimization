#!/usr/bin/env python3
"""SNR collapse curve: is count accuracy a single function of per-frame SNR / sqrt(N), across ALL
conditions (crowding, layer, N, joint/masked)? If the points collapse onto one curve when plotted vs
SNR/sqrt(N) (the discretization SNR) but NOT vs raw SNR, that's the quantitative law behind the story:
the LINEAR-sum count accuracy is set by per-frame SNR attenuated by sqrt(N)."""
from __future__ import annotations
import glob, math, os, pickle, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from experiments.glstm.probe_message_sum_decodability import fit_ridge, agg_sum

MS = PROJECT_ROOT / "outputs/frame_axis/probes/message_sum"


def _valid(reps, fl):
    try:
        lab = np.asarray([int(x) for x in fl])
    except (TypeError, ValueError):
        return None
    if lab.ndim != 1 or lab.shape[0] != reps.shape[0] or not set(lab.tolist()) <= {0, 1}:
        return None
    return lab


def exs_minimal(path, seq_len):
    out = []
    for v in torch.load(path, map_location="cpu").values():
        if int(v.get("seq_len", -1)) != seq_len or v.get("frame_labels") is None:
            continue
        reps = v["reps"].float().numpy(); lab = _valid(reps, v["frame_labels"])
        if lab is not None:
            out.append((reps, lab, int(v["gold"])))
    return out


def exs_layersweep(path, layer, seq_len):
    out = []
    for v in torch.load(path, map_location="cpu").values():
        if int(v.get("seq_len", -1)) != seq_len or v.get("frame_labels") is None or layer not in v["reps_by_layer"]:
            continue
        reps = v["reps_by_layer"][layer].float().numpy(); lab = _valid(reps, v["frame_labels"])
        if lab is not None:
            out.append((reps, lab, int(v["gold"])))
    return out


def exs_pkl(path, cond):
    d = pickle.load(open(path, "rb"))
    out = []
    for r, l, g in zip(d[cond], d["labels"], d["gold"]):
        if r is None:
            continue
        lab = _valid(r, l)
        if lab is not None:
            out.append((r, lab, int(g)))
    return out


def perframe_snr(exs):
    ev = np.stack([r[i] for r, l, g in exs for i in range(r.shape[0]) if l[i] == 1])
    nv = np.stack([r[i] for r, l, g in exs for i in range(r.shape[0]) if l[i] == 0])
    d = ev.mean(0) - nv.mean(0); dh = d / (np.linalg.norm(d) + 1e-9)
    sig = 0.5 * ((ev @ dh).std() + (nv @ dh).std())
    return float(abs(d @ dh) / (sig + 1e-9))


def sall_acc(exs, seeds=(0, 1, 2)):
    X = np.stack([agg_sum(r) for r, l, g in exs]); y = np.asarray([g for r, l, g in exs])
    return fit_ridge(X, y, list(seeds))["acc"]


def main():
    pts = []  # (label, N, snr, acc)
    def add(label, exs, N):
        if len(exs) >= 40 and len(set(g for _, _, g in exs)) >= 2:
            pts.append((label, N, perframe_snr(exs), sall_acc(exs)))
            print(f"  {label:<22} N={N} snr={pts[-1][2]:.3f} acc={pts[-1][3]:.3f} (n={len(exs)})", flush=True)

    print("collecting conditions ...", flush=True)
    for lbl, sub, N in [("crowded_5ch", "cache_crowded", 8), ("balanced_5ch", "cache_decrowded", 8),
                        ("1char", "cache_1char", 8), ("seq2", "cache_ns_seq2", 2),
                        ("seq4", "cache_ns_seq4", 4), ("seq6", "cache_ns_seq6", 6),
                        ("framesfirst", "cache_framesfirst_crowded", 8)]:
        p = MS / sub / "minimal_L19_steps_in_room.pt"
        if p.exists():
            add(lbl, exs_minimal(p, N), N)
    for lbl, sub in [("lsweep_crowd", "cache_layersweep_crowded"), ("lsweep_bal", "cache_layersweep_decrowded")]:
        p = MS / sub / "layersweep_steps_in_room.pt"
        if p.exists():
            cache = torch.load(p, map_location="cpu")
            for L in sorted(next(iter(cache.values()))["reps_by_layer"].keys()):
                add(f"{lbl}_L{L}", exs_layersweep(p, L, 8), 8)
    for pkl in sorted(glob.glob(str(PROJECT_ROOT / "outputs/frame_axis/probes/frame_isolation/*/reps.pkl")),
                      key=os.path.getmtime)[-2:]:
        for cond in ("joint", "masked"):
            add(f"iso_{cond}_{Path(pkl).parent.name[-6:]}", exs_pkl(pkl, cond), 8)

    out = PROJECT_ROOT / "outputs/frame_axis/probes/snr_collapse"
    out.mkdir(parents=True, exist_ok=True)
    rows = ["label,N,perframe_snr,snr_per_count,count_acc"]
    for lbl, N, snr, acc in pts:
        rows.append(f"{lbl},{N},{snr:.4f},{snr/math.sqrt(N):.4f},{acc:.4f}")
    (out / "collapse.csv").write_text("\n".join(rows) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        Ns = sorted(set(N for _, N, _, _ in pts)); cmap = {n: c for n, c in zip(Ns, plt.cm.viridis(np.linspace(0, 1, len(Ns))))}
        for lbl, N, snr, acc in pts:
            ax[0].scatter(snr, acc, color=cmap[N], s=40)
            ax[1].scatter(snr / math.sqrt(N), acc, color=cmap[N], s=40, label=f"N={N}")
        # erf prediction on panel 2 (adjacent-count Gaussian rounding)
        from math import erf
        xs = np.linspace(0.01, max(p[2] / math.sqrt(p[1]) for p in pts) * 1.1, 100)
        ax[1].plot(xs, [erf(0.5 * x / math.sqrt(2)) for x in xs], "k--", alpha=0.6, label="erf(0.5·SNR/√2)")
        ax[0].set_xlabel("per-frame SNR (raw)"); ax[0].set_ylabel("count acc (linear sum)")
        ax[0].set_title("vs RAW SNR — should NOT collapse (N-dependent)")
        ax[1].set_xlabel("per-frame SNR / √N  (discretization SNR)"); ax[1].set_ylabel("count acc")
        ax[1].set_title("vs SNR/√N — should COLLAPSE onto one curve")
        h, l = ax[1].get_legend_handles_labels(); seen = dict(zip(l, h)); ax[1].legend(seen.values(), seen.keys(), fontsize=8)
        for a in ax:
            a.set_ylim(0, 1.02); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "collapse.png", dpi=130); plt.close(fig)
        print(f"\nwrote {out}/collapse.png and collapse.csv  ({len(pts)} conditions)")
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
