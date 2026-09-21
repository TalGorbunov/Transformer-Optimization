#!/usr/bin/env python3
"""Dump frozen carrier states + per-frame fact labels for the arm-B GIN readout.

For every sample dir under --root (or the per-qtype train roots via --qtype-roots):
prefill the fenced/truncated prompt (park e_c, NO LoRA), capture carrier states at
--layer, label frames via facts.frame_labels, and append to one npz per invocation:
  X (n_frames, 3584) fp16 | y (n_frames,) | qtype (n_frames,) str | sid, fidx, N, gold

Usage (test split):
  python scripts/mmred_hf/dump_states.py --root data/mmred_hf/dirs/seq_len_32_test \
      --out outputs/mmred_hf/armB_dumps/seq_len_32_test.npz
Usage (train fit-set):
  python scripts/mmred_hf/dump_states.py --qtype-roots seq_len_8 --limit-per-qtype 100 \
      --out outputs/mmred_hf/armB_dumps/seq_len_8_trainfit.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/mmred_hf"))

from facts import frame_labels  # noqa: E402
from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.engine import CarrierEngine  # noqa: E402
from gnnformer.mmred_hf import qtype_from_dirname  # noqa: E402
from gnnformer.runtime import load_runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--qtype-roots", default=None, metavar="CONFIG",
                    help="use data/mmred_hf/dirs/<CONFIG>_train_<qtype> for all 24 qtypes")
    ap.add_argument("--limit-per-qtype", type=int, default=100)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--trunc", type=int, default=12)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--carrier-ckpt", default="checkpoints/carrier_token_room_k1_best.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rt = load_runtime()
    tk = torch.load(args.carrier_ckpt, map_location="cpu", weights_only=False)
    e_c = (tk["e_c"] if isinstance(tk, dict) else tk.e_c).float().to(rt.device)
    eng = CarrierEngine(rt, l_open=12, e_c=e_c)

    if args.qtype_roots:
        from gnnformer.mmred_hf import DC_QTYPES, NIAH_QTYPES
        dirs = []
        for qt in NIAH_QTYPES + DC_QTYPES:
            r = _REPO / f"data/mmred_hf/dirs/{args.qtype_roots}_train_{qt}"
            dirs += sorted(r.iterdir())[: args.limit_per_qtype]
    else:
        dirs = sorted(Path(args.root).iterdir())[: args.limit]

    X, Y, QT, SID, FI, NN, GOLD = [], [], [], [], [], [], []
    n_done = n_skip = 0
    t0 = time.time()
    for sd in dirs:
        qt = qtype_from_dirname(sd.name)
        if qt is None:
            n_skip += 1
            continue
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            labs, _space = frame_labels(qt, q0, states)
        except Exception:
            n_skip += 1
            continue
        rec = eng.prepare_sample(frames, q0, gold=0, task=qt, resize=args.resize,
                                 with_masks=True, with_trunc_cols=True)
        if rec is None:
            n_skip += 1
            continue
        with torch.no_grad():
            caches, *_ = eng.prefill_capture(rec, args.trunc)
        kk = rec["keep"]
        st_l = caches[min(args.layer, len(caches) - 1)]
        for t, c in enumerate([kk.index(p) for p in rec["cpos"]]):
            X.append(st_l[c].to(torch.float16).cpu().numpy())
            Y.append(labs[t])
            QT.append(qt)
            SID.append(sd.name)
            FI.append(t)
            NN.append(len(states))
            GOLD.append(str(a0))
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} samples ({n_skip} skip) {time.time()-t0:.0f}s "
                  f"{len(Y)} frames", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(X), y=np.array(Y),
                        qtype=np.array(QT), sid=np.array(SID),
                        fidx=np.array(FI), N=np.array(NN), gold=np.array(GOLD))
    print(f"DUMP DONE: {n_done} samples / {len(Y)} frames -> {out} "
          f"({out.stat().st_size/2**20:.0f} MB, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
