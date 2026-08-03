#!/usr/bin/env python3
"""Content probe: is per-frame ROOM IDENTITY linearly present in the carrier states?

Arm B only ever probed the binary relevance bit (gate->tally, steps family). This
answers the question the 2026-08-03 content-transport failure raises: for
char_at_frame samples ("In which room was C at step k?"), extract the FROZEN
carrier states (e_c injected, fenced lo phase, NO LoRA) at layer L, label each
frame with C's room (6-way), and fit a multinomial logistic probe (5 seeds,
50/50 split). High acc => content is in the states, the generative reader is the
gap. Low acc => e_c is a detection carrier; content needs a new distill objective.

Usage:
  python scripts/mmred_hf/content_probe.py \
      --data_root data/mmred_hf/dirs/seq_len_8_train_char_at_frame --limit 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gnnformer.carriers import load_carrier_layer_ckpt  # noqa: E402
from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.engine import CarrierEngine  # noqa: E402
from gnnformer.mmred_hf import ROOM_ORDER, _char_room, _match  # noqa: E402
from gnnformer.runtime import load_runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--carrier-ckpt", default="checkpoints/carrier_token_room_k1_best.pt")
    ap.add_argument("--layer-ckpt", default=None,
                    help="optional carrier_layer ckpt: probe POST-LoRA states instead")
    ap.add_argument("--output", default="outputs/mmred_hf/content_probe")
    args = ap.parse_args()

    rt = load_runtime()
    if args.layer_ckpt:
        from gnnformer.carriers import attach_lora
        from gnnformer.runtime import get_layers
        ck = load_carrier_layer_ckpt(Path(args.layer_ckpt))
        attach_lora(get_layers(rt.model), ck.l_open, rank=ck.rank, alpha=ck.alpha,
                    device=rt.device, state=ck.lora_state)
        e_c = ck.e_c.float().to(rt.device)
        l_open = ck.l_open
    else:
        import torch as _t
        tk = _t.load(args.carrier_ckpt, map_location="cpu", weights_only=False)
        e_c = (tk["e_c"] if isinstance(tk, dict) and "e_c" in tk else tk.e_c)
        e_c = e_c.float().to(rt.device)
        l_open = 12
    eng = CarrierEngine(rt, l_open=l_open, e_c=e_c)

    X, y = [], []
    n_done = 0
    dirs = sorted(Path(args.data_root).iterdir())
    for sd in dirs:
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            char = _match("char_at_frame", q0)[0]
        except Exception:
            continue
        rec = eng.prepare_sample(frames, q0, gold=0, task="char_at_frame", resize=args.resize,
                                 with_masks=True, with_trunc_cols=True)
        if rec is None:
            continue
        with torch.no_grad():
            caches, *_ = eng.prefill_capture(rec, args.layer if args.layer <= l_open else l_open)
        kk = rec["keep"]
        car_k = [kk.index(c) for c in rec["cpos"]]
        # states at the probe layer: caches[L] rows are keep-columns (truncated coords)
        st_l = caches[min(args.layer, len(caches) - 1)]
        for t, c in enumerate(car_k):
            room = _char_room(states[t], char)
            if room is None:
                continue
            X.append(st_l[c].float().cpu().numpy())
            y.append(ROOM_ORDER.index(room))
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done} samples, {len(y)} frames", flush=True)

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    print(f"probe set: {X.shape[0]} frames x {X.shape[1]} dims; "
          f"class counts {np.bincount(y, minlength=6).tolist()}")

    from sklearn.linear_model import LogisticRegression
    accs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        h = len(y) // 2
        tr, te = idx[:h], idx[h:]
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    maj = float(np.bincount(y).max()) / len(y)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    line = (f"CONTENT PROBE (room-of-C, 6-way, L{args.layer}, rs{args.resize}, "
            f"lora={'ON' if args.layer_ckpt else 'OFF'}, n_frames={len(y)}): "
            f"acc {np.mean(accs):.3f}±{np.std(accs):.3f} (majority {maj:.3f})")
    (out / "report.txt").write_text(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
