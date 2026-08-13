#!/usr/bin/env python3
"""Stage 2b: do the EXISTING carrier's MESSAGES carry the facts? (deployed layout)

Layout: [q_pf][frame_1+carrier]...[frame_N+carrier][q_pf tail] via
CarrierEngine.prepare_sample (park e_c, fence lo / open hi, posreset), q_pf = the
per-frame fact question (conditioning matched to stage 2). One full forward with
FenceHooks capturing qkv at L16 (no mask injection — the engine applies its own);
then per-carrier messages to every tail position, span-mean pooled, linear probes.

Gap stage2 (frame->replica messages, ceiling) vs stage2b (carrier->tail messages)
= what the CARRIER loses; decides whether stage-3 distillation is needed.

Usage: python scripts/mmred_hf/stage2b_carrier_msg.py --task occofr --limit 120
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

from stage1_alone import TASKS as _T1, build  # noqa: E402
TASKS = dict(_T1)
TASKS["match"] = ("seq_len_8_train_steps_in_room", "steps_in_room")
from stage2_pcw import pf_question  # noqa: E402
from gnnformer.carriers import ext_mask, make_masks  # noqa: E402
from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.engine import CarrierEngine  # noqa: E402
from gnnformer.fencing import FenceHooks, locate_word_token, recompute_messages  # noqa: E402
from gnnformer.mmred_hf import ROOM_ORDER, _match, _rooms  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    attention_dims, dequantize_linear_weight, get_layers, load_runtime,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--carrier-ckpt", default="checkpoints/carrier_token_room_k1_best.pt")
    ap.add_argument("--output", default="outputs/mmred_hf/stage2b")
    args = ap.parse_args()

    rt = load_runtime()
    layers = get_layers(rt.model)
    L = args.layer
    dims = attention_dims(rt.model)
    w_o = dequantize_linear_weight(layers[L].self_attn.o_proj)
    tk = torch.load(args.carrier_ckpt, map_location="cpu", weights_only=False)
    e_c = (tk["e_c"] if isinstance(tk, dict) else tk.e_c).float().to(rt.device)
    eng = CarrierEngine(rt, l_open=12, e_c=e_c)
    hooks = FenceHooks(layers, capture_layers=[L]).install()

    root_name, qtype = TASKS[args.task]
    import random as _rnd
    dirs = sorted((_REPO / f"data/mmred_hf/dirs/{root_name}").iterdir())
    _rnd.Random(0).shuffle(dirs)
    dirs = dirs[: args.limit]

    X, ys = [], []  # X: list of {locus: vec}
    n_done = n_skip = 0
    t0 = time.time()
    for si, sd in enumerate(dirs):
        try:
            _sid, frames, q0, states, _a = load_mmred_sample(sd)
            g = _match(qtype, q0)
        except Exception:
            n_skip += 1
            continue
        if args.task == "empty":
            room = ROOM_ORDER[si % 6]
            q_pf = f"Is the {room} empty? Answer Yes or No."
        elif args.task == "match":
            q_pf = q0  # the sample's own steps question (the distilled read)
        else:
            q_pf = pf_question(args.task, g, 0)
        rec = eng.prepare_sample(frames, q_pf, gold=0, task=qtype, resize=392,
                                 with_masks=True, with_trunc_cols=False)
        if rec is None:
            n_skip += 1
            continue
        # full deployed forward (fenced lo / open hi), qkv captured at L via hooks
        with torch.no_grad():
            _lg = eng.forward_logits(rec, False)
        seq = rec["seq"]
        NF = len(rec["cpos"])
        lo2, hi2 = make_masks(seq, rec["blocks"], rec["cpos"], rec["fin"])
        tail = list(range(rec["fin"], seq))
        labs = []
        for t in range(NF):
            if args.task == "match":
                labs.append(int(g[0] in _rooms(states[t]).get(g[1], [])))
                continue
            if args.task == "empty":
                labs.append(int(not _rooms(states[t]).get(room, [])))
            else:
                b = build(args.task, g, states, t)
                labs.append(None if b is None else b[2])
        common = dict(seq=seq, cos=hooks.cos, sin=hooks.sin, dims=dims, w_o=w_o,
                      q_proj=hooks.qkv[L]["q_proj"], k_proj=hooks.qkv[L]["k_proj"],
                      v_proj=hooks.qkv[L]["v_proj"], mask_full=hi2)
        # keyword locus in the TAIL copy of q_pf (room word / char name), else last token
        ids_full = None
        if args.task == "roomofc":
            kw = g[0]
        elif args.task == "occofr":
            kw = g[0]
        elif args.task == "trig":
            kw = g[2]
        elif args.task == "match":
            kw = g[1]  # the queried room word (the distilled locus)
        else:
            kw = room
        tail_span = (rec["fin"], seq)
        kw_c = None
        if kw is not None:
            kw_c = locate_word_token(rec["ids"], eng.tok, kw, tail_span)
        loci = {"kw": kw_c if kw_c is not None else seq - 1,
                "lastt": seq - 1}
        for t in range(NF):
            if labs[t] is None:
                continue
            row = {}
            for name, c in loci.items():
                msg = recompute_messages(
                    carrier_positions=[c],
                    vis_by_frame=[torch.tensor([rec["cpos"][t]], dtype=torch.long)],
                    **common)
                row[name] = np.asarray(msg[0], dtype=np.float32)
            X.append(row)
            ys.append(labs[t])
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done} samples ({n_skip} skip) {time.time()-t0:.0f}s", flush=True)

    hooks.remove()
    y = np.array(ys)
    from sklearn.linear_model import LogisticRegression
    maj = float(np.bincount(y).max()) / len(y)
    parts = []
    for name in X[0]:
        Xa = np.stack([r[name] for r in X])
        accs = []
        for seed in range(5):
            idx = np.random.default_rng(seed).permutation(len(y))
            h = len(y) // 2
            clf = LogisticRegression(max_iter=2000).fit(Xa[idx[:h]], y[idx[:h]])
            accs.append(float(clf.score(Xa[idx[h:]], y[idx[h:]])))
        parts.append(f"{name} {np.mean(accs):.3f}±{np.std(accs):.3f}")
    line = (f"STAGE2B CARRIER-MSG [{args.task}] L{L} n={len(y)}: "
            + "  ".join(parts) + f"  (maj {maj:.3f})")
    out = _REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}.txt").write_text(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
