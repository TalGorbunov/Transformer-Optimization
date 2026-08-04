#!/usr/bin/env python3
"""Stage 3: STATE-matching distillation — carrier h@L12 <- stage-2 span-mean L16 teacher.

Student: deployed carrier layout [q_pf][frame+carrier]xN[q_pf tail] (prepare_sample),
e_c trainable (init: park e_c), lo-phase forward (layers 0..11, fenced, posreset);
loss = MSE(h@L12 at carrier t, teacher_target(sample, t)) over the target mixture.
Prompt embeddings/masks/positions cached once per sample (only e_c changes).

Eval per epoch: held-out cosine-to-teacher + per-fact linear-probe acc on student
states + ZERO-SHOT transfer of teacher-fit heads (the convergence metric).

Usage:
  python scripts/mmred_hf/stage3_distill.py \
      --targets outputs/mmred_hf/stage3_targets/*.npz --epochs 12 \
      --output outputs/mmred_hf/stage3/k1_mix
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/mmred_hf"))

from gnnformer.data import load_mmred_sample  # noqa: E402
from gnnformer.engine import SDPA_BACKENDS, CarrierEngine  # noqa: E402
from gnnformer.runtime import get_layers, load_runtime  # noqa: E402
from torch.nn.attention import sdpa_kernel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux-ce", type=float, default=0.0,
                    help=">0: add aux-CE loss from per-fact linear heads on h (direct "
                         "decodability gradient to e_c; heads discarded at eval)")
    ap.add_argument("--student-layer", type=int, default=12, choices=(12, 16),
                    help="16: run open phase 12-15 (hi mask) and match h@L16 — "
                         "depth-aligned with the L16 teacher read")
    ap.add_argument("--carrier-ckpt", default="checkpoints/carrier_token_room_k1_best.pt")
    ap.add_argument("--output", default="outputs/mmred_hf/stage3/run")
    args = ap.parse_args()

    rt = load_runtime()
    tk = torch.load(args.carrier_ckpt, map_location="cpu", weights_only=False)
    e0 = (tk["e_c"] if isinstance(tk, dict) else tk.e_c).float()
    eng = CarrierEngine(rt, l_open=12, e_c=e0.to(rt.device))
    dev = rt.device
    layers = get_layers(rt.model)[: args.student_layer]

    # ---- load targets, group per sample
    by_sample = {}
    for p in sorted(sum([glob.glob(x) for x in args.targets], [])):
        d = np.load(p, allow_pickle=False)
        root = str(d["root"])
        for i in range(len(d["fidx"])):
            key = (root, str(d["sid"][i]), str(d["q_pf"][i]))
            by_sample.setdefault(key, {})[int(d["fidx"][i])] = (
                d["target"][i].astype(np.float32), int(d["y"][i]))
    keys = sorted(by_sample)
    print(f"{len(keys)} (sample,q_pf) pairs / "
          f"{sum(len(v) for v in by_sample.values())} frame targets", flush=True)

    # ---- prep cache (embeddings sans e_c, masks, positions)
    cache = []
    t0 = time.time()
    for key in keys:
        root, sid, q_pf = key
        try:
            _s, frames, _q, states, _a = load_mmred_sample(
                _REPO / f"data/mmred_hf/dirs/{root}/{sid}")
        except Exception:
            continue
        rec = eng.prepare_sample(frames, q_pf, gold=0, task="x", resize=392,
                                 with_masks=True, with_trunc_cols=False)
        if rec is None:
            continue
        tmap = by_sample[key]
        tgt = np.stack([tmap[t][0] for t in sorted(tmap)])
        tidx = sorted(tmap)
        fact = ("roomofc" if "char_at_frame" in root else
                "occofr" if "room_at_frame" in root else
                "trig" if "char_on_char" in root else "empty")
        cache.append(dict(emb=rec["emb"].cpu(), pos=rec["pos"].cpu(),
                          lo=rec["lo"], hi=rec["hi"], cpos=rec["cpos"], seq=rec["seq"],
                          tgt=torch.tensor(tgt), tidx=tidx, fact=fact,
                          y=[tmap[t][1] for t in tidx], key=key))
        if len(cache) % 100 == 0:
            print(f"  prep {len(cache)} {time.time()-t0:.0f}s", flush=True)
    print(f"prep done: {len(cache)} samples {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cache))
    n_tr = int(len(cache) * args.train_frac)
    tr_idx, ev_idx = order[:n_tr], order[n_tr:]

    e_c = torch.nn.Parameter(e0.clone().to(dev))
    # affine bridge: optimize LINEAR DECODABILITY, not raw state==target geometry
    # (raw MSE just shrinks norms: ep1-5 control showed loss down, cos/probe flat)
    W = torch.nn.Linear(e0.numel(), e0.numel(), bias=True).to(dev)
    torch.nn.init.eye_(W.weight)
    torch.nn.init.zeros_(W.bias)
    n_cls = {"roomofc": 6, "occofr": 7, "trig": 2, "empty": 2}
    heads = torch.nn.ModuleDict({f: torch.nn.Linear(e0.numel(), n)
                                 for f, n in n_cls.items()}).to(dev)
    opt = torch.optim.Adam([{"params": [e_c], "lr": args.lr},
                            {"params": W.parameters(), "lr": 1e-4},
                            {"params": heads.parameters(), "lr": 1e-3}])

    def student_states(c, grad):
        emb = c["emb"].to(dev).unsqueeze(0).clone().to(torch.bfloat16)
        cp = torch.tensor(c["cpos"], device=dev)
        emb[0, cp] = (e_c if grad else e_c.detach()).to(torch.bfloat16)
        lo = c["lo"].to(dev).to(torch.float32).view(1, 1, c["seq"], c["seq"])
        hi = c["hi"].to(dev).to(torch.float32).view(1, 1, c["seq"], c["seq"])
        pos = c["pos"].to(dev)
        cos_, sin_ = eng.text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        h = emb
        with sdpa_kernel(SDPA_BACKENDS):
            for li, ly in enumerate(layers):
                h = ly(h, attention_mask=(lo if li < 12 else hi),
                       position_embeddings=pe)[0]
        return h[0, cp][c["tidx"]]

    out = _REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    best = -1.0
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        tot = nb = 0.0
        for i in tr_idx:
            c = cache[i]
            opt.zero_grad()
            hs = student_states(c, True).float()
            loss = torch.nn.functional.mse_loss(W(hs), c["tgt"].to(dev))
            if args.aux_ce > 0:
                yy = torch.tensor(c["y"], device=dev)
                loss = loss + args.aux_ce * torch.nn.functional.cross_entropy(
                    heads[c["fact"]](hs), yy)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        # eval: cosine + probe on held-out
        with torch.no_grad():
            HS, WHS, TG, YY, FE = [], [], [], [], []
            for i in ev_idx:
                c = cache[i]
                hs_raw = student_states(c, False).float()
                hs = hs_raw.cpu()
                HS.append(hs)
                WHS.append(W(hs_raw).detach().cpu())
                TG.append(c["tgt"])
                YY += c["y"]
                FE += [c["fact"]] * len(c["y"])
            HS = torch.cat(HS).numpy()
            WH = torch.cat(WHS).numpy()
            TG = torch.cat(TG).numpy()
            y = np.array(YY)
        cos = float(np.mean(np.sum(WH * TG, 1) /
                            (np.linalg.norm(WH, axis=1) * np.linalg.norm(TG, axis=1) + 1e-8)))
        facts_ev = np.array(FE)
        per = []
        accs_all = []
        for f in sorted(set(facts_ev)):
            m = facts_ev == f
            ym, Hm = y[m], HS[m]
            if len(ym) < 40 or len(set(ym)) < 2:
                continue
            idx = np.random.default_rng(0).permutation(len(ym))
            hh = len(ym) // 2
            clf = LogisticRegression(max_iter=1000).fit(Hm[idx[:hh]], ym[idx[:hh]])
            a = float(clf.score(Hm[idx[hh:]], ym[idx[hh:]]))
            per.append(f"{f}:{a:.3f}")
            accs_all.append(a)
        acc = float(np.mean(accs_all)) if accs_all else 0.0
        print(f"[ep {ep}] loss {tot/max(nb,1):.4f} cos {cos:.3f} "
              f"probe(mean) {acc:.3f} [{' '.join(per)}]", flush=True)
        if acc > best:
            best = acc
            torch.save({"e_c": e_c.detach().cpu(), "W": W.state_dict(),
                        "epoch": ep, "probe": acc, "cos": cos}, out / "e_c_best.pt")
    print(f"BEST probe {best:.3f} -> {out}/e_c_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
