#!/usr/bin/env python3
"""E5/E5b/E5c + H3 of RESULTS [2026-07-03b/d/e/f]: CAUSAL dose / scrub experiments at deployed
carrier tokens, multi-layer, multi-offset.

Arms (all optional, composable):
  base                       no-op baseline
  single:lam                 dose (lam-1)*g*delta_L at the PRIMARY offset, entering args.layer+1 only
  multi:lam                  same dose at every message layer in --multi-layers (per-layer delta_L)
  randmulti:lam              multi-layer dose along per-layer random directions (magnitude-matched)
  scrub@o1[+o2...]           H3 causal carrier map: label-free removal of the delta-hat_{L,o} component
                             of token-at-offset-o's state at EVERY layer in --multi-layers, for each
                             offset set listed in --scrub-offsets (e.g. "15;13;10;13+15")
  scrubrand                  control: remove a random unit axis at the primary offset, all layers

Per arm: emitted answer (candidate-digit argmax), last-token rep at --probe-layer, final-layer
last-token rep (repaired-readout column). deltas come from a --messages-cache with per-offset msgs.
Tasks: count (steps) or co_occupancy (question/gold rebuilt exactly like the carrier probe).
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["count", "co_occupancy", "rooms_visited"], default="count")
    ap.add_argument("--kchannel", action="store_true",
                    help="rooms: build per-room delta_r from labels_raw; multi-lams become TALLY doses "
                         "(lam-1)*sum_r n_r*delta_r; scrub arms remove the span{delta_r} SUBSPACE (QR); "
                         "scrubrand removes a random K-dim subspace")
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--messages-cache", default="outputs/frame_axis/probes/carrier_message/count_msgcache/count/messages_cache.pt")
    ap.add_argument("--layer", type=int, default=16, help="single-site arms: delta layer; dose enters layer+1")
    ap.add_argument("--multi-layers", default="14,16,18,20")
    ap.add_argument("--offset", type=int, default=9, help="PRIMARY carrier offset (dose arms)")
    ap.add_argument("--scrub-offsets", default="", help="H3: ';'-separated offset sets, '+' joins, e.g. '15;13;10;13+15'")
    ap.add_argument("--scrub-offsets2", default="", help="second scrub window (offsets) applied at --multi-layers2")
    ap.add_argument("--multi-layers2", default="18,20")
    ap.add_argument("--scramble-offsets", default="", help="';'-separated offsets: add matched-norm fixed-direction noise "
                    "to that token at --scramble-layers (delta-free early-layer intervention)")
    ap.add_argument("--scramble-layers", default="2,4,6,8,10,12")
    ap.add_argument("--probe-layer", type=int, default=24)
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--lams-single", default="")
    ap.add_argument("--lams-multi", default="4,16")
    ap.add_argument("--scrub-random", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rand-lam", type=float, default=0.0)
    ap.add_argument("--rescue-lams", default="", help="scrub dhat at all --multi-layers THEN re-inject lam*g*delta at the LAST multi-layer; e.g. '1,4'")
    ap.add_argument("--setg", action="store_true", help="write-then-read: remove the count component at all multi-layers and write a CHOSEN g'!=gold; score emitted+repaired vs g'")
    ap.add_argument("--noise-sigmas", default="", help="add fresh Gaussian noise of sigma*||delta_L|| along the WHITENED axis at all multi-layers; e.g. '0.5,1,2,4,8'")
    ap.add_argument("--native-axes", default="", help="native_axes.pt from native_axis_probe; enables nat* arms at the layers it contains")
    ap.add_argument("--nat-lams", default="", help="natdose: add (lam-1)*g*||delta_L||*u_native at native-axis layers")
    ap.add_argument("--nat-scrub", action="store_true", help="natscrub: project out the native axis at its layers")
    ap.add_argument("--nat-set", action="store_true", help="natset: erase native component, write target*||delta_L||*u_native")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/probes/dprime_dose")
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    mls = [int(x) for x in str(args.multi_layers).replace(",", " ").split()]
    scrub_sets = [tuple(int(x) for x in s.split("+")) for s in str(args.scrub_offsets).split(";") if s.strip()]
    scrub_sets2 = [tuple(int(x) for x in s.split("+")) for s in str(args.scrub_offsets2).split(";") if s.strip()]
    mls2 = [int(x) for x in str(args.multi_layers2).replace(",", " ").split() if x.strip()]
    scram_offs = [int(x) for x in str(args.scramble_offsets).replace(";", " ").split() if x.strip()]
    scram_layers = [int(x) for x in str(args.scramble_layers).replace(",", " ").split() if x.strip()]
    need_offsets = sorted({int(args.offset)} | {o for st_ in scrub_sets for o in st_} | {o for st_ in scrub_sets2 for o in st_})
    _natL = []
    if args.native_axes:
        _natL = [int(L) for L in torch.load(args.native_axes, map_location="cpu", weights_only=False)["axes"].keys()]
    need_layers = sorted(set(mls + (mls2 if scrub_sets2 else []) + [int(args.layer)] + (scram_layers if scram_offs else []) + _natL))

    mc = torch.load(args.messages_cache, map_location="cpu", weights_only=False)
    delta, dhat, rnd = {}, {}, {}
    deltaK, Qk, Qkrand = {}, {}, {}       # kchannel: per-room deltas, subspace bases
    room_list = []
    rng = np.random.RandomState(0)
    if args.kchannel:
        lab_arr = np.array([[str(x) for x in row] for row in mc["labels_raw"]])
        room_list = sorted(set(lab_arr.reshape(-1)) - {"None"})
        for L in need_layers:
            deltaK[L], Qk[L] = {}, {}
            for o in need_offsets:
                M = mc["msgs"][L][o].astype(np.float32)
                flat = M.reshape(-1, M.shape[-1]); fl = lab_arr.reshape(-1)
                ds = {r_: flat[fl == r_].mean(0) - flat[fl != r_].mean(0) for r_ in room_list}
                deltaK[L][o] = ds
                B = np.stack([ds[r_] / (np.linalg.norm(ds[r_]) + 1e-12) for r_ in room_list], 1)
                Qk[L][o] = np.linalg.qr(B)[0].astype(np.float32)         # [H, K] orthonormal
            R = rng.randn(M.shape[-1], len(room_list)).astype(np.float32)
            Qkrand[L] = np.linalg.qr(R)[0].astype(np.float32)
            print(f"L{L}: K={len(room_list)} rooms " +
                  " ".join(f"|d({r_})|={np.linalg.norm(deltaK[L][int(args.offset)][r_]):.2f}" for r_ in room_list))
    else:
        lab = mc["labels"].astype(bool)
        for L in need_layers:
            delta[L], dhat[L] = {}, {}
            for o in need_offsets:
                if L not in mc["msgs"] or o not in mc["msgs"][L]:
                    continue                      # scramble-only layers/offsets need no delta
                M = mc["msgs"][L][o].astype(np.float32)
                d = M[lab].mean(0) - M[~lab].mean(0)
                delta[L][o] = d
                dhat[L][o] = d / (np.linalg.norm(d) + 1e-12)
            if delta[L]:
                d0 = next(iter(delta[L].values()))
                r = rng.randn(d0.shape[0]).astype(np.float32)
                ref = delta[L].get(int(args.offset), d0)
                rnd[L] = r * (np.linalg.norm(ref) / np.linalg.norm(r))
                print(f"L{L}: " + " ".join(f"|d(off{o})|={np.linalg.norm(delta[L][o]):.3f}" for o in delta[L]))
            else:
                print(f"L{L}: no cached deltas (scramble-only layer)")

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    dev = next(layers[0].parameters()).device
    if args.kchannel:
        deltaK_t = {L: {o: {r_: torch.tensor(deltaK[L][o][r_], device=dev, dtype=torch.bfloat16)
                            for r_ in room_list} for o in need_offsets} for L in need_layers}
        Qk_t = {L: {o: torch.tensor(Qk[L][o], device=dev, dtype=torch.float32) for o in need_offsets}
                for L in need_layers}
        Qkrand_t = {L: torch.tensor(Qkrand[L], device=dev, dtype=torch.float32) for L in need_layers}
        delta_t = dhat_t = rnd_t = rhat_t = None
    else:
        delta_t = {L: {o: torch.tensor(delta[L][o], device=dev, dtype=torch.bfloat16) for o in delta[L]}
               for L in need_layers}
        dhat_t = {L: {o: torch.tensor(dhat[L][o], device=dev, dtype=torch.float32) for o in dhat[L]}
                  for L in need_layers}
        rnd_t = {L: torch.tensor(rnd[L], device=dev, dtype=torch.bfloat16) for L in rnd}
        rhat_t = {L: torch.tensor(rnd[L] / (np.linalg.norm(rnd[L]) + 1e-12), device=dev, dtype=torch.float32)
                  for L in rnd}

    what_t = {}
    if args.noise_sigmas and not args.kchannel:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lab_ = mc["labels"].astype(int)
        for L in mls:
            M_ = mc["msgs"][L][int(args.offset)].astype(np.float32)
            X_ = M_.reshape(-1, M_.shape[-1]); y_ = lab_.reshape(-1)
            sub_ = np.random.RandomState(0).permutation(len(X_))[:4000]
            ld_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X_[sub_], y_[sub_])
            w_ = ld_.coef_[0] / (np.linalg.norm(ld_.coef_[0]) + 1e-12)
            what_t[L] = torch.tensor(w_, device=dev, dtype=torch.float32)
            print(f"L{L}: whitened axis fitted for noise arms (cos to dhat={float(w_ @ dhat[L][int(args.offset)]):.3f})")

    nat_t, nat_layers = {}, []
    if args.native_axes:
        na = torch.load(args.native_axes, map_location="cpu", weights_only=False)
        nat_layers = [int(L) for L in na["axes"].keys()]
        for L in nat_layers:
            ax = np.asarray(na["axes"][L]["axis"], dtype=np.float32)
            nat_t[L] = torch.tensor(ax / (np.linalg.norm(ax) + 1e-12), device=dev, dtype=torch.float32)
        print(f"native axes loaded for layers {nat_layers} (offset {na.get('offset')})")

    scram_dir_t = {}
    if scram_offs:
        H_ = 3584
        for L in scram_layers:
            v = np.random.RandomState(100 + L).randn(H_).astype(np.float32)
            scram_dir_t[L] = torch.tensor(v / np.linalg.norm(v), device=dev, dtype=torch.float32)

    tok = processor.tokenizer
    cand_ids, cand_vals = [], []
    for d_ in range(0, 9):
        enc = tok.encode(str(d_), add_special_tokens=False)
        if len(enc) == 1:
            cand_ids.append(int(enc[0])); cand_vals.append(d_)
    cand_ids_t = torch.tensor(cand_ids, dtype=torch.long)

    st = {"posmap": {}, "mode": "base", "param": None, "gold": 0, "probe_rep": None, "final_rep": None}

    def edit_at(L, hs):
        mode, prm, g = st["mode"], st["param"], st["gold"]
        P = st["posmap"]
        if mode == "base":
            return hs
        if mode in ("single", "multi"):
            if (mode == "single" and L != int(args.layer)) or (mode == "multi" and L not in mls):
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            if args.kchannel:                      # TALLY dose: (lam-1) * sum_r n_r * delta_r
                v = None
                for r_, n_r in st["nvis"].items():
                    if r_ in deltaK_t[L][int(args.offset)]:
                        t_ = float(n_r) * deltaK_t[L][int(args.offset)][r_]
                        v = t_ if v is None else v + t_
                if v is None:
                    return hs
                hs[0, pos, :] += (prm - 1.0) * v
            else:
                hs[0, pos, :] += (prm - 1.0) * g * delta_t[L][int(args.offset)]
            return hs
        if mode == "randmulti":
            if L not in mls:
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone(); hs[0, pos, :] += (prm - 1.0) * g * rnd_t[L]
            return hs
        if mode in ("scrub", "scrub2"):
            if L not in (mls if mode == "scrub" else mls2):
                return hs
            hs = hs.clone()
            for o in prm:
                pos = P[o]
                if hs.shape[1] <= pos:
                    continue
                h = hs[0, pos, :].float()
                if args.kchannel:                  # subspace scrub: h -= Q Q^T h  (span of all delta_r)
                    Q = Qk_t[L][o]
                    hs[0, pos, :] = (h - Q @ (Q.T @ h)).to(hs.dtype)
                else:
                    u = dhat_t[L][o]
                    hs[0, pos, :] = (h - (h @ u) * u).to(hs.dtype)
            return hs
        if mode == "rescue":
            if L not in mls:
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            h = hs[0, pos, :].float()
            u = dhat_t[L][int(args.offset)]
            h = h - (h @ u) * u                      # scrub at every layer...
            hs[0, pos, :] = h.to(hs.dtype)
            inj_L = min(mls) if float(prm) >= 0 else max(mls)   # +lam: inject EARLY (inside the
            if L == inj_L:                                       # carrier->last transfer window);
                hs[0, pos, :] += (abs(prm) * g * delta_t[L][int(args.offset)]).to(hs.dtype)
            return hs
        if mode == "setg":
            if L not in mls:
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            h = hs[0, pos, :].float()
            u = dhat_t[L][int(args.offset)]
            h = h - (h @ u) * u                      # erase the real count...
            hs[0, pos, :] = h.to(hs.dtype)
            hs[0, pos, :] += (float(st["setg_target"]) * delta_t[L][int(args.offset)]).to(hs.dtype)
            return hs                                 # ...write the chosen one
        if mode == "noisew":
            if L not in mls:
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            eps = float(st["noise_draw"].get(L, 0.0))
            dn = float(np.linalg.norm(delta[L][int(args.offset)]))
            hs[0, pos, :] += (eps * prm * dn * what_t[L]).to(hs.dtype)
            return hs
        if mode in ("natdose", "natscrub", "natset"):
            if L not in nat_t:
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            u = nat_t[L]
            sc = float(np.linalg.norm(delta[L][int(args.offset)])) if (L in delta and int(args.offset) in delta[L]) else 1.0
            h = hs[0, pos, :].float()
            if mode == "natdose":
                hs[0, pos, :] = (h + (prm - 1.0) * g * sc * u).to(hs.dtype)
            elif mode == "natscrub":
                hs[0, pos, :] = (h - (h @ u) * u).to(hs.dtype)
            else:  # natset
                h = h - (h @ u) * u
                hs[0, pos, :] = (h + float(st["setg_target"]) * sc * u).to(hs.dtype)
            return hs
        if mode == "scramble":
            if L not in scram_layers:
                return hs
            pos = P[prm]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            h = hs[0, pos, :].float()
            u = scram_dir_t[L]
            hs[0, pos, :] = (h + u * h.norm()).to(hs.dtype)   # matched-norm fixed-direction noise
            return hs
        if mode == "scrubrand":
            if L not in mls:
                return hs
            pos = P[int(args.offset)]
            if hs.shape[1] <= pos:
                return hs
            hs = hs.clone()
            h = hs[0, pos, :].float()
            if args.kchannel:                      # random K-dim subspace control
                Q = Qkrand_t[L]
                hs[0, pos, :] = (h - Q @ (Q.T @ h)).to(hs.dtype)
            else:
                u = rhat_t[L]
                hs[0, pos, :] = (h - (h @ u) * u).to(hs.dtype)
            return hs
        return hs

    def mk_pre(L):
        def pre_hook(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is None or st["mode"] == "base":
                return hargs, hkwargs
            hs = edit_at(L, hs)
            if hargs:
                return (hs,) + tuple(hargs[1:]), hkwargs
            hkwargs = dict(hkwargs); hkwargs["hidden_states"] = hs
            return hargs, hkwargs
        return pre_hook
    for L in need_layers:
        layers[L + 1].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    def probe_hook(_m, hargs, hkwargs):
        hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
        if hs is not None:
            st["probe_rep"] = hs[0, -1, :].float().cpu().clone()
        return hargs, hkwargs
    layers[int(args.probe_layer)].register_forward_pre_hook(probe_hook, with_kwargs=True)

    def final_hook(_m, _inp, outp):
        h = outp[0] if isinstance(outp, tuple) else outp
        st["final_rep"] = h[0, -1, :].float().cpu().clone()
    layers[-1].register_forward_hook(final_hook)

    arms = [("base", 1.0)]
    arms += [("single", float(x)) for x in str(args.lams_single).replace(",", " ").split() if x]
    arms += [("multi", float(x)) for x in str(args.lams_multi).replace(",", " ").split() if x]
    arms += [("scrub", t) for t in scrub_sets]
    arms += [("scrub2", t) for t in scrub_sets2]
    arms += [("scramble", o) for o in scram_offs]
    if args.scrub_random:
        arms.append(("scrubrand", 0.0))
    if args.rand_lam:
        arms.append(("randmulti", float(args.rand_lam)))
    arms += [("rescue", float(x)) for x in str(args.rescue_lams).replace(",", " ").split() if x]
    if args.setg:
        arms.append(("setg", 1.0))
    arms += [("noisew", float(x)) for x in str(args.noise_sigmas).replace(",", " ").split() if x]
    arms += [("natdose", float(x)) for x in str(args.nat_lams).replace(",", " ").split() if x]
    if args.nat_scrub:
        arms.append(("natscrub", 0.0))
    if args.nat_set:
        arms.append(("natset", 1.0))

    def arm_name(a):
        m, p = a
        if m == "scrub":
            return f"scrub@{'+'.join(map(str, p))}"
        if m == "scrub2":
            return f"scrubLATE@{'+'.join(map(str, p))}"
        if m == "scramble":
            return f"scramble@{p}"
        return f"{m}:{p:g}"

    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    res = {a: {"pred": [], "gold": []} for a in arms}
    reps = {a: [] for a in arms}; reps_final = {a: [] for a in arms}
    tok_ctr = {o: Counter() for o in need_offsets}
    n = 0
    for sd in all_dirs:
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            if args.task == "co_occupancy":
                from evaluations.scripts.patch_importence.probe_frame_to_carrier_message import pick_pair
                chars = sorted(eval_utils.extract_characters_from_states(states))
                if len(chars) < 2:
                    continue
                (c1, c2), gold = pick_pair(states, chars)
                q0 = f"In how many of the {len(frames)} frames were {c1} and {c2} in the same room?"
            elif args.task == "rooms_visited":
                from evaluations.scripts.patch_importence.probe_frame_to_carrier_message import char_room_at
                chars = sorted(eval_utils.extract_characters_from_states(states))
                present = lambda c: [t for t in range(len(states)) if char_room_at(states, t, c)]
                char = max(chars, key=lambda c: (len(present(c)), c))
                pres = present(char)
                if len(pres) < 2:
                    continue
                visits = [char_room_at(states, t, char) for t in pres]
                gold = len(set(visits))
                st["nvis"] = dict(Counter(visits))
                q0 = f"How many distinct rooms did {char} visit?"
            else:
                gold = int(str(a0).strip())
        except Exception:
            continue
        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
            ids = inputs["input_ids"][0].detach().cpu()
            seq = int(ids.shape[0])
            st["posmap"] = {o: seq - 1 - o for o in sorted(set(need_offsets) | set(scram_offs))}
            for o in need_offsets:
                tok_ctr[o][tok.decode([int(ids[seq - 1 - o])]).strip()] += 1
            st["gold"] = gold
            st["setg_target"] = int((gold + 1 + n % 7) % 9)
            for arm in arms:
                st["mode"], st["param"] = arm
                if arm[0] == "noisew":
                    rs = np.random.RandomState(10000 + n * 31 + int(arm[1] * 8))
                    st["noise_draw"] = {L: float(rs.randn()) for L in mls}
                st["probe_rep"] = None; st["final_rep"] = None
                with torch.no_grad():
                    outp = model(**inputs, use_cache=False)
                logits = outp.logits[0, -1].float().cpu()
                pred = int(cand_vals[int(torch.argmax(logits[cand_ids_t]).item())])
                res[arm]["pred"].append(pred); res[arm]["gold"].append(gold)
                res[arm].setdefault("target", []).append(st["setg_target"] if arm[0] == "setg" else gold)
                reps[arm].append(st["probe_rep"].half()); reps_final[arm].append(st["final_rep"].half())
            st["mode"] = "base"
            n += 1
            if n % 10 == 0:
                accs = {arm_name(a): np.mean(np.array(res[a]["pred"]) == np.array(res[a]["gold"])) for a in arms}
                print(f"  {n}: " + "  ".join(f"{k}={v:.3f}" for k, v in accs.items()), flush=True)
        except Exception as exc:
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            continue

    lines = [f"=== d' DOSE/SCRUB v4 ({args.task}; primary off{args.offset}; layers {mls}; n={n}; "
             f"carrier tokens {[ (o, tok_ctr[o].most_common(2)) for o in need_offsets ]}) ==="]
    rows = ["arm,n,acc,mae"]
    for arm in arms:
        p = np.array(res[arm]["pred"]); g = np.array(res[arm]["gold"])
        acc = float(np.mean(p == g)); mae = float(np.mean(np.abs(p - g)))
        by = " ".join(f"g{gv}:{np.mean(p[g == gv] == gv):.2f}" for gv in sorted(set(g.tolist())))
        if arm[0] in ("setg", "natset"):
            t_ = np.array(res[arm]["target"])
            lines.append(f"  {'setg vs TARGET':>13} acc={float(np.mean(p == t_)):.3f}  (emitted follows the WRITTEN count?)")
        lines.append(f"  {arm_name(arm):>13} acc={acc:.3f}  MAE={mae:.2f}   [{by}]")
        rows.append(f"{arm_name(arm)},{len(p)},{acc:.4f},{mae:.4f}")
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report)
    (out / "metrics.csv").write_text("\n".join(rows) + "\n")
    torch.save({"reps": {arm_name(a): torch.stack(v) for a, v in reps.items() if v},
                "reps_final": {arm_name(a): torch.stack(v) for a, v in reps_final.items() if v},
                "res": {arm_name(a): v for a, v in res.items()},
                "config": vars(args)}, out / "doses.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
