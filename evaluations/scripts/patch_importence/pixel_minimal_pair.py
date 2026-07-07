#!/usr/bin/env python3
"""Causal #3: PIXEL-level minimal pairs. Intervention in INPUT space — no activation surgery.

Per sample (steps count task, question 'how many frames was C in R'):
  orig      unmodified
  down      one EVIDENCE frame re-rendered with C moved OUT of R          (gold -> gold-1)
  up        one NON-evidence frame re-rendered with C moved INTO R        (gold -> gold+1)
  control   same frame as 'down', but a DIFFERENT character (not C) moved between rooms
            not involving R's count of C — gold unchanged, similar pixel-level churn

Measured per arm: emitted answer; carrier-token state entering layer L+1 projected on the whitened
axis w* and on delta-hat (fitted from --messages-cache at the same layer/offset).
Predictions: down shifts the carrier projection by ~ -1 frame's worth and moves emissions down at the
law's rate; control shifts ~0; up mirrors down.
"""
from __future__ import annotations
import argparse, random, re, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("render_mmred", str(_REPO / "datasets/mmred/render_mmred.py"))
_rm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rm)
render_frame, rooms_to_room2chars = _rm.render_frame, _rm.rooms_to_room2chars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--messages-cache", default="outputs/frame_axis/probes/carrier_message/count_msgcache/count/messages_cache.pt")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/probes/pixel_minimal_pair")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    L = int(args.layer)

    # axes from the messages cache (same protocol as the parity engine)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    mc = torch.load(args.messages_cache, map_location="cpu", weights_only=False)
    M = mc["msgs"][L][int(args.offset)].astype(np.float32)
    lab = mc["labels"].astype(int)
    X = M.reshape(-1, M.shape[-1]); y = lab.reshape(-1)
    dvec = X[y == 1].mean(0) - X[y == 0].mean(0)
    dhat = dvec / (np.linalg.norm(dvec) + 1e-12)
    sub = np.random.RandomState(0).permutation(len(X))[:4000]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[sub], y[sub])
    wstar = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)
    msg_step_w = float(np.mean(X[y == 1] @ wstar) - np.mean(X[y == 0] @ wstar))
    print(f"axes ready: one evidence frame's message step along w* = {msg_step_w:.3f}")

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    tok = processor.tokenizer
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)
            if len(tok.encode(str(d), add_special_tokens=False)) == 1]
    cand_vals = [d for d, _ in cand]; cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    st = {"pos": None, "h": None}
    def cap(_m, hargs, hkwargs):
        hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
        if hs is not None and st["pos"] is not None and hs.shape[1] > st["pos"]:
            st["h"] = hs[0, st["pos"], :].detach().float().cpu().numpy()
        return hargs, hkwargs
    layers[L + 1].register_forward_pre_hook(cap, with_kwargs=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="pixpair_"))

    def rerender(states, t, mutate):
        rooms = {r: list(occ) for r, occ in rooms_to_room2chars(states[t].get("rooms", {})).items()}
        mutate(rooms)
        p = tmpdir / f"edit_{t}.png"
        render_frame(rooms, int(states[t].get("step_id", t + 1)), str(p))
        return Image.open(p).convert("RGB")

    def fwd(frames, q0, seq_hint=None):
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
        seq = int(inputs["input_ids"].shape[1])
        st["pos"] = seq - 1 - args.offset; st["h"] = None
        with torch.no_grad():
            o = model(**inputs, use_cache=False)
        lg = o.logits[0, -1].float().cpu()
        pred = int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])
        return pred, st["h"]

    arms = ["orig", "down", "up", "control"]
    R_ = {a: {"pred": [], "gold": [], "projw": [], "projd": []} for a in arms}
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    n = 0; fails = 0
    rng = random.Random(args.sample_seed + 1)
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            m_ = re.search(r"did (\w+) spend in the (\w+)", q0)
            if not m_:
                continue
            C, R = m_.group(1), m_.group(2)
            evid = sorted(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            nonev = [t for t in range(len(frames)) if t not in set(evid)]
            if not evid or not nonev or gold < 1 or gold > 7:
                continue
            te = rng.choice(evid); tn = rng.choice(nonev)
            all_rooms = list(rooms_to_room2chars(states[te].get("rooms", {})).keys())
            other_rooms = [r for r in all_rooms if r != R]

            def mv_out(rooms):   # down: C leaves R in frame te
                rooms[R] = [c for c in rooms.get(R, []) if c != C]
                rooms[rng.choice(other_rooms)].append(C)
            def mv_in(rooms):    # up: C enters R in frame tn
                for r in rooms:
                    rooms[r] = [c for c in rooms[r] if c != C]
                rooms.setdefault(R, []).append(C)
            def mv_ctrl(rooms):  # control: some OTHER char shuffles between two non-R rooms in frame te
                oc = [c for r in other_rooms for c in rooms.get(r, []) if c != C]
                if not oc:
                    raise ValueError("no control char")
                c2 = rng.choice(oc)
                src = next(r for r in other_rooms if c2 in rooms.get(r, []))
                dst = rng.choice([r for r in other_rooms if r != src])
                rooms[src] = [c for c in rooms[src] if c != c2]
                rooms[dst].append(c2)

            variants = {"orig": (frames, gold)}
            fd = list(frames); fd[te] = rerender(states, te, mv_out); variants["down"] = (fd, gold - 1)
            fu = list(frames); fu[tn] = rerender(states, tn, mv_in); variants["up"] = (fu, gold + 1)
            fc = list(frames); fc[te] = rerender(states, te, mv_ctrl); variants["control"] = (fc, gold)

            for a in arms:
                fr, g_ = variants[a]
                pred, h = fwd(fr, q0)
                if h is None:
                    raise RuntimeError("no capture")
                R_[a]["pred"].append(pred); R_[a]["gold"].append(g_)
                R_[a]["projw"].append(float(h @ wstar)); R_[a]["projd"].append(float(h @ dhat))
            n += 1
            if n % 20 == 0:
                dw = np.array(R_["down"]["projw"]) - np.array(R_["orig"]["projw"])
                cw = np.array(R_["control"]["projw"]) - np.array(R_["orig"]["projw"])
                print(f"  {n}: dProj(down)={dw.mean():.3f}±{dw.std():.3f}  dProj(ctrl)={cw.mean():.3f}±{cw.std():.3f} "
                      f"(1-frame msg step={msg_step_w:.3f})", flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    lines = [f"=== PIXEL MINIMAL PAIRS (steps; L{L} off{args.offset}; n={n}; 1-frame msg step w*={msg_step_w:.3f}) ==="]
    po = np.array(R_["orig"]["pred"])
    for a in arms:
        p = np.array(R_[a]["pred"]); g = np.array(R_[a]["gold"])
        dw = np.array(R_[a]["projw"]) - np.array(R_["orig"]["projw"])
        moved = float(np.mean(p != po)) if a != "orig" else 0.0
        direc = float(np.mean(p < po)) if a == "down" else (float(np.mean(p > po)) if a == "up" else float("nan"))
        lines.append(f"  {a:>7} acc_vs_newgold={np.mean(p==g):.3f} MAE={np.mean(np.abs(p-g)):.2f} "
                     f"dProj_w*={dw.mean():+.3f}±{dw.std():.3f}  P(pred moved)={moved:.2f} P(moved in predicted dir)={direc:.2f}")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"res": R_, "msg_step_w": msg_step_w, "config": vars(args)}, out / "pixpairs.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
