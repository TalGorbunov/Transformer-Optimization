#!/usr/bin/env python3
"""Causal #5 (E5b): attention-EDGE path patching. Cut specific attention edges with a 4D mask and
measure emitted accuracy. Route exclusivity for the frame->carrier->last chain.

Arms:
  base       no cuts
  cutevid    carrier row blind to EVIDENCE frames' image tokens, at --msg-layers
  cutrand    carrier row blind to an equal number of random NON-evidence frames (control)
  cutall     carrier row blind to ALL image tokens, at --msg-layers
  cutlate    LAST-token row blind to the carrier position, at --late-layers (transfer-window causality)

Masks are additive 4D, cast to the runtime hidden-state dtype inside the hook (fp16/bf16-safe).
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups

MIN = -65504.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--msg-layers", default="14,16,18,20")
    ap.add_argument("--late-layers", default="17,18,19,20")
    ap.add_argument("--offset", type=int, default=9)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/probes/attn_edge_patch")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    msg_L = [int(x) for x in args.msg_layers.replace(",", " ").split()]
    late_L = [int(x) for x in args.late_layers.replace(",", " ").split()]

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    tok = processor.tokenizer
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)
            if len(tok.encode(str(d), add_special_tokens=False)) == 1]
    cand_vals = [d for d, _ in cand]; cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    st = {"masks": {}}   # layer -> 4D float32 mask (cast in hook)

    def mk_pre(L):
        def pre(_m, hargs, hkwargs):
            mk = st["masks"].get(L)
            if mk is None:
                return hargs, hkwargs
            hs = hargs[0] if hargs else hkwargs.get("hidden_states")
            if hs is not None and mk.dtype != hs.dtype:
                mk = mk.to(hs.dtype); st["masks"][L] = mk
            if len(hargs) >= 2:
                hargs = (hargs[0], mk) + tuple(hargs[2:])
            else:
                hkwargs = dict(hkwargs); hkwargs["attention_mask"] = mk
            return hargs, hkwargs
        return pre
    for L in sorted(set(msg_L + late_L)):
        layers[L].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    def causal_base(seq, dev):
        m = torch.zeros(seq, seq, dtype=torch.float32)
        m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
        return m.to(dev)

    def fwd(inputs):
        with torch.no_grad():
            o = model(**inputs, use_cache=False)
        lg = o.logits[0, -1].float().cpu()
        return int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])

    arms = ["base", "cutevid", "cutrand", "cutall", "cutlate"]
    res = {a: {"pred": [], "gold": []} for a in arms}
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    n = 0; fails = 0
    rng = np.random.RandomState(args.sample_seed)
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            evid = sorted(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            nonev = [t for t in range(len(frames)) if t not in set(evid)]
            if not evid or len(nonev) < len(evid):
                continue
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0))
            ids = inputs["input_ids"][0].detach().cpu()
            fg = image_token_groups(ids, expected_num_frames=len(frames), processor=processor)
            seq = int(ids.shape[0]); dev = next(model.parameters()).device
            car = seq - 1 - args.offset; last = seq - 1
            randfr = list(rng.choice(nonev, size=len(evid), replace=False))
            def cut_frames(fr_list):
                m = causal_base(seq, dev)
                keys = torch.tensor(sorted(int(p) for f in fr_list for p in fg[f]), dtype=torch.long)
                m[car, keys] = MIN
                return {L: m.view(1, 1, seq, seq) for L in msg_L}
            plans = {
                "base": {},
                "cutevid": cut_frames(evid),
                "cutrand": cut_frames(randfr),
                "cutall": cut_frames(list(range(len(frames)))),
            }
            mlate = causal_base(seq, dev); mlate[last, car] = MIN
            plans["cutlate"] = {L: mlate.view(1, 1, seq, seq) for L in late_L}
            for a in arms:
                st["masks"] = plans[a]
                p = fwd(inputs)
                res[a]["pred"].append(p); res[a]["gold"].append(gold)
            st["masks"] = {}
            n += 1
            if n % 20 == 0:
                accs = {a: float(np.mean(np.array(res[a]["pred"]) == np.array(res[a]["gold"]))) for a in arms}
                print(f"  {n}: " + "  ".join(f"{k}={v:.3f}" for k, v in accs.items()), flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    lines = [f"=== ATTENTION-EDGE PATCH (carrier off{args.offset}; msg layers {msg_L}; late {late_L}; n={n}) ==="]
    for a in arms:
        p = np.array(res[a]["pred"]); g = np.array(res[a]["gold"])
        lines.append(f"  {a:>8} acc={np.mean(p==g):.3f}  MAE={np.mean(np.abs(p-g)):.2f}  mean_pred={p.mean():.2f}")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"res": res, "config": vars(args)}, out / "edges.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
