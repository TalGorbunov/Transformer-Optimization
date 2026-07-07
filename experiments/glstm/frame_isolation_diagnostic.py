#!/usr/bin/env python3
"""Frame-isolation-mask diagnostic: can ONE forward with a block-diagonal frame mask recover MULTIPASS
per-frame quality? Question-first layout (frames see the query). We monkeypatch sdpa so that, in the
masked forward, each frame's vision tokens attend only to themselves + the preamble/question (NOT to
other frames). Then we compare per-frame is-evidence extraction + sigmoid-then-sum across:
  joint    (normal attention, baseline ~0.73)
  masked   (block-diagonal frame isolation, 1 forward)   <- the test
  multipass(N single-frame forwards, ~0.93)              <- the anchor (optional, --multipass)
If masked ~ multipass, cross-frame attention was the contaminant and the mask cures it in one pass.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import experiments.glstm.frame_axis_aggregator_adapter as fa
from experiments.glstm.cache_minimal_frame_reps import frame_labels
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import image_token_groups, get_layers

ORIG_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]
STATE = {"active": False, "iso": None}  # iso: [S,S] additive (0 / -inf off-block), set per example


def custom_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                     is_causal=None, **kwargs):
    li = getattr(module, "layer_idx", None)
    if li is None or not STATE["active"] or STATE["iso"] is None:
        return ORIG_SDPA(module, query, key, value, attention_mask, scaling=scaling,
                         dropout=dropout, is_causal=is_causal, **kwargs)
    q_len, k_len = query.shape[2], key.shape[2]
    iso = STATE["iso"]
    if q_len != k_len or q_len != iso.shape[0]:  # decode steps / mismatch -> faithful default
        return ORIG_SDPA(module, query, key, value, attention_mask, scaling=scaling,
                         dropout=dropout, is_causal=is_causal, **kwargs)
    dev = query.device
    causal = torch.triu(torch.full((q_len, k_len), float("-inf"), device=dev, dtype=torch.float32), 1)
    mask = (causal + iso.to(dev))[None, None]  # [1,1,S,S]
    if attention_mask is not None:
        mask = mask + attention_mask[..., :k_len].to(torch.float32)
    return ORIG_SDPA(module, query, key, value, mask.to(query.dtype), scaling=scaling, dropout=dropout,
                     is_causal=False, **kwargs)


def build_iso(seq_len, spans):
    """[S,S] additive: -inf where a frame-token query attends a DIFFERENT frame's token; else 0. Vectorized."""
    fid = torch.full((seq_len,), -1, dtype=torch.long)
    for fi, sp in enumerate(spans):
        fid[torch.tensor(sp, dtype=torch.long)] = fi
    is_frame = fid >= 0
    both_frame = is_frame.view(-1, 1) & is_frame.view(1, -1)
    diff_frame = fid.view(-1, 1) != fid.view(1, -1)
    iso = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    iso[both_frame & diff_frame] = float("-inf")
    return iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="steps_in_room")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    ap.add_argument("--split", default="all_uniform")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--read-layer", type=int, default=19)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--multipass", action="store_true", help="also run N single-frame forwards (the anchor)")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "probes" / "frame_isolation")
    args = ap.parse_args()
    device = base.resolve_device("cuda"); dtype = base.resolve_dtype("bfloat16", device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading model ...", flush=True)
    model, processor = base.load_model_and_processor("Qwen/Qwen2.5-VL-7B-Instruct", device, dtype, True)
    for p in model.parameters():
        p.requires_grad_(False)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = custom_attention  # monkeypatch (vision delegates via layer_idx None)
    target = get_layers(model)[args.read_layer]

    cap = {"spans": None, "cur": -1, "reps": None}
    def edit(hs):
        if cap["spans"] is None or hs.shape[1] <= cap["cur"]:
            return hs
        cap["reps"] = torch.stack([hs[0, idx, :].float().mean(0).cpu() for idx in cap["spans"]], 0)
        return hs
    def pre_hook(m, a, k):
        if len(a) >= 1:
            return (edit(a[0]),) + tuple(a[1:]), k
        k = dict(k); k["hidden_states"] = edit(k["hidden_states"]); return a, k
    target.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def forward_reps(inputs, spans, iso=None):
        cap["spans"] = spans; cap["cur"] = int(inputs["input_ids"].shape[1]) - 1; cap["reps"] = None
        STATE["active"] = iso is not None; STATE["iso"] = iso
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        STATE["active"] = False; STATE["iso"] = None
        return cap["reps"]

    splits = fa.declare_splits(args.data_root, args.split, [args.seq_len], [], 0.0, 0.0, 0, None, 12345)
    dirs = [Path(d) for d, _ in splits["train"]][:args.limit]
    import random
    rng = random.Random(0)
    data = {"joint": [], "masked": [], "multipass": [], "labels": [], "gold": []}
    t0 = time.time()
    for ix, d in enumerate(dirs):
        ex = fa.make_example(d, args.task, rng, eval_mode=True)
        if ex is None:
            continue
        frames, question, gold, nf, states = ex
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8")) if (d / "metadata.json").exists() else {}
        fl = frame_labels(args.task, states, meta)
        if fl is None or len(fl) != len(frames):
            continue
        inputs = fa.build_inputs(processor, frames, question, device)
        spans = image_token_groups(inputs["input_ids"][0].detach().cpu(), len(frames), processor=processor)
        if len(spans) != len(frames):
            continue
        S = int(inputs["input_ids"].shape[1])
        rj = forward_reps(inputs, spans, iso=None)
        rm = forward_reps(inputs, spans, iso=build_iso(S, spans))
        if rj is None or rm is None:
            continue
        data["joint"].append(rj.numpy()); data["masked"].append(rm.numpy())
        data["labels"].append([int(x) for x in fl]); data["gold"].append(int(gold))
        if args.multipass:
            mp = []
            ok = True
            for fi in range(len(frames)):
                si = fa.build_inputs(processor, [frames[fi]], question, device)
                sp = image_token_groups(si["input_ids"][0].detach().cpu(), 1, processor=processor)
                if len(sp) != 1:
                    ok = False; break
                r = forward_reps(si, sp, iso=None)
                mp.append(r[0].numpy())
            data["multipass"].append(np.stack(mp) if ok else None)
        if (ix + 1) % 40 == 0:
            print(f"  {ix+1}/{len(dirs)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- CPU eval: per-frame is-evidence + sigmoid-then-sum ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    from sklearn.model_selection import train_test_split

    def evaluate(reps_list):
        reps = [r for r in reps_list if r is not None]
        if len(reps) < 30:
            return None
        N = reps[0].shape[0]
        labels = [data["labels"][i] for i, r in enumerate(reps_list) if r is not None]
        gold = [data["gold"][i] for i, r in enumerate(reps_list) if r is not None]
        idx = np.arange(len(reps)); tr, te = train_test_split(idx, test_size=0.35, random_state=0)
        Xtr = np.stack([reps[i][f] for i in tr for f in range(N)])
        ytr = np.asarray([labels[i][f] for i in tr for f in range(N)])
        sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
        Xte = np.stack([reps[i][f] for i in te for f in range(N)])
        yte = np.asarray([labels[i][f] for i in te for f in range(N)])
        bacc = balanced_accuracy_score(yte, clf.predict(sc.transform(Xte)))
        cnt = [int(round(clf.predict_proba(sc.transform(reps[i]))[:, 1].sum())) for i in te]
        cacc = accuracy_score([gold[i] for i in te], cnt)
        # SHARP gate (kappa sweep) on standardized per-frame score: c = round(Sum sigmoid(kappa*z))
        sctr = np.concatenate([clf.decision_function(sc.transform(reps[i])) for i in tr])
        smean, sstd = float(sctr.mean()), float(sctr.std() + 1e-9)
        sharp = {}
        for kap in (5.0, 20.0):
            pred = []
            for i in te:
                z = (clf.decision_function(sc.transform(reps[i])) - smean) / sstd
                pred.append(int(round((1.0 / (1.0 + np.exp(-kap * z))).sum())))
            sharp[f"sharp_gate_k{int(kap)}"] = float(accuracy_score([gold[i] for i in te], pred))
        return {"perframe_bacc": float(bacc), "sigmoid_then_sum_acc": float(cacc), "n": len(reps), **sharp}

    import pickle
    with open(run_dir / "reps.pkl", "wb") as fh:
        pickle.dump({"joint": data["joint"], "masked": data["masked"], "multipass": data["multipass"],
                     "labels": data["labels"], "gold": data["gold"]}, fh)
    out = {k: evaluate(data[k]) for k in ("joint", "masked", "multipass")}
    lines = [f"n_examples={len(data['joint'])}",
             f"{'cond':<10} {'per-frame bacc':>15} {'soft-sum':>10} {'sharp k5':>10} {'sharp k20':>10}"]
    for k in ("joint", "masked", "multipass"):
        r = out[k]
        if r:
            lines.append(f"{k:<10} {r['perframe_bacc']:>15.3f} {r['sigmoid_then_sum_acc']:>10.3f} "
                         f"{r.get('sharp_gate_k5', float('nan')):>10.3f} {r.get('sharp_gate_k20', float('nan')):>10.3f}")
        else:
            lines.append(f"{k:<10}  (skipped)")
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    (run_dir / "report.txt").write_text(report + "\n")
    (run_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
