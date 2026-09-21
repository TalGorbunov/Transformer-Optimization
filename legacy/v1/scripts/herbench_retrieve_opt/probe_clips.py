#!/usr/bin/env python3
"""Clip-unit supply probe for the HERBench retrieve-opt sweep (Phase 1 deliverable).

Units = clips (or single frames for the A0 anchor). Per queried-pair question we build a
Q-first + block-fence + posreset sequence over that question's units (one question-replica
per UNIT, not per frame), run ONE forward, and recompute the per-unit message into the
replica carrier at the read layer(s) — exactly probe_supply.py's machinery, but with
UNIT-level spans (vis-per-unit = union of the unit's frame image tokens; one fence block +
one replica per unit). Big questions are chunked to <=K units/forward (the fence makes
units independent, so chunking is exact).

Per-unit binary gold: occurrence (label 1) vs not (0). Read the message at a small sweep of
carrier offsets from each replica's end and report the best-d' offset (mirrors the archived
per-token SNR sweep). Metrics: whitened LDA d' (QUESTION-grouped splits, so a question's own
pos/neg never straddle train/test) + AUROC + logistic gate acc; pooled AND per-verb (n>=20).

Reuses gnnformer as a library (no core edit): build_replica_probe_mask, reset_positions,
recompute_messages, find_question_spans, locate_word_token, image_token_groups, runtime.

Anchor (GATE): A0 (single-frame units @448, --input-mode armB) must land in the archived
band d' ~ 0.98-1.10 / AUROC ~ 0.80 (docs/archive/RESULTS_pre_fencing.md 2026-07-07d/e) or
the pipeline is suspect. The fence gain was NULL on real HERBench video, so the fenced
layout should reproduce the joint per-frame number.

Usage (A0 anchor):
  python scripts/herbench_retrieve_opt/probe_clips.py --input-mode armB \
    --data-root data/herbench_ac/armB_ev_fill16 --layers 12,14,16 --limit 134 \
    --output outputs/herbench_retrieve_opt/probe/A0
Usage (clip arm):
  python scripts/herbench_retrieve_opt/probe_clips.py --input-mode clips \
    --data-root data/herbench_retrieve_opt_clips/d0.5_r448 --layers 16 \
    --output outputs/herbench_retrieve_opt/probe/B_d0.5
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.fencing import (
    build_replica_probe_mask,
    recompute_messages,
    reset_positions,
)
from gnnformer.runtime import (
    attention_dims,
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)


# --------------------------------------------------------------------------- units
def question_text(pair: str, n_frames: int) -> str:
    unit = "clip" if n_frames > 1 else "frame"
    return (f"Look at this {unit} from an egocentric kitchen video. "
            f"Is the person performing the action '{pair}'? Answer yes or no.")


def load_units_clips(data_root: Path):
    """clips mode: each clip-dir = one unit. Group units by question_id. Frame images are
    stored as PATHS and opened lazily in the loop (opening thousands of JPEGs upfront is slow
    and defeats --limit)."""
    by_q = defaultdict(list)
    for d in sorted(data_root.iterdir()):
        mp = d / "meta.json"
        if not d.is_dir() or not mp.exists():
            continue
        m = json.loads(mp.read_text())
        paths = [d / f"frame_{i:02d}.jpg" for i in range(int(m["n_frames"]))]
        by_q[m["question_id"]].append({
            "frame_paths": paths, "label": int(m["label"]), "verb": m["verb"],
            "pair": m["pair"], "qid": m["question_id"], "clip_id": m["clip_id"]})
    return by_q


def load_units_armB(data_root: Path):
    """armB mode (A0 anchor): each armB frame = one single-frame unit; label = is_evidence.
    Frame paths only (opened lazily in the loop)."""
    by_q = defaultdict(list)
    for d in sorted(data_root.iterdir()):
        mp = d / "meta.json"
        if not d.is_dir() or not mp.exists():
            continue
        m = json.loads(mp.read_text())
        pair = m["pair"]
        verb = pair.split()[0]
        for fr in m["frames"]:
            i = fr["idx"]
            by_q[m["question_id"]].append({
                "frame_paths": [d / f"frame_{i:02d}.jpg"], "label": int(fr["is_evidence"]),
                "verb": verb, "pair": pair, "qid": m["question_id"],
                "clip_id": f"{m['question_id']}_f{i}"})
    return by_q


# --------------------------------------------------------------------------- d'
def dprime_grouped(X, y, groups, seeds=(0, 1, 2)):
    """Whitened shrinkage-LDA d' + AUROC-derived d' + logistic gate acc, with GROUP-disjoint
    60/40 splits (a group = one question). X [U,H], y [U], groups [U]. Mirrors
    gnnformer.metrics.dprime_pair but splits on `groups` instead of per-sample rows."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import norm
    X = np.asarray(X, np.float64); y = np.asarray(y, int); groups = np.asarray(groups)
    uniq = np.unique(groups)
    dws, das, gates = [], [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(uniq))
        tr_g = set(uniq[perm[: int(0.6 * len(uniq))]].tolist())
        tr = np.array([i for i in range(len(y)) if groups[i] in tr_g])
        te = np.array([i for i in range(len(y)) if groups[i] not in tr_g])
        if (len(tr) <= 3 or len(te) < 2
                or len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2):
            continue
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X[tr], y[tr])
        w = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)
        p = X[te] @ w
        pE, pN = p[y[te] == 1], p[y[te] == 0]
        dws.append(abs(pE.mean() - pN.mean()) / (0.5 * (pE.std() + pN.std()) + 1e-12))
        try:
            auc = min(max(roc_auc_score(y[te], p), 1e-4), 1 - 1e-4)
            das.append(np.sqrt(2) * norm.ppf(auc))
        except ValueError:
            pass
        lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        gates.append(lr.score(X[te], y[te]))
    if not dws:
        return dict(dprime=float("nan"), dprime_std=float("nan"), auc_dprime=float("nan"),
                    gate_acc=float("nan"), n=len(y), npos=int((y == 1).sum()))
    return dict(dprime=float(np.mean(dws)), dprime_std=float(np.std(dws)),
                auc_dprime=float(np.mean(das)) if das else float("nan"),
                gate_acc=float(np.mean(gates)), n=len(y), npos=int((y == 1).sum()))


# --------------------------------------------------------------------------- probe
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--input-mode", choices=("clips", "armB"), required=True)
    ap.add_argument("--layers", default="16")
    ap.add_argument("--resize", type=int, default=0, help="0 = keep extracted res")
    ap.add_argument("--max-units", type=int, default=10, help="units per forward (chunk big Qs)")
    ap.add_argument("--carrier-offsets", default="0,1,2,3",
                    help="sweep these offsets-from-replica-end; report best-d' offset")
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (0 = all)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    _t_start = time.time()
    rt = load_runtime(args.model) if args.model else load_runtime()
    print(f"[timing] load_runtime {time.time()-_t_start:.1f}s", flush=True)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.split(",")]
    offsets = [int(x) for x in args.carrier_offsets.split(",")]
    dims = attention_dims(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    ve_id = int(model.config.vision_end_token_id)
    w_o = {L: dequantize_linear_weight(layers[L].self_attn.o_proj) for L in Ls}

    from gnnformer.fencing import FenceHooks
    hooks = FenceHooks(layers, capture_layers=Ls).install()

    by_q = (load_units_clips if args.input_mode == "clips" else load_units_armB)(Path(args.data_root))
    qids = sorted(by_q)
    if args.limit:
        qids = qids[: args.limit]

    # message accumulators: per (layer, offset) a list of [H] vectors; parallel label/verb/group
    msgs = {L: {o: [] for o in offsets} for L in Ls}
    labels, verbs, groups, clip_ids = [], [], [], []
    n_units = n_skip_units = n_forward = 0

    for qi, qid in enumerate(qids):
        units_all = by_q[qid]
        pair = units_all[0]["pair"]
        obj_word = pair.split()[-1]
        # chunk this question's units into forwards of <= max_units
        for c0 in range(0, len(units_all), args.max_units):
            units = units_all[c0: c0 + args.max_units]
            NU = len(units)
            q0 = question_text(pair, max(len(u["frame_paths"]) for u in units))
            frames_flat = []
            for u in units:
                frs = [Image.open(p).convert("RGB") for p in u["frame_paths"]]
                if args.resize > 0:
                    frs = [f.resize((args.resize, args.resize)) for f in frs]
                frames_flat.append(frs)
            nfr_per_unit = [len(f) for f in frames_flat]

            # Q-first shared prefix, then per unit: [unit frames][q0 replica]. No trailing
            # final q0 (we read per-unit only) — the last unit's q0 IS its replica.
            content = [{"type": "text", "text": q0}]
            for frs in frames_flat:
                for f in frs:
                    content.append({"type": "image", "image": f})
                content.append({"type": "text", "text": q0})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = move_to_device(inputs, model.device)
            ids = inputs["input_ids"][0].tolist()
            seq = len(ids)
            total_frames = sum(nfr_per_unit)
            fg = image_token_groups(inputs["input_ids"][0].cpu(),
                                    expected_num_frames=total_frames, processor=processor)
            if len(fg) != total_frames:
                n_skip_units += NU; continue
            # group per-frame image tokens into per-UNIT visibility
            vis_by_unit, gi = [], 0
            for k in range(NU):
                toks = []
                for _ in range(nfr_per_unit[k]):
                    toks.extend(int(p) for p in fg[gi]); gi += 1
                vis_by_unit.append(torch.tensor(sorted(toks), dtype=torch.long))

            # STRUCTURAL span location from vision tokens (robust; text tokenization is
            # context-dependent). Frames within a unit are consecutive; a q0 replica sits
            # between units. Layout: [q0 prefix][imgs u0][q0][imgs u1][q0]...[imgs uN-1][q0].
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            vends = [p for p, t in enumerate(ids) if t == ve_id]
            if len(vstarts) != total_frames or len(vends) != total_frames:
                n_skip_units += NU; continue
            if NU < 2:
                n_skip_units += NU; continue        # need >=2 units to derive q0 length
            unit_first_vstart, unit_last_vend, fi = [], [], 0
            for k in range(NU):
                unit_first_vstart.append(vstarts[fi]); fi += nfr_per_unit[k]
                unit_last_vend.append(vends[fi - 1])
            qlen = unit_first_vstart[1] - (unit_last_vend[0] + 1)   # q0 in-context token length
            if qlen <= 0:
                n_skip_units += NU; continue
            fin_start = unit_last_vend[NU - 1] + 1 + qlen           # start of gen prompt
            boundaries = [unit_first_vstart[k + 1] for k in range(NU - 1)] + [fin_start]
            rep_spans = [(unit_last_vend[k] + 1, boundaries[k]) for k in range(NU)]
            unit_blocks = [(unit_first_vstart[k], boundaries[k]) for k in range(NU)]

            _sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
            _t0 = time.time()
            m = build_replica_probe_mask(seq, rep_spans, vis_by_unit,
                                         fence_frames=True, fence_blocks=True, blocks=unit_blocks)
            hooks.set_mask(m, model.device)
            with torch.inference_mode():
                base_pos, _ = rope_fn(inputs["input_ids"],
                                      image_grid_thw=inputs.get("image_grid_thw"),
                                      attention_mask=inputs.get("attention_mask"))
            pos_ids = reset_positions(base_pos, unit_blocks, fin_start)
            _sync(); _t1 = time.time()
            with torch.inference_mode():
                model(**inputs, position_ids=pos_ids)
            hooks.clear_mask()
            _sync(); _t2 = time.time()
            n_forward += 1

            # carrier positions per unit for each offset (offset 0 = last token of replica span).
            # Batch ALL offsets into one recompute_messages call per layer so the rotary q/k
            # are recomputed once per layer (not once per offset).
            all_carr, all_vis = [], []
            for o in offsets:
                for k, sp in enumerate(rep_spans):
                    c = max(min(sp[1] - 1 - o, sp[1] - 1), sp[0])
                    all_carr.append(c); all_vis.append(vis_by_unit[k])
            for L in Ls:
                mm = recompute_messages(
                    seq=seq, cos=hooks.cos, sin=hooks.sin, dims=dims, w_o=w_o[L],
                    q_proj=hooks.qkv[L]["q_proj"], k_proj=hooks.qkv[L]["k_proj"],
                    v_proj=hooks.qkv[L]["v_proj"], mask_full=m, vis_by_frame=all_vis,
                    carrier_positions=all_carr)  # [len(offsets)*NU, H]
                for i, o in enumerate(offsets):
                    msgs[L][o].append(mm[i * NU:(i + 1) * NU])
            _sync()
            if n_forward <= 6:
                print(f"    [timing] fwd#{n_forward} NU={NU} seq={seq}: "
                      f"mask+pos {_t1-_t0:.2f}s  forward {_t2-_t1:.2f}s  "
                      f"recompute {time.time()-_t2:.2f}s", flush=True)
            for u in units:
                labels.append(u["label"]); verbs.append(u["verb"]); groups.append(qid)
                clip_ids.append(u["clip_id"])
            n_units += NU
        if (qi + 1) % 20 == 0:
            print(f"  {qi+1}/{len(qids)} questions, {n_units} units, {n_forward} forwards "
                  f"(skip {n_skip_units})", flush=True)

    hooks.remove()
    y = np.array(labels); vb = np.array(verbs); grp = np.array(groups)
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    report = {"data_root": args.data_root, "input_mode": args.input_mode,
              "n_units": int(n_units), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
              "n_forwards": n_forward, "n_skip_units": n_skip_units,
              "layers": Ls, "offsets": offsets, "by_layer": {}}
    lines = [f"=== CLIP-UNIT SUPPLY PROBE ({args.input_mode}, n_units={n_units}, "
             f"pos={int((y==1).sum())} neg={int((y==0).sum())}, data={args.data_root}) ==="]
    for L in Ls:
        X_off = {o: np.concatenate(msgs[L][o], axis=0) for o in offsets}
        # pooled d' per offset -> pick best
        best = None
        per_off = {}
        for o in offsets:
            r = dprime_grouped(X_off[o], y, grp)
            per_off[o] = r
            if best is None or (r["dprime"] == r["dprime"] and r["dprime"] > per_off[best]["dprime"]):
                best = o
        rb = per_off[best]
        report["by_layer"][L] = {"best_offset": best, "pooled": rb,
                                 "per_offset": {o: per_off[o] for o in offsets}, "per_verb": {}}
        lines.append(f"L{L}: pooled d'={rb['dprime']:.3f}±{rb['dprime_std']:.3f} "
                     f"(auc-d' {rb['auc_dprime']:.3f}, gate {rb['gate_acc']:.3f}) "
                     f"@best-offset -{best}   [n={rb['n']}, pos={rb['npos']}]")
        # per-verb at the best offset
        Xb = X_off[best]
        for verb in sorted(set(vb.tolist())):
            msk = vb == verb
            if int((y[msk] == 1).sum()) < 20 or int((y[msk] == 0).sum()) < 20:
                continue
            rv = dprime_grouped(Xb[msk], y[msk], grp[msk])
            report["by_layer"][L]["per_verb"][verb] = rv
            lines.append(f"    verb={verb:<10s} d'={rv['dprime']:.3f} (gate {rv['gate_acc']:.3f}) "
                         f"n={rv['n']} pos={rv['npos']}")

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    # save raw messages+labels for offline re-analysis
    np.savez(out / "messages.npz",
             **{f"L{L}_o{o}": np.concatenate(msgs[L][o], axis=0) for L in Ls for o in offsets},
             y=y, verbs=vb, groups=grp, clip_ids=np.array(clip_ids))
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
