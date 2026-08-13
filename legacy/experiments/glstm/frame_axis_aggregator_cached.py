#!/usr/bin/env python3
"""Cached frame-axis aggregator (fast). Two phases:

  Phase 1 (cache, GPU, one-time): for every (dir, task) across all splits, run ONE frozen 7B forward
  (question-first + frames), extract the per-frame VISION-token reps entering decoder layer L_READ
  (mean-pooled per frame), and persist them. Reusable across aggregators / hyperparameters.

  Phase 2 (train, fast, ~seconds/epoch): train an adapter on the cached reps -- phi -> aggregator ->
  count-head -- with cross-entropy on the count. No 7B in the loop. Per-epoch validation, best-epoch
  by val accuracy, final test on IID + OOD. Readout = the count head (directly tests whether the
  frame-axis aggregator recovers the count and generalizes, without saturation).

This drops the live LM-injection readout (that needs the per-step forward); it isolates the core
claim -- can an explicit extract->aggregate over clean L19 reps recover the count -- and runs ~10-100x
faster, enabling cheap iteration. The character choice for rooms/co-occ is fixed deterministically per
(dir, task) so reps are cacheable and metrics are stable.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from models.model import image_token_groups
import experiments.glstm.frame_axis_aggregator_adapter as fa  # reuse splits/adapter/data helpers

TASKS = fa.TASKS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cached frame-axis aggregator (count-head readout).")
    p.add_argument("--aggregator", choices=["seqmodel", "deepsets", "pna", "sum", "summax", "logic"], default="seqmodel")
    p.add_argument("--phi", choices=["linear", "codebook"], default="linear")
    p.add_argument("--balanced-loss", action="store_true", help="inverse-frequency count weighting in CE")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--train-seq-lens", default="1,2,3,4,5,6")
    p.add_argument("--ood-seq-lens", default="7,8")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--ood-per", type=int, default=100)
    p.add_argument("--cap-per-seq", type=int, default=None, help="cap dirs per seq_len before splitting")
    p.add_argument("--val-cap", type=int, default=150)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--d-mem", type=int, default=256)
    p.add_argument("--n-counts", type=int, default=13)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--single-frame", action=argparse.BooleanOptionalAction, default=False,
                   help="De-superpose: extract each frame's rep from a forward containing ONLY that frame "
                        "(isolates aggregation from cross-frame superposition). Uses a separate '_sf' cache.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "frame_axis" / "adapter_cached" / "run")
    return p.parse_args()


@torch.inference_mode()
def extract_reps(model, processor, frames, question, read_layer, device, single_frame=False) -> Optional[torch.Tensor]:
    if single_frame:
        # De-superposed: one forward per frame (only that frame in context) -> no cross-frame mixing.
        # Same question conditioning as the joint pass, so the ONLY difference is the superposition.
        reps = []
        for fr in frames:
            inputs = fa.build_inputs(processor, [fr], question, device)
            spans = image_token_groups(inputs["input_ids"][0].detach().cpu(), 1, processor=processor)
            if len(spans) != 1:
                return None
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[read_layer][0]
            reps.append(hs[torch.tensor(spans[0], device=hs.device), :].float().mean(0))
        return torch.stack(reps, dim=0).half().cpu()
    inputs = fa.build_inputs(processor, frames, question, device)
    ids = inputs["input_ids"]
    spans = image_token_groups(ids[0].detach().cpu(), len(frames), processor=processor)
    if len(spans) != len(frames):
        return None
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[read_layer][0]  # [T, H] residual at L_READ boundary
    reps = [hs[torch.tensor(s, device=hs.device), :].float().mean(0) for s in spans]
    return torch.stack(reps, dim=0).half().cpu()


def build_cache(args, splits, emit) -> Dict[str, dict]:
    sf_suffix = "_sf" if getattr(args, "single_frame", False) else ""
    cache_path = args.cache_path or (PROJECT_ROOT / "outputs" / "frame_axis" / "cache" / f"L{args.read_layer}{sf_suffix}.pt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, dict] = {}
    if cache_path.exists() and not args.rebuild_cache:
        cache = torch.load(cache_path)
        emit(f"loaded cache {cache_path} ({len(cache)} entries)")
    # unique dirs across all splits
    dirs: Dict[str, int] = {}
    for items in splits.values():
        for dstr, sl in items:
            dirs[dstr] = sl
    needed = [(dstr, task) for dstr in dirs for task in TASKS
              if f"{Path(dstr).name}|{task}" not in cache]
    if not needed:
        emit(f"cache complete ({len(cache)} entries) -- skipping GPU phase")
        return cache
    emit(f"building cache: {len(needed)} (dir,task) to extract (read_layer={args.read_layer})")
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    model.eval()
    shared_rng = None
    import random
    for i, (dstr, task) in enumerate(needed):
        key = f"{Path(dstr).name}|{task}"
        ex = fa.make_example(Path(dstr), task, random.Random(0), eval_mode=True)
        if ex is None:
            cache[key] = None; continue
        frames, question, gold, n = ex
        try:
            reps = extract_reps(model, processor, frames, question, args.read_layer, device,
                                single_frame=bool(getattr(args, "single_frame", False)))
        except Exception as exc:
            emit(f"  cache skip {key}: {exc}"); cache[key] = None; continue
        cache[key] = None if reps is None else {"reps": reps, "gold": int(gold), "seq_len": dirs[dstr]}
        if (i + 1) % 500 == 0:
            emit(f"  cached {i+1}/{len(needed)}"); torch.save(cache, cache_path)
    torch.save(cache, cache_path)
    emit(f"cache written: {cache_path} ({len(cache)} entries)")
    del model
    torch.cuda.empty_cache()
    return cache


def main() -> int:
    import random
    args = parse_args()
    if args.smoke:
        args.epochs, args.ood_per, args.val_cap = 5, 8, 30
    rng = random.Random(int(args.seed))
    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.aggregator}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m: str) -> None:
        print(m, flush=True); log.write(m + "\n"); log.flush()

    train_seq = [int(x) for x in args.train_seq_lens.replace(",", " ").split()]
    ood_seq = [int(x) for x in args.ood_seq_lens.replace(",", " ").split()]
    mtps = 6 if args.smoke else None
    splits = fa.declare_splits(args.data_root, args.split, train_seq, ood_seq, args.val_frac,
                               args.test_frac, args.ood_per, mtps, args.split_seed,
                               cap_per_seq=args.cap_per_seq)
    emit(f"aggregator={args.aggregator} read_layer={args.read_layer} epochs={args.epochs} "
         f"train={len(splits['train'])} val={len(splits['val'])} test_iid={len(splits['test_iid'])} "
         f"test_ood={len(splits['test_ood'])}")

    cache = build_cache(args, splits, emit)

    # ---- phase 2: fast training on cached reps (CPU/GPU, no 7B) ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = next(v["reps"].shape[1] for v in cache.values() if v)  # H from a cached entry
    adapter = fa.FrameAggregatorAdapter(hidden, args.d_mem, args.aggregator, n_counts=args.n_counts,
                                        phi_mode=args.phi).to(dev).float()
    emit(f"hidden={hidden} aggregator={args.aggregator} phi={args.phi} balanced={args.balanced_loss} "
         f"adapter_params={sum(p.numel() for p in adapter.parameters()):,} device={dev}")
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    def get(dstr, task):
        return cache.get(f"{Path(dstr).name}|{task}")

    # class-balanced CE weights from the train split's gold counts
    weight = None
    if args.balanced_loss:
        freq = torch.zeros(args.n_counts)
        for dstr, sl in splits["train"]:
            for t in TASKS:
                e = get(dstr, t)
                if e:
                    freq[min(args.n_counts - 1, e["gold"])] += 1
        weight = (1.0 / freq.clamp(min=1)); weight = (weight / weight.sum() * args.n_counts).to(dev)
        emit(f"balanced-loss weights (per count): {[round(float(w),2) for w in weight]}")
    ce = nn.CrossEntropyLoss(weight=weight)

    def predict(entry):
        reps = entry["reps"].to(dev).float()
        m = adapter.encode(reps)
        return adapter.aux_head(adapter.aggregate(m))  # [n_counts]

    def evaluate(items, cap=None, records=None, split_name=""):
        adapter.eval()
        agg: Dict[Tuple[str, str], List[float]] = {}
        grids = {t: {} for t in TASKS}
        use = items if cap is None else items[:cap]
        with torch.no_grad():
            for (dstr, sl) in use:
                for task in TASKS:
                    e = get(dstr, task)
                    if not e:
                        continue
                    gold = e["gold"]; pred = int(predict(e).argmax())
                    ok = int(pred == gold)
                    a = agg.setdefault((task, "count"), [0, 0, 0.0, 0.0])
                    a[0] += 1; a[1] += ok; a[2] += gold; a[3] += pred
                    c, t = grids[task].get((sl, gold), (0, 0)); grids[task][(sl, gold)] = (c + ok, t + 1)
                    if records is not None:
                        records.append((task, split_name, int(sl), int(gold), int(pred)))
        return agg, grids

    def acc_bias(a):
        n, cor, gs, ps = a
        return cor / max(1, n), (ps - gs) / max(1, n), gs / max(1, n), ps / max(1, n)

    val_rows = ["epoch,task,val_acc,val_bias,mean_gold,mean_pred"]
    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        adapter.train()
        order = list(splits["train"]); rng.shuffle(order)
        opt.zero_grad(); seen = 0; run = 0.0; step = 0
        for i, (dstr, sl) in enumerate(order):
            task = TASKS[(i + epoch) % len(TASKS)]
            e = get(dstr, task)
            if not e:
                continue
            logits = predict(e)
            loss = ce(logits.unsqueeze(0), torch.tensor([min(args.n_counts - 1, e["gold"])], device=dev))
            (loss / args.grad_accum).backward()
            run += float(loss.detach()); seen += 1; step += 1
            if step % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
        vagg, _ = evaluate(splits["val"], cap=args.val_cap)
        accs = []
        for task in TASKS:
            if (task, "count") in vagg:
                acc, bias, mg, mp = acc_bias(vagg[(task, "count")])
                val_rows.append(f"{epoch},{task},{acc:.4f},{bias:+.3f},{mg:.3f},{mp:.3f}")
                accs.append(acc)
        mean_val = sum(accs) / max(1, len(accs))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            emit(f"epoch {epoch}: train_loss={run/max(1,seen):.3f} mean_val_acc={mean_val:.3f}")
        if mean_val > best_val:
            best_val, best_epoch = mean_val, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in adapter.state_dict().items()})
    (run_dir / "val_by_epoch.csv").write_text("\n".join(val_rows) + "\n", encoding="utf-8")
    emit(f"best_epoch={best_epoch} mean_val_acc={best_val:.3f}")

    if best_state is not None:
        adapter.load_state_dict(best_state)
    torch.save(best_state or adapter.state_dict(), run_dir / "adapter_best.pt")
    summary = ["split,task,n,accuracy,mean_gold,mean_pred,bias"]
    records: List[Tuple[str, str, int, int, int]] = []  # (task, split, seq_len, gold, pred)
    for sname in ("test_iid", "test_ood"):
        if not splits[sname]:
            continue
        agg, grids = evaluate(splits[sname], records=records, split_name=sname)
        for task in TASKS:
            if (task, "count") in agg:
                acc, bias, mg, mp = acc_bias(agg[(task, "count")])
                row = f"{sname},{task},{agg[(task,'count')][0]},{acc:.4f},{mg:.3f},{mp:.3f},{bias:+.3f}"
                summary.append(row); emit(row)
            if sname == "test_ood":
                tf.save_heatmap(grids[task], f"{task} (OOD, count-head)", run_dir / f"heatmap_{task}_ood.png")
    (run_dir / "summary.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    pred_rows = ["task,split,seq_len,gold,pred"] + [f"{t},{s},{sl},{g},{p}" for (t, s, sl, g, p) in records]
    (run_dir / "predictions.csv").write_text("\n".join(pred_rows) + "\n", encoding="utf-8")
    ceil_iid = extraction_ceilings(splits["test_iid"])  # {(task,count): mean extraction-bound max acc}
    for sp in ("test_iid", "test_ood"):
        rec_sp = [r for r in records if r[1] == sp]
        if rec_sp:
            make_diagnostic_plots(rec_sp, args.n_counts, run_dir, suffix=f"_{sp.replace('test_', '')}",
                                  ceilings=ceil_iid if sp == "test_iid" else None)
    emit(f"wrote {run_dir}/ (summary.csv, predictions.csv, val_by_epoch.csv, heatmaps, plots, adapter_best.pt)")
    log.close()
    return 0


# measured per-frame extraction accuracy (probes): steps/co-occ is-evidence ~0.94, rooms room-id 7-way 0.915.
# New Cat-1 tasks default to a placeholder; the live run MEASURES their per-frame p at L19 and overrides
# these via the p_extract argument (see measure_extraction_p / extraction_ceilings).
P_EXTRACT = {"steps_in_room": 0.94, "co_occupancy": 0.94, "rooms_visited": 0.915,
             "room_busy": 0.92, "char_accompanied": 0.92, "char_alone": 0.92}


def _p_for(task, p_extract):
    """Measured per-frame extraction p (if provided for this task) else the hardcoded default."""
    if p_extract and p_extract.get(task) is not None:
        return float(p_extract[task])
    return P_EXTRACT.get(task, 0.92)


def _sample_ceiling(task, dstr, states, sims=200, p_extract=None):
    """Monte-Carlo 'best acc if aggregation were perfect' for one sample: corrupt the true per-frame
    quantities with the measured extraction noise, apply the task's PERFECT aggregation. Returns (gold, ceil)."""
    name = Path(dstr).name
    rsel = random.Random(zlib.crc32((name + task).encode()))  # mirrors eval_mode char choice
    noise = random.Random(777)
    p = _p_for(task, p_extract)
    if task in tf.NEW_CAT1:  # answer = sum_t 1[per-frame predicate]; symmetric per-frame flip with prob 1-p
        res = tf.cat1_task(task, states, rsel)
        if res is None:
            return None
        ev = res[0]; true = sum(ev)
        cor = sum(int(sum((e if noise.random() < p else 1 - e) for e in ev) == true) for _ in range(sims))
        return true, cor / sims
    rooms = list(states[0]["rooms"].keys()); classes = rooms + ["not present"]
    if task == "steps_in_room":
        meta = json.loads((Path(dstr) / "metadata.json").read_text())
        C, R = meta.get("target_character"), meta.get("target_room")
        if not C or not R:
            return None
        ev = [int(tf.room_of(s, C) == R) for s in states]; true = sum(ev)
        cor = sum(int(sum((e if noise.random() < p else 1 - e) for e in ev) == true) for _ in range(sims))
        return true, cor / sims
    chars = rv.present_characters(states)
    if task == "rooms_visited":
        if not chars:
            return None
        C = rsel.choice(chars)
        seq = [tf.room_of(s, C) for s in states]
        true = len({r for r in seq if r != "not present"}); cor = 0
        for _ in range(sims):
            noisy = [(r if noise.random() < p else noise.choice([c for c in classes if c != r])) for r in seq]
            cor += int(len({r for r in noisy if r != "not present"}) == true)
        return true, cor / sims
    if len(chars) < 2:  # co_occupancy
        return None
    C, D = rsel.sample(chars, 2)
    ev = [int(tf.room_of(s, C) == tf.room_of(s, D) and tf.room_of(s, C) != "not present") for s in states]
    true = sum(ev)
    cor = sum(int(sum((e if noise.random() < p else 1 - e) for e in ev) == true) for _ in range(sims))
    return true, cor / sims


def extraction_ceilings(items, tasks=None, p_extract=None):
    """{(task, gold_count): mean extraction-bound ceiling} over the given dirs.
    tasks: restrict to a subset (default all TASKS). p_extract: {task: measured per-frame p} override."""
    from collections import defaultdict
    tasks = tasks or TASKS
    acc = defaultdict(list)
    for (dstr, sl) in items:
        states = rv.states_of(Path(dstr) / "qa.txt")
        if not states:
            continue
        for task in tasks:
            r = _sample_ceiling(task, dstr, states, p_extract=p_extract)
            if r is not None:
                acc[(task, r[0])].append(r[1])
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def _fit_binary_balacc(X, y, sample_idx, dev, seed=0, epochs=300, lr=0.05, wd=1e-3):
    """Logistic probe with a sample-disjoint 70/30 split; returns balanced accuracy (the per-frame
    extraction p) or None if a class is missing."""
    uniq = sorted(set(sample_idx)); random.Random(seed).shuffle(uniq)
    tr = set(uniq[: int(0.7 * len(uniq))])
    trm = torch.tensor([s in tr for s in sample_idx]); tem = ~trm
    Xtr, ytr, Xte, yte = X[trm].float(), y[trm], X[tem].float(), y[tem]
    if len(yte) == 0 or len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return None
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(dev); Xte = ((Xte - mu) / sd).to(dev)
    W = torch.zeros(Xtr.shape[1], 2, requires_grad=True, device=dev)
    b = torch.zeros(2, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd); lossf = nn.CrossEntropyLoss()
    yt = ytr.to(dev)
    for _ in range(epochs):
        opt.zero_grad(); lossf(Xtr @ W + b, yt).backward(); opt.step()
    pred = (Xte @ W + b).argmax(1).cpu()
    accs = [float((pred[yte == c] == c).float().mean()) for c in (0, 1) if int((yte == c).sum()) > 0]
    return sum(accs) / len(accs) if accs else None


def measure_extraction_p(model, processor, items, tasks, read_layer, device, cap=150, emit=print):
    """Measure per-frame extraction accuracy (balanced) at `read_layer` for each task via a linear
    probe over query-conditioned per-frame L_READ reps -> the per-frame binary predicate. Returns
    {task: p}. NOTE: caller must remove any residual hook first (probe needs the CLEAN reps)."""
    from collections import defaultdict
    import experiments.glstm.frame_axis_aggregator_adapter as fa
    feats = defaultdict(list); labs = defaultdict(list); sidx = defaultdict(list)
    for si, (dstr, sl) in enumerate(items[:cap]):
        d = Path(dstr)
        states = rv.states_of(d / "qa.txt")
        if not states:
            continue
        meta_path = d / "metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            frames = fa.pi.load_frames(d, states, meta)
        except Exception:
            continue
        if len(frames) != len(states):
            continue
        for task in tasks:
            rsel = random.Random(zlib.crc32((d.name + task).encode()))  # mirror eval/ceiling choice
            if task in tf.NEW_CAT1:
                res = tf.cat1_task(task, states, rsel)
                if res is None:
                    continue
                ev, question, _ = res
            else:
                qg = tf.question_and_gold(task, d, states, random.Random(zlib.crc32((d.name + task).encode())))
                if qg is None:
                    continue
                question = qg[0]
                ev = _per_frame_labels(task, d, states)
                if ev is None:
                    continue
            try:
                reps = fa.pi.per_frame_vision_reps(model, processor, frames, question, len(states), device)
            except Exception:
                continue
            if reps is None or read_layer >= reps.shape[0]:
                continue
            r = reps[read_layer]  # [n, H]
            for fi in range(len(states)):
                feats[task].append(r[fi].cpu().float()); labs[task].append(int(ev[fi])); sidx[task].append(si)
    out = {}
    for task in tasks:
        if len(labs[task]) >= 40:
            p = _fit_binary_balacc(torch.stack(feats[task]), torch.tensor(labs[task]), sidx[task], device)
            if p is not None:
                out[task] = round(p, 4)
                emit(f"  extraction p[{task}] @L{read_layer} = {p:.4f}  "
                     f"(frames={len(labs[task])}, pos={sum(labs[task])})")
    return out


def _per_frame_labels(task, d, states):
    """Per-frame binary evidence for the EXISTING binary Cat-1 tasks (for the extraction probe)."""
    if task == "steps_in_room":
        meta = json.loads((Path(d) / "metadata.json").read_text())
        C, R = meta.get("target_character"), meta.get("target_room")
        if not C or not R:
            return None
        return [int(tf.room_of(s, C) == R) for s in states]
    if task == "co_occupancy":
        chars = rv.present_characters(states)
        if len(chars) < 2:
            return None
        rsel = random.Random(zlib.crc32((Path(d).name + task).encode()))
        C, D = rsel.sample(chars, 2)
        return [int(tf.room_of(s, C) == tf.room_of(s, D) and tf.room_of(s, C) != "not present") for s in states]
    return None  # rooms_visited is multiclass (room-id), not handled here


def make_diagnostic_plots(records, n_counts, run_dir, suffix="", ceilings=None):
    """Three clean overall figures on the given records:
      acc_per_count{suffix}.png        - accuracy vs gold count, one line per task (+ dashed extraction ceiling)
      mean_pred_per_count{suffix}.png  - mean predicted vs gold count, one line per task, + y=x ideal
      confusion{suffix}.png            - one clean row-normalized confusion matrix per task (1x3)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    COLORS = {"steps_in_room": "tab:blue", "rooms_visited": "tab:orange", "co_occupancy": "tab:green",
              "room_busy": "tab:red", "char_accompanied": "tab:purple", "char_alone": "tab:brown"}
    all_tasks = list(dict.fromkeys(list(TASKS) + sorted({tk for (tk, *_rest) in records})))
    per_task = {t: [(g, p) for (tk, s, sl, g, p) in records if tk == t] for t in all_tasks}
    per_task = {t: r for t, r in per_task.items() if r}

    # --- 1. accuracy per count (solid = actual; dashed = extraction-bound ceiling) ---
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for t, rec in per_task.items():
        cs = sorted({g for g, _ in rec})
        acc = [np.mean([int(p == c) for g, p in rec if g == c]) for c in cs]
        ax.plot(cs, acc, "o-", color=COLORS.get(t, "tab:gray"), label=t)
        if ceilings is not None:
            cc = [c for c in cs if (t, c) in ceilings]
            if cc:
                ax.plot(cc, [ceilings[(t, c)] for c in cc], "--", color=COLORS.get(t, "tab:gray"), alpha=0.6)
    if ceilings is not None:
        ax.plot([], [], "k--", alpha=0.6, label="extraction-bound max (perfect agg)")
    ax.set_xlabel("gold count"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy per gold count (solid=actual, dashed=extraction ceiling)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(run_dir / f"acc_per_count{suffix}.png", bbox_inches="tight"); plt.close(fig)

    # --- 2. mean pred per count ---
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    hi = 0
    for t, rec in per_task.items():
        cs = sorted({g for g, _ in rec})
        mp = [np.mean([p for g, p in rec if g == c]) for c in cs]
        ax.plot(cs, mp, "o-", color=COLORS.get(t, "tab:gray"), label=t)
        hi = max(hi, max(cs))
    ax.plot([0, hi], [0, hi], "k--", alpha=0.6, label="ideal (y=x)")
    ax.set_xlabel("gold count"); ax.set_ylabel("mean predicted count")
    ax.set_title("Mean prediction per gold count"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(run_dir / f"mean_pred_per_count{suffix}.png", bbox_inches="tight"); plt.close(fig)

    # --- 3. confusion matrices (one per task) ---
    tasks = list(per_task)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.6), dpi=150)
    if len(tasks) == 1:
        axes = [axes]
    for ax, t in zip(axes, tasks):
        rec = per_task[t]
        K = max(max(g for g, _ in rec), max(p for _, p in rec)) + 1
        conf = np.zeros((K, K))
        for g, p in rec:
            conf[g, p] += 1
        conf_n = conf / np.clip(conf.sum(1, keepdims=True), 1, None)
        im = ax.imshow(conf_n, vmin=0, vmax=1, cmap="Blues", origin="lower")
        for g in range(K):
            for p in range(K):
                if conf[g, p] > 0:
                    ax.text(p, g, f"{conf_n[g, p]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if conf_n[g, p] > 0.5 else "black")
        ax.set_title(t); ax.set_xlabel("predicted"); ax.set_ylabel("gold")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
    fig.suptitle("Confusion (row-normalized)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(run_dir / f"confusion{suffix}.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
