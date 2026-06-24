#!/usr/bin/env python3
"""Text per-frame extraction: sweep POOLING x LAYER to find the fair text recipe.

The image-tuned recipe (read frame tokens, MEAN-pool, ~L19) superposes the multi-room text block, so
rooms/co-occ text extraction looked low (0.829 / 0.958) while steps (binary, survives pooling) was 0.997.
This sweeps poolings over each frame's TEXT token span, per layer, to test whether a text-appropriate
pooling recovers the per-frame fact:
  mean    : mean over the frame block          (current adapter default; superposes all rooms)
  last    : the LAST token of the frame block   (carrier hypothesis -- the segment summary)
  max     : max over the frame block
  target  : the token where the queried entity is named (direct localization; mildly task-aware)
Tasks: rooms_visited (7-way room-of-C) and co_occupancy (same-room binary, AUC). The queried entity is
read from metadata (query_character / query_pair) when present, else most-present (legacy data).
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import Counter
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from evaluations.scripts import probe_evidence_selection_linear as pr

POOLINGS = ["mean", "last", "max", "target"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--task", choices=["rooms_visited", "co_occupancy"], required=True)
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="8")
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_text_pooling_sweep")
    return p.parse_args()


def fit_multiclass(Xtr, ytr, Xte, yte, nc, dev, epochs=300, lr=0.05, wd=1e-3):
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(dev); Xte = ((Xte - mu) / sd).to(dev)
    W = torch.zeros(Xtr.shape[1], nc, requires_grad=True, device=dev)
    b = torch.zeros(nc, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd); lossf = nn.CrossEntropyLoss()
    yt = ytr.to(dev)
    for _ in range(epochs):
        opt.zero_grad(); lossf(Xtr @ W + b, yt).backward(); opt.step()
    return float(((Xte @ W + b).argmax(1).cpu() == yte).float().mean())


@torch.inference_mode()
def frame_reps_all_poolings(model, tokenizer, prompt, frame_tok_spans, target_tok, device):
    """Return {pooling: [L+1, n, H]} for the text prompt."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(device); mask = enc.attention_mask.to(device)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    layers = torch.stack([h[0] for h in out.hidden_states], dim=0).float()  # [L+1, T, H]
    T = layers.shape[1]
    pooled = {k: [] for k in POOLINGS}
    for fi, (a, b) in enumerate(frame_tok_spans):
        a = max(0, min(a, T - 1)); b = max(a + 1, min(b, T))
        seg = layers[:, a:b, :]
        pooled["mean"].append(seg.mean(1))
        pooled["last"].append(layers[:, b - 1, :])
        pooled["max"].append(seg.max(1).values)
        t = target_tok[fi]
        t = a if (t is None or t < a or t >= b) else t
        pooled["target"].append(layers[:, t, :])
    return {k: torch.stack(v, dim=1).half().cpu() for k, v in pooled.items()}  # [L+1, n, H]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    dev = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, dev)
    run_dir = args.output / f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"; run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w")
    def emit(m): print(m, flush=True); log.write(m + "\n"); log.flush()
    emit(f"TEXT pooling x layer sweep | task={args.task} data={args.data_root} model={args.model_name}")
    model, processor = base.load_model_and_processor(args.model_name, dev, dtype, bool(args.load_in_4bit))
    tok = processor.tokenizer

    rooms_vocab = None; absent = None
    feats = {k: [] for k in POOLINGS}; ys: List[int] = []; sids: List[int] = []; sid = 0
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            emit(f"seq_len={sl}: missing {sr}"); continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        rng.shuffle(dirs); dirs = dirs[: args.max_samples]
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            if not states:
                continue
            meta = json.loads((d / "metadata.json").read_text())
            rooms = list(states[0]["rooms"].keys())
            if rooms_vocab is None:
                rooms_vocab = {r: i for i, r in enumerate(rooms)}; absent = len(rooms_vocab)
            if args.task == "rooms_visited":
                C = meta.get("query_character")
                if not C:
                    freq = Counter(c for st in states for occ in st["rooms"].values() for c in occ)
                    if not freq:
                        continue
                    C = freq.most_common(1)[0][0]
                q = f"How many distinct rooms did {C} visit across the {len(states)} frames?"
                targets = [C]
            else:
                pair = meta.get("query_pair")
                if not pair:
                    freq = Counter(c for st in states for occ in st["rooms"].values() for c in occ)
                    if len(freq) < 2:
                        continue
                    pair = [c for c, _ in freq.most_common(2)]
                C, D = pair
                q = f"In how many of the {len(states)} frames were {C} and {D} in the same room?"
                targets = [C, D]
            prompt, char_spans = pr.build_probe_prompt(q, states)
            frame_tok_spans = pr.char_spans_to_token_spans(tok, prompt, char_spans)
            # target token = first occurrence of the (first) queried name inside each frame block
            target_tok = []
            for (ca, cb) in char_spans:
                pos = prompt.find(targets[0], ca, cb)
                target_tok.append(len(tok(prompt[:pos], add_special_tokens=False).input_ids) if pos >= 0 else None)
            try:
                reps = frame_reps_all_poolings(model, tok, prompt, frame_tok_spans, target_tok, dev)
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            for fi, st in enumerate(states):
                if args.task == "rooms_visited":
                    lab = rooms_vocab.get(tf.room_of(st, C), absent)
                else:
                    rc, rd = tf.room_of(st, C), tf.room_of(st, D)
                    lab = int(rc == rd and rc != "not present")
                for k in POOLINGS:
                    feats[k].append(reps[k][:, fi, :])
                ys.append(lab); sids.append(sid)
            sid += 1
        emit(f"seq_len={sl}: frames={len(ys)} samples={sid}")

    y = torch.tensor(ys)
    uniq = sorted(set(sids)); rng.shuffle(uniq); cut = int(0.7 * len(uniq)); trs = set(uniq[:cut])
    tr = torch.tensor([i for i, s in enumerate(sids) if s in trs])
    te = torch.tensor([i for i, s in enumerate(sids) if s not in trs])
    nL = feats["mean"][0].shape[0]
    nclass = (len(rooms_vocab) + 1) if args.task == "rooms_visited" else 2
    base_rate = max(float((y == c).float().mean()) for c in range(nclass) if (y == c).any())
    metric = "acc(7-way)" if args.task == "rooms_visited" else "auc(same-room)"
    emit(f"\nframes={len(y)} classes={nclass} majority/base={base_rate:.3f}  metric={metric}")
    rows = ["pooling,best_layer,score"]
    summary = {}
    for k in POOLINGS:
        F = torch.stack(feats[k], dim=0)  # [N, L+1, H]
        best = (-1, 0.0)
        for L in range(nL):
            XL = F[:, L, :].float()
            if args.task == "rooms_visited":
                s = fit_multiclass(XL[tr], y[tr], XL[te], y[te], nclass, dev)
            else:
                _, s = pr.fit_logreg(XL[tr], y[tr], XL[te], y[te])
            if s > best[1]:
                best = (L, s)
        summary[k] = best
        rows.append(f"{k},{best[0]},{best[1]:.4f}")
        emit(f"  pooling={k:7s} BEST L{best[0]:2d}  {metric}={best[1]:.3f}")
    (run_dir / "pooling_sweep.csv").write_text("\n".join(rows) + "\n")
    win = max(summary.items(), key=lambda kv: kv[1][1])
    emit(f"\nWINNER pooling={win[0]} L{win[1][0]} {metric}={win[1][1]:.3f}  (mean baseline={summary['mean'][1]:.3f})")
    log.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
