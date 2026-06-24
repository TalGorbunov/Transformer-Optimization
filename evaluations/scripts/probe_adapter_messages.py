#!/usr/bin/env python3
"""Does the TRAINED adapter's per-frame message preserve the per-frame evidence?

The raw L19 per-frame rep decodes "is C in R" at ~0.94 (probe_evidence_selection_*). Here we apply the
trained adapter's phi to each frame's rep and re-probe the SAME label from phi(rep). If phi-messages
decode ~0.94 too, the adapter preserves extraction (so the count error is downstream / inherent
compounding -> a sharper pool won't help). If phi-messages decode << 0.94, phi/pooling is losing the
per-frame signal -> attention pooling / per-frame supervision is justified.

Reports balanced-acc + AUC for (a) raw L19 rep and (b) adapter phi(rep), on the same frames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from evaluations.scripts import probe_evidence_selection_image as pi
from evaluations.scripts import probe_evidence_selection_linear as pr
from evaluations.scripts.probe_pertask_extraction import fit_multiclass
import experiments.glstm.frame_axis_aggregator_adapter as fa


def parse_args():
    p = argparse.ArgumentParser(description="Probe whether the trained adapter's phi preserves per-frame evidence.")
    p.add_argument("--adapter", required=True, help="path to adapter_best.pt (deepsets, mean-pool)")
    p.add_argument("--task", default="steps_in_room", choices=["steps_in_room", "co_occupancy", "rooms_visited"],
                   help="per-frame label: is-evidence (steps, binary), same-room (co_occ, binary), room-of-C (rooms, 7-way)")
    p.add_argument("--aggregator", default="deepsets")
    p.add_argument("--d-mem", type=int, default=256)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_adapter_messages")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    hidden = int(model.config.text_config.hidden_size) if hasattr(model.config, "text_config") else int(model.config.hidden_size)
    adapter = fa.FrameAggregatorAdapter(hidden, args.d_mem, args.aggregator).to(device).float()
    adapter.load_state_dict(torch.load(args.adapter, map_location=device))
    adapter.eval()
    emit(f"loaded adapter {args.adapter}; hidden={hidden} read_layer={args.read_layer}")

    raw_feats: List[torch.Tensor] = []
    msg_feats: List[torch.Tensor] = []
    labels: List[int] = []
    sample_idx: List[int] = []
    rooms_vocab = None; absent = None
    sid = 0
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        rng.shuffle(dirs); dirs = dirs[: args.max_samples]
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            meta = json.loads((d / "metadata.json").read_text())
            if not states:
                continue
            if args.task == "rooms_visited":
                if rooms_vocab is None:
                    rooms_vocab = {r: i for i, r in enumerate(list(states[0]["rooms"].keys()))}; absent = len(rooms_vocab)
                C = meta.get("query_character") or Counter(
                    c for s in states for occ in s["rooms"].values() for c in occ).most_common(1)[0][0]
                q = f"How many distinct rooms did {C} visit across the {len(states)} frames?"
                def label_fn(st, C=C):
                    return rooms_vocab.get(tf.room_of(st, C), absent)
            elif args.task == "co_occupancy":
                pair = meta.get("query_pair")
                if not pair:
                    freq = Counter(c for s in states for occ in s["rooms"].values() for c in occ)
                    if len(freq) < 2:
                        continue
                    pair = [c for c, _ in freq.most_common(2)]
                C, D = pair
                q = f"In how many of the {len(states)} frames were {C} and {D} in the same room?"
                def label_fn(st, C=C, D=D):
                    rc, rd = tf.room_of(st, C), tf.room_of(st, D)
                    return int(rc == rd and rc != "not present")
            else:
                C, R = meta.get("target_character"), meta.get("target_room")
                q = meta.get("question") or f"How many steps did {C} spend in the {R}?"
                if not C or not R:
                    continue
                def label_fn(st, C=C, R=R):
                    return int(tf.room_of(st, C) == R)
            try:
                frames = pi.load_frames(d, states, meta)
                reps = pi.per_frame_vision_reps(model, processor, frames, q, len(states), device)  # [L+1,n,H]
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if reps is None:
                continue
            layer = reps[args.read_layer]  # [n, H]
            with torch.no_grad():
                msgs = adapter.encode(layer.to(device).float())  # phi(rep) -> [n, d]
            for fi, st in enumerate(states):
                raw_feats.append(layer[fi].float().cpu())
                msg_feats.append(msgs[fi].float().cpu())
                labels.append(label_fn(st))
                sample_idx.append(sid)
            sid += 1
        emit(f"seq_len={sl}: frames={len(labels)} samples={sid}")

    y = torch.tensor(labels)
    emit(f"frames={len(y)} evidence={int(y.sum())} ({y.float().mean():.2%})")
    uniq = sorted(set(sample_idx)); rng.shuffle(uniq)
    cut = int(0.7 * len(uniq)); train_s = set(uniq[:cut])
    tr = torch.tensor([i for i, s in enumerate(sample_idx) if s in train_s])
    te = torch.tensor([i for i, s in enumerate(sample_idx) if s not in train_s])

    Xraw = torch.stack(raw_feats); Xmsg = torch.stack(msg_feats)
    if args.task == "rooms_visited":
        nc = len(rooms_vocab) + 1
        acc_r = fit_multiclass(Xraw[tr], y[tr], Xraw[te], y[te], nc, device)
        acc_m = fit_multiclass(Xmsg[tr], y[tr], Xmsg[te], y[te], nc, device)
        emit("")
        emit(f"RAW L19 rep        : 7-way acc={acc_r:.3f}")
        emit(f"adapter phi(rep)   : 7-way acc={acc_m:.3f}")
        emit(f"delta (msg - raw)  : acc={acc_m-acc_r:+.3f}")
        preserves = acc_m >= acc_r - 0.02
        (run_dir / "result.txt").write_text(f"raw acc={acc_r:.4f}\nmsg acc={acc_m:.4f}\n", encoding="utf-8")
    else:
        bacc_r, auc_r = pr.fit_logreg(Xraw[tr], y[tr], Xraw[te], y[te])
        bacc_m, auc_m = pr.fit_logreg(Xmsg[tr], y[tr], Xmsg[te], y[te])
        emit("")
        emit(f"RAW L19 rep        : balanced_acc={bacc_r:.3f}  auc={auc_r:.3f}")
        emit(f"adapter phi(rep)   : balanced_acc={bacc_m:.3f}  auc={auc_m:.3f}")
        emit(f"delta (msg - raw)  : balanced_acc={bacc_m-bacc_r:+.3f}  auc={auc_m-auc_r:+.3f}")
        preserves = bacc_m >= bacc_r - 0.02
        (run_dir / "result.txt").write_text(
            f"raw auc={auc_r:.4f} bacc={bacc_r:.4f}\nmsg auc={auc_m:.4f} bacc={bacc_m:.4f}\n", encoding="utf-8")
    emit("")
    emit("=> phi PRESERVES per-frame evidence." if preserves else "=> phi LOSES per-frame evidence.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
