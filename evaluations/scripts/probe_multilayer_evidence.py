#!/usr/bin/env python3
"""Does reading MULTIPLE layers beat the single best layer for per-frame evidence?

Single-layer probing plateaus at ~0.94 bal-acc / 0.98 AUC (L18-24). This concatenates several layers'
per-frame reps and re-probes is-evidence, to test whether multi-layer reads carry complementary signal
(would justify learned layer-mixing) or are redundant (layers in a residual stream are highly correlated
-> no gain -> ceiling is frozen perception, not the read).

Reports: best single layer, concat of a mid-late band, concat of all layers.
"""
from __future__ import annotations

import argparse
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--band", default="14,17,19,22,25", help="layers to concat for the mid-late band")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_multilayer_evidence")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    band = [int(x) for x in str(args.band).replace(",", " ").split()]
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))

    all_layer_feats: List[torch.Tensor] = []  # each [L+1, H]
    labels: List[int] = []; sample_idx: List[int] = []; sid = 0; n_layers = None
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        rng.shuffle(dirs); dirs = dirs[: args.max_samples]
        for d in dirs:
            states = rv.states_of(d / "qa.txt"); meta = json.loads((d / "metadata.json").read_text())
            C, R = meta.get("target_character"), meta.get("target_room")
            q = meta.get("question") or f"How many steps did {C} spend in the {R}?"
            if not states or not C or not R:
                continue
            try:
                reps = pi.per_frame_vision_reps(model, processor, pi.load_frames(d, states, meta), q, len(states), device)
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if reps is None:
                continue
            n_layers = reps.shape[0]
            for fi, st in enumerate(states):
                all_layer_feats.append(reps[:, fi, :].float().cpu())  # [L+1,H]
                labels.append(int(tf.room_of(st, C) == R)); sample_idx.append(sid)
            sid += 1
        emit(f"seq_len={sl}: frames={len(labels)}")

    y = torch.tensor(labels)
    F = torch.stack(all_layer_feats)  # [N, L+1, H]
    emit(f"frames={len(y)} evidence={y.float().mean():.2%} layers={n_layers}")
    uniq = sorted(set(sample_idx)); rng.shuffle(uniq); cut = int(0.7 * len(uniq)); train_s = set(uniq[:cut])
    tr = torch.tensor([i for i, s in enumerate(sample_idx) if s in train_s])
    te = torch.tensor([i for i, s in enumerate(sample_idx) if s not in train_s])

    def probe(X):
        return pr.fit_logreg(X[tr], y[tr], X[te], y[te])

    # best single layer
    best = (-1, 0.0, 0.0)
    for L in range(n_layers):
        b, a = probe(F[:, L, :])
        if a > best[2]:
            best = (L, b, a)
    emit(f"best single layer L{best[0]}: bal_acc={best[1]:.3f} auc={best[2]:.3f}")
    # mid-late band concat
    Xb = F[:, band, :].reshape(len(y), -1)
    bb, ba = probe(Xb)
    emit(f"concat band {band}: bal_acc={bb:.3f} auc={ba:.3f}")
    # all layers concat
    Xa = F.reshape(len(y), -1)
    ab, aa = probe(Xa)
    emit(f"concat ALL layers: bal_acc={ab:.3f} auc={aa:.3f}")
    emit("")
    gain = max(ba, aa) - best[2]
    emit(f"multi-layer AUC gain over best single = {gain:+.3f}")
    emit("=> multi-layer HELPS (complementary signal); learned layer-mix worth it." if gain > 0.01
         else "=> multi-layer REDUNDANT; ceiling is frozen perception, not the read. Single mid-late layer suffices.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
