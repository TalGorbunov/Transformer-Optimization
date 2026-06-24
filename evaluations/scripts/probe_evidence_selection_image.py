#!/usr/bin/env python3
"""Image-side counterpart of probe_evidence_selection_linear.py.

Same question: is "evidence vs distractor" (target character in target room, this frame) linearly
decodable from the frozen per-frame representation -- but now over the *vision tokens* of the
rendered image frames, i.e. the REAL multimodal pipeline, not the text proxy.

Question is placed FIRST (before the images) so each frame's vision tokens can attend back to it
(query-conditioned reps), matching the text probe. Per-frame vision-token spans come from
models.model.image_token_groups (contiguous <|image_pad|> runs). Mean-pool each frame's vision
tokens per layer -> one vector per frame; label is_evidence; per-layer logistic regression.

Compare best-layer AUC to the text probe (~0.997): if image-side AUC is much lower, the
distractor/selection gap in the real pipeline is vision-side per-frame encoding.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from evaluations.scripts import probe_evidence_selection_linear as pr  # reuse fit_logreg, auc_score
from models.model import image_token_groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe for evidence-vs-distractor in frozen per-frame VISION reps.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=100, help="cap per seq_len")
    p.add_argument("--image-sizes", default="0",
                   help="comma list of square resize targets (px) to sweep; 0=native 512. e.g. 224,336,448,560")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_evidence_selection_image")
    return p.parse_args()


def load_frames(d: Path, states: List[Dict[str, Any]], meta: Dict[str, Any], size: int = 0) -> List[Image.Image]:
    paths = meta.get("frame_paths") or meta.get("legacy_frame_paths") or [f"{i:03d}.png" for i in range(len(states))]
    frames: List[Image.Image] = []
    for i, p in enumerate(paths[: len(states)]):
        fp = d / p
        if not fp.is_file():
            fp = d / f"{i:03d}.png"
        im = Image.open(fp).convert("RGB")
        if size and size > 0:
            im = im.resize((size, size), Image.BICUBIC)  # resolution sweep: down/up-sample before tokenization
        frames.append(im)
    return frames


@torch.inference_mode()
def per_frame_vision_reps(model, processor, frames, question, n, device):
    preamble = f"Question: {question}\nHere are the {n} frames showing rooms in a house:"
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": preamble}]
                            + [{"type": "image", "image": im} for im in frames]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    inputs = base.move_inputs_to_device(dict(inputs), device)
    groups = image_token_groups(inputs["input_ids"][0].detach().cpu(), n, processor=processor)
    if len(groups) != n:
        return None
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    layers = torch.stack([h[0] for h in out.hidden_states], dim=0)  # [L+1, T, H]
    reps = []
    for g in groups:
        idx = torch.tensor(g, device=layers.device)
        reps.append(layers[:, idx, :].float().mean(dim=1))  # [L+1, H]
    return torch.stack(reps, dim=1)  # [L+1, n, H]


def fit_logreg_preds(Xtr, ytr, Xte, epochs=300, lr=0.05, wd=1e-3):
    """Same tiny logreg as pr.fit_logreg, but returns per-test logits (for crowding bucketing)."""
    d = Xtr.shape[1]
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    w = torch.zeros(d, 1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(1.0, pos)])
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = ytr.float().unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad(); lossf(Xtr @ w + b, yt).backward(); opt.step()
    with torch.no_grad():
        return (Xte @ w + b).squeeze(1)


def collect_at_size(model, processor, args, seq_lens, size, device, emit):
    """Collect per-frame reps/labels/crowding at a given resize. Re-seeds rng so the SAME dirs are used
    across sizes (fair comparison). crowd = number of distinct characters present in the frame."""
    rng = random.Random(int(args.seed))
    feats: List[torch.Tensor] = []; labels: List[int] = []; sample_idx: List[int] = []; crowd: List[int] = []
    n_layers = None; sid = 0
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            emit(f"  seq_len={sl}: missing {sr}, skip"); continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file()]
        rng.shuffle(dirs); dirs = dirs[: int(args.max_samples)]
        for d in dirs:
            states = rv.states_of(d / "qa.txt"); meta_path = d / "metadata.json"
            if not states or not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            C, R = meta.get("target_character"), meta.get("target_room")
            q = meta.get("question") or f"How many steps did {C} spend in the {R}?"
            if not C or not R:
                continue
            try:
                frames = load_frames(d, states, meta, size=size)
                reps = per_frame_vision_reps(model, processor, frames, q, len(states), device)
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if reps is None:
                continue
            n_layers = reps.shape[0]
            for fi, st in enumerate(states):
                feats.append(reps[:, fi, :].cpu())
                labels.append(int(tf.room_of(st, C) == R))
                crowd.append(len(rv.present_characters([st])))  # # distinct chars in this frame
                sample_idx.append(sid)
            sid += 1
    return feats, labels, sample_idx, crowd, n_layers


def main() -> int:
    args = parse_args()
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    sizes = [int(x) for x in str(args.image_sizes).replace(",", " ").split()]
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m: str) -> None:
        print(m, flush=True); log.write(m + "\n"); log.flush()

    emit(f"IMAGE probe evidence-vs-distractor | model={args.model_name} 4bit={args.load_in_4bit} {device}/{dtype}")
    emit(f"resolution sweep sizes(px, 0=native512)={sizes}  seq_lens={seq_lens}  max_samples={args.max_samples}")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))

    size_rows = ["image_size,best_layer,balanced_acc,auc"]
    crowd_rows = ["image_size,crowd,n_frames,is_evidence_acc"]
    for size in sizes:
        tag = f"{size}px" if size else "native512"
        emit(f"\n===== image_size={tag} =====")
        feats, labels, sample_idx, crowd, n_layers = collect_at_size(model, processor, args, seq_lens, size, device, emit)
        if not feats:
            emit("  no frames collected, skip"); continue
        y = torch.tensor(labels); F = torch.stack(feats, dim=0); cw = torch.tensor(crowd)
        rng = random.Random(int(args.seed))
        uniq = sorted(set(sample_idx)); rng.shuffle(uniq)
        cut = int(0.7 * len(uniq)); train_s = set(uniq[:cut])
        tr = torch.tensor([i for i, si in enumerate(sample_idx) if si in train_s])
        te = torch.tensor([i for i, si in enumerate(sample_idx) if si not in train_s])
        emit(f"  frames={len(y)} evidence={y.float().mean():.2%} layers={n_layers} train={len(tr)} test={len(te)}")
        best = (-1, 0.0, 0.0)
        for L in range(n_layers):
            XL = F[:, L, :]
            bacc, auc = pr.fit_logreg(XL[tr], y[tr], XL[te], y[te])
            emit(f"  L{L:2d}: balanced_acc={bacc:.4f} auc={auc:.4f}")
            if auc > best[2]:
                best = (L, bacc, auc)
        emit(f"  BEST layer={best[0]} balanced_acc={best[1]:.3f} auc={best[2]:.3f}")
        size_rows.append(f"{size},{best[0]},{best[1]:.4f},{best[2]:.4f}")
        # crowding bucketing at the best layer (Exp 3): per-test-frame correctness vs # chars in frame
        XB = F[:, best[0], :]
        logits = fit_logreg_preds(XB[tr], y[tr], XB[te])
        pred = (logits > 0).long(); yte = y[te]; cte = cw[te]
        emit(f"  crowding (best layer L{best[0]}): is-evidence acc by #chars-in-frame")
        for k in sorted(set(cte.tolist())):
            m = (cte == k)
            if m.any():
                acc = float((pred[m] == yte[m]).float().mean())
                emit(f"    crowd={k}: n={int(m.sum()):4d} acc={acc:.3f}")
                crowd_rows.append(f"{size},{k},{int(m.sum())},{acc:.4f}")
    (run_dir / "by_image_size.csv").write_text("\n".join(size_rows) + "\n", encoding="utf-8")
    (run_dir / "by_crowding.csv").write_text("\n".join(crowd_rows) + "\n", encoding="utf-8")
    emit("\nCompare best AUC to TEXT probe ~0.997. Exp1: AUC vs image_size (rises=resolution-bound). "
         "Exp3: acc vs crowd (drops=binding/superposition; flat=recognition).")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
