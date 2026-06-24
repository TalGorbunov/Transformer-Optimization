#!/usr/bin/env python3
"""Linear probe: is "evidence vs distractor" linearly decodable from the frozen per-frame
representation?  This isolates the SELECTION sub-problem from aggregation/readout.

Setup (text, query-conditioned): build a prompt with the QUESTION FIRST, then the frames as text
(so each frame token, in a causal LM, can attend back to the question). Run the frozen 7B, grab
hidden states at every layer, mean-pool each frame's token span -> one vector per frame per layer.
Label each frame is_evidence = (target character is in the target room that frame). Fit a per-layer
logistic regression (sample-disjoint train/test) and report balanced accuracy + AUC per layer.

Interpretation:
  high AUC at some layer  -> evidence/distractor IS linearly separable in the frozen rep; the
                            adapter's selection failure is architectural/training, not representational.
  AUC ~ chance everywhere -> the frozen rep does not carry separable evidence signal; "gating is
                            falsified" because there is nothing linear to gate on (retrieval deficit).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe for evidence-vs-distractor in frozen per-frame reps.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=120, help="cap per seq_len")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_evidence_selection_linear")
    return p.parse_args()


def build_probe_prompt(question: str, states: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
    """Question first, then frames. Returns (prompt, list of (char_start,char_end) per frame block)."""
    prompt = f"Question: {question}\nHere are the {len(states)} frames as text:\n"
    spans: List[Tuple[int, int]] = []
    for i, st in enumerate(states, start=1):
        block = f"Frame {i}:"
        for room, occ in st["rooms"].items():
            who = ", ".join(occ) if occ else "(empty)"
            block += f" {room}: {who};"
        block += "\n"
        start = len(prompt)
        prompt += block
        spans.append((start, len(prompt)))
    return prompt, spans


def char_spans_to_token_spans(tokenizer, prompt: str, char_spans: List[Tuple[int, int]]):
    """Map each frame's character span to a [tok_start, tok_end) range via cumulative-prefix lengths."""
    tok_spans = []
    prev_end_tok = len(tokenizer(prompt[: char_spans[0][0]], add_special_tokens=False).input_ids)
    for (_, c_end) in char_spans:
        end_tok = len(tokenizer(prompt[:c_end], add_special_tokens=False).input_ids)
        tok_spans.append((prev_end_tok, end_tok))
        prev_end_tok = end_tok
    return tok_spans


@torch.inference_mode()
def per_frame_reps(model, tokenizer, prompt: str, tok_spans, device: str):
    """Return tensor [num_layers, num_frames, hidden] of mean-pooled per-frame hidden states."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple (num_layers+1) of [1, T, H]
    T = ids.shape[1]
    layers = torch.stack([h[0] for h in hs], dim=0)  # [L+1, T, H]
    reps = []
    for (a, b) in tok_spans:
        a = max(0, min(a, T - 1)); b = max(a + 1, min(b, T))
        reps.append(layers[:, a:b, :].float().mean(dim=1))  # [L+1, H]
    return torch.stack(reps, dim=1)  # [L+1, num_frames, H]


def fit_logreg(Xtr, ytr, Xte, yte, epochs=300, lr=0.05, wd=1e-3):
    """Tiny torch logistic regression. Returns (balanced_acc, auc) on test."""
    d = Xtr.shape[1]
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    w = torch.zeros(d, 1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(1.0, pos)])  # balance classes
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = ytr.float().unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(Xtr @ w + b, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = (Xte @ w + b).squeeze(1)
        pred = (logits > 0).long()
        # balanced accuracy
        accs = []
        for c in (0, 1):
            m = (yte == c)
            if m.any():
                accs.append(float((pred[m] == c).float().mean()))
        bacc = sum(accs) / len(accs) if accs else 0.0
        # AUC via rank statistic
        auc = auc_score(logits, yte)
    return bacc, auc


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    rpos = ranks[labels == 1].sum()
    n_pos = len(pos); n_neg = len(neg)
    return float((rpos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> int:
    import random
    args = parse_args()
    rng = random.Random(int(args.seed))
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    device = base.resolve_device(args.device)
    dtype = base.resolve_dtype(args.dtype, device)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m: str) -> None:
        print(m, flush=True); log.write(m + "\n"); log.flush()

    emit(f"probe evidence-vs-distractor | model={args.model_name} 4bit={args.load_in_4bit} {device}/{dtype}")
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))
    tokenizer = processor.tokenizer

    # collect per-frame reps, labels, and a sample-id (for sample-disjoint split)
    feats: List[torch.Tensor] = []   # each [L+1, F, H]
    labels: List[int] = []
    sample_idx: List[int] = []
    n_layers = None
    sid = 0
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            emit(f"seq_len={sl}: missing {sr}, skip"); continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file()]
        rng.shuffle(dirs); dirs = dirs[: int(args.max_samples)]
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            if not states:
                continue
            meta_path = d / "metadata.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            C, R = meta.get("target_character"), meta.get("target_room")
            q = meta.get("question") or f"How many steps did {C} spend in the {R}?"
            if not C or not R:
                continue
            prompt, char_spans = build_probe_prompt(q, states)
            tok_spans = char_spans_to_token_spans(tokenizer, prompt, char_spans)
            try:
                reps = per_frame_reps(model, tokenizer, prompt, tok_spans, device)  # [L+1,F,H]
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            n_layers = reps.shape[0]
            for fi, st in enumerate(states):
                feats.append(reps[:, fi, :].cpu())
                labels.append(int(tf.room_of(st, C) == R))
                sample_idx.append(sid)
            sid += 1
        emit(f"seq_len={sl}: collected, total_frames={len(labels)} samples={sid}")

    y = torch.tensor(labels)
    s = torch.tensor(sample_idx)
    F = torch.stack(feats, dim=0)  # [N, L+1, H]
    emit(f"frames={len(y)} evidence={int(y.sum())} ({y.float().mean():.2%}) layers={n_layers}")

    # sample-disjoint split
    uniq = sorted(set(sample_idx))
    rng.shuffle(uniq)
    cut = int(0.7 * len(uniq))
    train_s = set(uniq[:cut])
    tr = torch.tensor([i for i, si in enumerate(sample_idx) if si in train_s])
    te = torch.tensor([i for i, si in enumerate(sample_idx) if si not in train_s])
    emit(f"train_frames={len(tr)} test_frames={len(te)}")

    rows = ["layer,balanced_acc,auc"]
    best = (-1, 0.0, 0.0)
    for L in range(n_layers):
        XL = F[:, L, :]
        bacc, auc = fit_logreg(XL[tr], y[tr], XL[te], y[te])
        rows.append(f"{L},{bacc:.4f},{auc:.4f}")
        if auc > best[2]:
            best = (L, bacc, auc)
        emit(f"layer {L:2d}: balanced_acc={bacc:.3f} auc={auc:.3f}")
    (run_dir / "probe_by_layer.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    base_rate = max(float(y.float().mean()), 1 - float(y.float().mean()))
    emit(f"\nBEST layer={best[0]} balanced_acc={best[1]:.3f} auc={best[2]:.3f} "
         f"(majority-class baseline acc={base_rate:.3f}, chance auc=0.5)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
