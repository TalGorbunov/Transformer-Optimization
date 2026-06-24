#!/usr/bin/env python3
"""Probe A: does any read over the frame's RAW tokens beat mean-pool+linear (~0.94) for is-evidence?

Tests whether POOLING or LINEARITY is the limit. Extracts each frame's raw L19 vision tokens (query-
conditioned) and probes is-evidence (C in R) with: mean+linear, max+linear, mean+MLP, attention-pool+MLP.
If the best caps at ~0.94 -> pooling/non-linearity is NOT the bottleneck (the info content is the limit).
"""
from __future__ import annotations
import argparse, json, random, sys, time
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
from evaluations.scripts import probe_evidence_selection_image as pi
from evaluations.scripts import probe_evidence_selection_linear as pr
from models.model import image_token_groups


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=70)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_token_extraction")
    return p.parse_args()


@torch.inference_mode()
def raw_tokens(model, processor, frames, question, read_layer, device):
    preamble = f"Question: {question}\nThe following are the {len(frames)} frames showing rooms in a house:"
    messages = [{"role": "user", "content": [{"type": "text", "text": preamble}] + [{"type": "image", "image": im} for im in frames]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = base.move_inputs_to_device(dict(inputs), device)
    spans = image_token_groups(inputs["input_ids"][0].detach().cpu(), len(frames), processor=processor)
    if len(spans) != len(frames):
        return None
    hs = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states[read_layer][0]  # [T,H]
    return [hs[torch.tensor(s, device=hs.device), :].float().cpu() for s in spans]  # list of [Tf,H]


class AttnMLP(nn.Module):
    def __init__(self, H, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.k = nn.Linear(H, d); self.v = nn.Linear(H, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, toks):  # [Tf,H]
        a = torch.softmax((self.k(toks) @ self.q) / (self.q.shape[0] ** 0.5), 0)  # [Tf]
        pooled = (a.unsqueeze(-1) * self.v(toks)).sum(0)
        return self.head(pooled).squeeze(-1)


def train_head(make, frames_tr, y_tr, frames_te, y_te, dev, epochs=40, lr=1e-3):
    head = make().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    pw = torch.tensor([(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        head.train(); opt.zero_grad(); loss = 0.0
        for i in range(len(frames_tr)):
            loss = loss + lossf(head(frames_tr[i].to(dev)).reshape(1), y_tr[i:i+1].float().to(dev))
        (loss / len(frames_tr)).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        logits = torch.stack([head(frames_te[i].to(dev)) for i in range(len(frames_te))]).cpu()
    pred = (logits > 0).long()
    accs = [float((pred[y_te == c] == c).float().mean()) for c in (0, 1) if (y_te == c).any()]
    return sum(accs) / len(accs), pr.auc_score(logits, y_te)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    dev = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, dev)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w")
    def emit(m): print(m, flush=True); log.write(m + "\n"); log.flush()
    model, processor = base.load_model_and_processor(args.model_name, dev, dtype, bool(args.load_in_4bit))

    toks_all: List[torch.Tensor] = []; labels: List[int] = []; sidx: List[int] = []; sid = 0
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
                tk = raw_tokens(model, processor, pi.load_frames(d, states, meta), q, args.read_layer, dev)
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if tk is None:
                continue
            for fi, st in enumerate(states):
                toks_all.append(tk[fi]); labels.append(int(tf.room_of(st, C) == R)); sidx.append(sid)
            sid += 1
        emit(f"seq_len={sl}: frames={len(labels)}")
    y = torch.tensor(labels)
    emit(f"frames={len(y)} evidence={y.float().mean():.2%} avg_tokens/frame={sum(t.shape[0] for t in toks_all)/len(toks_all):.0f}")
    uniq = sorted(set(sidx)); rng.shuffle(uniq); cut = int(0.7 * len(uniq)); trs = set(uniq[:cut])
    tr = [i for i, s in enumerate(sidx) if s in trs]; te = [i for i, s in enumerate(sidx) if s not in trs]
    yt = torch.tensor([labels[i] for i in tr]); ye = torch.tensor([labels[i] for i in te])

    mean = torch.stack([t.mean(0) for t in toks_all]); mx = torch.stack([t.max(0).values for t in toks_all])
    emit("")
    b, a = pr.fit_logreg(mean[tr], yt, mean[te], ye); emit(f"mean + linear : bal_acc={b:.3f} auc={a:.3f}  (baseline ~0.94)")
    b, a = pr.fit_logreg(mx[tr], yt, mx[te], ye);     emit(f"max  + linear : bal_acc={b:.3f} auc={a:.3f}")
    H = toks_all[0].shape[1]
    tr_t = [toks_all[i] for i in tr]; te_t = [toks_all[i] for i in te]
    b, a = train_head(lambda: AttnMLP(H, args.d), tr_t, yt, te_t, ye, dev); emit(f"attn-pool+MLP : bal_acc={b:.3f} auc={a:.3f}")
    emit("")
    emit("=> if best <= ~0.95: pooling/non-linearity is NOT the bottleneck (info content is the limit).")
    log.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
