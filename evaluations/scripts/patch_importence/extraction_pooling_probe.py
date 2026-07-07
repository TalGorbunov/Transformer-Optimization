#!/usr/bin/env python3
"""Exp-1 Stage A: does mean-pool LOSE per-frame entity info that per-token reps keep, under crowding?

Single frame, query-conditioned (question first so frame tokens are query-aware). Capture L19 per-TOKEN
vision reps. Decode is-evidence (target char in target room this frame) three ways, bucketed by #chars:
  - mean-pool : mean over frame tokens (the current lossy step)
  - max-pool  : elementwise max (a cheap "keep the strongest token" baseline)
  - attn-pool : a tiny LEARNED attention readout over the per-token reps (the 'don't pool' proxy)

If attn/max >> mean as crowding grows -> mean-pool superposes entities; keeping per-token reps (slots)
is the task-agnostic extraction fix. If attn ~ mean -> pooling wasn't the bottleneck (kills the thesis).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
from typing import List
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf
from evaluations.scripts import probe_evidence_selection_image as pi
from evaluations.scripts import probe_evidence_selection_linear as pr
from models.model import image_token_groups


@torch.inference_mode()
def per_token_reps(model, processor, frames, question, n, layer, device):
    """Question-first; return list of [T_f, d] per-frame token reps at `layer`."""
    preamble = f"Question: {question}\nHere are the {n} frames showing rooms in a house:"
    messages = [{"role": "user", "content": [{"type": "text", "text": preamble}]
                 + [{"type": "image", "image": im} for im in frames]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    inputs = base.move_inputs_to_device(dict(inputs), device)
    groups = image_token_groups(inputs["input_ids"][0].detach().cpu(), n, processor=processor)
    if len(groups) != n:
        return None
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0]                                   # [T, d]
    return [h[torch.tensor(g, device=h.device), :].float().cpu() for g in groups]


class AttnPool(torch.nn.Module):
    """Tiny learned attention readout over a frame's tokens -> is-evidence logit."""
    def __init__(self, d):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(d) * 0.02)
        self.proj = torch.nn.Linear(d, d)
        self.head = torch.nn.Linear(d, 1)
    def forward(self, toks):                                          # toks: list of [T_i, d]
        outs = []
        for t in toks:
            a = torch.softmax((self.proj(t) @ self.q) / (t.shape[1] ** 0.5), dim=0)  # [T]
            outs.append(self.head(a @ t))                            # weighted sum -> logit
        return torch.cat(outs).squeeze(-1)


def fit_attn(tok_tr, y_tr, tok_te, d, epochs=200, lr=1e-3):
    m = AttnPool(d)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    pw = torch.tensor([(len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum())])
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad(); lossf(m(tok_tr), y_tr.float()).backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(m(tok_te))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--datasets", default="data/mmred_steps_1char:1,data/mmred_steps_2char:2,data/mmred_steps_balanced:5",
                    help="comma list of root:crowdlabel")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--split", default="all_uniform")
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/probes/extraction_pooling")
    args = ap.parse_args()

    device = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, device)
    run = Path(args.output) / time.strftime("%Y%m%d_%H%M%S"); run.mkdir(parents=True, exist_ok=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, bool(args.load_in_4bit))

    # collect per-frame tokens + is_evidence label, grouped by crowd label
    by_crowd = defaultdict(lambda: {"tok": [], "y": []})
    import random
    for spec in str(args.datasets).split(","):
        root, clab = spec.split(":"); clab = int(clab)
        sr = Path(root) / f"seq_len_{args.seq_len}" / args.split
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        random.Random(args.seed).shuffle(dirs); dirs = dirs[: args.max_samples]
        for d in dirs:
            states = rv.states_of(d / "qa.txt"); meta = json.loads((d / "metadata.json").read_text())
            C, R = meta.get("target_character"), meta.get("target_room")
            if not states or not C or not R:
                continue
            try:
                frames = pi.load_frames(d, states, meta)
                toks = per_token_reps(model, processor, frames, meta.get("question") or f"Is {C} in the {R}?",
                                      len(states), args.layer, device)
            except Exception as exc:
                print(f"skip {d.name}: {exc}", flush=True); continue
            if toks is None or len(toks) != len(states):
                continue
            for t, st in zip(toks, states):
                by_crowd[clab]["tok"].append(t.half())
                by_crowd[clab]["y"].append(int(tf.room_of(st, C) == R))
        print(f"crowd={clab}: {len(by_crowd[clab]['y'])} frames", flush=True)

    rows = ["crowd,n,pos_rate,mean_auc,max_auc,attn_auc"]
    rng = np.random.RandomState(args.seed)
    for clab in sorted(by_crowd):
        toks = by_crowd[clab]["tok"]; y = torch.tensor(by_crowd[clab]["y"])
        if len(set(y.tolist())) < 2 or len(y) < 20:
            continue
        idx = rng.permutation(len(y)); cut = int(0.65 * len(idx)); tr, te = idx[:cut], idx[cut:]
        meanX = torch.stack([t.float().mean(0) for t in toks]); maxX = torch.stack([t.float().amax(0) for t in toks])
        _, mean_auc = pr.fit_logreg(meanX[tr], y[tr], meanX[te], y[te])
        _, max_auc = pr.fit_logreg(maxX[tr], y[tr], maxX[te], y[te])
        tok_tr = [toks[i].float() for i in tr]; tok_te = [toks[i].float() for i in te]
        attn_p = fit_attn(tok_tr, y[tr], tok_te, meanX.shape[1])
        attn_auc = pr.auc_score(attn_p, y[te])
        rows.append(f"{clab},{len(y)},{y.float().mean():.3f},{mean_auc:.4f},{max_auc:.4f},{attn_auc:.4f}")
        print(f"crowd={clab}: mean_auc={mean_auc:.3f} max_auc={max_auc:.3f} attn_auc={attn_auc:.3f}", flush=True)

    (run / "by_crowd.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = ("=== EXTRACTION POOLING PROBE (is-evidence AUC by crowding) ===\n" + "\n".join(rows) +
           "\n\nattn/max >> mean as crowd grows => mean-pool superposes; keep per-token (slots). "
           "attn ~ mean => pooling is not the extraction bottleneck.\n")
    (run / "report.txt").write_text(rep, encoding="utf-8"); print("\n" + rep)
    print(f"Wrote {run}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
