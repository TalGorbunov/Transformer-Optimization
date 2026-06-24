#!/usr/bin/env python3
"""Probe B: is the ~0.94 is-evidence ceiling the VISION ENCODER (perception) or the LM's query-conditioned
BINDING?

For each frame, get mean-pooled L19 frame reps TWICE: (q) with the question in context [query-conditioned],
and (noq) with a neutral prompt [query-independent perception]. Then a binding-aware probe
MLP(frame_rep, char_emb[X], room_emb[Y]) -> "is X in Y", trained on each rep type.

- query-independent (noq) >= query-conditioned (q) ~ high (e.g. ~0.98): the info IS in the vision tokens;
  the LM binding given the question is the limit -> NOT the vision encoder.
- noq ~ 0.94 too: the per-frame who's-where info is not cleanly in the vision tokens -> vision encoder.
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
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_perception_binding")
    return p.parse_args()


@torch.inference_mode()
def frame_reps(model, processor, frames, preamble, read_layer, device):
    messages = [{"role": "user", "content": [{"type": "text", "text": preamble}] + [{"type": "image", "image": im} for im in frames]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = base.move_inputs_to_device(dict(inputs), device)
    spans = image_token_groups(inputs["input_ids"][0].detach().cpu(), len(frames), processor=processor)
    if len(spans) != len(frames):
        return None
    hs = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states[read_layer][0]
    return torch.stack([hs[torch.tensor(s, device=hs.device), :].float().mean(0) for s in spans]).cpu()  # [n,H]


class Binding(nn.Module):
    def __init__(self, H, nch, nrm, emb):
        super().__init__()
        self.ce = nn.Embedding(nch, emb); self.re = nn.Embedding(nrm, emb)
        self.mlp = nn.Sequential(nn.Linear(H + 2 * emb, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, rep, ci, ri):
        return self.mlp(torch.cat([rep, self.ce(ci), self.re(ri)], -1)).squeeze(-1)


def run_probe(reps, triples, y, tr, te, H, nch, nrm, emb, dev, epochs=60, lr=1e-3):
    fidx = torch.tensor([t[0] for t in triples]); ci = torch.tensor([t[1] for t in triples]); ri = torch.tensor([t[2] for t in triples])
    R = reps.to(dev)
    net = Binding(H, nch, nrm, emb).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    pw = torch.tensor([(len(y[tr]) - y[tr].sum()) / max(1, y[tr].sum())], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = y.float().to(dev)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        out = net(R[fidx[tr]], ci[tr].to(dev), ri[tr].to(dev))
        lossf(out, yt[tr]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(R[fidx[te]], ci[te].to(dev), ri[te].to(dev)).cpu()
    pred = (logits > 0).long(); ye = y[te]
    accs = [float((pred[ye == c] == c).float().mean()) for c in (0, 1) if (ye == c).any()]
    return sum(accs) / len(accs), pr.auc_score(logits, ye)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    dev = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, dev)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w")
    def emit(m): print(m, flush=True); log.write(m + "\n"); log.flush()
    model, processor = base.load_model_and_processor(args.model_name, dev, dtype, bool(args.load_in_4bit))

    rooms_vocab = None; char_vocab = {}
    reps_q: List[torch.Tensor] = []; reps_n: List[torch.Tensor] = []
    triples = []; ys = []; sidx = []; fbase = 0; sid = 0
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
            if not states:
                continue
            rooms = list(states[0]["rooms"].keys())
            if rooms_vocab is None:
                rooms_vocab = {r: i for i, r in enumerate(rooms)}
            frames_pre_q = f"Question: {q}\nThe following are the {len(states)} frames showing rooms in a house:"
            frames_pre_n = f"The following are {len(states)} frames showing rooms in a house:"
            try:
                imgs = pi.load_frames(d, states, meta)
                rq = frame_reps(model, processor, imgs, frames_pre_q, args.read_layer, dev)
                rn = frame_reps(model, processor, imgs, frames_pre_n, args.read_layer, dev)
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if rq is None or rn is None:
                continue
            for fi, st in enumerate(states):
                reps_q.append(rq[fi]); reps_n.append(rn[fi])
                fidx = fbase + fi
                for X in rv.present_characters([st]):
                    if X not in char_vocab:
                        char_vocab[X] = len(char_vocab)
                    truer = tf.room_of(st, X)
                    if truer not in rooms_vocab:
                        continue
                    # positive (X in true room), negative (X in a random other room)
                    triples.append((fidx, char_vocab[X], rooms_vocab[truer])); ys.append(1); sidx.append(sid)
                    wrong = rng.choice([r for r in rooms if r != truer])
                    triples.append((fidx, char_vocab[X], rooms_vocab[wrong])); ys.append(0); sidx.append(sid)
            fbase += len(states); sid += 1
        emit(f"seq_len={sl}: frames={len(reps_q)} triples={len(triples)} chars={len(char_vocab)}")

    y = torch.tensor(ys)
    Rq = torch.stack(reps_q); Rn = torch.stack(reps_n); H = Rq.shape[1]
    emit(f"triples={len(y)} pos={y.float().mean():.2%} chars={len(char_vocab)} rooms={len(rooms_vocab)}")
    uniq = sorted(set(sidx)); rng.shuffle(uniq); cut = int(0.7 * len(uniq)); trs = set(uniq[:cut])
    tr = torch.tensor([i for i, s in enumerate(sidx) if s in trs]); te = torch.tensor([i for i, s in enumerate(sidx) if s not in trs])
    nch, nrm = len(char_vocab), len(rooms_vocab)
    bq, aq = run_probe(Rq, triples, y, tr, te, H, nch, nrm, args.emb, dev)
    bn, an = run_probe(Rn, triples, y, tr, te, H, nch, nrm, args.emb, dev)
    emit("")
    emit(f"query-CONDITIONED (question in context) : bal_acc={bq:.3f} auc={aq:.3f}")
    emit(f"query-INDEPENDENT (neutral prompt)      : bal_acc={bn:.3f} auc={an:.3f}  <- pure perception (probe binds)")
    emit("")
    if bn >= 0.97:
        emit("=> perception is HIGH -> who's-where IS in the vision tokens; the ~0.94 evidence ceiling is LM query-conditioned BINDING, not the vision encoder.")
    elif bn <= 0.95:
        emit("=> perception ~ evidence ceiling -> the per-frame who's-where info is genuinely limited in the vision tokens -> VISION ENCODER.")
    else:
        emit("=> intermediate; inspect.")
    log.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
