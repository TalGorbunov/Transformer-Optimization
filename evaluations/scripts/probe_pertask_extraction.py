#!/usr/bin/env python3
"""Per-task per-frame extraction ceiling for rooms_visited and co_occupancy (cf. steps_in_room=0.94).

For each task, query-condition the per-frame L19 rep on the task's question, then probe the per-frame
quantity that the task aggregates:
  rooms_visited : decode the target character's ROOM that frame (7-way: 6 rooms + absent) -> accuracy.
  co_occupancy  : decode "are C and D in the same room" that frame (binary) -> bal-acc + AUC.
Compares each to steps_in_room's 0.94 to see whether each task sits at its own extraction ceiling.
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
from evaluations.scripts import probe_evidence_selection_image as pi
from evaluations.scripts import probe_evidence_selection_linear as pr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    p.add_argument("--split", default="all_uniform")
    p.add_argument("--seq-lens", default="4,6,8")
    p.add_argument("--max-samples", type=int, default=90)
    p.add_argument("--read-layer", type=int, default=19)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text", action="store_true", help="frames as TEXT (token spans) instead of images")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "probe_pertask_extraction")
    return p.parse_args()


def reps_for(model, processor, frames, question, read_layer, device):
    r = pi.per_frame_vision_reps(model, processor, frames, question, len(frames), device)
    return None if r is None else r.half().cpu()  # [L+1, n, H] ALL layers


def text_reps_for(model, tokenizer, states, question, device):
    """TEXT-frame counterpart: per-frame reps from the text-frame token spans (frames given as text,
    not images). The state is explicit in the tokens, so this measures whether ANY extraction gap
    survives once frames are textual."""
    prompt, char_spans = pr.build_probe_prompt(question, states)
    tok_spans = pr.char_spans_to_token_spans(tokenizer, prompt, char_spans)
    if len(tok_spans) != len(states):
        return None
    r = pr.per_frame_reps(model, tokenizer, prompt, tok_spans, device)  # [L+1, n, H]
    return None if r is None else r.half().cpu()


def fit_multiclass(Xtr, ytr, Xte, yte, nc, dev, epochs=300, lr=0.05, wd=1e-3):
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(dev); Xte = ((Xte - mu) / sd).to(dev)
    W = torch.zeros(Xtr.shape[1], nc, requires_grad=True, device=dev)
    b = torch.zeros(nc, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd); lossf = nn.CrossEntropyLoss()
    yt = ytr.to(dev)
    for _ in range(epochs):
        opt.zero_grad(); lossf(Xtr @ W + b, yt).backward(); opt.step()
    pred = (Xte @ W + b).argmax(1).cpu()
    return float((pred == yte).float().mean())


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    seq_lens = [int(x) for x in str(args.seq_lens).replace(",", " ").split()]
    dev = base.resolve_device(args.device); dtype = base.resolve_dtype(args.dtype, dev)
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "run.log").open("w")
    def emit(m): print(m, flush=True); log.write(m + "\n"); log.flush()
    model, processor = base.load_model_and_processor(args.model_name, dev, dtype, bool(args.load_in_4bit))

    rooms_vocab = None
    rm_X: List[torch.Tensor] = []; rm_y: List[int] = []; rm_s: List[int] = []   # rooms: room-of-C (7-way)
    co_X: List[torch.Tensor] = []; co_y: List[int] = []; co_s: List[int] = []   # co-occ: same-room binary
    sid = 0
    for sl in seq_lens:
        sr = args.data_root / f"seq_len_{sl}" / args.split
        if not sr.is_dir():
            continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        rng.shuffle(dirs); dirs = dirs[: args.max_samples]
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            if not states:
                continue
            rooms = list(states[0]["rooms"].keys())
            if rooms_vocab is None:
                rooms_vocab = {r: i for i, r in enumerate(rooms)}; absent = len(rooms_vocab)
            present = [c for st in states for occ in st["rooms"].values() for c in occ]
            freq = Counter(present)
            if not freq:
                continue
            def get_reps(question):
                if args.text:
                    return text_reps_for(model, processor.tokenizer, states, question, dev)
                return reps_for(model, processor, pi.load_frames(d, states, json.loads((d/'metadata.json').read_text())),
                                question, args.read_layer, dev)
            # rooms_visited: main character = most-present
            C = freq.most_common(1)[0][0]
            try:
                rq = get_reps(f"How many distinct rooms did {C} visit across the {len(states)} frames?")
            except Exception as exc:
                emit(f"  skip {d.name}: {exc}"); continue
            if rq is not None:
                for fi, st in enumerate(states):
                    r = tf.room_of(st, C)
                    rm_X.append(rq[:, fi, :]); rm_y.append(rooms_vocab.get(r, absent)); rm_s.append(sid)
            # co_occupancy: two most-present chars
            if len(freq) >= 2:
                C2, D2 = [c for c, _ in freq.most_common(2)]
                try:
                    cq = get_reps(f"In how many of the {len(states)} frames were {C2} and {D2} in the same room?")
                except Exception:
                    cq = None
                if cq is not None:
                    for fi, st in enumerate(states):
                        rc, rd = tf.room_of(st, C2), tf.room_of(st, D2)
                        lab = int(rc == rd and rc != "not present")
                        co_X.append(cq[:, fi, :]); co_y.append(lab); co_s.append(sid)
            sid += 1
        emit(f"seq_len={sl}: rooms_frames={len(rm_y)} cooc_frames={len(co_y)}")

    def split(s):
        u = sorted(set(s)); rng.shuffle(u); cut = int(0.7 * len(u)); trn = set(u[:cut])
        return ([i for i, x in enumerate(s) if x in trn], [i for i, x in enumerate(s) if x not in trn])

    emit("")
    do_rooms, do_cooc = len(rm_X) > 0, len(co_X) > 0   # 1-char data has no pairs -> no co-occ
    if do_rooms:
        Xr = torch.stack(rm_X); yr = torch.tensor(rm_y); tr, te = split(rm_s); nL = Xr.shape[1]
        maj = Counter(rm_y).most_common(1)[0][1] / len(rm_y)
    if do_cooc:
        Xc = torch.stack(co_X); yc = torch.tensor(co_y); tr2, te2 = split(co_s); nL = Xc.shape[1]
    emit(f"per-layer sweep (rooms={do_rooms} cooc={do_cooc})  rooms majority={maj if do_rooms else float('nan'):.3f}")
    best_r = (-1, 0.0); best_c = (-1, 0.0); rows = ["layer,rooms_acc,cooc_auc"]
    for L in range(nL):
        ar = ac = float("nan")
        if do_rooms:
            ar = fit_multiclass(Xr[:, L, :].float()[tr], yr[tr], Xr[:, L, :].float()[te], yr[te], len(rooms_vocab) + 1, dev)
            if ar > best_r[1]:
                best_r = (L, ar)
        if do_cooc:
            _, ac = pr.fit_logreg(Xc[:, L, :].float()[tr2], yc[tr2], Xc[:, L, :].float()[te2], yc[te2])
            if ac > best_c[1]:
                best_c = (L, ac)
        emit(f"  L{L:2d}: rooms acc={ar:.3f} | co-occ auc={ac:.3f}")
        rows.append(f"{L},{ar:.4f},{ac:.4f}")
    (run_dir / "per_layer.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    emit("")
    if do_rooms:
        emit(f"BEST rooms_visited room-decode: L{best_r[0]} acc={best_r[1]:.3f}   (L19 reference; steps is-evidence peak ~L19-20 = 0.94)")
    if do_cooc:
        emit(f"BEST co_occupancy same-room:    L{best_c[0]} auc={best_c[1]:.3f}")
    log.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
