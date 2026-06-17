#!/usr/bin/env python3
"""Aggregation-vs-extraction probes for the two set-cardinality tasks (rooms_visited, co_occupancy).

Two decoders, each run as LINEAR (logistic) AND MLP, across layers, on the REAL 7B forward pass:

  (A) PER-FRAME evidence decoder — is the per-frame fact extracted at the frame tokens?
        rooms_visited: label = queried char's ROOM in that frame (multiclass)
        co_occupancy:  label = are the two queried chars in the SAME room in that frame (binary)
      If this decodes well but the answer (B) does not -> the per-frame info IS present and the
      failure is AGGREGATION, not extraction.

  (B) LAST-TOKEN answer decoder — is the final count present at the last prompt token, and at which
      DEPTH does it crystallize? (the 'stages' question: answer should emerge late, after frames->carrier)

Always run on 7B. Small by default (--limit 60). Honest baselines: majority-class per split.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Optional
import numpy as np
import torch
from nnsight import LanguageModel

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups


def char_room_at(states, t, char) -> Optional[str]:
    for room, occ in eval_utils.rooms_to_room2chars(states[t].get("rooms", {})).items():
        if char in occ:
            return room
    return None


def probe(x, y, seeds, clf_kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    accs, base = [], []
    for s in seeds:
        strat = y if len(set(y)) > 1 and min(Counter(y).values()) >= 2 else None
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=s, stratify=strat)
        if clf_kind == "mlp":
            clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, random_state=s))
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
        clf.fit(xtr, ytr)
        accs.append(accuracy_score(yte, clf.predict(xte)))
        mc = Counter(ytr).most_common(1)[0][0]
        base.append(accuracy_score(yte, [mc] * len(yte)))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(base))


def pick_pair(states, chars):
    """For co_occupancy: choose the char pair with the most same-room frames (>=1)."""
    best, best_gold = None, -1
    for c1, c2 in combinations(sorted(chars), 2):
        g = sum(1 for t in range(len(states))
                if char_room_at(states, t, c1) is not None
                and char_room_at(states, t, c1) == char_room_at(states, t, c2))
        if g > best_gold:
            best, best_gold = (c1, c2), g
    return best, best_gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rooms_visited", "co_occupancy"], required=True)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--layers", default="4,8,12,16,20,24,27")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--pool", default="meanmax", choices=["mean", "max", "meanmax"])
    ap.add_argument("--output", default="outputs/probe_aggregation_stages")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    lm = LanguageModel(gri._model(), tokenizer=gri._processor().tokenizer)
    layers = get_layers(lm.model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    seeds = [int(s) for s in str(args.seeds).replace(",", " ").split()]
    out = Path(args.output) / args.task
    out.mkdir(parents=True, exist_ok=True)

    frame_feats: Dict[int, List[np.ndarray]] = defaultdict(list)
    frame_labels: List[str] = []
    last_feats: Dict[int, List[np.ndarray]] = defaultdict(list)
    last_labels: List[int] = []
    n = 0
    for sd in iter_sample_dirs(Path(args.data_root)):
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        chars = sorted(eval_utils.extract_characters_from_states(states))
        if len(chars) < 2:
            continue

        if args.task == "rooms_visited":
            present = lambda c: [t for t in range(len(states)) if char_room_at(states, t, c)]
            char = max(chars, key=lambda c: (len(present(c)), c))
            pres = present(char)
            if len(pres) < 2:
                continue
            gold = len({char_room_at(states, t, char) for t in pres})
            question = f"How many distinct rooms did {char} visit?"
            frame_targets = {t: char_room_at(states, t, char) for t in pres}  # room str
        else:  # co_occupancy
            (c1, c2), gold = pick_pair(states, chars)
            if gold < 1:
                continue
            question = f"In how many of the {len(frames)} frames were {c1} and {c2} in the same room?"
            frame_targets = {}
            for t in range(len(states)):
                r1, r2 = char_room_at(states, t, c1), char_room_at(states, t, c2)
                if r1 is not None and r2 is not None:  # both present -> binary same/diff
                    frame_targets[t] = "same" if r1 == r2 else "diff"

        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, question))
            fg = image_token_groups(inputs["input_ids"][0].detach().cpu(),
                                    expected_num_frames=len(frames), processor=gri._processor())
            last_idx = int(inputs["input_ids"].shape[1]) - 1
            saved = {}
            with torch.no_grad():
                with lm.trace(inputs):
                    for L in probe_layers:
                        saved[L] = tgi._to_hidden_tensor(layers[L].output).save()
            per_layer_h = {L: tgi._materialize_saved(saved[L])[0] for L in probe_layers}  # [seq,dim]
        except Exception as exc:
            print(f"{sid} capture failed: {type(exc).__name__}: {exc}")
            continue

        # (B) last-token answer features
        for L in probe_layers:
            last_feats[L].append(per_layer_h[L][last_idx, :].float().cpu().numpy().astype(np.float32))
        last_labels.append(int(gold))

        # (A) per-frame evidence features
        for t, lab in frame_targets.items():
            if t >= len(fg) or lab is None:
                continue
            pos = [int(p) for p in fg[t]]
            if not pos:
                continue
            for L in probe_layers:
                h = per_layer_h[L][pos, :].float()
                v = torch.cat([h.mean(0), h.amax(0)], -1) if args.pool == "meanmax" else (
                    h.mean(0) if args.pool == "mean" else h.amax(0))
                frame_feats[L].append(v.cpu().numpy().astype(np.float32))
            frame_labels.append(str(lab))
        n += 1
        if n % 15 == 0:
            print(f"  scanned {n}: {len(last_labels)} last-tok, {len(frame_labels)} frame examples")

    lines = [f"=== AGGREGATION-STAGES PROBE ({args.task}, 7B) n_samples={len(last_labels)} ==="]
    rows = ["probe,layer,n,clf,acc,acc_std,majority,lift"]

    yb = np.array(last_labels)
    lines.append(f"\n(B) LAST-TOKEN answer decode  classes={sorted(set(yb.tolist()))}")
    for L in probe_layers:
        if len(last_feats[L]) < 12 or len(set(yb.tolist())) < 2:
            continue
        xb = np.stack(last_feats[L])
        for clf in ("logistic", "mlp"):
            acc, std, base = probe(xb, yb, seeds, clf)
            lines.append(f"  L{L:<2d} {clf:<8s}: acc={acc:.3f}±{std:.3f} (maj {base:.3f}) lift={acc-base:+.3f}")
            rows.append(f"last_token,{L},{len(yb)},{clf},{acc:.4f},{std:.4f},{base:.4f},{acc-base:.4f}")

    yf = np.array(frame_labels)
    lines.append(f"\n(A) PER-FRAME evidence decode  classes={sorted(set(yf.tolist()))}")
    for L in probe_layers:
        if len(frame_feats[L]) < 12 or len(set(yf.tolist())) < 2:
            continue
        xf = np.stack(frame_feats[L])
        for clf in ("logistic", "mlp"):
            acc, std, base = probe(xf, yf, seeds, clf)
            lines.append(f"  L{L:<2d} {clf:<8s}: acc={acc:.3f}±{std:.3f} (maj {base:.3f}) lift={acc-base:+.3f}")
            rows.append(f"per_frame,{L},{len(yf)},{clf},{acc:.4f},{std:.4f},{base:.4f},{acc-base:.4f}")

    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
