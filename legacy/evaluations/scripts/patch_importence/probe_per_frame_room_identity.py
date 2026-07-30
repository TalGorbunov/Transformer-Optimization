#!/usr/bin/env python3
"""DECISIVE diagnostic for the set-cardinality solution: is the queried character's ROOM decodable
from a SINGLE frame's tokens on 7B? If yes (high per-frame room-classification accuracy), then room
IDENTITY is extracted by the frozen model and the bottleneck is the DEDUP/COUNT computation at the
carrier → build a structured dedup readout. If no, identity isn't extracted → need to unfreeze backbone.

For each sample + queried character, for each frame where the character is present: capture that frame's
image-token hidden states (mean-pooled) at several layers, label = which room the character is in.
Linear-probe (multinomial logistic) the room, per layer.  ALWAYS run with --model 7B.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List
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
from models.model import get_layers, image_token_groups, DEFAULT_MODEL_ID


def char_room_at(states, t, char):
    for room, occ in eval_utils.rooms_to_room2chars(states[t].get("rooms", {})).items():
        if char in occ:
            return room
    return None


def probe(x, y, seeds, clf_kind="logistic"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    accs, base = [], []
    for s in seeds:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=s, stratify=y if len(set(y))>1 else None)
        if clf_kind == "mlp":
            clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=s))
        else:
            clf = LogisticRegression(max_iter=2000, random_state=s)
        clf.fit(xtr, ytr)
        accs.append(accuracy_score(yte, clf.predict(xte)))
        # majority-class baseline
        from collections import Counter
        mc = Counter(ytr).most_common(1)[0][0]
        base.append(accuracy_score(yte, [mc]*len(yte)))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(base))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=40, help="samples to scan")
    ap.add_argument("--layers", default="4,8,12,16,20,24")
    ap.add_argument("--model_name","--model",dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--clf",default="logistic",choices=["logistic","mlp"])
    ap.add_argument("--pool", default="mean", choices=["mean", "max", "meanmax"],
                    help="how to pool a frame's tokens; max/meanmax preserve a localized character signal better")
    ap.add_argument("--output", default="outputs/probe_per_frame_room_identity")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    lm = LanguageModel(gri._model(), tokenizer=gri._processor().tokenizer)
    layers = get_layers(lm.model)
    probe_layers = [int(x) for x in str(args.layers).replace(","," ").split()]
    seeds = [int(s) for s in str(args.seeds).replace(","," ").split()]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    feats: Dict[tuple, List[np.ndarray]] = defaultdict(list)  # layer -> [vec]
    labels: List[str] = []
    n = 0
    for sd in iter_sample_dirs(Path(args.data_root)):
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        chars = sorted(eval_utils.extract_characters_from_states(states))
        if not chars:
            continue
        # pick the character present in the most frames (matches the rooms_visited task selection)
        present = lambda c: [t for t in range(len(states)) if char_room_at(states, t, c)]
        char = max(chars, key=lambda c: (len(present(c)), c))
        pres = present(char)
        if len(pres) < 2:
            continue
        try:
            from PIL import Image
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, f"How many distinct rooms did {char} visit?"))
            fg = image_token_groups(inputs["input_ids"][0].detach().cpu(), expected_num_frames=len(frames), processor=gri._processor())
            with torch.no_grad():
                with lm.trace(inputs):
                    saved = {L: tgi._to_hidden_tensor(layers[L].output).save() for L in probe_layers}
            per_layer_h = {L: tgi._materialize_saved(saved[L])[0] for L in probe_layers}  # [seq,dim]
        except Exception as exc:
            print(f"{sid} capture failed: {type(exc).__name__}: {exc}")
            continue
        # one example per present frame: mean-pool that frame's tokens, label = its room
        for t in pres:
            room = char_room_at(states, t, char)
            if room is None or t >= len(fg):
                continue
            pos = [int(p) for p in fg[t]]
            if not pos:
                continue
            for L in probe_layers:
                h = per_layer_h[L][pos, :].float()
                if args.pool == "mean":
                    v = h.mean(0)
                elif args.pool == "max":
                    v = h.amax(0)
                else:
                    v = torch.cat([h.mean(0), h.amax(0)], dim=-1)
                feats[L].append(v.cpu().numpy().astype(np.float32))
            labels.append(room)
        n += 1
        if n % 10 == 0:
            print(f"  scanned {n} samples, {len(labels)} frame-examples")

    y = np.array(labels)
    lines = [f"=== PER-FRAME ROOM-IDENTITY probe (7B) — n_frame_examples={len(y)}, rooms={sorted(set(y.tolist()))} ==="]
    rows = ["layer,n,room_acc,room_acc_std,majority_baseline,lift"]
    for L in probe_layers:
        if len(feats[L]) < 12 or len(set(y.tolist())) < 2:
            continue
        x = np.stack(feats[L])
        acc, std, base = probe(x, y, seeds, args.clf)
        lines.append(f"  L{L:<2d}: room_acc={acc:.3f}±{std:.3f}  (majority {base:.3f})  lift={acc-base:+.3f}")
        rows.append(f"{L},{len(y)},{acc:.4f},{std:.4f},{base:.4f},{acc-base:.4f}")
    report = "\n".join(lines)+"\n"
    print(report)
    (out/"report.txt").write_text(report, encoding="utf-8")
    (out/"metrics.csv").write_text("\n".join(rows)+"\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
