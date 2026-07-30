#!/usr/bin/env python3
"""Simultaneous-extraction causal patch (steps_in_room).

Hypothesis (user): single-frame extraction is easy, but when N frames compete, attention spreads and
per-evidence-frame extraction degrades -> part of the count failure is *competed* extraction, not just
aggregation. Causal test: take each sample's EVIDENCE frames computed in an UNCOMPETED context
(evidence-frames-only forward pass = SOURCE) and patch those reps into the FULL N-frame forward pass
(TARGET) at one decoder layer. Read the count.

  - if patching the uncompeted evidence reps RAISES P(gold)/accuracy on the full run -> in-context
    extraction WAS degraded by competition (the user's hypothesis is causal).
  - if it does ~nothing -> in-context per-frame reps were fine; the failure is downstream AGGREGATION.

Readout = graded P(gold count) over the valid integer answers (good SNR even when exact-match is ~0),
plus exact-match accuracy and fix/break rates. Compares clean-full vs patched (both full context),
with clean-source (evidence-only) as an easy reference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch
from nnsight import LanguageModel

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.eval_mmred_text_frames_acc import room_of
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups


def softmax_probs(scores: Dict[int, float]) -> Dict[int, float]:
    ks = sorted(scores)
    v = torch.tensor([scores[k] for k in ks], dtype=torch.float64)
    p = torch.softmax(v, dim=0)
    return {k: float(p[i]) for i, k in enumerate(ks)}


def clean_scores(lm, inputs, prompt_len, cand):
    out = {}
    for c in cand:
        ids = tgi.token_ids_of_answer(str(c))
        si = tgi.append_answer_tokens_for_scoring(inputs, ids)
        out[c] = tgi.run_clean_sequence_logprob(lm, si, prompt_len=prompt_len, answer_token_ids=ids)
    return out


def patched_scores(lm, layers, tgt_inputs, src_inputs, layer_idx, tgt_pos, src_pos, prompt_len, cand):
    out = {}
    for c in cand:
        ids = tgi.token_ids_of_answer(str(c))
        tsi = tgi.append_answer_tokens_for_scoring(tgt_inputs, ids)
        ssi = tgi.append_answer_tokens_for_scoring(src_inputs, ids)
        sc = tgi.run_layer_corrupted_sequence_logprob(
            lm=lm, layers=layers, target_scoring_inputs=tsi, source_scoring_inputs=ssi,
            layer_idx=layer_idx, target_token_positions=tgt_pos, source_token_positions=src_pos,
            prompt_len=prompt_len, answer_token_ids=ids)
        out[c] = sc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_steps_balanced/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--layers", default="14,19", help="decoder layers to patch at")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", default="outputs/frame_axis/probes/simultaneous_extraction_patch")
    ap.add_argument("--tag", default="steps_5char")
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    proc = gri._processor()
    lm = LanguageModel(gri._model(), tokenizer=proc.tokenizer)
    layers = get_layers(lm.model)
    patch_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    agg = {"full": defaultdict(list), "source": defaultdict(list)}
    pat = {L: defaultdict(list) for L in patch_layers}
    rows = ["sid,gold,N,n_ev,full_pgold,full_pred," + ",".join(f"L{L}_pgold,L{L}_pred" for L in patch_layers)]
    # sample dirs are sorted by count; SHUFFLE so a --limit spans the full count range
    import random
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(0).shuffle(all_dirs)
    n = 0
    for sd in all_dirs:
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            meta = json.loads((sd / "metadata.json").read_text())
            C, R = meta.get("target_character"), meta.get("target_room")
            if not C or not R:
                continue
            ev = [i for i, st in enumerate(states) if room_of(st, C) == R]
            non = [i for i in range(len(states)) if i not in ev]
            if len(ev) < 1 or len(non) < 1:        # need competition to remove
                continue
            ev_frames = [frames[i] for i in ev]
            N = len(frames)
            cand = list(range(N + 1))

            tgt = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0, processor=proc))
            src = tgi.move_inputs_to_model_device(tgi.build_inputs(ev_frames, q0, processor=proc))
            tgt_len = int(tgt["input_ids"].shape[1]); src_len = int(src["input_ids"].shape[1])
            g_tgt = image_token_groups(tgt["input_ids"][0].detach().cpu(), expected_num_frames=N, processor=proc)
            g_src = image_token_groups(src["input_ids"][0].detach().cpu(), expected_num_frames=len(ev), processor=proc)

            tgt_pos, src_pos = [], []
            for j, fi in enumerate(ev):                 # match j-th evidence frame
                tp, sp = [int(p) for p in g_tgt[fi]], [int(p) for p in g_src[j]]
                if len(tp) == len(sp) and tp and sp:    # identical render -> equal token count
                    tgt_pos += tp; src_pos += sp
            if not tgt_pos:
                continue
        except Exception as exc:
            print(f"skip {sd.name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        try:
            full = softmax_probs(clean_scores(lm, tgt, tgt_len, cand))
            source = softmax_probs(clean_scores(lm, src, src_len, list(range(len(ev) + 1))))
            full_pred = max(full, key=full.get); full_pgold = full.get(gold, 0.0)
            agg["full"]["pgold"].append(full_pgold); agg["full"]["acc"].append(int(full_pred == gold))
            agg["source"]["pgold"].append(source.get(gold, 0.0)); agg["source"]["acc"].append(int(max(source, key=source.get) == gold))
            row = [sid, gold, N, len(ev), f"{full_pgold:.4f}", full_pred]
            for L in patch_layers:
                pp = softmax_probs(patched_scores(lm, layers, tgt, src, L, tgt_pos, src_pos, tgt_len, cand))
                ppred = max(pp, key=pp.get); pgold = pp.get(gold, 0.0)
                pat[L]["pgold"].append(pgold); pat[L]["acc"].append(int(ppred == gold))
                pat[L]["fix"].append(int(full_pred != gold and ppred == gold))
                pat[L]["break"].append(int(full_pred == gold and ppred != gold))
                row += [f"{pgold:.4f}", ppred]
            rows.append(",".join(str(x) for x in row))
        except Exception as exc:
            print(f"score-fail {sid}: {type(exc).__name__}: {exc}", flush=True)
            continue
        n += 1
        if n % 10 == 0:
            print(f"  {n}: full_pgold={np.mean(agg['full']['pgold']):.3f} "
                  + " ".join(f"L{L}={np.mean(pat[L]['pgold']):.3f}" for L in patch_layers), flush=True)

    def m(xs): return float(np.mean(xs)) if xs else float("nan")
    lines = [f"=== SIMULTANEOUS-EXTRACTION PATCH  tag={args.tag}  n={n} ===",
             "patch SOURCE = evidence-frames-only (uncompeted); TARGET = full N frames.",
             "if patched P(gold)/acc >> clean-full -> competed extraction is CAUSAL; if ~equal -> pure aggregation.\n",
             f"clean-full   : P(gold)={m(agg['full']['pgold']):.3f}  acc={m(agg['full']['acc']):.3f}",
             f"clean-source : P(gold)={m(agg['source']['pgold']):.3f}  acc={m(agg['source']['acc']):.3f}  (evidence-only reference)"]
    for L in patch_layers:
        lines.append(f"patch L{L:<2d}   : P(gold)={m(pat[L]['pgold']):.3f}  acc={m(pat[L]['acc']):.3f}  "
                     f"fix={m(pat[L]['fix']):.3f}  break={m(pat[L]['break']):.3f}")
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "per_sample.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
