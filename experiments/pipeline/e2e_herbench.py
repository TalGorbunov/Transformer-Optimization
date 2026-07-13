#!/usr/bin/env python3
"""P4b: retrieve-then-verify on HERBench armB (real video) — the regime-2 constructive test.

The per-frame verifier ALREADY exists: the look-again judge scores (lookagain.json, computed
2026-07-07 for curation, independent of any probe axis). Pipeline: tally = #frames with
P(yes) > --thr → render as the predicate-matched fact → frozen model verbalizes. Baselines:
the frozen armB open answer (rows from own_answer/armB) and MCQ.

Registered: MODEST lift only — bounded by the d′≈1 graded-evidence perception supply.
Also reports the tally-vs-gold ceiling directly (= what the verifier supply allows).
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.scripts.patch_importence import group_restoration_importance as gri

INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/herbench_ac/armB_ev_fill16")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--baseline-rows",
                    default="outputs/herbench/own_answer/armB_20260707_213726/rows.json")
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id

    base = {r["qid"]: r for r in json.loads(Path(args.baseline_rows).read_text())}
    rows = []
    dirs = sorted(d for d in Path(args.data_root).iterdir() if (d / "lookagain.json").exists())
    for i, sd in enumerate(dirs):
        meta = json.loads((sd / "meta.json").read_text())
        la = json.loads((sd / "lookagain.json").read_text())
        gold = int(meta["visible_count"])
        NF = int(meta["n_frames"])
        pair = meta.get("pair") or " ".join(meta.get("pair_words", []))
        tally = sum(1 for v in la.values() if v > args.thr)
        fact = (f"Note: the person is performing the action '{pair}' in exactly {tally} "
                f"of the {NF} frames.")
        q = (f"In how many of these {NF} frames is the person performing the action '{pair}'?")
        prompt = (f"{fact}\nRespond with a single integer. Output only the integer.\n"
                  f"Question: {q}\nAnswer: ")
        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([], prompt))
            plen = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                gen = model.generate(**inputs, do_sample=False, max_new_tokens=5,
                                     pad_token_id=pad)
            dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0]
            m = INT_RE.search(dec)
            rend = int(m.group(0)) if m else None
        except Exception as exc:
            print(f"{sd.name} render failed: {exc}", flush=True)
            rend = None
        b = base.get(meta["question_id"], {})
        rows.append({"qid": meta["question_id"], "gold": gold, "tally": tally,
                     "rendered": rend, "frozen_open": b.get("pred"),
                     "frozen_mcq_ok": b.get("mcq_ok")})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(dirs)}", flush=True)

    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    g = np.array([r["gold"] for r in rows])
    t = np.array([r["tally"] for r in rows])
    rd = np.array([r["rendered"] if r["rendered"] is not None else -99 for r in rows])
    fo = np.array([r["frozen_open"] if r["frozen_open"] is not None else -99 for r in rows])
    mc = np.array([r["frozen_mcq_ok"] for r in rows if r["frozen_mcq_ok"] is not None])
    lines = [f"=== E2E HERBench armB (judge-verify → tally → fact render; n={len(rows)}, "
             f"thr={args.thr}) ===",
             f"  tally (judge supply ceiling) exact = {float((t == g).mean()):.3f}  "
             f"MAE {float(np.abs(t - g).mean()):.2f}",
             f"  rendered answer exact = {float((rd == g).mean()):.3f}",
             f"  frozen open baseline = {float((fo == g).mean()):.3f}",
             f"  frozen MCQ baseline = {float(mc.mean()):.3f}"]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
