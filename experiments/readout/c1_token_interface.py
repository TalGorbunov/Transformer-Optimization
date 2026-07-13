#!/usr/bin/env python3
"""C1 [preapproved-smoke]: the TOKEN interface baseline for readout injection.

Write an oracle tally into the prompt as digits ("Counted occurrences so far: K.") with ZERO
training, and measure whether the frozen model verbalizes it. This defines the bar every
activation-level injection route (C2 digit-codebook / C3 native-geometry / C-control per-count
codebook) must beat, and measures the interface cost of the token route.

Arms (steps_in_room, N=8 images, deployed visual context):
  base_std      : standard prompt, no tally (control; reproduces the ~0.21 baseline)
  oracle_std    : tally = gold, standard "0 to 8" instruction  -> target = gold
  oracle_open   : tally = gold, open "single integer" instruction -> target = gold
  cf_in_open    : tally = counterfactual k in 0..8, k != gold, open instr -> target = k
  cf_ood_open   : tally = counterfactual k in {11,13,17,23,29,34,40}, open instr -> target = k
                  (digits-compose test: multi-token answers, unseen-count range)

Answer reading: BOTH single-forward digit argmax (0..8, comparable to every prior run) AND
greedy generation + first-integer parse (multi-digit capable — the B3/C answer-reader extension).
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri

OOD_TALLIES = [11, 13, 17, 23, 29, 34, 40]


def build_prompt(question: str, n_frames: int, tally, open_instr: bool,
                 phrasing: str = "tally", cr=None, pos: str = "pre", distract=None) -> str:
    """phrasing: how the oracle count is written into the prompt.
      tally      : 'Counted occurrences so far: K.' before the instruction (v1 arms)
      answer     : 'The correct answer is K.' before the instruction (explicit connective)
      fact       : 'Note: C spent exactly K steps in the R.' (semantic statement; needs cr=(C,R))
      after_q    : tally sentence AFTER the question, right before 'Answer: ' (legacy pos alias)
      fact_para1 : 'For reference, C was in the R in K of the frames.'
      fact_para2 : 'It is known that C appears in the R exactly K times.'
      fact_words : canonical fact with the count as an English number word
      fact_src   : source-attributed ('An automated frame counter reports: ...')
    pos: where the note goes — 'pre' (before instruction), 'after_q' (after the question),
         'top' (before the head line, i.e. maximally far from the answer slot).
    distract: optional (D, R2, J) — a same-form fact about ANOTHER character/room with count J,
         placed immediately before the true note (C1b binding test).
    """
    head = f"You will be shown {n_frames} frames describing steps in a house.\n"
    instr = ("Respond with a single integer. Output only the integer.\n" if open_instr else
             f"Respond with a single integer from 0 to {n_frames} (0 is allowed). "
             f"Output only the integer.\n")
    if tally is None:
        return head + instr + f"Question: {question}\nAnswer: "
    if phrasing == "after_q":       # legacy alias: tally note at the after_q position
        phrasing, pos = "tally", "after_q"
    if phrasing == "answer":
        note = f"The correct answer is {tally}.\n"
    elif phrasing == "fact" and cr is not None:
        note = f"Note: {cr[0]} spent exactly {tally} steps in the {cr[1]}.\n"
    elif phrasing == "fact_para1" and cr is not None:
        note = f"For reference, {cr[0]} was in the {cr[1]} in {tally} of the frames.\n"
    elif phrasing == "fact_para2" and cr is not None:
        note = f"It is known that {cr[0]} appears in the {cr[1]} exactly {tally} times.\n"
    elif phrasing == "fact_words" and cr is not None:
        from num2words import num2words
        note = f"Note: {cr[0]} spent exactly {num2words(tally)} steps in the {cr[1]}.\n"
    elif phrasing == "fact_src" and cr is not None:
        note = (f"An automated frame counter reports: {cr[0]} spent exactly {tally} steps "
                f"in the {cr[1]}.\n")
    else:
        note = f"Counted occurrences so far: {tally}.\n"
    if distract is not None:
        D, R2, J = distract
        note = f"Note: {D} spent exactly {J} steps in the {R2}.\n" + note
    if pos == "after_q":
        return head + instr + f"Question: {question}\n{note}Answer: "
    if pos == "top":
        return note + head + instr + f"Question: {question}\nAnswer: "
    return head + note + instr + f"Question: {question}\nAnswer: "


def frames_as_text(states) -> str:
    lines = []
    for i, st in enumerate(states, start=1):
        lines.append(f"Frame {i}:")
        for room, occ in st["rooms"].items():
            lines.append(f"  {room}: {', '.join(occ) if occ else '(empty)'}")
    return "\n".join(lines)


def load_states_only(sd: Path):
    import ast
    lines = (sd / "qa.txt").read_text(encoding="utf-8").splitlines()
    qi = next(i for i, ln in enumerate(lines) if ln.strip() == "question:")
    ai = next(i for i, ln in enumerate(lines) if ln.strip() == "answer:")
    states, q0 = [], None
    for ln in lines[qi + 1:ai]:
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
        elif s:
            q0 = s; break
    a0 = next(ln.strip() for ln in lines[ai + 1:] if ln.strip())
    return sd.name, q0, states, a0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--text-frames", action="store_true",
                    help="C-range: frames as TEXT (states-only data OK); prompt embeds the frame "
                         "text where images would go")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--arms", default="base_std,oracle_std,oracle_open,cf_in_open,cf_ood_open")
    ap.add_argument("--output", default="outputs/_scratch/c1_token_interface")
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    cand_ids, cand_vals = [], []
    for d in range(0, 9):
        enc = tok.encode(str(d), add_special_tokens=False)
        if len(enc) == 1:
            cand_ids.append(int(enc[0])); cand_vals.append(d)
    cand_ids_t = torch.tensor(cand_ids, dtype=torch.long)
    pad = tok.pad_token_id or tok.eos_token_id

    import re
    INT_RE = re.compile(r"-?\d+")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)

    rows = []          # per (sample, arm) record
    n = 0
    rng_cf = random.Random(1234)
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            if args.text_frames:
                sid, q0, states, a0 = load_states_only(sd)
                frames = []
            else:
                sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            continue
        hi = len(states) if args.text_frames else len(frames)
        cf_in = rng_cf.choice([k for k in range(0, hi + 1) if k != gold])
        cf_ood = rng_cf.choice(OOD_TALLIES)
        import re as _re
        m_cr = _re.search(r"did (\w+) spend in the (\w+)", q0)
        cr = (m_cr.group(1), m_cr.group(2)) if m_cr else None
        # C1b distractor: a same-form fact about another character/room, different count
        distract = None
        if cr is not None and states:
            from evaluations.helpers import utils as _eu
            _chars = sorted(_eu.extract_characters_from_states(states))
            _rooms = sorted(states[0].get("rooms", {}).keys())
            _dc = [c for c in _chars if c != cr[0]]
            _dr = [r for r in _rooms if r != cr[1]]
            if _dc and _dr:
                J = rng_cf.choice([k for k in range(0, hi + 1) if k not in (cf_in, cf_ood)])
                distract = (rng_cf.choice(_dc), rng_cf.choice(_dr), J)
        # arm: (tally, open_instr, target, phrasing, pos, use_distractor)
        arm_specs = {
            "base_std":    (None,   False, gold,   "tally", "pre", False),
            "oracle_std":  (gold,   False, gold,   "tally", "pre", False),
            "oracle_open": (gold,   True,  gold,   "tally", "pre", False),
            "cf_in_open":  (cf_in,  True,  cf_in,  "tally", "pre", False),
            "cf_ood_open": (cf_ood, True,  cf_ood, "tally", "pre", False),
            "ans_std":     (gold,   False, gold,   "answer", "pre", False),
            "ans_cf_in":   (cf_in,  True,  cf_in,  "answer", "pre", False),
            "ans_cf_ood":  (cf_ood, True,  cf_ood, "answer", "pre", False),
            "fact_std":    (gold,   False, gold,   "fact", "pre", False),
            "fact_cf_in":  (cf_in,  True,  cf_in,  "fact", "pre", False),
            "fact_cf_ood": (cf_ood, True,  cf_ood, "fact", "pre", False),
            "afterq_std":  (gold,   False, gold,   "after_q", "pre", False),
            "afterq_cf_ood": (cf_ood, True, cf_ood, "after_q", "pre", False),
            # --- C1b phrasing-robustness grid (fact family, counterfactual targets) ---
            "para1_cf_in":   (cf_in,  True, cf_in,  "fact_para1", "pre", False),
            "para1_cf_ood":  (cf_ood, True, cf_ood, "fact_para1", "pre", False),
            "para2_cf_ood":  (cf_ood, True, cf_ood, "fact_para2", "pre", False),
            "words_cf_in":   (cf_in,  True, cf_in,  "fact_words", "pre", False),
            "words_cf_ood":  (cf_ood, True, cf_ood, "fact_words", "pre", False),
            "src_cf_ood":    (cf_ood, True, cf_ood, "fact_src", "pre", False),
            "top_cf_ood":    (cf_ood, True, cf_ood, "fact", "top", False),
            "factq_cf_ood":  (cf_ood, True, cf_ood, "fact", "after_q", False),
            "dis_cf_in":     (cf_in,  True, cf_in,  "fact", "pre", True),
            "dis_cf_ood":    (cf_ood, True, cf_ood, "fact", "pre", True),
        }
        for arm in arms:
            tally, open_instr, target, phrasing, pos, use_dis = arm_specs[arm]
            if phrasing.startswith("fact") and cr is None:
                continue
            if use_dis and distract is None:
                continue
            prompt = build_prompt(q0, hi, tally, open_instr, phrasing, cr, pos=pos,
                                  distract=distract if use_dis else None)
            if args.text_frames:
                # frames as text replace the image block; drop the "You will be shown" head line
                prompt = (f"You are given {hi} frames describing steps in a house, as text.\n"
                          + frames_as_text(states) + "\n\n" + prompt.split("\n", 1)[1])
            try:
                inputs = tgi.move_inputs_to_model_device(
                    tgi.build_inputs_from_prompt(frames, prompt))
                with torch.no_grad():
                    outp = model(**inputs, use_cache=False)
                last_logits = outp.logits[0, -1].float().cpu()
                pred_argmax = int(cand_vals[int(torch.argmax(last_logits[cand_ids_t]).item())])
                plen = int(inputs["input_ids"].shape[1])
                with torch.no_grad():
                    gen = model.generate(**inputs, do_sample=False,
                                         max_new_tokens=args.max_new_tokens, pad_token_id=pad)
                dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0]
                m = INT_RE.search(dec)
                pred_gen = int(m.group(0)) if m else None
            except Exception as exc:
                print(f"{sid}/{arm} failed: {type(exc).__name__}: {exc}")
                continue
            rows.append({"sid": sid, "arm": arm, "gold": gold, "tally": tally,
                         "target": target, "pred_argmax": pred_argmax,
                         "pred_gen": pred_gen, "raw_gen": dec})
        n += 1
        if n % 20 == 0:
            print(f"  {n}/{args.limit} samples", flush=True)
        if n % 25 == 0:  # walltime insurance: partial rows survive a kill
            (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    # ---- report ----
    lines = [f"=== C1 TOKEN-INTERFACE SMOKE (steps, N=8, n={n}) ==="]
    summary = {}
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        if not rr:
            continue
        acc_gen = float(np.mean([r["pred_gen"] == r["target"] for r in rr]))
        acc_am = float(np.mean([r["pred_argmax"] == r["target"] for r in rr]))
        acc_gold_gen = float(np.mean([r["pred_gen"] == r["gold"] for r in rr]))
        mae = float(np.mean([abs((r["pred_gen"] if r["pred_gen"] is not None else -99)
                                 - r["target"]) for r in rr]))
        lines.append(f"  {arm:<12s} n={len(rr):>4d}  acc_vs_target(gen)={acc_gen:.3f}  "
                     f"acc_vs_target(argmax)={acc_am:.3f}  acc_vs_gold(gen)={acc_gold_gen:.3f}  "
                     f"MAE_vs_target={mae:.2f}")
        summary[arm] = {"n": len(rr), "acc_gen": acc_gen, "acc_argmax": acc_am,
                        "acc_gold_gen": acc_gold_gen, "mae_target": mae}
        # per-target breakdown for the tally arms
        if arm != "base_std":
            per = defaultdict(list)
            for r in rr:
                per[r["target"]].append(int(r["pred_gen"] == r["target"]))
            bt = "    by-target: " + " ".join(
                f"{t}:{np.mean(v):.2f}(n{len(v)})" for t, v in sorted(per.items()))
            lines.append(bt)
    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
