#!/usr/bin/env python3
"""P3: the E2E pipeline across the TASK ALGEBRA — same skeleton (per-frame verify → fixed
reduction → predicate-matched fact render → frozen verbalization), different reduction operator:

  rooms : per-frame "Which room is C in?" (name read from room-name logits) → SUPPORT SIZE of
          the answers (the union/distinct operator — deploys the dedup-semantics finding);
          fact: "C visited exactly K different rooms."
  cooc  : per-frame "Are C1 and C2 in the same room?" yes/no → SUM;
          fact: "C1 and C2 were in the same room in exactly K of the 8 frames."

N=8, verify ALL frames (no shortlist at this N; cost N+1 forwards). Reports exact-match/MAE of
(i) the raw reduction and (ii) the rendered answer, vs the frozen joint baseline measured in the
same run. Registered: rooms ≥ steps-level accuracy; cooc lower (look-again quality bottleneck).
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
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri

ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]
INT_RE = re.compile(r"-?\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rooms", "cooc"], required=True)
    ap.add_argument("--data_root", default="")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    droot = args.data_root or (f"data/mmred_{'rooms' if args.task == 'rooms' else 'cooc'}"
                               f"_balanced/seq_len_8/all_uniform")
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    yes_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("yes", "Yes")]
    no_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("no", "No")]
    room_first_ids = {r: tok.encode(r, add_special_tokens=False)[0] for r in ROOMS}

    def logit_read(frame, prompt, cand_ids):
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([frame], prompt))
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
        vals = {k: float(logits[i]) for k, i in cand_ids.items()}
        return max(vals, key=vals.get)

    def gen_answer(frames, prompt):
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt(frames, prompt))
        plen = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=5, pad_token_id=pad)
        dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0]
        m = INT_RE.search(dec)
        return int(m.group(0)) if m else None

    dirs = list(iter_sample_dirs(Path(droot)))
    random.Random(args.sample_seed).shuffle(dirs)
    rows = []
    n = 0
    for sd in dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            meta = json.loads((sd / "metadata.json").read_text())
        except Exception:
            continue
        NF = len(frames)
        if int(args.resize) > 0:
            frames = [f.resize((int(args.resize), int(args.resize))) for f in frames]
        try:
            if args.task == "rooms":
                C = meta.get("query_character") or meta.get("target_character")
                if not C:
                    continue
                per_rooms = []
                for fr in frames:
                    r = logit_read(fr, f"Look at this single frame.\nWhich room is {C} in "
                                       f"in this frame? Answer with the room name only.\nAnswer: ",
                                   room_first_ids)
                    per_rooms.append(r)
                tally = len(set(per_rooms))
                fact = f"Note: {C} visited exactly {tally} different rooms."
                per_detail = per_rooms
            else:
                pair = meta.get("query_pair")
                if not pair:
                    continue
                C1, C2 = pair
                votes = []
                for fr in frames:
                    v = logit_read(fr, f"Look at this single frame.\nAre {C1} and {C2} in the "
                                       f"same room in this frame? Answer yes or no.\nAnswer: ",
                                   {"yes": yes_ids[0], "no": no_ids[0]})
                    votes.append(int(v == "yes"))
                tally = int(sum(votes))
                fact = (f"Note: {C1} and {C2} were in the same room in exactly {tally} "
                        f"of the {NF} frames.")
                per_detail = votes
            # frozen joint baseline (same sample, no fact)
            base_prompt = (f"You will be shown {NF} frames describing steps in a house.\n"
                           f"Respond with a single integer from 0 to {NF} (0 is allowed). "
                           f"Output only the integer.\nQuestion: {q0}\nAnswer: ")
            base_pred = gen_answer(frames, base_prompt)
            # rendered answer (text-only, fact + question)
            rend_prompt = (f"{fact}\nRespond with a single integer. Output only the integer.\n"
                           f"Question: {q0}\nAnswer: ")
            rend_pred = gen_answer([], rend_prompt)
        except Exception as exc:
            print(f"{sid} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({"sid": sid, "gold": gold, "tally": tally, "rendered": rend_pred,
                     "frozen": base_pred, "detail": per_detail})
        n += 1
        if n % 25 == 0:
            acc_t = float(np.mean([r["tally"] == r["gold"] for r in rows]))
            print(f"  {n}: tally-exact {acc_t:.3f}", flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))

    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    g = np.array([r["gold"] for r in rows])
    t = np.array([r["tally"] for r in rows])
    rd = np.array([r["rendered"] if r["rendered"] is not None else -99 for r in rows])
    fz = np.array([r["frozen"] if r["frozen"] is not None else -99 for r in rows])
    lines = [f"=== E2E TASK ALGEBRA — {args.task} (N=8, verify-all, n={len(rows)}) ===",
             f"  tally (reduction) exact = {float((t == g).mean()):.3f}  MAE {float(np.abs(t-g).mean()):.2f}",
             f"  rendered answer  exact = {float((rd == g).mean()):.3f}",
             f"  frozen joint     exact = {float((fz == g).mean()):.3f}  MAE {float(np.abs(fz-g).mean()):.2f}",
             f"  forwards/sample = {len(rows[0]['detail']) + 2 if rows else 0}"]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
