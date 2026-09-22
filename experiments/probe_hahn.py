#!/usr/bin/env python3
"""The Hahn probe on official rows: how much does ONE frame's flip move the read, as a
function of N? Paired one-frame flips on steps_in_room rows (gold k -> k+1: the character is
moved INTO the asked room in one non-evidence frame; control: moved to a third room, k
unchanged), ||delta h|| at the answer position and at the flipped frame's <|vision_end|> slot,
digit-logit margins, and measured noise floors (replay, control, permutation).

Twin: legacy/v1/scripts/sparse/probe_hahn_gated.py (== armor/probe_hahn.py with no flags:
--gate none, arms plain/repjoint/fenced, layers 16/20/28, 392 px). No legacy cell ever ran on
official rows (every hahn/fixedk config used park or redux data), so there is no number to
reproduce; the anchor is the pre-registered band of outputs/armor/CAMPAIGN_BRIEF.md: joint
(plain) slope alpha in [0.7, 1.3] for delta ~ N^-alpha, fenced per-frame |alpha| < 0.2, replay
floor exactly 0. Members are re-rendered with the official renderer; the base member's frames
are md5-compared with the stored renders as an environment canary.

Arms: plain = paper layout, no fence, locus final; fenced = question-first + fence + posreset,
loci final + slot_t; gated = fenced + per-member oracle gate (base hides block t, evid member
exposes it: the flip as seen THROUGH the gate). Layer numbers follow the legacy convention
(hidden_states[L] = output of decoder module L-1); FenceHooks captures module L-1.
Outputs: pairs.csv, logits.csv, controls.csv, report.txt, config.json, audit_pair0/.
Usage:
  python experiments/probe_hahn.py --config seq_len_8 --limit 30 --arms plain fenced gated --output outputs/port/hahn_N8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.constants import ROOMS  # noqa: E402
from core.fence import FENCED_SDPA, FenceHooks, build_block_mask, hide_cols_for, layout_blocks, reset_positions, slot_positions  # noqa: E402
from core.mmred import as_sequence, evidence_frames, load_split, parse_question, recompute_answer, render_sequence, states, stratified_order, with_char_moved  # noqa: E402
from core.model import get_layers, get_rope_index_fn, load_runtime, move_to_device, special_ids  # noqa: E402
from core.prompt import build_messages  # noqa: E402
from experiments._port_check import PORT_CHECKS, legacy_p2_blocks, legacy_p2_messages, set_max_pixels  # noqa: E402

ANSWER_PREFIX = '{ "answer": "'      # teacher-forced assistant prefix; the next token is the digit


def _char_room(st, char):
    for r, occ in st["rooms"].items():
        if char in occ:
            return r
    return None


def build_pair(row, rng: random.Random, max_gold: int, gold_set) -> Optional[Dict[str, Any]]:
    st = states(row)
    gold = int(row["answer"])
    if gold_set is not None:
        if gold not in gold_set:
            return None
    elif gold > max_gold:
        return None
    char, room = parse_question("steps_in_room", row["question"])
    evid = evidence_frames("steps_in_room", row["question"], st)
    if evid is None or len(evid) != gold or recompute_answer("steps_in_room", row["question"], st) != str(gold):
        raise AssertionError(f"evidence/gold mismatch in {row['qid']}: {None if evid is None else len(evid)} vs {gold}")
    cands = [t for t in range(len(st)) if t not in evid]
    if not cands:
        return None
    flip_t = int(rng.choice(cands))
    cur = _char_room(st[flip_t], char)
    ctrl_room = str(rng.choice([r for r in ROOMS if r not in (cur, room)]))
    evid_states = list(st)
    evid_states[flip_t] = with_char_moved(st[flip_t], char, room)
    ctrl_states = list(st)
    ctrl_states[flip_t] = with_char_moved(st[flip_t], char, ctrl_room)

    def count(sts):
        return sum(1 for s in sts if char in s["rooms"].get(room, []))

    assert count(st) == gold and count(evid_states) == gold + 1 and count(ctrl_states) == gold, row["qid"]
    return dict(qid=row["qid"], question=row["question"], char=char, room=room, gold=gold, flip_t=flip_t,
                evid=set(evid), base_states=st, evid_states=evid_states, ctrl_states=ctrl_states)


def render_member(states_, out_dir: Path) -> List[bytes]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    paths = render_sequence(as_sequence(states_), out_dir)
    return [p.read_bytes() for p in paths]


def assert_pair_identity(a: List[bytes], b: List[bytes], flip_t: int, tag: str) -> None:
    for i, (x, y) in enumerate(zip(a, b)):
        same = hashlib.sha256(x).hexdigest() == hashlib.sha256(y).hexdigest()
        if i == flip_t and same:
            raise AssertionError(f"[{tag}] flipped frame {i} is byte-identical")
        if i != flip_t and not same:
            raise AssertionError(f"[{tag}] non-flipped frame {i} differs between members")


def to_frames(blobs: List[bytes]) -> List[Image.Image]:
    import io
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30, help="pairs to measure")
    ap.add_argument("--controls", type=int, default=12, help="pairs that also run replay/ctrl/perm controls")
    ap.add_argument("--max-gold", type=int, default=8, help="gold <= 8 so k+1 stays a single digit")
    ap.add_argument("--gold-set", nargs="*", type=int, default=None, help="explicit base golds (overrides --max-gold)")
    ap.add_argument("--arms", nargs="+", choices=["plain", "fenced", "gated"], default=["plain", "fenced", "gated"])
    ap.add_argument("--hs-layers", nargs="+", type=int, default=[16, 20, 28], help="legacy convention (1-based tuple index)")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter for the fenced/gated arms (frozen)")
    ap.add_argument("--port-check", choices=["p2"], default=None, help="fenced arm under the P2 legacy layout/prompt/392 px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    gold_set = set(args.gold_set) if args.gold_set else None
    pc = PORT_CHECKS["p2"] if args.port_check else None

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    sid = special_ids(processor)
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    prefix_ids = tok(ANSWER_PREFIX, add_special_tokens=False).input_ids
    if pc is not None:
        set_max_pixels(processor, pc.max_pixels)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
    capture_mods = [L - 1 for L in args.hs_layers]
    hooks = FenceHooks(layers, capture_layers=capture_mods).install()
    rng = random.Random(args.seed)

    def encode(frames, layout):
        if pc is not None:
            msgs = legacy_p2_messages(frames, question)
        else:
            msgs = build_messages(frames, question, layout=layout)
        enc = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        enc = dict(enc)
        if pc is None:                                   # teacher-force the JSON prefix; next token = the digit
            pre = torch.tensor([prefix_ids], dtype=enc["input_ids"].dtype)
            enc["input_ids"] = torch.cat([enc["input_ids"], pre], dim=1)
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return move_to_device(enc, rt.device)

    def capture(inputs, loci: Dict[str, int], mask=None, pos=None):
        cur = {k: v for k, v in inputs.items() if k != "attention_mask"} if mask is not None else dict(inputs)
        hooks.hidden.clear()
        if mask is not None:
            hooks.set_mask(mask, rt.device)
        try:
            with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
                outp = model(**cur, position_ids=pos.to(rt.device) if pos is not None else None, use_cache=False)
        finally:
            hooks.clear_mask()
        caps = {}
        for L, mod in zip(args.hs_layers, capture_mods):
            h = hooks.hidden[mod][0]
            for name, p in loci.items():
                caps[(L, name)] = h[p].float().cpu()
        lg = outp.logits[0, -1].float().cpu()
        digits = np.array([float(lg[t]) for t in digit_ids])
        del outp
        return caps, digits

    def dnorms(a, b):
        return {k: (float(torch.linalg.norm(a[k] - b[k])), float(torch.linalg.norm(a[k]))) for k in a}

    def margin_of(digits, gold):
        other = np.delete(digits, gold)
        return float(digits[gold] - other.max()), int(np.argmax(digits))

    pairs_f = open(out / "pairs.csv", "w", newline="")
    pairs_w = csv.writer(pairs_f)
    pairs_w.writerow(["qid", "n_frames", "gold", "flip_t", "flip_kind", "arm", "layer", "locus", "dnorm", "base_norm"])
    log_f = open(out / "logits.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["qid", "n_frames", "gold", "arm", "flip_kind", "margin_base", "margin_flip", "pred_base", "pred_flip",
                    "dlogit_digits"])
    ctrl_f = open(out / "controls.csv", "w", newline="")
    ctrl_w = csv.writer(ctrl_f)
    ctrl_w.writerow(["qid", "n_frames", "kind", "arm", "layer", "locus", "dnorm", "base_norm"])

    rows = stratified_order(load_split(args.config, args.split, args.data_root, ["steps_in_room"]), args.seed)
    rdir = Path(tempfile.mkdtemp(prefix="hahn_", dir=str(out)))
    n_done = n_skip = n_canary_same = n_canary = 0
    gold_hist: List[int] = []
    t0 = time.time()
    for row in rows:
        if n_done >= args.limit:
            break
        try:
            pair = build_pair(row, rng, args.max_gold, gold_set)
        except Exception as exc:
            print(f"  [skip] {row['qid']}: {exc}", flush=True)
            n_skip += 1
            continue
        if pair is None:
            n_skip += 1
            continue
        question, room, flip_t, gold = pair["question"], pair["room"], pair["flip_t"], pair["gold"]
        NF = len(pair["base_states"])
        do_controls = n_done < args.controls
        base_blobs = render_member(pair["base_states"], rdir / "base")
        evid_blobs = render_member(pair["evid_states"], rdir / "evid")
        assert_pair_identity(base_blobs, evid_blobs, flip_t, pair["qid"])
        ctrl_blobs = render_member(pair["ctrl_states"], rdir / "ctrl") if do_controls else None
        if ctrl_blobs is not None:
            assert_pair_identity(base_blobs, ctrl_blobs, flip_t, pair["qid"])
        # environment canary: fresh base render vs the stored official frames
        stored = Path(args.data_root) / "images" / row["split_dir"] / row["qid"]
        for i, b in enumerate(base_blobs):
            p = stored / f"frame_{i + 1:04d}.png"
            if p.exists():
                n_canary += 1
                n_canary_same += hashlib.md5(p.read_bytes()).hexdigest() == hashlib.md5(b).hexdigest()
        if n_done == 0:
            shutil.copytree(rdir, out / "audit_pair0", dirs_exist_ok=True)
        base_fr, evid_fr = to_frames(base_blobs), to_frames(evid_blobs)
        ctrl_fr = to_frames(ctrl_blobs) if ctrl_blobs else None

        # ---------------- plain arm: paper layout, no fence
        if "plain" in args.arms:
            in_b, in_e = encode(base_fr, "paper"), encode(evid_fr, "paper")
            if not torch.equal(in_b["input_ids"], in_e["input_ids"]):
                print(f"  [skip] {row['qid']}: plain layout differs", flush=True)
                n_skip += 1
                continue
            seq = int(in_b["input_ids"].shape[1])
            loci = {"final": seq - 1}
            cap_b, dig_b = capture(in_b, loci)
            cap_e, dig_e = capture(in_e, loci)
            for (L, name), (dn, bn) in dnorms(cap_b, cap_e).items():
                pairs_w.writerow([pair["qid"], NF, gold, flip_t, "evid", "plain", L, name, f"{dn:.6g}", f"{bn:.6g}"])
            if gold + 1 <= 9:
                mb, pb = margin_of(dig_b, gold)
                me, pe = margin_of(dig_e, gold + 1)
                log_w.writerow([pair["qid"], NF, gold, "plain", "evid", f"{mb:.4f}", f"{me:.4f}", pb, pe,
                                f"{float(np.linalg.norm(dig_b - dig_e)):.4f}"])
            if do_controls:
                cap_b2, _ = capture(in_b, loci)
                for (L, name), (dn, bn) in dnorms(cap_b, cap_b2).items():
                    ctrl_w.writerow([pair["qid"], NF, "replay", "plain", L, name, f"{dn:.6g}", f"{bn:.6g}"])
                in_c = encode(ctrl_fr, "paper")
                cap_c, dig_c = capture(in_c, loci)
                for (L, name), (dn, bn) in dnorms(cap_b, cap_c).items():
                    ctrl_w.writerow([pair["qid"], NF, "ctrl", "plain", L, name, f"{dn:.6g}", f"{bn:.6g}"])
                if gold + 1 <= 9:
                    mc, pcn = margin_of(dig_c, gold)
                    log_w.writerow([pair["qid"], NF, gold, "plain", "ctrl", f"{mb:.4f}", f"{mc:.4f}", pb, pcn,
                                    f"{float(np.linalg.norm(dig_b - dig_c)):.4f}"])

        # ---------------- fenced / gated arms: question-first + fence + posreset
        if {"fenced", "gated"} & set(args.arms):
            layout = "question-first"
            fin_b, fin_e = encode(base_fr, layout), encode(evid_fr, layout)
            ids_b = fin_b["input_ids"][0].cpu()
            if not torch.equal(ids_b, fin_e["input_ids"][0].cpu()):
                print(f"  [skip] {row['qid']}: fenced layout differs", flush=True)
                n_skip += 1
                continue
            if pc is not None:
                parsed = legacy_p2_blocks(ids_b.tolist(), tok, NF, sid["vision_start"])
                if parsed is None:
                    n_skip += 1
                    continue
                blocks, fin = parsed
            else:
                blocks, fin = layout_blocks(ids_b, layout, sid, im_end_id=im_end_id)
            if len(blocks) != NF:
                n_skip += 1
                continue
            seq = int(ids_b.shape[0])
            slots = slot_positions(ids_b, sid["vision_end"])
            loci = {"final": seq - 1, "slot_t": slots[flip_t]}
            with torch.inference_mode():
                base_pos, _ = rope_fn(fin_b["input_ids"], image_grid_thw=fin_b.get("image_grid_thw"),
                                      attention_mask=fin_b.get("attention_mask"))
            pos = reset_positions(base_pos, blocks, fin)
            m_open = build_block_mask(seq, blocks, hide_cols=[])
            arms_here = [a for a in ("fenced", "gated") if a in args.arms]
            for arm in arms_here:
                if arm == "fenced":
                    m_b = m_e = m_open
                else:
                    keep_b = [t in pair["evid"] for t in range(NF)]
                    keep_e = [t in (pair["evid"] | {flip_t}) for t in range(NF)]
                    m_b = build_block_mask(seq, blocks, hide_cols_for(blocks, keep_b))
                    m_e = build_block_mask(seq, blocks, hide_cols_for(blocks, keep_e))
                cap_b, dig_b = capture(fin_b, loci, m_b, pos)
                cap_e, dig_e = capture(fin_e, loci, m_e, pos)
                for (L, name), (dn, bn) in dnorms(cap_b, cap_e).items():
                    pairs_w.writerow([pair["qid"], NF, gold, flip_t, "evid", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                if gold + 1 <= 9:
                    mb, pb = margin_of(dig_b, gold)
                    me, pe = margin_of(dig_e, gold + 1)
                    log_w.writerow([pair["qid"], NF, gold, arm, "evid", f"{mb:.4f}", f"{me:.4f}", pb, pe,
                                    f"{float(np.linalg.norm(dig_b - dig_e)):.4f}"])
                if do_controls:
                    cap_b2, _ = capture(fin_b, loci, m_b, pos)
                    for (L, name), (dn, bn) in dnorms(cap_b, cap_b2).items():
                        ctrl_w.writerow([pair["qid"], NF, "replay", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                    fin_c = encode(ctrl_fr, layout)
                    if torch.equal(fin_c["input_ids"][0].cpu(), ids_b):
                        cap_c, _ = capture(fin_c, loci, m_b, pos)   # answer-preserving: base's gate
                        for (L, name), (dn, bn) in dnorms(cap_b, cap_c).items():
                            ctrl_w.writerow([pair["qid"], NF, "ctrl", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                    if arm == "fenced":
                        order = [i for i in range(NF) if i != flip_t]
                        rng.shuffle(order)
                        perm = list(range(NF))
                        j = 0
                        for i in range(NF):
                            if i != flip_t:
                                perm[i] = order[j]
                                j += 1
                        fin_p = encode([base_fr[perm[i]] for i in range(NF)], layout)
                        if torch.equal(fin_p["input_ids"][0].cpu(), ids_b):
                            cap_p, _ = capture(fin_p, loci, m_open, pos)
                            for (L, name), (dn, bn) in dnorms(cap_b, cap_p).items():
                                ctrl_w.writerow([pair["qid"], NF, "perm", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
        gold_hist.append(gold)
        n_done += 1
        pairs_f.flush(); log_f.flush(); ctrl_f.flush()
        if n_done % 5 == 0:
            print(f"  {n_done}/{args.limit} pairs (skip {n_skip}) {time.time() - t0:.0f}s", flush=True)

    hooks.remove()
    pairs_f.close(); log_f.close(); ctrl_f.close()
    shutil.rmtree(rdir, ignore_errors=True)
    hist = " ".join(f"g{g}:{c}" for g, c in sorted(defaultdict(int, {g: gold_hist.count(g) for g in set(gold_hist)}).items()))
    lines = [f"=== HAHN PROBE (official rows; pairs={n_done}, skip={n_skip}, {args.config}_{args.split}, "
             f"max_gold={args.max_gold}, gold_set={sorted(gold_set) if gold_set else 'none'}, controls={min(args.controls, n_done)}, "
             f"hs_layers={args.hs_layers}, arms={args.arms}, adapter={args.adapter or 'none'}, "
             f"port_check={pc.name if pc else None}) ===",
             f"[gold-hist] {hist}",
             f"[render canary] base frames md5-identical to stored renders: {n_canary_same}/{n_canary}"]
    acc: Dict[tuple, list] = defaultdict(list)
    with open(out / "pairs.csv") as fh:
        for r in csv.DictReader(fh):
            acc[(r["arm"], r["layer"], r["locus"])].append(float(r["dnorm"]))
    for k in sorted(acc):
        v = np.array(acc[k])
        lines.append(f"  {k[0]:7s} L{k[1]:>2s} {k[2]:7s} median dnorm {np.median(v):.4g}  (iqr {np.percentile(v, 25):.4g}"
                     f"-{np.percentile(v, 75):.4g}, n={len(v)})")
    floors: Dict[tuple, list] = defaultdict(list)
    with open(out / "controls.csv") as fh:
        for r in csv.DictReader(fh):
            floors[(r["kind"], r["arm"], r["locus"])].append(float(r["dnorm"]))
    for k in sorted(floors):
        v = np.array(floors[k])
        lines.append(f"  floor {k[0]:6s} {k[1]:7s} {k[2]:7s} median {np.median(v):.4g} max {v.max():.4g} n={len(v)}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
