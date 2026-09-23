#!/usr/bin/env python3
"""Single-frame influence at the answer ("Hahn" flip probe) on official MMReD rows, as a
function of N — DIAG cells D1 / D3 / D6 (docs/DIAGNOSTICS_2026-09-22.md §2).

Paired one-frame edits (experiments/_diag_common.build_pair): count qtype steps_in_room (gold
k -> k+1: the asked character moved INTO the asked room in one non-evidence frame; control = a
third room, k unchanged) and needle qtypes char_at_frame / n_char_at_frame (the asked step's
frame is edited so the answer changes; control = a comparable edit in another frame, answer
unchanged). Per pair: ||delta h|| at the answer row (`final`) and, under the fence, at the flipped
frame's <|vision_end|> slot (`slot_t`); first-token margins over the qtype's answer vocabulary;
measured floors (replay = same input twice, ctrl = the answer-preserving edit, perm = frame slots
permuted under the fence). Members are re-rendered with the official renderer; the base member's
frames are md5-compared with the stored renders as an environment canary.

Arms (layouts from core.prompt.build_messages, prompt text byte-identical across members):
  plain   paper layout (images then question), no fence           -> the deployed model
  qfirst  question-first (their --prefix_question), no fence       -> the method's prefix
  fenced  question-first + block fence + per-block position reset  -> loci final + slot_t
  gated   fenced + per-member oracle gate (count: base hides block t, flip exposes it; needle: the
          needle block is the only kept block in both members)
Knobs: --attn-sharpen TAU [--sharpen-from-layer L] (module.scaling *= TAU on decoder modules >= L;
0/1 = off), --attn-logn-sref S (scaling = base * ln(seq)/ln(S) per forward; 0 = off). Both are the
legacy S0/S10b/N3 knobs, eval-only.

Bands (pre-registered, outputs/diag/CAMPAIGN_BRIEF.md): plain/qfirst answer-row alpha in [0.5, 1.0]
for delta ~ N^-alpha; fenced slot |alpha| <= 0.05; replay floor exactly 0. No legacy cell ran on
official rows. Layer numbers follow the legacy convention (hidden_states[L] = output of decoder
module L-1); FenceHooks captures module L-1.
Outputs (run dir): pairs.csv, logits.csv, controls.csv, report.txt, config.json, audit_pair0/.
Usage:
  python experiments/probe_hahn.py --config seq_len_8 --qtypes steps_in_room --limit 30 \
      --arms plain qfirst fenced --output outputs/diag/hahn/steps_in_room/N8
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
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

from core.fence import FENCED_SDPA, FenceHooks, build_block_mask, hide_cols_for, layout_blocks, reset_positions, slot_positions  # noqa: E402
from core.mmred import as_sequence, load_split, render_sequence, stratified_order  # noqa: E402
from core.model import get_layers, get_rope_index_fn, load_runtime, special_ids  # noqa: E402
from experiments._diag_common import (ANSWER_PREFIX, FLIP_QTYPES, answer_vocab, build_pair, cond_tag,  # noqa: E402
                                      encode_prompt, gold_index, margin_of, set_logn, set_sharpen)
from experiments._port_check import PORT_CHECKS, legacy_p2_blocks, legacy_p2_messages, set_max_pixels  # noqa: E402

ARMS = ("plain", "qfirst", "fenced", "gated")


def render_member(states_, out_dir: Path) -> List[bytes]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    paths = render_sequence(as_sequence(states_), out_dir)
    return [p.read_bytes() for p in paths]


def assert_pair_identity(a: List[bytes], b: List[bytes], t: int, tag: str) -> None:
    for i, (x, y) in enumerate(zip(a, b)):
        same = hashlib.sha256(x).hexdigest() == hashlib.sha256(y).hexdigest()
        if i == t and same:
            raise AssertionError(f"[{tag}] edited frame {i} is byte-identical")
        if i != t and not same:
            raise AssertionError(f"[{tag}] non-edited frame {i} differs between members")


def pixel_md5(png_bytes: bytes) -> str:
    """md5 of the decoded RGB pixels: PNG bytes differ across matplotlib versions only in the
    'Software' text chunk (prepare_data --verify passes on pixel identity, 2026-09-21)."""
    import io
    return hashlib.md5(np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")).tobytes()).hexdigest()


def to_frames(blobs: List[bytes]) -> List[Image.Image]:
    import io
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/mmred_hf"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=["steps_in_room"], choices=FLIP_QTYPES)
    ap.add_argument("--limit", type=int, default=50, help="pairs per qtype")
    ap.add_argument("--controls", type=int, default=12, help="pairs per qtype that also run replay/ctrl/perm")
    ap.add_argument("--max-gold", type=int, default=None, help="count qtypes: skip golds above this (default: none)")
    ap.add_argument("--gold-set", nargs="*", type=int, default=None, help="count qtypes: explicit base golds")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=["plain", "qfirst", "fenced"])
    ap.add_argument("--hs-layers", nargs="+", type=int, default=[16, 20, 28], help="legacy convention (1-based tuple index)")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter (frozen) for every arm")
    ap.add_argument("--attn-sharpen", type=float, default=0.0, help="tau on decoder modules >= --sharpen-from-layer (0 = off)")
    ap.add_argument("--sharpen-from-layer", type=int, default=12)
    ap.add_argument("--attn-logn-sref", type=int, default=0, help="log-N logit scaling reference length in tokens (0 = off)")
    ap.add_argument("--max-seq-tokens", type=int, default=60000, help="skip fenced/gated arms above this prompt length")
    ap.add_argument("--port-check", choices=["p2"], default=None, help="fenced arm under the P2 legacy layout/prompt/392 px (steps_in_room only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.port_check and args.qtypes != ["steps_in_room"]:
        raise SystemExit("--port-check p2 is defined for steps_in_room only")
    out = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    out.mkdir(parents=True, exist_ok=True)
    cond = cond_tag(args.attn_sharpen, args.sharpen_from_layer, args.attn_logn_sref)
    (out / "config.json").write_text(json.dumps({**vars(args), "cond": cond}, indent=1, default=str))
    gold_set = set(args.gold_set) if args.gold_set else None
    pc = PORT_CHECKS["p2"] if args.port_check else None

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)                       # structural refs BEFORE the PEFT wrap
    rope_fn = get_rope_index_fn(model)
    sid = special_ids(processor)
    im_end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    prefix_ids = tok(ANSWER_PREFIX, add_special_tokens=False).input_ids
    base_scaling = set_sharpen(layers, args.attn_sharpen, args.sharpen_from_layer)
    if pc is not None:
        set_max_pixels(processor, pc.max_pixels)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
    capture_mods = [L - 1 for L in args.hs_layers]
    hooks = FenceHooks(layers, capture_layers=capture_mods).install()
    rng = random.Random(args.seed)
    vocab_cache: Dict[str, Any] = {}

    def encode(frames, layout, question):
        if pc is not None:
            from core.model import move_to_device
            enc = dict(processor.apply_chat_template(legacy_p2_messages(frames, question), add_generation_prompt=True,
                                                     tokenize=True, return_dict=True, return_tensors="pt"))
            return move_to_device(enc, rt.device)
        return encode_prompt(processor, frames, question, layout, prefix_ids, rt.device)

    def capture(inputs, loci: Dict[str, int], vocab_ids: List[int], mask=None, pos=None):
        cur = {k: v for k, v in inputs.items() if k != "attention_mask"} if mask is not None else dict(inputs)
        hooks.hidden.clear()
        set_logn(layers, int(inputs["input_ids"].shape[1]), args.attn_logn_sref, base_scaling)
        if mask is not None:
            hooks.set_mask(mask, rt.device)
        try:
            with torch.inference_mode(), (sdpa_kernel(FENCED_SDPA) if mask is not None else contextlib.nullcontext()):
                # a 4-D additive mask needs EFFICIENT/MATH; mask-free forwards keep the default (FLASH) backend
                outp = model(**cur, position_ids=pos.to(rt.device) if pos is not None else None, use_cache=False)
        finally:
            hooks.clear_mask()
        caps = {}
        for L, mod in zip(args.hs_layers, capture_mods):
            h = hooks.hidden[mod][0]
            for name, p in loci.items():
                caps[(L, name)] = h[p].float().cpu()
        lg = outp.logits[0, -1].float().cpu()
        vl = np.array([float(lg[t]) for t in vocab_ids])
        del outp
        return caps, vl

    def dnorms(a, b):
        return {k: (float(torch.linalg.norm(a[k] - b[k])), float(torch.linalg.norm(a[k]))) for k in a}

    pairs_f = open(out / "pairs.csv", "w", newline="")
    pairs_w = csv.writer(pairs_f)
    pairs_w.writerow(["qid", "qtype", "n_frames", "gold", "flip_t", "flip_kind", "arm", "layer", "locus", "dnorm", "base_norm"])
    log_f = open(out / "logits.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["qid", "qtype", "n_frames", "gold", "arm", "flip_kind", "margin_base", "margin_flip", "pred_base",
                    "pred_flip", "dlogit"])
    ctrl_f = open(out / "controls.csv", "w", newline="")
    ctrl_w = csv.writer(ctrl_f)
    ctrl_w.writerow(["qid", "qtype", "n_frames", "kind", "arm", "layer", "locus", "dnorm", "base_norm"])

    def write_margins(qid, qtype, NF, gold_lbl, arm, kind, vb, vf, gi_base, gi_flip):
        if gi_base is None or gi_flip is None:
            return
        mb, pb = margin_of(vb, gi_base)
        mf, pf = margin_of(vf, gi_flip)
        log_w.writerow([qid, qtype, NF, gold_lbl, arm, kind, f"{mb:.4f}", f"{mf:.4f}", pb, pf,
                        f"{float(np.linalg.norm(vb - vf)):.4f}"])

    rdir = Path(tempfile.mkdtemp(prefix="hahn_", dir=str(out)))
    totals = defaultdict(int)
    n_canary_same = n_canary = 0
    gold_hist: Dict[str, List[str]] = defaultdict(list)
    t0 = time.time()
    for qtype in args.qtypes:
        rows = stratified_order(load_split(args.config, args.split, args.data_root, [qtype]), args.seed)
        labels, vocab_ids = answer_vocab(qtype, tok)
        vocab_cache[qtype] = labels
        n_done = n_skip = 0
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
            question, flip_t, ctrl_t = pair["question"], pair["flip_t"], pair["ctrl_t"]
            gold_lbl, flip_lbl, kind = pair["gold"], pair["flip_gold"], pair["flip_kind"]
            gi_base, gi_flip = gold_index(qtype, gold_lbl, labels), gold_index(qtype, flip_lbl, labels)
            NF = len(pair["base_states"])
            do_controls = n_done < args.controls
            base_blobs = render_member(pair["base_states"], rdir / "base")
            flip_blobs = render_member(pair["flip_states"], rdir / "flip")
            assert_pair_identity(base_blobs, flip_blobs, flip_t, pair["qid"])
            ctrl_blobs = render_member(pair["ctrl_states"], rdir / "ctrl") if do_controls else None
            if ctrl_blobs is not None:
                assert_pair_identity(base_blobs, ctrl_blobs, ctrl_t, pair["qid"])
            stored = Path(args.data_root) / "images" / row["split_dir"] / str(row["qid"])
            for i, b in enumerate(base_blobs):
                p = stored / f"frame_{i + 1:04d}.png"
                if p.exists():
                    n_canary += 1
                    n_canary_same += pixel_md5(p.read_bytes()) == pixel_md5(b)   # pixels, not PNG bytes (Software chunk)
            if totals["pairs"] == 0:
                shutil.copytree(rdir, out / "audit_pair0", dirs_exist_ok=True)
            base_fr, flip_fr = to_frames(base_blobs), to_frames(flip_blobs)
            ctrl_fr = to_frames(ctrl_blobs) if ctrl_blobs else None
            skipped = False

            # ---------------- unfenced arms: plain (paper) / qfirst (question-first), no mask
            for arm, layout in (("plain", "paper"), ("qfirst", "question-first")):
                if arm not in args.arms:
                    continue
                in_b, in_f = encode(base_fr, layout, question), encode(flip_fr, layout, question)
                if not torch.equal(in_b["input_ids"], in_f["input_ids"]):
                    print(f"  [skip] {row['qid']}: {arm} layout differs", flush=True)
                    skipped = True
                    break
                seq = int(in_b["input_ids"].shape[1])
                loci = {"final": seq - 1}
                cap_b, vb = capture(in_b, loci, vocab_ids)
                cap_f, vf = capture(in_f, loci, vocab_ids)
                for (L, name), (dn, bn) in dnorms(cap_b, cap_f).items():
                    pairs_w.writerow([pair["qid"], qtype, NF, gold_lbl, flip_t, kind, arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                write_margins(pair["qid"], qtype, NF, gold_lbl, arm, kind, vb, vf, gi_base, gi_flip)
                if do_controls:
                    cap_b2, _ = capture(in_b, loci, vocab_ids)
                    for (L, name), (dn, bn) in dnorms(cap_b, cap_b2).items():
                        ctrl_w.writerow([pair["qid"], qtype, NF, "replay", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                    in_c = encode(ctrl_fr, layout, question)
                    if torch.equal(in_c["input_ids"], in_b["input_ids"]):
                        cap_c, vc = capture(in_c, loci, vocab_ids)
                        for (L, name), (dn, bn) in dnorms(cap_b, cap_c).items():
                            ctrl_w.writerow([pair["qid"], qtype, NF, "ctrl", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                        write_margins(pair["qid"], qtype, NF, gold_lbl, arm, "ctrl", vb, vc, gi_base, gi_base)
            if skipped:
                n_skip += 1
                continue

            # ---------------- fenced / gated arms: question-first + fence + posreset
            if {"fenced", "gated"} & set(args.arms):
                layout = "question-first"
                fin_b, fin_f = encode(base_fr, layout, question), encode(flip_fr, layout, question)
                ids_b = fin_b["input_ids"][0].cpu()
                if not torch.equal(ids_b, fin_f["input_ids"][0].cpu()):
                    print(f"  [skip] {row['qid']}: fenced layout differs", flush=True)
                    n_skip += 1
                    continue
                if int(ids_b.shape[0]) > args.max_seq_tokens:
                    totals["too_long"] += 1
                else:
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
                    for arm in [a for a in ("fenced", "gated") if a in args.arms]:
                        if arm == "fenced":
                            m_b = m_f = m_open
                        else:
                            evid_b = set(pair["evid"])
                            evid_f = evid_b | {flip_t}            # count: exposes t; needle: t == X already in
                            m_b = build_block_mask(seq, blocks, hide_cols_for(blocks, [t in evid_b for t in range(NF)]))
                            m_f = build_block_mask(seq, blocks, hide_cols_for(blocks, [t in evid_f for t in range(NF)]))
                        cap_b, vb = capture(fin_b, loci, vocab_ids, m_b, pos)
                        cap_f, vf = capture(fin_f, loci, vocab_ids, m_f, pos)
                        for (L, name), (dn, bn) in dnorms(cap_b, cap_f).items():
                            pairs_w.writerow([pair["qid"], qtype, NF, gold_lbl, flip_t, kind, arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                        write_margins(pair["qid"], qtype, NF, gold_lbl, arm, kind, vb, vf, gi_base, gi_flip)
                        if do_controls:
                            cap_b2, _ = capture(fin_b, loci, vocab_ids, m_b, pos)
                            for (L, name), (dn, bn) in dnorms(cap_b, cap_b2).items():
                                ctrl_w.writerow([pair["qid"], qtype, NF, "replay", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                            fin_c = encode(ctrl_fr, layout, question)
                            if torch.equal(fin_c["input_ids"][0].cpu(), ids_b):
                                cap_c, vc = capture(fin_c, loci, vocab_ids, m_b, pos)   # answer-preserving: base's gate
                                for (L, name), (dn, bn) in dnorms(cap_b, cap_c).items():
                                    ctrl_w.writerow([pair["qid"], qtype, NF, "ctrl", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
                                write_margins(pair["qid"], qtype, NF, gold_lbl, arm, "ctrl", vb, vc, gi_base, gi_base)
                            if arm == "fenced":
                                order = [i for i in range(NF) if i != flip_t]
                                rng.shuffle(order)
                                perm = list(range(NF))
                                j = 0
                                for i in range(NF):
                                    if i != flip_t:
                                        perm[i] = order[j]
                                        j += 1
                                fin_p = encode([base_fr[perm[i]] for i in range(NF)], layout, question)
                                if torch.equal(fin_p["input_ids"][0].cpu(), ids_b):
                                    cap_p, _ = capture(fin_p, loci, vocab_ids, m_open, pos)
                                    for (L, name), (dn, bn) in dnorms(cap_b, cap_p).items():
                                        ctrl_w.writerow([pair["qid"], qtype, NF, "perm", arm, L, name, f"{dn:.6g}", f"{bn:.6g}"])
            gold_hist[qtype].append(gold_lbl)
            n_done += 1
            totals["pairs"] += 1
            pairs_f.flush(); log_f.flush(); ctrl_f.flush()
            if n_done % 5 == 0:
                print(f"  {qtype}: {n_done}/{args.limit} pairs (skip {n_skip}) {time.time() - t0:.0f}s", flush=True)
        totals[f"skip_{qtype}"] = n_skip
        totals[f"pairs_{qtype}"] = n_done

    hooks.remove()
    pairs_f.close(); log_f.close(); ctrl_f.close()
    shutil.rmtree(rdir, ignore_errors=True)
    lines = [f"=== HAHN PROBE (official rows; {args.config}_{args.split}; qtypes={args.qtypes}; arms={args.arms}; "
             f"cond={cond}; adapter={args.adapter or 'none'}; port_check={pc.name if pc else None}; "
             f"hs_layers={args.hs_layers}; pairs={totals['pairs']}; too_long={totals['too_long']}) ==="]
    for qt in args.qtypes:
        hist = defaultdict(int)
        for g in gold_hist[qt]:
            hist[g] += 1
        lines.append(f"[{qt}] pairs {totals[f'pairs_{qt}']} skip {totals[f'skip_{qt}']} gold-hist " +
                     " ".join(f"{g}:{c}" for g, c in sorted(hist.items())))
    lines.append(f"[render canary] base frames pixel-identical to stored renders: {n_canary_same}/{n_canary}")
    acc: Dict[tuple, list] = defaultdict(list)
    with open(out / "pairs.csv") as fh:
        for r in csv.DictReader(fh):
            acc[(r["qtype"], r["arm"], r["layer"], r["locus"])].append(float(r["dnorm"]))
    for k in sorted(acc):
        v = np.array(acc[k])
        lines.append(f"  {k[0]:16s} {k[1]:7s} L{k[2]:>2s} {k[3]:7s} median dnorm {np.median(v):.4g}  "
                     f"(iqr {np.percentile(v, 25):.4g}-{np.percentile(v, 75):.4g}, n={len(v)})")
    floors: Dict[tuple, list] = defaultdict(list)
    with open(out / "controls.csv") as fh:
        for r in csv.DictReader(fh):
            floors[(r["qtype"], r["kind"], r["arm"], r["locus"])].append(float(r["dnorm"]))
    for k in sorted(floors):
        v = np.array(floors[k])
        lines.append(f"  floor {k[0]:16s} {k[1]:6s} {k[2]:7s} {k[3]:7s} median {np.median(v):.4g} max {v.max():.4g} n={len(v)}")
    marg: Dict[tuple, list] = defaultdict(list)
    with open(out / "logits.csv") as fh:
        for r in csv.DictReader(fh):
            if r["flip_kind"] != "ctrl":
                marg[(r["qtype"], r["arm"])].append((float(r["margin_base"]), r["pred_base"], r["pred_flip"]))
    for k in sorted(marg):
        v = marg[k]
        mb = np.median([x[0] for x in v])
        labels = vocab_cache[k[0]]
        gi = [gold_index(k[0], g, labels) for g in gold_hist[k[0]]]
        lines.append(f"  margin {k[0]:16s} {k[1]:7s} median base {mb:+.3f}  flip-changes-pred "
                     f"{np.mean([x[1] != x[2] for x in v]):.2f}  n={len(v)}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
