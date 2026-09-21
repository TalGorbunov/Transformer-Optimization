#!/usr/bin/env python3
"""SPARSE S3 — the Hahn probe + the evidence gate (copy-extension of
scripts/armor/probe_hahn.py per the brief §5; the armor original is untouched and
stays the anchor — NO-FLAG BEHAVIOR IS BYTE-IDENTICAL, verified against the
n2_hahn/p1fence_ep10_N8 first-3 pair dnorms before any full run).

SPARSE delta: --gate oracle (p1fence arm only). Each pair member gets the mask of
ITS OWN evidence set: the base member (gold k, flip frame t NON-evidence) hides
block t from the tail; the evid member (gold k+1, frame t evidence) exposes it.
That IS the point: the measured dnorm at the read locus is the flip as seen
THROUGH the gate. rep_t (verdict locus, own block) is mask-invariant by
construction. Controls: replay + ctrl flip both use the base member's gate (the
ctrl flip is answer-preserving, so its evidence set == base's).

All other arms/flags (plain/repjoint/fenced, --gold-set, --attn-logn-sref,
--peft-adapter) are inherited unchanged from the armor instrument.

Usage (S3 vs-N cell):
  python scripts/sparse/probe_hahn_gated.py --data_root <pool> --limit 50 \
      --arms p1fence --peft-adapter <P1g> --gate oracle --output outputs/sparse/s3/N32
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_RENDER_DIR = _REPO / "datasets" / "mmred"
if str(_RENDER_DIR) not in sys.path:
    sys.path.insert(0, str(_RENDER_DIR))

import render_mmred  # noqa: E402

from gnnformer.constants import ANCHOR_OFFSET, FRAME_RESIZE, ROOMS  # noqa: E402
from gnnformer.data import (  # noqa: E402
    build_count_prompt,
    build_prompt_inputs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_target_character_room,
    probe_evidence,
    rooms_to_room2chars,
)
from gnnformer.constants import MASK_MIN
from gnnformer.fencing import (  # noqa: E402
    FenceHooks,
    build_block_mask,
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    locate_word_token,
    reset_positions,
)

_LORAMECH = _REPO / "scripts" / "loramech"
if str(_LORAMECH) not in sys.path:
    sys.path.insert(0, str(_LORAMECH))
from train_sft_fenced import build_fenced_messages, parse_layout  # noqa: E402
from gnnformer.metrics import format_gold_histogram  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

PARK_ROOM_ORDER = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]


# ------------------------------------------------------------------ pair construction

def char_room_of(state, character):
    r2c = rooms_to_room2chars(state.get("rooms", {}))
    for room, chars in r2c.items():
        if character in chars:
            return room
    return None


def with_char_moved(state, character, new_room):
    """New state dict with `character` moved to `new_room` (removed from wherever)."""
    r2c = rooms_to_room2chars(state.get("rooms", {}))
    new = {room: [c for c in chars if c != character] for room, chars in r2c.items()}
    new.setdefault(new_room, [])
    new[new_room] = sorted(set(new[new_room] + [character]))
    out = dict(state)
    out["rooms"] = new
    return out


def render_states(states, room_order, out_dir):
    """Render every frame deterministically; -> list of PNG byte strings."""
    render_mmred.ROOMS = list(room_order)
    out_dir.mkdir(parents=True, exist_ok=True)
    blobs = []
    for i, st in enumerate(states):
        p = out_dir / f"{i:03d}.png"
        render_mmred.render_frame(
            rooms_to_room2chars(st.get("rooms", {})), int(st.get("step_id", i + 1)), str(p)
        )
        blobs.append(p.read_bytes())
    return blobs


def blobs_to_frames(blobs, resize):
    frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
    if resize > 0:
        frames = [f.resize((resize, resize)) for f in frames]
    return frames


def assert_pair_identity(base_blobs, flip_blobs, flip_t, tag):
    for i, (a, b) in enumerate(zip(base_blobs, flip_blobs)):
        ha, hb = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
        if i == flip_t:
            if ha == hb:
                raise AssertionError(f"[{tag}] flipped frame {i} is byte-identical")
        elif ha != hb:
            raise AssertionError(f"[{tag}] non-flipped frame {i} differs between members")


def build_pair(sd, rng, max_gold, gold_set=None):
    """-> dict with q0, room, char, gold, flip_t, evid, base/evid/ctrl state lists — or None."""
    _sid, _frames, q0, states, a0 = load_mmred_sample(sd)
    gold = int(str(a0).strip())
    if gold_set is not None:
        if gold not in gold_set:
            return None
    elif gold > max_gold:
        return None
    pe = probe_evidence("steps", q0, states, gold, ROOMS)
    if pe is None:
        return None
    evid, room = pe
    pr = parse_target_character_room(q0)
    if pr is None:
        return None
    char = pr[0]
    cands = [
        t for t in range(len(states))
        if t not in evid and char_room_of(states[t], char) not in (None, room)
    ]
    if not cands:
        return None
    flip_t = int(rng.choice(cands))
    cur = char_room_of(states[flip_t], char)
    wrong = [r for r in PARK_ROOM_ORDER if r not in (cur, room)]
    ctrl_room = str(rng.choice(wrong))
    base_states = list(states)
    evid_states = list(states)
    evid_states[flip_t] = with_char_moved(states[flip_t], char, room)
    ctrl_states = list(states)
    ctrl_states[flip_t] = with_char_moved(states[flip_t], char, ctrl_room)

    def count(sts):
        return sum(1 for st in sts if char in rooms_to_room2chars(st.get("rooms", {})).get(room, []))

    assert count(base_states) == gold, f"base gold mismatch in {sd.name}"
    assert count(evid_states) == gold + 1, f"evid-flip gold mismatch in {sd.name}"
    assert count(ctrl_states) == gold, f"ctrl-flip gold mismatch in {sd.name}"
    if len(evid) != gold:  # SPARSE gate contract: |evid| == k, skip+count on mismatch
        raise AssertionError(f"evidence/gold mismatch in {sd.name}: {len(evid)} vs {gold}")
    return dict(
        sample_id=sd.name, q0=q0, room=room, char=char, gold=gold, flip_t=flip_t,
        evid=set(evid),
        base_states=base_states, evid_states=evid_states, ctrl_states=ctrl_states,
    )


# ------------------------------------------------------------------------ forwards

def capture(model, inputs, hs_layers, loci, pos_ids=None, digit_ids=None):
    """One forward. -> ({(layer, locus_name): fp32 vec}, digit_logit_vec or None)."""
    with torch.inference_mode():
        out = model(**inputs, position_ids=pos_ids, output_hidden_states=True,
                    use_cache=False) if pos_ids is not None else model(
            **inputs, output_hidden_states=True, use_cache=False)
    caps = {}
    for L in hs_layers:
        h = out.hidden_states[L][0]
        for name, p in loci.items():
            caps[(L, name)] = h[p].float().cpu()
    digits = None
    if digit_ids is not None:
        lg = out.logits[0, -1].float().cpu()
        digits = np.array([float(lg[t]) for t in digit_ids])
    del out
    return caps, digits


def dnorms(caps_a, caps_b):
    return {k: (float(torch.linalg.norm(caps_a[k] - caps_b[k])),
                float(torch.linalg.norm(caps_a[k]))) for k in caps_a}


def margin_of(digits, gold):
    other = np.delete(digits, gold)
    return float(digits[gold] - other.max()), int(np.argmax(digits))


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--limit", type=int, default=50, help="pairs to measure")
    ap.add_argument("--controls", type=int, default=12,
                    help="pairs that also run replay/ctrl/perm controls")
    ap.add_argument("--max-gold", type=int, default=8,
                    help="single-digit slice: gold<=8 so flip answer k+1 stays one digit")
    ap.add_argument("--gold-set", default="",
                    help="explicit comma-set of base golds to accept (OVERRIDES "
                         "--max-gold). Digit margins are skipped for gold>8.")
    ap.add_argument("--attn-logn-sref", type=int, default=0,
                    help=">0 = log-N attention-logit scaling per forward (LM decoder "
                         "attn only). No flag = byte-identical behavior.")
    ap.add_argument("--gate", choices=["none", "oracle"], default="none",
                    help="SPARSE S3: oracle = per-member evidence gate on the p1fence "
                         "arm (base hides flip block t, evid member exposes it). "
                         "No flag = byte-identical behavior.")
    ap.add_argument("--gate-bonus", type=float, default=0.0,
                    help="SOFTGATE: finite penalty instead of MASK_MIN on the gated (non-evidence) "
                         "columns: -log(B) added to their logits (B=1 -> no penalty; 0 = hard gate, "
                         "byte-identical path). Requires --gate oracle.")
    ap.add_argument("--nfree-prompt", action="store_true",
                    help="S11: build the p1fence layout with the S9 N-free prompt "
                         "(single source: the sparse trainer's builder). Text is then "
                         "byte-identical across ALL N. No flag = byte-identical "
                         "legacy behavior.")
    ap.add_argument("--hs-layers", default="16,20,28")
    ap.add_argument("--resize", type=int, default=FRAME_RESIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--peft-adapter", type=Path, default=None,
                    help="restore a saved LoRA adapter dir (frozen) before probing")
    ap.add_argument("--arms", default="plain,repjoint,fenced",
                    help="comma-set of arms to run (p1fence = the trained-fenced layout)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    if not arms <= {"plain", "repjoint", "fenced", "p1fence"}:
        raise SystemExit(f"unknown arm in --arms: {arms}")
    if args.gate != "none" and arms != {"p1fence"}:
        raise SystemExit("--gate oracle is defined for --arms p1fence only")
    if args.gate_bonus > 0 and args.gate != "oracle":
        raise SystemExit("--gate-bonus requires --gate oracle")

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    hs_layers = [int(x) for x in args.hs_layers.split(",")]
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    gold_set = ({int(x) for x in args.gold_set.replace(",", " ").split()}
                if args.gold_set.strip() else None)
    # logN hook — attn modules captured PRE-wrap (PeftModel.model trap)
    _attn_mods = [ly.self_attn for ly in layers]
    _base_scaling = float(_attn_mods[0].scaling)

    def set_logn_scaling(seq_len):
        if args.attn_logn_sref <= 0:
            return
        import math

        s = _base_scaling * math.log(max(seq_len, 2)) / math.log(args.attn_logn_sref)
        for m_ in _attn_mods:
            m_.scaling = s
    # PEFT wrap LAST: PeftModel.model is the OUTER base model (the 137800/137801 lesson).
    if args.peft_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.peft_adapter),
                                          is_trainable=False)
        model.eval()
        print(f"peft adapter loaded (frozen): {args.peft_adapter}", flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    hooks = FenceHooks(layers).install()
    rng = random.Random(args.seed)
    pairs_f = open(out / "pairs.csv", "w", newline="")
    pairs_w = csv.writer(pairs_f)
    pairs_w.writerow(["sample_id", "n_frames", "gold", "flip_t", "flip_kind", "arm",
                      "layer", "locus", "dnorm", "base_norm"])
    log_f = open(out / "logits.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["sample_id", "n_frames", "gold", "flip_kind", "margin_base",
                    "margin_flip", "pred_base", "pred_flip", "dlogit_digits",
                    "base_digits", "flip_digits"])
    ctrl_f = open(out / "controls.csv", "w", newline="")
    ctrl_w = csv.writer(ctrl_f)
    ctrl_w.writerow(["sample_id", "n_frames", "kind", "arm", "layer", "locus",
                     "dnorm", "base_norm"])

    n_done = n_skip = 0
    gold_hist = []
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), args.seed):
        if n_done >= args.limit:
            break
        try:
            pair = build_pair(sd, rng, args.max_gold, gold_set=gold_set)
        except Exception as e:
            print(f"  [skip] {sd.name}: {e}", flush=True)
            n_skip += 1
            continue
        if pair is None:
            n_skip += 1
            continue
        NF = len(pair["base_states"])
        do_controls = n_done < args.controls
        rdir = out / "_render_tmp"
        base_blobs = render_states(pair["base_states"], PARK_ROOM_ORDER, rdir / "base")
        evid_blobs = render_states(pair["evid_states"], PARK_ROOM_ORDER, rdir / "evid")
        assert_pair_identity(base_blobs, evid_blobs, pair["flip_t"], pair["sample_id"])
        ctrl_blobs = None
        if do_controls:
            ctrl_blobs = render_states(pair["ctrl_states"], PARK_ROOM_ORDER, rdir / "ctrl")
            assert_pair_identity(base_blobs, ctrl_blobs, pair["flip_t"], pair["sample_id"])
        if n_done == 0:
            audit = out / "audit_pair0"
            if not audit.exists():
                import shutil
                shutil.copytree(rdir, audit)
        base_frames = blobs_to_frames(base_blobs, args.resize)
        evid_frames = blobs_to_frames(evid_blobs, args.resize)
        q0, room, flip_t, gold = pair["q0"], pair["room"], pair["flip_t"], pair["gold"]

        # ---------------- plain arm (deployed joint prompt) + A2 margins
        if "plain" in arms:
            prompt = build_count_prompt(q0, NF)
            in_base = move_to_device(build_prompt_inputs(processor, base_frames, prompt),
                                     model.device)
            in_evid = move_to_device(build_prompt_inputs(processor, evid_frames, prompt),
                                     model.device)
            if in_base["input_ids"].shape != in_evid["input_ids"].shape or not torch.equal(
                    in_base["input_ids"], in_evid["input_ids"]):
                print(f"  [skip] {sd.name}: token layout differs between members", flush=True)
                n_skip += 1
                continue
            seq_p = in_base["input_ids"].shape[1]
            loci_p = {"final": seq_p - 1}
            hooks.clear_mask()
            set_logn_scaling(seq_p)
            cap_pb, dig_b = capture(model, in_base, hs_layers, loci_p, digit_ids=digit_ids)
            cap_pe, dig_e = capture(model, in_evid, hs_layers, loci_p, digit_ids=digit_ids)
            for (L, name), (dn, bn) in dnorms(cap_pb, cap_pe).items():
                pairs_w.writerow([pair["sample_id"], NF, gold, flip_t, "evid", "plain",
                                  L, name, f"{dn:.6g}", f"{bn:.6g}"])
            if gold + 1 <= 9:  # single-digit margin protocol
                mb, pb = margin_of(dig_b, gold)
                me, pe_ = margin_of(dig_e, gold + 1)
                log_w.writerow([pair["sample_id"], NF, gold, "evid", f"{mb:.4f}",
                                f"{me:.4f}", pb, pe_,
                                f"{float(np.linalg.norm(dig_b - dig_e)):.4f}",
                                " ".join(f"{v:.3f}" for v in dig_b),
                                " ".join(f"{v:.3f}" for v in dig_e)])

        # ---------------- p1fence arm (the TRAINED fenced layout; SPARSE gate lives here)
        if "p1fence" in arms:
            def p1_inputs(frames):
                if args.nfree_prompt:  # S11: single-source S9 prompt builder
                    from train_sft_gated import build_fenced_messages as _bfm_sparse
                    msgs = _bfm_sparse(frames, q0, nfree=True)
                else:
                    msgs = build_fenced_messages(frames, q0)
                return move_to_device(dict(processor.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt")), model.device)

            p_base = p1_inputs(base_frames)
            p_evid = p1_inputs(evid_frames)
            p_ids = p_base["input_ids"][0].tolist()
            if p_evid["input_ids"][0].tolist() != p_ids:
                print(f"  [skip] {sd.name}: p1fence layout differs", flush=True)
                n_skip += 1
                continue
            parsed = parse_layout(p_ids, tok, q0, NF, vs_id)
            if parsed is None:
                print(f"  [skip] {sd.name}: p1fence layout parse failed", flush=True)
                n_skip += 1
                continue
            p_blocks, p_fin = parsed
            # verdict locus: room word inside flip_t's replica
            p_rep_t = locate_word_token(p_ids, tok, room, p_blocks[flip_t])
            if p_rep_t is None:
                n_skip += 1
                continue
            p_seq = len(p_ids)
            # SPARSE gate: per-member masks. gate=none keeps the single anchor mask.
            if args.gate == "oracle":
                def gated_mask(evid_set):
                    hide = []
                    for t, (a, b) in enumerate(p_blocks):
                        if t not in evid_set:
                            hide.extend(range(int(a), int(b)))
                    hard = build_block_mask(p_seq, p_blocks, hide_cols=hide)
                    if args.gate_bonus > 0:  # SOFTGATE: soften ONLY the gate entries, never the fence
                        import math
                        base = build_block_mask(p_seq, p_blocks, hide_cols=[])
                        soft = base.clone()
                        soft[(hard == MASK_MIN) & (base != MASK_MIN)] = -math.log(args.gate_bonus)
                        return soft
                    return hard

                p_mask_base = gated_mask(pair["evid"])
                p_mask_evid = gated_mask(pair["evid"] | {flip_t})
            else:
                p_mask_base = p_mask_evid = build_block_mask(p_seq, p_blocks,
                                                             hide_cols=[])
            with torch.inference_mode():
                p_pos_base, _ = rope_fn(p_base["input_ids"],
                                        image_grid_thw=p_base.get("image_grid_thw"),
                                        attention_mask=p_base.get("attention_mask"))
            p_pos = reset_positions(p_pos_base, p_blocks, p_fin)
            loci_q = {"final": p_seq - 1, "rep_t": p_rep_t}
            set_logn_scaling(p_seq)
            hooks.set_mask(p_mask_base, model.device)
            cap_qb, qdig_b = capture(model, p_base, hs_layers, loci_q, pos_ids=p_pos,
                                     digit_ids=digit_ids)
            hooks.set_mask(p_mask_evid, model.device)
            cap_qe, qdig_e = capture(model, p_evid, hs_layers, loci_q, pos_ids=p_pos,
                                     digit_ids=digit_ids)
            hooks.clear_mask()
            for (L, name), (dn, bn) in dnorms(cap_qb, cap_qe).items():
                pairs_w.writerow([pair["sample_id"], NF, gold, flip_t, "evid",
                                  "p1fence", L, name, f"{dn:.6g}", f"{bn:.6g}"])
            if "plain" not in arms and gold + 1 <= 9:  # one margin row/pair
                qmb, qpb = margin_of(qdig_b, gold)
                qme, qpe = margin_of(qdig_e, gold + 1)
                log_w.writerow([pair["sample_id"], NF, gold, "evid", f"{qmb:.4f}",
                                f"{qme:.4f}", qpb, qpe,
                                f"{float(np.linalg.norm(qdig_b - qdig_e)):.4f}",
                                " ".join(f"{v:.3f}" for v in qdig_b),
                                " ".join(f"{v:.3f}" for v in qdig_e)])
            if do_controls:
                hooks.set_mask(p_mask_base, model.device)
                cap_qb2, _ = capture(model, p_base, hs_layers, loci_q, pos_ids=p_pos)
                hooks.clear_mask()
                for (L, name), (dn, bn) in dnorms(cap_qb, cap_qb2).items():
                    ctrl_w.writerow([pair["sample_id"], NF, "replay", "p1fence", L,
                                     name, f"{dn:.6g}", f"{bn:.6g}"])
                p_ctrl = p1_inputs(blobs_to_frames(ctrl_blobs, args.resize))
                if p_ctrl["input_ids"][0].tolist() == p_ids:
                    # ctrl flip is answer-preserving: evidence set == base's -> base gate
                    hooks.set_mask(p_mask_base, model.device)
                    cap_qc, _ = capture(model, p_ctrl, hs_layers, loci_q, pos_ids=p_pos)
                    hooks.clear_mask()
                    for (L, name), (dn, bn) in dnorms(cap_qb, cap_qc).items():
                        ctrl_w.writerow([pair["sample_id"], NF, "ctrl", "p1fence", L,
                                         name, f"{dn:.6g}", f"{bn:.6g}"])

        # ---------------- replica layout (shared by repjoint + fenced)
        want_rep = bool({"repjoint", "fenced"} & arms)

        def rep_inputs(frames):
            c = [{"type": "text", "text": q0}]
            for f in frames:
                c.append({"type": "image", "image": f})
                c.append({"type": "text", "text": q0})
            c.append({"type": "text", "text": q0})
            return move_to_device(dict(processor.apply_chat_template(
                [{"role": "user", "content": c}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")), model.device)

        if want_rep:
            rin_base = rep_inputs(base_frames)
            rin_evid = rep_inputs(evid_frames)
            ids = rin_base["input_ids"][0].tolist()
            if rin_evid["input_ids"][0].tolist() != ids:
                print(f"  [skip] {sd.name}: replica layout differs", flush=True)
                n_skip += 1
                continue
            seq = len(ids)
            fg = image_token_groups(rin_base["input_ids"][0].cpu(),
                                    expected_num_frames=NF, processor=processor)
            spans = find_question_spans(ids, tok, q0, NF + 2)
            if len(fg) != NF or spans is None:
                print(f"  [skip] {sd.name}: layout parse failed", flush=True)
                n_skip += 1
                continue
            spans = spans[1:]
            rep_spans, fin_span = spans[:NF], spans[NF]
            rep_c = [locate_word_token(ids, tok, room, sp) for sp in rep_spans]
            if any(c is None for c in rep_c):
                n_skip += 1
                continue
            vis_by_frame = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long)
                            for g in fg]
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            if len(vstarts) != NF:
                n_skip += 1
                continue
            blocks = frame_blocks(vstarts, fin_span[0])
            loci_r = {"final": seq - 1, "anchor": seq - 1 - ANCHOR_OFFSET,
                      "rep_t": rep_c[flip_t]}

            m_base = build_replica_probe_mask(seq, rep_spans, vis_by_frame)
            m_fence = build_replica_probe_mask(seq, rep_spans, vis_by_frame,
                                               fence_frames=True, fence_blocks=True,
                                               blocks=blocks)
            with torch.inference_mode():
                base_pos, _ = rope_fn(rin_base["input_ids"],
                                      image_grid_thw=rin_base.get("image_grid_thw"),
                                      attention_mask=rin_base.get("attention_mask"))
            pos_fence = reset_positions(base_pos, blocks, fin_span[0])

        if "repjoint" in arms:
            hooks.set_mask(m_base, model.device)
            cap_rb, _ = capture(model, rin_base, hs_layers, loci_r)
            cap_re, _ = capture(model, rin_evid, hs_layers, loci_r)
            hooks.clear_mask()
            for (L, name), (dn, bn) in dnorms(cap_rb, cap_re).items():
                pairs_w.writerow([pair["sample_id"], NF, gold, flip_t, "evid",
                                  "repjoint", L, name, f"{dn:.6g}", f"{bn:.6g}"])

        if "fenced" in arms:
            hooks.set_mask(m_fence, model.device)
            cap_fb, _ = capture(model, rin_base, hs_layers, loci_r, pos_ids=pos_fence)
            cap_fe, _ = capture(model, rin_evid, hs_layers, loci_r, pos_ids=pos_fence)
            hooks.clear_mask()
            for (L, name), (dn, bn) in dnorms(cap_fb, cap_fe).items():
                pairs_w.writerow([pair["sample_id"], NF, gold, flip_t, "evid",
                                  "fenced", L, name, f"{dn:.6g}", f"{bn:.6g}"])

        # ---------------- controls
        if do_controls and "plain" in arms:
            hooks.clear_mask()
            cap_pb2, _ = capture(model, in_base, hs_layers, loci_p)
            for (L, name), (dn, bn) in dnorms(cap_pb, cap_pb2).items():
                ctrl_w.writerow([pair["sample_id"], NF, "replay", "plain", L, name,
                                 f"{dn:.6g}", f"{bn:.6g}"])
            ctrl_frames = blobs_to_frames(ctrl_blobs, args.resize)
            in_ctrl = move_to_device(build_prompt_inputs(processor, ctrl_frames, prompt),
                                     model.device)
            cap_pc, dig_c = capture(model, in_ctrl, hs_layers, loci_p,
                                    digit_ids=digit_ids)
            for (L, name), (dn, bn) in dnorms(cap_pb, cap_pc).items():
                ctrl_w.writerow([pair["sample_id"], NF, "ctrl", "plain", L, name,
                                 f"{dn:.6g}", f"{bn:.6g}"])
            if gold + 1 <= 9:
                mc, pc = margin_of(dig_c, gold)
                log_w.writerow([pair["sample_id"], NF, gold, "ctrl", f"{mb:.4f}",
                                f"{mc:.4f}", pb, pc,
                                f"{float(np.linalg.norm(dig_b - dig_c)):.4f}", "", ""])
        if do_controls and "fenced" in arms:
            order = [i for i in range(NF) if i != flip_t]
            rng.shuffle(order)
            perm = list(range(NF))
            j = 0
            for i in range(NF):
                if i != flip_t:
                    perm[i] = order[j]
                    j += 1
            perm_frames = [base_frames[perm[i]] for i in range(NF)]
            rin_perm = rep_inputs(perm_frames)
            if rin_perm["input_ids"][0].tolist() == ids:
                hooks.set_mask(m_fence, model.device)
                cap_fp, _ = capture(model, rin_perm, hs_layers, loci_r,
                                    pos_ids=pos_fence)
                hooks.clear_mask()
                for (L, name), (dn, bn) in dnorms(cap_fb, cap_fp).items():
                    ctrl_w.writerow([pair["sample_id"], NF, "perm", "fenced", L, name,
                                     f"{dn:.6g}", f"{bn:.6g}"])
            else:
                print(f"  [warn] {sd.name}: perm layout differs — perm control skipped",
                      flush=True)

        gold_hist.append(gold)
        n_done += 1
        pairs_f.flush(); log_f.flush(); ctrl_f.flush()
        if n_done % 5 == 0:
            print(f"  {n_done}/{args.limit} pairs (skip {n_skip}) {time.time()-t0:.0f}s",
                  flush=True)

    hooks.remove()
    pairs_f.close(); log_f.close(); ctrl_f.close()
    lines = [f"=== HAHN PROBE GATED (pairs={n_done}, skip={n_skip}, data={args.data_root}, "
             f"max_gold={args.max_gold}, controls={min(args.controls, n_done)}, "
             f"hs_layers={hs_layers}, arms={sorted(arms)}, "
             f"adapter={args.peft_adapter or 'none'}, "
             f"gold_set={sorted(gold_set) if gold_set else 'none'}, "
             f"logn_sref={args.attn_logn_sref}, gate={args.gate}, "
             f"nfree={args.nfree_prompt} gate_bonus={args.gate_bonus}) ===",
             "[gold-hist] " + format_gold_histogram(gold_hist)]
    import collections
    acc = collections.defaultdict(list)
    with open(out / "pairs.csv") as fh:
        for row in csv.DictReader(fh):
            acc[(row["arm"], row["layer"], row["locus"])].append(float(row["dnorm"]))
    for k in sorted(acc):
        v = np.array(acc[k])
        lines.append(f"  {k[0]:8s} L{k[1]:>2s} {k[2]:6s} median dnorm "
                     f"{np.median(v):.4g}  (iqr {np.percentile(v,25):.4g}"
                     f"–{np.percentile(v,75):.4g}, n={len(v)})")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
