#!/usr/bin/env python3
"""STAGE-2: the CARRIER LAYER (2026-07-18) — all-in-model aggregation.

Architecture (one forward, frozen 4-bit backbone):
  layers 0..L_OPEN-1 : fenced blocks — carrier_i reads ONLY {prefix+question, frame_i, itself}
                       (the proven supply mechanism, d' ~8-9)
  layers L_OPEN..27  : the fence OPENS between carriers — carrier_i additionally attends earlier
                       carriers (causal), and the tail (final question + answer position) attends
                       ALL carriers. Carriers get sequential positions (temporal order exists).
  loss               : plain LM cross-entropy on the gold answer digit at the answer position.
                       No gate, no tally, no fact sentence — the network must learn aggregation.

Trainable: e_c (carrier embedding, warm-startable from the distilled ckpt) + hand-rolled LoRA
(rank R on q/k/v/o_proj of layers >= L_OPEN, B zero-init so ep0 == no-LoRA). Everything else
frozen.

Anchors: ep0 emitted accuracy (carriers present, nothing trained) ~ frozen-model level; ceiling =
the gate->tally scaffold (0.991 @N=8 Q-first). Per-count accuracy reported to watch the
undercount clamp break.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import (iter_sample_dirs, iter_sample_dirs_shuffled,
                                       load_mmred_sample)
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from experiments.glstm.carrier_token_distill import (build_block_mask, reset_positions,
                                                     find_subseq, CARRIER_TOKEN, ROOMS)

MIN = -65504.0

from torch.nn.attention import sdpa_kernel, SDPBackend
# Prefer memory-efficient attention (O(seq) memory — the math kernel materializes ~30GB of
# scores at seq≈23k, which is what made N=128 evals h200-only); MATH stays as fallback.
EFF_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def make_masks(seq, blocks, cpos, fin_start):
    """lo = fenced blocks (carrier reads prefix+question, own frame, itself);
    hi = fence opened between carriers (causal) + tail reads all carriers.
    Rebuilt lazily at forward time in training mode (P1 2026-07-18): caching two
    seq^2 fp16 masks per sample is ~7 MB/sample — prohibitive at pooled-data scale."""
    mask_lo = build_block_mask(seq, blocks, hide_cols=cpos)
    mask_hi = mask_lo.clone()
    ct = torch.tensor(cpos, dtype=torch.long)
    for i in range(1, len(cpos)):              # carrier_i attends earlier carriers (causal)
        mask_hi[cpos[i], ct[:i]] = 0.0
    mask_hi[fin_start:, ct] = 0.0              # tail attends all carriers
    return mask_lo, mask_hi


def keep_cols(seq, blocks, cpos):
    """TRUNC campaign (2026-07-24): surviving columns for physical frame-drop —
    [prefix+question]+[carriers]+[tail]. Everything outside frame blocks survives;
    inside a block only its carrier does. Original indices, original order."""
    inblk = np.zeros(seq, dtype=bool)
    for a, b in blocks:
        inblk[a:b] = True
    cset = set(cpos)
    return [i for i in range(seq) if (not inblk[i]) or i in cset]


def frame_cols(seq, blocks, cpos):
    """Complement of keep_cols within the blocks: the frame tokens (incl. vision
    start/end furniture) that --drop-frame-kv hides from decoded rows."""
    cset = set(cpos)
    out = []
    for a, b in blocks:
        out.extend(i for i in range(a, b) if i not in cset)
    return out


def truncated_masks(keep, cpos):
    """TRUNC E5: direct construction of the truncated lo/hi masks over the keep sequence
    (no dense seq^2 intermediate). Derivation from build_block_mask/make_masks with frame
    cols removed: hi_t == plain causal (carrier own-frame edges are gone, everything else
    the tail/carriers see is causal); lo_t == causal with carrier COLUMNS hidden from every
    other row (each carrier still sees itself; carrier->earlier-carrier and tail->carrier
    edges are hi-only). Verified equal to index-selected dense masks in trunc_mask_smoke."""
    k = len(keep)
    idx = {p: j for j, p in enumerate(keep)}
    car = torch.tensor([idx[c] for c in cpos], dtype=torch.long)
    m = torch.zeros(k, k, dtype=torch.float16)
    m.masked_fill_(torch.triu(torch.ones(k, k, dtype=torch.bool), 1), MIN)
    hi_t = m.clone()
    lo_t = m
    lo_t[:, car] = MIN
    lo_t[car, car] = 0.0
    return lo_t, hi_t


def parse_task_labels(q0, states, gold):
    """Task-agnostic sample parsing (2026-07-18 E4/E5): the question decides the task.

    Returns (task, evid, aux) or None (sanity-check failure -> skip). evid feeds the
    scratchpad targets (A1) — legitimately available at TRAINING time only; aux carries
    the visited-room names for the rooms task (None otherwise).
    """
    mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
    if mm:
        nA, nB = mm.group(1), mm.group(2)
        evid = set()
        for t, st in enumerate(states):
            for occ in (st.get("rooms", {}) or {}).values():
                if nA in occ and nB in occ:
                    evid.add(t)
                    break
        return ("cooc", evid, None) if len(evid) == gold else None
    mm = re.search(r"How many frames was (\w+) in the (\w+) or the (\w+)", q0)
    if mm:                                 # composition eval: OR-union (never trained)
        c, r1, r2 = mm.groups()
        evid = {t for t, st in enumerate(states)
                if c in ((st.get("rooms", {}) or {}).get(r1, []) or [])
                or c in ((st.get("rooms", {}) or {}).get(r2, []) or [])}
        return ("union", evid, None) if len(evid) == gold else None
    mm = re.search(r"In which frame number \(1-\d+\) was (\w+) in the (\w+)", q0)
    if mm:                                 # C3 NIAH/which-frame (answer = 1-indexed frame)
        c, r = mm.group(1), mm.group(2)
        evid = {t for t, st in enumerate(states)
                if c in ((st.get("rooms", {}) or {}).get(r, []) or [])}
        return ("which", evid, None) if (len(evid) == 1 and gold - 1 in evid) else None
    mm = re.search(r"How many distinct rooms did (\w+) visit", q0)
    if mm:
        name = mm.group(1)
        rooms_v, evid = set(), set()
        for t, st in enumerate(states):
            for rname, occ in (st.get("rooms", {}) or {}).items():
                if name in occ:
                    rooms_v.add(rname)
                    evid.add(t)
        return ("rooms", evid, sorted(rooms_v)) if len(rooms_v) == gold else None
    mm = re.search(r"In how many of the \d+ frames does .+ appear", q0)
    if mm and states and isinstance(states[0], dict) and "natural" in states[0]:
        # P3a (2026-07-24): natural-images counting (mmred_natural_mm composed samples);
        # per-frame evidence from the judge-gated flags; counts like steps downstream.
        evid = {t for t, st in enumerate(states)
                if (st.get("natural", {}) or {}).get("evidence")}
        return ("steps", evid, None) if len(evid) == gold else None
    evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
    if not evid and gold != 0:
        return None
    return ("steps", evid, None)


def build_target(task, evid, aux, gold):
    """A1 verdict-scratchpad target: evidence verdicts (frames 1-indexed) then the count.
    Fixed short format; the integer after '->' is the parsed answer at eval."""
    if task == "rooms":
        return (f" rooms {', '.join(aux)} -> {gold}" if aux else " none -> 0")
    if evid:
        return f" frames {', '.join(str(i + 1) for i in sorted(evid))} -> {gold}"
    return " none -> 0"


def couple_offsets(token_texts, NF):
    """E-G position-coupling stream rule (arXiv:2405.20671 / 2410.15787, applied to the tally).

    Walk the token stream; token k gets (anchor, off): anchor = the 1-indexed carrier the
    stream is currently 'on', off = increment within that anchor span. The anchor jumps to
    carrier m right AFTER a token completes integer m at the START of a comma-separated
    segment (before any '('), so tally counts inside parens never re-anchor; '->' freezes
    the anchor for the tail. Deterministic from token strings alone — the SAME function
    drives teacher-forced target positions (training) and online greedy positions (decode),
    making train/eval consistency structural. Assign-then-update: the frame-index token
    itself still rides the previous anchor (the 'find the next evidence carrier' hop is then
    a short forward scan from the previous carrier, N-independent on average)."""
    out = []
    anchor, off, seg, tail = 1, 0, "", False
    for t in token_texts:
        out.append((anchor, off)); off += 1
        seg += t
        if "->" in seg:
            tail = True
        if not tail and "(" not in seg:
            mm = re.search(r"(\d+)", seg)
            if mm:
                m = int(mm.group(1))
                if 1 <= m <= NF and m != anchor:
                    anchor, off = m, 0
        if "," in t:
            seg = ""
    return out


def build_target_tally(task, evid, aux, gold):
    """RUNNING-TALLY variant (2026-07-19, R1): every verdict carries its running count —
    'frames 2 (1), 5 (2), 9 (3) -> 3' — so the final answer is a read-off of the last tally,
    not a post-hoc count of a long list (the measured failure mode at N>=32: verdict
    undercounts, MAE 1.44, while format/parse never breaks). Same '-> N' suffix, so the
    eval parser is unchanged."""
    if task == "rooms":
        if not aux:
            return " none -> 0"
        return (" rooms " + ", ".join(f"{r} ({k})" for k, r in enumerate(aux, 1))
                + f" -> {gold}")
    if evid:
        return (" frames " + ", ".join(f"{i + 1} ({k})" for k, i in enumerate(sorted(evid), 1))
                + f" -> {gold}")
    return " none -> 0"


SCRATCHPAD_FORMATS = ("poslist", "scan", "caption", "chunked")
CHUNK_FRAMES = 16


def frame_attr_labels(task, q0, states, evid):
    """FORMAT sweep (2026-07-22): per-evidence-frame room word for scan/caption targets.
    Room names kept capitalized exactly as in states/questions. Returns {} on parse
    failure (builders then fall back to 'yes' so no sample is ever skipped — the
    train/eval SPLIT must stay byte-identical across arms)."""
    out = {}

    def room_of(t, pred):
        st = states[t] if t < len(states) else {}
        for rname, occ in ((st.get("rooms", {}) or {}) if isinstance(st, dict) else {}).items():
            if pred(rname, occ or []):
                return rname
        return None

    if task == "cooc":
        mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
        if not mm:
            return {}
        nA, nB = mm.group(1), mm.group(2)
        for t in evid:
            r = room_of(t, lambda rn, oc: nA in oc and nB in oc)
            if r:
                out[t] = r
    elif task == "union":
        mm = re.search(r"How many frames was (\w+) in the (\w+) or the (\w+)", q0)
        if not mm:
            return {}
        c, r1, r2 = mm.group(1), mm.group(2), mm.group(3)
        for t in evid:
            st = states[t] if t < len(states) else {}
            rooms = (st.get("rooms", {}) or {}) if isinstance(st, dict) else {}
            out[t] = r1 if c in (rooms.get(r1, []) or []) else r2
    elif task == "which":
        mm = re.search(r"In which frame number \(1-\d+\) was (\w+) in the (\w+)", q0)
        if not mm:
            return {}
        for t in evid:
            out[t] = mm.group(2)
    elif task == "rooms":
        mm = re.search(r"How many distinct rooms did (\w+) visit", q0)
        if not mm:
            return {}
        name = mm.group(1)
        for t in evid:
            r = room_of(t, lambda rn, oc: name in oc)
            if r:
                out[t] = r
    else:                                  # steps: the queried room (constant over evid)
        if states and isinstance(states[0], dict) and "natural" in states[0]:
            con = (states[0].get("natural", {}) or {}).get("concept")
            return {t: str(con) for t in evid} if con else {}
        pr = eval_utils.parse_target_character_room(q0)
        if not pr:
            return {}
        for t in evid:
            out[t] = pr[1]
    return out


def build_target_scan(task, evid, gold, NF, labels, caption):
    """Arms B/C (2026-07-22): full-scan — EVERY frame gets a slot in frame order, evidence
    slots increment the tally inline, explicit 'END' terminator. B = 'yes' verdicts,
    C = room-word captions. The rooms task carries room words in BOTH arms (a bare yes
    cannot express distinct-room counting; tally increments only on FIRST visit). For
    'which' the total slot reads out the answer (the frame number), not the tally."""
    labels = labels or {}
    parts = []
    k = 0
    seen = set()
    for t in range(NF):
        if t in evid:
            w = labels.get(t, "yes") if (caption or task == "rooms") else "yes"
            if task == "rooms":
                if w not in seen:
                    seen.add(w)
                    k += 1
                    parts.append(f"f{t + 1}:{w}({k})")
                else:
                    parts.append(f"f{t + 1}:{w}")
            else:
                k += 1
                parts.append(f"f{t + 1}:{w}({k})")
        else:
            parts.append(f"f{t + 1}:-")
    return " scan: " + " ".join(parts) + f" | total: {gold} END"


def build_target_chunked(task, evid, gold, NF, labels):
    """Arm D (2026-07-22): blocks of CHUNK_FRAMES frames, positive-list per block (GLOBAL
    1-indexed frame numbers), '| sub k' per block, final 'total: a+b = c END'. Rooms lists
    only NEW rooms per block (subtotals of distinct-room increments); 'which' reads out the
    frame number without a sum expression."""
    labels = labels or {}
    parts = []
    subs = []
    seen = set()
    for c0 in range(0, NF, CHUNK_FRAMES):
        idx = [t for t in sorted(evid) if c0 <= t < c0 + CHUNK_FRAMES]
        if task == "rooms":
            items = []
            for t in idx:
                w = labels.get(t)
                if w and w not in seen:
                    seen.add(w)
                    items.append(w)
        else:
            items = [str(t + 1) for t in idx]
        subs.append(len(items))
        parts.append(f"c{len(subs)}: " + (", ".join(items) if items else "none")
                     + f" | sub {len(items)}")
    if task == "which":
        tail = f"total: {gold} END"
    else:
        tail = "total: " + "+".join(str(s) for s in subs) + f" = {gold} END"
    return " " + " ".join(parts) + " " + tail


def build_target_fmt(fmt, task, evid, aux, gold, NF, labels=None):
    """Dispatch for --scratchpad-format. 'poslist' is the unchanged l12v2 control format."""
    if fmt == "poslist":
        return build_target_tally(task, evid, aux, gold)
    if fmt in ("scan", "caption"):
        return build_target_scan(task, evid, gold, NF, labels, caption=(fmt == "caption"))
    if fmt == "chunked":
        return build_target_chunked(task, evid, gold, NF, labels)
    raise ValueError(f"unknown scratchpad format {fmt!r}")


def parse_scratchpad_answer(text, fmt):
    """Eval parser. poslist: integer after the LAST '->' (unchanged, backward-compatible).
    scan/caption/chunked: anchor on the LAST 'total:', take the LAST integer before 'END'
    (or before end-of-text if END never decoded — still a parse, format fidelity is
    reported separately via mean decode tokens / transcripts)."""
    if fmt == "poslist":
        mm = re.findall(r"->\s*(\d+)", text)
        return int(mm[-1]) if mm else None
    seg = text.rsplit("total:", 1)
    if len(seg) < 2:
        return None
    mm = re.findall(r"(\d+)", seg[1].split("END")[0])
    return int(mm[-1]) if mm else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform",
                    help="comma-separated list of roots = task MIXTURE (E4/E5); per-sample "
                         "task is inferred from the question")
    ap.add_argument("--limit", type=int, default=300, help="PER data root")
    ap.add_argument("--train-n", type=int, default=150)
    ap.add_argument("--l-open", type=int, default=17)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr-carrier", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--carrier-init", default=None,
                    help="path to carrier_best.pt (distilled) to warm-start e_c; default room-mean")
    ap.add_argument("--no-lora", action="store_true", help="ablation: train e_c only")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED",
                    help="stratified deterministic shuffle of sample dirs (class-balanced "
                         "prefixes; REQUIRED whenever --limit < the full dir count)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="No training: load --ckpt (e_c + LoRA), STREAM samples (no RAM "
                         "cache), report emitted accuracy. For length/task generalization "
                         "(E3/E4a). gold>9 allowed here (greedy multi-digit decode).")
    ap.add_argument("--ckpt", default=None, help="carrier_layer_best.pt for --eval-only")
    ap.add_argument("--decode-tokens", type=int, default=3,
                    help="eval-only: greedy decode up to this many tokens (multi-digit golds)")
    ap.add_argument("--dirs-file", default=None,
                    help="eval-only: file listing sample-dir paths (one per line) — stream "
                         "exactly these (in-dist held-out greedy eval of a cached-trainer run)")
    ap.add_argument("--alien-task", action="store_true",
                    help="eval-only (P2b MLVU, 2026-07-23): accept samples whose question "
                         "matches no MMRED task template — greedy decode + emitted-count "
                         "scoring only (evid empty; TF metrics meaningless, ignore them)")
    ap.add_argument("--dump-decodes", type=int, default=3,
                    help="eval-only scratchpad: print the first N decoded transcripts "
                         "(diagnostics; default 3)")
    ap.add_argument("--no-reset-positions", action="store_true",
                    help="posreset-necessity ablation (2026-07-27): run with NATURAL M-RoPE "
                         "positions instead of the PCW-style per-block reset. Eval-only use is a "
                         "TRAIN/TEST MISMATCH (the ckpt was trained with reset) — report it as "
                         "such, not as an architectural verdict.")
    ap.add_argument("--scratchpad-format", default=None, choices=SCRATCHPAD_FORMATS,
                    help="eval-only: override the scratchpad target format (default: "
                         "auto-detect from the ckpt's 'scratchpad_format' field, falling "
                         "back to poslist)")
    ap.add_argument("--drop-frame-kv", action="store_true",
                    help="TRUNC (2026-07-24, eval-only): decoded rows cannot attend frame "
                         "columns (mask-only change, prefill untouched; same shapes as "
                         "baseline so any token change is caused by the removed edges)")
    ap.add_argument("--truncate-at", type=int, default=None, metavar="L",
                    help="TRUNC (eval-only): physically drop frame rows from the hidden "
                         "states entering layer L — layers >=L and (with the cached decode "
                         "semantics) all decoding run on [question]+[carriers]+[tail] with "
                         "ORIGINAL position ids (index-selected, never renumbered). "
                         "Implies --drop-frame-kv for decoded rows.")
    ap.add_argument("--fast-decode", action="store_true",
                    help="TRUNC (eval-only): cached incremental decode — prefill once, "
                         "capture per-layer inputs at the keep columns, decode steps run "
                         "the layer stack on [keep||appended] only. Mathematically equal "
                         "to --drop-frame-kv (+ --truncate-at if set); reports the real "
                         "decode speedup. Requires one of the two flags above.")
    ap.add_argument("--exactness-check", action="store_true",
                    help="TRUNC E1: per sample decode baseline AND flagged arms, report "
                         "token-identity + per-arm wall-clock + peak VRAM")
    ap.add_argument("--chunked-prefill", action="store_true",
                    help="TRUNC E5: layers 0..L-1 as independent per-block short forwards "
                         "([prefix+question][frame_i][carrier_i] batched + one "
                         "[prefix+question][tail] chunk, ALL plain-causal masks — no dense "
                         "seq^2 mask anywhere); exact for question/carrier rows under the "
                         "fence; tail rows lose their (lo-phase) frame reads = full "
                         "truncation semantics. Requires --truncate-at == the ckpt's "
                         "L_open and --fast-decode.")
    ap.add_argument("--verify-chunked", action="store_true",
                    help="TRUNC E5 verification: per sample run dense-truncated AND "
                         "chunked prefill, report per-group hidden-state deltas at L_open "
                         "+ decode token identity")
    ap.add_argument("--dump-carrier-states", default=None, metavar="LAYERS",
                    help="TRUNC hybrid cell (2026-07-25): comma list of layers — per "
                         "sample run the (possibly truncated) prefill and dump carrier-row "
                         "hidden states entering each layer to carrier_states_cache.pt "
                         "(keys rep/labels/gold, replica_gate_tally.py-compatible; labels "
                         "from parsed evidence). Steps-task dirs only. Decode still runs "
                         "(use a small --decode-tokens to make it cheap).")
    ap.add_argument("--output", default="outputs/ladder/image_longN/carrier_layer")
    args = ap.parse_args()
    if args.eval_only and not args.ckpt:
        ap.error("--eval-only requires --ckpt")
    trunc_flags = (args.drop_frame_kv or args.truncate_at is not None
                   or args.fast_decode or args.exactness_check
                   or args.dump_carrier_states is not None)
    dump_layers = ([int(x) for x in args.dump_carrier_states.split(",")]
                   if args.dump_carrier_states else [])
    if trunc_flags and not args.eval_only:
        ap.error("--drop-frame-kv/--truncate-at/--fast-decode/--exactness-check are eval-only")
    if args.fast_decode and not (args.drop_frame_kv or args.truncate_at is not None):
        ap.error("--fast-decode requires --drop-frame-kv or --truncate-at")
    if args.exactness_check and not (args.drop_frame_kv or args.truncate_at is not None):
        ap.error("--exactness-check requires --drop-frame-kv or --truncate-at")
    if args.chunked_prefill and (args.truncate_at is None or not args.fast_decode):
        ap.error("--chunked-prefill requires --truncate-at and --fast-decode")
    if args.verify_chunked and not args.chunked_prefill:
        ap.error("--verify-chunked requires --chunked-prefill")
    ck = None
    scratch = False
    pcouple = False
    sfmt = args.scratchpad_format or "poslist"
    if args.eval_only:
        ck = torch.load(args.ckpt, map_location="cpu")
        args.l_open = int(ck["l_open"]); args.rank = int(ck["rank"])
        scratch = bool(ck.get("scratchpad"))
        pcouple = bool(ck.get("pos_couple"))
        sfmt = args.scratchpad_format or ck.get("scratchpad_format") or "poslist"
        if pcouple:
            print("[eval-only] POSITION-COUPLED decode active (E-G stream rule)", flush=True)
        if pcouple and trunc_flags:
            raise SystemExit("truncation flags are not supported with pos-coupled ckpts")
        if args.chunked_prefill and args.truncate_at != int(ck["l_open"]):
            raise SystemExit("--chunked-prefill is only exact when --truncate-at == L_open "
                             f"(fence layers); ckpt L_open={ck['l_open']}, got "
                             f"{args.truncate_at}")
        print(f"[eval-only] ckpt {args.ckpt} (trained ep {ck.get('epoch')}, "
              f"acc {ck.get('acc'):.3f}, L_open={args.l_open}, rank={args.rank}, "
              f"scratchpad={scratch}, fmt={sfmt})", flush=True)
        if scratch:
            args.decode_tokens = max(args.decode_tokens, 48)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    n_layers = len(layers)
    LO = args.l_open
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    tok = processor.tokenizer
    text_model = model.model.language_model
    dev = model.device
    D = cfg.hidden_size
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S")
                               + f"_L{LO}_r{args.rank}{'_nolora' if args.no_lora else ''}"
                               + ("_evalonly" if args.eval_only else ""))
    out.mkdir(parents=True, exist_ok=True)

    cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    vs_id = int(model.config.vision_start_token_id)
    rope_fn = getattr(model, "get_rope_index", None) or model.model.get_rope_index
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    # ---- trainable params ----
    if args.eval_only:
        e_c = nn.Parameter(ck["e_c"].float().to(dev))
    elif args.carrier_init:
        ck = torch.load(args.carrier_init, map_location="cpu")
        e_c = nn.Parameter(ck["e_c"].float().to(dev))
        print(f"[init] e_c from {args.carrier_init} (distilled d' {ck.get('dprime'):.2f})", flush=True)
    else:
        rows = []
        for r in ROOMS:
            tid = tok(" " + r, add_special_tokens=False).input_ids
            rows.append(text_model.embed_tokens.weight[tid[-1]].float())
        e_c = nn.Parameter(torch.stack(rows).mean(0).detach().float().to(dev))

    lora = {}
    handles = []
    if not args.no_lora:
        scale = args.alpha / args.rank
        for li in range(LO, n_layers):
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(layers[li].self_attn, nm)
                din = mod.in_features; dout = mod.out_features
                if args.eval_only:
                    A0, B0 = ck["lora"][f"{li}.{nm}"]
                    A = nn.Parameter(A0.float().to(dev)); B = nn.Parameter(B0.float().to(dev))
                else:
                    A = nn.Parameter(torch.randn(args.rank, din, device=dev) * 0.01)
                    B = nn.Parameter(torch.zeros(dout, args.rank, device=dev))
                lora[(li, nm)] = (A, B)

                def mk(A=A, B=B):
                    def hook(_m, inp, o):
                        x = inp[0]
                        return o + (scale * (x.float() @ A.T) @ B.T).to(o.dtype)
                    return hook
                handles.append(mod.register_forward_hook(mk()))
    lora_params = [p for ab in lora.values() for p in ab]
    opt = torch.optim.Adam([{"params": [e_c], "lr": args.lr_carrier},
                            {"params": lora_params, "lr": args.lr_lora}])
    n_par = D + sum(p.numel() for p in lora_params)
    print(f"[params] e_c {D} + lora {n_par - D} = {n_par}", flush=True)

    # ---- forward machinery (defined before prep so eval-only can STREAM) ----
    def _ext_mask(m, e):
        if e == 0:
            return m
        s0 = m.shape[0]
        big = torch.full((s0 + e, s0 + e), MIN, dtype=m.dtype)
        big[:s0, :s0] = m
        for j in range(e):
            r = s0 + j
            big[r, :s0] = m[s0 - 1]          # appended tokens read like the last tail row
            big[r, s0:r + 1] = 0.0           # + causal over appended tokens
        return big

    def forward_logits(d, grad, extra=(), dropkv=False, trunc=None):
        e = len(extra)
        seq = d["seq"] + e
        emb = d["emb"].to(dev).unsqueeze(0)
        if e:
            ext = text_model.embed_tokens(torch.tensor([list(extra)], device=dev))
            emb = torch.cat([emb, ext.to(emb.dtype)], dim=1)
        emb = emb.clone()
        stack = e_c.unsqueeze(0).repeat(len(d["cpos"]), 1).to(torch.bfloat16)
        emb[0, torch.tensor(d["cpos"], device=dev)] = stack if grad else stack.detach()
        if "lo" in d:
            lo_m, hi_m = d["lo"], d["hi"]
        else:                                   # training cache: masks rebuilt lazily
            lo_m, hi_m = make_masks(d["seq"], d["blocks"], d["cpos"], d["fin"])
        lo2, hi2 = _ext_mask(lo_m, e), _ext_mask(hi_m, e)
        if dropkv and e:                        # decoded rows lose frame cols (fresh big
            fc = torch.tensor(d["fcols"], dtype=torch.long)  # tensor when e>0 — safe)
            lo2[d["seq"]:, fc] = MIN
            hi2[d["seq"]:, fc] = MIN
        lo = lo2.to(dev).to(torch.float32).view(1, 1, seq, seq)
        hi = hi2.to(dev).to(torch.float32).view(1, 1, seq, seq)
        pos = d["pos"].to(dev)
        if e:
            if pcouple and d.get("task") != "rooms":
                texts = [tok.decode([t]) for t in extra]
                anch = couple_offsets(texts, len(d["cpos"]))
                cp = [int(pos[0, 0, c]) for c in d["cpos"]]
                vals = torch.tensor([cp[a - 1] + o for a, o in anch],
                                    device=dev).view(1, 1, e).expand(3, 1, e)
                pos = torch.cat([pos, vals], dim=2)
            else:
                inc = torch.arange(1, e + 1, device=dev).view(1, 1, e)
                pos = torch.cat([pos, pos[:, :, -1:] + inc], dim=2)
        cos_, sin_ = text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        h = emb
        with sdpa_kernel(EFF_SDPA):
            if trunc is None:
                for li, ly in enumerate(layers):
                    h = ly(h, attention_mask=(lo if li < LO else hi),
                           position_embeddings=pe)[0]
            else:
                for li in range(trunc):
                    h = layers[li](h, attention_mask=(lo if li < LO else hi),
                                   position_embeddings=pe)[0]
                # physical truncation: index-select rows/masks/positions — original
                # position ids preserved (P0.2); appended rows always survive
                kt = torch.tensor(d["keep"] + list(range(d["seq"], seq)), device=dev)
                k2 = kt.numel()
                h = h.index_select(1, kt)
                lo_t = lo[0, 0].index_select(0, kt).index_select(1, kt).view(1, 1, k2, k2)
                hi_t = hi[0, 0].index_select(0, kt).index_select(1, kt).view(1, 1, k2, k2)
                cos_t, sin_t = text_model.rotary_emb(h, pos.index_select(2, kt))
                pe_t = (cos_t.to(h.dtype), sin_t.to(h.dtype))
                for li in range(trunc, n_layers):
                    h = layers[li](h, attention_mask=(lo_t if li < LO else hi_t),
                                   position_embeddings=pe_t)[0]
        h = text_model.norm(h)
        return model.lm_head(h[0, -1].to(model.lm_head.weight.dtype)).float()

    def decode_answer(d, dropkv=False, trunc=None):
        """Greedy decode up to --decode-tokens digit tokens (multi-digit golds, E3).
        Returns (parsed int or None, 0-9-restricted first-token argmax)."""
        toks = []
        first_digit = None
        for step in range(args.decode_tokens):
            lg = forward_logits(d, False, extra=tuple(toks), dropkv=dropkv, trunc=trunc)
            if step == 0:
                first_digit = int(np.argmax([float(lg[t]) for t in digit_ids]))
            t = int(lg.argmax())
            if not tok.decode([t]).strip().isdigit():
                break
            toks.append(t)
        text = tok.decode(toks).strip()
        return (int(text) if text.isdigit() else None), first_digit

    def decode_scratchpad(d, dropkv=False, trunc=None):
        """A3 exam decode: greedy up to --decode-tokens, stop at EOS; parse per the ckpt's
        scratchpad format (poslist '->' / scan-family 'total:' anchor). Returns
        (parsed int or None, decoded text, decoded token ids)."""
        toks = []
        for _step in range(args.decode_tokens):
            lg = forward_logits(d, False, extra=tuple(toks), dropkv=dropkv, trunc=trunc)
            t = int(lg.argmax())
            if t == tok.eos_token_id:
                break
            toks.append(t)
        text = tok.decode(toks)
        return parse_scratchpad_answer(text, sfmt), text, toks

    def prefill_capture(d, trunc=None):
        """TRUNC fast decode: one prefill over the prompt (honoring --truncate-at),
        capturing every layer's INPUT hidden states at the keep columns. Prompt rows never
        attend appended rows (causal), so these states are exact for all decode steps.
        Returns (caches[n_layers] (k,D) bf16, step-0 logits, truncated lo/hi (k,k) cpu,
        pos_keep (3,1,k), last prompt position (3,1,1))."""
        seq = d["seq"]
        emb = d["emb"].to(dev).unsqueeze(0).clone()
        stack = e_c.unsqueeze(0).repeat(len(d["cpos"]), 1).to(torch.bfloat16)
        emb[0, torch.tensor(d["cpos"], device=dev)] = stack.detach()
        lo = d["lo"].to(dev).to(torch.float32).view(1, 1, seq, seq)
        hi = d["hi"].to(dev).to(torch.float32).view(1, 1, seq, seq)
        pos = d["pos"].to(dev)
        cos_, sin_ = text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        kt = torch.tensor(d["keep"], device=dev)
        k = kt.numel()
        pos_k = pos.index_select(2, kt)
        lo_t = lo[0, 0].index_select(0, kt).index_select(1, kt)
        hi_t = hi[0, 0].index_select(0, kt).index_select(1, kt)
        caches = []
        h = emb
        done = False
        pe_t = None
        with torch.no_grad(), sdpa_kernel(EFF_SDPA):
            for li, ly in enumerate(layers):
                if trunc is not None and li == trunc:
                    h = h.index_select(1, kt)
                    cos_t, sin_t = text_model.rotary_emb(h, pos_k)
                    pe_t = (cos_t.to(h.dtype), sin_t.to(h.dtype))
                    done = True
                caches.append((h[0].clone() if done else h[0].index_select(0, kt).clone()))
                if done:
                    h = ly(h, attention_mask=(lo_t if li < LO else hi_t).view(1, 1, k, k),
                           position_embeddings=pe_t)[0]
                else:
                    h = ly(h, attention_mask=(lo if li < LO else hi),
                           position_embeddings=pe)[0]
            hn = text_model.norm(h)
            lg0 = model.lm_head(hn[0, -1].to(model.lm_head.weight.dtype)).float()
        return caches, lg0, lo_t.cpu().to(torch.float16), hi_t.cpu().to(torch.float16), \
            pos_k, pos[:, :, -1:]

    def prefill_chunked(d, trunc):
        """TRUNC E5 chunked prefill: fence+posreset make layers 0..trunc-1 EXACTLY
        per-block computable for question/carrier rows ([prefix+question][frame_i]
        [carrier_i] chunks, batched over blocks — plain causal masks, carrier last row).
        Tail rows come from a [prefix+question][tail] chunk WITHOUT frames (full-truncation
        semantics; the dense path lets tail read frames in lo — that delta is E5-verified
        and E2-priced). Upper stack + decode masks built directly in truncated coords —
        no dense seq^2 mask anywhere. Same returns as prefill_capture."""
        seq = d["seq"]
        blocks, cpos, fin, keep = d["blocks"], d["cpos"], d["fin"], d["keep"]
        a0 = blocks[0][0]
        NF = len(cpos)
        assert all(c == b - 1 for c, (a, b) in zip(cpos, blocks)), "carrier must end block"
        bl = blocks[0][1] - blocks[0][0]
        assert all(b - a == bl for a, b in blocks), "posreset requires equal blocks"
        emb = d["emb"].to(dev).clone()
        emb[torch.tensor(cpos, device=dev)] = e_c.detach().to(torch.bfloat16)
        pos = d["pos"].to(dev)
        Lb = a0 + bl
        bat = torch.stack([torch.cat([emb[:a0], emb[a:b]]) for a, b in blocks])
        pos_b = torch.cat([torch.cat([pos[:, :, :a0], pos[:, :, a:b]], dim=2)
                           for a, b in blocks], dim=1)                    # (3,NF,Lb)
        Lt = a0 + (seq - fin)
        tl = torch.cat([emb[:a0], emb[fin:]]).unsqueeze(0)
        pos_t = torch.cat([pos[:, :, :a0], pos[:, :, fin:]], dim=2)       # (3,1,Lt)

        def _causal(n):
            m = torch.zeros(n, n, dtype=torch.float32, device=dev)
            m.masked_fill_(torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1),
                           MIN)
            return m.view(1, 1, n, n)
        mb, mt = _causal(Lb), _causal(Lt)
        cos_b, sin_b = text_model.rotary_emb(bat, pos_b)
        pe_b = (cos_b.to(bat.dtype), sin_b.to(bat.dtype))
        cos_t, sin_t = text_model.rotary_emb(tl, pos_t)
        pe_tl = (cos_t.to(tl.dtype), sin_t.to(tl.dtype))
        lo_t, hi_t = truncated_masks(keep, cpos)
        k = len(keep)
        kt = torch.tensor(keep, device=dev)
        pos_k = pos.index_select(2, kt)

        def _assemble(hb, ht):
            return torch.cat([ht[0, :a0], hb[:, -1, :], ht[0, a0:]], dim=0)
        caches = []
        hb, ht = bat, tl
        with torch.no_grad(), sdpa_kernel(EFF_SDPA):
            for li in range(trunc):
                caches.append(_assemble(hb, ht).clone())
                hb = layers[li](hb, attention_mask=mb, position_embeddings=pe_b)[0]
                ht = layers[li](ht, attention_mask=mt, position_embeddings=pe_tl)[0]
            h = _assemble(hb, ht).unsqueeze(0)
            lo4 = lo_t.to(dev).to(torch.float32).view(1, 1, k, k)
            hi4 = hi_t.to(dev).to(torch.float32).view(1, 1, k, k)
            cos_k, sin_k = text_model.rotary_emb(h, pos_k)
            pe_k = (cos_k.to(h.dtype), sin_k.to(h.dtype))
            for li in range(trunc, n_layers):
                caches.append(h[0].clone())
                h = layers[li](h, attention_mask=(lo4 if li < LO else hi4),
                               position_embeddings=pe_k)[0]
            hn = text_model.norm(h)
            lg0 = model.lm_head(hn[0, -1].to(model.lm_head.weight.dtype)).float()
        return caches, lg0, lo_t, hi_t, pos_k, pos[:, :, -1:]

    def decode_fast(d, trunc=None, chunked=False):
        """TRUNC fast decode: greedy scratchpad decode where every step runs the layer
        stack on [keep-cols cache || appended rows] only — the drop-frame-kv (+truncate)
        deployment semantics with a real speedup. Returns (parsed, text, toks, prefill_s)."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        tp = time.time()
        caches, lg0, lo_t, hi_t, pos_k, pos_last = (
            prefill_chunked(d, trunc) if chunked else prefill_capture(d, trunc))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        prefill_s = time.time() - tp
        k = pos_k.shape[2]
        toks = []
        with torch.no_grad(), sdpa_kernel(EFF_SDPA):
            for _step in range(args.decode_tokens):
                if _step == 0:
                    lg = lg0
                else:
                    e = len(toks)
                    h_app = text_model.embed_tokens(
                        torch.tensor([toks], device=dev)).to(torch.bfloat16)
                    inc = torch.arange(1, e + 1, device=dev).view(1, 1, e)
                    pos_step = torch.cat([pos_k, pos_last + inc], dim=2)
                    lo_s = _ext_mask(lo_t, e).to(dev).to(torch.float32).view(
                        1, 1, k + e, k + e)
                    hi_s = _ext_mask(hi_t, e).to(dev).to(torch.float32).view(
                        1, 1, k + e, k + e)
                    cos_, sin_ = text_model.rotary_emb(h_app, pos_step)
                    pe_s = (cos_.to(h_app.dtype), sin_.to(h_app.dtype))
                    hh = h_app
                    for li, ly in enumerate(layers):
                        hin = torch.cat([caches[li].unsqueeze(0), hh], dim=1)
                        hout = ly(hin, attention_mask=(lo_s if li < LO else hi_s),
                                  position_embeddings=pe_s)[0]
                        hh = hout[:, k:]
                    hn = text_model.norm(hh)
                    lg = model.lm_head(hn[0, -1].to(model.lm_head.weight.dtype)).float()
                t = int(lg.argmax())
                if t == tok.eos_token_id:
                    break
                toks.append(t)
        text = tok.decode(toks)
        return parse_scratchpad_answer(text, sfmt), text, toks, prefill_s

    # ---- preprocessing (cache per sample; eval-only STREAMS instead) ----
    data = []
    golds_seen = []
    eval_stats = {"n": 0, "raw": 0, "res": 0, "mae": 0.0, "mae_n": 0, "per": {}, "per_task": {}}
    n_done = n_skip = 0
    t0 = time.time()
    roots = [r.strip() for r in args.data_root.split(",") if r.strip()]
    root_done = {r: 0 for r in roots}
    sample_iter = []
    if args.dirs_file:
        dlist = [ln.strip() for ln in Path(args.dirs_file).read_text().splitlines() if ln.strip()]
        roots = ["dirsfile"]; root_done = {"dirsfile": 0}
        sample_iter = [("dirsfile", Path(ln)) for ln in dlist]
        print(f"[dirs-file] {len(dlist)} sample dirs from {args.dirs_file}", flush=True)
    else:
        for root in roots:
            rd = (iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs)
                  if args.shuffle_dirs is not None else iter_sample_dirs(Path(root)))
            sample_iter.extend((root, sd) for sd in rd)
    for root, sd in sample_iter:
        if root_done[root] >= args.limit:
            continue
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
        except Exception:
            n_skip += 1; continue
        pt = parse_task_labels(q0, states, gold)
        if pt is None and args.eval_only and args.alien_task:
            pt = ("steps", set(), None)     # unknown question type: decode-only scoring
        if pt is None:
            n_skip += 1; continue
        task, evid, aux = pt
        if gold > 9 and not args.eval_only:
            n_skip += 1; continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]

        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": CARRIER_TOKEN})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k2: (v.to(dev) if hasattr(v, "to") else v) for k2, v in inputs.items()}
        ids = inputs["input_ids"][0].tolist(); seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF, processor=processor)
        cpos = [p for p, t in enumerate(ids) if t == cid]
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        occ = None
        for pre in ("", " ", "\n"):
            needle = tok(pre + q0, add_special_tokens=False).input_ids
            o = find_subseq(ids, needle)
            if len(o) == 2:
                occ = o; break
        if len(fg) != NF or len(cpos) != NF or len(vstarts) != NF or occ is None:
            n_skip += 1; continue
        fin_start = occ[1]
        blocks = [(vstarts[i], (vstarts[i + 1] if i + 1 < NF else fin_start)) for i in range(NF)]

        mask_lo = mask_hi = None
        if n_done == 0 or args.eval_only:
            mask_lo, mask_hi = make_masks(seq, blocks, cpos, fin_start)
        if n_done == 0:
            def _alw(mm, r):
                return int((mm[r] == 0).sum())
            blk_last = [_alw(mask_lo, b - 1) for a, b in blocks]
            print(f"[mask-debug] seq={seq} lo allowed-keys @last-row-per-block: {blk_last} "
                  f"identical={len(set(blk_last)) == 1}; carrier rows lo "
                  f"{[_alw(mask_lo, c) for c in cpos]}; hi carrier rows "
                  f"{[_alw(mask_hi, c) for c in cpos]} (must be lo+0..+{NF-1}); "
                  f"tail last-row lo {_alw(mask_lo, seq-1)} hi {_alw(mask_hi, seq-1)} "
                  f"(hi-lo must be {NF})", flush=True)

        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            if args.no_reset_positions:
                # posreset-necessity ablation (2026-07-27): natural M-RoPE positions, i.e. block i
                # sits at its true sequence offset instead of block 0's range. base_pos is already
                # collision-free and ordered, so the carrier-position override and the tail +NF
                # shift below (which exist only to de-collide the RESET layout) are skipped.
                pos = base_pos.clone()
                if n_done == 0:
                    print(f"[pos-debug] seq={seq} NO-RESET arm: natural positions, "
                          f"max_pos {int(pos.max())}, block starts "
                          f"{[int(base_pos[0, 0, a]) for a, _ in blocks]}", flush=True)
            else:
                pos = reset_positions(base_pos, blocks, fin_start).clone()
            if n_done == 0 and not args.no_reset_positions:
                s0, e0 = blocks[0]
                same_len = all(b - a == e0 - s0 for a, b in blocks)
                blocks_eq = same_len and all(
                    torch.equal(pos[:, :, a:b], pos[:, :, s0:e0]) for a, b in blocks[1:])
                print(f"[pos-debug] seq={seq} max_pos {int(base_pos.max())} -> "
                      f"{int(pos.max())}, same_len={same_len}, blocks_identical={blocks_eq} "
                      f"(pre-carrier-override; carriers then get sequential positions)",
                      flush=True)
            if not args.no_reset_positions:
                blk0_max = int(pos[:, :, blocks[0][0]:blocks[0][1]].max())
                for i, c in enumerate(cpos):       # carriers: sequential ordered positions
                    pos[:, :, c] = blk0_max + 1 + i
                pos[:, :, fin_start:] += NF        # tail re-based after the carrier run
            emb = text_model.embed_tokens(inputs["input_ids"])
            img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb = emb.clone(); emb[0, im_mask] = img.to(emb.dtype)

        rec = {"emb": emb[0].to(torch.bfloat16), "pos": pos, "cpos": cpos,
               "blocks": blocks, "fin": fin_start,
               "seq": seq, "gold": gold, "task": task, "grp": f"{task}{NF}"}
        golds_seen.append(gold)
        if args.eval_only:
            rec["lo"] = mask_lo.to(torch.float16)
            rec["hi"] = mask_hi.to(torch.float16)
            if trunc_flags:
                rec["keep"] = keep_cols(seq, blocks, cpos)
                rec["fcols"] = frame_cols(seq, blocks, cpos)
                if n_done == 0:
                    ks, fs = set(rec["keep"]), set(rec["fcols"])
                    assert not (ks & fs) and len(ks) + len(fs) == seq, "keep/frame split"
                    print(f"[trunc-debug] seq={seq} keep={len(ks)} frame_cols={len(fs)} "
                          f"trunc_at={args.truncate_at} dropkv={args.drop_frame_kv} "
                          f"fast={args.fast_decode} exact={args.exactness_check}",
                          flush=True)
            if dump_layers:
                with torch.no_grad():
                    cD = prefill_capture(rec, args.truncate_at)
                a0d = rec["blocks"][0][0]
                ds = eval_stats.setdefault("dump", {"rep": {L: [] for L in dump_layers},
                                                    "labels": [], "gold": [], "sd": []})
                for L in dump_layers:
                    ds["rep"][L].append(
                        cD[0][L][a0d:a0d + NF].float().cpu().numpy().astype(np.float16))
                ds["labels"].append(
                    np.array([1 if t in evid else 0 for t in range(NF)], dtype=np.int64))
                ds["gold"].append(gold); ds["sd"].append(str(sd))
                del cD
            with torch.no_grad():
                if scratch:
                    if args.exactness_check:
                        cuda = torch.cuda.is_available()

                        def _arm(fn):
                            if cuda:
                                torch.cuda.reset_peak_memory_stats()
                                torch.cuda.synchronize()
                            ta = time.time()
                            r = fn()
                            if cuda:
                                torch.cuda.synchronize()
                            pk = (torch.cuda.max_memory_allocated() / 2**30) if cuda else 0.0
                            return r, time.time() - ta, pk
                        (vB, xB, kB), tB, pB = _arm(lambda: decode_scratchpad(rec))
                        (vM, xM, kM), tM, _pM = _arm(lambda: decode_scratchpad(
                            rec, dropkv=True, trunc=args.truncate_at))
                        ex = eval_stats.setdefault("exact", {
                            "n": 0, "ident": 0, "ans": 0, "fident": 0, "fans": 0,
                            "nf": 0, "tB": 0.0, "tM": 0.0, "tF": 0.0, "pB": 0.0, "pF": 0.0})
                        ex["n"] += 1; ex["ident"] += int(kB == kM)
                        ex["ans"] += int(vB == vM)
                        ex["tB"] += tB; ex["tM"] += tM; ex["pB"] = max(ex["pB"], pB)
                        line = (f"  [exact] gold={gold} N={NF} base({vB},{len(kB)}t,"
                                f"{tB:.1f}s) mask({vM},{len(kM)}t,{tM:.1f}s) "
                                f"ident={kB == kM}")
                        if kB != kM:
                            div = next((i for i, (a, b) in enumerate(zip(kB, kM))
                                        if a != b), min(len(kB), len(kM)))
                            line += f" FIRST-DIV@{div}"
                        if args.fast_decode:
                            (vF, xF, kF, pf_s), tF, pF = _arm(lambda: decode_fast(
                                rec, trunc=args.truncate_at))
                            ex["nf"] += 1; ex["fident"] += int(kF == kM)
                            ex["fans"] += int(vF == vM)
                            ex["tF"] += tF; ex["pF"] = max(ex["pF"], pF)
                            line += (f" fast({vF},{len(kF)}t,{tF:.1f}s,pf{pf_s:.1f}s) "
                                     f"fast==mask={kF == kM}")
                        print(line, flush=True)
                        val, txt, ndec = vB, xB, len(kB)   # eval_stats scores BASELINE
                        fd = None
                        eval_stats["dec_toks"] = eval_stats.get("dec_toks", 0) + ndec
                    elif args.verify_chunked:
                        cA = prefill_capture(rec, args.truncate_at)
                        cB = prefill_chunked(rec, args.truncate_at)
                        hA = cA[0][args.truncate_at].float()
                        hB = cB[0][args.truncate_at].float()
                        a0v = rec["blocks"][0][0]
                        NFv = len(rec["cpos"])
                        d_q = float((hA[:a0v] - hB[:a0v]).abs().max())
                        d_c = float((hA[a0v:a0v + NFv] - hB[a0v:a0v + NFv]).abs().max())
                        d_t = float((hA[a0v + NFv:] - hB[a0v + NFv:]).abs().max())
                        for lix in (0, 1, 2, 6, 11, args.truncate_at):
                            xa, xb = cA[0][lix].float(), cB[0][lix].float()
                            rowd = (xa - xb).abs().max(dim=1).values
                            wr = int(rowd.argmax())
                            wd = int((xa[wr] - xb[wr]).abs().argmax())
                            rel = (xa - xb).abs() / (xa.abs() + 1.0)
                            print(f"    [chunkdbg L{lix}] dq={float(rowd[:a0v].max()):.4f}"
                                  f" dc={float(rowd[a0v:a0v+NFv].max()):.4f}"
                                  f" dt={float(rowd[a0v+NFv:].max()):.4f}"
                                  f" worst-row={wr}/{rowd.numel()} |h[wr,wd]|="
                                  f"{float(xa[wr, wd].abs()):.0f} rel_q="
                                  f"{float(rel[:a0v].max()):.5f} rel_c="
                                  f"{float(rel[a0v:a0v+NFv].max()):.5f}", flush=True)
                        del cA, cB
                        vD, xD, kD, _ = decode_fast(rec, trunc=args.truncate_at)
                        val, txt, dtoks, _pf = decode_fast(rec, trunc=args.truncate_at,
                                                           chunked=True)
                        cv = eval_stats.setdefault("chunkv", {
                            "n": 0, "ident": 0, "ans": 0, "dq": 0.0, "dc": 0.0, "dt": 0.0})
                        cv["n"] += 1; cv["ident"] += int(kD == dtoks)
                        cv["ans"] += int(vD == val)
                        cv["dq"] = max(cv["dq"], d_q); cv["dc"] = max(cv["dc"], d_c)
                        cv["dt"] = max(cv["dt"], d_t)
                        print(f"  [chunkverify] gold={gold} N={NF} dq={d_q:.4f} "
                              f"dc={d_c:.4f} dt={d_t:.4f} h_std={float(hA.std()):.3f} "
                              f"tok-ident={kD == dtoks} dense={vD} chunked={val}",
                              flush=True)
                        fd = None
                        ndec = len(dtoks)
                        eval_stats["dec_toks"] = eval_stats.get("dec_toks", 0) + ndec
                    elif args.fast_decode:
                        cuda = torch.cuda.is_available()
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        tf0 = time.time()
                        val, txt, dtoks, pf_s = decode_fast(rec, trunc=args.truncate_at,
                                                            chunked=args.chunked_prefill)
                        if cuda:
                            torch.cuda.synchronize()
                        ft = eval_stats.setdefault("fastt", {"pf": 0.0, "dec": 0.0,
                                                             "tok": 0, "vram": 0.0})
                        ft["pf"] += pf_s
                        ft["dec"] += time.time() - tf0 - pf_s
                        ft["tok"] += len(dtoks)
                        if cuda:
                            ft["vram"] = max(ft["vram"],
                                             torch.cuda.max_memory_allocated() / 2**30)
                        fd = None
                        ndec = len(dtoks)
                        eval_stats["dec_toks"] = eval_stats.get("dec_toks", 0) + ndec
                    else:
                        val, txt, dtoks = decode_scratchpad(
                            rec,
                            dropkv=(args.drop_frame_kv or args.truncate_at is not None),
                            trunc=args.truncate_at)
                        fd = None
                        ndec = len(dtoks)
                        eval_stats["dec_toks"] = eval_stats.get("dec_toks", 0) + ndec
                    if n_done < args.dump_decodes:
                        print(f"  [decode-sample] gold={gold} parsed={val} text={txt!r}",
                              flush=True)
                else:
                    val, fd = decode_answer(
                        rec,
                        dropkv=(args.drop_frame_kv or args.truncate_at is not None)
                        if trunc_flags else False,
                        trunc=args.truncate_at)
            es = eval_stats
            es["n"] += 1
            es["raw"] += int(val == gold)
            # 2nd column: digit mode = 0-9-restricted hits; scratchpad mode = parse FAILS
            es["res"] += (int(val is None) if scratch else int(fd == gold))
            if val is not None:
                es["mae"] += abs(val - gold); es["mae_n"] = es.get("mae_n", 0) + 1
            elif fd is not None:
                es["mae"] += abs(fd - gold); es["mae_n"] = es.get("mae_n", 0) + 1
            pg = es["per"].setdefault(gold, [0, 0, 0])
            pg[0] += int(val == gold)
            pg[1] += (int(val is None) if scratch else int(fd == gold))
            pg[2] += 1
            tt = es["per_task"].setdefault(task, [0, 0])
            tt[0] += int(val == gold); tt[1] += 1
        else:
            rec["emb"] = rec["emb"].cpu(); rec["pos"] = pos.cpu()
            data.append(rec)
        n_done += 1
        root_done[root] += 1
        if n_done % 25 == 0:
            print(f"  prep {n_done} (skip {n_skip}) {time.time()-t0:.0f}s", flush=True)
    print(f"prep done: n={n_done} skip={n_skip} ({time.time()-t0:.0f}s)", flush=True)
    hist = {}
    for g in golds_seen:
        hist[g] = hist.get(g, 0) + 1
    print("[gold-hist] " + " ".join(f"g{g}:{c}" for g, c in sorted(hist.items())), flush=True)
    if not args.eval_only and len(roots) > 1:      # pooled data: per-(task,N) gold hist
        gh = {}
        for d in data:
            gh.setdefault(d["grp"], {})
            gh[d["grp"]][d["gold"]] = gh[d["grp"]].get(d["gold"], 0) + 1
        for gname in sorted(gh):
            print(f"[gold-hist {gname}] n={sum(gh[gname].values())}  "
                  + " ".join(f"g{g}:{c}" for g, c in sorted(gh[gname].items())), flush=True)

    if args.eval_only:
        es = eval_stats
        n = max(es["n"], 1)
        pc = " ".join(f"g{g}:{a}/{b}/{t}" for g, (a, b, t) in sorted(es["per"].items()))
        pt_s = " ".join(f"{t}:{h}/{n2}" for t, (h, n2) in sorted(es["per_task"].items()))
        col2 = ("parse-FAIL rate" if scratch else "0-9-restricted acc")
        mae_n = max(es.get("mae_n", n), 1)
        lines = [f"=== CARRIER LAYER EVAL-ONLY (ckpt={args.ckpt}, n={es['n']}, "
                 f"decode<={args.decode_tokens}, scratchpad={scratch}, fmt={sfmt}, "
                 f"data={args.dirs_file or args.data_root}) ===",
                 "[gold-hist] " + " ".join(f"g{g}:{c}" for g, c in sorted(hist.items())),
                 f"emitted RAW (greedy-parse) acc {es['raw']/n:.3f}   "
                 f"{col2} {es['res']/n:.3f}   MAE {es['mae']/mae_n:.2f} (over {mae_n} parsed)",
                 f"per-count raw/{'parsefail' if scratch else 'restricted'}/n: {pc}",
                 f"per-task raw: {pt_s}"]
        if scratch:
            lines.append(f"mean decode tokens {es.get('dec_toks', 0)/n:.1f} "
                         f"(cap {args.decode_tokens})")
        if trunc_flags:
            lines.append(f"[trunc-flags] dropkv={args.drop_frame_kv} "
                         f"truncate_at={args.truncate_at} fast={args.fast_decode} "
                         f"exact={args.exactness_check}")
        if "exact" in es:
            ex = es["exact"]
            lines.append(
                f"[exactness] mask-only identical {ex['ident']}/{ex['n']}, "
                f"answer-equal {ex['ans']}/{ex['n']}"
                + (f"; fast==mask {ex['fident']}/{ex['nf']}, "
                   f"fast answer-equal {ex['fans']}/{ex['nf']}" if ex["nf"] else ""))
            lines.append(
                f"[timing] decode s/sample: base {ex['tB']/ex['n']:.1f} "
                f"mask {ex['tM']/ex['n']:.1f}"
                + (f" fast {ex['tF']/ex['nf']:.1f} -> speedup base/fast "
                   f"{(ex['tB']/ex['n'])/max(ex['tF']/ex['nf'], 1e-9):.1f}x"
                   if ex["nf"] else ""))
            lines.append(f"[vram] peak GiB: base {ex['pB']:.1f} fast {ex['pF']:.1f}")
        if "chunkv" in es:
            cv = es["chunkv"]
            lines.append(
                f"[chunkverify] tok-ident {cv['ident']}/{cv['n']}, answer-equal "
                f"{cv['ans']}/{cv['n']}; max|dh_L*| question {cv['dq']:.4f} carriers "
                f"{cv['dc']:.4f} tail {cv['dt']:.4f}")
        if "fastt" in es:
            ft = es["fastt"]
            dec_s = ft["dec"] / n
            lines.append(
                f"[fast-timing] prefill {ft['pf']/n:.2f} s/sample, decode {dec_s:.2f} "
                f"s/sample, {ft['tok']/max(ft['dec'], 1e-9):.1f} tok/s, peak VRAM "
                f"{ft['vram']:.1f} GiB (chunked={args.chunked_prefill})")
        if "dump" in es:
            ds = es["dump"]
            torch.save({"rep": {L: np.stack(v) for L, v in ds["rep"].items()},
                        "labels": np.stack(ds["labels"]),
                        "gold": np.array(ds["gold"], dtype=np.int64),
                        "sd": ds["sd"],
                        "ckpt": args.ckpt, "truncate_at": args.truncate_at},
                       out / "carrier_states_cache.pt")
            lines.append(f"[dump] carrier_states_cache.pt: n={len(ds['gold'])} layers="
                         f"{sorted(ds['rep'])} (replica_gate_tally-compatible)")
            (out / "report.txt").write_text("\n".join(lines) + "\n")
            print(lines[-1])
        (out / "report.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines)); print("wrote", out)
        for h in handles:
            h.remove()
        return 0

    rng = np.random.default_rng(args.seed)
    grps_present = sorted({d["grp"] for d in data})
    if len(grps_present) > 1:      # mixture: stratified train/eval split per (task,N) group
        frac = args.train_n / max(n_done, 1)
        tr_l, ev_l = [], []
        for t in grps_present:
            tidx = np.array([i for i, d in enumerate(data) if d["grp"] == t])
            rng.shuffle(tidx)
            k = int(round(frac * len(tidx)))
            tr_l.append(tidx[:k]); ev_l.append(tidx[k:])
        tr_idx, ev_idx = np.concatenate(tr_l), np.concatenate(ev_l)
        rng.shuffle(tr_idx)
        print(f"[split] stratified by group: train {len(tr_idx)} eval {len(ev_idx)} "
              f"groups={grps_present}", flush=True)
    else:
        order = rng.permutation(n_done)
        tr_idx, ev_idx = order[:args.train_n], order[args.train_n:]

    def evaluate():
        hits = raw_hits = 0; mae = 0.0
        per = {}; ptask = {}
        with torch.no_grad():
            for i in ev_idx:
                lg = forward_logits(data[i], False)
                dg = int(np.argmax([float(lg[t]) for t in digit_ids]))
                g = data[i]["gold"]
                hits += (dg == g); mae += abs(dg - g)
                raw_hits += (int(lg.argmax()) == digit_ids[g])
                per.setdefault(g, [0, 0]); per[g][1] += 1; per[g][0] += (dg == g)
                tt = ptask.setdefault(data[i]["grp"], [0, 0])
                tt[1] += 1; tt[0] += (dg == g)
        n = len(ev_idx)
        pc = " ".join(f"g{g}:{c}/{t}" for g, (c, t) in sorted(per.items()))
        if len(ptask) > 1:
            pc += "  [" + " ".join(f"{t}:{c}/{n2}" for t, (c, n2)
                                   in sorted(ptask.items())) + "]"
        return hits / n, raw_hits / n, mae / n, pc

    acc0, raw0, mae0, pc0 = evaluate()
    print(f"[ep 0] emitted acc {acc0:.3f} (raw-argmax {raw0:.3f}, MAE {mae0:.2f})  {pc0}", flush=True)
    lines = [f"=== CARRIER LAYER (L_open={LO}, rank={args.rank}, nolora={args.no_lora}, "
             f"init={'distill' if args.carrier_init else 'room'}, n={n_done}, "
             f"train={len(tr_idx)}) ===",
             f"ep0 acc {acc0:.3f} raw {raw0:.3f} mae {mae0:.2f}  {pc0}"]
    best = (acc0, 0)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        tot = 0.0
        for step, i in enumerate(tr_idx):
            lg = forward_logits(data[i], True)
            loss = F.cross_entropy(lg.unsqueeze(0),
                                   torch.tensor([digit_ids[data[i]["gold"]]], device=dev))
            (loss / 4).backward()
            tot += float(loss)
            if (step + 1) % 4 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        acc, raw, mae, pc = evaluate()
        print(f"[ep {ep}] loss {tot/len(tr_idx):.4f}  emitted acc {acc:.3f} "
              f"(raw {raw:.3f}, MAE {mae:.2f})  {pc}", flush=True)
        lines.append(f"ep{ep} loss {tot/len(tr_idx):.4f} acc {acc:.3f} raw {raw:.3f} mae {mae:.2f}")
        if acc > best[0]:
            best = (acc, ep)
            torch.save({"e_c": e_c.detach().cpu(),
                        "lora": {f"{li}.{nm}": (A.detach().cpu(), B.detach().cpu())
                                 for (li, nm), (A, B) in lora.items()},
                        "l_open": LO, "rank": args.rank, "alpha": args.alpha,
                        "epoch": ep, "acc": acc},
                       out / "carrier_layer_best.pt")
    lines.append(f"BEST emitted acc {best[0]:.3f} @ ep {best[1]}  "
                 f"(scaffold ceiling 0.991; frozen baseline 0.207)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:])); print("wrote", out)
    for h in handles:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
