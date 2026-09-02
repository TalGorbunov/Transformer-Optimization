#!/usr/bin/env python3
"""SPARSE — gated fenced-SFT: the LORAMECH P1 trainer + the evidence gate + sharpness.

Extension of scripts/loramech/train_sft_fenced.py (copied per the SPARSE brief §5 —
all deltas live in scripts/sparse/; the loramech original is untouched and stays the
anchor). NO-FLAG BEHAVIOR IS THE ANCHOR CONTRACT: with none of the new flags this
script must reproduce the P1b numbers exactly (verified: eval-only P1b on
exam_ff_N8 = 1.000 before any full run).

SPARSE deltas (brief `outputs/sparse/CAMPAIGN_BRIEF.md`):
  --gate oracle          hide_cols := union of the spans of all NON-evidence blocks
                         (gnnformer.data.probe_evidence labels, |evid| == k asserted,
                         skip+count on mismatch), applied in TRAINING and EVAL
                         forwards. The read code becomes k/(k+C): N never enters.
  --gate-from-layer G    S2(b) one-forward variant: plain fence at layers < G, gated
                         fence at layers >= G (two FenceHooks over layer subsets —
                         gnnformer untouched). Default -1 = gate at ALL layers.
  --attn-sharpen TAU     S0: fixed attention-logit sharpening — scaling =
                         head_dim^-0.5 * TAU on the LM decoder attn modules...
  --sharpen-from-layer L ...restricted to module index >= L (default 0 = all layers).
                         Mutually exclusive with --attn-logn-sref.

Layout, recipe, decode: byte-identical to the loramech trainer (N x [frame_i + q]
blocks, block fence + posreset, count prompt, r8 a32 q/k/v/o+MLP lr 2e-4 accum 8,
answer-token loss, cache-free greedy decode, 4-D mask => EFFICIENT/MATH sdpa).

Usage (S1):
  python scripts/sparse/train_sft_gated.py --gate oracle --epochs 10 \
      --data_root <seq8>,<seq16> --exclude-dirs-file <exams...> --output outputs/sparse/s1
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import ROOMS  # noqa: E402
from gnnformer.data import (  # noqa: E402
    build_count_prompt,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    parse_task_labels,
    probe_evidence,
    read_dirs_file,
    rooms_to_room2chars,
)
from gnnformer.fencing import (  # noqa: E402
    FenceHooks,
    build_block_mask,
    find_subseq,
    frame_blocks,
    locate_word_token,
    reset_positions,
)
from gnnformer.metrics import format_gold_histogram  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402

_REDUX = _REPO / "scripts" / "redux"
if str(_REDUX) not in sys.path:
    sys.path.insert(0, str(_REDUX))
from tasks import (  # noqa: E402  (REDUX label module — count path stays byte-identical)
    build_prompt as task_prompt,
    derive_gold,
    evidence_count,
    parse_answer,
    replica_text,
    target_of,
    task_of_dirs_file,
)

FENCED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]  # 4-D mask: no FLASH
INTEGER_RE = re.compile(r"[+-]?\d+")
MAX_NEW = 4


def build_nfree_prompt(question):
    """S9: the N-free count prompt — minimal edit of build_count_prompt with the
    frame count and answer range removed (same opener, so parse_layout's needle
    is unchanged). Matches the upstream benchmark's no-N protocol in spirit."""
    return (
        "You will be shown a sequence of frames describing steps in a house.\n"
        "Respond with a single integer (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_fenced_messages(frames, question, answer=None, declare_n=None, nfree=False):
    """[frame_0, q, frame_1, q, ..., frame_{N-1}, q, count_prompt] (+ assistant answer).

    declare_n (WAVE 3, S7/S8): overrides ONLY the frame count used in the prompt
    TEXT ("You will be shown {N} frames ... 0 to {N}"); frames, mask, gate and
    positions are untouched. None = byte-identical legacy behavior.
    """
    content = []
    for im in frames:
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": question})
    content.append({"type": "text",
                    "text": (build_nfree_prompt(question) if nfree else
                             build_count_prompt(question, declare_n or len(frames)))})
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
    return msgs


def build_task_messages(frames, task, char, room, q0, answer=None, declare_n=None,
                        nfree=False):
    """REDUX task-aware fenced layout. count == build_fenced_messages byte-identical."""
    if task == "count":
        return build_fenced_messages(frames, q0, answer=answer, declare_n=declare_n,
                                     nfree=nfree)
    nf = len(frames)
    content = []
    for im in frames:
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": replica_text(task, char, room, nf)})
    content.append({"type": "text", "text": task_prompt(task, char, room, nf)})
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
    return msgs


def parse_layout(ids, tok, q0, nf, vs_id):
    """-> (blocks, fin_start) or None. (Unchanged from the loramech trainer.)"""
    vstarts = [p for p, t in enumerate(ids) if t == vs_id]
    if len(vstarts) != nf:
        return None
    fin_start = None
    for pre in ("", " ", "\n"):
        needle = tok(pre + "You will be shown", add_special_tokens=False).input_ids
        occ = find_subseq(ids, needle)
        if len(occ) == 1:
            fin_start = occ[0]
            break
    if fin_start is None or fin_start <= vstarts[-1] + 1:
        return None
    return frame_blocks(vstarts, fin_start), fin_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-cap", type=int, default=60)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-dirs-file", action="append", default=[])
    ap.add_argument("--eval-longn-limit", type=int, default=150)
    ap.add_argument("--exclude-dirs-file", action="append", default=[])
    ap.add_argument("--eval-only-adapter", type=Path, default=None)
    ap.add_argument("--tasks", default="count")
    ap.add_argument("--attn-logn-sref", type=int, default=0,
                    help="L5 log-N logit scaling (loramech contract; exclusive with "
                         "--attn-sharpen)")
    # ---- SPARSE deltas
    ap.add_argument("--gate", choices=["none", "oracle", "model"], default="none",
                    help="oracle: hide_cols = spans of all non-evidence blocks "
                         "(train AND eval forwards). Eval contract: an adapter "
                         "trained with the gate is evaluated with the gate. "
                         "model (S2a, eval-only): two-forward — ungated pass 1, "
                         "logistic gate (--gate-npz) on the replica-slot states, "
                         "gated decode; per-sample gate errors logged.")
    ap.add_argument("--gate-npz", type=Path, default=None,
                    help="train_gate.py output (w,b,mu,sd,layer) for --gate model")
    ap.add_argument("--gate-from-layer", type=int, default=-1,
                    help="S2b: apply the gated mask only at decoder layers >= this "
                         "index (plain fence below). -1 = gate at all layers.")
    ap.add_argument("--attn-sharpen", type=float, default=0.0,
                    help="S0: fixed sharpening factor TAU on attention scaling "
                         "(0 = off).")
    ap.add_argument("--declare-n", type=int, default=0,
                    help="S7: override the frame count in the prompt TEXT only "
                         "(eval path; 0 = off). Frames/mask/gate/positions untouched.")
    ap.add_argument("--virtual-n", action="store_true",
                    help="S8: training mixture = 50%% real gated samples + 50%% "
                         "synthetic evidence-only samples (k in 0..16 stratified x "
                         "declared N in {8,16,32,64,128}, k<=N; answer=k). Requires "
                         "--gate oracle. Eval paths unchanged.")
    ap.add_argument("--nfree-prompt", action="store_true",
                    help="S9: N-free count prompt (no frame count, no answer range "
                         "in the text) in TRAIN and EVAL. Adapter contract: an "
                         "adapter trained with this flag is evaluated with it.")
    ap.add_argument("--sharpen-from-layer", type=int, default=0,
                    help="S0: apply --attn-sharpen only to decoder modules with "
                         "index >= this (default 0 = all layers).")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/sparse/s1_gated"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.attn_sharpen > 0 and args.attn_logn_sref > 0:
        raise SystemExit("--attn-sharpen and --attn-logn-sref are mutually exclusive")
    if args.gate != "none" and args.tasks != "count":
        raise SystemExit("--gate requires --tasks count (evidence labels are count's)")
    if args.gate == "model":
        if args.gate_npz is None or args.eval_only_adapter is None:
            raise SystemExit("--gate model requires --gate-npz and --eval-only-adapter")
        if args.gate_from_layer >= 0:
            raise SystemExit("--gate model is full-depth only (no --gate-from-layer)")
    if args.virtual_n and (args.gate != "oracle" or args.tasks != "count"):
        raise SystemExit("--virtual-n requires --gate oracle and --tasks count")
    if args.nfree_prompt and args.declare_n:
        raise SystemExit("--nfree-prompt excludes --declare-n (no N in text)")
    # S9b: --nfree-prompt + --virtual-n = k-balanced evidence-subset oversampling
    # (the S8 coverage trick with the N~ dimension collapsed away).
    if args.nfree_prompt and args.tasks != "count":
        raise SystemExit("--nfree-prompt is count-only")
    gate_npz = None
    if args.gate_npz is not None:
        g_ = np.load(args.gate_npz)
        gate_npz = dict(w=g_["w"].astype(np.float32), b=float(g_["b"]),
                        mu=g_["mu"].astype(np.float32), sd=g_["sd"].astype(np.float32),
                        layer=int(g_["layer"]))

    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_gated"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (run_dir / "run.log").open("w", encoding="utf-8")

    def emit(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    # attn modules captured PRE-wrap (the 137800 PeftModel lesson)
    attn_mods = [ly.self_attn for ly in layers]
    base_scaling = float(attn_mods[0].scaling)

    def set_logn_scaling(seq_len):
        import math

        s = base_scaling * math.log(max(seq_len, 2)) / math.log(args.attn_logn_sref)
        for m_ in attn_mods:
            m_.scaling = s

    if args.attn_sharpen > 0:  # S0: static, set once (no per-seq dependence)
        for i_, m_ in enumerate(attn_mods):
            if i_ >= args.sharpen_from_layer:
                m_.scaling = base_scaling * args.attn_sharpen
        emit(f"[sharpen] tau={args.attn_sharpen} on layers >= {args.sharpen_from_layer} "
             f"({sum(1 for i_ in range(len(attn_mods)) if i_ >= args.sharpen_from_layer)}"
             f"/{len(attn_mods)} modules; base_scaling={base_scaling:.6g})")
    if args.eval_only_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.eval_only_adapter), is_trainable=False)
        emit(f"eval-only: adapter loaded from {args.eval_only_adapter}")
        args.epochs = 0
    elif args.epochs == 0:
        emit("frozen eval: no adapter, no LoRA")
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    emit("[template] FENCED frames-first: N x [frame+q] blocks + posreset + count prompt")
    emit(f"[gate] {args.gate}"
         + (f" from layer {args.gate_from_layer}" if args.gate_from_layer >= 0 else ""))

    # ---- data (baseline trainer's conventions: validity gate, exclusion, seeded split)
    excluded = set()
    for f in args.exclude_dirs_file:
        excluded.update(str(Path(p).resolve()) for p in read_dirs_file(Path(f)))
    if excluded:
        emit(f"[exclude] {len(excluded)} dirs across {len(args.exclude_dirs_file)} file(s)")
    samples, n_skip, n_excl = [], 0, 0
    for root in args.data_root.split(","):
        root = root.strip()
        if not root:
            continue
        lim = args.limit
        if "=" in root:
            root, lim = root.rsplit("=", 1)
            lim = int(lim)
        n_root = 0
        for sd in iter_sample_dirs_shuffled(Path(root), 0):
            if n_root >= lim:
                break
            if excluded and str(Path(sd).resolve()) in excluded:
                n_excl += 1
                continue
            try:
                _sid, _frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
            except Exception:
                n_skip += 1
                continue
            if parse_task_labels(q0, states, gold) is None:
                n_skip += 1
                continue
            samples.append((sd, gold))
            n_root += 1
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    assert all(t in ("count", "exists", "majority") for t in task_list), task_list
    samples = [(sd, gold, task_list[i % len(task_list)])
               for i, (sd, gold) in enumerate(samples)]
    emit(f"[tasks] {task_list}; per-task "
         + str({t: sum(1 for s in samples if s[2] == t) for t in task_list}))
    emit(f"[data] {len(samples)} samples (skip {n_skip}, excluded {n_excl}); gold-hist "
         + format_gold_histogram(g for _sd, g, _t in samples))
    split_rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(len(samples))
    n_tr = int(len(samples) * args.train_frac)
    n_va = int(len(samples) * args.val_frac)
    tr_idx, va_idx, te_idx = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    (run_dir / "train_dirs.txt").write_text("\n".join(str(samples[i][0]) for i in tr_idx) + "\n")
    (run_dir / "eval_dirs.txt").write_text("\n".join(str(samples[i][0]) for i in te_idx) + "\n")
    emit(f"[split] train={len(tr_idx)} val={len(va_idx)} test_iid={len(te_idx)}")

    # ---- S8 virtual-N synthetic mixture (train-split only; eval untouched)
    syn_specs = []
    if args.virtual_n:
        if args.nfree_prompt:  # S9b: no N~ dimension — pure k coverage
            grid = [(k, None) for k in range(0, 17)]
        else:
            grid = [(k, nn) for nn in (8, 16, 32, 64, 128) for k in range(0, 17) if k <= nn]
        srng = np.random.default_rng(args.seed + 1)
        for j in range(len(tr_idx)):
            k, nn = grid[j % len(grid)]
            cands = [int(i) for i in tr_idx if samples[i][1] >= k]
            if not cands:
                continue
            syn_specs.append((int(srng.choice(cands)), k, nn))
        emit(f"[virtual-n] {len(syn_specs)} synthetic specs over {len(grid)} (k,N~) "
             "cells; per-N~ "
             + str({nn: sum(1 for _s, _k, n2 in syn_specs if n2 == nn)
                    for nn in sorted({n2 for _s, _k, n2 in syn_specs},
                                     key=lambda x: (x is None, x))})
             + "; k-hist "
             + format_gold_histogram(k for _s, k, _n in syn_specs))

    # ---- hooks: single set (anchor path) or split at --gate-from-layer (S2b)
    gate_split = args.gate_from_layer if (args.gate == "oracle"
                                          and args.gate_from_layer >= 0) else None
    if gate_split is None:
        hooks_all = FenceHooks(layers).install()
        hooks_lo = hooks_hi = None
    else:
        hooks_all = None
        hooks_lo = FenceHooks(layers[:gate_split]).install()
        hooks_hi = FenceHooks(layers[gate_split:]).install()

    def set_masks(masks):
        if gate_split is None:
            hooks_all.set_mask(masks[0], rt.device)
        else:
            hooks_lo.set_mask(masks[0], rt.device)
            hooks_hi.set_mask(masks[1], rt.device)

    def clear_masks():
        for h in (hooks_all, hooks_lo, hooks_hi):
            if h is not None:
                h.clear_mask()

    def remove_hooks():
        for h in (hooks_all, hooks_lo, hooks_hi):
            if h is not None:
                h.remove()

    n_layout_skip = 0

    def oracle_evid(q0, states, char, room):
        """Frame-index evidence set; None when the gate is off. For --gate oracle it
        drives the mask; for --gate model it is the REFERENCE for gate-error
        accounting only (predict recomputes its own set). |evid| == k asserted
        (skip+count on mismatch, per the brief)."""
        if args.gate == "none":
            return None
        k = evidence_count(states, char, room)
        evid = {t for t, st in enumerate(states)
                if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
        if len(evid) != k:
            raise ValueError(f"oracle gate: |evid|={len(evid)} != k={k}")
        # cross-check vs the canonical label fn where its room list applies (park);
        # HF pools have rooms outside constants.ROOMS (e.g. Hallway) -> pe is None
        pe = probe_evidence("steps", q0, states, k, ROOMS)
        if pe is not None and set(pe[0]) != evid:
            raise ValueError("oracle gate: probe_evidence disagrees with occupancy set")
        return evid

    def load_frames(sd):
        """-> (frames, q0, char, room, states). Raises on missing target."""
        _sid, frames, q0, states, _a0 = load_mmred_sample(sd)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        tr = target_of(Path(str(sd)), q0)
        if tr is None:
            raise ValueError("no target character/room")
        return frames, q0, tr[0], tr[1], states

    def fenced_setup(inp, q0, nf, evid):
        """-> (masks tuple, pos) or None on layout-parse miss.
        masks = (mask,) on the single-hook path, (plain, gated) on the split path."""
        ids = inp["input_ids"][0].tolist()
        parsed = parse_layout(ids, tok, q0, nf, vs_id)
        if parsed is None:
            return None
        blocks, fin_start = parsed
        hide = []
        if evid is not None:
            for t, (a, b) in enumerate(blocks):
                if t not in evid:
                    hide.extend(range(int(a), int(b)))
        if gate_split is None:
            masks = (build_block_mask(len(ids), blocks, hide_cols=hide),)
        else:
            masks = (build_block_mask(len(ids), blocks, hide_cols=[]),
                     build_block_mask(len(ids), blocks, hide_cols=hide))
        with torch.no_grad():
            base_pos, _ = rope_fn(inp["input_ids"], image_grid_thw=inp.get("image_grid_thw"),
                                  attention_mask=inp.get("attention_mask"))
        return masks, reset_positions(base_pos, blocks, fin_start)

    def train_loss(frames, task, char, room, q0, gold, evid, declare_n=None):
        full = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0, answer=gold,
                                declare_n=declare_n, nfree=args.nfree_prompt),
            add_generation_prompt=False, tokenize=True, return_dict=True,
            return_tensors="pt")
        prompt = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0, declare_n=declare_n,
                                nfree=args.nfree_prompt),
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        full = move_to_device(dict(full), rt.device)
        if not frames:  # S8 synthetic k=0: text-only step, plain causal, no fence
            bnd0 = int(prompt["input_ids"].shape[1])
            labels0 = full["input_ids"].clone()
            labels0[:, :bnd0] = -100
            return model(**full, labels=labels0).loss
        setup = fenced_setup(full, q0, len(frames), evid)
        if setup is None:
            raise ValueError("layout parse failed")
        masks, pos = setup
        bnd = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :bnd] = -100
        full.pop("attention_mask", None)
        if args.attn_logn_sref > 0:
            set_logn_scaling(int(full["input_ids"].shape[1]))
        # NOTE: masks stay SET through backward — gradient checkpointing re-runs the
        # layer forwards during .backward(), and the pre-hook must inject the SAME
        # fence there (the 139063 diagnosis: clearing before backward makes the
        # recompute run UNGATED -> corrupted gradients -> the r1/r2 divergence).
        # The caller clears after opt/backward via clear_masks().
        set_masks(masks)
        with sdpa_kernel(FENCED_SDPA):
            return model(**full, position_ids=pos.to(rt.device), labels=labels).loss

    @torch.inference_mode()
    def infer_gate(inp, q0, room, nf):
        """S2a pass 1: ungated fenced forward -> LR gate on replica-slot states
        -> predicted evidence set (or None on layout/locus miss)."""
        ids = inp["input_ids"][0].tolist()
        parsed = parse_layout(ids, tok, q0, nf, vs_id)
        if parsed is None:
            return None
        blocks, _fin = parsed
        loci = [locate_word_token(ids, tok, room, b) for b in blocks]
        if any(p is None for p in loci):
            return None
        setup = fenced_setup(inp, q0, nf, None)
        if setup is None:
            return None
        masks, pos = setup
        cur = {k: v for k, v in inp.items() if k != "attention_mask"}
        set_masks(masks)
        try:
            with sdpa_kernel(FENCED_SDPA):
                hs = model(**cur, position_ids=pos.to(rt.device),
                           output_hidden_states=True, use_cache=False
                           ).hidden_states[gate_npz["layer"]][0]
        finally:
            clear_masks()
        x = hs[torch.tensor(loci)].float().cpu().numpy()
        sc = ((x - gate_npz["mu"]) / gate_npz["sd"]) @ gate_npz["w"] + gate_npz["b"]
        return {t for t in range(nf) if sc[t] > 0}

    @torch.inference_mode()
    def predict(frames, task, char, room, q0, evid):
        """-> (parsed answer or None, evidence set actually used for the mask)."""
        inp = processor.apply_chat_template(
            build_task_messages(frames, task, char, room, q0,
                                declare_n=args.declare_n or None,
                                nfree=args.nfree_prompt),
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")
        inp = move_to_device(dict(inp), rt.device)
        if args.gate == "model":
            evid = infer_gate(inp, q0, room, len(frames))
            if evid is None:
                return None, None
        ids = inp["input_ids"]
        grid = inp.get("image_grid_thw")
        out_toks = []
        if args.attn_logn_sref > 0:
            set_logn_scaling(int(ids.shape[1]))
        for _ in range(MAX_NEW):
            cur = {"input_ids": ids, "image_grid_thw": grid,
                   "pixel_values": inp.get("pixel_values"),
                   "attention_mask": torch.ones_like(ids)}
            setup = fenced_setup(cur, q0, len(frames), evid)
            if setup is None:
                return None, evid
            masks, pos = setup
            cur.pop("attention_mask", None)
            set_masks(masks)
            try:
                with sdpa_kernel(FENCED_SDPA):
                    logits = model(**cur, position_ids=pos.to(rt.device)).logits
            finally:
                clear_masks()
            nxt = int(logits[0, -1].argmax())
            if nxt == tok.eos_token_id:
                break
            out_toks.append(nxt)
            txt = tok.decode(out_toks)
            if "\n" in txt:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return parse_answer(task, tok.decode(out_toks, skip_special_tokens=True)), evid

    def evaluate(idx_list, cap=None):
        model.eval()
        use = idx_list if cap is None else idx_list[:cap]
        n = ok = 0
        for i in use:
            sd, _g, task = samples[i]
            try:
                frames, q0, char, room, states = load_frames(sd)
                gold = derive_gold(task, states, char, room, len(frames))
                evid = oracle_evid(q0, states, char, room)
                pred, _used = predict(frames, task, char, room, q0, evid)
            except Exception:
                continue
            n += 1
            ok += int(pred == gold)
        return ok / max(1, n), n

    if args.epochs:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        best_val, best_state, best_epoch = -1.0, None, -1
        vrows = ["epoch,val_acc,n"]
        rng = np.random.default_rng(args.seed)
        def syn_sample(spec):
            """Evidence-only synthetic sample: k of the source's evidence frames +
            a prompt declaring N~; answer = k (self-certifying — every shown frame
            is evidence)."""
            src, k, nn = spec
            sd_, _g_, _t_ = samples[src]
            frames_, q0_, char_, room_, states_ = load_frames(sd_)
            evid_ = oracle_evid(q0_, states_, char_, room_)
            idxs = sorted(evid_)
            if k > len(idxs):
                raise ValueError(f"source gold {len(idxs)} < k {k}")
            chosen = (sorted(rng.choice(len(idxs), size=k, replace=False).tolist())
                      if k else [])
            return [frames_[idxs[c]] for c in chosen], q0_, char_, room_, k, nn

        for epoch in range(args.epochs):
            model.train()
            rng.shuffle(tr_idx)
            if args.virtual_n:
                work = ([("real", int(i)) for i in tr_idx]
                        + [("syn", j) for j in range(len(syn_specs))])
                work = [work[o] for o in rng.permutation(len(work))]
            else:
                work = [("real", int(i)) for i in tr_idx]
            opt.zero_grad()
            run_loss, seen = 0.0, 0
            for step, (kind, i) in enumerate(work):
                sd = samples[i][0] if kind == "real" else samples[syn_specs[i][0]][0]
                try:
                    if kind == "real":
                        _sd, _g, task = samples[i]
                        frames, q0, char, room, states = load_frames(sd)
                        gold = derive_gold(task, states, char, room, len(frames))
                        evid = oracle_evid(q0, states, char, room)
                        loss = train_loss(frames, task, char, room, q0, gold, evid)
                    else:
                        sub, q0s, chars, rooms_, kk, nn = syn_sample(syn_specs[i])
                        loss = train_loss(sub, "count", chars, rooms_, q0s, kk,
                                          None, declare_n=nn)
                except Exception as exc:
                    emit(f"  train skip {Path(str(sd)).name} ({kind}): {exc}")
                    n_layout_skip += 1
                    clear_masks()
                    torch.cuda.empty_cache()
                    continue
                (loss / args.grad_accum).backward()
                clear_masks()
                run_loss += float(loss.detach())
                seen += 1
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    opt.zero_grad()
            vacc, vn = evaluate(list(va_idx), cap=args.val_cap)
            vrows.append(f"{epoch},{vacc:.4f},{vn}")
            emit(f"epoch {epoch}: train_loss={run_loss/max(1,seen):.3f} val_acc={vacc:.3f} "
                 f"(skips so far {n_layout_skip})")
            if vacc > best_val:
                best_val, best_epoch = vacc, epoch
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()
                                            if "lora" in k.lower()})
                model.save_pretrained(str(run_dir / "adapter"))
                emit(f"  adapter saved @ep{epoch} -> {run_dir / 'adapter'}")
            if args.patience and (epoch - best_epoch) >= args.patience:
                emit(f"early stop (best={best_epoch})")
                break
        (run_dir / "val_by_epoch.csv").write_text("\n".join(vrows) + "\n")
        emit(f"best_epoch={best_epoch} val_acc={best_val:.3f}")
        if best_state is not None:
            model.load_state_dict({k: v.to(rt.device) for k, v in best_state.items()},
                                  strict=False)

    acc, n = evaluate(list(te_idx))
    emit(f"TEST_IID: acc={acc:.4f} n={n}")
    (run_dir / "summary.csv").write_text(f"split,n,accuracy\ntest_iid,{n},{acc:.4f}\n")

    if args.eval_dirs_file:
        lrows = ["source,task,n,accuracy,parse_fail,mae,mean_signed_err,majority_baseline,skipped,gate_fn,gate_fp"]
        prows = ["source,sample_dir,task,n_frames,k,gold,pred,gate_fn,gate_fp"]
        for f in args.eval_dirs_file:
            e_task = task_of_dirs_file(f)
            dirs = read_dirs_file(Path(f))
            hits = pf = nn = mn = skp = 0
            mae = sgn = 0.0
            g_fn = g_fp = 0
            per: dict = {}
            per_k: dict = {}
            for sd in dirs[: args.eval_longn_limit]:
                try:
                    frames, q0, char, room, states = load_frames(sd)
                    k = evidence_count(states, char, room)
                    gold = derive_gold(e_task, states, char, room, len(frames))
                    evid = oracle_evid(q0, states, char, room)
                    pred, used = predict(frames, e_task, char, room, q0, evid)
                except Exception as exc:
                    emit(f"  eval skip {Path(str(sd)).name}: {exc}")
                    skp += 1
                    continue
                nn += 1
                hits += int(pred == gold)
                fn_ = fp_ = ""
                if args.gate == "model" and used is not None and evid is not None:
                    fn_, fp_ = len(evid - used), len(used - evid)
                    g_fn += fn_
                    g_fp += fp_
                if pred is None:
                    pf += 1
                elif e_task == "count":
                    mae += abs(int(pred) - int(gold))
                    sgn += int(pred) - int(gold)
                    mn += 1
                pg = per.setdefault(gold, [0, 0])
                pg[1] += 1
                pg[0] += int(pred == gold)
                kk = per_k.setdefault(k, [0, 0])
                kk[1] += 1
                kk[0] += int(pred == gold)
                prows.append(f"{f},{sd},{e_task},{len(frames)},{k},{gold},"
                             f"{'' if pred is None else pred},{fn_},{fp_}")
            pc = " ".join(f"g{g}:{c}/{t2}" for g, (c, t2) in sorted(per.items()))
            pk = " ".join(f"k{k_}:{c}/{t2}" for k_, (c, t2) in sorted(per_k.items()))
            maj = max((t2 for _c, t2 in per.values()), default=0) / max(nn, 1)
            gate_note = (f" gate_fn={g_fn} gate_fp={g_fp}" if args.gate == "model" else "")
            emit(f"LONGN {f} task={e_task}: n={nn} acc={hits/max(nn,1):.4f} "
                 f"parse_fail={pf/max(nn,1):.3f} mae={mae/max(mn,1):.2f} "
                 f"mse={sgn/max(mn,1):+.2f} majority={maj:.3f} skipped={skp}{gate_note}\n"
                 f"  per-gold {pc}\n  per-k {pk}")
            lrows.append(f"{f},{e_task},{nn},{hits/max(nn,1):.4f},{pf/max(nn,1):.3f},"
                         f"{mae/max(mn,1):.2f},{sgn/max(mn,1):+.3f},{maj:.3f},{skp},"
                         f"{g_fn},{g_fp}")
        (run_dir / "longn_eval.csv").write_text("\n".join(lrows) + "\n")
        (run_dir / "longn_predictions.csv").write_text("\n".join(prows) + "\n")
    remove_hooks()
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
