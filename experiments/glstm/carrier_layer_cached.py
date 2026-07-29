#!/usr/bin/env python3
"""STAGE-2 FAST TRAINER (2026-07-18): cached-activation variant of carrier_layer_lora.py.

Since the LoRA lives only in layers >= L_OPEN and e_c is FROZEN (loaded from the distilled
checkpoint), layers 0..L_OPEN-1 are fully static per sample. So: run them ONCE per sample at
prep (under the fenced lo-mask), cache h_{L_OPEN} (bf16, CPU RAM), and train only layers
L_OPEN..27 on cached states — ~3x cheaper steps, cache reused every epoch. Masks are rebuilt
per step from compact metadata (never cached; seq^2 would blow RAM at pooled scale).

Differences vs carrier_layer_lora.py (by design): e_c is NOT trainable (the frozen-e_c
ablation); checkpoint format is compatible (carrier_layer_best.pt with e_c/lora/l_open/rank)
so its --eval-only tooling works on the result.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, iter_sample_dirs_shuffled, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from experiments.glstm.carrier_token_distill import (reset_positions, find_subseq, CARRIER_TOKEN)
from experiments.glstm.carrier_layer_lora import (make_masks, parse_task_labels, build_target,
                                                  build_target_tally, couple_offsets,
                                                  build_target_fmt, frame_attr_labels,
                                                  parse_scratchpad_answer, SCRATCHPAD_FORMATS,
                                                  keep_cols, frame_cols, truncated_masks)
from torch.nn.attention import sdpa_kernel, SDPBackend

MIN = -65504.0
# EFFICIENT first (O(seq) memory — math OOMs at seq≳12k on 40GB: 28 heads × seq² scores),
# MATH fallback. Fixed list in BOTH normal forwards and checkpoint recomputes → consistent
# backend selection. Masks must be dtype-MATCHED to the query (bf16) for the efficient kernel.
MATH_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def ext_mask(m, e):
    """Append e teacher-forced rows: each reads like the last tail row + causal over appends."""
    if e == 0:
        return m
    s0 = m.shape[0]
    big = torch.full((s0 + e, s0 + e), MIN, dtype=m.dtype)
    big[:s0, :s0] = m
    for j in range(e):
        r = s0 + j
        big[r, :s0] = m[s0 - 1]
        big[r, s0:r + 1] = 0.0
    return big


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform",
                    help="comma-separated roots (task mixture, task inferred per question)")
    ap.add_argument("--limit", type=int, default=900, help="PER data root")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--l-open", type=int, default=17)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--carrier-ckpt", required=True,
                    help="distilled carrier_best.pt — e_c is loaded FROZEN")
    ap.add_argument("--scratchpad", action="store_true",
                    help="A1: teacher-forced verdict-scratchpad targets ('frames 2, 5 -> 2') "
                         "instead of the single answer digit; CE over the full target+EOS")
    ap.add_argument("--pos-couple", action="store_true",
                    help="E-G (2026-07-20): position-coupled tally — target/generated verdict "
                         "tokens carry the POSITION of the carrier they describe (couple_offsets "
                         "stream rule, shared with decode). Rooms stays uncoupled (name verdicts "
                         "carry no frame index). Implies --running-tally.")
    ap.add_argument("--running-tally", action="store_true",
                    help="R1 (2026-07-19): scratchpad variant with running counts per verdict "
                         "('frames 2 (1), 5 (2) -> 2') — answer becomes a read-off of the last "
                         "tally; targets long-list undercounts. Implies --scratchpad.")
    ap.add_argument("--scratchpad-format", default="poslist", choices=SCRATCHPAD_FORMATS,
                    help="FORMAT sweep (2026-07-22): gold scratchpad text shape — poslist = "
                         "the l12v2 control (build_target_tally, unchanged), scan/caption = "
                         "full-scan verdict/attribute slots + inline tally + END, chunked = "
                         "16-frame blocks with subtotals. Non-poslist implies --scratchpad; "
                         "saved into the ckpt for eval-side parser autodetection")
    ap.add_argument("--jitter-gap", type=int, default=0, metavar="G",
                    help="A2: TRAIN-only carrier position jitter for the hi phase — per step "
                         "draw carrier gaps ~ U{1..G} (0 = off; eval always gaps=1). Covers "
                         "the large carrier-carrier/tail-carrier RoPE distances of long N")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="A4: gradient-checkpoint the hi-phase layers during training "
                         "(long-N backward OOMs at seq>~4k on 40GB otherwise; ~2x fwd cost)")
    ap.add_argument("--no-qfirst", action="store_true",
                    help="C2 ablation: drop the LEADING question (question only after frames)")
    ap.add_argument("--no-posreset", action="store_true",
                    help="C2 ablation: keep base M-RoPE positions (no per-block reset, no "
                         "sequential carrier override, no tail re-base)")
    ap.add_argument("--shuffle-dirs", type=int, default=0, metavar="SEED")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truncate-at", type=int, default=None, metavar="L",
                    help="TRUNC E4 (2026-07-25): train WITH physical frame truncation — "
                         "the lo phase still runs full sequences (carriers read their "
                         "frames) but teacher-forced target rows get frame columns MASKED, "
                         "and the cache keeps ONLY [question]+[carriers]+[tail]+target rows "
                         "(~100x smaller); the hi phase runs on truncated coords with "
                         "ORIGINAL position ids. Must equal --l-open (the cache locus). "
                         "Deploy-matched exam: --truncate-at L --fast-decode in "
                         "carrier_layer_lora --eval-only.")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="P1.1 seeds (2026-07-23): pin the train/eval dir split (0 = arm A's) "
                         "while --seed varies init/jitter/epoch-shuffle; default = --seed "
                         "(byte-identical behavior to before)")
    ap.add_argument("--output", default="outputs/ladder/image_longN/carrier_layer_cached")
    args = ap.parse_args()
    if args.pos_couple:
        args.running_tally = True
    if args.scratchpad_format != "poslist":
        args.scratchpad = True
        if args.pos_couple:
            raise SystemExit("--pos-couple is poslist-only (couple_offsets assumes the "
                             "'frames i (k)' stream rule)")
    if args.running_tally:
        args.scratchpad = True
    if args.truncate_at is not None and args.truncate_at != args.l_open:
        raise SystemExit("--truncate-at must equal --l-open (the cache locus); truncation "
                         f"deeper than L_open needs full-row caches: got {args.truncate_at} "
                         f"vs l_open {args.l_open}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    LO = args.l_open
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    tok = processor.tokenizer
    text_model = model.model.language_model
    dev = model.device
    D = cfg.hidden_size
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_L{LO}_r{args.rank}")
    out.mkdir(parents=True, exist_ok=True)

    cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    vs_id = int(model.config.vision_start_token_id)
    rope_fn = getattr(model, "get_rope_index", None) or model.model.get_rope_index
    digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]

    ckc = torch.load(args.carrier_ckpt, map_location="cpu")
    e_c = ckc["e_c"].float().to(dev)                       # FROZEN by design
    print(f"[init] frozen e_c from {args.carrier_ckpt} (distilled d' {ckc.get('dprime'):.2f})",
          flush=True)

    scale = args.alpha / args.rank
    lora = {}
    handles = []
    for li in range(LO, len(layers)):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(layers[li].self_attn, nm)
            A = nn.Parameter(torch.randn(args.rank, mod.in_features, device=dev) * 0.01)
            B = nn.Parameter(torch.zeros(mod.out_features, args.rank, device=dev))
            lora[(li, nm)] = (A, B)

            def mk(A=A, B=B):
                def hook(_m, inp, o):
                    x = inp[0]
                    return o + (scale * (x.float() @ A.T) @ B.T).to(o.dtype)
                return hook
            handles.append(mod.register_forward_hook(mk()))
    lora_params = [p for ab in lora.values() for p in ab]
    opt = torch.optim.Adam(lora_params, lr=args.lr_lora)
    print(f"[params] lora {sum(p.numel() for p in lora_params)} (e_c frozen)", flush=True)

    # ---- prep: build inputs, run layers 0..LO-1 ONCE, cache h_LO + metadata ----
    data = []
    n_done = n_skip = 0
    cache_gb = 0.0
    t0 = time.time()
    for root in args.data_root.split(","):
        root = root.strip()
        lim = args.limit
        if "=" in root:                    # per-root override: path=LIMIT (A4)
            root, lim = root.rsplit("=", 1)
            lim = int(lim)
        it = (iter_sample_dirs_shuffled(Path(root), args.shuffle_dirs)
              if args.shuffle_dirs is not None else iter_sample_dirs(Path(root)))
        n_root = 0
        for sd in it:
            if n_root >= lim:
                break
            try:
                sid, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
            except Exception:
                n_skip += 1; continue
            parsed = parse_task_labels(q0, states, gold)
            if parsed is None or (gold > 9 and not args.scratchpad):
                n_skip += 1; continue      # scratchpad reads out multi-digit natively (A4)
            task, evid, aux = parsed
            tgt_ids = []
            anch = None
            if args.scratchpad:
                if args.scratchpad_format != "poslist":
                    labels = (frame_attr_labels(task, q0, states, evid)
                              if (args.scratchpad_format in ("caption", "chunked")
                                  or task == "rooms") else None)
                    tgt_str = build_target_fmt(args.scratchpad_format, task, evid, aux,
                                               gold, NF=len(frames), labels=labels)
                else:
                    tgt_str = (build_target_tally if args.running_tally
                               else build_target)(task, evid, aux, gold)
                tgt_ids = tok(tgt_str, add_special_tokens=False).input_ids + [tok.eos_token_id]
                if args.pos_couple and task != "rooms":
                    anch = couple_offsets([tok.decode([t]) for t in tgt_ids], len(frames))
                if n_done == 0:
                    rt = tok.decode(tgt_ids[:-1])
                    print(f"[target-debug] task={task} gold={gold} target={tgt_str!r} "
                          f"tokens={len(tgt_ids)} roundtrip={rt!r} ok={rt == tgt_str}",
                          flush=True)
                    if anch is not None:
                        print("[couple-debug] " + " ".join(
                            f"{tok.decode([t])!r}@c{a}+{o}"
                            for t, (a, o) in zip(tgt_ids, anch)), flush=True)
            NF = len(frames)
            if args.resize > 0:
                frames = [f.resize((args.resize, args.resize)) for f in frames]
            content = [] if args.no_qfirst else [{"type": "text", "text": q0}]
            for f in frames:
                content.append({"type": "image", "image": f})
                content.append({"type": "text", "text": CARRIER_TOKEN})
            content.append({"type": "text", "text": q0})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            ids = inputs["input_ids"][0].tolist(); seq = len(ids)
            fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                    processor=processor)
            cpos = [p for p, t in enumerate(ids) if t == cid]
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            occ = None
            n_occ = 1 if args.no_qfirst else 2
            for pre in ("", " ", "\n"):
                needle = tok(pre + q0, add_special_tokens=False).input_ids
                o = find_subseq(ids, needle)
                if len(o) == n_occ:
                    occ = o; break
            if len(fg) != NF or len(cpos) != NF or len(vstarts) != NF or occ is None:
                n_skip += 1; continue
            fin_start = occ[-1]
            blocks = [(vstarts[i], (vstarts[i + 1] if i + 1 < NF else fin_start))
                      for i in range(NF)]
            e_len = len(tgt_ids)
            mask_lo, _ = make_masks(seq, blocks, cpos, fin_start)
            mask_lo = ext_mask(mask_lo, e_len)
            keep = None
            if args.truncate_at is not None:
                keep = keep_cols(seq, blocks, cpos)
                if e_len:      # target rows never see frames (deploy-matched teacher forcing)
                    mask_lo[seq:, torch.tensor(frame_cols(seq, blocks, cpos),
                                               dtype=torch.long)] = MIN
            with torch.no_grad():
                base_pos, _d = rope_fn(inputs["input_ids"],
                                       image_grid_thw=inputs.get("image_grid_thw"),
                                       attention_mask=inputs.get("attention_mask"))
                if args.no_posreset:
                    pos = base_pos.clone()
                    blk0_max = int(pos.max())      # only used by jitter (off in C2 arms)
                else:
                    pos = reset_positions(base_pos, blocks, fin_start).clone()
                    blk0_max = int(pos[:, :, blocks[0][0]:blocks[0][1]].max())
                    for i, c in enumerate(cpos):
                        pos[:, :, c] = blk0_max + 1 + i
                    pos[:, :, fin_start:] += NF
                if e_len:
                    inc = torch.arange(1, e_len + 1,
                                       device=pos.device).view(1, 1, e_len).expand(3, 1, e_len)
                    pos = torch.cat([pos, pos[:, :, -1:] + inc], dim=2)
                    if anch is not None:      # E-G: coupled target positions in BOTH phases
                        cvals = [int(pos[0, 0, c]) for c in cpos]
                        pos[:, :, seq:] = torch.tensor(
                            [cvals[a - 1] + o for a, o in anch],
                            device=pos.device).view(1, 1, e_len).expand(3, 1, e_len)
                emb = text_model.embed_tokens(inputs["input_ids"])
                img = model.model.get_image_features(inputs["pixel_values"],
                                                     inputs["image_grid_thw"])
                img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
                im_mask = inputs["input_ids"][0] == model.config.image_token_id
                emb = emb.clone(); emb[0, im_mask] = img.to(emb.dtype)
                emb[0, torch.tensor(cpos, device=dev)] = e_c.to(emb.dtype)
                if e_len:
                    text = text_model.embed_tokens(torch.tensor([tgt_ids], device=dev))
                    emb = torch.cat([emb, text.to(emb.dtype)], dim=1)
                lo4 = mask_lo.to(dev).to(emb.dtype).view(1, 1, seq + e_len, seq + e_len)
                cos_, sin_ = text_model.rotary_emb(emb, pos.to(dev))
                pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
                h = emb
                with sdpa_kernel(MATH_SDPA):
                    for ly in layers[:LO]:
                        h = ly(h, attention_mask=lo4, position_embeddings=pe)[0]
            if args.truncate_at is not None:
                # physical truncation at the cache locus: keep-rows + target rows only;
                # ORIGINAL position ids index-selected (P0.2), coords remapped
                kt = torch.tensor(keep + list(range(seq, seq + e_len)), device=h.device)
                h = h.index_select(1, kt)
                pos = pos.index_select(2, kt.to(pos.device))
                a0t = blocks[0][0]
                cpos = list(range(a0t, a0t + NF))
                fin_start = a0t + NF
                seq = len(keep)
                if n_done == 0:
                    print(f"[trunc-debug] cache rows {int(kt.numel())} (keep {len(keep)} "
                          f"+ tgt {e_len}); carriers at [{a0t},{a0t + NF}) fin_t "
                          f"{fin_start}", flush=True)
            # readout slice at the tail of tgt_ids: poslist ends '... -> G' + EOS; the new
            # formats end '... G END' + EOS (the END token is part of the checked slice)
            ans_sfx = (f" {gold}" if args.scratchpad_format == "poslist" else f" {gold} END")
            ans_k = (len(tok(ans_sfx, add_special_tokens=False).input_ids) + 1
                     if e_len else 0)
            data.append({"h": h[0].to(torch.bfloat16).cpu(), "pos": pos.cpu(),
                         "blocks": blocks, "cpos": cpos, "fin": fin_start, "seq": seq,
                         "e": e_len, "tgt": tgt_ids, "ans_k": ans_k, "blk0max": blk0_max,
                         "anch": anch, "gold": gold, "task": task, "sd": str(sd),
                         "trunc": args.truncate_at is not None})
            cache_gb += (seq + e_len) * D * 2 / 1e9
            n_done += 1; n_root += 1
            if n_done % 100 == 0:
                print(f"  prep {n_done} (skip {n_skip}) {time.time()-t0:.0f}s cache {cache_gb:.1f}GB",
                      flush=True)
    from collections import Counter
    print(f"prep done: n={n_done} skip={n_skip} cache {cache_gb:.1f}GB "
          f"tasks {Counter(d['task'] for d in data)} "
          f"golds {Counter(d['gold'] for d in data)}", flush=True)

    def top_hidden(d, jitter=False):
        """hi-phase over cached h_LO (prompt + teacher-forced target rows). Returns final
        normed hidden (1, seq+e, D). jitter=A2: hi-phase-only carrier gap stretch."""
        seq = d["seq"]; e = d.get("e", 0)
        h = d["h"].to(dev).unsqueeze(0)
        if d.get("trunc"):
            _, mask_hi = truncated_masks(list(range(seq)), d["cpos"])
        else:
            _, mask_hi = make_masks(seq, d["blocks"], d["cpos"], d["fin"])
        # hi-phase stream runs fp32 (LoRA-hooked path); fp32 mask matches EFFICIENT exactly
        # and is universally accepted by the MATH fallback — do NOT dtype-match to h here
        hi4 = ext_mask(mask_hi, e).to(dev).to(torch.float32).view(1, 1, seq + e, seq + e)
        pos = d["pos"].to(dev)
        if jitter and args.jitter_gap > 1:
            pos = pos.clone()
            NF = len(d["cpos"])
            cum = torch.cumsum(torch.randint(1, args.jitter_gap + 1, (NF,)), 0)
            for i, c in enumerate(d["cpos"]):
                pos[:, :, c] = d["blk0max"] + int(cum[i])
            pos[:, :, d["fin"]:] += int(cum[-1]) - NF
        if d.get("anch") is not None and e:
            # E-G: target rows follow their (possibly jittered) carriers' positions
            cvals = [int(pos[0, 0, c]) for c in d["cpos"]]
            pos[:, :, d["seq"]:] = torch.tensor(
                [cvals[a - 1] + o for a, o in d["anch"]],
                device=pos.device).view(1, 1, e).expand(3, 1, e)
        cos_, sin_ = text_model.rotary_emb(h, pos)
        pe = (cos_.to(h.dtype), sin_.to(h.dtype))
        use_ckpt = args.grad_ckpt and torch.is_grad_enabled()

        def _blk(hh, _ly):
            # sdpa_kernel INSIDE: checkpoint recomputation runs outside the caller's
            # context manager — backend must be re-selected or metadata mismatches
            with sdpa_kernel(MATH_SDPA):
                return _ly(hh, attention_mask=hi4, position_embeddings=pe)[0]

        for ly in layers[LO:]:
            if use_ckpt:
                h = torch.utils.checkpoint.checkpoint(_blk, h, ly, use_reentrant=False)
            else:
                h = _blk(h, ly)
        return text_model.norm(h)

    def head(hs):
        return model.lm_head(hs.to(model.lm_head.weight.dtype)).float()

    split_rng = np.random.default_rng(args.seed if args.split_seed is None else args.split_seed)
    rng = np.random.default_rng(args.seed)          # epoch-shuffle rng, decoupled from the split
    order = split_rng.permutation(n_done)
    n_tr = int(n_done * args.train_frac)
    tr_idx, ev_idx = order[:n_tr], order[n_tr:]
    (out / "eval_dirs.txt").write_text(
        "\n".join(data[i]["sd"] for i in ev_idx) + "\n")
    (out / "train_dirs.txt").write_text(
        "\n".join(data[i]["sd"] for i in tr_idx) + "\n")

    def evaluate():
        """digit mode: emitted digit acc. scratchpad mode: teacher-forced COUNT-token acc
        (headline; greedy-decode acc comes from the eval-only exams) + TF-exact."""
        hits = 0; exact = 0; mae = 0.0; mae_n = 0
        per_task = {}
        with torch.no_grad():
            for i in ev_idx:
                d = data[i]
                g = d["gold"]; t = d["task"]
                if args.scratchpad:
                    hs = top_hidden(d)
                    seqp, e, ak = d["seq"], d["e"], d["ans_k"]
                    lg = head(hs[0, seqp - 1:seqp + e - 1])
                    preds = lg.argmax(-1).tolist()
                    tf_all = preds == d["tgt"]
                    cnt_ok = preds[-ak:-1] == d["tgt"][-ak:-1]
                    exact += tf_all; hits += cnt_ok
                    ptxt = tok.decode(preds[-ak:-1]).strip()
                    if ptxt.endswith("END"):        # scan/caption/chunked readout slice
                        ptxt = ptxt[:-3].strip()
                    if ptxt.isdigit():
                        mae += abs(int(ptxt) - g); mae_n += 1
                    ok = cnt_ok
                else:
                    lg = head(top_hidden(d)[0, -1])
                    dg = int(np.argmax([float(lg[t2]) for t2 in digit_ids]))
                    ok = (dg == g)
                    hits += ok; mae += abs(dg - g); mae_n += 1
                per_task.setdefault(t, [0, 0]); per_task[t][1] += 1; per_task[t][0] += ok
        pt = " ".join(f"{t}:{c}/{n}" for t, (c, n) in sorted(per_task.items()))
        if args.scratchpad:
            pt += f"  tf-exact {exact/len(ev_idx):.3f}"
        return hits / len(ev_idx), mae / max(mae_n, 1), pt, exact / len(ev_idx)

    acc0, mae0, pt0, ex0 = evaluate()
    print(f"[ep 0] acc {acc0:.3f} MAE {mae0:.2f}  [{pt0}]", flush=True)
    lines = [f"=== CARRIER LAYER CACHED (L_open={LO}, r={args.rank}, frozen e_c, n={n_done}, "
             f"train={len(tr_idx)}, scratchpad={args.scratchpad}, fmt={args.scratchpad_format}, "
             f"jitter={args.jitter_gap}, "
             f"noqfirst={args.no_qfirst}, noposreset={args.no_posreset}, "
             f"roots={args.data_root}) ===",
             f"ep0 acc {acc0:.3f} mae {mae0:.2f} [{pt0}]"]
    best = ((acc0, ex0), 0)   # tie-break on tf-exact (2026-07-21: TF-count saturates @ep1-2
                              # and the old acc-only rule kept selecting weaker-transcript ckpts)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        tot = 0.0
        te = time.time()
        for step, i in enumerate(tr_idx):
            d = data[i]
            if args.scratchpad:
                hs = top_hidden(d, jitter=True)
                lg = head(hs[0, d["seq"] - 1:d["seq"] + d["e"] - 1])
                loss = F.cross_entropy(lg, torch.tensor(d["tgt"], device=dev))
            else:
                lg = head(top_hidden(d, jitter=True)[0, -1])
                loss = F.cross_entropy(lg.unsqueeze(0),
                                       torch.tensor([digit_ids[d["gold"]]], device=dev))
            (loss / 8).backward()
            tot += float(loss)
            if (step + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        acc, mae, pt, ex = evaluate()
        print(f"[ep {ep}] loss {tot/len(tr_idx):.4f} acc {acc:.3f} MAE {mae:.2f} [{pt}] "
              f"({time.time()-te:.0f}s/ep)", flush=True)
        lines.append(f"ep{ep} loss {tot/len(tr_idx):.4f} acc {acc:.3f} mae {mae:.2f} [{pt}]")
        if (acc, ex) > best[0]:
            best = ((acc, ex), ep)
            torch.save({"e_c": e_c.detach().cpu(),
                        "lora": {f"{li}.{nm}": (A.detach().cpu(), B.detach().cpu())
                                 for (li, nm), (A, B) in lora.items()},
                        "l_open": LO, "rank": args.rank, "epoch": ep, "acc": acc,
                        "scratchpad": args.scratchpad, "jitter_gap": args.jitter_gap,
                        "running_tally": args.running_tally, "pos_couple": args.pos_couple,
                        "scratchpad_format": args.scratchpad_format,
                        "truncate_at": args.truncate_at},
                       out / "carrier_layer_best.pt")
    lines.append(f"BEST acc {best[0][0]:.3f} (tf-exact {best[0][1]:.3f}) @ ep {best[1]} "
                 f"(scaffold 0.998; frozen 0.219)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:])); print("wrote", out)
    for hd in handles:
        hd.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
