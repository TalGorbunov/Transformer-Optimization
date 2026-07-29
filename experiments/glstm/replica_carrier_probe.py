#!/usr/bin/env python3
"""REPLICA CARRIERS (2026-07-14): per-frame question replicas with masked attention — a
one-forward architectural fix for the query half of the joint-context tax.

Mechanism: after each frame's image tokens, insert a REPLICA of the question. A custom 4D
attention mask (all layers) makes each replica (a) attend ONLY {prefix text, its own frame,
itself} — so its query is computed from an uncontaminated residual — and (b) INVISIBLE to every
other token — so the original joint computation (frame encoding, final question, model answer)
is bit-for-bit undisturbed except for the replicas' own rows.

Per sample we then read, at L14/L16, the per-frame message into (1) each replica's room token
(the new mechanism) and (2) the final question's room token at off −9 (the in-run JOINT anchor).
Held-out d′ on both. Pre-registered prediction (from the 2×2): replica read ≈ clean-query ×
joint-values ≈ 3.3–4.0 (eval scale) vs joint anchor ≈ 2.0; GO if ≥3.

Steps task, mmred park images, 392px, N frames per sample. Forward-only (no generation).
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import (iter_sample_dirs, iter_sample_dirs_shuffled,
                                       load_mmred_sample)
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from experiments.glstm.dprime_vs_n import dprime_pair
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)

MIN = -65504.0


def find_subseq(hay: List[int], needle: List[int]) -> List[int]:
    out, n = [], len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            out.append(i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--layers", default="14,16")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--output", default="outputs/ladder/image_longN/replica_carrier")
    ap.add_argument("--no-mask", action="store_true",
                    help="UNMASKED arm: plain interleaved prompt, replicas fully visible/contaminated "
                         "(the prompt-engineering control; per-copy d' traces the contamination ladder)")
    ap.add_argument("--fence-frames", action="store_true",
                    help="Exp A (2026-07-15): FULL frame fencing — each frame's visual-token rows "
                         "attend only {prefix, own frame} (replicas already hidden). Every "
                         "(frame + replica) block becomes an isolated forward inside one sequence.")
    ap.add_argument("--unmix-dir", default=None,
                    help="Exp B (2026-07-15): dir with unmixer_L<L>.pt (encoding_unmixer --save-dir). "
                         "Applies g_k/g_v to FRAME visual-token k/v at --unmix-layer during the "
                         "forward (pre-rotary, inside k_proj/v_proj hooks) — the capture then sees "
                         "the un-mixed k/v, so the message recompute composes replica queries x "
                         "un-mixed values. Use with --no-mask (unmasked replica arm).")
    ap.add_argument("--unmix-layer", type=int, default=16)
    ap.add_argument("--natural", action="store_true",
                    help="mmred_natural cells (real photos): meta.json loader, evidence from "
                         "per-frame is_evidence flags, carrier locus = the concept word (e.g. "
                         "'dog') in each replica.")
    ap.add_argument("--task", choices=("steps", "cooc"), default="steps",
                    help="cooc (2026-07-17): co-occupancy questions ('were X and Y in the same "
                         "room?') — evidence derived from states (both names share a room), "
                         "carrier locus = the SECOND name token in each replica.")
    ap.add_argument("--question-first", action="store_true",
                    help="Q-first control (2026-07-17): also place the question BEFORE the frames "
                         "(shared prefix). Blocks then attend {prefix incl. question, own block} — "
                         "the layout the learned-carrier distillation needs (carrier conditions on "
                         "the question in-context).")
    ap.add_argument("--fence-blocks", action="store_true",
                    help="Exp A3 (2026-07-17): FULL block-diagonal fence — every token of block i "
                         "(vision markers + frame + replica) is forbidden from every token of all "
                         "other blocks. Closes the marker leak: vision_start/end tokens are neither "
                         "visual nor replica tokens, so under --fence-frames later blocks still "
                         "read earlier frames' content through them. Requires --fence-frames.")
    ap.add_argument("--reset-positions", action="store_true",
                    help="Exp A2 (2026-07-17): per-block M-RoPE reset — every fenced "
                         "(frame + replica) block gets block 0's position ids (PCW-style reuse; "
                         "safe because blocks cannot attend each other). Removes the long-context "
                         "position offset, making each block position-equivalent to an isolated "
                         "forward. Requires --fence-frames.")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED",
                    help="stratified deterministic shuffle of sample dirs (class-balanced "
                         "prefixes; REQUIRED whenever --limit < the full dir count)")
    args = ap.parse_args()
    if args.reset_positions and (args.no_mask or not args.fence_frames):
        ap.error("--reset-positions requires --fence-frames (blocks must be attention-isolated "
                 "before they may share positions)")
    if args.fence_blocks and (args.no_mask or not args.fence_frames):
        ap.error("--fence-blocks requires --fence-frames")

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    Ls = [int(x) for x in args.layers.split(",")]
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads, n_kv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope = cfg.rope_scaling["mrope_section"]
    tok = processor.tokenizer
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    import bitsandbytes.functional as bnbF
    oproj = {}
    for L in Ls:
        w = layers[L].self_attn.o_proj.weight
        oproj[L] = bnbF.dequantize_4bit(w.data, w.quant_state).float()

    # ---- hooks: mask injection (ALL layers) + qkv/pos capture (probe layers) ----
    holder = {"mask": None}

    def mask_pre(_m, hargs, hkwargs):
        mk = holder["mask"]
        if mk is None:
            return hargs, hkwargs
        hs = hargs[0] if hargs else hkwargs.get("hidden_states")
        if hs is not None and mk.dtype != hs.dtype:
            mk = mk.to(hs.dtype); holder["mask"] = mk
        if len(hargs) >= 2:
            return (hargs[0], mk) + tuple(hargs[2:]), hkwargs
        hkwargs = dict(hkwargs); hkwargs["attention_mask"] = mk
        return hargs, hkwargs

    qkv: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in Ls}
    posemb = {}
    handles = [ly.register_forward_pre_hook(mask_pre, with_kwargs=True) for ly in layers]

    # Exp B: un-mixer hooks. Registered BEFORE the qkv capture hooks (same modules) so the
    # capture — and everything downstream in the live forward — sees the un-mixed k/v.
    unmix = {"vis": None, "nets": {}}
    if args.unmix_dir:
        import torch.nn as nn
        UL = int(args.unmix_layer)
        ckpt = torch.load(Path(args.unmix_dir) / f"unmixer_L{UL}.pt", map_location="cpu")

        def _load_net(sdict):
            net = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512))
            net.load_state_dict(sdict); net.eval()
            return net.float().to(model.device)
        unmix["nets"][UL] = (_load_net(ckpt["gk"]), _load_net(ckpt["gv"]))
        print(f"[unmix] loaded g_k/g_v for L{UL} from {args.unmix_dir}", flush=True)

        def mk_unmix(L, which):
            def hook(_m, _i, o):
                vis = unmix["vis"]
                if vis is None:
                    return o
                net = unmix["nets"][L][0 if which == "k" else 1]
                o = o.clone()
                o[0, vis] = net(o[0, vis].float()).to(o.dtype)
                return o
            return hook
        handles.append(layers[UL].self_attn.k_proj.register_forward_hook(mk_unmix(UL, "k")))
        handles.append(layers[UL].self_attn.v_proj.register_forward_hook(mk_unmix(UL, "v")))

    def mk_qkv(L, nm):
        def hook(_m, _i, o):
            qkv[L][nm] = o.detach()
        return hook
    for L in Ls:
        for nm in ("q_proj", "k_proj", "v_proj"):
            handles.append(getattr(layers[L].self_attn, nm).register_forward_hook(mk_qkv(L, nm)))

    def pe_hook(_m, a_, k_):
        pe = k_.get("position_embeddings")
        if pe is None and len(a_) >= 1 and isinstance(a_[-1], tuple):
            pe = a_[-1]
        if pe is not None:
            posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
    handles.append(layers[Ls[0]].self_attn.register_forward_pre_hook(pe_hook, with_kwargs=True))

    ROOMS = ("Park", "Garden", "Bathroom", "Kitchen", "Office", "Bedroom")
    feats_rep = {L: [] for L in Ls}; feats_anc = {L: [] for L in Ls}
    labels_all, gold_all = [], []
    n_done = n_skip = 0
    if args.natural:
        dirs = sorted(d for d in Path(args.data_root).iterdir() if d.is_dir())
    elif args.shuffle_dirs is not None:
        dirs = iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs)
    else:
        dirs = iter_sample_dirs(Path(args.data_root))

    for sd in dirs:
        if n_done >= args.limit:
            break
        try:
            if args.natural:
                import json
                from PIL import Image
                meta = json.loads((sd / "meta.json").read_text())
                q0 = meta["question"]; gold = int(meta["answer"]); states = None
                frames = [Image.open(sd / f"frame_{i:02d}.jpg").convert("RGB")
                          for i in range(int(meta["n_frames"]))]
                nat_evid = {i for i, f in enumerate(meta["frames"]) if f["is_evidence"]}
                nat_word = meta["concept"]
            else:
                sid, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
        except Exception:
            n_skip += 1; continue
        if args.natural:
            evid = nat_evid
            room = nat_word
        elif args.task == "cooc":
            mm = re.search(r"were (\w+) and (\w+) in the same room", q0)
            if not mm:
                n_skip += 1; continue
            nA, nB = mm.group(1), mm.group(2)
            evid = set()
            for t, st in enumerate(states):
                for occ in (st.get("rooms", {}) or {}).values():
                    if nA in occ and nB in occ:
                        evid.add(t); break
            if len(evid) != gold:
                n_skip += 1; continue          # label sanity: derived evidence must match answer
            room = nB                          # carrier locus = second name token
        else:
            evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            room = next((r for r in ROOMS if r.lower() in q0.lower()), None)
        if not evid and gold != 0:
            n_skip += 1; continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        if room is None:
            n_skip += 1; continue

        content = []
        if args.question_first:
            content.append({"type": "text", "text": q0})
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        content.append({"type": "text", "text": q0})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        ids = inputs["input_ids"][0].tolist(); seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF, processor=processor)
        if len(fg) != NF:
            n_skip += 1; continue
        if args.unmix_dir:
            unmix["vis"] = torch.tensor(sorted(int(p) for g in fg for p in g),
                                        dtype=torch.long, device=model.device)

        # locate the N+1 question occurrences (replicas + final)
        spans = None
        exp_occ = NF + 2 if args.question_first else NF + 1
        for pre in ("", " ", "\n"):
            needle = tok(pre + q0, add_special_tokens=False).input_ids
            occ = find_subseq(ids, needle)
            if len(occ) == exp_occ:
                spans = [(o, o + len(needle)) for o in occ]; break
        if spans is None:
            n_skip += 1; continue
        if args.question_first:
            spans = spans[1:]                                   # drop the leading (prefix) question
        rep_spans, fin_span = spans[:NF], spans[NF]

        # room token inside each replica: last token in span whose decode contains the room word prefix
        def room_pos(a, b):
            for p in range(b - 1, a - 1, -1):
                if room[:4].lower() in tok.decode([ids[p]]).strip().lower():
                    return p
            return None
        rep_c = [room_pos(a, b) for a, b in rep_spans]
        if any(c is None for c in rep_c):
            n_skip += 1; continue
        anc_c = seq - 1 - 9                                     # off -9 convention

        vstarts = blk_ends = None
        if args.reset_positions or args.fence_blocks:
            vs_id = int(model.config.vision_start_token_id)
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            if len(vstarts) != NF:
                n_skip += 1; continue
            blk_ends = vstarts[1:] + [fin_span[0]]              # block i = [vstarts[i], blk_ends[i])

        # ---- 4D mask ----
        m = torch.zeros(seq, seq, dtype=torch.float32)
        m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
        rep_tok = torch.zeros(seq, dtype=torch.bool)
        for a, b in rep_spans:
            rep_tok[a:b] = True
        # replicas invisible to everyone else
        nonrep_rows = (~rep_tok).nonzero(as_tuple=True)[0]
        m[nonrep_rows.unsqueeze(1), rep_tok.nonzero(as_tuple=True)[0].unsqueeze(0)] = MIN
        # replica_i rows: forbid other frames' visual tokens and other replicas
        vis_by_frame = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
        all_vis = torch.cat(vis_by_frame)
        for i, (a, b) in enumerate(rep_spans):
            rows = torch.arange(a, b)
            own_vis = set(vis_by_frame[i].tolist())
            forb_vis = torch.tensor([int(p) for p in all_vis.tolist() if p not in own_vis], dtype=torch.long)
            if forb_vis.numel():
                m[rows.unsqueeze(1), forb_vis.unsqueeze(0)] = MIN
            for j, (a2, b2) in enumerate(rep_spans):
                if j != i:
                    m[rows.unsqueeze(1), torch.arange(a2, b2).unsqueeze(0)] = MIN
            # own span stays causal; prefix/suffix text stays visible (causal)
        if args.fence_frames:
            # frame i's visual rows: forbid all OTHER frames' visual tokens ->
            # frame rows see only {prefix, own frame} (replicas already hidden from them)
            for i in range(NF):
                rows_f = vis_by_frame[i]
                own = set(vis_by_frame[i].tolist())
                forb_f = torch.tensor([int(p) for p in all_vis.tolist() if p not in own],
                                      dtype=torch.long)
                if forb_f.numel():
                    m[rows_f.unsqueeze(1), forb_f.unsqueeze(0)] = MIN
        if args.fence_blocks:
            # full block-diagonal: every row of block i forbids every column of block j != i
            # (covers the vision start/end markers, which the finer rules above leave visible)
            for i in range(NF):
                rows_b = torch.arange(vstarts[i], blk_ends[i])
                for j in range(NF):
                    if j != i:
                        m[rows_b.unsqueeze(1),
                          torch.arange(vstarts[j], blk_ends[j]).unsqueeze(0)] = MIN
        if n_done == 0 and not args.no_mask:
            # debug: allowed-key counts per row class (sanity for the fence)
            def _row_allowed(r):
                return int((m[r] == 0).sum())
            fr0 = int(vis_by_frame[0][0]); frL = int(vis_by_frame[-1][0])
            ra0, raL = rep_spans[0][0], rep_spans[-1][0]
            print(f"[mask-debug] seq={seq} allowed-keys: frame0-row {_row_allowed(fr0)}, "
                  f"frameLast-row {_row_allowed(frL)}, replica0-row {_row_allowed(ra0)}, "
                  f"replicaLast-row {_row_allowed(raL)}, final-q-row {_row_allowed(anc_c)}",
                  flush=True)
        if args.no_mask:
            m = torch.zeros(seq, seq, dtype=torch.float32)
            m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
            holder["mask"] = None          # plain causal forward; m used only for recompute rows
        else:
            holder["mask"] = m.view(1, 1, seq, seq).to(model.device)

        pos_ids = None
        if args.reset_positions:
            ends = blk_ends
            rope_fn = getattr(model, "get_rope_index", None) or model.model.get_rope_index
            with torch.inference_mode():
                base_pos, _ = rope_fn(inputs["input_ids"],
                                      image_grid_thw=inputs.get("image_grid_thw"),
                                      attention_mask=inputs.get("attention_mask"))
            pos_ids = base_pos.clone()                          # (3, 1, seq)
            s0 = vstarts[0]
            for i in range(1, NF):
                si = vstarts[i]
                pos_ids[:, :, si:ends[i]] -= int(base_pos[0, 0, si]) - int(base_pos[0, 0, s0])
            blk0_max = int(pos_ids[:, :, s0:ends[0]].max())
            fs = fin_span[0]                                    # final question: right after block 0
            pos_ids[:, :, fs:] -= int(base_pos[0, 0, fs]) - (blk0_max + 1)
            if n_done == 0:
                same_len = all(ends[i] - vstarts[i] == ends[0] - s0 for i in range(NF))
                blocks_eq = same_len and all(
                    torch.equal(pos_ids[:, :, vstarts[i]:ends[i]], pos_ids[:, :, s0:ends[0]])
                    for i in range(1, NF))
                print(f"[pos-debug] seq={seq} max_pos {int(base_pos.max())} -> {int(pos_ids.max())}"
                      f", block starts {[int(base_pos[0, 0, v]) for v in vstarts]} -> "
                      f"{[int(pos_ids[0, 0, v]) for v in vstarts]}, same_len={same_len}, "
                      f"blocks_identical={blocks_eq}, final-q start {int(base_pos[0, 0, fs])} -> "
                      f"{int(pos_ids[0, 0, fs])}", flush=True)

        with torch.inference_mode():
            model(**inputs, position_ids=pos_ids) if pos_ids is not None else model(**inputs)
        holder["mask"] = None

        cos, sin = posemb["cos"], posemb["sin"]
        for L in Ls:
            q = qkv[L]["q_proj"].view(1, seq, n_heads, hd).transpose(1, 2)
            k = qkv[L]["k_proj"].view(1, seq, n_kv, hd).transpose(1, 2)
            v = qkv[L]["v_proj"].view(1, seq, n_kv, hd).transpose(1, 2)
            qr, kr = apply_multimodal_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float(), mrope)
            kr = repeat_kv(kr, n_heads // n_kv)[0]; vv = repeat_kv(v, n_heads // n_kv)[0].float()
            qr = qr[0]
            W = oproj[L].to(qr.device)

            def msg_at(c, allowed_mask_row, frame_idx):
                lg = torch.einsum("hd,htd->ht", qr[:, c], kr) / (hd ** 0.5)
                lg = lg + allowed_mask_row                     # additive mask row (0 / MIN)
                wgt = torch.softmax(lg, -1)
                fidx = vis_by_frame[frame_idx].to(qr.device)
                ctx = torch.einsum("ht,htd->hd", wgt[:, fidx], vv[:, fidx]).reshape(-1)
                return (W @ ctx).float().cpu().numpy()

            mrow = holder  # noqa
            full_mask = m.to(qr.device)
            rep_msgs, anc_msgs = [], []
            for i in range(NF):
                rep_msgs.append(msg_at(rep_c[i], full_mask[rep_c[i]], i))
                anc_msgs.append(msg_at(anc_c, full_mask[anc_c], i))
            feats_rep[L].append(np.stack(rep_msgs)); feats_anc[L].append(np.stack(anc_msgs))

        labels_all.append([1 if t in evid else 0 for t in range(NF)])
        gold_all.append(gold)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done} samples (skip {n_skip})", flush=True)

    for h in handles:
        h.remove()
    y = np.array(labels_all); gold_arr = np.array(gold_all)
    hist = {}
    for g in gold_all:
        hist[g] = hist.get(g, 0) + 1
    print("[gold-hist] " + " ".join(f"g{g}:{c}" for g, c in sorted(hist.items())), flush=True)
    variant = ("fenced" if args.fence_frames else "masked") if not args.no_mask else "unmasked"
    if args.fence_blocks:
        variant += "+blockfence"
    if args.reset_positions:
        variant += "+posreset"
    if args.question_first:
        variant += "+qfirst"
    if args.unmix_dir:
        variant += f"+unmix_L{args.unmix_layer}"
    lines = [f"=== REPLICA CARRIERS (n={n_done}, skip={n_skip}, variant={variant}, "
             f"data={args.data_root}) ==="]
    cache = {"labels": y, "gold": gold_arr, "rep": {}, "anc": {}}
    for L in Ls:
        Xr = np.stack(feats_rep[L]); Xa = np.stack(feats_anc[L])
        cache["rep"][L], cache["anc"][L] = Xr, Xa
        dr, sr, ar = dprime_pair(Xr, y)
        da, sa, aa = dprime_pair(Xa, y)
        lines.append(f"L{L}: REPLICA read d'={dr:.2f}±{sr:.2f} (auc {ar:.2f})   "
                     f"JOINT anchor d'={da:.2f}±{sa:.2f} (auc {aa:.2f})   ratio {dr/max(da,1e-9):.2f}x")
        per = []
        for i in range(Xr.shape[1]):
            try:
                di, _, _ = dprime_pair(Xr[:, [i], :], y[:, [i]])
                per.append(f"{di:.2f}")
            except Exception:
                per.append("--")
        lines.append(f"L{L}: per-copy d' (index 0..{Xr.shape[1]-1}): " + " ".join(per))
    torch.save(cache, out / "messages_cache.pt")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
