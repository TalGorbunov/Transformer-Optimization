#!/usr/bin/env python3
"""LEARNED CARRIER TOKEN (2026-07-17): distill the per-frame question replica into ONE trainable
embedding vector.

Layout (question-first):
  [prefix + question] [frame_1][c] [frame_2][c] ... [frame_N][c] [final question]
where c is a single placeholder token (<|box_start|>, id 151648) whose input embedding is replaced
by a trained vector e_c (shared across frames/samples). Full block-diagonal fence + per-block
M-RoPE reset, exactly as the replica probe's --fence-blocks --reset-positions (Exp A3, d' 6.34).

Why this is NOT the trained-query NO-GO (d' 0.4): that was a fixed vector injected AT L16 with no
context. Here the same-sized vector enters at the INPUT, so by L16 its query has been computed
in-context from this sample's question and its own frame — sample-specific and question-
conditioned, the two properties the dead version lacked.

Objectives (--objective):
  proxy   — BCE on a jointly-trained logistic head over the carrier's L16 message vs the per-frame
            evidence label (cheap go/no-go; eval is label-free d' regardless).
  distill — cosine loss matching the carrier's L16 message to the REPLICA room-token's message
            from an in-run teacher forward (A3 layout, no_grad) on the same sample. Task-agnostic:
            no evidence labels in the loss.

In-run anchor: the teacher replica messages' held-out d' is computed and reported — it must
reproduce ~6.3 (A3) for the run to be trusted (teacher is computed in the Q-FIRST replica layout,
so the Q-first control run also brackets it).

Eval: held-out d' (dprime_pair) on carrier messages, every epoch. Pre-registered: ceiling = the
teacher/A3 band ~6.3; GO >= 5; floor to crush = trained-query 0.4-0.5.

Frozen 4-bit backbone; only e_c (+ the proxy head) train. Gradients flow through bnb Linear4bit
to the input embedding (prompt-tuning style). Forward truncated at L16 for the student pass.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import List

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
from experiments.glstm.dprime_vs_n import dprime_pair
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)

MIN = -65504.0
CARRIER_TOKEN = "<|box_start|>"
ROOMS = ("Park", "Garden", "Bathroom", "Kitchen", "Office", "Bedroom")


def find_subseq(hay: List[int], needle: List[int]) -> List[int]:
    out, n = [], len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            out.append(i)
    return out


def build_block_mask(seq: int, blocks: List[tuple], hide_cols: List[int]) -> torch.Tensor:
    """Causal + full block-diagonal fence + globally hidden columns (carrier/replica spans)."""
    m = torch.zeros(seq, seq, dtype=torch.float32)
    m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1), MIN)
    if hide_cols:
        # globally hide (carrier/replica spans); own-block visibility is restored below when
        # each block's causal sub-mask is rewritten
        m[:, torch.tensor(sorted(hide_cols), dtype=torch.long)] = MIN
    for i, (a, b) in enumerate(blocks):
        rows = torch.arange(a, b)
        # re-open own-block columns (undo global hide within the block, causality intact)
        own = torch.arange(a, b)
        causal = torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1)
        blk = torch.zeros(b - a, b - a); blk.masked_fill_(causal, MIN)
        m[a:b, a:b] = blk
        for j, (a2, b2) in enumerate(blocks):
            if j != i:
                m[rows.unsqueeze(1), torch.arange(a2, b2).unsqueeze(0)] = MIN
    return m


def reset_positions(base_pos: torch.Tensor, blocks: List[tuple], fin_start: int) -> torch.Tensor:
    pos = base_pos.clone()
    s0, e0 = blocks[0]
    for (si, ei) in blocks[1:]:
        pos[:, :, si:ei] -= int(base_pos[0, 0, si]) - int(base_pos[0, 0, s0])
    blk0_max = int(pos[:, :, s0:e0].max())
    pos[:, :, fin_start:] -= int(base_pos[0, 0, fin_start]) - (blk0_max + 1)
    return pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--train-n", type=int, default=150)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--k", type=int, default=1, help="carrier tokens per frame")
    ap.add_argument("--objective", choices=("proxy", "distill"), default="proxy")
    ap.add_argument("--init", choices=("room", "random"), default="room")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/ladder/image_longN/carrier_token")
    ap.add_argument("--task", choices=("steps", "cooc"), default="steps",
                    help="cooc: co-occupancy evidence (both names share a room per states); "
                         "carrier/student layout is task-independent — only labels change.")
    ap.add_argument("--natural", action="store_true",
                    help="mmred_natural cells: meta.json loader, evidence from is_evidence flags")
    ap.add_argument("--eval-only", action="store_true",
                    help="No training: load --carrier-ckpt, stream samples (no RAM cache, no "
                         "teacher forward), report d' + ckpt-head tally + fresh-logistic tally. "
                         "For OOD / length-generalization of a trained carrier.")
    ap.add_argument("--carrier-ckpt", default=None)
    ap.add_argument("--carriers-at-end", action="store_true",
                    help="E-C layout freedom (2026-07-19): [frames][question][c x N] — no "
                         "leading question, carriers AFTER the question as separate tokens; "
                         "carrier_i reads {prefix, question, frame_i, itself}. Teacher becomes "
                         "the FRAMES-FIRST blockfence replica (A3 layout, anchor ~6.3).")
    ap.add_argument("--atend-qfirst", action="store_true",
                    help="E-C(b): with --carriers-at-end, RESTORE the leading question "
                         "([q0][frames][q0][c x N]) and the Q-FIRST teacher — isolates "
                         "question-conditioned frame encoding from carrier adjacency.")
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED",
                    help="stratified deterministic shuffle of sample dirs (class-balanced "
                         "prefixes; REQUIRED whenever --limit < the full dir count)")
    args = ap.parse_args()
    if args.eval_only and not args.carrier_ckpt:
        ap.error("--eval-only requires --carrier-ckpt")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    L = args.layer
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads, n_kv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope = cfg.rope_scaling["mrope_section"]
    tok = processor.tokenizer
    text_model = model.model.language_model
    dev = model.device
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S") + f"_{args.objective}_{args.init}_k{args.k}")
    out.mkdir(parents=True, exist_ok=True)

    import bitsandbytes.functional as bnbF
    w = layers[L].self_attn.o_proj.weight
    W_O = bnbF.dequantize_4bit(w.data, w.quant_state).float()

    cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    vs_id = int(model.config.vision_start_token_id)
    rope_fn = getattr(model, "get_rope_index", None) or model.model.get_rope_index

    # ---- trainable params (created up front so eval-only streaming can use them) ----
    D = cfg.hidden_size
    if args.eval_only:
        ck = torch.load(args.carrier_ckpt, map_location="cpu")
        e_c = nn.Parameter(ck["e_c"].float().to(dev))
        e_extra = (nn.Parameter(ck["e_extra"].float().to(dev))
                   if ck.get("e_extra") is not None else None)
        head_w = nn.Parameter(ck["head_w"].float().to(dev))
        head_b = nn.Parameter(ck["head_b"].float().to(dev))
        args.k = 1 + (e_extra.shape[0] if e_extra is not None else 0)
        print(f"[eval-only] loaded carrier from {args.carrier_ckpt} "
              f"(trained ep {ck.get('epoch')}, d' {ck.get('dprime'):.2f}, k={args.k})", flush=True)
    else:
        if args.init == "room":
            rows = []
            for r in ROOMS:
                tid = tok(" " + r, add_special_tokens=False).input_ids
                rows.append(text_model.embed_tokens.weight[tid[-1]].float())
            base_vec = torch.stack(rows).mean(0)
        else:
            base_vec = text_model.embed_tokens.weight.float().std() * torch.randn(D, device=dev)
        e_c = nn.Parameter(base_vec.detach().float().to(dev))
        e_extra = (nn.Parameter(base_vec.detach().float().to(dev).repeat(args.k - 1, 1)
                                + 0.01 * torch.randn(args.k - 1, D, device=dev))
                   if args.k > 1 else None)
        head_w = nn.Parameter(torch.zeros(D, device=dev)); head_b = nn.Parameter(torch.zeros(1, device=dev))

    # ---- teacher hooks (mask injection for the full-model replica forward) ----
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

    qkv = {}
    def mk_qkv(nm):
        def hook(_m, _i, o):
            qkv[nm] = o.detach()
        return hook
    posemb = {}
    def pe_hook(_m, a_, k_):
        pe = k_.get("position_embeddings")
        if pe is None and len(a_) >= 1 and isinstance(a_[-1], tuple):
            pe = a_[-1]
        if pe is not None:
            posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
    handles = [ly.register_forward_pre_hook(mask_pre, with_kwargs=True) for ly in layers]
    for nm in ("q_proj", "k_proj", "v_proj"):
        handles.append(getattr(layers[L].self_attn, nm).register_forward_hook(mk_qkv(nm)))
    handles.append(layers[L].self_attn.register_forward_pre_hook(pe_hook, with_kwargs=True))

    def messages_from_qkv(seq, mask_full, carriers, vis_by_frame, cos, sin, differentiable=False,
                          q_t=None, k_t=None, v_t=None):
        q = (qkv["q_proj"] if q_t is None else q_t).view(1, seq, n_heads, hd).transpose(1, 2)
        k = (qkv["k_proj"] if k_t is None else k_t).view(1, seq, n_kv, hd).transpose(1, 2)
        v = (qkv["v_proj"] if v_t is None else v_t).view(1, seq, n_kv, hd).transpose(1, 2)
        qr, kr = apply_multimodal_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float(), mrope)
        kr = repeat_kv(kr, n_heads // n_kv)[0]; vv = repeat_kv(v, n_heads // n_kv)[0].float()
        qr = qr[0]
        W = W_O.to(qr.device)
        msgs = []
        for i, c in enumerate(carriers):
            lg = torch.einsum("hd,htd->ht", qr[:, c], kr) / (hd ** 0.5)
            lg = lg + mask_full[c].to(qr.device)
            wgt = torch.softmax(lg, -1)
            fidx = vis_by_frame[i].to(qr.device)
            ctx = torch.einsum("ht,htd->hd", wgt[:, fidx], vv[:, fidx]).reshape(-1)
            msgs.append(W @ ctx)
        st = torch.stack(msgs)
        return st if differentiable else st.detach().cpu().numpy()

    def student_msgs(d, differentiable):
        seq = d["seq"]
        emb = d["emb"].to(dev).unsqueeze(0)
        if differentiable:
            emb = emb.clone()
        vecs = torch.cat([e_extra, e_c.unsqueeze(0)]) if args.k > 1 else e_c.unsqueeze(0)
        NF = len(d["carriers"])
        stack = vecs.repeat(NF, 1).to(torch.bfloat16)
        emb[0, torch.tensor(d["cpos"], device=dev)] = stack if differentiable else stack.detach()
        mask4 = d["mask"].to(dev).to(torch.float32).view(1, 1, seq, seq)
        pos = d["pos"].to(dev)
        cos_, sin_ = text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        h = emb
        for ly in layers[:L]:
            h = ly(h, attention_mask=mask4, position_embeddings=pe)[0]
        ln = layers[L].input_layernorm(h)
        at = layers[L].self_attn
        q_t, k_t, v_t = at.q_proj(ln), at.k_proj(ln), at.v_proj(ln)
        cos, sin = pe
        return messages_from_qkv(seq, d["mask"], d["carriers"], d["vis"], cos, sin,
                                 differentiable=differentiable, q_t=q_t, k_t=k_t, v_t=v_t)

    # ---- preprocessing: cache student inputs + teacher targets per sample ----
    # (eval-only: stream — compute messages immediately, never cache the heavy tensors)
    data = []
    Xs_stream, ys_stream, gold_stream = [], [], []
    n_done = n_skip = 0
    t0 = time.time()
    if args.natural:
        sample_dirs = sorted(d for d in Path(args.data_root).iterdir() if d.is_dir())
    elif args.shuffle_dirs is not None:
        sample_dirs = iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs)
    else:
        sample_dirs = iter_sample_dirs(Path(args.data_root))
    for sd in sample_dirs:
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
            import re as _re
            mm = _re.search(r"were (\w+) and (\w+) in the same room", q0)
            if not mm:
                n_skip += 1; continue
            nA, nB = mm.group(1), mm.group(2)
            evid = set()
            for t, st in enumerate(states):
                for occ in (st.get("rooms", {}) or {}).values():
                    if nA in occ and nB in occ:
                        evid.add(t); break
            if len(evid) != gold:
                n_skip += 1; continue
            room = nB                          # teacher replica locus (unused in eval-only)
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

        # ---------- STUDENT layout: Q-first + carrier placeholders ----------
        if args.carriers_at_end:            # E-C: [frames][question][c x N]
            content = [{"type": "text", "text": q0}] if args.atend_qfirst else []
            for f in frames:
                content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
            for _ in range(NF):
                content.append({"type": "text", "text": CARRIER_TOKEN * args.k})
        else:
            content = [{"type": "text", "text": q0}]
            for f in frames:
                content.append({"type": "image", "image": f})
                content.append({"type": "text", "text": CARRIER_TOKEN * args.k})
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
        n_occ = 1 if (args.carriers_at_end and not args.atend_qfirst) else 2
        for pre in ("", " ", "\n"):
            needle = tok(pre + q0, add_special_tokens=False).input_ids
            o = find_subseq(ids, needle)
            if len(o) == n_occ:
                occ = o; break
        if len(fg) != NF or len(cpos) != NF * args.k or len(vstarts) != NF or occ is None:
            n_skip += 1; continue
        fin_start = occ[-1]        # at-end without qfirst: occ[-1]==occ[0]; else final q0
        carriers = [cpos[(i + 1) * args.k - 1] for i in range(NF)]     # last carrier tok per frame
        blocks = [(vstarts[i], (vstarts[i + 1] if i + 1 < NF else fin_start)) for i in range(NF)]
        mask_s = build_block_mask(seq, blocks, hide_cols=cpos)
        if args.carriers_at_end:
            # carrier rows sit OUTSIDE all blocks: causal minus hidden-carrier cols. Fix them:
            # block out other frames, re-open own frame block + self (prefix+question stay
            # open via causal). Non-carrier rows already can't see carriers (hide_cols).
            for i in range(NF):
                a_i, b_i = blocks[i]
                for kk in range(args.k):
                    c = cpos[i * args.k + kk]
                    for j, (a, b) in enumerate(blocks):
                        if j != i:
                            mask_s[c, a:b] = MIN
                    mask_s[c, a_i:b_i] = 0.0
                    mask_s[c, c] = 0.0
            if n_done == 0:
                alw = [int((mask_s[c] == 0).sum()) for c in carriers]
                blk = blocks[0][1] - blocks[0][0]
                qspan = occ[0]  # question begins here; prefix = [0, first vstart)
                print(f"[mask-debug E-C] seq={seq} carrier allowed-keys {alw} "
                      f"identical={len(set(alw)) == 1}; block_len={blk} "
                      f"(expect prefix+question+one_block+self; prefix_end={vstarts[0]}, "
                      f"q_start={qspan})", flush=True)
        # no_grad (NOT inference_mode): these tensors are later consumed inside autograd graphs
        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos_s = reset_positions(base_pos, blocks, fin_start).clone()
            emb = text_model.embed_tokens(inputs["input_ids"])
            img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb = emb.clone(); emb[0, im_mask] = img.to(emb.dtype)
        vis_by_frame = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]

        if args.eval_only:
            rec = {"emb": emb[0].to(torch.bfloat16), "mask": mask_s.to(torch.float16),
                   "pos": pos_s, "carriers": carriers, "cpos": cpos, "vis": vis_by_frame,
                   "seq": seq}
            with torch.no_grad():
                Xs_stream.append(student_msgs(rec, False))
            ys_stream.append(np.array([1 if t in evid else 0 for t in range(NF)]))
            gold_stream.append(gold)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  eval {n_done} (skip {n_skip}) {time.time()-t0:.0f}s", flush=True)
            continue

        # ---------- TEACHER: replica layout (A3 = fence-blocks + posreset) ----------
        # Q-first by default; E-C uses the FRAMES-FIRST A3 teacher (no leading question).
        t_frames_first = args.carriers_at_end and not args.atend_qfirst
        contentT = [] if t_frames_first else [{"type": "text", "text": q0}]
        for f in frames:
            contentT.append({"type": "image", "image": f})
            contentT.append({"type": "text", "text": q0})
        contentT.append({"type": "text", "text": q0})
        inputsT = processor.apply_chat_template([{"role": "user", "content": contentT}],
                                                add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt")
        inputsT = {k2: (v.to(dev) if hasattr(v, "to") else v) for k2, v in inputsT.items()}
        idsT = inputsT["input_ids"][0].tolist(); seqT = len(idsT)
        fgT = image_token_groups(inputsT["input_ids"][0].cpu(), expected_num_frames=NF, processor=processor)
        vstartsT = [p for p, t in enumerate(idsT) if t == vs_id]
        spansT = None
        n_spans = NF + 1 if t_frames_first else NF + 2
        for pre in ("", " ", "\n"):
            needle = tok(pre + q0, add_special_tokens=False).input_ids
            o = find_subseq(idsT, needle)
            if len(o) == n_spans:
                spansT = [(x, x + len(needle)) for x in o]; break
        if len(fgT) != NF or len(vstartsT) != NF or spansT is None:
            n_skip += 1; continue
        if not t_frames_first:
            spansT = spansT[1:]
        repT, finT = spansT[:NF], spansT[NF]

        def room_pos(a, b):
            for p in range(b - 1, a - 1, -1):
                if room[:4].lower() in tok.decode([idsT[p]]).strip().lower():
                    return p
            return None
        repC = [room_pos(a, b) for a, b in repT]
        if any(c is None for c in repC):
            n_skip += 1; continue
        blocksT = [(vstartsT[i], (vstartsT[i + 1] if i + 1 < NF else finT[0])) for i in range(NF)]
        hideT = [p for a, b in repT for p in range(a, b)]
        mask_t = build_block_mask(seqT, blocksT, hide_cols=hideT)
        vis_by_frameT = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fgT]
        with torch.inference_mode():
            base_posT, _ = rope_fn(inputsT["input_ids"], image_grid_thw=inputsT.get("image_grid_thw"),
                                   attention_mask=inputsT.get("attention_mask"))
            pos_t = reset_positions(base_posT, blocksT, finT[0])
            holder["mask"] = mask_t.view(1, 1, seqT, seqT).to(dev)
            model(**inputsT, position_ids=pos_t)
            holder["mask"] = None
        t_msgs = messages_from_qkv(seqT, mask_t, repC, vis_by_frameT,
                                   posemb["cos"], posemb["sin"])

        data.append({
            "emb": emb[0].to(torch.bfloat16).cpu(), "mask": mask_s.to(torch.float16),
            "pos": pos_s.cpu(), "carriers": carriers, "cpos": cpos,
            "vis": vis_by_frame, "seq": seq,
            "teacher": t_msgs.astype(np.float32),
            "y": np.array([1 if t in evid else 0 for t in range(NF)]), "gold": gold,
        })
        n_done += 1
        if n_done % 25 == 0:
            print(f"  prep {n_done} (skip {n_skip}) {time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    print(f"prep done: n={n_done} skip={n_skip} ({time.time()-t0:.0f}s)", flush=True)
    hist = {}
    for g in (gold_stream if args.eval_only else [d["gold"] for d in data]):
        hist[g] = hist.get(g, 0) + 1
    print("[gold-hist] " + " ".join(f"g{g}:{c}" for g, c in sorted(hist.items())), flush=True)

    if args.eval_only:
        X = np.stack(Xs_stream); Y = np.stack(ys_stream); G = np.array(gold_stream)
        d_, s_, a_ = dprime_pair(X, Y)
        prob = 1 / (1 + np.exp(-(X @ head_w.detach().cpu().numpy() + float(head_b))))
        pred = (prob > 0.5).astype(int)
        ferr_ck = float((pred != Y).mean()); acc_ck = float((pred.sum(1) == G).mean())
        from sklearn.linear_model import LogisticRegression
        n, NF_, Dm = X.shape
        accs, ferrs = [], []
        for seed in range(5):
            r2 = np.random.default_rng(seed); idx = r2.permutation(n)
            tr, ev = idx[:n // 2], idx[n // 2:]
            clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, Dm), Y[tr].reshape(-1))
            pr = clf.predict(X[ev].reshape(-1, Dm)).reshape(len(ev), NF_)
            ferrs.append(1 - (pr == Y[ev]).mean()); accs.append((pr.sum(1) == G[ev]).mean())
        lines = [f"=== CARRIER EVAL-ONLY (ckpt={args.carrier_ckpt}, n={n_done}, NF={NF_}, "
                 f"data={args.data_root}) ===",
                 f"d' {d_:.2f}±{s_:.2f} (auc {a_:.2f})",
                 f"ckpt-head ZERO-SHOT: per-frame err {ferr_ck:.4f}, tally exact {acc_ck:.3f}",
                 f"fresh logistic (5 seeds): err {np.mean(ferrs):.4f}, "
                 f"exact {np.mean(accs):.3f}±{np.std(accs):.3f}"]
        (out / "report.txt").write_text("\n".join(lines) + "\n")
        np.savez(out / "messages_eval.npz", X=X, Y=Y, G=G)
        print("\n".join(lines)); print("wrote", out)
        return 0

    y_all = np.stack([d["y"] for d in data])
    T_all = np.stack([d["teacher"] for d in data])
    dT, sT, aT = dprime_pair(T_all, y_all)
    print(f"[anchor] teacher (Q-first replica blockfence+posreset) held-out d' = {dT:.2f}±{sT:.2f} "
          f"(auc {aT:.2f}) — expect ~6.3 (A3 band)", flush=True)

    params = [e_c, head_w, head_b] + ([e_extra] if e_extra is not None else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_done)
    tr_idx, ev_idx = order[:args.train_n], order[args.train_n:]
    dTe, sTe, _ = dprime_pair(T_all[ev_idx], y_all[ev_idx])
    print(f"[anchor] teacher EVAL-split d' = {dTe:.2f}±{sTe:.2f} (scale-matched ceiling for the "
          f"eval column)", flush=True)

    def eval_dprime():
        Xs, ys = [], []
        with torch.no_grad():
            for i in range(n_done):
                Xs.append(student_msgs(data[i], False))
                ys.append(data[i]["y"])
        X = np.stack(Xs); Y = np.stack(ys)
        # headline = held-out split only (e_c never trained on it); full-n for scale reference
        d_, s_, a_ = dprime_pair(X[ev_idx], Y[ev_idx])
        df, _, _ = dprime_pair(X, Y)
        return d_, s_, a_, df, X, Y

    d0, s0_, a0_, df0, _, _ = eval_dprime()
    print(f"[ep 0 / init={args.init}] carrier d' eval = {d0:.2f}±{s0_:.2f} (auc {a0_:.2f}, "
          f"full {df0:.2f})", flush=True)
    traj = [(0, float(d0))]
    lines = [f"=== CARRIER TOKEN (obj={args.objective}, init={args.init}, k={args.k}, n={n_done}, "
             f"train={len(tr_idx)}, L={L}, data={args.data_root}) ===",
             f"teacher anchor d' {dT:.2f}±{sT:.2f} full-n (expect ~6.3) / {dTe:.2f}±{sTe:.2f} "
             f"eval-split (the scale-matched ceiling)",
             f"ep0 d' eval {d0:.2f}±{s0_:.2f} full {df0:.2f}"]

    best = (float(d0), 0)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        tot = 0.0
        for step, i in enumerate(tr_idx):
            d = data[i]
            msgs = student_msgs(d, True)                          # (NF, D) float32, grads on
            if args.objective == "proxy":
                logits = msgs @ head_w + head_b
                loss = F.binary_cross_entropy_with_logits(
                    logits, torch.tensor(d["y"], dtype=torch.float32, device=dev))
            else:
                tgt = torch.tensor(d["teacher"], device=dev)
                loss = (1 - F.cosine_similarity(msgs, tgt, dim=-1)).mean()
            loss = loss / 4
            loss.backward()
            tot += float(loss) * 4
            if (step + 1) % 4 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        d_, s_, a_, df_, X, Y = eval_dprime()
        traj.append((ep, float(d_)))
        print(f"[ep {ep}] loss {tot/len(tr_idx):.4f}  carrier d' eval = {d_:.2f}±{s_:.2f} "
              f"(auc {a_:.2f}, full {df_:.2f})", flush=True)
        lines.append(f"ep{ep} loss {tot/len(tr_idx):.4f} d' eval {d_:.2f}±{s_:.2f} full {df_:.2f}")
        if d_ > best[0]:
            best = (float(d_), ep)
            torch.save({"e_c": e_c.detach().cpu(),
                        "e_extra": (e_extra.detach().cpu() if args.k > 1 else None),
                        "head_w": head_w.detach().cpu(), "head_b": head_b.detach().cpu(),
                        "epoch": ep, "dprime": float(d_)}, out / "carrier_best.pt")
            np.savez(out / "messages_best.npz", X=X, Y=Y)

    lines.append(f"BEST d' {best[0]:.2f} @ ep {best[1]}  (teacher {dT:.2f}; trained-query floor 0.4)")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "traj.csv").write_text("epoch,dprime\n" + "\n".join(f"{e},{v:.4f}" for e, v in traj) + "\n")
    print("\n".join(lines[-3:])); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
