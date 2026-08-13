#!/usr/bin/env python3
"""A1-fu1: text-MMRED MULTIPASS message cache — the definitive write-cap test.

Each frame's text block is fed ALONE (isolated forward, question suffix identical to the joint
text battery), and the frame->carrier messages are cached at the same question-token offsets as
the joint blockcache. If the text carrier d' rises under isolation (as image multipass does,
~2.5 -> 6+), the joint text supply is interference/binding-limited; if multipass d' stays at the
joint value (~2.45 block), the carrier-write process itself is capped — modality-independent.

Output: messages_cache.pt with the SAME schema as probe_frame_to_carrier_message.py
--save-messages, so block_read_completeness.py / probe_dprime_parity.py run on it unchanged.
  msgs[L][off] = [n, NF, hidden] f16; labels = per-frame evidence 0/1; gold = count.
model_correct here = "multipass solution" accuracy: sum of per-frame digit answers == gold.
perception (extra key) = per-frame single-frame answer correctness (0/1 vs frame evidence).
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)


def frame_block(st: dict, idx1: int, style: str) -> str:
    """Render ONE frame as text, idx1 = 1-based frame number (preserves joint headers)."""
    if style == "compact":
        pairs = sorted((ch, room) for room, occ in st["rooms"].items() for ch in occ)
        return f"Frame {idx1}: " + ", ".join(f"{ch}@{room}" for ch, room in pairs) + "."
    blk = [f"Frame {idx1}:"]
    for room, occ in st["rooms"].items():
        who = ", ".join(occ) if occ else "(empty)"
        blk.append(f"  {room}: {who}")
    return "\n".join(blk)


def build_single_frame_inputs(block: str, question: str, processor, hi: int):
    """Single-frame prompt with the SAME template as the joint text battery (n_frames=1)."""
    prompt = (f"You are given 1 frames describing steps in a house, as text.\n"
              + block + "\n\n"
              + f"Respond with a single integer from 0 to {hi} (0 is allowed). Output only the integer.\n"
              + f"Question: {question}\n"
              + "Answer: ")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ftok = getattr(build_single_frame_inputs, "_fast_tok", None)
    if ftok is None:
        from transformers import AutoTokenizer
        name = getattr(processor.tokenizer, "name_or_path", None) or "Qwen/Qwen2.5-VL-7B-Instruct"
        ftok = AutoTokenizer.from_pretrained(name, use_fast=True)
        assert ftok.is_fast
        build_single_frame_inputs._fast_tok = ftok
    enc = ftok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    if not getattr(build_single_frame_inputs, "_checked", False):
        ref = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt")
        assert torch.equal(enc["input_ids"], ref["input_ids"]), \
            "fast-tokenizer ids != runtime chat-template ids"
        build_single_frame_inputs._checked = True
        print("[multipass] tokenization self-check passed")
    offmap = enc["offset_mapping"][0].tolist()
    start = text.index(block); end = start + len(block)
    grp = [ti for ti, (a, b) in enumerate(offmap) if a >= start and b <= end and b > a]
    if not grp:
        raise RuntimeError("empty token group for the frame block")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, grp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--layers", default="14,16,18")
    ap.add_argument("--decode-offsets", default="0,2,5,8,9,11,13")
    ap.add_argument("--model_name", "--model", dest="model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--text-style", choices=["rooms", "compact"], default="rooms")
    ap.add_argument("--modality", choices=["text", "image"], default="text",
                    help="image = B1 multipass arm: each PNG frame fed alone (same question); "
                         "frame group = the image token block")
    ap.add_argument("--chunk-k", type=int, default=1,
                    help="P1e chunk-size sweep: process frames in groups of k per forward "
                         "(k=1 = multipass; k=N = joint). Image modality only. Messages for "
                         "each frame in the chunk are captured from that chunk's forward.")
    ap.add_argument("--pad-to-frames", type=int, default=0,
                    help="P1a position-only control: k=1 isolated frame, but preceded by "
                         "question-neutral filler TEXT sized like (pad_to_frames-1) frames' "
                         "tokens, so the frame sits at its joint-context position.")
    ap.add_argument("--companion", default="",
                    choices=["", "gray", "noise", "shuffle", "otherscene", "samescene"],
                    help="P1b companion-content ladder: k=2 chunks of [real frame, companion]. "
                         "gray = uniform gray image; noise = uniform pixel noise; shuffle = "
                         "patch-shuffled copy of a same-scene frame; otherscene = a frame from "
                         "a different sample; samescene = another frame of this sample (anchor).")
    ap.add_argument("--resize", type=int, default=0,
                    help="image modality: resize frames to <resize>px before the processor")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    probe_layers = [int(x) for x in str(args.layers).replace(",", " ").split()]
    DEC_OFF = [int(x) for x in str(args.decode_offsets).replace(",", " ").split()]
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope_section = (getattr(cfg, "rope_scaling", None) or {}).get("mrope_section", None)
    attn_scale = head_dim ** -0.5
    NF = int(args.n_frames)
    hidden = int(cfg.hidden_size)
    out = Path(args.output) / "count"; out.mkdir(parents=True, exist_ok=True)
    tok = processor.tokenizer
    cand_ids, cand_vals = [], []
    for d in range(0, 9):
        enc = tok.encode(str(d), add_special_tokens=False)
        if len(enc) == 1:
            cand_ids.append(int(enc[0])); cand_vals.append(d)
    cand_ids_t = torch.tensor(cand_ids, dtype=torch.long)

    # capture hooks (registered once; read after each forward)
    qkv: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in probe_layers}
    posemb: Dict[str, torch.Tensor] = {}
    for L in probe_layers:
        for nm in ("q_proj", "k_proj", "v_proj"):
            def mk(L, nm):
                def hook(_m, _i, o):
                    qkv[L][nm] = o.detach()[0]
                return hook
            getattr(layers[L].self_attn, nm).register_forward_hook(mk(L, nm))
    def mk_pe(_m, args_, kwargs_):
        pe = kwargs_.get("position_embeddings", None)
        if pe is None and len(args_) >= 1 and isinstance(args_[-1], tuple):
            pe = args_[-1]
        if pe is not None:
            posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
    layers[probe_layers[0]].self_attn.register_forward_pre_hook(mk_pe, with_kwargs=True)

    def frame_message(inputs, groups) -> tuple:
        """One forward over a chunk of frames; returns (per-group {off: msg} per layer list,
        digit pred). groups = list of token-position lists, one per frame in the chunk."""
        inputs = tgi.move_inputs_to_model_device(inputs)
        with torch.no_grad():
            outp = model(**inputs, use_cache=False)
        seq = int(inputs["input_ids"].shape[1])
        last_img = max(p for grp in groups for p in grp)
        q_span = list(range(last_img + 1, seq))
        carrier = q_span
        carrier_t = torch.tensor(carrier, dtype=torch.long)
        off_to_ci = {(len(carrier) - 1) - ci: ci for ci in range(len(carrier))}
        cos, sin = posemb["cos"], posemb["sin"]
        msgs = [dict() for _ in groups]     # per frame in chunk: {L: {off: vec}}
        for L in probe_layers:
            q = qkv[L]["q_proj"].view(1, seq, n_heads, head_dim).transpose(1, 2)
            k = qkv[L]["k_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
            v = qkv[L]["v_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
            if mrope_section is not None:
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
            k = repeat_kv(k, n_heads // n_kv); v = repeat_kv(v, n_heads // n_kv)
            qf = q[0].float().cpu(); kf = k[0].float().cpu()
            scores = torch.einsum("hcd,hkd->hck", qf[:, carrier_t], kf) * attn_scale
            key_idx = torch.arange(seq); allow = key_idx[None, :] <= carrier_t[:, None]
            scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
            A = torch.softmax(scores, dim=-1)                       # [H,|C|,S]
            vf = v[0].float().cpu()                                 # [H,S,hd]
            oproj = layers[L].self_attn.o_proj
            dev = next(oproj.parameters()).device
            for gi, grp in enumerate(groups):
                pos = torch.tensor(grp, dtype=torch.long)
                Asel = A[:, :, pos]; vsel = vf[:, pos, :]
                ctx = torch.einsum("hcj,hjd->hcd", Asel, vsel)      # [H,|C|,hd]
                ctx = ctx.permute(1, 0, 2).reshape(len(carrier), -1)
                with torch.no_grad():
                    mm = oproj(ctx.to(device=dev, dtype=torch.bfloat16)).float().cpu().numpy()
                msgs[gi][L] = {o: mm[ci].astype(np.float16)
                               for o, ci in off_to_ci.items() if o in DEC_OFF}
        last_logits = outp.logits[0, -1].float().cpu()
        pred = int(cand_vals[int(torch.argmax(last_logits[cand_ids_t]).item())])
        return msgs, pred

    dec = {L: {o: [] for o in DEC_OFF} for L in probe_layers}
    dec_gold: List[int] = []
    dec_labels: List[np.ndarray] = []
    dec_labels_raw: List[list] = []
    mp_sum_correct: List[int] = []
    perception: List[np.ndarray] = []
    n = 0
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    for sd in all_dirs:
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception:
            continue
        if states is None or len(states) < NF:
            continue
        try:
            gold = int(str(a0).strip())
        except Exception:
            continue
        evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
        if not evid:
            continue
        if args.modality == "image" and len(frames) < NF:
            continue
        try:
            per_dec = {L: {o: np.zeros((NF, hidden), dtype=np.float16) for o in DEC_OFF}
                       for L in probe_layers}
            rs = int(args.resize)

            def rz(fr):
                return fr.resize((rs, rs)) if rs > 0 else fr

            if args.companion:                      # P1b: [real frame, companion] pairs
                from PIL import Image as _Image
                preds = np.zeros(NF, dtype=np.int64)
                chunk_evid = np.zeros(NF, dtype=np.int64)
                if args.companion == "otherscene" and not globals().get("_prev_frames"):
                    globals()["_prev_frames"] = [rz(f) for f in frames]
                    continue                        # first sample seeds the buffer
                for t in range(NF):
                    fr = rz(frames[t])
                    if args.companion == "gray":
                        comp = _Image.new("RGB", fr.size, (128, 128, 128))
                    elif args.companion == "noise":
                        rngn = np.random.RandomState(hash(sid) % 2**31 + t)
                        comp = _Image.fromarray(
                            rngn.randint(0, 256, (*fr.size[::-1], 3), dtype=np.uint8))
                    elif args.companion == "shuffle":
                        src = np.asarray(rz(frames[(t + 1) % NF]))
                        P = 28
                        h, w = (src.shape[0] // P) * P, (src.shape[1] // P) * P
                        tiles = [src[i:i+P, j:j+P] for i in range(0, h, P) for j in range(0, w, P)]
                        rngs = np.random.RandomState(hash(sid) % 2**31 + t)
                        rngs.shuffle(tiles)
                        cols = w // P
                        rows_ = [np.concatenate(tiles[r*cols:(r+1)*cols], axis=1)
                                 for r in range(h // P)]
                        comp = _Image.fromarray(np.concatenate(rows_, axis=0))
                    elif args.companion == "otherscene":
                        comp = globals()["_prev_frames"][t % len(globals()["_prev_frames"])]
                    else:                            # samescene
                        comp = rz(frames[(t + 1) % NF])
                    inputs = tgi.build_inputs([fr, comp], q0)
                    groups = [[int(p) for p in g] for g in image_token_groups(
                        inputs["input_ids"][0].detach().cpu(), expected_num_frames=2,
                        processor=processor)]
                    msgs, pred_c = frame_message(inputs, groups)
                    preds[t] = pred_c
                    chunk_evid[t] = int(t in evid)
                    for L in probe_layers:            # capture the REAL frame (group 0) only
                        for o, m in msgs[0][L].items():
                            per_dec[L][o][t] = m
                if args.companion == "otherscene":
                    globals()["_prev_frames"] = [rz(f) for f in frames]
            elif int(args.pad_to_frames) > 1:        # P1a: isolated frame at joint positions
                from evaluations.helpers.patching_core import build_prompt as _bp
                preds = np.zeros(NF, dtype=np.int64)
                chunk_evid = np.zeros(NF, dtype=np.int64)
                base = ("The weather report for today mentions mild temperatures, light winds "
                        "and clear skies across the region. ")
                ntok = len(processor.tokenizer.encode(base, add_special_tokens=False))
                reps = max(1, round((int(args.pad_to_frames) - 1) * 196 / ntok))
                filler = base * reps
                for t in range(NF):
                    fr = rz(frames[t])
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": filler},
                        {"type": "image", "image": fr},
                        {"type": "text", "text": _bp(q0, num_frames=1)}]}]
                    inputs = dict(processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"))
                    groups = [[int(p) for p in g] for g in image_token_groups(
                        inputs["input_ids"][0].detach().cpu(), expected_num_frames=1,
                        processor=processor)]
                    msgs, pred_c = frame_message(inputs, groups)
                    preds[t] = pred_c
                    chunk_evid[t] = int(t in evid)
                    for L in probe_layers:
                        for o, m in msgs[0][L].items():
                            per_dec[L][o][t] = m
            else:
                K = max(1, int(args.chunk_k))
                n_chunks = (NF + K - 1) // K
                preds = np.zeros(n_chunks, dtype=np.int64)   # per-CHUNK digit answer
                chunk_evid = np.zeros(n_chunks, dtype=np.int64)
                for ci, c0 in enumerate(range(0, NF, K)):
                    idxs = list(range(c0, min(c0 + K, NF)))
                    if args.modality == "image":
                        frs = [rz(frames[t]) for t in idxs]
                        inputs = tgi.build_inputs(frs, q0)
                        groups = [[int(p) for p in g] for g in image_token_groups(
                            inputs["input_ids"][0].detach().cpu(),
                            expected_num_frames=len(frs), processor=processor)]
                    else:
                        assert K == 1, "chunking implemented for image modality only"
                        block = frame_block(states[idxs[0]], idxs[0] + 1, args.text_style)
                        inputs, grp = build_single_frame_inputs(block, q0, processor, hi=NF)
                        groups = [grp]
                    msgs, pred_c = frame_message(inputs, groups)
                    preds[ci] = pred_c
                    chunk_evid[ci] = sum(1 for t in idxs if t in evid)
                    for gi, t in enumerate(idxs):
                        for L in probe_layers:
                            for o, m in msgs[gi][L].items():
                                per_dec[L][o][t] = m
        except Exception as exc:
            print(f"{sid} capture failed: {type(exc).__name__}: {exc}")
            fail_count = globals().get("_fail_count", 0) + 1
            globals()["_fail_count"] = fail_count
            if fail_count >= 25 and n == 0:
                raise
            continue
        lab = np.array([1 if t in evid else 0 for t in range(NF)], dtype=np.int64)
        for L in probe_layers:
            for o in DEC_OFF:
                dec[L][o].append(per_dec[L][o])
        dec_gold.append(gold)
        dec_labels.append(lab)
        dec_labels_raw.append(["evid" if t in evid else "noev" for t in range(NF)])
        # chunked-tally solution: sum of per-chunk digit answers (chunk answer clipped to
        # the chunk size); K=1 reduces to the original multipass-sum
        Kc = max(1, int(args.chunk_k))
        mp_sum_correct.append(int(int(np.minimum(preds, Kc).sum()) == gold))
        # per-chunk perception: chunk answer == #evidence in chunk
        perception.append((np.minimum(preds, Kc) == chunk_evid).astype(np.int64))
        n += 1
        if n % 10 == 0:
            perc = float(np.concatenate(perception).mean())
            print(f"  scanned {n}: mp-sum acc {np.mean(mp_sum_correct):.3f}, "
                  f"per-frame perception {perc:.3f}", flush=True)

    cache_obj = {"msgs": {L: {o: np.stack(dec[L][o]) for o in DEC_OFF} for L in probe_layers},
                 "gold": np.array(dec_gold, dtype=np.int64),
                 "labels": np.stack(dec_labels),
                 "labels_raw": dec_labels_raw,
                 "model_correct": np.array(mp_sum_correct, dtype=np.int64),
                 "perception": np.stack(perception),
                 "layers": probe_layers, "offsets": DEC_OFF, "task": "count",
                 "data_root": str(args.data_root), "sample_seed": int(args.sample_seed),
                 "carrier": "per_token_multipass", "n_frames": NF,
                 "text_style": args.text_style, "mode": "multipass",
                 "modality": args.modality, "resize": int(args.resize),
                 "chunk_k": int(args.chunk_k)}
    torch.save(cache_obj, out / "messages_cache.pt")
    perc = float(np.concatenate(perception).mean()) if perception else float("nan")
    report = (f"=== TEXT MULTIPASS CACHE (isolated per-frame forwards) ===\n"
              f"n={len(dec_gold)} NF={NF} layers={probe_layers} offsets={DEC_OFF} "
              f"style={args.text_style}\n"
              f"per-frame perception accuracy (single-frame digit answer vs evidence): {perc:.3f}\n"
              f"multipass-solution accuracy (sum of clipped per-frame answers == gold): "
              f"{float(np.mean(mp_sum_correct)):.3f}\n"
              f"cache -> {out/'messages_cache.pt'}\n")
    (out / "report.txt").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
