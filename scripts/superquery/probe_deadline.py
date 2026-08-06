#!/usr/bin/env python3
"""EMISSION DEADLINE: at what layer does the model stop accepting new inputs
into its output plan — and do DEPTH-MATCHED codes extend the deadline?

Layout: fenced b=2 tree (N=8) + clean tail " Counts: 0 and 0. What is the total
count? Answer: " (the 1.000-measured pass-2 phrasing — v4c's 'Answer: (' force was
a confound: it echoes operands). GOLD lvl2 half-counts are written into the two
register positions at layer L in {0,4,8,12,16,20,24}; EM on the emitted total.

Code variants:
  emb   raw digit embedding, norm-matched (all prior experiments)
  depth real digit-token states harvested at layer L from text calibration prompts
        (has the 'processed token' features the output circuits may key on)
L=0/emb is equivalent to real input tokens (ceiling ~ pass-2 conditional 1.0).

Usage: python scripts/superquery/probe_deadline.py --output outputs/superquery/deadline_n8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/superquery"))

from probe_tree import leaf_sets, tree_levels  # noqa: E402

from gnnformer.constants import MASK_MIN, ROOMS  # noqa: E402
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence  # noqa: E402
from gnnformer.fencing import (  # noqa: E402
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    reset_positions,
)
from gnnformer.runtime import (  # noqa: E402
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NF = 8
ROOT = "data/mmred_images_park/seq_len_8/all_uniform"
SUFFIX = " Answer: ("
TAIL = " Counts: 0 and 0. What is the total count? Answer: "
WRITE_LS = (0, 4, 8, 12, 16, 20, 24)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    levels = tree_levels(NF, 2)
    lsets = leaf_sets(levels)
    n_nodes = sum(len(g) for g in levels)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    final_norm = text_model.norm
    dev = model.device
    W_U = dequantize_linear_weight(model.lm_head).float().to(dev)
    W_emb = text_model.embed_tokens.weight
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dig = {k: tok(f"{k}", add_special_tokens=False).input_ids[0] for k in range(NF + 1)}
    dig_ids = torch.tensor([dig[k] for k in range(NF + 1)], device=dev)
    dig_set = set(dig_ids.tolist())
    suf_ids_l = [tok(s, add_special_tokens=False).input_ids
                 for s in (SUFFIX, " Answer: ", " Answer:")]

    # ---------- harvest depth-matched digit codes: state of digit token k at layer L
    # from generic text contexts (5 partner values each)
    depth_code = {}   # (k, L) -> vector (fp32 cpu)
    with torch.no_grad():
        for k in range(5):
            acc = {L: [] for L in WRITE_LS if L > 0}
            for j in range(5):
                q = f"Two partial counts are {k} and {j}. What is the total count?"
                it = processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": q}]}],
                    add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt")
                it = move_to_device(it, dev)
                ids_t = it["input_ids"][0].tolist()
                p_k = ids_t.index(dig[k])
                hs = model(**it, output_hidden_states=True).hidden_states
                for L in acc:
                    acc[L].append(hs[L][0, p_k].float().cpu())
            for L in acc:
                depth_code[(k, L)] = torch.stack(acc[L]).mean(0)
    print("harvested depth-matched codes", flush=True)

    def prep(sd):
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence("steps", q0, states, gold, ROOMS)
            if pe_ is None:
                return None
            evid, _room = pe_
        except Exception:
            return None
        if (not evid and gold != 0) or len(frames) != NF:
            return None
        frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        for _ in range(n_nodes):
            content.append({"type": "text", "text": q0})
            content.append({"type": "text", "text": SUFFIX})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, dev)
        tail_ids = tok(TAIL, add_special_tokens=False).input_ids
        force = torch.tensor([tail_ids], device=dev)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], force], 1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(force)], 1)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            return None
        rep_spans = spans[1 : NF + 1]
        sq_raw = spans[NF + 1 :]
        sq_spans, ok = [], True
        for a, b in sq_raw:
            ext = None
            for si_ in suf_ids_l:
                if ids[b : b + len(si_)] == si_:
                    ext = b + len(si_)
                    break
            if ext is None:
                ok = False
                break
            sq_spans.append((a, ext))
        if not ok:
            return None
        tail_pos0 = seq - len(tail_ids)
        regs = [tail_pos0 + i for i, t in enumerate(tail_ids) if t == dig[0]]
        if len(regs) != 2:
            return None
        fin_start = sq_spans[0][0]
        blocks = frame_blocks(vstarts, fin_start)
        vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
        m = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                     fence_blocks=True, blocks=blocks)
        prefix_cols = torch.arange(0, blocks[0][0])
        si = 0
        span_of = {}
        for li, groups in enumerate(levels):
            for gi, g in enumerate(groups):
                a, b = sq_spans[si]
                span_of[(li, gi)] = (a, b)
                si += 1
                child_spans = ([rep_spans[c] for c in g] if li == 0
                               else [span_of[(li - 1, c)] for c in g])
                rws = torch.arange(a, b)
                m[rws] = MASK_MIN
                cols = torch.cat([prefix_cols] +
                                 [torch.arange(ca, cb) for ca, cb in child_spans])
                m[rws.unsqueeze(1), cols.unsqueeze(0)] = 0.0
                blk = torch.zeros(b - a, b - a)
                blk.masked_fill_(torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1),
                                 MASK_MIN)
                m[a:b, a:b] = blk
        ts = sq_spans[-1][1]
        rows = torch.arange(ts, seq)
        m[rows] = MASK_MIN
        m[rows.unsqueeze(1), prefix_cols.unsqueeze(0)] = 0.0
        blk = torch.zeros(seq - ts, seq - ts)
        blk.masked_fill_(torch.triu(torch.ones(seq - ts, seq - ts, dtype=torch.bool), 1),
                         MASK_MIN)
        m[ts:, ts:] = blk
        k2 = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]
        return dict(inputs=inputs, m=m, seq=seq, blocks=blocks,
                    fin_start=fin_start, regs=regs, k2=k2, gold=gold)

    def run_emit(rec, write_L, variant):
        inputs = rec["inputs"]
        base_pos, _ = rope_fn(inputs["input_ids"],
                              image_grid_thw=inputs.get("image_grid_thw"),
                              attention_mask=inputs.get("attention_mask"))
        pos = reset_positions(base_pos, rec["blocks"], rec["fin_start"]).clone().to(dev)
        emb = text_model.embed_tokens(inputs["input_ids"]).clone()
        img = model.model.get_image_features(inputs["pixel_values"],
                                             inputs["image_grid_thw"])
        img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
        im_mask = inputs["input_ids"][0] == model.config.image_token_id
        emb[0, im_mask] = img.to(emb.dtype)
        cos_, sin_ = text_model.rotary_emb(emb, pos)
        pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
        m4 = rec["m"].to(dev).to(emb.dtype).view(1, 1, rec["seq"], rec["seq"])

        def write(h, L):
            for p, k in zip(rec["regs"], rec["k2"]):
                if variant == "emb" or L == 0:
                    v = W_emb[dig[int(k)]].to(h.dtype)
                    scale = h[0, p].norm() / (v.norm() + 1e-6)
                    h[0, p] = v * scale
                else:
                    h[0, p] = depth_code[(int(k), L)].to(dev).to(h.dtype)
            return h

        h = emb
        if write_L == 0:
            h = write(h, 0)
        for li in range(28):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            if li == write_L and write_L > 0:
                h = write(h, write_L)
        lg = (final_norm(h[0, -1:]).float() @ W_U.T)[0]
        return int(lg.argmax().item())

    rows_out = []
    stats = {(L, v): [0, 0, Counter()] for L in WRITE_LS for v in ("emb", "depth")}
    n_done = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        rec = prep(sd)
        if rec is None:
            continue
        with torch.no_grad():
            for L in WRITE_LS:
                for v in ("emb", "depth"):
                    if L == 0 and v == "depth":
                        continue
                    t = run_emit(rec, L, v)
                    s = stats[(L, v)]
                    s[0] += int(t == dig[rec["gold"]])
                    s[1] += 1
                    s[2][tok.decode([t])] += 1
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{args.limit} {time.time()-t0:.0f}s", flush=True)

    for L in WRITE_LS:
        for v in ("emb", "depth"):
            c, n, ctr = stats[(L, v)]
            if n == 0:
                continue
            tops = "; ".join(f"{w!r}:{cc}" for w, cc in ctr.most_common(4))
            rows_out.append([L, v, c / n, n])
            print(f"[deadline L{L} {v}] EM {c/n:.3f} (n={n}) tops: {tops}", flush=True)
    with open(out / "deadline.csv", "w", newline="") as f:
        csv.writer(f).writerows([["write_L", "variant", "em", "n"], *rows_out])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
