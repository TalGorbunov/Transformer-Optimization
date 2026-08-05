#!/usr/bin/env python3
"""Self-quantization probe: can the model's OWN head read its aggregator nodes?

N=8, b=2 tree, blocks fenced at all layers. Every SQ replica is suffixed with
" Answer:" (a readout cue) — its extended-span last position is where the model
would answer if generating. At layers {12,16,20,24,27} the last-position states of
(a) in-block frame replicas (plain, verdict 0/1), (b) lvl1 SQs (pair count 0..2),
(c) lvl2 SQs (0..4), (d) root (0..8) are pushed through the model's FINAL NORM +
LM HEAD (logit lens). Reported per (level, layer): restricted-digit argmax accuracy
vs gold + fraction of unrestricted top-1 that is any digit. Zero trained parameters.

If lvl1 digit-ranking is high at some layer L*, repeater v3 = model-native (probe-free)
quantization at L* becomes buildable.

Usage: python scripts/superquery/probe_selfq.py --output outputs/superquery/selfq_n8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NF = 8
ROOT = "data/mmred_images_park/seq_len_8/all_uniform"
READ = (12, 16, 20, 24, 27)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=120)
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
    lm_head = model.lm_head
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dig = {k: tok(f"{k}", add_special_tokens=False).input_ids[0] for k in range(NF + 1)}
    dig_ids = torch.tensor([dig[k] for k in range(NF + 1)], device=dev)
    WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    word_ids = torch.tensor([tok(w, add_special_tokens=False).input_ids[0]
                             for w in WORDS], device=dev)
    # trailing space: the readout position's NATURAL next token is then a bare digit
    SUFFIXES = [" Answer: ", " Answer:", "Answer: ", "\nAnswer: "]
    suf_ids = [tok(s, add_special_tokens=False).input_ids for s in SUFFIXES]

    # hits[(group, L)] = [n_correct, n_total]; digit_frac same shape
    from collections import Counter, defaultdict
    hits = defaultdict(lambda: [0, 0])
    hits_w = defaultdict(lambda: [0, 0])      # digit OR count-word candidates
    dfrac = defaultdict(lambda: [0, 0])
    top1_ctr = defaultdict(Counter)           # what the model actually wanted to say
    n_done = n_skip = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence("steps", q0, states, gold, ROOMS)
            if pe_ is None:
                n_skip += 1
                continue
            evid, _room = pe_
        except Exception:
            n_skip += 1
            continue
        if (not evid and gold != 0) or len(frames) != NF:
            n_skip += 1
            continue
        frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        for _ in range(n_nodes):
            content.append({"type": "text", "text": q0})
            content.append({"type": "text", "text": " Answer: "})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, dev)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            n_skip += 1
            continue
        rep_spans = spans[1 : NF + 1]
        sq_spans_raw = spans[NF + 1 :]
        # extend each SQ span over its " Answer:" suffix
        sq_spans = []
        ok = True
        for a, b in sq_spans_raw:
            ext = None
            for si_ in suf_ids:
                if ids[b : b + len(si_)] == si_:
                    ext = b + len(si_)
                    break
            if ext is None:
                ok = False
                break
            sq_spans.append((a, ext))
        if not ok:
            n_skip += 1
            continue
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

        # golds per group
        y_frame = [1 if t in evid else 0 for t in range(NF)]
        y1 = [int(sum(1 for c in g if c in evid)) for g in levels[0]]
        y2 = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]
        groups = ([("frame", rep_spans[t][1] - 1, y_frame[t], 1) for t in range(NF)] +
                  [("lvl1", span_of[(0, gi)][1] - 1, y1[gi], 2)
                   for gi in range(len(levels[0]))] +
                  [("lvl2", span_of[(1, gi)][1] - 1, y2[gi], 4)
                   for gi in range(len(levels[1]))] +
                  [("root", span_of[(2, 0)][1] - 1, gold, 8)])

        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"],
                                  image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos = reset_positions(base_pos, blocks, fin_start).clone().to(dev)
            emb = text_model.embed_tokens(inputs["input_ids"]).clone()
            img = model.model.get_image_features(inputs["pixel_values"],
                                                 inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb[0, im_mask] = img.to(emb.dtype)
            cos_, sin_ = text_model.rotary_emb(emb, pos)
            pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
            m4 = m.to(dev).to(emb.dtype).view(1, 1, seq, seq)
            h = emb
            for li in range(max(READ) + 1):
                h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
                if li in READ:
                    pos_idx = torch.tensor([p for _g, p, _y, _mx in groups], device=dev)
                    lg = lm_head(final_norm(h[0, pos_idx]))  # (n_groups, vocab)
                    top1 = lg.argmax(-1)
                    for gi2, (gname, _p, y, mx) in enumerate(groups):
                        cand = dig_ids[: mx + 1]
                        pred = int(torch.argmax(lg[gi2, cand]).item())
                        hits[(gname, li)][0] += int(pred == y)
                        hits[(gname, li)][1] += 1
                        # digit-or-word: value k scored by max(logit_digit, logit_word)
                        sc = torch.maximum(lg[gi2, cand], lg[gi2, word_ids[: mx + 1]])
                        hits_w[(gname, li)][0] += int(int(torch.argmax(sc).item()) == y)
                        hits_w[(gname, li)][1] += 1
                        dfrac[(gname, li)][0] += int(top1[gi2].item() in
                                                     set(dig_ids.tolist()))
                        dfrac[(gname, li)][1] += 1
                        top1_ctr[(gname, li)][tok.decode([int(top1[gi2])])] += 1
        n_done += 1
        if n_done % 20 == 0:
            print(f"  {n_done}/{args.limit} (skip {n_skip}) {time.time()-t0:.0f}s",
                  flush=True)

    rows = []
    for gname in ("frame", "lvl1", "lvl2", "root"):
        for L in READ:
            c, t = hits[(gname, L)]
            cw, _tw = hits_w[(gname, L)]
            dc, dt = dfrac[(gname, L)]
            tops = "; ".join(f"{w!r}:{n2}" for w, n2 in
                             top1_ctr[(gname, L)].most_common(5))
            rows.append([gname, L, c / max(t, 1), cw / max(t, 1), dc / max(dt, 1),
                         t, tops])
            print(f"[selfq {gname} L{L}] digit-acc {c/max(t,1):.3f} "
                  f"+words {cw/max(t,1):.3f} top1-is-digit {dc/max(dt,1):.3f} "
                  f"(n={t}) top1s: {tops}", flush=True)
    with open(out / "selfq.csv", "w", newline="") as f:
        csv.writer(f).writerows([["group", "L", "digit_acc", "digitword_acc",
                                  "top1_digit_frac", "n", "top1_tokens"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
