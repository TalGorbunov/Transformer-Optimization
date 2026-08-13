#!/usr/bin/env python3
"""DEADLINE v2 — same-sequence patch ablation (Tal's control, 2026-08-06).

The deadline-v1 'depth-matched' codes came from OTHER prompts; late-layer states
are context-mixed, so late failure could be HARVEST MISMATCH, not a closed mouth.
Control: two runs of the SAME sequence differing only in register token identity.

  run A (donor): tail = " Counts: {a} and {b}. What is the total count? Answer: "
                 with the REAL gold digits as input tokens. Capture register states
                 at L in {4,8,12,16,20,24}; record run-A emission (= ceiling).
  run B (patched): tail digits are "0 and 0"; at layer L splice in run-A's states.
                 arm 'self'     : this sample's own donor states (exact control)
                 arm 'template' : per-(digit,L) donor states AVERAGED over the other
                                  samples (the DEPLOYABLE code — leave-one-out)
If self@L20 is high -> no deadline (v1 curve = mismatch artifact); template arm
then previews a single-forward deployment. If self@L20 fails -> deadline stands.

Usage: python scripts/superquery/probe_deadline2.py --output outputs/superquery/deadline2_n8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
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
WRITE_LS = (4, 8, 12, 16, 20, 24)


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
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dig = {k: tok(f"{k}", add_special_tokens=False).input_ids[0] for k in range(NF + 1)}
    dig_ids = torch.tensor([dig[k] for k in range(NF + 1)], device=dev)
    suf_ids_l = [tok(s, add_special_tokens=False).input_ids
                 for s in (SUFFIX, " Answer: ", " Answer:")]

    def prep(sd, a, b):
        """Build the sequence with tail digits (a, b). Returns rec or None."""
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
        tail = f" Counts: {a} and {b}. What is the total count? Answer: "
        tail_ids = tok(tail, add_special_tokens=False).input_ids
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
        for aa, bb in sq_raw:
            ext = None
            for si_ in suf_ids_l:
                if ids[bb : bb + len(si_)] == si_:
                    ext = bb + len(si_)
                    break
            if ext is None:
                ok = False
                break
            sq_spans.append((aa, ext))
        if not ok:
            return None
        tail_pos0 = seq - len(tail_ids)
        regs = [tail_pos0 + i for i, t in enumerate(tail_ids)
                if t in (dig[a], dig[b]) and i < len(tail_ids) - 8]
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
                aa, bb = sq_spans[si]
                span_of[(li, gi)] = (aa, bb)
                si += 1
                child_spans = ([rep_spans[c] for c in g] if li == 0
                               else [span_of[(li - 1, c)] for c in g])
                rws = torch.arange(aa, bb)
                m[rws] = MASK_MIN
                cols = torch.cat([prefix_cols] +
                                 [torch.arange(ca, cb) for ca, cb in child_spans])
                m[rws.unsqueeze(1), cols.unsqueeze(0)] = 0.0
                blk = torch.zeros(bb - aa, bb - aa)
                blk.masked_fill_(torch.triu(torch.ones(bb - aa, bb - aa,
                                                       dtype=torch.bool), 1), MASK_MIN)
                m[aa:bb, aa:bb] = blk
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

    def run(rec, capture_ls=(), patch=None, patch_L=None):
        """Forward; capture register states at capture_ls; or patch states at patch_L.
        Returns (emitted_token, captures{L: (2,H) fp32})."""
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
        h = emb
        caps = {}
        for li in range(28):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            if li in capture_ls:
                caps[li] = h[0, rec["regs"]].float().cpu()
            if patch is not None and li == patch_L:
                for j, p in enumerate(rec["regs"]):
                    h[0, p] = patch[j].to(dev).to(h.dtype)
        lg = (final_norm(h[0, -1:]).float() @ W_U.T)[0]
        return int(lg.argmax().item()), caps

    # ---------------- phase 1: donors (run A) for all samples
    t0 = time.time()
    samples = []
    em_A = 0
    donor = {}          # sid -> {L: (2,H)}
    with torch.no_grad():
        n_done = 0
        for sd in iter_sample_dirs_shuffled(Path(ROOT), args.shuffle_dirs):
            if n_done >= args.limit:
                break
            recA = prep(sd, 0, 0)   # probe golds first via prep(0,0) to get k2
            if recA is None:
                continue
            a, b = recA["k2"]
            recA = prep(sd, a, b)
            if recA is None or len(recA["regs"]) != 2:
                continue
            t, caps = run(recA, capture_ls=WRITE_LS)
            em_A += int(t == dig[recA["gold"]])
            donor[str(sd)] = caps
            samples.append((sd, a, b, recA["gold"]))
            n_done += 1
            if n_done % 10 == 0:
                print(f"  donors {n_done} {time.time()-t0:.0f}s", flush=True)
    nA = len(samples)
    print(f"[run-A ceiling] EM {em_A/nA:.3f} (n={nA})", flush=True)
    # per-(digit, L) leave-one-out template sums
    tmpl_sum = defaultdict(lambda: torch.zeros(3584))
    tmpl_cnt = defaultdict(int)
    for sid, a, b, _g in [(str(s), a, b, g) for s, a, b, g in samples]:
        for L in WRITE_LS:
            for j, k in enumerate((a, b)):
                tmpl_sum[(k, L)] += donor[sid][L][j]
                tmpl_cnt[(k, L)] += 1

    # ---------------- phase 2: patched runs (run B)
    stats = {(L, v): [0, 0] for L in WRITE_LS for v in ("self", "template")}
    with torch.no_grad():
        for sd, a, b, gold in samples:
            recB = prep(sd, 0, 0)
            if recB is None:
                continue
            sid = str(sd)
            for L in WRITE_LS:
                # self donor states
                t, _ = run(recB, patch=donor[sid][L], patch_L=L)
                stats[(L, "self")][0] += int(t == dig[gold])
                stats[(L, "self")][1] += 1
                # leave-one-out template states
                pt = []
                okt = True
                for j, k in enumerate((a, b)):
                    if tmpl_cnt[(k, L)] <= 1:
                        okt = False
                        break
                    pt.append((tmpl_sum[(k, L)] - donor[sid][L][j])
                              / (tmpl_cnt[(k, L)] - 1))
                if okt:
                    t, _ = run(recB, patch=pt, patch_L=L)
                    stats[(L, "template")][0] += int(t == dig[gold])
                    stats[(L, "template")][1] += 1

    rows = [["runA_ceiling", "", em_A / nA, nA]]
    for L in WRITE_LS:
        for v in ("self", "template"):
            c, n = stats[(L, v)]
            if n:
                rows.append([L, v, c / n, n])
                print(f"[deadline2 L{L} {v}] EM {c/n:.3f} (n={n})", flush=True)
    with open(out / "deadline2.csv", "w", newline="") as f:
        csv.writer(f).writerows([["write_L", "variant", "em", "n"], *rows])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
