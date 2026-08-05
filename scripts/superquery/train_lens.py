#!/usr/bin/env python3
"""Self-distilled ANSWER LENS: trained readout head for mid-forward quantization.

Raw logit lens is behaviorally dead at inner tree nodes (selfq v3: model predicts
<|im_end|>). This trains, per layer, an affine W on the node's readout-position
state so that lm_head(final_norm(W(h))) matches the TEACHER: the frozen model's own
final-layer answer distribution when the same content is posed in its native regime
(pair alone-pass: [frame_a][frame_b][question] + generation prompt).

No gold labels anywhere; codomain = vocabulary (task-agnostic by construction).
Loss arms (Tal's stage-3 gradient lesson): kl (plain distill) vs kl+ce (aux CE on
the TEACHER-ARGMAX token). Eval on held-out samples: pair digit-acc (floor: raw
lens 0.519, ceiling: ridge 0.983) + top1-is-digit + ZERO-SHOT transfer of the
pair-trained lens to frame nodes (verdict 0/1) and lvl2 nodes (0..4).

Usage: python scripts/superquery/train_lens.py --output outputs/superquery/lens_n8
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
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

NF = 8
ROOT = "data/mmred_images_park/seq_len_8/all_uniform"
READ = (16, 20, 24, 27)
TOPK = 256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--aux-ce", type=float, default=0.5)
    ap.add_argument("--train-frac", type=float, default=0.6)
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
    W_U = dequantize_linear_weight(model.lm_head).float()          # (V, H)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    dig_ids = torch.tensor([tok(f"{k}", add_special_tokens=False).input_ids[0]
                            for k in range(NF + 1)], device=dev)
    SUFFIX = " Answer: ("
    suf_ids_l = [tok(s, add_special_tokens=False).input_ids
                 for s in (SUFFIX, " Answer: ", " Answer:")]

    # ---------------- capture: student node states + native-regime teachers
    stu: dict = {}          # (group, L) -> list of (H,) fp32
    tea_pair, tea_frame = [], []   # (topk_ids, topk_probs) per node
    y_frame_all, y1_all, y2_all, gold_all, sid_of_pair = [], [], [], [], []
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
            content.append({"type": "text", "text": SUFFIX})
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

        y_frame = [1 if t in evid else 0 for t in range(NF)]
        y1 = [int(sum(1 for c in g if c in evid)) for g in levels[0]]
        y2 = [int(sum(1 for c in ls if c in evid)) for ls in lsets[1]]
        pos_groups = ([("frame", rep_spans[t][1] - 1) for t in range(NF)] +
                      [("lvl1", span_of[(0, gi)][1] - 1) for gi in range(4)] +
                      [("lvl2", span_of[(1, gi)][1] - 1) for gi in range(2)] +
                      [("root", span_of[(2, 0)][1] - 1)])

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
                    for gname, p in pos_groups:
                        stu.setdefault((gname, li), []).append(
                            h[0, p].float().cpu().numpy())
            # teachers: native-regime answer distributions
            def teacher(content_t):
                it = processor.apply_chat_template([{"role": "user", "content": content_t}],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
                it = move_to_device(it, dev)
                lg = model(**it).logits[0, -1].float()
                pr = torch.softmax(lg, -1)
                tv, ti = torch.topk(pr, TOPK)
                return ti.cpu().numpy(), tv.cpu().numpy()
            for gi in range(4):
                fa, fb = levels[0][gi]
                tea_pair.append(teacher([{"type": "image", "image": frames[fa]},
                                         {"type": "image", "image": frames[fb]},
                                         {"type": "text", "text": q0}]))
                sid_of_pair.append(n_done)
            for t in range(NF):
                tea_frame.append(teacher([{"type": "image", "image": frames[t]},
                                          {"type": "text", "text": q0}]))
        y_frame_all += y_frame
        y1_all += y1
        y2_all += y2
        gold_all.append(gold)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  {n_done}/{args.limit} (skip {n_skip}) {time.time()-t0:.0f}s",
                  flush=True)

    n = n_done
    print(f"capture done: {n} samples {time.time()-t0:.0f}s", flush=True)
    y1_np = np.array(y1_all)
    yf_np = np.array(y_frame_all)
    y2_np = np.array(y2_all)
    # teacher sanity: does the native regime actually answer with the right digit?
    tea_top1 = np.array([ti[0] for ti, _tv in tea_pair])
    dig_np = dig_ids.cpu().numpy()
    tea_acc = float(np.mean([tt == dig_np[y] if y < len(dig_np) else False
                             for tt, y in zip(tea_top1, y1_np)]))
    print(f"[teacher] pair top1==gold-digit: {tea_acc:.3f}", flush=True)

    # ---------------- train per-layer lenses on pair nodes
    H = 3584
    W_U_dev = W_U.to(dev)
    rngsplit = np.random.default_rng(0).permutation(n)
    tr_s = set(rngsplit[: int(n * args.train_frac)].tolist())
    idx_pair_tr = [i for i in range(len(tea_pair)) if sid_of_pair[i] in tr_s]
    idx_pair_ev = [i for i in range(len(tea_pair)) if sid_of_pair[i] not in tr_s]

    rows = []
    for arm, aux in (("kl", 0.0), ("kl+ce", args.aux_ce)):
        for L in READ:
            X = torch.tensor(np.stack(stu[("lvl1", L)]), device=dev)      # (P, H)
            Wl = torch.nn.Linear(H, H, bias=True).to(dev)
            torch.nn.init.eye_(Wl.weight)
            torch.nn.init.zeros_(Wl.bias)
            opt = torch.optim.Adam(Wl.parameters(), lr=1e-4)
            ti_all = torch.tensor(np.stack([tea_pair[i][0] for i in idx_pair_tr]),
                                  device=dev, dtype=torch.long)           # (T, K)
            tp_all = torch.tensor(np.stack([tea_pair[i][1] for i in idx_pair_tr]),
                                  device=dev, dtype=torch.float32)
            tp_all = tp_all / tp_all.sum(-1, keepdim=True)
            Xtr = X[idx_pair_tr]
            for ep in range(args.epochs):
                opt.zero_grad()
                hh = final_norm(Wl(Xtr).to(next(final_norm.parameters()).dtype)).float()
                lg = hh @ W_U_dev.T                                       # (T, V)
                lsm = torch.log_softmax(lg, -1)
                kl = -(tp_all * torch.gather(lsm, 1, ti_all)).sum(-1).mean()
                loss = kl
                if aux > 0:
                    loss = loss + aux * torch.nn.functional.cross_entropy(
                        lg, ti_all[:, 0])
                loss.backward()
                opt.step()
            with torch.no_grad():
                hh = final_norm(Wl(X[idx_pair_ev]).to(
                    next(final_norm.parameters()).dtype)).float()
                lg = hh @ W_U_dev.T
                top1 = lg.argmax(-1)
                cand = lg[:, dig_ids[:3]]
                pred = cand.argmax(-1).cpu().numpy()
                yv = y1_np[idx_pair_ev]
                acc = float((pred == yv).mean())
                t1d = float(np.mean([int(t.item()) in set(dig_np[:3].tolist())
                                     for t in top1]))
                # zero-shot transfer: frames (0/1) and lvl2 (0..4)
                def transfer(gname, y_np, mx):
                    Xt = torch.tensor(np.stack(stu[(gname, L)]), device=dev)
                    per = len(y_np) // n
                    ev_rows = [i for i in range(len(y_np))
                               if (i // per) not in tr_s]
                    hh2 = final_norm(Wl(Xt[ev_rows]).to(
                        next(final_norm.parameters()).dtype)).float()
                    lg2 = hh2 @ W_U_dev.T
                    pr2 = lg2[:, dig_ids[: mx + 1]].argmax(-1).cpu().numpy()
                    return float((pr2 == y_np[ev_rows]).mean())
                fr_acc = transfer("frame", yf_np, 1)
                l2_acc = transfer("lvl2", y2_np, 4)
            rows.append([arm, L, acc, t1d, fr_acc, l2_acc])
            print(f"[lens {arm} L{L}] pair-acc {acc:.3f} top1-is-digit {t1d:.3f} "
                  f"frame-transfer {fr_acc:.3f} lvl2-transfer {l2_acc:.3f}", flush=True)
            torch.save({"W": Wl.state_dict(), "L": L, "arm": arm},
                       out / f"lens_{arm.replace('+','_')}_L{L}.pt")

    with open(out / "lens.csv", "w", newline="") as f:
        csv.writer(f).writerows([["arm", "L", "pair_acc", "top1_digit_frac",
                                  "frame_transfer", "lvl2_transfer"], *rows,
                                 ["teacher_pair_top1", tea_acc, "", "", "", ""]])
    np.savez_compressed(out / "lens_data.npz",
                        **{f"stu|{g}|{L}": np.stack(v) for (g, L), v in stu.items()},
                        y_frame=yf_np, y1=y1_np, y2=y2_np, gold=np.array(gold_all),
                        tea_pair_ids=np.stack([t[0] for t in tea_pair]),
                        tea_pair_probs=np.stack([t[1] for t in tea_pair]),
                        tea_frame_ids=np.stack([t[0] for t in tea_frame]),
                        tea_frame_probs=np.stack([t[1] for t in tea_frame]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
