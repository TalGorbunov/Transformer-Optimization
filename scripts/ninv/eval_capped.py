#!/usr/bin/env python3
"""Capped-tree (flat-top) eval — depth extrapolation WITHOUT training (ninv).

The five-arm result (2026-08-11): trained levels are perfect and length-invariant,
but never-trained levels (>=5) are dead in every arm, so N>=32 EMITTED collapses.
This evaluates the zero-training alternative: DON'T build deep trees. Cap the tree
at the trained depth (4 => 16-leaf subtrees); an N=64 sample becomes FOUR depth-4
subtrees whose roots all speak digits (v2's node_tok anchoring). Two finishes:

  DIRECT   the answer tail's mask opens to ALL subtree roots (was: the single
           root); the readout must sum in-context. OOD for a readout trained on
           one root — measured, not assumed.
  TWO-PASS read each subtree root's emitted digit (lm_head argmax at its span
           tail — v2 trains exactly this emission, node_tok CE 0.02), then a
           text-only second pass: "Partial counts are a, b, c, d. What is the
           total count?" The pass-2 adder measured cond-EM 1.000 (twopass runs).

Also reports: per-root digit accuracy vs true subtree counts (the pass-1 fidelity
bound), and top-node probe decode using the heads' semantics (a 16-leaf subtree
root IS a lv4 node — in-range for everything trained).

Usage:
  python scripts/ninv/eval_capped.py --ckpt <v2 registers_best.pt> \
      --ns 32,64 --cap 4 --output outputs/ninv/<ts>_capped
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import evidence_bits, iter_hf_sample_dirs, load_hf_sample  # noqa: E402
from train_registers import attach_masked_lora, canonical_positions  # noqa: E402

from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.fencing import (  # noqa: E402
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
)
from gnnformer.runtime import (  # noqa: E402
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

ROOTS = {32: "data/mmred_hf/dirs/seq_len_32_test",
         64: "data/mmred_hf/dirs/seq_len_64_test",
         128: "data/mmred_hf/dirs/seq_len_128_test"}


def tree_levels_capped(n_leaves: int, cap: int):
    """b=2 levels, stopping after `cap` levels; the last level may hold several
    subtree roots (n_leaves / 2^cap of them)."""
    levels, cur = [], n_leaves
    while cur > 1 and len(levels) < cap:
        levels.append([list(range(i, min(i + 2, cur))) for i in range(0, cur, 2)])
        cur = len(levels[-1])
    return levels


def level_counts(bits, levels):
    out, prev = [], list(bits)
    for groups in levels:
        cur = [sum(prev[c] for c in g) for g in groups]
        out.append(cur)
        prev = cur
    return out


def build_mask_multi(seq, e, rep_spans, vis, blocks, sq_spans, arm_levels,
                     top_spans):
    """train_registers.build_mask with MULTIPLE top spans opened to the tail."""
    m_core = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                      fence_blocks=True, blocks=blocks)
    m = torch.full((seq + e, seq + e), MASK_MIN)
    m[:seq, :seq] = m_core
    prefix_cols = torch.arange(0, blocks[0][0])
    si = 0
    span_of = {}
    for li, groups in enumerate(arm_levels):
        for gi, g in enumerate(groups):
            a, b = sq_spans[si]
            span_of[(li, gi)] = (a, b)
            si += 1
            child_spans = ([rep_spans[c] for c in g] if li == 0
                           else [span_of[(li - 1, c)] for c in g])
            rows = torch.arange(a, b)
            m[rows] = MASK_MIN
            cols = torch.cat([prefix_cols]
                             + [torch.arange(ca, cb) for ca, cb in child_spans])
            m[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
            blk = torch.zeros(b - a, b - a)
            blk.masked_fill_(torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1),
                             MASK_MIN)
            m[a:b, a:b] = blk
    t0 = sq_spans[-1][1]
    tail_cols = torch.cat([prefix_cols]
                          + [torch.arange(ra, rb) for ra, rb in top_spans])
    for r in range(t0, seq + e):
        m[r] = MASK_MIN
        m[r, tail_cols] = 0.0
        m[r, t0 : r + 1] = 0.0
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ns", default="32,64")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-decode", type=int, default=4)
    ap.add_argument("--root-read", choices=("emit", "probe"), default="probe",
                    help="how to read each subtree root's count. 'emit' = lm_head "
                         "argmax at the span tail (degrades at long context: "
                         "0.40-0.45 measured, though the states probe at 0.99). "
                         "'probe' = ridge head fit on IN-LENGTH N=16 whole-tree "
                         "roots (cross-N transfer proven at 0.99) — the same "
                         "architecture as the frozen-twopass headline")
    ap.add_argument("--fit-n16", type=int, default=100,
                    help="in-length N=16 train samples for the probe head")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    print(f"[ckpt] {args.ckpt} arm={ck['leaf_input']} rows={ck.get('lora_rows')} "
          f"anchor={ck.get('token_anchor')} val_em={ck.get('val_em')}", flush=True)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    n_layers = len(layers)
    eos = tok.eos_token_id

    registers = ck["registers"].to(dev)
    row_holder: dict = {"rows": None}
    lora_mid = attach_masked_lora(layers[: ck["late_open"]], ck["mid_open"],
                                  rank=ck["rank"], alpha=ck["alpha"], device=dev,
                                  holder=row_holder, state=ck["lora_mid"])
    lora_late = attach_masked_lora(layers, ck["late_open"], rank=ck["rank"],
                                   alpha=ck["alpha"], device=dev,
                                   holder=row_holder, state=ck["lora_late"])
    lora_rows = ck.get("lora_rows", "all")

    def prep(sd, cap):
        try:
            _sid, frames, q0, states, a0 = load_hf_sample(sd, resize=args.resize)
            gold = int(str(a0).strip())
            bits = evidence_bits(q0, states)
            if bits is None or sum(bits) != gold:
                return None
        except Exception:
            return None
        NF = len(frames)
        levels = tree_levels_capped(NF, cap)
        n_nodes = sum(len(g) for g in levels)
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content += [{"type": "image", "image": f}, {"type": "text", "text": q0}]
        content += [{"type": "text", "text": q0}] * n_nodes
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        fg = image_token_groups(inputs["input_ids"][0], expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            return None
        rep = spans[1 : NF + 1]
        sq = spans[NF + 1 :]
        blocks = frame_blocks(vstarts, sq[0][0])
        n_top = len(levels[-1])
        top_spans = sq[-n_top:]
        with torch.no_grad():
            mv = move_to_device(inputs, dev)
            base_pos, _ = rope_fn(mv["input_ids"],
                                  image_grid_thw=mv.get("image_grid_thw"),
                                  attention_mask=mv.get("attention_mask"))
            img = model.model.get_image_features(mv["pixel_values"],
                                                 mv["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
        return dict(ids=inputs["input_ids"][0], img=img.to(torch.float16).cpu(),
                    base_pos=base_pos.cpu(), seq=len(ids), NF=NF, gold=gold,
                    bits=bits, levels=levels, lc=level_counts(bits, levels),
                    rep=rep, sq=sq, blocks=blocks, top_spans=top_spans,
                    vis=[torch.tensor(sorted(int(p) for p in g)) for g in fg])

    @torch.no_grad()
    def forward(d, extra_ids=()):
        seq, e = d["seq"], len(extra_ids)
        m = build_mask_multi(seq, e, d["rep"], d["vis"], d["blocks"], d["sq"],
                             d["levels"], d["top_spans"])
        # canonical_positions takes one "root" span for the tail start — use the
        # LAST top span; all node spans share canonical positions anyway
        pos, _, _ = canonical_positions(d["base_pos"], d["blocks"], d["sq"],
                                        d["sq"][-1], seq, e)
        ids = d["ids"].to(dev)
        emb = text_model.embed_tokens(ids.unsqueeze(0)).clone()
        emb[0, ids == model.config.image_token_id] = d["img"].to(dev).to(emb.dtype)
        for (a, b) in d["sq"]:
            emb[0, a:b] = registers[: b - a].to(emb.dtype)
        if e:
            et = torch.tensor(list(extra_ids), device=dev)
            emb = torch.cat([emb, text_model.embed_tokens(et.unsqueeze(0))], dim=1)
        if lora_rows != "all":
            rows = torch.zeros(seq + e)
            for (a, b) in d["sq"]:
                rows[a:b] = 1.0
            rows[d["sq"][-1][1]:] = 1.0
            if lora_rows == "leaves_nodes_tail":
                for (a, b) in d["rep"]:
                    rows[a:b] = 1.0
            row_holder["rows"] = rows.to(dev)
        pos = pos.to(dev)
        cos, sin = text_model.rotary_emb(emb, pos)
        pe = (cos.to(emb.dtype), sin.to(emb.dtype))
        m4 = m.to(dev).to(emb.dtype).view(1, 1, seq + e, seq + e)
        h = emb
        for li in range(n_layers):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
        hf = text_model.norm(h)
        lg_last = model.lm_head(hf[0, -1:].to(model.lm_head.weight.dtype)).float()[0]
        # per top span: BOTH readouts — emitted digits (lm_head argmax at the
        # tail) and the raw last-token state (for the probe head)
        root_digits, root_states = [], []
        for (a, b) in d["top_spans"]:
            lg2 = model.lm_head(hf[0, b - 2 : b].to(
                model.lm_head.weight.dtype)).float()
            two = tok.decode(lg2.argmax(-1).tolist()).strip()
            tail_txt = "".join(c for c in two if c.isdigit())
            root_digits.append(int(tail_txt) if tail_txt else -1)
            root_states.append(hf[0, b - 1].float().cpu().numpy())
        return lg_last, root_digits, root_states

    def emit_direct(d):
        toks = []
        for _ in range(args.max_decode):
            lg, _, _ = forward(d, extra_ids=toks)
            t = int(lg.argmax())
            if t == eos:
                break
            toks.append(t)
        s = tok.decode(toks).strip()
        return int(s) if s.isdigit() else -1

    @torch.no_grad()
    def add2(a, b):
        """The VALIDATED v4d 2-operand template (addsanity 1.000); my first-run
        comma-list template made the model CONCATENATE (emitted 33 for 3,3)."""
        q = f"Two partial counts are {a} and {b}. What is the total count?"
        it = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": q}]}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        it = move_to_device(it, dev)
        f2 = torch.tensor([tok("Answer: ", add_special_tokens=False).input_ids],
                          device=dev)
        it["input_ids"] = torch.cat([it["input_ids"], f2], 1)
        it["attention_mask"] = torch.cat([it["attention_mask"],
                                          torch.ones_like(f2)], 1)
        outp = ""
        for _ in range(args.max_decode):
            # pass 2 runs PURE FROZEN (like the frozen two-pass): zero the LoRA
            # delta by setting an all-zero row mask sized to the CURRENT prompt
            # (the holder otherwise still carries the tree forward's stale mask,
            # which both crashes on length and would wrongly apply node-LoRA here)
            row_holder["rows"] = torch.zeros(it["input_ids"].shape[1], device=dev)
            h = model(**it, output_hidden_states=True).hidden_states[-1][0, -1]
            lg = model.lm_head(text_model.norm(h.unsqueeze(0)).to(
                model.lm_head.weight.dtype)).float()[0]
            t = int(lg.argmax())
            piece = tok.decode([t])
            if t == eos or not piece.strip().isdigit():
                if outp:
                    break
                if t == eos:
                    break
            if piece.strip().isdigit():
                outp += piece.strip()
            nt = torch.tensor([[t]], device=dev)
            it["input_ids"] = torch.cat([it["input_ids"], nt], 1)
            it["attention_mask"] = torch.cat([it["attention_mask"],
                                              torch.ones_like(nt)], 1)
        return int(outp) if outp else -1

    def pass2(parts):
        """Pairwise reduction composing ONLY the validated 2-operand adder."""
        vals = list(parts)
        while len(vals) > 1:
            nxt = []
            for i in range(0, len(vals) - 1, 2):
                s = add2(vals[i], vals[i + 1])
                if s < 0:
                    return -1
                nxt.append(s)
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    # ---- probe head for root reading: fit on IN-LENGTH N=16 whole trees (their
    # root IS a 16-leaf depth-4 node — identical semantics to a capped subtree
    # root; cross-N state transfer measured at 0.99)
    root_head = None
    if args.root_read == "probe":
        sys.path.insert(0, str(_REPO / "scripts/ninv"))
        from transfer_matrix import fit_head
        t0 = time.time()
        fs, ys = [], []
        for sd in iter_hf_sample_dirs(
                Path("data/mmred_hf/dirs/seq_len_16_train_steps_in_room")):
            if len(ys) >= args.fit_n16:
                break
            d = prep(sd, args.cap)
            if d is None or len(d["top_spans"]) != 1:
                continue
            _, _, st = forward(d)
            fs.append(st[0])
            ys.append(d["gold"])
        X3 = np.array(fs)[:, None, :]
        y2 = np.array(ys)[:, None]
        root_head = fit_head(X3, y2)
        tr_acc = float((root_head(X3).reshape(-1) == np.array(ys)).mean())
        print(f"[root-head] fit on {len(ys)} in-length N=16 roots "
              f"(train acc {tr_acc:.3f}, {time.time()-t0:.0f}s)", flush=True)

    lines = [f"ckpt={args.ckpt} cap={args.cap} root_read={args.root_read}"]
    for n in [int(x) for x in args.ns.replace(",", " ").split()]:
        pool = iter_hf_sample_dirs(Path(ROOTS[n]))[: args.limit]
        t0 = time.time()
        em_d = em_2 = n_done = 0
        root_ok = root_n = 0
        g_seen, pd, p2v = [], [], []
        for sd in pool:
            d = prep(sd, args.cap)
            if d is None:
                continue
            _, rdig_emit, rstates = forward(d)
            if root_head is not None:
                rdig = [int(root_head(np.array(s)[None, None, :])[0, 0])
                        for s in rstates]
            else:
                rdig = rdig_emit
            true_tops = d["lc"][-1]
            root_ok += sum(int(a == b) for a, b in zip(rdig, true_tops))
            root_n += len(true_tops)
            v_direct = emit_direct(d)
            v_two = pass2(rdig) if all(r >= 0 for r in rdig) else -1
            em_d += int(v_direct == d["gold"])
            em_2 += int(v_two == d["gold"])
            g_seen.append(d["gold"])
            pd.append(v_direct)
            p2v.append(v_two)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  N={n}: {n_done} ({time.time()-t0:.0f}s)", flush=True)
        g = np.array(g_seen)
        le = g <= 16
        pda, p2a = np.array(pd), np.array(p2v)
        maj = float((g == 0).mean())
        print(f"[capped N={n} cap={args.cap}] n={n_done} "
              f"subtree-root digit acc {root_ok/max(root_n,1):.3f} "
              f"({root_ok}/{root_n})", flush=True)
        print(f"  DIRECT  EM {em_d/max(n_done,1):.3f}  EM(GT<=16) "
              f"{float((pda[le]==g[le]).mean()):.3f}  majority {maj:.3f}", flush=True)
        print(f"  TWOPASS EM {em_2/max(n_done,1):.3f}  EM(GT<=16) "
              f"{float((p2a[le]==g[le]).mean()):.3f}  "
              f"MAE(GT<=16) {float(np.abs(np.where(p2a[le]<0,99,p2a[le])-g[le]).mean()):.2f}",
              flush=True)
        print(f"  emitted(two-pass): {Counter(p2a.tolist()).most_common(8)}", flush=True)
        lines += [f"N={n} rootacc {root_ok/max(root_n,1):.3f} "
                  f"direct {em_d/max(n_done,1):.3f} twopass {em_2/max(n_done,1):.3f} "
                  f"twopass_le16 {float((p2a[le]==g[le]).mean()):.3f} maj {maj:.3f}"]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out, flush=True)
    lora_mid.remove()
    lora_late.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
