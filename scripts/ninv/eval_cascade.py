#!/usr/bin/env python3
"""Hard-requant cascade eval (ninv) — tokens as the inter-stage interface.

The P1.5 finding: lv5/6 trained on synthetic children do NOT accept real
children, even though both regimes' lv4 nodes "say" their digits — soft token
ALIGNMENT does not make states INTERCHANGEABLE. This eval tests the implied fix
with zero training: make the interface ACTUAL DECODED NUMBERS.

Pipeline per N=64 sample (N=32: same with 2 lv4 nodes):
  1. ONE full uncapped tree forward (the P1.5/v2 ckpt as-is). Levels 5-6 exist
     but are ignored.
  2. Read each lv4 node's subtree count with the IN-LENGTH-FIT PROBE on its
     last-token state @L27 (pre-norm) — this exact read is MEASURED at 1.000 on
     real N=64 interior lv4 nodes (evalP15 [b] table), unlike the capped
     layout's shifted top spans (0.50).
  3. Pairwise-compose the VALIDATED two-operand text adder (v4d template,
     cond-EM 1.000) on the decoded counts.
Every stage sits at measured >=0.99 fidelity, so EM(GT<=16) should approach the
in-length 0.92 — if it does, "hard tokens are the interface" is confirmed on the
trained system, completing the arm-B/C/D -> P1.5 chain of evidence.

Usage:
  python scripts/ninv/eval_cascade.py --ckpt <p15 registers_best.pt> \
      --ns 32,64 --output outputs/ninv/<ts>_cascade
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

from eval_capped import ROOTS, level_counts  # noqa: E402
from load_hf_sample import evidence_bits, iter_hf_sample_dirs, load_hf_sample  # noqa: E402
from train_registers import (  # noqa: E402
    attach_masked_lora,
    build_mask,
    canonical_positions,
    tree_levels_b2,
)

from gnnformer.fencing import find_question_spans, frame_blocks  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

READ_L = 27


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ns", default="32,64")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--fit-n16", type=int, default=120)
    ap.add_argument("--max-decode", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} val_em={ck.get('val_em')}",
          flush=True)

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
    lora_rows = ck.get("lora_rows", "leaves_nodes_tail")

    def prep(sd):
        try:
            _sid, frames, q0, states, a0 = load_hf_sample(sd, resize=args.resize)
            gold = int(str(a0).strip())
            bits = evidence_bits(q0, states)
            if bits is None or sum(bits) != gold:
                return None
        except Exception:
            return None
        NF = len(frames)
        levels = tree_levels_b2(NF)          # UNCAPPED — full tree
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
        # lv4 spans: skip lv1..lv3 groups in prompt order
        n_before = sum(len(g) for g in levels[:3])
        lv4_spans = sq[n_before : n_before + len(levels[3])] if len(levels) >= 4 else []
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
                    rep=rep, sq=sq, blocks=blocks, lv4_spans=lv4_spans,
                    vis=[torch.tensor(sorted(int(p) for p in g)) for g in fg])

    @torch.no_grad()
    def forward_states(d):
        """Full uncapped forward; return PRE-NORM lv4 last-token states @L27
        (the basis the 1.000 transfer was measured in)."""
        seq = d["seq"]
        m = build_mask(seq, 0, d["rep"], d["vis"], d["blocks"], d["sq"],
                       d["levels"], d["sq"][-1])
        pos, _, _ = canonical_positions(d["base_pos"], d["blocks"], d["sq"],
                                        d["sq"][-1], seq, 0)
        ids = d["ids"].to(dev)
        emb = text_model.embed_tokens(ids.unsqueeze(0)).clone()
        emb[0, ids == model.config.image_token_id] = d["img"].to(dev).to(emb.dtype)
        for (a, b) in d["sq"]:
            emb[0, a:b] = registers[: b - a].to(emb.dtype)
        rows = torch.zeros(seq)
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
        m4 = m.to(dev).to(emb.dtype).view(1, 1, seq, seq)
        h = emb
        states = None
        for li in range(n_layers):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            if li == READ_L:
                states = [h[0, b - 1].float().cpu().numpy()
                          for (a, b) in d["lv4_spans"]]
        return states

    @torch.no_grad()
    def add2(a, b):
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
            row_holder["rows"] = torch.zeros(it["input_ids"].shape[1], device=dev)
            h = model(**it, output_hidden_states=True).hidden_states[-1][0, -1]
            lg = model.lm_head(text_model.norm(h.unsqueeze(0)).to(
                model.lm_head.weight.dtype)).float()[0]
            t = int(lg.argmax())
            piece = tok.decode([t])
            if piece.strip().isdigit():
                outp += piece.strip()
            elif outp or t == eos:
                break
            nt = torch.tensor([[t]], device=dev)
            it["input_ids"] = torch.cat([it["input_ids"], nt], 1)
            it["attention_mask"] = torch.cat([it["attention_mask"],
                                              torch.ones_like(nt)], 1)
        return int(outp) if outp else -1

    def reduce_add(vals):
        vals = list(vals)
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

    # ---- fit the lv4 probe head in-length: N=16 whole trees, root = the lv4
    # node; SAME basis (pre-norm L27 last token) the 1.000 transfer used.
    # LOCAL ridge with clip 0..16 — transfer_matrix.fit_head hard-clips to 0..2
    # (built for pair counts) and silently crippled the first cascade run AND
    # the capped eval's probe numbers.
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    t0 = time.time()
    fs, ys = [], []
    for sd in iter_hf_sample_dirs(
            Path("data/mmred_hf/dirs/seq_len_16_train_steps_in_room")):
        if len(ys) >= args.fit_n16:
            break
        d = prep(sd)
        if d is None or len(d["lv4_spans"]) != 1:
            continue
        st = forward_states(d)
        fs.append(st[0])
        ys.append(d["gold"])
    Xf, yf = np.array(fs), np.array(ys)

    def mk_head(X, y):
        mu, sd_ = X.mean(0), X.std(0) + 1e-6
        pca = PCA(n_components=min(256, X.shape[0] - 1, X.shape[1]),
                  random_state=0).fit((X - mu) / sd_)
        rg = Ridge(alpha=10.0).fit(pca.transform((X - mu) / sd_), y)

        def h(X3):
            Z = X3.reshape(-1, X3.shape[-1])
            pr = np.clip(np.round(rg.predict(pca.transform((Z - mu) / sd_))), 0, 16)
            return pr.astype(int).reshape(X3.shape[:2])
        return h

    # honest power check: 80/20 heldout BEFORE refitting on everything (the
    # first cascade run's train acc 1.000 on 120 samples was interpolation —
    # a PCA-256 basis with n~=components memorizes; heldout is the real number)
    idx = np.random.default_rng(0).permutation(len(yf))
    tr_i, ev_i = idx[: int(len(yf) * 0.8)], idx[int(len(yf) * 0.8):]
    hh = mk_head(Xf[tr_i], yf[tr_i])
    held = float((hh(Xf[ev_i][:, None, :]).reshape(-1) == yf[ev_i]).mean())
    head = mk_head(Xf, yf)
    tr_acc = float((head(Xf[:, None, :]).reshape(-1) == yf).mean())
    print(f"[lv4-head] HELDOUT acc {held:.3f} (fit {len(tr_i)}, eval {len(ev_i)})",
          flush=True)
    print(f"[lv4-head] fit on {len(ys)} in-length roots (pre-norm L{READ_L}; "
          f"train acc {tr_acc:.3f}, {time.time()-t0:.0f}s)", flush=True)

    lines = [f"ckpt={args.ckpt} basis=prenorm-L{READ_L}"]
    for n in [int(x) for x in args.ns.replace(",", " ").split()]:
        pool = iter_hf_sample_dirs(Path(ROOTS[n]))[: args.limit]
        t0 = time.time()
        em = n_done = lv4_ok = lv4_n = 0
        preds, golds = [], []
        for sd in pool:
            d = prep(sd)
            if d is None:
                continue
            states = forward_states(d)
            counts = [int(head(np.array(s)[None, None, :])[0, 0]) for s in states]
            true4 = d["lc"][3]
            lv4_ok += sum(int(a == b) for a, b in zip(counts, true4))
            lv4_n += len(true4)
            v = reduce_add(counts) if len(counts) > 1 else counts[0]
            em += int(v == d["gold"])
            preds.append(v)
            golds.append(d["gold"])
            n_done += 1
            if n_done % 10 == 0:
                print(f"  N={n}: {n_done} ({time.time()-t0:.0f}s)", flush=True)
        g, p = np.array(golds), np.array(preds)
        le = g <= 16
        print(f"[cascade N={n}] n={n_done}  lv4 probe fidelity "
              f"{lv4_ok/max(lv4_n,1):.3f} ({lv4_ok}/{lv4_n})", flush=True)
        print(f"  EM {em/max(n_done,1):.3f}  EM(GT<=16) "
              f"{float((p[le]==g[le]).mean()):.3f} (n={int(le.sum())})  "
              f"majority {float((g==0).mean()):.3f}  "
              f"MAE(GT<=16) {float(np.abs(np.where(p[le]<0,99,p[le])-g[le]).mean()):.2f}",
              flush=True)
        print(f"  emitted: {Counter(p.tolist()).most_common(8)}", flush=True)
        lines.append(f"N={n} lv4fid {lv4_ok/max(lv4_n,1):.3f} "
                     f"em {em/max(n_done,1):.3f} "
                     f"em_le16 {float((p[le]==g[le]).mean()):.3f} "
                     f"maj {float((g==0).mean()):.3f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out, flush=True)
    lora_mid.remove()
    lora_late.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
