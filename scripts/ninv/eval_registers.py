#!/usr/bin/env python3
"""Phase 1 step-4 eval — the pre-registered protocol (STATE 2026-08-10 00:3x).

For a trained register ckpt (either arm):
  a. IN-LENGTH (the trainer's own val split, N in {8,16}): per-level probe decode
     of REGISTER states (ridge AND logistic on last-token states, train-split-fit,
     val-eval, metrics_skew reporting) + greedy EMITTED EM from the isolated
     answer position, reported against the majority-emission baseline.
  b. ZERO-SHOT N=64 (seq_len_64_test, steps_in_room): register decodability
     transfer — heads FIT on N in {8,16} train-split register states, TESTED on
     N=64 states per level (levels beyond the training depth reuse the same
     heads; labels clipped to each head's range) — plus emitted EM (full and
     GT<=16). THE flat-in-N claim on trained components.
  c. FRAME-PERMUTATION CANARY (N=8, first val sample): permute the frames, the
     answer-position logits must be bit-identical under the fence. Drift = leak.

The per-level probe answers the question the aux head cannot: does the register
STATE contain the count (probe decodes it) even where the shared 17-way aux head
fails? A root that probes well but emits badly is a READOUT problem; a root that
probes badly is a MERGE-DEPTH problem.

Usage:
  python scripts/ninv/eval_registers.py \
    --ckpt outputs/ninv/<...>/registers_best.pt --output outputs/ninv/<ts>_eval_armX \
    [--skip-n64] [--limit-n64 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import evidence_bits, iter_hf_sample_dirs, load_hf_sample  # noqa: E402
from metrics_skew import class_report, format_report  # noqa: E402
from train_registers import (  # noqa: E402
    attach_masked_lora,
    build_mask,
    canonical_positions,
    level_counts,
    tree_levels_b2,
)

from gnnformer.carriers import attach_lora  # noqa: E402
from gnnformer.runtime import (  # noqa: E402
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

N64_ROOT = "data/mmred_hf/dirs/seq_len_64_test"
READ_LAYERS = (20, 24, 27)


def fit_probe(Xtr, ytr, kind, hi):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    p = PCA(n_components=min(256, Xtr.shape[0] - 1, Xtr.shape[1]),
            random_state=0).fit((Xtr - mu) / sd)
    Z = p.transform((Xtr - mu) / sd)
    if kind == "ridge":
        m = Ridge(alpha=10.0).fit(Z, ytr)
        return lambda X: np.clip(np.round(m.predict(
            p.transform((X - mu) / sd))), 0, hi).astype(int)
    m = LogisticRegression(max_iter=1000).fit(Z, ytr)
    return lambda X: m.predict(p.transform((X - mu) / sd))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--limit-n64", type=int, default=50)
    ap.add_argument("--skip-n64", action="store_true")
    ap.add_argument("--max-decode", type=int, default=4)
    ap.add_argument("--canary-only", action="store_true",
                    help="run ONLY the three-tier permutation discrimination and "
                         "exit. The pre-registered bit-identity canary is the "
                         "wrong invariant for a TREE (a regrouping perm changes "
                         "intermediate pair contents legitimately); this mode "
                         "separates that from a real leak: T1 within-pair swap "
                         "[1,0,3,2,...] and T2 pair-block swap [2,3,0,1,...] "
                         "preserve every subtree's content and must match to "
                         "bf16 noise (<1e-2) — a large diff there IS a leak; "
                         "T3 regroup may move logits but the EMITTED ANSWER "
                         "must be invariant under all three")
    ap.add_argument("--canary-samples", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    arm = ck["leaf_input"]
    print(f"[ckpt] {args.ckpt}  arm={arm}  epoch={ck['epoch']}  "
          f"val_em={ck.get('val_em')}", flush=True)
    ckdir = Path(args.ckpt).parent
    train_dirs = [Path(l) for l in (ckdir / "train_dirs.txt").read_text().splitlines() if l.strip()]
    eval_dirs = [Path(l) for l in (ckdir / "eval_dirs.txt").read_text().splitlines() if l.strip()]

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
    lora_rows = ck.get("lora_rows", "all")   # pre-C/D ckpts have no field
    row_holder: dict = {"rows": None}
    if lora_rows == "all":
        lora_mid = attach_lora(layers[: ck["late_open"]], ck["mid_open"],
                               rank=ck["rank"], alpha=ck["alpha"], device=dev,
                               state=ck["lora_mid"])
        lora_late = attach_lora(layers, ck["late_open"], rank=ck["rank"],
                                alpha=ck["alpha"], device=dev, state=ck["lora_late"])
    else:
        lora_mid = attach_masked_lora(layers[: ck["late_open"]], ck["mid_open"],
                                      rank=ck["rank"], alpha=ck["alpha"], device=dev,
                                      holder=row_holder, state=ck["lora_mid"])
        lora_late = attach_masked_lora(layers, ck["late_open"], rank=ck["rank"],
                                       alpha=ck["alpha"], device=dev,
                                       holder=row_holder, state=ck["lora_late"])
        print(f"[lora] row-masked ({lora_rows})", flush=True)
    quant = None
    if arm == "quantized":
        qp = ck["quant_probe"]
        quant = (qp["w"].to(dev), float(qp["b"]), qp["yes"].to(dev), qp["no"].to(dev))
        print(f"[quant] codes at layers[{ck['quant_layer']}] (from ckpt)", flush=True)

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
        levels = tree_levels_b2(NF)
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
        from gnnformer.fencing import find_question_spans, frame_blocks
        spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            return None
        rep = spans[1 : NF + 1]
        sq = spans[NF + 1 :]
        blocks = frame_blocks(vstarts, sq[0][0])
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
                    rep=rep, sq=sq, blocks=blocks,
                    vis=[torch.tensor(sorted(int(p) for p in g)) for g in fg])

    @torch.no_grad()
    def forward(d, extra_ids=(), capture=False, perm=None):
        """One no-grad forward; extra_ids = already-generated answer tokens.
        Returns (last-position logits, {(level, L): [state per node]} if capture).
        perm: frame permutation (canary) — reorders img embeds AND evidence bits."""
        seq, e = d["seq"], len(extra_ids)
        root = d["sq"][-1]
        m = build_mask(seq, e, d["rep"], d["vis"], d["blocks"], d["sq"],
                       d["levels"], root)
        pos, _, _ = canonical_positions(d["base_pos"], d["blocks"], d["sq"],
                                        root, seq, e)
        if lora_rows != "all":
            rows = torch.zeros(seq + e)
            for (a, b) in d["sq"]:
                rows[a:b] = 1.0
            rows[d["sq"][-1][1]:] = 1.0
            if lora_rows == "leaves_nodes_tail":
                for (a, b) in d["rep"]:
                    rows[a:b] = 1.0
            row_holder["rows"] = rows.to(dev)
        ids = d["ids"].to(dev)
        emb = text_model.embed_tokens(ids.unsqueeze(0)).clone()
        im_mask = ids == model.config.image_token_id
        img = d["img"].to(dev)
        if perm is not None:
            # permute frames by permuting each frame's image-token embeddings;
            # frame token-counts are equal (uniform resize) so spans align
            groups = [g.tolist() if hasattr(g, "tolist") else list(g) for g in d["vis"]]
            flat = torch.cat([torch.tensor(groups[p]) for p in perm])
            orig = torch.cat([torch.tensor(g) for g in groups])
            pos_of = {int(p): i for i, p in enumerate(orig.tolist())}
            img = img[torch.tensor([pos_of[int(p)] for p in flat.tolist()])]
        emb[0, im_mask] = img.to(emb.dtype)
        for (a, b) in d["sq"]:
            emb[0, a:b] = registers[: b - a].to(emb.dtype)
        if e:
            et = torch.tensor(list(extra_ids), device=dev)
            emb = torch.cat([emb, text_model.embed_tokens(et.unsqueeze(0))], dim=1)
        pos = pos.to(dev)
        cos, sin = text_model.rotary_emb(emb, pos)
        pe = (cos.to(emb.dtype), sin.to(emb.dtype))
        m4 = m.to(dev).to(emb.dtype).view(1, 1, seq + e, seq + e)
        h = emb
        caps = {}
        for li in range(n_layers):
            h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            if quant is not None and li == ck["quant_layer"]:
                w, b, cy, cn = quant
                for (qa, qb) in d["rep"]:
                    span = h[0, qa:qb].float()
                    verdict = float(span.mean(0) @ w + b) > 0
                    h[0, qa:qb] = ((cy if verdict else cn)
                                   * span.norm(dim=-1).mean()).to(h.dtype)
            if capture and li in READ_LAYERS:
                si = 0
                for lv, groups in enumerate(d["levels"]):
                    for gi in range(len(groups)):
                        a, b = d["sq"][si + gi]
                        caps.setdefault((lv, li), []).append(
                            h[0, b - 1].float().cpu().numpy())
                    si += len(groups)
        hf = text_model.norm(h)
        lg = model.lm_head(hf[0, -1:].to(model.lm_head.weight.dtype)).float()[0]
        return lg, caps

    def emit(d, perm=None):
        toks = []
        for _ in range(args.max_decode):
            lg, _ = forward(d, extra_ids=toks, perm=perm)
            t = int(lg.argmax())
            if t == eos:
                break
            toks.append(t)
        s = tok.decode(toks).strip()
        return int(s) if s.isdigit() else -1

    lines = [f"ckpt={args.ckpt} arm={arm} epoch={ck['epoch']}"]

    if args.canary_only:
        # three-tier discrimination; see the --canary-only help text
        # T0 identity replay = THE control the first canary run lacked: the same
        # forward twice. If T0 diffs are comparable to T1/T2, the "leak" signal is
        # GPU non-determinism (bf16 SDPA reduction order), not information flow —
        # and the meaningful invariant is answer stability across replicates.
        tiers = {"T0 identity":    [0, 1, 2, 3, 4, 5, 6, 7],
                 "T1 within-pair": [1, 0, 3, 2, 5, 4, 7, 6],
                 "T2 pair-block":  [2, 3, 0, 1, 6, 7, 4, 5],
                 "T3 regroup":     [2, 4, 3, 6, 5, 0, 1, 7]}
        stats = {t: {"maxd": 0.0, "ans_flips": 0} for t in tiers}
        n_done = 0
        for sd in eval_dirs:
            if n_done >= args.canary_samples:
                break
            d = prep(sd)
            if d is None or d["NF"] != 8:
                continue
            lg0, _ = forward(d)
            a0 = emit(d)
            for tname, perm in tiers.items():
                lg, _ = forward(d, perm=perm)
                stats[tname]["maxd"] = max(stats[tname]["maxd"],
                                           float((lg0 - lg).abs().max()))
                stats[tname]["ans_flips"] += int(emit(d, perm=perm) != a0)
            n_done += 1
        leak = stats["T1 within-pair"]["maxd"] > 1e-2 or \
            stats["T2 pair-block"]["maxd"] > 1e-2
        flips = sum(s["ans_flips"] for s in stats.values())
        for tname, s in stats.items():
            print(f"[canary] {tname:<16} max|logit diff| {s['maxd']:.3e}   "
                  f"answer flips {s['ans_flips']}/{n_done}", flush=True)
        print(f"[canary] VERDICT: content-preserving tiers "
              f"{'EXCEED bf16 noise -> REAL LEAK, STOP' if leak else 'at FP noise -> no leak'}; "
              f"answer invariance {'FAILED' if flips else 'holds'} "
              f"({n_done} samples; arm={arm}"
              + (", note: an all-zero emitter passes T3 vacuously" if arm == "raw" else "")
              + ")", flush=True)
        (out / "report.txt").write_text(
            "\n".join([f"{t} maxd {s['maxd']:.3e} flips {s['ans_flips']}"
                       for t, s in stats.items()]
                      + [f"leak={leak} flips={flips} n={n_done}"]) + "\n")
        lora_mid.remove()
        lora_late.remove()
        return 1 if (leak or flips) else 0

    # -------- a. in-length: capture + emit on train/val splits
    print("[a] in-length capture + emitted EM...", flush=True)
    t0 = time.time()
    def run_split(dirs, do_emit):
        feats, labs, ems, golds = {}, {}, [], []
        for sd in dirs:
            d = prep(sd)
            if d is None:
                continue
            _, caps = forward(d, capture=True)
            for (lv, L), st in caps.items():
                feats.setdefault((lv, L), []).extend(st)
                labs.setdefault(lv, [] if lv not in labs else labs[lv])
            for lv, groups in enumerate(d["levels"]):
                labs.setdefault(lv, []).extend(d["lc"][lv])
            if do_emit:
                ems.append((emit(d), d["gold"]))
            golds.append(d["gold"])
        return feats, labs, ems, golds

    ftr, ltr, _, _ = run_split(train_dirs, do_emit=False)
    fev, lev, ems, golds_ev = run_split(eval_dirs, do_emit=True)
    print(f"    captured {len(train_dirs)} train / {len(eval_dirs)} val "
          f"({time.time()-t0:.0f}s)", flush=True)

    pred, gold_arr = np.array([p for p, _ in ems]), np.array([g for _, g in ems])
    maj = float((gold_arr == 0).mean())
    em_v = float((pred == gold_arr).mean())
    em_nz = float((pred[gold_arr > 0] == gold_arr[gold_arr > 0]).mean())
    print(f"[a] EMITTED in-length: EM {em_v:.3f} | majority-emission {maj:.3f} | "
          f"EM-on-gold>0 {em_nz:.3f} | emitted dist {Counter(pred.tolist()).most_common(6)}",
          flush=True)
    lines.append(f"in-length emitted EM {em_v:.3f} majority {maj:.3f} nz {em_nz:.3f}")

    print("[a] per-level probe decode of register states (fit=train, eval=val):")
    heads = {}
    for (lv, L) in sorted(ftr):
        hi = 2 ** (lv + 1)
        Xtr, ytr = np.array(ftr[(lv, L)]), np.array(ltr[lv])
        Xev, yev = np.array(fev[(lv, L)]), np.array(lev[lv])
        for kind in ("ridge", "logit"):
            h = fit_probe(Xtr, np.clip(ytr, 0, 16), kind, min(hi, 16))
            heads[(lv, L, kind)] = h
            r = class_report(h(Xev), np.clip(yev, 0, 16), n_classes=min(hi, 16) + 1)
            print(f"    lv{lv+1} @L{L} {kind:<6} "
                  + format_report(r, prefix="")[:120], flush=True)
            lines.append(f"lv{lv+1} L{L} {kind} raw {r['raw']:.3f} bal {r['balanced']:.3f}")

    # -------- c. permutation canary (before the long N=64 pass)
    d0 = next(d for d in (prep(sd) for sd in eval_dirs) if d is not None and d["NF"] == 8)
    lg_a, _ = forward(d0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(8)
    lg_b, _ = forward(d0, perm=perm.tolist())
    bit_ok = bool(torch.equal(lg_a, lg_b))
    print(f"[c] permutation canary (N=8, perm {perm.tolist()}): "
          f"bit-identical={bit_ok} max|diff|={float((lg_a-lg_b).abs().max()):.3e}",
          flush=True)
    lines.append(f"perm canary bit-identical={bit_ok}")
    if not bit_ok:
        print("    NOTE: answer logits depend on frame order -> fence/pos leak; "
              "STOP per the pre-registered plan (recorded, eval continues to "
              "gather the rest of the evidence).", flush=True)

    # -------- b. zero-shot N=64
    if not args.skip_n64:
        print("[b] zero-shot N=64...", flush=True)
        t0 = time.time()
        f64, l64, ems64 = {}, {}, []
        n_done = 0
        for sd in iter_hf_sample_dirs(Path(N64_ROOT)):
            if n_done >= args.limit_n64:
                break
            d = prep(sd)
            if d is None:
                continue
            _, caps = forward(d, capture=True)
            for (lv, L), st in caps.items():
                f64.setdefault((lv, L), []).extend(st)
            for lv, groups in enumerate(d["levels"]):
                l64.setdefault(lv, []).extend(d["lc"][lv])
            ems64.append((emit(d), d["gold"]))
            n_done += 1
            if n_done % 10 == 0:
                print(f"    {n_done} ({time.time()-t0:.0f}s)", flush=True)
        p64 = np.array([p for p, _ in ems64])
        g64 = np.array([g for _, g in ems64])
        le = g64 <= 16
        print(f"[b] EMITTED N=64: EM(GT<=16) "
              f"{float((p64[le]==g64[le]).mean()):.3f} (n={int(le.sum())})  "
              f"full EM {float((p64==g64).mean()):.3f} (n={len(g64)})  "
              f"majority {float((g64==0).mean()):.3f}  "
              f"MAE {float(np.abs(np.where(p64<0, 99, p64)-g64)[le].mean()):.2f}",
              flush=True)
        lines.append(f"N=64 emitted EM(GT<=16) {float((p64[le]==g64[le]).mean()):.3f}")
        print("[b] register decodability transfer (heads FIT in-length, TESTED at N=64):")
        for (lv, L) in sorted(f64):
            hi = min(2 ** (lv + 1), 16)
            X, y = np.array(f64[(lv, L)]), np.clip(np.array(l64[lv]), 0, hi)
            for kind in ("ridge", "logit"):
                if (lv, L, kind) in heads:
                    r = class_report(heads[(lv, L, kind)](X), y, n_classes=17)
                    print(f"    lv{lv+1} @L{L} {kind:<6} "
                          + format_report(r, prefix="")[:120], flush=True)
                    lines.append(f"N64 lv{lv+1} L{L} {kind} raw {r['raw']:.3f} "
                                 f"bal {r['balanced']:.3f}")
                else:
                    # deeper level than the training depth: reuse the deepest head
                    deepest = max(k[0] for k in heads if k[1] == L and k[2] == kind)
                    r = class_report(heads[(deepest, L, kind)](X), y, n_classes=17)
                    print(f"    lv{lv+1} @L{L} {kind:<6} (head from lv{deepest+1}) "
                          + format_report(r, prefix="")[:100], flush=True)

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print("wrote", out, flush=True)
    lora_mid.remove()
    lora_late.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
