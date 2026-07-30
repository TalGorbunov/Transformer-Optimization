#!/usr/bin/env python3
"""Learned carrier token (stage 1): distill the per-frame question replica into ONE
trainable embedding e_c.

Layout (Q-first): [q0][frame_1][c]...[frame_N][c][q0], full block fence + per-block
M-RoPE reset. The teacher is an in-run replica forward (the A3 probe layout); the
student runs layers 0..L-1 and reads each carrier's layer-L message.

Objectives: proxy (BCE on a jointly-trained logistic head) or distill (cosine match to
the teacher replica's room-token message; task-agnostic).

Anchor (RESULTS.md [2026-07-18] E1): distill @N=8 full prior, n=900 -> carrier eval
d' 11.45 = 96% of the scale-matched teacher; teacher anchor must reproduce ~13.5 @n900
(~6.3 @n300 truncated-prior). The historic E-C carriers-at-end ablations live only in
legacy/experiments/glstm/carrier_token_distill.py.

Usage:
  python scripts/train_carrier_token.py --objective distill --limit 900 --shuffle-dirs 0 \
      --output outputs/carrier/token
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.constants import CARRIER_TOKEN, ROOMS
from gnnformer.data import (
    iter_sample_dirs,
    iter_sample_dirs_shuffled,
    load_mmred_sample,
    load_natural_sample,
    probe_evidence,
)
from gnnformer.fencing import (
    FenceHooks,
    build_block_mask,
    find_question_spans,
    frame_blocks,
    locate_word_token,
    recompute_messages,
    reset_positions,
)
from gnnformer.metrics import dprime_pair, format_gold_histogram
from gnnformer.runtime import (
    attention_dims,
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--task", choices=("steps", "cooc"), default="steps")
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="stream samples, report d' + ckpt-head tally + fresh-logistic tally")
    ap.add_argument("--carrier-ckpt", default=None)
    ap.add_argument("--shuffle-dirs", type=int, default=None, metavar="SEED")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default="outputs/carrier/token")
    args = ap.parse_args()
    if args.eval_only and not args.carrier_ckpt:
        ap.error("--eval-only requires --carrier-ckpt")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    L = args.layer
    dims = attention_dims(model)
    n_heads, n_kv, hd, mrope = dims["n_heads"], dims["n_kv"], dims["head_dim"], dims["mrope_section"]
    text_model = model.model.language_model
    dev = model.device
    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S")
                               + f"_{args.objective}_{args.init}_k{args.k}")
    out.mkdir(parents=True, exist_ok=True)
    W_O = dequantize_linear_weight(layers[L].self_attn.o_proj)
    cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    vs_id = int(model.config.vision_start_token_id)
    rope_fn = get_rope_index_fn(model)
    D = dims["hidden_size"]

    # ---- trainable params ----
    if args.eval_only:
        ck = torch.load(args.carrier_ckpt, map_location="cpu")
        e_c = nn.Parameter(ck["e_c"].float().to(dev))
        e_extra = (nn.Parameter(ck["e_extra"].float().to(dev))
                   if ck.get("e_extra") is not None else None)
        head_w = nn.Parameter(ck["head_w"].float().to(dev))
        head_b = nn.Parameter(ck["head_b"].float().to(dev))
        args.k = 1 + (e_extra.shape[0] if e_extra is not None else 0)
        print(f"[eval-only] carrier from {args.carrier_ckpt} (ep {ck.get('epoch')}, "
              f"d' {ck.get('dprime'):.2f}, k={args.k})", flush=True)
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
        head_w = nn.Parameter(torch.zeros(D, device=dev))
        head_b = nn.Parameter(torch.zeros(1, device=dev))

    hooks = FenceHooks(layers, capture_layers=[L]).install()

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
        return recompute_messages(
            seq=seq, mask_full=d["mask"], carrier_positions=d["carriers"],
            vis_by_frame=d["vis"], cos=cos, sin=sin, dims=dims, w_o=W_O,
            q_proj=q_t, k_proj=k_t, v_proj=v_t, differentiable=differentiable)

    # ---- prep ----
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
                frames, q0, gold, evid, room = load_natural_sample(sd)
            else:
                _sid, frames, q0, states, a0 = load_mmred_sample(sd)
                gold = int(str(a0).strip())
                pe_ = probe_evidence(args.task, q0, states, gold, ROOMS)
                if pe_ is None:
                    n_skip += 1
                    continue
                evid, room = pe_
        except Exception:
            n_skip += 1
            continue
        if not evid and gold != 0:
            n_skip += 1
            continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]

        # ---------- STUDENT layout: Q-first + carrier placeholders ----------
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": CARRIER_TOKEN * args.k})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, dev)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        cpos = [p for p, t in enumerate(ids) if t == cid]
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        occ = find_question_spans(ids, tok, q0, 2)
        if len(fg) != NF or len(cpos) != NF * args.k or len(vstarts) != NF or occ is None:
            n_skip += 1
            continue
        fin_start = occ[-1][0]
        carriers = [cpos[(i + 1) * args.k - 1] for i in range(NF)]
        blocks = frame_blocks(vstarts, fin_start)
        mask_s = build_block_mask(seq, blocks, hide_cols=cpos)
        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            pos_s = reset_positions(base_pos, blocks, fin_start).clone()
            emb = text_model.embed_tokens(inputs["input_ids"])
            img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb = emb.clone()
            emb[0, im_mask] = img.to(emb.dtype)
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

        # ---------- TEACHER: Q-first replica layout (A3) ----------
        contentT = [{"type": "text", "text": q0}]
        for f in frames:
            contentT.append({"type": "image", "image": f})
            contentT.append({"type": "text", "text": q0})
        contentT.append({"type": "text", "text": q0})
        inputsT = processor.apply_chat_template([{"role": "user", "content": contentT}],
                                                add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt")
        inputsT = move_to_device(inputsT, dev)
        idsT = inputsT["input_ids"][0].tolist()
        seqT = len(idsT)
        fgT = image_token_groups(inputsT["input_ids"][0].cpu(), expected_num_frames=NF,
                                 processor=processor)
        vstartsT = [p for p, t in enumerate(idsT) if t == vs_id]
        spansT = find_question_spans(idsT, tok, q0, NF + 2)
        if len(fgT) != NF or len(vstartsT) != NF or spansT is None:
            n_skip += 1
            continue
        spansT = spansT[1:]
        repT, finT = spansT[:NF], spansT[NF]
        repC = [locate_word_token(idsT, tok, room, sp) for sp in repT]
        if any(c is None for c in repC):
            n_skip += 1
            continue
        blocksT = frame_blocks(vstartsT, finT[0])
        hideT = [p for a, b in repT for p in range(a, b)]
        mask_t = build_block_mask(seqT, blocksT, hide_cols=hideT)
        vis_by_frameT = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fgT]
        with torch.inference_mode():
            base_posT, _ = rope_fn(inputsT["input_ids"], image_grid_thw=inputsT.get("image_grid_thw"),
                                   attention_mask=inputsT.get("attention_mask"))
            pos_t = reset_positions(base_posT, blocksT, finT[0])
            hooks.set_mask(mask_t, dev)
            model(**inputsT, position_ids=pos_t)
            hooks.clear_mask()
        t_msgs = recompute_messages(
            seq=seqT, mask_full=mask_t, carrier_positions=repC, vis_by_frame=vis_by_frameT,
            cos=hooks.cos, sin=hooks.sin, dims=dims, w_o=W_O,
            q_proj=hooks.qkv[L]["q_proj"], k_proj=hooks.qkv[L]["k_proj"],
            v_proj=hooks.qkv[L]["v_proj"])

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
    hooks.remove()
    print(f"prep done: n={n_done} skip={n_skip} ({time.time()-t0:.0f}s)", flush=True)
    print("[gold-hist] " + format_gold_histogram(
        gold_stream if args.eval_only else [d["gold"] for d in data]), flush=True)

    if args.eval_only:
        X = np.stack(Xs_stream)
        Y = np.stack(ys_stream)
        G = np.array(gold_stream)
        d_, s_, a_ = dprime_pair(X, Y)
        prob = 1 / (1 + np.exp(-(X @ head_w.detach().cpu().numpy() + float(head_b))))
        pred = (prob > 0.5).astype(int)
        ferr_ck = float((pred != Y).mean())
        acc_ck = float((pred.sum(1) == G).mean())
        from sklearn.linear_model import LogisticRegression

        n, NF_, Dm = X.shape
        accs, ferrs = [], []
        for seed in range(5):
            r2 = np.random.default_rng(seed)
            idx = r2.permutation(n)
            tr, ev = idx[: n // 2], idx[n // 2 :]
            clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, Dm), Y[tr].reshape(-1))
            pr = clf.predict(X[ev].reshape(-1, Dm)).reshape(len(ev), NF_)
            ferrs.append(1 - (pr == Y[ev]).mean())
            accs.append((pr.sum(1) == G[ev]).mean())
        lines = [f"=== CARRIER TOKEN EVAL-ONLY (ckpt={args.carrier_ckpt}, n={n_done}, NF={NF_}, "
                 f"data={args.data_root}) ===",
                 f"d' {d_:.2f}±{s_:.2f} (auc {a_:.2f})",
                 f"ckpt-head ZERO-SHOT: per-frame err {ferr_ck:.4f}, tally exact {acc_ck:.3f}",
                 f"fresh logistic (5 seeds): err {np.mean(ferrs):.4f}, "
                 f"exact {np.mean(accs):.3f}±{np.std(accs):.3f}"]
        (out / "report.txt").write_text("\n".join(lines) + "\n")
        np.savez(out / "messages_eval.npz", X=X, Y=Y, G=G)
        print("\n".join(lines))
        print("wrote", out)
        return 0

    y_all = np.stack([d["y"] for d in data])
    T_all = np.stack([d["teacher"] for d in data])
    dT, sT, aT = dprime_pair(T_all, y_all)
    print(f"[anchor] teacher (Q-first replica blockfence+posreset) held-out d' = "
          f"{dT:.2f}±{sT:.2f} (auc {aT:.2f}) — expect the A3 band", flush=True)

    params = [e_c, head_w, head_b] + ([e_extra] if e_extra is not None else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_done)
    tr_idx, ev_idx = order[: args.train_n], order[args.train_n :]
    dTe, sTe, _ = dprime_pair(T_all[ev_idx], y_all[ev_idx])
    print(f"[anchor] teacher EVAL-split d' = {dTe:.2f}±{sTe:.2f} (scale-matched ceiling)", flush=True)

    def eval_dprime():
        Xs, ys = [], []
        with torch.no_grad():
            for i in range(n_done):
                Xs.append(student_msgs(data[i], False))
                ys.append(data[i]["y"])
        X = np.stack(Xs)
        Y = np.stack(ys)
        d_, s_, a_ = dprime_pair(X[ev_idx], Y[ev_idx])  # headline: held-out split only
        df, _, _ = dprime_pair(X, Y)
        return d_, s_, a_, df, X, Y

    d0, s0_, a0_, df0, _, _ = eval_dprime()
    print(f"[ep 0 / init={args.init}] carrier d' eval = {d0:.2f}±{s0_:.2f} "
          f"(auc {a0_:.2f}, full {df0:.2f})", flush=True)
    traj = [(0, float(d0))]
    lines = [f"=== CARRIER TOKEN (obj={args.objective}, init={args.init}, k={args.k}, "
             f"n={n_done}, train={len(tr_idx)}, L={L}, data={args.data_root}) ===",
             f"teacher anchor d' {dT:.2f}±{sT:.2f} full-n / {dTe:.2f}±{sTe:.2f} eval-split",
             f"ep0 d' eval {d0:.2f}±{s0_:.2f} full {df0:.2f}"]
    best = (float(d0), 0)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        tot = 0.0
        for step, i in enumerate(tr_idx):
            d = data[i]
            msgs = student_msgs(d, True)
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
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
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
    lines.append(f"BEST d' {best[0]:.2f} @ ep {best[1]}  (teacher {dT:.2f})")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "traj.csv").write_text("epoch,dprime\n" + "\n".join(f"{e},{v:.4f}" for e, v in traj) + "\n")
    print("\n".join(lines[-3:]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
