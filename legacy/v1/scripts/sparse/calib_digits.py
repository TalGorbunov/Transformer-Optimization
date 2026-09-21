#!/usr/bin/env python3
"""Minimal-fix ladder, rung A0 (2026-09-14): NO LoRA — is a re-calibration of the
share->digit readout enough once the oracle gate removes the (N-k) term?

Frozen Qwen2.5-VL + fence + ORACLE gate, prompt = the N-free count prompt (S9
protocol; no N in the text, so no declared-N calibration channel). One forward per
sample, no decode. Capture at the LAST PROMPT TOKEN (the answer position):
  h  = post-final-norm hidden state (hidden_states[-1][0, -1], 3584-d)
  z  = the frozen model's logits over the ten digit tokens '0'..'9'
Fit on the training pool (gold <= 9 only; single-digit readout by design):
  arm 'frozen'  : argmax_d z_d                       (0 trained params)
  arm 'bias'    : argmax_d z_d + b_d                  (10 params)
  arm 'rows'    : argmax_d (W h)_d + b_d, W init from lm_head digit rows (10x3584+10)
Eval on the exam ladder files (gold <= 9 subset; multi-digit golds are OUT of this
readout's range and are counted/reported separately). Headline = the gold 1-8 band,
comparable to the flat-line figure's band.
Outputs: run_dir/{states_train.npz, states_<evalfile>.npz, report.txt, results.csv,
per_gold.csv, config.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "sparse"))
import train_sft_gated as tg  # noqa: E402  (module-level helpers; main() not run)
from gnnformer.constants import ROOMS  # noqa: E402
from gnnformer.data import (  # noqa: E402
    iter_sample_dirs_shuffled, load_mmred_sample, parse_task_labels, probe_evidence,
    read_dirs_file, rooms_to_room2chars)
from gnnformer.fencing import FenceHooks, build_block_mask, reset_positions  # noqa: E402
from gnnformer.metrics import format_gold_histogram  # noqa: E402
from gnnformer.runtime import get_layers, get_rope_index_fn, load_runtime, move_to_device  # noqa: E402
from tasks import evidence_count, target_of  # noqa: E402

DIGITS = [str(d) for d in range(10)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root",
                    default="data/mmred_images_park/seq_len_8/all_uniform,"
                            "data/mmred_longN_park/seq_len_16/all_uniform")
    ap.add_argument("--limit", type=int, default=400, help="per root")
    ap.add_argument("--exclude-dirs-file", action="append", default=[])
    ap.add_argument("--eval-dirs-file", action="append", default=[])
    ap.add_argument("--eval-longn-limit", type=int, default=150)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--nfree-prompt", action="store_true", default=True)
    ap.add_argument("--fit-steps", type=int, default=600)
    ap.add_argument("--fit-lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/sparse/a0_calib"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = args.output / f"{time.strftime('%Y%m%d_%H%M%S')}_a0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log = (run_dir / "report.txt").open("w", encoding="utf-8")

    def emit(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    digit_ids = [tok(d, add_special_tokens=False).input_ids for d in DIGITS]
    assert all(len(x) == 1 for x in digit_ids), digit_ids
    digit_ids = [x[0] for x in digit_ids]
    hooks = FenceHooks(layers).install()
    emit(f"[setup] gate=oracle nfree={args.nfree_prompt} digits={digit_ids}")

    def oracle_evid(q0, states, char, room):
        k = evidence_count(states, char, room)
        evid = {t for t, st in enumerate(states)
                if char in rooms_to_room2chars(st.get("rooms", {})).get(room, [])}
        if len(evid) != k:
            raise ValueError(f"|evid|={len(evid)} != k={k}")
        pe = probe_evidence("steps", q0, states, k, ROOMS)
        if pe is not None and set(pe[0]) != evid:
            raise ValueError("probe_evidence disagrees")
        return evid, k

    def load_frames(sd):
        _sid, frames, q0, states, a0 = load_mmred_sample(sd)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        tr = target_of(Path(str(sd)), q0)
        if tr is None:
            raise ValueError("no target")
        return frames, q0, tr[0], tr[1], states, int(str(a0).strip())

    @torch.inference_mode()
    def capture(sd):
        """-> (h[3584] f32, z[10] f32, gold, k, nf)"""
        frames, q0, char, room, states, gold = load_frames(sd)
        evid, k = oracle_evid(q0, states, char, room)
        inp = processor.apply_chat_template(
            tg.build_fenced_messages(frames, q0, nfree=args.nfree_prompt),
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inp = move_to_device(dict(inp), rt.device)
        ids = inp["input_ids"][0].tolist()
        parsed = tg.parse_layout(ids, tok, q0, len(frames), vs_id)
        if parsed is None:
            raise ValueError("layout parse failed")
        blocks, fin_start = parsed
        hide = [c for t, (a, b) in enumerate(blocks) if t not in evid for c in range(int(a), int(b))]
        mask = build_block_mask(len(ids), blocks, hide_cols=hide)
        base_pos, _ = rope_fn(inp["input_ids"], image_grid_thw=inp.get("image_grid_thw"),
                              attention_mask=inp.get("attention_mask"))
        pos = reset_positions(base_pos, blocks, fin_start)
        cur = {kk: v for kk, v in inp.items() if kk != "attention_mask"}
        hooks.set_mask(mask, rt.device)
        try:
            with sdpa_kernel(tg.FENCED_SDPA):
                out = model(**cur, position_ids=pos.to(rt.device),
                            output_hidden_states=True, use_cache=False)
        finally:
            hooks.clear_mask()
        h = out.hidden_states[-1][0, -1].float().cpu().numpy()
        z = out.logits[0, -1, digit_ids].float().cpu().numpy()
        return h, z, gold, k, len(frames)

    def capture_pool(dirs, tag, cap):
        H, Z, G, K, NF, names = [], [], [], [], [], []
        skipped = 0
        for sd in dirs[:cap]:
            try:
                h, z, g, k, nf = capture(sd)
            except Exception as exc:
                emit(f"  skip {Path(str(sd)).name}: {exc}")
                skipped += 1
                continue
            H.append(h); Z.append(z); G.append(g); K.append(k); NF.append(nf)
            names.append(str(sd))
        H, Z, G, K, NF = map(np.asarray, (H, Z, G, K, NF))
        np.savez(run_dir / f"states_{tag}.npz", h=H, z=Z, gold=G, k=K, nf=NF,
                 names=np.array(names))
        emit(f"[capture:{tag}] n={len(G)} skipped={skipped} gold-hist "
             + format_gold_histogram(int(g) for g in G))
        return H, Z, G, K, NF

    # ---- training pool (same conventions as the trainer: shuffled, excluded exams)
    excluded = set()
    for f in args.exclude_dirs_file:
        excluded.update(str(Path(p).resolve()) for p in read_dirs_file(Path(f)))
    train_dirs = []
    for root in args.data_root.split(","):
        root = root.strip()
        n_root = 0
        for sd in iter_sample_dirs_shuffled(Path(root), 0):
            if n_root >= args.limit:
                break
            if excluded and str(Path(sd).resolve()) in excluded:
                continue
            try:
                _s, _f, q0, states, a0 = load_mmred_sample(sd)
                if parse_task_labels(q0, states, int(str(a0).strip())) is None:
                    continue
            except Exception:
                continue
            train_dirs.append(sd)
            n_root += 1
    emit(f"[data] train pool {len(train_dirs)} dirs (excluded {len(excluded)} exam dirs)")
    H, Z, G, K, NF = capture_pool(train_dirs, "train", len(train_dirs))
    lm_rows = model.lm_head.weight[digit_ids].detach().float().cpu()  # [10, d]

    # ---- fits (CPU), gold <= 9 only
    m9 = G <= 9
    Ht, Zt, Gt = torch.tensor(H[m9]), torch.tensor(Z[m9]), torch.tensor(G[m9])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Gt))
    n_va = int(len(Gt) * args.val_frac)
    va, tr = perm[:n_va], perm[n_va:]
    emit(f"[fit] gold<=9 train={len(tr)} val={len(va)} (dropped {int((~m9).sum())} gold>9)")

    def fit(kind):
        if kind == "bias":
            params = [torch.zeros(10, requires_grad=True)]
            def f(h, z): return z + params[0]
        else:
            W = lm_rows.clone().requires_grad_(True)
            b = torch.zeros(10, requires_grad=True)
            params = [W, b]
            def f(h, z): return h @ W.T + b
        opt = torch.optim.Adam(params, lr=args.fit_lr)
        best, best_state, bad = -1.0, None, 0
        for step in range(args.fit_steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(f(Ht[tr], Zt[tr]), Gt[tr])
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == args.fit_steps - 1:
                with torch.no_grad():
                    acc = float((f(Ht[va], Zt[va]).argmax(1) == Gt[va]).float().mean()) if n_va else float('nan')
                if n_va == 0 or acc > best:
                    best, best_state, bad = acc, [p.detach().clone() for p in params], 0
                else:
                    bad += 1
        emit(f"[fit:{kind}] val_acc={best:.3f} final_train_loss={float(loss):.3f}")
        st = best_state
        if kind == "bias":
            return lambda h, z: z + st[0]
        return lambda h, z: h @ st[0].T + st[1]

    arms = {"frozen": (lambda h, z: z), "bias": fit("bias"), "rows": fit("rows")}
    with torch.no_grad():
        for name, f in arms.items():
            acc = float((f(Ht[va], Zt[va]).argmax(1) == Gt[va]).float().mean()) if n_va else float('nan')
            emit(f"[val] {name}: {acc:.3f}")

    # ---- exam ladder
    rows = ["file,arm,n_total,n_gold_le9,acc_le9,n_band_1_8,acc_band_1_8"]
    pg_rows = ["file,arm,gold,correct,n"]
    for f_ in args.eval_dirs_file:
        dirs = read_dirs_file(Path(f_))
        He, Ze, Ge, Ke, NFe = capture_pool(dirs, Path(f_).stem, args.eval_longn_limit)
        m = Ge <= 9
        band = (Ge >= 1) & (Ge <= 8)
        with torch.no_grad():
            for name, fn in arms.items():
                pred = fn(torch.tensor(He), torch.tensor(Ze)).argmax(1).numpy()
                ok = pred == Ge
                acc9 = float(ok[m].mean()) if m.any() else float('nan')
                accb = float(ok[band].mean()) if band.any() else float('nan')
                rows.append(f"{f_},{name},{len(Ge)},{int(m.sum())},{acc9:.4f},{int(band.sum())},{accb:.4f}")
                per = " ".join(f"g{g}:{int(ok[Ge==g].sum())}/{int((Ge==g).sum())}" for g in sorted(set(Ge[m].tolist())))
                for g in sorted(set(Ge[m].tolist())):
                    pg_rows.append(f"{f_},{name},{g},{int(ok[Ge==g].sum())},{int((Ge==g).sum())}")
                emit(f"LADDER {Path(f_).name} {name}: gold<=9 acc={acc9:.4f} (n={int(m.sum())}/{len(Ge)}) "
                     f"band1-8 acc={accb:.4f} (n={int(band.sum())})\n  per-gold {per}")
    (run_dir / "results.csv").write_text("\n".join(rows) + "\n")
    (run_dir / "per_gold.csv").write_text("\n".join(pg_rows) + "\n")
    hooks.remove()
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
