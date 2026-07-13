#!/usr/bin/env python3
"""C2: digit-compositional soft-token injection vs C-control per-count codebook.

Can a LEARNED activation-level count representation be verbalized OOD by the frozen model?
Routes (all embedding-level soft tokens, frozen Qwen, text-MMRED N=40 context):
  digit_fact  : per-DIGIT codebook {v_0..v_9}; the count slot inside the semantic fact
                sentence ("Note: C spent exactly <K> steps in the R.") carries the digit
                vectors in sequence (17 -> v_1 v_7). Held-out two-digit counts REUSE the
                same digit vectors -> extrapolation is possible in principle.
  digit_ans   : same digit codebook, injected as bare soft tokens right before 'Answer: '
                (no fact framing) — the site C1 showed fails with REAL tokens.
  count_fact  : C-CONTROL, per-COUNT codebook (one vector per count value) at the fact slot.
                Held-out counts get linearly interpolated vectors between nearest trained
                counts (the charitable OOD variant; without it the control is undefined OOD).
  token_fact  : reference, no training — real digit tokens at the fact slot (= C1 bar).
C3 routes (native/continuous-geometry injection, same machinery):
  fourier_fact  : v(K) = W·φ(K), φ = fixed Fourier/linear features of the WHOLE count
                  (periods 10/40/100 + linear); W learned on train counts; held-out counts
                  extrapolate through the continuous basis. Single vector at all count-slot
                  positions.
  fourierE_fact : same, but W is INITIALIZED by ridge-fitting φ(d) -> E(digit d) on 0..9
                  (anchored to the model's own number geometry) before training.

Train counts {0..9,12,25,30}; held-out counts = two-digit in [10,40] not in train.
No trained readout head in v1 (report codebook param count; cos(v_d, E(d)) diagnostics —
if learned vectors converge to the token embeddings, that IS the "token interface is
necessary" result).
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from experiments.readout.c1_token_interface import load_states_only, frames_as_text

TRAIN_COUNTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 25, 30]
HELD_OUT = [k for k in range(10, 41) if k not in TRAIN_COUNTS]
COUNT_MARK = "‹K›"   # ‹K› placeholder, replaced by the count digits


def build_prompt_text(states, q0, K: int, site: str, processor) -> str:
    """Chat-templated prompt string with the count K written as digits at the chosen site."""
    hi = len(states)
    import re
    m = re.search(r"did (\w+) spend in the (\w+)", q0)
    C, R = m.group(1), m.group(2)
    head = f"You are given {hi} frames describing steps in a house, as text.\n"
    body = frames_as_text(states) + "\n\n"
    instr = "Respond with a single integer. Output only the integer.\n"
    if site in ("fact",):
        note = f"Note: {C} spent exactly {K} steps in the {R}.\n"
        prompt = head + body + note + instr + f"Question: {q0}\nAnswer: "
    else:  # answer region: bare digits right before the answer slot
        prompt = head + body + instr + f"Question: {q0}\n{K}\nAnswer: "
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def find_count_token_positions(text: str, K: int, ftok, site: str):
    """Token ids + positions of the digits of K at the injection site."""
    ks = str(K)
    if site == "fact":
        anchor = f"spent exactly {ks} steps"
        start = text.index(anchor) + len("spent exactly ")
    else:
        anchor = f"\n{ks}\nAnswer:"
        start = text.rindex(anchor) + 1
    end = start + len(ks)
    enc = ftok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offmap = enc["offset_mapping"][0].tolist()
    pos = [i for i, (a, b) in enumerate(offmap) if a >= start and b <= end and b > a]
    if len(pos) != len(ks):
        raise RuntimeError(f"count '{ks}' spans {len(pos)} tokens (expected {len(ks)})")
    return enc, pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_text_longN/seq_len_40/all_uniform")
    ap.add_argument("--routes", default="digit_fact,digit_ans,count_fact,token_fact")
    ap.add_argument("--reps-per-count", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--eval-per-count", type=int, default=4)
    ap.add_argument("--inject-level", choices=["embed", "residual"], default="embed",
                    help="P5: residual = write the codebook vectors into the RESIDUAL STREAM at "
                         "the count-slot positions at --inject-layers entries (count slot holds "
                         "same-length dummy digits so tokenization/positions are unchanged)")
    ap.add_argument("--inject-layers", default="14,15,16,17")
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    dev = next(model.parameters()).device
    from models.model import get_layers as _gl
    _layers = _gl(model)
    INJ_LAYERS = [int(x) for x in args.inject_layers.replace(",", " ").split()]
    ps_inj = {"pos": None, "vecs": None, "scale_probe": None}
    if args.inject_level == "residual":
        def mk_resid_hook(L):
            def hook(_m, hargs, hkwargs):
                hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
                if hs is None or hs.shape[1] <= 1:
                    return hargs, hkwargs
                if ps_inj["scale_probe"] is not None and L == INJ_LAYERS[0]:
                    ps_inj["scale_probe"].append(
                        float(hs[0, ps_inj["pos"]].float().norm(dim=-1).mean()))
                if ps_inj["vecs"] is None or ps_inj["pos"] is None:
                    return hargs, hkwargs
                if hs.shape[1] <= max(ps_inj["pos"]):
                    return hargs, hkwargs
                hs2 = hs.clone()
                for p, v in zip(ps_inj["pos"], ps_inj["vecs"]):
                    hs2[0, p] = v.to(hs.dtype)
                if "hidden_states" in hkwargs:
                    hkwargs["hidden_states"] = hs2
                else:
                    hargs = (hs2,) + tuple(hargs[1:])
                return hargs, hkwargs
            return hook
        for L in INJ_LAYERS:
            _layers[L].register_forward_pre_hook(mk_resid_hook(L), with_kwargs=True)
    emb = model.get_input_embeddings()
    E = emb.weight.detach()
    tok = processor.tokenizer
    from transformers import AutoTokenizer
    ftok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    digit_ids = []
    for d in range(10):
        ids = tok.encode(str(d), add_special_tokens=False)
        assert len(ids) == 1, f"digit {d} not single-token"
        digit_ids.append(ids[0])
    # sanity: multi-digit numbers split into digit tokens
    assert tok.encode("17", add_special_tokens=False) == [digit_ids[1], digit_ids[7]], \
        "tokenizer does not split numbers into digits — positional mapping invalid"
    emb_dtype = E.dtype
    H = E.shape[1]
    dig_norm = float(E[torch.tensor(digit_ids)].float().norm(dim=1).mean())

    # data pool
    dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.seed).shuffle(dirs)
    pool = []
    for sd in dirs:
        try:
            sid, q0, states, a0 = load_states_only(sd)
            if states and q0:
                pool.append((sid, q0, states))
        except Exception:
            continue
    print(f"{len(pool)} text samples in pool")

    def make_batch_item(sample, K, site):
        sid, q0, states = sample
        K_slot = int("5" * len(str(K))) if args.inject_level == "residual" else K
        text = build_prompt_text(states, q0, K_slot, site, processor)
        target_ids = tok.encode(str(K), add_special_tokens=False)
        full = text + str(K)
        enc, pos = find_count_token_positions(text, K_slot, ftok, site)
        enc_full = ftok(full, return_tensors="pt", add_special_tokens=False)
        ids_full = enc_full["input_ids"][0]
        n_prompt = enc["input_ids"].shape[1]
        assert ids_full[:n_prompt].tolist() == enc["input_ids"][0].tolist()
        assert ids_full[n_prompt:].tolist() == target_ids
        return ids_full, pos, target_ids, n_prompt

    results = {}
    routes = [r.strip() for r in args.routes.replace(",", " ").split() if r.strip()]
    rng = random.Random(args.seed + 1)

    def phi(K: int) -> torch.Tensor:
        feats = [1.0, K / 40.0]
        for T in (10.0, 40.0, 100.0):
            feats += [np.sin(2 * np.pi * K / T), np.cos(2 * np.pi * K / T)]
        return torch.tensor(feats, dtype=torch.float32, device=dev)

    NPHI = 8

    for route in routes:
        site = "fact" if route.endswith("_fact") else "ans"
        kind = route.split("_")[0]           # digit | count | token | fourier | fourierE
        print(f"\n=== ROUTE {route} (site={site}, kind={kind}) ===", flush=True)
        torch.manual_seed(args.seed)
        if kind == "digit":
            book = torch.nn.Parameter(torch.randn(10, H, device=dev, dtype=torch.float32)
                                      * dig_norm / np.sqrt(H))
            params = [book]
        elif kind == "count":
            book = torch.nn.Parameter(torch.randn(len(TRAIN_COUNTS), H, device=dev,
                                                  dtype=torch.float32) * dig_norm / np.sqrt(H))
            params = [book]
        elif kind in ("fourier", "fourierE"):
            W0 = torch.randn(NPHI, H, device=dev, dtype=torch.float32) * dig_norm / np.sqrt(H * NPHI)
            if kind == "fourierE":     # anchor: ridge-fit phi(d) -> E(d) on digits 0..9
                Phi = torch.stack([phi(d) for d in range(10)])            # [10, NPHI]
                Ed = E[torch.tensor(digit_ids)].float().to(dev)           # [10, H]
                A = Phi.T @ Phi + 1e-2 * torch.eye(NPHI, device=dev)
                W0 = torch.linalg.solve(A, Phi.T @ Ed)                    # [NPHI, H]
            book = torch.nn.Parameter(W0)
            params = [book]
        else:
            book, params = None, []

        def inject_embeds(ids_full, pos, K):
            x = emb(ids_full.unsqueeze(0).to(dev)).clone()
            if kind == "digit":
                for p, ch in zip(pos, str(K)):
                    x[0, p] = book[int(ch)].to(emb_dtype)
            elif kind in ("fourier", "fourierE"):
                v = phi(K) @ book                      # [H]
                for p in pos:
                    x[0, p] = v.to(emb_dtype)
            elif kind == "count":
                if K in TRAIN_COUNTS:
                    v = book[TRAIN_COUNTS.index(K)]
                else:  # charitable OOD: linear interpolation between nearest trained counts;
                    # outside the trained range, extrapolate along the last segment's line
                    lows = [c for c in TRAIN_COUNTS if c < K]
                    highs = [c for c in TRAIN_COUNTS if c > K]
                    if lows and highs:
                        lo, hi_ = max(lows), min(highs)
                    elif lows:
                        lo, hi_ = sorted(lows)[-2], sorted(lows)[-1]
                    else:
                        lo, hi_ = sorted(highs)[0], sorted(highs)[1]
                    w = (K - lo) / (hi_ - lo)
                    v = (1 - w) * book[TRAIN_COUNTS.index(lo)] + w * book[TRAIN_COUNTS.index(hi_)]
                # count codebook occupies the FIRST count-slot position; remaining digit
                # positions collapse onto the same vector (single-vector representation)
                for p in pos:
                    x[0, p] = v.to(emb_dtype)
            return x

        def apply_injection(ids_full, pos, K):
            """Returns inputs_embeds; residual mode arms ps_inj (caller disarms after forward)."""
            if args.inject_level == "residual" and kind != "token":
                x = emb(ids_full.unsqueeze(0).to(dev))
                if kind == "digit":
                    vecs = [book[int(ch)] for ch in str(K)]
                elif kind in ("fourier", "fourierE"):
                    v = phi(K) @ book
                    vecs = [v] * len(pos)
                else:                                  # count codebook (+OOD interpolation)
                    if K in TRAIN_COUNTS:
                        v = book[TRAIN_COUNTS.index(K)]
                    else:
                        lows = [c for c in TRAIN_COUNTS if c < K]
                        highs = [c for c in TRAIN_COUNTS if c > K]
                        if lows and highs:
                            lo, hi_ = max(lows), min(highs)
                        elif lows:
                            lo, hi_ = sorted(lows)[-2], sorted(lows)[-1]
                        else:
                            lo, hi_ = sorted(highs)[0], sorted(highs)[1]
                        w = (K - lo) / (hi_ - lo)
                        v = (1 - w) * book[TRAIN_COUNTS.index(lo)] + w * book[TRAIN_COUNTS.index(hi_)]
                    vecs = [v] * len(pos)
                ps_inj["pos"] = pos
                ps_inj["vecs"] = vecs
                return x
            return inject_embeds(ids_full, pos, K)

        # residual mode: rescale the random init to the residual-stream norm at the count slot
        if args.inject_level == "residual" and params:
            sample0 = pool[0]
            ids_full, pos, target_ids, n_prompt = make_batch_item(sample0, 25, site)
            ps_inj["pos"] = pos; ps_inj["vecs"] = None; ps_inj["scale_probe"] = []
            with torch.no_grad():
                model(inputs_embeds=emb(ids_full[:n_prompt].unsqueeze(0).to(dev)),
                      attention_mask=torch.ones(1, n_prompt, dtype=torch.long, device=dev),
                      use_cache=False)
            resid_norm = float(np.mean(ps_inj["scale_probe"])) if ps_inj["scale_probe"] else dig_norm
            ps_inj["scale_probe"] = None; ps_inj["pos"] = None
            with torch.no_grad():
                book.mul_(resid_norm / dig_norm)
            print(f"  [residual] init rescaled to residual norm {resid_norm:.1f} "
                  f"(emb digit norm {dig_norm:.2f})", flush=True)

        # ---- train ----
        if params:
            opt = torch.optim.Adam(params, lr=args.lr)
            train_items = [(rng.choice(pool), K) for K in TRAIN_COUNTS
                           for _ in range(args.reps_per_count)]
            for ep in range(args.epochs):
                rng.shuffle(train_items)
                tot, nb = 0.0, 0
                for sample, K in train_items:
                    try:
                        ids_full, pos, target_ids, n_prompt = make_batch_item(sample, K, site)
                    except Exception:
                        continue
                    x = apply_injection(ids_full, pos, K)
                    am = torch.ones(1, x.shape[1], dtype=torch.long, device=dev)
                    outp = model(inputs_embeds=x, attention_mask=am, use_cache=False)
                    logits = outp.logits[0, n_prompt - 1:-1].float()
                    loss = F.cross_entropy(logits, torch.tensor(target_ids, device=dev))
                    opt.zero_grad(); loss.backward(); opt.step()
                    ps_inj["pos"] = None; ps_inj["vecs"] = None
                    tot += float(loss); nb += 1
                    if nb % 100 == 0:
                        print(f"  ep{ep} step {nb}/{len(train_items)} loss {tot/nb:.3f}",
                              flush=True)
                print(f"  epoch {ep}: mean loss {tot/max(nb,1):.4f}", flush=True)

        # ---- eval (greedy decode from injected embeds) ----
        def eval_counts(counts, tag):
            per = {}
            recs = []
            for K in counts:
                hits = []
                for _ in range(args.eval_per_count):
                    sample = rng.choice(pool)
                    try:
                        ids_full, pos, target_ids, n_prompt = make_batch_item(sample, K, site)
                    except Exception:
                        continue
                    with torch.no_grad():
                        x = apply_injection(ids_full[:n_prompt], pos, K)
                        am = torch.ones(1, x.shape[1], dtype=torch.long, device=dev)
                        gen = model.generate(inputs_embeds=x, attention_mask=am,
                                             do_sample=False, max_new_tokens=4,
                                             pad_token_id=tok.eos_token_id)
                        ps_inj["pos"] = None; ps_inj["vecs"] = None
                    dec = tok.decode(gen[0], skip_special_tokens=True)
                    import re as _re
                    mm = _re.search(r"-?\d+", dec)
                    pred = int(mm.group(0)) if mm else None
                    hits.append(int(pred == K))
                    recs.append({"K": K, "pred": pred, "raw": dec[:20]})
                per[K] = float(np.mean(hits)) if hits else float("nan")
            acc = float(np.nanmean(list(per.values())))
            print(f"  [{tag}] acc {acc:.3f}  per-count "
                  + " ".join(f"{k}:{v:.2f}" for k, v in sorted(per.items())), flush=True)
            return acc, per, recs

        model.eval()
        acc_in, per_in, recs_in = eval_counts(TRAIN_COUNTS, "train-counts")
        acc_ood, per_ood, recs_ood = eval_counts(HELD_OUT, "held-out-counts")
        diag = {}
        if kind == "digit":
            with torch.no_grad():
                cos = [float(F.cosine_similarity(book[d].float(),
                                                 E[digit_ids[d]].float().to(dev), dim=0))
                       for d in range(10)]
            diag["cos_v_Ed"] = [round(c, 3) for c in cos]
            print("  cos(v_d, E(d)):", diag["cos_v_Ed"])
        results[route] = {"inject_level": args.inject_level,
                          "acc_train_counts": acc_in, "acc_held_out": acc_ood,
                          "per_count_in": per_in, "per_count_ood": per_ood,
                          "n_params": int(sum(p.numel() for p in params)), **diag}
        (out / f"recs_{route}.json").write_text(json.dumps(recs_in + recs_ood, indent=1))
        (out / "results.json").write_text(json.dumps(results, indent=1))

    lines = [f"=== C2 DIGIT-CODEBOOK INJECTION (text N=40, train {TRAIN_COUNTS}) ==="]
    for r, v in results.items():
        lines.append(f"  {r:<12s} params={v['n_params']:<7d} in-range {v['acc_train_counts']:.3f}"
                     f"  held-out {v['acc_held_out']:.3f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
