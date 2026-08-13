#!/usr/bin/env python3
"""Deploy the readout fix (Garcia-style digit-row repair) on the REAL frozen model.

Mechanism: the model emits a count via logit(k) = W_U[token k] . x, where x is the post-final-norm
last-token state. We fit a count-aligned linear readout on x (the SAME x the unembedding reads),
then REPLACE the count-token rows of W_U with it. Two measurements on held-out samples:

  (1) CONSTRAINED accuracy: argmax over count tokens, model's own W_U rows (= frozen model) vs the
      repaired rows. (does aligning the readout recover the under-read count?)
  (2) FREE GENERATION: actually overwrite W_U rows + add the readout bias, run the model's native
      greedy generate, parse the integer. (does the repair survive the real autoregressive pipeline,
      or only constrained scoring? -- Garcia: constrained works, generation does not.)

Baseline (frozen model) vs repaired, reported side by side.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri


def fit_raw_readout(Xtr, ytr, d, maxc, seed):
    """Standardized multinomial logistic (well-conditioned) -> convert to RAW-space rows W[maxc+1,d], b."""
    from sklearn.linear_model import LogisticRegression
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=seed).fit((Xtr - mu) / sd, ytr)
    W = np.zeros((maxc + 1, d), dtype=np.float64); b = np.full(maxc + 1, -1e9, dtype=np.float64)
    for i, c in enumerate(clf.classes_):
        c = int(c)
        W[c] = clf.coef_[i] / sd                                   # raw-space weight
        b[c] = float(clf.intercept_[i] - (clf.coef_[i] * mu / sd).sum())  # raw-space intercept
    return W, b


@torch.inference_mode()
def gen_int(model, proc, frames, q, device, maxn):
    inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q, processor=proc))
    out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    txt = proc.tokenizer.decode(out[0, int(inputs["input_ids"].shape[1]):], skip_special_tokens=True)
    m = re.search(r"\d+", txt)
    return int(m.group()) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_steps_balanced/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-count", type=int, default=8)
    ap.add_argument("--gen-check", type=int, default=30, help="free-gen spot check on N held-out samples (0=skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/probes/deploy_readout_fix")
    ap.add_argument("--tag", default="steps_5char")
    args = ap.parse_args()

    gri.configure_runtime("Qwen/Qwen2.5-VL-7B-Instruct")
    proc = gri._processor(); model = gri._model()
    dev = model.get_output_embeddings().weight.device
    lm_head = model.get_output_embeddings()
    cand = [tgi.token_ids_of_answer(str(k), processor=proc)[0] for k in range(args.max_count + 1)]
    out = Path(args.output) / args.tag; out.mkdir(parents=True, exist_ok=True)

    cap = {}
    def pre(mod, inp): cap["x"] = inp[0].detach()
    hk = lm_head.register_forward_pre_hook(pre)

    import random
    dirs = list(iter_sample_dirs(Path(args.data_root))); random.Random(0).shuffle(dirs)
    samples = []
    n = 0
    for sd in dirs:
        if n >= args.limit: break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd); gold = int(str(a0).strip())
        except Exception:
            continue
        try:
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frames, q0, processor=proc))
            li = int(inputs["input_ids"].shape[1]) - 1
            model(**inputs, use_cache=False)
            x = cap["x"][0, li, :].float().cpu().numpy()
        except Exception as e:
            print(f"{sid} fail: {type(e).__name__}: {e}", flush=True); continue
        samples.append(dict(dir=sd, frames=frames, q=q0, gold=gold, x=x)); n += 1
        if n % 25 == 0: print(f"  collected {n}", flush=True)
    hk.remove()

    X = np.stack([s["x"] for s in samples]); y = np.array([s["gold"] for s in samples])
    d = X.shape[1]
    rng = np.random.RandomState(args.seed); perm = rng.permutation(len(samples))
    cut = int(0.65 * len(perm)); tr, te = perm[:cut], perm[cut:]
    W, b = fit_raw_readout(X[tr], y[tr], d, args.max_count, args.seed)

    WU = lm_head.weight.detach().float().cpu().numpy()[cand]           # [K,d] model's own digit rows
    model_logits = X[te] @ WU.T                                        # model constrained count logits
    patched_logits = X[te] @ W.T + b                                   # repaired readout
    model_acc = float((model_logits.argmax(1) == y[te]).mean())
    patched_acc = float((patched_logits.argmax(1) == y[te]).mean())

    lines = [f"=== DEPLOY READOUT FIX  tag={args.tag}  n={len(samples)} (train {len(tr)}/test {len(te)}) ===",
             f"gold dist: {sorted(Counter(y.tolist()).items())}",
             "",
             "CONSTRAINED (argmax over count tokens, on the REAL last-token state):",
             f"  frozen model (its own W_U digit rows) : {model_acc:.3f}",
             f"  repaired digit rows (count-aligned)   : {patched_acc:.3f}   (delta {patched_acc-model_acc:+.3f})"]

    # ---- (2) FREE GENERATION on the real model: actually overwrite W_U rows + add bias, generate ----
    if args.gen_check > 0:
        te_g = te[: int(args.gen_check)]
        base_gen = [int(gen_int(model, proc, samples[i]["frames"], samples[i]["q"], dev, args.max_count) == samples[i]["gold"]) for i in te_g]
        orig = lm_head.weight.data[cand].clone()
        Wt = torch.tensor(W, dtype=torch.float32, device=lm_head.weight.device)
        # ONE global scale so repaired digit logits match the original digit-row magnitude: keeps the
        # readout's ranking among digits intact while letting them compete normally in free generation.
        gscale = float(orig.float().norm(dim=1).mean() / (Wt.norm(dim=1).mean() + 1e-8))
        lm_head.weight.data[cand] = (Wt * gscale).to(lm_head.weight.dtype)
        bfull = torch.zeros(lm_head.weight.shape[0], dtype=lm_head.weight.dtype, device=lm_head.weight.device)
        for k, t in enumerate(cand): bfull[t] = float(b[k]) * gscale
        def bias_hook(m, i, o): return o + bfull
        bh = lm_head.register_forward_hook(bias_hook)
        rep_gen = [int(gen_int(model, proc, samples[i]["frames"], samples[i]["q"], dev, args.max_count) == samples[i]["gold"]) for i in te_g]
        bh.remove(); lm_head.weight.data[cand] = orig                  # restore the frozen model
        lines += ["", f"FREE GENERATION (native greedy, n={len(te_g)} held-out):",
                  f"  frozen model           : {np.mean(base_gen):.3f}",
                  f"  repaired (weights+bias): {np.mean(rep_gen):.3f}   (delta {np.mean(rep_gen)-np.mean(base_gen):+.3f})",
                  "  (Garcia: digit-repair lifts CONSTRAINED but often NOT free generation)"]

    report = "\n".join(lines) + "\n"
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(f"Wrote {out}/report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
