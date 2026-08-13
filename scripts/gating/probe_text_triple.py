#!/usr/bin/env python3
"""P0 of the gating campaign: the only true G1 ablation that exists.

Three matched 1B checkpoints from `QwQZh/gated_attention` (`1B_baseline`,
`1B_gate_headwise`, `1B_gate_elementwise`) — identical data/params, gating the only
intended variable — probed for MMReD tally decodability on TEXT MMReD.

What it measures, per (model, root, layer, pool):
  * ridge-round exact / +-1 / R^2 on a held-out half (HEADLINE; logistic regression is
    NOT reported — it understates ordinal decodability, a known probe-family artifact).
  * a majority-class control so "exact" is readable against the gold prior.
And per (model, layer), on a small subset with eager attention:
  * F-Attn = fraction of attention mass on token 0 (their Table 4 sink metric)
  * M-Act  = mean max |hidden activation|
which is the sanity check that the gated checkpoints really are the sink-free ones.

The gate in these checkpoints is FUSED INTO q_proj (widened output, split into
query_states + gate_score), applied after SDPA before o_proj = the paper's G1.

Usage:
  python scripts/gating/probe_text_triple.py --model-root <snapshot> \
      --roots data/mmred_text_longN/seq_len_16/all_uniform ... \
      --limit 200 --output outputs/gating/p0_text_triple/<stamp>
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

VARIANTS = ("1B_baseline", "1B_gate_headwise", "1B_gate_elementwise")


# --------------------------------------------------------------------- text samples

def load_text_sample(sample_dir: Path) -> Optional[Tuple[str, List[str], str, int]]:
    """Text MMReD dir -> (sid, state_lines, question, gold). None if malformed.

    Same qa.txt grammar as gnnformer.data.load_mmred_sample, minus the PNG loading
    (these roots are text-only, so that loader cannot be reused)."""
    f = sample_dir / "qa.txt"
    if not f.exists():
        return None
    lines = f.read_text(encoding="utf-8").splitlines()
    q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "question:"), -1)
    a_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "answer:"), -1)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None
    states, question = [], None
    for ln in lines[q_idx + 1 : a_idx]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            ast.literal_eval(s)  # validate; we feed the raw line to the model
            states.append(s)
            continue
        question = s
        break
    if question is None or not states:
        return None
    ans = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if ans is None or not ans.isdigit():
        return None
    return sample_dir.name, states, question, int(ans)


def build_text_prompt(states: List[str], question: str) -> str:
    """PROMPT-CRITICAL: fixed once, before any number was seen. Mirrors the wording of
    gnnformer.data.build_count_prompt for the image task. Do not tune per model."""
    body = "\n".join(states)
    return (
        f"You will be shown {len(states)} frames describing steps in a house.\n"
        f"{body}\n"
        f"Question: {question}\n"
        f"Answer: "
    )


def iter_text_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "qa.txt").exists()]


# ------------------------------------------------------------------------- the probe

def ridge_round(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, alphas: Tuple[float, ...]
) -> Dict[str, float]:
    """Ridge regression on standardized features, alpha picked on the TRAIN half by
    5-fold CV; report rounded-exact / +-1 / R^2 on the held-out half."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rr = np.rint(pred)
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - ytr.mean()) ** 2).sum())
    return {
        "exact": float((rr == yte).mean()),
        "within1": float((np.abs(rr - yte) <= 1).mean()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "alpha": float(model[-1].alpha_),
    }


def majority_control(y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> Dict[str, float]:
    """Train-majority-class prediction on the held-out half — the floor `exact` is read
    against. Also the train-mean rounded, which is what a null ridge collapses to."""
    vals, cnts = np.unique(y[tr], return_counts=True)
    maj = float(vals[int(np.argmax(cnts))])
    m = np.rint(y[tr].mean())
    return {
        "majority_exact": float((y[te] == maj).mean()),
        "majority_within1": float((np.abs(y[te] - maj) <= 1).mean()),
        "trainmean_exact": float((y[te] == m).mean()),
    }


# ------------------------------------------------------------------------ extraction

@torch.no_grad()
def extract_hiddens(model, tok, prompts: List[str], device, max_len: int,
                    digit_ids: Optional[List[int]] = None):
    """-> (H_last [n, L+1, D], H_mean [n, L+1, D], token counts, emitted answers). One
    forward per sample (the sequences differ in length; batching would need padding and
    change the mean pool).

    `emitted` is what the model actually SAYS, not what a probe can decode from it:
    (a) argmax restricted to the single tokens '0'..'9' at the answer position, and
    (b) a free greedy digit decode (up to 4 tokens, stop at the first non-digit) so
    multi-digit counts are reachable. These are base LMs with no instruction tuning, so a
    low emitted score can mean "cannot follow the format" rather than "does not know" —
    which is exactly why the probe exists alongside it."""
    last_rows, mean_rows, ntoks, emitted = [], [], [], []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple (L+1) of [1, T, D]
        last_rows.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
        mean_rows.append(torch.stack([h[0].mean(0) for h in hs]).float().cpu().numpy())
        ntoks.append(int(ids["input_ids"].shape[1]))
        if digit_ids is not None:
            lg = out.logits[0, -1]
            single = int(np.argmax([float(lg[t]) for t in digit_ids]))
            seq = ids["input_ids"]
            toks: List[int] = []
            for _ in range(4):
                nxt = int(model(input_ids=seq, use_cache=False).logits[0, -1].argmax())
                if not tok.decode([nxt]).strip().isdigit():
                    break
                toks.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
            txt = tok.decode(toks).strip()
            emitted.append((single, int(txt) if txt.isdigit() else None))
        del out, hs
        if (i + 1) % 50 == 0:
            print(f"    hidden {i+1}/{len(prompts)}", flush=True)
    return np.stack(last_rows), np.stack(mean_rows), ntoks, emitted


@torch.no_grad()
def attn_stats(model, tok, prompts: List[str], device, max_len: int) -> Dict[str, List[float]]:
    """F-Attn (fraction of attention mass on token 0, averaged over heads and query
    positions > 0) and M-Act (mean max |hidden|) per layer. Needs eager attention."""
    f_attn: Optional[np.ndarray] = None
    m_act: Optional[np.ndarray] = None
    n = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**ids, output_attentions=True, output_hidden_states=True, use_cache=False)
        fa = np.array([float(a[0, :, 1:, 0].mean()) for a in out.attentions])
        ma = np.array([float(h[0].abs().max()) for h in out.hidden_states])
        f_attn = fa if f_attn is None else f_attn + fa
        m_act = ma if m_act is None else m_act + ma
        n += 1
        del out
        torch.cuda.empty_cache()
    return {"f_attn": (f_attn / max(n, 1)).tolist(), "m_act": (m_act / max(n, 1)).tolist(), "n": n}


# ------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-root", required=True, help="snapshot dir holding 1B_* subdirs")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--roots", nargs="+", required=True, help="text MMReD sample-dir roots")
    ap.add_argument("--limit", type=int, default=200, help="per-root sample cap")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-samples", type=int, default=6,
                    help="samples for the eager F-Attn/M-Act pass (smallest root only); 0 to skip")
    ap.add_argument("--attn-max-len", type=int, default=1536)
    ap.add_argument("--emit", action="store_true",
                    help="also measure what the model ACTUALLY EMITS (digit argmax + free "
                         "greedy digit decode), not just what a probe can decode from it")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data (identical order/split for every model) ----
    cells: Dict[str, Dict[str, Any]] = {}
    for r in args.roots:
        root = Path(r)
        dirs = iter_text_dirs(root)[: args.limit]
        recs = [s for s in (load_text_sample(d) for d in dirs) if s is not None]
        if not recs:
            print(f"[warn] no samples under {root}", flush=True)
            continue
        prompts = [build_text_prompt(st, q) for _, st, q, _ in recs]
        y = np.array([g for *_, g in recs], dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(recs))
        ntr = int(len(recs) * args.train_frac)
        cells[r] = {"prompts": prompts, "y": y, "tr": order[:ntr], "te": order[ntr:],
                    "sids": [s[0] for s in recs]}
        print(f"[data] {r}: n={len(recs)} train={ntr} test={len(recs)-ntr} "
              f"gold range {int(y.min())}..{int(y.max())} mean {y.mean():.2f}", flush=True)
    if not cells:
        raise SystemExit("no data")

    smallest = min(cells, key=lambda k: len(cells[k]["prompts"][0]))
    rows: List[Dict[str, Any]] = []
    emit_rows: List[Dict[str, Any]] = []
    sink: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    for var in args.variants:
        path = Path(args.model_root) / var
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to(dev).eval()
        cfg = model.config
        qw = model.state_dict()["model.layers.0.self_attn.q_proj.weight"].shape
        meta[var] = {
            "hidden": int(cfg.hidden_size), "layers": int(cfg.num_hidden_layers),
            "heads": int(cfg.num_attention_heads), "kv_heads": int(cfg.num_key_value_heads),
            "intermediate": int(cfg.intermediate_size),
            "headwise_gate": bool(cfg.headwise_attn_output_gate),
            "elementwise_gate": bool(cfg.elementwise_attn_output_gate),
            "q_proj_shape": list(qw),
            "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "attn_class": type(model.model.layers[0].self_attn).__name__,
        }
        print(f"[model] {var} {meta[var]} ({time.time()-t0:.0f}s)", flush=True)

        digit_ids = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
        for r, c in cells.items():
            H_last, H_mean, ntoks, emitted = extract_hiddens(
                model, tok, c["prompts"], dev, args.max_len,
                digit_ids=(digit_ids if args.emit else None))
            if args.emit:
                te = c["te"]
                y = c["y"]
                e1 = float(np.mean([emitted[i][0] == y[i] for i in te]))
                ef = float(np.mean([emitted[i][1] == y[i] for i in te]))
                nofmt = float(np.mean([emitted[i][1] is None for i in te]))
                emit_rows.append({"model": var, "root": r, "n_test": len(te),
                                  "emit_digit_argmax": e1, "emit_free": ef,
                                  "no_number_emitted": nofmt})
                print(f"  [{var} {r}] EMITTED digit-argmax {e1:.3f}  free-decode {ef:.3f} "
                      f"(no number: {nofmt:.3f})", flush=True)
            print(f"  [{var} {r}] tokens min/med/max "
                  f"{min(ntoks)}/{int(np.median(ntoks))}/{max(ntoks)}", flush=True)
            ctrl = majority_control(c["y"], c["tr"], c["te"])
            for pool, H in (("last", H_last), ("mean", H_mean)):
                for li in range(H.shape[1]):
                    res = ridge_round(H[:, li, :], c["y"], c["tr"], c["te"],
                                      alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6))
                    rows.append({"model": var, "root": r, "pool": pool, "layer": li,
                                 "n_train": len(c["tr"]), "n_test": len(c["te"]),
                                 "tok_med": int(np.median(ntoks)), **res, **ctrl})
            del H_last, H_mean

        if args.attn_samples > 0:
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                str(path), trust_remote_code=True, torch_dtype=torch.bfloat16,
                attn_implementation="eager").to(dev).eval()
            sink[var] = attn_stats(model, tok, cells[smallest]["prompts"][: args.attn_samples],
                                   dev, args.attn_max_len)
            print(f"  [{var}] F-Attn L0/mid/last "
                  f"{sink[var]['f_attn'][0]:.3f}/{sink[var]['f_attn'][len(sink[var]['f_attn'])//2]:.3f}/"
                  f"{sink[var]['f_attn'][-1]:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- write ----
    with (out / "probe_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if emit_rows:
        with (out / "emitted.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(emit_rows[0].keys()))
            w.writeheader(); w.writerows(emit_rows)
    (out / "sink_stats.json").write_text(json.dumps({"sink": sink, "meta": meta,
                                                     "attn_root": smallest}, indent=2))

    lines = [f"=== P0 TEXT TRIPLE (limit={args.limit} seed={args.seed} "
             f"train_frac={args.train_frac}) ===", ""]
    for var, m in meta.items():
        lines.append(f"{var}: {m}")
    lines.append("")
    for r in cells:
        y, te = cells[r]["y"], cells[r]["te"]
        lines.append(f"--- {r}  (n_test={len(te)}, gold mean {y.mean():.2f}, "
                     f"majority_exact {majority_control(y, cells[r]['tr'], te)['majority_exact']:.3f})")
        lines.append(f"{'pool':>5} {'layer':>5} " + " ".join(f"{v:>26}" for v in args.variants))
        for pool in ("last", "mean"):
            for li in sorted({x["layer"] for x in rows}):
                cellsr = []
                for var in args.variants:
                    m = [x for x in rows if x["model"] == var and x["root"] == r
                         and x["pool"] == pool and x["layer"] == li]
                    cellsr.append(f"ex {m[0]['exact']:.3f} +-1 {m[0]['within1']:.3f} R2 {m[0]['r2']:+.2f}"
                                  if m else "-")
                lines.append(f"{pool:>5} {li:>5} " + " ".join(f"{c:>26}" for c in cellsr))
        lines.append("")
    if sink:
        lines.append(f"--- F-Attn / M-Act per layer (eager, {args.attn_samples} samples from {smallest})")
        lines.append(f"{'layer':>5} " + " ".join(f"{v:>26}" for v in args.variants))
        nl = max(len(s["f_attn"]) for s in sink.values())
        for li in range(nl):
            cellsr = []
            for var in args.variants:
                s = sink.get(var)
                cellsr.append(f"F {s['f_attn'][li]:.4f}  M {s['m_act'][li]:8.1f}"
                              if s and li < len(s["f_attn"]) else "-")
            lines.append(f"{li:>5} " + " ".join(f"{c:>26}" for c in cellsr))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
