#!/usr/bin/env python3
"""P2h/P2i: END-TO-END TALLY PIPELINE — the constructive answer to the N-scaling walls.

Design (every stage justified by a measured campaign result):
  1. CHUNKED passes (k frames/forward, default 8): each chunk is an 8-frame joint pass with the
     full question, so per-frame carrier messages stay IN the N=8 training distribution at every
     N (kills the B2 threshold drift at its root; the chunk-size sweep quantifies the k-tax).
  2. MASS-NORMALIZED logistic gate trained ONCE on the existing N=8 joint cache ([2026-07-10i]:
     mass-norm removes magnitude dilution), with a PER-N CALIBRATED decision threshold fit on
     --calib-n labeled samples per N (N is known at inference; per-N calibration is legitimate).
  3. Tally = #frames above threshold.
  4. RENDER: the tally is written as the predicate-matched fact sentence pre-question (the only
     working token interface, C1b) and the FROZEN model verbalizes the answer (text-only render;
     C-range [2026-07-08e]: 0.99–1.00 over 0–40, C1 cf arms: the fact wins even against
     conflicting visual gold).

--mode retrieve (P2i): ONE joint pass over all N frames -> gate margins on (diluted) joint
messages -> shortlist frames above a HIGH-RECALL calibrated threshold -> isolated single-frame
look-again passes ("is C in R here? yes/no") ONLY on the shortlist -> tally = #yes. Cost scales
with evidence count, not N.

Eval: exact-match at each N vs the frozen joint model (B3), mp-sum (B1), law ceiling; cost in
forwards/sample. Outputs: results.csv, report.txt, rows.json.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import get_layers, image_token_groups
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)

L_READ = 16
OFFSET = 9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chunked", "retrieve", "adaptive", "twostage"],
                    default="chunked")
    ap.add_argument("--seq-lens", default="8,16,32,64,128")
    ap.add_argument("--chunk-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=150, help="eval samples per N")
    ap.add_argument("--calib-n", type=int, default=30, help="calibration samples per N")
    ap.add_argument("--train-cache",
                    default="", help="N=8 joint messages_cache.pt for gate training "
                                     "(default: newest valid under outputs/ladder/image_longN/joint/N8)")
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sample-seed", type=int, default=2, help="eval draw (2 = disjoint from "
                    "cache seed 0 and behavior seed 1)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))

    # ---- gate training (CPU, from the existing cache) ----
    from sklearn.linear_model import LogisticRegression
    tc = args.train_cache
    if not tc:
        for p in sorted(Path("outputs/ladder/image_longN/joint/N8").glob("*/count/messages_cache.pt")):
            if not (p.parent.parent / "CORRUPT.marker").exists():
                tc = str(p)
    c8 = torch.load(tc, map_location="cpu", weights_only=False)
    X8 = np.asarray(c8["msgs"][L_READ][OFFSET], dtype=np.float32)
    M8 = np.asarray(c8["mass"][L_READ][OFFSET], dtype=np.float32)
    X8 = X8 / np.clip(M8, 1e-6, None)[:, :, None]
    y8 = np.asarray(c8["labels"], dtype=int)
    gate = LogisticRegression(max_iter=2000, C=1.0)
    gate.fit(X8.reshape(-1, X8.shape[-1]), y8.reshape(-1))
    print(f"gate trained on {X8.shape[0]}x8 massnorm messages from {tc}", flush=True)

    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    layers = get_layers(model)
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    mrope_section = (getattr(cfg, "rope_scaling", None) or {}).get("mrope_section", None)
    attn_scale = head_dim ** -0.5
    tok = processor.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    import re
    INT_RE = re.compile(r"-?\d+")
    yes_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("yes", "Yes")]
    no_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("no", "No")]

    qkv: Dict[str, torch.Tensor] = {}
    posemb: Dict[str, torch.Tensor] = {}
    for nm in ("q_proj", "k_proj", "v_proj"):
        def mk(nm):
            def hook(_m, _i, o):
                qkv[nm] = o.detach()[0]
            return hook
        getattr(layers[L_READ].self_attn, nm).register_forward_hook(mk(nm))
    def mk_pe(_m, args_, kwargs_):
        pe = kwargs_.get("position_embeddings")
        if pe is None and len(args_) >= 1 and isinstance(args_[-1], tuple):
            pe = args_[-1]
        if pe is not None:
            posemb["cos"], posemb["sin"] = pe[0].detach(), pe[1].detach()
    layers[L_READ].self_attn.register_forward_pre_hook(mk_pe, with_kwargs=True)

    def chunk_margins(frames, q0):
        """Chunked passes -> per-frame gate margins (massnorm message @L16 off-9)."""
        K = int(args.chunk_k)
        margins = []
        n_fwd = 0
        for c0 in range(0, len(frames), K):
            frs = frames[c0:c0 + K]
            inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(frs, q0))
            with torch.no_grad():
                model(**inputs, use_cache=False)
            n_fwd += 1
            ids = inputs["input_ids"][0].detach().cpu()
            fg = image_token_groups(ids, expected_num_frames=len(frs), processor=processor)
            seq = int(ids.shape[0])
            last_img = max(int(p) for g in fg for p in g)
            qspan = list(range(last_img + 1, seq))
            carrier_t = torch.tensor(qspan, dtype=torch.long)
            ci = len(qspan) - 1 - OFFSET
            q = qkv["q_proj"].view(1, seq, n_heads, head_dim).transpose(1, 2)
            k = qkv["k_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
            v = qkv["v_proj"].view(1, seq, n_kv, head_dim).transpose(1, 2)
            cos, sin = posemb["cos"], posemb["sin"]
            if mrope_section is not None:
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
            k = repeat_kv(k, n_heads // n_kv); v = repeat_kv(v, n_heads // n_kv)
            qf = q[0].float().cpu(); kf = k[0].float().cpu()
            scores = torch.einsum("hcd,hkd->hck", qf[:, carrier_t], kf) * attn_scale
            allow = torch.arange(seq)[None, :] <= carrier_t[:, None]
            scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
            A = torch.softmax(scores, dim=-1)
            vf = v[0].float().cpu()
            oproj = layers[L_READ].self_attn.o_proj
            odev = next(oproj.parameters()).device
            for grp in fg:
                pos = torch.tensor(sorted(int(p) for p in grp), dtype=torch.long)
                Asel = A[:, ci:ci + 1, pos]
                mass = float(Asel.sum(-1).mean())
                ctx = torch.einsum("hcj,hjd->hcd", Asel, vf[:, pos, :])
                ctx = ctx.permute(1, 0, 2).reshape(1, -1)
                with torch.no_grad():
                    mm = oproj(ctx.to(device=odev, dtype=torch.bfloat16)).float().cpu().numpy()[0]
                feat = mm / max(mass, 1e-6)
                margins.append(float(gate.decision_function(feat[None])[0]))
        return np.array(margins), n_fwd

    def lookagain(frame, C, R):
        prompt = (f"Look at this single frame.\nIs {C} in the {R} in this frame? "
                  f"Answer yes or no.\nAnswer: ")
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([frame], prompt))
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
        py = torch.logsumexp(logits[yes_ids], 0)
        pn = torch.logsumexp(logits[no_ids], 0)
        return torch.sigmoid(py - pn).item()

    def render_answer(C, R, tally, q0):
        prompt = (f"Note: {C} spent exactly {tally} steps in the {R}.\n"
                  f"Respond with a single integer. Output only the integer.\n"
                  f"Question: {q0}\nAnswer: ")
        inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([], prompt))
        plen = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=5, pad_token_id=pad)
        dec = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)[0]
        m = INT_RE.search(dec)
        return int(m.group(0)) if m else None

    def load_pool(N):
        root = ("data/mmred_images_park/seq_len_8/all_uniform" if N == 8
                else f"data/mmred_longN_park/seq_len_{N}/all_uniform")
        dirs = list(iter_sample_dirs(Path(root)))
        random.Random(args.sample_seed).shuffle(dirs)
        return dirs

    rows_csv = ["mode,N,n,exact,mae,fwd_per_sample,threshold"]
    all_rows = []
    lines = [f"=== E2E TALLY PIPELINE mode={args.mode} k={args.chunk_k} gate=massnorm-L16-off9 ==="]
    for N in [int(x) for x in args.seq_lens.replace(",", " ").split()]:
        dirs = load_pool(N)
        calib_dirs = dirs[: args.calib_n]
        eval_dirs = dirs[args.calib_n: args.calib_n + args.limit]

        def sample_fields(sd):
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            m = re.search(r"did (\w+) spend in the (\w+)", q0)
            C, R = m.group(1), m.group(2)
            if int(args.resize) > 0:
                frames = [f.resize((int(args.resize), int(args.resize))) for f in frames]
            return sid, frames, q0, gold, evid, C, R

        # ---- calibrate threshold on calib set ----
        cal_margins, cal_labels = [], []
        cal_joint_margins = []
        for sd in calib_dirs:
            try:
                sid, frames, q0, gold, evid, C, R = sample_fields(sd)
            except Exception:
                continue
            if args.mode == "twostage":
                _saved_k = args.chunk_k
                args.chunk_k = len(frames)
                jmg, _ = chunk_margins(frames, q0)      # joint-pass margins
                args.chunk_k = _saved_k
                cal_joint_margins.append(jmg)
            mg, _ = chunk_margins(frames, q0)
            cal_margins.append(mg)
            cal_labels.append(np.array([1 if t in evid else 0 for t in range(len(mg))]))
        if args.mode == "adaptive":
            best_t = float("nan")
            cal_margins = cal_margins  # unused; margin ORDER + stability stop replace the threshold
        cm = np.concatenate(cal_margins); clab = np.concatenate(cal_labels)
        cand_t = np.quantile(cm, np.linspace(0.02, 0.98, 49))
        joint_t = float("nan")
        if args.mode == "twostage":       # stage-1 joint prefilter: FN<=1% recall-safe
            jm = np.concatenate(cal_joint_margins); jl = np.concatenate(cal_labels)
            joint_t = float(np.quantile(jm, 0.02))
            for t in sorted(np.quantile(jm, np.linspace(0.02, 0.98, 49))):
                fn = float((jm[jl == 1] <= t).mean()) if (jl == 1).any() else 0.0
                if fn <= 0.01:
                    joint_t = float(t)
                else:
                    break
            # stage-2 threshold on k=2 margins: FN<=2% (as retrieve)
            best_t = float(np.quantile(cm2 := np.concatenate(cal_margins), 0.02))
            cl2 = np.concatenate(cal_labels)
            for t in sorted(np.quantile(cm2, np.linspace(0.02, 0.98, 49))):
                fn = float((cm2[cl2 == 1] <= t).mean()) if (cl2 == 1).any() else 0.0
                if fn <= 0.02:
                    best_t = float(t)
                else:
                    break
        elif args.mode == "adaptive":
            best_t = float("nan")         # no threshold: margin order + stability stop
        elif args.mode == "chunked":
            best_t, best_err = 0.0, 1e9   # minimize |tally-gold| MAE on calib samples
            for t in cand_t:
                errs = [abs(int((mg > t).sum()) - int(lab.sum()))
                        for mg, lab in zip(cal_margins, cal_labels)]
                e = float(np.mean(errs))
                if e < best_err:
                    best_err, best_t = e, float(t)
        else:   # retrieve: high-recall shortlist threshold (FN <= 2%)
            best_t = float(cand_t[0])
            for t in sorted(cand_t):
                fn = float((cm[clab == 1] <= t).mean()) if (clab == 1).any() else 0.0
                if fn <= 0.02:
                    best_t = float(t)
                else:
                    break
        print(f"[N={N}] threshold {best_t:+.3f} (calib {len(cal_margins)} samples)", flush=True)

        n_ok, exact, maes, fwds = 0, [], [], []
        for sd in eval_dirs:
            try:
                sid, frames, q0, gold, evid, C, R = sample_fields(sd)
            except Exception:
                continue
            t0 = time.time()
            if args.mode == "chunked":
                mg, n_fwd = chunk_margins(frames, q0)
                tally = int((mg > best_t).sum())
            elif args.mode == "twostage":
                _saved_k = args.chunk_k
                args.chunk_k = len(frames)
                jmg, n_fwd = chunk_margins(frames, q0)          # 1 joint pass
                args.chunk_k = _saved_k
                surv = [t for t in range(len(jmg)) if jmg[t] > joint_t]
                # k=2 chunks over survivors only
                s_frames = [frames[t] for t in surv]
                if s_frames:
                    mg2, nf2 = chunk_margins(s_frames, q0)
                    n_fwd += nf2
                else:
                    mg2 = np.array([])
                votes = 0
                for i, t in enumerate(surv):
                    if mg2[i] > best_t:
                        votes += int(lookagain(frames[t], C, R) > 0.5)
                        n_fwd += 1
                tally = votes
            elif args.mode == "adaptive":
                mg, n_fwd = chunk_margins(frames, q0)
                order = np.argsort(-mg)
                window = int(np.ceil(0.15 * len(mg)))
                votes, consec_no = 0, 0
                for t in order:
                    yes = lookagain(frames[t], C, R) > 0.5
                    n_fwd += 1
                    if yes:
                        votes += 1; consec_no = 0
                    else:
                        consec_no += 1
                        if consec_no >= window:
                            break
                tally = votes
            else:
                mg, n_fwd = chunk_margins(frames, q0)   # retrieve uses ONE joint pass:
                # (chunk_k should be set = N at submit time for true retrieve; n_fwd counts it)
                short = [t for t in range(len(mg)) if mg[t] > best_t]
                votes = 0
                for t in short:
                    votes += int(lookagain(frames[t], C, R) > 0.5)
                    n_fwd += 1
                tally = votes
            pred = render_answer(C, R, tally, q0)
            n_fwd += 1
            all_rows.append({"mode": args.mode, "N": N, "sid": sid, "gold": gold,
                             "tally": tally, "pred": pred, "n_fwd": n_fwd,
                             "sec": round(time.time() - t0, 2)})
            exact.append(int(pred == gold))
            maes.append(abs((pred if pred is not None else -99) - gold))
            fwds.append(n_fwd)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  [N={N}] {n_ok}: exact {np.mean(exact):.3f}", flush=True)
                (out / "rows.json").write_text(json.dumps(all_rows, indent=1))
        ex, mae, fw = float(np.mean(exact)), float(np.mean(maes)), float(np.mean(fwds))
        lines.append(f"  N={N:<4d} n={n_ok:<4d} exact={ex:.3f}  MAE={mae:.2f}  "
                     f"forwards/sample={fw:.1f}  thr={best_t:+.3f}")
        rows_csv.append(f"{args.mode},{N},{n_ok},{ex:.4f},{mae:.4f},{fw:.2f},{best_t:.4f}")
        (out / "results.csv").write_text("\n".join(rows_csv) + "\n")
        (out / "report.txt").write_text("\n".join(lines) + "\n")

    (out / "rows.json").write_text(json.dumps(all_rows, indent=1))
    print("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
