#!/usr/bin/env python3
"""Two causal tests of the "softmax normalizer (sum_j alpha_j = 1) is the B->C bottleneck" hypothesis.

The carrier aggregates frame messages as  out = sum_j alpha_j v_j  with sum_j alpha_j = 1 (a MEAN).
A mean encodes the count k only as the FRACTION k/N, so adjacent counts are spaced ~||Delta||/N apart
-> they crowd as N grows. Two tests, both on the frozen model, frames-first layout so question
(carrier) tokens sit AFTER the frames and can attend to them:

TEST 1 - DILUTION LAW (no intervention).
  Record (gold k, seq_len N, predicted count). If aggregation is a mean, the model's sensitivity to the
  true count -- slope d(emitted)/d(gold) within a fixed N -- should fall ~1/N (counts squash toward a
  common middle at high N). A truly extensive process would keep the slope ~1 (flat in N).

TEST 2 - GAIN vs TEMPERATURE on the [question-query -> frame-key] attention sub-block (late band).
  * TEMP  : multiply that logit sub-block by beta BEFORE softmax. Changes WHICH frames get weight;
            still sums to 1 -> still a mean. (== attn_temp_frame_columns.py)
  * GAIN  : multiply that weight sub-block by g AFTER softmax. Keeps the relative frame weights but
            breaks sum=1 -> the frame block becomes EXTENSIVE (g=N turns the mean back into a sum).
  Prediction if sum=1 is the cause: GAIN recovers high-N accuracy where TEMP can't, and the best g
  scales with N. If GAIN <= TEMP, the normalizer is NOT the bottleneck (-> readout/superposition).

Readout = canonical greedy generation + parse first integer (base.build_prompt / extract_first_integer),
the same harness that produced the documented 85%->20% baseline curve.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch
import torch.nn as nn
import experiments.glstm.frame_axis_aggregator_adapter as fa
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from models.model import get_layers, image_token_groups
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as mq
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ORIG_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]  # captured before override; used to faithfully run vision attn

# mode in {"off","temp","gain"}; mult is beta (temp) or g (gain); q_pos/k_pos are 1-D LongTensors
STATE = {"mode": "off", "mult": 1.0, "lo": 0, "hi": 999, "q_pos": None, "k_pos": None,
         "instrument": False}
ATTN_BUF = []  # per-call carrier->frame attention stats during one generate() prefill (mechanism C/D)


def custom_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    # CRITICAL: this custom fn is dispatched globally, incl. the VISION tower (windowed attention with no
    # layer_idx). Only LM decoder layers have layer_idx -> for everything else, faithfully delegate to the
    # original sdpa implementation so image features are never corrupted.
    li = getattr(module, "layer_idx", None)
    if li is None:  # vision tower: faithfully delegate to sdpa
        kwargs.pop("dropout", None); kwargs.pop("scaling", None)
        return ORIG_SDPA(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    # Decide whether THIS LM call needs our manual eager path (intervention or instrumentation).
    q_len, k_len = query.shape[2], key.shape[2]
    in_band = (STATE["lo"] <= li < STATE["hi"] and STATE["q_pos"] is not None and q_len == k_len)
    active = (STATE["mode"] != "off" and STATE["mult"] != 1.0 and in_band)
    do_instr = bool(STATE.get("instrument")) and in_band and q_len > 1
    if not (active or do_instr):
        # baseline / out-of-band / decode steps: delegate to sdpa (faithful causal, exactly == --no-custom)
        kwargs.pop("dropout", None); kwargs.pop("scaling", None)
        return ORIG_SDPA(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    # ---- manual eager path (needed to touch post-softmax weights) ----
    # Do the math in float32 to MATCH sdpa's internal fp32 accumulation; bf16 QK^T diverges ~0.04/layer
    # which, compounded over the band, derails generation.
    key_states = mq.repeat_kv(key, module.num_key_value_groups).float()
    value_states = mq.repeat_kv(value, module.num_key_value_groups).float()
    attn_weights = torch.matmul(query.float(), key_states.transpose(2, 3)) * scaling
    qp, kp = STATE["q_pos"], STATE["k_pos"]
    have_pos = len(qp) > 0 and len(kp) > 0
    if active and STATE["mode"] == "temp" and have_pos:
        attn_weights[:, :, qp.unsqueeze(1), kp.unsqueeze(0)] *= STATE["mult"]
    # additive causal mask (faithful to eager_attention_forward). q_len==k_len here (in_band).
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :k_len].float()
    elif q_len > 1:
        causal = torch.triu(torch.ones(q_len, k_len, dtype=torch.bool, device=attn_weights.device),
                            diagonal=1)
        attn_weights = attn_weights.masked_fill(causal, float("-inf"))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    if do_instr and have_pos:  # mechanism C/D: native carrier->frame attention (post-softmax, pre-gain)
        try:
            w = attn_weights[0][:, qp][:, :, kp]   # [H,|q|,|frames|]
            fm = w.sum(-1)
            wn = w / (fm.unsqueeze(-1) + 1e-9)
            ent = -(wn * (wn + 1e-9).log()).sum(-1)
            ATTN_BUF.append((float(ent.mean()), float(fm.mean()),
                             float(w.max(-1).values.mean()), int(len(kp))))
        except Exception:
            pass
    if active and STATE["mode"] == "gain" and have_pos:
        attn_weights[:, :, qp.unsqueeze(1), kp.unsqueeze(0)] *= STATE["mult"]  # breaks sum_j alpha_j = 1
    attn_output = torch.matmul(attn_weights, value_states).to(query.dtype).transpose(1, 2).contiguous()
    if STATE.get("dbg", 0) > 0:  # numerical self-check: manual (mult=1 effect aside) vs sdpa baseline
        try:
            kwargs.pop("dropout", None); kwargs.pop("scaling", None)
            ref = ORIG_SDPA(module, query, key, value, attention_mask, scaling=scaling, dropout=0.0)[0]
            print(f"[selfcheck-LM] layer={li} q={q_len} mode={STATE['mode']} mult={STATE['mult']} "
                  f"max|manual-sdpa|={float((attn_output - ref).abs().max()):.4f} "
                  f"(0 => manual path faithful when no intervention)", flush=True)
        except Exception as e:
            print(f"[selfcheck-LM] err {e}", flush=True)
        STATE["dbg"] -= 1
    return attn_output, attn_weights


def build_inputs_frames_first(processor, frames, question, device):
    # EXACT canonical prompt (base.build_prompt) that produced the documented 85%->20% baseline curve.
    # Frames FIRST so question carriers attend to them (needed for the frame->carrier intervention).
    prompt = base.build_prompt(question, num_frames=len(frames))
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    return base.move_inputs_to_device(dict(inputs), device)


def frame_and_question_positions(processor, input_ids, n_frames):
    ids_cpu = input_ids[0].detach().cpu()
    spans = image_token_groups(ids_cpu, n_frames, processor=processor)
    if len(spans) != n_frames:
        return None, None
    frame_pos = sorted(int(i) for sp in spans for i in sp)
    last_img = max(frame_pos)
    special = set(processor.tokenizer.all_special_ids)
    seq_len = int(input_ids.shape[1])
    q_pos = [i for i in range(last_img + 1, seq_len) if int(ids_cpu[i]) not in special]
    return frame_pos, q_pos


def predict_count(model, inputs, processor, max_new_tokens=6):
    # canonical readout: greedy-generate a few tokens, parse the first integer (base.extract_first_integer).
    # The frame->carrier intervention applies during PREFILL (square attn); decode steps are skipped.
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out_ids[:, inputs["input_ids"].shape[-1]:]
    text = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
    pred = base.extract_first_integer(text)
    if STATE.get("dbg_txt", 0) > 0:
        print(f"[txt] raw={text!r} -> pred={pred}", flush=True)
        STATE["dbg_txt"] -= 1
    return pred if pred is not None else -1


def sweep(model, samples, processor, mode, mult, lo, hi, per_sample_gain_is_N=False):
    """Run one (mode,mult) setting over all samples; return overall acc + per-seqlen acc + preds list."""
    STATE["mode"], STATE["lo"], STATE["hi"] = mode, lo, hi
    correct = 0
    by = defaultdict(lambda: [0, 0])
    preds = []  # (gold, sl, pred)
    attn_by_sl = defaultdict(list)  # sl -> list of (entropy, frame_mass, maxw) for mechanism C/D
    instrument = (mode == "off") and bool(STATE.get("instr_enabled"))
    t0 = time.time()
    for inputs, gold, sl, qp, kp in samples:
        STATE["q_pos"], STATE["k_pos"] = qp, kp
        STATE["mult"] = float(sl) if per_sample_gain_is_N else mult
        if instrument:
            ATTN_BUF.clear()
            STATE["instrument"] = True
        pred = predict_count(model, inputs, processor)
        STATE["instrument"] = False
        if instrument and ATTN_BUF:
            ent = float(np.mean([b[0] for b in ATTN_BUF]))
            fm = float(np.mean([b[1] for b in ATTN_BUF]))
            mw = float(np.mean([b[2] for b in ATTN_BUF]))
            attn_by_sl[sl].append((ent, fm, mw))
        preds.append((gold, sl, pred))
        ok = int(pred == gold)
        correct += ok
        by[sl][0] += ok
        by[sl][1] += 1
    STATE["mode"], STATE["mult"] = "off", 1.0
    acc = correct / max(1, len(samples))
    by_sl = {sl: by[sl][0] / by[sl][1] for sl in sorted(by)}
    return {"overall": acc, "by_sl": by_sl, "preds": preds, "secs": time.time() - t0,
            "attn_by_sl": {sl: attn_by_sl[sl] for sl in sorted(attn_by_sl)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="steps_in_room")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mmred_images_park")
    ap.add_argument("--per-seqlen", type=int, default=30)
    ap.add_argument("--seq-lens", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--mults", default="2,4,8")  # beta (temp) and g (gain)
    ap.add_argument("--band", default="late")    # late = L14+ ; all = whole stack
    ap.add_argument("--no-custom", action="store_true",
                    help="diagnostic: skip registering custom attention (pure sdpa) to isolate the bug")
    ap.add_argument("--instrument", action="store_true",
                    help="enable mechanism C/D dispersion measurement on the baseline pass")
    ap.add_argument("--only", default="all", choices=["all", "dilution", "temp", "gain"],
                    help="run one test on its own GPU; baseline always computed for the comparison")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "outputs/frame_axis/probes/denom_gain_vs_temp")
    args = ap.parse_args()
    run_dir = args.output / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = base.resolve_device("cuda"); dtype = base.resolve_dtype("bfloat16", device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    mults = [float(x) for x in args.mults.split(",")]

    print(f"loading {args.model_name} ...", flush=True)
    model, processor = base.load_model_and_processor(args.model_name, device, dtype, True)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = get_layers(model)
    n_layers = len(layers)
    band = {"all": (0, n_layers), "late": (14, n_layers)}[args.band]
    if not args.no_custom:
        ALL_ATTENTION_FUNCTIONS["custom_gt"] = custom_attention
        model.config._attn_implementation = "custom_gt"
        for l in layers:
            l.self_attn.config._attn_implementation = "custom_gt"
    print(f"n_layers={n_layers} band={args.band}{band} no_custom={args.no_custom} "
          f"readout=generate+extract_first_integer", flush=True)

    import random
    rng = random.Random(0)
    samples = []
    for sl in seq_lens:
        splits = fa.declare_splits(args.data_root, "all_uniform", [sl], [], 0.0, 0.0, 0, None, 12345)
        dirs = [d for d, _ in splits["train"]]; rng.shuffle(dirs)
        got = 0
        for dstr in dirs:
            ex = fa.make_example(Path(dstr), args.task, rng, eval_mode=True)
            if ex is None:
                continue
            frames, question, gold, nf, states = ex
            try:
                inputs = build_inputs_frames_first(processor, frames, question, device)
            except Exception:
                continue
            fp, qp = frame_and_question_positions(processor, inputs["input_ids"], nf)
            if fp is None or not qp:
                continue
            samples.append((inputs, int(gold), int(nf),
                            torch.tensor(qp, device=device), torch.tensor(fp, device=device)))
            got += 1
            if got >= args.per_seqlen:
                break
    print(f"collected {len(samples)} samples; e.g. n_qtok={len(samples[0][3])} "
          f"n_frametok={len(samples[0][4])}", flush=True)

    report = []

    def emit(line):
        print(line, flush=True)
        report.append(line)

    # baseline is always needed (dilution analysis + the Test-2 comparison row)
    emit(f"\n########## BASELINE (mode=off)   only={args.only} ##########")
    STATE["instr_enabled"] = bool(args.instrument)
    STATE["dbg"] = 6  # self-check prints for the first few manual LM calls
    STATE["dbg_txt"] = 6  # print raw generated text for first few samples
    base_res = sweep(model, samples, processor, "off", 1.0, *band)
    emit(f"baseline overall acc={base_res['overall']:.3f}  ({base_res['secs']:.0f}s)")
    emit("baseline acc by seq_len: " + "  ".join(f"sl{sl}:{base_res['by_sl'].get(sl, float('nan')):.2f}"
                                                  for sl in seq_lens))
    preds = base_res["preds"]
    # quick degeneracy guard: if the model emits a near-constant value, the readout is broken, not a finding
    allp = [p for _, _, p in preds]
    emit(f"emitted-value distribution (sanity): {dict(sorted(__import__('collections').Counter(allp).items()))}")
    # mean emitted by (gold, N)
    cell = defaultdict(list)
    for g, sl, p in preds:
        cell[(g, sl)].append(p)
    emit("\nmean EMITTED count by (gold k row, seq_len N col)  [blank = no samples]:")
    golds = sorted({g for g, _ in cell})
    emit("k\\N  " + "  ".join(f"{sl:>4d}" for sl in seq_lens))
    for g in golds:
        row = []
        for sl in seq_lens:
            v = cell.get((g, sl))
            row.append(f"{np.mean(v):>4.1f}" if v else "   .")
        emit(f"{g:>2d}   " + "  ".join(row))
    # sensitivity slope d(emitted)/d(gold) within each N  -> predicted to fall ~1/N
    emit("\nsensitivity slope d(emitted)/d(gold) per seq_len N  (sum=1 mean => ~1/N; extensive => ~1):")
    slope_row = []
    for sl in seq_lens:
        xs = np.array([g for g, s, p in preds if s == sl], dtype=float)
        ys = np.array([p for g, s, p in preds if s == sl], dtype=float)
        if len(xs) >= 3 and xs.std() > 1e-6:
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = float("nan")
        slope_row.append((sl, slope))
    emit("N      " + "  ".join(f"{sl:>5d}" for sl, _ in slope_row))
    emit("slope  " + "  ".join(f"{s:>5.2f}" for _, s in slope_row))

    # ---- mechanism C/D: native carrier->frame attention dispersion & mass vs N (from baseline pass) ----
    attn_by_sl = base_res.get("attn_by_sl", {})
    disp = {}
    if attn_by_sl:
        emit("\nnative carrier->frame attention vs N (band {}):  entropy over frames, normalized entropy"
             " (ent/ln N), total frame-mass, max single-frame weight".format(band))
        emit("N        " + "  ".join(f"{sl:>6d}" for sl in seq_lens))
        rows = {"entropy": [], "norm_ent": [], "frame_mass": [], "max_w": []}
        for sl in seq_lens:
            vals = attn_by_sl.get(sl, [])
            if vals:
                e = float(np.mean([v[0] for v in vals])); fm = float(np.mean([v[1] for v in vals]))
                mw = float(np.mean([v[2] for v in vals]))
                ne = e / np.log(sl) if sl > 1 else float("nan")
            else:
                e = fm = mw = ne = float("nan")
            rows["entropy"].append(e); rows["norm_ent"].append(ne)
            rows["frame_mass"].append(fm); rows["max_w"].append(mw)
        for k, lab in (("entropy", "entropy "), ("norm_ent", "norm_ent"),
                       ("frame_mass", "fr_mass "), ("max_w", "max_w   ")):
            emit(f"{lab} " + "  ".join(f"{x:>6.3f}" for x in rows[k]))
        disp = {sl: {"entropy": rows["entropy"][i], "frame_mass": rows["frame_mass"][i],
                     "max_w": rows["max_w"][i]} for i, sl in enumerate(seq_lens)}

    # ---------------- TEST 2: gain vs temperature ----------------
    results = {("off", 1.0): base_res}
    run_temp = args.only in ("all", "temp")
    run_gain = args.only in ("all", "gain")
    rN = None
    if run_temp or run_gain:
        emit(f"\n########## TEST 2 - GAIN vs TEMPERATURE  (band={args.band}{band}) ##########")
    for mode in (["temp"] if run_temp else []) + (["gain"] if run_gain else []):
        for m in mults:
            r = sweep(model, samples, processor, mode, m, *band)
            results[(mode, m)] = r
            emit(f"[{mode:4s} mult={m:.1f}] overall={r['overall']:.3f}  "
                 f"by_sl={{ {', '.join(f'{k}:{v:.2f}' for k, v in r['by_sl'].items())} }}  "
                 f"({r['secs']:.0f}s)")
    if run_gain:  # per-sample g = N (gain only)
        rN = sweep(model, samples, processor, "gain", 1.0, *band, per_sample_gain_is_N=True)
        results[("gain", "N")] = rN
        emit(f"[gain mult=N ] overall={rN['overall']:.3f}  "
             f"by_sl={{ {', '.join(f'{k}:{v:.2f}' for k, v in rN['by_sl'].items())} }}  ({rN['secs']:.0f}s)")

    emit("\n===== acc by seq_len: baseline vs temp vs gain (the headline) =====")
    emit("setting        " + "  ".join(f"sl{sl}" for sl in seq_lens) + "   overall")
    def fmt(tag, r):
        return (f"{tag:14s} " + "  ".join(f"{r['by_sl'].get(sl, float('nan')):.2f}" for sl in seq_lens)
                + f"   {r['overall']:.3f}")
    emit(fmt("baseline", base_res))
    if run_temp:
        for m in mults:
            emit(fmt(f"temp x{m:.0f}", results[("temp", m)]))
    if run_gain:
        for m in mults:
            emit(fmt(f"gain x{m:.0f}", results[("gain", m)]))
        emit(fmt("gain x=N", rN))

    (run_dir / "report.txt").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "task": args.task, "band": args.band, "n_samples": len(samples),
        "baseline": {"overall": base_res["overall"], "by_sl": base_res["by_sl"]},
        "dilution_slope": {sl: s for sl, s in slope_row},
        "dispersion": disp,
        "test2": {f"{mode}:{m}": {"overall": r["overall"], "by_sl": r["by_sl"]}
                  for (mode, m), r in results.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {run_dir}/report.txt and summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
