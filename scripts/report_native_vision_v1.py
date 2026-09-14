"""Report the registered seven-arm V1 vision comparison on CPU Slurm.

Uses saved native predictions only. Paired bootstrap intervals condition on the
selected examples and one training seed; they do not establish seed replication.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import hashlib
import os
from pathlib import Path
from statistics import mean


ARMS = ("sum", "mean", "post_sum", "post_mean", "global", "hierarchical", "lora")
LABELS = {"sum": "Before / sum", "mean": "Before / mean", "post_sum": "After / sum",
          "post_mean": "After / mean", "global": "Global read", "hierarchical": "Hierarchical",
          "lora": "LoRA rank 16"}


def summarize(rows):
    parsed = [r for r in rows if r["prediction"] is not None]
    return dict(n=len(rows), correct=sum(r["exact"] for r in rows),
                exact=mean(r["exact"] for r in rows),
                parse_rate=len(parsed) / len(rows),
                mae_parsed=mean(abs(r["prediction"]-r["gold"]) for r in parsed) if parsed else None,
                bias_parsed=mean(r["prediction"]-r["gold"] for r in parsed) if parsed else None,
                first_token_nll=mean(r["gold_first_token_nll"] for r in rows),
                model_seconds=mean(r["model_seconds"] for r in rows),
                total_seconds=mean(r["model_seconds"] + r["preprocessing_seconds"] for r in rows))


def compare(candidate, control, ns, np, rng):
    def keyed(rows):
        rows = [r for r in rows if r["n_frames"] in ns]
        result = {(r["n_frames"], r["sid"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate sample keys")
        return result
    a, b = keyed(candidate), keyed(control)
    if a.keys() != b.keys() or not a:
        raise ValueError("Paired comparison requires identical nonempty sample sets")
    groups = defaultdict(list)
    wins = losses = both_correct = both_wrong = 0
    for key in sorted(a):
        if a[key]["gold"] != b[key]["gold"] or a[key]["path"] != b[key]["path"]:
            raise ValueError("Paired example identity/gold differs")
        x, y = bool(a[key]["exact"]), bool(b[key]["exact"])
        wins += x and not y
        losses += y and not x
        both_correct += x and y
        both_wrong += not x and not y
        groups[key[0], a[key]["gold"]].append(int(x)-int(y))
    bootstrap = np.zeros(10000, dtype=np.float64)
    for values in groups.values():
        values = np.array(values, dtype=np.float64)
        sampled = values[rng.integers(0, len(values), size=(10000, len(values)))]
        bootstrap += sampled.sum(axis=1)
    bootstrap /= len(a)
    aa, bb = summarize(list(a.values())), summarize(list(b.values()))
    return dict(n=len(a), exact_gain=aa["exact"]-bb["exact"],
                bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
                candidate_wins=wins, control_wins=losses, both_correct=both_correct,
                both_wrong=both_wrong,
                first_token_nll_difference=aa["first_token_nll"]-bb["first_token_nll"],
                model_latency_ratio=aa["model_seconds"]/bb["model_seconds"],
                total_latency_ratio=aa["total_seconds"]/bb["total_seconds"])



def validate_predictions(predictions, config, manifest):
    expected = {(n, r["sid"]): (r["path"], r["gold"])
                for n in (8, 16, 32, 64)
                for r in manifest["splits"][f"test_N{n}"]["samples"]}
    actual = {(r["n_frames"], r["sid"]): (r["path"], r["gold"])
              for r in predictions}
    if len(predictions) != 400 or len(actual) != 400 or actual != expected:
        raise ValueError("Prediction identities differ from the full manifest test set")
    if any(r["tag"] != "test" or r["mode"] != "all"
           or bool(r["exact"]) != (r["prediction"] == r["gold"]) for r in predictions):
        raise ValueError("Invalid prediction tag, mode, or exact flag")
    shared = dict(model="Qwen/Qwen2.5-VL-7B-Instruct", resize=392,
                  eval_ns=[8,16,32,64], limit_eval=100, max_new_tokens=4,
                  max_seq_tokens=16000, seed=0, data_seed=20260910)
    if any(config[k] != v for k, v in shared.items()):
        raise ValueError("Evaluation configuration differs from V1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/native_aggregation_vlm/v1"))
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit this report through CPU Slurm")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells, rows, runs = {}, {}, {}
    manifest_hashes = set()
    source_hashes = json.loads((a.root / "main" / "source_hashes.json").read_text())
    manifest_path = Path("/mnt/data/gabriele/gnn_transformer/native_aggregation_vision_pilot/main_manifest.json")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    expected_manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    for arm in ARMS:
        paths = sorted((a.root / "main" / arm).glob("*/summary.json"))
        if len(paths) != 1:
            raise ValueError(f"Need one canonical completed run for {arm}, found {paths}")
        path, = paths
        summary = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        predictions = json.loads((path.parent / "predictions.json").read_text())
        if config["seed"] != 0 or config["epochs"] != 9 or len(summary["training"]) != 9:
            raise ValueError(f"Incomplete or wrong training protocol: {path}")
        if config["arm"] != arm or any(r["mode"] != "all" for r in predictions):
            raise ValueError(f"Unexpected arm/mode: {path}")
        validate_predictions(predictions, config, manifest)
        training_config = dict(train_ns=[8,16], dev_ns=[8,16], limit_train=90, limit_dev=36,
                               layer_index=14, block_size=64, rank=64, routing_rank=8,
                               query_chunk_size=32, lora_layers=4, lr=.001, lr_lora=.0001,
                               accumulation=4, max_grad_norm=1., center_messages=False,
                               max_steps=0, checkpoint=None, eval_only=False,
                               lora_rank=16 if arm == "lora" else 8,
                               lora_alpha=32. if arm == "lora" else 16.)
        if any(config[k] != v for k, v in training_config.items()):
            raise ValueError(f"Training configuration differs for {arm}")
        if any(config["code_sha256"][k] != source_hashes[k] for k in
               ("scripts/native_aggregation_vlm.py", "scripts/native_aggregation.py", "gnnformer/native_aggregation.py")):
            raise ValueError(f"Training source changed for {arm}")
        if any(e["n"] != 180 or [m["n"] for m in e["dev"]] != [36,36]
               for e in summary["training"]):
            raise ValueError(f"Training/dev denominator differs for {arm}")
        manifest_hashes.add(config["manifest_sha256"])
        rows[arm], runs[arm] = predictions, str(path.parent)
        per_n = {}
        for n in (8, 16, 32, 64):
            selected = [r for r in predictions if r["n_frames"] == n]
            if len(selected) != 100:
                raise ValueError(f"Need 100 examples at N{n}: {path}")
            per_n[n] = summarize(selected)
        logged_gradients = []
        for line in (path.parent / "log.txt").read_text().splitlines():
            try:
                item = ast.literal_eval(line.split("] ", 1)[-1])
            except (ValueError, SyntaxError):
                continue
            if isinstance(item, dict) and "grad_norm" in item:
                logged_gradients.append(item)
        cells[arm] = dict(
                          descriptive_training_diagnostics={
                              "scope": "Post-hoc description; logged steps only; norms before joint clipping",
                              "first_logged_gradient": logged_gradients[0] if logged_gradients else None,
                              "n_logged_steps": len(logged_gradients),
                              "fraction_logged_steps_clipped": mean(x["grad_norm"] > 1 for x in logged_gradients) if logged_gradients else None},
                          by_n_gold={f"N{n}_K{k}": summarize([r for r in predictions if r["n_frames"] == n and r["gold"] == k])
                                     for n in (8,16,32,64) for k in range(9)},
                          selected_epoch=max(summary["training"], key=lambda e: (
                              sum(m["correct"] for m in e["dev"])/sum(m["n"] for m in e["dev"]),
                              -sum(m["gold_first_token_nll"]*m["n"] for m in e["dev"])/sum(m["n"] for m in e["dev"]))) ["epoch"],
                          parameters=summary["parameters"], selected_checkpoint=summary["selected_checkpoint"],
                          in_range=summarize([r for r in predictions if r["n_frames"] in (8, 16)]),
                          out_of_range=summarize([r for r in predictions if r["n_frames"] in (32, 64)]),
                          per_n=per_n, training=summary["training"],
                          peak_training_bytes=max(e["peak_cuda_allocated_bytes"] for e in summary["training"]),
                          peak_eval_bytes=max(e["peak_cuda_allocated_bytes"] for e in summary["results"]))
    if manifest_hashes != {expected_manifest_hash}:
        raise ValueError("Arms did not use an identical staged manifest")
    if len({cells[x]["parameters"]["total_trainable"] for x in ARMS[:4]}) != 1:
        raise ValueError("Processing-order factorial does not match parameters")
    base_paths = list((a.root / "base").glob("*/summary.json"))
    if len(base_paths) != 1:
        raise ValueError("Need the completed frozen reference")
    base_path, = base_paths
    base_config = json.loads((base_path.parent / "config.json").read_text())
    base_rows = json.loads((base_path.parent / "predictions.json").read_text())
    base_summary = json.loads(base_path.read_text())
    validate_predictions(base_rows, base_config, manifest)
    if (base_config["arm"] != "base" or base_config["epochs"] != 0
            or base_config["manifest_sha256"] != expected_manifest_hash
            or base_summary["training"] or base_summary["parameters"]["total_trainable"] != 0):
        raise ValueError("Invalid frozen reference")
    base = dict(run=str(base_path.parent), per_n={n: summarize([r for r in base_rows if r["n_frames"] == n])
                                               for n in (8,16,32,64)},
                in_range=summarize([r for r in base_rows if r["n_frames"] in (8,16)]),
                out_of_range=summarize([r for r in base_rows if r["n_frames"] in (32,64)]))
    comparisons = {}
    rng = np.random.default_rng(271828)
    controls = ("post_sum", "post_mean", "global", "hierarchical", "lora")
    for candidate in ("sum", "mean"):
        for control in controls:
            comparisons[f"{candidate}_vs_{control}"] = {
                name: compare(rows[candidate], rows[control], ns, np, rng)
                for name, ns in (("in_range", (8, 16)), ("out_of_range", (32, 64)))
            }

    descriptive_controls = {
        f"{control}_vs_lora": {name: compare(rows[control], rows["lora"], ns, np, rng)
                               for name, ns in (("in_range", (8,16)), ("out_of_range", (32,64)))}
        for control in ("post_sum", "post_mean", "global", "hierarchical")}

    def passes(candidate, control):
        pair = comparisons[f"{candidate}_vs_{control}"]
        return (pair["out_of_range"]["exact_gain"] >= .05-1e-12
                and pair["in_range"]["exact_gain"] >= -.05-1e-12)
    criteria = dict(V1_1_sum_processing_order=passes("sum", "post_sum"),
                    V1_2_mean_processing_order=passes("mean", "post_mean"),
                    V1_3_practical_screen={x: all(passes(x, c) for c in controls) for x in ("sum", "mean")})
    report = dict(scope="V1 single-seed direct MMReD Vision; dev-selected checkpoints; K<=8; combined length and nuisance-distribution shift",
                  metric_notes={"first_token_nll": "Full-vocabulary generation-policy NLL, after inherited repetition_penalty=1.05; also used for dev accuracy tie-breaking; not raw LM NLL", "global_timing": "Global control redundantly reconstructs the existing SDPA read; not a compute-optimal global baseline"},
                  uncertainty="10000 paired bootstrap replicates within N/gold strata; conditional on seed0; exploratory, not multiplicity-adjusted",
                  manifest_sha256=next(iter(manifest_hashes)), runs=runs, cells=cells, frozen_reference=base,
                  comparisons=comparisons, descriptive_control_comparisons=descriptive_controls, preregistered_criteria=criteria)
    (a.root / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# V1 native MMReD Vision results", "",
             "Seed 0; train N8/16; development-selected checkpoints; 100 test examples per length; K<=8.", "",
             "| Arm | N8 | N16 | N32 | N64 | Pooled N32/64 | Training parameters |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        c = cells[arm]
        numbers = [f'{c["per_n"][n]["correct"]}/100' for n in (8,16,32,64)]
        lines.append(f'| {LABELS[arm]} | '+" | ".join(numbers)+f' | {100*c["out_of_range"]["exact"]:.1f}% | {c["parameters"]["total_trainable"]:,} |')
    lines.append('| Frozen reference | '+" | ".join(f'{base["per_n"][n]["correct"]}/100' for n in (8,16,32,64))
                 +f' | {100*base["out_of_range"]["exact"]:.1f}% | 0 |')
    lines += ["", "## Registered contrasts", "",
              "| Candidate vs control | OOD gain | Paired 95% interval | In-range gain | OOD model latency ratio |", "|---|---:|---:|---:|---:|"]
    for key, pair in comparisons.items():
        ood, ind = pair["out_of_range"], pair["in_range"]
        lo, hi = ood["bootstrap95"]
        lines.append(f'| {key} | {100*ood["exact_gain"]:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {100*ind["exact_gain"]:+.1f} pp | {ood["model_latency_ratio"]:.2f}× |')
    lines += ["", "## Descriptive control comparisons", "",
              "These supplementary contrasts do not alter the preregistered criteria; seed-conditional, unadjusted intervals.", "",
              "| Control vs LoRA | OOD gain | Paired 95% interval | In-range gain |", "|---|---:|---:|---:|"]
    for key, pair in descriptive_controls.items():
        ood, ind = pair["out_of_range"], pair["in_range"]
        lo, hi = ood["bootstrap95"]
        lines.append(f'| {key} | {100*ood["exact_gain"]:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp | {100*ind["exact_gain"]:+.1f} pp |')
    lines += ["", "Criteria: `"+json.dumps(criteria, sort_keys=True)+"`", "",
              "These intervals describe paired-example uncertainty for one seed. They are not independent-seed confirmation.",
              "N16 train/dev have more same-character distractors than test (mean target-character counts 10.31/10.28 vs 6.12); this is a combined length/nuisance-distribution shift, not IID in-range testing or a clean length-only intervention. See data_audit/INDEX.md.",
              "The saved first-token NLL is generation-policy NLL after inherited repetition_penalty=1.05, not raw LM NLL; this also broke dev-accuracy ties.",
              "Policy likelihood, parsing, absolute/signed error, memory, dev histories and provenance are retained in analysis.json.",
              "The global control redundantly reconstructs the SDPA read; its timing is not an optimized global-attention baseline.",
              "Latency was measured during concurrent GPU jobs and is descriptive; claims of speed require dedicated profiling.", ""]
    (a.root / "REPORT.md").write_text("\n".join(lines))

    colors = dict(zip(ARMS, ("#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#333333")))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for arm in ARMS:
        c = cells[arm]
        xs = (8,16,32,64)
        for axis, key in ((axes[0,0], "exact"), (axes[0,1], "first_token_nll")):
            axis.plot(xs, [c["per_n"][n][key] for n in xs], marker="o", color=colors[arm], label=LABELS[arm])
            axis.set_xscale("log", base=2)
            axis.set_xticks(xs, xs)
            axis.axvspan(16.5, 70, color="#eeeeee", zorder=-1)
            axis.set_xlabel("Frames (training: 8 and 16)")
        history = c["training"]
        epochs = [e["epoch"] for e in history]
        dev = [sum(m["correct"] for m in e["dev"])/sum(m["n"] for m in e["dev"]) for e in history]
        axes[1,0].plot(epochs, dev, marker=".", color=colors[arm])
        ood = c["out_of_range"]
        axes[1,1].scatter(ood["model_seconds"], ood["exact"], color=colors[arm], s=45)
    for axis, key in ((axes[0,0], "exact"), (axes[0,1], "first_token_nll")):
        axis.plot((8,16,32,64), [base["per_n"][n][key] for n in (8,16,32,64)],
                  color="#888888", linestyle="--", marker="x", label="Frozen reference")
    axes[0,0].set(title="Native exact answer", ylim=(0,1), ylabel="Accuracy")
    axes[0,1].set(title="First answer-token policy likelihood", ylabel="NLL (repetition penalty 1.05)")
    axes[1,0].set(title="In-range development learning", xlabel="Epoch", ylabel="Accuracy", ylim=(0,1))
    axes[1,1].set(title="Long-input accuracy and measured cost", xlabel="Mean model seconds / example", ylabel="N32/64 accuracy", ylim=(0,1))
    for axis in axes.flat:
        axis.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Native MMReD Vision · Qwen2.5-VL-7B · seed 0", y=.935)
    fig.tight_layout(rect=(0,0,1,.9))
    fig.savefig(a.root / "comparison.png", dpi=170)
    fig.savefig(a.root / "comparison.pdf")
    print(json.dumps(criteria, sort_keys=True), flush=True)
    print("\n".join(lines[:13]), flush=True)


if __name__ == "__main__":
    main()
