#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.probes import run_oracle_count_multilayer_injection_seq8 as oracle
from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev


EXPERIMENT_NAME = "oracle_count_injection_site_sweep_seq8_7b"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME

BASELINE_CODEBOOK = "none"
LAYER_SPECIFIC = "layer_specific"
STATIC = "static"

TOKEN_GROUP_ALIASES = {
    "none": "room_char",
    "room_char": "room_char",
    "all_question_tokens": "all_question_tokens",
    "last_token": "last_token",
    "all_question_tokens+last_token": "question_plus_last",
    "question_plus_last": "question_plus_last",
    "room_char+all_question_tokens+last_token": "room_char_question_last",
    "room_char_question_last": "room_char_question_last",
}

CODEBOOK_TO_PARAM_TYPE = {
    LAYER_SPECIFIC: oracle.LAYER_CODEBOOK,
    STATIC: oracle.STATIC_CODEBOOK,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle count injection site sweep for MMReD seq_len=8 Qwen2.5-VL-7B. "
            "Qwen is frozen; only count-codebook residual vectors are trained."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--config-name", required=True)
    parser.add_argument("--token-group", default="room_char")
    parser.add_argument("--layer-start", type=int, default=14)
    parser.add_argument("--layer-end", type=int, default=17)
    parser.add_argument("--codebook-type", default=LAYER_SPECIFIC, choices=[BASELINE_CODEBOOK, LAYER_SPECIFIC, STATIC])
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--include-random-control", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true", default=False)
    return parser.parse_args()


def safe_name(text: str) -> str:
    safe = str(text)
    for old, new in (("/", "_"), ("+", "_"), (" ", "_"), (":", "_"), (".", "p")):
        safe = safe.replace(old, new)
    return safe


def layer_window_label(start: int, end: int) -> str:
    return f"{int(start)}-{int(end)}" if int(start) != int(end) else str(int(start))


def make_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    layer_label = "no_layers" if is_baseline(args) else f"l{int(args.layer_start)}_{int(args.layer_end)}"
    dirname = f"{stamp}_{safe_name(args.config_name)}_{safe_name(args.codebook_type)}_{safe_name(args.token_group)}_{layer_label}"
    return Path(args.output_root).resolve() / dirname


def is_baseline(args: argparse.Namespace) -> bool:
    return str(args.codebook_type) == BASELINE_CODEBOOK or str(args.config_name) == "baseline_no_injection"


def canonical_token_group(raw: str) -> str:
    key = str(raw).strip()
    if key not in TOKEN_GROUP_ALIASES:
        raise ValueError(f"Unknown token_group={raw!r}; valid={sorted(TOKEN_GROUP_ALIASES)}")
    return TOKEN_GROUP_ALIASES[key]


def limit_indices_balanced(
    indices: Sequence[int],
    records: Sequence[prev.SampleRecord],
    limit: int,
    seed: int,
) -> List[int]:
    values = [int(x) for x in indices]
    if int(limit) <= 0 or len(values) <= int(limit):
        return values
    by_count: Dict[int, List[int]] = defaultdict(list)
    for idx in values:
        by_count[int(records[idx].gold_count)].append(int(idx))
    for count in by_count:
        by_count[count].sort(key=lambda idx: prev.stable_hash_int(f"{seed}:{idx}:{records[idx].sample_id}"))
    rng = random.Random(int(seed))
    selected: List[int] = []
    while len(selected) < int(limit) and any(by_count.values()):
        for count in sorted(by_count):
            if len(selected) >= int(limit):
                break
            bucket = by_count[count]
            if bucket:
                selected.append(bucket.pop(0))
    rng.shuffle(selected)
    return sorted(selected)


def adapter_parameter_norm(adapter: torch.nn.Module) -> float:
    sq = 0.0
    with torch.no_grad():
        for param in adapter.parameters():
            sq += float(param.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(max(0.0, sq))


def make_random_control_like(
    trained_adapter: oracle.CountResidualInjectionAdapter,
    *,
    count_values: Sequence[int],
    hidden_size: int,
    inject_layers: Sequence[int],
    codebook_type: str,
    alpha: float,
    seed: int,
) -> oracle.CountResidualInjectionAdapter:
    random_adapter = oracle.CountResidualInjectionAdapter(
        count_values=count_values,
        hidden_size=int(hidden_size),
        inject_layers=[int(x) for x in inject_layers],
        param_type=CODEBOOK_TO_PARAM_TYPE[str(codebook_type)],
        alpha=float(alpha),
        normalize_injection_energy=False,
        reft_rank=1,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 314159)
    with torch.no_grad():
        for param in random_adapter.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.01)
        target_norm = adapter_parameter_norm(trained_adapter)
        current_norm = adapter_parameter_norm(random_adapter)
        if target_norm > 0 and current_norm > 0:
            scale = target_norm / current_norm
            for param in random_adapter.parameters():
                param.mul_(scale)
    for param in random_adapter.parameters():
        param.requires_grad_(False)
    return random_adapter


def make_eval_rows(
    *,
    eval_payload: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    count_values: Sequence[int],
    seq_len: int,
    config_name: str,
    token_group: str,
    requested_token_group: str,
    layer_window: str,
    alpha: float,
    codebook_type: str,
) -> List[Dict[str, Any]]:
    count_min = min(int(x) for x in count_values)
    count_max = max(int(x) for x in count_values)
    rows: List[Dict[str, Any]] = []
    for idx in eval_payload["indices"]:
        idx = int(idx)
        record = records[idx]
        pred = eval_payload["pred_by_idx"].get(idx, "")
        logits = eval_payload["logits_by_idx"].get(idx, [])
        gold = int(record.gold_count)
        gold_offset = gold - count_min
        pred_offset = int(pred) - count_min if pred != "" else -1
        gold_logit = logits[gold_offset] if 0 <= gold_offset < len(logits) else ""
        pred_logit = logits[pred_offset] if 0 <= pred_offset < len(logits) else ""
        rows.append(
            {
                "sample_id": record.sample_id,
                "sample_index": idx,
                "seq_len": int(seq_len),
                "evidence_count": int(record.evidence_count),
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": int(pred == gold) if pred != "" else "",
                "gold_logit": gold_logit,
                "pred_logit": pred_logit,
                "margin": eval_payload["gold_margin_by_idx"].get(idx, ""),
                "config_name": str(config_name),
                "token_group": str(requested_token_group),
                "resolved_token_group": str(token_group),
                "layer_window": str(layer_window),
                "alpha": float(alpha),
                "codebook_type": str(codebook_type),
                "total_injection_norm": eval_payload["energy_by_idx"].get(idx, 0.0),
                "total_injection_fro_norm": eval_payload["total_norm_by_idx"].get(idx, 0.0),
                "num_injected_tokens": eval_payload["position_count_by_idx"].get(idx, 0),
                "candidate_min": count_min,
                "candidate_max": count_max,
            }
        )
    return rows


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def summarize_sample_rows(rows: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config_name"])].append(row)
    summary_rows: List[Dict[str, Any]] = []
    per_count_rows: List[Dict[str, Any]] = []
    for config_name, config_rows in by_config.items():
        correct = [int(row["correct"]) for row in config_rows if row.get("correct") != ""]
        gold_logits = [value for row in config_rows if (value := finite_float(row.get("gold_logit"))) is not None]
        margins = [value for row in config_rows if (value := finite_float(row.get("margin"))) is not None]
        total_norms = [value for row in config_rows if (value := finite_float(row.get("total_injection_norm"))) is not None]
        acc_by_count: Dict[str, float] = {}
        for count in sorted({int(row["evidence_count"]) for row in config_rows}):
            count_rows = [row for row in config_rows if int(row["evidence_count"]) == int(count)]
            count_correct = [int(row["correct"]) for row in count_rows if row.get("correct") != ""]
            count_margins = [value for row in count_rows if (value := finite_float(row.get("margin"))) is not None]
            count_gold_logits = [value for row in count_rows if (value := finite_float(row.get("gold_logit"))) is not None]
            acc = float(np.mean(count_correct)) if count_correct else math.nan
            acc_by_count[str(count)] = acc
            per_count_rows.append(
                {
                    "config_name": config_name,
                    "evidence_count": int(count),
                    "n": len(count_correct),
                    "accuracy": acc,
                    "mean_margin": float(np.mean(count_margins)) if count_margins else math.nan,
                    "mean_gold_logit": float(np.mean(count_gold_logits)) if count_gold_logits else math.nan,
                    "token_group": config_rows[0].get("token_group", ""),
                    "layer_window": config_rows[0].get("layer_window", ""),
                    "alpha": config_rows[0].get("alpha", ""),
                    "codebook_type": config_rows[0].get("codebook_type", ""),
                }
            )
        summary_rows.append(
            {
                "config_name": config_name,
                "overall_acc": float(np.mean(correct)) if correct else math.nan,
                "n": len(correct),
                "acc_by_evidence_count": json.dumps(acc_by_count, sort_keys=True, separators=(",", ":")),
                "mean_gold_logit": float(np.mean(gold_logits)) if gold_logits else math.nan,
                "mean_margin": float(np.mean(margins)) if margins else math.nan,
                "mean_total_injection_norm": float(np.mean(total_norms)) if total_norms else 0.0,
                "token_group": config_rows[0].get("token_group", ""),
                "resolved_token_group": config_rows[0].get("resolved_token_group", ""),
                "layer_window": config_rows[0].get("layer_window", ""),
                "alpha": config_rows[0].get("alpha", ""),
                "codebook_type": config_rows[0].get("codebook_type", ""),
            }
        )
    summary_rows.sort(key=lambda row: str(row["config_name"]))
    per_count_rows.sort(key=lambda row: (str(row["config_name"]), int(row["evidence_count"])))
    return summary_rows, per_count_rows


def write_dynamic_csv(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    fields = list(leading)
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fields:
            fields.append(key)
    prev.write_csv(path, fields, rows)


def make_plots(output_dir: Path, summary_rows: Sequence[Dict[str, Any]], per_count_rows: Sequence[Dict[str, Any]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        names = [str(row["config_name"]) for row in summary_rows]
        accs = [float(row.get("overall_acc", math.nan)) for row in summary_rows]
        plt.figure(figsize=(max(6.0, 1.4 * len(names)), 4.5))
        plt.bar(np.arange(len(names)), accs)
        plt.xticks(np.arange(len(names)), names, rotation=25, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Overall Accuracy by Config")
        plt.ylim(0, max(1.0, max([x for x in accs if math.isfinite(x)] or [0.0]) * 1.1))
        plt.tight_layout()
        plt.savefig(plots_dir / "overall_accuracy_by_config.png", dpi=180, bbox_inches="tight")
        plt.close()

    configs = list(dict.fromkeys(str(row["config_name"]) for row in per_count_rows))
    counts = sorted({int(row["evidence_count"]) for row in per_count_rows})
    if configs and counts:
        for value_key, ylabel, filename in (
            ("accuracy", "Accuracy", "accuracy_vs_evidence_count_by_config.png"),
            ("mean_margin", "Mean margin", "margin_vs_evidence_count_by_config.png"),
        ):
            plt.figure(figsize=(7.5, 4.8))
            for config in configs:
                by_count = {int(row["evidence_count"]): row for row in per_count_rows if row["config_name"] == config}
                ys = [float(by_count.get(count, {}).get(value_key, math.nan)) for count in counts]
                plt.plot(counts, ys, marker="o", linewidth=1.8, label=config)
            plt.xlabel("Evidence count")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs Evidence Count")
            plt.xticks(counts)
            plt.grid(alpha=0.25)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
            plt.close()

        matrix = np.full((len(configs), len(counts)), np.nan, dtype=float)
        for i, config in enumerate(configs):
            for j, count in enumerate(counts):
                matches = [
                    row for row in per_count_rows if row["config_name"] == config and int(row["evidence_count"]) == int(count)
                ]
                if matches:
                    matrix[i, j] = float(matches[0].get("accuracy", math.nan))
        fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(counts)), max(3.5, 0.45 * len(configs))))
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(counts)
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(configs)
        ax.set_xlabel("Evidence count")
        ax.set_ylabel("Config")
        ax.set_title("Accuracy Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(float(matrix[i, j])):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "heatmap_config_x_evidence_count_accuracy.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if int(args.seq_len) != 8:
        raise ValueError("This experiment is intentionally seq_len=8 only.")
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    if int(args.layer_end) < int(args.layer_start):
        raise ValueError("--layer-end must be >= --layer-start")
    resolved_group = canonical_token_group(args.token_group)
    inject_layers = list(range(int(args.layer_start), int(args.layer_end) + 1))
    count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
    output_dir = make_output_dir(args)
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        run_config = {
            "experiment_name": EXPERIMENT_NAME,
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "seq_len": int(args.seq_len),
            "split": str(args.split),
            "output_root": os.fspath(Path(args.output_root).resolve()),
            "output_dir": os.fspath(output_dir),
            "config_name": str(args.config_name),
            "token_group": str(args.token_group),
            "resolved_token_group": resolved_group,
            "layer_start": int(args.layer_start),
            "layer_end": int(args.layer_end),
            "inject_layers": inject_layers if not is_baseline(args) else [],
            "codebook_type": str(args.codebook_type),
            "alpha": float(args.alpha),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "max_train_samples": int(args.max_train_samples),
            "max_eval_samples": int(args.max_eval_samples),
            "max_samples_per_count": int(args.max_samples_per_count),
            "margin": float(args.margin),
            "lambda_margin": float(args.lambda_margin),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        }
        prev.write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Run config: {json.dumps(run_config, sort_keys=True)}")

        sample_payload = oracle.load_sample_index_payload(args)
        sample_ids = sample_payload["sample_ids"]
        labels = sample_payload["labels"].long()
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        train_indices = limit_indices_balanced(splits["train"], records, int(args.max_train_samples), int(args.seed) + 11)
        val_indices = limit_indices_balanced(
            splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 17,
        )
        test_indices = limit_indices_balanced(
            splits["test"] or splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 23,
        )
        split_counts = prev.split_counts(
            {"train": train_indices, "val": val_indices, "test": test_indices},
            labels,
            count_values,
        )
        for split, row in split_counts.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in count_values))
        if not test_indices:
            raise RuntimeError("Test split is empty after limiting")
        if not is_baseline(args) and (not train_indices or not val_indices):
            raise RuntimeError("Train/val split is empty after limiting")

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_token_ids = prev.candidate_token_ids(
            processor.tokenizer,
            int(args.candidate_min),
            int(args.candidate_max),
        )
        print(f"Loaded frozen Qwen hidden_size={hidden_size} candidate_format={candidate_format}")

        metrics_rows: List[Dict[str, Any]] = []
        train_history_rows: List[Dict[str, Any]] = []
        layer_label = "" if is_baseline(args) else layer_window_label(int(args.layer_start), int(args.layer_end))

        print("Evaluating baseline_no_injection")
        baseline_eval = oracle.evaluate_adapter(
            method="baseline_no_injection",
            model=model,
            processor=processor,
            adapter=None,
            records=records,
            indices=test_indices,
            count_token_ids=count_token_ids,
            token_group="room_char",
            alpha=0.0,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=0,
        )
        metrics_rows.extend(
            make_eval_rows(
                eval_payload=baseline_eval,
                records=records,
                count_values=count_values,
                seq_len=int(args.seq_len),
                config_name="baseline_no_injection",
                token_group="none",
                requested_token_group="none",
                layer_window="",
                alpha=0.0,
                codebook_type=BASELINE_CODEBOOK,
            )
        )

        if not is_baseline(args):
            config = {
                "config_id": str(args.config_name),
                "param_type": CODEBOOK_TO_PARAM_TYPE[str(args.codebook_type)],
                "token_group": resolved_group,
                "layer_window": layer_label,
                "inject_layers": inject_layers,
                "margin_lambda": float(args.lambda_margin),
                "normalize_injection_energy": False,
            }
            train_args = argparse.Namespace(**vars(args))
            train_args.train_alpha = float(args.alpha)
            train_args.train_batch_size = int(args.batch_size)
            train_args.eval_batch_size = int(args.batch_size)
            train_args.max_train_steps = 0
            train_args.residual_l2 = float(args.weight_decay)
            train_args.weight_decay = 0.0
            train_args.reft_rank = 1
            print(f"Training {args.config_name}")
            adapter, history, checkpoint_path = oracle.train_one_config(
                config=config,
                args=train_args,
                output_dir=output_dir,
                model=model,
                processor=processor,
                records=records,
                train_indices=train_indices,
                val_indices=val_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                hidden_size=int(hidden_size),
                device=device,
            )
            train_history_rows.extend(history)

            if bool(args.include_random_control):
                random_adapter = make_random_control_like(
                    adapter,
                    count_values=count_values,
                    hidden_size=int(hidden_size),
                    inject_layers=inject_layers,
                    codebook_type=str(args.codebook_type),
                    alpha=float(args.alpha),
                    seed=int(args.seed),
                )
                random_name = f"{args.config_name}_random_control"
                print(f"Evaluating {random_name}")
                random_eval = oracle.evaluate_adapter(
                    method=random_name,
                    model=model,
                    processor=processor,
                    adapter=random_adapter,
                    records=records,
                    indices=test_indices,
                    count_token_ids=count_token_ids,
                    token_group=resolved_group,
                    alpha=float(args.alpha),
                    device=device,
                    batch_size=int(args.batch_size),
                    max_eval_samples=0,
                )
                metrics_rows.extend(
                    make_eval_rows(
                        eval_payload=random_eval,
                        records=records,
                        count_values=count_values,
                        seq_len=int(args.seq_len),
                        config_name=random_name,
                        token_group=resolved_group,
                        requested_token_group=str(args.token_group),
                        layer_window=layer_label,
                        alpha=float(args.alpha),
                        codebook_type=f"random_{args.codebook_type}",
                    )
                )
                random_adapter.cpu()

            print(f"Evaluating trained {args.config_name}")
            eval_payload = oracle.evaluate_adapter(
                method=str(args.config_name),
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                token_group=resolved_group,
                alpha=float(args.alpha),
                device=device,
                batch_size=int(args.batch_size),
                max_eval_samples=0,
            )
            metrics_rows.extend(
                make_eval_rows(
                    eval_payload=eval_payload,
                    records=records,
                    count_values=count_values,
                    seq_len=int(args.seq_len),
                    config_name=str(args.config_name),
                    token_group=resolved_group,
                    requested_token_group=str(args.token_group),
                    layer_window=layer_label,
                    alpha=float(args.alpha),
                    codebook_type=str(args.codebook_type),
                )
            )
            prev.write_json(
                output_dir / "checkpoint.json",
                {"trained_checkpoint": os.fspath(checkpoint_path), "codebook_type": str(args.codebook_type)},
            )
            adapter.cpu()

        summary_rows, per_count_rows = summarize_sample_rows(metrics_rows)
        write_dynamic_csv(
            output_dir / "metrics.csv",
            metrics_rows,
            [
                "sample_id",
                "seq_len",
                "evidence_count",
                "gold_answer",
                "pred_answer",
                "correct",
                "gold_logit",
                "pred_logit",
                "margin",
                "config_name",
                "token_group",
                "layer_window",
                "alpha",
                "codebook_type",
                "total_injection_norm",
            ],
        )
        write_dynamic_csv(
            output_dir / "summary.csv",
            summary_rows,
            [
                "config_name",
                "overall_acc",
                "acc_by_evidence_count",
                "mean_gold_logit",
                "mean_margin",
                "mean_total_injection_norm",
            ],
        )
        write_dynamic_csv(
            output_dir / "accuracy_by_evidence_count.csv",
            per_count_rows,
            ["config_name", "evidence_count", "n", "accuracy", "mean_margin", "mean_gold_logit"],
        )
        if train_history_rows:
            write_dynamic_csv(
                output_dir / "train_history.csv",
                train_history_rows,
                ["train_config_id", "epoch", "train_ce", "train_loss", "val_ce", "val_accuracy", "val_mae"],
            )
        if not bool(args.no_plots):
            make_plots(output_dir, summary_rows, per_count_rows)
        debug = {
            "runtime_seconds": time.time() - started,
            "source_cache": os.fspath(sample_payload["cache_path"]),
            "num_records": len(records),
            "splits": {key: len(value) for key, value in splits.items()},
            "limited_splits": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
            "output_dir": os.fspath(output_dir),
        }
        prev.write_json(output_dir / "debug.json", debug)
        print(f"Finished {EXPERIMENT_NAME} config={args.config_name} in {time.time() - started:.1f}s")
        print(f"Results: {output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
