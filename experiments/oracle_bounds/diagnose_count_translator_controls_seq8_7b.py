#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.oracle_bounds import translator_ablation_gold_count_seq8_7b as trans


EXPERIMENT_NAME = "translator_ablation_gold_count_seq8_7b_diagnostics"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_ALPHA_VALUES = "0.0,0.02,0.05,0.1,0.2,0.5,1.0"
CONTROL_TRAINED = "trained"
CONTROL_SHUFFLED = "shuffled_count"
CONTROL_RANDOM = "random_same_norm"
CONTROL_BASELINE = "baseline"


def parse_alpha_values(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw).replace(",", " ").split():
        if part.strip():
            values.append(float(part))
    if not values:
        raise ValueError("--alpha-values did not contain any numeric values")
    return sorted(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostics for trained gold-count codebook translators. "
            "Does not retrain; it loads a completed static/layer codebook checkpoint."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed trained run directory.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--method",
        choices=[trans.STATIC_CODEBOOK, trans.LAYER_CODEBOOK],
        default=None,
        help="static_count_codebook or layer_count_codebook.",
    )
    parser.add_argument("--token-group", default=None)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-samples", type=int, default=135)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha-values", default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--state-bottleneck", type=int, default=None)
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


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def resolve_args_from_run(args: argparse.Namespace) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    run_dir = Path(args.run_dir).resolve()
    run_config = read_json_if_exists(run_dir / "run_config.json")
    if args.model_name is None:
        args.model_name = str(run_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
    if args.dataset_root is None:
        args.dataset_root = Path(run_config.get("dataset_root", PROJECT_ROOT / "data/mmred_images_park"))
    if args.source_run is None:
        args.source_run = Path(run_config.get("source_run", trans.prev.DEFAULT_SOURCE_RUN))
    if args.seq_len is None:
        args.seq_len = int(run_config.get("seq_len", 8))
    if args.split is None:
        args.split = str(run_config.get("split", "all_uniform"))
    if args.method is None:
        args.method = str(run_config.get("method", ""))
    if args.token_group is None:
        args.token_group = str(run_config.get("token_group", "room_char"))
    if args.layer_start is None:
        args.layer_start = int(run_config.get("layer_start", 14))
    if args.layer_end is None:
        args.layer_end = int(run_config.get("layer_end", 17))
    if args.seed is None:
        args.seed = int(run_config.get("seed", 0))
    if args.rank is None:
        args.rank = int(run_config.get("rank", 16))
    if args.state_bottleneck is None:
        args.state_bottleneck = int(run_config.get("state_bottleneck", 64))
    args.run_dir = run_dir
    args.dataset_root = Path(args.dataset_root)
    args.source_run = Path(args.source_run)
    args.output_root = Path(args.output_root)
    args.method = trans.canonical_method(str(args.method))
    if args.method not in {trans.STATIC_CODEBOOK, trans.LAYER_CODEBOOK}:
        raise ValueError(f"Diagnostics only support static/layer codebooks, got method={args.method!r}")
    args.token_group = trans.canonical_token_group(str(args.token_group))
    return args, run_config


def make_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    layer_label = f"l{int(args.layer_start)}_{int(args.layer_end)}"
    dirname = (
        f"{stamp}_{trans.safe_name(Path(args.run_dir).name)}_"
        f"{trans.safe_name(args.method)}_{trans.safe_name(args.token_group)}_{layer_label}"
    )
    return Path(args.output_root).resolve() / dirname


def find_checkpoint(run_dir: Path) -> Path:
    manifest = read_json_if_exists(run_dir / "checkpoint.json")
    ckpt = manifest.get("trained_checkpoint")
    if ckpt:
        path = Path(str(ckpt))
        if path.is_file():
            return path
    candidates = sorted((run_dir / "checkpoints").glob("*_best.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Could not find trained checkpoint under {run_dir}")


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    vals = [float(x) for x in values if finite_float(x) is not None]
    return float(np.mean(vals)) if vals else float(default)


def safe_div(numerator: float, denominator: float, default: float = math.nan) -> float:
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-12:
        return float(default)
    return float(numerator) / float(denominator)


def load_trained_adapter(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    hidden_size: int,
    count_values: Sequence[int],
    inject_layers: Sequence[int],
) -> trans.TranslatorInjectionAdapter:
    checkpoint = trans.prev.load_torch(checkpoint_path)
    ckpt_hidden_size = int(checkpoint.get("hidden_size", hidden_size))
    if ckpt_hidden_size != int(hidden_size):
        print(f"Warning: checkpoint hidden_size={ckpt_hidden_size}, model hidden_size={hidden_size}; using checkpoint size")
    ckpt_count_values = [int(x) for x in checkpoint.get("count_values", count_values)]
    if list(ckpt_count_values) != [int(x) for x in count_values]:
        raise ValueError(f"Checkpoint count_values={ckpt_count_values} do not match requested {list(count_values)}")
    ckpt_config = dict(checkpoint.get("config", {}))
    rank = int(ckpt_config.get("rank", args.rank))
    state_bottleneck = int(ckpt_config.get("state_bottleneck", args.state_bottleneck))
    adapter = trans.make_adapter(
        method=str(args.method),
        count_values=ckpt_count_values,
        hidden_size=ckpt_hidden_size,
        inject_layers=[int(x) for x in inject_layers],
        alpha=1.0,
        rank=rank,
        state_bottleneck=state_bottleneck,
    )
    state = checkpoint.get("adapter_state_dict", checkpoint)
    adapter.load_state_dict(state)
    for param in adapter.parameters():
        param.requires_grad_(False)
    adapter.eval()
    return adapter


def make_random_same_norm_adapter(
    trained_adapter: trans.TranslatorInjectionAdapter,
    *,
    seed: int,
) -> trans.TranslatorInjectionAdapter:
    random_adapter = trans.make_adapter(
        method=trained_adapter.method,
        count_values=trained_adapter.count_values,
        hidden_size=trained_adapter.hidden_size,
        inject_layers=trained_adapter.inject_layers,
        alpha=trained_adapter.alpha,
        rank=trained_adapter.rank,
        state_bottleneck=trained_adapter.state_bottleneck,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 910031)
    with torch.no_grad():
        if trained_adapter.method == trans.STATIC_CODEBOOK:
            for count_pos in range(int(trained_adapter.codebook.shape[0])):
                target_norm = trained_adapter.codebook[count_pos].detach().float().norm()
                draw = torch.randn(trained_adapter.codebook[count_pos].shape, generator=generator, dtype=torch.float32)
                draw_norm = draw.norm()
                if float(draw_norm) > 0.0:
                    draw = draw * (target_norm / draw_norm)
                random_adapter.codebook[count_pos].copy_(draw.to(dtype=random_adapter.codebook.dtype))
        elif trained_adapter.method == trans.LAYER_CODEBOOK:
            for layer_pos in range(int(trained_adapter.codebook.shape[0])):
                for count_pos in range(int(trained_adapter.codebook.shape[1])):
                    target_norm = trained_adapter.codebook[layer_pos, count_pos].detach().float().norm()
                    draw = torch.randn(
                        trained_adapter.codebook[layer_pos, count_pos].shape,
                        generator=generator,
                        dtype=torch.float32,
                    )
                    draw_norm = draw.norm()
                    if float(draw_norm) > 0.0:
                        draw = draw * (target_norm / draw_norm)
                    random_adapter.codebook[layer_pos, count_pos].copy_(draw.to(dtype=random_adapter.codebook.dtype))
        else:
            raise ValueError(f"Unsupported random same-norm method={trained_adapter.method!r}")
    for param in random_adapter.parameters():
        param.requires_grad_(False)
    random_adapter.eval()
    return random_adapter


class HiddenNormRecorder:
    def __init__(self, inject_layers: Sequence[int]) -> None:
        self.inject_layers = [int(x) for x in inject_layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.inject_layers)}
        self._target_positions: List[List[int]] = []
        self._sum_norms: List[float] = []
        self._sum_sq_norms: List[float] = []
        self._site_counts: List[int] = []
        self._handles: List[Any] = []

    def set_context(self, target_positions: Sequence[Sequence[int]], batch_size: int) -> None:
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self._sum_norms = [0.0 for _ in range(int(batch_size))]
        self._sum_sq_norms = [0.0 for _ in range(int(batch_size))]
        self._site_counts = [0 for _ in range(int(batch_size))]

    def clear_context(self) -> None:
        self._target_positions = []

    def record(self, hidden_states: torch.Tensor) -> None:
        if not self._target_positions:
            return
        seq_len = int(hidden_states.shape[1])
        for batch_idx, positions in enumerate(self._target_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            if not valid or batch_idx >= len(self._sum_norms):
                continue
            pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
            values = hidden_states[batch_idx, pos_idx, :].detach().float()
            norms = values.norm(dim=-1)
            self._sum_norms[batch_idx] += float(norms.sum().cpu().item())
            self._sum_sq_norms[batch_idx] += float(values.pow(2).sum().cpu().item())
            self._site_counts[batch_idx] += len(valid)

    def last_stats(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for sum_norm, sum_sq, count in zip(self._sum_norms, self._sum_sq_norms, self._site_counts):
            hidden_norm_mean = float(sum_norm) / max(1, int(count))
            hidden_sqrt_energy = math.sqrt(max(0.0, float(sum_sq)))
            rows.append(
                {
                    "hidden_norm_mean": hidden_norm_mean,
                    "hidden_sqrt_energy": hidden_sqrt_energy,
                    "hidden_site_count": float(count),
                }
            )
        return rows

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = trans.prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"inject_layer={layer_idx} outside [0, {len(layers) - 1}]")

            def hook(_module: Any, _args: Any, output: Any, *, layer: int = int(layer_idx)) -> Any:
                hidden = trans.TranslatorInjectionAdapter._hidden_from_output(output)
                if hidden is not None:
                    self.record(hidden)
                return output

            self._handles.append(layers[int(layer_idx)].register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def evaluate_diagnostic_split(
    *,
    control: str,
    split_name: str,
    model: Any,
    processor: Any,
    adapter: Optional[trans.TranslatorInjectionAdapter],
    records: Sequence[Any],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    count_values: Sequence[int],
    token_group: str,
    alpha: float,
    device: str,
    batch_size: int,
    max_eval_samples: int,
    count_control: str,
    seed: int,
    collect_hidden_norm: bool,
) -> Dict[str, Any]:
    eval_indices = [int(x) for x in indices]
    if int(max_eval_samples) > 0:
        eval_indices = eval_indices[: int(max_eval_samples)]
    count_min = min(int(x) for x in count_values)
    recorder: Optional[HiddenNormRecorder] = None
    if collect_hidden_norm:
        if adapter is None:
            raise ValueError("collect_hidden_norm requires an adapter/injection layer definition")
        recorder = HiddenNormRecorder(adapter.inject_layers)
        recorder.register_hooks(model)
    if adapter is not None:
        adapter.to(device)
        adapter.eval()
        adapter.enabled = True
        adapter.set_alpha(float(alpha))
        adapter.register_hooks(model)

    rows: List[Dict[str, Any]] = []
    ce_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(trans.prev.chunked(eval_indices, int(batch_size)), start=1):
            batch_records = [records[int(idx)] for idx in batch_indices]
            batch = trans.prepare_translator_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                token_group=token_group if adapter is not None else "none",
            )
            inject_counts: Optional[torch.Tensor] = None
            if recorder is not None:
                recorder.set_context(batch.target_positions, int(batch.gold_counts.numel()))
            if adapter is not None:
                inject_counts = trans.control_counts_for_injection(
                    gold_counts=batch.gold_counts,
                    sample_indices=batch.sample_indices,
                    count_values=count_values,
                    seed=int(seed),
                    control=count_control,
                )
                adapter.set_context(inject_counts, batch.target_positions)
            with torch.no_grad():
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = trans.prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - int(count_min)
                ce_vec = F.cross_entropy(count_logits, gold_offsets, reduction="none")
                ce_total += float(ce_vec.sum().detach().cpu().item())
                n += int(batch.gold_counts.numel())
                pred_offsets = count_logits.argmax(dim=-1)
                gold_logits, _best_wrong, margins = trans.select_gold_logits_and_margins(count_logits, gold_offsets)
                adapter_energy = adapter.last_energy() if adapter is not None else [0.0 for _ in batch_indices]
                hidden_stats = recorder.last_stats() if recorder is not None else [{} for _ in batch_indices]
                logits_cpu = count_logits.detach().float().cpu()
                inject_counts_cpu = (
                    [int(x) for x in inject_counts.detach().cpu().tolist()] if inject_counts is not None else [None] * len(batch_indices)
                )
                for row_pos, idx in enumerate(batch_indices):
                    idx = int(idx)
                    record = records[idx]
                    pred = int(pred_offsets[row_pos].detach().cpu().item()) + int(count_min)
                    gold = int(record.gold_count)
                    gold_offset = gold - int(count_min)
                    pred_offset = pred - int(count_min)
                    logits = [float(v) for v in logits_cpu[row_pos].tolist()]
                    injection_energy = float(adapter_energy[row_pos]) if row_pos < len(adapter_energy) else 0.0
                    injection_sqrt_norm = math.sqrt(max(0.0, injection_energy))
                    hidden_row = hidden_stats[row_pos] if row_pos < len(hidden_stats) else {}
                    hidden_norm_mean = finite_float(hidden_row.get("hidden_norm_mean"))
                    hidden_sqrt_energy = finite_float(hidden_row.get("hidden_sqrt_energy"))
                    norm_ratio = (
                        safe_div(injection_sqrt_norm, hidden_sqrt_energy)
                        if hidden_sqrt_energy is not None
                        else math.nan
                    )
                    rows.append(
                        {
                            "control": str(control),
                            "split": str(split_name),
                            "alpha": float(alpha),
                            "sample_id": record.sample_id,
                            "sample_index": idx,
                            "seq_len": int(getattr(record, "seq_len", 8)) if hasattr(record, "seq_len") else "",
                            "evidence_count": int(record.evidence_count),
                            "gold_answer": gold,
                            "injected_count": inject_counts_cpu[row_pos] if inject_counts_cpu[row_pos] is not None else "",
                            "pred_answer": pred,
                            "correct": int(pred == gold),
                            "pred_matches_injected": (
                                int(pred == int(inject_counts_cpu[row_pos])) if inject_counts_cpu[row_pos] is not None else ""
                            ),
                            "gold_logit": logits[gold_offset] if 0 <= gold_offset < len(logits) else "",
                            "pred_logit": logits[pred_offset] if 0 <= pred_offset < len(logits) else "",
                            "margin": float(margins[row_pos].detach().cpu().item()),
                            "ce": float(ce_vec[row_pos].detach().cpu().item()),
                            "injection_norm": injection_energy,
                            "injection_energy": injection_energy,
                            "injection_sqrt_norm": injection_sqrt_norm,
                            "hidden_norm_mean": hidden_norm_mean if hidden_norm_mean is not None else "",
                            "hidden_sqrt_energy": hidden_sqrt_energy if hidden_sqrt_energy is not None else "",
                            "hidden_site_count": hidden_row.get("hidden_site_count", ""),
                            "norm_ratio": norm_ratio,
                            "norm_ratio_to_mean_hidden": (
                                safe_div(injection_sqrt_norm, hidden_norm_mean)
                                if hidden_norm_mean is not None
                                else math.nan
                            ),
                            "token_selection_ok": int(bool(batch.token_selection_ok[row_pos])) if adapter is not None else "",
                            "token_selection_error": batch.token_selection_errors[row_pos] if adapter is not None else "",
                            "num_injected_tokens": len(batch.target_positions[row_pos]) if adapter is not None else "",
                            "candidate_logits_json": trans.json_dumps_compact(logits),
                        }
                    )
            if adapter is not None:
                adapter.clear_context()
            if recorder is not None:
                recorder.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(
                    f"  eval {control} alpha={float(alpha):g} {split_name}: "
                    f"{min(len(eval_indices), batch_num * int(batch_size))}/{len(eval_indices)}"
                )
    finally:
        if adapter is not None:
            adapter.remove_hooks()
        if recorder is not None:
            recorder.remove_hooks()
    accuracy = finite_mean([row["correct"] for row in rows], default=math.nan)
    return {"rows": rows, "accuracy": accuracy, "ce": ce_total / max(1, n), "indices": eval_indices}


def summarize_eval_rows(rows: Sequence[Dict[str, Any]], *, control: str, alpha: float) -> Dict[str, Any]:
    count_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        count_rows[int(row["evidence_count"])].append(dict(row))
    acc_by_count: Dict[str, float] = {}
    n_by_count: Dict[str, int] = {}
    margin_by_count: Dict[str, float] = {}
    for count in sorted(count_rows):
        subset = count_rows[count]
        acc_by_count[str(count)] = finite_mean([row.get("correct") for row in subset])
        n_by_count[str(count)] = len(subset)
        margin_by_count[str(count)] = finite_mean([row.get("margin") for row in subset])
    return {
        "control": str(control),
        "alpha": float(alpha),
        "n": len(rows),
        "acc": finite_mean([row.get("correct") for row in rows]),
        "overall_accuracy": finite_mean([row.get("correct") for row in rows]),
        "mean_margin": finite_mean([row.get("margin") for row in rows]),
        "mean_gold_logit": finite_mean([row.get("gold_logit") for row in rows]),
        "mean_pred_logit": finite_mean([row.get("pred_logit") for row in rows]),
        "mean_injection_norm": finite_mean([row.get("injection_norm") for row in rows], default=0.0),
        "mean_injection_sqrt_norm": finite_mean([row.get("injection_sqrt_norm") for row in rows], default=0.0),
        "mean_hidden_norm": finite_mean([row.get("hidden_norm_mean") for row in rows]),
        "mean_hidden_sqrt_energy": finite_mean([row.get("hidden_sqrt_energy") for row in rows]),
        "mean_norm_ratio": finite_mean([row.get("norm_ratio") for row in rows]),
        "norm_ratio": finite_mean([row.get("norm_ratio") for row in rows]),
        "mean_pred_matches_injected": finite_mean([row.get("pred_matches_injected") for row in rows]),
        "acc_by_evidence_count": trans.json_dumps_compact(acc_by_count),
        "n_by_evidence_count": trans.json_dumps_compact(n_by_count),
        "margin_by_evidence_count": trans.json_dumps_compact(margin_by_count),
    }


def per_count_summary_rows(rows: Sequence[Dict[str, Any]], *, control: str, alpha: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_count: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_count[int(row["evidence_count"])].append(dict(row))
    for count in sorted(by_count):
        subset = by_count[count]
        out.append(
            {
                "control": str(control),
                "alpha": float(alpha),
                "evidence_count": int(count),
                "n": len(subset),
                "accuracy": finite_mean([row.get("correct") for row in subset]),
                "mean_margin": finite_mean([row.get("margin") for row in subset]),
                "mean_injection_norm": finite_mean([row.get("injection_norm") for row in subset], default=0.0),
                "mean_norm_ratio": finite_mean([row.get("norm_ratio") for row in subset]),
            }
        )
    return out


def aggregate_logit_decomposition(
    *,
    rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    count_values: Sequence[int],
) -> List[Dict[str, Any]]:
    baseline_by_idx = {int(row["sample_index"]): row for row in baseline_rows}
    count_min = min(int(x) for x in count_values)
    grouped: Dict[Tuple[str, float], List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        idx = int(row["sample_index"])
        base = baseline_by_idx.get(idx)
        if base is None:
            continue
        logits = json.loads(str(row["candidate_logits_json"]))
        base_logits = json.loads(str(base["candidate_logits_json"]))
        gold_offset = int(row["gold_answer"]) - int(count_min)
        if not (0 <= gold_offset < len(logits) and len(logits) == len(base_logits)):
            continue
        deltas = [float(logits[pos]) - float(base_logits[pos]) for pos in range(len(logits))]
        non_gold = [delta for pos, delta in enumerate(deltas) if pos != gold_offset]
        grouped[(str(row["control"]), float(row["alpha"]))].append(
            {
                "delta_gold_logit": deltas[gold_offset],
                "mean_delta_non_gold_logits": float(np.mean(non_gold)) if non_gold else math.nan,
                "max_delta_non_gold_logit": float(np.max(non_gold)) if non_gold else math.nan,
                "delta_margin": float(row["margin"]) - float(base["margin"]),
            }
        )
    out: List[Dict[str, Any]] = []
    for (control, alpha), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out.append(
            {
                "control": control,
                "alpha": alpha,
                "n": len(values),
                "delta_gold_logit": finite_mean([row["delta_gold_logit"] for row in values]),
                "mean_delta_non_gold_logits": finite_mean([row["mean_delta_non_gold_logits"] for row in values]),
                "max_delta_non_gold_logit": finite_mean([row["max_delta_non_gold_logit"] for row in values]),
                "delta_margin": finite_mean([row["delta_margin"] for row in values]),
            }
        )
    return out


def cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    a = vec_a.detach().float().view(-1)
    b = vec_b.detach().float().view(-1)
    denom = float(a.norm().cpu().item()) * float(b.norm().cpu().item())
    if denom <= 1e-12:
        return math.nan
    return float(torch.dot(a.cpu(), b.cpu()).item() / denom)


def codebook_vectors_by_count_layer(adapter: trans.TranslatorInjectionAdapter) -> List[Tuple[int, int, torch.Tensor]]:
    rows: List[Tuple[int, int, torch.Tensor]] = []
    with torch.no_grad():
        if adapter.method == trans.STATIC_CODEBOOK:
            for layer in adapter.inject_layers:
                for count_pos, count in enumerate(adapter.count_values):
                    rows.append((int(count), int(layer), adapter.codebook[count_pos].detach().float().cpu().clone()))
        elif adapter.method == trans.LAYER_CODEBOOK:
            for layer_pos, layer in enumerate(adapter.inject_layers):
                for count_pos, count in enumerate(adapter.count_values):
                    rows.append((int(count), int(layer), adapter.codebook[layer_pos, count_pos].detach().float().cpu().clone()))
        else:
            raise ValueError(f"Unsupported alignment method={adapter.method!r}")
    return rows


def lm_head_alignment_rows(
    *,
    model: Any,
    adapter: trans.TranslatorInjectionAdapter,
    count_token_ids: Dict[int, int],
) -> List[Dict[str, Any]]:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise RuntimeError("Could not access model output embeddings for LM-head alignment")
    weight = output_embeddings.weight
    answer_vectors: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for count, token_id in sorted(count_token_ids.items()):
            answer_vectors[int(count)] = weight[int(token_id)].detach().float().cpu().clone()
    rows: List[Dict[str, Any]] = []
    for count, layer, vector in codebook_vectors_by_count_layer(adapter):
        own = answer_vectors[int(count)]
        other_items = [(other_count, answer_vectors[other_count]) for other_count in sorted(answer_vectors) if other_count != count]
        cos_by_answer = {str(other_count): cosine(vector, answer_vectors[other_count]) for other_count in sorted(answer_vectors)}
        own_cos = cos_by_answer[str(count)]
        other_cos_values = [cos_by_answer[str(other_count)] for other_count, _ in other_items]
        own_minus_other_cos = {
            str(other_count): cosine(vector, own - other_vec) for other_count, other_vec in other_items
        }
        mean_other_vec = torch.stack([other_vec for _other_count, other_vec in other_items]).mean(dim=0)
        max_count = max(cos_by_answer, key=lambda key: -math.inf if finite_float(cos_by_answer[key]) is None else cos_by_answer[key])
        rows.append(
            {
                "count": int(count),
                "layer": int(layer),
                "norm": float(vector.norm().item()),
                "max_cos_to_answer_unembedding": cos_by_answer[str(max_count)],
                "max_cos_answer": int(max_count),
                "cos_to_own_answer": own_cos,
                "mean_cos_to_other_answers": finite_mean(other_cos_values),
                "own_minus_other_gap": (
                    own_cos - finite_mean(other_cos_values) if finite_float(own_cos) is not None else math.nan
                ),
                "cos_to_own_minus_mean_other": cosine(vector, own - mean_other_vec),
                "mean_cos_to_own_minus_other": finite_mean(own_minus_other_cos.values()),
                "max_cos_to_own_minus_other": max(
                    [value for value in own_minus_other_cos.values() if finite_float(value) is not None],
                    default=math.nan,
                ),
                "cos_to_answers_json": trans.json_dumps_compact(cos_by_answer),
                "cos_to_own_minus_each_other_json": trans.json_dumps_compact(own_minus_other_cos),
            }
        )
    return rows


def write_rows(path: Path, rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> None:
    trans.write_dynamic_csv(path, rows, leading)


def plot_line(
    *,
    path: Path,
    summary_rows: Sequence[Dict[str, Any]],
    controls: Sequence[str],
    y_key: str,
    ylabel: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for control in controls:
        rows = sorted([row for row in summary_rows if row["control"] == control], key=lambda row: float(row["alpha"]))
        if not rows:
            continue
        plt.plot([float(row["alpha"]) for row in rows], [float(row.get(y_key, math.nan)) for row in rows], marker="o", label=control)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if len(controls) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_plots(
    *,
    output_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    per_count_rows: Sequence[Dict[str, Any]],
    alignment_rows: Sequence[Dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_line(
        path=plot_dir / "accuracy_vs_alpha.png",
        summary_rows=summary_rows,
        controls=[CONTROL_TRAINED],
        y_key="acc",
        ylabel="accuracy",
        title="Trained Codebook Accuracy vs Alpha",
    )
    plot_line(
        path=plot_dir / "margin_vs_alpha.png",
        summary_rows=summary_rows,
        controls=[CONTROL_TRAINED],
        y_key="mean_margin",
        ylabel="mean margin",
        title="Trained Codebook Margin vs Alpha",
    )
    plot_line(
        path=plot_dir / "injection_norm_vs_alpha.png",
        summary_rows=summary_rows,
        controls=[CONTROL_TRAINED],
        y_key="mean_injection_norm",
        ylabel="mean injection energy",
        title="Trained Codebook Injection Norm vs Alpha",
    )
    plot_line(
        path=plot_dir / "trained_vs_shuffled_vs_random_accuracy_by_alpha.png",
        summary_rows=summary_rows,
        controls=[CONTROL_TRAINED, CONTROL_SHUFFLED, CONTROL_RANDOM],
        y_key="acc",
        ylabel="accuracy",
        title="Controls Accuracy vs Alpha",
    )
    plot_line(
        path=plot_dir / "norm_ratio_vs_alpha.png",
        summary_rows=summary_rows,
        controls=[CONTROL_TRAINED, CONTROL_SHUFFLED, CONTROL_RANDOM],
        y_key="mean_norm_ratio",
        ylabel="mean norm ratio",
        title="Injection / Hidden Norm Ratio vs Alpha",
    )

    selected_alphas = {0.0, 0.05, 0.1, 0.5, 1.0}
    rows = [
        row
        for row in per_count_rows
        if row["control"] == CONTROL_TRAINED and any(abs(float(row["alpha"]) - alpha) < 1e-9 for alpha in selected_alphas)
    ]
    if rows:
        plt.figure(figsize=(8, 4.8))
        for alpha in sorted({float(row["alpha"]) for row in rows}):
            subset = sorted([row for row in rows if abs(float(row["alpha"]) - alpha) < 1e-9], key=lambda row: int(row["evidence_count"]))
            plt.plot(
                [int(row["evidence_count"]) for row in subset],
                [float(row["accuracy"]) for row in subset],
                marker="o",
                label=f"alpha={alpha:g}",
            )
        plt.xlabel("evidence_count")
        plt.ylabel("accuracy")
        plt.title("Trained Accuracy by Evidence Count")
        plt.xticks(sorted({int(row["evidence_count"]) for row in rows}))
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "accuracy_by_evidence_count_selected_alphas.png", dpi=180)
        plt.close()

    if alignment_rows:
        counts = sorted({int(row["count"]) for row in alignment_rows})
        layers = sorted({int(row["layer"]) for row in alignment_rows})
        matrix = np.full((len(counts), len(layers)), np.nan, dtype=float)
        count_to_row = {count: pos for pos, count in enumerate(counts)}
        layer_to_col = {layer: pos for pos, layer in enumerate(layers)}
        for row in alignment_rows:
            matrix[count_to_row[int(row["count"])], layer_to_col[int(row["layer"])]] = float(row["cos_to_own_answer"])
        plt.figure(figsize=(max(5.5, 0.6 * len(layers)), 4.8))
        plt.imshow(matrix, aspect="auto", cmap="coolwarm")
        plt.colorbar(label="cos(e, own answer unembedding)")
        plt.xlabel("layer")
        plt.ylabel("count")
        plt.title("LM-head Cosine Heatmap")
        plt.xticks(range(len(layers)), layers)
        plt.yticks(range(len(counts)), counts)
        plt.tight_layout()
        plt.savefig(plot_dir / "lm_head_cosine_heatmap_count_x_layer.png", dpi=180)
        plt.close()


def print_quick_summary(summary_rows: Sequence[Dict[str, Any]]) -> None:
    order = {CONTROL_TRAINED: 0, CONTROL_SHUFFLED: 1, CONTROL_RANDOM: 2, CONTROL_BASELINE: 3}
    rows = sorted(
        [row for row in summary_rows if row["control"] in {CONTROL_TRAINED, CONTROL_SHUFFLED, CONTROL_RANDOM}],
        key=lambda row: (order.get(str(row["control"]), 99), float(row["alpha"])),
    )
    print("Quick summary:")
    print(f"{'control':<18} {'alpha':>7} {'acc':>8} {'margin':>10} {'inj_norm':>12} {'norm_ratio':>12}")
    for row in rows:
        print(
            f"{str(row['control']):<18} {float(row['alpha']):>7.3g} "
            f"{float(row.get('acc', math.nan)):>8.4f} "
            f"{float(row.get('mean_margin', math.nan)):>10.4f} "
            f"{float(row.get('mean_injection_norm', math.nan)):>12.4f} "
            f"{float(row.get('mean_norm_ratio', math.nan)):>12.4f}"
        )


def main() -> int:
    args = parse_args()
    args, source_run_config = resolve_args_from_run(args)
    alpha_values = parse_alpha_values(str(args.alpha_values))
    inject_layers = list(range(int(args.layer_start), int(args.layer_end) + 1))
    count_values = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
    output_dir = make_output_dir(args)
    log_handle, old_stdout, old_stderr = trans.prev.setup_logging(output_dir)
    started = time.time()
    exact_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    try:
        print(f"Output dir: {output_dir}")
        print(f"Exact command: {exact_command}")
        checkpoint_path = find_checkpoint(Path(args.run_dir))
        run_config = {
            **{key: os.fspath(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "experiment_name": EXPERIMENT_NAME,
            "alpha_values": alpha_values,
            "inject_layers": inject_layers,
            "count_values": count_values,
            "source_run_config": source_run_config,
            "checkpoint_path": os.fspath(checkpoint_path),
            "exact_command": exact_command,
            "output_dir": os.fspath(output_dir),
        }
        trans.prev.write_json(output_dir / "run_config.json", run_config)
        trans.prev.set_seed(int(args.seed))

        sample_payload = trans.load_sample_index_payload(args)
        sample_ids = sample_payload["sample_ids"]
        labels = sample_payload["labels"].long()
        records = trans.prev.load_records(args.dataset_root, args.split, int(args.seq_len), sample_ids)
        splits = trans.prev.stratified_split(sample_ids, labels, int(args.seed))
        test_indices = trans.split_limited_indices(
            splits["test"] or splits["val"] or splits["train"],
            records,
            int(args.max_eval_samples),
            int(args.seed) + 23,
        )
        if not test_indices:
            raise RuntimeError("Test split is empty after limiting")
        split_counts = trans.prev.split_counts({"test": test_indices}, labels, count_values)
        for split, row in split_counts.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in count_values))

        device = trans.prev.resolve_device(str(args.device))
        dtype = trans.prev.resolve_dtype(str(args.dtype), device)
        model, processor = trans.prev.load_model_and_processor(args, device=device, dtype=dtype)
        hidden_size = trans.prev.hidden_size_from_model(model)
        candidate_format, count_token_ids = trans.prev.candidate_token_ids(
            processor.tokenizer,
            int(args.candidate_min),
            int(args.candidate_max),
        )
        print(f"Loaded frozen Qwen hidden_size={hidden_size} candidate_format={candidate_format}")
        trained_adapter = load_trained_adapter(
            args=args,
            checkpoint_path=checkpoint_path,
            hidden_size=int(hidden_size),
            count_values=count_values,
            inject_layers=inject_layers,
        )
        random_adapter = make_random_same_norm_adapter(trained_adapter, seed=int(args.seed))

        print("Evaluating baseline")
        baseline_payload = evaluate_diagnostic_split(
            control=CONTROL_BASELINE,
            split_name="test",
            model=model,
            processor=processor,
            adapter=None,
            records=records,
            indices=test_indices,
            count_token_ids=count_token_ids,
            count_values=count_values,
            token_group=args.token_group,
            alpha=0.0,
            device=device,
            batch_size=int(args.batch_size),
            max_eval_samples=0,
            count_control=trans.COUNT_CONTROL_NONE,
            seed=int(args.seed),
            collect_hidden_norm=False,
        )
        baseline_rows = baseline_payload["rows"]

        trained_rows: List[Dict[str, Any]] = []
        shuffled_rows: List[Dict[str, Any]] = []
        random_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = [
            summarize_eval_rows(baseline_rows, control=CONTROL_BASELINE, alpha=0.0)
        ]
        per_count_rows: List[Dict[str, Any]] = per_count_summary_rows(
            baseline_rows,
            control=CONTROL_BASELINE,
            alpha=0.0,
        )

        for alpha in alpha_values:
            print(f"Evaluating trained codebook alpha={alpha:g}")
            payload = evaluate_diagnostic_split(
                control=CONTROL_TRAINED,
                split_name="test",
                model=model,
                processor=processor,
                adapter=trained_adapter,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                token_group=args.token_group,
                alpha=float(alpha),
                device=device,
                batch_size=int(args.batch_size),
                max_eval_samples=0,
                count_control=trans.COUNT_CONTROL_NONE,
                seed=int(args.seed),
                collect_hidden_norm=True,
            )
            rows = payload["rows"]
            trained_rows.extend(rows)
            summary_rows.append(summarize_eval_rows(rows, control=CONTROL_TRAINED, alpha=float(alpha)))
            per_count_rows.extend(per_count_summary_rows(rows, control=CONTROL_TRAINED, alpha=float(alpha)))

            print(f"Evaluating shuffled-count control alpha={alpha:g}")
            payload = evaluate_diagnostic_split(
                control=CONTROL_SHUFFLED,
                split_name="test",
                model=model,
                processor=processor,
                adapter=trained_adapter,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                token_group=args.token_group,
                alpha=float(alpha),
                device=device,
                batch_size=int(args.batch_size),
                max_eval_samples=0,
                count_control=trans.COUNT_CONTROL_SHUFFLED,
                seed=int(args.seed),
                collect_hidden_norm=True,
            )
            rows = payload["rows"]
            shuffled_rows.extend(rows)
            summary_rows.append(summarize_eval_rows(rows, control=CONTROL_SHUFFLED, alpha=float(alpha)))
            per_count_rows.extend(per_count_summary_rows(rows, control=CONTROL_SHUFFLED, alpha=float(alpha)))

            print(f"Evaluating random same-norm control alpha={alpha:g}")
            payload = evaluate_diagnostic_split(
                control=CONTROL_RANDOM,
                split_name="test",
                model=model,
                processor=processor,
                adapter=random_adapter,
                records=records,
                indices=test_indices,
                count_token_ids=count_token_ids,
                count_values=count_values,
                token_group=args.token_group,
                alpha=float(alpha),
                device=device,
                batch_size=int(args.batch_size),
                max_eval_samples=0,
                count_control=trans.COUNT_CONTROL_NONE,
                seed=int(args.seed),
                collect_hidden_norm=True,
            )
            rows = payload["rows"]
            random_rows.extend(rows)
            summary_rows.append(summarize_eval_rows(rows, control=CONTROL_RANDOM, alpha=float(alpha)))
            per_count_rows.extend(per_count_summary_rows(rows, control=CONTROL_RANDOM, alpha=float(alpha)))

        all_control_rows = [*trained_rows, *shuffled_rows, *random_rows]
        logit_rows = aggregate_logit_decomposition(
            rows=all_control_rows,
            baseline_rows=baseline_rows,
            count_values=count_values,
        )
        norm_ratio_rows = [
            {
                "control": row["control"],
                "alpha": row["alpha"],
                "sample_id": row["sample_id"],
                "sample_index": row["sample_index"],
                "evidence_count": row["evidence_count"],
                "gold_answer": row["gold_answer"],
                "injected_count": row["injected_count"],
                "injection_norm": row["injection_norm"],
                "injection_sqrt_norm": row["injection_sqrt_norm"],
                "hidden_norm_mean": row["hidden_norm_mean"],
                "hidden_sqrt_energy": row["hidden_sqrt_energy"],
                "hidden_site_count": row["hidden_site_count"],
                "norm_ratio": row["norm_ratio"],
                "norm_ratio_to_mean_hidden": row["norm_ratio_to_mean_hidden"],
            }
            for row in all_control_rows
        ]

        print("Computing LM-head alignment")
        alignment_rows = lm_head_alignment_rows(
            model=model,
            adapter=trained_adapter.cpu(),
            count_token_ids=count_token_ids,
        )

        alpha_sweep_rows = [row for row in summary_rows if row["control"] == CONTROL_TRAINED]
        write_rows(
            output_dir / "alpha_sweep_metrics.csv",
            alpha_sweep_rows,
            [
                "control",
                "alpha",
                "n",
                "acc",
                "overall_accuracy",
                "mean_margin",
                "mean_gold_logit",
                "mean_pred_logit",
                "mean_injection_norm",
                "mean_injection_sqrt_norm",
                "mean_hidden_norm",
                "mean_norm_ratio",
                "acc_by_evidence_count",
            ],
        )
        write_rows(
            output_dir / "trained_codebook_metrics.csv",
            trained_rows,
            ["control", "alpha", "sample_id", "evidence_count", "gold_answer", "pred_answer", "correct"],
        )
        write_rows(
            output_dir / "shuffled_count_metrics.csv",
            shuffled_rows,
            [
                "control",
                "alpha",
                "sample_id",
                "evidence_count",
                "gold_answer",
                "injected_count",
                "pred_answer",
                "correct",
                "pred_matches_injected",
                "margin",
                "injection_norm",
            ],
        )
        write_rows(
            output_dir / "random_same_norm_metrics.csv",
            random_rows,
            ["control", "alpha", "sample_id", "evidence_count", "gold_answer", "pred_answer", "correct", "margin", "injection_norm"],
        )
        write_rows(
            output_dir / "lm_head_alignment.csv",
            alignment_rows,
            [
                "count",
                "layer",
                "norm",
                "max_cos_to_answer_unembedding",
                "cos_to_own_answer",
                "mean_cos_to_other_answers",
                "own_minus_other_gap",
            ],
        )
        write_rows(
            output_dir / "logit_decomposition.csv",
            logit_rows,
            [
                "control",
                "alpha",
                "n",
                "delta_gold_logit",
                "mean_delta_non_gold_logits",
                "max_delta_non_gold_logit",
                "delta_margin",
            ],
        )
        write_rows(
            output_dir / "norm_ratios.csv",
            norm_ratio_rows,
            [
                "control",
                "alpha",
                "sample_id",
                "evidence_count",
                "gold_answer",
                "injected_count",
                "injection_norm",
                "injection_sqrt_norm",
                "hidden_norm_mean",
                "hidden_sqrt_energy",
                "norm_ratio",
            ],
        )
        write_rows(
            output_dir / "summary.csv",
            summary_rows,
            [
                "control",
                "alpha",
                "n",
                "acc",
                "mean_margin",
                "mean_gold_logit",
                "mean_pred_logit",
                "mean_injection_norm",
                "mean_norm_ratio",
                "acc_by_evidence_count",
            ],
        )
        write_rows(
            output_dir / "accuracy_by_evidence_count.csv",
            per_count_rows,
            ["control", "alpha", "evidence_count", "n", "accuracy", "mean_margin", "mean_injection_norm", "mean_norm_ratio"],
        )
        debug = {
            "runtime_seconds": time.time() - started,
            "source_cache": os.fspath(sample_payload["cache_path"]),
            "num_records": len(records),
            "limited_splits": {"test": len(test_indices)},
            "output_dir": os.fspath(output_dir),
            "checkpoint_path": os.fspath(checkpoint_path),
        }
        trans.prev.write_json(output_dir / "debug.json", debug)
        if not bool(args.no_plots):
            make_plots(
                output_dir=output_dir,
                summary_rows=summary_rows,
                per_count_rows=per_count_rows,
                alignment_rows=alignment_rows,
            )

        print(f"Finished {EXPERIMENT_NAME} in {time.time() - started:.1f}s")
        print(f"Output diagnostic directory: {output_dir}")
        print(f"Exact command used: {exact_command}")
        print_quick_summary(summary_rows)
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
