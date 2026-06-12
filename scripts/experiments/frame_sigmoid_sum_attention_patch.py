#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.model import get_layers
from scripts.experiments import pnamix_clean_aggregation_lora as text_base
from scripts.experiments import visual_fixed8_count_sweep_lora as visual_base


EXPERIMENT_NAME = "frame_sigmoid_sum_attention_patch"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
PATCH_LAYERS = (14, 15, 16, 17)

FROZEN_QWEN_BASELINE = "frozen_qwen_baseline"
FRAME_SOFTMAX_PATCH = "frame_softmax_patch"
FRAME_SIGMOID_SUM_PATCH = "frame_sigmoid_sum_patch"
LORA_14_17_BASELINE = "lora_14_17_baseline"
LORA_14_17_FRAME_SIGMOID_SUM_PATCH = "lora_14_17_frame_sigmoid_sum_patch"
LORA_14_17_FRAME_SOFTMAX_PATCH = "lora_14_17_frame_softmax_patch"

VARIANTS = (
    FROZEN_QWEN_BASELINE,
    FRAME_SOFTMAX_PATCH,
    FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_BASELINE,
    LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_FRAME_SOFTMAX_PATCH,
)
DEFAULT_VARIANTS = (
    FROZEN_QWEN_BASELINE,
    FRAME_SOFTMAX_PATCH,
    FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_BASELINE,
    LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
)
VARIANT_ALIASES = {
    "frozen": FROZEN_QWEN_BASELINE,
    FROZEN_QWEN_BASELINE: FROZEN_QWEN_BASELINE,
    "softmax": FRAME_SOFTMAX_PATCH,
    FRAME_SOFTMAX_PATCH: FRAME_SOFTMAX_PATCH,
    "sigmoid": FRAME_SIGMOID_SUM_PATCH,
    "sigmoid_sum": FRAME_SIGMOID_SUM_PATCH,
    FRAME_SIGMOID_SUM_PATCH: FRAME_SIGMOID_SUM_PATCH,
    "lora": LORA_14_17_BASELINE,
    "lora_14_17": LORA_14_17_BASELINE,
    LORA_14_17_BASELINE: LORA_14_17_BASELINE,
    "lora_sigmoid": LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
    "lora_14_17_sigmoid": LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_FRAME_SIGMOID_SUM_PATCH: LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
    "lora_softmax": LORA_14_17_FRAME_SOFTMAX_PATCH,
    "lora_14_17_softmax": LORA_14_17_FRAME_SOFTMAX_PATCH,
    LORA_14_17_FRAME_SOFTMAX_PATCH: LORA_14_17_FRAME_SOFTMAX_PATCH,
}

PATCH_VARIANTS = {
    FRAME_SOFTMAX_PATCH,
    FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_FRAME_SOFTMAX_PATCH,
}
SIGMOID_PATCH_VARIANTS = {
    FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
}
SOFTMAX_PATCH_VARIANTS = {
    FRAME_SOFTMAX_PATCH,
    LORA_14_17_FRAME_SOFTMAX_PATCH,
}
LORA_VARIANTS = {
    LORA_14_17_BASELINE,
    LORA_14_17_FRAME_SIGMOID_SUM_PATCH,
    LORA_14_17_FRAME_SOFTMAX_PATCH,
}


def parse_bool(value: Any) -> bool:
    return visual_base.parse_bool(value)


def split_tokens(raw_values: Sequence[Any]) -> List[str]:
    return visual_base.split_tokens(raw_values)


def parse_int_tokens(raw_values: Sequence[Any]) -> List[int]:
    return visual_base.parse_int_tokens(raw_values)


def parse_variants(raw_values: Sequence[Any]) -> List[str]:
    variants: List[str] = []
    for token in split_tokens(raw_values):
        if token not in VARIANT_ALIASES:
            raise ValueError(f"Unknown variant {token!r}; valid values are {sorted(VARIANT_ALIASES)}")
        variants.append(VARIANT_ALIASES[token])
    return list(dict.fromkeys(variants))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5-VL fixed-8 visual counting with frame-level softmax/sigmoid-sum "
            "attention aggregation patches on carrier tokens."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--source-dataset-root", type=Path, default=visual_base.DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--variants", nargs="+", default=[",".join(DEFAULT_VARIANTS)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=24680)
    parser.add_argument("--train-per-count", "--train_per_count", dest="train_per_count", type=int, default=30)
    parser.add_argument("--val-per-count", "--val_per_count", dest="val_per_count", type=int, default=10)
    parser.add_argument(
        "--iid-test-per-count",
        "--iid_test_per_count",
        dest="iid_test_per_count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--high-test-per-count",
        "--high_test_per_count",
        dest="high_test_per_count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--all-count-test-per-count",
        "--all_count_test_per_count",
        dest="all_count_test_per_count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--include-heldout",
        "--include_heldout",
        dest="include_heldout",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--include-long",
        "--include_long",
        dest="include_long",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Kept for compatibility; this fixed-8 patch experiment rejects long splits.",
    )
    parser.add_argument("--force-regenerate-dataset", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patch-lr", "--patch_lr", dest="patch_lr", type=float, default=1e-2)
    parser.add_argument("--lora-lr", "--lora_lr", dest="lora_lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--patch-layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj,k_proj,v_proj,o_proj"])
    parser.add_argument("--message-mode", choices=["auto", "exact", "approx"], default="auto")
    parser.add_argument("--patch-gamma-init", type=float, default=0.04)

    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-qlora-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--tiny-debug-model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", type=int, default=2)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--submit-mode", default="local")
    return parser.parse_args()


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    return visual_base.finite_mean(values, default=default)


def json_compact(value: Any) -> str:
    return visual_base.json_compact(value)


def image_token_id_from_processor(processor: Any) -> Optional[int]:
    token_id = getattr(processor, "image_token_id", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token_id is None and tokenizer is not None:
        token_id = getattr(tokenizer, "image_token_id", None)
    if token_id is None and tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return None if token_id is None else int(token_id)


def inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("inverse_softplus requires a positive value")
    return math.log(math.expm1(value))


def init_raw_gamma(effective_gamma: float) -> float:
    bounded = max(-0.199, min(0.199, float(effective_gamma)))
    return math.atanh(bounded / 0.2)


def dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.debug):
        train_per_count = min(int(args.train_per_count), 2)
        val_per_count = min(int(args.val_per_count), 1)
        iid_test_per_count = min(int(args.iid_test_per_count), 1)
        high_test_per_count = min(int(args.high_test_per_count), 1)
        all_count_test_per_count = min(int(args.all_count_test_per_count), 1)
    else:
        train_per_count = int(args.train_per_count)
        val_per_count = int(args.val_per_count)
        iid_test_per_count = int(args.iid_test_per_count)
        high_test_per_count = int(args.high_test_per_count)
        all_count_test_per_count = int(args.all_count_test_per_count)
    per_count_values = {
        "train_per_count": train_per_count,
        "val_per_count": val_per_count,
        "iid_test_per_count": iid_test_per_count,
        "high_test_per_count": high_test_per_count,
        "all_count_test_per_count": all_count_test_per_count,
    }
    if any(value <= 0 for value in per_count_values.values()):
        raise ValueError(f"All per-count sizes must be positive: {per_count_values}")
    all_counts = list(range(9))
    splits: Dict[str, Dict[str, Any]] = {
        "train_fixed8_clean": {
            "num_frames": 8,
            "counts": all_counts,
            "examples_per_count": train_per_count,
            "templates": list(visual_base.TRAIN_TEMPLATES),
            "source_partition": "train",
        },
        "val_fixed8_clean": {
            "num_frames": 8,
            "counts": all_counts,
            "examples_per_count": val_per_count,
            "templates": list(visual_base.TRAIN_TEMPLATES),
            "source_partition": "val",
        },
        "test_fixed8_iid": {
            "num_frames": 8,
            "counts": all_counts,
            "examples_per_count": iid_test_per_count,
            "templates": list(visual_base.TRAIN_TEMPLATES),
            "source_partition": "test",
        },
        "test_fixed8_high_count": {
            "num_frames": 8,
            "counts": [6, 7, 8],
            "examples_per_count": high_test_per_count,
            "templates": list(visual_base.TRAIN_TEMPLATES),
            "source_partition": "test",
        },
        "test_fixed8_all_counts": {
            "num_frames": 8,
            "counts": all_counts,
            "examples_per_count": all_count_test_per_count,
            "templates": list(visual_base.TRAIN_TEMPLATES),
            "source_partition": "test",
        },
    }
    if bool(args.include_heldout):
        splits["test_fixed8_heldout_template"] = {
            "num_frames": 8,
            "counts": all_counts,
            "examples_per_count": all_count_test_per_count,
            "templates": list(visual_base.HELDOUT_TEMPLATES),
            "source_partition": "test",
        }
    return {
        "dataset_seed": int(args.dataset_seed),
        "source_dataset_root": os.fspath(Path(args.source_dataset_root).resolve()),
        "source_split": str(args.source_split),
        **per_count_values,
        "include_long": False,
        "include_heldout": bool(args.include_heldout),
        "splits": splits,
        "evidence_positions_randomized": True,
        "hard_semantic_distractors": False,
        "neutral_rule": "queried character absent and queried room empty",
        "real_visual_frames": True,
        "closed_labels_0_to_8": True,
    }


def ensure_dataset(
    args: argparse.Namespace,
    dataset_base: Path,
) -> Tuple[Path, Dict[str, List[visual_base.VisualExample]], Dict[str, Any]]:
    original_dataset_config = visual_base.dataset_config
    visual_base.dataset_config = dataset_config
    try:
        dataset_dir, examples_by_split, manifest = visual_base.ensure_dataset(args, dataset_base)
    finally:
        visual_base.dataset_config = original_dataset_config
    expected = set(range(9))
    for split in ("train_fixed8_clean", "val_fixed8_clean", "test_fixed8_iid", "test_fixed8_all_counts"):
        observed = {int(example.gold_count) for example in examples_by_split.get(split, [])}
        if observed != expected:
            raise RuntimeError(f"{split}: expected closed labels 0..8, found {sorted(observed)}")
    high_counts = {int(example.gold_count) for example in examples_by_split.get("test_fixed8_high_count", [])}
    if high_counts != {6, 7, 8}:
        raise RuntimeError(f"test_fixed8_high_count: expected labels 6..8, found {sorted(high_counts)}")
    return dataset_dir, examples_by_split, manifest


class FrameAttentionPatch(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        inject_layers: Sequence[int],
        num_heads: int,
        image_token_id: Optional[int],
        message_mode: str,
        gamma_init: float,
    ) -> None:
        super().__init__()
        if mode not in {"frame_softmax", "frame_sigmoid_sum"}:
            raise ValueError(mode)
        self.mode = str(mode)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.layer_to_pos = {layer: pos for pos, layer in enumerate(self.inject_layers)}
        self.num_heads = int(num_heads)
        self.image_token_id = None if image_token_id is None else int(image_token_id)
        self.message_mode = str(message_mode)
        shape = (len(self.inject_layers), self.num_heads)
        if self.mode == "frame_sigmoid_sum":
            self.threshold = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.zeros(shape, dtype=torch.float32))
        self.raw_temperature = nn.Parameter(
            torch.full(shape, inverse_softplus(1.0 - 1e-4), dtype=torch.float32)
        )
        self.raw_gamma = nn.Parameter(torch.full(shape, init_raw_gamma(gamma_init), dtype=torch.float32))
        self._carrier_positions: Optional[List[List[int]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._prompt_last_indices: Optional[List[int]] = None
        self._handles: List[Any] = []
        self._last_stats: Dict[str, Dict[str, Any]] = {}
        self.hook_fire_counts: Dict[int, int] = defaultdict(int)
        self.message_mode_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_counts: Dict[str, int] = defaultdict(int)
        self.exact_failure_examples: List[str] = []

    def set_context(self, batch: visual_base.VisualBatch) -> None:
        self._carrier_positions = [[int(position) for position in row] for row in batch.carrier_positions]
        self._frame_groups = [
            [[int(position) for position in group] for group in row]
            for row in batch.frame_groups
        ]
        self._prompt_last_indices = [int(value) for value in batch.prompt_last_indices.detach().cpu().tolist()]
        input_ids = batch.inputs.get("input_ids")
        for batch_idx, groups in enumerate(self._frame_groups):
            errors: List[str] = []
            if len(groups) != 8 or any(not group for group in groups):
                errors.append(f"expected exactly 8 non-empty visual frame groups; got {[len(group) for group in groups]}")
            carriers = self._carrier_positions[batch_idx]
            if not carriers:
                errors.append("no room/character carrier token positions")
            prompt_last = self._prompt_last_indices[batch_idx]
            if prompt_last in carriers:
                errors.append("final prompt token was selected as a patch carrier")
            flattened = [position for group in groups for position in group]
            if self.image_token_id is not None and torch.is_tensor(input_ids):
                row_ids = input_ids[batch_idx].detach()
                bad = [
                    position
                    for position in flattened
                    if position < 0
                    or position >= int(row_ids.numel())
                    or int(row_ids[position].item()) != int(self.image_token_id)
                ]
                if bad:
                    errors.append(f"source frame groups include non-image-pad token positions: {bad[:10]}")
            if errors:
                raise AssertionError(f"FrameAttentionPatch context invalid for row={batch_idx}: {'; '.join(errors)}")
        self._last_stats = {
            "frame_patch_gamma_by_layer": {},
            "frame_patch_temperature_by_layer": {},
            "frame_patch_update_norm_by_layer": {},
            "frame_patch_delta_context_norm_by_layer": {},
            "frame_patch_visual_context_norm_by_layer": {},
            "frame_patch_normal_visual_context_norm_by_layer": {},
            "frame_patch_gate_mass_by_layer": {},
            "frame_patch_message_mode_by_layer": {},
        }

    def clear_context(self) -> None:
        self._carrier_positions = None
        self._frame_groups = None
        self._prompt_last_indices = None

    @staticmethod
    def _hidden_from_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        if args and torch.is_tensor(args[0]):
            return args[0]
        hidden = kwargs.get("hidden_states")
        return hidden if torch.is_tensor(hidden) else None

    @staticmethod
    def _repeat_kv(states: torch.Tensor, num_heads: int) -> torch.Tensor:
        if int(states.shape[1]) == int(num_heads):
            return states
        repeats = int(num_heads) // int(states.shape[1])
        return states.repeat_interleave(repeats, dim=1)

    @staticmethod
    def _base_layer(module: nn.Module) -> nn.Module:
        return getattr(module, "base_layer", module)

    @staticmethod
    def _replace_output(output: Any, patched_attn_output: torch.Tensor) -> Any:
        if torch.is_tensor(output):
            return patched_attn_output
        if isinstance(output, tuple):
            return (patched_attn_output,) + tuple(output[1:])
        if isinstance(output, list):
            return [patched_attn_output, *list(output[1:])]
        raise RuntimeError(f"Unsupported self-attention output type: {type(output).__name__}")

    @staticmethod
    def _first_output_tensor(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise RuntimeError(f"Unsupported self-attention output type: {type(output).__name__}")

    def _record_exact_failure(self, reason: str) -> None:
        key = str(reason).split(":", 1)[0][:100]
        self.exact_failure_counts[key] += 1
        if len(self.exact_failure_examples) < 20:
            self.exact_failure_examples.append(str(reason)[:500])
        if sum(self.exact_failure_counts.values()) <= 3:
            print(f"[frame_patch] exact message computation unavailable; using approx: {reason}")

    def _qkv_with_rope(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        kwargs: Dict[str, Any],
        *,
        require_exact: bool,
        force_approx: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        batch, seq_len, _hidden = hidden_states.shape
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)
        head_dim = int(getattr(attn, "head_dim", q.shape[-1] // int(getattr(attn, "num_heads", 1))))
        num_heads = int(getattr(attn, "num_heads", q.shape[-1] // head_dim))
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        mode = "approx"
        if not force_approx and (require_exact or self.message_mode in {"auto", "exact"}):
            position_embeddings = kwargs.get("position_embeddings")
            if (
                text_base.apply_multimodal_rotary_pos_emb is None
                or position_embeddings is None
                or not hasattr(attn, "rope_scaling")
            ):
                if require_exact or self.message_mode == "exact":
                    raise RuntimeError("exact Qwen multimodal RoPE unavailable")
            else:
                q, k = text_base.apply_multimodal_rotary_pos_emb(
                    q,
                    k,
                    position_embeddings[0],
                    position_embeddings[1],
                    attn.rope_scaling["mrope_section"],
                )
                mode = "exact"
        k = self._repeat_kv(k, num_heads)
        v = self._repeat_kv(v, num_heads)
        scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
        return q, k, v, scaling, mode

    def _qkv_with_fallback(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
        try:
            return self._qkv_with_rope(
                attn,
                hidden_states,
                kwargs,
                require_exact=self.message_mode == "exact",
            )
        except Exception as exc:
            self._record_exact_failure(f"layer {layer_idx}: {type(exc).__name__}: {exc}")
            if self.message_mode == "exact":
                raise
            return self._qkv_with_rope(
                attn,
                hidden_states,
                kwargs,
                require_exact=False,
                force_approx=True,
            )

    def _apply_query_masks(
        self,
        *,
        scores: torch.Tensor,
        attn: Any,
        kwargs: Dict[str, Any],
        batch_idx: int,
        carrier_idx: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        arange = torch.arange(seq_len, device=device)
        causal_allowed = arange.unsqueeze(0) <= carrier_idx.unsqueeze(1)
        sliding_window = getattr(attn, "sliding_window", None)
        if sliding_window is not None:
            causal_allowed &= arange.unsqueeze(0) >= carrier_idx.unsqueeze(1) - int(sliding_window)
        masked = scores.masked_fill(~causal_allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
        attention_mask = kwargs.get("attention_mask")
        if torch.is_tensor(attention_mask):
            if attention_mask.dim() == 4:
                if int(attention_mask.shape[-2]) == 1:
                    selected_mask = attention_mask[batch_idx : batch_idx + 1, :, :1, :].float()
                else:
                    selected_mask = attention_mask[batch_idx : batch_idx + 1, :, carrier_idx, :].float()
                masked = masked + selected_mask.squeeze(0)
            elif attention_mask.dim() == 2:
                valid = attention_mask[batch_idx].bool()
                masked = masked.masked_fill(~valid.view(1, 1, -1), torch.finfo(masked.dtype).min)
        return masked

    def _project_delta(self, attn: Any, delta_context: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        o_proj = self._base_layer(attn.o_proj)
        weight = getattr(o_proj, "weight", None)
        dtype = weight.dtype if torch.is_tensor(weight) and weight.dtype.is_floating_point else output_dtype
        projected = o_proj(delta_context.unsqueeze(0).to(dtype=dtype)).squeeze(0)
        return projected.to(dtype=output_dtype)

    def patch_output(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        output: Any,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Any:
        if self._carrier_positions is None or self._frame_groups is None or self._prompt_last_indices is None:
            return output
        layer_pos = self.layer_to_pos[int(layer_idx)]
        attn_output = self._first_output_tensor(output)
        batch, seq_len, _hidden = hidden_states.shape
        q, k, v, scaling, mode = self._qkv_with_fallback(attn, hidden_states, int(layer_idx), kwargs)
        num_heads = int(q.shape[1])
        head_dim = int(q.shape[-1])
        if num_heads != self.num_heads:
            raise RuntimeError(f"Configured num_heads={self.num_heads}, but layer {layer_idx} produced {num_heads}")
        patched = attn_output.clone()
        gamma_heads = 0.2 * torch.tanh(self.raw_gamma[layer_pos].float())
        temperature_heads = F.softplus(self.raw_temperature[layer_pos].float()) + 1e-4
        threshold_heads = self.threshold[layer_pos].float()
        self.hook_fire_counts[int(layer_idx)] += 1
        self.message_mode_counts[f"{int(layer_idx)}:{mode}"] += int(batch)

        per_sample_update_norm: List[float] = []
        per_sample_delta_norm: List[float] = []
        per_sample_visual_norm: List[float] = []
        per_sample_normal_norm: List[float] = []
        per_sample_gate_mass: List[float] = []

        for batch_idx in range(batch):
            carriers = [
                int(position)
                for position in self._carrier_positions[batch_idx]
                if 0 <= int(position) < seq_len and int(position) != self._prompt_last_indices[batch_idx]
            ]
            groups = self._frame_groups[batch_idx]
            if not carriers:
                per_sample_update_norm.append(0.0)
                per_sample_delta_norm.append(0.0)
                per_sample_visual_norm.append(0.0)
                per_sample_normal_norm.append(0.0)
                per_sample_gate_mass.append(0.0)
                continue
            carrier_idx = torch.tensor(carriers, device=hidden_states.device, dtype=torch.long)
            scores = torch.einsum(
                "hcd,hsd->hcs",
                q[batch_idx, :, carrier_idx, :].float(),
                k[batch_idx].float(),
            ) * float(scaling)
            scores = self._apply_query_masks(
                scores=scores,
                attn=attn,
                kwargs=kwargs,
                batch_idx=batch_idx,
                carrier_idx=carrier_idx,
                seq_len=seq_len,
                device=hidden_states.device,
            )
            visual_idx = torch.tensor(
                [position for group in groups for position in group],
                device=hidden_states.device,
                dtype=torch.long,
            )
            visual_scores_all = scores[:, :, visual_idx]
            visual_values_all = v[batch_idx, :, visual_idx, :].float()
            normal_weights = torch.softmax(visual_scores_all, dim=-1)
            normal_context = torch.einsum("hct,htd->hcd", normal_weights, visual_values_all)

            sample_updates: List[float] = []
            sample_deltas: List[float] = []
            sample_visuals: List[float] = []
            sample_normals: List[float] = []
            sample_gate_masses: List[float] = []
            for carrier_row, carrier_pos in enumerate(carriers):
                frame_scores: List[torch.Tensor] = []
                frame_values: List[torch.Tensor] = []
                for group in groups:
                    frame_idx = torch.tensor(group, device=hidden_states.device, dtype=torch.long)
                    token_scores = scores[:, carrier_row, frame_idx]
                    token_values = v[batch_idx, :, frame_idx, :].float()
                    weights = torch.softmax(token_scores, dim=-1)
                    frame_scores.append(torch.logsumexp(token_scores, dim=-1))
                    frame_values.append(torch.einsum("ht,htd->hd", weights, token_values))
                stacked_scores = torch.stack(frame_scores, dim=-1)
                stacked_values = torch.stack(frame_values, dim=1)
                if self.mode == "frame_softmax":
                    frame_weight = torch.softmax(stacked_scores / temperature_heads.view(num_heads, 1), dim=-1)
                    patched_context = torch.einsum("hf,hfd->hd", frame_weight, stacked_values)
                    gate_mass = frame_weight.sum(dim=-1)
                else:
                    gates = torch.sigmoid(
                        (stacked_scores - threshold_heads.view(num_heads, 1))
                        / temperature_heads.view(num_heads, 1)
                    )
                    patched_context = torch.einsum("hf,hfd->hd", gates, stacked_values)
                    gate_mass = gates.sum(dim=-1)
                normal_carrier_context = normal_context[:, carrier_row, :]
                delta_heads = patched_context - normal_carrier_context
                delta_flat = delta_heads.contiguous().view(num_heads * head_dim)
                update_heads = gamma_heads.view(num_heads, 1) * delta_heads
                update_flat = update_heads.contiguous().view(num_heads * head_dim)
                projected_update = self._project_delta(attn, update_flat, patched.dtype)
                patched[batch_idx, int(carrier_pos), :] = patched[batch_idx, int(carrier_pos), :] + projected_update
                sample_updates.append(float(projected_update.detach().float().norm().cpu().item()))
                sample_deltas.append(float(delta_flat.detach().float().norm().cpu().item()))
                sample_visuals.append(float(patched_context.detach().float().norm().cpu().item()))
                sample_normals.append(float(normal_carrier_context.detach().float().norm().cpu().item()))
                sample_gate_masses.append(float(gate_mass.detach().float().mean().cpu().item()))
            per_sample_update_norm.append(finite_mean(sample_updates, default=0.0))
            per_sample_delta_norm.append(finite_mean(sample_deltas, default=0.0))
            per_sample_visual_norm.append(finite_mean(sample_visuals, default=0.0))
            per_sample_normal_norm.append(finite_mean(sample_normals, default=0.0))
            per_sample_gate_mass.append(finite_mean(sample_gate_masses, default=0.0))

        key = str(int(layer_idx))
        self._last_stats["frame_patch_gamma_by_layer"][key] = [
            float(gamma_heads.detach().mean().cpu().item())
        ] * batch
        self._last_stats["frame_patch_temperature_by_layer"][key] = [
            float(temperature_heads.detach().mean().cpu().item())
        ] * batch
        self._last_stats["frame_patch_update_norm_by_layer"][key] = per_sample_update_norm
        self._last_stats["frame_patch_delta_context_norm_by_layer"][key] = per_sample_delta_norm
        self._last_stats["frame_patch_visual_context_norm_by_layer"][key] = per_sample_visual_norm
        self._last_stats["frame_patch_normal_visual_context_norm_by_layer"][key] = per_sample_normal_norm
        self._last_stats["frame_patch_gate_mass_by_layer"][key] = per_sample_gate_mass
        self._last_stats["frame_patch_message_mode_by_layer"][key] = [mode] * batch
        return self._replace_output(output, patched)

    def attach(self, model: Any) -> None:
        self.detach()
        layers = get_layers(model)
        for layer_idx in self.inject_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"patch layer={layer_idx} outside [0, {len(layers) - 1}]")
            attn = getattr(layers[int(layer_idx)], "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer={layer_idx} does not expose self_attn")

            def hook(
                module: Any,
                args: Tuple[Any, ...],
                kwargs: Dict[str, Any],
                output: Any,
                *,
                layer: int = int(layer_idx),
            ) -> Any:
                hidden = self._hidden_from_args(args, kwargs)
                if hidden is None:
                    return output
                return self.patch_output(module, hidden, output, layer, kwargs)

            self._handles.append(attn.register_forward_hook(hook, with_kwargs=True))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, by_layer in self._last_stats.items():
            row_payload: Dict[str, Any] = {}
            for layer, values in by_layer.items():
                row_payload[layer] = values[row] if isinstance(values, list) and row < len(values) else values
            out[name] = row_payload
        out["frame_patch_gamma"] = finite_mean(out.get("frame_patch_gamma_by_layer", {}).values(), default=0.0)
        out["frame_patch_temperature"] = finite_mean(
            out.get("frame_patch_temperature_by_layer", {}).values(),
            default=0.0,
        )
        out["frame_patch_update_norm"] = finite_mean(
            out.get("frame_patch_update_norm_by_layer", {}).values(),
            default=0.0,
        )
        out["frame_patch_gate_mass"] = finite_mean(
            out.get("frame_patch_gate_mass_by_layer", {}).values(),
            default=0.0,
        )
        return out


class VariantAdapter(nn.Module):
    def __init__(
        self,
        *,
        lora: Optional[text_base.MinimalAttentionLoRAAdapter],
        patch: Optional[FrameAttentionPatch],
    ) -> None:
        super().__init__()
        self.lora = lora
        self.patch = patch

    def attach(self, model: Any) -> None:
        if self.lora is not None:
            self.lora.attach(model)
        if self.patch is not None:
            self.patch.attach(model)

    def detach(self) -> None:
        if self.patch is not None:
            self.patch.detach()
        if self.lora is not None:
            self.lora.detach()

    def set_context(self, batch: visual_base.VisualBatch) -> None:
        if self.lora is not None:
            self.lora.set_context(batch)
        if self.patch is not None:
            self.patch.set_context(batch)

    def clear_context(self) -> None:
        if self.patch is not None:
            self.patch.clear_context()
        if self.lora is not None:
            self.lora.clear_context()

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if self.lora is not None:
            stats.update(self.lora.stats_for_row(row))
        if self.patch is not None:
            stats.update(self.patch.stats_for_row(row))
        return stats


def infer_num_heads(model: Any, layers: Sequence[int]) -> int:
    model_layers = get_layers(model)
    for layer_idx in layers:
        attn = getattr(model_layers[int(layer_idx)], "self_attn", None)
        if attn is not None and hasattr(attn, "num_heads"):
            return int(getattr(attn, "num_heads"))
    raise RuntimeError("Could not infer attention num_heads from requested patch layers")


def make_variant_adapter(
    *,
    variant: str,
    args: argparse.Namespace,
    lora_layers: Sequence[int],
    patch_layers: Sequence[int],
    lora_targets: Sequence[str],
    num_heads: int,
    image_token_id: Optional[int],
) -> Optional[VariantAdapter]:
    if variant == FROZEN_QWEN_BASELINE:
        return None
    lora: Optional[text_base.MinimalAttentionLoRAAdapter] = None
    if variant in LORA_VARIANTS:
        lora = text_base.MinimalAttentionLoRAAdapter(
            inject_layers=lora_layers,
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target_modules=lora_targets,
        )
    patch: Optional[FrameAttentionPatch] = None
    if variant in PATCH_VARIANTS:
        patch = FrameAttentionPatch(
            mode="frame_sigmoid_sum" if variant in SIGMOID_PATCH_VARIANTS else "frame_softmax",
            inject_layers=patch_layers,
            num_heads=int(num_heads),
            image_token_id=image_token_id,
            message_mode=str(args.message_mode),
            gamma_init=float(args.patch_gamma_init),
        )
    return VariantAdapter(lora=lora, patch=patch)


def count_trainable_parameters(module: Optional[nn.Module]) -> int:
    return 0 if module is None else int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def optimizer_groups(adapter: VariantAdapter, args: argparse.Namespace) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if adapter.lora is not None:
        params = [param for param in adapter.lora.parameters() if param.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(args.lora_lr), "name": "lora"})
    if adapter.patch is not None:
        params = [param for param in adapter.patch.parameters() if param.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(args.patch_lr), "name": "patch"})
    return groups


def freeze_model(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def trainable_parameter_report(model: Any, adapter: Optional[VariantAdapter]) -> Dict[str, Any]:
    model_trainable = [
        {"name": name, "numel": int(param.numel())}
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    adapter_trainable = []
    if adapter is not None:
        adapter_trainable = [
            {"name": name, "numel": int(param.numel())}
            for name, param in adapter.named_parameters()
            if param.requires_grad
        ]
    unexpected_model = [
        row["name"]
        for row in model_trainable
        if "lora_A" not in row["name"] and "lora_B" not in row["name"]
    ]
    if unexpected_model:
        raise RuntimeError(f"Unexpected trainable base-model parameters: {unexpected_model[:20]}")
    unexpected_adapter = [
        row["name"]
        for row in adapter_trainable
        if not (
            row["name"].startswith("lora.")
            or row["name"].startswith("patch.threshold")
            or row["name"].startswith("patch.raw_temperature")
            or row["name"].startswith("patch.raw_gamma")
        )
    ]
    if unexpected_adapter:
        raise RuntimeError(f"Unexpected trainable adapter parameters: {unexpected_adapter[:20]}")
    return {
        "model_trainable": model_trainable,
        "adapter_trainable": adapter_trainable,
        "model_trainable_total": int(sum(row["numel"] for row in model_trainable)),
        "adapter_trainable_total": int(sum(row["numel"] for row in adapter_trainable)),
        "allowed_trainable_parameter_families": ["lora_A", "lora_B", "patch.threshold", "patch.raw_temperature", "patch.raw_gamma"],
        "base_qwen_weights_trainable": False,
        "lm_head_trainable": False,
        "vision_encoder_trainable": False,
    }


def train_adapter(
    *,
    variant: str,
    args: argparse.Namespace,
    run_dir: Path,
    model: Any,
    processor: Any,
    adapter: VariantAdapter,
    train_examples: Sequence[visual_base.VisualExample],
    val_examples: Sequence[visual_base.VisualExample],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    answer_ids: Dict[int, Tuple[int, ...]],
    digit_token_ids: Dict[int, int],
    device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    groups = optimizer_groups(adapter, args)
    trainable = [param for group in groups for param in group["params"]]
    if not trainable:
        raise RuntimeError(f"{variant}: no trainable LoRA/patch parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_adapter.pt"
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_accuracy = -math.inf
    best_val_ce = math.inf
    history: List[Dict[str, Any]] = []
    count_values = sorted(digit_token_ids)
    for epoch in range(1, int(args.epochs) + 1):
        model.eval()
        adapter.train()
        order = visual_base.batch_indices(train_indices, 1, int(args.seed) + epoch * 1009, True)
        optimizer.zero_grad(set_to_none=True)
        total_ce = 0.0
        correct = 0
        n = 0
        for step, idxs in enumerate(order, start=1):
            idx = int(idxs[0])
            example = train_examples[idx]
            batch = visual_base.prepare_batch(
                examples=[example],
                sample_indices=[idx],
                processor=processor,
                device=device,
                answer_ids=answer_ids,
            )
            adapter.set_context(batch)
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = text_base.select_count_logits(outputs.logits, batch.prompt_last_indices, digit_token_ids)
            loss, _ = visual_base.answer_sequence_cross_entropy(outputs.logits, batch)
            torch.autograd.backward(loss / max(1, int(args.grad_accum)))
            pred = int(count_values[int(count_logits[0].argmax().detach().cpu().item())])
            correct += int(pred == int(example.gold_count))
            n += 1
            total_ce += float(loss.detach().cpu().item())
            adapter.clear_context()
            if step % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 25 == 0:
                print(
                    f"  {variant} epoch={epoch} step={step}/{len(order)} "
                    f"train_ce={total_ce / max(1, n):.4f} train_acc={correct / max(1, n):.4f}"
                )
        if len(order) % max(1, int(args.grad_accum)):
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        val_result = visual_base.evaluate_split(
            variant=variant,
            split_name="val_fixed8_clean",
            model=model,
            processor=processor,
            adapter=adapter,
            examples=val_examples,
            indices=val_indices,
            answer_ids=answer_ids,
            digit_token_ids=digit_token_ids,
            device=device,
            seed=int(args.seed) + 5000 + epoch,
        )
        history_row = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": total_ce / max(1, n),
            "train_ce": total_ce / max(1, n),
            "train_accuracy": correct / max(1, n),
            "train_steps": len(order),
            "val_ce": float(val_result["ce"]),
            "val_accuracy": float(val_result["accuracy"]),
            "val_mae": float(val_result["mae"]),
            "patch_lr": float(args.patch_lr),
            "lora_lr": float(args.lora_lr),
            "trainable_parameters": count_trainable_parameters(adapter),
        }
        history.append(history_row)
        print(
            f"  {variant} epoch={epoch} train_ce={history_row['train_ce']:.4f} "
            f"train_acc={history_row['train_accuracy']:.4f} val_ce={history_row['val_ce']:.4f} "
            f"val_acc={history_row['val_accuracy']:.4f}"
        )
        improved = history_row["val_accuracy"] > best_val_accuracy + 1e-9 or (
            abs(history_row["val_accuracy"] - best_val_accuracy) <= 1e-9
            and history_row["val_ce"] < best_val_ce
        )
        if improved:
            best_val_accuracy = float(history_row["val_accuracy"])
            best_val_ce = float(history_row["val_ce"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
            torch.save(
                {
                    "variant": variant,
                    "adapter_state_dict": best_state,
                    "history": history,
                    "note": "Adapter-only checkpoint; frozen Qwen weights are not stored.",
                },
                checkpoint_path,
            )
    if best_state is not None:
        adapter.load_state_dict(best_state)
    return history, checkpoint_path


def validate_args(args: argparse.Namespace, variants: Sequence[str], lora_layers: Sequence[int], patch_layers: Sequence[int]) -> None:
    if int(args.batch_size) != 1:
        raise ValueError("This visual experiment requires --batch-size 1")
    if bool(args.include_long):
        raise ValueError("frame_sigmoid_sum_attention_patch is fixed-8 only; do not pass --include-long")
    if int(args.candidate_min) != 0 or int(args.candidate_max) != 8:
        raise ValueError("This fixed-8 count experiment expects closed candidate labels 0..8")
    if not bool(args.tiny_debug_model):
        if tuple(lora_layers) != PATCH_LAYERS:
            raise ValueError(f"LoRA layers must be exactly {PATCH_LAYERS}; got {tuple(lora_layers)}")
        if tuple(patch_layers) != PATCH_LAYERS:
            raise ValueError(f"Patch layers must be exactly {PATCH_LAYERS}; got {tuple(patch_layers)}")
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"Unknown variants: {[variant for variant in variants if variant not in VARIANTS]}")
    lora_targets = set(split_tokens(args.lora_targets))
    if lora_targets != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError(f"LoRA targets must be q_proj,k_proj,v_proj,o_proj; got {sorted(lora_targets)}")


def write_patch_diagnostics(run_dir: Path, adapter: Optional[VariantAdapter]) -> None:
    if adapter is None or adapter.patch is None:
        return
    patch = adapter.patch
    gamma = 0.2 * torch.tanh(patch.raw_gamma.detach().float())
    temperature = F.softplus(patch.raw_temperature.detach().float()) + 1e-4
    payload = {
        "mode": patch.mode,
        "layers": list(patch.inject_layers),
        "hook_fire_counts": {str(key): value for key, value in sorted(patch.hook_fire_counts.items())},
        "message_mode_counts": dict(sorted(patch.message_mode_counts.items())),
        "exact_failure_counts": dict(sorted(patch.exact_failure_counts.items())),
        "exact_failure_examples": list(patch.exact_failure_examples),
        "learned_gamma_mean_by_layer": {
            str(layer): float(gamma[pos].mean().cpu().item())
            for pos, layer in enumerate(patch.inject_layers)
        },
        "learned_temperature_mean_by_layer": {
            str(layer): float(temperature[pos].mean().cpu().item())
            for pos, layer in enumerate(patch.inject_layers)
        },
    }
    if patch.mode == "frame_sigmoid_sum":
        threshold = patch.threshold.detach().float()
        payload["learned_threshold_mean_by_layer"] = {
            str(layer): float(threshold[pos].mean().cpu().item())
            for pos, layer in enumerate(patch.inject_layers)
        }
    visual_base.write_json(run_dir / "frame_attention_patch_diagnostics.json", payload)


def main() -> int:
    args = parse_args()
    variants = parse_variants(args.variants)
    lora_layers = parse_int_tokens(args.lora_layers)
    patch_layers = parse_int_tokens(args.patch_layers)
    lora_targets = split_tokens(args.lora_targets)
    validate_args(args, variants, lora_layers, patch_layers)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir, examples_by_split, dataset_manifest = ensure_dataset(args, output_root / "datasets")
    split_names = list(examples_by_split)
    print(f"Visual fixed-8 dataset: {dataset_dir}")
    for split, examples in examples_by_split.items():
        print(f"  {split}: n={len(examples)} counts={sorted({example.gold_count for example in examples})}")

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu --tiny-debug-model for debug")
    dtype = text_base.dtype_from_arg(str(args.dtype))
    model, processor, qlora_used, load_note = visual_base.load_model_and_processor(args, device, dtype)
    freeze_model(model)
    tokenizer = processor.tokenizer
    count_style, answer_ids = text_base.answer_token_ids(
        tokenizer,
        int(args.candidate_min),
        int(args.candidate_max),
    )
    digit_token_ids = visual_base.single_token_count_ids(answer_ids)
    hidden_size = text_base.hidden_size_from_model(model)
    layers = get_layers(model)
    num_heads = infer_num_heads(model, patch_layers)
    image_token_id = image_token_id_from_processor(processor)
    print(
        f"Loaded model={args.model_name} hidden_size={hidden_size} num_layers={len(layers)} "
        f"num_heads={num_heads} image_token_id={image_token_id} qlora_used={qlora_used} load_note={load_note}"
    )
    print(f"Count token style={count_style} ids={answer_ids}")

    train_indices = visual_base.limited_indices(
        examples_by_split["train_fixed8_clean"],
        int(args.max_train_examples),
        int(args.seed) + 101,
    )
    eval_indices: Dict[str, List[int]] = {}
    for pos, split in enumerate(split_names):
        limit = int(args.max_eval_examples)
        if split == "train_fixed8_clean":
            values = train_indices if limit <= 0 else train_indices[:limit]
        else:
            values = visual_base.limited_indices(examples_by_split[split], limit, int(args.seed) + 200 + pos)
        eval_indices[split] = values

    # Reuse visual_base summary/artifact writers, but teach their comparison plots this experiment's variants.
    visual_base.VARIANTS = VARIANTS
    all_split_rows: List[Dict[str, Any]] = []
    all_count_rows: List[Dict[str, Any]] = []
    for variant in variants:
        freeze_model(model)
        run_dir = visual_base.run_dir_for_variant(output_root, variant, str(args.run_prefix), bool(args.debug))
        handle, old_stdout, old_stderr = visual_base.setup_logging(run_dir)
        adapter: Optional[VariantAdapter] = None
        try:
            print(f"Starting variant={variant} run_dir={run_dir}")
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
            torch.manual_seed(int(args.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
            adapter = make_variant_adapter(
                variant=variant,
                args=args,
                lora_layers=lora_layers,
                patch_layers=patch_layers,
                lora_targets=lora_targets,
                num_heads=num_heads,
                image_token_id=image_token_id,
            )
            if adapter is not None:
                adapter.attach(model)
                adapter.to(device)
            report = trainable_parameter_report(model, adapter)
            trainable_parameters = count_trainable_parameters(adapter)
            config = {
                "experiment": EXPERIMENT_NAME,
                "variant": variant,
                "model_name": str(args.model_name),
                "run_dir": os.fspath(run_dir),
                "dataset_dir": os.fspath(dataset_dir),
                "dataset_manifest": dataset_manifest,
                "seed": int(args.seed),
                "dataset_seed": int(args.dataset_seed),
                "device": device,
                "dtype": str(args.dtype),
                "qlora_used": qlora_used,
                "load_note": load_note,
                "real_visual_frames": True,
                "runtime_assertions": {
                    "fixed8_required": True,
                    "exactly_8_nonempty_visual_frame_groups_required": True,
                    "source_groups_must_be_image_pad_tokens": True,
                    "room_character_carriers_required": True,
                    "final_prompt_token_patch_forbidden": True,
                },
                "lora": {
                    "enabled": variant in LORA_VARIANTS,
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "dropout": float(args.lora_dropout),
                    "layers": lora_layers,
                    "target_modules": lora_targets,
                    "base_qwen_weights_frozen": True,
                    "lm_head_trainable": False,
                    "vision_encoder_trainable": False,
                },
                "frame_attention_patch": {
                    "enabled": variant in PATCH_VARIANTS,
                    "mode": (
                        "frame_sigmoid_sum"
                        if variant in SIGMOID_PATCH_VARIANTS
                        else "frame_softmax"
                        if variant in SOFTMAX_PATCH_VARIANTS
                        else "none"
                    ),
                    "layers": patch_layers,
                    "source": "visual image-pad token spans grouped into exactly 8 real frames",
                    "target": "queried room and character carrier tokens only",
                    "patch_surface": "self-attention output; adds original_o_proj(delta_visual_context) only",
                    "message_mode": str(args.message_mode),
                    "gamma": "0.2 * tanh(raw_gamma)",
                    "initial_effective_gamma": float(args.patch_gamma_init),
                    "uses_gold_count": False,
                    "uses_evidence_labels": False,
                },
                "training": {
                    "epochs": 0 if variant == FROZEN_QWEN_BASELINE else int(args.epochs),
                    "patch_lr": float(args.patch_lr),
                    "lora_lr": float(args.lora_lr),
                    "batch_size": 1,
                    "grad_accum": int(args.grad_accum),
                    "grad_clip": float(args.grad_clip),
                    "loss": "Qwen answer-token cross entropy through frozen LM head",
                    "trainable_parameters": int(trainable_parameters),
                    "trainable_parameter_report": report,
                },
                "submit_mode": str(args.submit_mode),
                "debug": bool(args.debug),
                "tiny_debug_model": bool(args.tiny_debug_model),
            }
            visual_base.write_json(run_dir / "config.json", config)
            visual_base.write_json(run_dir / "trainable_parameters.json", report)
            print(f"Trainable parameters: {trainable_parameters:,}")
            print(
                "Trainable report: "
                f"model={report['model_trainable_total']:,} adapter={report['adapter_trainable_total']:,} "
                "allowed=LoRA and/or frame patch scalars"
            )
            history: List[Dict[str, Any]] = []
            if adapter is not None:
                history, checkpoint_path = train_adapter(
                    variant=variant,
                    args=args,
                    run_dir=run_dir,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    train_examples=examples_by_split["train_fixed8_clean"],
                    val_examples=examples_by_split["val_fixed8_clean"],
                    train_indices=train_indices,
                    val_indices=eval_indices["val_fixed8_clean"],
                    answer_ids=answer_ids,
                    digit_token_ids=digit_token_ids,
                    device=device,
                )
                print(f"Best checkpoint: {checkpoint_path}")
            else:
                print("Frozen Qwen baseline: evaluation only.")
            prediction_rows: List[Dict[str, Any]] = []
            for split in split_names:
                result = visual_base.evaluate_split(
                    variant=variant,
                    split_name=split,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    examples=examples_by_split[split],
                    indices=eval_indices[split],
                    answer_ids=answer_ids,
                    digit_token_ids=digit_token_ids,
                    device=device,
                    seed=int(args.seed) + 7000,
                )
                print(
                    f"  {split}: accuracy={result['accuracy']:.4f} "
                    f"mae={result['mae']:.4f} ce={result['ce']:.4f}"
                )
                prediction_rows.extend(result["rows"])
            split_rows, count_rows = visual_base.write_run_artifacts(
                run_dir=run_dir,
                variant=variant,
                rows=prediction_rows,
                history=history,
                split_names=split_names,
                no_plots=bool(args.no_plots),
            )
            write_patch_diagnostics(run_dir, adapter)
            all_split_rows.extend(split_rows)
            all_count_rows.extend(count_rows)
            print(f"Finished variant={variant}")
        finally:
            if adapter is not None:
                adapter.detach()
            visual_base.restore_logging(handle, old_stdout, old_stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary_dir = output_root / "debug" / "summary" if bool(args.debug) else output_root / "summary"
    visual_base.write_summary(summary_dir, all_split_rows, all_count_rows, bool(args.no_plots))
    print(f"Summary written to {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
