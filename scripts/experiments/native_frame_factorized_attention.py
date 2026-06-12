#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments import layerwise_frame_message_glstm as base
from models.model import get_layers
from scripts.experiments import layerwise_glstm_mechanism_ablation as scaffold


EXPERIMENT_NAME = "native_frame_factorized_attention"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DEFAULT_LAYERS = (14, 15, 16, 17)

NATIVE_ADDITIVE_SIGNED = "native_additive_signed"
NATIVE_REPLACE_VISUAL_SIGNED = "native_replace_visual_signed"
NATIVE_REPLACE_VISUAL_SOFTMAX = "native_replace_visual_softmax"
VARIANTS = (
    NATIVE_ADDITIVE_SIGNED,
    NATIVE_REPLACE_VISUAL_SIGNED,
    NATIVE_REPLACE_VISUAL_SOFTMAX,
)
VARIANT_ALIASES = {
    "all": "all",
    "additive": NATIVE_ADDITIVE_SIGNED,
    "additive_signed": NATIVE_ADDITIVE_SIGNED,
    "native_additive_signed": NATIVE_ADDITIVE_SIGNED,
    "replace": NATIVE_REPLACE_VISUAL_SIGNED,
    "replace_signed": NATIVE_REPLACE_VISUAL_SIGNED,
    "native_replace_visual_signed": NATIVE_REPLACE_VISUAL_SIGNED,
    "softmax": NATIVE_REPLACE_VISUAL_SOFTMAX,
    "replace_softmax": NATIVE_REPLACE_VISUAL_SOFTMAX,
    "native_replace_visual_softmax": NATIVE_REPLACE_VISUAL_SOFTMAX,
}

MISSING = math.nan
_ORIGINAL_RUN_VARIANT = scaffold.run_variant


FrameMemoryBatch = base.FrameMemoryBatch
ExperimentAdapter = base.ExperimentAdapter


def _install_constants() -> None:
    scaffold.EXPERIMENT_NAME = EXPERIMENT_NAME
    scaffold.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    scaffold.DEFAULT_LAYERS = DEFAULT_LAYERS
    scaffold.VARIANTS = VARIANTS
    scaffold.VARIANT_ALIASES = VARIANT_ALIASES
    scaffold.ASSOCIATIVE_VARIANTS = set()


_install_constants()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native frame-factorized visual-to-carrier attention ablation for Qwen visual counting."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--source-dataset-root", type=Path, default=base.DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--fallback-source-dataset-root", type=Path, default=base.DEFAULT_FALLBACK_SOURCE_DATASET_ROOT)
    parser.add_argument("--source-split", default="all_uniform")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--variants", nargs="+", default=[",".join(VARIANTS)])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=24680)
    parser.add_argument("--force-regenerate-dataset", action="store_true", default=False)
    parser.add_argument("--train-per-count", type=int, default=20)
    parser.add_argument("--val-per-count", type=int, default=20)
    parser.add_argument("--iid-test-per-count", type=int, default=20)
    parser.add_argument("--interpolation-test-per-count", type=int, default=20)
    parser.add_argument("--extrapolation-test-per-count", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--eval-candidate-scores", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--layers", nargs="+", default=["14,15,16,17"])
    parser.add_argument("--carrier-mode", default="room_character", choices=["room_character"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--frame-kv-lora", action="store_true", default=False)
    parser.add_argument("--reconstruction-tol", type=float, default=1e-4)
    parser.add_argument("--fail-on-reconstruction-error", action=argparse.BooleanOptionalAction, default=False)

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

    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--tiny-debug-model", action="store_true", default=False)
    parser.add_argument("--tiny-num-layers", type=int, default=18)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--prepare-dataset-only", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--no-aggregate-after-run", action="store_true", default=False)
    parser.add_argument("--submit-slurm", action="store_true", default=False)
    parser.add_argument("--skip-submit-smoke", action="store_true", default=False)
    return parser.parse_args()


def finite_float(value: Any) -> Optional[float]:
    return scaffold.finite_float(value)


def finite_mean(values: Iterable[Any], default: float = math.nan) -> float:
    return scaffold.finite_mean(values, default=default)


def _json_load_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str)):
        return value
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return None


class NativeFrameFactorizedAttention(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        layers: Sequence[int],
        additive_gamma_init: float = 1e-3,
        replacement_gamma_init: float = 0.0,
        reconstruction_tol: float = 1e-4,
        **_: Any,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        self.layers = [int(layer) for layer in layers]
        self.layer_to_pos = {int(layer): pos for pos, layer in enumerate(self.layers)}
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown native factorized attention variant {self.variant!r}")
        gamma_init = float(additive_gamma_init if self.variant == NATIVE_ADDITIVE_SIGNED else replacement_gamma_init)
        self.gamma = nn.Parameter(torch.full((len(self.layers),), gamma_init, dtype=torch.float32))
        self.reconstruction_tol = float(reconstruction_tol)
        self.enabled = True
        self.ablation_mode = "normal"
        self.ablation_seed = 0

        self._carrier_positions: Optional[List[List[int]]] = None
        self._carrier_identities: Optional[List[List[str]]] = None
        self._frame_groups: Optional[List[List[List[int]]]] = None
        self._frame_valid_mask: Optional[torch.Tensor] = None
        self._evidence_frame_indices: Optional[List[List[int]]] = None
        self._sample_ids: Optional[List[str]] = None
        self._slots = None
        self._original_forwards: List[Tuple[Any, Any]] = []
        self._hook_handles: List[Any] = []
        self._active_capture_layer: Optional[int] = None
        self._capture_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._last_stats: Dict[str, Dict[str, Any]] = {}
        self._last_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.hook_fire_counts: Dict[int, int] = defaultdict(int)

    def set_context(self, batch: FrameMemoryBatch) -> None:
        self._carrier_positions = [[int(pos) for pos in row] for row in batch.carrier_positions]
        self._carrier_identities = [[str(value) for value in row] for row in batch.carrier_identities]
        self._frame_groups = [[[int(pos) for pos in group] for group in row] for row in batch.frame_groups]
        self._frame_valid_mask = batch.frame_valid_mask
        self._evidence_frame_indices = [[int(x) for x in row] for row in batch.evidence_frame_indices]
        self._sample_ids = list(batch.sample_ids)
        self._slots = None
        self._capture_buffers = {}
        self._last_stats = {
            "within_frame_weight_sum_error_by_layer": {},
            "frame_message_norm_by_layer": {},
            "frame_key_norm_by_layer": {},
            "raw_frame_compatibility_mean_by_layer": {},
            "raw_frame_compatibility_abs_mean_by_layer": {},
            "across_frame_weight_sum_by_layer": {},
            "across_frame_weight_abs_sum_by_layer": {},
            "signed_positive_weight_fraction_by_layer": {},
            "signed_negative_weight_fraction_by_layer": {},
            "softmax_entropy_by_layer": {},
            "original_visual_update_norm_by_layer": {},
            "factorized_visual_update_norm_by_layer": {},
            "replacement_delta_norm_by_layer": {},
            "final_carrier_update_norm_by_layer": {},
            "read_norm_by_layer": {},
            "injection_norm_by_layer": {},
            "carrier_state_norm_by_layer": {},
            "update_to_carrier_ratio_by_layer": {},
            "injection_to_carrier_ratio_by_layer": {},
            "gamma_by_layer": {},
            "visual_reconstruction_error_by_layer": {},
            "factorized_reconstruction_error_by_layer": {},
            "noncarrier_factorized_update_max_by_layer": {},
            "invalid_frame_weight_max_by_layer": {},
            "softmax_weight_sum_error_by_layer": {},
            "replacement_subtracts_original_visual_by_layer": {},
            "additive_subtracts_original_visual_by_layer": {},
            "tensor_shapes_by_layer": {},
        }
        self._last_tensors = {}

    def clear_context(self) -> None:
        self._carrier_positions = None
        self._carrier_identities = None
        self._frame_groups = None
        self._frame_valid_mask = None
        self._evidence_frame_indices = None
        self._sample_ids = None
        self._slots = None
        self._capture_buffers = {}
        self._active_capture_layer = None

    def attach(self, model: Any) -> None:
        self.detach()
        layers = get_layers(model)
        for layer_idx in self.layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"native attention layer={layer_idx} outside [0, {len(layers) - 1}]")
            layer_module = layers[int(layer_idx)]
            attn = getattr(layer_module, "self_attn", None)
            if attn is None:
                raise RuntimeError(f"layer={layer_idx} has no self_attn")
            for name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, name, None)
                if proj is None:
                    raise RuntimeError(f"layer={layer_idx}.self_attn has no {name}")
                self._hook_handles.append(proj.register_forward_hook(self._capture_hook(int(layer_idx), name)))

            original_forward = layer_module.forward
            adapter = self

            def wrapped_forward(
                module_self: Any,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Any] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                _layer_idx: int = int(layer_idx),
                **kwargs: Any,
            ) -> Any:
                layer_kwargs = dict(kwargs)
                layer_kwargs.update(
                    {
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "past_key_values": past_key_values,
                        "output_attentions": output_attentions,
                        "use_cache": use_cache,
                        "cache_position": cache_position,
                        "position_embeddings": position_embeddings,
                    }
                )
                return adapter.forward_layer(module_self, hidden_states, int(_layer_idx), layer_kwargs)

            layer_module.forward = MethodType(wrapped_forward, layer_module)
            self._original_forwards.append((layer_module, original_forward))

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for layer_module, original_forward in reversed(self._original_forwards):
            layer_module.forward = original_forward
        self._original_forwards = []
        self._active_capture_layer = None
        self._capture_buffers = {}

    def _capture_hook(self, layer_idx: int, name: str) -> Any:
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            if self._active_capture_layer != int(layer_idx) or not torch.is_tensor(output):
                return
            self._capture_buffers.setdefault(int(layer_idx), {})[str(name)] = output

        return hook

    def _call_self_attn(self, module: Any, normed: torch.Tensor, kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[Any], bool]:
        attn = module.self_attn
        try:
            result = attn(
                hidden_states=normed,
                attention_mask=kwargs.get("attention_mask"),
                position_ids=kwargs.get("position_ids"),
                past_key_values=kwargs.get("past_key_values"),
                output_attentions=kwargs.get("output_attentions", False),
                use_cache=kwargs.get("use_cache", False),
                cache_position=kwargs.get("cache_position"),
                position_embeddings=kwargs.get("position_embeddings"),
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    not in {
                        "attention_mask",
                        "position_ids",
                        "past_key_values",
                        "output_attentions",
                        "use_cache",
                        "cache_position",
                        "position_embeddings",
                    }
                    and value is not None
                },
            )
        except TypeError:
            result = attn(normed, attention_mask=kwargs.get("attention_mask"))
        if torch.is_tensor(result):
            return result, None, False
        if isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]):
            weights = result[1] if len(result) > 1 else None
            return result[0], weights, True
        raise RuntimeError(f"Unsupported self-attention output type: {type(result).__name__}")

    def forward_layer(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Any:
        residual = hidden_states
        normed = module.input_layernorm(hidden_states)
        self._capture_buffers[int(layer_idx)] = {}
        self._active_capture_layer = int(layer_idx)
        try:
            attn_out, attn_weights, tuple_output = self._call_self_attn(module, normed, kwargs)
        finally:
            self._active_capture_layer = None
        h_attn = residual + attn_out
        if self.enabled and self._carrier_positions is not None:
            h_attn = self.apply_native_branch(module, normed, h_attn, int(layer_idx), kwargs)
        residual = h_attn
        mlp_hidden = module.post_attention_layernorm(h_attn)
        mlp_out = module.mlp(mlp_hidden)
        output_hidden = residual + mlp_out
        if tuple_output:
            outputs: Tuple[Any, ...] = (output_hidden,)
            if kwargs.get("output_attentions", False):
                outputs += (attn_weights,)
            return outputs
        return output_hidden

    def _captured_qkv(
        self,
        attn: Any,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, int]:
        del kwargs
        captured = self._capture_buffers.get(int(layer_idx), {})
        missing = [name for name in ("q_proj", "k_proj", "v_proj") if name not in captured]
        if missing:
            raise RuntimeError(f"layer={layer_idx}: q/k/v capture failed for {missing}")
        raw_q = captured["q_proj"]
        raw_k = captured["k_proj"]
        raw_v = captured["v_proj"]
        batch, seq_len, _ = raw_q.shape
        head_dim = int(getattr(attn, "head_dim", raw_q.shape[-1] // int(getattr(attn, "num_heads", 1))))
        num_heads = int(getattr(attn, "num_heads", raw_q.shape[-1] // head_dim))
        q = raw_q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = raw_k.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        v = raw_v.view(batch, seq_len, -1, head_dim).transpose(1, 2)
        position_embeddings = self._last_position_embeddings
        if (
            position_embeddings is not None
            and base.apply_multimodal_rotary_pos_emb is not None
            and hasattr(attn, "rope_scaling")
        ):
            q, k = base.apply_multimodal_rotary_pos_emb(
                q,
                k,
                position_embeddings[0],
                position_embeddings[1],
                attn.rope_scaling["mrope_section"],
            )
        k = base._repeat_kv(k, num_heads)
        v = base._repeat_kv(v, num_heads)
        scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
        return q, k, v, scaling, num_heads, head_dim

    def _allowed_and_scores(
        self,
        *,
        attn: Any,
        q_b: torch.Tensor,
        k_b: torch.Tensor,
        carrier_idx: torch.Tensor,
        seq_len: int,
        scaling: float,
        attention_mask: Any,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = q_b.device
        arange = torch.arange(seq_len, device=device)
        allowed = arange.unsqueeze(0) <= carrier_idx.unsqueeze(1)
        sliding_window = getattr(attn, "sliding_window", None)
        if sliding_window is not None:
            allowed &= arange.unsqueeze(0) >= (carrier_idx.unsqueeze(1) - int(sliding_window))
        scores = torch.einsum("hcd,hsd->hcs", q_b[:, carrier_idx, :].float(), k_b.float()) * float(scaling)
        scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
        if torch.is_tensor(attention_mask):
            if attention_mask.dim() == 4:
                selected_mask = attention_mask[batch_idx : batch_idx + 1, :, carrier_idx, :seq_len].float()
                scores = scores + selected_mask.squeeze(0)
                allowed = allowed & (selected_mask.squeeze(0).amax(dim=0) > torch.finfo(scores.dtype).min / 2)
            elif attention_mask.dim() == 2:
                src_valid = attention_mask[batch_idx, :seq_len].bool()
                scores = scores.masked_fill(~src_valid.view(1, 1, -1), torch.finfo(scores.dtype).min)
                allowed = allowed & src_valid.view(1, -1)
        return scores, allowed

    def _project_o(self, attn: Any, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return base._apply_o_proj(attn, x.to(dtype=dtype), gate_all=True).float()

    def apply_native_branch(
        self,
        module: Any,
        normed: torch.Tensor,
        h_attn: torch.Tensor,
        layer_idx: int,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if int(layer_idx) not in self.layer_to_pos:
            return h_attn
        if self._carrier_positions is None or self._frame_groups is None:
            return h_attn
        self.hook_fire_counts[int(layer_idx)] += 1
        attn = module.self_attn
        self._last_position_embeddings = kwargs.get("position_embeddings")
        q, k, v, scaling, num_heads, head_dim = self._captured_qkv(attn, int(layer_idx), kwargs)
        batch, _heads, seq_len, _dim = q.shape
        hidden = int(num_heads) * int(head_dim)
        max_carriers = max((len(row) for row in self._carrier_positions), default=0)
        max_frames = max((len(row) for row in self._frame_groups), default=0)
        max_frame_tokens = max(
            [len(group) for row in self._frame_groups for group in row if group] or [1]
        )
        if max_carriers <= 0 or max_frames <= 0:
            return h_attn

        device = h_attn.device
        messages = torch.zeros((batch, max_carriers, max_frames, num_heads, head_dim), device=device, dtype=torch.float32)
        frame_keys = torch.zeros_like(messages)
        original_per_frame = torch.zeros_like(messages)
        within_weights = torch.zeros((batch, max_carriers, max_frames, num_heads, max_frame_tokens), device=device, dtype=torch.float32)
        frame_token_positions = torch.full((batch, max_frames, max_frame_tokens), -1, device=device, dtype=torch.long)
        compat = torch.zeros((batch, max_carriers, max_frames, num_heads), device=device, dtype=torch.float32)
        valid_chf = torch.zeros((batch, max_carriers, max_frames, num_heads), device=device, dtype=torch.bool)
        original_visual_head = torch.zeros((batch, max_carriers, num_heads, head_dim), device=device, dtype=torch.float32)
        carrier_states = torch.zeros((batch, max_carriers, h_attn.shape[-1]), device=device, dtype=torch.float32)

        attention_mask = kwargs.get("attention_mask")
        for b in range(batch):
            carriers = [int(pos) for pos in self._carrier_positions[b] if 0 <= int(pos) < seq_len]
            if not carriers:
                continue
            c_idx = torch.tensor(carriers, device=device, dtype=torch.long)
            carrier_states[b, : len(carriers)] = h_attn[b, c_idx, :].float()
            scores, allowed = self._allowed_and_scores(
                attn=attn,
                q_b=q[b],
                k_b=k[b],
                carrier_idx=c_idx,
                seq_len=seq_len,
                scaling=float(scaling),
                attention_mask=attention_mask,
                batch_idx=b,
            )
            probs = torch.softmax(scores, dim=-1)
            visual_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
            for f, group in enumerate(self._frame_groups[b]):
                frame_positions = [int(x) for x in group if 0 <= int(x) < seq_len]
                if not frame_positions:
                    continue
                pos_tensor = torch.tensor(frame_positions, device=device, dtype=torch.long)
                frame_token_positions[b, f, : len(frame_positions)] = pos_tensor[:max_frame_tokens]
                visual_mask[pos_tensor] = True
                frame_allowed = allowed[:, pos_tensor].unsqueeze(0).expand(num_heads, -1, -1)
                frame_scores = scores[:, :, pos_tensor].masked_fill(~frame_allowed, torch.finfo(scores.dtype).min)
                has_valid = frame_allowed.any(dim=-1, keepdim=True)
                beta = torch.softmax(frame_scores, dim=-1)
                beta = torch.where(has_valid, beta, torch.zeros_like(beta))
                m = torch.einsum("hct,htd->hcd", beta, v[b, :, pos_tensor, :].float())
                kappa = torch.einsum("hct,htd->hcd", beta, k[b, :, pos_tensor, :].float())
                a = torch.einsum("hcd,hcd->hc", q[b, :, c_idx, :].float(), kappa.float()) * float(scaling)
                original_frame = torch.einsum("hct,htd->hcd", probs[:, :, pos_tensor], v[b, :, pos_tensor, :].float())
                messages[b, : len(carriers), f] = m.permute(1, 0, 2)
                frame_keys[b, : len(carriers), f] = kappa.permute(1, 0, 2)
                original_per_frame[b, : len(carriers), f] = original_frame.permute(1, 0, 2)
                within_weights[b, : len(carriers), f, :, : len(frame_positions)] = beta.permute(1, 0, 2)
                compat[b, : len(carriers), f] = a.permute(1, 0)
                valid_chf[b, : len(carriers), f] = has_valid.squeeze(-1).permute(1, 0)
            if bool(visual_mask.any()):
                visual_probs = probs * visual_mask.view(1, 1, -1).float()
                original = torch.einsum("hcs,hsd->hcd", visual_probs, v[b].float())
                original_visual_head[b, : len(carriers)] = original.permute(1, 0, 2)

        if self.variant == NATIVE_REPLACE_VISUAL_SOFTMAX:
            masked_compat = compat.masked_fill(~valid_chf, torch.finfo(compat.dtype).min)
            weights = torch.softmax(masked_compat, dim=2)
            weights = torch.where(valid_chf.any(dim=2, keepdim=True), weights, torch.zeros_like(weights))
            weights = weights.masked_fill(~valid_chf, 0.0)
        else:
            weights = compat.masked_fill(~valid_chf, 0.0)

        factorized_per_frame = weights.unsqueeze(-1) * messages
        factorized_head = factorized_per_frame.sum(dim=2)
        factorized_recon_error = float((factorized_per_frame.sum(dim=2) - factorized_head).detach().float().abs().max().cpu().item())
        original_recon_error = float((original_per_frame.sum(dim=2) - original_visual_head).detach().float().abs().max().cpu().item())
        frame_flat = factorized_head.reshape(batch * max_carriers, hidden)
        visual_flat = original_visual_head.reshape(batch * max_carriers, hidden)
        factorized_update = self._project_o(attn, frame_flat, normed.dtype).view(batch, max_carriers, -1)
        original_visual_update = self._project_o(attn, visual_flat, normed.dtype).view(batch, max_carriers, -1)

        if self.variant == NATIVE_ADDITIVE_SIGNED:
            raw_delta = factorized_update
        else:
            raw_delta = factorized_update - original_visual_update
        gamma = self.gamma[self.layer_to_pos[int(layer_idx)]].float()
        update = gamma * raw_delta
        out = h_attn.clone()
        update_mask = torch.zeros((batch, seq_len, 1), device=device, dtype=torch.float32)
        carrier_norms: List[float] = []
        update_norms: List[float] = []
        ratios: List[float] = []
        for b, positions in enumerate(self._carrier_positions):
            for c, pos in enumerate(positions):
                if c >= max_carriers or not (0 <= int(pos) < seq_len):
                    continue
                update_mask[b, int(pos), 0] = 1.0
                before = out[b, int(pos), :].float()
                delta = update[b, c].to(dtype=h_attn.dtype)
                out[b, int(pos), :] = out[b, int(pos), :] + delta
                update_norm = float(delta.detach().float().norm().cpu().item())
                carrier_norm = float(before.detach().float().norm().cpu().item())
                update_norms.append(update_norm)
                carrier_norms.append(carrier_norm)
                ratios.append(update_norm / max(carrier_norm, 1e-6))

        noncarrier_update = torch.zeros_like(h_attn).float()
        noncarrier_max = float((noncarrier_update * (1.0 - update_mask)).abs().max().cpu().item()) if noncarrier_update.numel() else 0.0
        weight_sums = weights.sum(dim=2)
        valid_weight_values = weights[valid_chf]
        if valid_weight_values.numel():
            positive_fraction = float((valid_weight_values > 0).float().mean().cpu().item())
            negative_fraction = float((valid_weight_values < 0).float().mean().cpu().item())
        else:
            positive_fraction = MISSING
            negative_fraction = MISSING
        if self.variant == NATIVE_REPLACE_VISUAL_SOFTMAX:
            entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=2)
            entropy_mean = float(entropy[valid_chf.any(dim=2)].mean().cpu().item()) if bool(valid_chf.any()) else MISSING
            softmax_sum_error = float((weight_sums[valid_chf.any(dim=2)] - 1.0).abs().max().cpu().item()) if bool(valid_chf.any()) else 0.0
        else:
            entropy_mean = MISSING
            softmax_sum_error = MISSING
        invalid_weight_max = float(weights.masked_fill(valid_chf, 0.0).abs().max().detach().cpu().item()) if weights.numel() else 0.0
        within_sum = within_weights.sum(dim=-1)
        within_error = float((within_sum[valid_chf] - 1.0).abs().max().cpu().item()) if bool(valid_chf.any()) else 0.0
        layer_key = str(int(layer_idx))
        stats_updates = {
            "within_frame_weight_sum_error_by_layer": within_error,
            "frame_message_norm_by_layer": _masked_norm_mean(messages, valid_chf),
            "frame_key_norm_by_layer": _masked_norm_mean(frame_keys, valid_chf),
            "raw_frame_compatibility_mean_by_layer": _masked_mean(compat, valid_chf),
            "raw_frame_compatibility_abs_mean_by_layer": _masked_mean(compat.abs(), valid_chf),
            "across_frame_weight_sum_by_layer": float(weight_sums.detach().float().mean().cpu().item()) if weight_sums.numel() else MISSING,
            "across_frame_weight_abs_sum_by_layer": float(weights.detach().float().abs().sum(dim=2).mean().cpu().item()) if weights.numel() else MISSING,
            "signed_positive_weight_fraction_by_layer": positive_fraction if self.variant != NATIVE_REPLACE_VISUAL_SOFTMAX else MISSING,
            "signed_negative_weight_fraction_by_layer": negative_fraction if self.variant != NATIVE_REPLACE_VISUAL_SOFTMAX else MISSING,
            "softmax_entropy_by_layer": entropy_mean,
            "original_visual_update_norm_by_layer": float(original_visual_update.detach().float().norm(dim=-1).mean().cpu().item()),
            "factorized_visual_update_norm_by_layer": float(factorized_update.detach().float().norm(dim=-1).mean().cpu().item()),
            "replacement_delta_norm_by_layer": float(raw_delta.detach().float().norm(dim=-1).mean().cpu().item()),
            "final_carrier_update_norm_by_layer": finite_mean(update_norms, default=0.0),
            "read_norm_by_layer": float(factorized_update.detach().float().norm(dim=-1).mean().cpu().item()),
            "injection_norm_by_layer": finite_mean(update_norms, default=0.0),
            "carrier_state_norm_by_layer": finite_mean(carrier_norms, default=0.0),
            "update_to_carrier_ratio_by_layer": finite_mean(ratios, default=0.0),
            "injection_to_carrier_ratio_by_layer": finite_mean(ratios, default=0.0),
            "gamma_by_layer": float(gamma.detach().cpu().item()),
            "visual_reconstruction_error_by_layer": original_recon_error,
            "factorized_reconstruction_error_by_layer": factorized_recon_error,
            "noncarrier_factorized_update_max_by_layer": noncarrier_max,
            "invalid_frame_weight_max_by_layer": invalid_weight_max,
            "softmax_weight_sum_error_by_layer": softmax_sum_error,
            "replacement_subtracts_original_visual_by_layer": int(self.variant != NATIVE_ADDITIVE_SIGNED),
            "additive_subtracts_original_visual_by_layer": int(False),
        }
        for key, value in stats_updates.items():
            self._last_stats[key][layer_key] = [value]
        self._last_stats["tensor_shapes_by_layer"][layer_key] = [{
            "within_frame_attention_weights": list(within_weights.shape),
            "frame_messages": list(messages.shape),
            "frame_keys": list(frame_keys.shape),
            "raw_frame_compatibility": list(compat.shape),
            "final_across_frame_weights": list(weights.shape),
            "original_visual_update": list(original_visual_update.shape),
            "factorized_visual_update": list(factorized_update.shape),
            "replacement_delta": list(raw_delta.shape),
            "final_carrier_update": list(update.shape),
        }]
        self._last_tensors[layer_key] = {
            "within_frame_attention_weights": within_weights.detach().float().cpu(),
            "frame_token_positions": frame_token_positions.detach().cpu(),
            "frame_messages": messages.detach().float().cpu(),
            "frame_keys": frame_keys.detach().float().cpu(),
            "raw_frame_compatibility": compat.detach().float().cpu(),
            "final_across_frame_weights": weights.detach().float().cpu(),
            "sum_across_frame_weights": weight_sums.detach().float().cpu(),
            "valid_frame_head_mask": valid_chf.detach().cpu(),
            "original_visual_update": original_visual_update.detach().float().cpu(),
            "factorized_visual_update": factorized_update.detach().float().cpu(),
            "replacement_delta": raw_delta.detach().float().cpu(),
            "final_carrier_update": update.detach().float().cpu(),
            "carrier_state_norm": carrier_states.detach().float().norm(dim=-1).cpu(),
            "update_to_carrier_norm_ratio": torch.tensor(ratios, dtype=torch.float32),
            "factorized_per_frame_contributions": factorized_per_frame.detach().float().cpu(),
            "original_visual_per_frame_contributions": original_per_frame.detach().float().cpu(),
        }
        return out

    def stats_for_row(self, row: int) -> Dict[str, Any]:
        del row
        out: Dict[str, Any] = {}
        for key, by_layer in self._last_stats.items():
            payload: Dict[str, Any] = {}
            for layer, values in by_layer.items():
                if isinstance(values, list) and values:
                    payload[str(layer)] = values[0]
                else:
                    payload[str(layer)] = values
            out[key] = payload
        gamma = {str(layer): float(self.gamma[pos].detach().cpu().item()) for layer, pos in self.layer_to_pos.items()}
        out["native_gamma"] = gamma
        out["memory_gamma"] = gamma
        out["memory_reconstruction_error"] = finite_mean(
            list(out.get("visual_reconstruction_error_by_layer", {}).values()),
            default=math.nan,
        )
        return out

    def diagnostic_tensors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._last_tensors


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    active = values.detach().float()[mask.bool()]
    return float(active.mean().cpu().item()) if active.numel() else MISSING


def _masked_norm_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    active = values.detach().float().norm(dim=-1)[mask.bool()]
    return float(active.mean().cpu().item()) if active.numel() else MISSING


def make_adapter(args: Any, variant: str, hidden_size: int, layers: Sequence[int]) -> ExperimentAdapter:
    del hidden_size
    if bool(getattr(args, "frame_kv_lora", False)):
        raise NotImplementedError("--frame-kv-lora is not part of this focused native attention ablation")
    lora = base.AttentionLoRAAdapter(
        inject_layers=layers,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=("q_proj", "o_proj"),
        gated=True,
    )
    native_attention = NativeFrameFactorizedAttention(
        variant=variant,
        layers=layers,
        additive_gamma_init=1e-3,
        replacement_gamma_init=0.0,
        reconstruction_tol=float(getattr(args, "reconstruction_tol", 1e-4)),
    )
    return ExperimentAdapter(lora=lora, memory=native_attention)


def trainable_parameter_summary(model: Any, adapter: ExperimentAdapter, variant: str) -> Dict[str, Any]:
    del variant
    total_model = sum(int(param.numel()) for param in model.parameters())
    trainable_model_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name
    ]
    if trainable_model_names:
        raise RuntimeError(f"Unexpected trainable Qwen base parameters: {trainable_model_names[:20]}")
    trainable: List[Dict[str, Any]] = []
    grouped = {"lora": 0, "native_attention": 0, "memory": 0, "other": 0}
    for name, param in adapter.named_parameters():
        if not param.requires_grad:
            continue
        count = int(param.numel())
        trainable.append({"name": name, "shape": list(param.shape), "numel": count})
        if "lora_" in name or ".lora" in name or "wrappers" in name:
            grouped["lora"] += count
        elif name == "memory.gamma" or name.endswith(".gamma"):
            grouped["native_attention"] += count
        else:
            grouped["other"] += count
    total_parameters = int(total_model + grouped["native_attention"] + grouped["other"])
    return {
        "total_parameter_count": total_parameters,
        "total_model_parameters_including_attached_lora": int(total_model),
        "trainable_model_parameter_tensors": trainable_model_names,
        "trainable_adapter_parameters": int(sum(row["numel"] for row in trainable)),
        "trainable_parameter_names": [row["name"] for row in trainable],
        "trainable_parameters": trainable,
        "groups": grouped,
    }


def diagnostic_scalar_rows(prediction_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    layer_keys = sorted(
        {
            key
            for row in prediction_rows
            for key, value in row.items()
            if key.endswith("_by_layer") and isinstance(_json_load_maybe(value), dict) and key != "tensor_shapes_by_layer"
        }
    )
    for row in prediction_rows:
        per_key = {key: _json_load_maybe(row.get(key, {})) for key in layer_keys}
        layers = sorted({str(layer) for payload in per_key.values() if isinstance(payload, dict) for layer in payload})
        for layer in layers:
            out = {
                "variant": row.get("variant"),
                "split": row.get("split"),
                "example_id": row.get("example_id"),
                "layer": int(layer),
                "num_frames": row.get("num_frames"),
                "true_count": row.get("gold_count"),
                "evidence_density": row.get("evidence_density"),
            }
            for key, payload in per_key.items():
                if isinstance(payload, dict) and layer in payload:
                    value = _scalar(payload[layer])
                    if value is not None:
                        out[key.replace("_by_layer", "")] = value
            rows.append(out)
    return rows


def assert_run_correctness(
    *,
    variant: str,
    adapter: ExperimentAdapter,
    prediction_rows: Sequence[Dict[str, Any]],
    dataset_manifest: Dict[str, Any],
    count_values: Sequence[int],
) -> None:
    if [int(x) for x in count_values] != list(range(9)):
        raise RuntimeError("Candidate counts are not exactly 0..8")
    for split, payload in dataset_manifest["splits"].items():
        if int(payload["n"]) != int(payload["expected_n"]):
            raise RuntimeError(f"{split}: dataset manifest has n != expected_n")
    max_noncarrier_lora = max(
        [finite_float(row.get("noncarrier_lora_update_max")) or 0.0 for row in prediction_rows],
        default=0.0,
    )
    if max_noncarrier_lora > 1e-5:
        raise RuntimeError(f"Carrier-gated LoRA updated non-carrier positions: max={max_noncarrier_lora}")
    if adapter.memory is None:
        raise RuntimeError("Native attention variants must attach the native attention adapter")
    for row in prediction_rows:
        hooks = _json_load_maybe(row.get("tensor_shapes_by_layer", {}))
        if isinstance(hooks, dict) and row["split"] in scaffold.MAIN_EVAL_SPLITS:
            missing = sorted(set(str(layer) for layer in DEFAULT_LAYERS) - set(hooks))
            if missing:
                raise RuntimeError(f"{row['example_id']}: missing native attention hook layers {missing}")
        checks = {
            "within_frame_weight_sum_error_by_layer": 1e-4,
            "visual_reconstruction_error_by_layer": 1e-4,
            "factorized_reconstruction_error_by_layer": 1e-5,
            "invalid_frame_weight_max_by_layer": 1e-7,
            "noncarrier_factorized_update_max_by_layer": 1e-7,
        }
        for key, tol in checks.items():
            payload = _json_load_maybe(row.get(key, {}))
            if isinstance(payload, dict):
                bad = {layer: value for layer, value in payload.items() if (finite_float(value) or 0.0) > tol}
                if bad:
                    raise RuntimeError(f"{key} failed tolerance {tol}: {bad}")
        if variant == NATIVE_REPLACE_VISUAL_SOFTMAX:
            payload = _json_load_maybe(row.get("softmax_weight_sum_error_by_layer", {}))
            if isinstance(payload, dict):
                bad = {layer: value for layer, value in payload.items() if (finite_float(value) or 0.0) > 1e-4}
                if bad:
                    raise RuntimeError(f"Softmax frame weights did not sum to one: {bad}")
        if variant == NATIVE_ADDITIVE_SIGNED:
            sub = _json_load_maybe(row.get("additive_subtracts_original_visual_by_layer", {}))
            if isinstance(sub, dict) and any(int(value) != 0 for value in sub.values() if str(value).strip() != ""):
                raise RuntimeError("Additive variant subtracted the original visual contribution")
        else:
            sub = _json_load_maybe(row.get("replacement_subtracts_original_visual_by_layer", {}))
            if isinstance(sub, dict) and any(int(value) != 1 for value in sub.values() if str(value).strip() != ""):
                raise RuntimeError("Replacement variant did not subtract the original visual contribution")


def validate_smoke_native_behavior(
    *,
    model: Any,
    processor: Any,
    adapter: ExperimentAdapter,
    examples: Dict[str, List[Any]],
    answer_ids: Dict[int, Tuple[int, ...]],
    device: str,
) -> None:
    if adapter.memory is None:
        raise RuntimeError("Smoke validation requires native attention adapter")
    adapter.attach(model)
    try:
        example = examples[scaffold.VAL_SPLIT][0]
        batch = scaffold.prepare_batch(
            examples=[example],
            sample_indices=[0],
            processor=processor,
            device=device,
            answer_ids=answer_ids,
        )
        adapter.set_context(batch)
        logits_a = model(**batch.inputs, use_cache=False).logits.detach().float().cpu()
        stats_a = adapter.stats_for_row(0)
        adapter.clear_context()

        no_evidence_example = scaffold.FrameMemoryExample(
            example_id=example.example_id + "_diag_no_evidence",
            split=example.split,
            frame_paths=example.frame_paths,
            num_frames=example.num_frames,
            gold_count=example.gold_count,
            evidence_frame_indices=tuple(),
            question=example.question,
            answer=example.answer,
            queried_character=example.queried_character,
            queried_room=example.queried_room,
            template_id=example.template_id,
            composition_key=example.composition_key,
            source_dataset_info=example.source_dataset_info,
        )
        batch_b = scaffold.prepare_batch(
            examples=[no_evidence_example],
            sample_indices=[0],
            processor=processor,
            device=device,
            answer_ids=answer_ids,
        )
        adapter.set_context(batch_b)
        logits_b = model(**batch_b.inputs, use_cache=False).logits.detach().float().cpu()
        adapter.clear_context()
        if not torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Evidence metadata affected native attention forward computation")
        hook_layers = set(str(layer) for layer in DEFAULT_LAYERS)
        if set(stats_a.get("tensor_shapes_by_layer", {})) != hook_layers:
            raise RuntimeError("Native attention hooks did not fire exactly once per selected layer in smoke forward")
        gamma = adapter.memory.gamma.detach().clone()
        try:
            adapter.memory.gamma.data.zero_()
            adapter.set_memory_enabled(False)
            adapter.set_context(batch)
            ref = model(**batch.inputs, use_cache=False).logits.detach().float().cpu()
            adapter.clear_context()
            adapter.set_memory_enabled(True)
            adapter.set_context(batch)
            got = model(**batch.inputs, use_cache=False).logits.detach().float().cpu()
            adapter.clear_context()
            if not torch.allclose(ref, got, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Replacement path with gamma=0 did not reproduce normal attention")
        finally:
            adapter.memory.gamma.data.copy_(gamma)
            adapter.set_memory_enabled(True)
    finally:
        adapter.detach()


def _metrics_by_density(rows: Sequence[Dict[str, Any]], variant: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    splits = sorted({str(row.get("split")) for row in rows})
    for split in splits:
        split_rows = [row for row in rows if str(row.get("split")) == split]
        densities = sorted({round(float(row.get("evidence_density", 0.0)), 6) for row in split_rows})
        for density in densities:
            data = [row for row in split_rows if round(float(row.get("evidence_density", 0.0)), 6) == density]
            true_values = [int(row["gold_count"]) for row in data]
            pred_values = [int(row["predicted_count"]) for row in data]
            correct = [int(p == t) for p, t in zip(pred_values, true_values)]
            abs_errors = [abs(int(p) - int(t)) for p, t in zip(pred_values, true_values)]
            out.append(
                {
                    "variant": variant,
                    "split": split,
                    "evidence_density": float(density),
                    "n": len(data),
                    "accuracy": finite_mean(correct, default=MISSING),
                    "mae": finite_mean(abs_errors, default=MISSING),
                    "mean_signed_error": finite_mean([int(p) - int(t) for p, t in zip(pred_values, true_values)], default=MISSING),
                }
            )
    return out


def run_variant(**kwargs: Any) -> Dict[str, Any]:
    summary = _ORIGINAL_RUN_VARIANT(**kwargs)
    run_dir = Path(summary["run_dir"])
    rows = scaffold.read_csv_rows(run_dir / "per_sample_predictions.csv")
    variant = str(summary["variant"])
    density_rows = _metrics_by_density(rows, variant)
    scaffold.write_csv_dynamic(run_dir / "metrics_by_evidence_density.csv", density_rows, leading=("variant", "split", "evidence_density"))
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["metrics_by_evidence_density"] = density_rows
        scaffold.write_json(metrics_path, payload)
    return summary


def plot_native_diagnostics(plots: Path, rows: Sequence[Dict[str, Any]]) -> None:
    for key, filename, ylabel in (
        ("gamma_by_layer", "gamma_by_layer", "Gamma"),
        ("original_visual_update_norm_by_layer", "original_visual_update_norm_by_layer", "Original visual update norm"),
        ("factorized_visual_update_norm_by_layer", "factorized_visual_update_norm_by_layer", "Factorized visual update norm"),
        ("replacement_delta_norm_by_layer", "replacement_delta_norm_by_layer", "Replacement delta norm"),
        ("final_carrier_update_norm_by_layer", "final_carrier_update_norm_by_layer", "Final carrier update norm"),
        ("raw_frame_compatibility_mean_by_layer", "raw_frame_compatibility_by_layer", "Raw frame compatibility"),
        ("across_frame_weight_abs_sum_by_layer", "across_frame_weight_abs_sum_by_layer", "Across-frame |weight| sum"),
        ("signed_positive_weight_fraction_by_layer", "signed_positive_weight_fraction_by_layer", "Positive signed weight fraction"),
        ("signed_negative_weight_fraction_by_layer", "signed_negative_weight_fraction_by_layer", "Negative signed weight fraction"),
        ("softmax_entropy_by_layer", "softmax_entropy_by_layer", "Softmax frame entropy"),
        ("within_frame_weight_sum_error_by_layer", "within_frame_weight_sum_error_by_layer", "Within-frame sum error"),
        ("visual_reconstruction_error_by_layer", "original_visual_reconstruction_error_by_layer", "Original visual reconstruction error"),
        ("factorized_reconstruction_error_by_layer", "factorized_reconstruction_error_by_layer", "Factorized reconstruction error"),
    ):
        scaffold.plot_diagnostic_line(plots, scaffold.diagnostic_layer_values(rows, key), filename, ylabel)
    scaffold.plot_by_layer_and_length(
        plots,
        rows,
        "final_carrier_update_norm_by_layer",
        "final_carrier_update_norm_by_layer_and_length",
        "Final carrier update norm",
    )
    scaffold.plot_by_layer_and_length(
        plots,
        rows,
        "update_to_carrier_ratio_by_layer",
        "update_to_carrier_ratio_by_layer_and_length",
        "Update/carrier norm ratio",
    )


def write_run_report(
    run_dir: Path,
    variant: str,
    split_metrics: Sequence[Dict[str, Any]],
    paired_by_length_rows: Sequence[Dict[str, Any]],
    intervention_summary: Sequence[Dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    del intervention_summary
    lines = [
        f"# Native Frame-Factorized Attention: {variant}",
        "",
        f"- Best checkpoint: `{checkpoint_path}`",
        "- Trainable parameters: carrier-gated LoRA on `q_proj` and `o_proj`, plus one native attention gamma per selected layer.",
        "- Native branch: Qwen q/k/v capture, within-frame visual softmax, frame key/message construction, native `o_proj`, carrier-only update.",
        "",
        "## Split Metrics",
        "",
        "| split | n | accuracy | MAE |",
        "|---|---:|---:|---:|",
    ]
    for row in split_metrics:
        lines.append(
            f"| `{row.get('split')}` | {row.get('n')} | {_fmt(row.get('accuracy'))} | {_fmt(row.get('mae'))} |"
        )
    if paired_by_length_rows:
        lines.extend(["", "## Paired Neutral Extension", "", "| length | accuracy | abs prediction drift | gold-score drift | update norm |", "|---:|---:|---:|---:|---:|"])
        for row in paired_by_length_rows:
            lines.append(
                f"| {row.get('version_length')} | {_fmt(row.get('accuracy'))} | "
                f"{_fmt(row.get('mean_absolute_prediction_drift'))} | {_fmt(row.get('gold_answer_score_change'))} | "
                f"{_fmt(row.get('memory_injection_norm'))} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Predictions: `{run_dir / 'per_sample_predictions.csv'}`",
            f"- Scalar diagnostics: `{run_dir / 'diagnostics' / 'scalar_diagnostics.csv'}`",
            f"- Tensor diagnostics: `{run_dir / 'diagnostics'}`",
            f"- PNG plots: `{run_dir / 'plots'}`",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = finite_float(value)
    if number is None or not math.isfinite(number):
        return "n/a"
    return f"{number:.4g}"


def _gamma_summary(parent: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for variant, run_dir in scaffold.variant_run_dirs(parent).items():
        rows = scaffold.read_csv_rows(run_dir / "per_sample_predictions.csv")
        values: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            payload = _json_load_maybe(row.get("native_gamma", row.get("memory_gamma", {})))
            if isinstance(payload, dict):
                for layer, value in payload.items():
                    number = finite_float(value)
                    if number is not None:
                        values[str(layer)].append(float(number))
        out[variant] = {layer: finite_mean(items, default=MISSING) for layer, items in sorted(values.items())}
    return out


def write_final_report(
    parent: Path,
    metric_rows: Sequence[Dict[str, Any]],
    paired_rows: Sequence[Dict[str, Any]],
    shared: Dict[str, Any],
) -> None:
    gamma = _gamma_summary(parent)
    iid = {variant: scaffold.metric_lookup(metric_rows, variant, scaffold.IID_TEST_SPLIT, "accuracy") for variant in VARIANTS}
    interp = {variant: scaffold.metric_lookup(metric_rows, variant, scaffold.LENGTH_INTERPOLATION_SPLIT, "accuracy") for variant in VARIANTS}
    extra = {variant: scaffold.metric_lookup(metric_rows, variant, scaffold.LENGTH_EXTRAPOLATION_SPLIT, "accuracy") for variant in VARIANTS}
    paired_l16 = [row for row in paired_rows if int(row.get("version_length", -1)) == 16]
    drift = {
        variant: finite_mean([row.get("abs_prediction_drift") for row in paired_l16 if row.get("variant") == variant], default=MISSING)
        for variant in VARIANTS
    }
    lines = [
        "# Native Frame-Factorized Attention Report",
        "",
        f"Dataset hash: `{shared.get('dataset_hash', '')}`",
        f"Output root: `{parent}`",
        "",
        "## Headline Metrics",
        "",
        "| variant | IID acc | interpolation acc | extrapolation acc | length-16 paired drift | gamma by layer |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        lines.append(
            f"| `{variant}` | {_fmt(iid[variant])} | {_fmt(interp[variant])} | {_fmt(extra[variant])} | "
            f"{_fmt(drift[variant])} | `{json.dumps(gamma.get(variant, {}), sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            "1. Separate additive correction branch: compare `native_additive_signed` against both replacement variants above. "
            "If it wins mainly by keeping the full original visual path, the additive branch is still carrying useful residual behavior.",
            "2. Direct visual replacement: `native_replace_visual_signed` tests this directly because gamma scales only "
            "`factorized_visual_update - original_visual_update` and initializes at exact normal attention.",
            "3. Signed versus softmax aggregation: compare `native_replace_visual_signed` with "
            "`native_replace_visual_softmax`; the softmax variant is the competitive-normalization control.",
            "4. Longer-sequence generalization: use the extrapolation column and paired neutral-extension drift. "
            "Lower drift and higher extrapolation accuracy indicate cleaner length generalization.",
            "5. Replacement coefficient movement: the gamma table shows how far each selected layer moved from zero "
            "toward full replacement; gamma near one means the learned path approached a complete visual substitution.",
            "",
            "## Artifacts",
            "",
            f"- Combined metrics: `{parent / 'combined_results.csv'}`",
            f"- Paired neutral-extension summary: `{parent / 'paired_extension_combined.csv'}`",
            f"- Cross-variant PNG plots: `{parent / 'comparison_plots'}`",
        ]
    )
    (parent / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_parent_outputs(parent: Path) -> Dict[str, Any]:
    summary = scaffold.aggregate_parent_outputs(parent)
    density_rows: List[Dict[str, Any]] = []
    for variant, run_dir in scaffold.variant_run_dirs(parent).items():
        path = run_dir / "metrics_by_evidence_density.csv"
        if path.is_file():
            density_rows.extend(scaffold.read_csv_rows(path))
    if density_rows:
        scaffold.write_csv_dynamic(
            parent / "metrics_by_evidence_density_combined.csv",
            density_rows,
            leading=("variant", "split", "evidence_density"),
        )
    scaffold.verify_no_pdfs(parent)
    return summary


def run_checked(cmd: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=os.fspath(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def submit_slurm_jobs(args: Any, parent_output_root: Path) -> Dict[str, Any]:
    parent_output_root.mkdir(parents=True, exist_ok=True)
    (parent_output_root / "slurm").mkdir(parents=True, exist_ok=True)
    scaffold.ensure_candidate_range(args)
    dataset_dir, _examples, dataset_manifest = scaffold.ensure_dataset(args, parent_output_root / "cache")
    syntax = run_checked([sys.executable, "-m", "py_compile", __file__], cwd=PROJECT_ROOT)
    if syntax.stdout:
        print(syntax.stdout)
    if syntax.stderr:
        print(syntax.stderr)
    if not bool(args.skip_submit_smoke):
        smoke_cmd = [
            sys.executable,
            __file__,
            "--smoke-test",
            "--variants",
            ",".join(VARIANTS),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--epochs",
            "1",
            "--max-train-examples",
            "2",
            "--max-eval-examples",
            "2",
            "--output-root",
            os.fspath(parent_output_root),
        ]
        smoke = run_checked(smoke_cmd, cwd=PROJECT_ROOT)
        print(smoke.stdout)
        if smoke.stderr:
            print(smoke.stderr)
    slurm_script = PROJECT_ROOT / "scripts" / "slurm" / "native_frame_factorized_attention.sbatch"
    aggregate_script = PROJECT_ROOT / "scripts" / "slurm" / "native_frame_factorized_attention_aggregate.sbatch"
    variant_jobs: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        job_name = f"nffa_{variant}"
        result = run_checked(
            ["sbatch", "--parsable", "--job-name", job_name, os.fspath(slurm_script), variant],
            cwd=PROJECT_ROOT,
        )
        job_id = result.stdout.strip().splitlines()[-1].split(";")[0]
        variant_jobs[variant] = {
            "job_id": job_id,
            "job_name": job_name,
            "slurm_stdout": os.fspath(parent_output_root / "slurm" / f"{job_name}-{job_id}.out"),
            "slurm_stderr": os.fspath(parent_output_root / "slurm" / f"{job_name}-{job_id}.err"),
        }
        print(f"Submitted {variant}: {job_id}")
    dependency = "afterok:" + ":".join(job["job_id"] for job in variant_jobs.values())
    aggregate_result = run_checked(
        [
            "sbatch",
            "--parsable",
            "--dependency",
            dependency,
            "--job-name",
            "nffa_aggregate",
            os.fspath(aggregate_script),
        ],
        cwd=PROJECT_ROOT,
    )
    aggregate_job_id = aggregate_result.stdout.strip().splitlines()[-1].split(";")[0]
    submitted = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": os.fspath(parent_output_root),
        "dataset_dir": os.fspath(dataset_dir),
        "dataset_hash": dataset_manifest.get("dataset_hash"),
        "variant_jobs": variant_jobs,
        "aggregate_job": {
            "job_id": aggregate_job_id,
            "job_name": "nffa_aggregate",
            "dependency": dependency,
            "slurm_stdout": os.fspath(parent_output_root / "slurm" / f"nffa_aggregate-{aggregate_job_id}.out"),
            "slurm_stderr": os.fspath(parent_output_root / "slurm" / f"nffa_aggregate-{aggregate_job_id}.err"),
        },
    }
    scaffold.write_json(parent_output_root / "submitted_jobs.json", submitted)
    return submitted


def install_overrides() -> None:
    _install_constants()
    scaffold.parse_args = parse_args
    scaffold.make_adapter = make_adapter
    scaffold.trainable_parameter_summary = trainable_parameter_summary
    scaffold.assert_run_correctness = assert_run_correctness
    scaffold.validate_smoke_slot_and_evidence_behavior = validate_smoke_native_behavior
    scaffold.diagnostic_scalar_rows = diagnostic_scalar_rows
    scaffold.plot_memory_diagnostics = plot_native_diagnostics
    scaffold.write_report = write_run_report
    scaffold.write_final_report = write_final_report
    scaffold.run_variant = run_variant


def main() -> int:
    install_overrides()
    args = scaffold.parse_args()
    parent_output_root = Path(args.output_root).resolve()
    parent_output_root.mkdir(parents=True, exist_ok=True)
    count_values = scaffold.ensure_candidate_range(args)
    if bool(args.aggregate_only):
        summary = aggregate_parent_outputs(parent_output_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if bool(args.submit_slurm):
        submitted = submit_slurm_jobs(args, parent_output_root)
        print(json.dumps(submitted, indent=2, sort_keys=True))
        return 0
    dataset_dir, examples, dataset_manifest = scaffold.ensure_dataset(args, parent_output_root / "cache")
    dataset_manifest = {**dataset_manifest, "dataset_dir": os.fspath(dataset_dir)}
    if bool(args.prepare_dataset_only):
        print(f"Prepared dataset {dataset_manifest['dataset_hash']} at {dataset_dir}")
        return 0
    variants = scaffold.parse_variants(args.variants)
    if bool(args.smoke_test):
        args.tiny_debug_model = True
        if str(args.device) == "cuda":
            args.device = "cpu"
        args.epochs = min(int(args.epochs), 1)
        args.grad_accum = 1
        args.max_train_examples = 2 if int(args.max_train_examples) <= 0 else min(int(args.max_train_examples), 2)
        args.max_eval_examples = 2 if int(args.max_eval_examples) <= 0 else min(int(args.max_eval_examples), 2)
    device = base.resolve_device(str(args.device))
    dtype = base.dtype_from_arg(str(args.dtype), device)
    model, processor, load_in_4bit, load_mode = base.load_model_and_processor(args, device=device, dtype=dtype)
    tokenizer = processor.tokenizer
    tokenization_mode, answer_ids = base.text_base.answer_token_ids(tokenizer, int(args.candidate_min), int(args.candidate_max))
    print(
        "Using Qwen native q/k/v/o_proj for frame-factorized visual-to-carrier attention; "
        "dataset generation, candidate scoring, carrier-gated LoRA, paired diagnostics, and optimizer setup "
        "are reused from layerwise_glstm_mechanism_ablation."
    )
    print(f"Model load mode={load_mode} load_in_4bit={load_in_4bit} answer_tokenization={tokenization_mode}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_summaries: List[Dict[str, Any]] = []
    for variant in variants:
        run_summaries.append(
            scaffold.run_variant(
                args=args,
                variant=variant,
                model=model,
                processor=processor,
                examples=examples,
                dataset_manifest=dataset_manifest,
                parent_output_root=parent_output_root,
                answer_ids=answer_ids,
                count_values=count_values,
                device=device,
                timestamp=timestamp,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scaffold.write_json(parent_output_root / f"{timestamp}_run_summaries.json", run_summaries)
    if not bool(args.no_aggregate_after_run) and not bool(args.smoke_test):
        aggregate_parent_outputs(parent_output_root)
    scaffold.verify_no_pdfs(parent_output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
