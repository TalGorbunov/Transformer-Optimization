#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
from torch import nn
from torch.utils.data import DataLoader

from experiments.carrier_probes import run_message_memory_adapter_stage1_stage3_seq8 as prev
from experiments.carrier_probes import run_shared_count_direction_memory_seq8 as shared


PREVIOUS_SHARED_RUN = PROJECT_ROOT / "outputs" / "shared_count_direction_memory_seq8_7b_20260527_203756"
STAGE1_CURRENT_FREE = shared.STAGE1_CURRENT_FREE
STAGE1_SHARED_RESIDUAL = shared.STAGE1_SHARED_RESIDUAL
STAGE3_CURRENT_FREE = shared.STAGE3_CURRENT_FREE
STAGE3_SHARED_RESIDUAL = shared.STAGE3_SHARED_RESIDUAL
CODEBOOK_LEARNED = "learned_count_codebook"
CODEBOOK_QWEN = "qwen_initialized_count_codebook"
MIDDLE_COUNTS = (3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer-aligned count-codebook memory experiment for MMReD seq_len=8."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--base-source-run", type=Path, default=prev.DEFAULT_BASE_SOURCE_RUN)
    parser.add_argument("--previous-shared-run", type=Path, default=PREVIOUS_SHARED_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in range(9)])
    parser.add_argument("--layers", nargs="+", default=[str(x) for x in prev.DEFAULT_LAYERS])
    parser.add_argument("--carriers", nargs="+", default=["target_char", "target_room"])
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--reuse-stage1-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-stage3-baselines", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--bottleneck-dim", "--d-m", type=int, default=256)
    parser.add_argument("--key-dim", "--d-k", type=int, default=64)
    parser.add_argument("--value-dim", "--d-v", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--frame-gate-bce-weight", type=float, default=0.1)
    parser.add_argument("--count-direction-mse-weight", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.1)

    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage1-patience", type=int, default=3)
    parser.add_argument("--stage1-batch-size", type=int, default=64)

    parser.add_argument("--stage3-epochs", type=int, default=3)
    parser.add_argument("--stage3-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-weight-decay", type=float, default=1e-2)
    parser.add_argument("--stage3-batch-size", type=int, default=1)
    parser.add_argument("--stage3-grad-accum", type=int, default=8)
    parser.add_argument("--stage3-patience", type=int, default=3)
    parser.add_argument("--stage3-grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-mem-ce-weight", type=float, default=0.2)
    parser.add_argument("--count-direction-mse-weight-stage3", type=float, default=0.1)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-5)
    parser.add_argument("--inject-layer", type=int, default=18)
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--train-gamma", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--taus", nargs="+", default=["1.0", "2.0", "4.0"])
    parser.add_argument("--lambda-count-cls", nargs="+", default=["0.0", "0.05"])
    parser.add_argument("--lambda-count-reg", nargs="+", default=["0.0", "0.05"])
    parser.add_argument(
        "--grid-mode",
        choices=["diagnostic", "full"],
        default="diagnostic",
        help=(
            "diagnostic runs a tau sweep at lambda=0.05/0.05 plus a lambda ablation at tau=2.0; "
            "full runs every tau/lambda product."
        ),
    )
    parser.add_argument("--default-tau", type=float, default=2.0)
    parser.add_argument("--default-lambda-count-cls", type=float, default=0.05)
    parser.add_argument("--default-lambda-count-reg", type=float, default=0.05)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--random-codebook-init-norm", type=float, default=1.0)
    parser.add_argument("--qwen-codebook-init-norm", type=float, default=None)
    parser.add_argument("--qwen-codebook-trainable", action=argparse.BooleanOptionalAction, default=True)

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


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"answer_aligned_count_codebook_memory_seq8_7b_{stamp}"


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(".", "p")


def codebook_method_name(init_kind: str, tau: float, lambda_cls: float, lambda_reg: float) -> str:
    return (
        f"stage3_{init_kind}_tau{float_tag(tau)}"
        f"_lcls{float_tag(lambda_cls)}_lreg{float_tag(lambda_reg)}"
    )


def method_family(method: str, configs: Dict[str, Dict[str, Any]]) -> str:
    if method == STAGE3_CURRENT_FREE:
        return "current_free_value"
    if method == STAGE3_SHARED_RESIDUAL:
        return "shared_count_direction_plus_small_residual"
    return str(configs.get(method, {}).get("variant", method))


def method_stage(method: str) -> str:
    if str(method).startswith("stage3_"):
        return "stage3"
    if method == "base_frozen_qwen":
        return "base"
    return ""


def method_config_id(method: str, configs: Dict[str, Dict[str, Any]]) -> str:
    if method in configs:
        return str(configs[method]["config_id"])
    if method in (STAGE3_CURRENT_FREE, STAGE3_SHARED_RESIDUAL):
        return "baseline"
    return ""


def short_label(method: str, configs: Dict[str, Dict[str, Any]]) -> str:
    if method == STAGE3_CURRENT_FREE:
        return "current_free"
    if method == STAGE3_SHARED_RESIDUAL:
        return "shared_residual"
    config = configs.get(method, {})
    if config:
        prefix = "learned" if config.get("init_kind") == CODEBOOK_LEARNED else "qwen_init"
        return f"{prefix} t={float(config['tau']):g} {float(config['lambda_count_cls']):g}/{float(config['lambda_count_reg']):g}"
    if method == "base_frozen_qwen":
        return "base_qwen"
    return str(method).replace("stage3_", "")


def add_method_metadata(rows: Sequence[Dict[str, Any]], configs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", ""))
        config = configs.get(method, {})
        new_row = dict(row)
        new_row["stage"] = method_stage(method)
        new_row["variant"] = method_family(method, configs)
        new_row["config_id"] = method_config_id(method, configs)
        new_row["tau"] = config.get("tau", "")
        new_row["lambda_count_cls"] = config.get("lambda_count_cls", "")
        new_row["lambda_count_reg"] = config.get("lambda_count_reg", "")
        new_row["codebook_init"] = config.get("init_kind", "")
        out.append(new_row)
    return out


def parse_float_list(values: Sequence[str]) -> List[float]:
    return [float(v) for v in prev.split_tokens(values)]


def nearest_value(value: float, candidates: Sequence[float]) -> float:
    if not candidates:
        return float(value)
    return min((float(x) for x in candidates), key=lambda x: abs(float(x) - float(value)))


def build_codebook_configs(args: argparse.Namespace) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    taus = sorted(dict.fromkeys(float(x) for x in args.taus))
    lambda_cls_values = sorted(dict.fromkeys(float(x) for x in args.lambda_count_cls))
    lambda_reg_values = sorted(dict.fromkeys(float(x) for x in args.lambda_count_reg))
    default_tau = nearest_value(float(args.default_tau), taus)
    default_cls = nearest_value(float(args.default_lambda_count_cls), lambda_cls_values)
    default_reg = nearest_value(float(args.default_lambda_count_reg), lambda_reg_values)
    triples: List[Tuple[float, float, float]] = []
    if str(args.grid_mode) == "full":
        triples = [(tau, l_cls, l_reg) for tau in taus for l_cls in lambda_cls_values for l_reg in lambda_reg_values]
    else:
        seen = set()
        for tau in taus:
            triple = (float(tau), float(default_cls), float(default_reg))
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)
        for l_cls in lambda_cls_values:
            for l_reg in lambda_reg_values:
                triple = (float(default_tau), float(l_cls), float(l_reg))
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)

    methods: List[str] = []
    configs: Dict[str, Dict[str, Any]] = {}
    for init_kind in (CODEBOOK_LEARNED, CODEBOOK_QWEN):
        for tau, l_cls, l_reg in triples:
            method = codebook_method_name(init_kind, tau, l_cls, l_reg)
            methods.append(method)
            configs[method] = {
                "variant": init_kind,
                "init_kind": init_kind,
                "tau": float(tau),
                "lambda_count_cls": float(l_cls),
                "lambda_count_reg": float(l_reg),
                "config_id": f"{init_kind}_tau{float_tag(tau)}_lcls{float_tag(l_cls)}_lreg{float_tag(l_reg)}",
            }
    return methods, configs


def stage3_to_stage1(method: str) -> str:
    if method == STAGE3_CURRENT_FREE:
        return STAGE1_CURRENT_FREE
    return STAGE1_SHARED_RESIDUAL


def load_stage1_from_checkpoint(
    *,
    args: argparse.Namespace,
    variant: str,
    checkpoint_path: Path,
    input_dim: int,
    candidate_min: int,
    candidate_max: int,
) -> Tuple[shared.Stage1SharedCountReadout, Dict[str, Any], Path]:
    ckpt = prev.load_torch(checkpoint_path)
    model = shared.make_stage1_model(
        args,
        variant,
        int(input_dim),
        int(candidate_max) - int(candidate_min) + 1,
        ckpt["x_mean"].float(),
        ckpt["x_std"].float(),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.cpu(), dict(ckpt.get("history", {})), checkpoint_path


def load_or_train_stage1(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    variant: str,
    x_messages: torch.Tensor,
    labels: torch.Tensor,
    frame_labels: torch.Tensor,
    splits: Dict[str, List[int]],
    candidate_min: int,
    candidate_max: int,
) -> Tuple[shared.Stage1SharedCountReadout, Dict[str, Any], Path]:
    checkpoint_path = Path(args.previous_shared_run) / "checkpoints" / f"{variant}_best.pt"
    if bool(args.reuse_stage1_checkpoints) and checkpoint_path.is_file():
        print(f"Loading Stage 1 {variant} from previous shared run: {checkpoint_path}")
        return load_stage1_from_checkpoint(
            args=args,
            variant=variant,
            checkpoint_path=checkpoint_path,
            input_dim=int(x_messages.shape[-1]),
            candidate_min=int(candidate_min),
            candidate_max=int(candidate_max),
        )
    print(f"Training Stage 1 {variant}")
    return shared.train_stage1_variant(
        args=args,
        output_dir=output_dir,
        variant=variant,
        x_messages=x_messages,
        labels=labels,
        frame_labels=frame_labels,
        splits=splits,
        candidate_min=int(candidate_min),
        candidate_max=int(candidate_max),
    )


def load_stage3_baseline_from_checkpoint(
    *,
    args: argparse.Namespace,
    variant: str,
    checkpoint_path: Path,
    stage1_model: shared.Stage1SharedCountReadout,
    hidden_size: int,
) -> Tuple[shared.Stage3SharedCountAdapter, Dict[str, Any], Path]:
    ckpt = prev.load_torch(checkpoint_path)
    adapter = shared.make_stage3_adapter(
        args=args,
        stage1_model=stage1_model,
        stage3_variant=variant,
        hidden_size=int(hidden_size),
    )
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    return adapter.cpu(), dict(ckpt.get("history", {})), checkpoint_path


class CountCodebookStage3Adapter(nn.Module):
    def __init__(
        self,
        *,
        core: shared.SharedCountMemoryCore,
        hidden_size: int,
        count_values: Sequence[int],
        tau: float,
        residual_scale: float,
        inject_layer: int,
        gamma_init: float,
        train_gamma: bool,
        codebook_trainable: bool,
    ) -> None:
        super().__init__()
        self.core = core
        self.hidden_size = int(hidden_size)
        self.tau = float(tau)
        self.residual_scale = float(residual_scale)
        self.residual_proj = nn.Linear(core.bottleneck_dim, self.hidden_size, bias=False)
        self.codebook = nn.Parameter(
            torch.zeros(len(count_values), self.hidden_size, dtype=torch.float32),
            requires_grad=bool(codebook_trainable),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32), requires_grad=bool(train_gamma))
        self.inject_layer = int(inject_layer)
        self.enabled = True
        self._x_messages: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._handles: List[Any] = []
        self.register_buffer("count_values", torch.tensor([int(x) for x in count_values], dtype=torch.float32))
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_key_query_similarity: Optional[torch.Tensor] = None
        self.last_soft_count: Optional[torch.Tensor] = None
        self.last_codebook_logits: Optional[torch.Tensor] = None
        self.last_codebook_probs: Optional[torch.Tensor] = None
        self.last_codebook_pred_count: Optional[torch.Tensor] = None
        self.last_codebook_delta_norm: Optional[torch.Tensor] = None
        self.last_residual_norm: Optional[torch.Tensor] = None
        self.last_delta_norm: Optional[torch.Tensor] = None
        self.last_alpha_for_loss: Optional[torch.Tensor] = None
        self.last_soft_count_for_loss: Optional[torch.Tensor] = None
        self.last_codebook_logits_for_loss: Optional[torch.Tensor] = None
        self.last_codebook_probs_for_loss: Optional[torch.Tensor] = None
        self.last_delta_for_loss: Optional[torch.Tensor] = None
        self.last_residual_for_loss: Optional[torch.Tensor] = None
        nn.init.zeros_(self.residual_proj.weight)

    def set_context(self, x_messages: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._x_messages = x_messages
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        self.last_alpha = None
        self.last_key_query_similarity = None
        self.last_soft_count = None
        self.last_codebook_logits = None
        self.last_codebook_probs = None
        self.last_codebook_pred_count = None
        self.last_codebook_delta_norm = None
        self.last_residual_norm = None
        self.last_delta_norm = None
        self.last_alpha_for_loss = None
        self.last_soft_count_for_loss = None
        self.last_codebook_logits_for_loss = None
        self.last_codebook_probs_for_loss = None
        self.last_delta_for_loss = None
        self.last_residual_for_loss = None

    def clear_context(self) -> None:
        self._x_messages = None
        self._target_positions = None
        self.last_alpha_for_loss = None
        self.last_soft_count_for_loss = None
        self.last_codebook_logits_for_loss = None
        self.last_codebook_probs_for_loss = None
        self.last_delta_for_loss = None
        self.last_residual_for_loss = None

    @staticmethod
    def _hidden_from_output(output: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        return None

    @staticmethod
    def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        if isinstance(output, list):
            return [hidden] + list(output[1:])
        return output

    def compute_delta(self, x_messages: torch.Tensor) -> torch.Tensor:
        x = self.core.standardize(x_messages)
        z = self.core.dropout(F.gelu(self.core.w_p(self.core.norm(x))))
        alpha = torch.sigmoid(self.core.w_alpha(z)).squeeze(-1)
        k = F.normalize(self.core.w_k(z), dim=-1)
        q = F.normalize(self.core.q0, dim=0)
        key_query_similarity = torch.matmul(k, q)
        soft_count = alpha.sum(dim=-1)
        count_values = self.count_values.to(soft_count.device).to(soft_count.dtype)
        codebook_logits = -float(self.tau) * torch.abs(soft_count.unsqueeze(-1) - count_values.view(1, -1))
        codebook_probs = F.softmax(codebook_logits, dim=-1)
        codebook_delta = torch.matmul(codebook_probs.to(self.codebook.dtype), self.codebook.to(codebook_probs.device))
        residual_frames = self.residual_proj(z)
        residual_delta = float(self.residual_scale) * torch.sum(alpha.unsqueeze(-1) * residual_frames, dim=1)
        delta = codebook_delta + residual_delta

        pred = count_values[codebook_probs.argmax(dim=-1)]
        self.last_alpha_for_loss = alpha
        self.last_soft_count_for_loss = soft_count
        self.last_codebook_logits_for_loss = codebook_logits
        self.last_codebook_probs_for_loss = codebook_probs
        self.last_delta_for_loss = delta
        self.last_residual_for_loss = residual_delta
        self.last_alpha = alpha.detach().float().cpu()
        self.last_key_query_similarity = key_query_similarity.detach().float().cpu()
        self.last_soft_count = soft_count.detach().float().cpu()
        self.last_codebook_logits = codebook_logits.detach().float().cpu()
        self.last_codebook_probs = codebook_probs.detach().float().cpu()
        self.last_codebook_pred_count = pred.detach().float().cpu()
        self.last_codebook_delta_norm = codebook_delta.detach().float().norm(dim=-1).cpu()
        self.last_residual_norm = residual_delta.detach().float().norm(dim=-1).cpu()
        self.last_delta_norm = delta.detach().float().norm(dim=-1).cpu()
        return delta

    def inject(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self._x_messages is None or self._target_positions is None:
            return hidden_states
        delta = self.compute_delta(self._x_messages.to(hidden_states.device))
        updates: List[torch.Tensor] = []
        seq_len = int(hidden_states.shape[1])
        for batch_idx, positions in enumerate(self._target_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            mask = hidden_states.new_zeros((seq_len, 1))
            if valid:
                pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
                mask.index_fill_(0, pos_idx, 1.0)
            update = (self.gamma.to(hidden_states.device) * delta[batch_idx]).to(hidden_states.dtype).unsqueeze(0)
            updates.append(mask * update)
        return hidden_states + torch.stack(updates, dim=0)

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        if self.inject_layer < 0 or self.inject_layer >= len(layers):
            raise ValueError(f"inject_layer={self.inject_layer} outside [0, {len(layers) - 1}]")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden = self._hidden_from_output(output)
            if hidden is None:
                return output
            return self._replace_hidden(output, self.inject(hidden))

        self._handles.append(layers[self.inject_layer].register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def make_codebook_adapter(
    *,
    args: argparse.Namespace,
    stage1_model: shared.Stage1SharedCountReadout,
    config: Dict[str, Any],
    hidden_size: int,
    count_values: Sequence[int],
) -> CountCodebookStage3Adapter:
    core = shared.SharedCountMemoryCore(
        input_dim=stage1_model.core.input_dim,
        bottleneck_dim=stage1_model.core.bottleneck_dim,
        key_dim=stage1_model.core.key_dim,
        value_dim=stage1_model.core.value_dim,
        dropout=float(args.dropout),
        residual_scale=float(args.residual_scale),
        variant="shared_count_direction_plus_small_residual",
    )
    core.load_state_dict(stage1_model.core.state_dict())
    core.variant = "shared_count_direction_plus_small_residual"
    return CountCodebookStage3Adapter(
        core=core,
        hidden_size=int(hidden_size),
        count_values=count_values,
        tau=float(config["tau"]),
        residual_scale=float(args.residual_scale),
        inject_layer=int(args.inject_layer),
        gamma_init=float(args.gamma_init),
        train_gamma=bool(args.train_gamma),
        codebook_trainable=bool(args.qwen_codebook_trainable) if config["init_kind"] == CODEBOOK_QWEN else True,
    )


def tokenization_report(
    tokenizer: Any,
    counts: Sequence[int],
    preferred_format: str,
) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    chosen: Dict[int, int] = {}
    preferred_prefix = "" if preferred_format == "plain" else " "
    for count in counts:
        plain_ids = tokenizer.encode(str(int(count)), add_special_tokens=False)
        spaced_ids = tokenizer.encode(f" {int(count)}", add_special_tokens=False)
        options = {
            "plain": plain_ids,
            "leading_space": spaced_ids,
        }
        single_token = {name: ids for name, ids in options.items() if len(ids) == 1}
        if preferred_format in single_token:
            name = preferred_format
            ids = single_token[name]
        elif single_token:
            name, ids = next(iter(single_token.items()))
        else:
            name = preferred_format
            ids = tokenizer.encode(f"{preferred_prefix}{int(count)}", add_special_tokens=False)
            raise RuntimeError(f"Count {count} is not single-token under plain or leading-space: {options}")
        chosen[int(count)] = int(ids[0])
        rows.append(
            {
                "count": int(count),
                "plain_ids": list(map(int, plain_ids)),
                "leading_space_ids": list(map(int, spaced_ids)),
                "chosen_format": name,
                "chosen_token_id": int(ids[0]),
            }
        )
    return chosen, rows


def get_output_head_vectors(model: Any, token_ids: Dict[int, int], hidden_size: int) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    weight = getattr(output_embeddings, "weight", None)
    if weight is None and hasattr(model, "lm_head"):
        weight = getattr(model.lm_head, "weight", None)
    if weight is None:
        raise RuntimeError("Could not locate model output embedding / LM-head weights")
    ordered = [int(token_ids[count]) for count in sorted(token_ids)]
    vectors = weight.detach().float().cpu().index_select(0, torch.tensor(ordered, dtype=torch.long))
    if int(vectors.shape[-1]) != int(hidden_size):
        raise RuntimeError(f"LM-head vector dim {vectors.shape[-1]} != hidden_size {hidden_size}")
    return vectors


def initialize_codebook(
    *,
    adapter: CountCodebookStage3Adapter,
    init_kind: str,
    qwen_vectors: Optional[torch.Tensor],
    qwen_scale: float,
    random_norm: float,
) -> Dict[str, Any]:
    with torch.no_grad():
        if init_kind == CODEBOOK_LEARNED:
            torch.manual_seed(torch.initial_seed())
            values = torch.randn_like(adapter.codebook)
            values = F.normalize(values.float(), dim=-1) * float(random_norm)
            adapter.codebook.copy_(values)
            return {"init_kind": init_kind, "init_norm": float(random_norm), "source": "random_normal_normalized"}
        if qwen_vectors is None:
            raise RuntimeError("qwen_vectors are required for qwen_initialized_count_codebook")
        centered = qwen_vectors.float() - qwen_vectors.float().mean(dim=0, keepdim=True)
        centered = F.normalize(centered, dim=-1) * float(qwen_scale)
        adapter.codebook.copy_(centered.to(adapter.codebook.device))
        return {"init_kind": init_kind, "init_norm": float(qwen_scale), "source": "lm_head_centered_count_vectors"}
    raise ValueError(f"Unknown codebook init kind: {init_kind}")


def previous_residual_norm_scale(previous_run: Path) -> float:
    path = Path(previous_run) / "memory_diagnostics.csv"
    if not path.is_file():
        return math.nan
    values: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("method") != STAGE3_SHARED_RESIDUAL:
                continue
            if row.get("diagnostic") != "mean_delta_norm":
                continue
            try:
                value = float(row.get("value", "nan"))
            except Exception:
                value = math.nan
            if math.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else math.nan


def values_mean(mapping: Dict[int, float]) -> float:
    vals = [float(v) for v in mapping.values() if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else math.nan


def corr_or_nan(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or float(np.std(xs)) <= 0 or float(np.std(ys)) <= 0:
        return math.nan
    return float(np.corrcoef(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))[0, 1])


def count_cls_loss(adapter: CountCodebookStage3Adapter, gold_offsets: torch.Tensor) -> torch.Tensor:
    if adapter.last_codebook_logits_for_loss is None:
        return gold_offsets.float().new_tensor(0.0)
    return F.cross_entropy(adapter.last_codebook_logits_for_loss.float(), gold_offsets.long())


def count_reg_loss(adapter: CountCodebookStage3Adapter, gold_counts: torch.Tensor, huber_delta: float) -> torch.Tensor:
    if adapter.last_soft_count_for_loss is None:
        return gold_counts.float().new_tensor(0.0)
    return F.huber_loss(
        adapter.last_soft_count_for_loss.float(),
        gold_counts.float(),
        reduction="mean",
        delta=float(huber_delta),
    )


def codebook_argmax_acc(adapter: CountCodebookStage3Adapter, gold_counts: torch.Tensor) -> float:
    if adapter.last_codebook_probs_for_loss is None:
        return math.nan
    pred_offsets = adapter.last_codebook_probs_for_loss.argmax(dim=-1)
    pred_counts = pred_offsets + int(adapter.count_values.min().detach().cpu().item())
    return prev.accuracy(
        [int(x) for x in gold_counts.detach().cpu().tolist()],
        [int(x) for x in pred_counts.detach().cpu().tolist()],
    )


def train_codebook_method(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    config: Dict[str, Any],
    model: Any,
    processor: Any,
    stage1_model: shared.Stage1SharedCountReadout,
    records: Sequence[prev.SampleRecord],
    x_messages: torch.Tensor,
    splits: Dict[str, List[int]],
    count_token_ids: Dict[int, int],
    device: str,
    hidden_size: int,
    count_values: Sequence[int],
    qwen_vectors: Optional[torch.Tensor],
    qwen_scale: float,
) -> Tuple[CountCodebookStage3Adapter, Dict[str, Any], Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{safe_name(method)}_best.pt"
    adapter = make_codebook_adapter(
        args=args,
        stage1_model=stage1_model,
        config=config,
        hidden_size=int(hidden_size),
        count_values=count_values,
    )
    init_info = initialize_codebook(
        adapter=adapter,
        init_kind=str(config["init_kind"]),
        qwen_vectors=qwen_vectors,
        qwen_scale=float(qwen_scale),
        random_norm=float(args.random_codebook_init_norm),
    )
    adapter.to(device)
    trainable_params = [param for param in adapter.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.stage3_lr),
        weight_decay=float(args.stage3_weight_decay),
    )
    train_indices = list(splits["train"])
    val_indices = list(splits["val"])
    lambda_cls = float(config["lambda_count_cls"])
    lambda_reg = float(config["lambda_count_reg"])
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_ce = math.inf
    best_val_acc = -math.inf
    best_epoch = 0
    bad_epochs = 0
    rows: List[Dict[str, Any]] = []
    try:
        for epoch in range(1, int(args.stage3_epochs) + 1):
            adapter.train()
            adapter.register_hooks(model)
            rng = random.Random(int(args.seed) + 3000 + epoch + shared.stable_variant_offset(method))
            shuffled = list(train_indices)
            rng.shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            train_ce = 0.0
            train_cls = 0.0
            train_reg = 0.0
            train_cb_acc = 0.0
            train_steps = 0
            for step, batch_indices in enumerate(prev.chunked(shuffled, int(args.stage3_batch_size)), start=1):
                batch_records = [records[idx] for idx in batch_indices]
                batch = prev.prepare_qwen_batch(
                    records=batch_records,
                    sample_indices=batch_indices,
                    processor=processor,
                    device=device,
                    carriers=args.carriers,
                )
                adapter.set_context(x_messages[batch_indices].to(device), batch.target_positions)
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce = F.cross_entropy(count_logits, gold_offsets)
                cls = count_cls_loss(adapter, gold_offsets)
                reg = count_reg_loss(adapter, batch.gold_counts, float(args.huber_delta))
                loss_raw = ce + float(lambda_cls) * cls + float(lambda_reg) * reg
                loss = loss_raw / max(1, int(args.stage3_grad_accum))
                loss.backward()
                train_ce += float(ce.detach().cpu().item())
                train_cls += float(cls.detach().cpu().item())
                train_reg += float(reg.detach().cpu().item())
                cb_acc = codebook_argmax_acc(adapter, batch.gold_counts)
                train_cb_acc += 0.0 if not math.isfinite(cb_acc) else float(cb_acc)
                train_steps += 1
                adapter.clear_context()
                if step % max(1, int(args.stage3_grad_accum)) == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(args.stage3_grad_clip))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 100 == 0:
                    print(
                        f"  {method} epoch={epoch} step={step} "
                        f"train_ce={train_ce / max(1, train_steps):.4f} "
                        f"count_cls={train_cls / max(1, train_steps):.4f} "
                        f"count_reg={train_reg / max(1, train_steps):.4f}"
                    )
            if train_steps % max(1, int(args.stage3_grad_accum)) != 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.stage3_grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            val_eval = evaluate_qwen_codebook(
                method=method,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=val_indices,
                x_messages=x_messages,
                count_token_ids=count_token_ids,
                args=args,
                device=device,
                batch_size=int(args.stage3_batch_size),
                save_gates=False,
            )
            row = {
                "epoch": int(epoch),
                "train_ce": train_ce / max(1, train_steps),
                "train_count_cls": train_cls / max(1, train_steps),
                "train_count_reg": train_reg / max(1, train_steps),
                "train_codebook_argmax_accuracy": train_cb_acc / max(1, train_steps),
                "val_ce": float(val_eval["ce"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "val_codebook_argmax_accuracy": float(val_eval["codebook_argmax_accuracy"]),
                "val_soft_count_mae": float(val_eval["soft_count_mae"]),
                "val_corr_s_true_count": float(val_eval["corr_s_true_count"]),
                "val_mean_delta_norm": float(val_eval["mean_delta_norm"]),
                "val_mean_residual_norm": float(val_eval["mean_residual_norm"]),
                "gamma": float(adapter.gamma.detach().cpu().item()),
                "tau": float(config["tau"]),
                "lambda_count_cls": lambda_cls,
                "lambda_count_reg": lambda_reg,
            }
            rows.append(row)
            print(
                f"  {method} epoch={epoch} train_ce={row['train_ce']:.4f} "
                f"val_ce={row['val_ce']:.4f} val_acc={row['val_accuracy']:.4f} "
                f"val_cb_acc={row['val_codebook_argmax_accuracy']:.4f} "
                f"val_s_mae={row['val_soft_count_mae']:.4f} gamma={row['gamma']:.4f}"
            )
            improved = row["val_ce"] < best_val_ce - 1e-9 or (
                abs(row["val_ce"] - best_val_ce) <= 1e-9 and row["val_accuracy"] > best_val_acc
            )
            if improved:
                best_val_ce = float(row["val_ce"])
                best_val_acc = float(row["val_accuracy"])
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if int(args.stage3_patience) > 0 and bad_epochs >= int(args.stage3_patience):
                break
    finally:
        adapter.remove_hooks()
    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter_cpu = adapter.cpu()
    ckpt = {
        "adapter_state_dict": adapter_cpu.state_dict(),
        "variant": str(method),
        "config": dict(config),
        "init_info": init_info,
        "history": {
            "rows": rows,
            "best_epoch": int(best_epoch),
            "best_val_ce": float(best_val_ce),
            "best_val_accuracy": float(best_val_acc),
        },
        "hidden_size": int(hidden_size),
        "inject_layer": int(args.inject_layer),
        "carriers": list(args.carriers),
    }
    torch.save(ckpt, best_path)
    print(f"Saved {method} checkpoint: {best_path}")
    return adapter_cpu, ckpt["history"], best_path


@torch.no_grad()
def evaluate_qwen_codebook(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: CountCodebookStage3Adapter,
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    x_messages: torch.Tensor,
    count_token_ids: Dict[int, int],
    args: argparse.Namespace,
    device: str,
    batch_size: int,
    save_gates: bool,
) -> Dict[str, Any]:
    adapter.to(device)
    adapter.eval()
    adapter.enabled = True
    adapter.register_hooks(model)
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_score_by_idx: Dict[int, float] = {}
    gate_by_idx: Dict[int, List[float]] = {}
    sum_alpha_by_idx: Dict[int, float] = {}
    key_query_by_idx: Dict[int, List[float]] = {}
    soft_count_by_idx: Dict[int, float] = {}
    codebook_pred_by_idx: Dict[int, int] = {}
    codebook_probs_by_idx: Dict[int, List[float]] = {}
    codebook_delta_norm_by_idx: Dict[int, float] = {}
    residual_norm_by_idx: Dict[int, float] = {}
    delta_norm_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(prev.chunked(list(indices), int(batch_size)), start=1):
            batch_records = [records[idx] for idx in batch_indices]
            batch = prev.prepare_qwen_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                carriers=args.carriers,
            )
            adapter.set_context(x_messages[batch_indices].to(device), batch.target_positions)
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
            gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
            ce_total += float(F.cross_entropy(count_logits, gold_offsets, reduction="sum").detach().cpu().item())
            n += int(batch.gold_counts.numel())
            pred_offsets = count_logits.argmax(dim=-1)
            logits_cpu = count_logits.detach().float().cpu()
            for row, idx in enumerate(batch_indices):
                idx = int(idx)
                pred_by_idx[idx] = int(pred_offsets[row].detach().cpu().item()) + min(count_token_ids.keys())
                values = [float(v) for v in logits_cpu[row].tolist()]
                logits_by_idx[idx] = values
                gold_offset = int(records[idx].gold_count) - min(count_token_ids.keys())
                gold_score_by_idx[idx] = float(values[gold_offset])
            if adapter.last_alpha is not None:
                sums = adapter.last_alpha.sum(dim=-1)
                for row, idx in enumerate(batch_indices):
                    idx = int(idx)
                    sum_alpha_by_idx[idx] = float(sums[row].item())
                    if save_gates:
                        gate_by_idx[idx] = [float(v) for v in adapter.last_alpha[row].tolist()]
                    if adapter.last_key_query_similarity is not None:
                        key_query_by_idx[idx] = [float(v) for v in adapter.last_key_query_similarity[row].tolist()]
                    if adapter.last_soft_count is not None:
                        soft_count_by_idx[idx] = float(adapter.last_soft_count[row].item())
                    if adapter.last_codebook_pred_count is not None:
                        codebook_pred_by_idx[idx] = int(round(float(adapter.last_codebook_pred_count[row].item())))
                    if adapter.last_codebook_probs is not None:
                        codebook_probs_by_idx[idx] = [float(v) for v in adapter.last_codebook_probs[row].tolist()]
                    if adapter.last_codebook_delta_norm is not None:
                        codebook_delta_norm_by_idx[idx] = float(adapter.last_codebook_delta_norm[row].item())
                    if adapter.last_residual_norm is not None:
                        residual_norm_by_idx[idx] = float(adapter.last_residual_norm[row].item())
                    if adapter.last_delta_norm is not None:
                        delta_norm_by_idx[idx] = float(adapter.last_delta_norm[row].item())
            adapter.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(f"  eval {method}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    finally:
        adapter.remove_hooks()
    y_true = [int(records[int(idx)].gold_count) for idx in indices if int(idx) in pred_by_idx]
    y_pred = [pred_by_idx[int(idx)] for idx in indices if int(idx) in pred_by_idx]
    s_values = [soft_count_by_idx[int(idx)] for idx in indices if int(idx) in soft_count_by_idx]
    s_golds = [float(records[int(idx)].gold_count) for idx in indices if int(idx) in soft_count_by_idx]
    cb_true = [int(records[int(idx)].gold_count) for idx in indices if int(idx) in codebook_pred_by_idx]
    cb_pred = [codebook_pred_by_idx[int(idx)] for idx in indices if int(idx) in codebook_pred_by_idx]
    entropies: List[float] = []
    for probs in codebook_probs_by_idx.values():
        arr = np.asarray(probs, dtype=float)
        arr = np.clip(arr, 1e-12, 1.0)
        entropies.append(float(-(arr * np.log(arr)).sum()))
    return {
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "gate_by_idx": gate_by_idx,
        "sum_alpha_by_idx": sum_alpha_by_idx,
        "key_query_by_idx": key_query_by_idx,
        "soft_count_by_idx": soft_count_by_idx,
        "codebook_pred_by_idx": codebook_pred_by_idx,
        "codebook_probs_by_idx": codebook_probs_by_idx,
        "codebook_delta_norm_by_idx": codebook_delta_norm_by_idx,
        "residual_norm_by_idx": residual_norm_by_idx,
        "delta_norm_by_idx": delta_norm_by_idx,
        "codebook_argmax_accuracy": prev.accuracy(cb_true, cb_pred),
        "soft_count_mae": prev.mae(s_golds, s_values),
        "corr_s_true_count": corr_or_nan(s_values, s_golds),
        "mean_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "mean_delta_norm": values_mean(delta_norm_by_idx),
        "mean_residual_norm": values_mean(residual_norm_by_idx),
        "mean_codebook_delta_norm": values_mean(codebook_delta_norm_by_idx),
    }


def build_gold_score_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    base_scores: Dict[int, float],
    stage3_scores_by_method: Dict[str, Dict[int, float]],
    counts: Sequence[int],
    configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, scores in stage3_scores_by_method.items():
        config = configs.get(method, {})
        for count in counts:
            deltas: List[float] = []
            drops: List[float] = []
            base_values: List[float] = []
            stage3_values: List[float] = []
            for idx in test_indices:
                idx = int(idx)
                if int(records[idx].gold_count) != int(count) or idx not in base_scores or idx not in scores:
                    continue
                base = float(base_scores[idx])
                stage3 = float(scores[idx])
                base_values.append(base)
                stage3_values.append(stage3)
                deltas.append(stage3 - base)
                drops.append(base - stage3)
            rows.append(
                {
                    "method": method,
                    "stage": "stage3",
                    "variant": method_family(method, configs),
                    "config_id": method_config_id(method, configs),
                    "tau": config.get("tau", ""),
                    "lambda_count_cls": config.get("lambda_count_cls", ""),
                    "lambda_count_reg": config.get("lambda_count_reg", ""),
                    "codebook_init": config.get("init_kind", ""),
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(deltas),
                    "mean_base_gold_score": float(np.mean(base_values)) if base_values else math.nan,
                    "mean_stage3_gold_score": float(np.mean(stage3_values)) if stage3_values else math.nan,
                    "mean_gold_score_drop_vs_base": float(np.mean(drops)) if drops else math.nan,
                    "median_gold_score_drop_vs_base": float(np.median(drops)) if drops else math.nan,
                    "mean_gold_score_delta_vs_base": float(np.mean(deltas)) if deltas else math.nan,
                    "median_gold_score_delta_vs_base": float(np.median(deltas)) if deltas else math.nan,
                }
            )
    return rows


def build_middle_rows(
    *,
    overall_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    overall = {row["method"]: row for row in overall_rows}
    rows: List[Dict[str, Any]] = []
    for method in methods:
        count_rows = [
            row for row in metric_rows if row.get("method") == method and int(row.get("evidence_count", -1)) in MIDDLE_COUNTS
        ]
        gold_mid = [
            float(row["mean_gold_score_delta_vs_base"])
            for row in gold_rows
            if row.get("method") == method
            and int(row.get("evidence_count", -1)) in MIDDLE_COUNTS
            and math.isfinite(float(row.get("mean_gold_score_delta_vs_base", math.nan)))
        ]
        n = sum(int(row.get("n", 0)) for row in count_rows)
        weighted_correct = sum(float(row.get("accuracy", math.nan)) * int(row.get("n", 0)) for row in count_rows)
        weighted_abs = sum(float(row.get("mae", math.nan)) * int(row.get("n", 0)) for row in count_rows)
        config = configs.get(method, {})
        rows.append(
            {
                "method": method,
                "stage": method_stage(method),
                "variant": method_family(method, configs),
                "config_id": method_config_id(method, configs),
                "tau": config.get("tau", ""),
                "lambda_count_cls": config.get("lambda_count_cls", ""),
                "lambda_count_reg": config.get("lambda_count_reg", ""),
                "codebook_init": config.get("init_kind", ""),
                "split": "test",
                "counts": "3,4,5,6",
                "n": int(n),
                "middle_accuracy_3_6": weighted_correct / max(1, n),
                "middle_mae_3_6": weighted_abs / max(1, n),
                "mean_gold_score_delta_3_6": float(np.mean(gold_mid)) if gold_mid else math.nan,
                "overall_accuracy": float(overall.get(method, {}).get("accuracy", math.nan)),
                "overall_mae": float(overall.get(method, {}).get("mae", math.nan)),
            }
        )
    return rows


def build_count_distribution_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    evals: Dict[str, Dict[str, Any]],
    counts: Sequence[int],
    configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, payload in evals.items():
        config = configs.get(method, {})
        probs_by_idx = payload.get("codebook_probs_by_idx", {})
        soft_by_idx = payload.get("soft_count_by_idx", {})
        pred_by_idx = payload.get("codebook_pred_by_idx", {})
        if not probs_by_idx:
            continue
        for true_count in counts:
            idxs = [int(idx) for idx in test_indices if int(records[int(idx)].gold_count) == int(true_count) and int(idx) in probs_by_idx]
            mean_soft = float(np.mean([float(soft_by_idx[idx]) for idx in idxs if idx in soft_by_idx])) if idxs else math.nan
            pred_vals = [int(pred_by_idx[idx]) for idx in idxs if idx in pred_by_idx]
            pred_acc = prev.accuracy([int(true_count)] * len(pred_vals), pred_vals) if pred_vals else math.nan
            mean_pred = float(np.mean(pred_vals)) if pred_vals else math.nan
            for codebook_count in counts:
                pos = int(codebook_count) - int(counts[0])
                values = [float(probs_by_idx[idx][pos]) for idx in idxs if pos < len(probs_by_idx[idx])]
                rows.append(
                    {
                        "method": method,
                        "stage": "stage3",
                        "variant": method_family(method, configs),
                        "config_id": method_config_id(method, configs),
                        "tau": config.get("tau", ""),
                        "lambda_count_cls": config.get("lambda_count_cls", ""),
                        "lambda_count_reg": config.get("lambda_count_reg", ""),
                        "codebook_init": config.get("init_kind", ""),
                        "split": "test",
                        "true_count": int(true_count),
                        "codebook_count": int(codebook_count),
                        "n": len(values),
                        "mean_probability": float(np.mean(values)) if values else math.nan,
                        "mean_soft_count": mean_soft,
                        "mean_predicted_codebook_count": mean_pred,
                        "codebook_argmax_accuracy": pred_acc,
                    }
                )
    return rows


def codebook_stats(
    *,
    method: str,
    adapter: CountCodebookStage3Adapter,
    eval_payload: Dict[str, Any],
    configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    codebook = adapter.codebook.detach().float().cpu()
    norms = codebook.norm(dim=-1).numpy()
    normed = F.normalize(codebook, dim=-1)
    cos = torch.matmul(normed, normed.t()).numpy()
    mask = ~np.eye(cos.shape[0], dtype=bool)
    offdiag = cos[mask]
    config = configs.get(method, {})
    return {
        "method": method,
        "stage": "stage3",
        "variant": method_family(method, configs),
        "config_id": method_config_id(method, configs),
        "tau": config.get("tau", ""),
        "lambda_count_cls": config.get("lambda_count_cls", ""),
        "lambda_count_reg": config.get("lambda_count_reg", ""),
        "codebook_init": config.get("init_kind", ""),
        "split": "test",
        "codebook_trainable": bool(adapter.codebook.requires_grad),
        "codebook_norm_mean": float(np.mean(norms)),
        "codebook_norm_min": float(np.min(norms)),
        "codebook_norm_max": float(np.max(norms)),
        "offdiag_cosine_mean": float(np.mean(offdiag)) if offdiag.size else math.nan,
        "offdiag_cosine_min": float(np.min(offdiag)) if offdiag.size else math.nan,
        "offdiag_cosine_max": float(np.max(offdiag)) if offdiag.size else math.nan,
        "codebook_argmax_accuracy": float(eval_payload.get("codebook_argmax_accuracy", math.nan)),
        "soft_count_mae": float(eval_payload.get("soft_count_mae", math.nan)),
        "corr_s_true_count": float(eval_payload.get("corr_s_true_count", math.nan)),
        "mean_entropy": float(eval_payload.get("mean_entropy", math.nan)),
        "mean_codebook_delta_norm": float(eval_payload.get("mean_codebook_delta_norm", math.nan)),
        "mean_residual_norm": float(eval_payload.get("mean_residual_norm", math.nan)),
        "mean_delta_norm": float(eval_payload.get("mean_delta_norm", math.nan)),
    }


def residual_norm_rows(
    *,
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
    evals: Dict[str, Dict[str, Any]],
    counts: Sequence[int],
    configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, payload in evals.items():
        config = configs.get(method, {})
        delta = payload.get("delta_norm_by_idx", {})
        residual = payload.get("residual_norm_by_idx", {})
        codebook = payload.get("codebook_delta_norm_by_idx", {})
        for count in counts:
            idxs = [int(idx) for idx in test_indices if int(records[int(idx)].gold_count) == int(count)]
            delta_values = [float(delta[idx]) for idx in idxs if idx in delta]
            residual_values = [float(residual[idx]) for idx in idxs if idx in residual]
            codebook_values = [float(codebook[idx]) for idx in idxs if idx in codebook]
            rows.append(
                {
                    "method": method,
                    "stage": method_stage(method),
                    "variant": method_family(method, configs),
                    "config_id": method_config_id(method, configs),
                    "tau": config.get("tau", ""),
                    "lambda_count_cls": config.get("lambda_count_cls", ""),
                    "lambda_count_reg": config.get("lambda_count_reg", ""),
                    "codebook_init": config.get("init_kind", ""),
                    "split": "test",
                    "evidence_count": int(count),
                    "n": len(delta_values),
                    "mean_delta_norm": float(np.mean(delta_values)) if delta_values else math.nan,
                    "mean_residual_norm": float(np.mean(residual_values)) if residual_values else math.nan,
                    "mean_codebook_delta_norm": float(np.mean(codebook_values)) if codebook_values else math.nan,
                }
            )
    return rows


def method_value(rows: Sequence[Dict[str, Any]], method: str, key: str) -> float:
    for row in rows:
        if row.get("method") == method:
            return float(row.get(key, math.nan))
    return math.nan


def mean_for_counts(rows: Sequence[Dict[str, Any]], method: str, key: str, counts: Sequence[int] = MIDDLE_COUNTS) -> float:
    wanted = set(int(c) for c in counts)
    values = [
        float(row.get(key, math.nan))
        for row in rows
        if row.get("method") == method
        and int(row.get("evidence_count", row.get("true_count", -1))) in wanted
        and math.isfinite(float(row.get(key, math.nan)))
    ]
    return float(np.mean(values)) if values else math.nan


def best_method_for_family(
    *,
    family: str,
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    codebook_diag_rows: Sequence[Dict[str, Any]],
) -> Optional[str]:
    candidates = [method for method in methods if configs.get(method, {}).get("init_kind") == family]
    if not candidates:
        return None

    def score(method: str) -> Tuple[float, float, float, float, float]:
        overall_acc = method_value(overall_rows, method, "accuracy")
        mid_acc = mean_for_counts(metric_rows, method, "accuracy")
        mid_delta = mean_for_counts(gold_rows, method, "mean_gold_score_delta_vs_base")
        cb_acc = method_value(codebook_diag_rows, method, "codebook_argmax_accuracy")
        mae = method_value(overall_rows, method, "mae")
        return (
            -math.inf if not math.isfinite(overall_acc) else overall_acc,
            -math.inf if not math.isfinite(mid_acc) else mid_acc,
            -math.inf if not math.isfinite(mid_delta) else mid_delta,
            -math.inf if not math.isfinite(cb_acc) else cb_acc,
            math.inf if not math.isfinite(mae) else -mae,
        )

    return max(candidates, key=score)


def display_methods_for_plots(
    stage3_methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    codebook_diag_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    methods = [STAGE3_CURRENT_FREE, STAGE3_SHARED_RESIDUAL]
    for family in (CODEBOOK_LEARNED, CODEBOOK_QWEN):
        best = best_method_for_family(
            family=family,
            methods=stage3_methods,
            configs=configs,
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            gold_rows=gold_rows,
            codebook_diag_rows=codebook_diag_rows,
        )
        if best is not None:
            methods.append(best)
    return [method for method in methods if method in set(stage3_methods)]


def plot_overall_acc_mae(
    output_dir: Path,
    overall_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
) -> None:
    rows = [row for method in methods for row in overall_rows if row.get("method") == method]
    if not rows:
        return
    labels = [short_label(str(row["method"]), configs) for row in rows]
    acc = [float(row["accuracy"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    xs = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(rows) * 1.6), 4.5))
    axes[0].bar(xs, acc, color="#3f6f8f")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, max(0.75, max(acc) * 1.15 if acc else 0.75))
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(xs, mae, color="#a75d45")
    axes[1].set_ylabel("MAE")
    axes[1].set_ylim(0, max(1.0, max(mae) * 1.15 if mae else 1.0))
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("Overall Accuracy and MAE by Variant")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_acc_mae_by_variant.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_line(
    *,
    output_dir: Path,
    filename: str,
    rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    xs_all = sorted({int(row[x_key]) for row in rows if str(row.get(x_key, "")).lstrip("-").isdigit()})
    if not xs_all:
        return
    plt.figure(figsize=(9, 5.5))
    for method in methods:
        by_x = {int(row[x_key]): row for row in rows if row.get("method") == method}
        ys = [float(by_x.get(x, {}).get(y_key, math.nan)) for x in xs_all]
        if any(math.isfinite(v) for v in ys):
            plt.plot(xs_all, ys, marker="o", linewidth=1.8, label=short_label(method, configs))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(xs_all)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_middle_accuracy(
    output_dir: Path,
    middle_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
) -> None:
    by_method = {row["method"]: row for row in middle_rows}
    labels = [short_label(method, configs) for method in methods]
    vals = [float(by_method.get(method, {}).get("middle_accuracy_3_6", math.nan)) for method in methods]
    xs = np.arange(len(methods))
    plt.figure(figsize=(max(8, len(methods) * 1.5), 4.8))
    plt.bar(xs, vals, color="#547a53")
    plt.ylabel("Mean accuracy, counts 3-6")
    plt.title("Middle-Count Accuracy by Variant")
    plt.xticks(xs, labels, rotation=30, ha="right")
    finite = [v for v in vals if math.isfinite(v)]
    plt.ylim(0, max(0.75, max(finite) * 1.15 if finite else 0.75))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "middle_count_accuracy_3_6_by_variant.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_codebook_probability_heatmap(
    output_dir: Path,
    count_distribution_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
) -> None:
    codebook_methods = [method for method in methods if method in configs]
    if not codebook_methods:
        return
    counts = sorted({int(row["true_count"]) for row in count_distribution_rows})
    if not counts:
        return
    fig, axes = plt.subplots(1, len(codebook_methods), figsize=(5.2 * len(codebook_methods), 4.8), squeeze=False)
    for ax, method in zip(axes[0], codebook_methods):
        matrix = np.full((len(counts), len(counts)), np.nan, dtype=float)
        for row in count_distribution_rows:
            if row.get("method") != method:
                continue
            i = counts.index(int(row["true_count"]))
            j = counts.index(int(row["codebook_count"]))
            matrix[i, j] = float(row["mean_probability"])
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(short_label(method, configs))
        ax.set_xlabel("Codebook count k")
        ax.set_ylabel("True count")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean p(k | s) by True Count")
    fig.tight_layout()
    fig.savefig(output_dir / "codebook_probability_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_codebook_cosine(
    output_dir: Path,
    codebooks: Dict[str, torch.Tensor],
    methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
) -> None:
    codebook_methods = [method for method in methods if method in codebooks]
    if not codebook_methods:
        return
    fig, axes = plt.subplots(1, len(codebook_methods), figsize=(5.0 * len(codebook_methods), 4.6), squeeze=False)
    for ax, method in zip(axes[0], codebook_methods):
        codebook = codebooks[method].float()
        cos = torch.matmul(F.normalize(codebook, dim=-1), F.normalize(codebook, dim=-1).t()).numpy()
        im = ax.imshow(cos, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="equal")
        ax.set_title(short_label(method, configs))
        ax.set_xlabel("a_k")
        ax.set_ylabel("a_k")
        ax.set_xticks(range(cos.shape[0]))
        ax.set_yticks(range(cos.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Codebook Cosine Similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "codebook_cosine_similarity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_required_plots(
    *,
    output_dir: Path,
    display_methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    middle_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    count_distribution_rows: Sequence[Dict[str, Any]],
    residual_rows: Sequence[Dict[str, Any]],
    codebooks: Dict[str, torch.Tensor],
) -> None:
    plot_overall_acc_mae(output_dir, overall_rows, display_methods, configs)
    plot_line(
        output_dir=output_dir,
        filename="per_count_accuracy_by_variant.png",
        rows=metric_rows,
        methods=display_methods,
        configs=configs,
        x_key="evidence_count",
        y_key="accuracy",
        xlabel="True evidence count",
        ylabel="Accuracy",
        title="Per-Count Accuracy by Variant",
    )
    plot_middle_accuracy(output_dir, middle_rows, display_methods, configs)
    plot_line(
        output_dir=output_dir,
        filename="gold_score_delta_by_count.png",
        rows=gold_rows,
        methods=display_methods,
        configs=configs,
        x_key="evidence_count",
        y_key="mean_gold_score_delta_vs_base",
        xlabel="True evidence count",
        ylabel="Mean gold-score delta vs base",
        title="Gold-Score Delta by Count",
    )
    plot_line(
        output_dir=output_dir,
        filename="soft_count_s_by_true_count.png",
        rows=count_distribution_rows,
        methods=[method for method in display_methods if method in configs],
        configs=configs,
        x_key="true_count",
        y_key="mean_soft_count",
        xlabel="True evidence count",
        ylabel="Mean soft count s",
        title="Soft Count s by True Count",
    )
    plot_line(
        output_dir=output_dir,
        filename="predicted_codebook_count_by_true_count.png",
        rows=count_distribution_rows,
        methods=[method for method in display_methods if method in configs],
        configs=configs,
        x_key="true_count",
        y_key="mean_predicted_codebook_count",
        xlabel="True evidence count",
        ylabel="Mean argmax_k p(k | s)",
        title="Predicted Codebook Count by True Count",
    )
    plot_codebook_probability_heatmap(output_dir, count_distribution_rows, display_methods, configs)
    plot_codebook_cosine(output_dir, codebooks, display_methods, configs)
    plot_line(
        output_dir=output_dir,
        filename="residual_norm_by_count.png",
        rows=residual_rows,
        methods=display_methods,
        configs=configs,
        x_key="evidence_count",
        y_key="mean_delta_norm",
        xlabel="True evidence count",
        ylabel="Mean ||injected delta||",
        title="Residual/Delta Norm by Count",
    )


def format_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> List[str]:
    table = [list(map(str, headers))] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[col]) for row in table) for col in range(len(headers))]
    lines: List[str] = []
    for idx, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * widths[col] for col in range(len(headers))))
    return lines


def corr_from_mapping(records: Sequence[prev.SampleRecord], indices: Sequence[int], values_by_idx: Dict[int, float]) -> float:
    xs = [float(values_by_idx[int(idx)]) for idx in indices if int(idx) in values_by_idx]
    ys = [float(records[int(idx)].gold_count) for idx in indices if int(idx) in values_by_idx]
    return corr_or_nan(xs, ys)


def write_summary(
    *,
    output_dir: Path,
    stage3_methods: Sequence[str],
    display_methods: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    overall_rows: Sequence[Dict[str, Any]],
    metric_rows: Sequence[Dict[str, Any]],
    middle_rows: Sequence[Dict[str, Any]],
    gold_rows: Sequence[Dict[str, Any]],
    codebook_diag_rows: Sequence[Dict[str, Any]],
    evals: Dict[str, Dict[str, Any]],
    checkpoints: Dict[str, Path],
    records: Sequence[prev.SampleRecord],
    test_indices: Sequence[int],
) -> None:
    overall = {row["method"]: row for row in overall_rows}
    middle = {row["method"]: row for row in middle_rows}
    learned_best = next((method for method in display_methods if configs.get(method, {}).get("init_kind") == CODEBOOK_LEARNED), None)
    qwen_best = next((method for method in display_methods if configs.get(method, {}).get("init_kind") == CODEBOOK_QWEN), None)
    shared_acc = float(overall.get(STAGE3_SHARED_RESIDUAL, {}).get("accuracy", math.nan))
    learned_acc = float(overall.get(learned_best or "", {}).get("accuracy", math.nan))
    qwen_acc = float(overall.get(qwen_best or "", {}).get("accuracy", math.nan))
    diag_by_method = {row["method"]: row for row in codebook_diag_rows}

    lines = [
        "Answer-aligned count codebook memory seq_len=8",
        "",
        "Overall accuracy and MAE:",
    ]
    table_rows = []
    for method in display_methods:
        row = overall.get(method, {})
        mid = middle.get(method, {})
        diag = diag_by_method.get(method, {})
        config = configs.get(method, {})
        corr_s = float(diag.get("corr_s_true_count", math.nan))
        if not math.isfinite(corr_s) and method in evals:
            corr_s = corr_from_mapping(records, test_indices, evals[method].get("sum_alpha_by_idx", {}))
        table_rows.append(
            [
                short_label(method, configs),
                f"{float(row.get('accuracy', math.nan)):.4f}",
                f"{float(row.get('mae', math.nan)):.4f}",
                f"{float(mid.get('middle_accuracy_3_6', math.nan)):.4f}",
                f"{float(mid.get('mean_gold_score_delta_3_6', math.nan)):.4f}",
                f"{corr_s:.4f}",
                f"{float(diag.get('codebook_argmax_accuracy', math.nan)):.4f}" if method in configs else "",
                f"{config.get('tau', '')}",
                f"{config.get('lambda_count_cls', '')}",
                f"{config.get('lambda_count_reg', '')}",
            ]
        )
    lines.extend(
        format_table(
            table_rows,
            ["method", "acc", "mae", "mid_acc_3_6", "mid_gold_delta", "corr_s_true", "cb_argmax_acc", "tau", "l_cls", "l_reg"],
        )
    )
    lines.append("")
    lines.append("Per-count accuracy:")
    for method in display_methods:
        rows = [
            [int(row["evidence_count"]), f"{float(row['accuracy']):.4f}", f"{float(row['mae']):.4f}", int(row["n"])]
            for row in metric_rows
            if row.get("method") == method
        ]
        lines.append(f"{short_label(method, configs)}:")
        lines.extend(format_table(rows, ["count", "acc", "mae", "n"]))
    lines.append("")
    if learned_best is not None:
        cfg = configs[learned_best]
        lines.append(
            "Best learned_count_codebook config: "
            f"tau={cfg['tau']}, lambda_count_cls={cfg['lambda_count_cls']}, "
            f"lambda_count_reg={cfg['lambda_count_reg']}."
        )
    if qwen_best is not None:
        cfg = configs[qwen_best]
        lines.append(
            "Best qwen_initialized_count_codebook config: "
            f"tau={cfg['tau']}, lambda_count_cls={cfg['lambda_count_cls']}, "
            f"lambda_count_reg={cfg['lambda_count_reg']}."
        )
    lines.append("")
    lines.append(
        "Does learned codebook beat shared-count residual? "
        f"{'Yes' if learned_acc > shared_acc else 'No'} "
        f"(accuracy {learned_acc:.4f} vs {shared_acc:.4f})."
    )
    lines.append(
        "Does Qwen-initialized codebook beat learned codebook? "
        f"{'Yes' if qwen_acc > learned_acc else 'No'} "
        f"(accuracy {qwen_acc:.4f} vs {learned_acc:.4f})."
    )
    if qwen_best is not None and learned_best is not None:
        qwen_mid = float(middle.get(qwen_best, {}).get("middle_accuracy_3_6", math.nan))
        learned_mid = float(middle.get(learned_best, {}).get("middle_accuracy_3_6", math.nan))
        qwen_competitive = qwen_acc >= learned_acc - 0.02 and qwen_mid >= learned_mid - 0.05
        lines.append(
            "Is Qwen-initialized codebook competitive with learned codebook? "
            f"{'Yes' if qwen_competitive else 'No'} "
            f"(overall {qwen_acc:.4f} vs {learned_acc:.4f}, middle {qwen_mid:.4f} vs {learned_mid:.4f})."
        )
    lines.append("")
    lines.append("Codebook diagnostics:")
    diag_rows = []
    for method in display_methods:
        if method not in configs:
            continue
        diag = diag_by_method.get(method, {})
        diag_rows.append(
            [
                short_label(method, configs),
                f"{float(diag.get('codebook_argmax_accuracy', math.nan)):.4f}",
                f"{float(diag.get('soft_count_mae', math.nan)):.4f}",
                f"{float(diag.get('corr_s_true_count', math.nan)):.4f}",
                f"{float(diag.get('offdiag_cosine_mean', math.nan)):.4f}",
                f"{float(diag.get('offdiag_cosine_max', math.nan)):.4f}",
                f"{float(diag.get('mean_delta_norm', math.nan)):.4f}",
            ]
        )
    lines.extend(
        format_table(
            diag_rows,
            ["method", "cb_argmax_acc", "soft_count_mae", "corr_s_true", "mean_offdiag_cos", "max_offdiag_cos", "delta_norm"],
        )
    )
    lines.append("")
    if learned_best is not None:
        shared_mid = float(middle.get(STAGE3_SHARED_RESIDUAL, {}).get("middle_accuracy_3_6", math.nan))
        learned_mid = float(middle.get(learned_best, {}).get("middle_accuracy_3_6", math.nan))
        shared_delta = float(middle.get(STAGE3_SHARED_RESIDUAL, {}).get("mean_gold_score_delta_3_6", math.nan))
        learned_delta = float(middle.get(learned_best, {}).get("mean_gold_score_delta_3_6", math.nan))
        learned_diag = diag_by_method.get(learned_best, {})
        detection_ok = float(learned_diag.get("codebook_argmax_accuracy", math.nan)) >= 0.5 and float(
            learned_diag.get("corr_s_true_count", math.nan)
        ) >= 0.85
        usage_better = learned_acc > shared_acc or learned_delta > shared_delta
        if usage_better and detection_ok:
            source_text = "better Qwen usage on top of a still-clean count signal"
        elif usage_better:
            source_text = "better Qwen usage, but count detection weakened"
        elif detection_ok:
            source_text = "count detection remained clean, but Qwen did not use the injected code better"
        else:
            source_text = "neither count detection nor Qwen usage clearly improved"
        lines.append(
            "Interpretation: improvement source looks like "
            f"{source_text} (learned mid_acc {learned_mid:.4f} vs shared {shared_mid:.4f}, "
            f"mid_gold_delta {learned_delta:.4f} vs shared {shared_delta:.4f})."
        )
    lines.append("")
    lines.append("All Stage 3 configs run:")
    for method in stage3_methods:
        lines.append(f"- {method}")
    lines.append("")
    lines.append("Checkpoints:")
    for method, path in checkpoints.items():
        lines.append(f"- {method}: {path}")
    (output_dir / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    prev.write_csv(path, fieldnames, rows)


def main() -> int:
    args = parse_args()
    args.layers = prev.parse_int_tokens(args.layers)
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    args.carriers = prev.split_tokens(args.carriers)
    args.taus = parse_float_list(args.taus)
    args.lambda_count_cls = parse_float_list(args.lambda_count_cls)
    args.lambda_count_reg = parse_float_list(args.lambda_count_reg)
    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir()).resolve()
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        codebook_methods, method_configs = build_codebook_configs(args)
        stage3_methods = [STAGE3_CURRENT_FREE, STAGE3_SHARED_RESIDUAL] + codebook_methods
        config = {
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "base_source_run": os.fspath(args.base_source_run),
            "previous_shared_run": os.fspath(args.previous_shared_run),
            "output_dir": os.fspath(output_dir),
            "split": str(args.split),
            "seq_len": int(args.seq_len),
            "evidence_counts": list(args.evidence_counts),
            "layers": list(args.layers),
            "carriers": list(args.carriers),
            "max_samples_per_count": int(args.max_samples_per_count),
            "candidate_min": int(args.candidate_min),
            "candidate_max": int(args.candidate_max),
            "stage3_methods": list(stage3_methods),
            "codebook_grid": list(method_configs.values()),
            "grid_mode": str(args.grid_mode),
            "bottleneck_dim": int(args.bottleneck_dim),
            "key_dim": int(args.key_dim),
            "value_dim": int(args.value_dim),
            "dropout": float(args.dropout),
            "frame_gate_bce_weight": float(args.frame_gate_bce_weight),
            "count_direction_mse_weight": float(args.count_direction_mse_weight),
            "residual_scale": float(args.residual_scale),
            "stage1_epochs": int(args.stage1_epochs),
            "stage1_patience": int(args.stage1_patience),
            "stage3_epochs": int(args.stage3_epochs),
            "stage3_patience": int(args.stage3_patience),
            "stage3_lr": float(args.stage3_lr),
            "stage3_grad_accum": int(args.stage3_grad_accum),
            "huber_delta": float(args.huber_delta),
            "random_codebook_init_norm": float(args.random_codebook_init_norm),
            "qwen_codebook_init_norm": args.qwen_codebook_init_norm,
            "qwen_codebook_trainable": bool(args.qwen_codebook_trainable),
            "inject_layer": int(args.inject_layer),
            "gamma_init": float(args.gamma_init),
            "train_gamma": bool(args.train_gamma),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
            "load_in_4bit": bool(args.load_in_4bit),
            "seed": int(args.seed),
        }
        prev.write_json(output_dir / "run_config.json", config)
        print(f"Output dir: {output_dir}")
        print(f"Config: {json.dumps(config, sort_keys=True)}")

        feature_data = prev.load_message_features(args, args.layers, args.evidence_counts)
        sample_ids = feature_data["sample_ids"]
        labels = feature_data["labels"]
        frame_labels = feature_data["frame_labels"]
        x_messages = feature_data["x_messages"]
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        counts = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        counts_by_split = prev.split_counts(splits, labels, counts)
        print(f"x_messages shape={tuple(x_messages.shape)} D_msg={int(x_messages.shape[-1])}")
        for split, row in counts_by_split.items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in counts))

        stage1_models: Dict[str, shared.Stage1SharedCountReadout] = {}
        stage1_histories: Dict[str, Any] = {}
        stage1_checkpoints: Dict[str, Path] = {}
        for variant in (STAGE1_CURRENT_FREE, STAGE1_SHARED_RESIDUAL):
            model_stage1, history, checkpoint = load_or_train_stage1(
                args=args,
                output_dir=output_dir,
                variant=variant,
                x_messages=x_messages,
                labels=labels,
                frame_labels=frame_labels,
                splits=splits,
                candidate_min=int(args.candidate_min),
                candidate_max=int(args.candidate_max),
            )
            stage1_models[variant] = model_stage1
            stage1_histories[variant] = history
            stage1_checkpoints[variant] = checkpoint

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        candidate_format, count_ids = prev.candidate_token_ids(
            processor.tokenizer,
            int(args.candidate_min),
            int(args.candidate_max),
        )
        chosen_count_ids, token_rows = tokenization_report(processor.tokenizer, counts, candidate_format)
        hidden_size = prev.hidden_size_from_model(model)
        qwen_vectors = get_output_head_vectors(model, chosen_count_ids, int(hidden_size))
        previous_scale = previous_residual_norm_scale(Path(args.previous_shared_run))
        qwen_scale = (
            float(args.qwen_codebook_init_norm)
            if args.qwen_codebook_init_norm is not None
            else (previous_scale if math.isfinite(previous_scale) and previous_scale > 0 else 50.0)
        )
        print(f"Loaded Qwen hidden_size={hidden_size} candidate_format={candidate_format} count_ids={count_ids}")
        print(f"Answer-token ids chosen for codebook init: {chosen_count_ids}")
        print(f"Qwen codebook init norm scale: {qwen_scale:.4f} (previous scale={previous_scale:.4f})")

        print("Evaluating base frozen Qwen on test split")
        base_eval = shared.evaluate_qwen_count_channel(
            method="base_frozen_qwen",
            model=model,
            processor=processor,
            adapter=None,
            records=records,
            indices=splits["test"],
            x_messages=x_messages,
            count_token_ids=count_ids,
            args=args,
            device=device,
            batch_size=int(args.stage3_batch_size),
            save_gates=False,
        )

        method_predictions: Dict[str, Dict[int, int]] = {"base_frozen_qwen": base_eval["pred_by_idx"]}
        stage3_eval_by_method: Dict[str, Dict[str, Any]] = {}
        stage3_histories: Dict[str, Any] = {}
        checkpoints: Dict[str, Path] = {**stage1_checkpoints}
        codebooks: Dict[str, torch.Tensor] = {}
        codebook_diag_rows: List[Dict[str, Any]] = []

        for variant in (STAGE3_CURRENT_FREE, STAGE3_SHARED_RESIDUAL):
            source_stage1 = stage3_to_stage1(variant)
            checkpoint_path = Path(args.previous_shared_run) / "checkpoints" / f"{variant}_best.pt"
            if bool(args.reuse_stage3_baselines) and checkpoint_path.is_file():
                print(f"Loading Stage 3 baseline {variant} from previous shared run: {checkpoint_path}")
                adapter, history, checkpoint = load_stage3_baseline_from_checkpoint(
                    args=args,
                    variant=variant,
                    checkpoint_path=checkpoint_path,
                    stage1_model=stage1_models[source_stage1],
                    hidden_size=int(hidden_size),
                )
            else:
                print(f"Training Stage 3 baseline {variant}")
                adapter, history, checkpoint = shared.train_stage3_variant(
                    args=args,
                    output_dir=output_dir,
                    stage3_variant=variant,
                    model=model,
                    processor=processor,
                    stage1_model=stage1_models[source_stage1],
                    records=records,
                    x_messages=x_messages,
                    splits=splits,
                    count_token_ids=count_ids,
                    device=device,
                    hidden_size=int(hidden_size),
                )
            stage3_histories[variant] = history
            checkpoints[variant] = checkpoint
            print(f"Evaluating {variant} on test split")
            stage3_eval = shared.evaluate_qwen_count_channel(
                method=variant,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=splits["test"],
                x_messages=x_messages,
                count_token_ids=count_ids,
                args=args,
                device=device,
                batch_size=int(args.stage3_batch_size),
                save_gates=True,
            )
            stage3_eval_by_method[variant] = stage3_eval
            method_predictions[variant] = stage3_eval["pred_by_idx"]
            adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        current_shared_delta_scale = values_mean(stage3_eval_by_method[STAGE3_SHARED_RESIDUAL].get("delta_norm_by_idx", {}))
        if args.qwen_codebook_init_norm is None and math.isfinite(current_shared_delta_scale) and current_shared_delta_scale > 0:
            qwen_scale = current_shared_delta_scale
            print(f"Updated Qwen codebook init norm scale from current baseline eval: {qwen_scale:.4f}")

        for method in codebook_methods:
            config_for_method = method_configs[method]
            print(f"Training {method}")
            adapter, history, checkpoint = train_codebook_method(
                args=args,
                output_dir=output_dir,
                method=method,
                config=config_for_method,
                model=model,
                processor=processor,
                stage1_model=stage1_models[STAGE1_SHARED_RESIDUAL],
                records=records,
                x_messages=x_messages,
                splits=splits,
                count_token_ids=count_ids,
                device=device,
                hidden_size=int(hidden_size),
                count_values=counts,
                qwen_vectors=qwen_vectors,
                qwen_scale=float(qwen_scale),
            )
            stage3_histories[method] = history
            checkpoints[method] = checkpoint
            print(f"Evaluating {method} on test split")
            eval_payload = evaluate_qwen_codebook(
                method=method,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=splits["test"],
                x_messages=x_messages,
                count_token_ids=count_ids,
                args=args,
                device=device,
                batch_size=int(args.stage3_batch_size),
                save_gates=True,
            )
            stage3_eval_by_method[method] = eval_payload
            method_predictions[method] = eval_payload["pred_by_idx"]
            codebooks[method] = adapter.codebook.detach().float().cpu().clone()
            codebook_diag_rows.append(
                codebook_stats(method=method, adapter=adapter, eval_payload=eval_payload, configs=method_configs)
            )
            adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raw_metric_rows, raw_overall_rows, raw_mean_rows = prev.metric_tables(
            records=records,
            test_indices=splits["test"],
            method_predictions=method_predictions,
            counts=counts,
        )
        metric_rows = add_method_metadata(raw_metric_rows, method_configs)
        overall_rows = add_method_metadata(raw_overall_rows, method_configs)
        mean_rows = add_method_metadata(raw_mean_rows, method_configs)
        stage3_scores_by_method = {
            method: payload.get("gold_score_by_idx", {}) for method, payload in stage3_eval_by_method.items()
        }
        gold_rows = build_gold_score_rows(
            records=records,
            test_indices=splits["test"],
            base_scores=base_eval.get("gold_score_by_idx", {}),
            stage3_scores_by_method=stage3_scores_by_method,
            counts=counts,
            configs=method_configs,
        )
        middle_rows = build_middle_rows(
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            gold_rows=gold_rows,
            methods=stage3_methods,
            configs=method_configs,
        )
        count_distribution_rows = build_count_distribution_rows(
            records=records,
            test_indices=splits["test"],
            evals=stage3_eval_by_method,
            counts=counts,
            configs=method_configs,
        )
        norm_rows = residual_norm_rows(
            records=records,
            test_indices=splits["test"],
            evals=stage3_eval_by_method,
            counts=counts,
            configs=method_configs,
        )
        display_methods = display_methods_for_plots(
            stage3_methods=stage3_methods,
            configs=method_configs,
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            gold_rows=gold_rows,
            codebook_diag_rows=codebook_diag_rows,
        )

        metric_fields = [
            "method",
            "stage",
            "variant",
            "config_id",
            "tau",
            "lambda_count_cls",
            "lambda_count_reg",
            "codebook_init",
            "split",
            "evidence_count",
            "n",
            "accuracy",
            "mae",
        ]
        write_csv(output_dir / "metrics.csv", metric_fields, metric_rows)
        write_csv(output_dir / "per_count_accuracy.csv", metric_fields, metric_rows)
        write_csv(
            output_dir / "middle_count_accuracy_3_6.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "tau",
                "lambda_count_cls",
                "lambda_count_reg",
                "codebook_init",
                "split",
                "counts",
                "n",
                "middle_accuracy_3_6",
                "middle_mae_3_6",
                "mean_gold_score_delta_3_6",
                "overall_accuracy",
                "overall_mae",
            ],
            middle_rows,
        )
        write_csv(
            output_dir / "overall_metrics.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "tau",
                "lambda_count_cls",
                "lambda_count_reg",
                "codebook_init",
                "split",
                "n",
                "accuracy",
                "mae",
                "mean_predicted_count",
            ],
            overall_rows,
        )
        write_csv(
            output_dir / "gold_score_deltas_by_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "tau",
                "lambda_count_cls",
                "lambda_count_reg",
                "codebook_init",
                "split",
                "evidence_count",
                "n",
                "mean_base_gold_score",
                "mean_stage3_gold_score",
                "mean_gold_score_drop_vs_base",
                "median_gold_score_drop_vs_base",
                "mean_gold_score_delta_vs_base",
                "median_gold_score_delta_vs_base",
            ],
            gold_rows,
        )
        write_csv(
            output_dir / "codebook_diagnostics.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "tau",
                "lambda_count_cls",
                "lambda_count_reg",
                "codebook_init",
                "split",
                "codebook_trainable",
                "codebook_norm_mean",
                "codebook_norm_min",
                "codebook_norm_max",
                "offdiag_cosine_mean",
                "offdiag_cosine_min",
                "offdiag_cosine_max",
                "codebook_argmax_accuracy",
                "soft_count_mae",
                "corr_s_true_count",
                "mean_entropy",
                "mean_codebook_delta_norm",
                "mean_residual_norm",
                "mean_delta_norm",
            ],
            codebook_diag_rows,
        )
        write_csv(
            output_dir / "count_distribution_by_true_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "tau",
                "lambda_count_cls",
                "lambda_count_reg",
                "codebook_init",
                "split",
                "true_count",
                "codebook_count",
                "n",
                "mean_probability",
                "mean_soft_count",
                "mean_predicted_codebook_count",
                "codebook_argmax_accuracy",
            ],
            count_distribution_rows,
        )
        write_csv(
            output_dir / "residual_norm_by_count.csv",
            [
                "method",
                "stage",
                "variant",
                "config_id",
                "tau",
                "lambda_count_cls",
                "lambda_count_reg",
                "codebook_init",
                "split",
                "evidence_count",
                "n",
                "mean_delta_norm",
                "mean_residual_norm",
                "mean_codebook_delta_norm",
            ],
            norm_rows,
        )
        write_csv(
            output_dir / "token_ids.csv",
            ["count", "plain_ids", "leading_space_ids", "chosen_format", "chosen_token_id"],
            token_rows,
        )

        debug = {
            "x_messages_shape": list(x_messages.shape),
            "D_msg": int(x_messages.shape[-1]),
            "split_counts": {split: {str(k): int(v) for k, v in row.items()} for split, row in counts_by_split.items()},
            "stage1": {
                method: {"history": stage1_histories[method], "checkpoint": os.fspath(stage1_checkpoints[method])}
                for method in stage1_checkpoints
            },
            "stage3": {
                method: {
                    "history": stage3_histories.get(method, {}),
                    "checkpoint": os.fspath(checkpoints[method]),
                    "config": method_configs.get(method, {}),
                }
                for method in stage3_methods
                if method in checkpoints
            },
            "display_methods": display_methods,
            "candidate_format": candidate_format,
            "count_token_ids": {str(k): int(v) for k, v in count_ids.items()},
            "codebook_token_ids": {str(k): int(v) for k, v in chosen_count_ids.items()},
            "qwen_codebook_init_norm_scale": qwen_scale,
            "source_cache": os.fspath(feature_data["cache_path"]),
            "runtime_seconds": time.time() - started,
        }
        prev.write_json(output_dir / "adapter_debug.json", debug)
        write_summary(
            output_dir=output_dir,
            stage3_methods=stage3_methods,
            display_methods=display_methods,
            configs=method_configs,
            overall_rows=overall_rows,
            metric_rows=metric_rows,
            middle_rows=middle_rows,
            gold_rows=gold_rows,
            codebook_diag_rows=codebook_diag_rows,
            evals=stage3_eval_by_method,
            checkpoints=checkpoints,
            records=records,
            test_indices=splits["test"],
        )
        if not bool(args.no_plots):
            make_required_plots(
                output_dir=output_dir,
                display_methods=display_methods,
                configs=method_configs,
                metric_rows=metric_rows,
                overall_rows=overall_rows,
                middle_rows=middle_rows,
                gold_rows=gold_rows,
                count_distribution_rows=count_distribution_rows,
                residual_rows=norm_rows,
                codebooks=codebooks,
            )
        print(f"Wrote outputs to {output_dir}")
        print(f"Total runtime seconds: {time.time() - started:.2f}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
