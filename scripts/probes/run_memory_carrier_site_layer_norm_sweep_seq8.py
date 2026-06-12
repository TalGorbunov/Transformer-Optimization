#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
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

from scripts.probes import run_message_memory_adapter_stage1_stage3_seq8 as prev


EXPERIMENT_NAME = "memory_carrier_site_layer_norm_sweep_seq8_7b"
DEFAULT_MESSAGE_MEMORY_RUN = PROJECT_ROOT / "outputs" / "message_memory_adapter_stage1_stage3_seq8_7b_20260526_212606"
DEFAULT_STAGE3_CHECKPOINT = DEFAULT_MESSAGE_MEMORY_RUN / "checkpoints" / "stage3_best.pt"
MIDDLE_COUNTS = (3, 4, 5, 6)
TOKEN_SITE_ORDER = (
    "room_char",
    "room_char_count_words",
    "semantic_question_tokens",
    "all_question_tokens",
    "last_token",
)
COUNT_WORDS = {
    "how",
    "many",
    "much",
    "number",
    "count",
    "counts",
    "step",
    "steps",
    "move",
    "moves",
    "times",
    "total",
    "answer",
}


class CarrierRetrievalAdapter(nn.Module):
    """Reuse the message-memory core, but retrieve a per-carrier-token value."""

    def __init__(
        self,
        *,
        core: prev.MessageMemoryCore,
        w_o: nn.Linear,
        hidden_size: int,
        inject_layers: Sequence[int],
        alpha: float,
        injection_mode: str,
        query_mode: str,
    ) -> None:
        super().__init__()
        self.core = core
        self.w_o = w_o
        self.hidden_size = int(hidden_size)
        self.inject_layers = [int(layer) for layer in inject_layers]
        self.alpha = float(alpha)
        self.injection_mode = str(injection_mode)
        self.query_mode = str(query_mode)
        self.w_q = nn.Linear(self.hidden_size, self.core.key_dim, bias=True)
        with torch.no_grad():
            self.w_q.weight.zero_()
            self.w_q.bias.copy_(self.core.q0.detach().float())
        for param in self.w_q.parameters():
            param.requires_grad_(False)
        self.enabled = True
        self._x_messages: Optional[torch.Tensor] = None
        self._target_positions: Optional[List[List[int]]] = None
        self._handles: List[Any] = []
        self._stat_attention_sum: Optional[torch.Tensor] = None
        self._stat_attention_steps: Optional[torch.Tensor] = None
        self._stat_token_norm_sum: Optional[torch.Tensor] = None
        self._stat_token_norm_count: Optional[torch.Tensor] = None
        self._stat_total_norm_sum: Optional[torch.Tensor] = None
        self._stat_fro_norm_sq_sum: Optional[torch.Tensor] = None
        self._stat_num_tokens: Optional[torch.Tensor] = None
        self._stat_num_layers: Optional[torch.Tensor] = None

    def configure(self, *, inject_layers: Sequence[int], alpha: float, injection_mode: str) -> None:
        layers = [int(layer) for layer in inject_layers]
        if not layers:
            raise ValueError("inject_layers must not be empty")
        self.inject_layers = layers
        self.alpha = float(alpha)
        self.injection_mode = str(injection_mode)

    def set_context(self, x_messages: torch.Tensor, target_positions: Sequence[Sequence[int]]) -> None:
        self._x_messages = x_messages
        self._target_positions = [[int(pos) for pos in positions] for positions in target_positions]
        batch = int(x_messages.shape[0])
        slots = int(x_messages.shape[1])
        device = x_messages.device
        self._stat_attention_sum = torch.zeros((batch, slots), dtype=torch.float32, device=device)
        self._stat_attention_steps = torch.zeros(batch, dtype=torch.float32, device=device)
        self._stat_token_norm_sum = torch.zeros(batch, dtype=torch.float32, device=device)
        self._stat_token_norm_count = torch.zeros(batch, dtype=torch.float32, device=device)
        self._stat_total_norm_sum = torch.zeros(batch, dtype=torch.float32, device=device)
        self._stat_fro_norm_sq_sum = torch.zeros(batch, dtype=torch.float32, device=device)
        self._stat_num_tokens = torch.tensor([len(set(pos)) for pos in self._target_positions], dtype=torch.float32, device=device)
        self._stat_num_layers = torch.zeros(batch, dtype=torch.float32, device=device)

    def clear_context(self) -> None:
        self._x_messages = None
        self._target_positions = None

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

    def _message_keys_values(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._x_messages is None:
            raise RuntimeError("Missing x_messages context")
        x = self.core.standardize(self._x_messages.to(hidden_states.device))
        z = F.gelu(self.core.w_p(self.core.norm(x)))
        k = self.core.w_k(z)
        v = self.core.w_v(z)
        return k.float(), v.float()

    def _queries(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        if self.query_mode == "checkpoint_q0":
            q0 = self.core.q0.detach().to(hidden_tokens.device).float()
            return q0.unsqueeze(0).expand(int(hidden_tokens.shape[0]), -1)
        if self.query_mode == "q0_bias_wq":
            return self.w_q(hidden_tokens.float())
        raise ValueError(f"Unsupported query_mode={self.query_mode!r}")

    def _scale_for_positions(self, num_tokens: int, hidden_states: torch.Tensor) -> torch.Tensor:
        scale = float(self.alpha)
        if self.injection_mode == "alpha_div_sqrt_num_tokens":
            scale = scale / math.sqrt(max(1, int(num_tokens)))
        elif self.injection_mode != "raw_alpha":
            raise ValueError(f"Unsupported injection_mode={self.injection_mode!r}")
        return hidden_states.new_tensor(scale, dtype=torch.float32)

    def inject(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self._x_messages is None or self._target_positions is None:
            return hidden_states
        k, v = self._message_keys_values(hidden_states)
        update = torch.zeros_like(hidden_states)
        seq_len = int(hidden_states.shape[1])
        denom = math.sqrt(float(self.core.key_dim))
        assert self._stat_attention_sum is not None
        assert self._stat_attention_steps is not None
        assert self._stat_token_norm_sum is not None
        assert self._stat_token_norm_count is not None
        assert self._stat_total_norm_sum is not None
        assert self._stat_fro_norm_sq_sum is not None
        assert self._stat_num_layers is not None
        for batch_idx, positions in enumerate(self._target_positions):
            valid = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
            if not valid:
                continue
            pos_idx = torch.tensor(valid, device=hidden_states.device, dtype=torch.long)
            h_tokens = hidden_states[batch_idx, pos_idx, :]
            q = self._queries(h_tokens)
            attn = torch.softmax(torch.matmul(q, k[batch_idx].transpose(0, 1)) / denom, dim=-1)
            z = torch.matmul(attn, v[batch_idx])
            delta = self.w_o(z.to(self.w_o.weight.dtype)).float()
            delta = delta * self._scale_for_positions(len(valid), hidden_states)
            update[batch_idx, pos_idx, :] = delta.to(hidden_states.dtype)

            norms = delta.norm(dim=-1)
            self._stat_attention_sum[batch_idx] += attn.detach().float().mean(dim=0)
            self._stat_attention_steps[batch_idx] += 1.0
            self._stat_token_norm_sum[batch_idx] += norms.detach().float().sum()
            self._stat_token_norm_count[batch_idx] += float(len(valid))
            self._stat_total_norm_sum[batch_idx] += norms.detach().float().sum()
            self._stat_fro_norm_sq_sum[batch_idx] += norms.detach().float().pow(2).sum()
            self._stat_num_layers[batch_idx] += 1.0
        return hidden_states + update

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"inject_layer={layer_idx} outside [0, {len(layers) - 1}]")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden = self._hidden_from_output(output)
            if hidden is None:
                return output
            return self._replace_hidden(output, self.inject(hidden))

        for layer_idx in self.inject_layers:
            self._handles.append(layers[int(layer_idx)].register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        if self._stat_attention_sum is None:
            return {}
        assert self._stat_attention_steps is not None
        assert self._stat_token_norm_sum is not None
        assert self._stat_token_norm_count is not None
        assert self._stat_total_norm_sum is not None
        assert self._stat_fro_norm_sq_sum is not None
        assert self._stat_num_tokens is not None
        assert self._stat_num_layers is not None
        steps = self._stat_attention_steps.clamp_min(1.0).unsqueeze(-1)
        token_counts = self._stat_token_norm_count.clamp_min(1.0)
        return {
            "mean_attention": (self._stat_attention_sum / steps).detach().float().cpu(),
            "mean_residual_norm_per_token": (self._stat_token_norm_sum / token_counts).detach().float().cpu(),
            "total_residual_norm_sum": self._stat_total_norm_sum.detach().float().cpu(),
            "total_residual_norm_fro": torch.sqrt(self._stat_fro_norm_sq_sum.clamp_min(0.0)).detach().float().cpu(),
            "num_tokens": self._stat_num_tokens.detach().float().cpu(),
            "num_layers": self._stat_num_layers.detach().float().cpu(),
        }


class WindowStage3ResidualAdapter(prev.Stage3ResidualAdapter):
    def __init__(self, *args: Any, inject_layers: Sequence[int], **kwargs: Any) -> None:
        layers = [int(layer) for layer in inject_layers]
        if not layers:
            raise ValueError("inject_layers must not be empty")
        kwargs["inject_layer"] = int(layers[0])
        super().__init__(*args, **kwargs)
        self.inject_layers = layers

    def set_inject_layers(self, inject_layers: Sequence[int]) -> None:
        layers = [int(layer) for layer in inject_layers]
        if not layers:
            raise ValueError("inject_layers must not be empty")
        self.inject_layers = layers
        self.inject_layer = int(layers[0])

    def register_hooks(self, model: Any) -> None:
        self.remove_hooks()
        layers = prev.get_layers(model)
        for layer_idx in self.inject_layers:
            if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
                raise ValueError(f"inject_layer={layer_idx} outside [0, {len(layers) - 1}]")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden = self._hidden_from_output(output)
            if hidden is None:
                return output
            return self._replace_hidden(output, self.inject(hidden))

        for layer_idx in self.inject_layers:
            self._handles.append(layers[int(layer_idx)].register_forward_hook(hook))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled carrier-token/layer/normalization sweep for message memory.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/mmred_images_park")
    parser.add_argument("--source-run", type=Path, default=prev.DEFAULT_SOURCE_RUN)
    parser.add_argument("--base-source-run", type=Path, default=prev.DEFAULT_BASE_SOURCE_RUN)
    parser.add_argument("--message-memory-run", type=Path, default=DEFAULT_MESSAGE_MEMORY_RUN)
    parser.add_argument("--stage3-checkpoint", type=Path, default=DEFAULT_STAGE3_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="all_uniform")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--evidence-counts", nargs="+", default=[str(x) for x in prev.DEFAULT_EVIDENCE_COUNTS])
    parser.add_argument("--layers", nargs="+", default=[str(x) for x in prev.DEFAULT_LAYERS])
    parser.add_argument("--token-sites", nargs="+", default=list(TOKEN_SITE_ORDER))
    parser.add_argument("--layer-windows", nargs="+", default=["12-15", "14-17", "16-19", "18-21", "14-21"])
    parser.add_argument("--alpha-values", nargs="+", default=["0.25", "0.5", "1.0", "2.0"])
    parser.add_argument("--injection-modes", nargs="+", default=["raw_alpha", "alpha_div_sqrt_num_tokens"])
    parser.add_argument("--query-mode", choices=["checkpoint_q0", "q0_bias_wq"], default="q0_bias_wq")
    parser.add_argument("--include-previous-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples-per-count", type=int, default=100)
    parser.add_argument("--candidate-min", type=int, default=0)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1)
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
    parser.add_argument("--smoke-limit-configs", type=int, default=0)
    parser.add_argument("--smoke-limit-samples", type=int, default=0)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"{EXPERIMENT_NAME}_{stamp}"


def parse_float_tokens(raw_values: Sequence[str]) -> List[float]:
    values: List[float] = []
    for part in prev.split_tokens(raw_values):
        values.append(float(part))
    return values


def parse_layer_windows(raw_values: Sequence[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for raw in prev.split_tokens(raw_values):
        text = str(raw)
        if "-" not in text:
            raise ValueError(f"Layer window must look like 14-17, got {text!r}")
        left, right = text.split("-", 1)
        start, end = int(left), int(right)
        if end < start:
            raise ValueError(f"Layer window end before start: {text!r}")
        specs.append({"layer_label": f"{start}-{end}", "inject_layers": list(range(start, end + 1))})
    return specs


def safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(".", "p").replace("-", "_")


def token_text(processor: Any, token_id: int) -> str:
    try:
        return str(processor.tokenizer.decode([int(token_id)], skip_special_tokens=True))
    except Exception:
        return ""


def is_semantic_token(text: str) -> bool:
    cleaned = str(text).strip().lower()
    if not cleaned:
        return False
    return any(ch.isalnum() for ch in cleaned)


def is_count_word_token(text: str) -> bool:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).strip()
    if not cleaned:
        return False
    pieces = [piece for piece in cleaned.split() if piece]
    return any(piece in COUNT_WORDS or piece.isdigit() for piece in pieces)


def prompt_text_and_question_positions(
    *,
    input_ids_1d: torch.Tensor,
    record: prev.SampleRecord,
    processor: Any,
) -> Tuple[str, int, List[int]]:
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    prompt_text = prev.core.build_prompt(record.question, num_frames=len(record.frame_paths))
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_text_start = prev.find_subsequence(input_ids, [int(token_id) for token_id in prompt_ids])
    if prompt_text_start is None:
        raise RuntimeError(f"sample_id={record.sample_id}: failed to locate prompt text")
    question_fragment = f"Question: {record.question}\n"
    question_start = prompt_text.index(question_fragment) + len("Question: ")
    question_end = question_start + len(record.question)
    question_positions = prev._positions_from_token_span(
        int(prompt_text_start),
        prev._token_span_from_char_span(prompt_text, (question_start, question_end), processor),
    )
    return prompt_text, int(prompt_text_start), question_positions


def locate_positions_for_site(
    *,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    record: prev.SampleRecord,
    processor: Any,
    token_site: str,
) -> Tuple[List[int], int, Dict[str, Any]]:
    room_char_positions, prompt_last, debug = prev.locate_target_positions(
        input_ids_1d=input_ids_1d,
        attention_mask_1d=attention_mask_1d,
        record=record,
        processor=processor,
        carriers=("target_char", "target_room"),
    )
    _prompt_text, _prompt_start, question_positions = prompt_text_and_question_positions(
        input_ids_1d=input_ids_1d,
        record=record,
        processor=processor,
    )
    input_ids = [int(token_id) for token_id in input_ids_1d.detach().cpu().tolist()]
    semantic_question_positions = [
        pos for pos in question_positions if 0 <= int(pos) < len(input_ids) and is_semantic_token(token_text(processor, input_ids[int(pos)]))
    ]
    count_word_positions = [
        pos for pos in question_positions if 0 <= int(pos) < len(input_ids) and is_count_word_token(token_text(processor, input_ids[int(pos)]))
    ]
    if not count_word_positions:
        count_word_positions = semantic_question_positions[:2]
    by_site = {
        "room_char": room_char_positions,
        "room_char_count_words": list(room_char_positions) + count_word_positions,
        "semantic_question_tokens": semantic_question_positions,
        "all_question_tokens": question_positions,
        "last_token": [int(prompt_last)],
    }
    if token_site not in by_site:
        raise ValueError(f"Unknown token_site={token_site!r}")
    requested = sorted({int(pos) for pos in by_site[token_site] if 0 <= int(pos) <= int(prompt_last)})
    if not requested:
        raise RuntimeError(f"sample_id={record.sample_id}: no positions for token_site={token_site}")
    debug = dict(debug)
    debug.update(
        {
            "token_site": token_site,
            "question_positions": list(question_positions),
            "semantic_question_positions": list(semantic_question_positions),
            "count_word_positions": list(count_word_positions),
            "last_token_position": int(prompt_last),
            "selected_positions": requested,
        }
    )
    return requested, int(prompt_last), debug


def prepare_carrier_batch(
    *,
    records: Sequence[prev.SampleRecord],
    sample_indices: Sequence[int],
    processor: Any,
    device: str,
    token_site: str,
) -> prev.QwenBatch:
    frames_by_record = [prev.load_frames(record) for record in records]
    conversations = [prev.build_conversation(record, frames) for record, frames in zip(records, frames_by_record)]
    try:
        if len(conversations) == 1:
            raw_inputs = processor.apply_chat_template(
                conversations[0],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            raw_inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
    finally:
        for frames in frames_by_record:
            for frame in frames:
                try:
                    frame.close()
                except Exception:
                    pass
    input_ids = raw_inputs["input_ids"]
    attention_mask = raw_inputs.get("attention_mask")
    target_positions: List[List[int]] = []
    prompt_last_indices: List[int] = []
    for batch_idx, record in enumerate(records):
        positions, prompt_last, _debug = locate_positions_for_site(
            input_ids_1d=input_ids[batch_idx],
            attention_mask_1d=attention_mask[batch_idx] if torch.is_tensor(attention_mask) else None,
            record=record,
            processor=processor,
            token_site=token_site,
        )
        target_positions.append(positions)
        prompt_last_indices.append(prompt_last)
    return prev.QwenBatch(
        inputs=prev.move_inputs_to_device(dict(raw_inputs), device),
        target_positions=target_positions,
        prompt_last_indices=torch.tensor(prompt_last_indices, device=device, dtype=torch.long),
        gold_counts=torch.tensor([int(record.gold_count) for record in records], device=device, dtype=torch.long),
        sample_indices=[int(idx) for idx in sample_indices],
    )


def make_carrier_adapter_from_checkpoint(
    checkpoint_path: Path,
    *,
    inject_layers: Sequence[int],
    alpha: float,
    injection_mode: str,
    query_mode: str,
) -> CarrierRetrievalAdapter:
    ckpt = prev.load_torch(Path(checkpoint_path))
    state = ckpt["adapter_state_dict"]
    input_dim = int(state["core.norm.weight"].numel())
    bottleneck_dim = int(state["core.w_p.weight"].shape[0])
    key_dim = int(state["core.q0"].numel())
    value_dim = int(state["core.w_v.weight"].shape[0])
    hidden_size = int(state["w_o.weight"].shape[0])
    core = prev.MessageMemoryCore(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        dropout=0.0,
    )
    core_state = {key.removeprefix("core."): value for key, value in state.items() if key.startswith("core.")}
    core.load_state_dict(core_state, strict=True)
    w_o = nn.Linear(value_dim, hidden_size, bias=False)
    with torch.no_grad():
        w_o.weight.copy_(state["w_o.weight"])
    adapter = CarrierRetrievalAdapter(
        core=core,
        w_o=w_o,
        hidden_size=hidden_size,
        inject_layers=inject_layers,
        alpha=alpha,
        injection_mode=injection_mode,
        query_mode=query_mode,
    )
    adapter.eval()
    return adapter


def make_previous_stage3_adapter(checkpoint_path: Path, *, inject_layers: Sequence[int]) -> WindowStage3ResidualAdapter:
    ckpt = prev.load_torch(Path(checkpoint_path))
    state = ckpt["adapter_state_dict"]
    input_dim = int(state["core.norm.weight"].numel())
    bottleneck_dim = int(state["core.w_p.weight"].shape[0])
    key_dim = int(state["core.q0"].numel())
    value_dim = int(state["core.w_v.weight"].shape[0])
    hidden_size = int(state["w_o.weight"].shape[0])
    core = prev.MessageMemoryCore(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        dropout=0.0,
    )
    adapter = WindowStage3ResidualAdapter(
        core=core,
        hidden_size=hidden_size,
        inject_layers=inject_layers,
        gamma_init=float(state.get("gamma", torch.tensor(1.0)).detach().cpu().item()),
        train_gamma=False,
    )
    adapter.load_state_dict(state, strict=True)
    adapter.set_inject_layers(inject_layers)
    adapter.eval()
    return adapter


def evaluate_base(
    *,
    method: str,
    model: Any,
    processor: Any,
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    count_token_ids: Dict[int, int],
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_score_by_idx: Dict[int, float] = {}
    ce_total = 0.0
    n = 0
    for batch_num, batch_indices in enumerate(prev.chunked(list(indices), int(batch_size)), start=1):
        batch_records = [records[idx] for idx in batch_indices]
        batch = prepare_carrier_batch(
            records=batch_records,
            sample_indices=batch_indices,
            processor=processor,
            device=device,
            token_site="room_char",
        )
        with torch.no_grad():
            outputs = model(**batch.inputs, use_cache=False)
            count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
            gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
            ce_total += float(F.cross_entropy(count_logits, gold_offsets, reduction="sum").detach().cpu().item())
            n += int(batch.gold_counts.numel())
            pred_offsets = count_logits.argmax(dim=-1)
            logits_cpu = count_logits.detach().float().cpu()
            for row, idx in enumerate(batch_indices):
                pred_by_idx[int(idx)] = int(pred_offsets[row].detach().cpu().item()) + min(count_token_ids.keys())
                values = [float(v) for v in logits_cpu[row].tolist()]
                logits_by_idx[int(idx)] = values
                gold_offset = int(records[int(idx)].gold_count) - min(count_token_ids.keys())
                gold_score_by_idx[int(idx)] = float(values[gold_offset])
        if batch_num == 1 or batch_num % 50 == 0:
            print(f"  eval {method}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    y_true = [int(records[int(idx)].gold_count) for idx in indices]
    y_pred = [pred_by_idx[int(idx)] for idx in indices]
    return {
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "mae": prev.mae(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
    }


def evaluate_original_stage3(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: WindowStage3ResidualAdapter,
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    x_messages: torch.Tensor,
    count_token_ids: Dict[int, int],
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    adapter.to(device)
    adapter.eval()
    adapter.enabled = True
    adapter.register_hooks(model)
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_score_by_idx: Dict[int, float] = {}
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
                carriers=("target_char", "target_room"),
            )
            adapter.set_context(x_messages[batch_indices].to(device), batch.target_positions)
            with torch.no_grad():
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce_total += float(F.cross_entropy(count_logits, gold_offsets, reduction="sum").detach().cpu().item())
                n += int(batch.gold_counts.numel())
                pred_offsets = count_logits.argmax(dim=-1)
                logits_cpu = count_logits.detach().float().cpu()
                for row, idx in enumerate(batch_indices):
                    pred_by_idx[int(idx)] = int(pred_offsets[row].detach().cpu().item()) + min(count_token_ids.keys())
                    values = [float(v) for v in logits_cpu[row].tolist()]
                    logits_by_idx[int(idx)] = values
                    gold_offset = int(records[int(idx)].gold_count) - min(count_token_ids.keys())
                    gold_score_by_idx[int(idx)] = float(values[gold_offset])
                    if adapter.last_delta_norm is not None:
                        delta_norm_by_idx[int(idx)] = float(adapter.last_delta_norm[row].item())
            adapter.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(f"  eval {method}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    finally:
        adapter.remove_hooks()
    y_true = [int(records[int(idx)].gold_count) for idx in indices]
    y_pred = [pred_by_idx[int(idx)] for idx in indices]
    return {
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "mae": prev.mae(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "delta_norm_by_idx": delta_norm_by_idx,
    }


def evaluate_carrier_variant(
    *,
    method: str,
    model: Any,
    processor: Any,
    adapter: CarrierRetrievalAdapter,
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    x_messages: torch.Tensor,
    count_token_ids: Dict[int, int],
    token_site: str,
    inject_layers: Sequence[int],
    alpha: float,
    injection_mode: str,
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    adapter.to(device)
    adapter.eval()
    adapter.enabled = True
    adapter.configure(inject_layers=inject_layers, alpha=alpha, injection_mode=injection_mode)
    adapter.register_hooks(model)
    pred_by_idx: Dict[int, int] = {}
    logits_by_idx: Dict[int, List[float]] = {}
    gold_score_by_idx: Dict[int, float] = {}
    mean_attention_by_idx: Dict[int, List[float]] = {}
    mean_residual_norm_by_idx: Dict[int, float] = {}
    total_residual_norm_by_idx: Dict[int, float] = {}
    total_residual_fro_by_idx: Dict[int, float] = {}
    num_tokens_by_idx: Dict[int, int] = {}
    ce_total = 0.0
    n = 0
    try:
        for batch_num, batch_indices in enumerate(prev.chunked(list(indices), int(batch_size)), start=1):
            batch_records = [records[idx] for idx in batch_indices]
            batch = prepare_carrier_batch(
                records=batch_records,
                sample_indices=batch_indices,
                processor=processor,
                device=device,
                token_site=token_site,
            )
            adapter.set_context(x_messages[batch_indices].to(device), batch.target_positions)
            with torch.no_grad():
                outputs = model(**batch.inputs, use_cache=False)
                count_logits = prev.select_count_logits(outputs.logits, batch.prompt_last_indices, count_token_ids)
                gold_offsets = batch.gold_counts.long() - min(count_token_ids.keys())
                ce_total += float(F.cross_entropy(count_logits, gold_offsets, reduction="sum").detach().cpu().item())
                n += int(batch.gold_counts.numel())
                pred_offsets = count_logits.argmax(dim=-1)
                logits_cpu = count_logits.detach().float().cpu()
                diag = adapter.diagnostics()
                for row, idx in enumerate(batch_indices):
                    idx = int(idx)
                    pred_by_idx[idx] = int(pred_offsets[row].detach().cpu().item()) + min(count_token_ids.keys())
                    values = [float(v) for v in logits_cpu[row].tolist()]
                    logits_by_idx[idx] = values
                    gold_offset = int(records[idx].gold_count) - min(count_token_ids.keys())
                    gold_score_by_idx[idx] = float(values[gold_offset])
                    if diag:
                        mean_attention_by_idx[idx] = [float(v) for v in diag["mean_attention"][row].tolist()]
                        mean_residual_norm_by_idx[idx] = float(diag["mean_residual_norm_per_token"][row].item())
                        total_residual_norm_by_idx[idx] = float(diag["total_residual_norm_sum"][row].item())
                        total_residual_fro_by_idx[idx] = float(diag["total_residual_norm_fro"][row].item())
                        num_tokens_by_idx[idx] = int(diag["num_tokens"][row].item())
            adapter.clear_context()
            if batch_num == 1 or batch_num % 50 == 0:
                print(f"  eval {method}: {min(len(indices), batch_num * int(batch_size))}/{len(indices)}")
    finally:
        adapter.remove_hooks()
    y_true = [int(records[int(idx)].gold_count) for idx in indices]
    y_pred = [pred_by_idx[int(idx)] for idx in indices]
    return {
        "ce": ce_total / max(1, n),
        "accuracy": prev.accuracy(y_true, y_pred),
        "mae": prev.mae(y_true, y_pred),
        "pred_by_idx": pred_by_idx,
        "logits_by_idx": logits_by_idx,
        "gold_score_by_idx": gold_score_by_idx,
        "mean_attention_by_idx": mean_attention_by_idx,
        "mean_residual_norm_by_idx": mean_residual_norm_by_idx,
        "total_residual_norm_by_idx": total_residual_norm_by_idx,
        "total_residual_fro_by_idx": total_residual_fro_by_idx,
        "num_tokens_by_idx": num_tokens_by_idx,
    }


def aggregate_counts(
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    pred_by_idx: Dict[int, int],
    selected_counts: Sequence[int],
) -> Tuple[float, float, int]:
    wanted = {int(count) for count in selected_counts}
    y_true = [int(records[int(idx)].gold_count) for idx in indices if int(records[int(idx)].gold_count) in wanted and int(idx) in pred_by_idx]
    y_pred = [int(pred_by_idx[int(idx)]) for idx in indices if int(records[int(idx)].gold_count) in wanted and int(idx) in pred_by_idx]
    if not y_true:
        return math.nan, math.nan, 0
    return prev.accuracy(y_true, y_pred), prev.mae(y_true, y_pred), len(y_true)


def summarize_predictions(
    *,
    method: str,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    eval_payload: Dict[str, Any],
    counts: Sequence[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pred_by_idx = eval_payload["pred_by_idx"]
    all_true = [int(records[int(idx)].gold_count) for idx in indices if int(idx) in pred_by_idx]
    all_pred = [int(pred_by_idx[int(idx)]) for idx in indices if int(idx) in pred_by_idx]
    mid_acc, mid_mae, mid_n = aggregate_counts(records, indices, pred_by_idx, MIDDLE_COUNTS)
    residual_values = [float(eval_payload.get("mean_residual_norm_by_idx", {}).get(int(idx), math.nan)) for idx in indices]
    total_values = [float(eval_payload.get("total_residual_norm_by_idx", {}).get(int(idx), math.nan)) for idx in indices]
    fro_values = [float(eval_payload.get("total_residual_fro_by_idx", {}).get(int(idx), math.nan)) for idx in indices]
    token_values = [float(eval_payload.get("num_tokens_by_idx", {}).get(int(idx), math.nan)) for idx in indices]
    finite_residual = [value for value in residual_values if math.isfinite(value)]
    finite_total = [value for value in total_values if math.isfinite(value)]
    finite_fro = [value for value in fro_values if math.isfinite(value)]
    finite_tokens = [value for value in token_values if math.isfinite(value)]
    summary = {
        **metadata,
        "method": method,
        "split": "test",
        "n": len(all_true),
        "accuracy": prev.accuracy(all_true, all_pred),
        "mae": prev.mae(all_true, all_pred),
        "mean_predicted_count": float(np.mean(all_pred)) if all_pred else math.nan,
        "ce": float(eval_payload.get("ce", math.nan)),
        "middle_count_accuracy_3_6": mid_acc,
        "middle_count_mae_3_6": mid_mae,
        "middle_count_n_3_6": mid_n,
        "mean_num_injected_tokens": float(np.mean(finite_tokens)) if finite_tokens else math.nan,
        "mean_injected_residual_norm_per_token": float(np.mean(finite_residual)) if finite_residual else math.nan,
        "mean_total_injected_residual_norm": float(np.mean(finite_total)) if finite_total else math.nan,
        "mean_total_injected_residual_fro": float(np.mean(finite_fro)) if finite_fro else math.nan,
    }
    per_count_rows: List[Dict[str, Any]] = []
    mean_rows: List[Dict[str, Any]] = []
    for count in counts:
        idxs = [int(idx) for idx in indices if int(records[int(idx)].gold_count) == int(count) and int(idx) in pred_by_idx]
        y_true = [int(records[idx].gold_count) for idx in idxs]
        y_pred = [int(pred_by_idx[idx]) for idx in idxs]
        per_count_rows.append(
            {
                **metadata,
                "method": method,
                "split": "test",
                "evidence_count": int(count),
                "n": len(y_true),
                "accuracy": prev.accuracy(y_true, y_pred) if y_true else math.nan,
                "mae": prev.mae(y_true, y_pred) if y_true else math.nan,
            }
        )
        mean_rows.append(
            {
                **metadata,
                "method": method,
                "split": "test",
                "evidence_count": int(count),
                "n": len(y_pred),
                "mean_predicted_count": float(np.mean(y_pred)) if y_pred else math.nan,
            }
        )
    distribution_rows: List[Dict[str, Any]] = []
    pred_counts = Counter(all_pred)
    for pred_count in counts:
        distribution_rows.append(
            {
                **metadata,
                "method": method,
                "split": "test",
                "predicted_count": int(pred_count),
                "n": int(pred_counts.get(int(pred_count), 0)),
                "fraction": float(pred_counts.get(int(pred_count), 0)) / max(1, len(all_pred)),
            }
        )
    sample_rows: List[Dict[str, Any]] = []
    for idx in indices:
        idx = int(idx)
        row = {
            **metadata,
            "method": method,
            "split": "test",
            "sample_index": idx,
            "sample_id": records[idx].sample_id,
            "evidence_count": int(records[idx].evidence_count),
            "gold_count": int(records[idx].gold_count),
            "pred_count": pred_by_idx.get(idx, ""),
            "correct": int(pred_by_idx.get(idx, -999) == int(records[idx].gold_count)),
            "gold_score": eval_payload.get("gold_score_by_idx", {}).get(idx, ""),
            "mean_injected_residual_norm_per_token": eval_payload.get("mean_residual_norm_by_idx", {}).get(idx, ""),
            "total_injected_residual_norm": eval_payload.get("total_residual_norm_by_idx", {}).get(idx, ""),
            "total_injected_residual_fro": eval_payload.get("total_residual_fro_by_idx", {}).get(idx, ""),
            "num_injected_tokens": eval_payload.get("num_tokens_by_idx", {}).get(idx, ""),
        }
        sample_rows.append(row)
    return summary, per_count_rows, mean_rows, distribution_rows, sample_rows


def retrieval_rows(
    *,
    method: str,
    metadata: Dict[str, Any],
    records: Sequence[prev.SampleRecord],
    indices: Sequence[int],
    eval_payload: Dict[str, Any],
    frame_labels: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in indices:
        idx = int(idx)
        weights = eval_payload.get("mean_attention_by_idx", {}).get(idx)
        if not weights:
            continue
        labels = frame_labels[idx].detach().cpu().tolist()
        for slot, value in enumerate(weights):
            rows.append(
                {
                    **metadata,
                    "method": method,
                    "split": "test",
                    "sample_index": idx,
                    "sample_id": records[idx].sample_id,
                    "evidence_count": int(records[idx].evidence_count),
                    "memory_slot": int(slot),
                    "is_evidence_frame": int(labels[slot]) if slot < len(labels) else "",
                    "mean_retrieval_weight": float(value),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    prev.write_csv(path, fieldnames, rows)


def best_row(rows: Sequence[Dict[str, Any]], predicate: Any, *, key: str = "accuracy", reverse: bool = True) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if predicate(row) and math.isfinite(float(row.get(key, math.nan)))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row.get(key, math.nan)), reverse=reverse)[0]


def heatmap_matrix(
    rows: Sequence[Dict[str, Any]],
    *,
    token_sites: Sequence[str],
    layer_windows: Sequence[str],
    value_key: str,
    reducer: str,
) -> np.ndarray:
    matrix = np.full((len(token_sites), len(layer_windows)), np.nan)
    for i, site in enumerate(token_sites):
        for j, window in enumerate(layer_windows):
            values = [
                float(row.get(value_key, math.nan))
                for row in rows
                if row.get("token_site") == site and row.get("layer_window") == window
            ]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            matrix[i, j] = min(values) if reducer == "min" else max(values)
    return matrix


def plot_heatmap(
    *,
    plot_dir: Path,
    rows: Sequence[Dict[str, Any]],
    token_sites: Sequence[str],
    layer_windows: Sequence[str],
    value_key: str,
    reducer: str,
    filename: str,
    title: str,
    cmap: str,
) -> None:
    matrix = heatmap_matrix(rows, token_sites=token_sites, layer_windows=layer_windows, value_key=value_key, reducer=reducer)
    fig, ax = plt.subplots(figsize=(max(8, len(layer_windows) * 1.2), max(4.8, len(token_sites) * 0.75)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(layer_windows)))
    ax.set_xticklabels(layer_windows, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(token_sites)))
    ax.set_yticklabels(token_sites)
    ax.set_xlabel("Layer window")
    ax.set_ylabel("Token site")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mode_bars(plot_dir: Path, rows: Sequence[Dict[str, Any]], token_sites: Sequence[str]) -> None:
    modes = ["raw_alpha", "alpha_div_sqrt_num_tokens"]
    x = np.arange(len(token_sites))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, mode in enumerate(modes):
        values: List[float] = []
        for site in token_sites:
            row = best_row(rows, lambda r, site=site, mode=mode: r.get("token_site") == site and r.get("injection_mode") == mode)
            values.append(float(row["accuracy"]) if row else math.nan)
        ax.bar(x + (offset - 0.5) * width, values, width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(token_sites, rotation=25, ha="right")
    ax.set_ylabel("Best accuracy")
    ax.set_title("Best Raw vs Token-Normalized Injection by Token Site")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "raw_vs_normalized_accuracy_by_token_site.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_count_curves(
    plot_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    per_count_rows: Sequence[Dict[str, Any]],
    token_sites: Sequence[str],
) -> None:
    selected: List[Tuple[str, str]] = []
    for site in token_sites:
        row = best_row(summary_rows, lambda r, site=site: r.get("token_site") == site)
        if row:
            selected.append((str(row["method"]), site))
    counts = sorted({int(row["evidence_count"]) for row in per_count_rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, label in selected:
        by_count = {int(row["evidence_count"]): row for row in per_count_rows if row.get("method") == method}
        values = [float(by_count.get(count, {}).get("accuracy", math.nan)) for count in counts]
        ax.plot(counts, values, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("Evidence count")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Count Accuracy for Best Variant of Each Token Site")
    ax.set_xticks(counts)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "per_count_accuracy_best_by_token_site.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_distribution(
    plot_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    distribution_rows: Sequence[Dict[str, Any]],
    counts: Sequence[int],
) -> None:
    best = best_row(summary_rows, lambda r: r.get("stage") == "carrier_sweep")
    methods = ["base_frozen_qwen"]
    labels = ["base"]
    if best:
        methods.append(str(best["method"]))
        labels.append("best")
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, (method, label) in enumerate(zip(methods, labels)):
        by_count = {
            int(row["predicted_count"]): float(row["fraction"])
            for row in distribution_rows
            if row.get("method") == method
        }
        values = [by_count.get(int(count), 0.0) for count in counts]
        ax.bar(x + (offset - (len(methods) - 1) / 2.0) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in counts])
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("Fraction")
    ax.set_title("Predicted Count Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "predicted_count_distribution_baseline_vs_best.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    *,
    output_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    per_count_rows: Sequence[Dict[str, Any]],
    distribution_rows: Sequence[Dict[str, Any]],
    token_sites: Sequence[str],
    layer_windows: Sequence[str],
    counts: Sequence[int],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = [row for row in summary_rows if row.get("stage") == "carrier_sweep"]
    plot_heatmap(
        plot_dir=plot_dir,
        rows=sweep_rows,
        token_sites=token_sites,
        layer_windows=layer_windows,
        value_key="accuracy",
        reducer="max",
        filename="heatmap_best_overall_accuracy.png",
        title="Best Overall Accuracy by Token Site and Layer Window",
        cmap="YlGnBu",
    )
    plot_heatmap(
        plot_dir=plot_dir,
        rows=sweep_rows,
        token_sites=token_sites,
        layer_windows=layer_windows,
        value_key="middle_count_accuracy_3_6",
        reducer="max",
        filename="heatmap_middle_count_accuracy.png",
        title="Best Middle-Count Accuracy by Token Site and Layer Window",
        cmap="YlGnBu",
    )
    plot_heatmap(
        plot_dir=plot_dir,
        rows=sweep_rows,
        token_sites=token_sites,
        layer_windows=layer_windows,
        value_key="mae",
        reducer="min",
        filename="heatmap_mae.png",
        title="Best MAE by Token Site and Layer Window",
        cmap="YlOrRd",
    )
    plot_mode_bars(plot_dir, sweep_rows, token_sites)
    plot_per_count_curves(plot_dir, sweep_rows, per_count_rows, token_sites)
    plot_predicted_distribution(plot_dir, summary_rows, distribution_rows, counts)


def write_summary(output_dir: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    sweep_rows = [row for row in summary_rows if row.get("stage") == "carrier_sweep"]
    best_overall = best_row(sweep_rows, lambda _row: True)
    best_middle = best_row(sweep_rows, lambda _row: True, key="middle_count_accuracy_3_6")
    best_norm = best_row(sweep_rows, lambda row: row.get("injection_mode") == "alpha_div_sqrt_num_tokens")
    norm_all_q = best_row(
        sweep_rows,
        lambda row: row.get("injection_mode") == "alpha_div_sqrt_num_tokens"
        and row.get("token_site") == "all_question_tokens",
    )
    norm_room = best_row(
        sweep_rows,
        lambda row: row.get("injection_mode") == "alpha_div_sqrt_num_tokens"
        and row.get("token_site") == "room_char",
    )
    room_1417 = best_row(sweep_rows, lambda row: row.get("token_site") == "room_char" and row.get("layer_window") == "14-17")
    room_1821 = best_row(sweep_rows, lambda row: row.get("token_site") == "room_char" and row.get("layer_window") == "18-21")
    broad = best_row(sweep_rows, lambda row: row.get("layer_window") == "14-21")
    non_broad = best_row(sweep_rows, lambda row: row.get("layer_window") != "14-21")

    def acc(row: Optional[Dict[str, Any]], key: str = "accuracy") -> float:
        return float(row.get(key, math.nan)) if row is not None else math.nan

    all_q_beats_norm = acc(norm_all_q) > acc(norm_room)
    earlier_beats_later = acc(room_1417) > acc(room_1821)
    broad_helps = acc(broad) > acc(non_broad)
    if all_q_beats_norm:
        interpretation = (
            "Normalized all-question injection still wins over room+char, so the benefit is less likely "
            "to be only residual-energy scaling and more consistent with carrier/question aggregation helping."
        )
    else:
        interpretation = (
            "After token-count normalization, all-question does not beat room+char, so any raw all-question gain "
            "should be treated as possible answer steering or residual-energy scaling rather than cleaner aggregation."
        )

    lines = [
        "Memory carrier site/layer/norm sweep seq_len=8 7B",
        "",
        "Best variants:",
        f"- Best overall: {best_overall['method'] if best_overall else 'none'} "
        f"acc={acc(best_overall):.4f}, mae={acc(best_overall, 'mae'):.4f}",
        f"- Best middle-count 3-6: {best_middle['method'] if best_middle else 'none'} "
        f"mid_acc={acc(best_middle, 'middle_count_accuracy_3_6'):.4f}, acc={acc(best_middle):.4f}",
        f"- Best normalized: {best_norm['method'] if best_norm else 'none'} "
        f"acc={acc(best_norm):.4f}, mae={acc(best_norm, 'mae'):.4f}",
        "",
        "Controlled questions:",
        f"1. Does all_question_tokens still beat room_char after sqrt(|S|) normalization? "
        f"{'Yes' if all_q_beats_norm else 'No'} "
        f"(all_question={acc(norm_all_q):.4f}, room_char={acc(norm_room):.4f}).",
        f"2. Do earlier layers 14-17 beat later layers 18-21 for room_char? "
        f"{'Yes' if earlier_beats_later else 'No'} "
        f"(14-17={acc(room_1417):.4f}, 18-21={acc(room_1821):.4f}).",
        f"3. Does broader 14-21 injection help? {'Yes' if broad_helps else 'No'} "
        f"(best_14-21={acc(broad):.4f}, best_non_broad={acc(non_broad):.4f}).",
        "",
        "Interpretation:",
        f"- {interpretation}",
        "- The adapter uses the existing message-memory checkpoint. Because that checkpoint has no trained hidden-state W_Q, "
        "the default query projection is initialized to reproduce the checkpoint q0 query while still applying per-token "
        "retrieval, residual scaling, and token-count normalization.",
        "",
        "Top 10 by overall accuracy:",
    ]
    top = sorted(sweep_rows, key=lambda row: float(row.get("accuracy", -math.inf)), reverse=True)[:10]
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"{rank}. {row['method']}: acc={float(row['accuracy']):.4f}, "
            f"mid={float(row['middle_count_accuracy_3_6']):.4f}, mae={float(row['mae']):.4f}, "
            f"mean_token_norm={float(row.get('mean_injected_residual_norm_per_token', math.nan)):.3f}, "
            f"mean_total_norm={float(row.get('mean_total_injected_residual_norm', math.nan)):.3f}"
        )
    (output_dir / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def field_union(rows: Sequence[Dict[str, Any]], leading: Sequence[str]) -> List[str]:
    keys = sorted({key for row in rows for key in row})
    return list(leading) + [key for key in keys if key not in set(leading)]


def main() -> int:
    args = parse_args()
    args.evidence_counts = prev.parse_int_tokens(args.evidence_counts)
    args.layers = prev.parse_int_tokens(args.layers)
    token_sites = prev.split_tokens(args.token_sites)
    unknown_sites = [site for site in token_sites if site not in TOKEN_SITE_ORDER]
    if unknown_sites:
        raise ValueError(f"Unknown token sites: {unknown_sites}")
    layer_specs = parse_layer_windows(args.layer_windows)
    alpha_values = parse_float_tokens(args.alpha_values)
    injection_modes = prev.split_tokens(args.injection_modes)
    unknown_modes = [mode for mode in injection_modes if mode not in {"raw_alpha", "alpha_div_sqrt_num_tokens"}]
    if unknown_modes:
        raise ValueError(f"Unknown injection modes: {unknown_modes}")

    output_dir = (args.output_dir if args.output_dir is not None else default_output_dir()).resolve()
    log_handle, old_stdout, old_stderr = prev.setup_logging(output_dir)
    started = time.time()
    try:
        prev.set_seed(int(args.seed))
        if not Path(args.stage3_checkpoint).is_file():
            raise FileNotFoundError(f"Missing message-memory Stage 3 checkpoint: {args.stage3_checkpoint}")

        configs: List[Dict[str, Any]] = []
        for token_site in token_sites:
            for spec in layer_specs:
                for injection_mode in injection_modes:
                    for alpha in alpha_values:
                        configs.append(
                            {
                                "token_site": token_site,
                                "layer_window": str(spec["layer_label"]),
                                "inject_layers": list(spec["inject_layers"]),
                                "injection_mode": injection_mode,
                                "alpha": float(alpha),
                            }
                        )
        if int(args.smoke_limit_configs) > 0:
            configs = configs[: max(1, int(args.smoke_limit_configs))]
            token_sites = sorted({str(config["token_site"]) for config in configs}, key=TOKEN_SITE_ORDER.index)
            layer_specs = [spec for spec in layer_specs if str(spec["layer_label"]) in {str(config["layer_window"]) for config in configs}]
        prev.write_json(output_dir / "configs.json", {"configs": configs})

        run_config = {
            "experiment_name": EXPERIMENT_NAME,
            "model_name": str(args.model_name),
            "dataset_root": os.fspath(args.dataset_root),
            "source_run": os.fspath(args.source_run),
            "base_source_run": os.fspath(args.base_source_run),
            "message_memory_run": os.fspath(args.message_memory_run),
            "stage3_checkpoint": os.fspath(args.stage3_checkpoint),
            "output_dir": os.fspath(output_dir),
            "split": str(args.split),
            "seq_len": int(args.seq_len),
            "evidence_counts": list(args.evidence_counts),
            "feature_layers": list(args.layers),
            "token_sites": list(token_sites),
            "layer_windows": [str(spec["layer_label"]) for spec in layer_specs],
            "alpha_values": list(alpha_values),
            "injection_modes": list(injection_modes),
            "query_mode": str(args.query_mode),
            "max_samples_per_count": int(args.max_samples_per_count),
            "candidate_min": int(args.candidate_min),
            "candidate_max": int(args.candidate_max),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
            "load_in_4bit": bool(args.load_in_4bit),
            "smoke_limit_configs": int(args.smoke_limit_configs),
            "smoke_limit_samples": int(args.smoke_limit_samples),
        }
        prev.write_json(output_dir / "run_config.json", run_config)
        print(f"Output dir: {output_dir}")
        print(f"Config: {json.dumps(run_config, sort_keys=True)}")

        feature_data = prev.load_message_features(args, args.layers, args.evidence_counts)
        sample_ids = feature_data["sample_ids"]
        labels = feature_data["labels"]
        frame_labels = feature_data["frame_labels"]
        x_messages = feature_data["x_messages"]
        records = prev.load_records(args.dataset_root, args.split, args.seq_len, sample_ids)
        splits = prev.stratified_split(sample_ids, labels, int(args.seed))
        test_indices = list(splits["test"])
        if int(args.smoke_limit_samples) > 0:
            test_indices = test_indices[: max(1, int(args.smoke_limit_samples))]
        counts = list(range(int(args.candidate_min), int(args.candidate_max) + 1))
        print(f"x_messages shape={tuple(x_messages.shape)} D_msg={int(x_messages.shape[-1])}")
        for split, row in prev.split_counts(splits, labels, counts).items():
            print(f"  {split}: " + ", ".join(f"{count}:{row.get(count, 0)}" for count in counts))
        print(f"Test samples used: {len(test_indices)}")

        device = prev.resolve_device(str(args.device))
        dtype = prev.resolve_dtype(str(args.dtype), device)
        model, processor = prev.load_model_and_processor(args, device=device, dtype=dtype)
        hidden_size = prev.hidden_size_from_model(model)
        candidate_format, count_ids = prev.candidate_token_ids(processor.tokenizer, int(args.candidate_min), int(args.candidate_max))
        print(f"Loaded Qwen hidden_size={hidden_size} candidate_format={candidate_format} count_ids={count_ids}")

        summary_rows: List[Dict[str, Any]] = []
        per_count_rows: List[Dict[str, Any]] = []
        mean_rows: List[Dict[str, Any]] = []
        distribution_rows: List[Dict[str, Any]] = []
        per_sample_rows: List[Dict[str, Any]] = []
        retrieval_weight_rows: List[Dict[str, Any]] = []

        print("Evaluating frozen Qwen baseline")
        base_eval = evaluate_base(
            method="base_frozen_qwen",
            model=model,
            processor=processor,
            records=records,
            indices=test_indices,
            count_token_ids=count_ids,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        base_meta = {
            "stage": "baseline",
            "source_checkpoint": "",
            "token_site": "none",
            "layer_window": "",
            "inject_layers": "",
            "injection_mode": "none",
            "alpha": "",
            "query_mode": "",
        }
        summary, per_count, means, dist, samples = summarize_predictions(
            method="base_frozen_qwen",
            metadata=base_meta,
            records=records,
            indices=test_indices,
            eval_payload=base_eval,
            counts=counts,
        )
        summary_rows.append(summary)
        per_count_rows.extend(per_count)
        mean_rows.extend(means)
        distribution_rows.extend(dist)
        per_sample_rows.extend(samples)

        if bool(args.include_previous_baseline):
            print("Evaluating previous Stage 3 message-memory baseline at room_char L18")
            previous_adapter = make_previous_stage3_adapter(Path(args.stage3_checkpoint), inject_layers=[18])
            previous_eval = evaluate_original_stage3(
                method="previous_stage3_room_char_L18",
                model=model,
                processor=processor,
                adapter=previous_adapter,
                records=records,
                indices=test_indices,
                x_messages=x_messages,
                count_token_ids=count_ids,
                device=device,
                batch_size=int(args.eval_batch_size),
            )
            prev_meta = {
                "stage": "previous_baseline",
                "source_checkpoint": os.fspath(args.stage3_checkpoint),
                "token_site": "room_char",
                "layer_window": "18",
                "inject_layers": "18",
                "injection_mode": "checkpoint_gamma",
                "alpha": "",
                "query_mode": "stage3_original",
            }
            summary, per_count, means, dist, samples = summarize_predictions(
                method="previous_stage3_room_char_L18",
                metadata=prev_meta,
                records=records,
                indices=test_indices,
                eval_payload=previous_eval,
                counts=counts,
            )
            summary_rows.append(summary)
            per_count_rows.extend(per_count)
            mean_rows.extend(means)
            distribution_rows.extend(dist)
            per_sample_rows.extend(samples)
            previous_adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        first = configs[0]
        adapter = make_carrier_adapter_from_checkpoint(
            Path(args.stage3_checkpoint),
            inject_layers=first["inject_layers"],
            alpha=float(first["alpha"]),
            injection_mode=str(first["injection_mode"]),
            query_mode=str(args.query_mode),
        )
        for config_idx, config in enumerate(configs, start=1):
            token_site = str(config["token_site"])
            layer_window = str(config["layer_window"])
            inject_layers = [int(layer) for layer in config["inject_layers"]]
            injection_mode = str(config["injection_mode"])
            alpha = float(config["alpha"])
            method = f"carrier__{token_site}__L{layer_window}__{injection_mode}__a{safe_name(alpha)}"
            print(
                f"Evaluating {method} ({config_idx}/{len(configs)}): "
                f"site={token_site} layers={inject_layers} mode={injection_mode} alpha={alpha}"
            )
            eval_payload = evaluate_carrier_variant(
                method=method,
                model=model,
                processor=processor,
                adapter=adapter,
                records=records,
                indices=test_indices,
                x_messages=x_messages,
                count_token_ids=count_ids,
                token_site=token_site,
                inject_layers=inject_layers,
                alpha=alpha,
                injection_mode=injection_mode,
                device=device,
                batch_size=int(args.eval_batch_size),
            )
            metadata = {
                "stage": "carrier_sweep",
                "source_checkpoint": os.fspath(args.stage3_checkpoint),
                "token_site": token_site,
                "layer_window": layer_window,
                "inject_layers": " ".join(str(layer) for layer in inject_layers),
                "injection_mode": injection_mode,
                "alpha": alpha,
                "query_mode": str(args.query_mode),
            }
            summary, per_count, means, dist, samples = summarize_predictions(
                method=method,
                metadata=metadata,
                records=records,
                indices=test_indices,
                eval_payload=eval_payload,
                counts=counts,
            )
            summary_rows.append(summary)
            per_count_rows.extend(per_count)
            mean_rows.extend(means)
            distribution_rows.extend(dist)
            per_sample_rows.extend(samples)
            retrieval_weight_rows.extend(
                retrieval_rows(
                    method=method,
                    metadata=metadata,
                    records=records,
                    indices=test_indices,
                    eval_payload=eval_payload,
                    frame_labels=frame_labels,
                )
            )
            adapter.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        leading = [
            "method",
            "stage",
            "token_site",
            "layer_window",
            "injection_mode",
            "alpha",
            "query_mode",
            "split",
        ]
        write_csv(output_dir / "metrics.csv", field_union(summary_rows, leading), summary_rows)
        write_csv(output_dir / "overall_metrics.csv", field_union(summary_rows, leading), summary_rows)
        write_csv(
            output_dir / "per_count_accuracy.csv",
            field_union(per_count_rows, leading + ["evidence_count"]),
            per_count_rows,
        )
        write_csv(
            output_dir / "mean_predicted_count_by_evidence_count.csv",
            field_union(mean_rows, leading + ["evidence_count"]),
            mean_rows,
        )
        write_csv(
            output_dir / "predicted_count_distribution.csv",
            field_union(distribution_rows, leading + ["predicted_count"]),
            distribution_rows,
        )
        write_csv(
            output_dir / "retrieval_weights.csv",
            field_union(retrieval_weight_rows, leading + ["sample_index", "memory_slot"]),
            retrieval_weight_rows,
        )
        write_csv(
            output_dir / "per_sample_predictions.csv",
            field_union(per_sample_rows, leading + ["sample_index", "sample_id"]),
            per_sample_rows,
        )

        debug = {
            "runtime_seconds": time.time() - started,
            "source_cache": os.fspath(feature_data["cache_path"]),
            "x_messages_shape": list(x_messages.shape),
            "candidate_format": candidate_format,
            "count_token_ids": {str(k): int(v) for k, v in count_ids.items()},
            "num_configs": len(configs),
            "best_overall": best_row([row for row in summary_rows if row.get("stage") == "carrier_sweep"], lambda _row: True),
        }
        prev.write_json(output_dir / "adapter_debug.json", debug)

        if not bool(args.no_plots):
            make_plots(
                output_dir=output_dir,
                summary_rows=summary_rows,
                per_count_rows=per_count_rows,
                distribution_rows=distribution_rows,
                token_sites=token_sites,
                layer_windows=[str(spec["layer_label"]) for spec in layer_specs],
                counts=counts,
            )
        write_summary(output_dir, summary_rows)
        print(f"Finished {EXPERIMENT_NAME}: {output_dir}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
