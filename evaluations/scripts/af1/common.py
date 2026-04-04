"""Shared AF1 constants and data structures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

NEG_INF = -1.0e9
DONOR_POLICY = "same_seq_len_validated_layout_seeded_shuffle_exclude_target"
VALID_MODES = ("full_af1", "wait_only", "mask_only")
VALID_INSTRUCTION_MASK_MODES = (
    "base",
    "vision_end_only",
    "vision_boundary_only",
    "prompt_only",
    "image_pad_only",
)
INSTRUCTION_TRANSFER_PROMPT_SPAN = "Output only the integer.\n"
SUMMARY_FIELDS = [
    "model",
    "mode",
    "seq_len",
    "wait_layer",
    "transfer_layers",
    "n_total",
    "n_used",
    "n_clean_correct",
    "clean_acc",
    "af1_acc",
    "af1_faith",
    "mean_clean_top1_score_drop",
    "mean_gold_answer_score_drop",
]
PER_SAMPLE_FIELDS = [
    "model",
    "mode",
    "sample_id",
    "seq_len",
    "used",
    "gold_answer",
    "clean_pred",
    "clean_correct",
    "clean_gold_prob",
    "clean_best_score",
    "clean_margin_over_second",
    "af1_clean_top1_score",
    "clean_top1_score_drop",
    "gold_answer_score_drop",
    "af1_pred",
    "af1_correct",
    "af1_gold_prob",
    "af1_best_score",
    "af1_margin_over_second",
    "carrier_index",
    "carrier_token",
    "wait_layer",
    "transfer_layers",
    "transfer_layer_indices",
    "k_donors",
    "num_frames",
    "num_frame_groups",
    "prompt_len",
    "image_tokens_per_frame",
    "room_text",
    "skipped_reason",
    "donor_ids",
    "layout_match_status",
    "layout_match_details",
]


@dataclass(frozen=True)
class SampleLayout:
    sample_id: str
    seq_len: int
    prompt_len: int
    carrier_index: int
    carrier_token_id: int
    carrier_token_text: str
    prompt_family_key: str
    frame_groups: Tuple[Tuple[int, ...], ...]
    image_tokens_per_frame: Tuple[int, ...]
    room_text: str
    room_positions: Tuple[int, ...]
    character_positions: Tuple[int, ...]
    instruction_positions: Tuple[int, ...]
    image_pad_positions: Tuple[int, ...]
    vision_start_positions: Tuple[int, ...]
    vision_end_positions: Tuple[int, ...]
    room_span_len: int
    prompt_input_ids: Tuple[int, ...]
    prompt_decoded_tokens: Tuple[str, ...]


@dataclass
class PreparedSample:
    sample_dir: Path
    sample_id: str
    frames: List[Any]
    question: str
    gold_answer: str
    layout: SampleLayout
    inputs_cpu: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class AttentionPolicy:
    prompt_len: int
    carrier_index: int
    wait_layer: int
    transfer_layers: int
    num_model_layers: int
    frame_group_by_token: Dict[int, Tuple[int, ...]]
    instruction_positions: Tuple[int, ...]
    image_pad_positions: Tuple[int, ...]
    vision_start_positions: Tuple[int, ...]
    vision_end_positions: Tuple[int, ...]
    instruction_mask_mode: str

    @property
    def transfer_layer_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.wait_layer, self.wait_layer + self.transfer_layers))

    @property
    def post_transfer_start(self) -> int:
        return self.wait_layer + self.transfer_layers

    def stage_for_layer(self, layer_idx: int) -> str:
        if layer_idx < self.wait_layer:
            return "wait"
        if layer_idx < self.wait_layer + self.transfer_layers:
            return "transfer"
        return "post_transfer"
