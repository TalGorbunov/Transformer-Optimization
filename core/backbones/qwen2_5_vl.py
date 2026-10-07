"""Qwen2.5-VL-7B-Instruct: the anchor backbone, as data. Every value here is what the pre-seam
code had inline (core/constants.py, core/model.py, experiments/train.py LORA_TARGETS,
experiments/gate_capture.py --hs-layers default, tests/test_model.py NATIVE_TOKENS)."""
from __future__ import annotations

from ..constants import IMAGE_PAD, MODEL_ID, TURN_END, VISION_END, VISION_START
from .base import BackboneSpec, Delimiters

QWEN2_5_VL_7B = BackboneSpec(
    name="qwen2.5-vl-7b",
    model_id=MODEL_ID,
    family="qwen2.5-vl",
    delimiters=Delimiters(start=VISION_START, end=VISION_END, pad=IMAGE_PAD, turn_end=TURN_END),
    block_rule="pair",
    rope="mrope",                       # 3-D M-RoPE via get_rope_index; reset_positions sees (3, 1, L)
    image_attention="causal",
    layer_types=None,                   # every decoder layer is global causal
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    probe_layers=(12, 20, 24),          # legacy hidden-state numbering (D4: L20 is the gate layer)
    tokens_per_frame_512=324,           # 512 px -> smart_resize 504 -> (36/2)^2 merged patches
    processor_kwargs={},                # native resolution: the processor's own defaults, untouched
    use_fast_processor=False,
    pixel_knob="max_pixels",            # default 12,845,056 px: a 1080p frame is shown whole (2,691 tokens)
    passthrough_keys=("pixel_values", "image_grid_thw"),
    res_modes={"lo": {"max_pixels": 151_200},       # what a 512 px MMReD frame gets (~180 tokens for a 1080p frame)
               "hi": {"max_pixels": 1_003_520}},    # the HERBench operating point (2026-09-30: full res buys nothing)
)
