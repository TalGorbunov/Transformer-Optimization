"""InternVL3.5-8B (HF-native port), as data. Every value is from experiments/backbone_probe.py on the
cached config + processor (outputs/scaleup/_backbones/OpenGVLab__InternVL3_5-8B-HF.json, 2026-09-28):
InternVLForConditionalGeneration; text model qwen3, 36 layers, 32 heads / 8 KV, hidden 4096, plain
1-D RoPE (rope_scaling None), every layer full attention; image processor GotOcr2ImageProcessorFast,
448 x 448, crop_to_patches False by default -> one tile = 256 <IMG_CONTEXT> tokens per frame (the
PROCESSOR resizes our 512 px frames to 448; that is the model's own preprocessing, not ours);
a frame tokenises as <img> 256 x <IMG_CONTEXT> </img> "\n" and frames follow each other directly.

The language model is Qwen3-8B: a different vision tower and VL wiring, NOT a different LM lineage —
every table says so (docs/SCALEUP_2026-09-23.md §4).
"""
from __future__ import annotations

from .base import BackboneSpec, Delimiters

INTERNVL3_5_8B = BackboneSpec(
    name="internvl3.5-8b",
    model_id="OpenGVLab/InternVL3_5-8B-HF",
    family="internvl3.5 (Qwen3-8B LM)",
    delimiters=Delimiters(start="<img>", end="</img>", pad="<IMG_CONTEXT>", turn_end="<|im_end|>"),
    block_rule="pair",
    sep_tokens=1,                        # the "\n" after every </img>
    rope="1d",
    image_attention="causal",
    layer_types=None,
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    probe_layers=(11, 15, 21, 26, 31),   # depth fractions 0.30 / 0.43 / 0.57 / 0.71 / 0.86 of 36 (Qwen's 12/20/24 of 28 + two)
    tokens_per_frame_512=256,
    processor_kwargs={},
    use_fast_processor=None,             # the default processor (GotOcr2ImageProcessorFast + fast tokenizer)
    assistant_prefix="<think>\n\n</think>\n\n",   # NON-THINKING MODE (decided by Tal, 2026-09-29). Qwen3's convention: without it the model opens a
                                         # <think> block on most MMReD rows and never reaches the answer in 24 tokens
                                         # (smoke 164447, 2026-09-28: 7/12 rows)
    passthrough_keys=("pixel_values",),
    res_modes={"lo": {"crop_to_patches": False},                                   # one 448 tile = 256 tokens
               "hi": {"crop_to_patches": True, "min_patches": 1, "max_patches": 12}},   # the model's own dynamic tiling
)
