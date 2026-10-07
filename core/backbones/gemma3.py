"""Gemma-3-12B-it, as data. Every value is from experiments/backbone_probe.py on the cached config +
processor (outputs/scaleup/_backbones/google__gemma-3-12b-it.json, 2026-09-28):
Gemma3ForConditionalGeneration; text model gemma3_text, 48 layers, 16 heads / 8 KV, hidden 3840,
head_dim 256, 1-D RoPE; layer_types = 40 sliding_attention (window 1024) + 8 full_attention (every
6th); image processor Gemma3ImageProcessor, 896 x 896, pan-and-scan off -> 256 <image_soft_token>
per frame (the processor resizes); a frame tokenises as <start_of_image> 256 x <image_soft_token>
<end_of_image> + ONE separator token ("\n\n\n\n" between frames, "\n\n" after the last); the
processor also returns token_type_ids (1 on image tokens), which the model's own mask builder needs
on the unfenced arms.

What differs from Qwen and is handled in core.backbones.base.fenced_setup:
  * image tokens attend each other in both directions -> the image run of each block is a full square;
  * local layers see 1024 tokens -> under the fence the window is taken in reset-position space, so
    prefix + own block (about 20 + 259 tokens) is inside it in every layer. On the UNFENCED arms the
    model's own index-based window applies: past ~4 frames most layers cannot see a question placed
    in the prefix. Gemma's plain / qfirst rows are therefore a different regime from Qwen's and are
    never pooled with them (docs/SCALEUP_2026-09-23.md §4).
No system role: the chat template folds a system prompt into the first user turn.
"""
from __future__ import annotations

from .base import BackboneSpec, Delimiters

GEMMA_3_12B = BackboneSpec(
    name="gemma-3-12b",
    model_id="google/gemma-3-12b-it",
    family="gemma-3",
    delimiters=Delimiters(start="<start_of_image>", end="<end_of_image>", pad="<image_soft_token>", turn_end="<end_of_turn>"),
    block_rule="pair",
    sep_tokens=1,                        # "\n\n\n\n" between frames / "\n\n" after the last: one token either way
    rope="1d",
    image_attention="bidirectional",
    layer_types="config",                # full / sliding per layer, window from the text config
    lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    probe_layers=(14, 21, 27, 34, 41),   # depth fractions 0.30 / 0.43 / 0.57 / 0.71 / 0.86 of 48
    tokens_per_frame_512=256,
    processor_kwargs={},
    torch_dtype="bfloat16",              # float16 (the checkpoint default under bnb) -> NaN logits
    use_fast_processor=None,             # default = fast tokenizer (the slow GemmaTokenizer needs sentencepiece)
    passthrough_keys=("pixel_values", "token_type_ids"),
    res_modes={"lo": {"do_pan_and_scan": False},                                   # one 896 tile = 256 tokens
               "hi": {"do_pan_and_scan": True, "pan_and_scan_min_crop_size": 256,   # the model's own pan-and-scan
                      "pan_and_scan_max_num_crops": 4, "pan_and_scan_min_ratio_to_activate": 1.2}},
)
