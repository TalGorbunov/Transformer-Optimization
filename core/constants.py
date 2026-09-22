"""Constants. Benchmark vocabulary is the OFFICIAL one (Fr0do/mmred `mmred/const.py`);
tests/test_mmred.py asserts equality with the installed `mmred` package when available.

Twin: legacy/v1/gnnformer/constants.py (park-era values dropped: ROOMS had "Park",
CARRIER_TOKEN / READ_LAYER / ANCHOR_OFFSET / L_OPEN belonged to the retired carrier
stack, FRAME_RESIZE=392 is gone — frames are used at native resolution).
"""
from __future__ import annotations

# --- the benchmark (order = upstream mmred.const) ---------------------------------------
ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]
CHARS = ["Sandra", "Mary", "John", "Daniel", "Michael"]
NOBODY = "Nobody"

# HF release: ef1e43ce/mmred. Train/val/test exist for seq_len <= 16; 32/64/128 are test-only.
HF_DATASET = "ef1e43ce/mmred"
SEQ_LENS = [1, 2, 4, 8, 16, 32, 64, 128]
TRAIN_SEQ_LENS = [1, 2, 4, 8, 16]          # the paper's protocol: train short, test all

# --- the model -------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"   # frozen, 4-bit nf4, bf16 compute, sdpa
VISION_START = "<|vision_start|>"           # one per image, before its image tokens
VISION_END = "<|vision_end|>"               # one per image, after its image tokens = the slot
IMAGE_PAD = "<|image_pad|>"                 # 324 per frame at the native 512 px renders

# --- masks -----------------------------------------------------------------------------
# Additive float mask: 0 = attend, MASK_MIN = forbidden. -65504 is the most negative bf16
# finite value; it is what the legacy fence used and what the parity test pins.
MASK_MIN = -65504.0
