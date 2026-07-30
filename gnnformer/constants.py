"""Project-wide constants. Values are load-bearing for reproducing RESULTS.md numbers."""

# Model ids. The thesis model is the 7B; the 32B is opt-in for heavy mechanistic runs.
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_32B = "Qwen/Qwen2.5-VL-32B-Instruct"

# Additive-attention-mask "minus infinity" that survives fp16/bf16.
MASK_MIN = -65504.0

# MMRED rooms (Park variant replaces Hallway).
ROOMS = ("Park", "Garden", "Bathroom", "Kitchen", "Office", "Bedroom")

# Carrier placeholder token (id 151648 in the Qwen2.5-VL tokenizer).
CARRIER_TOKEN = "<|box_start|>"

# Deployed message locus: layer whose attention messages are read/distilled,
# and the final-question room-token offset from the end of the prompt.
READ_LAYER = 16
ANCHOR_OFFSET = 9  # anchor position = seq - 1 - ANCHOR_OFFSET

# Separator layer: fenced supply in layers < L_OPEN, trained integration >= L_OPEN.
# E-H curve ([2026-07-23] RESULTS.md): inverted-U with peak at 12.
L_OPEN = 12

# Default frame resize used by every image experiment.
FRAME_RESIZE = 392
