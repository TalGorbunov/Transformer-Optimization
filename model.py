import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
hf_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
)

def find_blocks(m):
    """
    Returns (blocks, blocks_path_string) where blocks is a list-like ModuleList of transformer blocks.
    Tries common architectures; if it fails, raises with a helpful message.
    """
    candidates = [
        # SmolVLM2 text backbone (LlamaModel)
        ("text_model.layers", lambda x: getattr(getattr(x, "text_model", None), "layers", None)),
        # (optional) vision encoder blocks, if you ever want them
        ("vision_model.encoder.layers", lambda x: getattr(getattr(getattr(x, "vision_model", None), "encoder", None), "layers", None)),
        # Generic fallbacks for other architectures (when m is a HF model)
        ("model.layers", lambda x: getattr(getattr(x, "model", None), "layers", None)),
        ("transformer.h", lambda x: getattr(getattr(x, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda x: getattr(getattr(x, "gpt_neox", None), "layers", None)),
        ("decoder.layers", lambda x: getattr(getattr(x, "decoder", None), "layers", None)),
        ("model.decoder.layers", lambda x: getattr(getattr(getattr(x, "model", None), "decoder", None), "layers", None)),
    ]


    for path, getter in candidates:
        blocks = getter(m)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            return blocks, path

    # If we got here, we didn't find blocks in common places.
    # Print a short hint on what to do next.
    hint = []
    for name, mod in m.named_modules():
        if name.endswith(("layers", "h")):
            hint.append(name)
        if len(hint) >= 10:
            break

    raise RuntimeError(
        "Couldn't automatically find transformer blocks.\n"
        "Try inspecting lm.model.named_modules() for a ModuleList of blocks.\n"
        f"Some candidates I saw: {hint}"
    )