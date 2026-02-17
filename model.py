import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="cuda",  
)
model.eval()

def get_layers(m):
    """
    Returns (layers) where layers is a list-like ModuleList of transformer blocks.
    Tries common architectures; if it fails, raises with a helpful message.
    """
    candidates = [
        # Gemma 4 language layers
        ("gemma3_language_layers", lambda m: m.language_model.layers),   
        # SmolVLM2 text leyers
        ("text_model.layers", lambda x: getattr(getattr(x, "text_model", None), "layers", None)),
    ]


    for _, getter in candidates:
        try:
            layers = getter(m)
            if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except AttributeError:
            continue

    raise RuntimeError(
        "Couldn't automatically find transformer layers.\n"
    )
