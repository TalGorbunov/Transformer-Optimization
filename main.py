import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from nnsight import LanguageModel

MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    dtype=torch.float16,          
    device_map="auto",
).eval()

print("loaded", type(model))

# Wrap with nnsight
lm = LanguageModel(model, processor=processor)

# --- Helper: find transformer blocks list (works across many model families) ---
def find_blocks(m):
    """
    Returns (blocks, blocks_path_string) where blocks is a list-like ModuleList of transformer blocks.
    """
    # Common places: model.layers (Llama/Mistral), transformer.h (GPT2), gpt_neox.layers, etc.
    candidates = [
        ("model.layers",        lambda x: getattr(getattr(x, "model", None), "layers", None)),
        ("transformer.h",       lambda x: getattr(getattr(x, "transformer", None), "h", None)),
        ("gpt_neox.layers",     lambda x: getattr(getattr(getattr(x, "gpt_neox", None), "layers", None), "__len__", None) and x.gpt_neox.layers),
        ("model.decoder.layers",lambda x: getattr(getattr(getattr(x, "model", None), "decoder", None), "layers", None)),
    ]
    for path, getter in candidates:
        blocks = getter(m)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            return blocks, path
    raise RuntimeError("Couldn't automatically find transformer blocks. We'll need to inspect named_modules().")

blocks, blocks_path = find_blocks(lm.model)
n_layers = len(blocks)
print(f"Found blocks at: {blocks_path}  |  num_layers={n_layers}")

# --- Choose what to cache ---
# We'll cache:
# 1) embedding output (if accessible)
# 2) each block output (last hidden state after that layer)
# 3) final logits
#
# Note: in nnsight, you can do <node>.save() to store the tensor from that point in the graph.

prompt = "User: How many steps did Alice spend in kitchen?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to the same device as the model’s first parameter
first_param = next(lm.model.parameters())
device = first_param.device
inputs = {k: v.to(device) for k, v in inputs.items()}

cached = {}

with lm.trace(inputs) as tr:
    # Cache per-layer outputs (this is the most useful “activation cache”)
    layer_outs = []
    for i in range(n_layers):
        # blocks[i].output is a nnsight node representing that module's output during this trace
        layer_outs.append(blocks[i].output.save())

    # Cache logits (model output)
    logits = lm.model.output.logits.save()

# Materialize saved tensors from the trace
cached["layer_outs"] = [t.value for t in layer_outs]   # list: [batch, seq, hidden]
cached["logits"] = logits.value                         # [batch, seq, vocab]

print("Cached:")
print(" - layer_outs:", len(cached["layer_outs"]), cached["layer_outs"][0].shape)
print(" - logits:", cached["logits"].shape)

# Example: take last token hidden state from layer 0
last_tok_layer0 = cached["layer_outs"][0][0, -1]  # [hidden]
print("Last token hidden (layer0) shape:", last_tok_layer0.shape)
