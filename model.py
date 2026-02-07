import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-3-4b-it" 
device = "cuda"
dtype = torch.bfloat16

model = HookedTransformer.from_pretrained(
    "google/gemma-3-4b-it",
    device=device,
    dtype=dtype,
)

# Quick sanity run (logits)
prompt = "Hello from TransformerLens!"
logits = model(prompt)
print(logits.shape)

# If you want cached activations:
logits, cache = model.run_with_cache(prompt)
print(cache.keys())  # lots of hook names
