"""LORAMECH shared PEFT helpers for the mechanism probes (L2/L3/L4).

Two traps these helpers close:
1. Probes must run the TRAINED model — `load_peft_frozen` wraps the 4-bit runtime
   model with a saved adapter, eval-only. Attribute access (model.model.language_model,
   get_layers) delegates through PeftModel transparently; LoRA layers are injected
   in-place, so DecoderLayer objects captured before or after the wrap are identical.
2. `dequantize_linear_weight(o_proj)` on a LoRA-wrapped Linear4bit silently returns the
   BASE weight (lora.Linear4bit exposes base_layer.weight) — the adapter delta would be
   missing from any W_O message recompute. `effective_linear_weight` merges it:
   W = dequant(base) + scaling * B @ A.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.runtime import dequantize_linear_weight  # noqa: E402


def load_peft_frozen(model, adapter_dir):
    """Wrap `model` with a saved LoRA adapter, frozen (is_trainable=False)."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    print(f"peft adapter loaded (frozen): {adapter_dir}", flush=True)
    return model


def effective_linear_weight(mod):
    """fp32 effective weight of a (possibly LoRA-wrapped) 4-bit Linear."""
    base = getattr(mod, "base_layer", mod)
    w = dequantize_linear_weight(base)
    lora_a = getattr(mod, "lora_A", None)
    if lora_a is not None and "default" in lora_a:
        a = mod.lora_A["default"].weight.float()
        b = mod.lora_B["default"].weight.float()
        w = w + (b.to(w.device) @ a.to(w.device)) * float(mod.scaling["default"])
    return w
