"""Backbone registry. Experiments take --backbone <name> and go through get_backbone()."""
from __future__ import annotations

from typing import Dict

from .base import BackboneSpec, Delimiters, SpecialIds
from .gemma3 import GEMMA_3_12B
from .internvl import INTERNVL3_5_8B
from .qwen2_5_vl import QWEN2_5_VL_7B

BACKBONES: Dict[str, BackboneSpec] = {b.name: b for b in (QWEN2_5_VL_7B, INTERNVL3_5_8B, GEMMA_3_12B)}
DEFAULT_BACKBONE = QWEN2_5_VL_7B.name


def get_backbone(name: str) -> BackboneSpec:
    try:
        return BACKBONES[name]
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(BACKBONES)}") from None


__all__ = ["BACKBONES", "DEFAULT_BACKBONE", "BackboneSpec", "Delimiters", "SpecialIds", "get_backbone"]
