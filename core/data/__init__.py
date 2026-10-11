"""Dataset registry. Experiments take --dataset <name> and go through get_dataset()."""
from __future__ import annotations

from typing import Dict

from .base import DatasetSpec, Sample, stratified_order
from .mmred import MMReD
from .videoqa import HERBench, Minerva, VNBench, VNBench4

DATASETS: Dict[str, DatasetSpec] = {MMReD.name: MMReD(), HERBench.name: HERBench(), Minerva.name: Minerva(),
                                    VNBench.name: VNBench(), VNBench4.name: VNBench4()}
DEFAULT_DATASET = MMReD.name


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}") from None


__all__ = ["DATASETS", "DEFAULT_DATASET", "DatasetSpec", "Sample", "get_dataset", "stratified_order"]
