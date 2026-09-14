"""Stable, mergeable attention moments for fixed item-local scores and values.

This is a standard online-softmax accumulator, not a new attention algorithm.
It exposes both the normalized value and log partition to a later reader.
No question, decoder, model, data or task labels belong in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class AttentionMoments:
    maximum: Tensor
    scaled_mass: Tensor
    scaled_value: Tensor

    @property
    def slots(self) -> int:
        return self.maximum.shape[0]


def summarize(scores: Tensor, values: Tensor) -> AttentionMoments:
    """Scores [slots,items], values [items,width], FP32 or FP64.

    Callers must use identical fixed queries and item-local key/value maps in
    every chunk. Any position features must retain global coordinates.
    Empty chunks are identities; an entirely empty collection has no readout.
    """
    if (scores.ndim != 2 or values.ndim != 2 or scores.shape[1] != values.shape[0]
            or scores.device != values.device or scores.dtype != values.dtype
            or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape[0] == 0 or values.shape[1] == 0):
        raise ValueError('Expected matching FP32/FP64 scores [K,N] and values [N,D]')
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError('Scores and values must be finite; masked items must be removed')
    if scores.shape[1] == 0:
        return AttentionMoments(scores.new_full((scores.shape[0],), -torch.inf),
                                scores.new_zeros(scores.shape[0]),
                                values.new_zeros((scores.shape[0], values.shape[1])))
    maximum = scores.max(dim=-1).values
    weights = (scores - maximum[:, None]).exp()
    return AttentionMoments(maximum, weights.sum(dim=-1), weights @ values)


def merge(left: AttentionMoments, right: AttentionMoments) -> AttentionMoments:
    """Merge module-produced states; floating-point reduction order can differ."""
    if (left.maximum.shape != right.maximum.shape
            or left.scaled_value.shape != right.scaled_value.shape
            or left.maximum.dtype != right.maximum.dtype
            or left.maximum.device != right.maximum.device):
        raise ValueError('Incompatible attention moments')
    maximum = torch.maximum(left.maximum, right.maximum)
    # Finite zero is only a computational origin when BOTH chunks are empty.
    origin = torch.where(torch.isneginf(maximum), torch.zeros_like(maximum), maximum)
    lscale = (left.maximum - origin).exp()
    rscale = (right.maximum - origin).exp()
    mass = lscale * left.scaled_mass + rscale * right.scaled_mass
    value = lscale[:, None] * left.scaled_value + rscale[:, None] * right.scaled_value
    return AttentionMoments(maximum, mass, value)


def read_moments(state: AttentionMoments) -> tuple[Tensor, Tensor]:
    """Return normalized values [slots,width] and log mass [slots,1].

    Duplicating identical scored items c times preserves the first output and
    adds log(c) to the second. Weighted patch mass is not an object count;
    finite moments need not distinguish all cross-item relations.
    """
    if bool((state.scaled_mass <= 0).any()):
        raise ValueError('Cannot read an empty collection')
    return (state.scaled_value / state.scaled_mass[:, None],
            (state.maximum + state.scaled_mass.log())[:, None])
