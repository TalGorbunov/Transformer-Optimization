"""A fixed-scale consistency penalty for paired native aggregation residuals.

This is an auxiliary training loss, not an aggregation operator or a count
predictor. The caller establishes that adjacent examples have the same question,
answer, and causal prefix and differ only in the intended scene transformation.
No pairing metadata or labels are inferred here.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


def paired_residual_consistency(
    delta: Tensor, frozen_global: Tensor, *, eps: float = 1e-6
) -> Tensor:
    """Compare adjacent paired residuals without mixing causal positions.

    Both inputs have shape ``[2 * pairs, 2, hidden]``. The second axis contains
    the count-prediction and EOS-prediction positions, in that order. Rows
    ``(0, 1), (2, 3), ...`` are pairs. ``delta`` must be the FP32 branch output
    *before* its cast to the native residual-stream dtype. ``frozen_global`` is
    the unadapted global state, must require no gradient, and must be exactly
    equal within each pair at each position. States at different positions or
    in different pairs need not be equal.

    The returned FP32 scalar is the mean, over pairs and positions, of
    ``sum_h((delta_even - delta_odd)**2) /
    (sum_h(frozen_global_even.float()**2) + eps)``.
    Hidden dimensions are summed, not averaged. The detached denominator is
    fixed by the frozen input; there is no learned normalization or scale.
    Gradients flow to both residuals, including when they share parameters.

    The caller must validate scene/question/answer/prefix pairing independently.
    Tensor equality of the frozen states does not establish semantic pairing.
    """
    if not isinstance(delta, Tensor) or not isinstance(frozen_global, Tensor):
        raise ValueError("delta and frozen_global must be tensors")
    if delta.ndim != 3 or delta.shape[1] != 2 or delta.shape[2] == 0:
        raise ValueError("delta must have shape [2 * pairs, 2 positions, hidden > 0]")
    if delta.shape[0] == 0 or delta.shape[0] % 2:
        raise ValueError("delta must contain a positive even batch of adjacent pairs")
    if frozen_global.shape != delta.shape or frozen_global.device != delta.device:
        raise ValueError("frozen_global must have the same shape and device as delta")
    if delta.dtype != torch.float32:
        raise ValueError("delta must remain float32 before the native output cast")
    if not frozen_global.is_floating_point():
        raise ValueError("frozen_global must be floating point")
    if frozen_global.requires_grad:
        raise ValueError("frozen_global must be frozen and require no gradient")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive real scalar")
    if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(frozen_global).all()):
        raise ValueError("delta and frozen_global must be finite")
    if not torch.equal(frozen_global[0::2], frozen_global[1::2]):
        raise ValueError("frozen_global must be identical within each adjacent pair")

    # Explicitly preserve branch precision even under an enclosing autocast.
    with torch.autocast(device_type=delta.device.type, enabled=False):
        reference = frozen_global[0::2].detach().float()
        denominator = reference.square().sum(dim=-1) + float(eps)
        if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
            raise ValueError("the fixed float32 denominator must be finite and positive")
        difference = delta[0::2] - delta[1::2]
        loss = (difference.square().sum(dim=-1) / denominator).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("the float32 residual consistency loss must be finite")
    return loss
