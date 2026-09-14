"""Software-only conditional null-mean predictor and projected readout algebra.

This module neither fits a mean nor chooses null examples. The predictor sees
only a continuous current query, with no cardinality, question ID or token IDs.
Keep it separate from the existing ParallelLocalAggregation: native contracts
that count the original core's parameters must continue to receive that core.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _fp32_matrix(value, shape, device, name):
    _need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
          and value.shape == shape and value.device == device
          and bool(torch.isfinite(value).all()),
          name + ' must be a finite FP32 matrix with the declared shape/device')


class ConditionalNullMean(nn.Module):
    """Rank -> rank SiLU -> rank, with a signed, initially zero output.

    Default rank96 has18,624 parameters. All parameters/arithmetic stay FP32;
    move with .to(device=...), not by casting to a reduced precision. The first
    affine map has standard PyTorch initialization. The second weight AND bias
    start at zero. Initial gradients can therefore reach the second map before
    the first map, as expected for this initialization.

    The input [Q,rank] is q=Wq RMS(g), formed by the caller. The module stores
    neither the original aggregation core nor a table of queries/prefixes.
    """

    def __init__(self, rank: int = 96):
        super().__init__()
        _need(type(rank) is int and rank > 0, 'rank must be a positive integer')
        self.rank = rank
        self.fc1 = nn.Linear(rank, rank, bias=True, dtype=torch.float32)
        self.fc2 = nn.Linear(rank, rank, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, query):
        _need(isinstance(query, torch.Tensor) and query.ndim == 2
              and query.shape[0] > 0 and query.shape[1] == self.rank,
              'query must have shape [Q,rank], Q>0')
        _fp32_matrix(query, query.shape, self.fc1.weight.device, 'query')
        _need(all(p.dtype == torch.float32 and p.device == query.device
                  for p in self.parameters()), 'Predictor parameters must be FP32 on the query device')
        with torch.autocast(device_type=query.device.type, enabled=False):
            result = self.fc2(F.silu(self.fc1(query)))
        _need(bool(torch.isfinite(result).all()), 'Conditional null prediction is nonfinite')
        return result


def projected_null_readout(core, aggregate, global_states, null_mean, *,
                           n_elements, mode, anchor_n=16, output_dtype=None):
    """Apply explicit null correction before the existing nonlinear readout.

    Inputs: aggregate/null_mean [Q,rank] FP32, native global_states [Q,H].
    null_mean may come from the predictor or a separately supplied reference
    mean; this function never constructs/calibrates it and never detaches it.

    q = Wq RMS(g)
    r = Wagg(aggregate) + q          # retain the original affine bias/order
    b0 = linear(null_mean, Wagg.weight)  # deliberately NO Wagg bias
    coefficient = 0 / (N-anchor) / N / 1 for base/anchored/centered/offset
    delta = U SiLU(r - coefficient*b0)

    Return live tensors for inspection/gradients. `delta_float32` is the FP32
    branch result; `delta` has the requested dtype (default native g.dtype).
    The native caller must add g + delta_float32.to(g.dtype), preserving the
    cast-before-add contract. No native model, parameters or cache is mutated.

    `corrected_aggregate` exposes the centered-statistic algebra ONLY. The
    residual is computed through the projected subtraction above, not by
    reordering the original affine arithmetic through this diagnostic tensor.
    The nonlinear residual is not claimed to be additive over local sets.
    """
    _need(isinstance(core, ParallelLocalAggregation)
          and core.merge == 'sum' and core.post_activation == 'silu',
          'Require the original SUM/SiLU aggregation core')
    _need(type(n_elements) is int and n_elements >= 0,
          'n_elements must be a nonnegative integer')
    _need(type(anchor_n) is int and anchor_n >= 0,
          'anchor_n must be a nonnegative integer')
    _need(mode in ('base', 'anchored', 'centered', 'offset'), 'Unsupported null correction mode')
    core._global(global_states)
    shape = (global_states.shape[0], core.rank)
    _fp32_matrix(aggregate, shape, global_states.device, 'aggregate')
    _fp32_matrix(null_mean, shape, global_states.device, 'null_mean')
    dtype = global_states.dtype if output_dtype is None else output_dtype
    _need(isinstance(dtype, torch.dtype) and dtype.is_floating_point,
          'output_dtype must be floating point')
    coefficient = {'base': 0, 'anchored': n_elements-anchor_n,
                   'centered': n_elements, 'offset': 1}[mode]
    with torch.autocast(device_type=global_states.device.type, enabled=False):
        query = core.query(core.rms(global_states))
        preactivation = core.aggregate_projection(aggregate) + query
        projected_null = F.linear(null_mean, core.aggregate_projection.weight)
        # Preserve exact native/core identity whenever the coefficient is zero.
        corrected = preactivation if coefficient == 0 else preactivation-coefficient*projected_null
        corrected_aggregate = aggregate if coefficient == 0 else aggregate-coefficient*null_mean
        delta = core.up(F.silu(corrected))
    _need(bool(torch.isfinite(delta).all()) and bool(torch.isfinite(projected_null).all())
          and bool(torch.isfinite(corrected_aggregate).all()), 'Null readout produced nonfinite values')
    return dict(delta=delta.to(dtype), delta_float32=delta, query=query,
                preactivation=preactivation, projected_null=projected_null,
                corrected_preactivation=corrected, corrected_aggregate=corrected_aggregate,
                coefficient=coefficient, mode=mode, n_elements=n_elements, anchor_n=anchor_n)
