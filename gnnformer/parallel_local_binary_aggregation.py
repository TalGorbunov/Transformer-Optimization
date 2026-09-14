"""Held content-versus-count comparison; no native gate or answer algorithm.

The vector branch reads local states. The matched scalar control can read only
an exact binary count and global states. In particular, it does not sum repeated
floating-point payloads interspersed with zeros: that can leak item positions
through rounding. This class is a research control, not a new pooling operator.
"""
from __future__ import annotations

import torch

from .parallel_local_aggregation import _require
from .parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation


class ParallelLocalBinaryAggregation(ParallelLocalSemanticAggregation):
    """Use the same parameterization for binary-gated vector and scalar arms.

    ``mode`` must be saved explicitly with every checkpoint. It is not a learned
    parameter and introduces no extra buffer. Gates are detached FP32 zeros/ones
    from an external, scene-bound native measurement; labels never enter here.
    All parameters, nonlinearities and residual casting follow the semantic
    parent. Both arms evaluate a dense [N,Q,H] local map for nonempty input.

    Scalar aggregation is count * one query-derived payload. Captured scalar
    messages are descriptive only; their rounded sum is NOT the scalar input.
    Use forward rather than encode/aggregate for the scalar arm.
    """

    def __init__(self, hidden_size=3584, *, rank=96, mode="vector"):
        _require(mode in ("vector", "scalar"), "mode must be vector or scalar")
        super().__init__(hidden_size, rank=rank)
        self.mode = mode

    def _gates(self, gates, local_states):
        value = super()._gates(gates, local_states)
        _require(local_states.shape[0] <= 2**24,
                 "Exact FP32 binary count requires at most 2**24 items")
        _require(bool(((value == 0) | (value == 1)).all()),
                 "Binary aggregation requires exact zero/one measurements")
        return value

    def encode(self, local_states, global_states, *, gates, capture=False):
        _require(self.mode == "vector",
                 "Scalar aggregation requires forward: sum gates before payload multiplication")
        return super().encode(local_states, global_states, gates=gates, capture=capture)

    def forward(self, local_states, global_states, *, gates,
                output_dtype=None, capture=False):
        _require(self.mode in ("vector", "scalar"), "Unknown branch mode")
        _require(isinstance(capture, bool), "capture must be a boolean")
        self._global(global_states)
        self._local(local_states, global_states)
        detached_gates = self._gates(gates, local_states)
        dtype = self._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            self._floating_finite(query, "query")
            count = detached_gates.sum(dim=0)
            if self.mode == "vector":
                payload, messages = self._payload_and_messages(
                    local_states, query, detached_gates)
                aggregate = self.aggregate(messages)
                scalar_payload = None
            else:
                # No numeric use of local states or gate locations. Dense work
                # matches the vector map, but only one fixed row is consumed.
                n = local_states.shape[0]
                repeated = global_states.unsqueeze(0).expand(max(n, 1), -1, -1)
                dense = torch.tanh(self.local(self.rms(repeated))
                                   + query.unsqueeze(0) + self.local_bias)
                self._floating_finite(dense, "scalar payload")
                scalar_payload = dense[0]
                payload = dense[:n]
                messages = detached_gates.unsqueeze(-1) * payload
                aggregate = count.unsqueeze(-1) * scalar_payload
                self._floating_finite(aggregate, "scalar aggregate")
            delta, preactivation = self._readout(aggregate, query)
            result = delta.to(dtype)
        if capture:
            values = dict(gates=detached_gates, payload=payload,
                messages=messages, query=query, aggregate=aggregate,
                preactivation=preactivation, delta=delta, count=count)
            if scalar_payload is not None:
                values["scalar_payload"] = scalar_payload
            return result, values
        return result
