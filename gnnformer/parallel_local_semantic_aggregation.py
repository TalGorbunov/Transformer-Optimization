"""Conditional software prototype of a gated, bounded native residual branch.

The caller supplies frozen semantic gates; this module does not compute a gate,
load a vocabulary head, choose images, or decode an answer. Gates must already
align with every local item and causal query position. All payload coordinates
are gated before SUM pooling, with no ungated local-state path to the readout.

This prototype preserves the parameter shapes and initialization of
ParallelLocalAggregation. It does not establish an efficacy or reasoning result.
Unlike the ancestor's centered message map, a zero raw local state need not have
a zero payload: callers must give padded items explicit zero gates. An empty or
fully closed set has zero aggregate, but its learned query-conditioned residual
need not be zero after fitting.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .parallel_local_aggregation import ParallelLocalAggregation, _require


class ParallelLocalSemanticAggregation(ParallelLocalAggregation):
    """SUM of externally gated tanh payloads with the existing SiLU readout.

    q = Wq RMS(global)
    payload_i = tanh(Wlocal RMS(local_i) + q + b_local)
    message_i = stop_gradient(gate_i) * payload_i
    aggregate = sum_i message_i
    preactivation = Wagg aggregate + q + b_agg
    delta = U SiLU(preactivation)

    Inputs are [N,Q,H] and [Q,H]; supplied gates are exactly [N,Q], float32,
    finite and in [0,1]. Broadcast an original-query gate in the caller when
    multiple causal positions share it. States retain their gradient paths,
    while gates are always detached internally. Branch arithmetic and captured
    values are float32; only the returned delta is cast to output_dtype (which
    defaults to the global-state dtype). Captures are live and are not retained
    on the module. Finite checks can synchronize an accelerator; this bounded
    prototype makes no throughput claim.
    """

    def __init__(self, hidden_size: int = 3584, *, rank: int = 96,
                 merge: str = "sum", post_activation: str = "silu"):
        _require(merge == "sum", "Semantic aggregation requires merge=sum")
        _require(post_activation == "silu",
                 "Semantic aggregation requires post_activation=silu")
        super().__init__(hidden_size, rank=rank, merge=merge,
                         post_activation=post_activation)

    def _configuration(self):
        _require(self.merge == "sum", "Semantic aggregation requires merge=sum")
        _require(self.post_activation == "silu",
                 "Semantic aggregation requires post_activation=silu")

    def _parameters_on(self, device):
        self._configuration()
        super()._parameters_on(device)
        _require(all(bool(torch.isfinite(p).all()) for p in self.parameters()),
                 "All branch parameters must be finite")

    def _gates(self, gates, local_states):
        _require(isinstance(gates, torch.Tensor) and gates.dtype == torch.float32,
                 "gates must be a float32 tensor")
        _require(gates.shape == local_states.shape[:2],
                 "gates must have exact shape [N,Q] aligned to local_states")
        _require(gates.device == local_states.device,
                 "Gates and states must share a device")
        _require(bool(torch.isfinite(gates).all()), "gates must be finite")
        _require(bool(((gates >= 0) & (gates <= 1)).all()),
                 "gates must lie in [0,1]")
        return gates.detach()

    def _payload_and_messages(self, local_states, query, gates):
        payload = torch.tanh(self.local(self.rms(local_states))
                             + query.unsqueeze(0) + self.local_bias)
        self._floating_finite(payload, "payload")
        messages = gates.unsqueeze(-1) * payload
        self._floating_finite(messages, "messages")
        return payload, messages

    def _encode_with_query(self, local_states, query, *, gates):
        # Require gates even on this private override, so an inherited call
        # cannot accidentally select the ancestor's ungated message map.
        self._parameters_on(local_states.device)
        detached_gates = self._gates(gates, local_states)
        return self._payload_and_messages(local_states, query, detached_gates)[1]

    def encode(self, local_states, global_states, *, gates, capture=False):
        """Return messages, or (messages, live encode capture) when requested."""
        _require(isinstance(capture, bool), "capture must be a boolean")
        self._global(global_states)
        self._local(local_states, global_states)
        detached_gates = self._gates(gates, local_states)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            self._floating_finite(query, "query")
            payload, messages = self._payload_and_messages(
                local_states, query, detached_gates)
        if capture:
            return messages, {"gates": detached_gates, "payload": payload,
                              "messages": messages, "query": query}
        return messages

    def aggregate(self, messages):
        """SUM messages; N=0 gives exactly zero without forcing delta to zero."""
        self._configuration()
        self._floating_finite(messages, "messages")
        self._parameters_on(messages.device)
        aggregate = super().aggregate(messages)
        self._floating_finite(aggregate, "aggregate")
        return aggregate

    def _readout(self, aggregate, query):
        self._configuration()
        preactivation = self.aggregate_projection(aggregate.float()) + query
        self._floating_finite(preactivation, "preactivation")
        delta = self.up(F.silu(preactivation))
        self._floating_finite(delta, "delta")
        return delta, preactivation

    def _decode_with_query(self, aggregate, query):
        return self._readout(aggregate, query)[0]

    def forward(self, local_states, global_states, *, gates,
                output_dtype=None, capture=False):
        """Return delta, or (delta, live FP32 computation capture).

        Capture keys are gates, payload, messages, query, aggregate,
        preactivation and delta. Captured delta is before any native dtype cast.
        No state or capture is saved on this module between calls.
        """
        _require(isinstance(capture, bool), "capture must be a boolean")
        self._global(global_states)
        self._local(local_states, global_states)
        detached_gates = self._gates(gates, local_states)
        dtype = self._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            self._floating_finite(query, "query")
            payload, messages = self._payload_and_messages(
                local_states, query, detached_gates)
            aggregate = self.aggregate(messages)
            delta, preactivation = self._readout(aggregate, query)
            result = delta.to(dtype)
        if capture:
            return result, {"gates": detached_gates, "payload": payload,
                            "messages": messages, "query": query,
                            "aggregate": aggregate, "preactivation": preactivation,
                            "delta": delta}
        return result
