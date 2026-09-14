"""Standalone query-conditioned set aggregation over frozen local states.

This is a software prototype of a conditional DeepSets map and learned readout,
not a pretrained-model integration or an efficacy result. The caller supplies
aligned local states [N, Q, H] and global states [Q, H]. Each query is independent;
image selection, causality, local-stream packing and native residual injection
are the caller's responsibility.

All branch parameters and arithmetic remain float32. Output casting is explicit
and does not increase arithmetic precision. Zero raw local states give exactly
zero messages, but real negative images need not do so. Empty evidence produces
a zero aggregate; the learned post-merge readout may still produce a global
offset once its initially zero output projection is trained.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ParallelLocalAggregation(nn.Module):
    """Shared conditional map, SUM/MEAN pooling, and a learned native residual.

    q = Wq RMS(global)
    m_i = SiLU(Wlocal RMS(local_i) + q + b_local) - SiLU(q + b_local)
    z = sum_i m_i                         # or mean, zero when N=0
    t = Wagg z + q + b_rho
    delta = U activation(t)              # SiLU or identity

    RMS is tokenwise, has no learned affine term, and uses epsilon 1e-6.
    Wq, Wlocal and U are bias-free. The two readout options have exactly the
    same matrices and biases. SUM/MEAN scaling applies to z, not generally
    to delta after the nonlinear/global-conditioned readout.

    There is no learned state per image, no LoRA, and no model/cache hook.
    Inputs are not detached: callers may freeze local/global states themselves
    or differentiate through them. Parameters must remain float32; use
    .to(device=...) rather than casting the module to a reduced precision.
    """

    def __init__(self, hidden_size: int = 3584, *, rank: int = 96,
                 merge: str = "sum", post_activation: str = "silu"):
        super().__init__()
        _require(isinstance(hidden_size, int) and not isinstance(hidden_size, bool)
                 and hidden_size > 0, "hidden_size must be a positive integer")
        _require(isinstance(rank, int) and not isinstance(rank, bool) and rank > 0,
                 "rank must be a positive integer")
        _require(merge in ("sum", "mean"), "merge must be sum or mean")
        _require(post_activation in ("silu", "identity"),
                 "post_activation must be silu or identity")
        self.hidden_size = hidden_size
        self.rank = rank
        self.merge = merge
        self.post_activation = post_activation
        self.query = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.local = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.local_bias = nn.Parameter(torch.zeros(rank, dtype=torch.float32))
        self.aggregate_projection = nn.Linear(rank, rank, bias=True, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.aggregate_projection.bias)
        nn.init.zeros_(self.up.weight)

    @staticmethod
    def rms(states):
        """Fixed tokenwise RMS normalization, evaluated in float32."""
        states = states.float()
        return states * torch.rsqrt(states.square().mean(dim=-1, keepdim=True) + 1e-6)

    def _parameters_on(self, device):
        _require(all(p.dtype == torch.float32 for p in self.parameters()),
                 "All branch parameters must remain float32")
        _require(all(p.device == device for p in self.parameters()),
                 "Branch parameters and inputs must share a device")

    @staticmethod
    def _floating_finite(value, name):
        _require(isinstance(value, torch.Tensor) and value.is_floating_point(),
                 f"{name} must be a floating-point tensor")
        _require(bool(torch.isfinite(value).all()), f"{name} must be finite")

    def _global(self, global_states):
        self._floating_finite(global_states, "global_states")
        _require(global_states.ndim == 2 and global_states.shape[0] > 0
                 and global_states.shape[1] == self.hidden_size,
                 "global_states must have shape [Q,H] with Q>0 and the configured H")
        self._parameters_on(global_states.device)

    def _local(self, local_states, global_states):
        self._floating_finite(local_states, "local_states")
        _require(local_states.ndim == 3
                 and local_states.shape[1:] == global_states.shape,
                 "local_states must have shape [N,Q,H] aligned to global_states")
        _require(local_states.device == global_states.device,
                 "Local and global states must share a device")

    def _encode_with_query(self, local_states, query):
        # Match both SiLU operand shapes/strides. Separate [Q,R] versus
        # [N,Q,R] kernels can round differently even when raw local states are
        # zero, violating the exact numerical subtraction contract.
        baseline = (query + self.local_bias).unsqueeze(0).expand(
            local_states.shape[0], -1, -1).contiguous()
        return F.silu(self.local(self.rms(local_states)) + baseline) - F.silu(baseline)

    def encode(self, local_states, global_states):
        """Return float32 messages [N,Q,rank], including N=0.

        Finite inputs are checked explicitly in this bounded prototype; those
        checks can synchronize an accelerator. They are not a throughput claim.
        """
        self._global(global_states)
        self._local(local_states, global_states)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            return self._encode_with_query(local_states, query)

    def aggregate(self, messages):
        """Pool float32 messages to [Q,rank]; an empty set maps to zero."""
        self._floating_finite(messages, "messages")
        _require(messages.ndim == 3 and messages.shape[1] > 0
                 and messages.shape[2] == self.rank,
                 "messages must have shape [N,Q,rank] with Q>0")
        _require(self.merge in ("sum", "mean"), "merge must be sum or mean")
        # Sum handles N=0 without a NaN, while preserving an autograd path.
        total = messages.float().sum(dim=0)
        return total if self.merge == "sum" else total / max(messages.shape[0], 1)

    def _decode_with_query(self, aggregate, query):
        hidden = self.aggregate_projection(aggregate.float()) + query
        _require(self.post_activation in ("silu", "identity"),
                 "post_activation must be silu or identity")
        if self.post_activation == "silu":
            hidden = F.silu(hidden)
        return self.up(hidden)

    @staticmethod
    def _output_dtype(global_states, output_dtype):
        chosen = global_states.dtype if output_dtype is None else output_dtype
        _require(isinstance(chosen, torch.dtype) and chosen.is_floating_point,
                 "output_dtype must be a floating-point torch dtype")
        return chosen

    def decode(self, aggregate, global_states, *, output_dtype=None):
        """Decode [Q,rank] to [Q,H]; default output dtype follows global_states."""
        self._global(global_states)
        self._floating_finite(aggregate, "aggregate")
        _require(aggregate.shape == (global_states.shape[0], self.rank),
                 "aggregate must have shape [Q,rank] aligned to global_states")
        _require(aggregate.device == global_states.device,
                 "Aggregate and global states must share a device")
        dtype = self._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            return self._decode_with_query(aggregate, query).to(dtype)

    def forward(self, local_states, global_states, *, output_dtype=None):
        """Produce the residual, computing the shared global query only once."""
        self._global(global_states)
        self._local(local_states, global_states)
        dtype = self._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            messages = self._encode_with_query(local_states, query)
            aggregate = self.aggregate(messages)
            return self._decode_with_query(aggregate, query).to(dtype)

