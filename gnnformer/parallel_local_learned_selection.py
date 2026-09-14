"""Held learned selection over native local states; no local measurement.

The three modes share parameter shapes and a bounded tanh payload. Selection
uses only that payload, which already depends on the current global query.
Clip/sigmoid pool weighted SUMs; softmax is standard normalized attention over
valid items. No exact semantic neutrality or efficacy is assumed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_local_aggregation import ParallelLocalAggregation, _require

MODES = ('clip', 'sigmoid', 'softmax')


def selection_weights(scores, mode, valid_mask=None):
    """Map FP32 [N,Q] scores to live weights; exclude padding in every mode.

    Empty sets and all-padded query columns produce exact zero weights. The
    softmax bias is intentionally redundant under translation invariance; its
    parameter is retained to keep all three state-dict shapes identical.
    """
    _require(mode in MODES, 'Unknown learned selection mode')
    _require(isinstance(scores, torch.Tensor) and scores.dtype == torch.float32
             and scores.ndim == 2 and scores.shape[1] > 0
             and bool(torch.isfinite(scores).all()), 'scores must be finite FP32 [N,Q]')
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    _require(isinstance(valid_mask, torch.Tensor) and valid_mask.dtype == torch.bool
             and valid_mask.shape == scores.shape and valid_mask.device == scores.device,
             'valid_mask must be boolean [N,Q] on the score device')
    if mode == 'clip':
        weights = scores.clamp(0, 1)
    elif mode == 'sigmoid':
        weights = torch.sigmoid(4 * (scores - .5))
    elif scores.shape[0] == 0:
        weights = scores.clone()
    else:
        logits = (4 * (scores - .5)).masked_fill(~valid_mask, -torch.inf)
        # An entirely padded column must not enter an all-minus-inf softmax.
        logits = torch.where(valid_mask.any(dim=0, keepdim=True), logits,
                             torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=0)
    return weights * valid_mask.to(torch.float32)


class ParallelLocalLearnedSelection(ParallelLocalAggregation):
    """Shared rank-R payload, scalar selection, SUM and native SiLU readout.

    q = Wq RMS(g); p_i = tanh(Wl RMS(h_i) + q + b_local)
    s_i = w dot p_i + b; z = SUM_i gate(s)_i p_i
    delta = U SiLU(Wagg z + q + b_agg)

    Inputs are [N,Q,H]/[Q,H], with no per-image targets or identifiers. All
    arithmetic and parameters remain FP32; only the returned delta is cast.
    Initialized w=0,b=.5 gives clip/sigmoid identical .5 weights and derivatives;
    softmax starts at 1/number_valid. U starts at zero as in the frozen ancestor.
    Validity denotes padding only; its provenance is the caller's responsibility.
    """

    def __init__(self, hidden_size=3584, *, rank=96, mode='clip'):
        _require(mode in MODES, 'Unknown learned selection mode')
        super().__init__(hidden_size, rank=rank, merge='sum', post_activation='silu')
        self.mode = mode
        self.selection_weight = nn.Parameter(torch.zeros(rank, dtype=torch.float32))
        self.selection_bias = nn.Parameter(torch.full((1,), .5, dtype=torch.float32))

    def _parameters_on(self, device):
        _require(self.mode in MODES and self.merge == 'sum' and self.post_activation == 'silu',
                 'Learned selection configuration changed')
        super()._parameters_on(device)
        _require(all(bool(torch.isfinite(p).all()) for p in self.parameters()),
                 'Learned selection parameters must remain finite')

    def _encode(self, local_states, query, valid_mask):
        payload = torch.tanh(self.local(self.rms(local_states)) + query.unsqueeze(0)
                             + self.local_bias)
        scores = F.linear(payload, self.selection_weight.unsqueeze(0)).squeeze(-1) + self.selection_bias
        gates = selection_weights(scores, self.mode, valid_mask)
        messages = gates.unsqueeze(-1) * payload
        self._floating_finite(messages, 'messages')
        return dict(payload=payload, scores=scores, gates=gates, messages=messages)

    def _encode_with_query(self, local_states, query):
        return self._encode(local_states, query, None)['messages']

    def encode(self, local_states, global_states, *, valid_mask=None, capture=False):
        self._global(global_states); self._local(local_states, global_states)
        _require(type(capture) is bool, 'capture must be boolean')
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            values = dict(query=query, **self._encode(local_states, query, valid_mask))
        return (values['messages'], values) if capture else values['messages']

    def forward(self, local_states, global_states, *, valid_mask=None,
                output_dtype=None, capture=False):
        self._global(global_states); self._local(local_states, global_states)
        _require(type(capture) is bool, 'capture must be boolean')
        dtype = self._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            values = dict(query=query, **self._encode(local_states, query, valid_mask))
            aggregate = self.aggregate(values['messages'])
            preactivation = self.aggregate_projection(aggregate) + query
            delta = self.up(F.silu(preactivation))
            self._floating_finite(delta, 'delta')
            result = delta.to(dtype)
        if capture:
            return result, dict(values, aggregate=aggregate, preactivation=preactivation, delta=delta)
        return result
