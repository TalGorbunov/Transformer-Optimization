"""Privileged per-image joint codes with a learned SUM/SiLU native residual.

This diagnostic removes feature learning, not the join decision. Each input is
an unscaled person-by-requested-room one-hot code (or zero for another room),
constructed independently for each image. No bag answer enters this module.
The caller supplies frozen native globals and later applies the native head.
"""
from __future__ import annotations

import os
import torch
from torch import nn
from torch.nn import functional as F

from .parallel_local_aggregation import ParallelLocalAggregation, _require

CODE_DIM = 18
READOUT_KEYS = ('query.weight', 'aggregate_projection.weight',
                'aggregate_projection.bias', 'up.weight')


def local_joint_code(person_index, requested_room_index, *, rank=96):
    """Build one raw code from one image; None denotes an unrequested room.

    Person indices are 0..8; requested-room indices are the question's ordered
    0/1 roles. Resolving those roles and semantic provenance belongs to the
    caller. This helper has no bag, target-answer or predicted-label input.
    """
    _require(type(rank) is int and rank >= CODE_DIM, 'rank must be an integer >=18')
    _require(type(person_index) is int and 0 <= person_index < 9, 'Invalid person index')
    _require(requested_room_index is None or
             (type(requested_room_index) is int and requested_room_index in (0, 1)),
             'Requested room index must be 0, 1 or None')
    result = [0] * rank
    if requested_room_index is not None:
        result[2 * person_index + requested_room_index] = 1
    return result


class ParallelLocalJointCodeOracle(nn.Module):
    """z=.5 SUM_valid codes; delta=U SiLU(Wagg z + Wq RMS(g) + b).

    Only four FP32 readout tensors are stored. At H=3584,R=96 they contain
    697,440 trainable parameters; there is no local encoder or selector state.
    Payloads remain raw 0/1 codes, with no tanh or sqrt(rank) scaling.
    """

    rms = staticmethod(ParallelLocalAggregation.rms)

    def __init__(self, hidden_size=3584, *, rank=96):
        super().__init__()
        _require(type(hidden_size) is int and hidden_size > 0, 'hidden_size must be a positive integer')
        _require(type(rank) is int and rank >= CODE_DIM, 'rank must be an integer >=18')
        self.hidden_size, self.rank = hidden_size, rank
        self.query = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.aggregate_projection = nn.Linear(rank, rank, bias=True, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.aggregate_projection.bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, codes, global_states, *, valid_mask=None, output_dtype=None, capture=False):
        ParallelLocalAggregation._floating_finite(global_states, 'global_states')
        _require(global_states.ndim == 2 and global_states.shape[0] > 0
                 and global_states.shape[1] == self.hidden_size, 'global_states must be [Q,H] with Q>0')
        _require(all(p.dtype == torch.float32 and p.device == global_states.device
                     and bool(torch.isfinite(p).all()) for p in self.parameters()),
                 'Readout parameters must be finite FP32 on the input device')
        _require(isinstance(codes, torch.Tensor) and codes.dtype == torch.float32
                 and codes.ndim == 3 and codes.shape[1:] == (global_states.shape[0], self.rank)
                 and codes.device == global_states.device and bool(torch.isfinite(codes).all()),
                 'codes must be finite FP32 [N,Q,R] on the global-state device')
        _require(bool(((codes == 0) | (codes == 1)).all()) and bool((codes[..., CODE_DIM:] == 0).all())
                 and bool((codes[..., :CODE_DIM].sum(-1) <= 1).all()),
                 'Codes must be raw one-hot-or-zero in the first18 coordinates, with zero tail')
        if valid_mask is None:
            valid_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=codes.device)
        _require(isinstance(valid_mask, torch.Tensor) and valid_mask.dtype == torch.bool
                 and valid_mask.shape == codes.shape[:2] and valid_mask.device == codes.device,
                 'valid_mask must be boolean [N,Q] on the code device')
        _require(type(capture) is bool, 'capture must be boolean')
        dtype = ParallelLocalAggregation._output_dtype(global_states, output_dtype)
        with torch.autocast(device_type=global_states.device.type, enabled=False):
            query = self.query(self.rms(global_states))
            scores = torch.full(codes.shape[:2], .5, dtype=torch.float32, device=codes.device)
            gates = scores * valid_mask.float()
            messages = gates.unsqueeze(-1) * codes
            aggregate = messages.sum(0)
            preactivation = self.aggregate_projection(aggregate) + query
            delta = self.up(F.silu(preactivation))
            ParallelLocalAggregation._floating_finite(delta, 'delta')
            result = delta.to(dtype)
        if capture:
            return result, dict(query=query, payload=codes, scores=scores, gates=gates,
                                messages=messages, aggregate=aggregate,
                                preactivation=preactivation, delta=delta)
        return result


def self_test(torch_module):
    """Small CPU Slurm fixtures; no model, native head or saved-data access."""
    _require(torch_module is torch and bool(os.environ.get('SLURM_JOB_ID')),
             'Run joint-code oracle fixtures through the CPU Slurm check')
    checks = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        default = ParallelLocalJointCodeOracle()
        assert tuple(default.state_dict()) == READOUT_KEYS
        assert sum(p.numel() for p in default.parameters()) == 697440
        assert all(p.requires_grad and p.dtype == torch.float32 for p in default.parameters())
        assert not list(default.buffers())
        del default
        from .parallel_local_factor_binding import ParallelLocalFactorBinding
        parent = ParallelLocalFactorBinding(4, rank=18, interaction='product')
        core = ParallelLocalJointCodeOracle(4, rank=18)
        chosen = {k: parent.state_dict()[k].clone() for k in READOUT_KEYS}
        core.load_state_dict(chosen, strict=True)
        assert all(torch.equal(v, chosen[k]) for k, v in core.state_dict().items())
        checks.append('four_tensor_parameter_count_and_selected_parent_state_load')

        codes = torch.tensor([[local_joint_code(0, 0, rank=18)],
                              [local_joint_code(0, 1, rank=18)],
                              [local_joint_code(2, None, rank=18)]], dtype=torch.float32)
        g = torch.tensor([[1., .2, .3, .4]], dtype=torch.float16)
        delta, cap = core(codes, g, output_dtype=torch.float16, capture=True)
        assert torch.equal(delta, torch.zeros_like(delta)) and torch.equal(g + delta, g)
        assert torch.equal(cap['payload'], codes) and torch.equal(cap['aggregate'], .5 * codes.sum(0))
        assert torch.equal(cap['scores'], torch.full((3, 1), .5))
        assert torch.equal(cap['gates'], cap['scores']) and bool((cap['messages'][2] == 0).all())
        assert float(codes[0].square().sum()) == 1
        checks.append('zero_U_identity_and_unscaled_half_SUM')

        valid = torch.tensor([[True], [False], [True]])
        _, padded = core(codes, g, valid_mask=valid, capture=True)
        assert torch.equal(padded['gates'], .5 * valid.float())
        assert bool((padded['messages'][~valid] == 0).all())
        assert torch.equal(padded['aggregate'], .5 * codes[:1].sum(0))
        for values, mask in ((codes[:0], None), (codes, torch.zeros((3, 1), dtype=torch.bool))):
            _, empty = core(values, g, valid_mask=mask, capture=True)
            assert bool((empty['aggregate'] == 0).all()) and bool(torch.isfinite(empty['delta']).all())
        checks.append('padding_empty_and_all_padded_sets')

        cap['delta'].sum().backward()
        assert bool((core.up.weight.grad != 0).any())
        for name, p in core.named_parameters():
            if name != 'up.weight':
                assert p.grad is not None and bool((p.grad == 0).all())
        core.zero_grad(set_to_none=True)
        with torch.no_grad():
            core.query.weight.zero_()
            core.aggregate_projection.weight.copy_(torch.eye(18))
            core.aggregate_projection.bias.zero_()
            core.up.weight.fill_(.1)
        delta, active = core(codes, g, output_dtype=torch.float16, capture=True)
        delta.float().square().sum().backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   and bool((p.grad != 0).any()) for p in core.parameters())
        assert torch.equal(delta, active['delta'].half())
        checks.append('zero_U_gradient_boundary_and_live_readout_gradients')

        # These bags have the same person and requested-room marginals, but
        # different people occupy both rooms. No such answer computation is
        # part of forward: it must be learned by the unchanged readout.
        bag_a = [(0, 0), (0, 1), (1, 0), (1, 0), (2, 1), (2, 1)]
        bag_b = [(0, 0), (0, 0), (1, 0), (1, 1), (2, 1), (2, 1)]
        def table(bag):
            return torch.tensor([local_joint_code(person, room, rank=18)
                                 for person, room in bag], dtype=torch.float32).sum(0).reshape(9, 2)
        a, b = table(bag_a), table(bag_b)
        assert torch.equal(a.sum(0), b.sum(0)) and torch.equal(a.sum(1), b.sum(1))
        assert not torch.equal(a, b)
        assert torch.where((a > 0).all(1))[0].tolist() == [0]
        assert torch.where((b > 0).all(1))[0].tolist() == [1]
        assert local_joint_code(8, 1)[17] == 1 and sum(local_joint_code(8, None)) == 0
        checks.append('independent_local_codes_preserve_join_beyond_equal_marginals')

        bad = codes.clone(); bad[0, 0, 2] = 1
        try:
            core(bad, g)
        except ValueError:
            pass
        else:
            raise AssertionError('A multi-hot code must be rejected')
        try:
            core(codes * 2, g)
        except ValueError:
            pass
        else:
            raise AssertionError('Rescaled codes must be rejected')
        checks.append('multi_hot_and_scaling_rejection')
    return dict(passed=True, tests_passed=True, checks=checks, count=len(checks), device='cpu')
