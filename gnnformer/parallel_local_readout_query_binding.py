"""Product factors with explicit global query only in the post-SUM readout.

Local hidden states already contain their native question/prefix context. This
subclass removes only the additional explicit q inside both tanh factors; it
adds no parameters, normalization or learned state. Zero-U outputs match the
parent at initialization, although internal factors and learning paths differ.
"""
from __future__ import annotations

import os
import torch

from .parallel_local_aggregation import _require
from .parallel_local_factor_binding import ParallelLocalFactorBinding


class ParallelLocalReadoutQueryBinding(ParallelLocalFactorBinding):
    """Keep the parent's exact initialization, product, pairing and readout."""

    def __init__(self, hidden_size=3584, *, rank=96, interaction='product'):
        _require(interaction == 'product', 'Readout-only query binding requires product')
        super().__init__(hidden_size, rank=rank, interaction=interaction)

    def _encode(self, local_states, query, valid_mask):
        # The inherited outer forward retains its original live query tensor
        # in capture['query'] and aggregate_projection(aggregate) + query.
        return super()._encode(local_states, torch.zeros_like(query), valid_mask)


def self_test(torch_module):
    """Focused CPU Slurm fixtures; no model/head calls or fitted state reads."""
    _require(torch_module is torch and bool(os.environ.get('SLURM_JOB_ID')),
             'Run readout-query self-tests through the CPU Slurm check')
    checks = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        parent = ParallelLocalFactorBinding(4, rank=2, interaction='product')
        parent_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(24)
        core = ParallelLocalReadoutQueryBinding(4, rank=2)
        assert torch.equal(torch.random.get_rng_state(), parent_rng)
        assert all(torch.equal(v, parent.state_dict()[k]) for k, v in core.state_dict().items())
        assert [(n, p.requires_grad) for n, p in core.named_parameters()] == [
            (n, p.requires_grad) for n, p in parent.named_parameters()]
        assert sum(p.numel() for p in core.parameters()) == 45
        assert sum(p.numel() for p in core.parameters() if p.requires_grad) == 42
        assert not list(core.buffers()) and core.factor_derangement is False
        checks.append('exact_parent_initialization_rng_parameter_mask')

        h = torch.tensor([[[1., 0., 1., 0.]], [[0., 1., 1., 0.]], [[-1., -1., 1., 0.]]])
        g = torch.tensor([[1., 0., 1., 0.]])
        changed_g = torch.tensor([[0., 1., 1., 0.]])
        with torch.no_grad():
            core.query.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
            core.local.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                                  [1., 0., 0., 0.], [0., 1., 0., 0.]]))
            core.aggregate_projection.weight.copy_(torch.eye(2))
        parent.load_state_dict(core.state_dict())
        delta, cap = core(h, g, output_dtype=torch.float16, capture=True)
        parent_delta, parent_cap = parent(h, g, output_dtype=torch.float16, capture=True)
        assert torch.equal(delta, parent_delta) and torch.equal(delta, torch.zeros_like(delta))
        assert set(cap) == set(parent_cap)
        reference = parent._encode(h, torch.zeros_like(cap['query']), None)
        assert all(torch.equal(cap[k], v) for k, v in reference.items())
        assert torch.equal(cap['query'], core.query(core.rms(g)))
        assert torch.equal(cap['preactivation'], core.aggregate_projection(cap['aggregate']) + cap['query'])
        assert not torch.equal(cap['factor_a'], parent_cap['factor_a'])
        checks.append('shared_zero_U_output_and_exact_zero_query_parent_encoder')

        # Both zero outputs retain a live U gradient; its values need not match.
        cap['delta'].sum().backward(); parent_cap['delta'].sum().backward()
        assert bool((core.up.weight.grad != 0).any())
        assert not torch.equal(core.up.weight.grad, parent.up.weight.grad)
        for name, parameter in core.named_parameters():
            if name.startswith('selection_'):
                assert not parameter.requires_grad and parameter.grad is None
            elif name != 'up.weight':
                assert parameter.grad is not None and bool((parameter.grad == 0).all())
        core.zero_grad(set_to_none=True)
        with torch.no_grad():
            core.up.weight.copy_(torch.tensor([[.1, 0.], [0., .1], [.2, .1], [-.1, .2]]))
        delta, cap = core(h, g, output_dtype=torch.float16, capture=True)
        changed_delta, changed = core(h, changed_g, output_dtype=torch.float16, capture=True)
        encoder_keys = tuple(reference) + ('aggregate',)
        assert all(torch.equal(cap[k], changed[k]) for k in encoder_keys)
        assert not torch.equal(cap['query'], changed['query'])
        assert not torch.equal(cap['preactivation'], changed['preactivation'])
        assert not torch.equal(delta, changed_delta)
        assert torch.equal(delta, cap['delta'].half())
        checks.append('fixed_local_factors_and_live_post_SUM_query')

        delta.float().square().sum().backward()
        assert bool((core.query.weight.grad != 0).any())
        assert all(bool((part != 0).any()) for part in core.local.weight.grad.split(2))
        assert core.selection_weight.grad is None and core.selection_bias.grad is None
        checks.append('active_U_live_query_and_both_factor_gradients')

        valid = torch.tensor([[True, False], [False, True], [True, False]])
        hh = h.expand(-1, 2, -1).clone(); gg = g.expand(2, -1).clone()
        _, paired = core(hh, gg, valid_mask=valid, capture=True)
        with core.factor_pairing('cyclic'):
            _, cyclic = core(hh, gg, valid_mask=valid, capture=True)
        assert torch.equal(cyclic['factor_permutation_indices'], torch.tensor([[2, 0], [1, 1], [0, 2]]))
        assert torch.equal(paired['factor_a'], cyclic['factor_a'])
        assert torch.equal(paired['factor_b'], cyclic['factor_b'])
        assert torch.equal(cyclic['factor_b_used'][~valid], cyclic['factor_b'][~valid])
        assert torch.equal(cyclic['gates'], .5 * valid.float())
        assert bool((cyclic['messages'][~valid] == 0).all())
        for local, mask in ((h[:0], None), (h, torch.zeros((3, 1), dtype=torch.bool))):
            with core.factor_pairing('cyclic'):
                _, empty = core(local, g, valid_mask=mask, capture=True)
            assert bool((empty['aggregate'] == 0).all()) and bool(torch.isfinite(empty['delta']).all())
        before = {k: v.clone() for k, v in core.state_dict().items()}
        try:
            with core.factor_pairing('cyclic'):
                raise RuntimeError('fixture')
        except RuntimeError as error:
            assert str(error) == 'fixture'
        assert core.factor_derangement is False
        assert all(torch.equal(v, before[k]) for k, v in core.state_dict().items())
        checks.append('inherited_valid_pairing_empty_padding_and_exception_cleanup')

        try:
            ParallelLocalReadoutQueryBinding(4, rank=2, interaction='additive')
        except ValueError:
            pass
        else:
            raise AssertionError('Additive interaction must be rejected')
        checks.append('product_only_constructor')
    return dict(passed=True,tests_passed=True,checks=checks,count=len(checks),device='cpu')
