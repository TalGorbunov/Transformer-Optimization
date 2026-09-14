"""Matched nonlinear product/additive messages with a fixed half-weight SUM.

The factors have no assigned semantic roles. A diagnostic cyclic permutation
breaks their within-item pairing while preserving each query's valid marginals;
it is outside the ordinary forward path and never depends on labels or IDs.
The inherited current-query readout and FP32 arithmetic are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import os

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_local_aggregation import _require
from .parallel_local_learned_selection import (
    ParallelLocalLearnedSelection, selection_weights,
)

INTERACTIONS = ('product', 'additive')


class ParallelLocalFactorBinding(ParallelLocalLearnedSelection):
    """Two rank-R tanh factors, combined before the inherited SUM/readout.

    a = tanh(A RMS(h) + q + bias_a), b = tanh(B RMS(h) + q + bias_b)
    payload = a*b, or .5*(a+b); messages = .5*payload on valid items.

    A caller may condition the already-RMS local input with a prehook on
    ``local``; its input remains H-dimensional and receives no second RMS.
    Default H=3584,R=96 retains 1,385,857 parameters, of which 1,385,760
    are trainable. The inherited 97 selector coordinates stay frozen at 0/.5.
    Empty evidence has zero aggregate, but its readout can have a global offset.
    """

    def __init__(self, hidden_size=3584, *, rank=96, interaction='product'):
        _require(interaction in INTERACTIONS, 'Unknown factor interaction')
        super().__init__(hidden_size, rank=rank, mode='sigmoid')
        # Deliberately use the registered replacement initialization order.
        self.local = nn.Linear(hidden_size, 2 * rank, bias=False, dtype=torch.float32)
        self.local_bias = nn.Parameter(torch.zeros(2 * rank, dtype=torch.float32))
        self.selection_weight.requires_grad_(False)
        self.selection_bias.requires_grad_(False)
        self.interaction = interaction
        self._fixed_interaction = interaction
        self._factor_derangement = False

    def _parameters_on(self, device):
        super()._parameters_on(device)
        _require(self.mode == 'sigmoid' and self.interaction == self._fixed_interaction
                 and self.interaction in INTERACTIONS,
                 'Factor interaction or inherited selection mode changed')
        _require(type(self._factor_derangement) is bool, 'Invalid factor pairing flag')
        _require(self.local.weight.shape == (2 * self.rank, self.hidden_size)
                 and self.local.bias is None and self.local_bias.shape == (2 * self.rank,),
                 'Factor projection shape changed')
        _require(not self.selection_weight.requires_grad and not self.selection_bias.requires_grad
                 and bool((self.selection_weight == 0).all())
                 and bool((self.selection_bias == .5).all()),
                 'Legacy selector must remain frozen at weight 0 and bias .5')

    @property
    def factor_derangement(self):
        return self._factor_derangement

    def set_factor_derangement(self, enabled):
        """Enable only the fixed valid-item B-roll; no parameter/buffer changes."""
        _require(type(enabled) is bool, 'Factor derangement flag must be boolean')
        self._factor_derangement = enabled
        return self

    @contextmanager
    def factor_pairing(self, pairing):
        """Temporarily choose 'paired' or 'cyclic', restoring even on failure."""
        _require(pairing in ('paired', 'cyclic'), 'Unknown factor pairing')
        previous = self._factor_derangement
        self.set_factor_derangement(pairing == 'cyclic')
        try:
            yield self
        finally:
            self.set_factor_derangement(previous)

    def _encode(self, local_states, query, valid_mask):
        n_items, n_queries = local_states.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones((n_items, n_queries), dtype=torch.bool,
                                    device=local_states.device)
        _require(isinstance(valid_mask, torch.Tensor) and valid_mask.dtype == torch.bool
                 and valid_mask.shape == (n_items, n_queries)
                 and valid_mask.device == local_states.device,
                 'valid_mask must be boolean [N,Q] on the local-state device')
        projected = self.local(self.rms(local_states))
        first, second = projected.split(self.rank, dim=-1)
        bias_a, bias_b = self.local_bias.split(self.rank)
        pre_a = first + query.unsqueeze(0) + bias_a
        pre_b = second + query.unsqueeze(0) + bias_b
        self._floating_finite(pre_a, 'factor_preactivation_a')
        self._floating_finite(pre_b, 'factor_preactivation_b')
        factor_a, factor_b = torch.tanh(pre_a), torch.tanh(pre_b)
        indices = torch.arange(n_items, device=local_states.device).unsqueeze(1).expand(
            n_items, n_queries).clone()
        if self._factor_derangement:
            for q in range(n_queries):
                rows = torch.nonzero(valid_mask[:, q], as_tuple=False).flatten()
                indices[rows, q] = rows.roll(1)
        used_b = (factor_b.gather(0, indices.unsqueeze(-1).expand(-1, -1, self.rank))
                  if self._factor_derangement else factor_b)
        payload = (factor_a * used_b if self.interaction == 'product'
                   else .5 * (factor_a + used_b))
        scores = F.linear(payload, self.selection_weight.unsqueeze(0)).squeeze(-1) + self.selection_bias
        gates = selection_weights(scores, self.mode, valid_mask)
        messages = gates.unsqueeze(-1) * payload
        self._floating_finite(messages, 'messages')
        return dict(payload=payload, scores=scores, gates=gates, messages=messages,
                    factor_a=factor_a, factor_b=factor_b, factor_b_used=used_b,
                    factor_preactivation_a=pre_a, factor_preactivation_b=pre_b,
                    factor_permutation_indices=indices)


def self_test(torch_module):
    """Small CPU-only algebra/gradient fixtures, invoked by the Slurm check."""
    _require(torch_module is torch and bool(os.environ.get('SLURM_JOB_ID')),
             'Run factor-binding self-tests through the CPU Slurm check')
    checks = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        product = ParallelLocalFactorBinding(4, rank=2, interaction='product')
        torch.manual_seed(24)
        additive = ParallelLocalFactorBinding(4, rank=2, interaction='additive')
        assert all(torch.equal(v, additive.state_dict()[k]) for k, v in product.state_dict().items())
        assert product.local.weight.shape == (4, 4) and product.local_bias.shape == (4,)
        assert sum(p.numel() for p in product.parameters()) == 45
        assert sum(p.numel() for p in product.parameters() if p.requires_grad) == 42
        assert 4 * 3584 * 96 + 96 * 96 + 4 * 96 + 1 == 1385857
        assert not list(product.buffers()) and product.mode == 'sigmoid'
        checks.append('matched_initialization_shapes_and_trainable_mask')

        # Deterministic, nonconstant correlated factors make permutation decisive.
        with torch.no_grad():
            product.query.weight.zero_()
            product.local.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                                     [1., 0., 0., 0.], [0., 1., 0., 0.]]))
            product.aggregate_projection.weight.copy_(torch.eye(2))
        additive.load_state_dict(product.state_dict())
        h = torch.tensor([[[1., 0., 1., 0.]], [[0., 1., 1., 0.]], [[-1., -1., 1., 0.]]])
        g = torch.tensor([[.2, .3, .4, .5]])
        zero, paired = product(h, g, capture=True)
        assert torch.equal(zero, torch.zeros_like(zero))
        assert torch.equal(paired['payload'], paired['factor_a'] * paired['factor_b'])
        assert torch.equal(paired['messages'], .5 * paired['payload'])
        assert torch.equal(paired['factor_permutation_indices'][:, 0], torch.arange(3))
        assert bool((paired['payload'].abs() <= 1).all())
        with product.factor_pairing('cyclic'):
            _, cyclic = product(h, g, capture=True)
        assert torch.equal(cyclic['factor_permutation_indices'][:, 0], torch.tensor([2, 0, 1]))
        assert torch.equal(cyclic['factor_b_used'], paired['factor_b'].roll(1, dims=0))
        assert torch.equal(cyclic['payload'], paired['factor_a'] * cyclic['factor_b_used'])
        assert not torch.allclose(paired['aggregate'], cyclic['aggregate'])
        checks.append('product_identity_and_fixed_pairing_sensitivity')

        _, add_paired = additive(h, g, capture=True)
        with additive.factor_pairing('cyclic'):
            _, add_cyclic = additive(h, g, capture=True)
        assert torch.equal(add_paired['payload'], .5 * (add_paired['factor_a'] + add_paired['factor_b']))
        assert torch.allclose(add_paired['aggregate'], add_cyclic['aggregate'], atol=1e-6, rtol=1e-6)
        ref = .25 * (add_paired['factor_a'].double() + add_paired['factor_b'].double()).sum(0)
        perm = .25 * (add_cyclic['factor_a'].double() + add_cyclic['factor_b_used'].double()).sum(0)
        assert torch.allclose(ref, perm, atol=1e-10, rtol=0)
        checks.append('additive_pooled_permutation_invariance')

        ragged_h = h.expand(-1, 3, -1).clone()
        ragged_g = g.expand(3, -1).clone()
        valid = torch.tensor([[True, False, False], [False, True, False], [True, False, False]])
        with product.factor_pairing('cyclic'):
            _, cap = product(ragged_h, ragged_g, valid_mask=valid, capture=True)
        expected = torch.tensor([[2, 0, 0], [1, 1, 1], [0, 2, 2]])
        assert torch.equal(cap['factor_permutation_indices'], expected)
        assert torch.equal(cap['gates'], .5 * valid.float())
        assert torch.equal(cap['factor_b_used'][~valid], cap['factor_b'][~valid])
        assert torch.equal(cap['messages'][~valid], torch.zeros_like(cap['messages'][~valid]))
        for core in (product, additive):
            for empty_h, mask in ((h[:0], None), (h, torch.zeros((3, 1), dtype=torch.bool))):
                with core.factor_pairing('cyclic'):
                    _, empty = core(empty_h, g, valid_mask=mask, capture=True)
                assert torch.equal(empty['aggregate'], torch.zeros_like(empty['aggregate']))
                assert bool(torch.isfinite(empty['delta']).all())
        checks.append('ragged_padding_singleton_empty_and_all_padded')

        # Zero U blocks upstream gradients, while U itself has a live signal.
        for core, cap in ((product, paired), (additive, add_paired)):
            cap['delta'].sum().backward()
            assert core.up.weight.grad is not None and bool((core.up.weight.grad != 0).any())
            for name, parameter in core.named_parameters():
                if name.startswith('selection_'):
                    assert not parameter.requires_grad and parameter.grad is None
                elif name != 'up.weight':
                    assert parameter.grad is not None and torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        product.zero_grad(set_to_none=True)
        with torch.no_grad():
            product.up.weight.fill_(.1)
        live_h = h.clone().requires_grad_(True)
        with product.factor_pairing('cyclic'):
            delta, cap = product(live_h, g, output_dtype=torch.float16, capture=True)
        cap['factor_a'].retain_grad(); cap['factor_b'].retain_grad()
        delta.float().square().sum().backward()
        assert torch.equal(delta, cap['delta'].half())
        assert all(bool((v.grad != 0).any()) for v in (cap['factor_a'], cap['factor_b'], live_h))
        assert all(bool((part != 0).any()) for part in product.local.weight.grad.split(2))
        assert product.selection_weight.grad is None and product.selection_bias.grad is None
        checks.append('zero_U_gradient_boundary_live_factors_and_output_cast')

        before = {k: v.clone() for k, v in product.state_dict().items()}
        try:
            with product.factor_pairing('cyclic'):
                assert product._factor_derangement
                with product.factor_pairing('paired'):
                    assert not product._factor_derangement
                raise RuntimeError('fixture')
        except RuntimeError as error:
            assert str(error) == 'fixture'
        assert not product._factor_derangement
        assert all(torch.equal(v, before[k]) for k, v in product.state_dict().items())
        checks.append('scoped_pairing_restores_without_state_mutation')

        def rejects(call):
            try:
                call()
            except ValueError:
                return
            raise AssertionError('Expected contract rejection')

        rejects(lambda: product.set_factor_derangement(1))
        rejects(lambda: product(h, g, valid_mask=torch.ones((3, 1))))
        rejects(lambda: product(h * float('nan'), g))
        product.mode = 'clip'
        rejects(lambda: product(h, g))
        product.mode = 'sigmoid'
        product.selection_weight.requires_grad_(True)
        rejects(lambda: product(h, g))
        product.selection_weight.requires_grad_(False)
        checks.append('finite_input_mask_mode_and_selector_guards')
    return dict(passed=True, tests_passed=True, checks=checks, count=len(checks), device='cpu')
