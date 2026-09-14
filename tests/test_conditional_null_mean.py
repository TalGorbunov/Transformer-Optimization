"""Focused numerical tests; run only in a CPU Slurm allocation."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import unittest
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu' or os.environ.get('SLURM_JOB_GPUS'):
    raise SystemExit('Run conditional-null numerical tests in CPU Slurm')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.conditional_null_mean import ConditionalNullMean, projected_null_readout


class ConditionalNullTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261102)
        self.core = ParallelLocalAggregation(8, rank=4)
        self.predictor = ConditionalNullMean(rank=4)
        with torch.no_grad():
            self.core.up.weight.normal_(0, .15)
            self.core.aggregate_projection.bias.copy_(torch.tensor([.3, -.2, .5, -.4]))
        self.g = torch.randn(2, 8)
        self.local = torch.randn(5, 2, 8)
        self.z = self.core.aggregate(self.core.encode(self.local, self.g))

    def readout(self, mean, **kwargs):
        options = dict(n_elements=5, mode='centered')
        options.update(kwargs)
        return projected_null_readout(self.core, self.z, self.g, mean, **options)

    def activate_predictor(self):
        with torch.no_grad():
            self.predictor.fc2.weight.normal_(0, .1)
            self.predictor.fc2.bias.normal_(0, .1)

    def test_exact_parameter_count_zero_output_and_separate_core(self):
        predictor = ConditionalNullMean()
        self.assertEqual(sum(p.numel() for p in predictor.parameters()), 18624)
        self.assertEqual(set(predictor.state_dict()), {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'})
        self.assertTrue(torch.equal(predictor(torch.randn(3, 96)), torch.zeros(3, 96)))
        saved = {k:v.clone() for k,v in self.core.state_dict().items()}
        self.predictor.eval();self.predictor.requires_grad_(False)
        self.assertTrue(self.core.training)
        self.assertTrue(all(p.requires_grad for p in self.core.parameters()))
        self.assertFalse(any(m is self.predictor for m in self.core.modules()))
        self.assertFalse(any(m is self.core for m in self.predictor.modules()))
        for k,v in self.core.state_dict().items():self.assertTrue(torch.equal(v, saved[k]))

    def test_signed_output_and_input_contract(self):
        with torch.no_grad():self.predictor.fc2.bias.copy_(torch.tensor([-2., -1., 1., 2.]))
        result = self.predictor(torch.zeros(2, 4))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool((result[:, :2] < 0).all()) and bool((result[:, 2:] > 0).all()))
        bad = (torch.zeros(4), torch.zeros(2, 1, 4), torch.zeros(0, 4), torch.zeros(2, 3),
               torch.zeros(2, 4, dtype=torch.float16), torch.full((2, 4), float('nan')))
        for q in bad:
            with self.assertRaises(ValueError):self.predictor(q)
        with self.assertRaises(ValueError):ConditionalNullMean(rank=True)
        self.predictor.to(dtype=torch.bfloat16)
        with self.assertRaises(ValueError):self.predictor(torch.zeros(2, 4))

    def test_zero_predictor_preserves_original_decode_in_every_mode(self):
        mean = self.predictor(self.core.query(self.core.rms(self.g)))
        expected = self.core.decode(self.z, self.g, output_dtype=torch.float32)
        for mode in ('base', 'anchored', 'centered', 'offset'):
            value = self.readout(mean, mode=mode)
            self.assertTrue(torch.equal(value['delta'], expected))
            self.assertTrue(torch.equal(value['projected_null'], torch.zeros_like(mean)))

    def test_projected_formula_keeps_bias_and_supports_live_mean(self):
        mean = torch.randn(2, 4, requires_grad=True)
        value = self.readout(mean, n_elements=19, mode='anchored')
        q = self.core.query(self.core.rms(self.g))
        r = self.core.aggregate_projection(self.z)+q
        b = F.linear(mean, self.core.aggregate_projection.weight)
        self.assertEqual(value['coefficient'], 3)
        self.assertTrue(torch.equal(value['preactivation'], r))
        self.assertTrue(torch.equal(value['projected_null'], b))
        self.assertFalse(torch.equal(b, self.core.aggregate_projection(mean)))
        self.assertTrue(torch.equal(value['corrected_preactivation'], r-3*b))
        self.assertTrue(torch.equal(value['delta_float32'], self.core.up(F.silu(r-3*b))))
        gradient = torch.autograd.grad(value['delta'].square().sum(), mean)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()) and bool(gradient.ne(0).any()))

    def test_reference_mean_cancels_arbitrary_zero_hidden_origin(self):
        reference = torch.randn(7, 2, 8)
        query = self.core.query(self.core.rms(self.g))
        reference_messages = self.core.encode(reference, self.g)
        mean = reference_messages.double().mean(0).float()
        centered = self.core.encode(self.local, self.g)-mean
        actual_raw = F.silu(self.core.local(self.core.rms(self.local))+query+self.core.local_bias)
        reference_raw = F.silu(self.core.local(self.core.rms(reference))+query+self.core.local_bias)
        expected = actual_raw-reference_raw.double().mean(0).float()
        torch.testing.assert_close(centered, expected, atol=2e-7, rtol=2e-6)
        # This is a mean of nonlinear messages, not a nonlinear mean hidden state.
        pseudo = self.core.encode(reference.mean(0, keepdim=True), self.g)[0]
        self.assertFalse(torch.allclose(mean, pseudo, atol=1e-5, rtol=1e-5))

    def test_anchor_identity_and_empty_centered_statistic(self):
        mean = torch.randn(2, 4)
        base = self.readout(mean, mode='base', n_elements=16)
        anchored = self.readout(mean, mode='anchored', n_elements=16)
        self.assertTrue(torch.equal(base['delta'], anchored['delta']))
        empty = torch.zeros_like(self.z)
        value = projected_null_readout(self.core, empty, self.g, mean, n_elements=0, mode='centered')
        self.assertTrue(torch.equal(value['corrected_aggregate'], empty))
        self.assertTrue(torch.equal(value['delta'], self.core.decode(empty, self.g)))
        self.assertTrue(bool(value['delta'].ne(0).any())) # global offset is allowed

    def test_centered_statistic_additivity_before_nonlinear_readout(self):
        mean = torch.randn(2, 4)
        a = self.core.aggregate(self.core.encode(self.local[:2], self.g))
        b = self.core.aggregate(self.core.encode(self.local[2:], self.g))
        def corrected(z, n):
            return projected_null_readout(self.core, z, self.g, mean, n_elements=n, mode='centered')
        ca, cb, cab = corrected(a, 2), corrected(b, 3), corrected(a+b, 5)
        torch.testing.assert_close(ca['corrected_aggregate']+cb['corrected_aggregate'],
                                   cab['corrected_aggregate'], atol=1e-6, rtol=1e-6)
        # Neither the shared global offset nor the nonlinear decoder is additive.
        self.assertFalse(torch.allclose(ca['delta']+cb['delta'], cab['delta'], atol=1e-6, rtol=1e-6))

    def test_gradients_reach_only_predictor_when_core_frozen(self):
        self.activate_predictor();self.core.requires_grad_(False)
        saved = {k:v.clone() for k,v in self.core.state_dict().items()}
        local, g = self.local.detach(), self.g.detach()
        z = self.core.aggregate(self.core.encode(local, g))
        mean = self.predictor(self.core.query(self.core.rms(g)))
        result = projected_null_readout(self.core, z, g, mean, n_elements=5, mode='centered')
        result['delta'].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in self.core.parameters()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            and bool(p.grad.ne(0).any()) for p in self.predictor.parameters()))
        for k,v in self.core.state_dict().items():self.assertTrue(torch.equal(v, saved[k]))

    def test_zero_output_initialization_gradient_route(self):
        mean = self.predictor(torch.randn(2, 4))
        mean.sum().backward()
        self.assertTrue(torch.equal(self.predictor.fc1.weight.grad, torch.zeros_like(self.predictor.fc1.weight)))
        self.assertTrue(torch.equal(self.predictor.fc1.bias.grad, torch.zeros_like(self.predictor.fc1.bias)))
        self.assertTrue(bool(self.predictor.fc2.weight.grad.ne(0).any()))
        self.assertTrue(bool(self.predictor.fc2.bias.grad.ne(0).all()))

    def test_native_cast_before_add_and_explicit_output_dtype(self):
        g = torch.ones(2, 8, dtype=torch.float16)
        with torch.no_grad():
            self.core.query.weight.zero_();self.core.aggregate_projection.weight.zero_()
            self.core.aggregate_projection.bias.fill_(1.)
            self.core.up.weight.fill_(.0004884/(4*F.silu(torch.tensor(1.)).item()))
        result = projected_null_readout(self.core, torch.zeros(2, 4), g, torch.zeros(2, 4),
                                        n_elements=16, mode='anchored')
        self.assertEqual(result['delta'].dtype, torch.float16)
        self.assertEqual(result['delta_float32'].dtype, torch.float32)
        native = g+result['delta_float32'].to(g.dtype)
        self.assertTrue(torch.equal(native, g+result['delta']))
        self.assertFalse(torch.equal(native, (g.float()+result['delta_float32']).half()))
        fp32 = projected_null_readout(self.core, torch.zeros(2, 4), g, torch.zeros(2, 4),
                                      n_elements=16, mode='anchored', output_dtype=torch.float32)
        self.assertTrue(torch.equal(fp32['delta'], result['delta_float32']))

    def test_readout_rejects_invalid_shapes_types_modes_and_core(self):
        mean = torch.zeros(2, 4)
        for change in (dict(n_elements=True), dict(n_elements=-1), dict(anchor_n=-1),
                       dict(mode='mean'), dict(output_dtype=torch.int64)):
            with self.assertRaises(ValueError):self.readout(mean, **change)
        with self.assertRaises(ValueError):self.readout(mean.half())
        with self.assertRaises(ValueError):self.readout(torch.zeros(1, 4))
        self.core.merge = 'mean'
        with self.assertRaises(ValueError):self.readout(mean)


if __name__ == '__main__':
    unittest.main()
