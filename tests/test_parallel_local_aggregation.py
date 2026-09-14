"""Focused standalone math tests. Run only in a CPU Slurm allocation.

No pretrained models, downloads, training experiments or efficacy measurements.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "cpu":
    raise SystemExit("Run these tests in a Slurm cpu allocation")
if os.environ.get("SLURM_JOB_GPUS"):
    raise SystemExit("These checks require no GPU allocation")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation


class ParallelLocalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260922)
        self.local = torch.randn(5, 3, 8)
        self.global_states = torch.randn(3, 8)

    @staticmethod
    def active(**kwargs):
        module = ParallelLocalAggregation(8, rank=4, **kwargs)
        with torch.no_grad():
            module.up.weight.normal_(0, .2)
        return module

    def test_parameter_matching_and_native_size(self):
        models = [ParallelLocalAggregation(8, rank=4, merge=merge, post_activation=activation)
                  for merge in ("sum", "mean") for activation in ("silu", "identity")]
        expected = {"local_bias": (4,), "query.weight": (4, 8), "local.weight": (4, 8),
                    "aggregate_projection.weight": (4, 4), "aggregate_projection.bias": (4,),
                    "up.weight": (8, 4)}
        for model in models:
            self.assertEqual({k: tuple(v.shape) for k, v in model.state_dict().items()}, expected)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 3 * 8 * 4 + 4 * 4 + 2 * 4)
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        native = ParallelLocalAggregation()
        self.assertEqual(sum(p.numel() for p in native.parameters()), 1_041_600)

    def test_zero_initialization_and_initial_gradient_gate(self):
        module = ParallelLocalAggregation(8, rank=4)
        out = module(self.local, self.global_states)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        (out * torch.randn_like(out)).sum().backward()
        self.assertGreater(float(module.up.weight.grad.abs().sum()), 0)
        for name, parameter in module.named_parameters():
            if name != "up.weight":
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

    def test_zero_local_message_with_nonzero_bias(self):
        module = self.active()
        with torch.no_grad():
            module.local_bias.copy_(torch.tensor([.9, -.7, .4, -.1]))
        messages = module.encode(torch.zeros_like(self.local), self.global_states)
        self.assertTrue(torch.equal(messages, torch.zeros_like(messages)))
        # A neutral local message is not a promise of a zero final residual.
        torch.testing.assert_close(module(torch.zeros_like(self.local), self.global_states),
                                   module.decode(torch.zeros(3, 4), self.global_states))
        awkward = ParallelLocalAggregation(8, rank=13)
        with torch.no_grad():
            awkward.local_bias.normal_(0, .4)
        for n, q in ((1, 1), (3, 2), (7, 5), (0, 2)):
            messages = awkward.encode(torch.zeros(n, q, 8), torch.randn(q, 8))
            self.assertTrue(torch.equal(messages, torch.zeros_like(messages)))

    def test_empty_evidence_learned_global_offset(self):
        for activation in ("silu", "identity"):
            module = self.active(post_activation=activation)
            with torch.no_grad():
                module.query.weight.zero_()
                module.aggregate_projection.bias.fill_(1)
                module.up.weight.fill_(.2)
            local = torch.empty(0, 3, 8)
            messages = module.encode(local, self.global_states)
            self.assertEqual(tuple(messages.shape), (0, 3, 4))
            for merge in ("sum", "mean"):
                module.merge = merge
                pooled = module.aggregate(messages)
                self.assertTrue(torch.equal(pooled, torch.zeros(3, 4)))
                out = module(local, self.global_states)
                self.assertTrue(bool(torch.isfinite(out).all()))
                self.assertGreater(float(out.min()), 0)
                torch.testing.assert_close(out, module.decode(pooled, self.global_states))
            module.zero_grad(set_to_none=True)
            module(local, self.global_states).sum().backward()
            self.assertGreater(float(module.aggregate_projection.bias.grad.abs().sum()), 0)
            self.assertGreater(float(module.query.weight.grad.abs().sum()), 0)

    def test_permutation_and_query_independence(self):
        module = self.active()
        out = module(self.local, self.global_states)
        permuted = module(self.local[torch.tensor([3, 0, 4, 1, 2])], self.global_states)
        torch.testing.assert_close(out, permuted, rtol=2e-6, atol=2e-7)
        one = module(self.local[:, 1:2], self.global_states[1:2])
        torch.testing.assert_close(one, out[1:2], rtol=2e-6, atol=2e-7)
        changed = self.local.clone()
        changed[:, 1:] = 100
        changed_global = self.global_states.clone()
        changed_global[1:] = -100
        independent = module(changed, changed_global)
        torch.testing.assert_close(independent[:1], out[:1], rtol=0, atol=0)

    def test_sum_mean_and_duplication_are_latent_relationships(self):
        module = self.active()
        messages = module.encode(self.local, self.global_states)
        pooled_sum = module.aggregate(messages)
        duplicated = module.aggregate(torch.cat((messages, messages), 0))
        torch.testing.assert_close(duplicated, 2 * pooled_sum)
        module.merge = "mean"
        pooled_mean = module.aggregate(messages)
        torch.testing.assert_close(pooled_sum, len(messages) * pooled_mean)
        torch.testing.assert_close(module.aggregate(torch.cat((messages, messages), 0)), pooled_mean)
        torch.testing.assert_close(module(self.local.repeat(2, 1, 1), self.global_states),
                                   module(self.local, self.global_states), rtol=2e-6, atol=2e-7)
        # A learned nonlinear readout need not commute with scaling.
        tiny = ParallelLocalAggregation(1, rank=1)
        with torch.no_grad():
            tiny.query.weight.zero_()
            tiny.aggregate_projection.weight.fill_(1)
            tiny.aggregate_projection.bias.zero_()
            tiny.up.weight.fill_(1)
        one = tiny.decode(torch.ones(1, 1), torch.zeros(1, 1))
        two = tiny.decode(torch.full((1, 1), 2.), torch.zeros(1, 1))
        self.assertGreater(float((two - 2 * one).abs()), .1)

    def test_reference_formula_and_exposed_composition(self):
        def rms(x):
            x = x.float()
            return x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        for merge in ("sum", "mean"):
            for activation in ("silu", "identity"):
                module = self.active(merge=merge, post_activation=activation)
                with torch.no_grad():
                    module.local_bias.normal_(0, .3)
                    module.aggregate_projection.bias.normal_(0, .3)
                q = F.linear(rms(self.global_states), module.query.weight)
                base = q + module.local_bias
                expected_messages = F.silu(F.linear(rms(self.local), module.local.weight) + base) - F.silu(base)
                z = expected_messages.sum(0)
                if merge == "mean":
                    z = z / len(self.local)
                t = F.linear(z, module.aggregate_projection.weight, module.aggregate_projection.bias) + q
                expected = F.linear(F.silu(t) if activation == "silu" else t, module.up.weight)
                actual = module(self.local, self.global_states)
                torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
                torch.testing.assert_close(module.encode(self.local, self.global_states),
                                           expected_messages, rtol=3e-6, atol=3e-7)
                composed = module.decode(module.aggregate(module.encode(self.local, self.global_states)),
                                         self.global_states)
                torch.testing.assert_close(composed, actual, rtol=0, atol=0)

    def test_active_gradients_reach_every_parameter_and_both_inputs(self):
        for activation in ("silu", "identity"):
            module = self.active(post_activation=activation)
            local = self.local.clone().requires_grad_()
            global_states = self.global_states.clone().requires_grad_()
            out = module(local, global_states)
            (out * torch.randn_like(out)).sum().backward()
            for name, tensor in list(module.named_parameters()) + [("local_input", local), ("global_input", global_states)]:
                self.assertIsNotNone(tensor.grad, name)
                self.assertTrue(bool(torch.isfinite(tensor.grad).all()), name)
                self.assertGreater(float(tensor.grad.abs().sum()), 0, name)

    def test_fp32_arithmetic_under_autocast_and_output_dtype(self):
        module = self.active()
        local = self.local.to(torch.bfloat16)
        global_states = self.global_states.to(torch.bfloat16)
        reference = module(local, global_states, output_dtype=torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            messages = module.encode(local, global_states)
            aggregate = module.aggregate(messages)
            actual = module(local, global_states, output_dtype=torch.float32)
            decoded = module.decode(aggregate, global_states, output_dtype=torch.float32)
        self.assertEqual(messages.dtype, torch.float32)
        self.assertEqual(aggregate.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(decoded, reference, rtol=0, atol=0)
        self.assertEqual(module(local, global_states).dtype, torch.bfloat16)
        self.assertEqual(module(local, global_states, output_dtype=torch.float64).dtype, torch.float64)
        module.half()
        with self.assertRaisesRegex(ValueError, "remain float32"):
            module(local, global_states)

    def test_shapes_finite_inputs_and_configuration_errors(self):
        for arguments in (dict(hidden_size=0), dict(rank=0), dict(rank=True),
                          dict(merge="max"), dict(post_activation="relu")):
            with self.assertRaises(ValueError):
                ParallelLocalAggregation(**arguments)
        module = self.active()
        for local, global_states in ((self.local[0], self.global_states),
                                     (self.local[:, :2], self.global_states),
                                     (self.local, self.global_states.unsqueeze(0)),
                                     (torch.empty(2, 0, 8), torch.empty(0, 8)),
                                     (self.local.long(), self.global_states),
                                     (self.local * float("nan"), self.global_states)):
            with self.assertRaises(ValueError):
                module(local, global_states)
        with self.assertRaises(ValueError):
            module.aggregate(torch.zeros(3, 2, 5))
        with self.assertRaises(ValueError):
            module.decode(torch.zeros(3, 5), self.global_states)
        with self.assertRaises(ValueError):
            module.decode(torch.full((3, 4), float("inf")), self.global_states)
        with self.assertRaises(ValueError):
            module(self.local, self.global_states, output_dtype=torch.int64)


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))))
    unittest.main(verbosity=2)

