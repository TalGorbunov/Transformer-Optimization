"""Semantic aggregation algebra and gradient tests; CPU Slurm execution only.

No pretrained head, semantic-gate computation, fitting or efficacy measurement.
"""
from __future__ import annotations

import math
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

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation


class SemanticAggregationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261120)
        self.local = torch.randn(5, 3, 8)
        self.global_states = torch.randn(3, 8)
        self.gates = torch.tensor([[.4, .7, 1.], [0., 0., 0.], [.8, .2, .6],
                                   [1., .4, .3], [.5, .6, .9]])

    @staticmethod
    def active():
        module = ParallelLocalSemanticAggregation(8, rank=4)
        with torch.no_grad():
            module.up.weight.normal_(0, .2)
            module.local_bias.copy_(torch.tensor([.9, -.7, .4, -.1]))
        return module

    def test_parameter_shape_count_and_paired_initialization(self):
        torch.manual_seed(73)
        ancestor = ParallelLocalAggregation(8, rank=4)
        torch.manual_seed(73)
        module = ParallelLocalSemanticAggregation(8, rank=4)
        self.assertEqual(set(ancestor.state_dict()), set(module.state_dict()))
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, ancestor.state_dict()[name]))
            self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(list(module.named_buffers()), [])
        native = ParallelLocalSemanticAggregation()
        self.assertEqual(sum(p.numel() for p in native.parameters()), 1_041_600)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 120)

    def test_hand_computed_two_coordinate_example(self):
        module = ParallelLocalSemanticAggregation(2, rank=2)
        with torch.no_grad():
            module.query.weight.zero_()
            module.local.weight.copy_(torch.eye(2))
            module.aggregate_projection.weight.copy_(torch.eye(2))
            module.up.weight.copy_(torch.eye(2))
        local = torch.tensor([[[1., 0.]], [[0., 1.]], [[-1., 0.]]])
        gates = torch.tensor([[.25], [0.], [1.]])
        result, cap = module(local, torch.zeros(1, 2), gates=gates, capture=True)
        a = math.tanh(1. / math.sqrt(.5 + 1e-6))
        z = -.75 * a
        expected_payload = torch.tensor([[[a, 0.]], [[0., a]], [[-a, 0.]]])
        torch.testing.assert_close(cap["payload"], expected_payload)
        torch.testing.assert_close(cap["aggregate"], torch.tensor([[z, 0.]]))
        torch.testing.assert_close(result, torch.tensor([[z / (1. + math.exp(-z)), 0.]]))
        self.assertTrue(torch.equal(cap["messages"][1], torch.zeros(1, 2)))

    def test_closed_item_has_no_local_payload_bypass(self):
        module = self.active()
        original, cap = module(self.local, self.global_states, gates=self.gates, capture=True)
        changed = self.local.clone()
        changed[1] = torch.randn_like(changed[1]) * 10
        torch.testing.assert_close(module(changed, self.global_states, gates=self.gates),
                                   original, rtol=0, atol=0)
        self.assertTrue(torch.equal(cap["messages"][1], torch.zeros(3, 4)))
        changed[0] = -self.local[0]
        self.assertGreater(float((module(changed, self.global_states, gates=self.gates)
                                  - original).abs().max()), 1e-5)
        zeros = torch.zeros_like(self.gates)
        out, closed = module(self.local, self.global_states, gates=zeros, capture=True)
        self.assertTrue(torch.equal(closed["messages"], torch.zeros(5, 3, 4)))
        self.assertTrue(torch.equal(closed["aggregate"], torch.zeros(3, 4)))
        torch.testing.assert_close(out, module.decode(torch.zeros(3, 4), self.global_states))
        self.assertGreater(float(out.abs().max()), 0)

    def test_gate_is_detached_but_open_payload_and_readout_are_live(self):
        module = self.active()
        local = self.local.clone().requires_grad_()
        global_states = self.global_states.clone().requires_grad_()
        gates = self.gates.clone().requires_grad_()
        out, cap = module(local, global_states, gates=gates, capture=True)
        out.square().sum().backward()
        self.assertIsNone(gates.grad)
        self.assertTrue(gates.requires_grad)
        self.assertFalse(cap["gates"].requires_grad)
        self.assertTrue(cap["payload"].requires_grad)
        self.assertTrue(cap["delta"].requires_grad)
        self.assertTrue(torch.equal(local.grad[1], torch.zeros_like(local.grad[1])))
        self.assertGreater(float(local.grad[0].abs().sum()), 0)
        self.assertGreater(float(global_states.grad.abs().sum()), 0)
        for name, value in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(value.grad)
                self.assertTrue(bool(torch.isfinite(value.grad).all()))
                self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_zero_up_native_identity_and_initial_gradient_route(self):
        module = ParallelLocalSemanticAggregation(8, rank=4)
        gates = self.gates.clone().requires_grad_()
        delta = module(self.local, self.global_states, gates=gates)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        native = self.global_states.half()
        self.assertTrue(torch.equal(native + delta.half(), native))
        (delta * torch.randn_like(delta)).sum().backward()
        self.assertIsNone(gates.grad)
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad)
            if name == "up.weight":
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            else:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

    def test_empty_set_and_explicit_padding_gate(self):
        module = self.active()
        with torch.no_grad():
            module.query.weight.zero_()
            module.aggregate_projection.weight.copy_(torch.eye(4))
            module.aggregate_projection.bias.fill_(1)
            module.up.weight.fill_(.2)
            module.local_bias.fill_(.5)
        empty = torch.empty(0, 3, 8)
        out, cap = module(empty, self.global_states, gates=torch.empty(0, 3), capture=True)
        self.assertEqual(tuple(cap["messages"].shape), (0, 3, 4))
        self.assertTrue(torch.equal(cap["aggregate"], torch.zeros(3, 4)))
        self.assertGreater(float(out.min()), 0)
        # A raw-zero padded item is neutral only when its explicit gate is zero.
        pad = torch.zeros(1, 3, 8)
        torch.testing.assert_close(module(pad, self.global_states, gates=torch.zeros(1, 3)),
                                   out, rtol=0, atol=0)
        open_pad = module(pad, self.global_states, gates=torch.ones(1, 3))
        self.assertTrue(bool((open_pad > out).all()))

    def test_permutation_equivariance_and_query_independence(self):
        module = self.active()
        out, cap = module(self.local, self.global_states, gates=self.gates, capture=True)
        perm = torch.tensor([4, 1, 3, 0, 2])
        perm_out, perm_cap = module(self.local[perm], self.global_states,
                                   gates=self.gates[perm], capture=True)
        torch.testing.assert_close(perm_cap["messages"], cap["messages"][perm])
        torch.testing.assert_close(perm_out, out)
        changed_local, changed_global = self.local.clone(), self.global_states.clone()
        changed_local[:, 2] *= -1
        changed_global[2] *= -1
        changed_out = module(changed_local, changed_global, gates=self.gates)
        torch.testing.assert_close(changed_out[:2], out[:2], rtol=0, atol=0)

    def test_capture_bounds_encode_decode_and_no_retained_state(self):
        module = self.active()
        buffers_before = dict(module.named_buffers())
        out, cap = module(self.local, self.global_states, gates=self.gates, capture=True)
        self.assertEqual(set(cap), {"gates", "payload", "messages", "query", "aggregate",
                                   "preactivation", "delta"})
        self.assertTrue(bool((cap["payload"].abs() <= 1).all()))
        self.assertTrue(bool((cap["messages"].abs() <= self.gates.unsqueeze(-1)).all()))
        messages, encoding = module.encode(self.local, self.global_states,
                                           gates=self.gates, capture=True)
        self.assertEqual(set(encoding), {"gates", "payload", "messages", "query"})
        torch.testing.assert_close(messages, cap["messages"], rtol=0, atol=0)
        torch.testing.assert_close(module.decode(module.aggregate(messages), self.global_states),
                                   out, rtol=0, atol=0)
        self.assertEqual(dict(module.named_buffers()), buffers_before)
        self.assertFalse(any(key.startswith("last_") for key in vars(module)))

    def test_expanded_gates_native_cast_and_autocast_boundary(self):
        module = self.active()
        gates = torch.tensor([1., 0., .4, .2, .8])[:, None].expand(5, 3)
        for dtype in (torch.float16, torch.bfloat16):
            local, global_states = self.local.to(dtype), self.global_states.to(dtype)
            expected = module(local.float(), global_states.float(), gates=gates)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out, cap = module(local, global_states, gates=gates, capture=True)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(all(value.dtype == torch.float32 for value in cap.values()))
            torch.testing.assert_close(cap["delta"], expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(out, expected.to(dtype)))
            double = module(local, global_states, gates=gates, output_dtype=torch.float64)
            self.assertEqual(double.dtype, torch.float64)
            torch.testing.assert_close(double, expected.double(), rtol=0, atol=0)

    def test_invalid_gates_are_rejected_without_implicit_broadcast(self):
        module = self.active()
        bad = [torch.ones(5), torch.ones(5, 1), torch.ones(4, 3),
               torch.ones(5, 3, 1), torch.ones(5, 3, dtype=torch.float16),
               torch.ones(5, 3, dtype=torch.float64), torch.ones(5, 3, dtype=torch.int64),
               torch.ones(5, 3, dtype=torch.bool), None, [[1.] * 3] * 5]
        for value in (-.01, 1.01, float("nan"), float("inf")):
            gates = self.gates.clone()
            gates[0, 0] = value
            bad.append(gates)
        for gates in bad:
            with self.subTest(gate=repr(gates)):
                with self.assertRaises(ValueError):
                    module(self.local, self.global_states, gates=gates)
                with self.assertRaises(ValueError):
                    module.encode(self.local, self.global_states, gates=gates)
        with self.assertRaises(TypeError):
            module(self.local, self.global_states)

    def test_invalid_states_parameters_and_configuration(self):
        for kwargs in ({"merge": "mean"}, {"post_activation": "identity"}):
            with self.assertRaises(ValueError):
                ParallelLocalSemanticAggregation(8, rank=4, **kwargs)
        for field, value in (("merge", "mean"), ("post_activation", "identity")):
            module = self.active()
            setattr(module, field, value)
            for call in (
                lambda: module(self.local, self.global_states, gates=self.gates),
                lambda: module.encode(self.local, self.global_states, gates=self.gates),
                lambda: module.aggregate(torch.zeros(5, 3, 4)),
                lambda: module.decode(torch.zeros(3, 4), self.global_states),
            ):
                with self.assertRaises(ValueError):
                    call()
        module = self.active()
        bad_local = self.local.clone()
        bad_local[1, 0, 0] = float("nan")  # Closed gates cannot hide invalid states.
        with self.assertRaises(ValueError):
            module(bad_local, self.global_states, gates=self.gates)
        for local, global_states in ((self.local[:, :2], self.global_states),
                                     (self.local, self.global_states[:, :7]),
                                     (self.local[:, :0], self.global_states[:0]),
                                     (self.local, torch.full((3, 8), float("inf")))):
            with self.assertRaises(ValueError):
                module(local, global_states, gates=self.gates)
        with self.assertRaises(ValueError):
            module(self.local, self.global_states, gates=self.gates, output_dtype=torch.int64)
        with self.assertRaises(ValueError):
            module(self.local, self.global_states, gates=self.gates, capture="yes")
        with torch.no_grad():
            module.up.weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            module(self.local, self.global_states, gates=self.gates)
        module = self.active().half()
        with self.assertRaises(ValueError):
            module(self.local, self.global_states, gates=self.gates)
        with self.assertRaises(ValueError):
            module.aggregate(torch.zeros(5, 3, 4))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    unittest.main()
