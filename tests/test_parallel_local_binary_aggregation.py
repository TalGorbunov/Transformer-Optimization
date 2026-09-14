"""Meaningful control-input and gradient tests; execute only on Slurm CPU."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import unittest

if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "cpu":
    raise SystemExit("Run in a Slurm CPU allocation")
if os.environ.get("SLURM_JOB_GPUS"):
    raise SystemExit("This suite requires no GPU")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gnnformer.parallel_local_binary_aggregation import ParallelLocalBinaryAggregation
from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation


class BinaryAggregationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261122)
        self.local = torch.randn(16, 3, 8)
        self.global_state = torch.randn(3, 8)
        self.gates = torch.zeros(16, 3)
        self.gates[[0, 2, 5, 8, 11, 15]] = 1

    def active(self, mode):
        core = ParallelLocalBinaryAggregation(8, rank=4, mode=mode)
        with torch.no_grad():
            core.up.weight.normal_(0, .2)
        return core

    def test_same_parameters_and_zero_native_identity(self):
        torch.manual_seed(42)
        vector = ParallelLocalBinaryAggregation(8, rank=4, mode="vector")
        torch.manual_seed(42)
        scalar = ParallelLocalBinaryAggregation(8, rank=4, mode="scalar")
        self.assertEqual(set(vector.state_dict()), set(scalar.state_dict()))
        for k, v in vector.state_dict().items():
            self.assertTrue(torch.equal(v, scalar.state_dict()[k]))
        self.assertFalse(list(vector.buffers()))
        self.assertFalse(list(scalar.buffers()))
        for core in (vector, scalar):
            delta = core(self.local, self.global_state, gates=self.gates)
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertEqual(sum(p.numel() for p in ParallelLocalBinaryAggregation().parameters()), 1041600)

    def test_vector_matches_frozen_parent(self):
        core = self.active("vector")
        parent = ParallelLocalSemanticAggregation(8, rank=4)
        parent.load_state_dict(core.state_dict())
        a, ac = core(self.local, self.global_state, gates=self.gates, capture=True)
        b, bc = parent(self.local, self.global_state, gates=self.gates, capture=True)
        self.assertTrue(torch.equal(a, b))
        for k in bc:
            self.assertTrue(torch.equal(ac[k], bc[k]), k)

    def test_scalar_ignores_contents_and_gate_positions_exactly(self):
        core = self.active("scalar")
        a, cap = core(self.local, self.global_state, gates=self.gates, capture=True)
        changed_local = torch.randn_like(self.local) * 1000
        changed_gates = self.gates.flip(0)
        b, other = core(changed_local, self.global_state, gates=changed_gates, capture=True)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(cap["count"], torch.full((3,), 6.)))
        self.assertTrue(torch.equal(cap["aggregate"], 6 * cap["scalar_payload"]))
        self.assertTrue(torch.equal(cap["aggregate"], other["aggregate"]))
        self.assertTrue(torch.equal(cap["query"], other["query"]))
        # A change to the count is allowed to affect the control.
        less = changed_gates.clone(); less[0] = 0
        self.assertGreater(float((core(changed_local, self.global_state, gates=less) - a).abs().max()), 0)
        with self.assertRaises(ValueError):
            core.encode(self.local, self.global_state, gates=self.gates)

    def test_equal_count_can_preserve_different_vector_contents(self):
        core = self.active("vector")
        first = core(self.local, self.global_state, gates=self.gates)
        changed = self.local.clone(); changed[0] *= -1
        self.assertGreater(float((first-core(changed, self.global_state, gates=self.gates)).abs().max()), 1e-5)
        closed = self.local.clone(); closed[1] *= -100
        self.assertTrue(torch.equal(first, core(closed, self.global_state, gates=self.gates)))

    def test_gradients_respect_scalar_information_boundary(self):
        for mode in ("vector", "scalar"):
            core = self.active(mode)
            local = self.local.clone().requires_grad_()
            global_state = self.global_state.clone().requires_grad_()
            gates = self.gates.clone().requires_grad_()
            core(local, global_state, gates=gates).square().sum().backward()
            self.assertIsNone(gates.grad)
            self.assertGreater(float(global_state.grad.abs().sum()), 0)
            if mode == "scalar":
                self.assertIsNone(local.grad)
            else:
                self.assertTrue(torch.equal(local.grad[1], torch.zeros_like(local.grad[1])))
                self.assertGreater(float(local.grad[0].abs().sum()), 0)
            for name, p in core.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
                self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_empty_closed_and_dtype(self):
        for mode in ("vector", "scalar"):
            core = self.active(mode)
            for local, gates in ((self.local[:0], self.gates[:0]), (self.local, self.gates*0)):
                out, cap = core(local.half(), self.global_state.half(), gates=gates, capture=True)
                self.assertEqual(out.dtype, torch.float16)
                self.assertTrue(torch.equal(cap["aggregate"], torch.zeros(3, 4)))
                self.assertTrue(torch.equal(out, core.decode(torch.zeros(3, 4), self.global_state.half())))

    def test_invalid_measurements_are_not_silently_closed(self):
        core = self.active("vector")
        for value in (.5, float("nan"), -1., 2.):
            gates = self.gates.clone(); gates[0, 0] = value
            with self.assertRaises(ValueError):
                core(self.local, self.global_state, gates=gates)
        with self.assertRaises(ValueError):
            core(self.local, self.global_state, gates=self.gates.double())
        with self.assertRaises(ValueError):
            ParallelLocalBinaryAggregation(mode="unknown")

if __name__ == "__main__":
    unittest.main()
