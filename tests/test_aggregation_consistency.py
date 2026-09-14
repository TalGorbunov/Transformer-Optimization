"""Standalone loss checks; run only in a CPU Slurm allocation.

No pretrained models, data loading, optimizer steps, or efficacy measurements.
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

from gnnformer.aggregation_consistency import paired_residual_consistency


class AggregationConsistencyTests(unittest.TestCase):
    @staticmethod
    def inputs():
        delta = torch.tensor([[[3., -4.], [-2., 1.]],
                              [[0., 0.], [0., 0.]]])
        global_states = torch.tensor([[[3., 4.], [1., 2.]],
                                      [[3., 4.], [1., 2.]]])
        return delta, global_states

    def test_analytic_squared_l2_and_independent_position_normalizers(self):
        delta, global_states = self.inputs()
        expected = .5 * (25. / (25. + 1e-6) + 5. / (5. + 1e-6))
        actual = paired_residual_consistency(delta, global_states)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.ndim, 0)
        self.assertAlmostEqual(float(actual), expected, places=6)
        scaled = paired_residual_consistency(delta, global_states * 2)
        self.assertAlmostEqual(float(scaled), .25, places=6)

    def test_count_and_eos_differences_cannot_cancel(self):
        delta = torch.tensor([[[2., -3.], [-2., 3.]],
                              [[0., 0.], [0., 0.]]])
        global_states = torch.ones_like(delta)
        self.assertTrue(torch.equal((delta[0] - delta[1]).mean(0), torch.zeros(2)))
        self.assertAlmostEqual(float(paired_residual_consistency(delta, global_states)),
                               13. / (2. + 1e-6), places=5)

    def test_identical_signed_or_zero_residuals_are_exact_noops(self):
        for value in (torch.tensor([[2., -3.], [-7., 5.]]), torch.zeros(2, 2)):
            delta = torch.stack((value, value)).requires_grad_()
            global_states = torch.zeros_like(delta)
            loss = paired_residual_consistency(delta, global_states)
            self.assertEqual(float(loss), 0.)
            loss.backward()
            self.assertTrue(torch.equal(delta.grad, torch.zeros_like(delta)))
            self.assertIsNone(global_states.grad)

    def test_both_sides_receive_opposite_gradients_and_pair_swap_is_symmetric(self):
        delta, global_states = self.inputs()
        delta.requires_grad_()
        loss = paired_residual_consistency(delta, global_states)
        loss.backward()
        self.assertGreater(float(delta.grad[0].abs().sum()), 0.)
        torch.testing.assert_close(delta.grad[0], -delta.grad[1], rtol=0, atol=0)
        # The squared-loss derivative's factor 2 cancels the two-position mean.
        expected = (delta.detach()[0] - delta.detach()[1]) / torch.tensor([[25. + 1e-6], [5. + 1e-6]])
        torch.testing.assert_close(delta.grad[0], expected)
        swapped = paired_residual_consistency(delta.flip(0), global_states.flip(0))
        torch.testing.assert_close(swapped, loss, rtol=0, atol=0)

    def test_mean_over_pairs_does_not_reweight_hidden_dimensions(self):
        delta, global_states = self.inputs()
        one = paired_residual_consistency(delta, global_states)
        # The extra pair has a different frozen query and zero discrepancy.
        zeros = torch.zeros_like(delta)
        two = paired_residual_consistency(torch.cat((delta, zeros)),
                                          torch.cat((global_states, global_states * 3)))
        torch.testing.assert_close(two, one / 2, rtol=0, atol=0)
        duplicated_features = paired_residual_consistency(delta.repeat(1, 1, 2),
                                                          global_states.repeat(1, 1, 2))
        self.assertAlmostEqual(float(duplicated_features), float(one), places=6)

    def test_gradient_reaches_shared_trainable_residual_parameters(self):
        weight = torch.nn.Parameter(torch.tensor([[.4, -.2], [.1, .7]]))
        features = torch.tensor([[[1., 0.], [0., 1.]],
                                 [[0., 1.], [-1., 0.]]])
        delta = features @ weight.T
        loss = paired_residual_consistency(delta, torch.ones_like(delta))
        loss.backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(bool(torch.isfinite(weight.grad).all()))
        self.assertGreater(float(weight.grad.abs().sum()), 0.)

    def test_native_half_global_is_promoted_before_square_and_autocast_is_disabled(self):
        delta = torch.tensor([[[300., 400.], [-300., -400.]],
                              [[0., 0.], [0., 0.]]])
        global_states = torch.tensor([[[300., 400.], [300., 400.]],
                                      [[300., 400.], [300., 400.]]], dtype=torch.float16)
        reference = paired_residual_consistency(delta, global_states)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = paired_residual_consistency(delta, global_states)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(float(actual), 1.)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_difference_smaller_than_native_half_resolution_is_retained(self):
        delta = torch.ones(2, 2, 1)
        delta[0] += 2. ** -15
        self.assertTrue(torch.equal(delta[0].half(), delta[1].half()))
        self.assertGreater(float(paired_residual_consistency(delta, torch.ones_like(delta))), 0.)
        with self.assertRaisesRegex(ValueError, "float32"):
            paired_residual_consistency(delta.half(), torch.ones_like(delta))

    def test_frozen_state_must_match_within_pair_and_have_no_gradient(self):
        delta, global_states = self.inputs()
        changed = global_states.clone()
        changed[1, 1, 0] += .01
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_residual_consistency(delta, changed)
        with self.assertRaisesRegex(ValueError, "no gradient"):
            paired_residual_consistency(delta, global_states.requires_grad_())

    def test_shape_dtype_and_epsilon_rejections(self):
        delta, global_states = self.inputs()
        cases = [(delta[:1], global_states[:1]),
                 (delta[:0], global_states[:0]),
                 (delta[:, :1], global_states[:, :1]),
                 (delta[:, :, :0], global_states[:, :, :0]),
                 (delta[0], global_states[0]),
                 (delta, global_states[:, :, :1]),
                 (delta.double(), global_states),
                 (delta, global_states.long()),
                 (None, global_states)]
        for first, second in cases:
            with self.subTest(first_shape=getattr(first, "shape", None), second_shape=second.shape):
                with self.assertRaises(ValueError):
                    paired_residual_consistency(first, second)
        for eps in (0., -1., float("nan"), float("inf"), True, "1e-6"):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ValueError, "eps"):
                    paired_residual_consistency(delta, global_states, eps=eps)

    def test_nonfinite_inputs_and_float32_overflow_fail_explicitly(self):
        delta, global_states = self.inputs()
        for value in (float("nan"), float("inf"), -float("inf")):
            for which in ("delta", "global"):
                changed_delta, changed_global = delta.clone(), global_states.clone()
                if which == "delta":
                    changed_delta[0, 0, 0] = value
                else:
                    changed_global[:, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    paired_residual_consistency(changed_delta, changed_global)
        huge_global = torch.full_like(global_states, 1e30)
        with self.assertRaisesRegex(ValueError, "denominator"):
            paired_residual_consistency(delta, huge_global)
        huge_delta = delta.clone()
        huge_delta[0] = 1e30
        with self.assertRaisesRegex(ValueError, "loss must be finite"):
            paired_residual_consistency(huge_delta, global_states)


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))))
    unittest.main()
