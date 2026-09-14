"""CPU-only software tests for native final-norm fusion and token broadcast."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "cpu":
    raise SystemExit("Run native integration tests in a Slurm cpu allocation")
if os.environ.get("SLURM_JOB_GPUS"):
    raise SystemExit("Native integration unit tests require no GPU allocation")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.parallel_local_native import ParallelLocalNative, GlobalBroadcastLogitsProcessor


class FakeNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.8, 1.2, hidden), requires_grad=False)

    def forward(self, hidden_states):
        return hidden_states * self.weight.to(hidden_states.dtype)


class FakeModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.norm = FakeNorm(hidden)
        self.head = nn.Linear(hidden, 11, bias=False)
        self.requires_grad_(False)

    def forward(self, hidden_states, keyword=False):
        normalized = self.norm(hidden_states=hidden_states) if keyword else self.norm(hidden_states)
        return self.head(normalized)


class NativeHookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260923)
        self.model = FakeModel()
        self.branch = ParallelLocalAggregation(8, rank=4)
        self.hidden = torch.randn(4, 7, 8)

    def activate(self):
        with torch.no_grad():
            self.branch.up.weight.normal_(0, .2)

    def test_zero_init_and_context_removal_preserve_native_model(self):
        baseline = self.model(self.hidden)
        parameters = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        branch_training = self.branch.training
        controller = ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3)
        with controller:
            self.model.eval()
            self.assertEqual(self.branch.training, branch_training)
            self.assertEqual(set(self.model.state_dict()), set(parameters))
            self.assertFalse(any(module is self.branch for module in self.model.modules()))
            torch.testing.assert_close(self.model(self.hidden), baseline, rtol=0, atol=0)
            self.assertEqual(controller.calls, 1)
            self.assertEqual(controller.last_query_position, 6)
            self.assertIsNone(controller.export_last_capture())
        self.assertFalse(controller.active)
        torch.testing.assert_close(self.model(self.hidden), baseline, rtol=0, atol=0)
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, parameters[name]))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))
        controller.close()

    def test_only_final_global_query_changes_without_mutating_input(self):
        self.activate()
        original = self.hidden.clone()
        baseline = self.model(self.hidden)
        expected_hidden = self.hidden.clone()
        expected_hidden[-1, -1:] += self.branch(self.hidden[:-1, -1:], self.hidden[-1, -1:])
        expected = self.model(expected_hidden)
        with ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3) as controller:
            for keyword in (False, True):
                actual = self.model(self.hidden, keyword=keyword)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual[:-1], baseline[:-1]))
                self.assertTrue(torch.equal(actual[-1, :-1], baseline[-1, :-1]))
                self.assertFalse(torch.equal(actual[-1, -1], baseline[-1, -1]))
                self.assertTrue(torch.equal(self.hidden, original))
            self.assertEqual(controller.calls, 2)

    def test_active_branch_gradients_and_frozen_native_gradients(self):
        self.activate()
        with ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3):
            out = self.model(self.hidden)
            (out[-1, -1] * torch.randn(11)).sum().backward()
        for name, parameter in self.branch.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_capture_is_detached_exported_normal_and_independent(self):
        self.activate()
        controller = ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3, capture=True)
        with controller, torch.inference_mode():
            self.model(self.hidden)
            exported = controller.export_last_capture(cpu=True)
            self.assertEqual(tuple(exported["local_states"].shape), (3, 1, 8))
            self.assertEqual(tuple(exported["global_states"].shape), (1, 8))
            for value in exported.values():
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                self.assertFalse(torch.is_inference(value))
                self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(exported["fused_global"],
                                       exported["global_states"] + exported["delta"], rtol=0, atol=0)
            exported["local_states"].fill_(1000)
            again = controller.export_last_capture()
            self.assertFalse(torch.equal(again["local_states"], exported["local_states"]))
            controller.capture = False
            self.model(self.hidden)
            self.assertIsNone(controller.export_last_capture())

    def test_error_removes_hook_and_reentry_resets_metadata(self):
        self.activate()
        baseline = self.model(self.hidden)
        controller = ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with controller:
                self.model(self.hidden)
                raise RuntimeError("intentional")
        self.assertFalse(controller.active)
        torch.testing.assert_close(self.model(self.hidden), baseline, rtol=0, atol=0)
        with controller:
            self.assertEqual(controller.calls, 0)
            with self.assertRaisesRegex(ValueError, "already active"):
                controller.__enter__()
            self.model(self.hidden)
            self.assertEqual(controller.calls, 1)

    def test_shape_norm_freeze_and_native_dtype_contracts(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            ParallelLocalNative(nn.LayerNorm(8), self.branch, n_local_rows=3)
        for count in (-1, True, 1.2):
            with self.assertRaises(ValueError):
                ParallelLocalNative(self.model.norm, self.branch, n_local_rows=count)
        with ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3):
            for hidden in (self.hidden[:3], self.hidden[:, :0], self.hidden[:, :, :7],
                           self.hidden[0], self.hidden.long()):
                with self.assertRaises(ValueError):
                    self.model.norm(hidden)
        native = self.hidden.to(torch.bfloat16)
        self.activate()
        with ParallelLocalNative(self.model.norm, self.branch, n_local_rows=3, capture=True) as controller:
            result = self.model.norm(native)
            capture = controller.export_last_capture()
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(capture["delta"].dtype, torch.bfloat16)
        self.assertEqual(capture["fused_global"].dtype, torch.bfloat16)

    def test_zero_local_rows_retain_global_offset(self):
        self.activate()
        with torch.no_grad():
            self.branch.aggregate_projection.bias.fill_(.8)
        hidden = self.hidden[-1:]
        expected = self.model.norm(hidden + torch.zeros_like(hidden))
        with ParallelLocalNative(self.model.norm, self.branch, n_local_rows=0, capture=True) as controller:
            actual = self.model.norm(hidden)
            capture = controller.export_last_capture()
        self.assertEqual(tuple(capture["local_states"].shape), (0, 1, 8))
        self.assertTrue(torch.equal(actual[:, :-1], expected[:, :-1]))
        self.assertGreater(float(capture["delta"].abs().sum()), 0)


class BroadcastTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([[2, 3, 4], [8, 9, 7], [0, 5, 6]])
        self.scores = torch.tensor([[8., 2., 1., 0.], [0., 9., 2., 1.], [2., 1., 3., 8.]])

    def test_fresh_storage_shared_global_choice_and_unchanged_inputs(self):
        processor = GlobalBroadcastLogitsProcessor(n_local_rows=2, prompt_length=3)
        ids, scores = self.ids.clone(), self.scores.clone()
        result = processor(self.ids, self.scores)
        self.assertTrue(torch.equal(result, scores[-1:].expand_as(scores)))
        self.assertEqual(result.argmax(-1).tolist(), [3, 3, 3])
        self.assertTrue(torch.equal(self.ids, ids))
        self.assertTrue(torch.equal(self.scores, scores))
        result[0, 0] = -100
        self.assertTrue(torch.equal(self.scores, scores))
        self.assertEqual(float(result[1, 0]), float(scores[-1, 0]))

    def test_shared_generated_suffix_validation(self):
        processor = GlobalBroadcastLogitsProcessor(n_local_rows=2, prompt_length=3)
        generated = torch.cat((self.ids, torch.tensor([[3, 2], [3, 2], [3, 2]])), 1)
        processor(generated, self.scores)
        wrong = generated.clone()
        wrong[0, -1] = 1
        with self.assertRaisesRegex(ValueError, "prefixes"):
            processor(wrong, self.scores)
        self.assertEqual(processor.calls, 1)

    def test_masked_scores_and_invalid_shapes(self):
        processor = GlobalBroadcastLogitsProcessor(n_local_rows=2, prompt_length=3)
        masked = self.scores.clone()
        masked[-1, 0] = -torch.inf
        actual = processor(self.ids, masked)
        self.assertTrue(bool(torch.isneginf(actual[:, 0]).all()))
        for ids, scores in ((self.ids[:2], self.scores), (self.ids, self.scores[:2]),
                            (self.ids[:, :2], self.scores), (self.ids.float(), self.scores),
                            (self.ids, self.scores.long()), (self.ids, self.scores * float("nan")),
                            (self.ids, self.scores * float("inf"))):
            with self.assertRaises(ValueError):
                processor(ids, scores)
        masked[-1].fill_(-torch.inf)
        with self.assertRaisesRegex(ValueError, "finite candidate"):
            processor(self.ids, masked)
        for args in (dict(n_local_rows=-1, prompt_length=3),
                     dict(n_local_rows=2, prompt_length=0)):
            with self.assertRaises(ValueError):
                GlobalBroadcastLogitsProcessor(**args)

    def test_processor_autograd_reads_only_global_scores(self):
        scores = self.scores.clone().requires_grad_()
        processor = GlobalBroadcastLogitsProcessor(n_local_rows=2, prompt_length=3)
        processor(self.ids, scores).sum().backward()
        self.assertTrue(torch.equal(scores.grad[:-1], torch.zeros_like(scores.grad[:-1])))
        self.assertTrue(torch.equal(scores.grad[-1], torch.full_like(scores.grad[-1], 3.)))


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))))
    unittest.main(verbosity=2)

