"""CPU-only V6 mathematics, gradient isolation, RNG and causal observer checks."""
from __future__ import annotations
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run these tests in a Slurm CPU allocation")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from gnnformer.independent_vision_aggregation import IndependentVisionAggregation, attach_independent_vision_aggregation
from gnnformer.local_evidence_distillation import LocalEvidenceHead, LastPromptMessageObserver, last_prompt_messages, local_teacher_kl
spec = importlib.util.spec_from_file_location("v5_mock_flow", ROOT / "tests/test_independent_vision_aggregation.py")
mock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock)


class LocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.branch = IndependentVisionAggregation(8, rank=3, merge="sum")
        mock.activate(self.branch)
        self.hidden = torch.randn(1, 7, 8, requires_grad=True)
        self.memory = (torch.randn(2, 8), torch.randn(4, 8))
        self.ends = torch.tensor([1, 3])
        self.positions = torch.arange(7)
        self.language = torch.ones(7, dtype=torch.bool)
        self.teacher = torch.tensor([[.8, .15, .05], [.1, .85, .05]])

    def messages(self, hidden=None, branch=None):
        return last_prompt_messages(branch or self.branch, self.hidden if hidden is None else hidden,
            self.memory, self.ends, self.positions, self.language,
            prompt_length=5, total_length=7, expected_images=2)

    def test_rng_independence_and_exact_parameter_count(self):
        before = torch.get_rng_state().clone()
        head = LocalEvidenceHead(seed=6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 291)
        self.assertTrue(torch.equal(head.bias, torch.zeros(3)))
        twin = LocalEvidenceHead(seed=6)
        self.assertTrue(torch.equal(head.weight, twin.weight))
        self.assertFalse(torch.equal(head.weight, LocalEvidenceHead(seed=7).weight))

    def test_formula_matches_native_prompt_row(self):
        self.branch.capture_last_query_messages = True
        self.branch(self.hidden[:, :5], self.memory, self.ends, self.positions[:5], self.language[:5])
        expected = self.branch.export_last_query_diagnostics()["frame_messages"]
        torch.testing.assert_close(self.messages(), expected, rtol=2e-6, atol=1e-7)

    def test_suffix_hidden_cannot_enter_auxiliary_query(self):
        expected = self.messages()
        altered = self.hidden.detach().clone()
        altered[:, 5:] = 1000 * torch.randn_like(altered[:, 5:])
        torch.testing.assert_close(self.messages(altered), expected, rtol=0, atol=0)
        expected.square().sum().backward()
        self.assertTrue(torch.equal(self.hidden.grad[:, 5:], torch.zeros_like(self.hidden.grad[:, 5:])))

    def test_visibility_language_and_prefill_guards(self):
        kwargs = dict(prompt_length=5, total_length=7, expected_images=2)
        with self.assertRaisesRegex(ValueError, "complete"):
            last_prompt_messages(self.branch, self.hidden, self.memory, torch.tensor([1,5]),
                                 self.positions, self.language, **kwargs)
        mask = self.language.clone(); mask[4] = False
        with self.assertRaisesRegex(ValueError, "language query"):
            last_prompt_messages(self.branch, self.hidden, self.memory, self.ends,
                                 self.positions, mask, **kwargs)
        with self.assertRaisesRegex(ValueError, "uncached"):
            last_prompt_messages(self.branch, self.hidden, self.memory, self.ends,
                                 self.positions + 8, self.language, **kwargs)

    def test_mean_images_kl_and_zero_probability_handling(self):
        head = LocalEvidenceHead(3, seed=6)
        messages = self.messages()
        probabilities = torch.tensor([[1.,0.,0.], [0.,1.,0.]])
        loss, logits = local_teacher_kl(head, messages, probabilities, detach_messages=False)
        expected = -(F.log_softmax(logits, -1)[0,0] + F.log_softmax(logits,-1)[1,1]) / 2
        torch.testing.assert_close(loss, expected)
        doubled, _ = local_teacher_kl(head, messages.repeat(2,1), probabilities.repeat(2,1), detach_messages=False)
        torch.testing.assert_close(loss, doubled)

    def test_detached_auxiliary_exact_count_gradient_control(self):
        baseline = copy.deepcopy(self.branch)
        control = copy.deepcopy(self.branch)
        aligned = copy.deepcopy(self.branch)
        head = LocalEvidenceHead(3, seed=6)
        def count_loss(branch):
            output = branch(self.hidden, self.memory, self.ends, self.positions, self.language)
            return output.square().sum() + output.sum() * .07
        count_loss(baseline).backward()
        kl, _ = local_teacher_kl(head, self.messages(branch=control), self.teacher, detach_messages=True)
        (count_loss(control) + kl).backward()
        for a, b in zip(baseline.parameters(), control.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)
        kl, _ = local_teacher_kl(LocalEvidenceHead(3, seed=6), self.messages(branch=aligned),
                                 self.teacher, detach_messages=False)
        kl.backward()
        for component in (aligned.query, aligned.memory, aligned.read):
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in component.parameters()))
        self.assertIsNone(aligned.up.weight.grad)

    def test_native_causal_suffix_and_observer_removal(self):
        model = mock.FakeQwen()
        branch = attach_independent_vision_aggregation(model, layer_index=1, rank=3, merge="sum")
        mock.activate(branch)
        observer = LastPromptMessageObserver(branch)
        pixels = torch.randn(4, 8)
        ids = torch.cat((mock.IDS, torch.tensor([[4,5]])), 1)
        state_before = {k: v.clone() for k,v in branch.state_dict().items()}
        observer.begin(prompt_length=11, total_length=13, expected_images=2)
        first = model(ids, pixel_values=pixels, image_grid_thw=mock.GRID).logits
        messages, metadata = observer.consume()
        self.assertEqual(metadata["branch_calls"], 1)
        self.assertIsNone(observer.messages)
        self.assertEqual(model.model.visual.calls, 1)
        changed = ids.clone(); changed[:, 11:] = torch.tensor([[9,12]])
        observer.begin(prompt_length=11, total_length=13, expected_images=2)
        model(changed, pixel_values=pixels, image_grid_thw=mock.GRID)
        altered, _ = observer.consume()
        torch.testing.assert_close(messages, altered, rtol=0, atol=0)
        inactive = model(ids, pixel_values=pixels, image_grid_thw=mock.GRID).logits
        observer.remove()
        removed = model(ids, pixel_values=pixels, image_grid_thw=mock.GRID).logits
        torch.testing.assert_close(first, inactive, rtol=0, atol=0)
        torch.testing.assert_close(first, removed, rtol=0, atol=0)
        for key,value in branch.state_dict().items():
            torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)
        branch.remove()

    def test_observer_rejects_reentry_and_clears_graph(self):
        observer = LastPromptMessageObserver(self.branch)
        observer.begin(prompt_length=5, total_length=7, expected_images=2)
        with self.assertRaises(ValueError):
            observer.begin(prompt_length=5, total_length=7, expected_images=2)
        self.branch(self.hidden, self.memory, self.ends, self.positions, self.language)
        with self.assertRaises(ValueError):
            self.branch(self.hidden, self.memory, self.ends, self.positions, self.language)
        observer.cancel()
        self.assertIsNone(observer.messages)
        observer.remove()


if __name__ == "__main__":
    unittest.main()
