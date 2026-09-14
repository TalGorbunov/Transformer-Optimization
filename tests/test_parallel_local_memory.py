"""Slurm CPU tests for explicit multi-query native memory writes.

Tiny causal attention exercises actual K/V persistence and input gradients.
This is not a Qwen/backend numerical-equivalence or efficacy experiment.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu':
    raise SystemExit('Run memory-controller tests in a Slurm cpu allocation')
if os.environ.get('SLURM_JOB_GPUS'):
    raise SystemExit('Memory-controller unit tests require no GPU allocation')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
from gnnformer.parallel_local_memory import ParallelLocalMemory


class FrozenReadLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.7, 1.3, hidden), requires_grad=False)
        self.extra = object()

    def forward(self, hidden_states):
        return hidden_states * self.weight.to(hidden_states.dtype), self.extra


class FrozenFinalBlock(nn.Module):
    """One causal attention layer, with a native concatenating K/V cache."""
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.8, 1.1, hidden), requires_grad=False)
        self.last_input = None

    def forward(self, hidden_states, cache=None):
        self.last_input = hidden_states
        weight = self.weight.to(hidden_states.dtype)
        q = hidden_states * .25
        k, v = hidden_states * weight, hidden_states * weight.flip(0)
        past = 0 if cache is None else cache[0].shape[1]
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=1), torch.cat((cache[1], v), dim=1)
        scores = q @ k.transpose(-1, -2) / hidden_states.shape[-1] ** .5
        permitted = torch.arange(k.shape[1])[None, :] <= (past + torch.arange(q.shape[1]))[:, None]
        scores = scores.masked_fill(~permitted[None], float('-inf'))
        attention = scores.float().softmax(-1).to(hidden_states.dtype) @ v
        return hidden_states + .4 * torch.tanh(attention), (k, v)


class FrozenNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.9, 1.2, hidden), requires_grad=False)

    def forward(self, hidden_states):
        return hidden_states * self.weight.to(hidden_states.dtype)


class TinyNativeModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.read = FrozenReadLayer(hidden)
        self.last = FrozenFinalBlock(hidden)
        self.norm = FrozenNorm(hidden)
        self.head = nn.Linear(hidden, 11, bias=False)
        self.requires_grad_(False)

    def forward(self, hidden_states, cache=None, keyword=False):
        read = self.read(hidden_states)
        if read[1] is not self.read.extra:
            raise AssertionError('Decoder output extras were changed')
        final, new_cache = self.last(read[0], cache)
        normalized = self.norm(hidden_states=final) if keyword else self.norm(final)
        return self.head(normalized), new_cache


class IndependentDeltas(ParallelLocalAggregation):
    """An independent live residual at each query for a causal gradient test."""
    def __init__(self, delta):
        super().__init__(delta.shape[-1], rank=4)
        self.injected = delta

    def forward(self, local_states, global_states, *, output_dtype=None):
        if global_states.shape != self.injected.shape:
            raise ValueError('Independent delta layout changed')
        return self.injected.to(output_dtype)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261008)
        self.model = TinyNativeModel()
        self.branch = ParallelLocalAggregation(8, rank=4)
        self.hidden = torch.randn(4, 5, 8)

    def activate(self):
        with torch.no_grad():
            self.branch.up.weight.normal_(0, .2)

    def controller(self, placement, **kwargs):
        defaults = dict(n_local_rows=3, write_location=placement,
                        query_indices=[2, 3, 4], stream_positions=[2, 3, 4])
        defaults.update(kwargs)
        return ParallelLocalMemory(self.model.read, self.model.norm, self.branch, **defaults)

    def test_zero_identity_cleanup_and_no_native_model_registration(self):
        baseline, baseline_cache = self.model(self.hidden)
        original = {key: value.clone() for key, value in self.model.state_dict().items()}
        counts = (len(self.model.read._forward_hooks), len(self.model.norm._forward_pre_hooks))
        for placement in ('pre_last', 'post_last'):
            controller = self.controller(placement)
            with controller:
                self.model.eval()
                self.assertTrue(self.branch.training)
                self.assertFalse(any(module is self.branch for module in self.model.modules()))
                output, cache = self.model(self.hidden, keyword=True)
                self.assertTrue(torch.equal(output, baseline))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(cache, baseline_cache)))
                controller.assert_complete()
                self.assertEqual((controller.calls, controller.read_calls, controller.norm_calls), (1, 1, 1))
                self.assertIsNone(controller.last_capture)
            controller.close()
            self.assertFalse(controller.active)
            self.assertEqual(counts, (len(self.model.read._forward_hooks), len(self.model.norm._forward_pre_hooks)))
            self.assertTrue(torch.equal(self.model(self.hidden)[0], baseline))
        self.assertEqual(set(original), set(self.model.state_dict()))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_same_reads_and_delta_both_locations_native_addition(self):
        self.activate()
        original = self.hidden.clone()
        baseline, native_cache = self.model(self.hidden)
        common = self.model.read(self.hidden)[0]
        captures = {}
        for placement in ('pre_last', 'post_last'):
            with self.controller(placement, capture=True) as controller:
                output, cache = self.model(self.hidden)
                capture = controller.export_last_capture()
                captures[placement] = capture
                self.assertTrue(torch.equal(output[:-1], baseline[:-1]))
                self.assertTrue(torch.equal(output[-1, :2], baseline[-1, :2]))
                self.assertTrue(torch.equal(capture['write_output'],
                    capture['write_input'] + capture['delta'].to(capture['write_input'].dtype)))
                self.assertTrue(torch.equal(self.hidden, original))
                self.assertTrue(all(torch.equal(a[:-1], b[:-1]) for a, b in zip(cache, native_cache)))
                if placement == 'pre_last':
                    expected = common.clone()
                    expected[-1, 2:] += capture['delta']
                    self.assertTrue(torch.equal(self.model.last.last_input, expected))
                    self.assertFalse(torch.equal(cache[0][-1, 2:], native_cache[0][-1, 2:]))
                else:
                    self.assertTrue(torch.equal(self.model.last.last_input, common))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(cache, native_cache)))
        for key in ('local_states', 'global_states', 'delta'):
            self.assertTrue(torch.equal(captures['pre_last'][key], captures['post_last'][key]))

    def test_all_historical_queries_make_full_and_cached_equivalent(self):
        self.activate()
        for placement in ('pre_last', 'post_last'):
            with self.controller(placement) as controller, torch.no_grad():
                full, full_cache = self.model(self.hidden)
                controller.configure_queries([2], [2])
                first, cache = self.model(self.hidden[:, :3])
                outputs = [first]
                for position in (3, 4):
                    controller.configure_queries([0], [position])
                    current, cache = self.model(self.hidden[:, position:position + 1], cache=cache)
                    outputs.append(current)
                torch.testing.assert_close(torch.cat(outputs, 1), full, atol=2e-6, rtol=2e-6)
                for actual, wanted in zip(cache, full_cache):
                    torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-6)
                controller.assert_complete()
            # Deliberately omitting earlier writes breaks the early reference.
            if placement == 'pre_last':
                with self.controller(placement, query_indices=[4], stream_positions=[4]):
                    incomplete = self.model(self.hidden)[0]
                self.assertGreater(float((incomplete[-1, -1] - full[-1, -1]).abs().max()), 1e-6)

    def test_later_loss_has_past_delta_gradient_only_for_early_write(self):
        for placement in ('pre_last', 'post_last'):
            delta = (.2 * torch.randn(3, 8)).requires_grad_()
            branch = IndependentDeltas(delta)
            with ParallelLocalMemory(self.model.read, self.model.norm, branch,
                    n_local_rows=3, write_location=placement, query_indices=[2, 3, 4],
                    stream_positions=[2, 3, 4], capture=True, detach_captures=False) as controller:
                output = self.model(self.hidden)[0]
                self.assertIs(controller.last_capture['delta'], delta)
                gradient = torch.autograd.grad(output[-1, -1].square().sum(), controller.last_capture['delta'])[0]
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient[-1].abs().sum()), 0)
                if placement == 'pre_last':
                    self.assertGreater(float(gradient[0].abs().sum()), 1e-6)
                else:
                    self.assertTrue(torch.equal(gradient[:-1], torch.zeros_like(gradient[:-1])))

    def test_future_write_cannot_change_past_logits(self):
        for placement in ('pre_last', 'post_last'):
            delta = torch.randn(3, 8)
            branch = IndependentDeltas(delta)
            with ParallelLocalMemory(self.model.read, self.model.norm, branch,
                    n_local_rows=3, write_location=placement, query_indices=[2, 3, 4],
                    stream_positions=[2, 3, 4]):
                original = self.model(self.hidden)[0]
                branch.injected = delta.clone()
                branch.injected[-1] += 10
                changed = self.model(self.hidden)[0]
                self.assertTrue(torch.equal(original[:, :4], changed[:, :4]))
                self.assertFalse(torch.equal(original[-1, -1], changed[-1, -1]))

    def test_native_parameters_frozen_and_core_gradient_routes_live(self):
        self.activate()
        for placement in ('pre_last', 'post_last'):
            self.branch.zero_grad(set_to_none=True)
            with self.controller(placement, capture=True, detach_captures=False):
                self.model(self.hidden)[0][-1, -1].square().sum().backward()
            for name, parameter in self.branch.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_detached_export_and_inference_capture_are_normal_copies(self):
        self.activate()
        with self.controller('pre_last', capture=True) as controller, torch.inference_mode():
            self.model(self.hidden)
            exported = controller.export_last_capture(cpu=True)
            for key, value in exported.items():
                if isinstance(value, torch.Tensor):
                    self.assertFalse(value.requires_grad)
                    self.assertFalse(value.is_inference())
                    self.assertNotEqual(value.data_ptr(), controller.last_capture[key].data_ptr())
            self.assertEqual(exported['query_indices'], [2, 3, 4])
            self.assertEqual(exported['stream_positions'], [2, 3, 4])
        self.assertIsNotNone(exported)

    def test_error_cleanup_reentry_and_other_hooks_preserved(self):
        seen = []
        external = self.model.norm.register_forward_pre_hook(lambda module, args: seen.append(1))
        controller = self.controller('pre_last')
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with controller:
                self.model.read(self.hidden)
                with self.assertRaisesRegex(ValueError, 'between read'):
                    controller.configure_queries([0], [7])
                raise RuntimeError('synthetic')
        self.assertFalse(controller.active)
        self.assertIn(external.id, self.model.norm._forward_pre_hooks)
        with controller:
            self.model(self.hidden)
            controller.assert_complete()
        self.assertEqual(len(seen), 1)
        external.remove()

    def test_reject_bad_layout_overlapping_owners_and_unpaired_hooks(self):
        for queries, streams in (([], []), ([True], [1]), ([2, 2], [2, 2]),
                                 ([2], [1]), ([1, 2], [1, 4]), ([2], [2, 3])):
            with self.assertRaises(ValueError):
                self.controller('pre_last', query_indices=queries, stream_positions=streams)
        with self.controller('pre_last'):
            with self.assertRaisesRegex(ValueError, 'already owns'):
                with self.controller('post_last'):
                    pass
        with self.assertRaisesRegex(ValueError, 'outside'):
            with self.controller('pre_last', query_indices=[5], stream_positions=[5]):
                self.model(self.hidden)
        with self.assertRaisesRegex(ValueError, 'without the common read'):
            with self.controller('post_last'):
                self.model.norm(self.hidden)
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            with self.controller('pre_last') as controller:
                self.model.read(self.hidden)
                controller.assert_complete()
        self.assertFalse(self.model.read._forward_hooks)
        self.assertFalse(self.model.norm._forward_pre_hooks)

    def test_bfloat16_native_carry_and_float32_delta(self):
        self.activate()
        self.model.to(dtype=torch.bfloat16)
        hidden = self.hidden.to(torch.bfloat16)
        for placement in ('pre_last', 'post_last'):
            with self.controller(placement, capture=True) as controller:
                self.model(hidden)
                capture = controller.last_capture
                self.assertEqual(capture['delta'].dtype, torch.float32)
                self.assertEqual(capture['write_input'].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(capture['write_output'],
                    capture['write_input'] + capture['delta'].to(torch.bfloat16)))
        self.assertTrue(all(p.dtype == torch.float32 for p in self.branch.parameters()))

    def test_no_local_rows_supports_learned_global_offset(self):
        self.activate()
        with self.controller('pre_last', n_local_rows=0, capture=True) as controller:
            self.model(self.hidden[-1:])
            self.assertEqual(controller.last_capture['local_states'].shape, (0, 3, 8))
            self.assertGreater(float(controller.last_capture['delta'].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
