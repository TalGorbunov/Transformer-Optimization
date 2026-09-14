"""Held numerical tests: execute only in a CPU Slurm allocation."""
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu':
    raise SystemExit('CPU Slurm required; no numerical execution on login')

import torch
from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection, selection_weights, MODES
from scripts.native_learned_selection_runtime import LearnedSelectionNative, prepare_scene


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(6, dtype=torch.float16), requires_grad=False)
        self.eval()
    def forward(self, hidden_states): return hidden_states * self.weight


class LearnedSelectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(731)
        self.local = torch.randn(4, 3, 6)
        self.global_states = torch.randn(3, 6)

    def core(self, mode='clip', *, active=True):
        core = ParallelLocalLearnedSelection(hidden_size=6, rank=4, mode=mode)
        if active:
            with torch.no_grad(): core.up.weight.normal_(0, .2)
        return core

    def test_parameter_count_shapes_initialization_and_dtype(self):
        full = ParallelLocalLearnedSelection()
        self.assertEqual(sum(p.numel() for p in full.parameters()), 1041697)
        self.assertTrue(all(p.dtype == torch.float32 for p in full.parameters()))
        self.assertTrue(torch.equal(full.selection_weight, torch.zeros(96)))
        self.assertTrue(torch.equal(full.selection_bias, torch.tensor([.5])))
        self.assertTrue(torch.equal(full.up.weight, torch.zeros_like(full.up.weight)))
        shapes = {mode: {k: tuple(v.shape) for k, v in self.core(mode).state_dict().items()} for mode in MODES}
        self.assertEqual(shapes['clip'], shapes['sigmoid']); self.assertEqual(shapes['clip'], shapes['softmax'])
        with self.assertRaisesRegex(ValueError, 'float32'):
            self.core().half()(self.local.half(), self.global_states.half())

    def test_initial_clip_sigmoid_value_and_full_gradient_match(self):
        clip = self.core(); sigmoid = self.core('sigmoid'); sigmoid.load_state_dict(clip.state_dict())
        l0 = self.local.clone().requires_grad_(); g0 = self.global_states.clone().requires_grad_()
        l1 = self.local.clone().requires_grad_(); g1 = self.global_states.clone().requires_grad_()
        y0, c0 = clip(l0, g0, capture=True); y1, c1 = sigmoid(l1, g1, capture=True)
        self.assertTrue(torch.equal(c0['gates'], torch.full((4, 3), .5)))
        self.assertTrue(torch.equal(y0, y1)); self.assertTrue(torch.equal(c0['messages'], c1['messages']))
        y0.square().sum().backward(); y1.square().sum().backward()
        torch.testing.assert_close(l0.grad, l1.grad, rtol=0, atol=0)
        torch.testing.assert_close(g0.grad, g1.grad, rtol=0, atol=0)
        for a, b in zip(clip.parameters(), sigmoid.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        s = torch.full((2, 3), .5, requires_grad=True)
        for mode in ('clip', 'sigmoid'):
            derivative, = torch.autograd.grad(selection_weights(s, mode).sum(), s)
            self.assertTrue(torch.equal(derivative, torch.ones_like(s)))

    def test_empty_all_masked_singleton_and_padding(self):
        for mode in MODES:
            core = self.core(mode)
            _, empty = core(self.local[:0], self.global_states, capture=True)
            self.assertEqual(empty['gates'].shape, (0, 3))
            self.assertTrue(torch.equal(empty['aggregate'], torch.zeros(3, 4)))
            mask = torch.tensor([[True, False, True], [False, False, False],
                                 [True, False, False], [False, False, False]])
            x = self.local.clone().requires_grad_()
            y, cap = core(x, self.global_states, valid_mask=mask, capture=True)
            self.assertTrue(torch.equal(cap['gates'][~mask], torch.zeros_like(cap['gates'][~mask])))
            self.assertTrue(torch.equal(cap['aggregate'][1], torch.zeros(4)))
            y.square().sum().backward()
            self.assertTrue(torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask])))
            self.assertTrue(bool(torch.isfinite(y).all()))
            _, one = core(self.local[:1], self.global_states, capture=True)
            torch.testing.assert_close(one['gates'], torch.full((1, 3), 1. if mode == 'softmax' else .5))
            if mode == 'softmax':
                torch.testing.assert_close(cap['gates'].sum(0), torch.tensor([1., 0., 1.]))

    def test_bounds_clip_zero_region_and_finite_precision_saturation(self):
        scores = torch.tensor([[-1000., -2., -.1], [.1, .5, .9], [1.1, 2., 1000.]], requires_grad=True)
        for mode in MODES:
            g = selection_weights(scores, mode)
            self.assertTrue(bool(torch.isfinite(g).all() and ((g >= 0) & (g <= 1)).all()))
        clipped = selection_weights(scores, 'clip')
        self.assertTrue(torch.equal(clipped[0], torch.zeros(3)))
        derivative, = torch.autograd.grad(clipped.sum(), scores)
        self.assertTrue(torch.equal(derivative[0], torch.zeros(3)))
        self.assertTrue(torch.equal(derivative[1], torch.ones(3)))
        self.assertTrue(torch.equal(derivative[2], torch.zeros(3)))
        # FP32 sigmoid may saturate to literal zero/one; do not assume strict positivity.
        smooth = selection_weights(scores, 'sigmoid')
        self.assertEqual(float(smooth[0, 0]), 0.); self.assertEqual(float(smooth[-1, -1]), 1.)

    def test_repeat_set_sum_versus_normalized_attention_and_permutation(self):
        for mode in MODES:
            core = self.core(mode)
            with torch.no_grad(): core.selection_weight.normal_(0, .1)
            _, base = core(self.local, self.global_states, capture=True)
            _, repeated = core(self.local.repeat(2, 1, 1), self.global_states, capture=True)
            expected = base['aggregate'] * (1 if mode == 'softmax' else 2)
            torch.testing.assert_close(repeated['aggregate'], expected, rtol=1e-6, atol=1e-6)
            _, permuted = core(self.local[[3, 1, 0, 2]], self.global_states, capture=True)
            torch.testing.assert_close(permuted['aggregate'], base['aggregate'], rtol=1e-6, atol=1e-6)

    def test_live_gate_gradients_and_softmax_bias_redundancy(self):
        for mode in MODES:
            core = self.core(mode); x = self.local.clone().requires_grad_(); g = self.global_states.clone().requires_grad_()
            y = core(x, g); y.square().sum().backward()
            for p in (core.selection_weight, core.query.weight, core.local.weight, core.up.weight):
                self.assertIsNotNone(p.grad); self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(float(p.grad.abs().sum()), 0.)
            self.assertGreater(float(x.grad.abs().sum()), 0.); self.assertGreater(float(g.grad.abs().sum()), 0.)
            if mode == 'softmax':
                self.assertLess(float(core.selection_bias.grad.abs().max()), 1e-5)
                with torch.no_grad(): core.selection_bias.add_(.25)
                torch.testing.assert_close(core(self.local, self.global_states), y.detach(), rtol=1e-5, atol=1e-6)

    def test_zero_up_identity_and_first_gradient_route(self):
        norm = Norm(); h = torch.randn(5, 4, 6).half()
        for mode in MODES:
            core = self.core(mode, active=False)
            with LearnedSelectionNative(norm, core, n_local_rows=4, selection_mode=mode, capture=True) as hook:
                out = norm(h)
            self.assertTrue(torch.equal(out, h)); self.assertEqual(hook.calls, 1)
            out[-1, -1].float().square().sum().backward()
            self.assertGreater(float(core.up.weight.grad.abs().sum()), 0.)
            for name, parameter in core.named_parameters():
                if name != 'up.weight':
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

    def test_native_cast_single_branch_call_and_frozen_head_gradient(self):
        norm = Norm(); head = torch.nn.Linear(6, 7, bias=False, dtype=torch.float16).eval().requires_grad_(False)
        core = self.core(); h = torch.randn(5, 4, 6).half().requires_grad_(); original = h.detach().clone()
        calls = []; handle = core.register_forward_hook(lambda *_: calls.append(1))
        try:
            with LearnedSelectionNative(norm, core, n_local_rows=4, selection_mode='clip', capture=True) as hook:
                output = norm(h); logits = head(output)
        finally: handle.remove()
        self.assertEqual(calls, [1]); self.assertEqual(hook.calls, 1)
        self.assertTrue(torch.equal(h.detach(), original)); self.assertTrue(torch.equal(output[:-1], original[:-1]))
        self.assertTrue(torch.equal(output[-1, :-1], original[-1, :-1]))
        self.assertTrue(torch.equal(output[-1, -1:], original[-1, -1:] + hook.last_capture['delta'].half()))
        self.assertEqual(hook.last_capture['delta'].dtype, torch.float32)
        self.assertTrue(torch.equal(hook.last_capture['native_query_hidden'], original[:, -1:]))
        logits[-1, -1].float().square().sum().backward()
        self.assertIsNone(head.weight.grad); self.assertIsNone(norm.weight.grad)
        self.assertGreater(float(core.selection_weight.grad.abs().sum()), 0.)
        self.assertIsNone(next(iter(norm.children()), None))
        exported = hook.export_last_capture(cpu=True)
        self.assertFalse(any(t.requires_grad or torch.is_inference(t) for t in exported.values()))

    def test_current_prefix_selection_and_last_query_only(self):
        norm = Norm(); core = self.core(); h = torch.randn(5, 4, 6).half()
        with torch.no_grad(): core.selection_weight.copy_(torch.tensor([.2, -.1, .1, -.15]))
        with LearnedSelectionNative(norm, core, n_local_rows=4, selection_mode='clip', capture=True) as hook:
            first = norm(h); gates0 = hook.last_capture['gates'].clone()
            current = h[:, -1:].clone(); current[0] *= -1
            cached = norm(current); gates1 = hook.last_capture['gates'].clone()
        self.assertFalse(torch.equal(gates0, gates1)); self.assertEqual(hook.calls, 2)
        full = h.clone(); full[:, -1:] = current
        with LearnedSelectionNative(norm, core, n_local_rows=4, selection_mode='clip'):
            replay = norm(full)
        self.assertTrue(torch.equal(replay[:, -1:], cached))
        self.assertTrue(torch.equal(first[:-1], h[:-1]))
        self.assertTrue(torch.equal(replay[:, :-1], full[:, :-1]))

    def test_lifecycle_mode_guards_overflow_and_input_boundary(self):
        norm = Norm(); core = self.core(); h = torch.randn(5, 4, 6).half()
        hook = LearnedSelectionNative(norm, core, n_local_rows=4, selection_mode='clip')
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with hook:
                with self.assertRaisesRegex(ValueError, 'Nested'):
                    LearnedSelectionNative(norm, core, n_local_rows=4, selection_mode='clip').__enter__()
                norm(h); raise RuntimeError('synthetic')
        self.assertFalse(hook.active); self.assertEqual(len(norm._forward_pre_hooks), 0)
        self.assertNotIn(norm, LearnedSelectionNative._owners)
        core.mode = 'sigmoid'; hook.selection_mode = 'sigmoid'
        with self.assertRaisesRegex(ValueError, 'mode changed'): hook.__enter__()
        core.mode = 'clip'; hook.selection_mode = 'clip'
        with hook: norm(h)
        with torch.no_grad(): core.up.weight.fill_(1e9)
        with self.assertRaisesRegex(ValueError, 'finite cast'):
            with hook: norm(h)
        self.assertFalse(hook.active)
        with self.assertRaisesRegex(ValueError, 'labels/roles/answers'):
            prepare_scene(None, dict(sid='x', n_frames=1, question='Original question', image_files=[], answer='Mary'))
        with self.assertRaisesRegex(ValueError, 'valid_mask'):
            self.core()(self.local, self.global_states, valid_mask=torch.ones(4, 3))


if __name__ == '__main__': unittest.main()
