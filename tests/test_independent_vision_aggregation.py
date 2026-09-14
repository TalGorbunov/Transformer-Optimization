"""Focused CPU correctness checks; no pretrained weights, downloads or efficacy.

Run only through CPU Slurm:
  .venv/bin/python tests/test_independent_vision_aggregation.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run prototype tests in a Slurm CPU allocation")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from gnnformer.independent_vision_aggregation import (
    IndependentVisionAggregation, attach_independent_vision_aggregation, image_layout,
)

TOKENS = dict(image_token_id=28, vision_start_token_id=29, vision_end_token_id=30,
              video_token_id=31, spatial_merge_size=2)
IDS = torch.tensor([[1, 29, 28, 28, 30, 2, 29, 28, 28, 30, 3]])
GRID = torch.tensor([[1, 2, 4], [1, 2, 4]])


def activate(branch):
    with torch.no_grad():
        branch.up.weight.normal_(0, .12)


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.hidden = torch.randn(1, 7, 8)
        self.memory = torch.randn(3, 8)
        self.positions = torch.arange(7)
        self.language = torch.ones(7, dtype=torch.bool)

    def test_parameter_matching_and_zero_output(self):
        a = IndependentVisionAggregation(8, rank=3, merge="sum")
        b = IndependentVisionAggregation(8, rank=3, merge="mean")
        self.assertEqual(sum(p.numel() for p in a.parameters()), 3 * 8 * 3 + 3 * 3 + 3)
        self.assertEqual({k: tuple(v.shape) for k, v in a.state_dict().items()},
                         {k: tuple(v.shape) for k, v in b.state_dict().items()})
        out = a(self.hidden, [self.memory], torch.tensor([1]), self.positions, self.language)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        out.sum().backward()
        self.assertIsNotNone(a.up.weight.grad)
        self.assertGreater(float(a.up.weight.grad.abs().sum()), 0)

    def test_zero_read_null_with_nonzero_bias_and_up(self):
        a = IndependentVisionAggregation(8, rank=3)
        activate(a)
        with torch.no_grad():
            a.read.bias.fill_(.73)
        out = a(self.hidden, [torch.zeros_like(self.memory)], torch.tensor([0]), self.positions, self.language)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_duplicate_image_sum_scales_mean_invariant(self):
        a = IndependentVisionAggregation(8, rank=3)
        activate(a)
        one = a(self.hidden, [self.memory], torch.tensor([0]), self.positions, self.language)
        two = a(self.hidden, [self.memory, self.memory.clone()], torch.tensor([0, 0]),
                self.positions, self.language)
        self.assertGreater(float(one.abs().sum()), 0)
        torch.testing.assert_close(two, 2 * one)
        a.merge = "mean"
        mean = a(self.hidden, [self.memory, self.memory.clone()], torch.tensor([0, 0]),
                 self.positions, self.language)
        torch.testing.assert_close(mean, one)

    def test_future_image_cannot_change_earlier_outputs_or_gradients(self):
        a = IndependentVisionAggregation(8, rank=3)
        activate(a)
        future = torch.randn(4, 8, requires_grad=True)
        out = a(self.hidden, [self.memory, future], torch.tensor([1, 4]), self.positions, self.language)
        changed = a(self.hidden, [self.memory, future * 13 + 9], torch.tensor([1, 4]),
                    self.positions, self.language)
        torch.testing.assert_close(out[:, :5], changed[:, :5], rtol=0, atol=0)
        self.assertTrue(torch.equal(out[:, :2], torch.zeros_like(out[:, :2])))
        out[:, :5].sum().backward()
        self.assertTrue(future.grad is None or torch.equal(future.grad, torch.zeros_like(future.grad)))

    def test_nonlanguage_positions_are_exact_zero_and_chunks_agree(self):
        a = IndependentVisionAggregation(8, rank=3, query_chunk_size=1)
        activate(a)
        language = torch.tensor([True, False, False, True, False, True, True])
        small = a(self.hidden, [self.memory], torch.tensor([0]), self.positions, language)
        a.query_chunk_size = 64
        large = a(self.hidden, [self.memory], torch.tensor([0]), self.positions, language)
        torch.testing.assert_close(small, large)
        self.assertTrue(torch.equal(small[:, ~language], torch.zeros_like(small[:, ~language])))

    def test_detached_stats_and_opt_in_final_query_vectors(self):
        a = IndependentVisionAggregation(8, rank=3)
        activate(a)
        self.assertIsNone(a.export_last_query_diagnostics())
        a.capture_last_query_messages = True
        out = a(self.hidden, [self.memory, self.memory * 2], torch.tensor([1, 4]),
                self.positions, self.language)
        self.assertEqual(a.last_call_stats["n_visible"].tolist(), [0, 0, 1, 1, 1, 2, 2])
        for tensor in a.last_call_stats.values():
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        diagnostic = a.export_last_query_diagnostics(cpu=True)
        self.assertEqual(tuple(diagnostic["frame_messages"].shape), (2, 3))
        torch.testing.assert_close(diagnostic["merged_message"], diagnostic["frame_messages"].sum(0))
        torch.testing.assert_close(diagnostic["residual"], out[0, -1].detach())
        a.capture_last_query_messages = False
        a(self.hidden, [self.memory], torch.tensor([1]), self.positions, self.language)
        self.assertIsNone(a.export_last_query_diagnostics())

    def test_layout_complete_delimiters_and_grid_contract(self):
        sizes, ends, language = image_layout(IDS, GRID, **TOKENS)
        self.assertEqual(sizes, [2, 2])
        self.assertEqual(ends.tolist(), [4, 9])
        self.assertEqual(torch.nonzero(language).flatten().tolist(), [0, 5, 10])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            image_layout(IDS[:, :9], GRID, **TOKENS)
        with self.assertRaises(ValueError):
            image_layout(IDS, torch.tensor([[1, 2, 2], [1, 2, 4]]), **TOKENS)
        with self.assertRaisesRegex(ValueError, "Videos"):
            image_layout(torch.tensor([[31]]), None, **TOKENS)


class FakeCache:
    def __init__(self):
        self.values = {}

    def get_seq_length(self, layer_idx=0):
        return 0 if layer_idx not in self.values else self.values[layer_idx][0].shape[-2]

    def update(self, key, value, layer_idx):
        if layer_idx in self.values:
            old_key, old_value = self.values[layer_idx]
            key, value = torch.cat((old_key, key), dim=-2), torch.cat((old_value, value), dim=-2)
        self.values[layer_idx] = (key, value)
        return key, value


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_merge_size = 2
        self.merger = nn.Identity()
        self.calls = 0

    def forward(self, pixels, grid_thw):
        self.calls += 1
        # The merger sees packed data. Final visual output reverses that order.
        packed = self.merger(pixels.flip(0))
        return packed.flip(0)


class FakeAttention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.sliding_window = None
        self.q_proj, self.k_proj = nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)
        self.v_proj, self.o_proj = nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)
        self.calls = 0

    def forward(self, hidden_states, past_key_values=None, cache_position=None, **kwargs):
        self.calls += 1
        q, k, v = (projection(hidden_states).unsqueeze(1) for projection in
                   (self.q_proj, self.k_proj, self.v_proj))
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        allowed = torch.arange(k.shape[-2])[None, :] <= cache_position[:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0)
        return self.o_proj(out.squeeze(1)), None


class FakeLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = FakeAttention(layer_idx)
        self.norm = nn.LayerNorm(8)

    def forward(self, hidden, cache, positions):
        return hidden + self.self_attn(self.norm(hidden), past_key_values=cache, cache_position=positions)[0]


class FakeQwen(nn.Module):
    """Small native-shaped flow with real causal attention and per-layer KV cache."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2_5_vl",
            **{k: v for k, v in TOKENS.items() if k != "spatial_merge_size"})
        self.embed = nn.Embedding(32, 8)
        self.model = nn.Module()
        self.model.visual = FakeVisual()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([FakeLayer(i) for i in range(2)])
        self.lm_head = nn.Linear(8, 19, bias=False)
        self.requires_grad_(False)

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None, past_key_values=None,
                cache_position=None, use_cache=False, attention_mask=None, **kwargs):
        cache = past_key_values if past_key_values is not None else FakeCache() if use_cache else None
        start = 0 if cache is None else cache.get_seq_length()
        positions = torch.arange(start, start + input_ids.shape[1]) if cache_position is None else cache_position
        hidden = self.embed(input_ids)
        if pixel_values is not None:
            features = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            hidden = hidden.clone()
            hidden[0, input_ids[0] == TOKENS["image_token_id"]] = features
        for layer in self.model.language_model.layers:
            hidden = layer(hidden, cache, positions)
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=cache)


class HookFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(112)
        self.model = FakeQwen()
        self.pixels = torch.randn(4, 8)

    def attach(self, merge="sum"):
        return attach_independent_vision_aggregation(self.model, layer_index=0, rank=3, merge=merge)

    def test_zero_off_and_removal_preserve_native_forward(self):
        expected = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID).logits
        branch = self.attach()
        zero = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID).logits
        torch.testing.assert_close(zero, expected, rtol=0, atol=0)
        activate(branch)
        branch.mode = "off"
        off = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID).logits
        torch.testing.assert_close(off, expected, rtol=0, atol=0)
        branch.remove()
        self.assertFalse(hasattr(self.model.model.language_model.layers[0].self_attn, "independent_vision_aggregation"))
        restored = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID).logits
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    def test_nonzero_cached_equals_full_and_vision_is_not_repeated(self):
        for merge in ("sum", "mean"):
            with self.subTest(merge=merge):
                branch = self.attach(merge)
                activate(branch)
                full_ids = torch.cat((IDS, torch.tensor([[4, 5]])), dim=1)
                full = self.model(full_ids, pixel_values=self.pixels, image_grid_thw=GRID).logits
                calls = self.model.model.visual.calls
                prefill = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID, use_cache=True)
                self.assertEqual(self.model.model.visual.calls, calls + 1)
                self.assertEqual(branch.memory_image_count, 2)
                tail = self.model(torch.tensor([[4, 5]]), past_key_values=prefill.past_key_values,
                                  image_grid_thw=GRID, use_cache=True)
                self.assertEqual(self.model.model.visual.calls, calls + 1)
                self.assertEqual(tail.past_key_values.get_seq_length(), full_ids.shape[1])
                torch.testing.assert_close(prefill.logits, full[:, :IDS.shape[1]], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(tail.logits, full[:, IDS.shape[1]:], rtol=2e-5, atol=2e-6)
                branch.remove()

    def test_final_visual_order_and_reset_next_example(self):
        branch = self.attach()
        activate(branch)
        prefill = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID, use_cache=True)
        memory = branch._controller.image_memory
        torch.testing.assert_close(torch.cat(memory), self.pixels, rtol=0, atol=0)
        self.assertFalse(any(m.requires_grad for m in memory))
        changed = self.pixels * 3 + 2
        second = self.model(IDS, pixel_values=changed, image_grid_thw=GRID, use_cache=True)
        torch.testing.assert_close(torch.cat(branch._controller.image_memory), changed, rtol=0, atol=0)
        self.assertIsNot(prefill.past_key_values, second.past_key_values)
        state = branch.state_dict()
        self.assertEqual(set(state), {"query.weight", "memory.weight", "read.weight", "read.bias", "up.weight"})
        branch.reset_memory()
        self.assertEqual(branch.memory_image_count, 0)
        with self.assertRaisesRegex(ValueError, "foreign"):
            self.model(torch.tensor([[4]]), past_key_values=second.past_key_values, use_cache=True)

    def test_receiving_attention_output_norm_is_recorded_before_addition(self):
        attn = self.model.model.language_model.layers[0].self_attn
        ordinary = []
        handle = attn.register_forward_hook(lambda module, args, output:
            ordinary.append(output[0][0, -1].detach().norm()))
        try:
            branch = self.attach()
            activate(branch)
            branch.capture_last_query_messages = True
            self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID)
            torch.testing.assert_close(branch.last_call_stats["native_output_norm"][-1], ordinary[-1])
            torch.testing.assert_close(branch.last_query_diagnostics["native_output_norm"], ordinary[-1])
            self.assertFalse(branch.last_call_stats["native_output_norm"].requires_grad)
        finally:
            handle.remove()

    def test_future_pixels_do_not_leak_before_complete_image(self):
        branch = self.attach()
        activate(branch)
        full = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID).logits
        changed = self.pixels.clone()
        changed[2:] = changed[2:] * 17 + 6
        other = self.model(IDS, pixel_values=changed, image_grid_thw=GRID).logits
        # First six positions precede every patch of image2; native flow and
        # image-memory branch must both ignore its changed pixels there.
        torch.testing.assert_close(full[:, :6], other[:, :6], rtol=0, atol=0)

    def test_text_prefill_clears_old_images_and_native_calls_once(self):
        branch = self.attach()
        image = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID, use_cache=True)
        counts = [layer.self_attn.calls for layer in self.model.model.language_model.layers]
        self.model(torch.tensor([[4]]), past_key_values=image.past_key_values, use_cache=True)
        self.assertEqual([layer.self_attn.calls for layer in self.model.model.language_model.layers],
                         [c + 1 for c in counts])
        self.model(torch.tensor([[1, 2, 3]]), use_cache=True)
        self.assertEqual(branch.memory_image_count, 0)

    def test_gradients_flow_to_memory_projection_without_vision_graph(self):
        branch = self.attach()
        activate(branch)
        out = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID)
        out.logits[:, -1].square().sum().backward()
        for name, parameter in branch.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(branch.memory.weight.grad.abs().sum()), 0)

    def test_unsupported_or_foreign_state_is_rejected_and_cleared(self):
        branch = self.attach()
        cache = self.model(IDS, pixel_values=self.pixels, image_grid_thw=GRID, use_cache=True).past_key_values
        with self.assertRaisesRegex(ValueError, "unpadded"):
            self.model(torch.tensor([[4]]), past_key_values=cache, attention_mask=torch.zeros(1, 12), use_cache=True)
        self.assertEqual(branch.memory_image_count, 0)
        with self.assertRaisesRegex(ValueError, "B1"):
            self.model(IDS.expand(2, -1), pixel_values=self.pixels, image_grid_thw=GRID)
        with self.assertRaisesRegex(ValueError, "Videos"):
            self.model(torch.tensor([[1, 2]]), video_grid_thw=torch.tensor([[1, 2, 2]]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.model(IDS[:, :9], pixel_values=self.pixels, image_grid_thw=GRID)


if __name__ == "__main__":
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    unittest.main()
