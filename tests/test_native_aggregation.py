"""CPU correctness checks for the native per-token aggregation pilot.

Run on an allocated CPU node: .venv/bin/python tests/test_native_aggregation.py

All models are tiny, randomly initialized, and constructed without downloads.
These tests establish mathematical, causal, and KV-cache correctness; they do
not establish benchmark accuracy or research claims.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.native_aggregation import (  # noqa: E402
    NativeAggregation,
    attach_native_aggregation,
    blockwise_attention,
    reconstruct_global_attention,
)


def _repeat_kv(x: torch.Tensor, heads: int) -> torch.Tensor:
    return x.repeat_interleave(heads // x.shape[1], dim=1)


def _causal_mask(q: torch.Tensor, k: torch.Tensor, positions=None) -> torch.Tensor:
    if positions is None:
        positions = torch.arange(k.shape[-2] - q.shape[-2], k.shape[-2])
    visible = torch.arange(k.shape[-2])[None, :] <= positions[:, None]
    return visible[None, None].expand(q.shape[0], q.shape[1], -1, -1).clone()


def _reference(q, k, v, mask, scaling=None):
    return F.scaled_dot_product_attention(
        q,
        _repeat_kv(k, q.shape[1]),
        _repeat_kv(v, q.shape[1]),
        attn_mask=mask,
        dropout_p=0.0,
        scale=scaling,
    )


class BlockwiseAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)

    def _check(self, q, k, v, *, block_size, attention_mask=None,
               query_positions=None, reference_mask=None, scaling=None):
        reads, log_z, valid = blockwise_attention(
            q, k, v, block_size=block_size, attention_mask=attention_mask,
            query_positions=query_positions, scaling=scaling,
        )
        shape = (*q.shape[:3], (k.shape[-2] + block_size - 1) // block_size)
        self.assertEqual(tuple(reads.shape), (*shape, q.shape[-1]))
        self.assertEqual(tuple(log_z.shape), shape)
        self.assertEqual(tuple(valid.shape), shape)
        self.assertEqual(valid.dtype, torch.bool)
        self.assertTrue(torch.isfinite(reads).all())
        self.assertTrue(torch.isfinite(log_z[valid]).all())
        self.assertTrue(torch.isneginf(log_z[~valid]).all())
        self.assertEqual(torch.count_nonzero(reads[~valid]).item(), 0)
        if reference_mask is None:
            reference_mask = _causal_mask(q, k, query_positions)
        expected = _reference(q, k, v, reference_mask, scaling)
        actual = reconstruct_global_attention(reads, log_z)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        return reads, log_z, valid

    def test_global_sdpa_reconstruction_gqa_and_partial_blocks(self):
        q = torch.randn(2, 4, 11, 8)
        k, v = torch.randn(2, 2, 11, 8), torch.randn(2, 2, 11, 8)
        for size in (1, 4, 11, 32):
            with self.subTest(block_size=size):
                self._check(q, k, v, block_size=size)

    def test_cached_queries_use_absolute_cache_positions(self):
        q = torch.randn(2, 4, 3, 8)
        k, v = torch.randn(2, 2, 11, 8), torch.randn(2, 2, 11, 8)
        # Non-contiguous positions also catch a silent reliance on query length.
        positions = torch.tensor([3, 7, 10])
        self._check(q, k, v, block_size=4, query_positions=positions)

    def test_padding_masks_and_fully_masked_query_rows(self):
        q = torch.randn(2, 4, 9, 8)
        k, v = torch.randn(2, 2, 9, 8), torch.randn(2, 2, 9, 8)
        padding = torch.ones(2, 9, dtype=torch.bool)
        padding[0, :3] = False
        padding[1, -2:] = False
        reference = _causal_mask(q, k) & padding[:, None, None, :]
        for mask in (padding, padding.long()):
            with self.subTest(dtype=mask.dtype):
                self._check(q, k, v, block_size=4, attention_mask=mask,
                            reference_mask=reference)

    def test_head_specific_boolean_masks(self):
        q = torch.randn(2, 4, 7, 8)
        k, v = torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
        allowed = _causal_mask(q, k)
        allowed[:, 0, :, :4] = False
        allowed[:, 1, 4:, 4:] = False
        self._check(q, k, v, block_size=4, attention_mask=allowed,
                    reference_mask=allowed)

    def test_additive_mask_preserves_finite_attention_bias(self):
        q = torch.randn(2, 4, 7, 8)
        k, v = torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
        allowed = _causal_mask(q, k)
        allowed[0, :, :, 0] = False
        bias = torch.randn(2, 4, 7, 7) * 0.4
        for sentinel in (-float("inf"), torch.finfo(q.dtype).min):
            with self.subTest(sentinel=sentinel):
                mask = bias.masked_fill(~allowed, sentinel)
                reference = bias.masked_fill(~allowed, -float("inf"))
                self._check(q, k, v, block_size=3, attention_mask=mask,
                            reference_mask=reference, scaling=0.7)

    def test_future_blocks_are_zero_and_cannot_change_past_reads(self):
        q = torch.randn(1, 4, 9, 8)
        k, v = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
        reads, log_z, valid = self._check(q, k, v, block_size=4)
        self.assertFalse(valid[:, :, :4, 1:].any())
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 5:] = torch.randn_like(changed_k[:, :, 5:]) * 30
        changed_v[:, :, 5:] = torch.randn_like(changed_v[:, :, 5:]) * 30
        other_reads, other_log_z, _ = blockwise_attention(
            q, changed_k, changed_v, block_size=4,
        )
        torch.testing.assert_close(reads[:, :, :5], other_reads[:, :, :5])
        torch.testing.assert_close(log_z[:, :, :5], other_log_z[:, :, :5])

    def test_fully_masked_rows_have_finite_backward(self):
        q = torch.randn(1, 4, 6, 8, requires_grad=True)
        k = torch.randn(1, 2, 6, 8, requires_grad=True)
        v = torch.randn(1, 2, 6, 8, requires_grad=True)
        allowed = _causal_mask(q, k)
        allowed[:, :, :2] = False
        reads, log_z, _ = blockwise_attention(
            q, k, v, attention_mask=allowed, block_size=4,
        )
        merged = reconstruct_global_attention(reads, log_z)
        self.assertEqual(torch.count_nonzero(merged[:, :, :2]).item(), 0)
        merged.square().sum().backward()
        for tensor in (q, k, v):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)


_FAMILIES = ("qwen2", "qwen3", "qwen2_5_vl_text")


def _tiny_model(family):
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    torch.manual_seed(202)
    kwargs = dict(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    if family == "qwen2":
        config = Qwen2Config(**kwargs)
        model_class = Qwen2ForCausalLM
    elif family == "qwen3":
        config = Qwen3Config(**kwargs, head_dim=8)
        model_class = Qwen3ForCausalLM
    else:
        config = Qwen2_5_VLTextConfig(
            **kwargs, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        )
        model_class = Qwen2_5_VLTextModel
    config._attn_implementation = "sdpa"
    return model_class(config).eval()


def _positions(family, batch, start, length):
    positions = torch.arange(start, start + length)
    if family == "qwen2_5_vl_text":
        # Distinct axes exercise actual M-RoPE, not just text-equivalent RoPE.
        return torch.stack((positions, positions // 2, positions // 3))[:, None].expand(-1, batch, -1)
    return positions[None].expand(batch, -1)


def _forward(model, family, ids, *, start=0, mask=None, cache=None, use_cache=False):
    return model(
        input_ids=ids, attention_mask=mask,
        position_ids=_positions(family, ids.shape[0], start, ids.shape[1]),
        cache_position=torch.arange(start, start + ids.shape[1]),
        past_key_values=cache, use_cache=use_cache, return_dict=True,
    )


def _output(result):
    return result.logits if hasattr(result, "logits") else result.last_hidden_state


def _attach(model, **kwargs):
    return attach_native_aggregation(
        model, layer_index=1, block_size=4, rank=8,
        query_chunk_size=3, **kwargs,
    )


def _enable_nonzero(module):
    with torch.no_grad():
        torch.manual_seed(303)
        module.up.weight.normal_(mean=0.0, std=0.05)


class DecoderIntegrationTests(unittest.TestCase):
    def test_centered_zero_reads_are_exact_noops(self):
        torch.manual_seed(505)
        hidden = torch.randn(1, 7, 32)
        query = torch.randn(1, 4, 7, 8)
        key = torch.randn(1, 2, 7, 8)
        inputs = (hidden, query, key, torch.zeros_like(key), None,
                  torch.arange(7), 8 ** -0.5)
        for nonlinear in (False, True):
            for merge in ("sum", "mean"):
                with self.subTest(nonlinear=nonlinear, merge=merge):
                    module = NativeAggregation(
                        32, 32, block_size=3, rank=8, query_chunk_size=2,
                        nonlinear=nonlinear, merge=merge, center_messages=True,
                    )
                    _enable_nonzero(module)
                    delta = module(*inputs)
                    self.assertTrue(torch.isfinite(delta).all())
                    self.assertEqual(torch.count_nonzero(delta).item(), 0)
                    module.center_messages = False
                    self.assertGreater(module(*inputs).abs().sum().item(), 0)

    def test_centered_native_gradients_and_cached_decode(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])
        model = _tiny_model("qwen2")
        model.requires_grad_(False)
        with torch.no_grad():
            baseline = _output(_forward(model, "qwen2", ids))
        module = _attach(model, center_messages=True)
        result = _output(_forward(model, "qwen2", ids))
        torch.testing.assert_close(result, baseline, atol=2e-6, rtol=2e-5)
        loss = F.cross_entropy(result[:, :-1].reshape(-1, 97), ids[:, 1:].reshape(-1))
        loss.backward()
        self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
        for parameter in module.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        _enable_nonzero(module)
        with torch.no_grad():
            full = _output(_forward(model, "qwen2", ids))
            self.assertGreater((full - baseline).abs().max().item(), 1e-5)
            first = _forward(model, "qwen2", ids[:, :5], use_cache=True)
            cache, pieces = first.past_key_values, [_output(first)]
            for pos in range(5, ids.shape[1]):
                result = _forward(
                    model, "qwen2", ids[:, pos:pos + 1], start=pos,
                    cache=cache, use_cache=True,
                )
                cache = result.past_key_values
                self.assertEqual(cache.get_seq_length(), pos + 1)
                pieces.append(_output(result))
            torch.testing.assert_close(torch.cat(pieces, dim=1), full,
                                       atol=3e-6, rtol=3e-5)

    def test_zero_initialization_preserves_backbone_output(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                baseline = _output(_forward(model, family, ids))
                module = _attach(model)
                self.assertEqual(torch.count_nonzero(module.up.weight).item(), 0)
                actual = _output(_forward(model, family, ids))
                torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-5)

    def test_native_loss_reaches_zero_initialized_output_projection(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])
        for family in _FAMILIES:
            with self.subTest(family=family):
                model = _tiny_model(family)
                model.requires_grad_(False)
                module = _attach(model)
                result = _output(_forward(model, family, ids))
                if family == "qwen2_5_vl_text":
                    # The raw language trunk has no LM head: attach a fixed
                    # vocabulary projection only to test its gradient path.
                    generator = torch.Generator().manual_seed(404)
                    readout = torch.randn(97, result.shape[-1], generator=generator)
                    result = F.linear(result, readout)
                loss = F.cross_entropy(result[:, :-1].reshape(-1, 97), ids[:, 1:].reshape(-1))
                loss.backward()
                self.assertIsNotNone(module.up.weight.grad)
                self.assertTrue(torch.isfinite(module.up.weight.grad).all())
                self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
                frozen = [p for p in model.parameters() if not p.requires_grad]
                self.assertTrue(frozen)
                self.assertTrue(all(p.grad is None for p in frozen))

    def test_enabled_full_prefill_matches_token_and_chunk_cached_decode(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18, 5, 19]])
        for family in _FAMILIES:
            for schedule in ((5, 1, 1, 1, 1, 1, 1), (3, 2, 4, 2)):
                with self.subTest(family=family, schedule=schedule), torch.no_grad():
                    model = _tiny_model(family)
                    baseline = _output(_forward(model, family, ids))
                    module = _attach(model)
                    _enable_nonzero(module)
                    full = _output(_forward(model, family, ids))
                    self.assertGreater((full - baseline).abs().max().item(), 1e-5)
                    cache, offset, pieces = None, 0, []
                    for width in schedule:
                        result = _forward(
                            model, family, ids[:, offset:offset + width],
                            start=offset, cache=cache, use_cache=True,
                        )
                        cache = result.past_key_values
                        offset += width
                        self.assertEqual(cache.get_seq_length(), offset)
                        pieces.append(_output(result))
                    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=3e-6, rtol=3e-5)

    def test_enabled_adapter_cannot_leak_future_tokens(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])
        changed = ids.clone()
        changed[:, 5:] = torch.tensor([[33, 24, 72, 51]])
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                _enable_nonzero(_attach(model))
                original = _output(_forward(model, family, ids))
                altered = _output(_forward(model, family, changed))
                torch.testing.assert_close(original[:, :5], altered[:, :5], atol=2e-6, rtol=2e-5)

    def test_padding_content_cannot_change_visible_outputs(self):
        ids = torch.tensor([[0, 0, 14, 7, 32, 6, 13, 4, 18], [1, 9, 14, 7, 32, 6, 13, 0, 0]])
        mask = ids.ne(0)
        changed = ids.clone()
        changed[~mask] = 62
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                _enable_nonzero(_attach(model))
                original = _output(_forward(model, family, ids, mask=mask))
                altered = _output(_forward(model, family, changed, mask=mask))
                self.assertTrue(torch.isfinite(original).all())
                torch.testing.assert_close(original[mask], altered[mask], atol=3e-6, rtol=3e-5)
                cache, pieces = None, []
                for pos in range(ids.shape[1]):
                    result = _forward(
                        model, family, ids[:, pos:pos + 1], start=pos,
                        mask=mask[:, :pos + 1], cache=cache, use_cache=True,
                    )
                    cache = result.past_key_values
                    pieces.append(_output(result))
                cached = torch.cat(pieces, dim=1)
                torch.testing.assert_close(cached[mask], original[mask], atol=3e-6, rtol=3e-5)

    def test_off_and_remove_restore_original_forward(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                baseline = _output(_forward(model, family, ids))
                module = _attach(model)
                _enable_nonzero(module)
                module.mode = "off"
                disabled = _output(_forward(model, family, ids))
                torch.testing.assert_close(disabled, baseline, atol=2e-6, rtol=2e-5)
                module.mode = "all"
                module.remove()
                restored = _output(_forward(model, family, ids))
                torch.testing.assert_close(restored, baseline, atol=2e-6, rtol=2e-5)



_VARIANTS = ("local_pre", "local_post", "global", "hierarchical")


def _variant_kwargs(variant):
    return dict(variant=variant, merge="mean" if variant in ("global", "hierarchical") else "sum")


class AggregationVariantMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(606)

    def _inputs(self, masked=True):
        hidden = torch.randn(2, 7, 12, dtype=torch.float64)
        query = torch.randn(2, 2, 7, 4, dtype=torch.float64)
        key = torch.randn(2, 1, 7, 4, dtype=torch.float64)
        value = torch.randn_like(key)
        mask = _causal_mask(query, key) if masked else None
        if mask is not None:
            mask[0, :, 0] = False
            mask[0, 0, :2] = False
            mask[1, 1, :, :3] = False
        return hidden, query, key, value, mask, torch.arange(7), 0.5

    def _module(self, variant, **kwargs):
        config = dict(block_size=3, rank=5, query_chunk_size=2, head_dim=4,
                      **_variant_kwargs(variant))
        config.update(kwargs)
        module = NativeAggregation(12, 8, **config).double()
        _enable_nonzero(module)
        return module

    def test_linear_pre_post_agree_with_masks_and_both_scales(self):
        inputs = self._inputs()
        for merge in ("sum", "mean"):
            with self.subTest(merge=merge):
                pre = self._module("local_pre", merge=merge, nonlinear=False)
                post = self._module("local_post", merge=merge, nonlinear=False)
                post.load_state_dict(pre.state_dict(), strict=True)
                torch.testing.assert_close(pre(*inputs), post(*inputs), atol=2e-12, rtol=2e-12)

    def test_equal_local_reads_agree_with_nonlinearity(self):
        inputs = list(self._inputs(masked=False))
        inputs[3] = torch.ones_like(inputs[3])
        for merge in ("sum", "mean"):
            pre = self._module("local_pre", merge=merge)
            post = self._module("local_post", merge=merge)
            post.load_state_dict(pre.state_dict(), strict=True)
            torch.testing.assert_close(pre(*inputs), post(*inputs), atol=2e-12, rtol=2e-12)

    def test_heterogeneous_reads_distinguish_pre_post(self):
        hidden = torch.zeros(1, 1, 1, dtype=torch.float64)
        query = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        key = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
        value = torch.tensor([[[[-2.0], [2.0]]]], dtype=torch.float64)
        inputs = hidden, query, key, value, None, torch.tensor([1]), 1.0
        for merge in ("sum", "mean"):
            pre = NativeAggregation(1, 1, block_size=1, rank=1, merge=merge).double()
            post = NativeAggregation(1, 1, block_size=1, rank=1, merge=merge,
                                     variant="local_post").double()
            with torch.no_grad():
                pre.query_down.weight.zero_()
                pre.read_down.weight.fill_(1)
                pre.read_down.bias.zero_()
                pre.up.weight.fill_(1)
            post.load_state_dict(pre.state_dict(), strict=True)
            self.assertGreater(pre(*inputs).item(), 0.7)
            self.assertEqual(post(*inputs).item(), 0.0)

    def test_global_matches_direct_sdpa_and_initial_hierarchy(self):
        inputs = self._inputs()
        hidden, query, key, value, mask, _, scaling = inputs
        global_module = self._module("global")
        hierarchical = self._module("hierarchical")
        missing, unexpected = hierarchical.load_state_dict(global_module.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertEqual(set(missing), {"route_query.weight", "route_read.weight",
                                       "route_read.bias", "route_out.weight"})
        fused = _reference(query, key, value, mask, scaling).transpose(1, 2).flatten(-2)
        expected = global_module.up(F.silu(global_module.read_down(fused)
                                            + global_module.query_down(hidden)))
        expected = expected * mask.any(dim=-1).any(dim=1)[..., None]
        torch.testing.assert_close(global_module(*inputs), expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(hierarchical(*inputs), global_module(*inputs), atol=2e-12, rtol=2e-12)

    def test_hierarchical_weights_per_head_learnable(self):
        module = self._module("hierarchical")
        query = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
        query[:, 1] = 1.0
        reads = torch.tensor([-1., 0., 1., 2.], dtype=torch.float64)[None,None,None,:,None].expand(1,2,1,4,4).clone()
        log_z = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
        log_z[..., -1] = -torch.inf
        valid = torch.isfinite(log_z)
        initial = module.hierarchical_weights(query, reads, log_z, valid)
        expected = torch.tensor([1/3, 1/3, 1/3, 0.], dtype=torch.float64).expand_as(initial)
        torch.testing.assert_close(initial, expected)
        reward = torch.arange(4, dtype=torch.float64)
        (initial * reward).sum().backward()
        self.assertGreater(module.route_out.weight.grad.abs().sum().item(), 0)
        module.zero_grad(set_to_none=True)
        with torch.no_grad():
            module.route_query.weight.fill_(0.4)
            module.route_read.weight.fill_(0.3)
            module.route_read.bias.zero_()
            module.route_out.weight.fill_(0.3)
        learned = module.hierarchical_weights(query, reads, log_z, valid)
        torch.testing.assert_close(learned.sum(-1), torch.ones_like(learned.sum(-1)))
        self.assertEqual(torch.count_nonzero(learned[..., -1]).item(), 0)
        self.assertGreater((learned[:,0] - learned[:,1]).abs().max().item(), 1e-4)
        (learned * reward).sum().backward()
        for parameter in (module.route_query.weight, module.route_read.weight,
                          module.route_read.bias, module.route_out.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_all_variants_masked_rows_finite_backward(self):
        for variant in _VARIANTS:
            with self.subTest(variant=variant):
                inputs = list(self._inputs())
                for index in range(4):
                    inputs[index].requires_grad_(True)
                module = self._module(variant)
                if variant == "hierarchical":
                    with torch.no_grad():
                        module.route_out.weight.fill_(0.2)
                result = module(*inputs)
                self.assertEqual(torch.count_nonzero(result[0,0]).item(), 0)
                self.assertTrue(torch.isfinite(result).all())
                result.square().sum().backward()
                for tensor in inputs[:4]:
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_variant_configurations_rejected(self):
        with self.assertRaises(ValueError):
            NativeAggregation(12, 8, variant="unknown")
        for variant in ("global", "hierarchical"):
            with self.assertRaises(ValueError):
                NativeAggregation(12, 8, variant=variant, merge="sum", head_dim=4)
        with self.assertRaises(ValueError):
            NativeAggregation(12, 8, variant="hierarchical", merge="mean")
        with self.assertRaises(ValueError):
            NativeAggregation(12, 8, variant="hierarchical", merge="mean", head_dim=3)
        for variant in ("local_post", "global", "hierarchical"):
            with self.assertRaises(ValueError):
                NativeAggregation(12, 8, head_dim=4, center_messages=True, **_variant_kwargs(variant))


class AggregationVariantDecoderTests(unittest.TestCase):
    def test_all_variants_zero_init_gradients_cache_causality(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])
        changed = ids.clone()
        changed[:, 5:] = torch.tensor([[33, 24, 72, 51]])
        for family in _FAMILIES:
            for variant in _VARIANTS:
                with self.subTest(family=family, variant=variant):
                    model = _tiny_model(family)
                    model.requires_grad_(False)
                    with torch.no_grad():
                        baseline = _output(_forward(model, family, ids))
                    module = _attach(model, **_variant_kwargs(variant))
                    actual = _output(_forward(model, family, ids))
                    torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-5)
                    logits = actual
                    if family == "qwen2_5_vl_text":
                        generator = torch.Generator().manual_seed(404)
                        readout = torch.randn(97, actual.shape[-1], generator=generator)
                        logits = F.linear(actual, readout)
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1,97), ids[:,1:].reshape(-1))
                    loss.backward()
                    self.assertIsNotNone(module.up.weight.grad)
                    self.assertTrue(torch.isfinite(module.up.weight.grad).all())
                    self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
                    for parameter in module.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                    _enable_nonzero(module)
                    if variant == "hierarchical":
                        with torch.no_grad():
                            module.route_out.weight.normal_(0,0.1)
                    with torch.no_grad():
                        full = _output(_forward(model, family, ids))
                        self.assertGreater((full-baseline).abs().max().item(), 1e-5)
                        altered = _output(_forward(model, family, changed))
                        torch.testing.assert_close(full[:,:5], altered[:,:5], atol=3e-6, rtol=3e-5)
                        for schedule in ((4,1,1,1,1,1), (3,2,4)):
                            cache, offset, pieces = None, 0, []
                            for width in schedule:
                                result = _forward(model, family, ids[:,offset:offset+width],
                                                  start=offset, cache=cache, use_cache=True)
                                cache = result.past_key_values
                                offset += width
                                self.assertEqual(cache.get_seq_length(), offset)
                                pieces.append(_output(result))
                            torch.testing.assert_close(torch.cat(pieces,dim=1), full, atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
