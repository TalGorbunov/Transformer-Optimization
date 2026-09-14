"""CPU Slurm tests for native-width value-lifting placement controls."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gnnformer.native_aggregation import NativeAggregation
from gnnformer.value_lifting import ValueLifting, attach_value_lifting
from test_native_aggregation import _FAMILIES, _causal_mask, _forward, _output, _repeat_kv, _tiny_model


def enable(module):
    with torch.no_grad():
        torch.manual_seed(1001)
        module.up.weight.normal_(0, .05)
        module.lift_up.weight.normal_(0, .03)
        module.lift_up.bias.normal_(0, .03)


class ValueLiftingMathTests(unittest.TestCase):
    def test_parameter_count_identity_lift_and_common_initialization(self):
        module = ValueLifting(3584, 3584, 512, 128, rank=64, lift_rank=8)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 696896)
        self.assertEqual(sum(p.numel() for p in module.lift_down.parameters())
                         + sum(p.numel() for p in module.lift_up.parameters()), 8704)
        native_values = torch.randn(1, 4, 3, 128)
        torch.testing.assert_close(module.lift_values(native_values), native_values, atol=0, rtol=0)
        torch.manual_seed(123)
        global_module = NativeAggregation(32, 32, rank=8, variant="global", merge="mean")
        torch.manual_seed(123)
        lifted = ValueLifting(32, 32, 16, 8, rank=8, lift_rank=3)
        for name, expected in global_module.state_dict().items():
            torch.testing.assert_close(lifted.state_dict()[name], expected, atol=0, rtol=0)

    def test_linear_activation_before_equals_after_with_trained_lift(self):
        torch.manual_seed(111)
        query = torch.randn(2, 4, 5, 8, dtype=torch.float64)
        key, value = torch.randn(2, 2, 7, 8, dtype=torch.float64), torch.randn(2, 2, 7, 8, dtype=torch.float64)
        hidden = torch.randn(2, 5, 32, dtype=torch.float64)
        positions = torch.tensor([0, 1, 3, 4, 6])
        allowed = _causal_mask(query, key, positions)
        allowed[0, :, :2] = False
        allowed[1, 2, :, 2:] = False
        before = ValueLifting(32, 32, 16, 8, rank=8, lift_rank=3, activation="identity").double()
        after = ValueLifting(32, 32, 16, 8, rank=8, lift_rank=3,
                             placement="after", activation="identity").double()
        enable(before)
        after.load_state_dict(before.state_dict(), strict=True)
        inputs = hidden, query, key, value, allowed, positions, .7
        torch.testing.assert_close(before(*inputs), after(*inputs), atol=2e-12, rtol=2e-12)

    def test_known_nonlinear_pre_and_post_difference(self):
        query = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        key = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
        value = torch.tensor([[[[-2.], [2.]]]], dtype=torch.float64)
        before = ValueLifting(1, 1, 1, 1, rank=1, lift_rank=1).double()
        after = ValueLifting(1, 1, 1, 1, rank=1, lift_rank=1, placement="after").double()
        after.load_state_dict(before.state_dict(), strict=True)
        pre_read, _ = before.attention_read(query, key, value)
        post_read, _ = after.attention_read(query, key, value)
        torch.testing.assert_close(pre_read, F.silu(value).mean(-2, keepdim=True))
        self.assertGreater(pre_read.item(), .7)
        self.assertEqual(post_read.item(), 0.)

    def test_native_sdpa_reference_padding_head_masks_and_additive_biases(self):
        torch.manual_seed(222)
        query = torch.randn(2, 4, 5, 8, dtype=torch.float64)
        key, value = torch.randn(2, 2, 7, 8, dtype=torch.float64), torch.randn(2, 2, 7, 8, dtype=torch.float64)
        positions = torch.tensor([0, 1, 3, 4, 6])
        causal = _causal_mask(query, key, positions)
        padding = torch.ones(2, 7, dtype=torch.bool)
        padding[0, :2] = False
        heads = causal.clone()
        heads[:, 1, :, :4] = False
        bias = torch.randn(2, 1, 1, 7, dtype=torch.float64) * .2
        cases = [(padding, causal & padding[:, None, None]),
                 (heads, heads),
                 (bias, torch.where(causal, bias, -torch.inf))]
        for sentinel in (-float("inf"), torch.finfo(torch.float64).min):
            masked_bias = torch.randn(2, 4, 5, 7, dtype=torch.float64).masked_fill(~heads, sentinel)
            cases.append((masked_bias, masked_bias.masked_fill(~heads, -torch.inf)))
        originals = key.clone(), value.clone()
        for placement in ("before", "after"):
            module = ValueLifting(32, 32, 16, 8, rank=8, lift_rank=3,
                                  placement=placement, activation="identity").double()
            for mask, reference in cases:
                with self.subTest(placement=placement, mask_shape=tuple(mask.shape)):
                    actual, valid = module.attention_read(query, key, value, mask, positions, .7)
                    expected = F.scaled_dot_product_attention(query, _repeat_kv(key, 4),
                        _repeat_kv(value, 4), attn_mask=reference, dropout_p=0., scale=.7)
                    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertEqual(valid.shape, (2, 5))
        torch.testing.assert_close(key, originals[0], atol=0, rtol=0)
        torch.testing.assert_close(value, originals[1], atol=0, rtol=0)

    def test_fully_masked_rows_and_backward_are_finite(self):
        for placement in ("before", "after"):
            with self.subTest(placement=placement):
                torch.manual_seed(333)
                hidden = torch.randn(2, 5, 32, requires_grad=True)
                query = torch.randn(2, 4, 5, 8, requires_grad=True)
                key = torch.randn(2, 2, 5, 8, requires_grad=True)
                value = torch.randn(2, 2, 5, 8, requires_grad=True)
                mask = _causal_mask(query, key)
                mask[0, :, :2] = False
                module = ValueLifting(32, 32, 16, 8, rank=8, lift_rank=3, placement=placement)
                enable(module)
                result = module(hidden, query, key, value, mask, torch.arange(5), .5)
                self.assertEqual(torch.count_nonzero(result[0, :2]).item(), 0)
                result.square().sum().backward()
                for tensor in (hidden, query, key, value):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_future_values_and_keys_cannot_affect_earlier_reads(self):
        torch.manual_seed(444)
        query = torch.randn(1, 4, 9, 8)
        key, value = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
        for placement in ("before", "after"):
            with self.subTest(placement=placement):
                module = ValueLifting(32, 32, 16, 8, rank=8, lift_rank=3, placement=placement)
                enable(module)
                original, _ = module.attention_read(query, key, value)
                changed_key, changed_value = key.clone(), value.clone()
                changed_key[:, :, 5:] = torch.randn_like(changed_key[:, :, 5:]) * 20
                changed_value[:, :, 5:] = torch.randn_like(changed_value[:, :, 5:]) * 20
                changed, _ = module.attention_read(query, changed_key, changed_value)
                torch.testing.assert_close(original[:, :, :5], changed[:, :, :5], atol=2e-6, rtol=2e-5)


class ValueLiftingDecoderTests(unittest.TestCase):
    def test_zero_init_native_gradients_and_full_cached_causal_parity(self):
        ids = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18, 5, 19]])
        changed = ids.clone()
        changed[:, 5:] = torch.tensor([[33, 24, 72, 51, 30, 40]])
        for family in _FAMILIES:
            for placement in ("before", "after"):
                with self.subTest(family=family, placement=placement):
                    model = _tiny_model(family)
                    model.requires_grad_(False)
                    with torch.no_grad():
                        baseline = _output(_forward(model, family, ids))
                    module = attach_value_lifting(model, 1, rank=8, lift_rank=3, placement=placement)
                    result = _output(_forward(model, family, ids))
                    torch.testing.assert_close(result, baseline, atol=2e-6, rtol=2e-5)
                    logits = result
                    if family == "qwen2_5_vl_text":
                        generator = torch.Generator().manual_seed(505)
                        logits = F.linear(result, torch.randn(97, result.shape[-1], generator=generator))
                    F.cross_entropy(logits[:, :-1].reshape(-1, 97), ids[:, 1:].reshape(-1)).backward()
                    self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
                    for parameter in module.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                    enable(module)
                    with torch.no_grad():
                        full = _output(_forward(model, family, ids))
                        self.assertGreater((full-baseline).abs().max().item(), 1e-5)
                        altered = _output(_forward(model, family, changed))
                        torch.testing.assert_close(full[:, :5], altered[:, :5], atol=3e-6, rtol=3e-5)
                        for schedule in ((4, 1, 1, 1, 1, 1, 1, 1), (4, 3, 4)):
                            offset, cache, pieces = 0, None, []
                            for width in schedule:
                                result = _forward(model, family, ids[:, offset:offset+width],
                                    start=offset, cache=cache, use_cache=True)
                                offset += width
                                cache = result.past_key_values
                                self.assertEqual(cache.get_seq_length(), offset)
                                pieces.append(_output(result))
                            torch.testing.assert_close(torch.cat(pieces, 1), full, atol=3e-6, rtol=3e-5)
                        module.mode = "off"
                        torch.testing.assert_close(_output(_forward(model, family, ids)), baseline, atol=2e-6, rtol=2e-5)
                        module.mode = "all"
                        module.remove()
                        module.remove()
                        torch.testing.assert_close(_output(_forward(model, family, ids)), baseline, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
