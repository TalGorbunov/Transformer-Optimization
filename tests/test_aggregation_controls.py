"""Correctness checks for V1 attribution controls; run only on CPU Slurm.

Uses tiny random Qwen2/Qwen3/Qwen2.5-VL text trunks without downloads. These
checks establish intervention/cache/state semantics, not research efficacy.
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
for directory in (_REPO, _REPO / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gnnformer.aggregation_controls import (  # noqa: E402
    HiddenOnlyAdapter,
    attach_hidden_only_adapter,
    attach_middle_and_upper_lora,
)
from gnnformer.carriers import attach_lora  # noqa: E402
from gnnformer.runtime import get_layers  # noqa: E402
from test_native_aggregation import (  # noqa: E402
    _FAMILIES, _forward, _output, _tiny_model,
)


_IDS = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])


def _layers(model):
    return model.layers if hasattr(model, "layers") else get_layers(model)


def _loss(output, family):
    if family == "qwen2_5_vl_text":
        # A fixed vocabulary projection is used only for the bare text trunk's
        # gradient test; real VLM runs retain their original vocabulary head.
        generator = torch.Generator().manual_seed(404)
        output = F.linear(output, torch.randn(97, output.shape[-1], generator=generator))
    return F.cross_entropy(output[:, :-1].reshape(-1, 97), _IDS[:, 1:].reshape(-1))


def _activate_hidden(module):
    with torch.no_grad():
        torch.manual_seed(707)
        module.up.weight.normal_(0, 0.05)


def _activate_lora(group):
    with torch.no_grad():
        torch.manual_seed(808)
        for _, matrix_b in group.params.values():
            matrix_b.normal_(0, 0.03)


class HiddenAdapterTests(unittest.TestCase):
    def test_parameter_budget_and_zero_initialization(self):
        module = HiddenOnlyAdapter(3584, rank=96)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 688224)
        self.assertEqual(torch.count_nonzero(module.up.weight).item(), 0)
        self.assertEqual(set(module.state_dict()), {"down.weight", "down.bias", "up.weight"})
        for hidden_size, rank in ((0, 8), (32, 0)):
            with self.assertRaises(ValueError):
                HiddenOnlyAdapter(hidden_size, rank=rank)

    def test_zero_init_native_gradient_and_state_round_trip(self):
        for family in _FAMILIES:
            with self.subTest(family=family):
                model = _tiny_model(family)
                model.requires_grad_(False)
                with torch.no_grad():
                    baseline = _output(_forward(model, family, _IDS))
                module = attach_hidden_only_adapter(model, 1, rank=12)
                actual = _output(_forward(model, family, _IDS))
                torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-5)
                _loss(actual, family).backward()
                self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(torch.count_nonzero(module.down.weight.grad).item(), 0)
                self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                _activate_hidden(module)
                with torch.no_grad():
                    expected = _output(_forward(model, family, _IDS))
                    self.assertGreater((expected - baseline).abs().max().item(), 1e-5)
                buffer = io.BytesIO()
                torch.save(module.state_dict(), buffer)
                with torch.no_grad():
                    module.up.weight.zero_()
                buffer.seek(0)
                module.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(
                        _output(_forward(model, family, _IDS)), expected,
                        atol=2e-6, rtol=2e-5,
                    )

    def test_exact_normalized_input_and_attention_output_insertion_site(self):
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                attn = _layers(model)[1].self_attn
                captured = {}

                def before(_module, args, kwargs):
                    captured["hidden"] = (args[0] if args else kwargs["hidden_states"]).clone()

                def after(_module, _args, output):
                    captured["attention"] = output[0].clone()

                pre_handle = attn.register_forward_pre_hook(before, with_kwargs=True)
                post_handle = attn.register_forward_hook(after)
                _forward(model, family, _IDS)
                hidden, ordinary = captured["hidden"], captured["attention"]
                module = attach_hidden_only_adapter(model, 1, rank=12)
                _activate_hidden(module)
                _forward(model, family, _IDS)
                torch.testing.assert_close(captured["hidden"], hidden)
                torch.testing.assert_close(
                    captured["attention"], ordinary + module(hidden),
                    atol=2e-6, rtol=2e-5,
                )
                pre_handle.remove()
                post_handle.remove()

    def test_cached_decode_causality_padding_and_removal(self):
        changed = _IDS.clone()
        changed[:, 5:] = torch.tensor([[33, 24, 72, 51]])
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                baseline = _output(_forward(model, family, _IDS))
                module = attach_hidden_only_adapter(model, 1, rank=12)
                _activate_hidden(module)
                full = _output(_forward(model, family, _IDS))
                altered = _output(_forward(model, family, changed))
                torch.testing.assert_close(full[:, :5], altered[:, :5], atol=3e-6, rtol=3e-5)
                for schedule in ((4, 1, 1, 1, 1, 1), (3, 2, 4)):
                    cache, offset, pieces = None, 0, []
                    for width in schedule:
                        result = _forward(
                            model, family, _IDS[:, offset:offset + width],
                            start=offset, cache=cache, use_cache=True,
                        )
                        offset += width
                        cache = result.past_key_values
                        self.assertEqual(cache.get_seq_length(), offset)
                        pieces.append(_output(result))
                    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=3e-6, rtol=3e-5)
                padding = torch.ones_like(_IDS, dtype=torch.bool)
                padding[:, :2] = False
                padded = _IDS.clone()
                padded[:, :2] = 62
                original = _output(_forward(model, family, _IDS, mask=padding))
                altered = _output(_forward(model, family, padded, mask=padding))
                torch.testing.assert_close(original[padding], altered[padding], atol=3e-6, rtol=3e-5)
                module.mode = "off"
                torch.testing.assert_close(_output(_forward(model, family, _IDS)), baseline)
                module.mode = "all"
                module.remove()
                module.remove()
                self.assertFalse(hasattr(_layers(model)[1].self_attn, "hidden_only_adapter"))
                torch.testing.assert_close(_output(_forward(model, family, _IDS)), baseline)

    def test_modes_use_cache_history_including_one_token_prefill(self):
        model = _tiny_model("qwen2")
        module = attach_hidden_only_adapter(model, 1, rank=12)
        _activate_hidden(module)
        calls = []
        handle = module.register_forward_hook(lambda _m, _a, _o: calls.append(True))
        with torch.no_grad():
            for mode in ("prefill", "decode"):
                module.mode = mode
                calls.clear()
                first = _forward(model, "qwen2", _IDS[:, :1], use_cache=True)
                self.assertEqual(len(calls), int(mode == "prefill"))
                second = _forward(
                    model, "qwen2", _IDS[:, 1:2], start=1,
                    cache=first.past_key_values, use_cache=True,
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(second.past_key_values.get_seq_length(), 2)
        handle.remove()
        module.mode = "invalid"
        with self.assertRaises(ValueError):
            _forward(model, "qwen2", _IDS)

    def test_duplicate_attachment_and_invalid_index_are_rejected(self):
        model = _tiny_model("qwen2")
        with self.assertRaises(ValueError):
            attach_hidden_only_adapter(model, -1)
        attach_hidden_only_adapter(model, 1)
        with self.assertRaises(ValueError):
            attach_hidden_only_adapter(model, 1)


class PlacementLoraTests(unittest.TestCase):
    def _attach(self, model):
        return attach_middle_and_upper_lora(
            _layers(model), middle_layer_index=1, upper_layers=1,
            middle_rank=4, middle_alpha=8, upper_rank=2, upper_alpha=4,
            device="cpu",
        )

    def test_actual_indices_parameter_counts_and_matched_upper_initialization(self):
        model = _tiny_model("qwen2")
        layers = _layers(model)
        torch.manual_seed(909)
        reference = attach_lora(layers, 2, rank=2, alpha=4, device="cpu")
        expected = {key: tuple(p.detach().clone() for p in pair)
                    for key, pair in reference.params.items()}
        reference.remove()
        torch.manual_seed(909)
        combined = self._attach(model)
        self.assertEqual({index for index, _ in combined.params}, {1, 2})
        self.assertEqual(len(combined.params), 8)
        self.assertEqual(combined.num_parameters(), 1344)
        self.assertEqual(len({id(p) for p in combined.parameters()}), len(combined.parameters()))
        for key, pair in expected.items():
            for actual, target in zip(combined.params[key], pair):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
        for index, name in combined.params:
            self.assertEqual(combined.params[index, name][0].shape[0], 4 if index == 1 else 2)
        self.assertEqual(set(combined.state()), {f"{index}.{name}" for index, name in combined.params})

    def test_native_gradient_cache_and_flat_state_round_trip(self):
        for family in _FAMILIES:
            with self.subTest(family=family):
                model = _tiny_model(family)
                model.requires_grad_(False)
                with torch.no_grad():
                    baseline = _output(_forward(model, family, _IDS))
                group = self._attach(model)
                actual = _output(_forward(model, family, _IDS))
                torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-5)
                _loss(actual, family).backward()
                total_b_gradient = 0.0
                for matrix_a, matrix_b in group.params.values():
                    for parameter in (matrix_a, matrix_b):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    total_b_gradient += matrix_b.grad.abs().sum().item()
                self.assertGreater(total_b_gradient, 0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                _activate_lora(group)
                with torch.no_grad():
                    full = _output(_forward(model, family, _IDS))
                    self.assertGreater((full - baseline).abs().max().item(), 1e-5)
                    cache, pieces = None, []
                    for position in range(_IDS.shape[1]):
                        result = _forward(
                            model, family, _IDS[:, position:position + 1],
                            start=position, cache=cache, use_cache=True,
                        )
                        cache = result.past_key_values
                        self.assertEqual(cache.get_seq_length(), position + 1)
                        pieces.append(_output(result))
                    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=3e-6, rtol=3e-5)
                buffer = io.BytesIO()
                torch.save(group.state(), buffer)
                with torch.no_grad():
                    for matrix_a, matrix_b in group.params.values():
                        matrix_a.zero_()
                        matrix_b.zero_()
                buffer.seek(0)
                state = torch.load(buffer, weights_only=True)
                with torch.no_grad():
                    for (index, name), pair in group.params.items():
                        for parameter, saved in zip(pair, state[f"{index}.{name}"]):
                            parameter.copy_(saved)
                    torch.testing.assert_close(_output(_forward(model, family, _IDS)), full)
                    group.remove()
                    group.remove()
                    torch.testing.assert_close(_output(_forward(model, family, _IDS)), baseline)

    def test_only_selected_layer_projections_receive_lora(self):
        model = _tiny_model("qwen2")
        layers = _layers(model)
        inputs = torch.randn(1, 4, 32)
        with torch.no_grad():
            before = [{name: getattr(layer.self_attn, name)(inputs).clone()
                       for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
                      for layer in layers]
            group = self._attach(model)
            _activate_lora(group)
            for index, layer in enumerate(layers):
                for name, original in before[index].items():
                    output = getattr(layer.self_attn, name)(inputs)
                    if index == 0:
                        torch.testing.assert_close(output, original, atol=0, rtol=0)
                    else:
                        self.assertGreater((output - original).abs().max().item(), 1e-6)

    def test_overlapping_or_invalid_placements_are_rejected_before_hooks(self):
        model = _tiny_model("qwen2")
        layers = _layers(model)
        for index in (-1, 2, 3):
            with self.subTest(index=index), self.assertRaises(ValueError):
                attach_middle_and_upper_lora(
                    layers, middle_layer_index=index, upper_layers=1, device="cpu",
                )
        self.assertTrue(all(not getattr(layer.self_attn, name)._forward_hooks
                            for layer in layers
                            for name in ("q_proj", "k_proj", "v_proj", "o_proj")))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
