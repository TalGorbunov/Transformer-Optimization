"""CPU Slurm correctness gate for diagnostic-only final-query interchanges."""
from pathlib import Path
import sys
import unittest
import torch
REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from gnnformer.native_aggregation import attach_native_aggregation
from scripts.probe_native_vision_channels import FinalQueryChannels, match_recipient_norm
from test_native_aggregation import _FAMILIES, _forward, _output, _tiny_model
IDS = torch.tensor([[1, 9, 14, 7, 32, 6, 13, 4, 18]])

def attach(model):
    branch = attach_native_aggregation(model, 1, variant="global", merge="mean",
        block_size=4, rank=8, query_chunk_size=3)
    with torch.no_grad():
        torch.manual_seed(919)
        branch.up.weight.normal_(0, .08)
    return branch

class ChannelInterchangeTests(unittest.TestCase):
    def test_identity_row_locality_and_cached_interchanges(self):
        donor_ids = IDS.clone()
        donor_ids[:, 2:5] = torch.tensor([[28, 46, 37]])
        for family in _FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                model = _tiny_model(family)
                branch = attach(model)
                with FinalQueryChannels(branch, IDS.shape[1]) as own:
                    baseline = _output(_forward(model, family, IDS))
                self.assertEqual(own.offsets, {"hidden": 9, "read": 9})
                self.assertEqual(set(own.captured), {"hidden", "read"})
                with FinalQueryChannels(branch, IDS.shape[1], own.captured):
                    identity = _output(_forward(model, family, IDS))
                torch.testing.assert_close(identity, baseline, atol=2e-6, rtol=2e-5)
                with FinalQueryChannels(branch, donor_ids.shape[1]) as donor:
                    _forward(model, family, donor_ids)
                for names in (("hidden",), ("read",), ("hidden", "read")):
                    replacement = {name: donor.captured[name] for name in names}
                    with FinalQueryChannels(branch, IDS.shape[1], replacement):
                        patched = _output(_forward(model, family, IDS))
                    torch.testing.assert_close(patched[:, :-1], baseline[:, :-1], atol=2e-6, rtol=2e-5)
                    self.assertGreater((patched[:, -1] - baseline[:, -1]).abs().max().item(), 1e-6)
                    prefix = _forward(model, family, IDS[:, :-1], use_cache=True)
                    with FinalQueryChannels(branch, 1, replacement) as cached_channels:
                        result = _forward(model, family, IDS[:, -1:], start=8,
                            cache=prefix.past_key_values, use_cache=True)
                    self.assertEqual(result.past_key_values.get_seq_length(), 9)
                    self.assertEqual(cached_channels.offsets, {"hidden": 1, "read": 1})
                    torch.testing.assert_close(_output(result)[:, -1], patched[:, -1], atol=3e-6, rtol=3e-5)
                self.assertFalse(branch.query_down._forward_pre_hooks)
                self.assertFalse(branch.read_down._forward_pre_hooks)

    def test_wrong_query_count_fails_and_removes_hooks(self):
        model = _tiny_model("qwen2")
        branch = attach(model)
        with self.assertRaises(RuntimeError), torch.no_grad():
            with FinalQueryChannels(branch, 8):
                _forward(model, "qwen2", IDS)
        self.assertFalse(branch.query_down._forward_pre_hooks)
        self.assertFalse(branch.read_down._forward_pre_hooks)
        with self.assertRaises(ValueError), torch.no_grad():
            with FinalQueryChannels(branch, 9, {"hidden": torch.zeros(1, 3)}):
                _forward(model, "qwen2", IDS)
        self.assertFalse(branch.query_down._forward_pre_hooks)
        self.assertFalse(branch.read_down._forward_pre_hooks)

    def test_norm_matching_and_zero_guard(self):
        donor, recipient = torch.randn(1, 32), torch.randn(1, 32) * 3
        matched = match_recipient_norm(donor, recipient)
        torch.testing.assert_close(matched.norm(), recipient.norm())
        torch.testing.assert_close(matched / matched.norm(), donor / donor.norm())
        for left, right in ((torch.zeros_like(donor), recipient), (donor, torch.zeros_like(recipient))):
            with self.assertRaises(ValueError):
                match_recipient_norm(left, right)

    def test_rejects_non_global_and_disabled_branch(self):
        model = _tiny_model("qwen2")
        branch = attach(model)
        branch.mode = "off"
        with self.assertRaises(ValueError):
            FinalQueryChannels(branch, 9)
        branch.mode = "all"
        branch.variant = "local_pre"
        with self.assertRaises(ValueError):
            FinalQueryChannels(branch, 9)

if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
