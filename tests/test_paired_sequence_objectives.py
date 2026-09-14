"""CPU Slurm tests for arbitrary-length paired native training objectives."""
from pathlib import Path
import os
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gnnformer.paired_sequence_objectives import sequence_layout, pack_feature_sequences, sequence_objectives
from gnnformer.aggregation_consistency import paired_residual_consistency
from gnnformer.aggregation_path_bound import paired_native_path_bound


class SequenceObjectives(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def fixture(self, sequences):
        layout = sequence_layout(sequences)
        count = len(layout['targets'])
        delta = torch.randn(count, 4, requires_grad=True)
        z = torch.randn(count, 3, requires_grad=True)
        w = torch.randn(3, 3, requires_grad=True)
        u = torch.randn(4, 3, requires_grad=True)
        g = torch.randn(count, 4).half()
        g[layout['right']] = g[layout['left']]
        logits = torch.randn(count, 16, requires_grad=True)
        return (logits, delta, z, w, u, g, layout)

    def test_two_position_value_and_gradient_compatibility(self):
        args = self.fixture([[1, 15], [1, 15], [2, 15], [2, 15]])
        logits, delta, z, w, u, g, layout = args
        result = sequence_objectives(*args)
        old = dict(ce=torch.nn.functional.cross_entropy(logits, torch.tensor(layout['targets'])),
            residual=paired_residual_consistency(delta.reshape(4, 2, 4), g.reshape(4, 2, 4)),
            path=paired_native_path_bound(z.reshape(4, 2, 3), w, u, g.reshape(4, 2, 4)))
        for key in old:
            torch.testing.assert_close(result[key], old[key])
            a = torch.autograd.grad(result[key], (logits, delta, z, w, u), retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(old[key], (logits, delta, z, w, u), retain_graph=True, allow_unused=True)
            for x, y in zip(a, b):
                if x is None or y is None:
                    self.assertIs(x, y)
                else:
                    torch.testing.assert_close(x, y)

    def test_ragged_prefixes_and_sequence_weights(self):
        layout = sequence_layout([[1, 15], [1, 15], [1, 0, 15], [1, 0, 15]])
        self.assertEqual(layout['offsets'], [0, 2, 4, 7, 10])
        self.assertEqual(layout['left'], [0, 1, 4, 5, 6])
        self.assertEqual(layout['right'], [2, 3, 7, 8, 9])
        self.assertEqual(layout['prefixes'], [[[], [1]], [[], [1]], [[], [1], [1, 0]], [[], [1], [1, 0]]])
        for a, b in zip(layout['offsets'], layout['offsets'][1:]):
            self.assertAlmostEqual(sum(layout['scene_weights'][a:b]), .25)
        args = self.fixture(layout['target_sequences']); result = sequence_objectives(*args)
        positions = result['ce_positions']; expected = sum(positions[a:b].mean() for a,b in zip(layout['offsets'], layout['offsets'][1:])) / 4
        torch.testing.assert_close(result['ce'], expected)
        for key in ('residual', 'path'):
            values = result[key + '_positions']
            torch.testing.assert_close(result[key], (values[:2].mean() + values[2:].mean()) / 2)

    def test_pad_only_items_and_preserve_causal_order(self):
        local = [torch.ones(1, 2, 4).half(), 2 * torch.ones(3, 3, 4).half()]
        global_states = [3 * torch.ones(2, 4).half(), 4 * torch.ones(3, 4).half()]
        h, g = pack_feature_sequences(local, global_states)
        self.assertEqual(tuple(h.shape), (3, 5, 4)); self.assertEqual(tuple(g.shape), (5, 4))
        self.assertTrue(torch.equal(h[0, :2], local[0][0]))
        self.assertTrue((h[1:, :2] == 0).all()); self.assertTrue((h[:, 2:] == 2).all())
        self.assertTrue(torch.equal(g, torch.cat(global_states)))

    def test_zero_output_path_gradient_is_finite_and_zero(self):
        args = list(self.fixture([[1, 15], [1, 15], [1, 0, 15], [1, 0, 15]]))
        args[4] = torch.zeros_like(args[4], requires_grad=True)
        result = sequence_objectives(*args)
        self.assertEqual(float(result['path']), 0.)
        for grad in torch.autograd.grad(result['path'], (args[2], args[3], args[4])):
            self.assertTrue(torch.isfinite(grad).all() and (grad == 0).all())

    def test_same_scene_pair_has_exact_zero_penalties(self):
        args = list(self.fixture([[1, 0, 15], [1, 0, 15]])); layout = args[-1]
        args[1] = args[1].detach(); args[2] = args[2].detach()
        for value in (args[1], args[2]): value[layout['right']] = value[layout['left']]
        result = sequence_objectives(*args)
        self.assertEqual(float(result['residual']), 0.); self.assertEqual(float(result['path']), 0.)

    def test_reject_mismatched_prefix_layout_or_query(self):
        for seq in ([], [[1]], [[1], [1, 2]], [[True], [True]]):
            with self.assertRaises(ValueError): sequence_layout(seq)
        args = list(self.fixture([[1, 15], [1, 15]])); args[-1]['left'].reverse()
        with self.assertRaises(ValueError): sequence_objectives(*args)
        args = list(self.fixture([[1, 15], [1, 15]])); args[-2][2, 0] += 1
        with self.assertRaises(ValueError): sequence_objectives(*args)


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
    torch.set_num_threads(2)
    unittest.main()
