"""Numerical algebra fixtures; run only in a Slurm CPU allocation."""
import os
import unittest
import torch
from gnnformer.attention_moments import summarize, merge, read_moments


class AttentionMomentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
            raise RuntimeError('Run numerical tests on CPU Slurm')
        torch.set_num_threads(2)

    def sample(self, dtype=torch.float64):
        generator = torch.Generator().manual_seed(24)
        return (torch.randn(4, 11, generator=generator, dtype=dtype),
                torch.randn(11, 7, generator=generator, dtype=dtype))

    def test_full_matches_independent_reference(self):
        scores, values = self.sample()
        mean, mass = read_moments(summarize(scores, values))
        torch.testing.assert_close(mean, scores.softmax(-1) @ values, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(mass[:, 0], scores.logsumexp(-1), atol=1e-12, rtol=1e-12)

    def test_partition_merges_and_gradients(self):
        scores, values = self.sample()
        first = [scores.clone().requires_grad_(), values.clone().requires_grad_()]
        second = [scores.clone().requires_grad_(), values.clone().requires_grad_()]
        direct = read_moments(summarize(*first))
        a = summarize(second[0][:, :3], second[1][:3])
        b = summarize(second[0][:, 3:8], second[1][3:8])
        c = summarize(second[0][:, 8:], second[1][8:])
        partitioned = read_moments(merge(merge(a, b), c))
        other_order = read_moments(merge(c, merge(b, a)))
        for target, actual, permuted in zip(direct, partitioned, other_order):
            torch.testing.assert_close(actual, target, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(permuted, target, atol=1e-12, rtol=1e-12)
        loss = lambda pair: pair[0].square().sum() + .37 * pair[1].square().sum()
        gd = torch.autograd.grad(loss(direct), first)
        gm = torch.autograd.grad(loss(partitioned), second)
        for actual, target in zip(gm, gd):
            torch.testing.assert_close(actual, target, atol=1e-11, rtol=1e-11)

    def test_replication_preserves_mean_but_changes_mass(self):
        scores, values = self.sample()
        original = read_moments(summarize(scores, values))
        repeated = read_moments(summarize(scores.repeat(1, 3), values.repeat(3, 1)))
        torch.testing.assert_close(repeated[0], original[0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(repeated[1], original[1] + torch.tensor(3.).double().log(), atol=1e-12, rtol=1e-12)

    def test_empty_chunk_identity_and_empty_read_rejection(self):
        scores, values = self.sample()
        empty = summarize(scores[:, :0], values[:0])
        full = summarize(scores, values)
        for result in (merge(empty, full), merge(full, empty)):
            for actual, target in zip(read_moments(result), read_moments(full)):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            read_moments(merge(empty, empty))

    def test_extreme_finite_scores_and_item_permutation(self):
        scores, values = self.sample(torch.float32)
        scores = scores * 1000 + torch.tensor([10000., -10000., 500., -500.])[:, None]
        direct = read_moments(summarize(scores, values))
        perm = torch.tensor([10, 0, 7, 2, 4, 6, 1, 8, 3, 9, 5])
        actual = read_moments(summarize(scores[:, perm], values[perm]))
        for a, b in zip(actual, direct):
            self.assertTrue(bool(torch.isfinite(a).all()))
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(direct[0], scores.softmax(-1) @ values, atol=1e-5, rtol=1e-6)

    def test_invalid_interfaces_rejected(self):
        scores, values = self.sample()
        for s, v in ((scores.half(), values.half()), (scores, values[:-1]), (scores, values.float())):
            with self.assertRaises(ValueError):
                summarize(s, v)
        for bad in (float('inf'), float('-inf'), float('nan')):
            changed = scores.clone(); changed[0, 0] = bad
            with self.assertRaises(ValueError):
                summarize(changed, values)
        with self.assertRaises(ValueError):
            merge(summarize(scores, values), summarize(scores[:2], values))


if __name__ == '__main__':
    unittest.main()
