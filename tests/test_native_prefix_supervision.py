"""Meaningful CPU fixtures; invoked by the Slurm-only target preparation stage."""
from __future__ import annotations
import copy
import unittest
import torch
from gnnformer.native_prefix_supervision import (
    FixedPrefixSupervision, build_projection, population_normalization, reference_fp64,
    world_targets, PEOPLE, ROOMS, PAIRS, CHANNEL_NAMES, CHANNELS, HIDDEN_SIZE)


def frame(index, positions):
    return dict(step_id=index + 1, rooms={r: [p for p, x in zip(PEOPLE, positions) if x == r] for r in ROOMS})


class PrefixSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projection = build_projection()

    def test_semantics_and_prefixes(self):
        sequence = [frame(0, [ROOMS[0]] * 5), frame(1, [ROOMS[0], ROOMS[1], ROOMS[0], ROOMS[0], ROOMS[0]])]
        result = world_targets(sequence)
        self.assertEqual(len(CHANNEL_NAMES), 40)
        self.assertEqual(len(set(CHANNEL_NAMES)), 40)
        self.assertEqual(PAIRS[0], ('Daniel', 'John'))
        self.assertEqual(result['local'][0][30:], [1] * 10)
        self.assertEqual(result['local'][1][30:], [0,1,1,1,0,0,0,1,1,1])
        for t, row in enumerate(result['local']):
            for p in range(5):
                self.assertEqual(sum(row[6*p:6*p+6]), 1)
                self.assertEqual(sum(result['prefix'][t][6*p:6*p+6]), t + 1)
        self.assertEqual(result['prefix'][1], [a+b for a,b in zip(*result['local'])])

    def test_causal_targets_and_additivity(self):
        a = frame(0, [ROOMS[0]] * 5); b = frame(1, [ROOMS[0], ROOMS[1], ROOMS[0], ROOMS[0], ROOMS[0]])
        changed = frame(1, [ROOMS[0], ROOMS[2], ROOMS[0], ROOMS[0], ROOMS[0]])
        x = world_targets([a, b]); y = world_targets([a, changed])
        self.assertEqual(x['prefix'][0], y['prefix'][0])
        self.assertNotEqual(x['prefix'][1], y['prefix'][1])
        b0 = copy.deepcopy(b); b0['step_id'] = 1
        suffix = world_targets([b0])
        self.assertEqual(x['prefix'][-1], [u+v for u,v in zip(x['prefix'][0], suffix['prefix'][0])])

    def test_world_weighting_and_constant_channels(self):
        x = torch.zeros(1, 40); y = torch.full((3, 40), 2.)
        norm = population_normalization([x, y])
        self.assertTrue(torch.equal(norm['mean'], torch.ones(40)))
        self.assertTrue(torch.equal(norm['scale'], torch.ones(40)))
        self.assertEqual(norm['worlds'], 2)
        constant = population_normalization([torch.full((2, 40), 7.), torch.full((5, 40), 7.)])
        self.assertTrue(torch.equal(constant['variance_fp64'], torch.zeros(40, dtype=torch.float64)))
        self.assertTrue(torch.equal(constant['scale'], torch.ones(40)))

    def test_private_projection_and_orthonormality(self):
        rng = torch.random.get_rng_state().clone()
        actual = build_projection()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(actual, self.projection))
        self.assertTrue(torch.allclose(actual.double() @ actual.double().T, torch.eye(40, dtype=torch.float64), atol=2e-6, rtol=0))

    def test_live_gradient_and_fp64_reference(self):
        mean = torch.linspace(-1, 1, 40); scale = torch.linspace(.5, 2, 40)
        probe = FixedPrefixSupervision(self.projection, mean, scale)
        h = torch.linspace(-.2, .3, 3 * HIDDEN_SIZE).reshape(1, 3, HIDDEN_SIZE).requires_grad_()
        raw = torch.arange(120).reshape(3, 40) % 4
        loss = probe.loss(h, raw); loss.backward()
        reference = reference_fp64(h, raw, self.projection, mean, scale)
        self.assertLess(abs(float(loss) - float(reference['loss'])), 2e-5)
        expected = (2 / 120) * ((reference['prediction'] - reference['normalized_target']) @ self.projection.double())
        self.assertTrue(torch.allclose(h.grad.reshape(3, -1).double(), expected, atol=2e-6, rtol=2e-5))
        self.assertGreater(float(h.grad.norm()), 0)
        self.assertEqual(list(probe.parameters()), [])
        self.assertTrue(all(not v.requires_grad for v in probe.buffers()))
        self.assertTrue(torch.allclose(probe.decode(h).reshape(3, 40).double(), reference['decoded'], atol=2e-6, rtol=2e-5))

    def test_native_cast_autocast_and_raw_units(self):
        probe = FixedPrefixSupervision(self.projection, torch.full((40,), 2.), torch.full((40,), 3.))
        h = torch.zeros(2, HIDDEN_SIZE, dtype=torch.float16, requires_grad=True)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            prediction = probe(h); loss = probe.loss(h, torch.ones(2, 40, dtype=torch.int64))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.equal(probe.decode(h), torch.full((2, 40), 2.)))
        loss.backward(); self.assertEqual(h.grad.dtype, torch.float16)
        self.assertGreater(float(h.grad.float().norm()), 0)

    def test_invalid_world_and_rows(self):
        good = frame(0, [ROOMS[0]] * 5)
        for bad in ([], [dict(good, step_id=2)], [dict(step_id=1, rooms={})]):
            with self.assertRaises(ValueError): world_targets(bad)
        duplicate = copy.deepcopy(good); duplicate['rooms'][ROOMS[1]].append(PEOPLE[0])
        with self.assertRaises(ValueError): world_targets([duplicate])
        with self.assertRaises(ValueError): world_targets([good, frame(1, [ROOMS[0]] * 5)])
        with self.assertRaises(ValueError): world_targets([good, frame(1, [ROOMS[1], ROOMS[1], ROOMS[0], ROOMS[0], ROOMS[0]])])
        probe = FixedPrefixSupervision(self.projection, torch.zeros(40), torch.ones(40))
        for bad in (torch.zeros(0,HIDDEN_SIZE), torch.zeros(2,HIDDEN_SIZE-1), torch.full((1,HIDDEN_SIZE),float('nan'))):
            with self.assertRaises(ValueError): probe(bad)
        with self.assertRaises(ValueError): probe.loss(torch.zeros(2,HIDDEN_SIZE), torch.zeros(3,40))
        with self.assertRaises(ValueError): FixedPrefixSupervision(self.projection, torch.zeros(40), torch.zeros(40))


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrefixSupervisionTests)
    result = unittest.TestResult(); suite.run(result)
    return dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                failures=[str(item[0])+': '+item[1] for item in result.failures],
                errors=[str(item[0])+': '+item[1] for item in result.errors], skipped=result.skipped)


if __name__ == '__main__':
    import os
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('CPU Slurm-only numerical fixtures')
    unittest.main()
