"""Native-width software fixtures, executed only by the CPU Slurm check."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torch
from gnnformer.native_visual_memory import NativeVisualMemory, position_features
from gnnformer.attention_moments import AttentionMoments

PROPOSAL = 'docs/paper/NATIVE_AGGREGATION_VISUAL_MEMORY_CORE_PROPOSAL.md'
PROPOSAL_SHA = '9e6190564ba3b4aeb764c7ecaf0c451b47bb394feccdccfdd21976c53f08a68f'
OWN = ('gnnformer/native_visual_memory.py', 'tests/test_native_visual_memory.py',
       'slurm/native_visual_memory_check.sbatch', PROPOSAL)
INHERITED = {'gnnformer/attention_moments.py': '2513044c9ea490098217fc55ac68aac5c798ce3eed7ca746e47a447a3e0fd804'}


def cpu_slurm():
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu'
            or os.environ.get('SLURM_JOB_GPUS') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4):
        raise RuntimeError('Only a four-core CPU Slurm allocation may execute these fixtures')


def sample(dtype=torch.float32):
    generator = torch.Generator().manual_seed(105)
    x = .3 * torch.randn(9, 3584, generator=generator, dtype=torch.float32)
    frame = torch.tensor([0, 0, 1, 8, 16, 31, 63, 64, 127], dtype=torch.int64)
    row = torch.arange(9, dtype=torch.int64) % 3
    col = torch.arange(9, dtype=torch.int64) % 5
    return (x.to(dtype), frame, row, col, torch.full_like(frame, 3), torch.full_like(frame, 5))


def subset(items, selection):
    return tuple(x[selection] for x in items)


def reference(core, items, retain_mass):
    x, frames, rows, cols, heights, widths = items
    # Scalar stdlib position formula and FP64 softmax, independent of the core
    # coordinate constructor and the frozen stable moment implementation.
    positions = []
    for f, r, c, h, w in zip(frames.tolist(), rows.tolist(), cols.tolist(), heights.tolist(), widths.tolist()):
        values = []
        for coordinate in (f / 128, (r + .5) / h, (c + .5) / w):
            for frequency in (1, 2, 4, 8):
                angle = 2 * math.pi * coordinate * frequency
                values.extend((math.sin(angle), math.cos(angle)))
        positions.append(values)
    values = x.double()
    normalized = values / (values.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    keys = torch.cat((normalized, torch.tensor(positions, dtype=torch.float64)), -1) @ core.key_weight.double().T
    scores = core.queries.double() @ keys.T / math.sqrt(128)
    mean = scores.softmax(-1) @ values
    log_mass = scores.logsumexp(-1)[:, None]
    return dict(tokens=mean + (log_mass if retain_mass else torch.zeros_like(log_mass)) * core.mass_direction.double(),
                normalized_value=mean, log_mass=log_mass)


class NativeVisualMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu_slurm()
        torch.set_num_threads(4)

    def test_native_parameter_count_and_private_exact_initialization(self):
        before = torch.random.get_rng_state().clone()
        core = NativeVisualMemory()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(list(core.state_dict()), ['key_weight', 'queries', 'mass_direction'])
        self.assertEqual(sum(p.numel() for p in core.parameters()), 469504)
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in core.parameters()))
        generator = torch.Generator().manual_seed(24)
        expected = torch.empty(128, 3608).uniform_(-1 / math.sqrt(3608), 1 / math.sqrt(3608), generator=generator)
        queries = torch.randn(32, 128, generator=generator)
        self.assertTrue(torch.equal(core.key_weight, expected))
        self.assertTrue(torch.equal(core.queries, queries))
        self.assertEqual(int(torch.count_nonzero(core.mass_direction)), 0)
        other = NativeVisualMemory()
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in core.state_dict().items()))

    def test_independent_fp64_formula_and_raw_identity_values(self):
        core = NativeVisualMemory()
        with torch.no_grad():
            core.mass_direction.copy_(torch.linspace(-.03, .03, 3584))
        items = sample()
        for retain in (False, True):
            actual, expected = core(*items, retain_mass=retain), reference(core, items, retain)
            self.assertEqual(actual['tokens'].shape, (32, 3584))
            for name in expected:
                torch.testing.assert_close(actual[name].double(), expected[name], rtol=3e-5, atol=3e-6)
        # Zero scores have exactly uniform weights. Values must remain raw x,
        # not the RMS-normalized key features or a hidden value projection.
        with torch.no_grad():
            core.key_weight.zero_()
        out = core(*items, retain_mass=False)
        torch.testing.assert_close(out['tokens'], items[0].mean(0).expand(32, -1), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(out['log_mass'], torch.full((32, 1), math.log(9)), rtol=1e-6, atol=1e-6)

    def test_partition_permutation_and_parameter_input_gradients(self):
        direct, chunks = NativeVisualMemory(), NativeVisualMemory()
        with torch.no_grad():
            direct.mass_direction.copy_(torch.linspace(-.02, .02, 3584))
        chunks.load_state_dict(direct.state_dict())
        items = sample()
        a = (items[0].clone().requires_grad_(), *items[1:])
        b = (items[0].clone().requires_grad_(), *items[1:])
        full = direct(*a, retain_mass=True)
        states = [chunks.summarize(*subset(b, slice(lo, hi))) for lo, hi in ((0, 2), (2, 6), (6, 9))]
        joined = chunks.read(chunks.merge(states[2], chunks.merge(states[0], states[1])), retain_mass=True)
        probe = torch.linspace(-.7, .9, 32 * 3584).reshape(32, 3584)
        def loss(out):
            return (out['tokens'] * probe).sum() / 1000 + .07 * out['log_mass'].square().mean()
        expected = torch.autograd.grad(loss(full), (a[0], *direct.parameters()))
        actual = torch.autograd.grad(loss(joined), (b[0], *chunks.parameters()))
        for x, y in zip(actual, expected):
            self.assertTrue(bool(torch.isfinite(x).all()) and float(x.norm()) > 0)
            torch.testing.assert_close(x, y, rtol=3e-4, atol=3e-5)
        for key in ('tokens', 'normalized_value', 'log_mass'):
            torch.testing.assert_close(joined[key], full[key], rtol=2e-5, atol=2e-6)
        permutation = torch.tensor([8, 2, 5, 0, 7, 4, 1, 6, 3])
        permuted = direct(*subset(items, permutation), retain_mass=True)
        torch.testing.assert_close(permuted['tokens'], full['tokens'], rtol=2e-5, atol=2e-6)

    def test_zero_initialization_parity_and_live_mass_gradient(self):
        core = NativeVisualMemory()
        items = sample()
        normalized = core(*items, retain_mass=False)
        mass = core(*items, retain_mass=True)
        self.assertTrue(torch.equal(normalized['tokens'], mass['tokens']))
        self.assertEqual(int(torch.count_nonzero(normalized['mass_feature'])), 0)
        self.assertTrue(torch.equal(mass['mass_feature'], mass['log_mass']))
        probe = torch.linspace(-.1, .9, 32 * 3584).reshape(32, 3584)
        gn = torch.autograd.grad((normalized['tokens'] * probe).sum(), tuple(core.parameters()))
        gm = torch.autograd.grad((mass['tokens'] * probe).sum(), tuple(core.parameters()))
        self.assertEqual(int(torch.count_nonzero(gn[-1])), 0)
        self.assertGreater(float(gm[-1].norm()), 0)
        for x, y in zip(gn[:-1], gm[:-1]):
            self.assertTrue(bool(torch.isfinite(x).all()) and float(x.norm()) > 0)
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_literal_replication_after_nonzero_mass_direction(self):
        core = NativeVisualMemory()
        with torch.no_grad():
            core.mass_direction.copy_(torch.linspace(-.04, .04, 3584))
        items = sample()
        duplicate = tuple(x.repeat((3, 1) if x.ndim == 2 else (3,)) for x in items)
        a = core(*items, retain_mass=True)
        b = core(*duplicate, retain_mass=True)
        torch.testing.assert_close(b['normalized_value'], a['normalized_value'], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(b['log_mass'], a['log_mass'] + math.log(3), rtol=2e-6, atol=2e-6)
        expected = a['tokens'] + math.log(3) * core.mass_direction[None, :]
        torch.testing.assert_close(b['tokens'], expected, rtol=2e-5, atol=3e-6)
        normalized = core(*duplicate, retain_mass=False)
        torch.testing.assert_close(normalized['tokens'], a['normalized_value'], rtol=2e-5, atol=2e-6)

    def test_global_coordinates_are_not_reset_at_chunk_boundaries(self):
        core = NativeVisualMemory()
        items = sample()
        with torch.no_grad():
            core.key_weight.zero_()
            core.queries.zero_()
            core.key_weight[0, 3584] = 2
            core.queries[:, 0] = math.sqrt(128)
        first, last = subset(items, slice(0, 5)), subset(items, slice(5, None))
        correct = core.read(core.merge(core.summarize(*first), core.summarize(*last)), retain_mass=False)
        changed = (last[0], last[1] - last[1][0], *last[2:])
        wrong = core.read(core.merge(core.summarize(*first), core.summarize(*changed)), retain_mass=False)
        self.assertGreater(float((correct['tokens'] - wrong['tokens']).abs().max()), 1e-4)
        positions = position_features(*items[1:])
        self.assertEqual(positions.shape, (9, 24))
        for index in range(9):
            scalar = []
            for coordinate in (int(items[1][index]) / 128,
                    (int(items[2][index]) + .5) / int(items[4][index]),
                    (int(items[3][index]) + .5) / int(items[5][index])):
                for frequency in (1, 2, 4, 8):
                    scalar.extend((math.sin(2 * math.pi * frequency * coordinate), math.cos(2 * math.pi * frequency * coordinate)))
            torch.testing.assert_close(positions[index], torch.tensor(scalar), rtol=0, atol=5e-6)

    def test_native_fp16_promotion_and_outer_autocast(self):
        core = NativeVisualMemory()
        items = sample(torch.float16)
        target = core(*items, retain_mass=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = core(*items, retain_mass=True)
        for key in actual:
            self.assertEqual(actual[key].dtype, torch.float32)
            torch.testing.assert_close(actual[key], target[key], rtol=0, atol=0)
        promoted = core(items[0].float(), *items[1:], retain_mass=True)
        self.assertTrue(torch.equal(promoted['tokens'], target['tokens']))

    def test_empty_invalid_and_no_question_interface(self):
        core = NativeVisualMemory()
        items = sample()
        empty = core.summarize(*subset(items, slice(0, 0)))
        full = core.summarize(*items)
        for state in (core.merge(empty, full), core.merge(full, empty)):
            for key, value in core.read(full, retain_mass=True).items():
                torch.testing.assert_close(core.read(state, retain_mass=True)[key], value, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            core.read(core.merge(empty, empty), retain_mass=False)
        invalid = [(items[0][:, :-1], *items[1:]), (items[0].double(), *items[1:]),
                   (items[0], items[1].float(), *items[2:]),
                   (items[0], items[1] + 128, *items[2:]),
                   (items[0], items[1] - 1, *items[2:]),
                   (*items[:4], torch.zeros_like(items[4]), items[5]),
                   (items[0], items[1], items[4], *items[3:])]
        for bad in (float('nan'), float('inf'), float('-inf'), torch.finfo(torch.float32).max):
            changed = items[0].clone(); changed[0, 0] = bad
            invalid.append((changed, *items[1:]))
        for case in invalid:
            with self.assertRaises(ValueError):
                core(*case, retain_mass=True)
        with self.assertRaises(ValueError):
            core(*items, retain_mass=1)
        with self.assertRaises(TypeError):
            core(*items, retain_mass=True, question='not an input')
        self.assertEqual(list(inspect.signature(core.forward).parameters),
                         ['features', 'frame_index', 'raster_row', 'raster_col', 'grid_height', 'grid_width', 'retain_mass'])
        wrong = AttentionMoments(full.maximum[:1], full.scaled_mass[:1], full.scaled_value[:1])
        with self.assertRaises(ValueError):
            core.read(wrong, retain_mass=True)
        with torch.no_grad():
            core.mass_direction[0] = float('nan')
        with self.assertRaises(ValueError):
            core(*items, retain_mass=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    cpu_slurm()
    started = time.perf_counter()
    out = REPO / 'outputs/native_aggregation_vlm/native_visual_memory' / ('check_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    frozen = {name: digest(REPO / name) for name in OWN}
    if frozen[PROPOSAL] != PROPOSAL_SHA or any(digest(REPO / name) != value for name, value in INHERITED.items()):
        raise RuntimeError('Prospective source/accumulator binding changed')
    (out / 'source').mkdir()
    for name, value in {**frozen, **INHERITED}.items():
        target = out / 'source' / name.replace('/', '_')
        target.write_bytes((REPO / name).read_bytes())
        if digest(target) != value:
            raise RuntimeError('Source copy differs')
    policy = dict(cpu_seconds=90, cpu_cores=4, memory_gib=16, native_model_calls=0, native_head_calls=0,
                  optimizer_steps=0, no_native_integration=True, no_fit_or_inference_release=True)
    save(out / 'request.json', dict(source_sha256=frozen, inherited_source_sha256=INHERITED, policy=policy))
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeVisualMemoryTests))
        findings = dict(tests_run=result.testsRun, failures=[dict(test=str(t), trace=s) for t, s in result.failures],
                        errors=[dict(test=str(t), trace=s) for t, s in result.errors], skipped=result.skipped)
        save(out / 'test_results.json', findings)
        if not result.wasSuccessful() or result.testsRun != 8 or time.perf_counter() - started > 90:
            raise RuntimeError('Native memory fixture or fixed90-second cap failed')
        if frozen != {name: digest(REPO / name) for name in OWN}:
            raise RuntimeError('Native memory sources changed during check')
        (out / 'REPORT.md').write_text('# Native visual memory core software check\n\nEight native-width CPU fixture groups passed. This is algebra, interface and gradient evidence only. No images, decoder, native head, optimizer, fitting or task evaluation ran.\n')
        save(out / 'summary.json', dict(passed=True, completed=True, protocol='native_visual_memory_core_check',
            source_sha256=frozen, inherited_source_sha256=INHERITED, policy=policy,
            test_results_file=str(out / 'test_results.json'), test_results_sha256=digest(out / 'test_results.json'),
            report_file=str(out / 'REPORT.md'), report_sha256=digest(out / 'REPORT.md'),
            torch_version=torch.__version__, elapsed_seconds=time.perf_counter() - started,
            parameters=469504, slots=32, key_width=128, hidden_size=3584, tests_run=8))
    except BaseException as exc:
        save(out / 'failure.json', dict(type=type(exc).__name__, message=str(exc),
            source_sha256=frozen, inherited_source_sha256=INHERITED, policy=policy,
            elapsed_seconds=time.perf_counter() - started, partial_evidence_retained=True))
        raise


if __name__ == '__main__':
    main()
