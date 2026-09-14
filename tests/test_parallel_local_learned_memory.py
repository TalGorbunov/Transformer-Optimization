"""CPU Slurm fixtures for the held learned-memory controller; no native model.

Only this test class runs. Tiny causal fixture classes are imported from the
immutable V11 test file, whose source is included in the snapshot. No fixture
result establishes actual Qwen/Cosmos numerical equivalence or efficacy.
"""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
OWN = ('tests/test_parallel_local_learned_memory.py',
       'slurm/native_learned_memory_controller_check.sbatch',
       'gnnformer/parallel_local_learned_memory.py',
       'tests/test_parallel_local_memory.py', 'gnnformer/parallel_local_memory.py',
       'gnnformer/parallel_local_learned_selection.py',
       'gnnformer/parallel_local_aggregation.py', 'gnnformer/parallel_local_native.py')
PROTOCOL = 'learned_memory_controller_cpu_fixtures'
START = time.perf_counter()
OUT = None
FROZEN = None


def source_hashes():
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in OWN}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'cpu'
        or os.environ.get('SLURM_JOB_GPUS')):
    raise SystemExit('Run these numerical fixtures only in CPU Slurm without a GPU allocation')
if __name__ == '__main__':
    OUT = REPO / 'outputs/native_aggregation_vlm/learned_memory' / ('controller_check_' + os.environ['SLURM_JOB_ID'])
    OUT.mkdir(parents=True, exist_ok=False)
    FROZEN = source_hashes()
    for name, digest in FROZEN.items():
        target = OUT / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed during snapshot: ' + name)
    save(OUT / 'source_hashes.json', FROZEN)
    save(OUT / 'request.json', dict(protocol=PROTOCOL, argv=sys.argv,
        slurm_job_id=os.environ['SLURM_JOB_ID'], partition=os.environ['SLURM_JOB_PARTITION'],
        source_sha256=FROZEN, no_pretrained_model_loaded=True))

sys.path.insert(0, str(REPO))
try:
    import torch
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    from gnnformer.parallel_local_learned_memory import ParallelLocalLearnedMemory
    from gnnformer.parallel_local_memory import ParallelLocalMemory
    from gnnformer.parallel_local_native import ParallelLocalNative
    spec = importlib.util.spec_from_file_location('frozen_v11_memory_fixtures', REPO / 'tests/test_parallel_local_memory.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
except BaseException as exc:
    if OUT is not None:
        save(OUT / 'failure.json', dict(type=type(exc).__name__, message=str(exc), source_sha256=FROZEN))
    raise


class LearnedMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261126)
        torch.set_num_threads(4)
        self.model = fixtures.TinyNativeModel(8).half().eval()
        self.core = ParallelLocalLearnedSelection(8, rank=4, mode='clip')
        self.hidden = (torch.randn(4, 5, 8) * .5).half()

    def activate(self, core=None):
        with torch.no_grad():
            (self.core if core is None else core).up.weight.normal_(0, .2)

    def controller(self, placement, **changes):
        options = dict(n_local_rows=3, write_location=placement, query_indices=[1, 3, 4],
                       stream_positions=[1, 3, 4], capture=True, detach_captures=False)
        options.update(changes)
        return ParallelLocalLearnedMemory(self.model.read, self.model.norm, self.core, **options)

    def assert_clean(self):
        self.assertFalse(self.model.read._forward_hooks)
        self.assertFalse(self.model.norm._forward_pre_hooks)
        self.assertNotIn(self.model.read, ParallelLocalMemory._owners)
        self.assertNotIn(self.model.norm, ParallelLocalMemory._owners)

    def test_zero_identity_single_call_all_modes_and_native_state(self):
        baseline, reference_kv = self.model(self.hidden)
        native_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        for mode in ('clip', 'sigmoid', 'softmax'):
            self.core = ParallelLocalLearnedSelection(8, rank=4, mode=mode)
            for placement in ('pre_last', 'post_last'):
                calls = []
                hook = self.core.register_forward_hook(lambda m, a, out: calls.append(a[0].shape))
                try:
                    with self.controller(placement, capture=False) as control:
                        output, kv = self.model(self.hidden, keyword=True)
                        self.assertTrue(torch.equal(output, baseline))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(kv, reference_kv)))
                        self.assertEqual(calls, [torch.Size([3, 3, 8])])
                        self.assertEqual((control.calls, control.core_calls, control.read_calls, control.norm_calls), (1, 1, 1, 1))
                        self.assertIsNone(control.last_capture)
                        self.assertFalse(any(m is self.core for m in self.model.modules()))
                        control.assert_complete()
                finally:
                    hook.remove()
                self.assert_clean()
        self.assertEqual(set(native_state), set(self.model.state_dict()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, native_state[name]))

    def test_actual_capture_boundaries_and_untouched_rows(self):
        self.activate()
        original = self.hidden.clone()
        baseline, reference_kv = self.model(self.hidden)
        common = self.model.read(self.hidden)[0]
        captures = {}
        for placement in ('pre_last', 'post_last'):
            norm_inputs = []
            observer = self.model.norm.register_forward_pre_hook(lambda m, a: norm_inputs.append(a[0]))
            try:
                with self.controller(placement, query_indices=[1, 3], stream_positions=[1, 3]) as control:
                    output, kv = self.model(self.hidden)
                    cap = control.last_capture
                    captures[placement] = cap
                    self.assertTrue(torch.equal(cap['common_query_hidden'], common[:, [1, 3]]))
                    self.assertTrue(torch.equal(cap['local_states'], common[:-1, [1, 3]]))
                    self.assertTrue(torch.equal(cap['global_states'], common[-1, [1, 3]]))
                    self.assertTrue(torch.equal(cap['native_query_hidden'], norm_inputs[0][:, [1, 3]]))
                    self.assertTrue(torch.equal(cap['final_block_input_queries'], self.model.last.last_input[:, [1, 3]]))
                    self.assertTrue(torch.equal(cap['write_output'], cap['write_input'] + cap['native_delta']))
                    self.assertTrue(torch.equal(cap['native_delta'], cap['delta'].half()))
                    self.assertEqual(cap['delta'].dtype, torch.float32)
                    self.assertEqual(cap['native_delta'].dtype, torch.float16)
                    self.assertTrue(torch.equal(cap['fused_query_hidden'][-1], cap['norm_input_after_write']))
                    self.assertTrue(torch.equal(cap['messages'], cap['gates'].unsqueeze(-1) * cap['payload']))
                    self.assertTrue(torch.equal(self.model.last.last_input[:-1], common[:-1]))
                    self.assertTrue(torch.equal(self.model.last.last_input[-1, [0, 2, 4]], common[-1, [0, 2, 4]]))
                    self.assertTrue(torch.equal(output[:-1], baseline[:-1]))
                    self.assertTrue(torch.equal(output[-1, :1], baseline[-1, :1]))
                    self.assertTrue(all(torch.equal(a[:-1], b[:-1]) for a, b in zip(kv, reference_kv)))
                    if placement == 'pre_last':
                        self.assertFalse(torch.equal(kv[0][-1], reference_kv[0][-1]))
                    else:
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(kv, reference_kv)))
                        self.assertTrue(torch.equal(output[-1, [0, 2, 4]], baseline[-1, [0, 2, 4]]))
                    control.assert_complete()
            finally:
                observer.remove()
            self.assert_clean()
        for key in ('query', 'payload', 'scores', 'gates', 'messages', 'aggregate', 'preactivation', 'delta'):
            self.assertTrue(torch.equal(captures['pre_last'][key], captures['post_last'][key]), key)
        self.assertTrue(torch.equal(self.hidden, original))

    def test_live_delta_cast_and_temporal_selection_gradients(self):
        for mode in ('clip', 'sigmoid', 'softmax'):
            self.core = ParallelLocalLearnedSelection(8, rank=4, mode=mode)
            self.activate()
            for placement in ('pre_last', 'post_last'):
                self.core.zero_grad(set_to_none=True)
                returned = []
                observer = self.core.register_forward_hook(lambda m, a, out: returned.append(out))
                try:
                    with self.controller(placement) as control:
                        output, _ = self.model(self.hidden)
                        cap = control.last_capture
                        self.assertIs(cap['delta'], returned[0][0])
                        self.assertIs(cap['delta'], returned[0][1]['delta'])
                        self.assertTrue(cap['native_delta'].requires_grad)
                        loss = torch.nn.functional.cross_entropy(output[-1, 3:4].float(), torch.tensor([2]))
                        grad, native_grad = torch.autograd.grad(loss, (cap['delta'], cap['native_delta']), retain_graph=True)
                        self.assertTrue(bool(torch.isfinite(grad).all() and torch.isfinite(native_grad).all()))
                        self.assertGreater(float(grad[1].abs().sum()), 0.)
                        self.assertTrue(torch.equal(grad[2], torch.zeros_like(grad[2])))
                        if placement == 'pre_last':
                            self.assertGreater(float(grad[0].abs().sum()), 0.)
                        else:
                            self.assertTrue(torch.equal(grad[0], torch.zeros_like(grad[0])))
                        loss.backward()
                        self.assertGreater(float(self.core.selection_weight.grad.abs().sum()), 0.)
                        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
                        self.assertTrue(all(p.dtype == torch.float32 for p in self.core.parameters()))
                finally:
                    observer.remove()
                self.assert_clean()

    def test_full_history_and_cached_current_queries(self):
        self.activate()
        for placement in ('pre_last', 'post_last'):
            with self.controller(placement, query_indices=[2, 3, 4], stream_positions=[2, 3, 4]) as control:
                with torch.no_grad():
                    full, full_kv = self.model(self.hidden)
                    control.configure_queries([2], [2])
                    initial, kv = self.model(self.hidden[:, :3])
                    pieces = [initial]
                    for t in (3, 4):
                        control.configure_queries([0], [t])
                        output, kv = self.model(self.hidden[:, t:t+1], cache=kv)
                        pieces.append(output)
                    torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=3e-3, atol=3e-3)
                    for actual, expected in zip(kv, full_kv):
                        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
                    self.assertEqual(control.core_calls, 4)
                    control.assert_complete()
            self.assert_clean()

    def test_mutation_layout_and_overlap_rejection(self):
        mutations = dict(selection_mode='sigmoid', write_location='post_last', n_local_rows=2,
            branch=ParallelLocalLearnedSelection(8, rank=4), final_norm=fixtures.FrozenNorm(8).half().eval(),
            penultimate_layer=fixtures.FrozenReadLayer(8).half().eval(), query_indices=(3,), stream_positions=(99,), capture=False)
        for key, value in mutations.items():
            with self.assertRaises(ValueError):
                with self.controller('pre_last') as control:
                    setattr(control, key, value)
                    self.model(self.hidden)
            self.assertFalse(control.active)
            self.assert_clean()
        with self.assertRaises(ValueError):
            with self.controller('pre_last') as control:
                self.core.mode = 'sigmoid'
                self.model(self.hidden)
        self.core.mode = 'clip'
        self.assert_clean()
        for queries, positions in (([], []), ([True], [1]), ([1, 1], [1, 1]), ([2], [1]), ([1, 2], [1, 4])):
            with self.assertRaises(ValueError):
                self.controller('pre_last', query_indices=queries, stream_positions=positions)
        with self.controller('pre_last'):
            with self.assertRaises(ValueError):
                self.controller('post_last')
            with self.assertRaises(ValueError):
                with ParallelLocalMemory(self.model.read, self.model.norm, self.core, n_local_rows=3,
                        write_location='post_last', query_indices=[1], stream_positions=[1]):
                    pass
        self.assert_clean()
        with ParallelLocalNative(self.model.norm, self.core, n_local_rows=3):
            with self.assertRaises(ValueError):
                self.controller('pre_last')
        self.assert_clean()

    def test_exception_cleanup_preserves_external_hooks_and_reentry(self):
        seen = []
        external = self.model.norm.register_forward_pre_hook(lambda m, a: seen.append(1))
        control = self.controller('pre_last')
        def fail(*args):
            raise RuntimeError('synthetic upper-block failure')
        fault = self.model.last.register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                with control:
                    self.model(self.hidden)
            self.assertFalse(control.active)
            self.assertIn(external.id, self.model.norm._forward_pre_hooks)
        finally:
            fault.remove()
        with control:
            self.model(self.hidden)
            control.assert_complete()
        self.assertEqual(len(seen), 1)
        external.remove()
        self.assert_clean()
        with self.assertRaisesRegex(ValueError, 'between read'):
            with self.controller('pre_last') as control:
                self.model.read(self.hidden)
                control.configure_queries([0], [6])
        self.assert_clean()
        with self.assertRaisesRegex(ValueError, 'without the common read'):
            with self.controller('post_last'):
                self.model.norm(self.hidden)
        self.assert_clean()

    def test_fp16_validation_cast_and_addition_overflow(self):
        for value in (self.hidden.float(), self.hidden.clone().fill_(float('nan'))):
            with self.assertRaises(ValueError):
                with self.controller('pre_last'):
                    self.model(value)
            self.assert_clean()
        with torch.no_grad():
            self.core.query.weight.zero_(); self.core.local.weight.zero_(); self.core.local_bias.zero_()
            self.core.aggregate_projection.weight.zero_(); self.core.aggregate_projection.bias.fill_(1.)
            self.core.up.weight.fill_(1e6)
        for placement in ('pre_last', 'post_last'):
            with self.assertRaisesRegex(ValueError, 'cast overflowed'):
                with self.controller(placement):
                    self.model(self.hidden)
            self.assert_clean()
        with torch.no_grad():
            self.model.read.weight.fill_(1.)
            self.core.up.weight.fill_(10000. / (4 * float(torch.nn.functional.silu(torch.tensor(1.)))))
        with self.assertRaisesRegex(ValueError, 'addition overflowed'):
            with self.controller('pre_last'):
                self.model(self.hidden.clone().fill_(60000.))
        self.assert_clean()

    def test_detached_inference_export_and_empty_actual_set(self):
        self.activate()
        with self.controller('pre_last', detach_captures=True) as control, torch.inference_mode():
            self.model(self.hidden)
            exported = control.export_last_capture(cpu=True)
            for key, value in exported.items():
                if isinstance(value, torch.Tensor):
                    self.assertFalse(value.requires_grad)
                    self.assertFalse(torch.is_inference(value))
                    self.assertNotEqual(value.data_ptr(), control.last_capture[key].data_ptr())
            self.assertEqual(exported['query_indices'], [1, 3, 4])
            self.assertEqual(exported['selection_mode'], 'clip')
        self.assert_clean()
        for placement in ('pre_last', 'post_last'):
            with self.controller(placement, n_local_rows=0) as control:
                self.model(self.hidden[-1:])
                cap = control.last_capture
                self.assertEqual(tuple(cap['payload'].shape), (0, 3, 4))
                self.assertTrue(torch.equal(cap['aggregate'], torch.zeros_like(cap['aggregate'])))
                self.assertGreater(float(cap['delta'].abs().sum()), 0.)
                control.assert_complete()
            self.assert_clean()


def main():
    log = io.StringIO()
    try:
        if len(sys.argv) != 1:
            raise ValueError('This fixed CPU fixture driver accepts no arguments')
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LearnedMemoryTests)
        with redirect_stdout(log), redirect_stderr(log):
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (OUT / 'tests.log').write_text(log.getvalue())
        unchanged = source_hashes() == FROZEN
        passed = result.wasSuccessful() and result.testsRun == 8 and not result.skipped and unchanged
        summary = dict(schema_version=1, protocol=PROTOCOL, completed=True, passed=passed,
            tests_run=result.testsRun, failures=[dict(test=str(t), traceback=s) for t, s in result.failures],
            errors=[dict(test=str(t), traceback=s) for t, s in result.errors], skipped=result.skipped,
            source_sha256=FROZEN, sources_unchanged=unchanged,
            source_ledger_file=str(OUT/'source_hashes.json'), source_ledger_sha256=hashlib.sha256((OUT/'source_hashes.json').read_bytes()).hexdigest(),
            log_file=str(OUT/'tests.log'), log_sha256=hashlib.sha256((OUT/'tests.log').read_bytes()).hexdigest(),
            slurm_job_id=os.environ['SLURM_JOB_ID'], partition='cpu', torch_version=str(torch.__version__),
            elapsed_seconds=time.perf_counter()-START, no_pretrained_model_loaded=True,
            no_gpu_work=True, no_native_equivalence_or_efficacy_claim=True)
        save(OUT/'summary.json', summary)
        print(json.dumps(dict(completed=True, passed=passed, directory=str(OUT), tests_run=result.testsRun)))
        return 0 if passed else 1
    except BaseException as exc:
        (OUT/'tests.log').write_text(log.getvalue())
        save(OUT/'failure.json', dict(type=type(exc).__name__, message=str(exc), source_sha256=FROZEN,
            elapsed_seconds=time.perf_counter()-START, partial_outputs_retained=True))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
