"""CPU-Slurm fixtures for loss-selected replay of every causal memory write.

The actual tiny Qwen SDPA block matches the established V11 CPU fixture. No
pretrained model, dataset, GPU, optimizer, or accuracy evaluation is involved.
Source snapshots precede imports/tests; all outcomes are saved immutably.
"""
from pathlib import Path
import hashlib
import inspect
import io
import json
import os
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OWN = ('tests/test_native_learned_memory_training.py',
       'slurm/native_learned_memory_training_check.sbatch',
       'scripts/native_learned_memory_training.py',
       'scripts/native_vision_v11_last_block.py',
       'tests/test_native_vision_v11_last_block.py')
PROTOCOL = 'native_learned_memory_selected_loss_cpu_check'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def make_suite():
    import torch
    import transformers
    from torch import nn
    from torch.nn import functional as F
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer, Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm)
    from transformers.masking_utils import create_causal_mask
    from scripts.native_vision_v11_last_block import replay_last_block
    from scripts.native_learned_memory_training import replay_learned_memory

    class MemoryTrainingTests(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(11)
            torch.set_num_threads(4)

        def fixture(self, *, batch=1, length=7, dtype=torch.float32):
            # Exact architecture and seed convention of the existing V11 fixture.
            config = Qwen2_5_VLTextConfig(vocab_size=23, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, use_sliding_window=False,
                layer_types=['full_attention', 'full_attention'], attention_dropout=0.,
                rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]})
            config._attn_implementation = 'sdpa'
            modules = (Qwen2_5_VLDecoderLayer(config, layer_idx=1),
                       Qwen2RMSNorm(32, eps=config.rms_norm_eps), nn.Linear(32, 23, bias=False),
                       Qwen2_5_VLRotaryEmbedding(config=config))
            for module in modules:
                module.eval().to(dtype=dtype)
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            values = dict(hidden_states=torch.randn(batch, length, 32).to(dtype),
                input_ids=torch.arange(length).expand(batch, -1).clone(),
                attention_mask=torch.ones(batch, length, dtype=torch.long),
                position_ids=torch.arange(length).view(1, 1, -1).expand(3, batch, -1).clone())
            return modules, values

        def replay(self, modules, values, deltas, placement, writes=None, losses=None):
            return replay_learned_memory(*modules, **values, deltas=deltas, placement=placement,
                write_indices=[[3, 4, 5]] if writes is None else writes,
                loss_indices=[[3, 4, 5]] if losses is None else losses)

        def test_all_writes_supervised_recovers_v11_logits_and_gradients(self):
            modules, values = self.fixture(batch=2)
            indices = [[2, 4, 5], [1, 3]]; base = torch.randn(5, 32)*.1
            target = torch.tensor([2, 7, 11, 3, 5])
            for placement in ('pre_last', 'post_last'):
                old_delta = base.clone().requires_grad_(); new_delta = base.clone().requires_grad_()
                old = replay_last_block(*modules, **values, query_indices=indices,
                                        deltas=old_delta, placement=placement)
                new = self.replay(modules, values, new_delta, placement, indices, indices)
                self.assertTrue(torch.equal(new['logits'], old['logits']))
                self.assertTrue(torch.equal(new['block_input'], old['block_input']))
                self.assertTrue(torch.equal(new['final_norm_input'], old['final_norm_input']))
                before = torch.autograd.grad(F.cross_entropy(old['logits'], target), old_delta)[0]
                after = torch.autograd.grad(F.cross_entropy(new['logits'], target), new_delta)[0]
                self.assertTrue(torch.equal(before, after))
                self.assertEqual(new['write_offsets'], [0, 3, 5])
                self.assertEqual(new['loss_offsets'], [0, 3, 5])
                self.assertEqual(new['loss_to_write_indices'].tolist(), list(range(5)))
                self.assertIs(new['deltas'], new_delta)
                self.assertFalse(new['use_cache'])

        def test_last_loss_keeps_unsupervised_earlier_write_gradients(self):
            modules, values = self.fixture(); base = torch.randn(3, 32)*.1
            for placement in ('pre_last', 'post_last'):
                delta = base.clone().requires_grad_()
                result = self.replay(modules, values, delta, placement, losses=[[5]])
                self.assertEqual(result['logits'].shape, (1, 23))
                self.assertEqual(result['loss_to_write_indices'].tolist(), [2])
                gradient = torch.autograd.grad(F.cross_entropy(result['logits'], torch.tensor([7])), delta)[0]
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient[2].norm()), 1e-7)
                if placement == 'pre_last':
                    self.assertGreater(float(gradient[0].norm()), 1e-7)
                    self.assertGreater(float(gradient[1].norm()), 1e-7)
                else:
                    self.assertTrue(torch.equal(gradient[:2], torch.zeros_like(gradient[:2])))
                self.assertEqual(result['write_fused_query_hidden'].shape, (3, 32))

        def test_future_and_other_scene_writes_have_zero_value_and_gradient_effect(self):
            modules, values = self.fixture(batch=2)
            writes = [[2, 4, 6], [1, 3, 5]]; losses = [[4], []]
            base = torch.randn(6, 32)*.1
            for placement in ('pre_last', 'post_last'):
                delta = base.clone().requires_grad_()
                result = self.replay(modules, values, delta, placement, writes, losses)
                gradient = torch.autograd.grad(F.cross_entropy(result['logits'], torch.tensor([7])), delta)[0]
                self.assertTrue(torch.equal(gradient[2:], torch.zeros_like(gradient[2:])))
                self.assertGreater(float(gradient[1].norm()), 1e-7)
                self.assertEqual(bool((gradient[0] != 0).any()), placement == 'pre_last')
                changed = base.clone(); changed[2:] += torch.linspace(-1., 1., 32)
                altered = self.replay(modules, values, changed, placement, writes, losses)
                self.assertTrue(torch.equal(result['logits'], altered['logits']))

        def test_detaching_unsupervised_history_cuts_only_its_temporal_credit(self):
            modules, values = self.fixture(); base = torch.randn(3, 32)*.1
            for placement in ('pre_last', 'post_last'):
                delta = base.clone().requires_grad_()
                live = self.replay(modules, values, delta, placement, losses=[[5]])
                detached = torch.cat((delta[:2].detach(), delta[2:]), dim=0)
                fixed = self.replay(modules, values, detached, placement, losses=[[5]])
                self.assertTrue(torch.equal(live['logits'], fixed['logits']))
                a = torch.autograd.grad(F.cross_entropy(live['logits'], torch.tensor([7])), delta)[0]
                b = torch.autograd.grad(F.cross_entropy(fixed['logits'], torch.tensor([7])), delta)[0]
                self.assertTrue(torch.equal(b[:2], torch.zeros_like(b[:2])))
                self.assertTrue(torch.equal(a[2], b[2]))
                self.assertEqual(bool((a[:2] != 0).any()), placement == 'pre_last')

        def test_ragged_padding_maps_and_only_selected_rows_reach_head(self):
            modules, values = self.fixture(batch=3, length=8)
            values['attention_mask'][1, :2] = 0; values['attention_mask'][2, -2:] = 0
            pos = values['attention_mask'].cumsum(-1)-1; pos.masked_fill_(values['attention_mask'] == 0, 1)
            values['position_ids'] = pos.unsqueeze(0).expand(4, -1, -1).clone()
            writes = [[], [3, 4, 6], [1, 3, 5]]; losses = [[], [4], [3, 5]]
            delta = torch.randn(6, 32)*.1; changed = {k: v.clone() for k, v in values.items()}
            changed['hidden_states'][~values['attention_mask'].bool()] += 100.
            for placement in ('pre_last', 'post_last'):
                calls = dict(block=[], norm=[], head=[]); handles = []
                for name, module in zip(('block', 'norm', 'head'), modules[:3]):
                    def hook(module, args, kwargs, key=name):
                        x = args[0] if args else kwargs['hidden_states']
                        calls[key].append(tuple(x.shape))
                    handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
                try:
                    result = self.replay(modules, values, delta, placement, writes, losses)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(calls, dict(block=[(3, 8, 32)], norm=[(3, 32)], head=[(3, 32)]))
                self.assertEqual(result['logits'].shape, (3, 23))
                self.assertEqual(result['write_offsets'], [0, 0, 3, 6])
                self.assertEqual(result['loss_offsets'], [0, 0, 1, 3])
                self.assertEqual(result['flat_write_batch_indices'].tolist(), [1, 1, 1, 2, 2, 2])
                self.assertEqual(result['flat_write_indices'].tolist(), [3, 4, 6, 1, 3, 5])
                self.assertEqual(result['flat_loss_batch_indices'].tolist(), [1, 2, 2])
                self.assertEqual(result['flat_loss_indices'].tolist(), [4, 3, 5])
                self.assertEqual(result['loss_to_write_indices'].tolist(), [1, 4, 5])
                other = self.replay(modules, changed, delta, placement, writes, losses)
                self.assertTrue(torch.equal(result['logits'], other['logits']))

        def test_native_fp16_cast_precedes_add_and_keeps_backward(self):
            modules, values = self.fixture(dtype=torch.float16)
            values['hidden_states'].fill_(1.)
            delta = torch.full((3, 32), .0004884, dtype=torch.float32, requires_grad=True)
            result = self.replay(modules, values, delta, 'pre_last', losses=[[5]])
            actual = result['block_input'][0, [3, 4, 5]]
            wrong = (values['hidden_states'][0, [3, 4, 5]].float()+delta).half()
            self.assertTrue(torch.equal(actual, torch.ones_like(actual)))
            self.assertFalse(torch.equal(actual, wrong))
            F.cross_entropy(result['logits'].float(), torch.tensor([7])).backward()
            self.assertTrue(bool(torch.isfinite(delta.grad).all()))
            self.assertEqual(result['logits'].dtype, torch.float16)

        def test_native_parameters_and_input_tensors_remain_unchanged(self):
            modules, values = self.fixture()
            originals = {k: v.clone() for k, v in values.items()}
            states = [{k: v.clone() for k, v in module.state_dict().items()} for module in modules]
            versions = [[p._version for p in module.parameters()] for module in modules]
            for placement in ('pre_last', 'post_last'):
                delta = (torch.randn(3, 32)*.1).requires_grad_()
                result = self.replay(modules, values, delta, placement, losses=[[5]])
                F.cross_entropy(result['logits'], torch.tensor([7])).backward()
                self.assertIsNotNone(delta.grad)
                for module, state, version in zip(modules, states, versions):
                    self.assertTrue(all(torch.equal(state[k], v) for k, v in module.state_dict().items()))
                    self.assertEqual([p._version for p in module.parameters()], version)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.parameters()))
            self.assertTrue(all(torch.equal(v, originals[k]) for k, v in values.items()))

        def test_malformed_write_loss_indices_and_padding_are_rejected(self):
            modules, values = self.fixture(); delta = torch.zeros(3, 32)
            bad_writes = ([[3, 3, 5]], [[5, 4, 3]], [[True, 4, 5]], [[-1, 4, 5]], [[3, 4, 7]], [[]], [])
            for writes in bad_writes:
                with self.subTest(writes=writes), self.assertRaises(ValueError):
                    self.replay(modules, values, delta, 'pre_last', writes, [[5]])
            bad_losses = ([[5, 5]], [[5, 3]], [[True]], [[2]], [[7]], [[]], [])
            for losses in bad_losses:
                with self.subTest(losses=losses), self.assertRaises(ValueError):
                    self.replay(modules, values, delta, 'pre_last', losses=losses)
            values['attention_mask'][0, 3] = 0
            with self.assertRaises(ValueError):
                self.replay(modules, values, delta, 'pre_last', losses=[[5]])
            values['attention_mask'][0, 3] = 1; values['attention_mask'][0, 5] = 0
            with self.assertRaises(ValueError):
                self.replay(modules, values, delta, 'pre_last', losses=[[5]])

        def test_invalid_delta_layout_and_trainable_backbone_are_rejected(self):
            modules, values = self.fixture(); delta = torch.zeros(3, 32)
            nonfinite = delta.clone(); nonfinite[0, 0] = float('nan')
            for invalid in (delta.half(), delta[:2], nonfinite):
                with self.assertRaises(ValueError):
                    self.replay(modules, values, invalid, 'pre_last', losses=[[5]])
            with self.assertRaises(ValueError):
                self.replay(modules, values, delta, 'unknown', losses=[[5]])
            bad = {k: v.clone() for k, v in values.items()}; bad['position_ids'] = bad['position_ids'][:, :, :-1]
            with self.assertRaises(ValueError):
                self.replay(modules, bad, delta, 'pre_last', losses=[[5]])
            next(modules[0].parameters()).requires_grad_(True)
            with self.assertRaises(ValueError):
                self.replay(modules, values, delta, 'pre_last', losses=[[5]])

    native_sources = {str(Path(inspect.getfile(symbol)).resolve()): sha(inspect.getfile(symbol))
                      for symbol in (Qwen2_5_VLTextConfig, Qwen2_5_VLDecoderLayer,
                                     Qwen2_5_VLRotaryEmbedding, Qwen2RMSNorm, create_causal_mask)}
    metadata = dict(torch_version=str(torch.__version__), transformers_version=str(transformers.__version__),
                    native_source_sha256=native_sources, attention_backend='sdpa', hidden_size=32,
                    vocabulary_size=23, seed=11, cpu_threads=4, full_pretrained_model_loaded=False)
    return unittest.defaultTestLoader.loadTestsFromTestCase(MemoryTrainingTests), metadata


def main():
    if not (os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
            and not os.environ.get('SLURM_JOB_GPUS')):
        raise RuntimeError('Run these tensor/model tests only through CPU Slurm')
    out = ROOT/'outputs/native_aggregation_vlm/learned_memory'/('training_check_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False); (out/'source').mkdir()
    start = time.perf_counter(); frozen = {}; result = None; error = None; metadata = {}; log = io.StringIO()
    try:
        for name in OWN:
            source = ROOT/name; frozen[name] = sha(source)
            copied = out/'source'/name.replace('/', '_'); copied.write_bytes(source.read_bytes())
            if sha(copied) != frozen[name]:
                raise RuntimeError('Source changed while snapshotting: '+name)
        save(out/'source_hashes.json', frozen)
        suite, metadata = make_suite()
        save(out/'native_runtime.json', metadata)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if result.testsRun != 9 or result.skipped or not result.wasSuccessful():
            raise RuntimeError('All nine native memory training CPU tests must pass')
        if {name: sha(ROOT/name) for name in OWN} != frozen:
            raise RuntimeError('Source changed during CPU tests')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        log_path = out/'unit_tests.log'; log_path.write_text(log.getvalue()+(error['traceback'] if error else ''))
        passed = error is None and result is not None and result.wasSuccessful() and result.testsRun == 9 and not result.skipped
        summary = dict(protocol=PROTOCOL, passed=passed, completed=passed, cpu_only=True, no_fit=True,
            slurm_job_id=os.environ['SLURM_JOB_ID'], source_sha256=frozen,
            tests_run=0 if result is None else result.testsRun,
            failures=0 if result is None else len(result.failures), errors=0 if result is None else len(result.errors),
            skipped=0 if result is None else len(result.skipped), tests_expected=9, runtime=metadata,
            log_file=str(log_path), log_sha256=sha(log_path), seconds=time.perf_counter()-start, failure=error)
        save(out/'summary.json', summary)
        print(log_path.read_text(), end=''); print(json.dumps(dict(passed=passed, directory=str(out))))
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
