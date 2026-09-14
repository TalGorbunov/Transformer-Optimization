"""Matched native MMReD continuation with training-only prefix supervision.

Numerical execution is Slurm-only. Existing native input/training helpers are
reused without modification; no hidden state is replaced during inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.stage_mmred_official_recovery import need, sha, save, object_sha

PROTOCOL = 'mmred_native_prefix_supervision'
OUT = REPO / 'outputs/native_aggregation_vlm/mmred_prefix_supervision'
DATA = Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision')
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/mmred_prefix_supervision')
TRAIN_PLAN = REPO / 'outputs/native_aggregation_vlm/mmred_official_native_training/check_444010/plan.json'
TRAIN_PLAN_SHA = '94b0c0eeed2130e256c63a5a4777599562b5be4f9b97b79e6356d21892d0b368'
EVAL_PLAN = REPO / 'outputs/native_aggregation_vlm/mmred_official_native_evaluation/check_444051/plan.json'
EVAL_PLAN_SHA = 'b044c69a3a558a5369b093ee23b9cdbbde24923f019eecad19f14f7558b31cb2'
ARMS = ('answer', 'local', 'prefix')
SEEDS = (25, 26)
WITNESSES = (0, 5, 800, 810, 1600, 1612, 2400, 2405, 2410, 3203, 3210, 3222)
ROUNDTRIP = (0, 3222)
POLICY = dict(protocol=PROTOCOL, arms=list(ARMS), seeds=list(SEEDS), layer=27,
    channels=40, hidden_width=3584, projection_seed=20260914, auxiliary_weight=.1,
    hidden_normalization='none', train_worlds=4000, epochs=3, presentations=12000,
    accumulation=8, updates=1500, lr=2e-4, betas=[.9, .999], eps=1e-8,
    weight_decay=.01, clip_norm=1., profile_accumulation=12,
    profile_updates_per_arm=2, profile_witnesses=list(WITNESSES),
    profile_gpu_seconds=900, main_gpu_seconds=7200, cpu_cores=4,
    training_diagnostics=100, maximum_new_tokens=50,
    native_forward_unchanged=True, oracle_inputs_at_inference=False,
    trainable_scope='original 224 language-attention QKVO LoRA tensors',
    no_checkpoint_selection=True, no_evaluation_during_training=True)
OWN = ('scripts/train_mmred_prefix_supervision.py',
       'gnnformer/native_prefix_supervision.py',
       'scripts/prepare_mmred_prefix_supervision.py',
       'tests/test_native_prefix_supervision.py',
       'slurm/mmred_prefix_supervision_prepare.sbatch',
       'slurm/mmred_prefix_supervision_profile.sbatch',
       'slurm/mmred_prefix_supervision_train.sbatch',
       'scripts/report_mmred_prefix_supervision.py',
       'slurm/mmred_prefix_supervision_report.sbatch',
       'docs/paper/MMRED_PREFIX_SUPERVISION_EXECUTION_PROTOCOL.md')


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=sha(path))


def verify_ref(value):
    need(sha(value['file']) == value['sha256'], 'Bound file changed: ' + value['file'])
    return Path(value['file'])


def verify_release(path):
    release = read(path)
    need(release['protocol'] == PROTOCOL and release['policy'] == POLICY,
         'Frozen training release/policy differs')
    need(set(OWN) <= set(release['source_sha256']), 'Incomplete source release')
    for name, digest in release['source_sha256'].items():
        need(sha(REPO / name) == digest, 'Released source changed: ' + name)
    return release


def verify_main_release(args):
    """Require the independent audit of this exact profile before model loading."""
    from scripts.report_mmred_prefix_supervision import verify_report
    release = read(args.main_release)
    need(release['protocol'] == PROTOCOL and release['policy'] == POLICY
         and release['source_release'] == ref(args.release)
         and release['targets'] == ref(args.targets), 'Main release does not bind this implementation')
    profile = read(verify_ref(release['profile_summary']))
    need(profile['passed'] is profile['completed'] is True
         and profile['phase'] == 'profile' and profile['protocol'] == PROTOCOL
         and profile['source_release'] == release['source_release']
         and profile['targets'] == release['targets'], 'Exact passed software profile required')
    analysis = read(verify_ref(profile['analysis']))
    need(analysis['policy'] == POLICY and analysis['phase'] == 'profile'
         and analysis['passed'] is analysis['completed'] is True,
         'Profile analysis or training policy differs')
    audit = verify_report(verify_ref(release['profile_audit']))
    need(audit['phase'] == 'profile' and audit['producer_summary'] == release['profile_summary']
         and audit['source_release'] == release['source_release']
         and audit['targets'] == release['targets'], 'Independent audit belongs to another profile')
    need(release['projection'] == profile['future_main_projection']
         == analysis['future_main_projection'] == audit['future_main_projection'],
         'Released forecast differs from independently reconstructed timing evidence')
    need(set(release['projection']) == set(ARMS) and all(
        v['passed'] is True and v['seconds'] <= v['cap_seconds'] == POLICY['main_gpu_seconds']
        for v in release['projection'].values()), 'All matched fits need a passing resource forecast')
    return release


def training_order(seed):
    rng = random.Random(seed)
    result = []
    for epoch in range(3):
        indices = list(range(4000))
        rng.shuffle(indices)
        result.extend(dict(index=i, epoch=epoch, micro_index=epoch * 4000 + j)
                      for j, i in enumerate(indices))
    return result


def gradient_summary(torch, params, gradients=None):
    values = {k: p.grad for k, p in params.items()} if gradients is None else dict(zip(params, gradients))
    present = {k: v is not None for k, v in values.items()}
    finite = {k: v is None or bool(v.isfinite().all()) for k, v in values.items()}
    norms = [0.] * 28
    for name, value in values.items():
        if value is not None:
            layer = int(re.search(r'\.layers\.(\d+)\.', name).group(1))
            norms[layer] += float(value.detach().float().square().sum())
    return dict(all_present=all(present.values()), all_finite=all(finite.values()),
                present=present, finite=finite, layer_l2=[math.sqrt(x) for x in norms])


class Runner:
    def __init__(self, args, out, started):
        import torch
        from gnnformer.runtime import load_runtime
        from gnnformer.native_prefix_supervision import FixedPrefixSupervision
        from scripts import prepare_mmred_prefix_supervision as preparation
        from scripts import evaluate_mmred_official_native_memory as old
        self.torch, self.old, self.training = torch, old, old.training
        self.profile, self.p = old.profile, old.p
        self.args, self.out, self.started = args, out, started
        self.cap = POLICY['profile_gpu_seconds' if args.profile else 'main_gpu_seconds']
        self.release = verify_release(args.release)
        need(sha(TRAIN_PLAN) == TRAIN_PLAN_SHA and sha(EVAL_PLAN) == EVAL_PLAN_SHA,
             'Fixed native parent plans changed')
        self.plan = self.training.verify_plan(TRAIN_PLAN)
        self.evaluation = old.verify_plan(EVAL_PLAN)
        self.proof = self.evaluation['main_arms']['ordinary']
        self.target_plan = preparation.verify_stage(args.targets)
        need(self.target_plan['training_plan'] == ref(TRAIN_PLAN), 'Target training-world owner differs')
        for name, digest in {**self.target_plan['inherited_source_sha256'],
                             **self.target_plan['source_sha256']}.items():
            need(self.release['source_sha256'].get(name) == digest,
                 'Training release omits or changes prepared-target source: ' + name)
        self.items = read(self.plan['cases_file'])
        self.target_packet = torch.load(verify_ref(self.target_plan['targets']), map_location='cpu', weights_only=True)
        self.target_rows = self.target_packet['records']
        self.projection_packet = torch.load(verify_ref(self.target_plan['projection']), map_location='cpu', weights_only=True)
        need(len(self.items) == len(self.target_rows) == 4000, 'Exact 4000 target/input rows required')
        for i, row in enumerate(self.target_rows):
            need(row['index'] == i and row['sid'] == self.items[i]['row']['sid']
                 and row['n'] == self.items[i]['row']['n'], 'Target/input ordering differs')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        need(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'NVIDIA B200', 'One B200 required')
        self.data, self.ckpt = DATA / out.name, CKPT / out.name
        self.data.mkdir(parents=True, exist_ok=False)
        self.ckpt.mkdir(parents=True, exist_ok=False)
        self.loaded = load_runtime(str(self.profile.preparation.MODEL), use_4bit=True,
                                   attn_implementation='sdpa', device_map='cuda')
        self.model = self.loaded.model.eval().requires_grad_(False)
        self.hardware = self.p.frozen_joint.live_identity(torch, self.loaded, self.evaluation)
        need(self.p.installed({}) == self.evaluation['packages'], 'Native packages changed')
        self.base_refs = list(self.model.named_parameters())
        self.base = self.p.base_metadata(self.base_refs)
        save(out / 'base_before.json', self.base)
        self.p.install_lora(torch, self.model, out / 'actual_peft_config.json')
        self.params = self.p.adapter_parameters(self.model)
        self.initial_ref = self.proof['final_checkpoint']
        self.initial = torch.load(verify_ref(self.initial_ref), map_location='cpu', weights_only=True)
        need(old.state_proof(torch, self.initial, 'ordinary', self.proof['peft_config']) == self.proof['state_proof'],
             'Competent ordinary initial checkpoint differs')
        self.profile.restore(torch, self.params, self.initial['adapter'])
        self.contract = self.p.adapter_contract(torch, self.model)
        need(len(self.params) == 224 and sum(v.numel() for v in self.params.values()) == 10092544,
             'Exact unchanged language-attention LoRA scope required')
        self.cores = {}
        for arm in ('local', 'prefix'):
            norm = self.projection_packet['normalization'][arm]
            core = FixedPrefixSupervision(self.projection_packet['projection'], norm['mean'], norm['scale']).to(self.model.device)
            need(not list(core.parameters()), 'Training projection must have no trainable parameters')
            self.cores[arm] = core
        self.counts = Counter(dict.fromkeys(('model', 'backbone', 'vision', 'language', 'norm', 'head',
                                            'backward', 'auxiliary_gradient_probes', 'optimizer'), 0))
        self.layers = [0] * 28
        self.hooks = ExitStack()
        for module, key in ((self.model, 'model'), (self.model.model, 'backbone'),
                            (self.model.model.visual, 'vision'), (self.model.model.language_model, 'language'),
                            (self.model.model.language_model.norm, 'norm'), (self.model.lm_head, 'head')):
            def bump(*_, key=key): self.counts[key] += 1
            self.hooks.callback(module.register_forward_pre_hook(bump).remove)
        for index, module in enumerate(self.model.model.language_model.layers):
            def bump(*_, index=index): self.layers[index] += 1
            self.hooks.callback(module.register_forward_pre_hook(bump).remove)
        self.capture_active = False
        self.retained, self.naturals, self.states = [], [], {}
        self.timings = dict(teacher=[], optimizer=[], checkpoint=[], generation=[])
        save(out / 'config.json', dict(protocol=PROTOCOL, policy=POLICY, source_release=ref(args.release),
            main_release=None if args.profile else ref(args.main_release),
            targets=ref(args.targets), target_plan=self.target_plan, training_plan=ref(TRAIN_PLAN),
            evaluation_plan=ref(EVAL_PLAN), initial=self.initial_ref, hardware=self.hardware,
            peft_config=self.proof['peft_config'], contract=self.contract,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            native_identity_sha256=self.evaluation['native_identity_sha256'],
            data_directory=str(self.data), checkpoint_directory=str(self.ckpt)))
        self.setup_seconds = time.perf_counter() - started

    def guard(self):
        need(time.perf_counter() - self.started < self.cap - 20, 'Prospective Slurm work deadline reached')
        self.p.check_base(self.base_refs, self.base, self.model)

    def reset(self, seed, training=True):
        self.profile.restore(self.torch, self.params, self.initial['adapter'])
        for value in self.params.values(): value.grad = None
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.profile.model_mode(self.model, self.params, training)

    def optimizer(self):
        return self.torch.optim.AdamW(list(self.params.values()), lr=2e-4, betas=(.9, .999),
                                      eps=1e-8, weight_decay=.01)

    def state(self, arm, seed, update, optimizer=None, suffix=''):
        tick = time.perf_counter()
        value = dict(arm=arm, seed=seed, updates=update,
            adapter=self.profile.cpu_state(self.torch, self.params),
            peft_config=self.proof['peft_config'], initial=self.initial_ref,
            source_release=ref(self.args.release), targets=ref(self.args.targets), policy=POLICY,
            rng=dict(cpu=self.torch.random.get_rng_state(), cuda=self.torch.cuda.get_rng_state_all()))
        if optimizer is not None: value['optimizer'] = self.p.cpu_tree(self.torch, optimizer.state_dict())
        path = self.ckpt / f'{arm}_seed{seed}_update{update}{suffix}.pt'
        self.torch.save(value, path)
        result = dict(ref(path), arm=arm, seed=seed, updates=update)
        self.states[path.name] = result
        self.timings['checkpoint'].append(time.perf_counter() - tick)
        return result

    def teacher(self, index, arm, state, *, accumulation=8, backward=False,
                retain=None, capture=True, auxiliary_probe=False):
        torch = self.torch
        torch.cuda.synchronize()
        tick = time.perf_counter()
        case, features = self.training.load_inputs(torch, self.model, self.items[index])
        row = self.target_rows[index]
        positions = row['image_end_positions'].to(self.model.device)
        ids = case['teacher_input_ids'].to(self.model.device)
        need(torch.equal((ids[0] == 151653).nonzero().flatten(), positions)
             and positions.numel() == row['n'] and int(positions.max()) < case['metadata']['prompt_width'],
             'Exact native visible image-end positions required')
        evidence = dict(parameter_state=state, feature_packet=self.items[index]['features'],
                        compact_packet=self.items[index]['compact'])
        active = {}
        def observe(module, args, kwargs):
            need(not active, 'Exactly one selected decoder boundary per teacher call')
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            need(hidden is not None and hidden.ndim == 3 and hidden.shape[0] == 1,
                 'Native selected hidden state missing')
            active['hidden'] = hidden[0].index_select(0, positions)
        handle = None
        try:
            if capture:
                need(not self.capture_active, 'Nested prefix capture is forbidden')
                self.capture_active = True
                handle = self.model.model.language_model.layers[27].register_forward_pre_hook(observe, with_kwargs=True)
            result = self.profile.ordinary_teacher(torch, self.model, case, features, evidence=evidence)
        finally:
            if handle is not None: handle.remove()
            self.capture_active = False
        aux = result['loss'].new_zeros(())
        weight = .1 if arm in ('local', 'prefix') else 0.
        prediction = None
        if capture:
            target_arm = arm if arm in ('local', 'prefix') else 'local'
            core = self.cores[target_arm]
            raw = row[target_arm].to(self.model.device)
            hidden = active['hidden']
            prediction = core(hidden)
            normalized = (raw.float() - core.mean) / core.scale
            diagnostic_aux = (prediction - normalized).square().mean()
            if weight: aux = diagnostic_aux
            if retain and prediction.requires_grad: prediction.retain_grad()
            evidence['prefix_supervision'] = dict(arm=arm, target_arm=target_arm, layer=27,
                positions=positions, hidden_rows=hidden, projection=core.projection,
                mean=core.mean, scale=core.scale, raw_targets=raw,
                normalized_targets=normalized, predictions=prediction,
                decoded=prediction * core.scale + core.mean,
                auxiliary_loss=float(aux.detach()), diagnostic_auxiliary_loss=float(diagnostic_aux.detach()),
                auxiliary_weight=weight, accumulation=accumulation, base_ce=float(result['loss'].detach()),
                no_injected_states=True, targets_descriptor=self.target_plan['targets'])
        total = result['loss'] + weight * aux
        need(bool(total.isfinite()), 'Finite native and auxiliary training loss required')
        evidence.update(total_loss=float(total.detach()), auxiliary_loss=float(aux.detach()),
                        auxiliary_weight=weight, backward_loss=float(total.detach()) / accumulation)
        probe = None
        probe_seconds = 0.
        if auxiliary_probe:
            torch.cuda.synchronize()
            probe_tick = time.perf_counter()
            need(weight > 0 and capture, 'Auxiliary path probe requires a supervised arm')
            gradients = torch.autograd.grad(aux, tuple(self.params.values()), allow_unused=True, retain_graph=True)
            self.counts['auxiliary_gradient_probes'] += 1
            probe = gradient_summary(torch, self.params, gradients)
            need(probe['all_finite'] and all(x > 0 for x in probe['layer_l2'][:27])
                 and probe['layer_l2'][27] == 0., 'Auxiliary gradients must reach lower layers and not block27')
            evidence['auxiliary_gradient_probe'] = probe
            del gradients
            if prediction is not None:
                prediction.grad = None
            torch.cuda.synchronize()
            probe_seconds = time.perf_counter() - probe_tick
        if backward:
            (total / accumulation).backward()
            self.counts['backward'] += 1
            gradients = gradient_summary(torch, self.params)
            need(gradients['all_present'] and gradients['all_finite'], 'All224 live adapter gradients must be present and finite')
        else:
            gradients = None
        if capture and retain:
            evidence['prefix_supervision']['prediction_gradient'] = None if prediction.grad is None else prediction.grad
            evidence['prefix_supervision']['total_loss'] = float(total.detach())
        descriptor = None
        retention_seconds = 0.
        if retain:
            retain_tick = time.perf_counter()
            file = self.data / (retain + '.pt')
            descriptor = self.profile.retain(torch, file, index, arm, 'teacher', evidence,
                backward_completed=backward, gradient_summary=gradients)
            self.retained.append(dict(descriptor, index=index, arm=arm, key=retain,
                                      parameter_state=state, backward=backward))
            retention_seconds = time.perf_counter() - retain_tick
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        self.timings['teacher'].append(dict(arm=arm, n=row['n'], backward=backward,
            retained=bool(retain), seconds=elapsed, retention_seconds=retention_seconds,
            auxiliary_probe_seconds=probe_seconds,
            training_work_seconds=elapsed-retention_seconds-probe_seconds))
        return dict(index=index, arm=arm, n=row['n'], mean_ce=evidence['loss'],
            auxiliary_loss=float(aux.detach()), total_loss=float(total.detach()),
            backward_loss=float(total.detach()) / accumulation, target_rows=len(case['target_ids']),
            gradients=gradients, seconds=elapsed, retained=descriptor,
            logits=result['logits'].detach().cpu() if not backward else None)

    def step(self, optimizer, arm, update, gradient_file=None):
        torch = self.torch
        tick = time.perf_counter()
        gradient_save_seconds = 0.
        if gradient_file:
            torch.save({name: p.grad.detach().cpu().clone() for name, p in self.params.items()}, gradient_file)
            gradient_save_seconds = time.perf_counter() - tick
        norm = torch.nn.utils.clip_grad_norm_(list(self.params.values()), 1.)
        need(bool(norm.isfinite()), 'Nonfinite accumulated gradient norm')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.counts['optimizer'] += 1
        need(all(bool(v.isfinite().all()) for v in self.params.values()), 'Updated adapter nonfinite')
        steps = [float(optimizer.state[v]['step']) for v in self.params.values()]
        need(set(steps) == {float(update)}, 'AdamW update count differs')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        self.timings['optimizer'].append(elapsed-gradient_save_seconds)
        self.guard()
        return dict(arm=arm, update=update, gradient_norm=float(norm), parameter_steps=steps,
            seconds=elapsed, gradient_save_seconds=gradient_save_seconds,
            gradients=None if gradient_file is None else ref(gradient_file))

    def generate(self, index, arm, state, key):
        from scripts import mmred_official_answer as answer
        torch = self.torch
        need(not self.capture_active, 'Auxiliary hook cannot be installed during generation')
        need(not self.model.training and all(not p.requires_grad and p.grad is None for p in self.params.values()),
             'Natural generation requires frozen adapter and empty gradients')
        torch.cuda.synchronize()
        tick = time.perf_counter()
        case, features = self.training.load_inputs(torch, self.model, self.items[index])
        evidence = dict(parameter_state=state, feature_packet=self.items[index]['features'],
                        compact_packet=self.items[index]['compact'], prefix_supervision_active=False)
        result = self.profile.ordinary_generate(torch, self.model, self.loaded.processor, case, features, evidence)
        ids = result['generated_ids']
        content = ids[:-1] if ids[-1] in answer.EOS_IDS else ids
        text = self.loaded.processor.tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        score = answer.score_answer(text, case['metadata']['atype'], json.loads(case['target_text'])['answer'], ids)
        descriptor = self.profile.retain(torch, self.data / (key + '.pt'), index, arm, 'cold', evidence,
                                          result=result, primary_text=text, score=score)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        record = dict(descriptor, index=index, arm=arm, sid=self.items[index]['row']['sid'],
            n=self.items[index]['row']['n'], qtype=self.items[index]['row']['qtype'],
            parameter_state=state, generated_ids=ids, primary_text=text, score=score,
            seconds=elapsed, auxiliary_hook_installed=False)
        self.naturals.append(record)
        self.timings['generation'].append(dict(n=record['n'], tokens=len(ids), seconds=elapsed))
        self.guard()
        return record

    def finish(self, extra):
        self.hooks.close()
        self.guard()
        need(not self.capture_active and self.counts['vision'] == 0, 'Native inference/no-extra-vision invariant failed')
        for key in ('backbone', 'language', 'norm', 'head'):
            need(self.counts[key] == self.counts['model'], 'One complete native path per call required')
        need(self.layers == [self.counts['model']] * 28, 'No replay or hidden extra decoder steps allowed')
        save(self.out / 'base_after.json', self.p.base_metadata(self.base_refs))
        need(read(self.out / 'base_before.json') == read(self.out / 'base_after.json'), 'Frozen base parameters changed')
        analysis = dict(protocol=PROTOCOL, policy=POLICY, completed=True, passed=True,
            initial=self.initial_ref, states=self.states, retained_teachers=self.retained, natural=self.naturals,
            counters=dict(self.counts), decoder_layer_calls=self.layers, timings=self.timings,
            setup_seconds=self.setup_seconds, elapsed_seconds=time.perf_counter() - self.started,
            peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=self.torch.cuda.max_memory_reserved(), **extra)
        save(self.out / 'analysis.json', analysis)
        manifest = {str(path): sha(path) for root in (self.out, self.data, self.ckpt)
                    for path in root.iterdir() if path.is_file()}
        save(self.out / 'artifacts.json', manifest)
        return dict(analysis=ref(self.out / 'analysis.json'), artifacts=ref(self.out / 'artifacts.json'),
                    counters=dict(self.counts), **extra)


def software_profile(run):
    torch = run.torch
    parity, baseline, final = [], {}, {}
    run.reset(25, training=False)
    for index in ROUNDTRIP:
        with torch.no_grad(): baseline[index] = run.teacher(index, 'answer', run.initial_ref,
            retain=f'baseline_{index}', capture=False)['logits']
        run.generate(index, 'answer', run.initial_ref, f'baseline_natural_{index}')
    for arm in ARMS:
        run.reset(25, training=False)
        for index in ROUNDTRIP:
            with torch.no_grad(): observed = run.teacher(index, arm, run.initial_ref,
                retain=f'{arm}_initial_{index}')['logits']
            equal = torch.equal(baseline[index], observed)
            parity.append(dict(arm=arm, index=index, logits_bit_equal=equal))
            need(equal, 'Passive supervision capture must preserve native logits exactly')
        run.reset(25, training=True)
        optimizer = run.optimizer()
        optimizer.zero_grad(set_to_none=True)
        state = run.state(arm, 25, 0, optimizer)
        for update in (1, 2):
            if update == 2: state = run.state(arm, 25, 1, optimizer)
            for j, index in enumerate(WITNESSES):
                run.teacher(index, arm, state, accumulation=12, backward=True,
                    retain=f'{arm}_update{update}_micro{j}',
                    auxiliary_probe=update == 1 and j == 0 and arm != 'answer')
            gradfile = run.data / f'{arm}_update{update}_gradients.pt'
            record = run.step(optimizer, arm, update, gradfile)
            save(run.out / f'{arm}_update{update}.json', record)
        final[arm] = run.state(arm, 25, 2, optimizer)
        run.profile.model_mode(run.model, run.params, False)
        before = {}
        for index in ROUNDTRIP:
            with torch.no_grad(): before[index] = run.teacher(index, arm, final[arm],
                retain=f'{arm}_roundtrip_before_{index}')['logits']
        run.profile.restore(torch, run.params, run.initial['adapter'])
        restored = torch.load(verify_ref(final[arm]), map_location='cpu', weights_only=True)
        run.profile.restore(torch, run.params, restored['adapter'])
        for index in ROUNDTRIP:
            with torch.no_grad(): after = run.teacher(index, arm, final[arm],
                retain=f'{arm}_roundtrip_after_{index}')['logits']
            need(torch.equal(before[index], after), 'Checkpoint reload changed native teacher logits')
            run.generate(index, arm, final[arm], f'{arm}_final_natural_{index}')
        del optimizer
    generated = sum(len(r['generated_ids']) for r in run.naturals)
    need(run.counts['model'] == 92 + generated and run.counts['backward'] == 72
         and run.counts['optimizer'] == 6 and run.counts['auxiliary_gradient_probes'] == 2
         and len(run.naturals) == 8 and len(run.retained) == 92, 'Complete software profile inventory differs')
    projections = {}
    for arm in ARMS:
        by_n = {n: max(r['training_work_seconds'] for r in run.timings['teacher']
                        if r['arm'] == arm and r['backward'] and r['n'] == n) for n in (1,2,4,8,16)}
        train = sum(2400 * by_n[n] for n in by_n)
        update = 1500 * max(run.timings['optimizer'])
        checkpoints = 4 * max(run.timings['checkpoint'])
        # Conservatively price 100 maximum-length responses using observed
        # per-token generation work, without counting saved-audit throughput as production latency.
        generation = 100 * 50 * max(r['seconds'] / r['tokens'] for r in run.timings['generation'])
        retained = 6 * max(r['retention_seconds'] for r in run.timings['teacher'])
        estimate = run.setup_seconds + 1.25 * (train + update + checkpoints + generation + retained + 4*max(by_n.values())) + 60
        projections[arm] = dict(seconds=estimate, per_n_micro_seconds=by_n,
            training_seconds=train, optimizer_seconds=update, checkpoint_seconds=checkpoints,
            retained_teacher_seconds=retained,
            training_diagnostic_seconds=generation, cap_seconds=7200, passed=estimate <= 7200,
            conservative_instrumented_forecast=True, empirical_not_guarantee=True)
    return run.finish(dict(phase='profile', parity=parity, final_checkpoints=final,
        future_main_projection=projections, no_fit_release=True,
        no_validation_test_inference=True, objective_achieved=False))


def main_fit(run, task_index):
    torch = run.torch
    arm, seed = ARMS[task_index % 3], SEEDS[task_index // 3]
    verify_main_release(run.args)
    run.reset(seed, training=True)
    optimizer = run.optimizer()
    optimizer.zero_grad(set_to_none=True)
    initial = run.state(arm, seed, 0, optimizer)
    order = training_order(seed)
    save(run.out / 'order.json', order)
    trace, updates_path = run.out / 'training.jsonl', run.out / 'updates.jsonl'
    state = initial
    with trace.open('x') as stream, updates_path.open('x') as update_stream:
        for entry in order:
            m, index = entry['micro_index'], entry['index']
            if m == 11992: state = run.state(arm, seed, 1499, optimizer)
            descriptor = state if m in (0,11999) else dict(updates=m//8, arm=arm, seed=seed)
            before = list(run.layers)
            record = run.teacher(index, arm, descriptor, backward=True,
                retain=f'training_{m}' if m in (0,11999) else None,
                capture=arm != 'answer' or m in (0,11999))
            need([a-b for a,b in zip(run.layers,before)] == [1]*28, 'One native forward per presentation required')
            record.pop('logits')
            # Per-tensor booleans are audited on retained calls; retain compact full-trace group information.
            grads = record.pop('gradients')
            record.update(entry, seed=seed, gradients_all_present=grads['all_present'],
                          gradients_all_finite=grads['all_finite'], gradient_layer_l2=grads['layer_l2'])
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            if (m+1)%8 == 0:
                update = (m+1)//8
                result = run.step(optimizer, arm, update)
                update_stream.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
                if update%50 == 0:
                    stream.flush(); update_stream.flush()
                    print(json.dumps(dict(arm=arm,seed=seed,update=update,mean_ce=record['mean_ce'],
                        auxiliary_loss=record['auxiliary_loss'],elapsed=time.perf_counter()-run.started)),flush=True)
    final = run.state(arm, seed, 1500, optimizer)
    run.profile.model_mode(run.model, run.params, False)
    before = {}
    for index in ROUNDTRIP:
        with torch.no_grad(): before[index] = run.teacher(index, arm, final,
            retain=f'roundtrip_before_{index}')['logits']
    run.profile.restore(torch, run.params, run.initial['adapter'])
    restored = torch.load(verify_ref(final),map_location='cpu',weights_only=True)
    run.profile.restore(torch, run.params, restored['adapter'])
    for index in ROUNDTRIP:
        with torch.no_grad(): after = run.teacher(index, arm, final,
            retain=f'roundtrip_after_{index}')['logits']
        need(torch.equal(before[index],after), 'Final checkpoint reload differs')
    for index in run.plan['diagnostic_indices']:
        run.generate(index, arm, final, f'natural_{index}')
    generated = sum(len(row['generated_ids']) for row in run.naturals)
    need(run.counts['model'] == 12004+generated and run.counts['backward'] == 12000
         and run.counts['optimizer'] == 1500 and len(run.naturals) == 100
         and len(run.retained) == 6, 'Complete matched fit inventory differs')
    return run.finish(dict(phase='run',arm=arm,seed=seed,task_index=task_index,
        initial_continuation_checkpoint=initial,final_checkpoint=final,
        order=ref(run.out/'order.json'),training_trace=ref(trace),updates_trace=ref(updates_path),
        training_diagnostic_correct=sum(r['score']['correct'] for r in run.naturals),
        no_validation_test_inference=True,requires_independent_audit=True,objective_achieved=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--profile',action='store_true'); mode.add_argument('--run',action='store_true')
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--main-release',type=Path)
    parser.add_argument('--task-index',type=int,choices=range(6))
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,
         'All model/tensor work requires four-core GPU Slurm')
    if args.run:
        need(args.main_release and args.task_index is not None
             and os.environ.get('SLURM_ARRAY_TASK_ID')==str(args.task_index), 'Released matched-array task required')
        verify_main_release(args)
    started=time.perf_counter()
    tag=f'profile_{os.environ["SLURM_JOB_ID"]}' if args.profile else f'run_{os.environ["SLURM_ARRAY_JOB_ID"]}_{args.task_index}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False)
    release=verify_release(args.release)
    (out/'source').mkdir()
    for name,digest in release['source_sha256'].items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes())
        need(sha(path)==digest,'Source archive differs')
    save(out/'request.json',dict(protocol=PROTOCOL,policy=POLICY,phase='profile' if args.profile else 'run',
        source_release=ref(args.release),targets=ref(args.targets),task_index=args.task_index,
        main_release=None if args.profile else ref(args.main_release)))
    runner=None
    try:
        runner=Runner(args,out,started)
        result=software_profile(runner) if args.profile else main_fit(runner,args.task_index)
        verify_release(args.release)
        save(out/'summary.json',dict(result,protocol=PROTOCOL,passed=True,completed=True,
            source_release=ref(args.release),targets=ref(args.targets),
            main_release=None if args.profile else ref(args.main_release),
            elapsed_seconds=time.perf_counter()-started))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,
            counters={} if runner is None else dict(runner.counts),
            retained=[] if runner is None else runner.retained,source_release=ref(args.release),
            partial_evidence_preserved=True))
        raise


if __name__=='__main__':main()
