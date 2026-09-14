"""Paired V9 cached-state training with unchanged native parallel evaluation.

Both conditions use the same scenes, pair order, initial parameters and native
count/EOS cross-entropy. The sole objective difference is endpoint residual consistency versus the
native path bound, both with coefficient one. Equal coefficients do not imply
equal effective regularization strength. The auxiliary loss is absent at inference.
All tensor, tokenizer, data-audit and model work requires a Slurm allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import train_native_vision_v7 as v7
from scripts import native_vision_v7_runtime as native
from scripts.stage_native_vision_v7_features import MODEL, need, read, sha, object_sha

DATA = Path('/mnt/data/gabriele/gnn_transformer')
OUT = REPO / 'outputs/native_aggregation_vlm/v9'
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v9')
TRAIN = DATA / 'v7_balanced/main_manifest.json'
SCHEDULE = DATA / 'v7_balanced/schedule.json'
FRESH = DATA / 'v9_fresh'
POLICY = dict(
    protocol='v9_paired_native_path_bound', native_arm='parallel',
    conditions=['residual', 'path'], seeds=[12, 13], epochs=40,
    pair_slots_per_epoch=972, scene_slots_per_epoch=1944,
    scene_presentations=77760, pair_presentations=38880, batch_size=16,
    pairs_per_batch=8, steps=4860, dev_steps=[972, 1944, 2916, 3888, 4860],
    lr=.001, warmup=50, final_lr=.00001, weight_decay=0., clip_norm=1.,
    rank=96, hidden_size=3584, parameters=1041600, merge='sum', post_activation='silu',
    native_dtype='torch.float16', branch_dtype='torch.float32',
    regularizer_coefficients={'residual': 1., 'path': 1.}, consistency_epsilon=1e-6,
    path_reduction='mean_B_squared_over_frozen_global_squared_L2_plus_epsilon',
    path_displacement='aggregate_projection.weight_times_z16_minus_z8_no_bias',
    bound_squared_factor=1.21, bound_descriptive_atol=1e-7, bound_descriptive_rtol=1e-4,
    bound_scope='FP32_branch_residual_along_observed_chords_only_no_accuracy_certificate',
    equal_coefficients_do_not_match_effective_regularization_strength=True,
    consistency_reduction='mean_over_pairs_and_separate_count_EOS_positions_of_squared_L2_ratio',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',
    target_positions=['count', 'EOS'], ce_reduction='equal_mean_over_16_scenes_and_two_positions',
    max_new_tokens=4, exact_requires_eos=True,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',
    native_eos=[151645, 151643], target_eos=151645, profile_steps=32,
)
TEST_FILES = ('tests/test_aggregation_consistency.py', 'tests/test_aggregation_path_bound.py',
              'tests/test_native_vision_v9_pairs.py')
OWN = tuple(sorted(set(p for p in v7.OWN if not p.startswith('slurm/')) | {
    'scripts/train_native_vision_v9.py', 'scripts/native_vision_v9_pairs.py',
    'gnnformer/aggregation_consistency.py', 'gnnformer/aggregation_path_bound.py', *TEST_FILES,
    'slurm/native_vision_v9_train_check.sbatch',
    'slurm/native_vision_v9_train_profile.sbatch', 'slurm/native_vision_v9_train.sbatch',
}))
save = v7.save


def sources():
    return {name: sha(REPO / name) for name in OWN}


def snapshot(out):
    (out / 'code').mkdir()
    for name in OWN:
        (out / 'code' / name.replace('/', '_')).write_bytes((REPO / name).read_bytes())
    save(out / 'source_hashes.json', sources())


def index(out, title):
    (out / 'INDEX.md').write_text('# ' + title + '\n\n'
        '[Plan or configuration](plan.json) · [Summary](summary.json) · '
        '[Source hashes](source_hashes.json)\n')


def bind_file(bindings, path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    need(expected is None or digest == expected, 'Referenced artifact changed: ' + str(path))
    need(str(path) not in bindings or bindings[str(path)] == digest, 'Inconsistent artifact binding')
    bindings[str(path)] = digest
    return {'file': str(path), 'sha256': digest}


def bind_manifest(bindings, path):
    bind_file(bindings, path)
    manifest = read(path)
    need(manifest['schema_version'] == 1, 'Manifest schema differs')
    for field in ('audit', 'inventory', 'stage_plan'):
        if field + '_file' in manifest:
            bind_file(bindings, manifest[field + '_file'], manifest[field + '_sha256'])
    if 'stage_audit_file' in manifest:
        bind_file(bindings, manifest['stage_audit_file'])
    for source, digest in manifest.get('generator_code_sha256', {}).items():
        bind_file(bindings, REPO / source, digest)
    return manifest


def test_helpers():
    results = []
    for name in TEST_FILES:
        result = subprocess.run([sys.executable, str(REPO / name)], capture_output=True, text=True)
        results.append(dict(file=name, sha256=sha(REPO / name), returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        need(result.returncode == 0, 'CPU helper tests failed: ' + name + '\n' + result.stderr)
    return dict(passed=True, suites=results)


def validate_fresh(bindings, train_manifest):
    contents = set()
    train_dev = {r['content_sha256'] for split in train_manifest['splits'].values()
                 for r in split['samples']}
    for purpose, expected in (('main', {'test_N16': 108, 'test_N32': 108, 'test_N64': 108}),
                              ('count', {'test_N32': 64, 'test_N64': 64})):
        manifest = bind_manifest(bindings, FRESH / (purpose + '_manifest.json'))
        need(Path(manifest['dataset_root']).resolve() == FRESH and manifest['purpose'] == purpose,
             'Fresh evaluation manifest identity differs')
        need({k: len(v['samples']) for k, v in manifest['splits'].items()} == expected,
             'Fresh evaluation cells differ')
        audit = read(manifest['audit_file'])
        need(all(audit.get(k) is True for k in (
            'all_fresh_content_exclusions_passed', 'all_generated_semantics_and_extensions_passed',
            'all_image_hashes_and_dimensions_passed', 'all_image_semantic_links_passed',
            'all_pair_extensions_passed', 'all_qa_hashes_and_gold_recounts_passed',
            'training_dev_and_schedule_unchanged')), 'Fresh semantic audit did not pass')
        for cell, split in manifest['splits'].items():
            ks = Counter(r['gold'] for r in split['samples'])
            wanted = {k: (12 if purpose == 'main' else 8) for k in (range(9) if purpose == 'main' else range(9, 17))}
            need(dict(ks) == wanted, 'Fresh gold balance differs')
            for row in split['samples']:
                need(row['n_frames'] == int(cell.split('_N')[1]), 'Fresh N/cell mismatch')
                need(row['content_sha256'] not in contents | train_dev, 'Fresh context overlaps another selected context')
                contents.add(row['content_sha256'])
    need(len(contents) == 452, 'Fresh evaluation cardinality differs')


def bind_native_profile(bindings, directory):
    directory = Path(directory).resolve()
    summary_path = directory / 'summary.json'
    binding = bind_file(bindings, summary_path)
    summary = read(summary_path)
    need(summary['completed'] and summary['computational_integrity_passed']
         and summary['zero_identity_passed'], 'Native integration profile failed')
    bind_file(bindings, summary['plan_file'], summary['plan_sha256'])
    sidecar = Path(summary['plan_file']).with_suffix('.sha256')
    need(sidecar.read_text().strip() == summary['plan_sha256'], 'Native plan sidecar changed')
    bind_file(bindings, sidecar)
    for source, digest in summary['source_sha256'].items():
        bind_file(bindings, REPO / source, digest)
    for item in summary['files'].values():
        bind_file(bindings, item['path'], item['sha256'])
    return dict(directory=str(directory), **binding,
                strict_numerical_gates_passed=summary['strict_numerical_gates_passed'])


def check(args):
    out = OUT / ('train_check_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out)
    index(out, 'V9 paired training CPU plan')
    import torch
    from transformers import AutoProcessor
    from scripts.native_vision_v9_pairs import load_pairs, presentation_order
    torch.set_num_threads(4)
    begin = time.perf_counter()
    helper_tests = test_helpers()
    bindings = {}
    train_manifest = bind_manifest(bindings, TRAIN)
    bind_file(bindings, SCHEDULE)
    need(len(train_manifest['splits']['dev_N16']['samples']) == 72, 'Native dev set must remain the same 72 N16 scenes')
    validate_fresh(bindings, train_manifest)
    cache_path = args.parallel_cache.resolve()
    cache_binding = bind_file(bindings, cache_path)
    cache = v7.cache_binding(cache_path)
    bind_file(bindings, cache['plan_file'], cache['plan_sha256'])
    need(read(cache['plan_file'])['protocol'] == 'v7_parallel_local_training_features', 'Parallel cache required')
    pairing = load_pairs(TRAIN, cache_path, SCHEDULE)
    pairs = pairing['pairs']
    need(len(pairs) == 972, 'Exactly 972 pair slots required')
    processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True, use_fast=False)
    for scene in cache['scenes'].values():
        need(scene['target_ids'] == native.encode_target(processor.tokenizer, scene['gold']), 'Frozen native count/EOS targets differ')
    states, feature_index = v7.load_features(torch, cache, 'cpu')
    for pair in pairs:
        left, right = (cache['scenes'][sid] for sid in pair['sids'])
        need(left['global_feature_ids'] == right['global_feature_ids'], 'Paired causal global identities differ')
        left_g = states[[feature_index[f] for f in left['global_feature_ids']]]
        right_g = states[[feature_index[f] for f in right['global_feature_ids']]]
        need(torch.equal(left_g, right_g) and not left_g.requires_grad, 'Paired frozen global tensors differ')
    for path, digest in {r['file']: r['file_sha256'] for r in cache['features'].values()}.items():
        bind_file(bindings, path, digest)
    orders = {str(seed): object_sha(presentation_order(pairs, seed)) for seed in POLICY['seeds']}
    need(len(set(orders.values())) == 2, 'Distinct registered seeds yielded identical orders')
    native_profile = bind_native_profile(bindings, args.native_profile)
    diagnostic = bind_diagnostic(bindings, args.diagnostic_summary, cache_path, args.prior_finalization)
    save(out / 'pairing.json', pairing)
    plan = dict(schema_version=1, policy=POLICY, source_sha256=sources(),
        artifact_bindings=bindings, cache_binding=cache_binding,
        pairing_file=str(out / 'pairing.json'), pairing_sha256=sha(out / 'pairing.json'),
        pairing_object_sha256=object_sha(pairing), order_sha256=orders,
        train_manifest=str(TRAIN), schedule_file=str(SCHEDULE),
        fresh_manifests={p: str(FRESH / (p + '_manifest.json')) for p in ('main', 'count')},
        native_profile=native_profile, diagnostic_summary=diagnostic,
        tests=helper_tests, actual_frozen_pair_equality_passed=True,
        no_model_loaded=True, elapsed_seconds=time.perf_counter() - begin)
    save(out / 'plan.json', plan)
    (out / 'plan.sha256').write_text(sha(out / 'plan.json') + '\n')
    verify_plan(out / 'plan.json')
    save(out / 'summary.json', dict(passed=True, plan_file=str(out / 'plan.json'),
        plan_sha256=sha(out / 'plan.json'), source_sha256=sources(), tests=helper_tests,
        no_model_loaded=True, elapsed_seconds=time.perf_counter() - begin))
    index(out, 'V9 paired training CPU plan')
    print(json.dumps(dict(passed=True, plan_file=str(out / 'plan.json'))), flush=True)


def verify_plan(path):
    path = Path(path).resolve()
    need(sha(path) == path.with_suffix('.sha256').read_text().strip(), 'V9 plan sidecar differs')
    plan = read(path)
    need(plan['schema_version'] == 1 and plan['policy'] == POLICY and plan['source_sha256'] == sources(),
         'Frozen V9 source/policy differs')
    for artifact, digest in plan['artifact_bindings'].items():
        need(sha(artifact) == digest, 'Frozen artifact changed: ' + artifact)
    need(sha(plan['pairing_file']) == plan['pairing_sha256'], 'Frozen pairing changed')
    from scripts.native_vision_v9_pairs import load_pairs
    rebuilt = load_pairs(plan['train_manifest'], plan['cache_binding']['file'], plan['schedule_file'])
    need(object_sha(rebuilt) == plan['pairing_object_sha256'] == object_sha(read(plan['pairing_file'])),
         'Frozen pairing no longer matches training metadata')
    return plan


def verify_release(path, plan_path, plan):
    need(path is not None, 'A root-reviewed measured main release is required')
    released = read(path)
    need(released['passed'] is True and released['plan_sha256'] == sha(plan_path)
         and released['source_sha256'] == sources(), 'Main resource release differs')
    need(released['per_main_seconds_cap'] == 2700 and released['main_block_gpu_seconds_cap'] == 11700,
         'Main resource envelope differs')
    need(set(released['profiles']) == set(POLICY['conditions']), 'Both condition profiles required')
    data_release = released['data_release']
    need(sha(data_release['file']) == data_release['sha256'], 'Independent report/data release changed')
    report_release = read(data_release['file'])
    need(report_release['passed'] is True and report_release['source_sha256'] == released['report_source_sha256'],
         'Independent report/data release source differs')
    for source, digest in released['report_source_sha256'].items():
        need(sha(REPO / source) == digest, 'Frozen independent reporter changed')
    for artifact, digest in report_release['data_bindings'].items():
        need(sha(artifact) == digest, 'Independent report data binding changed')
    for condition, item in released['profiles'].items():
        summary_path = Path(item['directory']) / 'summary.json'
        need(sha(summary_path) == item['summary_sha256'], 'Training profile changed')
        profile = read(summary_path)
        need(profile['profile'] and profile['condition'] == condition and profile['seed'] == 12
             and profile['passed'] and profile['computational_integrity_passed']
             and profile['no_dev_or_test_evaluation'] and profile['steps'] == 32
             and profile['plan_sha256'] == sha(plan_path) and profile['source_sha256'] == sources()
             and profile['active_cache_replay']['passed'], 'Unmatched or failed training profile')
        zero = profile['zero_initialization_auxiliary_gradients']
        need(all(zero.get(key) is True for key in ('passed', 'up_exactly_zero',
             'all_residual_gradients_zero', 'all_path_gradients_zero'))
             and zero['residual_loss'] == zero['path_loss'] == 0., 'Profile zero-U auxiliary gate failed')
    need(set(released['projections']) == set(POLICY['conditions']), 'Both measured cost projections required')
    need(all(v['passed'] is True and 0 < v['projected_seconds'] <= 2700
             for v in released['projections'].values()), 'Measured main runtime projection failed')
    return dict(file=str(Path(path).resolve()), sha256=sha(path))


def losses(torch, branch, local, global_states, targets, norm, head, condition):
    from gnnformer.aggregation_consistency import paired_residual_consistency
    from gnnformer.aggregation_path_bound import native_path_aggregate, paired_native_path_bound
    need(condition in POLICY['conditions'], 'Unknown regularizer')
    need(tuple(global_states.shape) == (32, 3584) and tuple(targets.shape) == (32,),
         'Training must contain 16 scenes with separate count/EOS states')
    # This is the exact unchanged deployed branch.forward used for native CE.
    delta = branch(local, global_states, output_dtype=torch.float32)
    consistency = paired_residual_consistency(delta.reshape(16, 2, 3584),
        global_states.reshape(16, 2, 3584), eps=POLICY['consistency_epsilon'])
    # Both conditions compute this same extra training-only graph.
    z = native_path_aggregate(branch, local, global_states).reshape(16, 2, 96)
    path_loss = paired_native_path_bound(z, branch.aggregate_projection.weight, branch.up.weight,
        global_states.reshape(16, 2, 3584), eps=POLICY['consistency_epsilon'])
    logits = head(norm((global_states + delta.to(global_states.dtype)).unsqueeze(0)))[0]
    token_ce = torch.nn.functional.cross_entropy(logits.float(), targets, reduction='none')
    ce = token_ce.mean()
    with torch.no_grad():
        ce_positions = token_ce.reshape(16, 2).mean(0)
        paired_delta = delta.detach().reshape(16, 2, 3584)
        paired_global = global_states.detach().float().reshape(16, 2, 3584)
        g_squared = paired_global[0::2].square().sum(-1)
        denominator = g_squared + POLICY['consistency_epsilon']
        residual_difference = paired_delta[1::2] - paired_delta[0::2]
        residual_norm = torch.linalg.vector_norm(residual_difference, dim=-1)
        residual_penalties = residual_difference.square().sum(-1) / denominator
        displacement = torch.nn.functional.linear(z[1::2] - z[0::2], branch.aggregate_projection.weight)
        displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
        B = (displacement.abs() * torch.linalg.vector_norm(branch.up.weight, dim=0)).sum(-1)
        path_penalties = B.square() / denominator
        bounded = POLICY['bound_squared_factor'] * path_penalties
        holds = residual_penalties <= bounded + POLICY['bound_descriptive_atol'] + POLICY['bound_descriptive_rtol'] * bounded
        residual_scale = torch.linalg.vector_norm(paired_delta, dim=-1) / torch.sqrt(
            paired_global.square().sum(-1) + POLICY['consistency_epsilon'])
        need(all(bool(torch.isfinite(x).all()) for x in (B, residual_norm, displacement_norm,
             g_squared, residual_penalties, path_penalties, residual_scale)), 'Nonfinite loss diagnostic')
        residual_positions, path_positions = residual_penalties.mean(0), path_penalties.mean(0)
        diagnostics = dict(B_norms=B.cpu().tolist(), residual_difference_norms=residual_norm.cpu().tolist(),
            preactivation_difference_norms=displacement_norm.cpu().tolist(),
            frozen_global_squared_norms=g_squared.cpu().tolist(),
            residual_pair_position_losses=residual_penalties.cpu().tolist(),
            path_pair_position_losses=path_penalties.cpu().tolist(), bound_inequality_holds=holds.cpu().tolist(),
            B_mean=float(B.mean()), B_max=float(B.max()),
            residual_difference_norm_mean=float(residual_norm.mean()), residual_difference_norm_max=float(residual_norm.max()),
            preactivation_difference_norm_mean=float(displacement_norm.mean()), preactivation_difference_norm_max=float(displacement_norm.max()),
            bound_violation_count=int((~holds).sum()), bound_inequality_all_hold=bool(holds.all()),
            residual_scale_mean=float(residual_scale.mean()))
    components = dict(ce_count_loss=float(ce_positions[0]), ce_eos_loss=float(ce_positions[1]),
        consistency_count_loss=float(residual_positions[0]), consistency_eos_loss=float(residual_positions[1]),
        path_count_loss=float(path_positions[0]), path_eos_loss=float(path_positions[1]),
        regularizer_diagnostics=diagnostics)
    selected = consistency if condition == 'residual' else path_loss
    total = ce + POLICY['regularizer_coefficients'][condition] * selected
    need(all(bool(torch.isfinite(x)) for x in (ce, consistency, path_loss, total)), 'Nonfinite training objective')
    return total, ce, consistency, path_loss, components


def zero_auxiliary_gradients(torch, branch, consistency, path_loss):
    """Exact zero-U start; autograd.grad leaves optimizer .grad buffers untouched."""
    need(bool((branch.up.weight == 0).all()), 'Profile must start from exact zero U')
    need(float(consistency) == float(path_loss) == 0., 'Initial auxiliary losses must be zero')
    flags = {}
    for label, value in (('residual', consistency), ('path', path_loss)):
        gradients = torch.autograd.grad(value, tuple(branch.parameters()), retain_graph=True, allow_unused=True)
        flags['all_' + label + '_gradients_zero'] = all(
            gradient is None or bool(torch.isfinite(gradient).all() and (gradient == 0).all())
            for gradient in gradients)
    need(all(flags.values()) and all(p.grad is None for p in branch.parameters()),
         'Auxiliary zero-U gradients must vanish and preserve .grad buffers')
    return dict(passed=True, residual_loss=0., path_loss=0., up_exactly_zero=True, **flags)


def evaluate(torch, model, processor, branch, rows, out, label, dataout):
    """Unchanged native parallel generation; archive all raw logits in FP32."""
    records, raw = [], []
    begin = time.perf_counter()
    for cell, sample in rows:
        bundle = native.prepare_scene(processor, sample, 'parallel')
        result = native.generate_native(model, processor, branch, bundle)
        records.append(v7.prediction_record(torch, processor, sample, result, len(raw), cell))
        raw.append(result['raw_logits'].detach().to(device='cpu', dtype=torch.float32))
        if len(records) % 36 == 0:
            print(json.dumps(dict(evaluation=label, processed=len(records), total=len(rows))), flush=True)
    rawfile = dataout / (label + '_raw.pt')
    torch.save(dict(schema_version=1, raw_logits=raw), rawfile)
    result = dict(label=label, rows=records, n=len(records),
        exact_count=sum(r['exact'] for r in records),
        first_token_nll=sum(r['first_token_nll'] for r in records) / len(records),
        seconds=time.perf_counter() - begin, raw_file=str(rawfile), raw_sha256=sha(rawfile),
        raw_dtype='torch.float32')
    save(out / (label + '.json'), result)
    return result


def bind_diagnostic(bindings, path, cache_path, finalization_path):
    summary_binding = bind_file(bindings, path)
    summary = read(path)
    need(summary.get('passed') is True and summary.get('diagnostic_only') is True
         and summary.get('no_fit') is True and summary.get('vlm_forward_calls') == 0
         and summary.get('synthetic_points') == 432, 'A completed 432-point V8 CPU diagnostic is required')
    bind_file(bindings, summary['analysis_file'], summary['analysis_sha256'])
    bind_file(bindings, summary['raw_file'], summary['raw_sha256'])
    for source, digest in summary['source_sha256'].items():
        bind_file(bindings, REPO / source, digest)
    analysis = read(summary['analysis_file'])
    bind_file(bindings, analysis['report_file'], analysis['report_sha256'])
    bind_file(bindings, analysis['report_completion_file'], analysis['report_completion_sha256'])
    report, completion = read(analysis['report_file']), read(analysis['report_completion_file'])
    need(report['passed'] and report['audit_passed'] and completion['passed']
         and completion['analysis_sha256'] == analysis['report_sha256']
         and Path(completion['analysis_file']).resolve() == Path(analysis['report_file']).resolve(),
         'V8 report completion/analysis identity failed')
    for source, digest in report['source_sha256'].items():
        bind_file(bindings, REPO / source, digest)
    cache_binding = analysis['training_cache']
    bind_file(bindings, cache_binding['file'], cache_binding['sha256'])
    bind_file(bindings, analysis['training_plan_file'], analysis['training_plan_sha256'])
    need(Path(cache_binding['file']).resolve() == Path(cache_path).resolve(),
         'Response-surface diagnostic used a different frozen training cache')
    need(len(analysis['selected_checkpoints']) == 4, 'Need all four V8 selected cores')
    for selected in analysis['selected_checkpoints']:
        run = Path(selected['run_directory'])
        bind_file(bindings, run / 'config.json', selected['config_sha256'])
        bind_file(bindings, run / 'summary.json', selected['summary_sha256'])
        bind_file(bindings, selected['checkpoint'], selected['checkpoint_sha256'])
    final_binding = bind_file(bindings, finalization_path)
    finalized = read(finalization_path)
    need(finalized['passed'] and finalized['finalized'], 'V8 finalization did not complete verification')
    bind_file(bindings, finalized['final_acceptance_file'], finalized['final_acceptance_sha256'])
    acceptance = read(finalized['final_acceptance_file'])
    need(acceptance['completed'] and acceptance['verification_passed'], 'V8 final report/post verification failed')
    for item in acceptance['artifacts'].values():
        bind_file(bindings, item['path'], item['sha256'])
    need(acceptance['artifacts']['analysis'] == dict(path=analysis['report_file'], sha256=analysis['report_sha256']),
         'Finalized V8 report differs from diagnostic ancestor')
    for source, digest in finalized['source_sha256'].items():
        bind_file(bindings, REPO / source, digest)
    return dict(**summary_binding, finalization=final_binding, software_completion_passed=True,
                selection_on_diagnostic_predictions=False)


def run(args):
    job = os.environ['SLURM_JOB_ID']
    runid = f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out = OUT / runid
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out)
    (out / 'INDEX.md').write_text('# V9 paired aggregation run\n\n'
        '[Configuration](config.json) · [Training](training.json) · [Summary](summary.json)\n')
    import torch
    import transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.native_vision_v9_pairs import presentation_order
    from scripts.stage_native_vision_v7_features import runtime_identity, model_metadata, native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4)
    start = time.perf_counter()
    args.plan = args.plan.resolve()
    plan = verify_plan(args.plan)
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds'], 'Unregistered condition/seed')
    need(not args.profile or args.seed == 12, 'Profiles use only registered seed 12')
    release = None if args.profile else verify_release(args.main_release, args.plan, plan)
    coefficient = POLICY['regularizer_coefficients'][args.condition]
    cache = v7.cache_binding(plan['cache_binding']['file'])
    dataout = DATA / 'native_aggregation_vlm_v9' / runid
    dataout.mkdir(parents=True, exist_ok=False)
    ckpt = CKPT / runid
    ckpt.mkdir(parents=True, exist_ok=False)
    loadstart = time.perf_counter()
    runtime = load_runtime(str(MODEL), use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    model = runtime.model
    model.eval()
    model.requires_grad_(False)
    native.native_contract(model)
    need(runtime_identity() == cache['runtime'] and model_metadata() == cache['model']
         and fingerprint(runtime.processor, str(transformers.__version__)) == cache['processor'],
         'Loaded frozen model/runtime/processor differs from cached-state provenance')
    need(native_api(runtime.processor)[2] == read(cache['plan_file'])['native_api'], 'Native installed code differs')
    need(native.generation_policy(model, runtime.processor.tokenizer)[1]['native_eos_token_ids'] == POLICY['native_eos'],
         'Native EOS contract changed')
    states, feature_index = v7.load_features(torch, cache, runtime.device)
    loadseconds = time.perf_counter() - loadstart
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    branch = ParallelLocalAggregation().to(runtime.device)
    native.native_contract(model, branch)
    initialized = v7.state_info(branch)
    initial = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
    optimizer = torch.optim.AdamW(branch.parameters(), lr=POLICY['lr'], weight_decay=0.)
    pairing = read(plan['pairing_file'])
    order = presentation_order(pairing['pairs'], args.seed)
    need(object_sha(order) == plan['order_sha256'][str(args.seed)], 'Frozen paired presentation order differs')
    save(out / 'presentations.json', order)
    config = dict(run_id=runid, arm='parallel', condition=args.condition, seed=args.seed,
        profile=args.profile, policy=POLICY, regularizer=args.condition, regularizer_coefficient=coefficient,
        plan_file=str(args.plan), plan_sha256=sha(args.plan), source_sha256=sources(),
        cache_binding=plan['cache_binding'], pairing_file=plan['pairing_file'],
        pairing_sha256=plan['pairing_sha256'], diagnostic_summary=plan['diagnostic_summary'],
        initialized=initialized, initialized_sha256=object_sha(initialized),
        presentations_sha256=sha(out / 'presentations.json'), order_sha256=object_sha(order),
        model=cache['model'], runtime=cache['runtime'], processor=cache['processor'],
        native_dtypes=cache['native_dtypes'],
        hardware=dict(name=torch.cuda.get_device_name(0), cuda=torch.version.cuda, torch=str(torch.__version__)),
        model_and_features_load_seconds=loadseconds, checkpoint_directory=str(ckpt),
        data_directory=str(dataout), main_release=release)
    save(out / 'config.json', config)
    (out / 'INDEX.md').write_text('# V9 paired aggregation run\n\n'
        '[Configuration](config.json) · [Training](training.json) · [Summary](summary.json)\n')
    train_manifest = read(plan['train_manifest'])
    devrows = [('dev_N16', r) for r in train_manifest['splits']['dev_N16']['samples']]
    trainlog, developments = [], []
    best = None
    totalsteps = POLICY['profile_steps'] if args.profile else POLICY['steps']
    norm, head = model.model.language_model.norm, model.lm_head
    nonzero_gradient_seen = {name: False for name, _ in branch.named_parameters()}
    first_gradients = []
    zero_initialization = None
    for step in range(1, totalsteps + 1):
        begin = time.perf_counter()
        rows = order[(step - 1) * 16:step * 16]
        need(len(rows) == 16 and all(rows[i]['pair_id'] == rows[i + 1]['pair_id']
             and rows[i]['pair_side'] == 'N8' and rows[i + 1]['pair_side'] == 'N16'
             for i in range(0, 16, 2)), 'A training batch split or reversed a registered pair')
        sids = [r['sid'] for r in rows]
        local, g, targets = v7.batch_states(torch, cache, states, feature_index, sids, 'parallel')
        optimizer.zero_grad(set_to_none=True)
        rate = v7.lr(step)
        for group in optimizer.param_groups:
            group['lr'] = rate
        total, ce, consistency, path_loss, components = losses(torch, branch, local, g, targets, norm, head, args.condition)
        if args.profile and step == 1:
            zero_initialization = zero_auxiliary_gradients(torch, branch, consistency, path_loss)
        selected_regularizer = consistency if args.condition == 'residual' else path_loss
        total.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in branch.parameters()),
             'Missing or nonfinite branch gradient')
        if args.profile or step <= 2:
            for name, parameter in branch.named_parameters():
                nonzero_gradient_seen[name] |= bool((parameter.grad != 0).any())
        if step <= 2:
            first_gradients.append(dict(step=step, pre_clip={name: v7.tensor_info(p.grad)
                for name, p in branch.named_parameters()}))
        gradient_norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), POLICY['clip_norm'])
        need(bool(torch.isfinite(gradient_norm)), 'Nonfinite gradient norm')
        optimizer.step()
        need(all(bool(torch.isfinite(p).all()) for p in branch.parameters()), 'Nonfinite updated branch')
        need(not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'Backbone gradient state changed')
        torch.cuda.synchronize()
        trainlog.append(dict(step=step, lr=rate, loss=float(total), ce_loss=float(ce),
            consistency_loss=float(consistency), path_loss=float(path_loss), regularizer=args.condition,
            regularizer_coefficient=coefficient, selected_regularizer_loss=float(selected_regularizer),
            weighted_regularizer_loss=float(coefficient * selected_regularizer), gradient_norm=float(gradient_norm),
            clipped=float(gradient_norm) > POLICY['clip_norm'], seconds=time.perf_counter() - begin,
            target_ids=targets.detach().cpu().tolist(), sids=sids,
            pair_ids=[rows[i]['pair_id'] for i in range(0, 16, 2)],
            epochs=[rows[i]['epoch'] for i in range(0, 16, 2)], **components))
        if step % 100 == 0:
            print(json.dumps(dict(step=step, ce=float(ce), consistency=float(consistency), path=float(path_loss), total=float(total))), flush=True)
        if not args.profile and step in POLICY['dev_steps']:
            checkpoint = ckpt / f'step_{step}.pt'
            torch.save(dict(branch=branch.state_dict(), step=step, config=config), checkpoint)
            dev = evaluate(torch, model, runtime.processor, branch, devrows, out, f'dev_{step}', dataout)
            entry = dict(step=step, exact_count=dev['exact_count'], nll=dev['first_token_nll'],
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                parameter_sha256=object_sha(v7.state_info(branch)), dev_file=str(out / f'dev_{step}.json'),
                dev_sha256=sha(out / f'dev_{step}.json'))
            developments.append(entry)
            if best is None or (-entry['exact_count'], entry['nll'], step) < (-best['exact_count'], best['nll'], best['step']):
                best = entry
            save(out / 'training.json', trainlog)
            save(out / 'selection.json', dict(development=developments, selected=best))
    save(out / 'training.json', trainlog)
    save(out / 'first_gradients.json', first_gradients)
    if args.profile:
        need(all(nonzero_gradient_seen.values()), 'Profile did not exercise every branch parameter gradient')
        updated = v7.state_info(branch)
        need(updated != initialized, 'Profile updates did not change parameters')
        path = ckpt / 'profile.pt'
        torch.save(branch.state_dict(), path)
        branch.load_state_dict(initial)
        need(v7.state_info(branch) == initialized, 'Initial-state restoration failed')
        branch.load_state_dict(torch.load(path, map_location=runtime.device, weights_only=True))
        need(v7.state_info(branch) == updated, 'Updated-state restoration failed')
        replay = v7.active_cache_replay(torch, model, runtime.processor, branch, cache,
            states, feature_index, train_manifest, 'parallel', dataout)
        save(out / 'active_cache_replay.json', replay)
        need(replay['passed'], 'Active captured-hidden native head replay failed')
        result = dict(config, passed=True, completed=True, computational_integrity_passed=True,
            steps=32, active_cache_replay=replay, no_dev_or_test_evaluation=True,
            zero_initialization_auxiliary_gradients=zero_initialization,
            all_parameter_gradients_exercised=nonzero_gradient_seen,
            frozen_backbone_gradient_state_preserved=True, checkpoint_roundtrip_passed=True,
            profile_checkpoint=str(path), profile_checkpoint_sha256=sha(path),
            step_seconds_max_steady=max(r['seconds'] for r in trainlog[4:]))
    else:
        need(best is not None and len(developments) == 5, 'Missing registered dev selection')
        blob = torch.load(best['checkpoint'], map_location=runtime.device, weights_only=True)
        branch.load_state_dict(blob['branch'])
        need(object_sha(v7.state_info(branch)) == best['parameter_sha256'], 'Selected checkpoint restoration differs')
        testrows = []
        for purpose in ('main', 'count'):
            manifest = read(plan['fresh_manifests'][purpose])
            for cell, split in manifest['splits'].items():
                testrows.extend((purpose + '_' + cell, r) for r in split['samples'])
        need(len(testrows) == 452, 'Incomplete fresh native evaluation')
        evaluate(torch, model, runtime.processor, branch, testrows, out, 'test', dataout)
        result = dict(config, passed=True, completed=True, computational_integrity_passed=True,
            steps=4860, selected=best, development=developments, native_test_count=452,
            test_file=str(out / 'test.json'), test_sha256=sha(out / 'test.json'))
    result.update(training_seconds=sum(r['seconds'] for r in trainlog),
        training_file=str(out / 'training.json'), training_sha256=sha(out / 'training.json'),
        first_gradients_file=str(out / 'first_gradients.json'), first_gradients_sha256=sha(out / 'first_gradients.json'),
        elapsed_seconds=time.perf_counter() - start)
    verify_plan(args.plan)
    if release is not None:
        need(sha(release['file']) == release['sha256'], 'Main release changed during execution')
        verify_release(args.main_release, args.plan, plan)
    save(out / 'summary.json', result)
    print(json.dumps(dict(run_id=runid, completed=True, elapsed_seconds=result['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--profile', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--parallel-cache', type=Path)
    parser.add_argument('--native-profile', type=Path)
    parser.add_argument('--diagnostic-summary', type=Path)
    parser.add_argument('--prior-finalization', type=Path, default=REPO /
        'outputs/native_aggregation_vlm/v8/finalization/final_441998/summary.json')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--main-release', type=Path)
    parser.add_argument('--condition', choices=POLICY['conditions'])
    parser.add_argument('--seed', type=int, choices=POLICY['seeds'])
    args = parser.parse_args()
    native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION') == 'cpu' and not os.environ.get('SLURM_JOB_GPUS'),
             'The V9 freeze requires a CPU-only Slurm allocation')
        need(all(getattr(args, name) is not None for name in ('parallel_cache', 'native_profile', 'diagnostic_summary')),
             '--check requires --parallel-cache, --native-profile, and --diagnostic-summary')
        check(args)
    else:
        need(args.plan is not None and args.condition is not None and args.seed is not None,
             'GPU modes require --plan, --condition, and --seed')
        run(args)


if __name__ == '__main__':
    main()
