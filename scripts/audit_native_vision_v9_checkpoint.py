"""All-selected-checkpoint V9 native state/cache audit, without model selection.

Four complete main runs, two fixed old software scenes each, eight ordinary
max4 generations plus five forced-prefix forwards/case: at most72 VLM/32 visual
calls. Every cache/full numerical failure remains descriptive. Actual captured
h through the same branch/native norm/head retains TV<=.02/top1 replay checks.
No training, gold prefixes, teacher scores, test-case selection or backend edits.
CPU self-test precedes main outcomes; CPU selection freezes all four completed
runs and the exact software inputs before GPU execution. All heavy work Slurm.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts import native_vision_v7_runtime as native
from scripts import profile_native_vision_v7_runtime as profile
from scripts import probe_native_vision_parallel_local_mixed as mixed
from scripts.stage_native_vision_v6_teacher import MODEL, need, read, save, sha
OUT = REPO/'outputs/native_aggregation_vlm/v9/checkpoint_audit'
MAIN = REPO/'outputs/native_aggregation_vlm/v9'
DATA = Path('/mnt/data/gabriele/gnn_transformer/v9_checkpoint_audit')
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v9')
STEPS = [972, 1944, 2916, 3888, 4860]
FORCED = ['Therefore', ':']
OWN = ('scripts/audit_native_vision_v9_checkpoint.py', 'slurm/native_vision_v9_checkpoint_selftest.sbatch',
       'slurm/native_vision_v9_checkpoint_check.sbatch', 'slurm/native_vision_v9_checkpoint.sbatch')
DEPENDENCIES = tuple(dict.fromkeys((*OWN, *profile.OWN, 'scripts/native_vision_v9_pairs.py', 'gnnformer/aggregation_consistency.py',
    'gnnformer/aggregation_path_bound.py', 'scripts/train_native_vision_v9.py')))


EXPECTED_POLICY = dict(
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

def sources(): return {name: sha(REPO/name) for name in DEPENDENCIES}


def snapshot(out):
    (out/'source').mkdir()
    for name in DEPENDENCIES: (out/'source'/name.replace('/', '_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json', sources())


def select_dev(entries):
    need(len(entries) == 5 and [x['step'] for x in entries] == STEPS, 'Expected all five registered dev checks')
    for row in entries:
        need(isinstance(row['exact_count'], int) and 0 <= row['exact_count'] <= 72
             and math.isfinite(row['nll']) and row['nll'] >= 0, 'Invalid dev selection statistic')
    return min(entries, key=lambda row: (-row['exact_count'], row['nll'], row['step']))


def verify_objective(row, condition):
    """Validate logged objective arithmetic, not unavailable past model states."""
    need(condition in ('residual', 'path'), 'Unknown selected-model condition')
    fields = ('loss', 'ce_loss', 'consistency_loss', 'path_loss', 'selected_regularizer_loss',
              'weighted_regularizer_loss', 'gradient_norm', 'ce_count_loss', 'ce_eos_loss',
              'consistency_count_loss', 'consistency_eos_loss', 'path_count_loss', 'path_eos_loss')
    need(all(isinstance(row[key], (int, float)) and not isinstance(row[key], bool)
             and math.isfinite(row[key]) and row[key] >= 0 for key in fields),
         'Nonfinite/negative recorded objective or gradient')
    close = lambda a,b: math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6)
    for prefix, scalar in (('ce', 'ce_loss'), ('consistency', 'consistency_loss'), ('path', 'path_loss')):
        need(close(row[scalar], (row[prefix+'_count_loss'] + row[prefix+'_eos_loss'])/2),
             'Count/EOS loss reduction differs')
    selected = row['consistency_loss'] if condition == 'residual' else row['path_loss']
    need(row['regularizer'] == condition and row['regularizer_coefficient'] == 1.
         and close(row['selected_regularizer_loss'], selected)
         and close(row['weighted_regularizer_loss'], selected)
         and close(row['loss'], row['ce_loss'] + selected)
         and row['clipped'] == (row['gradient_norm'] > 1.), 'Selected objective or clipping differs')
    d = row['regularizer_diagnostics']
    arrays = ('B_norms', 'residual_difference_norms', 'preactivation_difference_norms',
              'frozen_global_squared_norms', 'residual_pair_position_losses', 'path_pair_position_losses')
    for key in arrays:
        need(isinstance(d[key], list) and len(d[key]) == 8 and all(isinstance(x,list) and len(x) == 2 for x in d[key]),
             'Per-pair/causal-position diagnostics must be8x2')
        need(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
                 for pair in d[key] for x in pair), 'Nonfinite or negative pair diagnostic')
    flags = d['bound_inequality_holds']
    need(isinstance(flags,list) and len(flags)==8 and all(isinstance(x,list) and len(x)==2
         and all(type(v) is bool for v in x) for x in flags), 'Bound flags must be8x2 booleans')
    failed = sum(not v for pair in flags for v in pair)
    need(d['bound_violation_count'] == failed and type(d['bound_inequality_all_hold']) is bool
         and d['bound_inequality_all_hold'] == (failed == 0), 'Descriptive bound flags disagree')
    for prefix, field in (('residual','consistency_loss'),('path','path_loss')):
        need(close(sum(v for pair in d[prefix+'_pair_position_losses'] for v in pair)/16,row[field]),
             'Pair/position mean disagrees with objective')
    # A numerical bound violation stays descriptive; it must never drop a model.
    return True


def self_test():
    import torch
    torch.set_num_threads(4)
    entries = [dict(step=step, exact_count=10, nll=1.) for step in STEPS]
    need(select_dev(entries)['step'] == STEPS[0], 'Earliest exact/NLL tie rule differs')
    entries[2]['nll'] = .8
    need(select_dev(entries)['step'] == STEPS[2], 'NLL tie-break differs')
    entries[4].update(exact_count=11, nll=2.)
    need(select_dev(entries)['step'] == STEPS[4], 'Exact count must dominate NLL')
    for malformed in (entries[:4], entries + entries[:1], [dict(x, nll=float('nan')) for x in entries]):
        try: select_dev(malformed)
        except ValueError: pass
        else: raise AssertionError('Malformed/incomplete dev selection accepted')
    logits = torch.tensor([1., 2., 3.])
    same = profile.metric(torch, logits, logits+7.)
    wrong = profile.metric(torch, logits, logits.flip(0))
    need(same['numerical_rule_passed'] and same['full_vocabulary_tv'] < 1e-15
         and not wrong['numerical_rule_passed'] and not wrong['top1_equal'], 'Numerical diagnostic rules differ')
    import copy
    fixture = dict(ce_loss=2., ce_count_loss=1., ce_eos_loss=3., consistency_loss=.5,
        consistency_count_loss=.25, consistency_eos_loss=.75, path_loss=1., path_count_loss=.5,
        path_eos_loss=1.5, gradient_norm=2., clipped=True, regularizer_coefficient=1.,
        regularizer_diagnostics={key:[[0.,0.] for _ in range(8)] for key in
            ('B_norms','residual_difference_norms','preactivation_difference_norms','frozen_global_squared_norms')})
    fixture['regularizer_diagnostics'].update(residual_pair_position_losses=[[.25,.75] for _ in range(8)],
        path_pair_position_losses=[[.5,1.5] for _ in range(8)],bound_inequality_holds=[[False,True] for _ in range(8)],
        bound_violation_count=8,bound_inequality_all_hold=False)
    for condition, loss in (('residual',.5),('path',1.)):
        row=dict(fixture,regularizer=condition,selected_regularizer_loss=loss,weighted_regularizer_loss=loss,loss=2.+loss)
        need(verify_objective(row,condition), 'Valid distinct objectives were rejected')
        for changed in (dict(row,selected_regularizer_loss=7.),dict(row,regularizer_coefficient=0.),
                        dict(row,consistency_eos_loss=8.)):
            try:verify_objective(changed,condition)
            except ValueError:pass
            else:raise AssertionError('Incorrect objective arithmetic accepted')
        changed=copy.deepcopy(row);changed['regularizer_diagnostics']['B_norms'].pop()
        try:verify_objective(changed,condition)
        except ValueError:pass
        else:raise AssertionError('Incomplete pair-position diagnostics accepted')
    return dict(passed=True, tests=['complete_five_dev_selection', 'exact_then_NLL_then_earliest',
        'reject_missing_duplicate_nonfinite', 'full_vocabulary_shift_invariance', 'retain_top1_failure',
        'both_regularizers_and_position_reductions', 'reject_wrong_selected_loss_coefficient_position',
        'retain_descriptive_bound_failure', 'reject_incomplete_pair_diagnostics'])


def unit(args):
    out = OUT/f'selftest_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False); snapshot(out)
    tests = self_test()
    save(out/'summary.json', dict(tests, source_sha256=sources(), scope='Synthetic audit software only'))
    profile.local.index(out, 'V9 checkpoint audit software check', [('Summary', 'summary.json'), ('Frozen sources', 'source_hashes.json')])
    print(json.dumps(dict(passed=True, directory=str(out))), flush=True)


def verify_v9_pairing(config, train_plan, cache, directory):
    from scripts.native_vision_v9_pairs import load_pairs
    policy = config['policy']
    need(policy == EXPECTED_POLICY, 'V9 complete registered policy differs')
    coefficient = EXPECTED_POLICY['regularizer_coefficients'][config['condition']]
    need(config['regularizer'] == config['condition'] and config['regularizer_coefficient'] == coefficient,
         'Condition regularizer/coefficient differs')
    need(config['pairing_file'] == train_plan['pairing_file']
         and config['pairing_sha256'] == train_plan['pairing_sha256']
         and sha(config['pairing_file']) == config['pairing_sha256'], 'Pairing file binding differs')
    pairing = read(config['pairing_file'])
    rebuilt = load_pairs(train_plan['train_manifest'], config['cache_binding']['file'], train_plan['schedule_file'])
    need(native.object_sha(pairing) == native.object_sha(rebuilt) == train_plan['pairing_object_sha256']
         and train_plan['actual_frozen_pair_equality_passed'] is True, 'Frozen pair metadata/global equality changed')
    pairs = pairing['pairs']
    need(len(pairs) == 972 and [p['slot'] for p in pairs] == list(range(972)), 'Canonical pair coverage differs')
    for pair in pairs:
        left, right = (cache['scenes'][sid] for sid in pair['sids'])
        need(left['n_frames'] == 8 and right['n_frames'] == 16
             and left['question'] == right['question'] == pair['question']
             and left['gold'] == right['gold'] == pair['gold']
             and left['target_ids'] == right['target_ids'] == pair['target_ids']
             and left['global_feature_ids'] == right['global_feature_ids'] == pair['global_feature_ids'],
             'Paired target/question/causal global feature identity differs')
    rng = random.Random(config['seed']); expected_order = []
    for epoch in range(1, 41):
        slots = list(range(972)); rng.shuffle(slots)
        for slot in slots:
            pair = pairs[slot]
            common = {key: pair[key] for key in ('slot', 'pair_id', 'question', 'gold', 'replica')}
            expected_order.extend(dict(common, epoch=epoch, sid=sid, pair_side=side)
                                  for side, sid in zip(('N8', 'N16'), pair['sids']))
    actual_order = read(directory/'presentations.json')
    need(actual_order == expected_order and len(actual_order) == 77760
         and native.object_sha(actual_order) == config['order_sha256'] == train_plan['order_sha256'][str(config['seed'])],
         'Persistent RNG/pair-order reconstruction differs')
    history = read(directory/'training.json')
    need(len(history) == 4860, 'Incomplete paired optimizer history')
    for step, row in enumerate(history, 1):
        batch = expected_order[(step-1)*16:step*16]
        sids = [item['sid'] for item in batch]
        targets = [token for sid in sids for token in cache['scenes'][sid]['target_ids']]
        need(row['step'] == step and row['sids'] == sids and row['target_ids'] == targets
             and row['pair_ids'] == [batch[i]['pair_id'] for i in range(0,16,2)]
             and row['epochs'] == [batch[i]['epoch'] for i in range(0,16,2)], 'Paired batch/target ledger differs')
        rate = .001*step/50 if step <= 50 else .00001+.00099*(1+math.cos(math.pi*(step-50)/4810))/2
        need(math.isclose(row['lr'], rate, rel_tol=1e-10, abs_tol=1e-12), 'Optimizer schedule differs')
        verify_objective(row, config['condition'])
    artifacts = {config['pairing_file']: config['pairing_sha256']}
    need(config['diagnostic_summary'] == train_plan['diagnostic_summary'], 'Diagnostic dependency differs')
    diagnostic = config['diagnostic_summary']
    need(sha(diagnostic['file']) == diagnostic['sha256'], 'Completed diagnostic dependency changed')
    d = read(diagnostic['file'])
    need(d['passed'] and d['diagnostic_only'] and d['no_fit'] and d['vlm_forward_calls'] == 0
         and d['synthetic_points'] == 432, 'Diagnostic completion binding is invalid')
    artifacts[diagnostic['file']] = diagnostic['sha256']
    final_binding = diagnostic['finalization']
    need(sha(final_binding['file']) == final_binding['sha256'], 'V8 finalization dependency changed')
    finalized = read(final_binding['file'])
    need(finalized['passed'] and finalized['finalized']
         and sha(finalized['final_acceptance_file']) == finalized['final_acceptance_sha256'],
         'Completed V8 finalization binding differs')
    acceptance = read(finalized['final_acceptance_file'])
    need(acceptance['completed'] and acceptance['verification_passed'], 'Prior V8 verification did not pass')
    artifacts[final_binding['file']] = final_binding['sha256']
    artifacts[finalized['final_acceptance_file']] = finalized['final_acceptance_sha256']
    release_binding = config['main_release']
    need(sha(release_binding['file']) == release_binding['sha256'], 'Main resource release changed')
    release = read(release_binding['file'])
    need(release['passed'] and release['plan_sha256'] == config['plan_sha256']
         and release['source_sha256'] == config['source_sha256']
         and set(release['profiles']) == {'residual', 'path'}, 'Main condition release differs')
    artifacts[release_binding['file']] = release_binding['sha256']
    for condition, binding in release['profiles'].items():
        path = str(Path(binding['directory'])/'summary.json')
        need(sha(path) == binding['summary_sha256'], 'Condition profile changed')
        p = read(path)
        need(p['profile'] and p['condition'] == condition and p['arm'] == 'parallel'
             and p['seed'] == 12 and p['steps'] == 32 and p['passed'] and p['computational_integrity_passed']
             and p['no_dev_or_test_evaluation'] and p['active_cache_replay']['passed']
             and p['plan_sha256'] == config['plan_sha256'] and p['source_sha256'] == config['source_sha256'],
             'Condition profile/source release differs')
        zero = p['zero_initialization_auxiliary_gradients']
        need(all(zero.get(key) is True for key in ('passed', 'up_exactly_zero',
             'all_residual_gradients_zero', 'all_path_gradients_zero'))
             and zero['residual_loss'] == zero['path_loss'] == 0., 'Profile zero-U auxiliary gradient gate differs')
        artifacts[path] = binding['summary_sha256']
    summary = read(directory/'summary.json')
    for field in ('training', 'first_gradients'):
        path, digest = summary[field+'_file'], summary[field+'_sha256']
        need(Path(path).resolve() == directory/(field+'.json') and sha(path) == digest, 'Run training/gradient artifact changed')
        artifacts[path] = digest
    return artifacts


def verify_run(torch, directory):
    """Selection uses every dev result, never a software-case or test outcome."""
    directory = Path(directory).resolve()
    need(directory.parent == MAIN, 'Require a canonical V9 main run directory')
    config = read(directory/'config.json'); summary = read(directory/'summary.json')
    arm, condition, seed = config['arm'], config['condition'], config['seed']; run_id = config['run_id']
    need(arm == 'parallel' and condition in ('residual', 'path') and seed in (12, 13) and config['profile'] is False
         and directory.name == run_id and run_id.startswith(f'run_{condition}_s{seed}_'), 'Main run identity differs')
    need(all(summary[key] == value for key, value in config.items()) and summary['passed'] is True
         and summary['completed'] is True and summary['computational_integrity_passed'] is True
         and summary['steps'] == 4860 and summary['native_test_count'] == 452, 'Incomplete canonical main run')
    policy = config['policy']
    need(policy['steps'] == 4860 and policy['epochs'] == 40 and policy['scene_slots_per_epoch'] == 1944
         and policy['batch_size'] == 16 and policy['parameters'] == 1041600 and policy['dev_steps'] == STEPS
         and policy['merge'] == 'sum' and policy['post_activation'] == 'silu'
         and policy['native_eos'] == [151645, 151643] and policy['target_eos'] == 151645,
         'Registered architecture/training/selection policy differs')
    need(sha(config['plan_file']) == config['plan_sha256'], 'Frozen training CPU plan changed')
    need(Path(config['plan_file']).with_suffix('.sha256').read_text().strip() == config['plan_sha256'],
         'Training CPU plan sidecar changed')
    train_plan = read(config['plan_file'])
    need(train_plan['source_sha256'] == config['source_sha256'] and train_plan['policy'] == policy,
         'Main source/policy differs from CPU release')
    cpu = read(Path(config['plan_file']).parent/'summary.json')
    need(cpu['passed'] and cpu['plan_sha256'] == config['plan_sha256'], 'Missing matching completed training CPU gate')
    for name, digest in config['source_sha256'].items():
        need(sha(REPO/name) == digest and sha(directory/'code'/name.replace('/', '_')) == digest,
             'Frozen training source/code snapshot changed')
    for path, digest in train_plan['artifact_bindings'].items(): need(sha(path) == digest, 'Main dataset/audit changed')
    binding = config['cache_binding']; need(sha(binding['file']) == binding['sha256'], 'Feature cache changed')
    cache = read(binding['file'])
    need(cache['complete'] and cache['training_only'] and len(cache['scenes']) == 1890
         and sha(cache['plan_file']) == cache['plan_sha256'],
         'Main cache/feature-plan binding differs')
    feature_plan = read(cache['plan_file'])
    for key in ('model', 'runtime', 'processor', 'native_dtypes'):
        need(cache[key] == config[key], 'Recorded native identity differs from training features')
    if 'runtime_extension' in cache:
        extension = cache['runtime_extension']; ledger = read(extension['release_file'])
        need(sha(extension['release_file']) == extension['release_sha256']
             and extension == dict(ledger, release_file=extension['release_file'], release_sha256=extension['release_sha256'])
             and ledger['original_profile_passed'] is False and ledger['cap_seconds'] == 360,
             'Explicit parallel feature-runtime extension differs')
        for name, digest in ledger['dispatcher_source_sha256'].items(): need(sha(REPO/name) == digest, 'Runtime dispatcher source changed')
    pair_artifacts = verify_v9_pairing(config, train_plan, cache, directory)
    history = read(directory/'training.json'); presentations = read(directory/'presentations.json')
    need(len(history) == 4860 and len(presentations) == 77760
         and sha(directory/'presentations.json') == config['presentations_sha256'], 'Training presentation coverage differs')
    for step, row in enumerate(history, 1):
        need(row['step'] == step and row['sids'] == [x['sid'] for x in presentations[(step-1)*16:step*16]]
             and len(row['target_ids']) == 32 and math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']),
             'Recorded updates differ from the frozen presentation order')
    selection = read(directory/'selection.json'); entries = selection['development']
    need(summary['development'] == entries and summary['selected'] == selection['selected'], 'Canonical selection records disagree')
    artifacts = dict(pair_artifacts)
    for entry in entries:
        step = entry['step']; devpath = directory/f'dev_{step}.json'; dev = read(devpath)
        need(Path(entry['dev_file']).resolve() == devpath and sha(devpath) == entry['dev_sha256']
             and dev['n'] == len(dev['rows']) == 72 and len({x['sid'] for x in dev['rows']}) == 72,
             'Incomplete or changed development sweep')
        exact = 0
        for row in dev['rows']:
            parsed = int(row['text'].strip()) if re.fullmatch(r'[0-9]+', row['text'].strip()) else None
            need(row['prediction'] == parsed and row['exact'] == (row['completed'] and parsed == row['gold'])
                 and math.isfinite(row['first_token_nll']), 'Development complete-answer selection statistic differs')
            exact += row['exact']
        nll = sum(row['first_token_nll'] for row in dev['rows'])/72
        need(exact == entry['exact_count'] == dev['exact_count'] and nll == entry['nll'] == dev['first_token_nll']
             and sha(dev['raw_file']) == dev['raw_sha256'], 'Development selection/raw evidence differs')
        checkpoint = Path(entry['checkpoint']).resolve()
        need(checkpoint == CKPT/run_id/f'step_{step}.pt' and sha(checkpoint) == entry['checkpoint_sha256'],
             'Development checkpoint provenance differs')
        artifacts[str(devpath)] = sha(devpath); artifacts[str(checkpoint)] = sha(checkpoint)
    selected = select_dev(entries)
    need(selected == summary['selected'], 'Selected checkpoint is not the registered dev optimum')
    saved = torch.load(selected['checkpoint'], map_location='cpu', weights_only=True)
    need(saved['step'] == selected['step'] and saved['config'] == config
         and native.object_sha({name: native.tensor_info(value) for name, value in saved['branch'].items()}) == selected['parameter_sha256'],
         'Saved selected branch differs from the selected epoch/config/parameter digest')
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    check_core = ParallelLocalAggregation(); check_core.load_state_dict(saved['branch'], strict=True)
    need(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in saved['branch'].values()),
         'Selected core must contain finite FP32 tensors')
    need(sha(summary['test_file']) == summary['test_sha256'], 'Completed test artifact changed')
    for name in ('config.json', 'summary.json', 'training.json', 'presentations.json', 'selection.json'):
        artifacts[str(directory/name)] = sha(directory/name)
    artifacts[config['plan_file']] = config['plan_sha256']
    artifacts[binding['file']] = binding['sha256']
    artifacts[cache['plan_file']] = cache['plan_sha256']
    artifacts.update(train_plan['artifact_bindings'])
    artifacts[str(Path(train_plan['native_profile']['directory'])/'summary.json')] = train_plan['native_profile']['sha256']
    if 'runtime_extension' in cache:
        artifacts[cache['runtime_extension']['release_file']] = cache['runtime_extension']['release_sha256']
    return dict(run_id=run_id, arm=arm, condition=condition, seed=seed, directory=str(directory), config=config,
        selected=selected, artifact_sha256=artifacts, native_api=feature_plan['native_api'],
        native_profile=train_plan['native_profile']['directory'], native_profile_sha256=train_plan['native_profile']['sha256'])


def check(args):
    import torch, transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    out = OUT/f'check_{os.environ["SLURM_JOB_ID"]}'; out.mkdir(parents=True, exist_ok=False); snapshot(out)
    tests = self_test()
    unit = read(args.source_check/'summary.json')
    need(unit['passed'] and unit['source_sha256'] == sources(), 'Require matching pre-main audit self-test/source freeze')
    need(args.runs and len(args.runs) == 4 and len({x.resolve() for x in args.runs}) == 4, 'Require all four distinct main runs')
    runs = [verify_run(torch, path) for path in args.runs]
    runs.sort(key=lambda x: (x['condition'], x['seed']))
    need({(x['condition'], x['seed']) for x in runs} == {(a, s) for a in ('residual', 'path') for s in (12, 13)},
         'Missing a registered selected model')
    first = runs[0]
    for run in runs:
        need(all(run['config'][key] == first['config'][key] for key in ('policy', 'source_sha256', 'plan_file', 'plan_sha256',
             'model', 'runtime', 'processor', 'native_dtypes')), 'Main models used different frozen protocols/native runtimes')
        need(run['native_profile'] == first['native_profile'] and run['native_profile_sha256'] == first['native_profile_sha256'],
             'Native software release differs across mains')
    for seed in (12, 13):
        a, b = [x for x in runs if x['seed'] == seed]
        need(a['config']['initialized_sha256'] == b['config']['initialized_sha256']
             and a['config']['presentations_sha256'] == b['config']['presentations_sha256'], 'Paired initialization/order differs')
    profile_dir = Path(first['native_profile']); summary = read(profile_dir/'summary.json')
    need(sha(profile_dir/'summary.json') == first['native_profile_sha256'] and summary['completed']
         and summary['computational_integrity_passed'] and summary['zero_identity_passed'], 'Native integration release differs')
    parent = profile.verify(summary['plan_file'])
    need(parent['model'] == first['config']['model'] and parent['runtime'] == first['config']['runtime']
         and parent['processor'] == first['config']['processor'], 'Prepared input/model ancestor differs')
    processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True, use_fast=False)
    from scripts.probe_native_vision_v2_prefix import fingerprint
    need(fingerprint(processor, str(transformers.__version__)) == parent['processor'], 'Actual CPU processor differs')
    forced = []
    for text in FORCED:
        ids = processor.tokenizer(text, add_special_tokens=False)['input_ids']
        need(len(ids) == 1 and ids[0] not in processor.tokenizer.all_special_ids
             and processor.tokenizer.decode(ids, skip_special_tokens=False) == text, 'Fixed forced token is unsupported')
        forced.extend(ids)
    blob = torch.load(parent['prepared_file'], map_location='cpu', weights_only=True)
    owner, fn, _ = profile.local.native_api(processor)
    def rope(**kwargs): return fn(owner, **kwargs)
    cases = []
    for run in runs:
        for n in (16, 64):
            case_id = f"{run['run_id']}__N{n}"; key = f"{run['arm']}_N{n}"; base = blob['bundles'][key]
            prefixes = []
            for step in range(3):
                item = profile.prefixed_bundle(base, forced[:step])
                layout = native.audit_layout(rope, item)
                prefixes.append(dict(input_identity=item['metadata']['input_identity'], layout=layout['metadata']))
            cases.append(dict(case_id=case_id, run_id=run['run_id'], arm=run['arm'], condition=run['condition'], n_frames=n,
                              bundle_key=key, sid=base['metadata']['sid'], prefix_identities=prefixes))
    plan = dict(schema_version=1, protocol='all_selected_v9_native_checkpoint_audit', source_sha256=sources(), tests=tests,
        source_check=str(args.source_check.resolve()), source_check_sha256=sha(args.source_check/'summary.json'),
        runs=runs, cases=cases, prepared_file=parent['prepared_file'], prepared_sha256=parent['prepared_sha256'],
        parent_plan_file=summary['plan_file'], parent_plan_sha256=sha(summary['plan_file']),
        model=parent['model'], runtime=parent['runtime'], processor=parent['processor'], native_api=parent['native_api'],
        forced_text=FORCED, forced_ids=forced, maximum_calls=dict(model=72, visual=32),
        forced_calls=dict(model=40, visual=24), ordinary_generations=8, cache_full_rows=656,
        policy='All four selected checkpoints; replayTV<=.02/top1 is binding; every cache/full error remains descriptive',
        backend='native_bitsandbytes_dispatch_unmodified', original_mixed_numerical_gate_passed=False)
    path = out/'plan.json'; save(path, plan); path.with_suffix('.sha256').write_text(sha(path)+'\n')
    verify(path)
    save(out/'summary.json', dict(passed=True, plan_file=str(path), plan_sha256=sha(path), source_sha256=sources(), selected_models=4))
    profile.local.index(out, 'All four selected V9 checkpoints: CPU freeze', [('Plan', 'plan.json'), ('Summary', 'summary.json')])
    print(json.dumps(dict(passed=True, plan=str(path), plan_sha256=sha(path))), flush=True)


def verify(path):
    path = Path(path).resolve()
    need(path.is_relative_to(OUT) and sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Audit plan sidecar differs')
    plan = read(path)
    need(plan['schema_version'] == 1 and plan['protocol'] == 'all_selected_v9_native_checkpoint_audit'
         and plan['source_sha256'] == sources(), 'Frozen audit source/protocol differs')
    for name, digest in plan['source_sha256'].items():
        need(sha(path.parent/'source'/name.replace('/', '_')) == digest, 'Audit CPU source snapshot differs')
    need(sha(plan['prepared_file']) == plan['prepared_sha256'] and sha(plan['parent_plan_file']) == plan['parent_plan_sha256']
         and sha(Path(plan['source_check'])/'summary.json') == plan['source_check_sha256'], 'Prepared/native/source ancestor changed')
    for run in plan['runs']:
        for filename, digest in run['artifact_sha256'].items(): need(sha(filename) == digest, 'Frozen selected-model evidence changed')
        for filename, digest in run['config']['source_sha256'].items(): need(sha(REPO/filename) == digest, 'Main source changed')
    need(len(plan['runs']) == 4 and len(plan['cases']) == 8 and plan['forced_text'] == FORCED
         and plan['maximum_calls'] == dict(model=72, visual=32), 'Complete registered audit coverage differs')
    need({(r['condition'], r['seed']) for r in plan['runs']} ==
         {(c,s) for c in ('residual','path') for s in (12,13)}
         and all(r['arm'] == 'parallel' for r in plan['runs'])
         and plan['cache_full_rows'] == 656 and plan['forced_calls'] == dict(model=40,visual=24),
         'Registered V9 condition/parallel-row audit coverage differs')
    return plan


def run(args):
    import torch, transformers, importlib.metadata
    from gnnformer.runtime import load_runtime, get_rope_index_fn, move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4); started = time.perf_counter(); plan = verify(args.plan)
    cpu = read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] and cpu['plan_sha256'] == sha(args.plan), 'Completed matching CPU selection gate required')
    need(dict(torch_version=str(torch.__version__), transformers_version=str(transformers.__version__),
         bitsandbytes_version=importlib.metadata.version('bitsandbytes')) == plan['runtime'], 'Native runtime changed')
    parent = profile.verify(plan['parent_plan_file'])
    need(parent['model'] == plan['model'] and parent['processor'] == plan['processor'], 'Native software ancestor differs')
    blob = torch.load(plan['prepared_file'], map_location='cpu', weights_only=True)
    job = os.environ['SLURM_JOB_ID']; out = OUT/f'run_{job}'; out.mkdir(parents=True, exist_ok=False); snapshot(out)
    data = DATA/f'run_{job}'; data.mkdir(parents=True, exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    profile.local.index(out, 'All-selected V9 native checkpoint audit', [('Frozen plan', 'plan.json'),
        ('Summary after completion', 'summary.json'), ('Cached/full comparisons', 'comparisons.json'), ('Native replay', 'replay.json')])
    tick = time.perf_counter(); rt = load_runtime(str(MODEL), use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    model = rt.model; model.eval(); model.requires_grad_(False); native.native_contract(model)
    need(fingerprint(rt.processor, str(transformers.__version__)) == plan['processor'], 'Loaded processor differs')
    _, _, api = profile.local.native_api(rt.processor); need(api == plan['native_api'], 'Installed native model code changed')
    for selected in plan['runs']:
        for path, digest in selected['native_api']['source_sha256'].items():
            need(sha(path) == digest, 'Native feature extraction backend source changed')
    _, policy = native.generation_policy(model, rt.tokenizer)
    need(policy['native_eos_token_ids'] == [151645, 151643] and policy['target_eos_token_id'] == 151645,
         'Native selected-model EOS contract differs')
    torch.cuda.synchronize(); load_seconds = time.perf_counter()-tick
    norm, head, rope = model.model.language_model.norm, model.lm_head, get_rope_index_fn(model)
    branch = ParallelLocalAggregation().to(device=rt.device)
    comparisons, replays, generations, artifacts, model_results = [], [], [], [], []
    with profile.NativeAudit(model, data) as audit:
        for selected in plan['runs']:
            run_id = selected['run_id']; selected_checkpoint = selected['selected']
            need(sha(selected_checkpoint['checkpoint']) == selected_checkpoint['checkpoint_sha256'], 'Selected checkpoint changed')
            saved = torch.load(selected_checkpoint['checkpoint'], map_location=rt.device, weights_only=True)
            branch.load_state_dict(saved['branch'], strict=True); native.native_contract(model, branch)
            before = native.object_sha({name: native.tensor_info(value) for name, value in branch.state_dict().items()})
            need(before == selected_checkpoint['parameter_sha256'], 'Loaded selected branch differs')
            model_start = time.perf_counter(); first_comparison, first_replay = len(comparisons), len(replays)
            for case in [x for x in plan['cases'] if x['run_id'] == run_id]:
                case_id = case['case_id']; bundle = blob['bundles'][case['bundle_key']]
                meta = bundle['metadata']; batch, width = meta['row_count'], meta['prompt_width']
                audit.context = dict(run_id=run_id, case_id=case_id, phase='ordinary_generation')
                offset = len(audit.records)
                generated = native.generate_native(model, rt.processor, branch, bundle, capture=True)
                profile.audit_sequence(torch, model, bundle, audit.records[offset:], generated['generated_ids'])
                genpath = data/f'{case_id}__generation.pt'; torch.save(generated, genpath)
                generations.append(dict(run_id=run_id, case_id=case_id, sid=case['sid'], arm=selected['arm'], condition=selected['condition'], seed=selected['seed'],
                    n_frames=case['n_frames'], generated_ids=generated['generated_ids'], text=generated['text'],
                    raw_text=generated['raw_text'], completed=generated['completed'], truncated=generated['truncated'],
                    counters=generated['counters'], model_seconds=generated['model_seconds'], path=str(genpath), sha256=sha(genpath)))
                def execute(tag, inputs, *, cached, expected_positions, expected_mask, fusion):
                    audit.context = dict(run_id=run_id, case_id=case_id, phase='forced', tag=tag)
                    old_count = len(audit.records)
                    with torch.inference_mode(): result = model(**inputs, use_cache=cached, logits_to_keep=1)
                    torch.cuda.synchronize()
                    need(len(audit.records) == old_count+1, 'Each forced state must have exactly one native forward')
                    record = audit.records[-1]; raw = torch.load(record['path'], map_location='cpu', weights_only=True)
                    need(torch.equal(raw['position_ids'], expected_positions.cpu())
                         and torch.equal(raw['attention_mask'], expected_mask.cpu()), 'Forced native mask/position identity differs')
                    need(record['visual'] == int(inputs.get('pixel_values') is not None), 'Forced native vision invocation differs')
                    capture = fusion.export_last_capture(cpu=False)
                    h, local = capture['global_states'], capture['local_states']
                    with torch.inference_mode():
                        delta = branch(local, h, output_dtype=h.dtype)
                        # Direct native norm.forward avoids only the observation hook during this extra readout.
                        replayed = head(norm.forward((h+delta).unsqueeze(0)))[0, 0]
                    measured = profile.metric(torch, result.logits[-1, -1], replayed)
                    measured.update(run_id=run_id, case_id=case_id, tag=tag, condition=selected['condition'], seed=selected['seed'],
                        delta_maximum_absolute=float((delta.float()-capture['delta'].float()).abs().max()),
                        source_state_file=record['path'], source_state_sha256=record['sha256'], binding_gate=True)
                    replays.append(measured)
                    replay_path = data/f'{case_id}__{tag}__replay.pt'
                    torch.save(dict(capture=native.normal_copies(capture, cpu=True), replayed_logits=replayed.detach().cpu(),
                                    native_global_logits=result.logits[-1, -1].detach().cpu()), replay_path)
                    artifacts.append(dict(run_id=run_id, case_id=case_id, tag=tag, path=str(replay_path), sha256=sha(replay_path)))
                    return result.past_key_values, raw
                base_item = move_to_device(bundle['inputs'], rt.device)
                initial = native.audit_layout(rope, bundle)
                need(dict(input_identity=meta['input_identity'], layout=initial['metadata']) == case['prefix_identities'][0],
                     'Frozen selected software input differs')
                with native._controller(norm, branch, meta, capture=True) as fusion:
                    cache, _ = execute('prefill', base_item, cached=True, expected_positions=initial['position_ids'],
                                       expected_mask=bundle['inputs']['attention_mask'], fusion=fusion)
                    need(torch.equal(model.model.rope_deltas.detach().cpu(), initial['rope_deltas']), 'Initial native rope delta differs')
                    for step in (1, 2):
                        full = profile.prefixed_bundle(bundle, plan['forced_ids'][:step])
                        layout = native.audit_layout(rope, full)
                        need(dict(input_identity=full['metadata']['input_identity'], layout=layout['metadata']) == case['prefix_identities'][step],
                             'Frozen forced-prefix inputs differ')
                        full_item = move_to_device(full['inputs'], rt.device)
                        cache_position = torch.tensor([width+step-1], device=rt.device)
                        prepared = model.prepare_inputs_for_generation(full_item['input_ids'], past_key_values=cache,
                            attention_mask=full_item['attention_mask'], cache_position=cache_position, use_cache=True,
                            pixel_values=full_item['pixel_values'], image_grid_thw=full_item['image_grid_thw'])
                        expected = torch.cat(((full_item['attention_mask'].cumsum(-1)-1)[:, -1:].unsqueeze(0),
                                              layout['position_ids'][:, :, -1:].to(rt.device)), dim=0)
                        need(prepared['input_ids'].shape == (batch, 1)
                             and bool((prepared['input_ids'] == plan['forced_ids'][step-1]).all())
                             and prepared.get('past_key_values') is cache and prepared.get('pixel_values') is None
                             and torch.equal(prepared['position_ids'], expected) and prepared.pop('use_cache', True) is True,
                             'Native forced-token preparation changed cache/history/mRoPE')
                        previous = cache
                        cache, cached_raw = execute(f'cached_{step}', prepared, cached=True,
                            expected_positions=expected, expected_mask=full_item['attention_mask'], fusion=fusion)
                        need(cache is previous and cache.get_seq_length() == width+step, 'Native cached state identity/length differs')
                        del previous
                        saved_delta = model.model.rope_deltas.detach().clone()
                        try:
                            unused, full_raw = execute(f'full_{step}', full_item, cached=False,
                                expected_positions=layout['position_ids'], expected_mask=full_item['attention_mask'], fusion=fusion)
                            need(unused is None and cache.get_seq_length() == width+step
                                 and torch.equal(model.model.rope_deltas, saved_delta)
                                 and torch.equal(saved_delta.detach().cpu(), layout['rope_deltas']), 'Full reference disturbed retained native state')
                        finally:
                            model.model.rope_deltas = saved_delta
                        need(cached_raw['native_logits'].shape == full_raw['native_logits'].shape, 'Cache/full row coverage differs')
                        for row_index, (a, b) in enumerate(zip(cached_raw['native_logits'], full_raw['native_logits'])):
                            result = profile.metric(torch, a, b)
                            ha = cached_raw['pre_final_rms_hidden'][row_index].double()
                            hb = full_raw['pre_final_rms_hidden'][row_index].double()
                            result.update(run_id=run_id, case_id=case_id, n_frames=case['n_frames'], arm=selected['arm'], condition=selected['condition'], seed=selected['seed'],
                                step=step, row_index=row_index, row_kind=meta['row_kinds'][row_index], descriptive_only=True,
                                hidden_rms=float((ha-hb).square().mean().sqrt()),
                                hidden_relative_l2=float((ha-hb).norm()/hb.norm()) if float(hb.norm()) else None,
                                hidden_cosine=float(torch.dot(ha,hb)/(ha.norm()*hb.norm())) if float(ha.norm()*hb.norm()) else None)
                            comparisons.append(result)
                    need(fusion.calls == 5, 'Each forced software case needs exactly five native branch calls')
                del cache, base_item
            after = native.object_sha({name: native.tensor_info(value) for name, value in branch.state_dict().items()})
            need(before == after and sha(selected_checkpoint['checkpoint']) == selected_checkpoint['checkpoint_sha256'],
                 'Audit changed selected parameters/checkpoint')
            need(len(comparisons)-first_comparison == 164 and len(replays)-first_replay == 10,
                 'Each parallel selected model requires164 cache/full rows and10 captured-state replays')
            model_results.append(dict(run_id=run_id, arm=selected['arm'], condition=selected['condition'], seed=selected['seed'],
                checkpoint=selected_checkpoint['checkpoint'], checkpoint_sha256=selected_checkpoint['checkpoint_sha256'],
                selected_step=selected_checkpoint['step'], selected_parameter_sha256=after,
                cache_full_rows=len(comparisons)-first_comparison, replay_rows=len(replays)-first_replay,
                replay_gate_passed=all(x['numerical_rule_passed'] for x in replays[first_replay:]),
                cache_full_numeric_passed=all(x['numerical_rule_passed'] for x in comparisons[first_comparison:]),
                seconds=time.perf_counter()-model_start))
            print(json.dumps(model_results[-1]), flush=True)
        counts = dict(audit.counts); records = list(audit.records)
    expected_generation = sum(len(x['generated_ids']) for x in generations)
    need(len(model_results) == 4 and len(generations) == 8 and len(comparisons) == 656 and len(replays) == 40
         and counts == dict(model=40+expected_generation, visual=32, language=40+expected_generation,
                            norm=40+expected_generation, attention=40+expected_generation)
         and counts['model'] <= 72, 'Incomplete registered all-model audit')
    need(verify(args.plan) == plan, 'Frozen audit/selected sources changed during execution')
    for name, value in (('forwards.json', records), ('generations.json', generations), ('comparisons.json', comparisons),
                        ('replay.json', replays), ('artifacts.json', artifacts)): save(out/name, value)
    replay_failures = [x for x in replays if not x['numerical_rule_passed']]
    cache_failures = [x for x in comparisons if not x['numerical_rule_passed']]
    summary = dict(schema_version=1, completed=True, computational_integrity_passed=True,
        native_replay_gate_passed=not replay_failures, passed=not replay_failures,
        strict_cache_numerical_gate_passed=not cache_failures, original_mixed_numerical_gate_passed=False,
        replay_failures=replay_failures, cache_numerical_failures=cache_failures,
        plan_file=str(Path(args.plan).resolve()), plan_sha256=sha(args.plan), source_sha256=sources(),
        backend=plan['backend'], model=plan['model'], runtime=plan['runtime'], processor=plan['processor'],
        runs=model_results, calls=counts, ordinary_generations=8, forced_forwards=40,
        replay_comparisons=40, cache_full_rows=656, model_load_seconds=load_seconds,
        seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(0), slurm_job_id=job,
        files={name: dict(path=str(out/name), sha256=sha(out/name)) for name in
               ('forwards.json', 'generations.json', 'comparisons.json', 'replay.json', 'artifacts.json')},
        limitations=['Every selected model retained; software outcomes never choose/drop a checkpoint.',
            'Old software scenes and forced tokens are not fresh model accuracy or reasoning-composition evidence.',
            'Cache/full numerical failures are descriptive under the unchanged native engineering release.',
            'Captured-state branch/norm/head replay retains the registered TV.02/top1 criterion.',
            'All actual causal masks/logits/last-query states saved; full KV tensors are checked in GPU memory but not archived.',
            'All timing includes diagnostic capture, copying, hashing, readout replay and serialization.'])
    save(out/'summary.json', summary)
    lines = ['# V9 selected-checkpoint software audit', '',
        f"All four models completed. Native replay failures: {len(replay_failures)}/40; descriptive cache/full failures: {len(cache_failures)}/656.", '',
        '| Condition | Seed | Selected step | Replay passed | Cache/full passed |', '|---|---:|---:|---|---|']
    for row in model_results:
        lines.append(f"| {row['condition']} | {row['seed']} | {row['selected_step']} | {row['replay_gate_passed']} | {row['cache_full_numeric_passed']} |")
    lines += ['', 'The original mixed/cache failure remains failed. No software result changes checkpoint selection.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    profile.local.index(data, 'Selected V9 native state evidence', [('Summary', str(out/'summary.json')), ('Forward inventory', str(out/'forwards.json'))])
    print(json.dumps(dict(directory=str(out), passed=not replay_failures, replay_failures=len(replay_failures),
                         cache_failures=len(cache_failures), calls=counts)), flush=True)
    if replay_failures: raise SystemExit('Native selected-state replay gate failed; every model and failure is preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true'); mode.add_argument('--check', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--source-check', type=Path); parser.add_argument('--runs', type=Path, nargs=4); parser.add_argument('--plan', type=Path)
    args = parser.parse_args(); native.require_slurm(gpu=args.run)
    if args.self_test or args.check:
        need(os.environ.get('SLURM_JOB_PARTITION') == 'cpu' and not os.environ.get('SLURM_JOB_GPUS'), 'CPU-only audit preparation required')
        if args.self_test: unit(args)
        else:
            need(args.source_check is not None, 'Pre-main source/self-test check required')
            args.source_check = args.source_check.resolve(); check(args)
    else:
        need(args.plan is not None, 'Exact completed CPU selected-checkpoint plan required')
        args.plan = args.plan.resolve(); run(args)


if __name__ == '__main__': main()
