"""All-selected-checkpoint V10 native state/cache audit, without model selection.

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
OUT = REPO/'outputs/native_aggregation_vlm/v10/checkpoint_audit'
MAIN = REPO/'outputs/native_aggregation_vlm/v10'
DATA = Path('/mnt/data/gabriele/gnn_transformer/v10_checkpoint_audit')
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v10')
STEPS = [918, 1836, 2754, 3672, 4590]
FORCED = ['Therefore', ':']
OWN = ('scripts/audit_native_vision_v10_checkpoint.py', 'slurm/native_vision_v10_checkpoint_selftest.sbatch',
       'slurm/native_vision_v10_checkpoint_check.sbatch', 'slurm/native_vision_v10_checkpoint.sbatch')
DEPENDENCIES = tuple(dict.fromkeys((*OWN, *profile.OWN, 'scripts/native_vision_v10_pairs.py', 'gnnformer/aggregation_consistency.py',
    'gnnformer/aggregation_path_bound.py', 'gnnformer/paired_sequence_objectives.py',
    'scripts/stage_native_vision_v10_features.py', 'scripts/cache_native_vision_v10_features.py',
    'scripts/train_native_vision_v10.py')))


EXPECTED_POLICY = dict(
    protocol='v10_supported_answer_native_residual_consistency', native_arm='parallel',
    conditions=['ce', 'consistency'], seeds=[14, 15], epochs=40,
    pair_slots_per_epoch=918, scene_slots_per_epoch=1836, unique_training_scenes=1782,
    scene_presentations=73440, pair_presentations=36720, target_positions=177120,
    batch_size=16, pairs_per_batch=8, steps=4590,
    dev_steps=[918, 1836, 2754, 3672, 4590], dev_examples=64, test_examples=272,
    lr=.001, warmup=50, final_lr=.00001, weight_decay=0., clip_norm=1.,
    rank=96, hidden_size=3584, parameters=1041600, merge='sum', post_activation='silu',
    native_dtype='torch.float16', branch_dtype='torch.float32',
    consistency_coefficients={'ce': 0., 'consistency': 1.}, consistency_epsilon=1e-6,
    ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',
    within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',
    path_loss_role='diagnostic_only_no_loss_weight_no_inference_role',
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',
    max_new_tokens=4, exact_requires_eos=True,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',
    native_eos=[151645, 151643], target_eos=151645, profile_steps=32,
    training_count_support=list(range(17)), development_count_support=list(range(16)),
    test_count_support=list(range(17)), maximum_training_N=16, test_N=[32, 64],
    bootstrap_seed=20261008, bootstrap_replicates=10000,
)

def sources(): return {name: sha(REPO/name) for name in DEPENDENCIES}


def snapshot(out):
    (out/'source').mkdir()
    for name in DEPENDENCIES: (out/'source'/name.replace('/', '_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json', sources())


def select_dev(entries):
    need(len(entries) == 5 and [x['step'] for x in entries] == STEPS, 'Expected all five registered dev checks')
    for row in entries:
        need(isinstance(row['exact_count'], int) and 0 <= row['exact_count'] <= 64
             and math.isfinite(row['nll']) and row['nll'] >= 0, 'Invalid dev selection statistic')
    return min(entries, key=lambda row: (-row['exact_count'], row['nll'], row['step']))


def verify_objective(row, condition):
    """Recompute sequence weighting from saved per-position loss diagnostics."""
    from gnnformer.paired_sequence_objectives import sequence_layout
    need(condition in ('ce','consistency'),'Unknown selected-model condition')
    coefficient=EXPECTED_POLICY['consistency_coefficients'][condition]
    fields=('loss','ce_loss','consistency_loss','path_loss','weighted_consistency_loss','gradient_norm')
    need(all(isinstance(row[k],(int,float)) and not isinstance(row[k],bool) and math.isfinite(row[k]) and row[k]>=0
             for k in fields),'Invalid recorded objective/gradient')
    close=lambda a,b: math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6)
    need(row['consistency_coefficient']==coefficient
         and close(row['weighted_consistency_loss'],coefficient*row['consistency_loss'])
         and close(row['loss'],row['ce_loss']+coefficient*row['consistency_loss'])
         and type(row['clipped']) is bool and row['clipped']==(row['gradient_norm']>1.),
         'Loss must be native CE plus registered residual consistency; path is diagnostic only')
    lengths=row['scene_lengths'];offsets=row['scene_offsets']
    need(len(lengths)==16 and all(type(v) is int and v in (2,3) for v in lengths),'Wrong per-scene target lengths')
    expected=[0]
    for length in lengths:expected.append(expected[-1]+length)
    need(offsets==expected and len(row['target_ids'])==offsets[-1] and len(row['sids'])==16,
         'Ragged scene offsets/targets differ')
    layout=sequence_layout([row['target_ids'][a:b] for a,b in zip(offsets,offsets[1:])])
    need(row['pair_lengths']==lengths[::2] and row['prefix_ids']==[p for seq in layout['prefixes'] for p in seq],
         'Pair lengths or actual strict-prefix ledger differs')
    need(all(seq[-1]==151645 and all(t not in (151643,151645) for t in seq[:-1]) for seq in layout['target_sequences']),
         'Each canonical target needs one terminal EOS')
    positions=row['position_losses'];pair_count=len(layout['left'])
    for key,count in (('ce',offsets[-1]),('consistency',pair_count),('path',pair_count)):
        need(isinstance(positions[key],list) and len(positions[key])==count
             and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in positions[key]),
             'Invalid per-position diagnostic coverage')
    scene_ce=[sum(positions['ce'][a:b])/(b-a) for a,b in zip(offsets,offsets[1:])]
    need(len(row['per_scene_ce'])==16 and all(close(x,y) for x,y in zip(scene_ce,row['per_scene_ce']))
         and close(sum(scene_ce)/16,row['ce_loss']),'Scene-balanced CE reduction differs')
    cursor=0;pair_losses={'consistency':[],'path':[]}
    for length in lengths[::2]:
        for key in pair_losses:pair_losses[key].append(sum(positions[key][cursor:cursor+length])/length)
        cursor+=length
    for key,values in pair_losses.items():
        need(len(row['per_pair_'+key])==8 and all(close(x,y) for x,y in zip(values,row['per_pair_'+key]))
             and close(sum(values)/8,row[key+'_loss']),'Pair-balanced valid-prefix reduction differs')
    for key in ('pair_position_denominator','residual_difference_norms','B_norms'):
        values=row[key]
        need(len(values)==pair_count and all(isinstance(v,(int,float)) and not isinstance(v,bool)
             and math.isfinite(v) and (v>0 if key=='pair_position_denominator' else v>=0) for v in values),
             'Invalid residual/path norm or denominator')
    for key,norms in (('consistency',row['residual_difference_norms']),('path',row['B_norms'])):
        need(all(close(n*n/d,v) for n,d,v in zip(norms,row['pair_position_denominator'],positions[key])),
             'Reported per-prefix norm/denominator differs from penalty')
    identity=[row['sids'][i]==row['sids'][i+1] for i in range(0,16,2)]
    need(row['saturated_identity_pairs']==identity and all(type(x) is bool for x in row['saturated_identity_pairs']),
         'Saturated identity-pair ledger differs')
    cursor=0
    for same,length in zip(identity,lengths[::2]):
        if same:
            need(all(v==0. for key in ('consistency','path') for v in positions[key][cursor:cursor+length]),
                 'Same-SID auxiliary terms must be exactly zero and remain in the denominator')
        cursor+=length
    return True

def self_test():
    import copy
    import torch
    from gnnformer.paired_sequence_objectives import sequence_layout
    torch.set_num_threads(4)
    entries=[dict(step=step,exact_count=10,nll=1.) for step in STEPS]
    need(select_dev(entries)['step']==STEPS[0],'Earliest exact/NLL tie rule differs')
    entries[2]['nll']=.8;need(select_dev(entries)['step']==STEPS[2],'NLL tie-break differs')
    entries[4].update(exact_count=11,nll=2.);need(select_dev(entries)['step']==STEPS[4],'Exact count must dominate NLL')
    for malformed in (entries[:4],entries+entries[:1],[dict(x,nll=float('nan')) for x in entries],
                      [dict(x,exact_count=65) for x in entries]):
        try:select_dev(malformed)
        except ValueError:pass
        else:raise AssertionError('Malformed/incomplete development selection accepted')
    logits=torch.tensor([1.,2.,3.]);same=profile.metric(torch,logits,logits+7.);wrong=profile.metric(torch,logits,logits.flip(0))
    need(same['numerical_rule_passed'] and same['full_vocabulary_tv']<1e-15
         and not wrong['numerical_rule_passed'] and not wrong['top1_equal'],'Numerical diagnostic rules differ')
    sequences=[];sids=[]
    for pair in range(8):
        seq=[17,151645] if pair<4 else [17,22,151645]
        sequences.extend([seq,seq]);sids.extend([f'a{pair}',f'a{pair}' if pair==7 else f'b{pair}'])
    layout=sequence_layout(sequences);lengths=[len(seq) for seq in sequences];pair_lengths=lengths[::2]
    residual=[];path=[];norms=[];bounds=[]
    for pair,length in enumerate(pair_lengths):
        residual.extend([0. if pair==7 else 1.]*length);path.extend([0. if pair==7 else 4.]*length)
        norms.extend([0. if pair==7 else 1.]*length);bounds.extend([0. if pair==7 else 2.]*length)
    base=dict(ce_loss=2.,consistency_loss=7/8,path_loss=3.5,gradient_norm=2.,clipped=True,
        scene_lengths=lengths,scene_offsets=layout['offsets'],pair_lengths=pair_lengths,
        prefix_ids=[p for seq in layout['prefixes'] for p in seq],target_ids=layout['targets'],sids=sids,
        per_scene_ce=[2.]*16,per_pair_consistency=[1.]*7+[0.],per_pair_path=[4.]*7+[0.],
        position_losses=dict(ce=[2.]*len(layout['targets']),consistency=residual,path=path),
        pair_position_denominator=[1.]*len(residual),residual_difference_norms=norms,B_norms=bounds,
        saturated_identity_pairs=[False]*7+[True])
    for condition,coefficient in (('ce',0.),('consistency',1.)):
        row=dict(base,consistency_coefficient=coefficient,weighted_consistency_loss=coefficient*base['consistency_loss'],
                 loss=2.+coefficient*base['consistency_loss'])
        need(verify_objective(row,condition),'Valid ragged objective was rejected')
        malformed=[dict(row,loss=row['loss']+row['path_loss']),dict(row,consistency_coefficient=1.-coefficient)]
        wrong=copy.deepcopy(row);wrong['prefix_ids'][-1].append(99);malformed.append(wrong)
        wrong=copy.deepcopy(row);wrong['per_pair_consistency'][-1]=.1;malformed.append(wrong)
        wrong=copy.deepcopy(row);wrong['position_losses']['path'].pop();malformed.append(wrong)
        for changed in malformed:
            try:verify_objective(changed,condition)
            except ValueError:pass
            else:raise AssertionError('Wrong weight, prefix, identity or ragged coverage was accepted')
    return dict(passed=True,tests=['complete64_example_development_selection','exact_NLL_earliest',
        'retain_descriptive_numerical_failure','ragged_scene_pair_weighting','zero_identity_in_denominator',
        'reject_path_objective_weight','reject_wrong_coefficient_prefix_identity_coverage'])

def unit(args):
    out = OUT/f'selftest_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False); snapshot(out)
    tests = self_test()
    save(out/'summary.json', dict(tests, source_sha256=sources(), scope='Synthetic audit software only'))
    profile.local.index(out, 'V10 checkpoint audit software check', [('Summary', 'summary.json'), ('Frozen sources', 'source_hashes.json')])
    print(json.dumps(dict(passed=True, directory=str(out))), flush=True)


def verify_v10_pairing(config, train_plan, cache, directory):
    from scripts.native_vision_v10_pairs import load_pairs
    need(config['policy']==EXPECTED_POLICY,'V10 full registered policy differs')
    need(config['consistency_coefficient']==EXPECTED_POLICY['consistency_coefficients'][config['condition']],
         'Condition residual coefficient differs')
    need(config['pairing_file']==train_plan['pairing_file'] and config['pairing_sha256']==train_plan['pairing_sha256']
         and sha(config['pairing_file'])==config['pairing_sha256'],'Frozen pairing binding differs')
    pairing=read(config['pairing_file'])
    rebuilt=load_pairs(train_plan['train_manifest'],config['cache_binding']['file'],train_plan['schedule_file'])
    need(native.object_sha(pairing)==native.object_sha(rebuilt)==train_plan['pairing_object_sha256']
         and train_plan['actual_frozen_pair_equality_passed'] is True,'Pair metadata/query equality differs')
    pairs=pairing['pairs'];need(len(pairs)==918 and [p['slot'] for p in pairs]==list(range(918)),'Pair coverage differs')
    for pair in pairs:
        a,b=[cache['scenes'][sid] for sid in pair['sids']];k=pair['gold']
        need([a['n_frames'],b['n_frames']]==([8,16] if k<=8 else [16,16])
             and a['question']==b['question']==pair['question'] and a['gold']==b['gold']==k
             and a['target_ids']==b['target_ids']==pair['target_ids']
             and a['global_feature_ids']==b['global_feature_ids']==pair['global_feature_ids'],
             'Paired complete targets/question/prefix identities differ')
    rng=random.Random(config['seed']);expected_order=[]
    for epoch in range(1,41):
        slots=list(range(918));rng.shuffle(slots)
        for slot in slots:
            p=pairs[slot]
            for side,sid in enumerate(p['sids']):
                expected_order.append(dict(epoch=epoch,slot=slot,pair_id=p['pair_id'],question=p['question'],gold=p['gold'],
                    sid=sid,pair_side=side,pair_kind=p['pair_kind'],n_frames=p['n_frames'][side],replica=p['replicas'][side]))
    actual=read(directory/'presentations.json')
    need(actual==expected_order and len(actual)==73440
         and native.object_sha(actual)==config['order_sha256']==train_plan['order_sha256'][str(config['seed'])],
         'Persistent RNG/order/weight reconstruction differs')
    history=read(directory/'training.json');need(len(history)==4590,'Incomplete optimizer history')
    for step,row in enumerate(history,1):
        batch=expected_order[(step-1)*16:step*16];sids=[x['sid'] for x in batch]
        targets=[t for sid in sids for t in cache['scenes'][sid]['target_ids']]
        need(row['step']==step and row['sids']==sids and row['target_ids']==targets
             and row['pair_ids']==[batch[i]['pair_id'] for i in range(0,16,2)]
             and row['epochs']==[batch[i]['epoch'] for i in range(0,16,2)],'Actual update/prefix-target order differs')
        rate=.001*step/50 if step<=50 else .00001+.00099*(1+math.cos(math.pi*(step-50)/4540))/2
        need(math.isclose(row['lr'],rate,rel_tol=1e-10,abs_tol=1e-12),'Optimizer schedule differs')
        verify_objective(row,config['condition'])
    need(sum(len(r['target_ids']) for r in history)==177120,'Complete valid target-position coverage differs')
    artifacts={config['pairing_file']:config['pairing_sha256']}
    need(config['prior_result']==train_plan['prior_result'],'Prior-result dependency differs')
    prior=config['prior_result'];need(prior['previous_path_objective_failed'] is True,'Path efficacy failure must remain failed')
    for label in ('finalization','diagnostic'):
        b=prior[label];need(sha(b['file'])==b['sha256'],'Frozen prior dependency changed');artifacts[b['file']]=b['sha256']
    finalized=read(prior['finalization']['file'])
    need(finalized['passed'] and finalized['finalized'] and sha(finalized['final_acceptance_file'])==finalized['final_acceptance_sha256'],
         'V9 completed finalization binding differs')
    accepted=read(finalized['final_acceptance_file'])
    need(accepted['completed'] and accepted['verification_passed'] and accepted['accepted_vision_milestone'] is False,
         'Prior V9 failure was changed to acceptance')
    artifacts[finalized['final_acceptance_file']]=finalized['final_acceptance_sha256']
    diagnostic=read(prior['diagnostic']['file'])
    need(diagnostic['passed'] and diagnostic['diagnostic_only'] and diagnostic['no_fit']
         and diagnostic['vlm_forward_calls']==0 and diagnostic['synthetic_points']==432,'Prior diagnostic differs')
    binding=config['main_release'];need(sha(binding['file'])==binding['sha256'],'Main release changed')
    release=read(binding['file'])
    need(release['passed'] and release['plan_sha256']==config['plan_sha256'] and release['source_sha256']==config['source_sha256']
         and set(release['profiles'])=={'ce','consistency'} and release['per_main_seconds_cap']==2700
         and release['main_block_gpu_seconds_cap']==16200,'V10 main objective/resource release differs')
    artifacts[binding['file']]=binding['sha256']
    for condition,b in release['profiles'].items():
        path=str(Path(b['directory'])/'summary.json');need(sha(path)==b['summary_sha256'],'Training profile changed')
        p=read(path)
        need(p['profile'] and p['condition']==condition and p['arm']=='parallel' and p['seed']==14 and p['steps']==32
             and p['passed'] and p['computational_integrity_passed'] and p['no_dev_or_test_evaluation']
             and p['active_cache_replay']['passed'] and p['active_cache_replay']['counts_covered']==[0,9,10,16]
             and p['active_cache_replay']['prefixes_checked']==10
             and p['plan_sha256']==config['plan_sha256'] and p['source_sha256']==config['source_sha256'],
             'Unmatched training profile or full-prefix software coverage')
        zero=p['zero_initialization_auxiliary_gradients']
        need(all(zero.get(k) is True for k in ('passed','up_exactly_zero','all_residual_gradients_zero','all_path_gradients_zero'))
             and zero['residual_loss']==zero['path_loss']==0.,'Zero-U auxiliary gradient gate differs')
        artifacts[path]=b['summary_sha256']
    summary=read(directory/'summary.json')
    for field in ('training','first_gradients'):
        path,digest=summary[field+'_file'],summary[field+'_sha256']
        need(Path(path).resolve()==directory/(field+'.json') and sha(path)==digest,'Training/gradient evidence changed')
        artifacts[path]=digest
    return artifacts

def verify_run(torch, directory):
    """Selection uses every dev result, never a software-case or test outcome."""
    directory = Path(directory).resolve()
    need(directory.parent == MAIN, 'Require a canonical V10 main run directory')
    config = read(directory/'config.json'); summary = read(directory/'summary.json')
    arm, condition, seed = config['arm'], config['condition'], config['seed']; run_id = config['run_id']
    need(arm == 'parallel' and condition in ('ce', 'consistency') and seed in (14, 15) and config['profile'] is False
         and directory.name == run_id and run_id.startswith(f'run_{condition}_s{seed}_'), 'Main run identity differs')
    need(all(summary[key] == value for key, value in config.items()) and summary['passed'] is True
         and summary['completed'] is True and summary['computational_integrity_passed'] is True
         and summary['steps'] == 4590 and summary['native_test_count'] == 272, 'Incomplete canonical main run')
    policy = config['policy']
    need(policy['steps'] == 4590 and policy['epochs'] == 40 and policy['scene_slots_per_epoch'] == 1836
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
    need(cache['complete'] and cache['training_only'] and len(cache['scenes']) == 1782
         and sha(cache['plan_file']) == cache['plan_sha256'],
         'Main cache/feature-plan binding differs')
    feature_plan = read(cache['plan_file'])
    for key in ('model', 'runtime', 'processor', 'native_dtypes'):
        need(cache[key] == config[key], 'Recorded native identity differs from training features')
    from scripts.stage_native_vision_v10_features import verify_plan as verify_feature_plan
    from scripts.cache_native_vision_v10_features import verify_profile as verify_feature_profile
    need('runtime_extension' not in cache and cache['protocol']=='v10_parallel_local_training_features',
         'Require the new V10 training feature cache, not the older extension')
    need(verify_feature_plan(Path(cache['plan_file']),pixels=False)==feature_plan
         and cache['scenes']==feature_plan['scenes'] and set(cache['features'])==set(feature_plan['features']),
         'V10 feature inventory/provenance differs')
    verified_profile=verify_feature_profile(cache['profile_directory'],feature_plan,Path(cache['plan_file']))
    need(sha(Path(cache['profile_directory'])/'summary.json')==cache['profile_summary_sha256'],
         'Feature profile binding changed')
    for name,digest in cache['source_sha256'].items():need(sha(REPO/name)==digest,'Cache source changed')
    pair_artifacts = verify_v10_pairing(config, train_plan, cache, directory)
    history = read(directory/'training.json'); presentations = read(directory/'presentations.json')
    need(len(history) == 4590 and len(presentations) == 73440
         and sha(directory/'presentations.json') == config['presentations_sha256'], 'Training presentation coverage differs')
    for step, row in enumerate(history, 1):
        need(row['step'] == step and row['sids'] == [x['sid'] for x in presentations[(step-1)*16:step*16]]
             and len(row['target_ids']) == sum(row['scene_lengths']) and math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']),
             'Recorded updates differ from the frozen presentation order')
    selection = read(directory/'selection.json'); entries = selection['development']
    need(summary['development'] == entries and summary['selected'] == selection['selected'], 'Canonical selection records disagree')
    artifacts = dict(pair_artifacts)
    for entry in entries:
        step = entry['step']; devpath = directory/f'dev_{step}.json'; dev = read(devpath)
        need(Path(entry['dev_file']).resolve() == devpath and sha(devpath) == entry['dev_sha256']
             and dev['n'] == len(dev['rows']) == 64 and len({x['sid'] for x in dev['rows']}) == 64,
             'Incomplete or changed development sweep')
        exact = 0
        for row in dev['rows']:
            parsed = int(row['text'].strip()) if re.fullmatch(r'[0-9]+', row['text'].strip()) else None
            need(row['prediction'] == parsed and row['exact'] == (row['completed'] and parsed == row['gold'])
                 and math.isfinite(row['first_token_nll']), 'Development complete-answer selection statistic differs')
            exact += row['exact']
        nll = sum(row['first_token_nll'] for row in dev['rows'])/64
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
    artifacts[str(Path(cache['profile_directory'])/'summary.json')]=cache['profile_summary_sha256']
    for item in feature_plan['source_files'].values():artifacts[item['path']]=item['sha256']
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
    need({(x['condition'], x['seed']) for x in runs} == {(a, s) for a in ('ce', 'consistency') for s in (14, 15)},
         'Missing a registered selected model')
    first = runs[0]
    for run in runs:
        need(all(run['config'][key] == first['config'][key] for key in ('policy', 'source_sha256', 'plan_file', 'plan_sha256',
             'model', 'runtime', 'processor', 'native_dtypes')), 'Main models used different frozen protocols/native runtimes')
        need(run['native_profile'] == first['native_profile'] and run['native_profile_sha256'] == first['native_profile_sha256'],
             'Native software release differs across mains')
    for seed in (14, 15):
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
    plan = dict(schema_version=1, protocol='all_selected_v10_native_checkpoint_audit', source_sha256=sources(), tests=tests,
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
    profile.local.index(out, 'All four selected V10 checkpoints: CPU freeze', [('Plan', 'plan.json'), ('Summary', 'summary.json')])
    print(json.dumps(dict(passed=True, plan=str(path), plan_sha256=sha(path))), flush=True)


def verify(path):
    path = Path(path).resolve()
    need(path.is_relative_to(OUT) and sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Audit plan sidecar differs')
    plan = read(path)
    need(plan['schema_version'] == 1 and plan['protocol'] == 'all_selected_v10_native_checkpoint_audit'
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
         {(c,s) for c in ('ce','consistency') for s in (14,15)}
         and all(r['arm'] == 'parallel' for r in plan['runs'])
         and plan['cache_full_rows'] == 656 and plan['forced_calls'] == dict(model=40,visual=24),
         'Registered V10 condition/parallel-row audit coverage differs')
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
    profile.local.index(out, 'All-selected V10 native checkpoint audit', [('Frozen plan', 'plan.json'),
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
    lines = ['# V10 selected-checkpoint software audit', '',
        f"All four models completed. Native replay failures: {len(replay_failures)}/40; descriptive cache/full failures: {len(cache_failures)}/656.", '',
        '| Condition | Seed | Selected step | Replay passed | Cache/full passed |', '|---|---:|---:|---|---|']
    for row in model_results:
        lines.append(f"| {row['condition']} | {row['seed']} | {row['selected_step']} | {row['replay_gate_passed']} | {row['cache_full_numeric_passed']} |")
    lines += ['', 'The original mixed/cache failure remains failed. No software result changes checkpoint selection.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    profile.local.index(data, 'Selected V10 native state evidence', [('Summary', str(out/'summary.json')), ('Forward inventory', str(out/'forwards.json'))])
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
