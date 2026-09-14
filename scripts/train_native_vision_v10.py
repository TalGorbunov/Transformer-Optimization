"""V10 native training with complete answer sequences and matched residual consistency.

The deployed parallel SUM/SiLU operator is unchanged. All state harvesting,
training and evaluation use Slurm; native tests always read the image inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts import train_native_vision_v7 as v7
from scripts import native_vision_v7_runtime as native
from scripts.stage_native_vision_v7_features import MODEL, need, read, sha, object_sha
DATA = Path('/mnt/data/gabriele/gnn_transformer')
OUT = REPO / 'outputs/native_aggregation_vlm/v10'
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v10')
TRAIN = DATA / 'v10_balanced/main_manifest.json'
SCHEDULE = DATA / 'v10_balanced/schedule.json'
PAIRING = DATA / 'v10_balanced/pairing.json'
FRESH = DATA / 'v10_fresh'
POLICY = dict(
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

TEST_FILES = ('tests/test_paired_sequence_objectives.py', 'scripts/native_vision_v10_pairs.py')
OWN = tuple(sorted(set(name for name in v7.OWN if not name.startswith('slurm/')) | {
    'scripts/train_native_vision_v10.py', 'scripts/native_vision_v10_pairs.py',
    'gnnformer/paired_sequence_objectives.py', 'gnnformer/aggregation_consistency.py',
    'gnnformer/aggregation_path_bound.py', *TEST_FILES,
    'slurm/native_vision_v10_train_check.sbatch', 'slurm/native_vision_v10_train_profile.sbatch',
    'slurm/native_vision_v10_train.sbatch',
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



def lr(step):
    need(1 <= step <= POLICY['steps'], 'Step out of range')
    if step <= 50: return .001 * step / 50
    return 1e-5 + (.001 - 1e-5) * (1 + math.cos(math.pi * (step-50) / 4540)) / 2


def cache_binding(path):
    from scripts.stage_native_vision_v10_features import verify_plan
    cache = read(path)
    need(cache['complete'] and cache['training_only'] and len(cache['scenes']) == 1782,
         'Incomplete V10 training-only feature cache')
    need(sha(cache['plan_file']) == cache['plan_sha256'], 'Feature plan changed')
    plan = verify_plan(Path(cache['plan_file']), pixels=False)
    need(plan['protocol'] == 'v10_parallel_local_training_features' and cache['scenes'] == plan['scenes']
         and set(cache['features']) == set(plan['features']), 'Feature cache inventory differs')
    for name, digest in cache['source_sha256'].items():
        need(sha(REPO/name) == digest, 'Feature source changed')
    for key in ('model','runtime','processor'):
        need(cache[key] == plan[key], 'Cached native identity differs: ' + key)
    need(cache['native_dtypes'] == dict(norm='torch.float16',lm_head='torch.float16')
         and plan['native_dtype'] == 'torch.float16', 'Cached native dtype differs')
    profile_path = Path(cache['profile_directory'])/'summary.json'
    need(sha(profile_path) == cache['profile_summary_sha256'], 'Feature profile changed')
    profile = read(profile_path)
    need(profile['passed'] and profile['completed'] and profile['computational_integrity_passed']
         and profile['numerical_gate_passed'] and profile['runtime_projection_passed']
         and profile['plan_sha256'] == cache['plan_sha256'], 'Feature profile gate failed')
    return cache


def validate_fresh(bindings, train_manifest):
    manifest = bind_manifest(bindings, FRESH/'main_manifest.json')
    need(Path(manifest['dataset_root']).resolve() == FRESH and manifest['purpose'] == 'main'
         and {k:len(v['samples']) for k,v in manifest['splits'].items()} == {'test_N32':136,'test_N64':136},
         'Fresh test identity or cells differ')
    audit = read(manifest['audit_file'])
    need(all(audit.get(k) is True for k in ('passed','all_current_contexts_disjoint',
         'all_other_contexts_fresh_against_25_priors','all_generated_semantics_and_extensions_passed',
         'all_qa_hashes_and_gold_recounts_passed','all_image_hashes_and_dimensions_passed',
         'all_image_semantic_links_passed')), 'Fresh semantic audit failed')
    used = {r['content_sha256'] for split in train_manifest['splits'].values() for r in split['samples']}
    for cell, split in manifest['splits'].items():
        need(Counter(r['gold'] for r in split['samples']) == Counter({k:8 for k in range(17)}), 'Fresh K support differs')
        for row in split['samples']:
            need(row['content_sha256'] not in used and row['n_frames'] == int(cell.split('_N')[1]), 'Test overlap or length differs')
            used.add(row['content_sha256'])


def bind_prior(bindings, finalization_path, diagnostic_path):
    final = bind_file(bindings, finalization_path)
    need(Path(finalization_path).resolve() == REPO/'outputs/native_aggregation_vlm/v9/finalization/final_442084/summary.json',
         'Require the completed canonical V9 result')
    summary = read(finalization_path)
    need(summary['passed'] and summary['finalized'], 'V9 final verification did not complete')
    bind_file(bindings, summary['final_acceptance_file'], summary['final_acceptance_sha256'])
    accepted = read(summary['final_acceptance_file'])
    need(accepted['completed'] and accepted['verification_passed'] and not accepted['accepted_vision_milestone'],
         'Prior failed efficacy decision changed')
    for item in accepted['artifacts'].values(): bind_file(bindings, item['path'], item['sha256'])
    report_path = accepted['artifacts']['analysis']['path']; report = read(report_path)
    need(not report['primary_both_seeds'] and not report['practical_both_seeds'], 'Prior V9 failure must be retained')
    for mapping in (summary['source_sha256'], report['source_sha256']):
        for source, digest in mapping.items(): bind_file(bindings, REPO/source, digest)
    diagnostic = bind_file(bindings, diagnostic_path); d = read(diagnostic_path)
    need(d['passed'] and d['diagnostic_only'] and d['no_fit'] and d['vlm_forward_calls'] == 0
         and d['synthetic_points'] == 432, 'Completed V9 fixed-state diagnostic required')
    bind_file(bindings, d['analysis_file'], d['analysis_sha256']); bind_file(bindings, d['raw_file'], d['raw_sha256'])
    analysis = read(d['analysis_file'])
    need(Path(analysis['report_file']).resolve() == Path(report_path).resolve()
         and analysis['report_sha256'] == sha(report_path), 'Diagnostic ancestry differs')
    bind_file(bindings, analysis['report_completion_file'], analysis['report_completion_sha256'])
    for source, digest in d['source_sha256'].items(): bind_file(bindings, REPO/source, digest)
    return dict(finalization=final, diagnostic=diagnostic, previous_path_objective_failed=True,
                new_training_cache=True, no_selection_on_prior_diagnostic_predictions=True)


def check(args):
    out = OUT/('train_check_'+os.environ['SLURM_JOB_ID']); out.mkdir(parents=True, exist_ok=False)
    snapshot(out); index(out, 'V10 paired sequence training CPU plan')
    import torch
    from transformers import AutoProcessor
    from scripts.native_vision_v10_pairs import load_pairs, presentation_order
    torch.set_num_threads(4); begin=time.perf_counter(); helpers=test_helpers(); bindings={}
    train_manifest=bind_manifest(bindings, TRAIN)
    for path in (SCHEDULE, PAIRING): bind_file(bindings, path)
    dev=train_manifest['splits']['dev_N16']['samples']
    need(len(dev)==64 and Counter(r['gold'] for r in dev)==Counter({k:4 for k in range(16)}), 'Development support differs')
    validate_fresh(bindings,train_manifest)
    cache_path=args.parallel_cache.resolve(); cache_ref=bind_file(bindings,cache_path); cache=cache_binding(cache_path)
    bind_file(bindings,cache['plan_file'],cache['plan_sha256'])
    feature_plan=read(cache['plan_file'])
    for item in feature_plan['source_files'].values(): bind_file(bindings,item['path'],item['sha256'])
    bind_file(bindings,Path(cache['profile_directory'])/'summary.json',cache['profile_summary_sha256'])
    pairing=load_pairs(TRAIN,cache_path,SCHEDULE)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    for scene in cache['scenes'].values():
        need(scene['target_ids']==native.encode_target(processor.tokenizer,scene['gold']), 'Cached native full-answer target differs')
    states,feature_index=v7.load_features(torch,cache,'cpu')
    for pair in pairing['pairs']:
        a,b=[cache['scenes'][sid] for sid in pair['sids']]
        ga=states[[feature_index[fid] for fid in a['global_feature_ids']]]
        gb=states[[feature_index[fid] for fid in b['global_feature_ids']]]
        need(torch.equal(ga,gb) and not ga.requires_grad, 'Paired frozen causal query tensors differ')
    for path,digest in {r['file']:r['file_sha256'] for r in cache['features'].values()}.items(): bind_file(bindings,path,digest)
    orders={str(seed):object_sha(presentation_order(pairing['pairs'],seed)) for seed in POLICY['seeds']}
    need(len(set(orders.values()))==2, 'Registered seed orders coincide')
    native_profile=bind_native_profile(bindings,args.native_profile)
    prior=bind_prior(bindings,args.prior_finalization,args.diagnostic_summary)
    save(out/'pairing.json',pairing)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=sources(),artifact_bindings=bindings,
        cache_binding=cache_ref,pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),
        pairing_object_sha256=object_sha(pairing),order_sha256=orders,train_manifest=str(TRAIN),schedule_file=str(SCHEDULE),
        original_pairing_file=str(PAIRING),fresh_manifests={'main':str(FRESH/'main_manifest.json')},
        native_profile=native_profile,prior_result=prior,tests=helpers,actual_frozen_pair_equality_passed=True,
        no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin)
    save(out/'plan.json',plan); (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    verify_plan(out/'plan.json')
    save(out/'summary.json',dict(passed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=sources(),tests=helpers,no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify_plan(path):
    path=Path(path).resolve(); need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'V10 plan sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['policy']==POLICY and plan['source_sha256']==sources(), 'V10 source/policy changed')
    for artifact,digest in plan['artifact_bindings'].items(): need(sha(artifact)==digest,'Frozen artifact changed: '+artifact)
    need(sha(plan['pairing_file'])==plan['pairing_sha256'],'Paired schedule changed')
    from scripts.native_vision_v10_pairs import load_pairs
    rebuilt=load_pairs(plan['train_manifest'],plan['cache_binding']['file'],plan['schedule_file'])
    need(object_sha(rebuilt)==plan['pairing_object_sha256']==object_sha(read(plan['pairing_file'])),'Pair metadata changed')
    return plan


def verify_release(path,plan_path,plan):
    need(path is not None,'A measured V10 main release is required')
    release=read(path)
    need(release['passed'] and release['plan_sha256']==sha(plan_path) and release['source_sha256']==sources()
         and release['per_main_seconds_cap']==2700 and release['main_block_gpu_seconds_cap']==16200,'Main release differs')
    data=release['data_release'];need(sha(data['file'])==data['sha256'],'Independent data release changed')
    report_release=read(data['file'])
    need(report_release['passed'] and report_release['source_sha256']==release['report_source_sha256'],'Report/data source differs')
    for name,digest in release['report_source_sha256'].items(): need(sha(REPO/name)==digest,'Independent report source changed')
    for name,digest in report_release['data_bindings'].items(): need(sha(name)==digest,'Report data binding changed')
    need(set(release['profiles'])==set(POLICY['conditions']),'Both matched training profiles required')
    for condition,item in release['profiles'].items():
        p=Path(item['directory'])/'summary.json';need(sha(p)==item['summary_sha256'],'Training profile changed');s=read(p)
        need(s['profile'] and s['condition']==condition and s['seed']==14 and s['passed']
             and s['computational_integrity_passed'] and s['steps']==32 and s['no_dev_or_test_evaluation']
             and s['plan_sha256']==sha(plan_path) and s['source_sha256']==sources()
             and s['active_cache_replay']['passed'],'Unmatched or failed training profile')
        need(s['active_cache_replay']['counts_covered']==[0,9,10,16]
             and s['active_cache_replay']['prefixes_checked']==10,'Full native prefix replay coverage differs')
        zero=s['zero_initialization_auxiliary_gradients']
        need(all(zero.get(k) is True for k in ('passed','up_exactly_zero','all_residual_gradients_zero','all_path_gradients_zero'))
             and zero['residual_loss']==zero['path_loss']==0.,'Initial auxiliary gradient gate failed')
    need(set(release['projections'])==set(POLICY['conditions']) and all(
        p['passed'] and 0<p['projected_seconds']<=2700 for p in release['projections'].values()),'Measured main cost exceeds cap')
    return dict(file=str(Path(path).resolve()),sha256=sha(path))


def batch_states(torch,cache,states,index,sids):
    from gnnformer.paired_sequence_objectives import sequence_layout,pack_feature_sequences
    local=[];globals_=[];targets=[]
    for sid in sids:
        scene=cache['scenes'][sid];ids=scene['local_feature_ids'];gids=scene['global_feature_ids'];tokens=scene['target_ids']
        need(len(ids)==scene['n_frames'] and len(gids)==len(tokens) and all(len(r)==len(tokens) for r in ids),
             'Individual scene prefix segmentation differs')
        local.append(states[torch.tensor([[index[f] for f in row] for row in ids],device=states.device)])
        globals_.append(states[torch.tensor([index[f] for f in gids],device=states.device)]);targets.append(tokens)
    layout=sequence_layout(targets);h,g=pack_feature_sequences(local,globals_)
    need(g.shape[0]==layout['offsets'][-1] and all(len(tokens)==b-a for tokens,a,b in
         zip(targets,layout['offsets'],layout['offsets'][1:])),'Packed causal sequence offsets differ')
    return h,g,layout


def losses(torch,branch,local,g,layout,norm,head,condition,sids):
    from gnnformer.paired_sequence_objectives import sequence_objectives
    from gnnformer.aggregation_path_bound import native_path_aggregate
    delta=branch(local,g,output_dtype=torch.float32)
    z=native_path_aggregate(branch,local,g)
    logits=head(norm((g+delta.to(g.dtype)).unsqueeze(0)))[0]
    result=sequence_objectives(logits,delta,z,branch.aggregate_projection.weight,branch.up.weight,g,layout,
        eps=POLICY['consistency_epsilon'])
    ce,consistency,path=result['ce'],result['residual'],result['path']
    total=ce+POLICY['consistency_coefficients'][condition]*consistency
    with torch.no_grad():
        offsets=layout['offsets'];lengths=[b-a for a,b in zip(offsets,offsets[1:])];pair_lengths=lengths[::2]
        per_scene=[float(result['ce_positions'][a:b].mean()) for a,b in zip(offsets,offsets[1:])]
        per_pair={};cursor=0
        for label,values in (('consistency',result['residual_positions']),('path',result['path_positions'])):
            cursor=0;per_pair[label]=[]
            for length in pair_lengths:
                per_pair[label].append(float(values[cursor:cursor+length].mean()));cursor+=length
        identity=[sids[i]==sids[i+1] for i in range(0,len(sids),2)]
        cursor=0
        for same,length in zip(identity,pair_lengths):
            if same: need(bool((result['residual_positions'][cursor:cursor+length]==0).all()
                              and (result['path_positions'][cursor:cursor+length]==0).all()),'Saturated identity penalty is nonzero')
            cursor+=length
        left=torch.tensor(layout['left'],device=g.device);right=torch.tensor(layout['right'],device=g.device)
        norms=torch.linalg.vector_norm(delta[right]-delta[left],dim=-1)
        components=dict(scene_lengths=lengths,scene_offsets=offsets,pair_lengths=pair_lengths,
            prefix_ids=[prefix for row in layout['prefixes'] for prefix in row],
            per_scene_ce=per_scene,per_pair_consistency=per_pair['consistency'],per_pair_path=per_pair['path'],
            position_losses=dict(ce=result['ce_positions'].tolist(),consistency=result['residual_positions'].tolist(),path=result['path_positions'].tolist()),
            pair_position_denominator=result['denominator'].tolist(),residual_difference_norms=norms.tolist(),
            B_norms=result['bound'].tolist(),saturated_identity_pairs=identity)
    need(all(bool(torch.isfinite(v)) for v in (total,ce,consistency,path)),'Nonfinite objective')
    return total,ce,consistency,path,components


def active_cache_replay(torch,model,processor,branch,cache,states,index,manifest,dataout):
    from scripts.cache_native_vision_v7_features import replay_metrics
    observations=[];raw=[]
    for k in (0,9,10,16):
        sample=sorted((r for r in manifest['splits']['train_N16']['samples'] if r['gold']==k),key=lambda r:r['sid'])[0]
        h,g,layout=batch_states(torch,cache,states,index,[sample['sid'],sample['sid']])
        targets=cache['scenes'][sample['sid']]['target_ids']
        for t in range(len(targets)):
            prefix=targets[:t];bundle=native.prepare_scene(processor,sample,'parallel',prefix_ids=prefix)
            actual=native.forward_native(model,bundle,branch,capture=True,cpu=False)
            with torch.inference_mode():
                cached=model.lm_head(model.model.language_model.norm((g[t:t+1]+branch(h[:,t:t+1],g[t:t+1])).unsqueeze(0)))[0]
                ag=actual['global_states'];al=actual['local_states']
                replay=model.lm_head(model.model.language_model.norm((ag+branch(al,ag)).unsqueeze(0)))[0]
            logits=actual['global_logits'].unsqueeze(0);metrics=replay_metrics(torch,logits,replay);drift=replay_metrics(torch,logits,cached)
            observations.append(dict(sid=sample['sid'],gold=k,n_frames=16,prefix_ids=prefix,position=t,
                native_head_replay=metrics,cached_vs_native_descriptive=drift,cached_vs_native_gate=False,
                counters=actual['counters'],native_metadata=actual['metadata']))
            raw.append(dict(native=logits.detach().cpu(),replayed=replay.detach().cpu(),cached=cached.detach().cpu()))
    need(len(observations)==10 and {r['gold'] for r in observations}=={0,9,10,16},'Native strict-prefix coverage differs')
    path=dataout/'active_cache_replay.pt';torch.save(raw,path)
    return dict(passed=all(r['native_head_replay']['passed'] for r in observations),observations=observations,
        counts_covered=[0,9,10,16],prefixes_checked=10,raw_file=str(path),raw_sha256=sha(path),
        cached_numerical_failures=sum(not r['cached_vs_native_descriptive']['passed'] for r in observations))


def run(args):
    job = os.environ['SLURM_JOB_ID']
    runid = f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out = OUT / runid
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out)
    (out / 'INDEX.md').write_text('# V10 paired aggregation run\n\n'
        '[Configuration](config.json) · [Training](training.json) · [Summary](summary.json)\n')
    import torch
    import transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.native_vision_v10_pairs import presentation_order
    from scripts.stage_native_vision_v10_features import runtime_identity, model_metadata, native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4)
    start = time.perf_counter()
    args.plan = args.plan.resolve()
    plan = verify_plan(args.plan)
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds'], 'Unregistered condition/seed')
    need(not args.profile or args.seed == 14, 'Profiles use only registered seed 14')
    release = None if args.profile else verify_release(args.main_release, args.plan, plan)
    coefficient = POLICY['consistency_coefficients'][args.condition]
    cache = cache_binding(plan['cache_binding']['file'])
    dataout = DATA / 'native_aggregation_vlm_v10' / runid
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
        profile=args.profile, policy=POLICY, consistency_coefficient=coefficient, slurm_job_id=job,
        plan_file=str(args.plan), plan_sha256=sha(args.plan), source_sha256=sources(),
        cache_binding=plan['cache_binding'], pairing_file=plan['pairing_file'],
        pairing_sha256=plan['pairing_sha256'], prior_result=plan['prior_result'],
        initialized=initialized, initialized_sha256=object_sha(initialized),
        presentations_sha256=sha(out / 'presentations.json'), order_sha256=object_sha(order),
        model=cache['model'], runtime=cache['runtime'], processor=cache['processor'],
        native_dtypes=cache['native_dtypes'],
        hardware=dict(name=torch.cuda.get_device_name(0), cuda=torch.version.cuda, torch=str(torch.__version__)),
        model_and_features_load_seconds=loadseconds, checkpoint_directory=str(ckpt),
        data_directory=str(dataout), main_release=release)
    save(out / 'config.json', config)
    (out / 'INDEX.md').write_text('# V10 paired aggregation run\n\n'
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
             and rows[i]['pair_side'] == 0 and rows[i + 1]['pair_side'] == 1
             for i in range(0, 16, 2)), 'A training batch split or reversed a registered pair')
        sids = [r['sid'] for r in rows]
        local, g, layout = batch_states(torch, cache, states, feature_index, sids)
        optimizer.zero_grad(set_to_none=True)
        rate = lr(step)
        for group in optimizer.param_groups:
            group['lr'] = rate
        total, ce, consistency, path_loss, components = losses(torch, branch, local, g, layout, norm, head, args.condition, sids)
        if args.profile and step == 1:
            zero_initialization = zero_auxiliary_gradients(torch, branch, consistency, path_loss)
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
            consistency_loss=float(consistency), path_loss=float(path_loss),
            consistency_coefficient=coefficient, weighted_consistency_loss=float(coefficient * consistency), gradient_norm=float(gradient_norm),
            clipped=float(gradient_norm) > POLICY['clip_norm'], seconds=time.perf_counter() - begin,
            target_ids=layout['targets'], sids=sids,
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
        replay = active_cache_replay(torch, model, runtime.processor, branch, cache,
            states, feature_index, train_manifest, dataout)
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
        for purpose in ('main',):
            manifest = read(plan['fresh_manifests'][purpose])
            for cell, split in manifest['splits'].items():
                testrows.extend((purpose + '_' + cell, r) for r in split['samples'])
        need(len(testrows) == 272, 'Incomplete fresh native evaluation')
        evaluate(torch, model, runtime.processor, branch, testrows, out, 'test', dataout)
        result = dict(config, passed=True, completed=True, computational_integrity_passed=True,
            steps=4590, selected=best, development=developments, native_test_count=272,
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
        'outputs/native_aggregation_vlm/v9/finalization/final_442084/summary.json')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--main-release', type=Path)
    parser.add_argument('--condition', choices=POLICY['conditions'])
    parser.add_argument('--seed', type=int, choices=POLICY['seeds'])
    args = parser.parse_args()
    native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION') == 'cpu' and not os.environ.get('SLURM_JOB_GPUS'),
             'The V10 freeze requires a CPU-only Slurm allocation')
        need(all(getattr(args, name) is not None for name in ('parallel_cache', 'native_profile', 'diagnostic_summary')),
             '--check requires --parallel-cache, --native-profile, and --diagnostic-summary')
        check(args)
    else:
        need(args.plan is not None and args.condition is not None and args.seed is not None,
             'GPU modes require --plan, --condition, and --seed')
        run(args)


if __name__ == '__main__':
    main()
