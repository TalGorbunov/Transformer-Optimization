"""V11 native full-sequence training with matched common reads and alternative write sites.

The SUM/SiLU core is unchanged; the actual frozen last block stays live. All state harvesting,
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
from scripts import native_vision_v11_runtime as deployment
from scripts import native_vision_v11_batches as batches
from scripts import train_native_vision_v10 as previous
from scripts.stage_native_vision_v7_features import MODEL, need, read, sha, object_sha
DATA = Path('/mnt/data/gabriele/gnn_transformer')
OUT = REPO / 'outputs/native_aggregation_vlm/v11'
CKPT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v11')
TRAIN = DATA / 'v10_balanced/main_manifest.json'
SCHEDULE = DATA / 'v10_balanced/schedule.json'
PAIRING = DATA / 'v10_balanced/pairing.json'
FRESH = DATA / 'v11_fresh'
POLICY = dict(
    protocol='v11_common_read_native_memory_placement', native_arm='parallel',
    conditions=['pre_last', 'post_last'], seeds=[16, 17], epochs=40,
    pair_slots_per_epoch=918, scene_slots_per_epoch=1836, unique_training_scenes=1782,
    scene_presentations=73440, pair_presentations=36720, target_positions=177120,
    batch_size=16, pairs_per_batch=8, steps=4590,
    dev_steps=[918, 1836, 2754, 3672, 4590], dev_examples=64, test_examples=272,
    lr=.001, warmup=50, final_lr=.00001, weight_decay=0., clip_norm=1.,
    rank=96, hidden_size=3584, parameters=1041600, merge='sum', post_activation='silu',
    native_dtype='torch.float16', branch_dtype='torch.float32',
    consistency_coefficients={'pre_last': 1., 'post_last': 1.}, consistency_epsilon=1e-6,
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
    bootstrap_seed=20261011, bootstrap_replicates=10000,
    read_block_index=26, final_block_index=27, decoder_blocks=28,
    read_boundary='common_native_penultimate_output',
    training_replay='complete_global_sequences_all_historical_writes_actual_frozen_final_block',
)

PROFILE_CALLS=dict(training_updates=32,cached_sequence_replays=4,native_full_prefix_forwards=10,
    same_batch_last_block_replays=10,same_state_native_head_replays=10,extra_native_shaped_head_forwards=20,timing_generations=2,
    maximum_timing_model_forwards=8,maximum_full_model_forwards=18,visual_forwards=12,
    standalone_last_block_forwards=46,maximum_total_last_block_forwards=64)
TEST_FILES=('tests/test_native_vision_v11_batches.py','scripts/native_vision_v11_pairs.py')
OWN=tuple(sorted(set(previous.OWN) | {
    'scripts/train_native_vision_v11.py','scripts/native_vision_v11_runtime.py',
    'scripts/native_vision_v11_batches.py','scripts/native_vision_v11_pairs.py',
    'scripts/native_vision_v11_last_block.py','gnnformer/parallel_local_memory.py',
    'scripts/profile_native_vision_v11_memory.py','tests/test_native_vision_v11_last_block.py',*TEST_FILES,
    'slurm/native_vision_v11_train_check.sbatch','slurm/native_vision_v11_train_profile.sbatch',
    'slurm/native_vision_v11_train.sbatch'}))
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


def evaluate(torch, model, processor, branch, rows, out, label, dataout, condition):
    """Unchanged native parallel generation; archive all raw logits in FP32."""
    records, raw = [], []
    begin = time.perf_counter()
    for cell, sample in rows:
        bundle = native.prepare_scene(processor, sample, 'parallel')
        result = deployment.generate_native(model, processor, branch, bundle, write_location=condition)
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
    from scripts.stage_native_vision_v11_features import verify_plan,READ_BOUNDARY
    from scripts.cache_native_vision_v11_features import verify_profile
    cache=read(path)
    need(cache['complete'] and cache['training_only'] and len(cache['scenes'])==1782,
        'Incomplete V11 penultimate training cache')
    need(sha(cache['plan_file'])==cache['plan_sha256'],'Feature plan changed')
    plan=verify_plan(Path(cache['plan_file']),pixels=False)
    need(plan['protocol']==cache['protocol']=='v11_parallel_local_penultimate_training_features'
        and cache['read_boundary']==READ_BOUNDARY and cache['scenes']==plan['scenes']
        and set(cache['features'])==set(plan['features']) and len(cache['global_prompts'])==54,
        'Cache read boundary or scene/query/prompt inventory differs')
    for name,digest in cache['source_sha256'].items():need(sha(REPO/name)==digest,'Feature source changed')
    for key in ('model','runtime','processor','native_api','software_gate'):
        need(cache[key]==plan[key],'Feature identity differs: '+key)
    need(cache['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'), 'Native dtypes differ')
    profile=Path(cache['profile_directory']);need(sha(profile/'summary.json')==cache['profile_summary_sha256'],'Feature profile changed')
    verify_profile(profile,plan,Path(cache['plan_file']))
    return cache


def load_prompts(torch,cache,feature_plan,states,feature_index,device):
    from scripts.cache_native_vision_v11_features import validate_prompt
    prompts={}
    paths={r['file']:r['file_sha256'] for r in cache['global_prompts'].values()}
    for path,digest in paths.items():
        need(sha(path)==digest,'Full global prompt file changed')
        blob=torch.load(path,map_location='cpu',weights_only=True)
        need(blob['schema_version']==1,'Global prompt schema differs')
        for gid,descriptor in cache['global_prompts'].items():
            if descriptor['file']!=path:continue
            value=blob['prompts'][descriptor['key']]
            info=validate_prompt(torch,feature_plan,gid,value,states[feature_index[gid]].detach().cpu())
            need(info==descriptor['tensor_info'] and info['hidden_states']['sha256']==descriptor['state_sha256'],
                'Global prompt tensor identities differ')
            prompts[gid]={k:v.detach().to(device).clone() for k,v in value.items()}
    need(len(prompts)==54 and set(prompts)==set(feature_plan['global_prompts'])
        and not any(torch.is_inference(p['hidden_states']) or p['hidden_states'].requires_grad for p in prompts.values()),
        'Missing or nonordinary full global prompt states')
    return prompts


def bind_native_profile(bindings,directory):
    from scripts.stage_native_vision_v11_features import verify_software
    result=verify_software(Path(directory)/'summary.json',deep=True)
    bind_file(bindings,result['file'],result['sha256']);bind_file(bindings,result['plan_file'],result['plan_sha256'])
    for name,digest in result['source_sha256'].items():bind_file(bindings,REPO/name,digest)
    return result


def bind_prior(bindings,finalization_path):
    expected=REPO/'outputs/native_aggregation_vlm/v10/finalization/final_442190/summary.json'
    need(Path(finalization_path).resolve()==expected,'Require completed canonical V10 result')
    summary=read(finalization_path);need(summary['passed'] and summary['finalized'],'V10 verification incomplete')
    final=bind_file(bindings,finalization_path)
    acceptance=bind_file(bindings,summary['final_acceptance_file'],summary['final_acceptance_sha256'])
    decision=read(acceptance['file'])
    need(decision['completed'] and decision['verification_passed'] and not decision['accepted_vision_milestone'],
        'Prior negative V10 decision must remain unchanged')
    return dict(finalization=final,acceptance=acceptance,prior_milestone_failed=True,
        prior_is_historical_not_matched_control=True,new_penultimate_cache=True)


def check(args):
    out=OUT/('train_check_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
    snapshot(out);index(out,'V11 complete-sequence training CPU plan; profiles only until independent main release')
    import torch
    from transformers import AutoProcessor
    from scripts.native_vision_v11_pairs import load_pairs,presentation_order
    from scripts.profile_native_vision_v11_memory import old
    torch.set_num_threads(4);begin=time.perf_counter();helpers=test_helpers();bindings={}
    manifest=bind_manifest(bindings,TRAIN)
    for path in (SCHEDULE,PAIRING):bind_file(bindings,path)
    dev=manifest['splits']['dev_N16']['samples']
    need(len(dev)==64 and Counter(r['gold'] for r in dev)==Counter({k:4 for k in range(16)}),'Development support differs')
    cache_path=args.parallel_cache.resolve();cache_ref=bind_file(bindings,cache_path);cache=cache_binding(cache_path)
    bind_file(bindings,cache['plan_file'],cache['plan_sha256']);feature_plan=read(cache['plan_file'])
    for item in feature_plan['source_files'].values():bind_file(bindings,item['path'],item['sha256'])
    bind_file(bindings,Path(cache['profile_directory'])/'summary.json',cache['profile_summary_sha256'])
    pairing=load_pairs(TRAIN,cache_path,SCHEDULE)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    for scene in cache['scenes'].values():
        need(scene['target_ids']==native.encode_target(processor.tokenizer,scene['gold']),'Cached full native target differs')
    states,feature_index=v7.load_features(torch,cache,'cpu')
    prompts=load_prompts(torch,cache,feature_plan,states,feature_index,'cpu')
    for pair in pairing['pairs']:
        h,g,layout,replay=batches.batch_states(torch,cache,feature_plan,states,feature_index,prompts,pair['sids'])
        need(torch.equal(g[:len(pair['target_ids'])],g[len(pair['target_ids']):])
            and len(replay['query_indices'])==2,'Actual paired states or reconstructed prefixes differ')
    for path,digest in {r['file']:r['file_sha256'] for r in list(cache['features'].values())+list(cache['global_prompts'].values())}.items():
        bind_file(bindings,path,digest)
    orders={str(seed):object_sha(presentation_order(pairing['pairs'],seed)) for seed in POLICY['seeds']}
    need(len(set(orders.values()))==2,'Different seed orders coincide')
    native_profile=bind_native_profile(bindings,args.native_profile)
    need(native_profile['file']==cache['software_gate']['file'] and native_profile['sha256']==cache['software_gate']['sha256'],
        'Training and cache do not share the same native software gate')
    prior=bind_prior(bindings,args.prior_finalization)
    software=read(native_profile['plan_file']);timing_scenes,timing_manifest=old.mixed.selected_sources()
    bind_file(bindings,timing_manifest['path'],timing_manifest['sha256'])
    expected={c['sid'] for c in software['cases'][:2]}
    timing_scenes=sorted((r for r in timing_scenes if r['sid'] in expected),key=lambda r:r['n_frames'])
    need(len(timing_scenes)==2 and [r['n_frames'] for r in timing_scenes]==[16,64], 'Old fixed timing fixture identities differ')
    save(out/'pairing.json',pairing)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=sources(),artifact_bindings=bindings,
        cache_binding=cache_ref,pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),
        pairing_object_sha256=object_sha(pairing),order_sha256=orders,train_manifest=str(TRAIN),schedule_file=str(SCHEDULE),
        original_pairing_file=str(PAIRING),fresh_manifests={},fresh_test_bound_by_future_independent_release=True,
        native_profile=native_profile,prior_result=prior,tests=helpers,actual_frozen_pair_equality_passed=True,
        complete_global_reconstruction_passed=True,timing_scenes=timing_scenes,profile_calls=PROFILE_CALLS,
        no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify_plan(out/'plan.json')
    save(out/'summary.json',dict(passed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        source_sha256=sources(),tests=helpers,no_model_loaded=True,profile_calls=PROFILE_CALLS,
        fresh_test_bound_by_future_independent_release=True,elapsed_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify_release(path,plan_path,plan):
    # A profile-only CPU plan does not authorize a main. Fresh tests and the
    # independent reporter must be frozen in a separate measured release.
    need(path is not None,'A separately frozen V11 fresh-data/report/resource main release is required')
    release=read(path)
    need(release.get('protocol')=='v11_native_memory_main_release' and release['passed']
        and release['plan_sha256']==sha(plan_path) and release['source_sha256']==sources(), 'Main release source or training plan differs')
    need(set(release['profiles'])==set(POLICY['conditions']) and release['per_main_seconds_cap']>0
        and release['campaign_gpu_seconds_cap']>0 and release['campaign_budget']['passed'],'Both measured profiles and campaign accounting required')
    for condition,item in release['profiles'].items():
        p=Path(item['directory'])/'summary.json';need(sha(p)==item['summary_sha256'],'Training profile changed');s=read(p)
        need(s['completed'] and s['profile'] and s['condition']==condition and s['seed']==16 and s['passed']
            and s['computational_integrity_passed'] and s['steps']==32 and s['no_dev_or_test_evaluation']
            and s['plan_sha256']==sha(plan_path) and s['source_sha256']==sources() and s['active_cache_replay']['passed']
            and s['active_cache_replay']['temporal_credit']['passed'] and s['checkpoint_roundtrip_passed']
            and all(s['all_parameter_gradients_exercised'].values()),'Unmatched or failed training profile')
    data=release['data_release'];need(sha(data['file'])==data['sha256'],'Independent fresh data release changed')
    audit=read(data['file']);need(audit['passed'] and audit['source_sha256']==release['report_source_sha256'],
        'Independent V11 reporter/data audit failed')
    for source,digest in release['report_source_sha256'].items():need(sha(REPO/source)==digest,'Independent reporter source changed')
    for artifact,digest in audit['data_bindings'].items():need(sha(artifact)==digest,'Independently audited artifact changed')
    fresh=release['fresh_manifest'];need(sha(fresh['file'])==fresh['sha256']
        and str(Path(fresh['file']).resolve()) in audit['data_bindings'],'Fresh tests were not independently frozen')
    manifest=read(fresh['file'])
    need(Path(manifest['dataset_root']).resolve()==FRESH and manifest['purpose']=='main'
        and {k:len(v['samples']) for k,v in manifest['splits'].items()}=={'test_N32':136,'test_N64':136},
        'Fresh test root, scope or size differs')
    current={r['content_sha256'] for split in read(TRAIN)['splits'].values() for r in split['samples']}
    for cell,split in manifest['splits'].items():
        need(Counter(r['gold'] for r in split['samples'])==Counter({k:8 for k in range(17)}),'Fresh count support differs')
        for row in split['samples']:
            need(row['content_sha256'] not in current and row['n_frames']==int(cell.split('_N')[1]),'Fresh train/dev/test overlap or length differs')
            current.add(row['content_sha256'])
    need(set(release['projections'])==set(POLICY['conditions']) and all(p['passed']
        and 0<p['projected_seconds']<=release['per_main_seconds_cap'] for p in release['projections'].values()),'Measured main cost exceeds release')
    return dict(file=str(Path(path).resolve()),sha256=sha(path),fresh_manifest=fresh)


def verify_plan(path):
    path=Path(path).resolve(); need(path.is_relative_to(OUT) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'V11 plan path/sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['profile_calls']==PROFILE_CALLS and plan['fresh_manifests']=={}
         and plan['fresh_test_bound_by_future_independent_release'] is True,'V11 source/policy/profile scope changed')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'code'/name.replace('/','_'))==digest,'Frozen CPU source copy changed')
    for artifact,digest in plan['artifact_bindings'].items(): need(sha(artifact)==digest,'Frozen artifact changed: '+artifact)
    need(sha(plan['pairing_file'])==plan['pairing_sha256'],'Paired schedule changed')
    from scripts.native_vision_v11_pairs import load_pairs
    rebuilt=load_pairs(plan['train_manifest'],plan['cache_binding']['file'],plan['schedule_file'])
    need(object_sha(rebuilt)==plan['pairing_object_sha256']==object_sha(read(plan['pairing_file'])),'Pair metadata changed')
    return plan




def losses(torch,branch,local,g,layout,replay_inputs,model,condition,sids,*,capture_initial=False):
    result=batches.forward_objectives(torch,branch,local,g,layout,replay_inputs,
        model.model.language_model.layers[-1],model.model.language_model.norm,model.lm_head,
        model.model.language_model.rotary_emb,condition,eps=POLICY['consistency_epsilon'])
    delta=result['delta'];ce,consistency,path=result['ce'],result['residual'],result['path'];total=result['total']
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
            B_norms=result['bound'].tolist(),saturated_identity_pairs=identity,
            replay_sequence_lengths=replay_inputs['attention_mask'].sum(-1).tolist(),
            replay_query_indices=replay_inputs['query_indices'],packed_sequence_width=replay_inputs['input_ids'].shape[1])
    if capture_initial:components['initial_native_query_logits']=v7.tensor_info(result['replay']['logits'])
    need(all(bool(torch.isfinite(v)) for v in (total,ce,consistency,path)),'Nonfinite objective')
    return total,ce,consistency,path,components



def active_cache_replay(torch,model,processor,branch,cache,feature_plan,states,feature_index,prompts,manifest,dataout,condition):
    from contextlib import nullcontext
    from gnnformer.runtime import get_layers,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_memory import ParallelLocalMemory
    from scripts.native_vision_v11_last_block import replay_last_block
    from scripts.profile_native_vision_v11_memory import Capture,cpu_copy,verify_write,old
    layers=get_layers(model);language=model.model.language_model;norm=language.norm;head=model.lm_head
    observations=[];raws=[];temporal=None
    for k in (0,9,10,16):
        sample=sorted((r for r in manifest['splits']['train_N16']['samples'] if r['gold']==k),key=lambda r:r['sid'])[0]
        sid=sample['sid'];h,g,layout,inputs=batches.batch_states(torch,cache,feature_plan,states,feature_index,prompts,[sid,sid])
        with (nullcontext() if k==10 else torch.no_grad()):
            result=batches.forward_objectives(torch,branch,h,g,layout,inputs,layers[-1],norm,head,language.rotary_emb,condition)
        cached_logits=cpu_copy(result['replay']['logits']);cached_delta=cpu_copy(result['delta'])
        if k==10:
            loss=torch.nn.functional.cross_entropy(result['replay']['logits'][1:2].float(),
                torch.tensor([layout['targets'][1]],device=model.device))
            gradient=torch.autograd.grad(loss,result['delta'],allow_unused=False)[0]
            need(gradient.shape==(6,3584) and bool(torch.isfinite(gradient).all()) and bool(gradient[1].ne(0).any()),
                'Full reconstructed sequence has no finite current-delta CE gradient')
            earlier=bool(gradient[0].ne(0).any());future=bool(gradient[2].ne(0).any());other=bool(gradient[3:].ne(0).any())
            need(earlier==(condition=='pre_last') and not future and not other,'Full sequence temporal or cross-scene credit structure differs')
            temporal=dict(passed=True,sid=sid,condition=condition,target_position=1,target_token_id=layout['targets'][1],
                earlier_delta_nonzero=earlier,current_delta_nonzero=True,future_delta_nonzero=future,other_scene_delta_nonzero=other,
                independent_delta_output=True,gradient_norms=[float(torch.linalg.vector_norm(v)) for v in gradient],
                loss=float(loss.detach()))
            torch.save(dict(gradient=cpu_copy(gradient),delta=cached_delta,logits=cached_logits,layout=layout,replay_inputs=cpu_copy(inputs)),
                dataout/'temporal_credit.pt')
        del result
        targets=cache['scenes'][sid]['target_ids']
        for t in range(len(targets)):
            bundle=native.prepare_scene(processor,sample,'parallel',prefix_ids=targets[:t]);meta=bundle['metadata']
            actual_layout=native.audit_layout(get_rope_index_fn(model),bundle);physical=list(range(meta['original_prompt_width']-1,meta['prompt_width']))
            controller=ParallelLocalMemory(layers[-2],norm,branch,n_local_rows=16,write_location=condition,
                query_indices=physical,stream_positions=physical,capture=True)
            with controller:
                with Capture(model) as captured:
                    with torch.inference_mode():
                        native_output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
                controller.assert_complete();raw=captured.value;raw['fusion']=controller.export_last_capture(cpu=True)
            need(not controller.active and controller.calls==controller.read_calls==controller.norm_calls==1,'Native prefix hook coverage differs')
            need(torch.equal(raw['position_ids'],actual_layout['position_ids'])
                and torch.equal(raw['attention_mask'],bundle['inputs']['attention_mask']), 'Native full-prefix mask/positions differ')
            mask_audit=old.mask_check(torch,raw['last_block_mask'],raw['attention_mask'],raw['cache_position'])
            write=verify_write(raw,condition,physical,physical)
            raw['native_logits']=cpu_copy(native_output.logits);raw['input_ids']=cpu_copy(bundle['inputs']['input_ids'])
            need(native_output.logits.dtype==torch.float16 and native_output.past_key_values is None,'Native full-prefix dtype/cache differs')
            with torch.no_grad():
                head_logits=head.forward(norm.forward(raw['actual_norm_input'][:,-1:,:].to(model.device)))
                replay=replay_last_block(layers[-1],norm,head,language.rotary_emb,
                    hidden_states=raw['common_penultimate_hidden'].to(model.device),input_ids=raw['input_ids'].to(model.device),
                    attention_mask=raw['attention_mask'].to(model.device),position_ids=raw['position_ids'].to(model.device),
                    query_indices=[[] for _ in range(16)]+[physical],deltas=raw['fusion']['delta'].to(model.device),placement=condition)
                block_logits=head.forward(norm.forward(replay['final_norm_input'][:,-1:,:]))
            def same(a,b):
                return a is None and b is None or isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu())
            need(same(replay['attention_mask'],raw['last_block_mask']) and same(replay['cache_position'],raw['cache_position'])
                and all(same(a,b) for a,b in zip(replay['position_embeddings'],raw['position_embeddings']))
                and same(replay['block_input'],raw['actual_last_block_input']), 'Same-batch native replay layout or historical writes differ')
            actual=raw['native_logits'][-1,-1]
            head_metric=old.metric(torch,actual,head_logits[-1,-1].detach().cpu())
            block_metric=old.metric(torch,actual,block_logits[-1,-1].detach().cpu())
            cached_metric=old.metric(torch,actual,cached_logits[t]);cached_metric['binding']=False
            common=raw['fusion']['global_states'][-1];local=raw['fusion']['local_states'][:,-1]
            hidden_errors=dict(global_l2=float(torch.linalg.vector_norm(g[t].float().cpu()-common.float())),
                local_l2=float(torch.linalg.vector_norm(h[:,t].float().cpu()-local.float())))
            observations.append(dict(sid=sid,gold=k,n_frames=16,prefix_ids=targets[:t],position=t,write_location=condition,
                native_head_replay=head_metric,same_batch_last_block_replay=block_metric,cached_vs_native_descriptive=cached_metric,
                cached_hidden_errors_descriptive=hidden_errors,cached_vs_native_gate=False,native_mask_audit=mask_audit,
                historical_writes=write,native_metadata=meta))
            raw.update(cached_complete_sequence_logits=cached_logits,cached_delta=cached_delta,
                same_state_head_logits=cpu_copy(head_logits),same_batch_last_block_logits=cpu_copy(block_logits))
            path=dataout/f'active_replay_K{k}_t{t}.pt';torch.save(raw,path)
            raws.append(dict(path=str(path),sha256=sha(path)))
            del raw,replay,native_output,head_logits,block_logits
    need(len(observations)==10 and temporal is not None,'Incomplete fixed training-only prefix/credit coverage')
    return dict(passed=all(r['native_head_replay']['numerical_rule_passed'] and r['same_batch_last_block_replay']['numerical_rule_passed']
        for r in observations),observations=observations,counts_covered=[0,9,10,16],prefixes_checked=10,
        cached_complete_sequence_replays=4,native_full_prefix_forwards=10,same_batch_last_block_replays=10,
        extra_native_shaped_head_forwards=20,raw_artifacts=raws,temporal_credit=temporal,
        temporal_raw_file=str(dataout/'temporal_credit.pt'),temporal_raw_sha256=sha(dataout/'temporal_credit.pt'),
        cached_numerical_failures=sum(not r['cached_vs_native_descriptive']['numerical_rule_passed'] for r in observations))


def profile_timing(torch,model,processor,branch,plan,condition,dataout):
    rows=[]
    for sample in plan['timing_scenes']:
        started=time.perf_counter();bundle=native.prepare_scene(processor,sample,'parallel');preprocessing=time.perf_counter()-started
        result=deployment.generate_native(model,processor,branch,bundle,write_location=condition)
        steps=len(result['generated_ids']);need(1<=steps<=4 and result['counters']['visual']==1,'Natural timing execution differs')
        path=dataout/f'timing_N{sample["n_frames"]}.pt';torch.save(result,path)
        rows.append(dict(sid=sample['sid'],n_frames=sample['n_frames'],generated_tokens=steps,counters=result['counters'],
            preprocessing_seconds=preprocessing,generation_seconds=result['model_seconds'],
            four_token_seconds_bound=preprocessing+result['model_seconds']*4/steps,
            raw_file=str(path),raw_sha256=sha(path),natural_eos_retained=True,no_accuracy_scoring=True))
    need([r['n_frames'] for r in rows]==[16,64],'Timing fixture lengths differ')
    return dict(passed=True,rows=rows,selection='same fixed old software N16K3/N64K6 scenes',
        T16=rows[0]['four_token_seconds_bound'],T64=rows[1]['four_token_seconds_bound'],
        bound_rule='preprocessing + measured natural generation seconds * 4 / observed tokens',maximum_new_tokens=4)


def run(args):
    job = os.environ['SLURM_JOB_ID']
    runid = f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out = OUT / runid
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out)
    (out / 'INDEX.md').write_text('# V11 paired aggregation run\n\n'
        '[Configuration](config.json) · [Training](training.json) · [Summary](summary.json)\n')
    import torch
    import transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.native_vision_v11_pairs import presentation_order
    from scripts.stage_native_vision_v11_features import runtime_identity, model_metadata, native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4)
    start = time.perf_counter()
    args.plan = args.plan.resolve()
    plan = verify_plan(args.plan)
    cpu=read(args.plan.parent/'summary.json')
    need(cpu['passed'] and cpu['plan_sha256']==sha(args.plan) and cpu['source_sha256']==sources(),'Completed exact CPU training freeze required')
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds'], 'Unregistered condition/seed')
    need(not args.profile or args.seed == 16, 'Profiles use only registered seed 16')
    release = None if args.profile else verify_release(args.main_release, args.plan, plan)
    coefficient = POLICY['consistency_coefficients'][args.condition]
    cache = cache_binding(plan['cache_binding']['file'])
    dataout = DATA / 'native_aggregation_vlm_v11' / runid
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
    feature_plan=read(cache['plan_file'])
    prompts=load_prompts(torch,cache,feature_plan,states,feature_index,runtime.device)
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
        profile=args.profile, policy=POLICY, consistency_coefficient=coefficient, write_location=args.condition, slurm_job_id=job,
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
    (out / 'INDEX.md').write_text('# V11 paired aggregation run\n\n'
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
    parameter_versions={name:p._version for name,p in model.named_parameters()}
    execution_counts=dict(model=0,visual=0,last_block=0,head=0)
    def counter(name):
        def hook(*_):execution_counts[name]+=1
        return hook
    handles=[]
    if args.profile:
        handles=[module.register_forward_pre_hook(counter(name)) for name,module in (
            ('model',model),('visual',model.model.visual),('last_block',model.model.language_model.layers[-1]),('head',model.lm_head))]
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, totalsteps + 1):
        begin = time.perf_counter()
        rows = order[(step - 1) * 16:step * 16]
        need(len(rows) == 16 and all(rows[i]['pair_id'] == rows[i + 1]['pair_id']
             and rows[i]['pair_side'] == 0 and rows[i + 1]['pair_side'] == 1
             for i in range(0, 16, 2)), 'A training batch split or reversed a registered pair')
        sids = [r['sid'] for r in rows]
        local,g,layout,replay_inputs=batches.batch_states(torch,cache,feature_plan,states,feature_index,prompts,sids)
        optimizer.zero_grad(set_to_none=True)
        rate = lr(step)
        for group in optimizer.param_groups:
            group['lr'] = rate
        total, ce, consistency, path_loss, components = losses(torch,branch,local,g,layout,replay_inputs,model,args.condition,sids,capture_initial=(step==1))
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
        if args.profile:save(out/'training.json',trainlog)
        if step % 100 == 0:
            print(json.dumps(dict(step=step, ce=float(ce), consistency=float(consistency), path=float(path_loss), total=float(total))), flush=True)
        if not args.profile and step in POLICY['dev_steps']:
            checkpoint = ckpt / f'step_{step}.pt'
            torch.save(dict(branch=branch.state_dict(), step=step, config=config), checkpoint)
            dev = evaluate(torch, model, runtime.processor, branch, devrows, out, f'dev_{step}', dataout,args.condition)
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
        replay=active_cache_replay(torch,model,runtime.processor,branch,cache,feature_plan,states,feature_index,prompts,train_manifest,dataout,args.condition)
        save(out / 'active_cache_replay.json', replay)
        timing=profile_timing(torch,model,runtime.processor,branch,plan,args.condition,dataout)
        save(out/'native_timing.json',timing)
        expected_models=10+sum(r['generated_tokens'] for r in timing['rows'])
        need(execution_counts==dict(model=expected_models,visual=12,last_block=46+expected_models,head=46+expected_models)
            and expected_models<=18,'Actual training/native/replay forward inventory differs')
        result = dict(config, passed=replay['passed'], completed=True, computational_integrity_passed=True,
            steps=32, active_cache_replay=replay, native_timing=timing, no_dev_or_test_evaluation=True,
            profile_calls=PROFILE_CALLS,actual_calls=execution_counts,extra_native_shaped_head_forwards=20,
            standalone_last_block_forwards=46,peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
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
        manifest=read(release['fresh_manifest']['file']);testrows=[]
        for cell,split in manifest['splits'].items():testrows.extend(('main_'+cell,r) for r in split['samples'])
        need(len(testrows) == 272, 'Incomplete fresh native evaluation')
        evaluate(torch, model, runtime.processor, branch, testrows, out, 'test', dataout,args.condition)
        result = dict(config, passed=True, completed=True, computational_integrity_passed=True,
            steps=4590, selected=best, development=developments, native_test_count=272,
            test_file=str(out / 'test.json'), test_sha256=sha(out / 'test.json'))
    for handle in handles:handle.remove()
    need(parameter_versions=={name:p._version for name,p in model.named_parameters()}
        and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone state changed')
    result.update(frozen_backbone_unchanged=True,training_seconds=sum(r['seconds'] for r in trainlog),
        training_file=str(out / 'training.json'), training_sha256=sha(out / 'training.json'),
        first_gradients_file=str(out / 'first_gradients.json'), first_gradients_sha256=sha(out / 'first_gradients.json'),
        elapsed_seconds=time.perf_counter() - start)
    verify_plan(args.plan)
    if release is not None:
        need(sha(release['file']) == release['sha256'], 'Main release changed during execution')
        verify_release(args.main_release, args.plan, plan)
    save(out / 'summary.json', result)
    need(result['passed'],'Binding same-state or same-batch replay failed; preserve all numerical evidence')
    print(json.dumps(dict(run_id=runid, completed=True, elapsed_seconds=result['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--profile', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--parallel-cache', type=Path)
    parser.add_argument('--native-profile', type=Path)
    parser.add_argument('--prior-finalization', type=Path, default=REPO /
        'outputs/native_aggregation_vlm/v10/finalization/final_442190/summary.json')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--main-release', type=Path)
    parser.add_argument('--condition', choices=POLICY['conditions'])
    parser.add_argument('--seed', type=int, choices=POLICY['seeds'])
    args = parser.parse_args()
    native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION') == 'cpu' and not os.environ.get('SLURM_JOB_GPUS'),
             'The V11 freeze requires a CPU-only Slurm allocation')
        need(all(getattr(args, name) is not None for name in ('parallel_cache', 'native_profile')),
             '--check requires --parallel-cache and --native-profile')
        check(args)
    else:
        need(args.plan is not None and args.condition is not None and args.seed is not None,
             'GPU modes require --plan, --condition, and --seed')
        run(args)


if __name__ == '__main__':
    main()
