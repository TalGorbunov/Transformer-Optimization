"""V13 matched learned-null training, with frozen native parallel inference.

Both arms learn from complete native answer CE and paired residual consistency.
A separate occurrence-weighted null regression updates only the predictor; the
main objective differentiates through its query and prediction normally.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import train_native_vision_v7 as v7
from scripts import train_native_vision_v10 as previous
from scripts import native_vision_v7_runtime as native
from scripts.stage_native_vision_v7_features import MODEL,need,read,sha,object_sha
DATA=Path('/mnt/data/gabriele/gnn_transformer')
OUT=REPO/'outputs/native_aggregation_vlm/v13'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v13')
TRAIN=DATA/'v10_balanced/main_manifest.json'
SCHEDULE=DATA/'v10_balanced/schedule.json'
PAIRING=DATA/'v10_balanced/pairing.json'
FRESH=DATA/'v11_fresh'
PRIOR=REPO/'outputs/native_aggregation_vlm/v12/reference_study/report_442359/summary.json'
POLICY=dict(
    protocol='v13_learned_null_centering',native_arm='parallel',conditions=['centered','offset'],seeds=[16,17],epochs=40,
    pair_slots_per_epoch=918,scene_slots_per_epoch=1836,unique_training_scenes=1782,
    scene_presentations=73440,pair_presentations=36720,target_positions=177120,
    batch_size=16,pairs_per_batch=8,steps=4590,dev_steps=[918,1836,2754,3672,4590],dev_examples=64,test_examples=272,
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,
    optimizer='one_AdamW_two_parameter_groups_core_and_predictor',clipping='independent_norm1_core_and_predictor',
    rank=96,hidden_size=3584,core_parameters=1041600,predictor_parameters=18624,parameters=1060224,
    merge='sum',post_activation='silu',native_dtype='torch.float16',branch_dtype='torch.float32',predictor_dtype='torch.float32',
    correction_coefficients={'centered':'actual_scene_N','offset':1},
    predictor_input='continuous_live_q_equals_Wq_RMS_g_no_N_ID_or_prefix_lookup',
    predictor_initialization='core_then_predictor_from_same_seed_second_weight_and_bias_exact_zero',
    consistency_coefficients={'centered':1.,'offset':1.},consistency_epsilon=1e-6,
    ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',
    within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',
    path_loss_role='diagnostic_only_on_corrected_aggregate_no_loss_weight',
    auxiliary_groups=972,auxiliary_batch_size=16,auxiliary_presentations=73440,auxiliary_coefficient=1.,
    auxiliary_seed_offset=20261105,auxiliary_order='persistent_random.Random(seed+20261105)_sorted_group_permutation_each_cycle_carried_batches',
    auxiliary_target='no_grad_core_messages_FP64_mean_over24_occurrences_cast_FP32',
    auxiliary_denominator='detached_FP32_message_squared_mean_over24_times96_plus1e-6',
    auxiliary_reduction='mean16_of_mean96_squared_prediction_error_divided_by_group_denominator',
    auxiliary_query_and_target_detached=True,main_predictor_and_query_fully_differentiable=True,
    gradient_isolation_steps=[1,2,32],
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',
    max_new_tokens=4,exact_requires_eos=True,reject_nonterminal_special_tokens=True,
    selection='max_dev_exact_then_min_raw_first_token_NLL_then_earliest',native_eos=[151645,151643],target_eos=151645,
    profile_steps=32,training_count_support=list(range(17)),development_count_support=list(range(16)),
    test_count_support=list(range(17)),maximum_training_N=16,test_N=[32,64],
    bootstrap_seed=20261106,bootstrap_replicates=10000,
    per_main_seconds_cap=2700,campaign_gpu_seconds_cap=16200,profile_seconds_cap=300,
)
PROFILE_CALLS=dict(training_updates=32,native_prefill_forwards=10,visual_forwards=10,
    same_state_native_shaped_head_replays=10,cached_state_head_projections=10,standalone_head_calls=20,natural_generations=0)
TEST_FILES=('tests/test_conditional_null_mean.py','tests/test_paired_sequence_objectives.py','scripts/native_vision_v10_pairs.py')
OWN=tuple(sorted(set(previous.OWN)|{
    'scripts/train_native_vision_v13.py','gnnformer/conditional_null_mean.py','gnnformer/parallel_local_learned_null.py',
    'scripts/native_vision_v13_runtime.py','scripts/profile_native_vision_v13_null.py',*TEST_FILES,
    'slurm/native_vision_v13_train_check.sbatch','slurm/native_vision_v13_train_profile.sbatch','slurm/native_vision_v13_train.sbatch'}))
save=v7.save
bind_file=previous.bind_file
bind_manifest=previous.bind_manifest
batch_states=previous.batch_states
lr=previous.lr


def sources():
    from scripts import cache_native_vision_v13_null_features as worker
    from scripts import merge_native_vision_v13_precision as precision
    from scripts import profile_native_vision_v13_null as software
    return {name:sha(REPO/name) for name in sorted(set(OWN)|set(worker.OWN)|set(precision.sources())|set(software.OWN))}


def snapshot(out):
    (out/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'code'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed while snapshotting')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V13 learned null training\n\n[Plan](plan.json) · [Configuration](config.json) · [Summary](summary.json) · [Sources](source_hashes.json)\n')
    return frozen


def state_info(core,predictor):return dict(branch=v7.state_info(core),predictor=v7.state_info(predictor))


def presentation_order(pairs,seed):
    need(seed in POLICY['seeds'] and len(pairs)==918,'Unregistered V13 pair order')
    rng=random.Random(seed);rows=[]
    for epoch in range(1,41):
        permutation=list(range(918));rng.shuffle(permutation)
        for slot in permutation:
            p=pairs[slot]
            for side,sid in enumerate(p['sids']):
                rows.append(dict(epoch=epoch,slot=slot,pair_id=p['pair_id'],question=p['question'],gold=p['gold'],sid=sid,
                    pair_side=side,pair_kind=p['pair_kind'],n_frames=p['n_frames'][side],replica=p['replicas'][side]))
    need(len(rows)==73440,'Incomplete ordinary training order')
    return rows


def auxiliary_order(groups,seed):
    need(seed in POLICY['seeds'] and len(groups)==972,'Unregistered auxiliary inventory')
    canonical=sorted(groups);rng=random.Random(seed+POLICY['auxiliary_seed_offset']);rows=[];cycle=0
    while len(rows)<73440:
        cycle+=1;permutation=list(range(972));rng.shuffle(permutation)
        for slot in permutation:
            if len(rows)==73440:break
            rows.append(dict(cycle=cycle,slot=slot,group_id=canonical[slot]))
    need(Counter(r['cycle'] for r in rows)==Counter({**{c:972 for c in range(1,76)},76:540}), 'Auxiliary cycle weights differ')
    return rows


def schedule_self_test():
    groups={str(i):{} for i in range(972)};a=auxiliary_order(groups,16);b=auxiliary_order(groups,17)
    need(a==auxiliary_order(groups,16) and a!=b,'Auxiliary order is not deterministic and seed-specific')
    need([r['cycle'] for r in a[960:976]]==[1]*12+[2]*4,'Auxiliary stream dropped the carried cycle boundary')
    rng=random.Random(16+20261105)
    for cycle in (1,2):
        indices=list(range(972));rng.shuffle(indices)
        need([r['slot'] for r in a[(cycle-1)*972:cycle*972]]==indices,'Auxiliary RNG reset between cycles')
    pairs=[dict(pair_id=str(i),question='q',gold=i%17,sids=[f'a{i}',f'b{i}'],pair_kind='test',n_frames=[8,16],replicas=[0,0]) for i in range(918)]
    order=presentation_order(pairs,16);need(len(order)==73440 and Counter(r['epoch'] for r in order[1824:1840])=={1:12,2:4},'Ordinary batch carry differs')
    need(all(order[i]['pair_id']==order[i+1]['pair_id'] and order[i]['pair_side']==0 and order[i+1]['pair_side']==1 for i in range(0,len(order),2)), 'Broken adjacent pairs')
    return dict(passed=True,tests=['deterministic_independent_seeded_streams','persistent_cycle_RNG','no_dropped_cycle_tail','intact_pair_epoch_carry'])


def test_helpers():
    suites=[]
    for name in TEST_FILES:
        r=subprocess.run([sys.executable,str(REPO/name)],capture_output=True,text=True)
        suites.append(dict(file=name,sha256=sha(REPO/name),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        need(r.returncode==0,'CPU helper failed: '+name+'\n'+r.stderr)
    return dict(passed=True,suites=suites,schedules=schedule_self_test(),training_graph=training_graph_self_test())


def cache_binding(path):
    from scripts.merge_native_vision_v13_precision import verify_cache
    cache=verify_cache(path,verify_tensors=False)
    need(cache['protocol']=='v13_training_null_union_cache' and cache['complete'] and cache['training_only']
        and len(cache['features'])==53322 and len(cache['scenes'])==1782 and len(cache['auxiliary_groups'])==972,'V13 union inventory differs')
    parent=read(cache['parent_cache']['file'])
    need(sha(cache['parent_cache']['file'])==cache['parent_cache']['sha256'] and cache['scenes']==parent['scenes']
        and all(cache['features'][fid]==record for fid,record in parent['features'].items()),'Original V10 tensors/scenes were replaced')
    need(cache['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),'Native feature dtypes differ')
    return cache


def pairing_binding(cache):
    from scripts.native_vision_v10_pairs import load_pairs
    pairing=load_pairs(TRAIN,Path(cache['parent_cache']['file']),SCHEDULE)
    return dict(pairing,protocol='v13_unchanged_v10_pairing_new_seeds',seeds=POLICY['seeds'],parent_cache=cache['parent_cache'])


def bind_software(path,bindings):
    from scripts.profile_native_vision_v13_null import verify_profile
    path=Path(path);path=path/'summary.json' if path.is_dir() else path
    summary=verify_profile(path);binding=bind_file(bindings,path)
    need(summary['passed'] and summary['completed'],'Completed V13 native software required')
    bind_file(bindings,summary['plan_file'],summary['plan_sha256'])
    for name,digest in summary['source_sha256'].items():bind_file(bindings,REPO/name,digest)
    return dict(**binding,plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],source_sha256=summary['source_sha256'])


def bind_prior(path,bindings):
    need(Path(path).resolve()==PRIOR,'Require the complete canonical V12 diagnostic')
    ref=bind_file(bindings,path);summary=read(path)
    need(summary['completed'] and summary['audit_passed'] and summary['no_practical_milestone_claim'] and summary['records']==408,'Prior diagnostic incomplete')
    bind_file(bindings,summary['analysis_file'],summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    bind_file(bindings,analysis['plan_file'],analysis['plan_sha256'])
    need(analysis['no_fit'] and analysis['exploratory_reused_test'],'Prior diagnostic scope changed')
    for name,digest in summary['source_sha256'].items():bind_file(bindings,REPO/name,digest)
    return dict(summary=ref,analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'],no_prior_milestone_claim=True)


def check(args):
    out=OUT/f'train_check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4);begin=time.perf_counter();tests=test_helpers();bindings={}
    manifest=bind_manifest(bindings,TRAIN)
    for p in (SCHEDULE,PAIRING):bind_file(bindings,p)
    need(Counter(r['gold'] for r in manifest['splits']['dev_N16']['samples'])==Counter({k:4 for k in range(16)}),'Dev count support differs')
    cache_path=args.parallel_cache.resolve();cache_ref=bind_file(bindings,cache_path);cache=cache_binding(cache_path)
    for key in ('parent_cache','parent_plan','inventory_plan'):bind_file(bindings,cache[key]['file'],cache[key]['sha256'])
    bind_file(bindings,cache['plan_file'],cache['plan_sha256'])
    bind_file(bindings,Path(cache['profile_directory'])/'summary.json',cache['profile_summary_sha256'])
    pairing=pairing_binding(cache);processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    for scene in cache['scenes'].values():need(scene['target_ids']==native.encode_target(processor.tokenizer,scene['gold']),'Full native training target differs')
    states,index=v7.load_features(torch,cache,'cpu')
    for pair in pairing['pairs']:
        a,b=[cache['scenes'][sid] for sid in pair['sids']]
        need(torch.equal(states[[index[f] for f in a['global_feature_ids']]],states[[index[f] for f in b['global_feature_ids']]]),'Paired frozen queries differ')
    for filename,digest in {r['file']:r['file_sha256'] for r in cache['features'].values()}.items():bind_file(bindings,filename,digest)
    for gid,group in cache['auxiliary_groups'].items():
        need(group['auxiliary_only'] and not group['zero_answer_ce_for_assigned_prefix'] and len(group['local_feature_ids'])==24
            and group['reference_count']==24 and group['global_feature_id'] in cache['features']
            and all(fid in cache['features'] for fid in group['local_feature_ids']),'Auxiliary feature ownership differs')
    orders={str(seed):object_sha(presentation_order(pairing['pairs'],seed)) for seed in POLICY['seeds']}
    auxiliary_orders={str(seed):object_sha(auxiliary_order(cache['auxiliary_groups'],seed)) for seed in POLICY['seeds']}
    software=bind_software(args.native_profile,bindings);prior=bind_prior(args.diagnostic_summary,bindings)
    save(out/'pairing.json',pairing)
    plan=dict(schema_version=1,policy=POLICY,profile_calls=PROFILE_CALLS,source_sha256=frozen,artifact_bindings=bindings,
        cache_binding=cache_ref,pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=object_sha(pairing),
        order_sha256=orders,auxiliary_order_sha256=auxiliary_orders,auxiliary_groups_sha256=object_sha(cache['auxiliary_groups']),
        train_manifest=str(TRAIN),schedule_file=str(SCHEDULE),original_pairing_file=str(PAIRING),fresh_manifests={},
        fresh_test_bound_by_future_independent_release=True,native_profile=software,prior_result=prior,tests=tests,
        actual_frozen_pair_equality_passed=True,no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify_plan(out/'plan.json')
    save(out/'summary.json',dict(passed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),source_sha256=frozen,
        tests=tests,no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify_plan(path):
    path=Path(path).resolve();need(path.with_suffix('.sha256').read_text().strip()==sha(path),'V13 plan sidecar differs');plan=read(path)
    need(plan['schema_version']==1 and plan['policy']==POLICY and plan['source_sha256']==sources() and plan['profile_calls']==PROFILE_CALLS,'Frozen V13 source/policy differs')
    for filename,digest in plan['artifact_bindings'].items():need(sha(filename)==digest,'Bound input changed: '+filename)
    need(sha(plan['pairing_file'])==plan['pairing_sha256'],'Pairing file changed')
    cache=cache_binding(plan['cache_binding']['file']);pairing=pairing_binding(cache)
    need(object_sha(pairing)==plan['pairing_object_sha256']==object_sha(read(plan['pairing_file'])),'Ordinary pairing changed')
    need(object_sha(cache['auxiliary_groups'])==plan['auxiliary_groups_sha256'],'Auxiliary grouping changed')
    for seed in POLICY['seeds']:
        need(object_sha(presentation_order(pairing['pairs'],seed))==plan['order_sha256'][str(seed)]
            and object_sha(auxiliary_order(cache['auxiliary_groups'],seed))==plan['auxiliary_order_sha256'][str(seed)],'Paired or auxiliary order changed')
    return plan


def learned_batch(torch,core,predictor,local,g,counts,condition):
    """Apply the same deployed projected arithmetic, grouped by actual item N.

    Raw-zero padding belongs only to batching and is excluded from the centering
    coefficient. Grouping also avoids changing the core's unmodified interface.
    """
    from gnnformer.conditional_null_mean import projected_null_readout
    need(condition in POLICY['conditions'] and len(counts)==g.shape[0] and all(type(n) is int and 0<n<=local.shape[0] for n in counts), 'Actual per-position cardinalities differ')
    delta=torch.zeros_like(g,dtype=torch.float32);corrected=torch.zeros((g.shape[0],core.rank),device=g.device,dtype=torch.float32)
    nulls=torch.zeros_like(corrected)
    for n in sorted(set(counts)):
        ids=torch.tensor([i for i,v in enumerate(counts) if v==n],device=g.device)
        h=local[:n,ids];global_=g[ids]
        need(bool((local[n:,ids]==0).all()),'Nonzero local evidence was dropped as padding')
        z=core.aggregate(core.encode(h,global_));q=core.query(core.rms(global_));c=predictor(q)
        result=projected_null_readout(core,z,global_,c,n_elements=n,mode=condition,output_dtype=torch.float32)
        delta=delta.index_copy(0,ids,result['delta_float32']);corrected=corrected.index_copy(0,ids,result['corrected_aggregate'])
        nulls=nulls.index_copy(0,ids,c)
    return delta,corrected,nulls


def auxiliary_loss(torch,core,predictor,local,g):
    """Only predictor parameters enter this regression graph; no false answer CE."""
    need(local.shape[0]==24 and local.shape[1:]==g.shape and not local.requires_grad and not g.requires_grad,'Null occurrence/global feature shapes differ')
    with torch.no_grad():
        q=core.query(core.rms(g));messages=core.encode(local,g)
        target=messages.double().mean(0).float()
        denominator=messages.float().square().mean(dim=(0,2))+1e-6
    predicted=predictor(q.detach());mse=(predicted-target.detach()).square().mean(-1)
    normalized=mse/denominator.detach();loss=normalized.mean()
    need(bool(torch.isfinite(loss)) and not q.requires_grad and not target.requires_grad and not denominator.requires_grad,'Auxiliary detached target/query contract failed')
    with torch.no_grad():
        # Diagnostic geometry only: omit the affine bias, preserve the live
        # main/auxiliary losses above, and never feed these values to training.
        weight=core.aggregate_projection.weight
        projected_error=torch.nn.functional.linear(predicted.detach()-target,weight)
        projected_prediction=torch.nn.functional.linear(predicted.detach(),weight)
        projected_target=torch.nn.functional.linear(target,weight)
        projected=dict(per_group_projected_mse=projected_error.square().mean(-1).tolist(),
            per_group_projected_error_norms=projected_error.norm(dim=-1).tolist(),
            projected_prediction_norms=projected_prediction.norm(dim=-1).tolist(),
            projected_target_norms=projected_target.norm(dim=-1).tolist(),query_norms=q.norm(dim=-1).tolist(),
            aggregate_projection_weight_norm=float(weight.norm()),readout_weight_norm=float(core.up.weight.norm()),
            projected_diagnostics_detached=True,projected_diagnostics_include_bias=False)
    return loss,dict(per_group_mse=mse.detach().tolist(),per_group_denominator=denominator.detach().tolist(),
        per_group_loss=normalized.detach().tolist(),target_norms=target.norm(dim=-1).tolist(),
        prediction_norms=predicted.detach().norm(dim=-1).tolist(),query_detached=True,target_detached=True,denominator_detached=True,**projected)


def auxiliary_states(torch,cache,states,index,group_ids):
    groups=[cache['auxiliary_groups'][gid] for gid in group_ids]
    h=torch.stack([states[torch.tensor([index[f] for f in row['local_feature_ids']],device=states.device)] for row in groups],dim=1)
    g=states[torch.tensor([index[row['global_feature_id']] for row in groups],device=states.device)]
    need(h.shape==(24,len(group_ids),3584) and g.shape==(len(group_ids),3584),'Null calibration batch shape differs')
    return h,g


def losses(torch,core,predictor,local,g,layout,norm,head,condition,sids,counts):
    from gnnformer.paired_sequence_objectives import sequence_objectives
    delta,z,nulls=learned_batch(torch,core,predictor,local,g,counts,condition)
    logits=head(norm((g+delta.to(g.dtype)).unsqueeze(0)))[0]
    result=sequence_objectives(logits,delta,z,core.aggregate_projection.weight,core.up.weight,g,layout,eps=1e-6)
    ce,consistency,path=result['ce'],result['residual'],result['path'];main=ce+consistency
    with torch.no_grad():
        offsets=layout['offsets'];lengths=[b-a for a,b in zip(offsets,offsets[1:])];pair_lengths=lengths[::2]
        per_pair={}
        for label,values in [('consistency',result['residual_positions']),('path',result['path_positions'])]:
            cursor=0;per_pair[label]=[]
            for length in pair_lengths:per_pair[label].append(float(values[cursor:cursor+length].mean()));cursor+=length
        identity=[sids[i]==sids[i+1] for i in range(0,len(sids),2)];cursor=0
        for same,length in zip(identity,pair_lengths):
            if same:need(bool((result['residual_positions'][cursor:cursor+length]==0).all()),'Saturated identity consistency must be exactly zero')
            cursor+=length
        left=torch.tensor(layout['left'],device=g.device);right=torch.tensor(layout['right'],device=g.device)
        components=dict(scene_lengths=lengths,scene_offsets=offsets,pair_lengths=pair_lengths,
            prefix_ids=[prefix for row in layout['prefixes'] for prefix in row],position_n_frames=counts,
            per_scene_ce=[float(result['ce_positions'][a:b].mean()) for a,b in zip(offsets,offsets[1:])],
            per_pair_consistency=per_pair['consistency'],per_pair_path=per_pair['path'],
            position_losses=dict(ce=result['ce_positions'].tolist(),consistency=result['residual_positions'].tolist(),path=result['path_positions'].tolist()),
            pair_position_denominator=result['denominator'].tolist(),residual_difference_norms=(delta[right]-delta[left]).norm(dim=-1).tolist(),
            B_norms=result['bound'].tolist(),saturated_identity_pairs=identity,null_prediction_norms=nulls.norm(dim=-1).tolist())
    need(all(bool(torch.isfinite(v)) for v in (main,ce,consistency,path)),'Nonfinite main objective')
    return main,ce,consistency,path,components


def gradient_isolation(torch,core,predictor,main,auxiliary,step):
    core_params=tuple(core.parameters());pred_params=tuple(predictor.parameters());params=core_params+pred_params
    main_grad=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True)
    aux_grad=torch.autograd.grad(auxiliary,params,retain_graph=True,allow_unused=True)
    need(all(g is None for g in aux_grad[:len(core_params)]),'Auxiliary regression leaked into the core graph')
    need(all(p.grad is None for p in params),'Gradient inspection mutated optimizer buffers')
    def stats(values):
        present=[g for g in values if g is not None]
        need(all(bool(torch.isfinite(g).all()) for g in present),'Nonfinite isolated gradients')
        return dict(norm=float(torch.stack([g.float().square().sum() for g in present]).sum().sqrt()) if present else 0.,
            connected_tensors=len(present),nonzero_tensors=sum(bool((g!=0).any()) for g in present))
    mcore,mpred=stats(main_grad[:len(core_params)]),stats(main_grad[len(core_params):]);apred=stats(aux_grad[len(core_params):])
    need(mpred['connected_tensors']==len(pred_params) and apred['connected_tensors']==len(pred_params),'Predictor missing a main/auxiliary gradient path')
    return dict(step=step,passed=True,auxiliary_core_gradients_all_none=True,optimizer_buffers_untouched=True,
        main_core=mcore,main_predictor=mpred,auxiliary_predictor=apred,
        zero_initial_main_predictor_gradient=(mpred['norm']==0.) if step==1 else None)


def training_graph_self_test():
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261105);core=ParallelLocalAggregation(hidden_size=8,rank=4);pred=ConditionalNullMean(rank=4)
        with torch.no_grad():core.up.weight.normal_(std=.1);pred.fc2.weight.normal_(std=.1)
        g=torch.randn(2,8);h=torch.randn(24,2,8);aux,detail=auxiliary_loss(torch,core,pred,h,g)
        delta,_,_=learned_batch(torch,core,pred,h,g,[24,24],'centered');main=delta.square().sum()
        checks=gradient_isolation(torch,core,pred,main,aux,2)
        need(checks['main_predictor']['norm']>0 and checks['auxiliary_predictor']['norm']>0,'Synthetic live predictor routes not exercised')
        direct=torch.autograd.grad(main,core.query.weight,retain_graph=True)[0]
        need(bool((direct!=0).any()) and detail['target_detached'] and detail['query_detached'],'Main query path is detached or auxiliary target is live')
        # False padding N must not change the operator: actual12 with zeros has
        # the same output as a truly12-item tensor, even though packed width24.
        short=h[:12];padded=torch.cat([short,torch.zeros_like(short)])
        a=learned_batch(torch,core,pred,short,g,[12,12],'centered')[0]
        b=learned_batch(torch,core,pred,padded,g,[12,12],'centered')[0]
        need(torch.equal(a,b),'Centering accidentally used padded rather than actual N')
    return dict(passed=True,tests=['auxiliary_core_graph_disconnected','main_query_and_predictor_live','no_grad_target_statistics','actual_N_not_padding_width'])


def strict_prediction(torch,processor,sample,result,index,cell):
    row=v7.prediction_record(torch,processor,sample,result,index,cell);ids=result['generated_ids']
    complete=ids[-1] in POLICY['native_eos'];need(not any(i in POLICY['native_eos'] for i in ids[:-1]) and (complete or len(ids)==4),'Unexpected native stop')
    bodyids=ids[:-1] if complete else ids;clean=not any(i in set(processor.tokenizer.all_special_ids) for i in bodyids)
    body=processor.tokenizer.decode(bodyids,skip_special_tokens=False);value=int(body.strip()) if clean and re.fullmatch('[0-9]+',body.strip()) else None
    row.update(answer_body=body,no_nonterminal_special_tokens=clean,prediction=value,parseable=value is not None,
        parsed_count_correct=value==sample['gold'],exact=complete and value==sample['gold'],pair_id=sample.get('pair_id'))
    return row


def evaluate(torch,model,processor,core,predictor,rows,out,label,dataout,condition):
    from scripts import native_vision_v13_runtime as deployment
    records=[];raw=[];begin=time.perf_counter()
    for cell,sample in rows:
        bundle=native.prepare_scene(processor,sample,'parallel')
        result=deployment.generate_native(model,processor,core,predictor,bundle,mode=condition)
        value=result['raw_logits'].detach().cpu().float()
        need(torch.equal(value,value.half().float()) and bool(torch.isfinite(value).all()),'Raw archive is not exact native FP16 promotion')
        records.append(strict_prediction(torch,processor,sample,result,len(raw),cell));raw.append(value)
        if len(records)%36==0:print(json.dumps(dict(evaluation=label,processed=len(records),total=len(rows))),flush=True)
    rawfile=dataout/(label+'_raw.pt');torch.save(dict(schema_version=1,raw_logits=raw),rawfile)
    result=dict(label=label,rows=records,n=len(records),exact_count=sum(r['exact'] for r in records),
        first_token_nll=sum(r['first_token_nll'] for r in records)/len(records),seconds=time.perf_counter()-begin,
        raw_file=str(rawfile),raw_sha256=sha(rawfile),raw_dtype='torch.float32')
    save(out/(label+'.json'),result);return result


def active_cache_replay(torch,model,processor,core,predictor,cache,states,index,manifest,dataout,condition):
    from scripts import native_vision_v13_runtime as deployment
    from scripts.cache_native_vision_v7_features import replay_metrics
    observations=[];raw=[]
    for k in (0,9,10,16):
        sample=sorted((r for r in manifest['splits']['train_N16']['samples'] if r['gold']==k),key=lambda r:r['sid'])[0]
        h,g,layout=batch_states(torch,cache,states,index,[sample['sid'],sample['sid']]);targets=cache['scenes'][sample['sid']]['target_ids']
        for t in range(len(targets)):
            bundle=native.prepare_scene(processor,sample,'parallel',prefix_ids=targets[:t])
            actual=deployment.forward_native(model,bundle,core,predictor,mode=condition,capture=True,cpu=False)
            with torch.inference_mode():
                cached_delta=learned_batch(torch,core,predictor,h[:,t:t+1],g[t:t+1],[16],condition)[0]
                cached=model.lm_head(model.model.language_model.norm((g[t:t+1]+cached_delta.to(g.dtype)).unsqueeze(0)))[0]
                ag,al=actual['global_states'],actual['local_states']
                replay_delta=learned_batch(torch,core,predictor,al,ag,[16],condition)[0]
                full=actual['native_query_hidden'].clone()
                need(full.shape==(17,1,3584),'Same-captured native head must preserve every local/global row')
                full[-1:]=full[-1:]+replay_delta.to(full.dtype).unsqueeze(0)
                replay=model.lm_head(model.model.language_model.norm(full))[-1]
            logits=actual['global_logits'].unsqueeze(0);metrics=replay_metrics(torch,logits,replay);drift=replay_metrics(torch,logits,cached)
            observations.append(dict(sid=sample['sid'],gold=k,n_frames=16,prefix_ids=targets[:t],position=t,
                native_head_replay=metrics,cached_vs_native_descriptive=drift,cached_vs_native_gate=False,
                counters=actual['counters'],native_metadata=actual['metadata']))
            raw.append(dict(native=logits.detach().cpu(),replayed=replay.detach().cpu(),cached=cached.detach().cpu()))
    need(len(observations)==10 and sum(r['counters']['model'] for r in observations)==10
        and sum(r['counters']['visual'] for r in observations)==10,'Registered profile forward inventory differs')
    path=dataout/'active_cache_replay.pt';torch.save(raw,path)
    return dict(passed=all(r['native_head_replay']['passed'] for r in observations),observations=observations,
        counts_covered=[0,9,10,16],prefixes_checked=10,raw_file=str(path),raw_sha256=sha(path),
        cached_numerical_failures=sum(not r['cached_vs_native_descriptive']['passed'] for r in observations),
        native_model_calls=10,vision_calls=10,standalone_head_calls=20)


def verify_release(path,plan_path,plan):
    need(path is not None,'V13 mains require an explicit measured release')
    release=read(path)
    need(release['protocol']=='v13_learned_null_main_release' and release['passed'] and release['plan_sha256']==sha(plan_path)
        and release['source_sha256']==sources() and release['per_main_seconds_cap']==2700
        and release['main_block_gpu_seconds_cap']==16200 and release['campaign_budget']['passed'],'Main source/resource release differs')
    data=release['data_release'];need(sha(data['file'])==data['sha256'],'Independent data release changed');d=read(data['file'])
    need(d['passed'] and d['source_sha256']==release['report_source_sha256'],'Independent data/report release failed')
    for name,digest in release['report_source_sha256'].items():need(sha(REPO/name)==digest,'Report source changed')
    for name,digest in d['data_bindings'].items():need(sha(name)==digest,'Independent data binding changed')
    fresh=release['fresh_manifest'];need(Path(fresh['file']).resolve()==FRESH/'main_manifest.json' and sha(fresh['file'])==fresh['sha256']
        and d['data_bindings'][str(Path(fresh['file']).resolve())]==fresh['sha256'],'Fresh272 manifest not bound by independent release')
    manifest=read(fresh['file']);need({k:len(v['samples']) for k,v in manifest['splits'].items()}=={'test_N32':136,'test_N64':136},'Fresh test count differs')
    need(set(release['profiles'])==set(POLICY['conditions']) and set(release['projections'])==set(POLICY['conditions']),'Both matched profiles/projections required')
    for condition,item in release['profiles'].items():
        p=Path(item['directory'])/'summary.json';need(sha(p)==item['summary_sha256'],'Training profile changed');s=read(p)
        need(s['profile'] and s['condition']==condition and s['seed']==16 and s['passed'] and s['completed']
            and s['steps']==32 and s['computational_integrity_passed'] and s['no_dev_or_test_evaluation']
            and s['plan_sha256']==sha(plan_path) and s['source_sha256']==sources()
            and s['active_cache_replay']['passed'] and s['profile_calls']==PROFILE_CALLS,'Matched training profile failed')
        need(s['active_cache_replay']['counts_covered']==[0,9,10,16] and s['active_cache_replay']['prefixes_checked']==10
            and s['checkpoint_roundtrip_passed'] and s['gradient_isolation_passed'],'Training computation gate failed')
        projection=release['projections'][condition]
        need(projection['passed'] and 0<projection['projected_seconds']<=2700,'Measured main time exceeds cap')
    return dict(file=str(Path(path).resolve()),sha256=sha(path),fresh_manifest=fresh)


def zero_main_gradients(torch,core,predictor,consistency,path):
    need(bool((core.up.weight==0).all()) and bool((predictor.fc2.weight==0).all()) and bool((predictor.fc2.bias==0).all()),'Zero-U/C2 initialization changed')
    need(float(consistency)==float(path)==0.,'Initial paired regularizers are nonzero')
    params=tuple(core.parameters())+tuple(predictor.parameters());flags={}
    for label,value in [('residual',consistency),('path',path)]:
        grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
        flags['all_'+label+'_gradients_zero']=all(g is None or bool(torch.isfinite(g).all() and (g==0).all()) for g in grads)
    need(all(flags.values()) and all(p.grad is None for p in params),'Initial regularizer gradients differ')
    return dict(passed=True,up_exactly_zero=True,predictor_second_layer_exactly_zero=True,residual_loss=0.,path_loss=0.,**flags)


def run(args):
    job=os.environ['SLURM_JOB_ID'];runid=f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out=OUT/runid;out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch,transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    from scripts.stage_native_vision_v10_features import runtime_identity,model_metadata,native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.perf_counter();args.plan=args.plan.resolve();plan=verify_plan(args.plan)
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds'] and (not args.profile or args.seed==16),'Unregistered run condition/seed')
    release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    cache=cache_binding(plan['cache_binding']['file']);dataout=DATA/'native_aggregation_vlm_v13'/runid;dataout.mkdir(parents=True,exist_ok=False)
    ckpt=CKPT/runid;ckpt.mkdir(parents=True,exist_ok=False);loadstart=time.perf_counter()
    runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=runtime.model.eval().requires_grad_(False)
    native.native_contract(model)
    need(runtime_identity()==cache['runtime'] and model_metadata()==cache['model']
        and fingerprint(runtime.processor,str(transformers.__version__))==cache['processor'],'Model/runtime/processor differs from cache')
    need(native_api(runtime.processor)[2]==cache['native_api'] and native.generation_policy(model,runtime.processor.tokenizer)[1]['native_eos_token_ids']==POLICY['native_eos'],'Native API or EOS policy differs')
    states,feature_index=v7.load_features(torch,cache,runtime.device);loadseconds=time.perf_counter()-loadstart
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    core=ParallelLocalAggregation().to(runtime.device);predictor=ConditionalNullMean().to(runtime.device)
    native.native_contract(model,core)
    need(sum(p.numel() for p in core.parameters())==1041600 and sum(p.numel() for p in predictor.parameters())==18624,'Learned parameter counts differ')
    initialized=state_info(core,predictor)
    initial=dict(branch={k:v.detach().cpu().clone() for k,v in core.state_dict().items()},predictor={k:v.detach().cpu().clone() for k,v in predictor.state_dict().items()})
    optimizer=torch.optim.AdamW([dict(params=list(core.parameters()),name='branch'),dict(params=list(predictor.parameters()),name='predictor')],lr=.001,weight_decay=0.)
    pairing=read(plan['pairing_file']);order=presentation_order(pairing['pairs'],args.seed);aux_order=auxiliary_order(cache['auxiliary_groups'],args.seed)
    need(object_sha(order)==plan['order_sha256'][str(args.seed)] and object_sha(aux_order)==plan['auxiliary_order_sha256'][str(args.seed)],'Frozen training streams differ')
    save(out/'presentations.json',order);save(out/'auxiliary_presentations.json',aux_order)
    config=dict(run_id=runid,arm='parallel',condition=args.condition,seed=args.seed,profile=args.profile,policy=POLICY,
        consistency_coefficient=1.,auxiliary_coefficient=1.,slurm_job_id=job,plan_file=str(args.plan),plan_sha256=sha(args.plan),source_sha256=frozen,
        cache_binding=plan['cache_binding'],pairing_file=plan['pairing_file'],pairing_sha256=plan['pairing_sha256'],prior_result=plan['prior_result'],
        auxiliary_groups_sha256=plan['auxiliary_groups_sha256'],initialized=initialized,initialized_sha256=object_sha(initialized),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        auxiliary_presentations_sha256=sha(out/'auxiliary_presentations.json'),auxiliary_order_sha256=object_sha(aux_order),
        model=cache['model'],runtime=cache['runtime'],processor=cache['processor'],native_dtypes=cache['native_dtypes'],
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch=str(torch.__version__)),
        model_and_features_load_seconds=loadseconds,checkpoint_directory=str(ckpt),data_directory=str(dataout),main_release=release)
    save(out/'config.json',config)
    manifest=read(plan['train_manifest']);devrows=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples']]
    trainlog=[];developments=[];best=None;first_gradients=[];isolation=[];zero=None
    totalsteps=32 if args.profile else 4590
    norm,head=model.model.language_model.norm,model.lm_head
    groups={'branch':core,'predictor':predictor}
    nonzero={group:{name:False for name,_ in module.named_parameters()} for group,module in groups.items()}
    model_versions={name:p._version for name,p in model.named_parameters()}
    for step in range(1,totalsteps+1):
        begin=time.perf_counter();rows=order[(step-1)*16:step*16];auxrows=aux_order[(step-1)*16:step*16]
        need(len(rows)==len(auxrows)==16 and all(rows[i]['pair_id']==rows[i+1]['pair_id'] and rows[i]['pair_side']==0 and rows[i+1]['pair_side']==1 for i in range(0,16,2)), 'Intact paired batch required')
        sids=[r['sid'] for r in rows];auxids=[r['group_id'] for r in auxrows]
        local,g,layout=batch_states(torch,cache,states,feature_index,sids)
        counts=[row['n_frames'] for row,tokens in zip(rows,layout['target_sequences']) for _ in tokens]
        ah,ag=auxiliary_states(torch,cache,states,feature_index,auxids)
        optimizer.zero_grad(set_to_none=True);rate=lr(step)
        for group in optimizer.param_groups:group['lr']=rate
        main,ce,consistency,path_loss,components=losses(torch,core,predictor,local,g,layout,norm,head,args.condition,sids,counts)
        aux,auxdetails=auxiliary_loss(torch,core,predictor,ah,ag);total=main+aux
        if step==1:zero=zero_main_gradients(torch,core,predictor,consistency,path_loss)
        if step in POLICY['gradient_isolation_steps']:
            item=gradient_isolation(torch,core,predictor,main,aux,step)
            if step==1:need(item['zero_initial_main_predictor_gradient'],'Zero-U main predictor gradient must vanish')
            isolation.append(item)
        total.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for module in groups.values() for p in module.parameters()),'Missing/nonfinite learned gradient')
        if args.profile or step<=32:
            for group,module in groups.items():
                for name,p in module.named_parameters():nonzero[group][name]|=bool((p.grad!=0).any())
        if step<=2:first_gradients.append(dict(step=step,pre_clip={group:{name:v7.tensor_info(p.grad) for name,p in module.named_parameters()} for group,module in groups.items()}))
        gradient_norms={group:float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.)) for group,module in groups.items()}
        need(all(math.isfinite(v) for v in gradient_norms.values()),'Nonfinite parameter-group gradient norm');optimizer.step()
        need(all(bool(torch.isfinite(p).all()) for module in groups.values() for p in module.parameters()),'Nonfinite updated parameters')
        need(not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone gradient state changed')
        torch.cuda.synchronize()
        trainlog.append(dict(step=step,lr=rate,loss=float(total),main_loss=float(main),ce_loss=float(ce),consistency_loss=float(consistency),path_loss=float(path_loss),
            auxiliary_loss=float(aux),consistency_coefficient=1.,auxiliary_coefficient=1.,weighted_consistency_loss=float(consistency),weighted_auxiliary_loss=float(aux),
            gradient_norms=gradient_norms,clipped_groups={k:v>1. for k,v in gradient_norms.items()},seconds=time.perf_counter()-begin,
            target_ids=layout['targets'],sids=sids,pair_ids=[r['pair_id'] for r in rows[::2]],epochs=[r['epoch'] for r in rows[::2]],
            auxiliary_group_ids=auxids,auxiliary_cycles=[r['cycle'] for r in auxrows],auxiliary=auxdetails,**components))
        if step%100==0:print(json.dumps(dict(step=step,ce=float(ce),consistency=float(consistency),auxiliary=float(aux),total=float(total))),flush=True)
        if not args.profile and step in POLICY['dev_steps']:
            checkpoint=ckpt/f'step_{step}.pt';torch.save(dict(branch=core.state_dict(),predictor=predictor.state_dict(),step=step,config=config),checkpoint)
            dev=evaluate(torch,model,runtime.processor,core,predictor,devrows,out,f'dev_{step}',dataout,args.condition)
            entry=dict(step=step,exact_count=dev['exact_count'],nll=dev['first_token_nll'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                parameter_sha256=object_sha(state_info(core,predictor)),dev_file=str(out/f'dev_{step}.json'),dev_sha256=sha(out/f'dev_{step}.json'))
            developments.append(entry)
            if best is None or (-entry['exact_count'],entry['nll'],step)<(-best['exact_count'],best['nll'],best['step']):best=entry
            save(out/'training.json',trainlog);save(out/'selection.json',dict(development=developments,selected=best))
    save(out/'training.json',trainlog);save(out/'first_gradients.json',first_gradients);save(out/'gradient_isolation.json',isolation)
    need(len(isolation)==3 and all(r['passed'] for r in isolation) and isolation[-1]['main_predictor']['norm']>0,'Main predictor path not exercised after learning')
    need(model_versions=={name:p._version for name,p in model.named_parameters()},'Frozen backbone tensor version changed')
    if args.profile:
        need(all(v for group in nonzero.values() for v in group.values()),'Profile did not exercise every learned tensor')
        updated=state_info(core,predictor);need(updated!=initialized,'Profile did not update parameters')
        checkpoint=ckpt/'profile.pt';torch.save(dict(branch=core.state_dict(),predictor=predictor.state_dict(),step=32,config=config),checkpoint)
        core.load_state_dict(initial['branch']);predictor.load_state_dict(initial['predictor']);need(state_info(core,predictor)==initialized,'Initial-state restore failed')
        saved=torch.load(checkpoint,map_location=runtime.device,weights_only=True);core.load_state_dict(saved['branch']);predictor.load_state_dict(saved['predictor'])
        need(state_info(core,predictor)==updated,'Updated-state checkpoint restore failed')
        replay=active_cache_replay(torch,model,runtime.processor,core,predictor,cache,states,feature_index,manifest,dataout,args.condition)
        save(out/'active_cache_replay.json',replay);need(replay['passed'],'Native captured-state head replay failed')
        result=dict(config,passed=True,completed=True,computational_integrity_passed=True,steps=32,profile_calls=PROFILE_CALLS,
            active_cache_replay=replay,no_dev_or_test_evaluation=True,zero_initialization_auxiliary_gradients=zero,
            all_parameter_gradients_exercised=nonzero,frozen_backbone_gradient_state_preserved=True,checkpoint_roundtrip_passed=True,
            profile_checkpoint=str(checkpoint),profile_checkpoint_sha256=sha(checkpoint),step_seconds_max_steady=max(r['seconds'] for r in trainlog[4:]))
    else:
        need(best is not None and len(developments)==5,'Missing development selection')
        saved=torch.load(best['checkpoint'],map_location=runtime.device,weights_only=True);core.load_state_dict(saved['branch']);predictor.load_state_dict(saved['predictor'])
        need(object_sha(state_info(core,predictor))==best['parameter_sha256'],'Selected combined checkpoint restoration differs')
        fresh=read(release['fresh_manifest']['file']);testrows=[('main_'+cell,r) for cell,split in fresh['splits'].items() for r in split['samples']]
        need(len(testrows)==272,'Fresh native test inventory differs');evaluate(torch,model,runtime.processor,core,predictor,testrows,out,'test',dataout,args.condition)
        result=dict(config,passed=True,completed=True,computational_integrity_passed=True,steps=4590,selected=best,development=developments,
            native_test_count=272,test_file=str(out/'test.json'),test_sha256=sha(out/'test.json'))
    result.update(training_seconds=sum(r['seconds'] for r in trainlog),training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),
        first_gradients_file=str(out/'first_gradients.json'),first_gradients_sha256=sha(out/'first_gradients.json'),
        gradient_isolation_passed=True,gradient_isolation_file=str(out/'gradient_isolation.json'),gradient_isolation_sha256=sha(out/'gradient_isolation.json'),
        elapsed_seconds=time.perf_counter()-start)
    verify_plan(args.plan)
    if release is not None:
        need(sha(release['file'])==release['sha256'],'Main release changed');verify_release(args.main_release,args.plan,plan)
    save(out/'summary.json',result);print(json.dumps(dict(run_id=runid,completed=True,elapsed_seconds=result['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true');modes.add_argument('--profile',action='store_true');modes.add_argument('--run',action='store_true')
    parser.add_argument('--parallel-cache',type=Path);parser.add_argument('--native-profile',type=Path)
    parser.add_argument('--diagnostic-summary',type=Path,default=PRIOR);parser.add_argument('--plan',type=Path);parser.add_argument('--main-release',type=Path)
    parser.add_argument('--condition',choices=POLICY['conditions']);parser.add_argument('--seed',type=int,choices=POLICY['seeds']);args=parser.parse_args()
    native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU source freeze requires CPU Slurm')
        need(args.parallel_cache is not None and args.native_profile is not None,'--check needs the union cache and completed V13 native software')
        check(args)
    else:
        need(args.plan is not None and args.condition is not None and args.seed is not None,'GPU modes need frozen plan/condition/seed');run(args)


if __name__=='__main__':main()
