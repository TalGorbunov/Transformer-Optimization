"""V14 empirical-bank core training followed by frozen-core mean distillation.

Only the core learns during CE+residual consistency; all reference messages and
queries remain live. The final core then supplies a fixed972-group regression
table. Only the predictor learns in phase2. Final-only native evaluation uses
the immutable V13 predictor runtime, without reference rows or an answer oracle.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import train_native_vision_v13 as ancestor
from scripts import train_native_vision_v7 as v7
from scripts import native_vision_v7_runtime as native
from scripts.stage_native_vision_v7_features import MODEL,need,read,sha,object_sha
DATA=ancestor.DATA
OUT=REPO/'outputs/native_aggregation_vlm/v14'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v14')
TRAIN=ancestor.TRAIN;SCHEDULE=ancestor.SCHEDULE;PAIRING=ancestor.PAIRING
FRESH=DATA/'v14_fresh'
PRIOR=REPO/'outputs/native_aggregation_vlm/v13/finalization/final_442481/final_acceptance.json'
GEOMETRY=REPO/'outputs/native_aggregation_vlm/v13/calibration_geometry/geometry_442483/summary.json'
SOFTWARE=REPO/'outputs/native_aggregation_vlm/v13/null_software/profile_442413/summary.json'
POLICY=dict(
    protocol='v14_exact_bank_then_frozen_core_distillation',native_arm='parallel',conditions=['centered','offset'],seeds=[18,19],
    epochs=40,pair_slots_per_epoch=918,scene_slots_per_epoch=1836,unique_training_scenes=1782,
    scene_presentations=73440,pair_presentations=36720,target_positions=177120,batch_size=16,pairs_per_batch=8,steps=4590,
    lr=.001,warmup=50,final_lr=.00001,weight_decay=0.,clip_norm=1.,core_optimizer='AdamW_core_only',
    rank=96,hidden_size=3584,core_parameters=1041600,predictor_parameters=18624,parameters=1060224,
    merge='sum',post_activation='silu',native_dtype='torch.float16',branch_dtype='torch.float32',predictor_dtype='torch.float32',
    correction_coefficients={'centered':'actual_scene_N','offset':1},
    core_null_mean='live_FP64_mean_over24_occurrences_cast_FP32',core_reference_query_and_messages_fully_differentiable=True,
    core_predictor_present=False,core_auxiliary_loss_present=False,consistency_coefficients={'centered':1.,'offset':1.},
    consistency_epsilon=1e-6,ce_reduction='mean_over_scenes_of_mean_over_valid_native_target_positions',
    consistency_reduction='mean_over_pairs_of_mean_over_corresponding_strict_prefix_positions',
    denominator='detached_squared_L2_of_identical_frozen_global_state_plus_fixed_epsilon',
    pair_order='persistent_random.Random(seed)_fresh_canonical_pair_permutation_each_epoch',within_pair_order='registered_pair_sides_0_then_1',
    saturation_rule='retain54_N16_K16_same_SID_pairs_with_exact_zero_regularizer',path_loss_role='diagnostic_only_on_corrected_aggregate',
    reference_gradient_audit_steps=[1,2,32],core_then_student_no_overlap=True,
    student_groups=972,student_steps=8000,student_batch_size=64,student_presentations=512000,
    student_optimizer='AdamW_predictor_only',student_lr=.001,student_warmup=100,student_final_lr=.00001,
    student_weight_decay=0.,student_clip_norm=1.,student_initialization_seed_offset=20261110,student_order_seed_offset=20261111,
    student_initialization='independent_private_Torch_seed_then_zero_C2_weight_and_bias',
    student_order='persistent_random.Random(seed+20261111)_sorted_group_permutation_each_cycle_carried_batches',
    target_precompute_groups=972,target_precompute_batch_size=64,target_mean='FP64_occurrence_mean_cast_FP32',
    target_denominator='frozen_FP32_message_squared_mean_over24_times96_plus1e-6',
    student_loss='mean64_of_mean96_squared_error_divided_by_fixed_group_denominator',
    student_core_query_target_and_denominator_frozen=True,
    conversion_groups=972,conversion_unique_scenes=1782,conversion_ordinary_positions=4266,conversion_batch_scenes=32,
    conversion_no_VLM_head_accuracy_or_selection=True,
    full_answer_targets='canonical_native_numeral_tokens_plus_exactly_one_terminal_EOS',max_new_tokens=4,exact_requires_eos=True,
    reject_nonterminal_special_tokens=True,native_eos=[151645,151643],target_eos=151645,
    selection='fixed_final_core4590_student8000',dev_label='dev_final',dev_examples=64,dev_descriptive_only=True,test_examples=272,
    profile_core_steps=32,profile_student_steps=32,training_count_support=list(range(17)),development_count_support=list(range(16)),
    test_count_support=list(range(17)),maximum_training_N=16,test_N=[32,64],bootstrap_seed=20261109,bootstrap_replicates=10000,
    per_main_seconds_cap=2700,campaign_gpu_seconds_cap=16200,profile_seconds_cap=300)
PROFILE_CALLS=dict(core_training_updates=32,student_training_updates=32,target_precompute_groups=972,conversion_groups=972,
    conversion_ordinary_positions=4266,native_prefill_forwards=10,visual_forwards=10,same_state_native_shaped_head_replays=10,
    cached_state_head_projections=10,standalone_head_calls=20,natural_generations=0)
TEST_FILES=('tests/test_native_vision_v14_training.py','tests/test_conditional_null_mean.py','tests/test_paired_sequence_objectives.py','scripts/native_vision_v10_pairs.py')
OWN=tuple(sorted(set(ancestor.OWN)|{'scripts/train_native_vision_v14.py',*TEST_FILES,
    'slurm/native_vision_v14_train_check.sbatch','slurm/native_vision_v14_train_profile.sbatch','slurm/native_vision_v14_train.sbatch'}))
save=ancestor.save;bind_file=ancestor.bind_file;bind_manifest=ancestor.bind_manifest;batch_states=ancestor.batch_states
lr=ancestor.lr;cache_binding=ancestor.cache_binding;state_info=ancestor.state_info
auxiliary_states=ancestor.auxiliary_states;evaluate=ancestor.evaluate;active_cache_replay=ancestor.active_cache_replay


def sources():return {**ancestor.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    (out/'code').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'code'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed while snapshotting')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V14 exact-bank training and distillation\n\n[Plan](plan.json) · [Configuration](config.json) · [Summary](summary.json) · [Sources](source_hashes.json)\n')
    return frozen


def presentation_order(pairs,seed):
    need(seed in POLICY['seeds'] and len(pairs)==918,'Unregistered V14 pair order')
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


def pairing_binding(cache):
    from scripts.native_vision_v10_pairs import load_pairs
    pairing=load_pairs(TRAIN,Path(cache['parent_cache']['file']),SCHEDULE)
    return dict(pairing,protocol='v14_unchanged_v10_pairing_new_seeds',seeds=POLICY['seeds'],parent_cache=cache['parent_cache'])


def student_order(groups,seed):
    need(seed in POLICY['seeds'] and len(groups)==972,'Unregistered student inventory')
    canonical=sorted(groups);rng=random.Random(seed+20261111);rows=[];cycle=0
    while len(rows)<512000:
        cycle+=1;slots=list(range(972));rng.shuffle(slots)
        for slot in slots:
            if len(rows)==512000:break
            rows.append(dict(cycle=cycle,slot=slot,group_id=canonical[slot]))
    need(Counter(r['cycle'] for r in rows)==Counter({**{c:972 for c in range(1,527)},527:728}),'Carried student cycle weights differ')
    return rows


def student_lr(step):
    need(1<=step<=8000,'Student optimizer step out of range')
    return .001*step/100 if step<=100 else .00001+.00099*(1+math.cos(math.pi*(step-100)/7900))/2


def make_student(torch,seed,rank=96):
    from gnnformer.conditional_null_mean import ConditionalNullMean
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed+20261110);model=ConditionalNullMean(rank)
    return model


def test_helpers():
    suites=[]
    for name in TEST_FILES:
        result=subprocess.run([sys.executable,str(REPO/name)],capture_output=True,text=True,cwd=REPO)
        suites.append(dict(file=name,sha256=sha(REPO/name),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        need(result.returncode==0,'CPU helper failed: '+name+'\n'+result.stderr)
    return dict(passed=True,suites=suites)


def bind_software(path,bindings):
    need(Path(path).resolve()==SOFTWARE,'Reuse canonical completed V13 software442413')
    return ancestor.bind_software(path,bindings)


def bind_prior(path,bindings):
    need(Path(path).resolve()==PRIOR,'Require the completed V13 failure')
    result=read(path);need(result['completed'] and result['verification_passed'] and not result['accepted_vision_milestone'],'Prior finalization differs')
    ref=bind_file(bindings,path)
    for item in result['artifacts'].values():bind_file(bindings,item['path'],item['sha256'])
    geometry=read(GEOMETRY);need(geometry['passed'] and geometry['completed'] and geometry['no_fit'] and geometry['no_accuracy']
        and geometry['total_groups']==3888 and geometry['total_ordinary_positions']==17064,'Complete V13 geometry required')
    for item in geometry['models']:bind_file(bindings,item['file'],item['sha256'])
    for name,digest in geometry['source_sha256'].items():bind_file(bindings,REPO/name,digest)
    return dict(finalization=ref,geometry=bind_file(bindings,GEOMETRY),no_prior_milestone_claim=True)


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
    student_orders={str(seed):object_sha(student_order(cache['auxiliary_groups'],seed)) for seed in POLICY['seeds']}
    software=bind_software(args.native_profile,bindings);prior=bind_prior(args.diagnostic_summary,bindings)
    save(out/'pairing.json',pairing)
    plan=dict(schema_version=1,policy=POLICY,profile_calls=PROFILE_CALLS,source_sha256=frozen,artifact_bindings=bindings,
        cache_binding=cache_ref,pairing_file=str(out/'pairing.json'),pairing_sha256=sha(out/'pairing.json'),pairing_object_sha256=object_sha(pairing),
        order_sha256=orders,student_order_sha256=student_orders,null_groups_sha256=object_sha(cache['auxiliary_groups']),
        train_manifest=str(TRAIN),schedule_file=str(SCHEDULE),original_pairing_file=str(PAIRING),fresh_manifests={},
        fresh_test_bound_by_future_independent_release=True,native_profile=software,prior_result=prior,tests=tests,
        actual_frozen_pair_equality_passed=True,no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify_plan(out/'plan.json')
    save(out/'summary.json',dict(passed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),source_sha256=frozen,
        tests=tests,no_model_loaded=True,elapsed_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify_plan(path):
    path=Path(path).resolve();need(path.with_suffix('.sha256').read_text().strip()==sha(path),'V14 plan sidecar differs');plan=read(path)
    need(plan['schema_version']==1 and plan['policy']==POLICY and plan['source_sha256']==sources() and plan['profile_calls']==PROFILE_CALLS,'Frozen V14 source/policy differs')
    for filename,digest in plan['artifact_bindings'].items():need(sha(filename)==digest,'Bound input changed: '+filename)
    need(sha(plan['pairing_file'])==plan['pairing_sha256'],'Pairing file changed')
    cache=cache_binding(plan['cache_binding']['file']);pairing=pairing_binding(cache)
    need(object_sha(pairing)==plan['pairing_object_sha256']==object_sha(read(plan['pairing_file'])),'Ordinary pairing changed')
    need(object_sha(cache['auxiliary_groups'])==plan['null_groups_sha256'],'Auxiliary grouping changed')
    for seed in POLICY['seeds']:
        need(object_sha(presentation_order(pairing['pairs'],seed))==plan['order_sha256'][str(seed)]
            and object_sha(student_order(cache['auxiliary_groups'],seed))==plan['student_order_sha256'][str(seed)],'Paired or auxiliary order changed')
    return plan


def losses(torch,core,local,g,reference_local,layout,norm,head,condition,sids,counts):
    from gnnformer.paired_sequence_objectives import sequence_objectives
    result_readout=empirical_batch(torch,core,local,g,reference_local,counts,condition)
    delta,z,nulls=result_readout["delta"],result_readout["corrected_aggregate"],result_readout["mean"]
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
            B_norms=result['bound'].tolist(),saturated_identity_pairs=identity,empirical_mean_norms=nulls.norm(dim=-1).tolist())
    need(all(bool(torch.isfinite(v)) for v in (main,ce,consistency,path)),'Nonfinite main objective')
    return main,ce,consistency,path,components,result_readout


def group_lookup(cache):
    lookup={group['global_feature_id']:gid for gid,group in cache['auxiliary_groups'].items()}
    need(len(lookup)==972,'Each strict native global view requires exactly one null group')
    return lookup


def position_groups(cache,sids,lookup):
    return [lookup[fid] for sid in sids for fid in cache['scenes'][sid]['global_feature_ids']]


def empirical_batch(torch,core,local,g,reference_local,counts,condition):
    """Exact occurrence mean remains live; only cached native inputs are frozen."""
    from gnnformer.conditional_null_mean import projected_null_readout
    need(condition in POLICY['conditions'] and local.ndim==3 and g.ndim==2
        and reference_local.shape==(24,g.shape[0],core.hidden_size) and len(counts)==g.shape[0]
        and all(type(n) is int and 0<n<=local.shape[0] for n in counts),'Empirical input shapes/cardinalities differ')
    messages=core.encode(reference_local,g)
    mean=messages.double().mean(0).float()  # Deliberately not detached.
    delta=torch.zeros_like(g,dtype=torch.float32)
    corrected=torch.zeros((g.shape[0],core.rank),device=g.device,dtype=torch.float32)
    preactivation=torch.zeros_like(corrected)
    for n in sorted(set(counts)):
        ids=torch.tensor([i for i,v in enumerate(counts) if v==n],device=g.device)
        need(bool((local[n:,ids]==0).all()),'Padding contains dropped real evidence')
        aggregate=core.aggregate(core.encode(local[:n,ids],g[ids]))
        result=projected_null_readout(core,aggregate,g[ids],mean[ids],n_elements=n,mode=condition,output_dtype=torch.float32)
        delta=delta.index_copy(0,ids,result['delta_float32'])
        corrected=corrected.index_copy(0,ids,result['corrected_aggregate'])
        preactivation=preactivation.index_copy(0,ids,result['corrected_preactivation'])
    need(bool(torch.isfinite(delta).all()) and bool(torch.isfinite(mean).all()),'Nonfinite empirical readout')
    return dict(delta=delta,corrected_aggregate=corrected,corrected_preactivation=preactivation,mean=mean,reference_messages=messages)


def reference_gradient_audit(torch,core,main,readout,step):
    messages=readout['reference_messages'];mean=readout['mean']
    need(messages.requires_grad and mean.requires_grad,'Empirical reference route was detached')
    dm=torch.autograd.grad(main,messages,retain_graph=True,allow_unused=False)[0]
    dq,dl=torch.autograd.grad(messages,(core.query.weight,core.local.weight),grad_outputs=dm,retain_graph=True,allow_unused=False)
    need(all(bool(torch.isfinite(x).all()) for x in (dm,dq,dl)) and all(p.grad is None for p in core.parameters()),
        'Reference route gradient inspection is nonfinite or changed optimizer buffers')
    norms=dict(reference_message=float(dm.norm()),reference_query_weight=float(dq.norm()),reference_local_weight=float(dl.norm()))
    if step==1:need(all(v==0 for v in norms.values()),'Zero-U should initially block reference gradients')
    else:need(all(v>0 for v in norms.values()),'Live reference query/message route not exercised')
    return dict(step=step,passed=True,mean_and_messages_live=True,optimizer_buffers_untouched=True,
        core_reference_gradients=norms,zero_up_expected=step==1,cached_native_features_frozen=True)


def zero_core_gradients(torch,core,consistency,path):
    need(bool((core.up.weight==0).all()) and float(consistency)==float(path)==0.,'Initial U/residual regularizer differs')
    result={}
    for label,value in [('residual',consistency),('path',path)]:
        grads=torch.autograd.grad(value,tuple(core.parameters()),retain_graph=True,allow_unused=True)
        result['all_'+label+'_gradients_zero']=all(x is None or bool(torch.isfinite(x).all() and (x==0).all()) for x in grads)
    need(all(result.values()) and all(p.grad is None for p in core.parameters()),'Initial regularizer gradients differ')
    return dict(passed=True,up_exactly_zero=True,residual_loss=0.,path_loss=0.,**result)


def tensor_digest_table(blob):
    import torch
    return {key:v7.tensor_info(value) for key,value in blob.items() if isinstance(value,torch.Tensor)}


def precompute_targets(torch,core,cache,states,index,dataout):
    need(not any(p.requires_grad or p.grad is not None for p in core.parameters()),'Precomputation requires a frozen clean core')
    begin=time.perf_counter();ident=v7.state_info(core);core_sha=object_sha(ident);ids=sorted(cache['auxiliary_groups'])
    queries=[];messages=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            h,g=auxiliary_states(torch,cache,states,index,ids[start:start+64])
            queries.append(core.query(core.rms(g)));messages.append(core.encode(h,g).transpose(0,1).contiguous())
        query=torch.cat(queries);message=torch.cat(messages);mean=message.double().mean(1).float()
        denominator=message.float().square().mean(dim=(1,2))+1e-6
        projected_messages=torch.nn.functional.linear(message,core.aggregate_projection.weight)
        projected=projected_messages.double()
        variance=(projected-projected.mean(1,keepdim=True)).square().sum(dim=(1,2))/24
    need(query.shape==(972,96) and message.shape==(972,24,96) and mean.shape==(972,96)
        and all(bool(torch.isfinite(t).all()) and not t.requires_grad for t in (query,message,mean,denominator,variance)),
        'Frozen target inventory or finite/detach contract differs')
    blob=dict(schema_version=1,group_ids=ids,query=query.detach().cpu(),messages=message.detach().cpu(),mean=mean.detach().cpu(),
        denominator=denominator.detach().cpu(),projected_messages=projected_messages.detach().cpu(),projected_population_variance=variance.detach().cpu(),
        core_parameter_sha256=core_sha,cache_sha256=sha(DATA/'v13_null_features/feature_cache.json'))
    path=dataout/'frozen_targets.pt';torch.save(blob,path)
    metadata=dict(schema_version=1,groups=972,occurrences=24,group_ids=ids,core_parameter_sha256=core_sha,
        cache_sha256=blob['cache_sha256'],file=str(path),sha256=sha(path),tensors=tensor_digest_table(blob),
        mean_rule=POLICY['target_mean'],denominator_rule=POLICY['target_denominator'],
        covariance_rule='FP32 projected messages; FP64 centered population trace with divisor24',all_targets_frozen=True)
    save(dataout/'target_metadata.json',metadata)
    need(v7.state_info(core)==ident,'Target precomputation changed core')
    torch.cuda.synchronize()
    return blob,metadata,time.perf_counter()-begin


def student_loss(torch,predictor,query,target,denominator):
    need(not query.requires_grad and not target.requires_grad and not denominator.requires_grad,
        'Student targets/query/normalization must remain fixed')
    predicted=predictor(query);mse=(predicted-target).square().mean(-1);values=mse/denominator;loss=values.mean()
    need(bool(torch.isfinite(loss)),'Nonfinite distillation loss')
    return loss,dict(per_group_mse=mse.detach().tolist(),per_group_denominator=denominator.tolist(),
        per_group_loss=values.detach().tolist(),target_norms=target.norm(dim=-1).tolist(),
        prediction_norms=predicted.detach().norm(dim=-1).tolist(),query_detached=True,target_detached=True,denominator_detached=True)


def conversion_states(torch,cache,states,index,sids):
    """Gather independent unique scenes; conversion has no paired objective."""
    from gnnformer.paired_sequence_objectives import pack_feature_sequences
    local=[];globals_=[];targets=[];offsets=[0]
    need(len(sids)>0 and len(set(sids))==len(sids),'Conversion requires distinct nonempty scene IDs')
    for sid in sids:
        scene=cache['scenes'][sid];ids=scene['local_feature_ids'];gids=scene['global_feature_ids'];tokens=list(scene['target_ids'])
        need(tokens and all(type(token) is int and token>=0 for token in tokens)
            and len(ids)==scene['n_frames'] and len(gids)==len(tokens) and all(len(row)==len(tokens) for row in ids),
            'Individual conversion scene prefix segmentation differs')
        local.append(states[torch.tensor([[index[f] for f in row] for row in ids],device=states.device)])
        globals_.append(states[torch.tensor([index[f] for f in gids],device=states.device)])
        targets.append(tokens);offsets.append(offsets[-1]+len(tokens))
    h,g=pack_feature_sequences(local,globals_)
    need(g.shape[0]==offsets[-1] and all(len(tokens)==b-a for tokens,a,b in zip(targets,offsets,offsets[1:])),
        'Packed independent conversion offsets differ')
    return h,g,dict(target_sequences=targets,offsets=offsets)


def conversion_diagnostics(torch,core,predictor,table,cache,states,index,condition,dataout):
    """Training features only. No native head, fitted classifier or accuracy oracle."""
    from gnnformer.conditional_null_mean import projected_null_readout
    need(not any(p.requires_grad or p.grad is not None for p in core.parameters()),'Core must stay frozen throughout conversion diagnostics')
    begin=time.perf_counter();before=state_info(core,predictor);device=states.device
    group_ids=table['group_ids'];group_index={gid:i for i,gid in enumerate(group_ids)};lookup=group_lookup(cache)
    query=table['query'].to(device);target=table['mean'].to(device);denominator=table['denominator'].to(device)
    with torch.no_grad():
        prediction=predictor(query);error=prediction-target;projected=torch.nn.functional.linear(error,core.aggregate_projection.weight)
        mse=error.square().mean(-1);error_sq=projected.double().square().sum(-1)
    variance=table['projected_population_variance'];groups=[]
    for i,gid in enumerate(group_ids):
        description=cache['auxiliary_groups'][gid];v=float(variance[i]);e=float(error_sq[i])
        groups.append(dict(group_id=gid,question=description['question'],prefix_ids=description['prefix_ids'],
            mse=float(mse[i]),denominator=float(denominator[i]),normalized_mse=float(mse[i]/denominator[i]),
            projected_error_squared=e,projected_error_norm=math.sqrt(e),projected_population_variance=v,
            projected_error_squared_over_variance=e/v if v>0 else None,zero_variance=v==0,
            target_norm=float(target[i].norm()),prediction_norm=float(prediction[i].norm()),query_norm=float(query[i].norm())))
    ordinary=[];raw=[];sids=sorted(cache['scenes'])
    for start in range(0,len(sids),32):
        selected=sids[start:start+32];local,g,layout=conversion_states(torch,cache,states,index,selected)
        counts=[cache['scenes'][sid]['n_frames'] for sid,tokens in zip(selected,layout['target_sequences']) for _ in tokens]
        gids=position_groups(cache,selected,lookup);ti=torch.tensor([group_index[gid] for gid in gids],device=device)
        with torch.no_grad():
            mu=target[ti];native_prediction=predictor(core.query(core.rms(g)))
            bank_delta=torch.zeros_like(g,dtype=torch.float32);student_delta=torch.zeros_like(bank_delta)
            bank_pre=torch.zeros((len(gids),96),device=device);student_pre=torch.zeros_like(bank_pre)
            for n in sorted(set(counts)):
                chosen=torch.tensor([i for i,v in enumerate(counts) if v==n],device=device)
                need(bool((local[n:,chosen]==0).all()),'Ordinary substitution dropped real evidence')
                z=core.aggregate(core.encode(local[:n,chosen],g[chosen]))
                a=projected_null_readout(core,z,g[chosen],mu[chosen],n_elements=n,mode=condition,output_dtype=torch.float32)
                b=projected_null_readout(core,z,g[chosen],native_prediction[chosen],n_elements=n,mode=condition,output_dtype=torch.float32)
                bank_delta[chosen]=a['delta_float32'];student_delta[chosen]=b['delta_float32']
                bank_pre[chosen]=a['corrected_preactivation'];student_pre[chosen]=b['corrected_preactivation']
            projected_error=torch.nn.functional.linear(native_prediction-mu,core.aggregate_projection.weight)
            coefficients=torch.tensor([n if condition=='centered' else 1 for n in counts],device=device)
            algebra=student_pre-bank_pre+coefficients[:,None]*projected_error
            pre_norm=(student_pre-bank_pre).double().norm(dim=-1);delta_norm=(student_delta-bank_delta).double().norm(dim=-1)
            algebra_norm=algebra.double().norm(dim=-1)
        for scene_index,sid in enumerate(selected):
            a,b=layout['offsets'][scene_index:scene_index+2];scene=cache['scenes'][sid]
            for pos,k in enumerate(range(a,b)):
                ordinary.append(dict(sid=sid,n_frames=scene['n_frames'],gold=scene['gold'],position=pos,prefix_ids=scene['target_ids'][:pos],
                    group_id=gids[k],preactivation_difference_norm=float(pre_norm[k]),residual_difference_norm=float(delta_norm[k]),
                    FP32_substitution_algebra_error_norm=float(algebra_norm[k]),empirical_residual_norm=float(bank_delta[k].norm()),
                    distilled_residual_norm=float(student_delta[k].norm())))
        raw.append(dict(sids=selected,offsets=layout['offsets'],group_ids=gids,bank_preactivation=bank_pre.cpu(),
            distilled_preactivation=student_pre.cpu(),bank_residual=bank_delta.cpu(),distilled_residual=student_delta.cpu(),
            projected_error=projected_error.cpu(),coefficients=coefficients.cpu()))
    need(len(groups)==972 and len(sids)==1782 and len(ordinary)==4266,'Complete conversion diagnostic coverage differs')
    with torch.no_grad():
        need(all(bool(torch.isfinite(t).all()) for row in raw for t in row.values() if isinstance(t,torch.Tensor)),
            'Nonfinite substitution tensor')
    rawpath=dataout/'conversion_raw.pt';torch.save(dict(group_ids=group_ids,predictions=prediction.cpu(),projected_error=projected.cpu(),ordinary=raw),rawpath)
    total_error=sum(r['projected_error_squared'] for r in groups);total_variance=sum(r['projected_population_variance'] for r in groups)
    value=dict(schema_version=1,condition=condition,groups=groups,ordinary_positions=ordinary,groups_count=972,unique_scenes=1782,
        ordinary_positions_count=4266,ratio_of_mean_projected_error_squared_to_mean_variance=total_error/total_variance if total_variance>0 else None,
        mean_normalized_mse=sum(r['normalized_mse'] for r in groups)/972,raw_file=str(rawpath),raw_sha256=sha(rawpath),
        all_groups_and_positions_retained=True,no_VLM_or_head_calls=True,no_accuracy=True,no_fit=True,no_selection=True,
        empirical_targets_are_training_bank_not_population=True,ordinary_weight='unique_scene_and_valid_prefix_once',
        native_batch_route_differences_descriptive=True)
    save(dataout/'conversion_diagnostics.json',value)
    need(state_info(core,predictor)==before,'Conversion diagnostics changed learned parameters')
    torch.cuda.synchronize()
    return dict(file=str(dataout/'conversion_diagnostics.json'),sha256=sha(dataout/'conversion_diagnostics.json'),
        raw_file=str(rawpath),raw_sha256=sha(rawpath),groups=972,ordinary_positions=4266,
        ratio_of_mean_projected_error_squared_to_mean_variance=value['ratio_of_mean_projected_error_squared_to_mean_variance'],
        mean_normalized_mse=value['mean_normalized_mse'],no_accuracy=True),time.perf_counter()-begin

def verify_release(path,plan_path,plan):
    need(path is not None,'V14 mains require a measured release')
    release=read(path)
    need(release['protocol']=='v14_exact_bank_distillation_main_release' and release['passed'] is True
        and release['plan_sha256']==sha(plan_path) and release['source_sha256']==sources()
        and release['per_main_seconds_cap']==2700 and release['main_block_gpu_seconds_cap']==16200
        and release['campaign_budget']['passed'] is True,'Main source/resource release differs')
    ref=release['data_release'];need(sha(ref['file'])==ref['sha256'],'Independent data release changed');checked=read(ref['file'])
    need(checked['passed'] is True and checked['source_sha256']==release['report_source_sha256'],'Independent report/data gate failed')
    for name,digest in checked['source_sha256'].items():need(sha(REPO/name)==digest,'Report source changed')
    for name,digest in checked['data_bindings'].items():need(sha(name)==digest,'Data binding changed')
    fresh=release['fresh_manifest'];need(Path(fresh['file']).resolve()==FRESH/'main_manifest.json'
        and sha(fresh['file'])==fresh['sha256'] and checked['data_bindings'][str(Path(fresh['file']).resolve())]==fresh['sha256'],
        'Fresh272 data not independently bound')
    need({k:len(v['samples']) for k,v in read(fresh['file'])['splits'].items()}=={'test_N32':136,'test_N64':136},'Fresh test inventory differs')
    need(set(release['profiles'])==set(release['projections'])==set(POLICY['conditions']),'Both matched profiles required')
    for condition,item in release['profiles'].items():
        filename=Path(item['directory'])/'summary.json';need(sha(filename)==item['summary_sha256'],'Profile changed');s=read(filename)
        need(s['profile'] is True and s['condition']==condition and s['seed']==18 and s['passed'] and s['completed']
            and s['steps']==s['student_steps']==32 and s['computational_integrity_passed'] and s['no_dev_or_test_evaluation']
            and s['plan_sha256']==sha(plan_path) and s['source_sha256']==sources() and s['profile_calls']==PROFILE_CALLS
            and s['active_cache_replay']['passed'] and s['checkpoint_roundtrip_passed'] and s['core_freeze_passed']
            and s['reference_gradient_audit_passed'],'Training/distillation profile gate failed')
        for key in ('core_step_seconds_max_steady','student_step_seconds_max_steady','target_precompute_seconds','conversion_diagnostics_seconds'):
            need(math.isfinite(s[key]) and s[key]>0,'Missing measured phase timing')
        need(release['projections'][condition]['passed'] and 0<release['projections'][condition]['projected_seconds']<=2700,'Projected main exceeds cap')
    return dict(file=str(Path(path).resolve()),sha256=sha(path),fresh_manifest=fresh)


def run(args):
    job=os.environ['SLURM_JOB_ID'];runid=f'{"profile" if args.profile else "run"}_{args.condition}_s{args.seed}_{job}'
    out=OUT/runid;out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    import torch,transformers
    from gnnformer.runtime import load_runtime
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from scripts.stage_native_vision_v10_features import runtime_identity,model_metadata,native_api
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.perf_counter();args.plan=args.plan.resolve();plan=verify_plan(args.plan)
    need(args.condition in POLICY['conditions'] and args.seed in POLICY['seeds'] and (not args.profile or args.seed==18),'Unregistered V14 condition/seed')
    release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    cache=cache_binding(plan['cache_binding']['file']);dataout=DATA/'native_aggregation_vlm_v14'/runid;dataout.mkdir(parents=True,exist_ok=False)
    ckpt=CKPT/runid;ckpt.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=runtime.model.eval().requires_grad_(False)
    native.native_contract(model)
    need(runtime_identity()==cache['runtime'] and model_metadata()==cache['model']
        and fingerprint(runtime.processor,str(transformers.__version__))==cache['processor'],'Model/runtime/processor differs from immutable cache')
    need(native_api(runtime.processor)[2]==cache['native_api'] and native.generation_policy(model,runtime.processor.tokenizer)[1]['native_eos_token_ids']==POLICY['native_eos'],
        'Native API/decoding differs')
    states,feature_index=v7.load_features(torch,cache,runtime.device);torch.cuda.synchronize();loadseconds=time.perf_counter()-tick
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);core=ParallelLocalAggregation().to(runtime.device)
    native.native_contract(model,core);initial_core={k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
    student_preview=make_student(torch,args.seed);student_initial_info=v7.state_info(student_preview);del student_preview
    initialized=dict(branch=v7.state_info(core),predictor=student_initial_info)
    need(sum(p.numel() for p in core.parameters())==1041600,'Core count changed')
    optimizer=torch.optim.AdamW(core.parameters(),lr=.001,weight_decay=0.)
    pairing=read(plan['pairing_file']);order=presentation_order(pairing['pairs'],args.seed);student_rows=student_order(cache['auxiliary_groups'],args.seed)
    need(object_sha(order)==plan['order_sha256'][str(args.seed)] and object_sha(student_rows)==plan['student_order_sha256'][str(args.seed)],'Phase presentation order differs')
    save(out/'presentations.json',order);save(out/'student_presentations.json',student_rows)
    config=dict(run_id=runid,arm='parallel',condition=args.condition,seed=args.seed,profile=args.profile,policy=POLICY,
        consistency_coefficient=1.,slurm_job_id=job,plan_file=str(args.plan),plan_sha256=sha(args.plan),source_sha256=frozen,
        cache_binding=plan['cache_binding'],pairing_file=plan['pairing_file'],pairing_sha256=plan['pairing_sha256'],prior_result=plan['prior_result'],
        null_groups_sha256=plan['null_groups_sha256'],initialized=initialized,initialized_sha256=object_sha(initialized),
        student_initialization_seed=args.seed+20261110,student_initialized_sha256=object_sha(student_initial_info),
        presentations_sha256=sha(out/'presentations.json'),order_sha256=object_sha(order),
        student_presentations_sha256=sha(out/'student_presentations.json'),student_order_sha256=object_sha(student_rows),
        model=cache['model'],runtime=cache['runtime'],processor=cache['processor'],native_dtypes=cache['native_dtypes'],
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch=str(torch.__version__)),
        model_and_features_load_seconds=loadseconds,checkpoint_directory=str(ckpt),data_directory=str(dataout),main_release=release)
    save(out/'config.json',config)
    totalsteps=32 if args.profile else 4590;student_steps=32 if args.profile else 8000
    trainlog=[];studentlog=[];core_gradients=[];student_gradients=[];gradient_audit=[];lookup=group_lookup(cache)
    norm,head=model.model.language_model.norm,model.lm_head
    exercised={name:False for name,_ in core.named_parameters()};model_versions={name:p._version for name,p in model.named_parameters()}
    setup_before_core_seconds=time.perf_counter()-start
    for step in range(1,totalsteps+1):
        begin=time.perf_counter();rows=order[(step-1)*16:step*16];sids=[r['sid'] for r in rows]
        need(len(rows)==16 and all(rows[i]['pair_id']==rows[i+1]['pair_id'] and rows[i]['pair_side']==0 and rows[i+1]['pair_side']==1 for i in range(0,16,2)),
            'Core batches must retain intact adjacent pairs')
        local,g,layout=batch_states(torch,cache,states,feature_index,sids)
        counts=[r['n_frames'] for r,tokens in zip(rows,layout['target_sequences']) for _ in tokens]
        reference_ids=position_groups(cache,sids,lookup);rh,rg=auxiliary_states(torch,cache,states,feature_index,reference_ids)
        need(torch.equal(g,rg) and not any(x.requires_grad for x in (local,g,rh,rg)),'Actual/reference native global identity differs')
        optimizer.zero_grad(set_to_none=True);rate=lr(step)
        for group in optimizer.param_groups:group['lr']=rate
        loss,ce,consistency,path,components,readout=losses(torch,core,local,g,rh,layout,norm,head,args.condition,sids,counts)
        if step==1:zero=zero_core_gradients(torch,core,consistency,path)
        if step in POLICY['reference_gradient_audit_steps']:gradient_audit.append(reference_gradient_audit(torch,core,loss,readout,step))
        loss.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters()),'Missing/nonfinite core gradient')
        if step<=32:
            for name,p in core.named_parameters():exercised[name]|=bool((p.grad!=0).any())
        if step<=2:core_gradients.append(dict(step=step,pre_clip={name:v7.tensor_info(p.grad) for name,p in core.named_parameters()}))
        gradient_norm=float(torch.nn.utils.clip_grad_norm_(core.parameters(),1.));need(math.isfinite(gradient_norm),'Core gradient norm nonfinite')
        optimizer.step();need(all(bool(torch.isfinite(p).all()) for p in core.parameters()),'Nonfinite updated core')
        torch.cuda.synchronize()
        trainlog.append(dict(step=step,lr=rate,loss=float(loss),main_loss=float(loss),ce_loss=float(ce),consistency_loss=float(consistency),path_loss=float(path),
            consistency_coefficient=1.,weighted_consistency_loss=float(consistency),gradient_norm=gradient_norm,clipped=gradient_norm>1.,
            seconds=time.perf_counter()-begin,target_ids=layout['targets'],sids=sids,pair_ids=[r['pair_id'] for r in rows[::2]],
            epochs=[r['epoch'] for r in rows[::2]],reference_group_ids=reference_ids,reference_mean_live=True,**components))
        if step%500==0:print(json.dumps(dict(phase='core',step=step,loss=float(loss),ce=float(ce))),flush=True)
    log_tick=time.perf_counter()
    save(out/'training.json',trainlog);save(out/'first_gradients.json',core_gradients);save(out/'reference_gradient_audit.json',gradient_audit)
    phase_log_io_seconds=time.perf_counter()-log_tick
    need(len(gradient_audit)==3 and all(r['passed'] for r in gradient_audit) and all(exercised.values()),'Core reference/parameter gradient gate incomplete')
    for p in core.parameters():p.grad=None
    core.eval().requires_grad_(False);del optimizer,loss,ce,consistency,path,readout,local,g,rh,rg
    frozen_core=v7.state_info(core);core_sha=object_sha(frozen_core);core_checkpoint=ckpt/'core_frozen.pt'
    torch.save(dict(branch=core.state_dict(),step=totalsteps,config=config),core_checkpoint)
    save(out/'core_phase.json',dict(completed=True,step=totalsteps,core_checkpoint=str(core_checkpoint),core_checkpoint_sha256=sha(core_checkpoint),
        core_parameter_sha256=core_sha,training_sha256=sha(out/'training.json'),reference_gradient_audit_sha256=sha(out/'reference_gradient_audit.json')))
    table,target_metadata,precompute_seconds=precompute_targets(torch,core,cache,states,feature_index,dataout)
    need(table['core_parameter_sha256']==core_sha,'Frozen target core differs')
    predictor=make_student(torch,args.seed).to(runtime.device)
    need(v7.state_info(predictor)==student_initial_info and sum(p.numel() for p in predictor.parameters())==18624,'Independent student initialization changed')
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=.001,weight_decay=0.)
    q=table['query'].to(runtime.device);target=table['mean'].to(runtime.device);denominator=table['denominator'].to(runtime.device)
    group_index={gid:i for i,gid in enumerate(table['group_ids'])};student_exercised={name:False for name,_ in predictor.named_parameters()}
    for step in range(1,student_steps+1):
        begin=time.perf_counter();rows=student_rows[(step-1)*64:step*64];ids=[r['group_id'] for r in rows]
        need(len(rows)==64,'Student cycle tail dropped');indices=torch.tensor([group_index[gid] for gid in ids],device=runtime.device)
        optimizer.zero_grad(set_to_none=True);rate=student_lr(step)
        for group in optimizer.param_groups:group['lr']=rate
        loss,details=student_loss(torch,predictor,q[indices],target[indices],denominator[indices]);loss.backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in predictor.parameters())
            and not any(p.requires_grad or p.grad is not None for p in core.parameters()),'Distillation gradient leak/nonfinite gradient')
        if step<=32:
            for name,p in predictor.named_parameters():student_exercised[name]|=bool((p.grad!=0).any())
        if step<=2:student_gradients.append(dict(step=step,pre_clip={name:v7.tensor_info(p.grad) for name,p in predictor.named_parameters()}))
        gradient_norm=float(torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.));need(math.isfinite(gradient_norm),'Student gradient norm nonfinite')
        optimizer.step();need(all(bool(torch.isfinite(p).all()) for p in predictor.parameters()),'Nonfinite updated student')
        torch.cuda.synchronize()
        studentlog.append(dict(step=step,lr=rate,loss=float(loss),gradient_norm=gradient_norm,clipped=gradient_norm>1.,
            seconds=time.perf_counter()-begin,group_ids=ids,cycles=[r['cycle'] for r in rows],**details))
        if step%1000==0:print(json.dumps(dict(phase='student',step=step,loss=float(loss))),flush=True)
    log_tick=time.perf_counter()
    save(out/'distillation.json',studentlog);save(out/'student_first_gradients.json',student_gradients)
    phase_log_io_seconds+=time.perf_counter()-log_tick
    need(v7.state_info(core)==frozen_core and all(student_exercised.values()),'Frozen core changed or student parameter route unexercised')
    need(torch.equal(q.cpu(),table['query']) and torch.equal(target.cpu(),table['mean']) and torch.equal(denominator.cpu(),table['denominator'])
        and sha(target_metadata['file'])==target_metadata['sha256'] and tensor_digest_table(table)==target_metadata['tensors']
        and read(dataout/'target_metadata.json')==target_metadata,'Frozen query/target/denominator or target archive changed during student phase')
    for p in predictor.parameters():p.grad=None
    predictor.eval();del optimizer,loss
    conversion,conversion_seconds=conversion_diagnostics(torch,core,predictor,table,cache,states,feature_index,args.condition,dataout)
    need(v7.state_info(core)==frozen_core,'Conversion changed frozen core')
    freeze=dict(schema_version=1,core_step=totalsteps,student_step=student_steps,core_checkpoint=str(core_checkpoint),core_checkpoint_sha256=sha(core_checkpoint),
        core_parameter_sha256=core_sha,core_unchanged_after_distillation=True,core_unchanged_after_diagnostics=True,
        fixed_query_target_denominator_and_archive_unchanged=True,
        target_file=target_metadata['file'],target_sha256=target_metadata['sha256'],target_metadata_file=str(dataout/'target_metadata.json'),
        target_metadata_sha256=sha(dataout/'target_metadata.json'),student_initialization_seed=args.seed+20261110,
        student_initialized_sha256=object_sha(student_initial_info))
    save(out/'core_freeze.json',freeze)
    combined=state_info(core,predictor);checkpoint=ckpt/('profile.pt' if args.profile else 'final.pt')
    torch.save(dict(branch=core.state_dict(),predictor=predictor.state_dict(),step=totalsteps,student_step=student_steps,config=config),checkpoint)
    # Exercise combined save/restore in both phases without selecting a checkpoint.
    core.load_state_dict(initial_core);predictor.load_state_dict(make_student(torch,args.seed).state_dict())
    need(state_info(core,predictor)==initialized,'Initial state restore failed')
    saved=torch.load(checkpoint,map_location=runtime.device,weights_only=True);core.load_state_dict(saved['branch']);predictor.load_state_dict(saved['predictor'])
    need(state_info(core,predictor)==combined and object_sha(v7.state_info(core))==core_sha,'Final checkpoint restore failed')
    result=dict(config,passed=True,completed=True,computational_integrity_passed=True,steps=totalsteps,student_steps=student_steps,
        core_freeze_passed=True,core_freeze_file=str(out/'core_freeze.json'),core_freeze_sha256=sha(out/'core_freeze.json'),
        reference_gradient_audit_passed=True,reference_gradient_audit_file=str(out/'reference_gradient_audit.json'),
        reference_gradient_audit_sha256=sha(out/'reference_gradient_audit.json'),
        zero_initialization_core_gradients=zero,all_parameter_gradients_exercised=dict(branch=exercised,predictor=student_exercised),
        checkpoint_roundtrip_passed=True,conversion_diagnostics=conversion,target_metadata=target_metadata,
        target_precompute_seconds=precompute_seconds,conversion_diagnostics_seconds=conversion_seconds,
        core_step_seconds_max_steady=max(r['seconds'] for r in trainlog[4:]),student_step_seconds_max_steady=max(r['seconds'] for r in studentlog[4:]),
        training_seconds=sum(r['seconds'] for r in trainlog),student_training_seconds=sum(r['seconds'] for r in studentlog),
        training_file=str(out/'training.json'),training_sha256=sha(out/'training.json'),distillation_file=str(out/'distillation.json'),distillation_sha256=sha(out/'distillation.json'),
        first_gradients_file=str(out/'first_gradients.json'),first_gradients_sha256=sha(out/'first_gradients.json'),
        student_first_gradients_file=str(out/'student_first_gradients.json'),student_first_gradients_sha256=sha(out/'student_first_gradients.json'))
    manifest=read(plan['train_manifest'])
    if args.profile:
        replay=active_cache_replay(torch,model,runtime.processor,core,predictor,cache,states,feature_index,manifest,dataout,args.condition)
        save(out/'active_cache_replay.json',replay);need(replay['passed'],'Same-captured native head replay failed')
        result.update(profile_calls=PROFILE_CALLS,active_cache_replay=replay,no_dev_or_test_evaluation=True,
            profile_checkpoint=str(checkpoint),profile_checkpoint_sha256=sha(checkpoint))
    else:
        devrows=[('dev_N16',r) for r in manifest['splits']['dev_N16']['samples']]
        dev=evaluate(torch,model,runtime.processor,core,predictor,devrows,out,'dev_final',dataout,args.condition)
        selected=dict(step=4590,student_step=8000,exact_count=dev['exact_count'],nll=dev['first_token_nll'],checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),parameter_sha256=object_sha(combined),dev_file=str(out/'dev_final.json'),dev_sha256=sha(out/'dev_final.json'))
        save(out/'selection.json',dict(rule=POLICY['selection'],development=[selected],selected=selected,dev_descriptive_only=True))
        fresh=read(release['fresh_manifest']['file']);testrows=[('main_'+cell,r) for cell,split in fresh['splits'].items() for r in split['samples']]
        need(len(testrows)==272,'Fresh272 inventory differs');evaluate(torch,model,runtime.processor,core,predictor,testrows,out,'test',dataout,args.condition)
        result.update(selected=selected,development=[selected],dev_descriptive_only=True,native_dev_count=64,native_test_count=272,
            test_file=str(out/'test.json'),test_sha256=sha(out/'test.json'))
    need(model_versions=={name:p._version for name,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters()),
        'Frozen native backbone changed')
    need(state_info(core,predictor)==combined,'Native evaluation changed trained parameters')
    validation_tick=time.perf_counter();verify_plan(args.plan)
    if release is not None:need(sha(release['file'])==release['sha256'],'Main release changed');verify_release(args.main_release,args.plan,plan)
    result.update(frozen_backbone_gradient_state_preserved=True,setup_before_core_seconds=setup_before_core_seconds,
        setup_includes_model_and_feature_load=True,phase_log_io_seconds=phase_log_io_seconds,
        final_validation_seconds=time.perf_counter()-validation_tick,elapsed_seconds=time.perf_counter()-start)
    save(out/'summary.json',result);print(json.dumps(dict(run_id=runid,completed=True,elapsed_seconds=result['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--profile',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--parallel-cache',type=Path);parser.add_argument('--native-profile',type=Path,default=SOFTWARE)
    parser.add_argument('--diagnostic-summary',type=Path,default=PRIOR);parser.add_argument('--plan',type=Path);parser.add_argument('--main-release',type=Path)
    parser.add_argument('--condition',choices=POLICY['conditions']);parser.add_argument('--seed',type=int,choices=POLICY['seeds']);args=parser.parse_args()
    native.require_slurm(gpu=not args.check)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU source freeze needs CPU Slurm')
        need(args.parallel_cache is not None,'--check requires the immutable V13 union cache');check(args)
    else:
        need(args.plan is not None and args.condition is not None and args.seed is not None,'GPU modes require plan/condition/seed');run(args)


if __name__=='__main__':main()
