"""Prospective V13 missing-null-feature harvest and exact V10 union merge.

A separate CPU freeze fixes eight missing and eight old local profile inputs.
Only 19,664 absent native final-norm-input states are harvested. No feature from
V10 is replaced, no model is fitted, and occurrence weights remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_vision_v13_null_features as inventory_stage
from scripts import stage_native_vision_v10_features as ancestor
from scripts.cache_native_vision_v10_features import replay_metrics
from scripts.stage_native_vision_v10_features import MODEL,need,read,save,sha,object_sha,tensor_info,row_inputs,pack,runtime_identity,native_api
DATA=Path('/mnt/data/gabriele/gnn_transformer/v13_null_features')
OUT=REPO/'outputs/native_aggregation_vlm/v13/null_cache'
INVENTORY=DATA/'stage_442353/plan.json'
INVENTORY_SHA='fa6e3014dff2a083cf45dab1888c065bb52e55244e06379dbca44ae1e03a88ef'
PROTOCOL='v13_missing_null_native_features'
STRATA=('1','9','10','16')
POLICY=dict(profile_strata=list(STRATA),missing_per_stratum=2,reused_per_stratum=2,profile_small_rows=4,
    profile_stress_rows=64,profile_distinct_missing=8,profile_distinct_reused=8,profile_model_calls=8,
    profile_vision_calls=8,profile_standalone_head_calls=8,shards=4,rows_per_shard=4916,harvest_batch_size=64,
    model_calls_per_shard=77,total_harvest_model_calls=308,total_successful_model_calls=316,
    profile_seconds_cap=180,shard_seconds_cap=600,campaign_gpu_seconds_cap=2700,
    read_boundary='actual native final norm input',native_dtype='torch.float16',hidden_size=3584,
    selection='per prefix: two lowest missing-null FIDs then two lowest existing V10 local FIDs; no outputs',
    old_control_membership='ordinary V10 local features; null membership recorded, never assumed',
    projection='load + pixels + 1.5 * 77 * max(four stress64 harvest_seconds) + 30 <=600')
OWN=tuple(dict.fromkeys(('scripts/cache_native_vision_v13_null_features.py',
    'slurm/native_vision_v13_null_cache_check.sbatch','slurm/native_vision_v13_null_feature_profile.sbatch',
    'slurm/native_vision_v13_null_feature_harvest.sbatch','slurm/native_vision_v13_null_feature_merge.sbatch',
    *inventory_stage.OWN,*ancestor.OWN)))


def require_slurm(gpu=False):
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')==('gpu' if gpu else 'cpu'),
        'Run all worker/check/merge execution through the correct Slurm partition')
    if gpu:need(os.environ.get('SLURM_JOB_GPUS'),'Require an assigned GPU')
    else:need(not os.environ.get('SLURM_JOB_GPUS'),'CPU operation must not allocate a GPU')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen);return frozen


def index(out,title,links):
    (out/'INDEX.md').write_text('# '+title+'\n\n'+' · '.join('['+a+']('+b+')' for a,b in links)+'\n')


def inventory_inputs():
    need(sha(INVENTORY)==INVENTORY_SHA and INVENTORY.with_suffix('.sha256').read_text().strip()==INVENTORY_SHA,'Canonical V13 inventory changed')
    staged=read(INVENTORY)
    need(staged['protocol']==inventory_stage.PROTOCOL and staged['source_sha256']==inventory_stage.sources()
        and staged['inventory_sha256']==object_sha(staged['inventory']) and staged['inventory']['counts']==inventory_stage.EXPECTED,
        'Frozen inventory source/content differs')
    completed=inventory_stage.OUT/f'check_{staged["slurm_job_id"]}'/'summary.json';summary=read(completed)
    need(summary['passed'] and summary['plan_sha256']==INVENTORY_SHA and summary['source_sha256']==staged['source_sha256'],
        'Completed canonical inventory CPU check required')
    for item in staged['source_bindings'].values():need(sha(item['file'])==item['sha256'],'Inventory ancestor binding changed')
    cache=read(staged['source_bindings']['cache']['file']);parent=ancestor.verify_plan(cache['plan_file'],pixels=False)
    need(sha(cache['plan_file'])==cache['plan_sha256'] and staged['inventory']==inventory_stage.inventory(cache,parent),
        'Reference occurrence inventory or actual input identities changed')
    return staged,cache,parent


def select_profile(missing,parent_features,target_token_ids,required_null_ids):
    result=[]
    for text in STRATA:
        prefix=target_token_ids[text][:-1]
        new=sorted(fid for fid,row in missing.items() if row['prefix_ids']==prefix)
        old=sorted(fid for fid,row in parent_features.items() if row['kind']=='local' and row['prefix_ids']==prefix)
        need(len(new)>=2 and len(old)>=2 and not set(new)&set(parent_features),'Missing/reused profile populations overlap or are too small')
        ids=new[:2]+old[:2]
        result.append(dict(stratum=text,prefix_ids=prefix,feature_ids=ids,missing_feature_ids=new[:2],reused_feature_ids=old[:2],
            reused_null_membership={fid:fid in required_null_ids for fid in old[:2]}))
    need(len({fid for row in result for fid in row['feature_ids']})==16,'Profile does not have sixteen distinct inputs')
    return result


def projection_seconds(observations,load_seconds,pixel_seconds):
    stress=[r['harvest_seconds'] for r in observations if r['case']=='stress64']
    need(len(stress)==4 and all(math.isfinite(x) and x>0 for x in stress)
        and all(math.isfinite(x) and x>=0 for x in (load_seconds,pixel_seconds)),'Missing or invalid timing coverage')
    projected=load_seconds+pixel_seconds+1.5*77*max(stress)+30
    return [projected]*4


def self_test():
    missing={};old={};targets={}
    for i,text in enumerate(STRATA):
        prefix=[i+1];targets[text]=prefix+[99]
        for j in (2,0,1):missing[f'n{i}{j}']=dict(prefix_ids=prefix,kind='local');old[f'o{i}{j}']=dict(prefix_ids=prefix,kind='local')
    chosen=select_profile(missing,old,targets,set())
    need(chosen[0]['feature_ids']==['n00','n01','o00','o01'] and all(not any(r['reused_null_membership'].values()) for r in chosen),
        'Selection must use canonical IDs and disclose non-null controls')
    observed=[dict(case='stress64',harvest_seconds=x) for x in (1.,2.,3.,4.)]
    need(projection_seconds(observed,10.,20.)==[522.]*4,'Projection must use the shared worst stress batch')
    return dict(passed=True,tests=['canonical_disjoint_profile_inputs','nonnull_control_disclosure','shared_worst_stress_projection'])


def check(args):
    require_slurm();job=os.environ['SLURM_JOB_ID'];begin=time.perf_counter();out=OUT/f'check_{job}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);index(out,'V13 missing-feature worker CPU freeze',[('Summary','summary.json'),('Sources','source_hashes.json')])
    tests=self_test();staged,cache,parent=inventory_inputs();inv=staged['inventory']
    registration=read(args.registration)
    need(registration.get('protocol')==PROTOCOL and registration.get('source_sha256')==frozen and registration.get('policy')==POLICY
        and registration.get('inventory_plan_sha256')==INVENTORY_SHA and registration.get('worker_stage_authorized') is True,
        'Exact-source prospective worker/profile registration required')
    selected=select_profile(inv['missing_features'],parent['features'],parent['target_token_ids'],set(inv['required_local_feature_ids']))
    features=dict(inv['missing_features'])
    controls={fid:cache['features'][fid] for row in selected for fid in row['reused_feature_ids']}
    for fid in controls:features[fid]=parent['features'][fid]
    layouts={r['layout_id']:parent['layouts'][r['layout_id']] for r in features.values()}
    pairs={r['pair_id']:parent['pairs'][r['pair_id']] for r in features.values()}
    shards=inv['missing_feature_shards'];need(list(map(len,shards))==[4916]*4,'Missing-state shard sizes changed')
    need(all(parent['eos_token_id'] not in r['prefix_ids'] and r['phase']=='local_prefix' and r['kind']=='local'
        and r['prefix_ids'] in parent['strict_prefix_vocabulary'] for r in features.values()),'Auxiliary input is not an ordinary strict prefix')
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,
        inventory_plan=dict(file=str(INVENTORY),sha256=INVENTORY_SHA),inventory_sha256=staged['inventory_sha256'],
        registration=dict(file=str(Path(args.registration).resolve()),sha256=sha(args.registration)),
        parent_cache=staged['source_bindings']['cache'],parent_plan=staged['source_bindings']['parent_plan'],
        model=parent['model'],runtime=parent['runtime'],processor=parent['processor'],native_api=parent['native_api'],
        read_boundary=inventory_stage.READ_BOUNDARY,native_dtype='torch.float16',features=features,layouts=layouts,pairs=pairs,
        missing_feature_ids=sorted(inv['missing_features']),shards=shards,profile_groups=selected,profile_reused_controls=controls,
        target_token_ids=parent['target_token_ids'],strict_prefix_vocabulary=parent['strict_prefix_vocabulary'],
        eos_token_id=parent['eos_token_id'],pad_token_id=parent['pad_token_id'],pixels_file=parent['pixels_file'],
        pixels_sha256=parent['pixels_sha256'],pixel_info=parent['pixel_info'],native_dtypes=cache['native_dtypes'],
        fixed_counts=inv['counts'],tests=tests,no_model_loaded=True,no_tensor_loaded=True,no_training_release=True,slurm_job_id=job)
    target=DATA/f'worker_stage_{job}';target.mkdir(parents=True,exist_ok=False)
    need(sources()==frozen,'Worker source changed during CPU freeze')
    save(target/'plan.json',plan);(target/'plan.sha256').write_text(sha(target/'plan.json')+'\n')
    verify_plan(target/'plan.json',pixels=False)
    save(out/'summary.json',dict(passed=True,completed=True,plan_file=str(target/'plan.json'),plan_sha256=sha(target/'plan.json'),
        source_sha256=frozen,policy=POLICY,tests=tests,profile_distinct_features=16,missing_features=19664,no_tensor_loaded=True,
        no_model_loaded=True,no_training_release=True,seconds=time.perf_counter()-begin,slurm_job_id=job))
    index(target,'V13 missing-feature execution plan',[('Plan','plan.json')]);print(json.dumps(dict(passed=True,plan_file=str(target/'plan.json'))),flush=True)


def verify_plan(path,*,pixels=False):
    path=Path(path).resolve();need(path.is_relative_to(DATA) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Worker plan path/sidecar differs')
    plan=read(path);need(plan['schema_version']==1 and plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources(),
        'Worker source/protocol/policy changed')
    for field in ('inventory_plan','registration','parent_cache','parent_plan'):
        need(sha(plan[field]['file'])==plan[field]['sha256'],'Frozen worker ancestor changed: '+field)
    registration=read(plan['registration']['file'])
    need(registration.get('protocol')==PROTOCOL and registration.get('source_sha256')==plan['source_sha256']
        and registration.get('policy')==POLICY and registration.get('inventory_plan_sha256')==INVENTORY_SHA
        and registration.get('worker_stage_authorized') is True,'Worker authorization changed')
    staged=read(plan['inventory_plan']['file']);inv=staged['inventory'];parent=read(plan['parent_plan']['file']);cache=read(plan['parent_cache']['file'])
    need(plan['inventory_plan']['sha256']==INVENTORY_SHA and staged['source_sha256']==inventory_stage.sources()
        and plan['inventory_sha256']==staged['inventory_sha256']==object_sha(inv) and inv['counts']==inventory_stage.EXPECTED,
        'Worker is not bound to the frozen exact inventory')
    need(plan['read_boundary']==inventory_stage.READ_BOUNDARY and plan['native_dtype']=='torch.float16'
        and plan['no_model_loaded'] is True and plan['no_tensor_loaded'] is True and plan['no_training_release'] is True,
        'Worker read boundary or stage-only contract changed')
    need(plan['fixed_counts']==inv['counts'] and plan['missing_feature_ids']==sorted(inv['missing_features'])
        and plan['shards']==inv['missing_feature_shards'] and list(map(len,plan['shards']))==[4916]*4,'Missing-state inventory changed')
    selected=select_profile(inv['missing_features'],parent['features'],parent['target_token_ids'],set(inv['required_local_feature_ids']))
    need(plan['profile_groups']==selected,'Response-independent profile selection changed')
    controls={fid:cache['features'][fid] for row in selected for fid in row['reused_feature_ids']}
    expected=dict(inv['missing_features']);expected.update({fid:parent['features'][fid] for fid in controls})
    need(plan['features']==expected and plan['profile_reused_controls']==controls
        and plan['layouts']=={r['layout_id']:parent['layouts'][r['layout_id']] for r in expected.values()}
        and plan['pairs']=={r['pair_id']:parent['pairs'][r['pair_id']] for r in expected.values()},'Actual native descriptor/layout/pair identity changed')
    need(all(plan[k]==parent[k] for k in ('model','runtime','processor','native_api','target_token_ids','strict_prefix_vocabulary','eos_token_id',
        'pad_token_id','pixels_file','pixels_sha256','pixel_info')) and plan['native_dtypes']==cache['native_dtypes'], 'Native V10 extraction contract changed')
    need(plan['model']==ancestor.model_metadata() and plan['runtime']==runtime_identity(),'Installed native runtime/model differs')
    for name,digest in parent['source_sha256'].items():need(sha(REPO/name)==digest,'Frozen V10 extraction source changed')
    if pixels:need(sha(plan['pixels_file'])==plan['pixels_sha256'],'Immutable V10 processed pixels changed')
    return plan


def load_controls(torch,plan):
    result={};locations=plan['profile_reused_controls']
    for path,digest in {r['file']:r['file_sha256'] for r in locations.values()}.items():
        need(sha(path)==digest,'Original V10 control state file changed');blob=torch.load(path,map_location='cpu',weights_only=True)
        for fid,location in locations.items():
            if location['file']!=path:continue
            value=blob['states'][location['row']]
            need(blob['feature_ids'][location['row']]==fid and tensor_info(value)['sha256']==location['state_sha256']
                and value.shape==(3584,) and value.dtype==torch.float16,'Original V10 control tensor changed')
            result[fid]=value.clone()
    need(len(result)==8,'Missing immutable control features');return result


def hidden_difference(torch,new,old):
    difference=new.float()-old.float();rms=float(difference.square().mean().sqrt());base=float(old.float().square().mean().sqrt())
    return dict(exact=torch.equal(new,old),rms=rms,l2=float(torch.linalg.vector_norm(difference)),
        maximum_absolute=float(difference.abs().max()),relative_rms=rms/max(base,1e-12),binding=False)


def gpu(args):
    profile=args.profile
    require_slurm(gpu=True)
    need(profile != (args.shard is not None),'Choose exactly one profile or shard')
    if not profile:need(0<=args.shard<4 and args.profile_directory is not None,'Shard0..3 and passing profile are required')
    job=os.environ['SLURM_JOB_ID'];name=f'profile_{job}' if profile else f'shard{args.shard}_{job}'
    out=OUT/name;out.mkdir(parents=True,exist_ok=False);snapshot(out)
    index(out,'V13 feature attempt',[('Source snapshot','source_hashes.json'),('Summary after completion','summary.json')])
    import torch,transformers
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.perf_counter();plan=verify_plan(args.plan,pixels=True)
    cpu=read(OUT/f'check_{plan["slurm_job_id"]}'/'summary.json')
    need(cpu['passed'] and cpu['plan_sha256']==sha(args.plan) and cpu['source_sha256']==sources(),'Completed exact worker CPU freeze required')
    need(torch.cuda.device_count()==1,'Exactly one visible GPU is required')
    coverage=plan['profile_groups']
    reference=None if profile else verify_profile(args.profile_directory,plan,args.plan)
    data=DATA/name;data.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    pixel_start=time.perf_counter()
    blob=torch.load(plan['pixels_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==1 and set(blob['pixels'])==set(plan['pixel_info']),'Pixel coverage differs')
    pixels=blob['pixels']
    for key,value in pixels.items():need(tensor_info(value)==plan['pixel_info'][key],'Processed pixel tensor changed')
    pixel_seconds=time.perf_counter()-pixel_start
    control_start=time.perf_counter();controls=load_controls(torch,plan) if profile else {}
    control_seconds=time.perf_counter()-control_start
    load_start=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.eval();model.requires_grad_(False)
    need(fingerprint(runtime.processor,str(transformers.__version__))==plan['processor'],'Native processor differs')
    need(runtime_identity()==plan['runtime'],'Native runtime differs')
    _,_,native=native_api(runtime.processor);need(native==plan['native_api'],'Native implementation changed')
    norm=model.model.language_model.norm;language=model.model.language_model
    dtypes=dict(norm=str(norm.weight.dtype),lm_head=str(model.lm_head.weight.dtype))
    need(dtypes==dict(norm='torch.float16',lm_head='torch.float16'),'Expected actual native FP16 norm/head')
    need(not any(p.requires_grad for p in model.parameters()),'Backbone must be completely frozen')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    hardware=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
                  cuda_version=torch.version.cuda,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if reference is not None:
        need(reference['native_dtypes']==dtypes and reference['hardware']==hardware,
             'Native numerical backend differs from the software profile')
    config=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,read_boundary=plan['read_boundary'],
                parent_cache=plan['parent_cache'],inventory_plan=plan['inventory_plan'],
                profile=profile,shard=args.shard,plan_file=str(Path(args.plan).resolve()),
                plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],model=plan['model'],
                runtime=plan['runtime'],processor=plan['processor'],hardware=hardware,native_dtypes=dtypes,
                profile_directory=None if profile else str(Path(args.profile_directory).resolve()),
                profile_summary_sha256=None if profile else sha(Path(args.profile_directory)/'summary.json'),
                native_api=plan['native_api'],prefix_coverage=coverage,per_shard_seconds_cap=POLICY['shard_seconds_cap'],
                slurm_job_id=job)
    save(out/'config.json',config)
    index(out,'V13 native training-feature harvest',[('Configuration','config.json'),('Batch observations','observations.json'),
          ('Summary after completion','summary.json')])
    captured={};counts=dict(model=0,vision=0,language=0,norm=0,last_block=0);recording=False
    def model_hook(*_):counts['model']+=1
    def visual_hook(*_):counts['vision']+=1
    def language_hook(module,args,kwargs):
        counts['language']+=1
        captured['positions']=kwargs['position_ids'].detach().cpu().clone()
        captured['mask']=kwargs['attention_mask'].detach().cpu().clone()
    def last_block_hook(*_):counts['last_block']+=1
    def norm_hook(module,args):
        if recording:
            counts['norm']+=1;captured['hidden']=args[0][:,-1,:].detach().clone()
    handles=[model.register_forward_pre_hook(model_hook),model.model.visual.register_forward_pre_hook(visual_hook),
             language.register_forward_pre_hook(language_hook,with_kwargs=True),norm.register_forward_pre_hook(norm_hook),
             language.layers[-1].register_forward_pre_hook(last_block_hook)]
    rope=get_rope_index_fn(model);observations=[];raw=[];state_chunks=[];state_ids=[];small_states={}
    def batch(ids,stratum,kind):
        phase='local_prefix'
        nonlocal recording
        begin=time.perf_counter();rows=[row_inputs(torch,plan,pixels,fid) for fid in ids]
        inputs=pack(torch,rows,plan['pad_token_id'])
        expected_pos,expected_delta=rope(input_ids=inputs['input_ids'],image_grid_thw=inputs.get('image_grid_thw'),
                                          attention_mask=inputs['attention_mask'])
        for i,fid in enumerate(ids):
            layout=plan['layouts'][plan['features'][fid]['layout_id']];length=len(layout['input_ids'])
            need(torch.equal(expected_pos[:,i,-length:],torch.tensor(layout['position_ids'])[:,0]),
                 'Packed valid positions differ from independently staged row positions')
        identity=dict(feature_ids=ids,input_ids=tensor_info(inputs['input_ids']),
                      attention_mask=tensor_info(inputs['attention_mask']),position_ids=tensor_info(expected_pos),
                      image_sha256=[plan['pairs'][plan['features'][fid]['pair_id']]['image_sha256'] for fid in ids]
                                   if phase.startswith('local') else [],rope_deltas=expected_delta.tolist())
        item=move_to_device(inputs,runtime.device);torch.cuda.synchronize();prepared_at=time.perf_counter()
        captured.clear();before=dict(counts);recording=True
        with torch.inference_mode():output=model(**item,use_cache=False,logits_to_keep=1)
        recording=False;torch.cuda.synchronize();forward_seconds=time.perf_counter()-prepared_at
        need({key:counts[key]-before[key] for key in counts}==
             dict(model=1,vision=1,language=1,norm=1,last_block=1),'Native forward counts differ')
        need(torch.equal(captured['positions'],expected_pos) and torch.equal(captured['mask'],inputs['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(),expected_delta),'Executed native positions/mask differ')
        h=captured['hidden']
        need(h.shape==(len(ids),3584) and h.dtype==torch.float16 and bool(torch.isfinite(h).all())
             and output.logits.shape[:2]==(len(ids),1) and output.past_key_values is None,'Native feature/forward contract differs')
        states=h.cpu().clone();harvest_seconds=time.perf_counter()-begin
        observation=dict(phase=phase,stratum=stratum,case=kind,batch_size=len(ids),identity=identity,
                         preprocessing_seconds=prepared_at-begin,forward_seconds=forward_seconds,
                         harvest_seconds=harvest_seconds,states=tensor_info(states),
                         native_model_forwards=1,native_visual_forwards=int(phase.startswith('local')))
        if profile:
            with torch.inference_mode():replayed=model.lm_head.forward(norm.forward(h.unsqueeze(1)))[:,0]
            native_logits=output.logits[:,0];metrics=replay_metrics(torch,native_logits,replayed)
            observation['replay']=metrics
            observation['standalone_head_calls']=1
            observation['old_cache_hidden_descriptive']=[dict(row=i,feature_id=fid,**hidden_difference(torch,states[i],controls[fid]))
                for i,fid in enumerate(ids) if fid in controls]
            observation['old_cache_comparison_is_binding']=False
            if kind=='small':
                for fid,value in zip(ids,states):small_states[fid]=value.clone()
            else:
                differences=[hidden_difference(torch,value,small_states[fid]) for fid,value in zip(ids,states)]
                observation['small_vs_stress_hidden_descriptive']=differences
                observation['small_vs_stress_is_descriptive']=True
            raw.append(dict(phase=phase,stratum=stratum,case=kind,feature_ids=ids,states=states,
                            immutable_old_states={fid:controls[fid] for fid in set(ids)&set(controls)},
                            native_logits=native_logits.cpu(),replayed_logits=replayed.cpu()))
        else:
            state_ids.extend(ids);state_chunks.append(states)
        observations.append(observation)
        print(json.dumps(dict(phase=phase,case=kind,n=len(ids),seconds=harvest_seconds)),flush=True)
    try:
        if profile:
            for group in plan['profile_groups']:
                ids=group['feature_ids'];stratum=group['stratum']
                batch(ids,stratum,'small');batch([ids[i%4] for i in range(64)],stratum,'stress64')
        else:
            ids=plan['shards'][args.shard]
            for offset in range(0,len(ids),64):batch(ids[offset:offset+64],'mixed_numeric_prefix',f'batch_{offset//64:04d}')
    finally:
        recording=False
        for handle in handles:handle.remove()
    need(verify_plan(args.plan,pixels=False)==plan,'Plan/source changed during harvesting')
    save(out/'observations.json',observations)
    summary=dict(config,completed=True,computational_integrity_passed=True,calls=counts,model_load_seconds=load_seconds,
                 pixel_load_verify_seconds=pixel_seconds,reused_control_load_seconds=control_seconds,observations_file=str(out/'observations.json'),
                 observations_sha256=sha(out/'observations.json'),total_seconds=time.perf_counter()-start,
                 peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    if profile:
        need(len(observations)==8 and counts==dict(model=8,vision=8,language=8,norm=8,last_block=8),'Profile call coverage differs')
        raw_path=data/'raw.pt';torch.save(dict(schema_version=1,observations=raw),raw_path)
        projections=projection_seconds(observations,load_seconds,pixel_seconds)
        numerical=all(row['replay']['passed'] for row in observations)
        summary.update(raw_file=str(raw_path),raw_sha256=sha(raw_path),numerical_gate_passed=numerical,
                       shard_projected_seconds=projections,runtime_projection_passed=max(projections)<=POLICY['shard_seconds_cap'],
                       profile_feature_count=16,profile_missing_features=8,profile_reused_features=8,standalone_head_calls=8,
                       reused_state_and_batch_route_differences_descriptive=True,
                       passed=numerical and max(projections)<=POLICY['shard_seconds_cap'],
                       no_quality_or_accuracy_gate=True)
    else:
        need(state_ids==plan['shards'][args.shard] and len(set(state_ids))==len(state_ids),'Shard feature order/coverage differs')
        need(len(observations)==77 and counts==dict(model=77,vision=77,language=77,norm=77,last_block=77),'Shard call budget differs')
        states=torch.cat(state_chunks);path=data/'features.pt'
        torch.save(dict(schema_version=1,feature_ids=state_ids,states=states),path)
        summary.update(passed=True,feature_count=len(state_ids),features_file=str(path),features_sha256=sha(path),
                       states=tensor_info(states),feature_ids_sha256=object_sha(state_ids),
                       feature_state_sha256={fid:tensor_info(row)['sha256'] for fid,row in zip(state_ids,states)})
    summary['total_seconds']=time.perf_counter()-start
    summary['execution_within_cap']=summary['total_seconds']<=(180 if profile else 600)
    summary['passed']=summary['passed'] and summary['execution_within_cap']
    save(out/'summary.json',summary)
    index(data,'V13 feature tensors',[('Profile raw tensors' if profile else 'Native feature states',
                                   'raw.pt' if profile else 'features.pt')])
    print(json.dumps(dict(directory=str(out),passed=summary['passed'],calls=counts,total_seconds=summary['total_seconds'])),flush=True)
    if not summary['passed']:raise SystemExit('V13 feature profile failed; all raw observations retained')


def verify_snapshot(directory,plan):
    directory=Path(directory).resolve()
    need(directory.is_relative_to(OUT) and read(directory/'source_hashes.json')==plan['source_sha256'],
        'Attempt source ledger or output directory differs')
    for name,digest in plan['source_sha256'].items():
        need(sha(directory/'source'/name.replace('/','_'))==digest,'Attempt source snapshot changed')


def verify_profile(directory,plan,path):
    """Validate a completed GPU profile using hashes and small metadata only."""
    directory=Path(directory).resolve();verify_snapshot(directory,plan);result=read(directory/'summary.json')
    need(result['profile'] is True and result['shard'] is None and result['completed'] is True and result['passed'] is True
        and result['computational_integrity_passed'] is True and result['numerical_gate_passed'] is True
        and result['runtime_projection_passed'] is True and result['execution_within_cap'] is True,
        'A completed passing GPU software profile is required')
    need(result['protocol']==PROTOCOL and result['policy']==POLICY and result['plan_sha256']==sha(path)
        and result['source_sha256']==plan['source_sha256'] and result['prefix_coverage']==plan['profile_groups'],
        'Profile exact-source/plan/selection identity differs')
    need(all(result[key]==plan[key] for key in ('model','runtime','processor','native_api','read_boundary','parent_cache','inventory_plan'))
        and result['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'), 'Profile native provenance changed')
    need(read(directory/'config.json')=={key:result[key] for key in read(directory/'config.json')}
        and sha(directory/'plan.json')==sha(path),'Profile config/plan snapshot changed')
    need(sha(result['raw_file'])==result['raw_sha256'] and sha(result['observations_file'])==result['observations_sha256'],
        'Profile raw evidence changed')
    rows=read(result['observations_file']);expected=[(g,k) for g in plan['profile_groups'] for k in ('small','stress64')]
    need(len(rows)==8,'Profile must contain exactly eight batches')
    for row,(group,kind) in zip(rows,expected):
        ids=group['feature_ids'] if kind=='small' else [group['feature_ids'][i%4] for i in range(64)]
        need(row['phase']=='local_prefix' and row['stratum']==group['stratum'] and row['case']==kind
            and row['batch_size']==len(ids) and row['identity']['feature_ids']==ids
            and row['native_model_forwards']==row['native_visual_forwards']==row['standalone_head_calls']==1,
            'Profile batch order, input identity or call count differs')
        metrics=row['replay'];tv=metrics['tv'];same=metrics['top1_equal']
        need(len(tv)==len(same)==len(ids) and all(type(x) is bool for x in same) and all(same)
            and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=.02 for x in tv)
            and metrics['maximum_tv']==max(tv) and metrics['all_top1_equal'] is True and metrics['passed'] is True,
            'Native same-captured-state head gate failed or changed')
        expected_old=[(i,fid) for i,fid in enumerate(ids) if fid in plan['profile_reused_controls']]
        need([(x['row'],x['feature_id']) for x in row['old_cache_hidden_descriptive']]==expected_old
            and row['old_cache_comparison_is_binding'] is False
            and all(x['binding'] is False for x in row['old_cache_hidden_descriptive']),
            'Old-cache comparisons must preserve control identities and remain descriptive')
        if kind=='stress64':
            need(row['small_vs_stress_is_descriptive'] is True and len(row['small_vs_stress_hidden_descriptive'])==64
                and all(x['binding'] is False for x in row['small_vs_stress_hidden_descriptive']),
                'Batch-route comparisons must remain descriptive')
    projected=projection_seconds(rows,result['model_load_seconds'],result['pixel_load_verify_seconds'])
    need(result['shard_projected_seconds']==projected and max(projected)<=600 and result['per_shard_seconds_cap']==600
        and result['calls']==dict(model=8,vision=8,language=8,norm=8,last_block=8)
        and result['profile_feature_count']==16 and result['profile_missing_features']==result['profile_reused_features']==8
        and result['standalone_head_calls']==8 and result['reused_state_and_batch_route_differences_descriptive'] is True
        and result['no_quality_or_accuracy_gate'] is True and 0<result['total_seconds']<=180,
        'Profile measured projection or registered coverage/cap differs')
    return result


def verify_profile_tensors(torch,result,plan):
    """Independent CPU replay of recorded metrics, with no model or head calls."""
    raw=torch.load(result['raw_file'],map_location='cpu',weights_only=True)
    rows=read(result['observations_file']);need(raw['schema_version']==1 and len(raw['observations'])==8,'Profile raw schema differs')
    controls=load_controls(torch,plan);small={}
    for item,row in zip(raw['observations'],rows):
        ids=row['identity']['feature_ids'];states=item['states']
        need(item['feature_ids']==ids and all(item[k]==row[k] for k in ('phase','stratum','case'))
            and tensor_info(states)==row['states'] and states.shape==(len(ids),3584) and states.dtype==torch.float16
            and bool(torch.isfinite(states).all()),'Profile raw state identity differs')
        need(item['native_logits'].dtype==item['replayed_logits'].dtype==torch.float16, 'Profile raw native logits dtype differs')
        recomputed=replay_metrics(torch,item['native_logits'],item['replayed_logits']);recorded=row['replay']
        need(all(recomputed[key]==recorded[key] for key in ('top1_equal','all_top1_equal','passed'))
            and len(recomputed['tv'])==len(recorded['tv'])
            and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12) for a,b in zip(recomputed['tv'],recorded['tv']))
            and math.isclose(recomputed['maximum_tv'],recorded['maximum_tv'],rel_tol=1e-9,abs_tol=1e-12),
            'Recorded native head comparisons differ from raw tensors beyond CPU/GPU FP64 reduction precision')
        need(set(item['immutable_old_states'])==set(ids)&set(controls)
            and all(torch.equal(value,controls[fid]) for fid,value in item['immutable_old_states'].items()),
            'Profile old controls differ from the original cache tensors')
        expected=[dict(row=i,feature_id=fid,**hidden_difference(torch,states[i],controls[fid]))
            for i,fid in enumerate(ids) if fid in controls]
        need(expected==row['old_cache_hidden_descriptive'],'Old cached state differences were changed')
        if row['case']=='small':small.update({fid:value.clone() for fid,value in zip(ids,states)})
        else:
            need([hidden_difference(torch,value,small[fid]) for fid,value in zip(ids,states)]==row['small_vs_stress_hidden_descriptive'],
                'Small/stress descriptive differences were changed')
    return dict(passed=True,batches=8,distinct_missing=8,distinct_reused=8,old_state_comparisons=136,
        small_stress_comparisons=256,head_replays_recomputed=8,no_model_loaded=True)


def validate_locations(torch,locations):
    """Hash every original/new tensor row; never rewrite or cast cached states."""
    files={};verified=0
    for fid,location in locations.items():files.setdefault(location['file'],[]).append((fid,location))
    for path,entries in files.items():
        need(len({entry['file_sha256'] for _,entry in entries})==1 and sha(path)==entries[0][1]['file_sha256'],
            'Cached tensor file changed or conflicting hashes supplied')
        blob=torch.load(path,map_location='cpu',weights_only=True)
        need(blob['schema_version']==1 and blob['states'].shape==(len(blob['feature_ids']),3584)
            and blob['states'].dtype==torch.float16 and bool(torch.isfinite(blob['states']).all()),'Cached native state format differs')
        for fid,location in entries:
            row=location['row'];need(type(row) is int and 0<=row<len(blob['feature_ids']) and blob['feature_ids'][row]==fid
                and tensor_info(blob['states'][row])['sha256']==location['state_sha256'],'Cached state ID/row/bytes changed')
            verified+=1
    return dict(passed=True,files=len(files),features=verified,native_dtype='torch.float16',hidden_size=3584)


def validate_observed_inputs(torch,plan,row,ids):
    """Independently reconstruct left-padded token/mask rows from frozen layouts."""
    layouts=[plan['layouts'][plan['features'][fid]['layout_id']] for fid in ids]
    width=max(len(layout['input_ids']) for layout in layouts)
    input_ids=torch.full((len(ids),width),plan['pad_token_id'],dtype=torch.long);mask=torch.zeros_like(input_ids)
    for i,layout in enumerate(layouts):
        length=len(layout['input_ids']);input_ids[i,-length:]=torch.tensor(layout['input_ids']);mask[i,-length:]=1
    need(row['identity']['feature_ids']==ids and row['identity']['input_ids']==tensor_info(input_ids)
        and row['identity']['attention_mask']==tensor_info(mask)
        and row['identity']['image_sha256']==[plan['pairs'][plan['features'][fid]['pair_id']]['image_sha256'] for fid in ids],
        'Observed actual image/question/strict-prefix input differs from the frozen layout')


def merge(args):
    require_slurm();job=os.environ['SLURM_JOB_ID'];start=time.perf_counter();out=OUT/f'merge_{job}'
    out.mkdir(parents=True,exist_ok=False);snapshot(out);index(out,'V13 exact native cache union',[('Summary','summary.json'),('Sources','source_hashes.json')])
    import torch
    torch.set_num_threads(4);plan=verify_plan(args.plan,pixels=False)
    cpu=read(OUT/f'check_{plan["slurm_job_id"]}'/'summary.json')
    need(cpu['passed'] and cpu['plan_sha256']==sha(args.plan),'Completed exact worker CPU freeze required')
    cache=read(plan['parent_cache']['file']);staged=read(plan['inventory_plan']['file']);inv=staged['inventory']
    old_audit=validate_locations(torch,cache['features']);need(old_audit['features']==33658,'Old V10 state coverage differs')
    locations=dict(cache['features']);seen=set();summaries=[];common=None;profile=None;profile_audit=None
    need(len(args.merge)==4 and len({Path(x).resolve() for x in args.merge})==4,'Four distinct shard directories are required')
    for directory in map(lambda p:Path(p).resolve(),args.merge):
        verify_snapshot(directory,plan);summary=read(directory/'summary.json');shard=summary['shard']
        need(type(shard) is int and shard in range(4) and shard not in seen,'Duplicate/invalid shard index');seen.add(shard)
        need(summary['protocol']==PROTOCOL and summary['policy']==POLICY and summary['profile'] is False
            and summary['completed'] is True and summary['passed'] is True and summary['computational_integrity_passed'] is True
            and summary['execution_within_cap'] is True and 0<summary['total_seconds']<=600
            and summary['plan_sha256']==sha(args.plan) and summary['source_sha256']==plan['source_sha256'],
            'A completed bounded exact-plan shard is required')
        need(all(summary[k]==plan[k] for k in ('model','runtime','processor','native_api','read_boundary','parent_cache','inventory_plan'))
            and summary['native_dtypes']==plan['native_dtypes'] and summary['prefix_coverage']==plan['profile_groups'],
            'Shard model/native/source provenance differs')
        config=read(directory/'config.json')
        need(config=={key:summary[key] for key in config} and sha(directory/'plan.json')==sha(args.plan),'Shard config/plan copy changed')
        if common is None:
            common=summary;profile=verify_profile(summary['profile_directory'],plan,args.plan)
            need(sha(Path(summary['profile_directory'])/'summary.json')==summary['profile_summary_sha256'],'Shard profile binding changed')
            profile_audit=verify_profile_tensors(torch,profile,plan)
            for row in read(profile['observations_file']):validate_observed_inputs(torch,plan,row,row['identity']['feature_ids'])
        need(all(summary[k]==common[k] for k in ('profile_directory','profile_summary_sha256','hardware','native_dtypes'))
            and summary['hardware']==profile['hardware'],'Shards must use the same passing profile and native backend')
        need(sha(summary['features_file'])==summary['features_sha256']
            and sha(summary['observations_file'])==summary['observations_sha256'],'Shard raw state/observation files changed')
        observations=read(summary['observations_file']);ids=plan['shards'][shard]
        batches=[ids[offset:offset+64] for offset in range(0,len(ids),64)]
        need(len(observations)==len(batches)==77 and summary['calls']==dict(model=77,vision=77,language=77,norm=77,last_block=77),
            'Shard exact call inventory differs')
        blob=torch.load(summary['features_file'],map_location='cpu',weights_only=True);states=blob['states']
        need(blob['schema_version']==1 and blob['feature_ids']==ids and states.shape==(4916,3584) and states.dtype==torch.float16
            and bool(torch.isfinite(states).all()) and summary['feature_count']==4916 and tensor_info(states)==summary['states']
            and object_sha(ids)==summary['feature_ids_sha256'] and set(summary['feature_state_sha256'])==set(ids),
            'Shard native tensor/feature identities differ')
        offset=0
        for b,(observation,batch_ids) in enumerate(zip(observations,batches)):
            need(observation['phase']=='local_prefix' and observation['stratum']=='mixed_numeric_prefix'
                and observation['case']==f'batch_{b:04d}' and observation['batch_size']==len(batch_ids)
                and observation['native_model_forwards']==observation['native_visual_forwards']==1,
                'Shard batch order/call metadata differs')
            validate_observed_inputs(torch,plan,observation,batch_ids)
            end=offset+len(batch_ids);need(tensor_info(states[offset:end])==observation['states'],'Saved shard state differs from captured batch')
            offset=end
        need(offset==4916,'Observed shard row count differs')
        for row,(fid,value) in enumerate(zip(ids,states)):
            digest=tensor_info(value)['sha256'];need(fid not in locations and digest==summary['feature_state_sha256'][fid],
                'Missing feature overlaps old cache or native bytes differ')
            locations[fid]=dict(file=summary['features_file'],file_sha256=summary['features_sha256'],row=row,state_sha256=digest)
        summaries.append(dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),shard=shard))
    summaries.sort(key=lambda x:x['shard'])
    need(seen==set(range(4)) and len(locations)==53322 and set(locations)==set(cache['features'])|set(plan['missing_feature_ids'])
        and all(locations[fid]==entry for fid,entry in cache['features'].items()),'Exact V10 union coverage/preservation failed')
    need(len(inv['auxiliary_groups'])==972 and all(len(group['local_feature_ids'])==24
        and all(fid in locations for fid in group['local_feature_ids']) and group['global_feature_id'] in cache['features']
        for group in inv['auxiliary_groups'].values()),'Auxiliary occurrence/global support incomplete')
    need(verify_plan(args.plan,pixels=False)==plan,'Plan/source changed during merge')
    result=dict(schema_version=1,protocol='v13_training_null_union_cache',complete=True,training_only=True,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        native_dtypes=common['native_dtypes'],hardware=common['hardware'],read_boundary=plan['read_boundary'],
        parent_cache=plan['parent_cache'],parent_plan=plan['parent_plan'],inventory_plan=plan['inventory_plan'],inventory_sha256=plan['inventory_sha256'],
        target_token_ids=cache['target_token_ids'],scenes=cache['scenes'],features=locations,feature_count=53322,
        original_feature_count=33658,new_feature_count=19664,counts=inv['counts'],null_banks=inv['banks'],
        auxiliary_groups=inv['auxiliary_groups'],required_null_feature_ids=inv['required_local_feature_ids'],
        required_global_feature_ids=inv['required_global_feature_ids'],strict_prefix_vocabulary=inv['strict_prefix_vocabulary'],
        shards=summaries,parent_shards=cache['shards'],profile_directory=common['profile_directory'],
        profile_summary_sha256=common['profile_summary_sha256'],preserved_v10_features_exact=True,
        preserve_occurrence_multiplicity=True,no_new_global_states=True,no_training_release=True,
        merge_directory=str(out),merge_slurm_job_id=job)
    target=DATA/'feature_cache.json';need(not target.exists() and not target.with_suffix('.sha256').exists(),
        'Preserve an existing or partial canonical V13 cache')
    save(target,result);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    save(out/'summary.json',dict(passed=True,completed=True,computational_integrity_passed=True,plan_file=str(Path(args.plan).resolve()),
        plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],cache_file=str(target),cache_sha256=sha(target),
        feature_count=53322,original_feature_count=33658,new_feature_count=19664,counts=inv['counts'],
        old_tensor_audit=old_audit,profile_tensor_audit=profile_audit,shards=summaries,preserved_v10_features_exact=True,
        successful_artifact_calls=dict(model=316,vision=316,standalone_head=8,last_block=316),
        campaign_allocation_accounting_required=True,no_training_release=True,seconds=time.perf_counter()-start,slurm_job_id=job))
    print(json.dumps(dict(passed=True,cache_file=str(target),cache_sha256=sha(target),feature_count=53322)),flush=True)


def verify_cache(path,*,verify_tensors=False):
    """Read-only public union validator. Tensor audit is explicitly CPU-Slurm-only."""
    path=Path(path).resolve();need(path==DATA/'feature_cache.json' and sha(path)==path.with_suffix('.sha256').read_text().strip(),
        'Canonical V13 cache path/hash differs')
    result=read(path);plan=verify_plan(result['plan_file'],pixels=False)
    need(result['schema_version']==1 and result['protocol']=='v13_training_null_union_cache' and result['complete'] is True
        and result['training_only'] is True and result['plan_sha256']==sha(result['plan_file'])
        and result['source_sha256']==plan['source_sha256'],'Union source/completion identity differs')
    cache=read(plan['parent_cache']['file']);inv=read(plan['inventory_plan']['file'])['inventory']
    need(all(result[k]==plan[k] for k in ('model','runtime','processor','native_api','native_dtypes','read_boundary','parent_cache','parent_plan',
        'inventory_plan','inventory_sha256')) and result['feature_count']==len(result['features'])==53322
        and result['original_feature_count']==33658 and result['new_feature_count']==19664 and result['counts']==inv['counts'],
        'Union native identity or coverage differs')
    need(result['scenes']==cache['scenes'] and result['target_token_ids']==cache['target_token_ids']
        and all(result['features'][fid]==entry for fid,entry in cache['features'].items())
        and set(result['features'])==set(cache['features'])|set(inv['missing_features'])
        and result['null_banks']==inv['banks'] and result['auxiliary_groups']==inv['auxiliary_groups']
        and result['required_null_feature_ids']==inv['required_local_feature_ids']
        and result['required_global_feature_ids']==inv['required_global_feature_ids']
        and result['strict_prefix_vocabulary']==inv['strict_prefix_vocabulary']
        and result['preserved_v10_features_exact'] is True and result['preserve_occurrence_multiplicity'] is True,
        'Union changed original V10 states/scenes or auxiliary occurrence ordering')
    profile=verify_profile(result['profile_directory'],plan,result['plan_file'])
    need(sha(Path(result['profile_directory'])/'summary.json')==result['profile_summary_sha256']
        and profile['hardware']==result['hardware'],'Union profile/native hardware binding changed')
    verify_snapshot(result['merge_directory'],plan);summary=read(Path(result['merge_directory'])/'summary.json')
    need(summary['passed'] and summary['completed'] and summary['cache_file']==str(path) and summary['cache_sha256']==sha(path)
        and summary['feature_count']==53322 and summary['preserved_v10_features_exact'] is True,'Completed independent CPU union merge required')
    need([r['shard'] for r in result['shards']]==[0,1,2,3] and result['parent_shards']==cache['shards'],'Union shard provenance differs')
    for entry in result['shards']:
        directory=Path(entry['directory']);need(sha(directory/'summary.json')==entry['summary_sha256'],'Union shard summary changed')
        shard=read(directory/'summary.json');need(shard['passed'] and shard['completed'] and shard['shard']==entry['shard']
            and shard['plan_sha256']==result['plan_sha256'] and shard['source_sha256']==result['source_sha256'],'Union shard identity differs')
        for row,fid in enumerate(plan['shards'][entry['shard']]):
            need(result['features'][fid]==dict(file=shard['features_file'],file_sha256=shard['features_sha256'],row=row,
                state_sha256=shard['feature_state_sha256'][fid]),'Union new feature locator differs from independently merged shard')
    if verify_tensors:
        require_slurm();import torch
        torch.set_num_threads(4);validate_locations(torch,result['features']);verify_profile_tensors(torch,profile,plan)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true');modes.add_argument('--profile',action='store_true')
    modes.add_argument('--shard',type=int);modes.add_argument('--merge',type=Path,nargs=4)
    parser.add_argument('--registration',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--profile-directory',type=Path)
    args=parser.parse_args()
    if args.check:need(args.registration is not None,'Prospective exact-source registration is required');check(args)
    else:
        need(args.plan is not None,'Frozen CPU worker plan required');args.plan=args.plan.resolve()
        if args.merge:merge(args)
        else:gpu(args)


if __name__=='__main__':main()
