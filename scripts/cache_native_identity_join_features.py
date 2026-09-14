"""Native learned-selection join feature profile, harvest and independent merge.

All numerical work runs under Slurm. No branch, local classifier, fitting or
answer-accuracy gate is used. The new runtime's passed software proof is required
before GPU extraction. Native head replays bind fixed captured states; different
native batch-route hidden states are retained as descriptive comparisons.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import math
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_identity_join_features import (DATA,CKPT,OUT,MODEL,PROTOCOL,PHASES,need,read,save,sha,oid,
    sources,snapshot,verify_plan,row_inputs,pack,tensor_info,native_api,runtime_identity,profile_selection)
PROFILE_CAP=300
SHARD_CAP=1200


def replay_metrics(torch,a,b):
    need(a.shape==b.shape and a.ndim==2 and a.dtype==b.dtype==torch.float16
         and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Native FP16 logit replay shape/dtype differs')
    tv=.5*(torch.softmax(a.double(),-1)-torch.softmax(b.double(),-1)).abs().sum(-1)
    same=a.argmax(-1)==b.argmax(-1)
    return dict(tv=tv.cpu().tolist(),top1_equal=same.cpu().tolist(),maximum_tv=float(tv.max()),
                all_top1_equal=bool(same.all()),passed=bool((tv<=.02).all() and same.all()))


def projections(plan,observations,load,pixels):
    need(all(math.isfinite(v) and v>=0 for v in (load,pixels)),'Invalid load timing')
    phase_seconds={phase:max(row['complete_batch_seconds'] for row in observations if row['phase']==phase) for phase in PHASES}
    need(all(math.isfinite(v) and v>0 for v in phase_seconds.values()),'Nonpositive measured phase timing')
    result=[]
    for shard in plan['shards']:
        counts=Counter(plan['features'][fid]['phase'] for fid in shard)
        result.append(load+pixels+1.5*sum(math.ceil(counts[p]/64)*phase_seconds[p] for p in PHASES)+30)
    return result


def verify_software(path,plan):
    from scripts import profile_native_learned_selection as software
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path
    summary=software.verify_profile(path)
    need(summary==read(path),'Software verifier must return its unchanged saved summary')
    for name,h in summary['source_sha256'].items():need(sha(REPO/name)==h,'Software verifier/runtime source changed')
    need(summary['passed'] is True and summary['completed'] is True
         and summary['native_identity']==plan['native_identity']
         and summary['native_identity_sha256']==plan['native_identity_sha256'],'Passed new native software/backbone identity required')
    return dict(file=str(path),sha256=sha(path),source_sha256=summary['source_sha256'],
                native_identity_sha256=summary['native_identity_sha256'])


def check_saved_metrics(torch,raw,observation):
    need(raw['states'].shape==(observation['batch_size'],3584) and raw['native_logits'].shape==(observation['batch_size'],152064),
         'Full native hidden/vocabulary shape differs')
    metrics=replay_metrics(torch,raw['native_logits'],raw['replayed_logits'])
    archived=observation['replay']
    need(metrics['passed'] and archived['passed'] is True and metrics['top1_equal']==archived['top1_equal']
         and len(metrics['tv'])==len(archived['tv']),'Native replay failed')
    need(all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(metrics['tv'],archived['tv'])),
         'GPU/CPU FP64 replay probabilities differ beyond roundoff')
    need(tensor_info(raw['states'])==observation['states'] and raw['states'].dtype==torch.float16
         and raw['feature_ids']==observation['feature_ids'] and raw['input_identity']==observation['input_identity'],'Profile captured states/owners/inputs differ')
    return metrics


def verify_profile(path,plan_path,plan,*,verify_tensors=False,ancestors=True):
    directory=Path(path).resolve();directory=directory.parent if directory.is_file() else directory
    summary=read(directory/'summary.json')
    need(summary['profile'] is True and summary['passed'] is True and summary['completed'] is True
         and summary['source_sha256']==plan['source_sha256'] and summary['plan_sha256']==sha(plan_path)
         and summary['native_identity']==plan['native_identity'] and summary['native_identity_sha256']==plan['native_identity_sha256'],
         'Require exact passed feature profile')
    if ancestors:need(summary['software']==verify_software(summary['software']['file'],plan),'New software release changed')
    else:need(sha(summary['software']['file'])==summary['software']['sha256'],'Bound new software summary changed')
    for name,h in plan['source_sha256'].items():need(sha(directory/'source'/name.replace('/','_'))==h,'Profile source snapshot differs')
    need(sha(summary['observations_file'])==summary['observations_sha256'],'Profile observation index changed')
    observations=read(summary['observations_file'])
    phases=[row for row in observations if row['route']=='feature']
    native=[row for row in observations if row['route']=='native']
    need([(r['phase'],r['case']) for r in phases]==[(p,k) for p in PHASES for k in ('small','stress64')]
         and [r['case'] for r in native]==[r['case_id'] for r in plan['native_cases']],'Fixed profile order/coverage differs')
    for row in phases:
        base=plan['profile_groups'][row['phase']];expected=base if row['case']=='small' else [base[i%len(base)] for i in range(64)]
        need(row['feature_ids']==expected and row['batch_size']==len(expected),'Feature profile input coverage differs')
    for row,case in zip(native,plan['native_cases']):
        need(row['feature_ids']==case['feature_ids'] and row['batch_size']==17
             and row['input_identity']==plan['native_layouts'][case['case_id']]['input_identity'],'Actual native profile input coverage differs')
    need(sum(r['batch_size'] for r in observations)==434 and summary['calls']==dict(model=16,vision=12,language=16,norm=16,head=16,standalone_heads=16),
         'Exact feature/native profile call inventory differs')
    for row in observations:
        if ancestors or verify_tensors:need(sha(row['raw_file'])==row['raw_sha256'],'Profile raw archive changed')
        m=row['replay'];need(m['passed'] is True and len(m['tv'])==len(m['top1_equal'])==row['batch_size']
            and all(type(x) is bool and x for x in m['top1_equal']) and all(math.isfinite(x) and 0<=x<=.02 for x in m['tv']), 'Native head binding failed')
    need(summary['shard_projected_seconds']==projections(plan,phases,summary['model_load_seconds'],summary['pixel_load_verify_seconds'])
         and max(summary['shard_projected_seconds'])<=SHARD_CAP and summary['seconds']<=PROFILE_CAP,'Feature profile resource projection failed')
    model_copy=summary['native_model_file'];need(Path(model_copy).resolve().is_relative_to(CKPT.resolve())
         and sha(model_copy)==summary['native_model_sha256'],'Native model tensor copy changed or is outside checkpoint root')
    if verify_tensors:
        import torch
        for row in observations:
            raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True);check_saved_metrics(torch,raw,row)
            if row['route']=='native':need(raw['metadata']['layout']==plan['native_layouts'][row['case']]['layout'],'Actual native profile layout differs')
        weights=torch.load(model_copy,map_location='cpu',weights_only=True)
        need(weights['native_identity']==plan['native_identity'] and tensor_info(weights['norm_weight'])==plan['native_identity']['norm_weight']
             and tensor_info(weights['head_weight'])==plan['native_identity']['head_weight'],'Saved actual native FP16 model weights differ')
    return summary


def gpu(args,out,frozen):
    import torch,transformers
    from scripts import native_learned_selection_runtime as runtime
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    torch.set_num_threads(4);start=time.perf_counter();plan=verify_plan(args.plan,pixels=True)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One registered B200 required')
    if args.profile:
        need(args.software_summary is not None,'New native software proof is required before feature GPU profile')
        software=verify_software(args.software_summary,plan);reference=None
    else:
        need(args.shard is not None and 0<=args.shard<4 and args.profile_directory is not None,'Shard0..3 and passed feature profile required')
        reference=verify_profile(args.profile_directory,args.plan,plan);software=reference['software']
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    config=dict(schema_version=1,protocol=PROTOCOL,profile=args.profile,shard=args.shard,plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),
        source_sha256=frozen,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],software=software,
        profile_directory=None if reference is None else str(Path(args.profile_directory).resolve()),
        profile_summary_sha256=None if reference is None else sha(Path(args.profile_directory)/'summary.json'),
        slurm_job_id=os.environ['SLURM_JOB_ID'],profile_seconds_cap=PROFILE_CAP,shard_seconds_cap=SHARD_CAP)
    save(out/'config.json',config)
    tick=time.perf_counter();blob=torch.load(plan['pixels_file'],map_location='cpu',weights_only=True);pixels=blob['pixels']
    need(blob['schema_version']==1 and set(pixels)==set(plan['pixel_info']),'Compact pixel inventory differs')
    for key,value in pixels.items():need(tensor_info(value)==plan['pixel_info'][key],'Processed pixel tensor changed')
    native_bundles=torch.load(plan['native_bundles_file'],map_location='cpu',weights_only=True)['bundles'] if args.profile else None
    pixel_seconds=time.perf_counter()-tick
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model;model.eval();model.requires_grad_(False);norm=model.model.language_model.norm;head=model.lm_head
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and runtime_identity()==plan['runtime'],'Loaded native processor/runtime differs')
    _,_,api=native_api(loaded.processor);need(api==plan['native_api'],'Loaded native API differs')
    need(tensor_info(norm.weight)==plan['native_identity']['norm_weight'] and tensor_info(head.weight)==plan['native_identity']['head_weight']
         and norm.variance_epsilon==plan['native_identity']['rms_norm_eps'],'Actual FP16 native norm/head identity differs')
    need(norm.weight.dtype==head.weight.dtype==torch.float16 and not any(p.requires_grad for p in model.parameters()),'Frozen native FP16 model required')
    native_file=None
    if args.profile:
        checkpoint_dir=CKPT/out.name;checkpoint_dir.mkdir(parents=True,exist_ok=False);native_file=checkpoint_dir/'native_norm_head.pt'
        torch.save(dict(schema_version=1,native_identity=plan['native_identity'],norm_weight=norm.weight.detach().cpu().clone(),
            head_weight=head.weight.detach().cpu().clone(),rms_norm_eps=norm.variance_epsilon),native_file)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    hardware=dict(gpu=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if reference is not None:need(hardware==reference['hardware'],'Shard native hardware/backend differs from measured profile')
    versions={name:p._version for name,p in model.named_parameters()};observations=[];counts=dict(model=0,vision=0,language=0,norm=0,head=0,standalone_heads=0)
    state_chunks=[];state_ids=[];capture={};active=False;rope=get_rope_index_fn(model);prior_rope=getattr(model.model,'rope_deltas',None)
    def count(name):
        def hook(*_):
            if active:counts[name]+=1
        return hook
    def language(module,args,kwargs):
        if active:
            counts['language']+=1;capture['positions']=kwargs['position_ids'].detach().cpu().clone();capture['mask']=kwargs['attention_mask'].detach().cpu().clone()
    def hidden(module,args):
        if active:counts['norm']+=1;capture['hidden']=args[0][:,-1:,:].detach().clone()
    def execute(ids,phase,case):
        nonlocal active
        begin=time.perf_counter();inputs=pack(torch,[row_inputs(torch,plan,pixels,fid) for fid in ids],plan['pad_token_id'])
        positions,deltas=rope(input_ids=inputs['input_ids'],image_grid_thw=inputs.get('image_grid_thw'),attention_mask=inputs['attention_mask'])
        for i,fid in enumerate(ids):
            layout=plan['layouts'][plan['features'][fid]['layout_id']];length=len(layout['input_ids'])
            need(torch.equal(positions[:,i,-length:],torch.tensor(layout['position_ids'])[:,0]),'Packed actual valid mRoPE differs')
        identity={key:tensor_info(value) for key,value in inputs.items()};capture.clear();before=dict(counts);active=True
        with torch.inference_mode():output=model(**move_to_device(inputs,model.device),use_cache=False,logits_to_keep=1)
        active=False;capture['native_logits']=output.logits.detach().clone();torch.cuda.synchronize()
        expected=dict(model=1,vision=int(phase.startswith('local')),language=1,norm=1,head=1,standalone_heads=0)
        need({k:counts[k]-before[k] for k in counts}==expected,'Actual feature model call inventory differs')
        need(torch.equal(capture['positions'],positions) and torch.equal(capture['mask'],inputs['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(),deltas),'Actual feature positions/mask/rope differ')
        h=capture['hidden'];need(h.shape==(len(ids),1,3584) and h.dtype==torch.float16 and bool(torch.isfinite(h).all())
            and output.logits.shape==(len(ids),1,152064) and output.logits.dtype==torch.float16 and output.past_key_values is None,'Native feature shape/dtype differs')
        states=h[:,0].cpu().clone();raw=dict(schema_version=1,feature_ids=ids,states=states,input_identity=identity,
            position_ids=tensor_info(positions),rope_deltas=deltas.tolist())
        observation=dict(route='feature',phase=phase,case=case,feature_ids=ids,batch_size=len(ids),input_identity=identity,
            states=tensor_info(states),position_ids=tensor_info(positions),rope_deltas=deltas.tolist())
        if args.profile:
            with torch.inference_mode():replayed=head.forward(norm.forward(h))[:,0]
            counts['standalone_heads']+=1;native_logits=output.logits[:,0]
            raw.update(native_logits=native_logits.cpu().clone(),replayed_logits=replayed.cpu().clone())
            observation['replay']=replay_metrics(torch,native_logits,replayed)
        path=data/f'feature_{len(observations):04d}.pt';torch.save(raw,path)
        observation.update(raw_file=str(path),raw_sha256=sha(path),complete_batch_seconds=time.perf_counter()-begin)
        observations.append(observation)
        if not args.profile:state_ids.extend(ids);state_chunks.append(states)
        save(out/f'progress_{len(observations):04d}.json',dict(observations=observations,calls=counts))
        print(f'{phase} {case}: {len(ids)} rows {observation["complete_batch_seconds"]:.3f}s',flush=True)
    failure=None
    try:
        with ExitStack() as stack:
            for handle in (model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('vision')),
                model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),norm.register_forward_pre_hook(hidden),head.register_forward_pre_hook(count('head'))):
                stack.callback(handle.remove)
            if args.profile:
                for phase in PHASES:
                    ids=plan['profile_groups'][phase];execute(ids,phase,'small');execute([ids[i%len(ids)] for i in range(64)],phase,'stress64')
                for case in plan['native_cases']:
                    begin=time.perf_counter();bundle=runtime.append_observed_prefix(native_bundles[case['sid']],case['prefix_ids'])
                    active=False
                    result=runtime.forward_native(model,bundle,native_identity_sha256=plan['native_identity_sha256'],capture=True,cpu=False)
                    need(result['counters']==dict(model=1,visual=1,language=1,norm=1,head=1,selection=0,probe_head=0),'Actual full native prefix call differs')
                    for key in ('model','language','norm','head'):counts[key]+=1
                    counts['vision']+=1
                    h=result['native_query_hidden'];need(h.shape==(17,1,3584) and h.dtype==torch.float16,'Native fullbatch query capture differs')
                    with torch.inference_mode():replayed=head.forward(norm.forward(h))[:,0]
                    counts['standalone_heads']+=1;native_logits=result['native_logits'][:,0]
                    states=h[:,0].detach().cpu().clone();identity={k:tensor_info(v) for k,v in bundle['inputs'].items()}
                    need(identity==plan['native_layouts'][case['case_id']]['input_identity'],'Executed native profile inputs differ from CPU')
                    raw=dict(schema_version=1,feature_ids=case['feature_ids'],states=states,native_logits=native_logits.cpu().clone(),
                        replayed_logits=replayed.cpu().clone(),metadata=result['metadata'],input_identity=identity)
                    path=data/f'native_{case["case_id"]}.pt';torch.save(raw,path)
                    observation=dict(route='native',phase='native',case=case['case_id'],feature_ids=case['feature_ids'],batch_size=17,
                        states=tensor_info(states),input_identity=identity,replay=replay_metrics(torch,native_logits,replayed),
                        raw_file=str(path),raw_sha256=sha(path),complete_batch_seconds=time.perf_counter()-begin)
                    observations.append(observation);save(out/f'progress_{len(observations):04d}.json',dict(observations=observations,calls=counts))
            else:
                assigned=set(plan['shards'][args.shard])
                for phase in PHASES:
                    ids=[fid for fid in plan['groups'][phase] if fid in assigned]
                    for offset in range(0,len(ids),64):execute(ids[offset:offset+64],phase,f'batch_{offset//64:04d}')
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
        torch.save({key:value.detach().cpu().clone() for key,value in capture.items()},data/'failure_capture.pt')
        raise
    finally:
        active=False;model.model.rope_deltas=prior_rope
        save(out/'observations.json',observations)
        save(out/'attempt_status.json',dict(failed=failure is not None,failure=failure,calls=counts,seconds=time.perf_counter()-start,
            native_versions_unchanged=versions=={name:p._version for name,p in model.named_parameters()}))
    need(versions=={name:p._version for name,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen native model changed')
    need(sources()==frozen and sha(args.plan)==config['plan_sha256'],'Sources/plan changed during extraction')
    result=dict(config,completed=True,calls=counts,model_load_seconds=load_seconds,pixel_load_verify_seconds=pixel_seconds,
        observations_file=str(out/'observations.json'),observations_sha256=sha(out/'observations.json'),
        hardware=hardware,
        frozen_native_unchanged=True,peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start)
    if args.profile:
        phases=[r for r in observations if r['route']=='feature'];projected=projections(plan,phases,load_seconds,pixel_seconds)
        numerical=all(r['replay']['passed'] for r in observations)
        need(counts==dict(model=16,vision=12,language=16,norm=16,head=16,standalone_heads=16)
             and sum(r['batch_size'] for r in observations)==434,'Registered profile inventory differs')
        result.update(native_model_file=str(native_file),native_model_sha256=sha(native_file),shard_projected_seconds=projected,
            numerical_gate_passed=numerical,runtime_projection_passed=max(projected)<=SHARD_CAP,
            passed=numerical and max(projected)<=SHARD_CAP and result['seconds']<=PROFILE_CAP,
            head_binding_rows=434,cached_native_hidden_comparison='Descriptive only, computed at complete merge',no_accuracy_score=True)
    else:
        need(state_ids==plan['shards'][args.shard] and len(set(state_ids))==len(state_ids),'Shard ownership/order differs')
        states=torch.cat(state_chunks);path=data/'features.pt';torch.save(dict(schema_version=1,feature_ids=state_ids,states=states),path)
        result.update(features_file=str(path),features_sha256=sha(path),feature_count=len(state_ids),states=tensor_info(states),
            feature_ids_sha256=oid(state_ids),feature_state_sha256={fid:tensor_info(row)['sha256'] for fid,row in zip(state_ids,states)})
        result['seconds']=time.perf_counter()-start;result['passed']=result['seconds']<=SHARD_CAP
    result['seconds']=time.perf_counter()-start
    result['passed']=result['passed'] and result['seconds']<=(PROFILE_CAP if args.profile else SHARD_CAP)
    return result


def selftest():
    fake=dict(shards=[['a']*65,[],[],[]],features={'a':dict(phase='local_empty')})
    observations=[dict(phase=p,complete_batch_seconds=t) for p in PHASES for t in (2.,1.)]
    need(projections(fake,observations,3.,4.)==[43.,37.,37.,37.],'Projection must charge the largest cold/small/stress cost to every batch')
    need(all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in ((0.,1e-16),(.001,.001+1e-15))), 'Probability roundoff comparison differs')
    return dict(passed=True,tests=['complete_batch_projection','empty_shard_load_floor','FP64_probability_roundoff'])


def verify_cache(path,*,verify_tensors=False,ancestors=True,require_completed=True):
    path=Path(path).resolve();cache=read(path)
    need(cache['protocol']=='identity_join_learned_training_feature_cache' and cache['complete'] is True and cache['training_only'] is True
         and cache['source_sha256']==sources(),'Complete new training feature cache required')
    if require_completed:
        summary=read(OUT/path.parent.name/'summary.json')
        need(summary['passed'] is True and summary['completed'] is True and summary['cache_file']==str(path)
             and summary['cache_sha256']==sha(path) and summary['source_sha256']==cache['source_sha256'],'Completed exact CPU merge required')
    need(sha(cache['plan_file'])==cache['plan_sha256'],'Feature plan changed');plan=verify_plan(cache['plan_file'],pixels=False,ancestors=ancestors)
    need(cache['scenes']==plan['scenes'] and cache['training_pairs']==plan['training_pairs'] and set(cache['features'])==set(plan['features'])
         and cache['native_identity']==plan['native_identity'] and cache['native_identity_sha256']==plan['native_identity_sha256'], 'Cache scene/feature/native identity differs')
    profile=verify_profile(cache['profile_directory'],cache['plan_file'],plan,verify_tensors=verify_tensors,ancestors=ancestors)
    need(cache['profile_summary_sha256']==sha(Path(cache['profile_directory'])/'summary.json') and cache['native_model_file']==profile['native_model_file']
         and cache['native_model_sha256']==profile['native_model_sha256'],'Native profile/model copy binding differs')
    need(sha(cache['cached_native_hidden_file'])==cache['cached_native_hidden_sha256'],'Descriptive native/cache evidence changed')
    comparisons=read(cache['cached_native_hidden_file'])
    need(comparisons['descriptive_only'] is True and comparisons['head_not_reexecuted'] is True and len(comparisons['rows'])==136,
         'Descriptive native/cache coverage differs')
    all_ids=[]
    for item in cache['shards']:
        summary=read(Path(item['directory'])/'summary.json')
        need(sha(Path(item['directory'])/'summary.json')==item['summary_sha256'] and summary['passed'] is True and summary['completed'] is True
             and summary['source_sha256']==plan['source_sha256'] and summary['plan_sha256']==cache['plan_sha256']
             and summary['profile_summary_sha256']==cache['profile_summary_sha256'],'Shard provenance differs')
        expected=plan['shards'][item['shard']]
        phase_counts=Counter(plan['features'][fid]['phase'] for fid in expected)
        calls=sum(math.ceil(phase_counts[p]/64) for p in PHASES);vision=sum(math.ceil(phase_counts[p]/64) for p in PHASES if p.startswith('local'))
        need(summary['calls']==dict(model=calls,vision=vision,language=calls,norm=calls,head=calls,standalone_heads=0)
             and summary['hardware']==profile['hardware'] and summary['seconds']<=SHARD_CAP,'Harvest call/backend/resource inventory differs')
        need(summary['shard']==item['shard'] and summary['feature_count']==len(expected)
            and summary['feature_ids_sha256']==oid(expected) and sha(summary['features_file'])==summary['features_sha256'],'Shard data inventory differs')
        for row,fid in enumerate(expected):
            need(cache['features'][fid]==dict(file=summary['features_file'],file_sha256=summary['features_sha256'],row=row,
                state_sha256=summary['feature_state_sha256'][fid]),'Exact cache pointer differs')
        all_ids.extend(expected)
        if verify_tensors:
            import torch
            blob=torch.load(summary['features_file'],map_location='cpu',weights_only=True)
            need(blob['feature_ids']==expected and tensor_info(blob['states'])==summary['states'] and blob['states'].dtype==torch.float16
                 and blob['states'].shape==(len(expected),3584) and bool(torch.isfinite(blob['states']).all()),'Shard actual tensor differs')
            for fid,row in zip(expected,blob['states']):need(tensor_info(row)['sha256']==summary['feature_state_sha256'][fid],'Actual per-feature state differs')
    need(len(cache['shards'])==4 and sorted(item['shard'] for item in cache['shards'])==list(range(4))
         and len(set(all_ids))==len(all_ids)==len(plan['features']),'Complete unique cache coverage differs')
    return cache


def merge(args,out,frozen):
    import torch
    torch.set_num_threads(4);tests=selftest();plan=verify_plan(args.plan,pixels=True)
    need(args.profile_directory is not None and len(args.shard_directories)==4,'Exact profile and four shards required')
    profile=verify_profile(args.profile_directory,args.plan,plan,verify_tensors=True);features={};shards=[];native_states={}
    observations=read(profile['observations_file'])
    wanted={fid for row in observations if row['route']=='native' for fid in row['feature_ids']}
    seen=set()
    for directory in args.shard_directories:
        directory=Path(directory).resolve();summary=read(directory/'summary.json');index=summary['shard']
        need(type(index) is int and index in range(4) and index not in seen,'Duplicate/invalid shard');seen.add(index)
        need(summary['passed'] is True and summary['completed'] is True and summary['source_sha256']==frozen and summary['plan_sha256']==sha(args.plan)
             and summary['profile_summary_sha256']==sha(Path(args.profile_directory)/'summary.json') and summary['seconds']<=SHARD_CAP,'Shard release/source/resource differs')
        need(sha(summary['features_file'])==summary['features_sha256'] and sha(summary['observations_file'])==summary['observations_sha256'],'Shard actual archive changed')
        blob=torch.load(summary['features_file'],map_location='cpu',weights_only=True);ids=plan['shards'][index]
        need(blob['feature_ids']==ids and blob['states'].shape==(len(ids),3584) and blob['states'].dtype==torch.float16
             and tensor_info(blob['states'])==summary['states'],'Shard native FP16 state coverage differs')
        observed=[]
        for batch in read(summary['observations_file']):
            need(batch['route']=='feature' and sha(batch['raw_file'])==batch['raw_sha256'],'Harvest batch evidence changed')
            raw=torch.load(batch['raw_file'],map_location='cpu',weights_only=True)
            start=len(observed);n=len(batch['feature_ids'])
            need(raw['feature_ids']==batch['feature_ids']==ids[start:start+n]
                 and torch.equal(raw['states'],blob['states'][start:start+n]) and tensor_info(raw['states'])==batch['states'], 'Saved batch/final state identity differs')
            observed.extend(raw['feature_ids'])
        need(observed==ids and summary['feature_ids_sha256']==oid(ids),'Complete raw harvest ownership differs')
        for row,fid in enumerate(ids):
            value=blob['states'][row];h=tensor_info(value)['sha256'];need(h==summary['feature_state_sha256'][fid],'Exact per-state native hash differs')
            features[fid]=dict(file=summary['features_file'],file_sha256=summary['features_sha256'],row=row,state_sha256=h)
            if fid in wanted:native_states[fid]=value.clone()
        shards.append(dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),shard=index))
    need(set(features)==set(plan['features']) and len(features)==sum(len(x) for x in plan['shards']),'Union cache incomplete')
    comparisons=[]
    for observation in observations:
        if observation['route']!='native':continue
        raw=torch.load(observation['raw_file'],map_location='cpu',weights_only=True)
        for fid,native in zip(raw['feature_ids'],raw['states']):
            diff=native_states[fid].double()-native.double()
            comparisons.append(dict(case=observation['case'],feature_id=fid,maximum_absolute=float(diff.abs().max()),
                rms=float(diff.square().mean().sqrt()),bitwise_equal=torch.equal(native_states[fid],native)))
    need(len(comparisons)==136,'Native/cache batch-route comparison coverage differs')
    save(out/'cached_native_hidden.json',dict(descriptive_only=True,rows=comparisons,head_not_reexecuted=True))
    cache=dict(schema_version=1,protocol='identity_join_learned_training_feature_cache',complete=True,training_only=True,
        source_sha256=frozen,plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),features=features,scenes=plan['scenes'],
        training_pairs=plan['training_pairs'],native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_model_file=profile['native_model_file'],native_model_sha256=profile['native_model_sha256'],
        profile_directory=str(Path(args.profile_directory).resolve()),profile_summary_sha256=sha(Path(args.profile_directory)/'summary.json'),
        shards=sorted(shards,key=lambda row:row['shard']),state_dtype='torch.float16',read_boundary=plan['read_boundary'],
        cached_native_hidden_file=str(out/'cached_native_hidden.json'),cached_native_hidden_sha256=sha(out/'cached_native_hidden.json'),
        no_binary_measurement=True,no_local_labels=True,no_feature_reuse=True)
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);path=data/'feature_cache.json';save(path,cache)
    verify_cache(path,verify_tensors=False,require_completed=False)
    return dict(passed=True,completed=True,source_sha256=frozen,cache_file=str(path),cache_sha256=sha(path),feature_count=len(features),
        training_scene_count=6048,training_target_positions=13440,head_binding_rows=434,cached_native_hidden_rows=136,
        cached_native_hidden_differences_descriptive=True,no_binary_measurement=True,no_local_labels=True,tests=tests)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--profile',action='store_true');group.add_argument('--shard',type=int);group.add_argument('--merge',action='store_true')
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--software-summary',type=Path)
    parser.add_argument('--profile-directory',type=Path);parser.add_argument('--shard-directories',type=Path,nargs='*',default=[]);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Slurm with4CPUcores required')
    if args.merge:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'Merge must run on Slurm CPU')
    else:need(os.environ.get('SLURM_JOB_PARTITION')=='gpu','Feature extraction must run on Slurm GPU')
    label='merge' if args.merge else 'profile' if args.profile else f'shard{args.shard}'
    out=OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);start=time.perf_counter()
    save(out/'request.json',dict(plan=str(args.plan),profile=args.profile,shard=args.shard,merge=args.merge,source_sha256=frozen))
    try:
        result=merge(args,out,frozen) if args.merge else gpu(args,out,frozen)
        need(sources()==frozen,'Sources changed during execution');result['elapsed_including_finalization_seconds']=time.perf_counter()-start
        save(out/'summary.json',result)
        if not result['passed']:raise RuntimeError('Feature profile/shard gate failed; all raw evidence retained')
        print(str(out),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
