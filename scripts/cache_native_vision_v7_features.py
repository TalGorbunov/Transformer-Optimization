"""Native FP16 pre-final-norm feature harvesting; Slurm GPU or CPU merge only.

No trained branch, no count prediction quality gate, no test/development data.
A fixed profile validates native head replay and projects four bounded shards.
"""
from __future__ import annotations
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time
import json
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v7_features import (
    DATA,OUT,MODEL,PHASES,need,read,save,sha,object_sha,sources,snapshot,index,
    tensor_info,verify_plan,row_inputs,pack,runtime_identity,native_api)


def verify_profile(directory,plan,path):
    directory=Path(directory)
    need(directory.is_relative_to(OUT),'Profile directory is outside V7 feature outputs')
    result=read(directory/'summary.json')
    need(result['profile'] is True and result['completed'] is True and result['passed'] is True
         and result['plan_sha256']==sha(path) and result['source_sha256']==plan['source_sha256'],
         'Require a completed passing exact-plan profile')
    need(sha(result['raw_file'])==result['raw_sha256'] and sha(result['observations_file'])==result['observations_sha256'],
         'Profile raw evidence changed')
    return result


def replay_metrics(torch,a,b):
    a=a.double();b=b.double()
    need(a.shape==b.shape and a.ndim==2 and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
         'Nonfinite/malformed native replay logits')
    tv=.5*(torch.softmax(a,-1)-torch.softmax(b,-1)).abs().sum(-1)
    same=a.argmax(-1)==b.argmax(-1)
    return dict(tv=tv.cpu().tolist(),top1_equal=same.cpu().tolist(),
                maximum_tv=float(tv.max()),all_top1_equal=bool(same.all()),
                passed=bool((tv<=.02).all() and same.all()))


def gpu(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);start=time.perf_counter();plan=verify_plan(args.plan,pixels=True)
    profile=args.profile
    need(profile != (args.shard is not None),'Choose exactly one profile or shard')
    reference=None if profile else verify_profile(args.profile_directory,plan,args.plan)
    if not profile:need(0<=args.shard<4,'Shard must be 0..3')
    job=os.environ['SLURM_JOB_ID'];name=f'profile_{job}' if profile else f'shard{args.shard}_{job}'
    out=OUT/name;out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/name;data.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    pixel_start=time.perf_counter()
    blob=torch.load(plan['pixels_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==1 and set(blob['pixels'])==set(plan['pixel_info']),'Pixel coverage differs')
    pixels=blob['pixels']
    for key,value in pixels.items():need(tensor_info(value)==plan['pixel_info'][key],'Processed pixel tensor changed')
    pixel_seconds=time.perf_counter()-pixel_start
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
    config=dict(schema_version=1,profile=profile,shard=args.shard,plan_file=str(Path(args.plan).resolve()),
                plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],model=plan['model'],
                runtime=plan['runtime'],processor=plan['processor'],hardware=hardware,native_dtypes=dtypes,
                profile_directory=None if profile else str(Path(args.profile_directory).resolve()),
                profile_summary_sha256=None if profile else sha(Path(args.profile_directory)/'summary.json'),
                slurm_job_id=job)
    save(out/'config.json',config)
    index(out,'V7 native training-feature harvest',[('Configuration','config.json'),('Batch observations','observations.json'),
          ('Summary after completion','summary.json')])
    captured={};counts=dict(model=0,vision=0,language=0,norm=0);recording=False
    def model_hook(*_):counts['model']+=1
    def visual_hook(*_):counts['vision']+=1
    def language_hook(module,args,kwargs):
        counts['language']+=1
        captured['positions']=kwargs['position_ids'].detach().cpu().clone()
        captured['mask']=kwargs['attention_mask'].detach().cpu().clone()
    def norm_hook(module,args):
        if recording:
            counts['norm']+=1;captured['hidden']=args[0][:,-1,:].detach().clone()
    handles=[model.register_forward_pre_hook(model_hook),model.model.visual.register_forward_pre_hook(visual_hook),
             language.register_forward_pre_hook(language_hook,with_kwargs=True),norm.register_forward_pre_hook(norm_hook)]
    rope=get_rope_index_fn(model);observations=[];raw=[];state_chunks=[];state_ids=[];small_states={}
    def batch(ids,phase,kind):
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
             dict(model=1,vision=int(phase.startswith('local')),language=1,norm=1),'Native forward counts differ')
        need(torch.equal(captured['positions'],expected_pos) and torch.equal(captured['mask'],inputs['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(),expected_delta),'Executed native positions/mask differ')
        h=captured['hidden']
        need(h.shape==(len(ids),3584) and h.dtype==torch.float16 and bool(torch.isfinite(h).all())
             and output.logits.shape[:2]==(len(ids),1) and output.past_key_values is None,'Native feature/forward contract differs')
        states=h.cpu().clone();harvest_seconds=time.perf_counter()-begin
        observation=dict(phase=phase,case=kind,batch_size=len(ids),identity=identity,
                         preprocessing_seconds=prepared_at-begin,forward_seconds=forward_seconds,
                         harvest_seconds=harvest_seconds,states=tensor_info(states),
                         native_model_forwards=1,native_visual_forwards=int(phase.startswith('local')))
        if profile:
            with torch.inference_mode():replayed=model.lm_head(norm(h.unsqueeze(1)))[:,0]
            native_logits=output.logits[:,0];metrics=replay_metrics(torch,native_logits,replayed)
            observation['replay']=metrics
            if kind=='small':
                for fid,value in zip(ids,states):small_states[fid]=value.clone()
            else:
                differences=[float((value.float()-small_states[fid].float()).square().mean().sqrt())
                             for fid,value in zip(ids,states)]
                observation['small_vs_stress_hidden_rms']=differences
                observation['small_vs_stress_is_descriptive']=True
            raw.append(dict(phase=phase,case=kind,feature_ids=ids,states=states,
                            native_logits=native_logits.cpu(),replayed_logits=replayed.cpu()))
        else:
            state_ids.extend(ids);state_chunks.append(states)
        observations.append(observation)
        print(json.dumps(dict(phase=phase,case=kind,n=len(ids),seconds=harvest_seconds)),flush=True)
    try:
        if profile:
            for phase in PHASES:
                ids=plan['profile_groups'][phase]
                batch(ids,phase,'small');batch([ids[i%len(ids)] for i in range(64)],phase,'stress64')
        else:
            assigned=set(plan['shards'][args.shard])
            for phase in PHASES:
                ids=[fid for fid in plan['groups'][phase] if fid in assigned]
                for offset in range(0,len(ids),64):batch(ids[offset:offset+64],phase,f'batch_{offset//64:04d}')
    finally:
        recording=False
        for handle in handles:handle.remove()
    need(verify_plan(args.plan,pixels=False)==plan,'Plan/source changed during harvesting')
    save(out/'observations.json',observations)
    summary=dict(config,completed=True,computational_integrity_passed=True,model_load_seconds=load_seconds,
                 pixel_load_verify_seconds=pixel_seconds,observations_file=str(out/'observations.json'),
                 observations_sha256=sha(out/'observations.json'),total_seconds=time.perf_counter()-start,
                 peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    if profile:
        need(len(observations)==8 and counts['model']==8 and counts['vision']==4,'Profile call coverage differs')
        raw_path=data/'raw.pt';torch.save(dict(schema_version=1,observations=raw),raw_path)
        stress={row['phase']:row['harvest_seconds'] for row in observations if row['case']=='stress64'}
        projections=[]
        for shard in plan['shards']:
            histogram=Counter(plan['features'][fid]['phase'] for fid in shard)
            projections.append(load_seconds+pixel_seconds+1.5*sum(math.ceil(histogram[p]/64)*stress[p] for p in PHASES)+30)
        numerical=all(row['replay']['passed'] for row in observations)
        summary.update(raw_file=str(raw_path),raw_sha256=sha(raw_path),numerical_gate_passed=numerical,
                       shard_projected_seconds=projections,runtime_projection_passed=max(projections)<=300,
                       profile_feature_count=32,passed=numerical and max(projections)<=300,
                       no_quality_or_accuracy_gate=True)
    else:
        need(state_ids==plan['shards'][args.shard] and len(set(state_ids))==len(state_ids),'Shard feature order/coverage differs')
        states=torch.cat(state_chunks);path=data/'features.pt'
        torch.save(dict(schema_version=1,feature_ids=state_ids,states=states),path)
        summary.update(passed=True,feature_count=len(state_ids),features_file=str(path),features_sha256=sha(path),
                       states=tensor_info(states),feature_ids_sha256=object_sha(state_ids),
                       feature_state_sha256={fid:tensor_info(row)['sha256'] for fid,row in zip(state_ids,states)})
    summary['total_seconds']=time.perf_counter()-start
    save(out/'summary.json',summary)
    index(data,'V7 feature tensors',[('Profile raw tensors' if profile else 'Native feature states',
                                   'raw.pt' if profile else 'features.pt')])
    print(json.dumps(summary,indent=2),flush=True)
    if not summary['passed']:raise SystemExit('V7 feature profile failed; all raw observations retained')


def merge(args):
    import torch
    start=time.perf_counter();plan=verify_plan(args.plan,pixels=True)
    need(len(args.merge)==4 and len(set(args.merge))==4,'Require four distinct shard output directories')
    summaries=[];locations={};common=None;seen_shards=set()
    for value in args.merge:
        directory=Path(value);summary=read(directory/'summary.json')
        need(directory.is_relative_to(OUT) and summary['profile'] is False and summary['completed']
             and summary['passed'] and summary['computational_integrity_passed']
             and summary['plan_sha256']==sha(args.plan) and summary['source_sha256']==plan['source_sha256'],
             'Incomplete/incompatible shard')
        profile=verify_profile(summary['profile_directory'],plan,args.plan)
        need(sha(Path(summary['profile_directory'])/'summary.json')==summary['profile_summary_sha256'],
             'Bound profile changed')
        invariant={key:summary[key] for key in ('model','runtime','processor','hardware','native_dtypes',
                                                'profile_directory','profile_summary_sha256')}
        if common is None:common=invariant
        need(invariant==common,'Shards used different native runtimes/profiles')
        shard=summary['shard'];need(shard in range(4) and shard not in seen_shards,'Duplicate/invalid shard')
        seen_shards.add(shard)
        need(sha(summary['features_file'])==summary['features_sha256'] and
             sha(summary['observations_file'])==summary['observations_sha256'],'Shard artifact changed')
        blob=torch.load(summary['features_file'],map_location='cpu',weights_only=True);ids=blob['feature_ids'];states=blob['states']
        need(blob['schema_version']==1 and ids==plan['shards'][shard]
             and object_sha(ids)==summary['feature_ids_sha256'] and tensor_info(states)==summary['states']
             and states.shape==(len(ids),3584) and states.dtype==torch.float16 and bool(torch.isfinite(states).all()),
             'Shard tensor/ID contract differs')
        for row,(fid,state) in enumerate(zip(ids,states)):
            digest=tensor_info(state)['sha256']
            need(fid not in locations and digest==summary['feature_state_sha256'][fid],'Feature identity/hash differs')
            locations[fid]=dict(file=summary['features_file'],file_sha256=summary['features_sha256'],row=row,state_sha256=digest)
        summaries.append(dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),shard=shard))
    need(set(locations)==set(plan['features']) and seen_shards==set(range(4)),'Incomplete merged cache')
    result=dict(schema_version=1,protocol=plan['protocol'],complete=True,training_only=True,
                plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
                source_sha256=plan['source_sha256'],teacher_binding=plan['teacher_binding'],
                model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_dtypes=common['native_dtypes'],
                hardware=common['hardware'],feature_count=len(locations),counts=plan['counts'],
                features=locations,scenes=plan['scenes'],shards=summaries,
                profile_directory=common['profile_directory'],profile_summary_sha256=common['profile_summary_sha256'])
    target=DATA/'feature_cache.json';save(target,result);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'merge_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    save(out/'summary.json',dict(passed=True,cache_file=str(target),cache_sha256=sha(target),
                                feature_count=len(locations),counts=plan['counts'],seconds=time.perf_counter()-start))
    index(out,'V7 complete training-feature cache',[('Summary','summary.json'),('Source hashes','source_hashes.json')])
    print(json.dumps(dict(cache_file=str(target),cache_sha256=sha(target),features=len(locations))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--profile',action='store_true');parser.add_argument('--shard',type=int)
    parser.add_argument('--profile-directory',type=Path);parser.add_argument('--merge',type=Path,nargs=4)
    args=parser.parse_args();need(os.environ.get('SLURM_JOB_ID'),'All feature work requires Slurm')
    if args.merge:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
             and not args.profile and args.shard is None,'Merge is CPU only')
        merge(args)
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and os.environ.get('SLURM_JOB_GPUS'),
             'Feature profile/harvest requires one GPU Slurm allocation')
        gpu(args)


if __name__=='__main__':main()

