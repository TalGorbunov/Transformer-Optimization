"""Released full joint-feature harvesting and CPU merge for V7.

This NEW program consumes the immutable joint stage/profile; it never changes
their original profile-only authorization. A separate CPU release freezes this
program and authorizes exactly four <=480-second GPU shards. All features remain
training-only ordinary joint VLM pre-final-norm states; no model is trained here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_vision_v7_joint_features as joint
from scripts import stage_native_vision_v7_features as parallel
from scripts.cache_native_vision_v7_features import replay_metrics
need,read,save,sha,object_sha=joint.need,joint.read,joint.save,joint.sha,joint.object_sha
tensor_info=joint.tensor_info
DATA,OUT=joint.DATA,joint.OUT
OWN=('scripts/harvest_native_vision_v7_joint_features.py',
     'slurm/native_vision_v7_joint_features_release.sbatch',
     'slurm/native_vision_v7_joint_features_array.sbatch',
     'slurm/native_vision_v7_joint_features_merge.sbatch')
WALL_SECONDS=480
BLOCK_GPU_SECONDS=2340
PROFILE_ACCOUNTED_GPU_SECONDS=38


def union_sources(plan):
    result=dict(plan['source_sha256'])
    for name in OWN:
        digest=sha(REPO/name)
        need(name not in result or result[name]==digest,'Conflicting ancestor/new source binding')
        result[name]=digest
    return result


def snapshot(out,ledger):
    (out/'source').mkdir()
    for name,digest in ledger.items():
        need(sha(REPO/name)==digest,'Source changed before snapshot')
        target=out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',ledger)


def profile_binding(directory,plan,plan_file):
    directory=Path(directory).resolve()
    need(directory.is_relative_to(OUT),'Joint profile is outside canonical output root')
    summary_file=directory/'summary.json';summary=read(summary_file)
    need(summary['profile'] is True and summary['completed'] is True and summary['passed'] is True
         and summary['computational_integrity_passed'] is True and summary['numerical_gate_passed'] is True
         and summary['source_sha256']==plan['source_sha256'] and summary['plan_sha256']==sha(plan_file)
         and Path(summary['plan_file']).resolve()==Path(plan_file).resolve(),'Incomplete or incompatible joint profile')
    config_file=directory/'config.json';config=read(config_file)
    for key in config:
        need(summary[key]==config[key],'Profile config/summary mismatch: '+key)
    need(read(directory/'source_hashes.json')==plan['source_sha256'],'Profile source ledger changed')
    for name,digest in plan['source_sha256'].items():
        need(sha(directory/'source'/name.replace('/','_'))==digest,'Profile frozen source snapshot changed')
    files=dict(summary=dict(path=str(summary_file),sha256=sha(summary_file)),
               config=dict(path=str(config_file),sha256=sha(config_file)),
               observations=dict(path=summary['observations_file'],sha256=summary['observations_sha256']),
               raw=dict(path=summary['raw_file'],sha256=summary['raw_sha256']))
    for value in files.values():need(sha(value['path'])==value['sha256'],'Joint profile artifact changed')
    return summary,files


def check_profile_raw(torch,profile,plan):
    observations=read(profile['observations_file'])
    raw=torch.load(profile['raw_file'],map_location='cpu',weights_only=True)
    need(raw['schema_version']==1 and len(raw['observations'])==len(observations)==8,'Profile raw call count differs')
    stress={};total=0
    for index,(observed,values) in enumerate(zip(observations,raw['observations'])):
        group=joint.GROUPS[index//2];case='B1_first' if index%2==0 else 'B4_all'
        expected=plan['profile_groups'][group][:1] if index%2==0 else plan['profile_groups'][group]
        need(observed['group']==values['group']==group and observed['case']==values['case']==case
             and observed['identity']['feature_ids']==values['feature_ids']==expected,
             'Fixed profile input selection/order changed')
        states=values['states'];native=values['native_logits'];replay=values['replayed_logits'];count=len(expected)
        need(states.shape==(count,3584) and states.dtype==torch.float16 and bool(torch.isfinite(states).all())
             and tensor_info(states)==observed['states'] and native.shape[0]==replay.shape[0]==count
             and native.dtype==replay.dtype==torch.float16,'Profile native tensor coverage/dtype differs')
        actual=replay_metrics(torch,native,replay);stored=observed['replay']
        need(actual['passed'] and actual['passed']==stored['passed']
             and actual['top1_equal']==stored['top1_equal'] and actual['all_top1_equal']==stored['all_top1_equal']
             and len(actual['tv'])==len(stored['tv'])
             and all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(actual['tv'],stored['tv']))
             and math.isclose(actual['maximum_tv'],stored['maximum_tv'],rel_tol=1e-10,abs_tol=1e-12),
             'Raw profile does not reproduce the fixed norm/head replay gate')
        need(observed['native_model_forwards']==observed['native_visual_forwards']==1,
             'Profile forward counts differ')
        if case=='B4_all':stress[group]=observed['harvest_seconds']
        total+=count
    need(total==20 and profile['profile_calls']==8 and profile['distinct_feature_rows']==16
         and profile['total_output_rows']==20,'Profile denominator differs')
    projections=[]
    for shard in plan['prospective_shards']:
        histogram=Counter(plan['features'][fid]['phase'] for fid in shard)
        projected=profile['model_load_seconds']+profile['pixel_load_verify_seconds']+1.5*sum(
            math.ceil(histogram[group]/4)*stress[group] for group in joint.GROUPS)+30
        projections.append(projected)
    need(all(math.isfinite(value) and value<=WALL_SECONDS for value in projections)
         and all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
                 for a,b in zip(projections,profile['prospective_shard_projected_seconds'])),
         'Joint profile does not fit the prospective480-second shard cap')
    return dict(passed=True,raw_replay_recomputed=True,profile_calls=8,total_output_rows=20,
                shard_projected_seconds=projections,
                B1_B4_stability='Retained descriptive comparison; never reclassified as a strict replay gate')


def freeze(args):
    import torch
    started=time.monotonic();plan,_=joint.verify_plan(args.plan,pixels=True);ledger=union_sources(plan)
    profile,bindings=profile_binding(args.profile_directory,plan,args.plan)
    checked=check_profile_raw(torch,profile,plan)
    need(PROFILE_ACCOUNTED_GPU_SECONDS+4*WALL_SECONDS<=BLOCK_GPU_SECONDS,'Joint total GPU reservation exceeds envelope')
    for path,digest in plan['native_api']['source_sha256'].items():
        need(sha(path)==digest,'Installed native backend source changed')
    wrapper=(REPO/OWN[2]).read_text()
    need('#SBATCH -t 00:08:00' in wrapper and '#SBATCH --array=0-3%4' in wrapper
         and '#SBATCH --gres=gpu:1' in wrapper,'Joint harvest wrapper allocation differs')
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'harvest_release_{job}';out.mkdir(parents=True,exist_ok=False)
    snapshot(out,ledger)
    release=dict(schema_version=1,protocol='v7_joint_full_training_feature_release',
        passed=True,full_harvest_authorized=True,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        parent_source_sha256=plan['source_sha256'],source_sha256=ledger,profile=bindings,
        profile_directory=str(Path(args.profile_directory).resolve()),profile_checks=checked,
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],
        native_dtypes=profile['native_dtypes'],hardware=profile['hardware'],
        shards=plan['prospective_shards'],feature_count=3780,counts=plan['counts'],batch_size=4,
        wall_seconds_per_shard=WALL_SECONDS,max_parallel_shards=4,block_gpu_seconds=BLOCK_GPU_SECONDS,
        initial_profile_accounted_gpu_seconds=PROFILE_ACCOUNTED_GPU_SECONDS,
        maximum_reserved_gpu_seconds=PROFILE_ACCOUNTED_GPU_SECONDS+4*WALL_SECONDS,
        profile_gpu_accounting_source='Registered Slurm elapsed38s for profile441831, including process overhead',
        scope='New computational/resource release; no count-accuracy eligibility or training',
        slurm_job_id=job,seconds=time.monotonic()-started)
    target=out/'release.json';save(target,release);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    parallel.index(out,'V7 joint full-cache CPU release',[('Release','release.json'),('Frozen source ledger','source_hashes.json')])
    need(union_sources(plan)==ledger,'New harvest source changed during release')
    print(json.dumps(dict(passed=True,release_file=str(target),sha256=sha(target),
                         projections=checked['shard_projected_seconds']),indent=2))


def verify_release(path,*,pixels=False):
    path=Path(path).resolve()
    need(path.is_relative_to(OUT) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Release path/sidecar changed')
    release=read(path);plan,parent=joint.verify_plan(release['plan_file'],pixels=pixels)
    need(release['schema_version']==1 and release['protocol']=='v7_joint_full_training_feature_release'
         and release['passed'] is True and release['full_harvest_authorized'] is True
         and release['plan_sha256']==sha(release['plan_file'])
         and release['source_sha256']==union_sources(plan)
         and release['parent_source_sha256']==plan['source_sha256']
         and release['shards']==plan['prospective_shards'],'Harvest release/source/input changed')
    need(read(path.parent/'source_hashes.json')==release['source_sha256'],'Release source ledger changed')
    for name,digest in release['source_sha256'].items():
        need(sha(path.parent/'source'/name)==digest,'Release frozen source copy changed')
    profile,bindings=profile_binding(release['profile_directory'],plan,release['plan_file'])
    need(bindings==release['profile'] and release['wall_seconds_per_shard']==480
         and release['max_parallel_shards']==4 and release['block_gpu_seconds']==2340
         and release['maximum_reserved_gpu_seconds']==1958
         and max(release['profile_checks']['shard_projected_seconds'])<=480,'Release budget/profile changed')
    for key in ('model','runtime','processor'):
        need(release[key]==plan[key],'Released native identity differs from parent plan: '+key)
    need(release['native_dtypes']==profile['native_dtypes'] and release['hardware']==profile['hardware']
         and release['feature_count']==3780 and release['counts']==plan['counts'] and release['batch_size']==4
         and release['initial_profile_accounted_gpu_seconds']==38,'Released execution contract differs')
    return release,plan,parent


def gpu(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter()
    release,plan,parent=verify_release(args.release,pixels=True)
    need(args.shard in range(4),'Joint shard must be0..3')
    job=os.environ['SLURM_JOB_ID'];name=f'harvest_shard{args.shard}_{job}'
    out=OUT/name;out.mkdir(parents=True,exist_ok=False);snapshot(out,release['source_sha256'])
    data=DATA/name;data.mkdir(parents=True,exist_ok=False)
    (out/'release.json').write_bytes(Path(args.release).read_bytes())
    pixel_start=time.perf_counter();blob=torch.load(plan['pixels_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==1 and set(blob['pixels'])==set(plan['pixel_info']),'Shared pixel coverage changed')
    pixels=blob['pixels']
    for key,value in pixels.items():need(tensor_info(value)==plan['pixel_info'][key],'Shared native pixel tensor changed')
    pixel_seconds=time.perf_counter()-pixel_start
    load_start=time.perf_counter();runtime=load_runtime(str(parallel.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.eval();model.requires_grad_(False)
    need(fingerprint(runtime.processor,str(transformers.__version__))==release['processor']
         and parallel.runtime_identity()==release['runtime'],'Joint runtime/processor changed')
    _,_,native=joint.native_api(runtime.processor);need(native==plan['native_api'],'Installed native implementation changed')
    norm=model.model.language_model.norm;language=model.model.language_model
    dtypes=dict(norm=str(norm.weight.dtype),lm_head=str(model.lm_head.weight.dtype))
    hardware=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
                  cuda_version=torch.version.cuda,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    need(dtypes==release['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16')
         and hardware==release['hardware'] and not any(p.requires_grad for p in model.parameters()),
         'Native dtype/hardware/frozen state differs from released profile')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    config=dict(schema_version=1,profile=False,shard=args.shard,release_file=str(Path(args.release).resolve()),
        release_sha256=sha(args.release),plan_file=release['plan_file'],plan_sha256=release['plan_sha256'],
        source_sha256=release['source_sha256'],model=release['model'],runtime=release['runtime'],
        processor=release['processor'],native_dtypes=dtypes,hardware=hardware,slurm_job_id=job)
    save(out/'config.json',config)
    parallel.index(out,'V7 joint training-feature shard',[('Configuration','config.json'),('Batch observations','observations.jsonl'),
                   ('Summary after completion','summary.json')])
    capture={};counts=dict(model=0,vision=0,language=0,norm=0)
    def model_hook(*_):counts['model']+=1
    def vision_hook(*_):counts['vision']+=1
    def language_hook(module,args,kwargs):
        counts['language']+=1;capture['positions']=kwargs['position_ids'].detach().cpu().clone()
        capture['mask']=kwargs['attention_mask'].detach().cpu().clone()
    def norm_hook(module,args):
        counts['norm']+=1;capture['hidden']=args[0][:,-1,:].detach().clone()
    handles=[model.register_forward_pre_hook(model_hook),model.model.visual.register_forward_pre_hook(vision_hook),
             language.register_forward_pre_hook(language_hook,with_kwargs=True),norm.register_forward_pre_hook(norm_hook)]
    rope=get_rope_index_fn(model);assigned=set(release['shards'][args.shard]);ids_out=[];chunks=[];batches=0
    raw_path=out/'observations.jsonl'
    try:
        with raw_path.open('x') as stream:
            for group in joint.GROUPS:
                ids=[fid for fid in plan['groups'][group] if fid in assigned]
                for offset in range(0,len(ids),4):
                    selected=ids[offset:offset+4];begin=time.perf_counter()
                    inputs=joint.pack(torch,[joint.row_inputs(torch,plan,pixels,fid) for fid in selected],plan['pad_token_id'])
                    positions,deltas=rope(input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],
                                           attention_mask=inputs['attention_mask'])
                    for i,fid in enumerate(selected):
                        layout=plan['layouts'][plan['features'][fid]['layout_id']];length=len(layout['input_ids'])
                        need(tensor_info(positions[:,i:i+1,-length:])==layout['position_ids_info'],
                             'Actual packed row mRoPE differs from staged identity')
                    identity=dict(feature_ids=selected,input_ids=tensor_info(inputs['input_ids']),
                        attention_mask=tensor_info(inputs['attention_mask']),image_grid_thw=tensor_info(inputs['image_grid_thw']),
                        position_ids=tensor_info(positions),rope_deltas=deltas.tolist(),
                        image_sha256=[plan['scenes'][plan['features'][fid]['sid']]['image_sha256'] for fid in selected])
                    item=move_to_device(inputs,runtime.device);torch.cuda.synchronize();prepared_at=time.perf_counter()
                    before=dict(counts);capture.clear()
                    with torch.inference_mode():output=model(**item,use_cache=False,logits_to_keep=1)
                    torch.cuda.synchronize();forward_seconds=time.perf_counter()-prepared_at
                    need({key:counts[key]-before[key] for key in counts}==dict(model=1,vision=1,language=1,norm=1),
                         'Joint native forward counters changed')
                    need(torch.equal(capture['positions'],positions) and torch.equal(capture['mask'],inputs['attention_mask'])
                         and torch.equal(model.model.rope_deltas.detach().cpu(),deltas),'Executed native layout changed')
                    hidden=capture['hidden']
                    need(hidden.shape==(len(selected),3584) and hidden.dtype==torch.float16 and bool(torch.isfinite(hidden).all())
                         and output.logits.shape[:2]==(len(selected),1) and output.past_key_values is None
                         and bool(torch.isfinite(output.logits).all()),'Nonfinite/malformed native state/logits')
                    states=hidden.cpu().clone();chunks.append(states);ids_out.extend(selected);batches+=1
                    observation=dict(group=group,batch_size=len(selected),identity=identity,states=tensor_info(states),
                        preparation_seconds=prepared_at-begin,forward_seconds=forward_seconds,
                        batch_seconds=time.perf_counter()-begin,native_model_forwards=1,native_visual_forwards=1)
                    stream.write(json.dumps(observation,allow_nan=False)+'\n');stream.flush()
                    if batches%25==0:print(json.dumps(dict(shard=args.shard,features=len(ids_out),batches=batches)),flush=True)
                    del output,inputs,item,hidden,states
    finally:
        for handle in handles:handle.remove()
    need(ids_out==release['shards'][args.shard] and len(set(ids_out))==len(ids_out)
         and counts==dict(model=batches,vision=batches,language=batches,norm=batches),'Joint shard final coverage differs')
    need(verify_release(args.release,pixels=False)[0]==release,'Release/source changed during harvest')
    states=torch.cat(chunks);path=data/'features.pt'
    torch.save(dict(schema_version=1,feature_ids=ids_out,states=states),path)
    summary=dict(config,completed=True,computational_integrity_passed=True,passed=True,
        feature_count=len(ids_out),feature_ids_sha256=object_sha(ids_out),features_file=str(path),features_sha256=sha(path),
        states=tensor_info(states),feature_state_sha256={fid:tensor_info(row)['sha256'] for fid,row in zip(ids_out,states)},
        observations_file=str(raw_path),observations_sha256=sha(raw_path),batches=batches,
        native_forward_counts=counts,model_load_seconds=load_seconds,pixel_load_verify_seconds=pixel_seconds,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),total_seconds=time.perf_counter()-started)
    save(out/'summary.json',summary)
    parallel.index(data,'V7 joint native training states',[('Feature tensor file','features.pt')])
    print(json.dumps(dict(output=str(out),passed=True,features=len(ids_out),seconds=summary['total_seconds'])),flush=True)


def merge(args):
    import torch
    started=time.monotonic();release,plan,parent=verify_release(args.release,pixels=True)
    need(len(args.merge)==4 and len(set(args.merge))==4,'Four distinct joint shard directories required')
    profile,_=profile_binding(release['profile_directory'],plan,release['plan_file'])
    check_profile_raw(torch,profile,plan)
    locations={};seen=set();summaries=[]
    for value in args.merge:
        directory=Path(value).resolve();need(directory.is_relative_to(OUT),'Joint shard outside canonical outputs')
        summary=read(directory/'summary.json');config=read(directory/'config.json')
        need(summary['completed'] is True and summary['passed'] is True and summary['computational_integrity_passed'] is True
             and summary['profile'] is False and summary['release_sha256']==sha(args.release)
             and summary['source_sha256']==release['source_sha256'],'Incomplete/incompatible joint shard')
        need(all(summary[key]==config[key] for key in config),'Joint shard config/summary differs')
        for key in ('model','runtime','processor','native_dtypes','hardware','plan_file','plan_sha256'):
            need(summary[key]==release[key],'Joint shard identity differs: '+key)
        need(read(directory/'source_hashes.json')==release['source_sha256'],'Joint shard source ledger differs')
        for name,digest in release['source_sha256'].items():
            need(sha(directory/'source'/name)==digest,'Joint shard frozen source copy changed')
        shard=summary['shard'];need(shard in range(4) and shard not in seen,'Duplicate/invalid joint shard')
        seen.add(shard)
        need(sha(summary['features_file'])==summary['features_sha256'] and
             sha(summary['observations_file'])==summary['observations_sha256'],'Joint shard tensor/observation changed')
        blob=torch.load(summary['features_file'],map_location='cpu',weights_only=True);ids=blob['feature_ids'];states=blob['states']
        need(blob['schema_version']==1 and ids==release['shards'][shard]
             and object_sha(ids)==summary['feature_ids_sha256'] and tensor_info(states)==summary['states']
             and states.shape==(len(ids),3584) and states.dtype==torch.float16 and bool(torch.isfinite(states).all()),
             'Joint shard state/ID contract differs')
        observations=[json.loads(line) for line in Path(summary['observations_file']).read_text().splitlines()]
        need([fid for row in observations for fid in row['identity']['feature_ids']]==ids
             and len(observations)==summary['batches'],'Joint actual batch order differs from saved features')
        offset=0
        for row in observations:
            count=row['batch_size'];batch_ids=row['identity']['feature_ids']
            need(count==len(batch_ids) and all(plan['features'][fid]['phase']==row['group'] for fid in batch_ids)
                 and 1<=count<=4 and tensor_info(states[offset:offset+count])==row['states']
                 and row['native_model_forwards']==row['native_visual_forwards']==1,'Joint batch state/counters differ')
            offset+=count
        need(offset==len(ids)==summary['feature_count'],'Joint batch/tensor/summary row counts differ')
        need(summary['native_forward_counts']==dict(model=len(observations),vision=len(observations),
             language=len(observations),norm=len(observations)),'Joint total counters differ')
        for row,(fid,state) in enumerate(zip(ids,states)):
            digest=tensor_info(state)['sha256'];need(fid not in locations and digest==summary['feature_state_sha256'][fid],
                                                   'Joint state duplicated or changed')
            locations[fid]=dict(file=summary['features_file'],file_sha256=summary['features_sha256'],row=row,state_sha256=digest)
        summaries.append(dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),shard=shard))
    need(set(locations)==set(plan['features']) and len(locations)==3780 and seen==set(range(4)),
         'Joint merged inventory incomplete')
    cache=dict(schema_version=1,protocol='v7_joint_training_features',complete=True,training_only=True,
        plan_file=release['plan_file'],plan_sha256=release['plan_sha256'],source_sha256=release['source_sha256'],
        parent_source_sha256=release['parent_source_sha256'],release_file=str(Path(args.release).resolve()),
        release_sha256=sha(args.release),model=release['model'],runtime=release['runtime'],processor=release['processor'],
        native_dtypes=release['native_dtypes'],hardware=release['hardware'],feature_count=3780,counts=plan['counts'],
        features=locations,scenes=plan['scenes'],shards=summaries,balanced_ancestors=plan['balanced_ancestors'],
        profile_directory=release['profile_directory'],profile_summary_sha256=release['profile']['summary']['sha256'])
    target=DATA/'feature_cache.json';save(target,cache);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'harvest_merge_{job}';out.mkdir(parents=True,exist_ok=False)
    snapshot(out,release['source_sha256'])
    save(out/'summary.json',dict(passed=True,cache_file=str(target),cache_sha256=sha(target),feature_count=3780,
        release_file=str(Path(args.release).resolve()),release_sha256=sha(args.release),
        counts=plan['counts'],seconds=time.monotonic()-started))
    parallel.index(out,'V7 complete joint feature cache',[('Summary','summary.json'),('Frozen source hashes','source_hashes.json')])
    print(json.dumps(dict(cache_file=str(target),cache_sha256=sha(target),features=3780)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true');mode.add_argument('--shard',type=int)
    mode.add_argument('--merge',type=Path,nargs=4)
    parser.add_argument('--plan',type=Path);parser.add_argument('--profile-directory',type=Path)
    parser.add_argument('--release',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID'),'All joint cache work requires Slurm')
    if args.freeze or args.merge:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'Release/merge require CPU Slurm')
        if args.freeze:
            need(args.plan is not None and args.profile_directory is not None,'Release needs exact plan and profile')
            freeze(args)
        else:
            need(args.release is not None,'Merge needs a frozen full-harvest release')
            merge(args)
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and os.environ.get('SLURM_JOB_GPUS')
             and args.release is not None,'Harvest requires one GPU Slurm allocation and CPU release')
        gpu(args)


if __name__=='__main__':main()

