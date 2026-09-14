"""One shared frozen native-feature cache; CPU release, four GPU shards, CPU merge."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_mmred_official_training as compact
from scripts import profile_mmred_official_native_memory as producer
from scripts import report_mmred_official_native_memory as auditor
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=producer.p
PROTOCOL='mmred_official_shared_native_features'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_FEATURE_HARVEST.md'
PROPOSAL_SHA='969e740859565f374d8afba61909ff6adacd234af55f815f43e0c953f0884522'
OWN=('scripts/harvest_mmred_official_training_features.py',PROPOSAL,'slurm/mmred_official_features_check.sbatch',
     'slurm/mmred_official_features_harvest.sbatch','slurm/mmred_official_features_merge.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_features'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_features')
PROFILE=producer.OUT/'profile_443978'
PROFILE_RELEASE=producer.OUT/'source_release.json'
PROFILE_RELEASE_SHA='f3fefd045dc58d337ef05ff20e9cdaf314270b9315b0bffa29163935577a80f8'
PROFILE_CONFIG_SHA='f007f8866f2828c3f7044e1f0ab5cc9e50ccf89b84692554853bbe0f1f6a15a2'
PROFILE_ANALYSIS_SHA='5b4405d62ac2e0da5291c2c110bfe1c8bf6ddd0728bfd7cb7690724aa7023b10'
PROFILE_FAILURE_SHA='9a41831122107c3d314a7541163f4bc848865921bb96ac3184bd9cdaf8633df8'
AUDIT=producer.OUT/'report_443990/summary.json'
AUDIT_SHA='183087eb0bd6ec18a0df4bb19b4dda009b20163863d25db39658460552a3ea2b'
AUDIT_ANALYSIS_SHA='cdaabf9222cda0573045ef5f231852284ef87ce098c1d910f6f0da9b7e7667a1'
POLICY=dict(protocol=PROTOCOL,check_cpu_seconds=120,shard_gpu_seconds=1800,merge_cpu_seconds=600,
    shards=4,maximum_array_attempts=1,maximum_GPUs=4,total_GPU_seconds=7200,cpu_cores=4,memory_gib=16,
    worlds=4000,worlds_per_shard=1000,frames=24800,frames_per_shard=6200,
    native_feature_rows=4860800,feature_width=3584,feature_dtype='torch.float16',raw_feature_bytes=34842214400,
    per_world_vision_batching=True,vision_calls_per_shard=1000,decoder_calls=0,language_calls=0,head_calls=0,
    no_adapters_or_training=True,no_validation_test_processing=True,no_cached_prefix_release=True,no_training_or_inference_release=True)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held harvest proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    need(sha(PROFILE_RELEASE)==PROFILE_RELEASE_SHA and sha(AUDIT)==AUDIT_SHA,'Frozen profile/audit source references changed')
    release=read(PROFILE_RELEASE);audit=read(AUDIT);result={}
    for mapping in (compact.sources(),compact.inherited_sources(),release['source_sha256'],release['inherited_source_sha256'],
                    audit['source_sha256'],audit['inherited_source_sha256']):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Feature source closure conflict');result[name]=digest
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen feature dependency changed')
    return result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Bound feature input changed')
    bindings[str(path)]=digest;return digest


def check_time(started,phase):
    need(time.perf_counter()-started<POLICY[{'check':'check_cpu_seconds','shard':'shard_gpu_seconds','merge':'merge_cpu_seconds'}[phase]],
         'Fixed feature preparation cap exceeded')


def profile_metadata(bindings,full_audit=False):
    bind(AUDIT,bindings,AUDIT_SHA);summary=read(AUDIT)
    need(summary['analysis_sha256']==AUDIT_ANALYSIS_SHA,'Fixed independent audit analysis alias changed')
    bind(summary['analysis_file'],bindings,AUDIT_ANALYSIS_SHA)
    result=auditor.verify_report(AUDIT) if full_audit else read(summary['analysis_file'])
    need(summary['passed'] is summary['completed'] is True and result['passed'] is result['cpu_numerical_passed'] is True
         and result['cold_training_components_passed'] is True and result['original_profile_passed'] is False
         and result['original_failure_preserved'] is True and result['cache_parity_passed'] is False
         and result['all_scheduled_numerical_evidence_collected'] is True and result['profile']==summary['profile']
         and not (AUDIT.parent/'failure.json').exists(),'Passed independent cold/training audit with preserved original failure required')
    for name,digest in [('config.json',PROFILE_CONFIG_SHA),('analysis.json',PROFILE_ANALYSIS_SHA),('failure.json',PROFILE_FAILURE_SHA)]:
        bind(PROFILE/name,bindings,digest)
    config=read(PROFILE/'config.json');original=read(PROFILE/'analysis.json');descriptor=result['profile']
    need(descriptor['directory']==str(PROFILE) and descriptor['config_sha256']==PROFILE_CONFIG_SHA
         and descriptor['analysis_sha256']==PROFILE_ANALYSIS_SHA and descriptor['failure_sha256']==PROFILE_FAILURE_SHA
         and original['passed'] is False and not (PROFILE/'summary.json').exists(),'Original failed profile identity differs')
    records=original['feature_records'];need(len(records)==8 and [r['index'] for r in records]==list(range(8)), 'Eight actual feature measurements required')
    for r in records:
        need(r['n']==config['cases'][r['index']]['n'] and r['seconds']>0 and r['tensor']['dtype']=='torch.float16'
             and r['tensor']['shape']==[196*r['n'],3584],'Measured native feature shape/dtype/timing differs')
        timing=PROFILE/f'feature_{r["index"]}_timing.json';bind(timing,bindings);need(read(timing)==r,'Feature timing publication differs')
    bind(PROFILE_RELEASE,bindings,PROFILE_RELEASE_SHA)
    return config,original,dict(file=str(AUDIT),sha256=AUDIT_SHA,analysis_file=summary['analysis_file'],analysis_sha256=AUDIT_ANALYSIS_SHA)


def project(records,shard_plans,original):
    F={n:max(r['seconds'] for r in original['feature_records'] if r['n']==n) for n in compact.LENGTHS}
    C={(n,q):max(r['staging_seconds'] for r in records if r['n']==n and r['qtype']==q)
       for n in compact.LENGTHS for q in compact.QTYPES}
    details=[dict(n=n,qtype=q,worlds_per_shard=50,compact_staging_max_seconds=C[(n,q)],profile_feature_max_seconds=F[n])
             for n in compact.LENGTHS for q in compact.QTYPES]
    variable=sum(50*(r['compact_staging_max_seconds']+r['profile_feature_max_seconds']) for r in details)
    shards=[]
    for i in range(4):
        setup=max(original['setup_seconds'],shard_plans[i]['setup_seconds']);seconds=setup+1.25*variable+60
        shards.append(dict(index=i,worlds=1000,frames=6200,setup_seconds=setup,projected_seconds=seconds,cap_seconds=1800,eligible=seconds<=1800))
    return dict(formula='max(profile_setup,compact_shard_setup)+1.25*sum_cells50*(max_compact_staging_cell+max_profile_feature_N)+60',
        cells=details,profile_setup_seconds=original['setup_seconds'],shards=shards,resource_eligible=all(r['eligible'] for r in shards),
        deliberate_conservative_IO_overlap=True,empirical_estimate_not_guarantee=True,unmeasured_timings_invented=False)


def make_plan(phase,bindings,artifacts,**extra):
    return dict(protocol=PROTOCOL,phase=phase,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        input_bindings=bindings,artifacts=artifacts,no_training_or_inference_release=True,**extra)


def prepare_release(path,out,started):
    bindings={};path=Path(path).resolve();stage=compact.verify_stage(path);summary=read(path)
    compact_ref=dict(file=str(path),sha256=bind(path,bindings),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])
    bind(summary['plan_file'],bindings,summary['plan_sha256'])
    for key in ('rows_file','cases_file','diagnostic_sids_file','envelopes_file','target_inventory_file','timings_file'):
        bind(stage[key],bindings,stage['artifacts'][stage[key]])
    rows=read(stage['rows_file']);records=read(stage['cases_file']);train,assigned,diagnostics=compact.selection(rows)
    need(train==rows and [{k:r[k] for k in assigned[0]} for r in records]==assigned
         and read(stage['diagnostic_sids_file'])==diagnostics,'Exact original4000 compact cohort/order required')
    shards={}
    for descriptor in stage['shard_reports']:
        bind(descriptor['file'],bindings,descriptor['sha256']);bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
        shard=read(descriptor['plan_file']);shards[shard['shard_index']]=shard
    need(set(shards)==set(range(4)),'Four measured compact shards required')
    for r in records:
        need(r['omitted_pixel_values']['shape']==[r['n']*784,1176] and r['omitted_pixel_values']['dtype']=='torch.float32'
             and r['grid_identity']['shape']==[r['n'],3] and r['grid_identity']['dtype']=='torch.int64',
             'Training vision shapes exceed fixed392px native preparation')
    config,original,audit_ref=profile_metadata(bindings,full_audit=True)
    need(config['native_identity']==stage['native_identity'] and config['native_identity_sha256']==stage['native_identity_sha256'],
         'Compact/native profile identities differ')
    bind(compact.PARENT,bindings,compact.PARENT_SHA);parent_summary=read(compact.PARENT)
    bind(parent_summary['plan_file'],bindings,compact.PARENT_PLAN_SHA);native=read(parent_summary['plan_file'])
    projection=project(records,shards,original);save(out/'projection.json',projection)
    save(out/'shards.json',[dict(index=i,case_indices=[r['index'] for r in records if r['shard_index']==i]) for i in range(4)])
    artifacts={str(f):sha(f) for f in out.glob('*.json')}
    plan=make_plan('check',bindings,artifacts,compact_stage=compact_ref,independent_profile_audit=audit_ref,
        rows_file=stage['rows_file'],cases_file=stage['cases_file'],diagnostic_sids_file=stage['diagnostic_sids_file'],
        compact_envelopes_file=stage['envelopes_file'],shards_file=str(out/'shards.json'),projection_file=str(out/'projection.json'),
        projection=projection,resource_eligible=projection['resource_eligible'],native_identity=stage['native_identity'],
        native_identity_sha256=stage['native_identity_sha256'],precision=native['precision'],packages=native['packages'],
        system_prompt=native['system_prompt'],profile_hardware=config['hardware'],
        original_profile_passed=False,cold_training_components_passed=True,no_decoder_envelope_gate_for_feature_harvest=True)
    save(out/'plan.json',plan);check_time(started,'check')
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),resource_eligible=projection['resource_eligible'])


def verify_plan(path,require_eligible=True):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='check' and summary['protocol']==PROTOCOL
         and summary['plan_file']==str(path) and summary['plan_sha256']==sha(path) and summary['policy']==POLICY
         and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed feature CPU release check required')
    need(plan['protocol']==PROTOCOL and plan['phase']=='check' and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and plan['no_training_or_inference_release'] is True,
         'Feature release policy/source differs')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Feature release metadata changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Release source archive changed')
    config,original,_=profile_metadata({})
    records=read(plan['cases_file']);compact_plan=read(plan['compact_stage']['plan_file'])
    shard_plans={read(d['plan_file'])['shard_index']:read(d['plan_file']) for d in compact_plan['shard_reports']}
    expected=project(records,shard_plans,original)
    need(expected==plan['projection']==read(plan['projection_file']) and plan['resource_eligible']==expected['resource_eligible'],
         'Resource projection changed')
    need(not require_eligible or expected['resource_eligible'],'Measured release exceeds1800s per shard; GPU harvest held')
    need(config['native_identity']==plan['native_identity'] and config['hardware']==plan['profile_hardware'], 'Profile identity changed')
    return plan


def counters_expected(worlds):return dict(model=0,backbone=0,language=0,norm=0,head=0,vision=worlds)


def harvest(index,release_path,out,data,started,progress):
    plan=verify_plan(release_path);records=read(plan['cases_file']);rows=read(plan['rows_file'])
    selected=[r for r in records if r['shard_index']==index]
    need(len(selected)==1000 and sum(r['n'] for r in selected)==6200,'Exact balanced feature shard required')
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    loaded=load_runtime(str(compact.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
    need(hardware==plan['profile_hardware'] and p.installed({})==plan['packages'],'Actual native hardware/packages differ from fixed profile')
    refs=list(model.named_parameters());base=p.base_metadata(refs);p.check_base(refs,base,model)
    need(not p.adapter_parameters(model) and not any(v.training for v in model.modules()),'Entire unchanged base must be frozen/eval with no adapter')
    owner,fn,api=p.backend.native_api(loaded.processor);rope=lambda **kw:fn(owner,**kw)
    counts=Counter(counters_expected(0));progress['counters']=counts;features=[];setup=time.perf_counter()-started
    save(out/'base_before.json',base);save(out/'hardware.json',hardware)
    with ExitStack() as hooks, (out/'features.jsonl').open('x') as stream:
        def count(key):
            def callback(*_):
                counts[key]+=1
                need(key=='vision','Feature harvest unexpectedly called a decoder/model/norm/head')
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.language_model,'language'),
                           (model.model.language_model.norm,'norm'),(model.lm_head,'head'),(model.model.visual,'vision')):
            hooks.callback(module.register_forward_pre_hook(count(key)).remove)
        for compact_record in selected:
            check_time(started,'shard');tick=time.perf_counter();idx=compact_record['index'];row=rows[idx]
            need(row['sid']==compact_record['sid'] and row['raw_row_sha256']==compact_record['raw_row_sha256'],'Feature row ownership changed')
            need(sha(compact_record['file'])==compact_record['sha256'],'Consumed compact packet changed')
            packet=torch.load(compact_record['file'],map_location='cpu',weights_only=True)
            need(set(packet)=={'case','omitted_pixel_values'} and packet['omitted_pixel_values']==compact_record['omitted_pixel_values']
                 and object_sha(compact.tensor_tree(torch,p,packet['case']))==compact_record['compact_identity_sha256'],
                 'Compact tensor/metadata identity differs')
            loaded_at=time.perf_counter();case=compact.preparation.prepare_joint(loaded.processor,row,plan['system_prompt'],rope)
            prepared_at=time.perf_counter();pixels=case['inputs']['pixel_values'];pixel_info=p.tensor_info(pixels)
            need(pixel_info==packet['omitted_pixel_values'],'Reprocessed pixels differ from CPU-prepared native tensor')
            grid_info=p.tensor_info(case['inputs']['image_grid_thw']);need(grid_info==compact_record['grid_identity'],'Native image grid differs')
            del case['inputs']['pixel_values']
            need(compact.tensor_tree(torch,p,case)==compact.tensor_tree(torch,p,packet['case'])
                 and object_sha(compact.tensor_tree(torch,p,case))==compact_record['compact_identity_sha256'],
                 'Reprocessed native/teacher/text/coordinate packet differs')
            coordinate_identity={k:p.tensor_info(v) for k,v in case['coordinates'].items()}
            proof_at=time.perf_counter();torch.cuda.synchronize();vision_start=time.perf_counter();before=counts['vision']
            with torch.inference_mode():value=producer.extract_features(torch,model,pixels,case['inputs']['image_grid_thw'])
            torch.cuda.synchronize();vision_end=time.perf_counter()
            need(counts['vision']==before+1 and value.dtype==torch.float16 and value.shape==(row['n']*196,3584)
                 and bool(torch.isfinite(value).all()),'One finite native FP16 feature collection required')
            tensor=p.tensor_info(value);input_identity=dict(pixel_values=pixel_info,image_grid_thw=grid_info)
            compact_ref=dict(file=compact_record['file'],sha256=compact_record['sha256'],compact_identity_sha256=compact_record['compact_identity_sha256'])
            file=data/f'features_{idx:04d}.pt'
            ownership=dict(index=idx,sid=row['sid'],n=row['n'],input_identity=input_identity,compact=compact_ref,
                coordinate_identity=coordinate_identity,native_identity_sha256=plan['native_identity_sha256'])
            torch.save(dict(features=value.cpu(),**ownership),file)
            descriptor=dict(index=idx,sid=row['sid'],n=row['n'],qtype=row['qtype'],shard_index=index,
                file=str(file),sha256=sha(file),bytes=file.stat().st_size,tensor=tensor,finite=True,
                raw_feature_bytes=value.numel()*value.element_size(),input_identity=input_identity,compact=compact_ref,
                coordinate_identity=coordinate_identity,native_identity_sha256=plan['native_identity_sha256'])
            p.check_base(refs,base,model);check_time(started,'shard');torch.cuda.synchronize()
            descriptor.update(compact_load_seconds=loaded_at-tick,full_reprocess_seconds=prepared_at-loaded_at,
                input_proof_seconds=proof_at-prepared_at,vision_seconds=vision_end-vision_start)
            # Publish the durable ownership record before stopping the complete-world timer.
            stream.write(json.dumps(descriptor,sort_keys=True,allow_nan=False)+'\n');stream.flush()
            end=time.perf_counter();descriptor.update(output_and_audit_seconds=end-vision_end,full_world_seconds=end-tick)
            features.append(descriptor)
            progress['completed_worlds']=len(features)
            del value,pixels,packet,case
    need(dict(counts)==counters_expected(1000) and sum(r['raw_feature_bytes'] for r in features)==POLICY['raw_feature_bytes']//4,
         'Exact feature-only call/byte inventory required')
    p.check_base(refs,base,model);save(out/'base_after.json',p.base_metadata(refs));save(out/'features.json',features)
    save(out/'counters.json',dict(counts));save(out/'timings.json',dict(setup_seconds=setup,
        full_world_seconds=sum(r['full_world_seconds'] for r in features),maximum_world_seconds=max(r['full_world_seconds'] for r in features),
        vision_seconds=sum(r['vision_seconds'] for r in features),full_reprocess_seconds=sum(r['full_reprocess_seconds'] for r in features)))
    bindings={str(Path(release_path).resolve()):sha(release_path),plan['cases_file']:sha(plan['cases_file']),plan['rows_file']:sha(plan['rows_file'])}
    artifacts={str(f):sha(f) for f in out.glob('*.json')};artifacts[str(out/'features.jsonl')]=sha(out/'features.jsonl')
    result=make_plan('shard',bindings,artifacts,release_plan_file=str(Path(release_path).resolve()),release_plan_sha256=sha(release_path),
        feature_index_file=str(out/'features.json'),timings_file=str(out/'timings.json'),counters=dict(counts),
        native_identity_sha256=plan['native_identity_sha256'],shard_index=index,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
        feature_file_hashes_are_in_bound_index=True,all_features_finite_native_FP16=True)
    save(out/'plan.json',result)
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),shard_index=index,
        array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],worlds=1000,vision_calls=1000)


def verify_report(path,phase,streaming=False):
    path=Path(path).resolve();summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==PROTOCOL and summary['phase']==phase
         and summary['policy']==POLICY and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed feature report required')
    need(sha(summary['plan_file'])==summary['plan_sha256'],'Feature report plan changed');plan=read(summary['plan_file'])
    need(plan['protocol']==PROTOCOL and plan['phase']==phase and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and plan['no_training_or_inference_release'] is True,'Feature report source/policy differs')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Feature metadata changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Archived feature source changed')
    need(plan['feature_index_file'] in plan['artifacts'],'Feature index must be bound');features=read(plan['feature_index_file'])
    if not streaming:
        for r in features:need(sha(r['file'])==r['sha256'] and Path(r['file']).stat().st_size==r['bytes'],'Saved feature file changed')
    need(0<summary['elapsed_seconds']<=POLICY['shard_gpu_seconds' if phase=='shard' else 'merge_cpu_seconds'],'Feature report time exceeds cap')
    return plan,features,dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])


def merge(paths,release_path,out,started):
    release_path=Path(release_path).resolve();release=verify_plan(release_path);compact_records=read(release['cases_file'])
    features=[];indices=set();arrays=set();reports=[];bindings={str(release_path):sha(release_path)}
    for path in paths:
        check_time(started,'merge');plan,items,descriptor=verify_report(path,'shard');i=plan['shard_index']
        need(i in range(4) and i not in indices,'Four distinct feature shards required');indices.add(i);arrays.add(plan['array_job_id'])
        expected=[r for r in compact_records if r['shard_index']==i]
        need(len(items)==1000 and [r['index'] for r in items]==[r['index'] for r in expected]
             and plan['release_plan_file']==str(release_path) and plan['release_plan_sha256']==sha(release_path)
             and plan['native_identity_sha256']==release['native_identity_sha256'] and plan['counters']==counters_expected(1000)
             and plan['all_features_finite_native_FP16'] is True,'Feature shard release/call/ownership differs')
        for r,c in zip(items,expected):
            need((r['sid'],r['n'],r['qtype'],r['shard_index'])==(c['sid'],c['n'],c['qtype'],c['shard_index'])
                 and r['tensor']['shape']==[r['n']*196,3584] and r['tensor']['dtype']=='torch.float16' and r['finite'] is True
                 and r['raw_feature_bytes']==r['n']*196*3584*2 and r['bytes']>=r['raw_feature_bytes']
                 and r['input_identity']==dict(pixel_values=c['omitted_pixel_values'],image_grid_thw=c['grid_identity'])
                 and r['compact']==dict(file=c['file'],sha256=c['sha256'],compact_identity_sha256=c['compact_identity_sha256'])
                 and r['native_identity_sha256']==release['native_identity_sha256']
                 and 0<r['vision_seconds']<=r['full_world_seconds'] and 0<r['full_reprocess_seconds']<=r['full_world_seconds'],
                 'Native feature shape/finite/input/timing metadata differs')
        features.extend(items);reports.append(descriptor)
        bind(descriptor['file'],bindings,descriptor['sha256']);bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
        for file,digest in plan['artifacts'].items():bind(file,bindings,digest)
    features.sort(key=lambda r:r['index'])
    need(indices==set(range(4)) and len(arrays)==1 and [r['index'] for r in features]==list(range(4000))
         and len({r['sid'] for r in features})==4000 and sum(r['raw_feature_bytes'] for r in features)==34842214400,
         'Complete ordered shared4000 feature cache required')
    save(out/'feature_index.json',features);save(out/'timings.json',dict(full_world_seconds=sum(r['full_world_seconds'] for r in features),
        vision_seconds=sum(r['vision_seconds'] for r in features),full_reprocess_seconds=sum(r['full_reprocess_seconds'] for r in features),
        saved_bytes=sum(r['bytes'] for r in features),raw_feature_bytes=34842214400,vision_calls=4000,
        model_calls=0,decoder_calls=0,head_calls=0,shared_by_all_three_arms=True))
    (out/'REPORT.md').write_text('# Shared original MMReD features\n\nAll4,000 original training worlds have one bound finite native FP16 feature packet. Four thousand vision calls and no decoder/model/head calls were made. Merge verified file hashes and source-bound metadata without loading or copying tensor payloads. The original cache-parity failure remains preserved. No training or inference is released.\n')
    artifacts={str(f):sha(f) for f in out.glob('*.json')};artifacts[str(out/'REPORT.md')]=sha(out/'REPORT.md')
    plan=make_plan('merge',bindings,artifacts,release_plan_file=str(release_path),release_plan_sha256=sha(release_path),
        feature_index_file=str(out/'feature_index.json'),timings_file=str(out/'timings.json'),shard_reports=reports,
        compact_stage=release['compact_stage'],independent_profile_audit=release['independent_profile_audit'],
        rows_file=release['rows_file'],cases_file=release['cases_file'],diagnostic_sids_file=release['diagnostic_sids_file'],
        compact_envelopes_file=release['compact_envelopes_file'],native_identity=release['native_identity'],
        native_identity_sha256=release['native_identity_sha256'],precision=release['precision'],packages=release['packages'],
        system_prompt=release['system_prompt'],feature_file_hashes_are_in_bound_index=True,
        all_features_finite_native_FP16=True,original_profile_passed=False,cold_training_components_passed=True,
        array_job_id=next(iter(arrays)),no_merge_tensor_load_or_copy=True)
    save(out/'plan.json',plan)
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),worlds=4000,vision_calls=4000,array_job_id=next(iter(arrays)))


def verify_stage(path,streaming=False):
    plan,features,_=verify_report(path,'merge',streaming=streaming)
    release=verify_plan(plan['release_plan_file'])
    need(plan['release_plan_sha256']==sha(plan['release_plan_file']) and len(features)==4000
         and [r['index'] for r in features]==list(range(4000)) and sum(r['raw_feature_bytes'] for r in features)==34842214400
         and plan['compact_stage']==release['compact_stage'] and plan['native_identity']==release['native_identity']
         and plan['all_features_finite_native_FP16'] is True,'Complete exact native feature cache required')
    return plan


def main():
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--shard',type=int,choices=range(4));group.add_argument('--merge',nargs=4,type=Path)
    ap.add_argument('--compact',type=Path);ap.add_argument('--plan',type=Path);args=ap.parse_args()
    phase='check' if args.check else 'shard' if args.shard is not None else 'merge';started=time.perf_counter()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm allocation required')
    if phase=='shard':
        p.native.require_slurm(gpu=True)
        need(os.environ.get('SLURM_ARRAY_JOB_ID') and int(os.environ.get('SLURM_ARRAY_TASK_COUNT','0'))==4
             and int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))==args.shard and args.plan is not None,'One four-task GPU feature array required')
        array=os.environ['SLURM_ARRAY_JOB_ID'];tag=f'shard_{array}_{args.shard}'
        need(all(path.name.split('_')[1]==array for path in OUT.glob('shard_*')),'Only one feature harvest array is released')
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU-only release/merge required')
        need(args.compact is not None if phase=='check' else args.plan is not None,'Required compact summary/release plan missing')
        tag=f'{phase}_{os.environ["SLURM_JOB_ID"]}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**frozen,**inherited}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Feature source archive changed')
    save(out/'request.json',dict(phase=phase,shard=args.shard,compact=None if args.compact is None else str(args.compact.resolve()),
        plan=None if args.plan is None else str(args.plan.resolve()),merge=None if args.merge is None else [str(p.resolve()) for p in args.merge],policy=POLICY))
    progress={}
    try:
        if phase=='check':result=prepare_release(args.compact,out,started)
        elif phase=='shard':
            data=DATA/tag;data.mkdir(parents=True,exist_ok=False);result=harvest(args.shard,args.plan,out,data,started,progress)
        else:result=merge(args.merge,args.plan,out,started)
        check_time(started,phase);need(sources()==frozen and inherited_sources()==inherited,'Feature source closure changed')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,phase=phase,policy=POLICY,
            source_sha256=frozen,inherited_source_sha256=inherited,elapsed_seconds=time.perf_counter()-started,
            no_training_or_inference_release=True,**result))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),phase=phase,elapsed_seconds=time.perf_counter()-started,
            source_sha256=frozen,progress=progress,partial_outputs_retained=True,no_training_or_inference_release=True));raise


if __name__=='__main__':main()
