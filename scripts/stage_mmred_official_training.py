"""Prepare compact original-MMReD training inputs, without native feature calls.

All processor, image and tensor work is CPU Slurm-only. Merge uses JSON/hashes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import prepare_mmred_official_native_v3 as preparation
from scripts import render_mmred_official_shards as rendering
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save

PROTOCOL='mmred_official_compact_training_preparation'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_TRAINING_PREPARATION.md'
PROPOSAL_SHA='b4ececc8f71ad818c6d36df8492ceb4eb9c94ce7445a4eac91d3ee15de1a1ef9'
OWN=('scripts/stage_mmred_official_training.py',PROPOSAL,
     'slurm/mmred_official_training_prepare.sbatch','slurm/mmred_official_training_merge.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_training'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_training')
PARENT=preparation.OUT/'check_443974/summary.json'
PARENT_SHA='25bfcf524f6dd9b8568f219d0f2e4f5217bea8fac98b4fe0151c9a17882d97f7'
PARENT_PLAN_SHA='147070a5819d51802b588f8e3466baa5821d5d2702d08b1cce11627c6903a864'
PARENT_RELEASE=preparation.OUT/'source_release.json'
PARENT_RELEASE_SHA='8ed154d9bc65049c94d9a39b3edb11abd941700952410f725465bb50bf7db0f8'
RENDER=rendering.OUT/'merge_443964/summary.json'
RENDER_SHA='2f1fe71b030085f6836bb7ad822ad24ad1114086bb9b91cc63d10aa7694c4c4f'
RENDER_PLAN_SHA='29bd4a1acc84b7590f2903d2535112422ffb4a7395bd6fc318e636bf7518a713'
CORE=REPO/'outputs/native_aggregation_vlm/native_visual_memory/check_443957/summary.json'
CORE_SHA='e06e95f992dfc5218a7e695938a79767501d40ce6d31549cab3efbab7b83089f'
LENGTHS=(1,2,4,8,16)
QTYPES=preparation.QTYPES
WIDTHS=('ordinary_prompt','ordinary_teacher','compressed_prompt','compressed_teacher','target_rows')
POLICY=dict(protocol=PROTOCOL,shards=4,worlds_per_shard=1000,frames_per_shard=6200,
    worlds=4000,frame_occurrences=24800,cells=20,rows_per_cell=200,diagnostic_sids=100,
    cpu_seconds_per_shard=900,merge_cpu_seconds=600,total_allocation_seconds=4200,cpu_cores=4,memory_gib=16,
    maximum_array_attempts=1,lengths=list(LENGTHS),qtypes=list(QTYPES),memory_slots=32,
    resize=392,native_feature_rows_per_frame=196,native_feature_width=3584,
    maximum_target_tokens=50,no_pixels_saved=True,no_feature_extraction=True,no_model_or_head_calls=True,
    no_validation_test_processing=True,no_training_or_inference_release=True)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Compact preparation proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    need(sha(PARENT_RELEASE)==PARENT_RELEASE_SHA and sha(CORE)==CORE_SHA,'Frozen preparation/core source descriptors changed')
    parent=read(PARENT_RELEASE);core=read(CORE);result={}
    for record in (parent,core):
        for name,digest in {**record['source_sha256'],**record['inherited_source_sha256']}.items():
            need(name not in result or result[name]==digest,'Source closure conflict');result[name]=digest
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen imported source changed')
    return result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound input changed');bindings[str(path)]=digest
    return digest


def check_time(started,phase):
    need(time.perf_counter()-started<POLICY['cpu_seconds_per_shard' if phase=='shard' else 'merge_cpu_seconds'],
         'Fixed compact preparation CPU cap exceeded')


def selection(rows):
    train=[r for r in rows if r['pilot_role']=='train'];counts=Counter();assigned=[]
    need(len(train)==len({r['sid'] for r in train})==4000,'Exactly4000 distinct original training rows required')
    for index,row in enumerate(train):
        key=(row['n'],row['qtype']);ordinal=counts[key];counts[key]+=1
        assigned.append(dict(index=index,sid=row['sid'],n=row['n'],qtype=row['qtype'],cell_index=ordinal,shard_index=ordinal%4))
    need(counts==Counter({(n,q):200 for n in LENGTHS for q in QTYPES}), 'Original20x200 training cells required')
    diagnostics=[r['sid'] for n in LENGTHS for q in QTYPES
                 for r in [x for x in train if x['n']==n and x['qtype']==q][:5]]
    need(len(diagnostics)==len(set(diagnostics))==100,'Exact first-five-per-cell diagnostic100 required')
    for i in range(4):
        subset=[r for r in assigned if r['shard_index']==i]
        need(len(subset)==1000 and sum(r['n'] for r in subset)==6200
             and Counter((r['n'],r['qtype']) for r in subset)==Counter({(n,q):50 for n in LENGTHS for q in QTYPES}),
             'Balanced1000-world/6200-frame shard differs')
    return train,assigned,diagnostics


def dependencies(bindings):
    bind(PARENT,bindings,PARENT_SHA);summary=read(PARENT)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==preparation.PROTOCOL
         and summary['source_sha256']==preparation.sources() and not (PARENT.parent/'failure.json').exists(),
         'Passed exact V3 preparation required')
    bind(summary['plan_file'],bindings,PARENT_PLAN_SHA);need(summary['plan_sha256']==PARENT_PLAN_SHA,'Preparation plan alias differs')
    plan=read(summary['plan_file'])
    need(plan['protocol']==preparation.PROTOCOL and plan['policy']==preparation.POLICY
         and plan['source_sha256']==preparation.sources() and plan['inherited_source_sha256']==preparation.inherited_sources()
         and plan['no_model_forward'] is plan['no_training_release'] is True,'Preparation policy/source scope differs')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for path,digest in mapping.items():bind(path,bindings,digest)
    for name,digest in {**plan['source_sha256'],**plan['inherited_source_sha256']}.items():
        bind(PARENT.parent/'source'/name.replace('/','_'),bindings,digest)
    bind(RENDER,bindings,RENDER_SHA);render=rendering.verify_stage(RENDER,streaming=True)
    need(read(RENDER)['plan_sha256']==RENDER_PLAN_SHA and plan['render_plan']==dict(file=str(RENDER),sha256=RENDER_SHA)
         and plan['rows_file']==render['rows_file'],'Same passed render merge required')
    bind(read(RENDER)['plan_file'],bindings,RENDER_PLAN_SHA)
    rows=read(plan['rows_file']);train,assigned,diagnostics=selection(rows)
    need(preparation.select_cases(rows)==[next(r for r in rows if r['sid']==item['sid']) for item in plan['cases']],
         'Reference profile cases no longer follow frozen selection')
    for item in plan['cases']:bind(item['file'],bindings,item['sha256'])
    bind(CORE,bindings,CORE_SHA);core=read(CORE)
    need(core['passed'] is core['completed'] is True and core['slots']==32 and core['hidden_size']==3584
         and core['key_width']==128 and core['parameters']==469504 and core['tests_run']==8
         and not (CORE.parent/'failure.json').exists(),'Fixed passed memory-format fixture required')
    for field in ('report','test_results'):bind(core[field+'_file'],bindings,core[field+'_sha256'])
    for name,digest in {**core['source_sha256'],**core['inherited_source_sha256']}.items():
        bind(CORE.parent/'source'/name.replace('/','_'),bindings,digest)
    bind(PARENT_RELEASE,bindings,PARENT_RELEASE_SHA)
    return plan,train,assigned,diagnostics


def tensor_tree(torch,p,value):
    if isinstance(value,torch.Tensor):return {'tensor':p.tensor_info(value)}
    if isinstance(value,dict):return {k:tensor_tree(torch,p,v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tensor_tree(torch,p,v) for v in value]
    need(value is None or type(value) in (str,int,float,bool),'Unexpected compact packet value')
    return value


def widths(case):
    text=case['text'];L=len(case['target_ids'])
    compressed=text['prefix']['opening_ids'].shape[1]+32+text['prefix']['end_ids'].shape[1]+text['suffix']['input_ids'].shape[1]
    W=case['inputs']['input_ids'].shape[1];S=case['teacher_input_ids'].shape[1]
    need(case['metadata']['prompt_width']==W and case['metadata']['teacher_width']==S and S==W+L-1
         and case['target_positions']==list(range(W-1,S)) and 1<L<=50 and case['target_ids'][-1]==151645,
         'Whole JSON/EOS teacher width/positions differ')
    return dict(ordinary_prompt=W,ordinary_teacher=S,compressed_prompt=compressed,compressed_teacher=compressed+L-1,target_rows=L)


def reference_cases(torch,p,plan):
    records=[];identities={}
    for item in plan['cases']:
        need(sha(item['file'])==item['sha256'],'Reference tensor packet changed')
        case=torch.load(item['file'],map_location='cpu',weights_only=True)
        need(case['metadata']==item['metadata'] and case['target_ids']==item['target_ids'],'Reference packet ownership differs')
        identities[item['sid']]=object_sha(tensor_tree(torch,p,case))
        records.append(dict(index=item['index'],sid=item['sid'],n=item['n'],qtype=item['qtype'],**widths(case)))
    return records,identities


def envelopes(records,references):
    uncovered=[];maxima=[]
    for record in records:
        candidates=[r for r in references if r['n']==record['n']]
        covered=[r['index'] for r in candidates if all(record[k]<=r[k] for k in WIDTHS)]
        if not covered:
            uncovered.append(dict(index=record['index'],sid=record['sid'],n=record['n'],qtype=record['qtype'],
                widths={k:record[k] for k in WIDTHS},excess_above_component_max={k:max(0,record[k]-max(r[k] for r in candidates)) for k in WIDTHS}))
    for n in LENGTHS:
        group=[r for r in records if r['n']==n]
        for key in WIDTHS:
            maximum=max(r[key] for r in group);owner=next(r for r in group if r[key]==maximum)
            maxima.append(dict(n=n,metric=key,value=maximum,sid=owner['sid'],index=owner['index']))
    return dict(reference_cases=references,metrics=list(WIDTHS),requires_one_jointly_covering_reference=True,
        uncovered_count=len(uncovered),uncovered=uncovered,first_tie_maxima=maxima,
        fit_profile_envelope_eligible=not uncovered,prepared_shape_comparison_only=True,
        requires_passed_scope_specific_software_audit=True,gpu_profile_outcome_is_not_preparation_gate=True,
        no_training_or_inference_release=True)


def target_inventory(records):
    result={}
    for r in records:
        target=r['target_text'];entry=dict(ids=r['target_ids'],length=r['target_rows'])
        need(target not in result or {k:result[target][k] for k in entry}==entry,'Canonical target tokenization differs')
        if target not in result:result[target]=dict(entry,worlds=0)
        result[target]['worlds']+=1
    return dict(targets=result,maximum_target_tokens=max(r['target_rows'] for r in records),
        supervised_rows_per_epoch=sum(r['target_rows'] for r in records),worlds=len(records))


def timings(records):
    groups=[]
    for n in LENGTHS:
        for q in QTYPES:
            group=[r for r in records if r['n']==n and r['qtype']==q]
            groups.append(dict(n=n,qtype=q,worlds=len(group),
                prepare_seconds_sum=sum(r['prepare_seconds'] for r in group),prepare_seconds_max=max(r['prepare_seconds'] for r in group),
                pixel_identity_seconds_sum=sum(r['pixel_identity_seconds'] for r in group),
                reprocess_and_identity_seconds_max=max(r['prepare_seconds']+r['pixel_identity_seconds'] for r in group),
                staging_seconds_sum=sum(r['staging_seconds'] for r in group),staging_seconds_max=max(r['staging_seconds'] for r in group)))
    return dict(cells=groups,compact_bytes=sum(r['bytes'] for r in records),
        omitted_pixel_bytes=sum(r['pixel_bytes'] for r in records),
        preparation_and_identity_cost_must_be_added_to_future_feature_profile=True)


def common_plan(phase,bindings,artifacts,**extra):
    return dict(protocol=PROTOCOL,phase=phase,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        preparation=dict(file=str(PARENT),sha256=PARENT_SHA,plan_sha256=PARENT_PLAN_SHA),
        render=dict(file=str(RENDER),sha256=RENDER_SHA,plan_sha256=RENDER_PLAN_SHA),core_report=dict(file=str(CORE),sha256=CORE_SHA),
        input_bindings=bindings,artifacts=artifacts,no_training_or_inference_release=True,**extra)


def shard(index,out,data,started):
    bindings={};plan,train,assigned,diagnostics=dependencies(bindings)
    import torch
    from transformers import AutoProcessor,__version__ as tf_version
    from scripts import diagnose_native_identity_join_joint_lora_v2 as p
    from scripts.stage_native_vision_v6_teacher import model_metadata
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(str(preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=p.backend.native_api(processor);rope=lambda **kw:fn(owner,**kw);identity=plan['native_identity']
    need(api==identity['native_api'] and p.fingerprint(processor,str(tf_version))==identity['processor']
         and p.backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'], 'Pinned native CPU environment changed')
    references,reference_ids=reference_cases(torch,p,plan);records=[];setup=time.perf_counter()-started
    journal=out/'cases.jsonl'
    with journal.open('x') as stream:
        for assignment in assigned:
            if assignment['shard_index']!=index:continue
            check_time(started,'shard');row=train[assignment['index']];tick=time.perf_counter()
            case=preparation.prepare_joint(processor,row,plan['system_prompt'],rope);prepared=time.perf_counter()-tick
            identity_tick=time.perf_counter();pixels=case['inputs']['pixel_values'];pixel_identity=p.tensor_info(pixels)
            identity_seconds=time.perf_counter()-identity_tick
            full_tree=tensor_tree(torch,p,case)
            if row['sid'] in reference_ids:need(object_sha(full_tree)==reference_ids[row['sid']],'Reprocessed reference case differs from frozen V3')
            expected=dict(full_tree);expected['inputs']=dict(full_tree['inputs']);del expected['inputs']['pixel_values']
            del case['inputs']['pixel_values'];need(tensor_tree(torch,p,case)==expected,'Compaction changed more than pixel_values')
            file=data/f'case_{assignment["index"]:04d}.pt'
            torch.save(dict(case=case,omitted_pixel_values=pixel_identity),file)
            restored=torch.load(file,map_location='cpu',weights_only=True)
            need(set(restored)=={'case','omitted_pixel_values'} and restored['omitted_pixel_values']==pixel_identity
                 and tensor_tree(torch,p,restored['case'])==expected,'Compact CPU serialization roundtrip differs')
            descriptor=dict(assignment,file=str(file),sha256=sha(file),bytes=file.stat().st_size,
                raw_row_sha256=row['raw_row_sha256'],metadata_sha256=object_sha(case['metadata']),
                compact_identity_sha256=object_sha(expected),omitted_pixel_values=pixel_identity,
                grid_identity=p.tensor_info(case['inputs']['image_grid_thw']),pixel_bytes=pixels.numel()*pixels.element_size(),
                target_text=case['target_text'],target_ids=case['target_ids'],**widths(case),
                prepare_seconds=prepared,pixel_identity_seconds=identity_seconds,
                reference_case_exact=None if row['sid'] not in reference_ids else True)
            descriptor['staging_seconds']=time.perf_counter()-tick
            stream.write(json.dumps(descriptor,sort_keys=True,allow_nan=False)+'\n');stream.flush();records.append(descriptor)
            del restored,case,pixels,full_tree,expected
    need(len(records)==1000 and sum(r['n'] for r in records)==6200,'Complete compact shard required')
    save(out/'cases.json',records);save(out/'reference_cases.json',references);save(out/'diagnostic_sids.json',diagnostics)
    save(out/'target_inventory.json',target_inventory(records));save(out/'timings.json',timings(records));save(out/'envelopes.json',envelopes(records,references))
    artifacts={str(f):sha(f) for f in out.glob('*.json')};artifacts[str(journal)]=sha(journal)
    result=common_plan('shard',bindings,artifacts,shard_index=index,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
        cases_file=str(out/'cases.json'),reference_cases_file=str(out/'reference_cases.json'),diagnostic_sids_file=str(out/'diagnostic_sids.json'),
        target_inventory_file=str(out/'target_inventory.json'),timings_file=str(out/'timings.json'),envelopes_file=str(out/'envelopes.json'),
        setup_seconds=setup,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        compact_file_hashes_are_in_bound_cases=True)
    save(out/'plan.json',result)
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),shard_index=index,
        array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],worlds=1000,frames=6200)


def verify_report(path,phase,streaming=False):
    path=Path(path).resolve();summary=read(path)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==PROTOCOL and summary['phase']==phase
         and summary['policy']==POLICY and summary['source_sha256']==sources() and summary['inherited_source_sha256']==inherited_sources()
         and not (path.parent/'failure.json').exists(),'Passed exact compact preparation required')
    need(0<summary['elapsed_seconds']<=POLICY['cpu_seconds_per_shard' if phase=='shard' else 'merge_cpu_seconds']
         and sha(summary['plan_file'])==summary['plan_sha256'],'Compact plan/time changed')
    plan=read(summary['plan_file'])
    need(plan['phase']==phase and plan['protocol']==PROTOCOL and plan['policy']==POLICY
         and plan['source_sha256']==sources() and plan['inherited_source_sha256']==inherited_sources()
         and plan['no_training_or_inference_release'] is True,'Compact plan policy/sources differ')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Bound compact input/artifact changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Archived compact source changed')
    need(plan['cases_file'] in plan['artifacts'],'Case descriptor inventory is not bound');records=read(plan['cases_file'])
    if not streaming:
        for record in records:need(sha(record['file'])==record['sha256'] and Path(record['file']).stat().st_size==record['bytes'],'Compact tensor packet changed')
    return plan,records,dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'])


def merge(paths,out,started):
    bindings={};parent,train,assigned,diagnostics=dependencies(bindings);records=[];indices=set();arrays=set();reports=[];reference=None
    for path in paths:
        check_time(started,'merge');plan,items,descriptor=verify_report(path,'shard');index=plan['shard_index']
        need(index in range(4) and index not in indices,'Four distinct compact shard indices required');indices.add(index);arrays.add(plan['array_job_id'])
        expected=[r for r in assigned if r['shard_index']==index]
        need(len(items)==1000 and [{k:r[k] for k in expected[0]} for r in items]==expected,'Exact original shard membership/order differs')
        current=read(plan['reference_cases_file'])
        need(reference is None or reference==current,'Prepared reference envelopes differ between shards');reference=current
        need(read(plan['diagnostic_sids_file'])==diagnostics and read(plan['target_inventory_file'])==target_inventory(items)
             and read(plan['timings_file'])==timings(items) and read(plan['envelopes_file'])==envelopes(items,current),
             'Shard derived inventory differs')
        need(plan['native_identity']==parent['native_identity'] and plan['native_identity_sha256']==parent['native_identity_sha256'],
             'Shard native identity differs')
        for item in items:
            row=train[item['index']]
            need(item['raw_row_sha256']==row['raw_row_sha256'] and item['target_text']==preparation.canonical_target(row['atype'],row['answer'])
                 and item['target_ids']==parent['targets']['targets'][item['target_text']]['ids']
                 and item['target_rows']==len(item['target_ids']) and item['bytes']>0 and item['pixel_bytes']>0
                 and 0<item['prepare_seconds']<=item['staging_seconds'] and item['pixel_identity_seconds']>=0,
                 'Original row/target/timing ownership differs')
        records.extend(items);reports.append(descriptor)
        bind(descriptor['file'],bindings,descriptor['sha256']);bind(descriptor['plan_file'],bindings,descriptor['plan_sha256'])
        for file,digest in plan['artifacts'].items():bind(file,bindings,digest)
    records.sort(key=lambda r:r['index'])
    need(indices==set(range(4)) and len(arrays)==1 and len(records)==4000
         and [{k:r[k] for k in assigned[0]} for r in records]==assigned and sum(r['n'] for r in records)==24800,
         'Complete single-array original4000 order required')
    need({r['sid'] for r in records if r['reference_case_exact'] is True}=={r['sid'] for r in reference},'All eight reprocessed reference cases must match')
    envelope=envelopes(records,reference);save(out/'rows.json',train);save(out/'cases.json',records);save(out/'diagnostic_sids.json',diagnostics)
    save(out/'envelopes.json',envelope);save(out/'target_inventory.json',target_inventory(records));save(out/'timings.json',timings(records))
    analysis=dict(passed=True,worlds=4000,frame_occurrences=24800,diagnostic_sids=100,compact_bytes=sum(r['bytes'] for r in records),
        omitted_pixel_bytes=sum(r['pixel_bytes'] for r in records),fit_profile_envelope_eligible=envelope['fit_profile_envelope_eligible'],
        envelope_excess_worlds=envelope['uncovered_count'],no_merge_tensor_or_image_load=True,no_training_or_inference_release=True)
    save(out/'analysis.json',analysis)
    (out/'REPORT.md').write_text('# Original MMReD compact training inputs\n\nAll4,000 original training rows and the fixed100 diagnostic SIDs are bound. Only pixel_values was omitted from each prepared case; its exact tensor identity is retained. Width excess is descriptive and holds a later fit rather than failing this input stage. No native features, model calls, training or inference were released.\n')
    artifacts={str(f):sha(f) for f in out.glob('*.json')};artifacts[str(out/'REPORT.md')]=sha(out/'REPORT.md')
    plan=common_plan('merge',bindings,artifacts,rows_file=str(out/'rows.json'),cases_file=str(out/'cases.json'),
        diagnostic_sids_file=str(out/'diagnostic_sids.json'),envelopes_file=str(out/'envelopes.json'),
        target_inventory_file=str(out/'target_inventory.json'),timings_file=str(out/'timings.json'),
        shard_reports=reports,array_job_id=next(iter(arrays)),analysis=analysis,native_identity=parent['native_identity'],
        native_identity_sha256=parent['native_identity_sha256'],system_prompt=parent['system_prompt'],
        compact_file_hashes_are_in_bound_cases=True)
    save(out/'plan.json',plan)
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),worlds=4000,frames=24800,
        array_job_id=next(iter(arrays)),fit_profile_envelope_eligible=envelope['fit_profile_envelope_eligible'])


def verify_stage(path,streaming=False):
    plan,records,_=verify_report(path,'merge',streaming=streaming)
    need(len(records)==4000 and plan['analysis']['worlds']==4000 and sum(r['n'] for r in records)==24800,
         'Complete compact training cohort required')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--shard',type=int,choices=range(4));group.add_argument('--merge',nargs=4,type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')
         and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core CPU Slurm only')
    phase='shard' if args.shard is not None else 'merge';started=time.perf_counter()
    if phase=='shard':
        need(os.environ.get('SLURM_ARRAY_JOB_ID') and int(os.environ.get('SLURM_ARRAY_TASK_COUNT','0'))==4
             and int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))==args.shard,'Exactly four CPU array tasks required')
        array=os.environ['SLURM_ARRAY_JOB_ID'];tag=f'shard_{array}_{args.shard}'
        need(all(p.name.split('_')[1]==array for p in OUT.glob('shard_*')),'Only one compact preparation array is released')
    else:tag=f'merge_{os.environ["SLURM_JOB_ID"]}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);frozen=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**frozen,**inherited}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Source copy changed')
    save(out/'request.json',dict(phase=phase,shard=args.shard,merge=None if args.merge is None else [str(p.resolve()) for p in args.merge],
        policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited))
    try:
        if phase=='shard':
            data=DATA/tag;data.mkdir(parents=True,exist_ok=False);value=shard(args.shard,out,data,started)
        else:value=merge(args.merge,out,started)
        check_time(started,phase);need(sources()==frozen and inherited_sources()==inherited,'Compact preparation source changed')
        save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,phase=phase,policy=POLICY,
            source_sha256=frozen,inherited_source_sha256=inherited,elapsed_seconds=time.perf_counter()-started,
            no_training_or_inference_release=True,**value))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),phase=phase,source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True,no_training_or_inference_release=True));raise


if __name__=='__main__':main()
