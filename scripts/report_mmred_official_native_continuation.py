"""All1000-output audit of a resource-only continuation; old failures stay failed."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import continue_mmred_official_native_memory as producer
from scripts import report_mmred_official_native_evaluation as previous
from scripts import report_mmred_official_native_training as training_audit
from scripts import report_mmred_official_native_memory as reference
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=reference.p
read=reference.read;bind=reference.bind;record=reference.record;packet=reference.packet
original_producer=previous.producer
PROTOCOL='mmred_official_native_continuation_independent_audit'
PROPOSAL='docs/paper/MMRED_OFFICIAL_NATIVE_CONTINUATION_AUDIT_PROPOSAL.md'
PROPOSAL_SHA='02a675f7b14318b04c323f990ada4aad297807d6cb73a077d196273a0eb735be'
OWN=('scripts/report_mmred_official_native_continuation.py',PROPOSAL,'slurm/mmred_official_native_continuation_report.sbatch')
OUT=producer.OUT
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_continuation_audit')
POLICY=dict(previous.POLICY,original_worlds_per_arm=3,new_worlds_per_arm=997,
    new_gpu_seconds_cap=14340,cumulative_gpu_seconds_cap_per_arm=14400,original_array_gpu_seconds=98,
    original_failures_preserved=True,original_outputs_rerun=False)
PREDICTION_KEYS=previous.PREDICTION_KEYS+('origin','source_run','source_execution_index')
expected_cohort=previous.expected_cohort
projection=previous.projection
main_report_metadata=previous.main_report_metadata
final_state=previous.final_state
audit_natural=previous.audit_natural


def inherited_sources():
    result={}
    for mapping in (producer.inherited_sources(),producer.sources(),previous.sources()):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Continuation audit source conflict');result[name]=digest
    return result


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held evaluation audit proposal changed')
    return {name:sha(REPO/name) for name in OWN}

def snapshot(out,inherited):
    own=sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        need(sha(REPO/name)==digest,'Evaluation audit source changed')
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Evaluation audit source archive differs')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',inherited);return own

def inputs(torch,path,bindings):
    path=Path(path).resolve();directory=path if path.is_dir() else path.parent
    need(directory.parent==OUT and re.fullmatch(r'run_\d+_[012]',directory.name),'Registered evaluation array/arm directory required')
    config=record(directory/'config.json',bindings);original=record(directory/'analysis.json',bindings)
    need(config['protocol']==producer.PROTOCOL and config['policy']==producer.POLICY
         and config['source_sha256']==producer.sources() and config['inherited_source_sha256']==producer.inherited_sources()
         and config['arm'] in original_producer.profile.ARMS,'Exact frozen evaluation policy/source required')
    arm=config['arm'];need(directory.name==f'run_{config["array_job_id"]}_{original_producer.profile.ARMS.index(arm)}','Evaluation directory/arm differs')
    reference.archive(directory,config['source_sha256'],config['inherited_source_sha256'],bindings)
    bind(config['plan_file'],bindings,config['plan_sha256']);plan=producer.verify_plan(config['plan_file'])
    reference.archive(Path(config['plan_file']).parent,plan['source_sha256'],plan['inherited_source_sha256'],bindings)
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():bind(file,bindings,digest)
    summary=record(directory/'summary.json',bindings)
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='run'
         and summary['protocol']==producer.PROTOCOL and summary['policy']==producer.POLICY
         and summary['source_sha256']==config['source_sha256'] and summary['inherited_source_sha256']==config['inherited_source_sha256']
         and summary['analysis_file']==str(directory/'analysis.json') and summary['analysis_sha256']==sha(directory/'analysis.json')
         and summary['counters']==original['counters']
         and not (directory/'failure.json').exists(),'Complete computational evaluation required')
    need(summary['artifacts_file']==str(directory/'artifacts.json'),'Evaluation artifact manifest path differs')
    artifacts=record(directory/'artifacts.json',bindings,summary['artifacts_sha256'])
    for file,digest in artifacts.items():bind(file,bindings,digest)
    need(record(directory/'base_before.json',bindings)==record(directory/'base_after.json',bindings),'Frozen native base changed')
    need(Path(config['data_directory'])==producer.DATA/directory.name,'Evaluation data outside designated root')
    rows=record(plan['rows_file'],bindings);items=record(plan['cases_file'],bindings);order,prefix=expected_cohort(rows)
    need(record(plan['order_file'],bindings)==order and plan['prefix_indices']==prefix,'Prospective first-per-length execution order differs')
    stage=original_producer.features.verify_stage(plan['feature_stage']['file'],streaming=True)
    stage_summary=record(plan['feature_stage']['file'],bindings,plan['feature_stage']['sha256'])
    record(stage_summary['plan_file'],bindings,stage_summary['plan_sha256'])
    prepared=original_producer.features.verify_preparation(stage['compact_stage']['file'],streaming=True)
    for value in (stage,prepared):
        for mapping in (value['input_bindings'],value['artifacts']):
            for file,digest in mapping.items():bind(file,bindings,digest)
    compacts=record(stage['cases_file'],bindings);features=record(stage['feature_index_file'],bindings)
    need(len(compacts)==len(features)==len(items)==1000 and rows==record(stage['rows_file'],bindings)
         and items==[dict(compact=c,features=f,row=r) for c,f,r in zip(compacts,features,rows)]
         and stage['compact_stage']==plan['compact_stage'] and plan['native_identity']==stage['native_identity']
         and plan['native_identity_sha256']==stage['native_identity_sha256'],'Exact feature/compact/original-row ownership differs')
    for i,item in enumerate(items):
        c,f,r=item['compact'],item['features'],item['row']
        need(c['index']==f['index']==i and all(c[k]==f[k]==r[k] for k in ('sid','n','qtype','pilot_role'))
             and c['raw_row_sha256']==r['raw_row_sha256'] and re.fullmatch(r'[0-9a-f]{64}',r['world_sha256']) is not None
             and f['compact']==dict(file=c['file'],sha256=c['sha256'],compact_identity_sha256=c['compact_identity_sha256'])
             and f['input_identity']==dict(pixel_values=c['omitted_pixel_values'],image_grid_thw=c['grid_identity'])
             and f['tensor']['shape']==[196*r['n'],3584] and f['tensor']['dtype']=='torch.float16' and f['finite'],
             'Evaluation compact/native feature metadata differs')
    mains={}
    for name,parent in plan['main_arms'].items():
        value=main_report_metadata(parent['main_report'],bindings);mains[name]=value
        need(value['arm']==name and value['run']==parent['main_run'] and value['final_checkpoint']==parent['final_checkpoint'],
             'Unconditional final training endpoint differs')
    need(set(mains)==set(original_producer.profile.ARMS),'All three completed fixed training endpoints required')
    owner=plan['main_arms'][arm]
    need(config['final_checkpoint']==original['final_checkpoint']==owner['final_checkpoint']
         and config['main_report']==owner['main_report'] and config['peft_config']==owner['peft_config']
         and config['native_identity']==plan['native_identity'] and config['native_identity_sha256']==plan['native_identity_sha256']
         and config['feature_stage']==plan['feature_stage'] and config['compact_stage']==plan['compact_stage']
         and config['main_run']==owner['main_run'] and config['main_plan']==owner['main_plan']
         and config['profile_preparation']==owner['profile_preparation'] and config['statistics_protocol']==plan['statistics_protocol'],
         'Actual evaluation checkpoint/native ownership differs')
    boundary_summary=record(plan['boundary_report']['file'],bindings,plan['boundary_report']['sha256'])
    boundary=record(boundary_summary['analysis_file'],bindings,boundary_summary['analysis_sha256'])
    need(plan['projection']=={name:projection(boundary,name,rows) for name in original_producer.profile.ARMS}
         and all(v['passed'] for v in plan['projection'].values()),'Independent initial evaluation projection differs')
    from transformers import AutoProcessor,__version__ as version
    processor=AutoProcessor.from_pretrained(str(original_producer.profile.preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(p.fingerprint(processor,str(version))==plan['native_identity']['processor'] and processor.tokenizer.eos_token_id==151645,
         'Actual evaluation tokenizer ownership differs')
    return directory,config,original,plan,items,rows,processor,stage,mains,boundary


def original_evidence(config,plan,rows,boundary,bindings):
    arm=config['arm'];ref=plan['original_runs'][arm]
    need(config['original_plan']==plan['original_plan'] and config['resource_only_continuation'], 'Explicit resource-only original plan binding required')
    need(config['original_run']==ref and ref['arm']==arm and ref['array_job_id']=='444052'
         and ref['allocation_id']=='444052_'+str(original_producer.profile.ARMS.index(arm))
         and ref['allocation_seconds']=={'ordinary':33,'normalized':32,'mass':33}[arm], 'Exact original failed allocation required')
    directory=Path(ref['directory']);need(directory==original_producer.OUT/f'run_{ref["array_job_id"]}_{original_producer.profile.ARMS.index(arm)}',
        'Original resource-failed run directory differs')
    values={}
    for key,name in (('config','config.json'),('failure','failure.json'),('request','request.json'),
                     ('prefix_natural','prefix_natural.json'),('prefix_projection','prefix_projection.json')):
        need(Path(ref[key]['file'])==directory/name,'Original evidence descriptor path differs')
        values[key]=record(ref[key]['file'],bindings,ref[key]['sha256'])
    old=values['config'];failure=values['failure'];request=values['request'];prefix=values['prefix_natural'];gate=values['prefix_projection']
    need(not (directory/'summary.json').exists() and not (directory/'analysis.json').exists()
         and old['protocol']==original_producer.PROTOCOL and old['policy']==original_producer.POLICY
         and old['source_sha256']==original_producer.sources() and old['inherited_source_sha256']==original_producer.inherited_sources()
         and old['arm']==arm and old['array_job_id']=='444052', 'Original failed evaluator source/policy must remain unchanged')
    reference.archive(directory,old['source_sha256'],old['inherited_source_sha256'],bindings)
    old_plan=record(plan['original_plan']['file'],bindings,plan['original_plan']['sha256'])
    need(old['plan_file']==plan['original_plan']['file'] and old['plan_sha256']==plan['original_plan']['sha256']
         and old_plan==original_producer.verify_plan(plan['original_plan']['file']),'Original passed CPU evaluation plan changed')
    for key in ('cases_file','rows_file','order_file','prefix_indices','main_arms','feature_stage','compact_stage','native_identity',
                'native_identity_sha256','statistics_protocol','boundary_report','projection'):
        need(plan[key]==old_plan[key],'Resource continuation changed original '+key)
    need(all(old[k]==config[k] for k in ('final_checkpoint','main_report','main_run','main_plan','feature_stage','compact_stage',
        'profile_preparation','peft_config','state_proof','provenance','native_identity','native_identity_sha256','statistics_protocol')),
        'Original/new fitted model or input provenance differs')
    need(failure['type']=='ValueError' and failure['phase']=='run'
         and failure['message']=='Measured first three worlds hold remaining evaluation beyond10800s'
         and failure['source_sha256']==old['source_sha256'] and failure['no_automatic_retry']
         and request['phase']=='run' and request['arm']==original_producer.profile.ARMS.index(arm)
         and request['plan']==old['plan_file'] and request['source_sha256']==old['source_sha256'],
         'Only preserved original resource-gate failures are eligible for this separate completion')
    _,indices=expected_cohort(rows)
    need([r['index'] for r in prefix]==indices and [r['execution_index'] for r in prefix]==[0,1,2]
         and failure['progress']['natural']==prefix,'Exact original three saved outcomes required')
    G=sum(len(r['generated_ids']) for r in prefix);counters=dict(model=G,backbone=G,visual=0,language=G,norm=G,head=G,backward=0,optimizer=0)
    need(ref['counters']==failure['progress']['counters']==counters,'Original three-output native counters differ')
    for row in prefix:
        need(record(directory/f'natural_{row["index"]:04d}.json',bindings)==row
             and row['fifty_token_seconds']==row['seconds']*50/len(row['generated_ids']), 'Original durable outcome/timing metadata changed')
        bind(row['file'],bindings,row['sha256'])
    expected=projection(boundary,arm,rows,prefix,gate['elapsed_at_prefix_gate'])
    need(gate==expected and gate['passed'] is False and gate['projected_seconds']>10800
         and gate['N32_generation_measured'], 'Original resource failure was not reproduced unchanged')
    return old,prefix,gate,counters


def resources(config,out):
    wrappers=[name for name in producer.OWN if name.endswith('_run.sbatch')]
    need(len(wrappers)==1,'One continuation run wrapper required')
    match=re.search(r'^#SBATCH -J ([^\n]+)$',(REPO/wrappers[0]).read_text(),re.M)
    need(match is not None,'Continuation wrapper must declare its actual accounting name')
    new_name=match.group(1);old_name='mmred_official_native_evaluation_run';array=str(config['array_job_id'])
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(raw)
    rows=[];names={old_name,new_name,'mmred_official_native_evaluation',producer.PROTOCOL}
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=8 or fields[1] not in names:continue
        identifier,name,partition,state,exit_code,seconds,tres,limit=fields
        need((name==old_name and re.fullmatch(r'444052_[012]',identifier))
             or (name==new_name and re.fullmatch(re.escape(array)+r'_[012]',identifier)), 'Additional evaluation/continuation allocation is not released')
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=exit_code,seconds=int(seconds),
            gpus=int(generic[0]) if generic else sum(map(int,typed)),time_limit_minutes=int(limit),tres=tres))
    old=[r for r in rows if r['name']==old_name];new=[r for r in rows if r['name']==new_name]
    need(len(old)==len(new)==3 and {r['job_id'] for r in old}=={'444052_'+str(i) for i in range(3)}
         and {r['job_id'] for r in new}=={array+'_'+str(i) for i in range(3)}, 'Exact old/new three-arm arrays required')
    old_seconds={'444052_0':33,'444052_1':32,'444052_2':33}
    need(all(r['partition']=='gpu' and r['state']=='FAILED' and r['exit_code']=='1:0' and r['gpus']==1
         and r['seconds']==old_seconds[r['job_id']] and r['time_limit_minutes']==180 for r in old)
         and sum(r['seconds'] for r in old)==98, 'Original98 GPU-seconds and failed allocation states must be preserved')
    need(all(r['partition']=='gpu' and r['seconds']<=14340 and r['time_limit_minutes']==239 and r['gpus'] in (0,1) for r in new),
         'Continuation per-arm14340-second allocation bound differs')
    ledger=[]
    for index,arm in enumerate(original_producer.profile.ARMS):
        before=next(r for r in old if r['job_id']=='444052_'+str(index));after=next(r for r in new if r['job_id']==array+'_'+str(index))
        need(before['seconds']+after['seconds']<=14400,'Cumulative per-arm14400-second cap exceeded')
        ledger.append(dict(arm=arm,original=before,continuation=after,cumulative_seconds=before['seconds']+after['seconds']))
    own=next(r for r in ledger if r['arm']==config['arm'])
    need(own['continuation']['state']=='COMPLETED' and own['continuation']['exit_code']=='0:0' and own['continuation']['gpus']==1,
         'Own continuation allocation did not complete')
    return dict(passed=True,original_array_job_id='444052',continuation_array_job_id=array,own=own,all_arms=ledger,
        original_total_gpu_seconds=98,maximum_new_seconds_per_arm=14340,maximum_cumulative_seconds_per_arm=14400,
        maximum_cumulative_total_gpu_seconds=43200,raw_file=str(file),raw_sha256=sha(file))


def continuation_projection(boundary,arm,old_ref,old_gate,new_setup=None):
    reserve=max(old_gate['elapsed_at_prefix_gate'],boundary['projection']['arms'][arm]['setup_seconds'])
    setup=reserve if new_setup is None else new_setup
    units=old_gate['fifty_token_seconds_by_n'];counts=old_gate['remaining_counts']
    need(set(units)==set(counts)=={'8','16','32'} and sum(counts.values())==997,'Exact remaining evaluation timing population differs')
    work=sum(counts[str(n)]*units[str(n)] for n in (8,16,32));new_seconds=setup+1.25*work+60
    total=old_ref['allocation_seconds']+new_seconds
    need(all(math.isfinite(v) and v>0 for v in (setup,work,new_seconds,total)),'Finite continuation projection required')
    return dict(passed=new_seconds<=14340 and total<=14400,projected_seconds=total,new_projected_seconds=new_seconds,
        cap_seconds=14400,continuation_cap_seconds=14340,original_allocated_seconds=old_ref['allocation_seconds'],
        new_setup_seconds=setup,new_setup_is_measured=new_setup is not None,cpu_setup_reserve_seconds=reserve,
        remaining_worlds=997,remaining_counts=counts,fifty_token_seconds_by_n=units,
        original_projection=old_gate,original_projection_preserved_false=True,maximum_new_tokens=50,
        accuracy_not_used=True,empirical_estimate_not_guarantee=True)


def audit(torch,args,out,data,bindings,progress,started):
    directory,config,original,plan,items,rows,processor,stage,mains,boundary=inputs(torch,args.run,bindings)
    own=snapshot(out,inherited_sources());arm=config['arm'];tests=training_audit.parser_proof();save(out/'parser_proof.json',tests)
    old_config,old_rows,old_gate,old_counters=original_evidence(config,plan,rows,boundary,bindings)
    old_ref=plan['original_runs'][arm];final,state_proof=final_state(torch,config,original,plan,bindings)
    old_final,old_state_proof=final_state(torch,old_config,old_config,plan,bindings)
    need(state_proof==old_state_proof,'Original and continuation must use the identical final checkpoint');del old_final
    parent=plan['main_arms'][arm]['profile_preparation']
    native_plan=record(parent['plan_file'],bindings,parent['plan_sha256']);modules=reference.native_modules(torch,native_plan,bindings)
    need(original['protocol']==producer.PROTOCOL and original['arm']==arm and original['passed'] is original['completed'] is True
         and original['all1000_completed'] and original['requires_independent_cpu_audit']
         and original['computational_completion_only'] and not original['objective_achieved']
         and original['statistics_protocol']==plan['statistics_protocol'], 'Complete resource-only continuation analysis required')
    order,prefix=expected_cohort(rows);remaining=[i for i in range(1000) if i not in prefix]
    need(record(plan['remaining_order_file'],bindings)==remaining
         and [r['index'] for r in original['natural']]==list(range(1000)), 'Exact disjoint original3 and remaining997 outputs required')
    imported={r['index']:r for r in old_rows}
    expected_imported=[dict(row,origin='original',source_run=old_ref['directory'],source_execution_index=row['execution_index']) for row in old_rows]
    need(record(directory/'imported_natural.json',bindings)==expected_imported, 'Imported manifest differs from exact original metadata')
    metrics=[];predictions=[];totals=Counter();tables=defaultdict(Counter)
    progress.update(natural_audits=metrics,completed_worlds=0)
    for row in original['natural']:
        need(time.perf_counter()-started<7200,'Fixed7200-second continuation CPU audit cap exceeded')
        index=row['index'];item=items[index];is_original=index in imported
        need(row['execution_index']==order.index(index) and all(row[k]==item['row'][k] for k in ('sid','n','qtype','pilot_role'))
             and row['max_reserved_bytes']>=row['max_allocated_bytes']>0 and math.isfinite(row['seconds']) and row['seconds']>0
             and row['fifty_token_seconds']==row['seconds']*50/len(row['generated_ids']), 'Combined output metadata/history cost differs')
        if is_original:
            before=imported[index]
            need({k:row[k] for k in before}==before and row['origin']=='original' and row['source_run']==old_ref['directory']
                 and row['source_execution_index']==before['execution_index'], 'Imported outcome was changed, rerun or reassigned')
            context=old_config
        else:
            need(row['origin']=='continuation' and row['source_run']==str(directory)
                 and row['source_execution_index']==remaining.index(index)
                 and Path(row['file'])==Path(config['data_directory'])/f'natural_{index:04d}.pt', 'New997 output ownership differs')
            need(record(directory/f'natural_{index:04d}.json',bindings)==row,'New durable scene metadata differs');context=original
        raw=packet(torch,row,bindings);case,features=training_audit.load_case(torch,item,bindings)
        need(case['metadata']['n']==row['n'] and case['metadata']['qtype']==row['qtype']
             and case['metadata']['raw_row_sha256']==item['row']['raw_row_sha256']
             and case['metadata']['question']==item['row']['question'] and case['metadata']['atype']==item['row']['atype']
             and case['target_text']==original_producer.features.preparation.canonical_target(item['row']['atype'],item['row']['answer']),
             'Actual official input/target metadata differs')
        metric=audit_natural(torch,raw,row,case,dict(item,_features=features),arm,context,final,processor,modules,data,f'natural_{index}')
        metric.update(origin=row['origin'],source_run=row['source_run'],source_execution_index=row['source_execution_index'])
        file=out/f'natural_{index:04d}.json';save(file,metric)
        metrics.append(dict(index=index,sid=row['sid'],origin=row['origin'],passed=metric['passed'],head_calls=metric['head_calls'],
            head_rows=metric['head_rows'],memory_passed=None if metric['memory'] is None else metric['memory']['passed'],file=str(file),sha256=sha(file)))
        predictions.append({k:row[k] for k in PREDICTION_KEYS});progress['completed_worlds']=len(metrics)
        score=metric['score'];counts=dict(contexts=1,correct=int(score['correct']),format_valid=int(score['format_valid']),
            semantic_match=int(score['semantic_match']),completed=int(score['completed']),truncated=int(score['truncated']),output_tokens=score['output_tokens'])
        totals.update(counts);tables[f'{row["pilot_role"]}:{row["n"]}:{row["qtype"]}'].update(counts);del raw,case,features,metric
    G=sum(v['head_rows'] for v in metrics);oldG=sum(v['head_rows'] for v in metrics if v['origin']=='original');newG=G-oldG
    def counts(tokens):return dict(model=tokens,backbone=tokens,visual=0,language=tokens,norm=tokens,head=tokens,backward=0,optimizer=0)
    need(len(metrics)==len(predictions)==1000 and Counter(v['origin'] for v in metrics)==Counter(original=3,continuation=997)
         and 3<=oldG<=150 and 997<=newG<=49850 and G<=50000 and original['head_rows']==newG
         and original['counters']==counts(newG) and old_counters==counts(oldG) and original['combined_counters']==counts(G)
         and original['combined_head_rows']==G and original['imported_counters']==counts(oldG) and original['decoder_layer_calls']==[newG]*28
         and original['imported_decoder_layer_counts_recorded'] is False and original['combined_expected_decoder_layer_calls']==[G]*28
         and totals['contexts']==1000 and totals['output_tokens']==G and len(tables)==20 and all(v['contexts']==50 for v in tables.values()),
         'Actual original/new/combined native work inventory differs')
    expected_lineage=dict(original_run=old_ref,original_plan=plan['original_plan'],original_failure_preserved=True,
        imported_worlds=3,new_worlds=997,combined_worlds=1000,imported_indices=prefix,new_indices=remaining,
        imported_counters=counts(oldG),new_counters=counts(newG),combined_counters=counts(G),imported_decoder_layer_counts_recorded=False,
        original_allocated_seconds=old_ref['allocation_seconds'],continuation_array_job_id=config['array_job_id'],
        continuation_cap_seconds=14340,total_per_arm_cap_seconds=14400)
    need(original['original_run']==old_ref and original['continuation_lineage']==expected_lineage
         and read(directory/'summary.json')['continuation_lineage']==expected_lineage, 'Producer combined continuation lineage differs')
    cpu_projection=continuation_projection(boundary,arm,old_ref,old_gate)
    measured_projection=continuation_projection(boundary,arm,old_ref,old_gate,original['setup_seconds'])
    need(plan['continuation_projection'][arm]==cpu_projection and cpu_projection['passed']
         and original['continuation_projection']==measured_projection and measured_projection['passed']
         and record(directory/'continuation_projection.json',bindings)==measured_projection
         and original['prefix_projection']==old_gate and original['original_projection']==plan['projection'][arm],
         'Independent original-plus-new resource projection differs')
    allocation=resources(config,out);summary=read(directory/'summary.json')
    need(0<summary['elapsed_seconds']<=14340 and old_ref['failure_elapsed_seconds']==read(old_ref['failure']['file'])['elapsed_seconds'],
         'Separate original/continuation elapsed records differ')
    own_allocation=allocation['own'];timing_file=stage['timings_file'];feature_timings=record(timing_file,bindings)
    lineage=dict(original_array_job_id='444052',original_run=old_ref,original_indices=prefix,continuation_indices=remaining,
        original_allocation_seconds=own_allocation['original']['seconds'],continuation_allocation_seconds=own_allocation['continuation']['seconds'],
        cumulative_allocation_seconds=own_allocation['cumulative_seconds'],cumulative_cap_seconds=14400,
        original_resource_gate_passed=False,original_failures_preserved=True,original_outputs_rerun=False,
        original_counters=counts(oldG),continuation_counters=counts(newG),combined_counters=counts(G),
        imported_decoder_layer_counts_recorded=False,combined_layer_counts_are_expected_not_observed=True)
    cost=dict(allocation=allocation,setup_seconds=None,continuation_setup_seconds=original['setup_seconds'],original_setup_seconds_available=False,
        evaluation_elapsed_seconds=old_ref['failure_elapsed_seconds']+summary['elapsed_seconds'],
        original_evaluation_elapsed_seconds=old_ref['failure_elapsed_seconds'],continuation_evaluation_elapsed_seconds=summary['elapsed_seconds'],
        inference_seconds=sum(v['seconds'] for v in original['natural']),actual_output_tokens=G,
        peak_allocated_bytes=max(v['max_allocated_bytes'] for v in original['natural']),peak_reserved_bytes=max(v['max_reserved_bytes'] for v in original['natural']),
        main_training=mains[arm]['resources'],shared_evaluation_features=dict(stage=plan['feature_stage'],timings_file=timing_file,
            timings_sha256=sha(timing_file),timings=feature_timings),projection=cpu_projection,prefix_projection=old_gate,
        continuation_projection=measured_projection,empirical_timing_not_production_benchmark=True)
    save(out/'predictions.json',predictions);save(out/'natural_audits.json',metrics)
    return dict(protocol=PROTOCOL,passed=all(v['passed'] for v in metrics),completed=True,arm=arm,policy=POLICY,source_sha256=own,
        inherited_source_sha256=inherited_sources(),run=dict(directory=str(directory),config_file=str(directory/'config.json'),
            config_sha256=sha(directory/'config.json'),analysis_file=str(directory/'analysis.json'),analysis_sha256=sha(directory/'analysis.json'),
            plan_file=config['plan_file'],plan_sha256=config['plan_sha256']),original_run_passed=False,continuation_run_passed=True,
        original_cache_profile_passed=False,cache_reuse_qualified=False,state_audit=state_proof,continuation_lineage=lineage,
        main_report=plan['main_arms'][arm]['main_report'],final_checkpoint=original['final_checkpoint'],evalfeature_stage=plan['feature_stage'],
        native_identity_sha256=plan['native_identity_sha256'],statistics_protocol=plan['statistics_protocol'],
        predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
        original_rows_file=plan['rows_file'],original_rows_sha256=sha(plan['rows_file']),
        natural_audits_file=str(out/'natural_audits.json'),natural_audits_sha256=sha(out/'natural_audits.json'),
        counters=counts(G),cpu_head_calls=G,cpu_head_rows=G,memory_reconstructions=0 if arm=='ordinary' else 1000,
        scores=dict(total=dict(totals),by_panel_length_task={k:dict(v) for k,v in sorted(tables.items())}),resources=cost,tests=tests,
        all_scheduled_numerical_evidence_collected=True,raw_inputs_verified_in_cpu_audit=True,raw_files_rehashed_in_handoff=False,
        no_accuracy_release_gate=True,no_checkpoint_selection=True,no_further_release=True)


def verify_report(path):
    """Verify compact immutable audit output; do not rehash large raw head files."""
    path=Path(path).resolve();summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['passed'] is summary['completed'] is True
         and summary['source_sha256']==sources() and summary['policy']==POLICY
         and not (path.parent/'failure.json').exists(),'Completed independent evaluation audit required')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'],'Evaluation audit analysis changed')
    result=read(summary['analysis_file'])
    need(result['protocol']==PROTOCOL and result['passed'] is result['completed'] is True and result['arm']==summary['arm']
         and result['run']==summary['run'] and result['source_sha256']==summary['source_sha256']
         and result['inherited_source_sha256']==summary['inherited_source_sha256'] and result['policy']==POLICY
         and result['all_scheduled_numerical_evidence_collected'] and result['raw_inputs_verified_in_cpu_audit']
         and result['raw_files_rehashed_in_handoff'] is False and result['no_further_release'],
         'Complete evaluation audit handoff differs')
    for key in ('input_bindings','artifacts'):
        need(sha(summary[key+'_file'])==summary[key+'_sha256'],'Evaluation audit immutable manifest changed')
    reference.archive(path.parent,summary['source_sha256'],summary['inherited_source_sha256'],{})
    need(summary['inherited_source_sha256']==inherited_sources(),'Evaluator source closure changed')
    for name in ('predictions','original_rows','natural_audits'):
        need(sha(result[name+'_file'])==result[name+'_sha256'],'Lean evaluation audit output changed')
    for ref in (result['evalfeature_stage'],result['final_checkpoint'],result['main_report']):
        need(sha(ref['file'])==ref['sha256'],'Evaluation stage/checkpoint reference changed')
    for file_key,sha_key in (('config_file','config_sha256'),('analysis_file','analysis_sha256'),('plan_file','plan_sha256')):
        need(sha(result['run'][file_key])==result['run'][sha_key],'Evaluation producer metadata changed')
    rows=read(result['original_rows_file']);predictions=read(result['predictions_file']);expected_cohort(rows)
    need(len(predictions)==1000 and all(row['index']==i and all(row[k]==rows[i][k] for k in ('sid','n','qtype','pilot_role'))
         and set(row)==set(PREDICTION_KEYS) for i,row in enumerate(predictions)),'Complete lean prediction/source inventory differs')
    lineage=result['continuation_lineage']
    need(result['original_run_passed'] is False and result['continuation_run_passed'] is True
         and lineage['original_failures_preserved'] and lineage['original_outputs_rerun'] is False
         and lineage['original_resource_gate_passed'] is False and lineage['original_array_job_id']=='444052'
         and lineage['cumulative_allocation_seconds']<=14400,'Successful handoff must preserve original resource failures')
    _,indices=expected_cohort(rows);remaining=[i for i in range(1000) if i not in indices]
    need(lineage['original_indices']==indices and lineage['continuation_indices']==remaining
         and all(row['origin']==('original' if row['index'] in indices else 'continuation')
             and row['source_run']==(lineage['original_run']['directory'] if row['index'] in indices else result['run']['directory'])
             and row['source_execution_index']==(indices.index(row['index']) if row['index'] in indices else remaining.index(row['index']))
             for row in predictions),'Lean imported/new origin inventory differs')
    for key in ('failure','config','prefix_natural','prefix_projection','request'):
        ref=lineage['original_run'][key];need(sha(ref['file'])==ref['sha256'],'Original continuation lineage changed')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,
        'CPU-only four-core evaluation audit required')
    import torch
    torch.set_num_threads(4);started=time.perf_counter();out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';data=DATA/out.name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);bindings={};progress={}
    save(out/'request.json',dict(run=str(args.run.resolve()),policy=POLICY,source_sha256=sources()))
    try:
        result=audit(torch,args,out,data,bindings,progress,started);elapsed=time.perf_counter()-started
        need(elapsed<=7200,'Fixed7200-second per-arm CPU audit cap exceeded')
        result['resources']['cpu_audit_elapsed_seconds']=elapsed;save(out/'analysis.json',result);save(out/'input_bindings.json',bindings)
        (out/'REPORT.md').write_text('Independent evaluation CPU audit: '+str(result['passed'])+'.\n\nAll1,000 original validation/test outcomes retained under the unchanged strict typed JSON/EOS parser. Computational validity is separate from scientific qualification; the paired statistical report consumes all three audited arms.\n')
        artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()};save(out/'artifacts.json',artifacts)
        elapsed=time.perf_counter()-started;need(elapsed<=7200,'Fixed CPU audit cap exceeded during publication')
        summary=dict(protocol=PROTOCOL,passed=result['passed'],completed=True,arm=result['arm'],policy=POLICY,run=result['run'],
            source_sha256=result['source_sha256'],inherited_source_sha256=result['inherited_source_sha256'],
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),input_bindings_file=str(out/'input_bindings.json'),
            input_bindings_sha256=sha(out/'input_bindings.json'),artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            raw_inputs_verified_in_cpu_audit=True,raw_files_rehashed_in_handoff=False,no_further_release=True,elapsed_seconds=elapsed)
        save(out/'summary.json',summary);need(result['passed'],'Continuation CPU numerical audit failed after complete evidence collection')
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,
            input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),
            progress=progress,no_further_release=True));raise


if __name__=='__main__':main()
