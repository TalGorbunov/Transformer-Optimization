"""Independent all-row cold evaluation audit; numerical work is CPU Slurm only."""
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
from scripts import evaluate_mmred_official_native_memory as producer
from scripts import report_mmred_official_native_training as training_audit
from scripts import report_mmred_official_native_memory as reference
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=reference.p
read=reference.read;bind=reference.bind;record=reference.record;packet=reference.packet
PROTOCOL='mmred_official_native_evaluation_independent_audit'
PROPOSAL='docs/paper/MMRED_OFFICIAL_NATIVE_EVALUATION_AUDIT_PROPOSAL.md'
PROPOSAL_SHA='aeb2fec7fd83a7b9abbba3cd82e6751944ac710bebfb5e090e62104e52fcb2bd'
OWN=('scripts/report_mmred_official_native_evaluation.py',PROPOSAL,'slurm/mmred_official_native_evaluation_report.sbatch')
OUT=producer.OUT
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_evaluation_audit')
POLICY=dict(cpu_seconds_per_arm=7200,cpu_cores=4,memory_gib=16,worlds_per_arm=1000,
    validation_worlds=400,test_worlds=600,maximum_new_tokens=50,cpu_head_calls_cap_per_arm=50000,
    cpu_head_rows_cap_per_arm=50000,memory_reconstructions_per_memory_arm=1000,
    cpu_head_tv_max=.02,core_atol=2e-4,core_rtol=2e-4,model_calls=0,decoder_calls=0,
    vision_calls=0,backward_calls=0,optimizer_steps=0,cold_only=True,
    no_accuracy_release_gate=True,no_checkpoint_selection=True,no_further_release=True)
PREDICTION_KEYS=('index','sid','n','qtype','pilot_role','generated_ids','primary_text','score','seconds',
    'max_allocated_bytes','max_reserved_bytes','file','sha256')


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


def expected_cohort(rows):
    cells={(role,n,q):50 for role,ns in (('val',(8,16)),('test',(8,16,32))) for n in ns
        for q in ('char_at_frame','steps_in_room','spend_together','where_spend')}
    need(len(rows)==len({r['sid'] for r in rows})==1000
         and Counter((r['pilot_role'],r['n'],r['qtype']) for r in rows)==Counter(cells),'Exact original20x50 evaluation cohort required')
    prefix=[next(i for i,row in enumerate(rows) if row['n']==n) for n in (8,16,32)]
    return prefix+[i for i in range(1000) if i not in prefix],prefix


def projection(boundary,arm,rows,completed=(),elapsed=None):
    value=boundary['projection']['arms'][arm]
    initial={n:value['generation_fifty_token_seconds_by_n'][str(n)] for n in (8,16)};initial[32]=2*initial[16]
    units=dict(initial);done={r['index'] for r in completed}
    need(len(done)==len(completed),'Duplicate measured evaluation timing case')
    for row in completed:units[row['n']]=max(units[row['n']],row['fifty_token_seconds'])
    counts=Counter(row['n'] for index,row in enumerate(rows) if index not in done)
    setup=value['setup_seconds'] if elapsed is None else elapsed
    seconds=setup+1.25*sum(counts[n]*units[n] for n in (8,16,32))+60
    need(all(math.isfinite(v) and v>0 for v in [setup,seconds,*units.values()]),'Finite positive evaluation timing population required')
    return dict(passed=seconds<=10800,projected_seconds=seconds,cap_seconds=10800,setup_or_elapsed_seconds=setup,
        elapsed_at_prefix_gate=elapsed,remaining_worlds=sum(counts.values()),remaining_counts={str(n):counts[n] for n in (8,16,32)},
        initial_fifty_token_seconds_by_n={str(n):initial[n] for n in (8,16,32)},
        fifty_token_seconds_by_n={str(n):units[n] for n in (8,16,32)},completed_indices=[r['index'] for r in completed],
        N32_initial_estimate='2 * audited boundary N16 allowance',N32_generation_measured=any(r['n']==32 for r in completed),
        empirical_estimate_not_guarantee=True,accuracy_not_used=True,maximum_new_tokens=50)


def main_report_metadata(descriptor,bindings):
    summary=record(descriptor['file'],bindings,descriptor['sha256'])
    need(summary['protocol']==training_audit.PROTOCOL and summary['passed'] is summary['completed'] is True
         and summary['analysis_file']==descriptor['analysis_file'] and summary['analysis_sha256']==descriptor['analysis_sha256']
         and summary['source_sha256']==training_audit.sources() and not (Path(descriptor['file']).parent/'failure.json').exists(),
         'Completed independent final-training audit required')
    analysis=record(descriptor['analysis_file'],bindings,descriptor['analysis_sha256'])
    need(analysis['passed'] and analysis['original_run_passed'] and analysis['all_scheduled_numerical_evidence_collected']
         and analysis['run']==summary['run'] and analysis['arm']==summary['arm'] and analysis['no_inference_release']
         and analysis['original_cache_profile_passed'] is False and analysis['cache_reuse_qualified'] is False,
         'Bound independent training audit handoff differs')
    for key in ('input_bindings','artifacts'):record(summary[key+'_file'],bindings,summary[key+'_sha256'])
    reference.archive(Path(descriptor['file']).parent,summary['source_sha256'],summary['inherited_source_sha256'],bindings)
    return analysis


def inputs(torch,path,bindings):
    path=Path(path).resolve();directory=path if path.is_dir() else path.parent
    need(directory.parent==OUT and re.fullmatch(r'run_\d+_[012]',directory.name),'Registered evaluation array/arm directory required')
    config=record(directory/'config.json',bindings);original=record(directory/'analysis.json',bindings)
    need(config['protocol']==producer.PROTOCOL and config['policy']==producer.POLICY
         and config['source_sha256']==producer.sources() and config['inherited_source_sha256']==producer.inherited_sources()
         and config['arm'] in producer.profile.ARMS,'Exact frozen evaluation policy/source required')
    arm=config['arm'];need(directory.name==f'run_{config["array_job_id"]}_{producer.profile.ARMS.index(arm)}','Evaluation directory/arm differs')
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
    stage=producer.features.verify_stage(plan['feature_stage']['file'],streaming=True)
    stage_summary=record(plan['feature_stage']['file'],bindings,plan['feature_stage']['sha256'])
    record(stage_summary['plan_file'],bindings,stage_summary['plan_sha256'])
    prepared=producer.features.verify_preparation(stage['compact_stage']['file'],streaming=True)
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
    need(set(mains)==set(producer.profile.ARMS),'All three completed fixed training endpoints required')
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
    need(plan['projection']=={name:projection(boundary,name,rows) for name in producer.profile.ARMS}
         and all(v['passed'] for v in plan['projection'].values()),'Independent initial evaluation projection differs')
    from transformers import AutoProcessor,__version__ as version
    processor=AutoProcessor.from_pretrained(str(producer.profile.preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(p.fingerprint(processor,str(version))==plan['native_identity']['processor'] and processor.tokenizer.eos_token_id==151645,
         'Actual evaluation tokenizer ownership differs')
    return directory,config,original,plan,items,rows,processor,stage,mains,boundary


def final_state(torch,config,original,plan,bindings):
    arm=config['arm'];owner=plan['main_arms'][arm];ref=owner['final_checkpoint'];value=packet(torch,ref,bindings)
    reference.adapter_state(torch,value['adapter'])
    need(value['arm']==arm and value['updates']==ref['updates']==1500 and value['final_checkpoint']
         and value['peft_config']==owner['peft_config'],'Only exact final1500 checkpoint allowed')
    if arm=='ordinary':need(value['memory'] is None,'Ordinary checkpoint has no memory parameters')
    else:
        reference.memory_state(torch,value['memory'])
        if arm=='normalized':need(bool((value['memory']['mass_direction']==0).all()),'Normalized endpoint acquired mass contribution')
    state_proof=dict(arm=arm,updates=1500,adapter=reference.state_identity(value['adapter']),
        memory=None if value['memory'] is None else reference.state_identity(value['memory']),peft_config=value['peft_config'],
        finite=True,checkpoint_used_for_inference_only=True)
    need(state_proof==owner['state_proof']==config['state_proof'],'Actual checkpoint tensors differ from prepared endpoint')
    contract=config['contract'];expected_tensors={name:dict(shape=list(tensor.shape),dtype=str(tensor.dtype)) for name,tensor in value['adapter'].items()}
    need({name:{k:v for k,v in field.items() if k!='object_id'} for name,field in contract['tensors'].items()}==expected_tensors
         and {k:v for k,v in contract.items() if k!='tensors'}==dict(targets=list(p.TARGETS),trainable_parameters=10092544,
             active_adapter='default',scaling=2.,rank=16,alpha=32,dropout=.05),'Actual frozen inference adapter contract differs')
    provenance=dict(native_identity_sha256=plan['native_identity_sha256'],lora=ref,
        memory=None if arm=='ordinary' else dict(checkpoint=ref,tensors=reference.state_identity(value['memory'])))
    need(original['provenance']==config['provenance']==provenance,'Actual cold generation provenance differs')
    return value,state_proof


def audit_natural(torch,raw,row,case,item,arm,original,final,processor,modules,data,key):
    metric=training_audit.natural_audit(torch,raw,row,case,item,arm,original,final,processor,modules,data,key)
    metric.pop('training_diagnostic_only');metric.pop('memory_reconstruction_scope')
    core=None
    if arm!='ordinary':
        observed=reference.memory_links(torch,raw['evidence'],case,item['features'],original['final_checkpoint'])
        core=reference.memory_audit(torch,observed,item['_features'],case['coordinates'],final['memory'],arm,data,key)
    metric.update(passed=metric['passed'] and (core is None or core['passed']),memory=core,pilot_role=row['pilot_role'],
        memory_reconstruction_scope='every_evaluation_world' if arm!='ordinary' else 'not_applicable',evaluation_only=True)
    return metric


def resources(config,out):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(raw)
    rows=[];array=str(config['array_job_id']);own=array+'_'+str(producer.profile.ARMS.index(config['arm']))
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=8 or fields[1] not in ('mmred_official_native_evaluation','mmred_official_native_evaluation_run'):continue
        identifier,name,partition,state,exit_code,seconds,tres,limit=fields
        need(re.fullmatch(re.escape(array)+r'_[012]',identifier) is not None,'Additional evaluation array/allocation is not released')
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=exit_code,seconds=int(seconds),
            gpus=int(generic[0]) if generic else sum(map(int,typed)),time_limit_minutes=int(limit),tres=tres))
    need(len(rows)==3 and {r['job_id'] for r in rows}=={array+'_'+str(i) for i in range(3)}
         and all(r['name']=='mmred_official_native_evaluation_run' and r['partition']=='gpu' and r['seconds']<=10800 and r['time_limit_minutes']==180 and r['gpus'] in (0,1) for r in rows),
         'Exact three-arm bounded evaluation allocation required')
    current=next(r for r in rows if r['job_id']==own)
    need(current['state']=='COMPLETED' and current['exit_code']=='0:0' and current['gpus']==1,'Own evaluation allocation did not complete')
    return dict(passed=True,array_job_id=array,own=current,all_array_rows=rows,maximum_per_arm_gpu_seconds=10800,
        maximum_total_gpu_seconds=32400,raw_file=str(file),raw_sha256=sha(file))


def audit(torch,args,out,data,bindings,progress,started):
    directory,config,original,plan,items,rows,processor,stage,mains,boundary=inputs(torch,args.run,bindings)
    inherited={**config['inherited_source_sha256'],**config['source_sha256']};own=snapshot(out,inherited)
    arm=config['arm'];tests=training_audit.parser_proof();save(out/'parser_proof.json',tests)
    final,state_proof=final_state(torch,config,original,plan,bindings)
    parent=plan['main_arms'][arm]['profile_preparation']
    native_plan=record(parent['plan_file'],bindings,parent['plan_sha256']);modules=reference.native_modules(torch,native_plan,bindings)
    need(original['protocol']==producer.PROTOCOL and original['arm']==arm and original['passed'] is original['completed'] is True
         and original['all1000_completed'] and original['all_three_arms_required'] and original['requires_independent_cpu_audit']
         and original['computational_completion_only'] and not original['objective_achieved'] and original['accuracy_not_used_for_execution']
         and original['statistics_protocol']==plan['statistics_protocol'] and original['no_checkpoint_copy'],
         'Complete accuracy-independent evaluation analysis required')
    need([r['index'] for r in original['natural']]==list(range(1000)),'All1000 source-ordered native outcomes required')
    order,prefix=expected_cohort(rows);metrics=[];predictions=[];totals=Counter();tables=defaultdict(Counter)
    progress.update(natural_audits=metrics,completed_worlds=0)
    for row in original['natural']:
        need(time.perf_counter()-started<POLICY['cpu_seconds_per_arm'],'Fixed7200-second CPU evaluation audit cap exceeded')
        index=row['index'];item=items[index]
        need(all(row[k]==item['row'][k] for k in ('sid','n','qtype','pilot_role'))
             and order[row['execution_index']]==index and row['max_reserved_bytes']>=row['max_allocated_bytes']>0
             and math.isfinite(row['seconds']) and row['seconds']>0
             and row['fifty_token_seconds']==row['seconds']*50/len(row['generated_ids']),
             'Native output source/cost/execution ownership differs')
        need(record(directory/f'natural_{index:04d}.json',bindings)==row,'Durable scene metadata differs')
        raw=packet(torch,row,bindings);case,features=training_audit.load_case(torch,item,bindings)
        need(case['metadata']['n']==row['n'] and case['metadata']['qtype']==row['qtype']
             and case['metadata']['raw_row_sha256']==item['row']['raw_row_sha256']
             and case['metadata']['question']==item['row']['question'] and case['metadata']['atype']==item['row']['atype']
             and case['target_text']==producer.features.preparation.canonical_target(item['row']['atype'],item['row']['answer']),
             'Actual official question/target metadata differs')
        metric=audit_natural(torch,raw,row,case,dict(item,_features=features),arm,original,final,processor,modules,data,f'natural_{index}')
        file=out/f'natural_{index:04d}.json';save(file,metric)
        metrics.append(dict(index=index,sid=row['sid'],passed=metric['passed'],head_calls=metric['head_calls'],head_rows=metric['head_rows'],
            memory_passed=None if metric['memory'] is None else metric['memory']['passed'],file=str(file),sha256=sha(file)))
        predictions.append({k:row[k] for k in PREDICTION_KEYS});progress['completed_worlds']=len(metrics)
        score=metric['score'];counts=dict(contexts=1,correct=int(score['correct']),format_valid=int(score['format_valid']),
            semantic_match=int(score['semantic_match']),completed=int(score['completed']),truncated=int(score['truncated']),output_tokens=score['output_tokens'])
        totals.update(counts);tables[f'{row["pilot_role"]}:{row["n"]}:{row["qtype"]}'].update(counts)
        del raw,case,features,metric
    G=sum(v['head_rows'] for v in metrics);expected=dict(model=G,backbone=G,visual=0,language=G,norm=G,head=G,backward=0,optimizer=0)
    need(len(metrics)==len(predictions)==1000 and 1000<=G<=50000 and sum(v['head_calls'] for v in metrics)==G
         and original['head_rows']==G and original['counters']==expected and original['decoder_layer_calls']==[G]*28
         and totals['contexts']==1000 and totals['output_tokens']==G and len(tables)==20 and all(v['contexts']==50 for v in tables.values()),
         'Complete all-world native/CPU head inventory and fixed denominators differ')
    measured=[original['natural'][i] for i in prefix]
    saved_prefix=record(directory/'prefix_natural.json',bindings);gate=record(directory/'prefix_projection.json',bindings)
    projected=projection(boundary,arm,rows,measured,gate['elapsed_at_prefix_gate'])
    need(saved_prefix==measured and gate==projected==original['prefix_projection'] and gate['passed'] and gate['N32_generation_measured']
         and original['original_projection']==plan['projection'][arm] and 0<original['setup_seconds']<gate['elapsed_at_prefix_gate'],
         'Independent measured prefix resource projection differs')
    allocation=resources(config,out);summary=read(directory/'summary.json')
    need(0<summary['elapsed_seconds']<=10800,'Evaluation elapsed cap differs')
    timing_file=stage['timings_file'];feature_timings=record(timing_file,bindings)
    cost=dict(allocation=allocation,setup_seconds=original['setup_seconds'],evaluation_elapsed_seconds=summary['elapsed_seconds'],
        inference_seconds=sum(v['seconds'] for v in original['natural']),actual_output_tokens=G,
        peak_allocated_bytes=max(v['max_allocated_bytes'] for v in original['natural']),
        peak_reserved_bytes=max(v['max_reserved_bytes'] for v in original['natural']),
        main_training=mains[arm]['resources'],shared_evaluation_features=dict(stage=plan['feature_stage'],
            timings_file=timing_file,timings_sha256=sha(timing_file),timings=feature_timings),
        projection=plan['projection'][arm],prefix_projection=gate,empirical_timing_not_production_benchmark=True)
    save(out/'predictions.json',predictions);save(out/'natural_audits.json',metrics)
    passed=all(v['passed'] for v in metrics)
    result=dict(protocol=PROTOCOL,passed=passed,completed=True,arm=arm,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,
        run=dict(directory=str(directory),config_file=str(directory/'config.json'),config_sha256=sha(directory/'config.json'),
            analysis_file=str(directory/'analysis.json'),analysis_sha256=sha(directory/'analysis.json'),plan_file=config['plan_file'],plan_sha256=config['plan_sha256']),
        original_run_passed=True,original_cache_profile_passed=False,cache_reuse_qualified=False,state_audit=state_proof,
        main_report=plan['main_arms'][arm]['main_report'],final_checkpoint=original['final_checkpoint'],evalfeature_stage=plan['feature_stage'],
        native_identity_sha256=plan['native_identity_sha256'],statistics_protocol=plan['statistics_protocol'],
        predictions_file=str(out/'predictions.json'),predictions_sha256=sha(out/'predictions.json'),
        original_rows_file=plan['rows_file'],original_rows_sha256=sha(plan['rows_file']),
        natural_audits_file=str(out/'natural_audits.json'),natural_audits_sha256=sha(out/'natural_audits.json'),
        counters=expected,cpu_head_calls=G,cpu_head_rows=G,memory_reconstructions=0 if arm=='ordinary' else 1000,
        scores=dict(total=dict(totals),by_panel_length_task={k:dict(v) for k,v in sorted(tables.items())}),
        resources=cost,tests=tests,all_scheduled_numerical_evidence_collected=True,
        raw_inputs_verified_in_cpu_audit=True,raw_files_rehashed_in_handoff=False,
        no_accuracy_release_gate=True,no_checkpoint_selection=True,no_further_release=True)
    return result


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
    need(summary['inherited_source_sha256']=={**producer.inherited_sources(),**producer.sources()},'Evaluator source closure changed')
    for name in ('predictions','original_rows','natural_audits'):
        need(sha(result[name+'_file'])==result[name+'_sha256'],'Lean evaluation audit output changed')
    for ref in (result['evalfeature_stage'],result['final_checkpoint'],result['main_report']):
        need(sha(ref['file'])==ref['sha256'],'Evaluation stage/checkpoint reference changed')
    for file_key,sha_key in (('config_file','config_sha256'),('analysis_file','analysis_sha256'),('plan_file','plan_sha256')):
        need(sha(result['run'][file_key])==result['run'][sha_key],'Evaluation producer metadata changed')
    rows=read(result['original_rows_file']);predictions=read(result['predictions_file']);expected_cohort(rows)
    need(len(predictions)==1000 and all(row['index']==i and all(row[k]==rows[i][k] for k in ('sid','n','qtype','pilot_role'))
         and set(row)==set(PREDICTION_KEYS) for i,row in enumerate(predictions)),'Complete lean prediction/source inventory differs')
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
        save(out/'summary.json',summary);need(result['passed'],'Evaluation CPU numerical audit failed after complete evidence collection')
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,
            input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),
            progress=progress,no_further_release=True));raise


if __name__=='__main__':main()
