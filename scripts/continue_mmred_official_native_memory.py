"""One resource-only continuation preserving the three original evaluation outcomes."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_mmred_official_native_memory as original
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
training=original.training;profile=original.profile;runtime=original.runtime;answers=original.answers
p=original.p;read=original.read;bind=original.bind
state_proof=original.state_proof;frozen_adapter_contract=original.frozen_adapter_contract
PROTOCOL='mmred_official_native_evaluation_continuation'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_EVALUATION_CONTINUATION.md'
PROPOSAL_SHA='f44eac17b623829ff3896bfce914beef7a19773284448342b5fb2268b66536c5'
OWN=('scripts/continue_mmred_official_native_memory.py',PROPOSAL,
     'slurm/mmred_official_native_continuation_check.sbatch','slurm/mmred_official_native_continuation_run.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native_evaluation_continuation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_evaluation_continuation')
PARENT=original.OUT/'check_444051/plan.json'
PARENT_SHA='b044c69a3a558a5369b093ee23b9cdbbde24923f019eecad19f14f7558b31cb2'
RELEASE=original.OUT/'source_release.json'
RELEASE_SHA='1f047c45163c3dcdffc493c8216dfc0268ca6c0b8c618256b7e0dde1d43f5aa5'
FAILURE_SHAS=('79eaa22366cce901ebf6ab9f0aa7f7ea1937c79fdbed3462c759061cb47129ca',
 'cf34ddbfe8a3810822a746731764e9ff96bb2c41cee24d951d94d83f1c1a7aec',
 'd247fb13ebe442b83e835fd1ac11efec6adc757329b80191bb917ab84d1f53b3')
OLD_SECONDS=(33,32,33)
POLICY=dict(protocol=PROTOCOL,arms=list(profile.ARMS),worlds=1000,imported_worlds=3,new_worlds=997,
    validation_worlds=400,test_worlds=600,lengths=[8,16,32],checkpoint_update=1500,cpu_seconds=600,
    gpu_seconds_per_arm=14340,total_gpu_seconds_per_arm=14400,original_allocated_seconds=list(OLD_SECONDS),
    maximum_new_GPU_seconds=43020,maximum_total_GPU_seconds=43200,maximum_arrays=1,maximum_attempts_per_arm=1,
    cpu_cores=4,memory_gib=16,maximum_new_tokens=50,maximum_new_model_head_calls_per_arm=49850,
    maximum_combined_model_head_calls_per_arm=50000,vision_calls=0,backward_calls=0,optimizer_steps=0,
    cold_full_prefix=True,no_checkpoint_selection=True,no_accuracy_release_gate=True,
    all_validation_and_test_unconditional=True,original_failure_preserved=True,requires_independent_cpu_audit=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Continuation proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={**original.inherited_sources(),**original.sources()}
    need(not set(result)&set(OWN) and all(sha(REPO/name)==digest for name,digest in result.items()),'Continuation dependency changed')
    return result


def archive(out):
    own=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Continuation source archive differs')
    return own,inherited


def call_counts(tokens):
    return dict(model=tokens,backbone=tokens,visual=0,language=tokens,norm=tokens,head=tokens,backward=0,optimizer=0)


def original_runs(parent,bindings):
    bind(PARENT,bindings,PARENT_SHA);release_ref=bind(RELEASE,bindings,RELEASE_SHA);release=read(RELEASE)
    need(release['source_sha256']==original.sources() and release['inherited_source_sha256']==original.inherited_sources()
         and release['policy']==original.POLICY,'Original source release changed')
    rows=read(parent['rows_file']);boundary=read(read(parent['boundary_report']['file'])['analysis_file']);result={}
    for i,arm in enumerate(profile.ARMS):
        directory=original.OUT/f'run_444052_{i}';refs={}
        for key in ('failure','config','prefix_natural','prefix_projection','request'):
            refs[key]=bind(directory/(key+'.json'),bindings,FAILURE_SHAS[i] if key=='failure' else None)
        failure=read(refs['failure']['file']);config=read(refs['config']['file']);natural=read(refs['prefix_natural']['file'])
        projection=read(refs['prefix_projection']['file']);request=read(refs['request']['file']);proof=parent['main_arms'][arm]
        need(failure['type']=='ValueError' and failure['message']=='Measured first three worlds hold remaining evaluation beyond10800s'
             and failure['phase']=='run' and failure['no_automatic_retry'] and failure['partial_evidence_retained']
             and failure['source_sha256']==original.sources() and not (directory/'summary.json').exists()
             and not (directory/'analysis.json').exists(),'Only the exact preserved resource-gate failure is eligible')
        need(config['protocol']==original.PROTOCOL and config['policy']==original.POLICY and config['arm']==arm
             and config['array_job_id']=='444052' and config['plan_file']==str(PARENT) and config['plan_sha256']==PARENT_SHA
             and config['source_sha256']==original.sources() and config['inherited_source_sha256']==original.inherited_sources(),
             'Original inference configuration differs')
        for key in ('main_report','main_run','main_plan','final_checkpoint','profile_preparation','peft_config','state_proof'):
            need(config[key]==proof[key],'Original endpoint ownership differs: '+key)
        for key in ('native_identity','native_identity_sha256','feature_stage','compact_stage','statistics_protocol'):
            need(config[key]==parent[key],'Original native input/protocol identity differs: '+key)
        need(request==dict(phase='run',features=None,main_reports=None,plan=str(PARENT),arm=i,policy=original.POLICY,
             source_sha256=original.sources(),inherited_source_sha256=original.inherited_sources()),'Original request differs')
        need(len(natural)==3 and [r['index'] for r in natural]==parent['prefix_indices']==[0,400,800]
             and [r['execution_index'] for r in natural]==[0,1,2]
             and failure['progress']['natural']==natural and failure['progress']['completed_worlds']==3
             and set(failure['progress'])=={'natural','completed_worlds','counters'},'Exactly original three complete outcomes required')
        for row in natural:
            index=row['index'];need(all(row[key]==rows[index][key] for key in ('sid','n','qtype','pilot_role'))
                and 1<=len(row['generated_ids'])<=50 and math.isfinite(row['seconds']) and row['seconds']>0
                and row['fifty_token_seconds']==row['seconds']*50/len(row['generated_ids']), 'Original per-world identity/timing differs')
            need(Path(row['file'])==Path(config['data_directory'])/f'natural_{index:04d}.pt','Original raw path differs')
            bind(row['file'],bindings,row['sha256']);small=directory/f'natural_{index:04d}.json';bind(small,bindings)
            need(read(small)==row,'Original published row differs')
        need({str(f) for f in Path(config['data_directory']).glob('natural_*.pt')}=={r['file'] for r in natural},
             'Unexpected original trajectory or partial packet')
        counts=call_counts(sum(len(r['generated_ids']) for r in natural));need(failure['progress']['counters']==counts,'Original native call inventory differs')
        need(projection==original.project(boundary,arm,rows,completed=natural,elapsed=projection['elapsed_at_prefix_gate'])
             and projection['passed'] is False and projection['remaining_worlds']==997
             and projection['elapsed_at_prefix_gate']<=failure['elapsed_seconds']<=OLD_SECONDS[i], 'Original failed forecast differs')
        for name,digest in {**original.sources(),**original.inherited_sources()}.items():
            bind(directory/'source'/name.replace('/','_'),bindings,digest)
        result[arm]=dict(directory=str(directory),arm=arm,array_job_id='444052',allocation_id=f'444052_{i}',
            allocation_seconds=OLD_SECONDS[i],failure_elapsed_seconds=failure['elapsed_seconds'],**refs,
            original_source_release=release_ref,counters=counts,imported_decoder_layer_counts_recorded=False)
    return result


def allocation_check(out):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'original_sacct.psv';file.write_text(raw);records=[]
    for line in raw.splitlines():
        f=line.split('|')
        if len(f)!=8:continue
        need(f[1]!='mmred_official_native_continuation_run','No continuation GPU allocation before its sole CPU release')
        if f[1] not in ('mmred_official_native_evaluation','mmred_official_native_evaluation_run'):continue
        identifier,name,partition,state,code,seconds,tres,limit=f
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        records.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=code,seconds=int(seconds),
            gpus=int(generic[0]) if generic else sum(map(int,typed)),time_limit_minutes=int(limit),tres=tres))
    need(len(records)==3 and {r['job_id'] for r in records}=={f'444052_{i}' for i in range(3)},'Exactly one original three-arm array required')
    records.sort(key=lambda r:r['job_id'])
    need(all(r['name']=='mmred_official_native_evaluation_run' and r['partition']=='gpu' and r['state']=='FAILED'
         and r['exit_code']=='1:0' and r['gpus']==1 and r['seconds']==OLD_SECONDS[i] and r['time_limit_minutes']==180
         for i,r in enumerate(records)),'Original allocation state/seconds differ')
    return dict(passed=True,rows=records,original_allocated_GPU_seconds=98,raw_file=str(file),raw_sha256=sha(file))


def project(parent,arm,old_run,new_setup):
    old=read(old_run['prefix_projection']['file']);boundary=read(read(parent['boundary_report']['file'])['analysis_file'])
    reserve=max(old['elapsed_at_prefix_gate'],boundary['projection']['arms'][arm]['setup_seconds'])
    setup=reserve if new_setup is None else new_setup
    units=old['fifty_token_seconds_by_n'];counts=old['remaining_counts']
    work=sum(counts[str(n)]*units[str(n)] for n in (8,16,32));new_seconds=setup+1.25*work+60
    total=old_run['allocation_seconds']+new_seconds
    need(all(math.isfinite(v) and v>0 for v in (setup,work,new_seconds,total)),'Finite positive continuation timing required')
    return dict(passed=new_seconds<=14340 and total<=14400,projected_seconds=total,new_projected_seconds=new_seconds,
        cap_seconds=14400,continuation_cap_seconds=14340,original_allocated_seconds=old_run['allocation_seconds'],
        new_setup_seconds=setup,new_setup_is_measured=new_setup is not None,cpu_setup_reserve_seconds=reserve,
        remaining_worlds=997,remaining_counts=counts,fifty_token_seconds_by_n=units,
        original_projection=old,original_projection_preserved_false=True,maximum_new_tokens=50,
        accuracy_not_used=True,empirical_estimate_not_guarantee=True)


def check(args,out,started):
    need(sha(PARENT)==PARENT_SHA,'Fixed passed original CPU plan changed');parent=original.verify_plan(PARENT);bindings={}
    previous=original_runs(parent,bindings);ledger=allocation_check(out);save(out/'original_allocations.json',ledger)
    remaining=[i for i in range(1000) if i not in parent['prefix_indices']]
    need(len(remaining)==997 and len(set(remaining)|set(parent['prefix_indices']))==1000,'Exact disjoint remaining997 required')
    projections={arm:project(parent,arm,previous[arm],None) for arm in profile.ARMS};save(out/'continuation_projection.json',projections)
    need(all(v['passed'] for v in projections.values()),'Resource-only continuation does not fit cumulative cap')
    disk=dict(free_bytes=shutil.disk_usage(DATA.parent).free,approximate_allowance_bytes=300000000000,
        not_a_rigorous_byte_bound=True,no_space_preallocated=True)
    save(out/'disk_preflight.json',disk);need(disk['free_bytes']>=disk['approximate_allowance_bytes'],'Approximate remaining raw-output allowance unavailable')
    save(out/'remaining_order.json',remaining)
    plan=dict(parent);plan.update(protocol=PROTOCOL,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        original_plan=dict(file=str(PARENT),sha256=PARENT_SHA),original_runs=previous,
        original_allocations_file=str(out/'original_allocations.json'),remaining_order_file=str(out/'remaining_order.json'),
        continuation_projection=projections,continuation_projection_file=str(out/'continuation_projection.json'),
        input_bindings={**parent['input_bindings'],**bindings},artifacts={str(f):sha(f) for f in out.iterdir() if f.is_file()})
    save(out/'plan.json',plan);need(time.perf_counter()-started<600,'Continuation CPU cap exceeded')
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),continuation_projection=projections,
        original_allocated_GPU_seconds=98,imported_worlds_per_arm=3,new_worlds_per_arm=997)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='check' and summary['protocol']==PROTOCOL
         and summary['plan_file']==str(path) and summary['plan_sha256']==sha(path) and not (path.parent/'failure.json').exists(),
         'Passed continuation CPU preparation required')
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and plan['original_plan']==dict(file=str(PARENT),sha256=PARENT_SHA),
         'Continuation source/policy/parent changed')
    parent=original.verify_plan(PARENT);bindings={};previous=original_runs(parent,bindings)
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Continuation bound input/artifact changed')
    for name,digest in {**sources(),**inherited_sources()}.items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Continuation archived source changed')
    for key,value in parent.items():
        if key not in ('protocol','policy','source_sha256','inherited_source_sha256','input_bindings','artifacts'):
            need(plan[key]==value,'Original preparation field changed: '+key)
    need(plan['original_runs']==previous and all(plan['input_bindings'].get(f)==d for f,d in {**parent['input_bindings'],**bindings}.items())
         and read(plan['remaining_order_file'])==[i for i in range(1000) if i not in parent['prefix_indices']]
         and plan['continuation_projection']=={arm:project(parent,arm,previous[arm],None) for arm in profile.ARMS}
         and all(v['passed'] for v in plan['continuation_projection'].values()),'Continuation origin/order/forecast changed')
    return plan


def run(args,out,data,started,progress):
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.native_visual_memory import NativeVisualMemory
    torch.set_num_threads(4);plan=verify_plan(args.plan);arm=profile.ARMS[args.arm];proof=plan['main_arms'][arm]
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    need(os.environ.get('SLURM_ARRAY_TASK_ID')==str(args.arm) and os.environ.get('SLURM_ARRAY_TASK_COUNT')=='3','One fixed three-arm array required')
    items=read(plan['cases_file']);rows=read(plan['rows_file']);order=read(plan['remaining_order_file'])
    old_run=plan['original_runs'][arm]
    imported=[dict(row,origin='original',source_run=old_run['directory'],source_execution_index=row['execution_index'])
        for row in read(old_run['prefix_natural']['file'])]
    save(out/'imported_natural.json',imported)
    boundary=read(read(plan['boundary_report']['file'])['analysis_file']);final=proof['final_checkpoint']
    loaded=load_runtime(str(profile.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
    need(p.installed({})==plan['packages'],'Native packages differ')
    refs=list(model.named_parameters());base=p.base_metadata(refs);save(out/'base_before.json',base)
    contract=p.install_lora(torch,model,out/'actual_peft_config.json');params=p.adapter_parameters(model)
    need(sha(final['file'])==final['sha256'],'Final checkpoint changed')
    state=torch.load(final['file'],map_location='cpu',weights_only=True)
    need(state_proof(torch,state,arm,proof['peft_config'])==proof['state_proof']
         and p.canonical_peft_config(model)==proof['peft_config'],'CPU-bound checkpoint/PEFT proof differs')
    profile.restore(torch,params,state['adapter']);memory=None if arm=='ordinary' else NativeVisualMemory().to(model.device)
    if memory is not None:profile.restore(torch,dict(memory.named_parameters()),state['memory']);memory.eval().requires_grad_(False)
    profile.model_mode(model,params,False);p.check_base(refs,base,model)
    live={**params,**({} if memory is None else {'memory.'+k:v for k,v in memory.named_parameters()})}
    versions={k:(id(v),v._version) for k,v in live.items()}
    need(all(not v.requires_grad and v.grad is None for v in model.parameters())
         and all(not v.requires_grad and v.grad is None for v in live.values()),'Entire fitted model/core must be frozen')
    provenance=dict(native_identity_sha256=plan['native_identity_sha256'],lora=final,
        memory=None if memory is None else dict(checkpoint=final,tensors=profile.tensor_state(dict(memory.named_parameters()))))
    config=dict(protocol=PROTOCOL,policy=POLICY,arm=arm,array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        hardware=hardware,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        main_report=proof['main_report'],main_run=proof['main_run'],main_plan=proof['main_plan'],final_checkpoint=final,
        feature_stage=plan['feature_stage'],compact_stage=plan['compact_stage'],profile_preparation=proof['profile_preparation'],
        peft_config=proof['peft_config'],contract=contract,state_proof=proof['state_proof'],provenance=provenance,
        statistics_protocol=plan['statistics_protocol'],data_directory=str(data),no_checkpoint_copy=True,
        original_plan=plan['original_plan'],original_run=old_run,resource_only_continuation=True)
    save(out/'config.json',config);del state
    counts=Counter(dict.fromkeys(('model','backbone','visual','language','norm','head','backward','optimizer'),0));layers=[0]*28
    natural=[];progress.update(counters=counts,natural=natural,imported_natural=imported,imported_counters=old_run['counters']);gate=None
    def check_run():
        need(time.perf_counter()-started<14340 and old_run['allocation_seconds']+time.perf_counter()-started<14400,
             'Continuation or cumulative GPU cap exceeded');p.check_base(refs,base,model)
        need(versions=={k:(id(v),v._version) for k,v in live.items()} and frozen_adapter_contract(torch,model)==contract
             and all(not v.requires_grad and v.grad is None for v in live.values()),'Fitted endpoint/PEFT state changed')
    with ExitStack() as hooks,(out/'natural.jsonl').open('x') as stream:
        def bump(key):
            def callback(*_):
                counts[key]+=1;need(key!='visual','Evaluation must use shared frozen features, without vision')
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.visual,'visual'),
                           (model.model.language_model,'language'),(model.model.language_model.norm,'norm'),(model.lm_head,'head')):
            hooks.callback(module.register_forward_pre_hook(bump(key)).remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def callback(*_,index=index):layers[index]+=1
            hooks.callback(layer.register_forward_pre_hook(callback).remove)
        torch.cuda.synchronize();setup=time.perf_counter()-started;gate=project(plan,arm,old_run,setup)
        save(out/'continuation_projection.json',gate);progress['continuation_projection']=gate
        need(gate['passed'],'Actual new setup plus preserved old bounds holds continuation beyond cumulative cap')
        for execution_index,index in enumerate(order):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            evidence=dict(parameter_state=final,feature_packet=items[index]['features']);file=data/f'natural_{index:04d}.pt'
            result=None;text=None;score=None
            try:
                case,value=training.load_inputs(torch,model,items[index])
                if memory is None:
                    with torch.inference_mode():result=profile.ordinary_generate(torch,model,loaded.processor,case,value,evidence)
                else:
                    with torch.inference_mode():capture=memory(value,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
                    evidence.update(memory_capture=capture,coordinate_identity={k:p.tensor_info(v) for k,v in case['coordinates'].items()})
                    result=runtime.generate_cold(model,loaded.processor,case['text'],capture['tokens'].unsqueeze(0),provenance=provenance,max_tokens=50,evidence=evidence)
                ids=result['generated_ids'];need(1<=len(ids)<=50,'Fixed native generated-token bound differs')
                content=ids[:-1] if ids[-1] in answers.EOS_IDS else ids
                text=loaded.processor.tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                score=answers.score_answer(text,case['metadata']['atype'],json.loads(case['target_text'])['answer'],ids)
                descriptor=profile.retain(torch,file,index,arm,'cold',evidence,result=result,primary_text=text,score=score)
                check_run();row=dict(index=index,execution_index=execution_index+3,origin='continuation',source_run=str(out),
                    source_execution_index=execution_index,sid=rows[index]['sid'],n=rows[index]['n'],
                    qtype=rows[index]['qtype'],pilot_role=rows[index]['pilot_role'],**descriptor,generated_ids=ids,primary_text=text,score=score,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();torch.cuda.synchronize()
                row['seconds']=time.perf_counter()-tick;row['fifty_token_seconds']=row['seconds']*50/len(ids)
                natural.append(row);save(out/f'natural_{index:04d}.json',row);progress['completed_worlds']=len(natural)
                del case,value,evidence,result
                if memory is not None:del capture
            except BaseException:
                if 'evidence' in locals():
                    partial=profile.retain(torch,file.with_name(file.stem+'_partial.pt'),index,arm,'cold',evidence,partial=True,result=result,primary_text=text,score=score)
                    progress.setdefault('partial_packets',[]).append(partial)
                raise
    generated=sum(len(r['generated_ids']) for r in natural);expected=dict(model=generated,backbone=generated,visual=0,
        language=generated,norm=generated,head=generated,backward=0,optimizer=0)
    need(len(natural)==997 and [r['index'] for r in natural]==order and generated<=49850 and gate is not None and gate['passed']
         and dict(counts)==expected and layers==[generated]*28,'Complete evaluation native work inventory differs')
    check_run();need(profile.tensor_state(params)==proof['state_proof']['adapter']
        and (memory is None or profile.tensor_state(dict(memory.named_parameters()))==proof['state_proof']['memory']),
        'Exact fitted endpoint bytes changed')
    new_natural=natural;natural=sorted(imported+new_natural,key=lambda r:r['index'])
    need([r['index'] for r in natural]==list(range(1000)) and len({r['sid'] for r in natural})==1000,
         'Original3 plus disjoint new997 must cover every original world once')
    combined_tokens=generated+sum(len(r['generated_ids']) for r in imported);combined=call_counts(combined_tokens)
    need(combined_tokens<=50000 and combined=={key:expected[key]+old_run['counters'][key] for key in expected},
         'Combined original/new native inventory differs')
    lineage=dict(original_run=old_run,original_plan=plan['original_plan'],original_failure_preserved=True,
        imported_worlds=3,new_worlds=997,combined_worlds=1000,imported_indices=[r['index'] for r in imported],
        new_indices=order,imported_counters=old_run['counters'],new_counters=expected,combined_counters=combined,
        imported_decoder_layer_counts_recorded=False,original_allocated_seconds=old_run['allocation_seconds'],
        continuation_array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],continuation_cap_seconds=14340,total_per_arm_cap_seconds=14400)
    save(out/'base_after.json',p.base_metadata(refs));need(read(out/'base_before.json')==read(out/'base_after.json'),'Frozen native base changed')
    analysis=dict(protocol=PROTOCOL,arm=arm,passed=True,completed=True,final_checkpoint=final,natural=natural,provenance=provenance,
        counters=expected,decoder_layer_calls=layers,head_rows=generated,setup_seconds=setup,original_projection=plan['projection'][arm],
        prefix_projection=read(old_run['prefix_projection']['file']),continuation_projection=gate,original_run=old_run,
        continuation_lineage=lineage,imported_counters=old_run['counters'],combined_counters=combined,combined_head_rows=combined_tokens,
        imported_decoder_layer_counts_recorded=False,combined_expected_decoder_layer_calls=[combined_tokens]*28,all1000_completed=True,all_three_arms_required=True,requires_independent_cpu_audit=True,
        computational_completion_only=True,objective_achieved=False,accuracy_not_used_for_execution=True,
        statistics_protocol=plan['statistics_protocol'],no_checkpoint_copy=True)
    save(out/'analysis.json',analysis)
    artifacts={str(f):sha(f) for directory in (out,data) for f in directory.iterdir() if f.is_file()};save(out/'artifacts.json',artifacts)
    check_run()
    return dict(config,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),counters=expected,head_rows=generated,
        worlds=1000,new_worlds=997,imported_worlds=3,combined_counters=combined,combined_head_rows=combined_tokens,
        continuation_lineage=lineage,requires_independent_cpu_audit=True,computational_completion_only=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    ap.add_argument('--plan',type=Path);ap.add_argument('--arm',type=int,choices=range(3));args=ap.parse_args()
    p.native.require_slurm(gpu=args.run);need(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm only')
    phase='check' if args.check else 'run';started=time.perf_counter()
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and args.plan is None and args.arm is None,'Fixed CPU continuation check only')
        need(not any(OUT.glob('check_*')),'One CPU continuation release attempt');tag=f'check_{os.environ["SLURM_JOB_ID"]}'
    else:
        need(args.plan and args.arm is not None and os.environ.get('SLURM_ARRAY_JOB_ID'),'One continuation three-arm array required')
        array=os.environ['SLURM_ARRAY_JOB_ID'];need(all(x.name.split('_')[1]==array for x in OUT.glob('run_*')),'Only one continuation array is released')
        tag=f'run_{array}_{args.arm}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);own,inherited=archive(out)
    save(out/'request.json',dict(phase=phase,plan=None if args.plan is None else str(args.plan.resolve()),arm=args.arm,
        policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,original_plan=dict(file=str(PARENT),sha256=PARENT_SHA)))
    progress={}
    try:
        if args.check:result=check(args,out,started)
        else:
            data=DATA/tag;data.mkdir(parents=True,exist_ok=False);result=run(args,out,data,started,progress)
        need(time.perf_counter()-started<POLICY['cpu_seconds' if args.check else 'gpu_seconds_per_arm']
             and sources()==own and inherited_sources()==inherited,'Continuation source or allocation cap changed')
        if args.run:need(OLD_SECONDS[args.arm]+time.perf_counter()-started<14400,'Cumulative per-arm GPU cap exceeded')
        save(out/'summary.json',dict(result,protocol=PROTOCOL,phase=phase,policy=POLICY,passed=True,completed=True,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),phase=phase,progress=progress,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited,
            original_failure_preserved=True,partial_evidence_retained=True,no_automatic_retry=True));raise


if __name__=='__main__':main()
