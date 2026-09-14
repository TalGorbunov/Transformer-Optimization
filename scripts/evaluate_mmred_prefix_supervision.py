"""Seven fixed native checkpoints on fresh main1400; no inference supervision."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_mmred_official_native_memory as old
from scripts import report_mmred_official_native_continuation as prior_audit
from scripts import stage_mmred_fresh_native as features
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=old.p;profile=old.profile;training=old.training;answers=old.answers;read=old.read
PROTOCOL='mmred_prefix_supervision_fresh_evaluation'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_prefix_supervision_evaluation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_prefix_supervision_evaluation')
PROPOSAL='docs/paper/MMRED_PREFIX_SUPERVISION_FRESH_EVALUATOR.md'
STATISTICS='docs/paper/MMRED_PREFIX_SUPERVISION_FRESH_STATISTICS.md'
STATISTICS_SHA='687b1bf5eb192a204aaa5da10b6f0527c2a9e16feecc9bed94610a2c254f4e9e'
STATISTICS_RELEASE=REPO/'outputs/native_aggregation_vlm/mmred_prefix_supervision/fresh_statistics_release.json'
TRAINING_RELEASE=REPO/'outputs/native_aggregation_vlm/mmred_prefix_supervision/source_release.json'
OWN=('scripts/evaluate_mmred_prefix_supervision.py',PROPOSAL,
     'slurm/mmred_prefix_supervision_evaluation_check.sbatch','slurm/mmred_prefix_supervision_evaluation_run.sbatch')
TASKS=(('ordinary',None),('answer',25),('answer',26),('local',25),('local',26),('prefix',25),('prefix',26))
POLICY=dict(protocol=PROTOCOL,tasks=[dict(task_index=i,arm=a,seed=s) for i,(a,s) in enumerate(TASKS)],
    worlds=1400,lengths=[8,16,32],worlds_by_length={'8':400,'16':400,'32':600},
    primary_N32_worlds=400,N32_control_worlds=200,checkpoint_update=1500,
    cpu_seconds=600,gpu_seconds_per_task=3600,maximum_GPU_seconds=25200,
    maximum_arrays=1,maximum_attempts_per_task=1,maximum_concurrent_evaluation_GPUs=3,
    cpu_cores=4,memory_gib=16,maximum_new_tokens=50,maximum_model_head_calls_per_task=70000,
    vision_calls=0,backward_calls=0,optimizer_steps=0,logical_visual_worlds_per_task=1400,
    logical_visual_frames_per_task=28800,logical_visual_tokens_per_task=5644800,
    inference_supervision_modules=0,inference_oracle_hooks=0,auxiliary_target_tensor_files_opened=0,
    all_seven_checkpoints_required=True,no_checkpoint_selection=True,no_accuracy_release_gate=True,
    diagnostic_questions_evaluated=0,first_length_witnesses_retained_once=True,
    empirical_complete_world_timing=True,not_a_fifty_token_worst_case_bound=True,
    approximate_raw_free_space_bytes=1000000000000,requires_independent_cpu_audit=True)


def task_key(arm,seed):return arm if seed is None else f'{arm}_seed{seed}'


def main_auditor():
    from scripts import report_mmred_prefix_supervision_main
    return report_mmred_prefix_supervision_main


def sources():
    need(sha(REPO/STATISTICS)==STATISTICS_SHA,'Prospective statistics changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    fo,fi=features.source_maps();release=read(TRAINING_RELEASE);result={}
    for mapping in (old.sources(),old.inherited_sources(),prior_audit.sources(),prior_audit.inherited_sources(),
                    fo,fi,main_auditor().sources(),main_auditor().inherited_sources(),
                    release['source_sha256'],release.get('inherited_source_sha256',{}),
                    {STATISTICS:STATISTICS_SHA}):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Fresh evaluator source conflict');result[name]=digest
    need(not set(OWN)&set(result) and all(sha(REPO/name)==digest for name,digest in result.items()),'Frozen source ancestor changed')
    return result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound fresh evaluator input changed: '+str(path))
    bindings[str(path)]=digest;return dict(file=str(path),sha256=digest)


def bind_ref(value,bindings):
    bind(value['file'],bindings,value['sha256']);return value


def archive(out):
    own=sources();inherited=inherited_sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Fresh evaluator source archive differs')
    return own,inherited


def counts(tokens):return dict(model=tokens,backbone=tokens,visual=0,language=tokens,norm=tokens,head=tokens,backward=0,optimizer=0)


def work_order(rows,populations):
    need(len(rows)==1400 and [r['index'] for r in rows]==list(range(1400))
         and len({r['sid'] for r in rows})==len({r['world_sha256'] for r in rows})==1400
         and all(r['pilot_role']=='fresh_main' and r['cohort']=='main' for r in rows)
         and Counter(r['n'] for r in rows)==Counter({8:400,16:400,32:600}), 'Exact main1400 fresh worlds required')
    need(populations['main']==list(range(1400)) and len(populations['main_N32_primary'])==400
         and len(populations['main_N32_controls'])==200,'Fixed primary/control populations differ')
    primary=[i for i,r in enumerate(rows) if r['n']==32 and r['qtype'] in ('spend_together','where_spend')]
    controls=[i for i,r in enumerate(rows) if r['n']==32 and r['qtype'] in ('steps_in_room','char_at_frame')]
    need(primary==populations['main_N32_primary'] and controls==populations['main_N32_controls']
         and populations['main_by_length']=={str(n):[i for i,r in enumerate(rows) if r['n']==n] for n in (8,16,32)},
         'Source-ordered primary/control/length aliases differ')
    prefix=[next(i for i,r in enumerate(rows) if r['n']==n) for n in (8,16,32)]
    return prefix+[i for i in range(1400) if i not in prefix],prefix


def project(reference,rows,*,setup,completed=(),elapsed=None):
    need(math.isfinite(setup) and setup>0,'Positive measured or reserved setup required')
    units={int(n):v for n,v in reference['complete_world_seconds_by_n'].items()};done=set()
    for row in completed:
        need(row['index'] not in done and 0<=row['index']<1400 and row['n']==rows[row['index']]['n']
             and math.isfinite(row['seconds']) and row['seconds']>0,'Complete prefix timing ownership differs')
        done.add(row['index']);units[row['n']]=max(units[row['n']],row['seconds'])
    remaining=Counter(r['n'] for i,r in enumerate(rows) if i not in done)
    base=setup if elapsed is None else elapsed;total=base+1.25*sum(remaining[n]*units[n] for n in (8,16,32))+60
    need(all(math.isfinite(v) and v>0 for v in (*units.values(),base,total)),'Finite positive resource estimate required')
    return dict(passed=total<=3600,projected_seconds=total,cap_seconds=3600,setup_seconds=setup,
        elapsed_at_prefix_gate=elapsed,base_seconds=base,remaining_worlds=sum(remaining.values()),
        remaining_counts={str(n):remaining[n] for n in (8,16,32)},complete_world_seconds_by_n={str(n):units[n] for n in (8,16,32)},
        completed_indices=[r['index'] for r in completed],margin=1.25,reserve_seconds=60,
        timing_population='all1000 audited original ordinary complete-world measurements plus retained fresh prefix',
        maximum_new_tokens=50,no_token_length_rescaling=True,not_a_fifty_token_worst_case_bound=True,
        empirical_estimate_not_guarantee=True,accuracy_not_used=True)


def checkpoint_proof(torch,state,owner):
    arm,seed=owner['arm'],owner['seed'];adapter=state['adapter']
    expected={target+suffix for target in p.TARGETS for suffix in ('.lora_A.default.weight','.lora_B.default.weight')}
    need(state['arm']==arm and state['updates']==1500 and state['peft_config']==owner['peft_config']
         and set(adapter)==expected and len(adapter)==224 and sum(v.numel() for v in adapter.values())==10092544
         and all(v.dtype==torch.float32 and bool(v.isfinite().all()) for v in adapter.values()),'Exact finite final224 adapter state required')
    if arm=='ordinary':
        need(seed is None and state['final_checkpoint'] is True and state['memory'] is None,'Original fitted ordinary endpoint required')
        need(old.state_proof(torch,state,'ordinary',owner['peft_config'])==owner['original_state_proof'],'Original endpoint proof differs')
    else:
        need(state['seed']==seed and state['initial']==owner['initial_checkpoint']
             and state['source_release']==owner['source_release'] and state['targets']==owner['targets']
             and state['policy']==owner['training_policy'] and 'memory' not in state,
             'Continuation checkpoint training lineage differs')
    tensors=profile.tensor_state(adapter)
    need(tensors==owner['final_adapter_tensors'],'Audit-bound final adapter bytes differ')
    return dict(arm=arm,seed=seed,updates=1500,adapter=tensors,peft_config=owner['peft_config'],
        tensors=224,parameters=10092544,finite=True,adapter_only_restored=True,no_inference_supervision=True)


def timing_reference(path,bindings):
    descriptor=bind(path,bindings);summary=read(path);report=prior_audit.verify_report(path)
    need(report['arm']=='ordinary' and report['passed'] is report['completed'] is True
         and report['all_scheduled_numerical_evidence_collected'],'Completed original ordinary audit required')
    for key in ('analysis','input_bindings','artifacts'):bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
    bind(report['predictions_file'],bindings,report['predictions_sha256']);records=read(report['predictions_file'])
    need(len(records)==1000 and [r['index'] for r in records]==list(range(1000)),'All original timing rows required')
    selected=[dict(index=r['index'],n=r['n'],seconds=r['seconds'],tokens=len(r['generated_ids'])) for r in records]
    need(Counter(r['n'] for r in selected)==Counter({8:400,16:400,32:200})
         and all(math.isfinite(r['seconds']) and r['seconds']>0 and 1<=r['tokens']<=50 for r in selected),
         'Complete prior timing population differs')
    run=report['run'];bind(run['config_file'],bindings,run['config_sha256']);config=read(run['config_file'])
    bind(run['plan_file'],bindings,run['plan_sha256']);parent=read(run['plan_file'])
    owner=parent['main_arms']['ordinary'];endpoint=report['final_checkpoint']
    need(config['arm']=='ordinary' and endpoint==config['final_checkpoint']==owner['final_checkpoint']
         and config['peft_config']==owner['peft_config'],'Original timing/anchor checkpoint identity differs')
    setup=report['resources']['continuation_setup_seconds'];need(math.isfinite(setup) and setup>0,'Prior measured setup absent')
    reference=dict(audit=descriptor,predictions=dict(file=report['predictions_file'],sha256=report['predictions_sha256']),
        complete_world_seconds_by_n={str(n):max(r['seconds'] for r in selected if r['n']==n) for n in (8,16,32)},
        observed_token_range_by_n={str(n):[min(r['tokens'] for r in selected if r['n']==n),max(r['tokens'] for r in selected if r['n']==n)] for n in (8,16,32)},
        continuation_setup_seconds=setup,worlds=1000,rows=selected,score_fields_used=False,
        original_resource_failure_preserved=True,continuation_lineage=report['continuation_lineage'])
    anchor=dict(arm='ordinary',seed=None,task_index=0,task_key='ordinary',main_report=report['main_report'],
        final_checkpoint=endpoint,peft_config=owner['peft_config'],original_state_proof=owner['state_proof'],
        final_adapter_tensors=owner['state_proof']['adapter'],main_run=owner['main_run'],main_plan=owner['main_plan'],
        profile_preparation=owner['profile_preparation'],reference_evaluation_audit=descriptor)
    return reference,anchor,config


def check(args,out,started,progress):
    import torch
    torch.set_num_threads(4);bindings={};progress['input_bindings']=bindings
    feature_ref=bind(args.features,bindings);stage=features.verify_stage(args.features,streaming=True)
    feature_summary=read(args.features);bind(feature_summary['plan_file'],bindings,feature_summary['plan_sha256'])
    for key in ('rows_file','cases_file','compact_cases_file','feature_index_file','populations_file','timings_file'):
        bind(stage[key],bindings)
    rows=read(stage['rows_file']);items=read(stage['cases_file']);populations=read(stage['populations_file'])
    order,prefix=work_order(rows,populations)
    need(len(items)==1400 and all(item['row']==rows[i] and item['compact']['index']==item['features']['index']==i
         and item['compact']['sid']==item['features']['sid']==rows[i]['sid'] for i,item in enumerate(items)),
         'Complete shared native feature/input inventory differs')
    reference,anchor,previous_config=timing_reference(args.reference_evaluation_audit,bindings)
    need(previous_config['native_identity']==stage['native_identity']
         and previous_config['native_identity_sha256']==stage['native_identity_sha256'],'Fresh and original native precision/processor identities differ')
    owners={'ordinary':anchor};common={}
    matched=main_auditor().verify_matched_reports(args.main_reports)
    for path,audit in zip(args.main_reports,matched):
        ref=bind(path,bindings);summary=read(path)
        arm,seed=audit['arm'],audit['seed'];key=task_key(arm,seed)
        need((arm,seed) in TASKS[1:] and key not in owners and audit['phase']=='run'
             and audit['passed'] is audit['completed'] is audit['no_efficacy_gate'] is audit['no_fresh_predictions'] is True
             and audit['no_inference_supervision'] and audit['trainable_tensors']==224
             and audit['trainable_parameters']==10092544 and audit['auxiliary_trainable_parameters']==0,
             'Six distinct completed supervised continuation audits required')
        for label in ('producer_summary','producer_config','source_release','main_release','profile_audit','training_plan','evaluation_plan'):
            bind_ref(audit[label],bindings)
        # Targets are metadata lineage only: do not open their tensors or construct their module.
        need(audit['targets']['sha256'] and audit['targets']['file'],'Training target lineage missing')
        config=read(audit['producer_config']['file']);final=audit['final_checkpoint']
        need(final==dict(file=final['file'],sha256=final['sha256'],arm=arm,seed=seed,updates=1500)
             and audit['task_index']==3*(seed-25)+('answer','local','prefix').index(arm)
             and audit['initial_checkpoint']==anchor['final_checkpoint']
             and audit['native_identity_sha256']==stage['native_identity_sha256']
             and audit['peft_config']==anchor['peft_config'] and config['peft_config']==audit['peft_config'],
             'Matched final endpoint/native/configuration differs')
        for label in ('source_release','main_release','profile_audit','targets','training_plan','evaluation_plan'):
            need(label not in common or common[label]==audit[label],'Continuation arms must share fixed training lineage');common[label]=audit[label]
        owners[key]=dict(arm=arm,seed=seed,task_index=TASKS.index((arm,seed)),task_key=key,main_report=ref,
            final_checkpoint=final,initial_checkpoint=audit['initial_checkpoint'],
            initial_continuation_checkpoint=audit['initial_continuation_checkpoint'],
            final_adapter_tensors=audit['final_adapter_tensors'],peft_config=audit['peft_config'],
            producer_summary=audit['producer_summary'],producer_config=audit['producer_config'],
            training_policy=config['policy'],**{label:audit[label] for label in common})
        # Bind the auditor's compact successful publication and immutable manifests.
        for label in ('analysis','input_bindings','artifacts'):
            value=summary.get(label)
            if isinstance(value,dict):bind_ref(value,bindings)
            else:bind(summary[label+'_file'],bindings,summary[label+'_sha256'])
    need(set(owners)=={task_key(a,s) for a,s in TASKS},'All seven fixed checkpoints required')
    for key,owner in owners.items():
        final=owner['final_checkpoint'];bind_ref(final,bindings)
        state=torch.load(final['file'],map_location='cpu',weights_only=True);owner['state_proof']=checkpoint_proof(torch,state,owner);del state
    bind(TRAINING_RELEASE,bindings,common['source_release']['sha256'])
    stats=bind(REPO/STATISTICS,bindings,STATISTICS_SHA);stats_release=bind(STATISTICS_RELEASE,bindings)
    save(out/'timing_reference.json',reference)
    reserved_setup=max(reference['continuation_setup_seconds'],time.perf_counter()-started)
    forecast=project(reference,rows,setup=reserved_setup);save(out/'projection.json',forecast)
    free=shutil.disk_usage(DATA.parent).free
    disk=dict(free_bytes=free,operational_allowance_bytes=1000000000000,passed=free>=1000000000000,
        not_a_rigorous_byte_bound=True,no_space_preallocated=True,no_accuracy_condition=True)
    save(out/'disk_preflight.json',disk);save(out/'order.json',order)
    stage_timings=read(stage['timings_file'])
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited_sources(),
        feature_stage=feature_ref,feature_plan=dict(file=feature_summary['plan_file'],sha256=feature_summary['plan_sha256']),
        compact_stage=stage['compact_stage'],fresh_cohort=stage['fresh_cohort'],
        rows_file=stage['rows_file'],cases_file=stage['cases_file'],feature_index_file=stage['feature_index_file'],
        populations_file=stage['populations_file'],order_file=str(out/'order.json'),prefix_indices=prefix,
        checkpoints=owners,reference_evaluation_audit=reference['audit'],reference_timing_file=str(out/'timing_reference.json'),
        projection=forecast,projection_file=str(out/'projection.json'),resource_eligible=forecast['passed'] and disk['passed'],
        native_identity=stage['native_identity'],native_identity_sha256=stage['native_identity_sha256'],
        precision=stage['precision'],packages=stage['packages'],system_prompt=stage['system_prompt'],
        reference_hardware=previous_config['hardware'],statistics_protocol=stats,statistics_release=stats_release,
        training_source_release=dict(file=str(TRAINING_RELEASE),sha256=sha(TRAINING_RELEASE)),
        shared_visual_work=dict(stage=feature_ref,timings_file=stage['timings_file'],timings_sha256=sha(stage['timings_file']),
            timings=stage_timings,actual_preparation_once=True,logical_charge_per_checkpoint=dict(vision=1400,frames=28800,feature_tokens=5644800),
            measured_shared_allocation_in_separate_accounting=True,not_production_latency=True),
        input_bindings=bindings,artifacts={str(f):sha(f) for f in out.iterdir() if f.is_file()},
        no_accuracy_release_gate=True,no_inference_supervision=True)
    save(out/'plan.json',plan)
    need(time.perf_counter()-started<600,'CPU preparation cap exceeded')
    return dict(plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),worlds=1400,tasks=7,
        resource_eligible=plan['resource_eligible'],projection=forecast,no_GPU_release_if_ineligible=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='check' and summary['protocol']==PROTOCOL
         and summary['plan_file']==str(path) and summary['plan_sha256']==sha(path) and not (path.parent/'failure.json').exists(),
         'Completed CPU evaluation preparation required')
    need(plan['policy']==POLICY and plan['source_sha256']==sources() and plan['inherited_source_sha256']==inherited_sources()
         and plan['statistics_protocol']==dict(file=str(REPO/STATISTICS),sha256=STATISTICS_SHA)
         and plan['resource_eligible'] is summary['resource_eligible'] is True,'Frozen source or resource eligibility differs')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():need(sha(file)==digest,'Evaluation bound input/artifact changed')
    for name,digest in {**sources(),**inherited_sources()}.items():
        need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Evaluation source archive changed')
    for key,value in plan.items():
        if key.endswith('_file'):need(value in plan['input_bindings'] or value in plan['artifacts'],'External plan file lacks direct binding: '+key)
    rows=read(plan['rows_file']);order,prefix=work_order(rows,read(plan['populations_file']))
    reference=read(plan['reference_timing_file'])
    need(read(plan['order_file'])==order and plan['prefix_indices']==prefix
         and plan['projection']==project(reference,rows,setup=plan['projection']['setup_seconds'])
         and plan['projection']['passed'] and len(plan['checkpoints'])==7,'Fixed execution inventory/projection differs')
    return plan


def hook_signature(model):
    return {name:(tuple(m._forward_pre_hooks),tuple(m._forward_hooks),tuple(m._backward_hooks)) for name,m in model.named_modules()}


def run(args,out,data,started,progress):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);plan=verify_plan(args.plan);arm,seed=TASKS[args.task];key=task_key(arm,seed);owner=plan['checkpoints'][key]
    need(os.environ.get('SLURM_ARRAY_TASK_ID')==str(args.task) and os.environ.get('SLURM_ARRAY_TASK_COUNT')=='7'
         and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One task of fixed seven-checkpoint B200 array required')
    items=read(plan['cases_file']);rows=read(plan['rows_file']);order=read(plan['order_file']);reference=read(plan['reference_timing_file'])
    loaded=load_runtime(str(profile.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
    need(hardware==plan['reference_hardware'] and p.installed({})==plan['packages'],'Native hardware/packages differ')
    refs=list(model.named_parameters());base=p.base_metadata(refs);save(out/'base_before.json',base)
    contract=p.install_lora(torch,model,out/'actual_peft_config.json');params=p.adapter_parameters(model);final=owner['final_checkpoint']
    need(sha(final['file'])==final['sha256'],'Final checkpoint changed')
    state=torch.load(final['file'],map_location='cpu',weights_only=True)
    need(checkpoint_proof(torch,state,owner)==owner['state_proof'] and p.canonical_peft_config(model)==owner['peft_config'],
         'CPU-bound actual checkpoint/configuration differs')
    profile.restore(torch,params,state['adapter']);del state
    profile.model_mode(model,params,False);p.check_base(refs,base,model)
    need(profile.tensor_state(params)==owner['state_proof']['adapter'],'Restored adapter bytes differ before first fresh call')
    versions={name:(id(value),value._version) for name,value in params.items()};baseline_hooks=hook_signature(model)
    need(all(not before and not after and not backward for before,after,backward in baseline_hooks.values()),
         'Unexpected model hook before inference inventory installation')
    provenance=dict(native_identity_sha256=plan['native_identity_sha256'],lora=final,memory=None,
        arm=arm,seed=seed,task_key=key,no_inference_supervision=True)
    config=dict(protocol=PROTOCOL,policy=POLICY,arm=arm,seed=seed,task_key=key,task_index=args.task,
        array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        source_sha256=sources(),inherited_source_sha256=inherited_sources(),hardware=hardware,
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        main_report=owner['main_report'],checkpoint_owner=owner,final_checkpoint=final,
        feature_stage=plan['feature_stage'],compact_stage=plan['compact_stage'],fresh_cohort=plan['fresh_cohort'],
        peft_config=owner['peft_config'],contract=contract,state_proof=owner['state_proof'],provenance=provenance,
        statistics_protocol=plan['statistics_protocol'],statistics_release=plan['statistics_release'],
        data_directory=str(data),supervision_module_constructed=False,supervision_hook_installed=False,
        auxiliary_target_tensors_opened=False,checkpoint_optimizer_used=False,baseline_hook_signature=baseline_hooks)
    save(out/'config.json',config)
    counters=Counter(counts(0));layers=[0]*28;natural=[];progress.update(counters=counters,decoder_layer_calls=layers,natural=natural)
    def check_run():
        need(time.perf_counter()-started<3580,'Fixed work deadline leaves20 seconds for failure publication')
        p.check_base(refs,base,model)
        need(versions=={name:(id(value),value._version) for name,value in params.items()}
             and old.frozen_adapter_contract(torch,model)==contract,'Fitted adapter/base/PEFT state changed')
    gate=None
    with ExitStack() as hooks,(out/'natural.jsonl').open('x') as stream:
        def bump(name):
            def callback(*_):
                counters[name]+=1;need(name!='visual','Shared-feature inference must not execute vision')
            return callback
        for module,name in ((model,'model'),(model.model,'backbone'),(model.model.visual,'visual'),
             (model.model.language_model,'language'),(model.model.language_model.norm,'norm'),(model.lm_head,'head')):
            hooks.callback(module.register_forward_pre_hook(bump(name)).remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def callback(*_,index=index):layers[index]+=1
            hooks.callback(layer.register_forward_pre_hook(callback).remove)
        expected_hooks=hook_signature(model);torch.cuda.synchronize();setup=time.perf_counter()-started
        measured_setup_projection=project(reference,rows,setup=setup);save(out/'setup_projection.json',measured_setup_projection)
        need(measured_setup_projection['passed'],'Current setup plus prior complete-world timings exceed3600s; no fresh predictions started')
        for execution_index,index in enumerate(order):
            check_run();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            item=items[index];file=data/f'natural_{index:04d}.pt';result=None;text=None;score=None
            evidence=dict(parameter_state=final,feature_packet=item['features'],compact_packet=item['compact'],
                prefix_supervision_active=False,supervision_hook_installed=False,provenance=provenance)
            try:
                case,value=training.load_inputs(torch,model,item);torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
                call_start=time.perf_counter()
                with torch.inference_mode():result=profile.ordinary_generate(torch,model,loaded.processor,case,value,evidence)
                torch.cuda.synchronize();generation_seconds=time.perf_counter()-call_start
                need(hook_signature(model)==expected_hooks,'Natural generation left an unexpected hook')
                ids=result['generated_ids'];need(1<=len(ids)<=50,'Native token cap differs')
                content=ids[:-1] if ids[-1] in answers.EOS_IDS else ids
                text=loaded.processor.tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                score=answers.score_answer(text,case['metadata']['atype'],json.loads(case['target_text'])['answer'],ids)
                descriptor=profile.retain(torch,file,index,arm,'cold',evidence,result=result,primary_text=text,score=score,
                    seed=seed,task_key=key,task_index=args.task)
                check_run();row=dict(index=index,execution_index=execution_index,arm=arm,seed=seed,task_key=key,
                    sid=rows[index]['sid'],n=rows[index]['n'],qtype=rows[index]['qtype'],pilot_role=rows[index]['pilot_role'],
                    **descriptor,generated_ids=ids,primary_text=text,score=score,actual_output_tokens=len(ids),
                    prompt_tokens=int(case['inputs']['input_ids'].shape[-1]),visual_tokens=196*rows[index]['n'],
                    load_transfer_seconds=load_seconds,generation_with_observation_seconds=generation_seconds,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
                    raw_bytes=file.stat().st_size,auxiliary_hook_installed=False)
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
                save(out/f'natural_{index:04d}_observation.json',row);torch.cuda.synchronize()
                row['seconds']=time.perf_counter()-tick
                row['post_generation_evidence_seconds']=row['seconds']-load_seconds-generation_seconds
                natural.append(row);save(out/f'natural_{index:04d}.json',row);progress['completed_worlds']=len(natural)
                del case,value,evidence,result
                if execution_index==2:
                    gate=project(reference,rows,setup=setup,completed=natural,elapsed=time.perf_counter()-started)
                    save(out/'prefix_natural.json',natural);save(out/'prefix_projection.json',gate)
                    need([r['index'] for r in natural]==plan['prefix_indices'] and gate['passed'],
                         'Measured first three complete-world timings hold remaining evaluation beyond3600s')
            except BaseException:
                if 'evidence' in locals():
                    partial=profile.retain(torch,file.with_name(file.stem+'_partial.pt'),index,arm,'cold',evidence,
                        partial=True,result=result,primary_text=text,score=score,seed=seed,task_key=key)
                    progress.setdefault('partial_packets',[]).append(partial)
                raise
    need(hook_signature(model)==baseline_hooks,'Evaluation inventory hooks were not removed')
    tokens=sum(r['actual_output_tokens'] for r in natural)
    need(len(natural)==1400 and tokens<=70000 and dict(counters)==counts(tokens) and layers==[tokens]*28
         and gate is not None and gate['passed'],'Complete seven-task native work inventory differs')
    check_run();need(profile.tensor_state(params)==owner['state_proof']['adapter'],'Final adapter bytes changed during inference')
    save(out/'base_after.json',p.base_metadata(refs));need(read(out/'base_after.json')==read(out/'base_before.json'),'Frozen base changed')
    natural.sort(key=lambda row:row['index']);need([row['index'] for row in natural]==list(range(1400)),'Complete source order required')
    analysis=dict(protocol=PROTOCOL,policy=POLICY,arm=arm,seed=seed,task_key=key,task_index=args.task,
        passed=True,completed=True,final_checkpoint=final,natural=natural,provenance=provenance,
        counters=dict(counters),decoder_layer_calls=layers,head_rows=tokens,setup_seconds=setup,
        original_projection=plan['projection'],setup_projection=measured_setup_projection,prefix_projection=gate,
        all1400_completed=True,all_seven_checkpoints_required=True,statistics_protocol=plan['statistics_protocol'],
        shared_visual_work=plan['shared_visual_work'],actual_visual_calls=0,logical_visual_calls=1400,
        inference_seconds=sum(r['seconds'] for r in natural),generation_with_observation_seconds=sum(r['generation_with_observation_seconds'] for r in natural),
        load_transfer_seconds=sum(r['load_transfer_seconds'] for r in natural),
        post_generation_evidence_seconds=sum(r['post_generation_evidence_seconds'] for r in natural),
        raw_bytes=sum(r['raw_bytes'] for r in natural),empirical_timing_not_production_benchmark=True,
        no_inference_supervision=True,no_accuracy_release_gate=True,computational_completion_only=True,
        requires_independent_cpu_audit=True,no_further_release=True,objective_achieved=False)
    save(out/'analysis.json',analysis)
    artifacts={str(f):sha(f) for root in (out,data) for f in root.iterdir() if f.is_file()};save(out/'artifacts.json',artifacts)
    need(time.perf_counter()-started<3600,'Final publication exceeded fixed one-hour allocation')
    return dict(config,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
        counters=dict(counters),head_rows=tokens,worlds=1400,computational_completion_only=True,requires_independent_cpu_audit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--features',type=Path);parser.add_argument('--reference-evaluation-audit',type=Path)
    parser.add_argument('--main-reports',type=Path,nargs=6);parser.add_argument('--plan',type=Path)
    parser.add_argument('--task',type=int,choices=range(7));args=parser.parse_args()
    p.native.require_slurm(gpu=args.run);need(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm execution required')
    phase='check' if args.check else 'run';started=time.perf_counter()
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and args.features and args.reference_evaluation_audit and args.main_reports,
             'CPU preparation needs fresh features, original ordinary audit and six main audits')
        need(not any(OUT.glob('check_*')),'One CPU preparation attempt');tag=f'check_{os.environ["SLURM_JOB_ID"]}'
    else:
        need(args.plan and args.task is not None and os.environ.get('SLURM_ARRAY_JOB_ID'),'Fixed seven-task evaluation array required')
        array=os.environ['SLURM_ARRAY_JOB_ID'];need(all(x.name.split('_')[1]==array for x in OUT.glob('run_*')),'Only one evaluation array released')
        tag=f'run_{array}_{args.task}'
    out=OUT/tag;out.mkdir(parents=True,exist_ok=False);own,inherited=archive(out)
    save(out/'request.json',dict(phase=phase,features=None if args.features is None else str(args.features.resolve()),
        reference_evaluation_audit=None if args.reference_evaluation_audit is None else str(args.reference_evaluation_audit.resolve()),
        main_reports=None if args.main_reports is None else [str(v.resolve()) for v in args.main_reports],
        plan=None if args.plan is None else str(args.plan.resolve()),task=args.task,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited))
    progress={}
    try:
        if args.check:result=check(args,out,started,progress)
        else:
            data=DATA/tag;data.mkdir(parents=True,exist_ok=False);result=run(args,out,data,started,progress)
        need(time.perf_counter()-started<POLICY['cpu_seconds' if args.check else 'gpu_seconds_per_task']
             and sources()==own and inherited_sources()==inherited,'Evaluation cap/source changed')
        save(out/'summary.json',dict(result,protocol=PROTOCOL,phase=phase,policy=POLICY,passed=True,completed=True,
            elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited))
    except BaseException as exc:
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),phase=phase,
            progress=progress,elapsed_seconds=time.perf_counter()-started,source_sha256=own,inherited_source_sha256=inherited,
            original_outputs_preserved=True,no_automatic_retry=True));raise


if __name__=='__main__':main()
