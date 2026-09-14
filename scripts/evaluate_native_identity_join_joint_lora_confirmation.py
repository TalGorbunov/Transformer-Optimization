"""DRAFT conditional N16 confirmation of the final ordinary joint LoRA model.

No training, model selection, longer-length inference or extra profile. The
CPU preparation and passed main audit must precede a separate GPU release.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
from scripts import stage_native_identity_join_joint_lora_confirmation as preparation
from scripts import train_native_identity_join_joint_lora as training
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts import report_native_identity_join_joint_lora as audit
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_confirmation/evaluation'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_lora_confirmation/evaluation')
PROTOCOL='identity_join_joint_lora_confirmation_N16'
JOB='identity_join_joint_lora_confirmation_n16'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_LORA_CONFIRMATION_EVALUATION_PROPOSAL.md'
PROPOSAL_SHA='4326cb037deeafea3601d15b47b26957200a32930d17c27ae9e86e0268d50e3c'
PREPARATION=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_confirmation/preparation/check_443925/summary.json'
PREPARATION_SHA='3874a01fdfb17e3b99a3d78c86e6d2a9a7c29fac950115bf6853724176aea968'
PREPARATION_PLAN_SHA='7f2c4b6f2d91d7bc2cd761fbfa00bcd1b09aa9802aaf57fb6e6dfb077ccf40ea'
OWN=('scripts/evaluate_native_identity_join_joint_lora_confirmation.py',
     'scripts/report_native_identity_join_joint_lora_confirmation.py',PROPOSAL,
     'slurm/native_identity_join_joint_lora_confirmation_run.sbatch',
     'slurm/native_identity_join_joint_lora_confirmation_report.sbatch')
POLICY=dict(protocol=PROTOCOL,contexts=270,n_frames=16,checkpoint_step=324,maximum_new_tokens=4,
    panels={'A':108,'B':108,'C':54},families={'A':36,'B':36,'C':18},
    whole_threshold={'A':98,'B':98,'C':49},family_threshold={'A':33,'B':33,'C':16},
    model_calls_cap=1080,vision_calls=270,native_head_rows_cap=1080,
    run_seconds=900,report_seconds=1800,max_attempts=1,cpu_cores=4,memory_gib=16,
    no_fitting=True,no_checkpoint_selection=True,no_profile=True,no_extra_gpu_heads=True,
    no_longer_lengths=True,ordinary_joint_images_first=True,confirmation_seed=91726342,
    cpu_native_tv_max=.02,cpu_argmax_gate=False,resource_cap_is_draft=False)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA and POLICY['resource_cap_is_draft'] is False,'Evaluation resource/source release is not finalized')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for group in (preparation.sources(),preparation.inherited_sources(),training.sources(),training.inherited_sources()):
        for name,digest in group.items():
            need(name not in result or result[name]==digest,'Conflicting evaluation source ancestor');result[name]=digest
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Evaluation ancestor changed');return result


def snapshot(out):
    own=sources();(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Evaluation source copy changed')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return own


def verify_main_audit(path,bindings=None):
    """Hash/JSON-only handoff; never replays a tensor or reconstructs a model."""
    bindings={} if bindings is None else bindings;path=audit.summary_path(path);p.bind(path,bindings);summary=read(path)
    need(summary['protocol']==audit.PROTOCOL and summary['phase']=='main'
         and summary['passed'] is summary['completed'] is summary['main_audit_passed'] is True
         and summary['source_sha256']==audit.sources() and summary['inherited_source_sha256']==audit.inherited_sources()
         and summary['policy']==audit.POLICY and not (path.parent/'failure.json').exists(),'Passed exact independent main audit required')
    audit.archive(path.parent,summary['source_sha256'],summary['inherited_source_sha256'],bindings)
    for key in ('analysis','artifacts','input_bindings','report'):p.bind(summary[key+'_file'],bindings,summary[key+'_sha256'])
    for file,digest in read(summary['input_bindings_file']).items():p.bind(file,bindings,digest)
    for file,digest in read(summary['artifacts_file']).items():p.bind(file,bindings,digest)
    analysis=read(summary['analysis_file']);competence=analysis['competence']
    need(analysis['passed'] is analysis['completed'] is analysis['main_audit_passed'] is True and analysis['policy']==audit.POLICY
         and analysis['all2592_teacher_CE_and_ownership_audited'] is analysis['all_numerical_evidence_collected'] is True
         and competence==summary['competence'] and competence['passed'] is True and competence['contexts']==216
         and competence['whole_correct']>=206 and min(competence['orientation_correct'][k] for k in ('original','flipped'))>=103
         and competence['families']==18 and competence['complete_families']>=16,'Actual complete native216 training competence required')
    main_path=Path(summary['main_summary_file']);p.bind(main_path,bindings,summary['main_summary_sha256']);main=read(main_path)
    need(main['protocol']==training.PROTOCOL and main['phase']=='main' and main['passed'] is main['completed'] is main['computational_passed'] is True
         and main['source_sha256']==training.sources() and main['inherited_source_sha256']==training.inherited_sources()
         and main['policy']==training.POLICY and main['competence']==competence and main['downstream_release'] is False
         and main['confirmation_predictions_accessed'] is False and not (main_path.parent/'failure.json').exists(),'Fixed final main producer/audit join differs')
    audit.archive(main_path.parent,main['source_sha256'],main['inherited_source_sha256'],bindings)
    p.bind(main['endpoint_file'],bindings,main['endpoint_sha256']);endpoint=read(main['endpoint_file'])
    p.bind(endpoint['final_adapter_file'],bindings,endpoint['final_adapter_sha256']);final=read(endpoint['final_adapter_file'])
    need(endpoint['competence']==competence and endpoint['completed_updates']==324 and endpoint['completed_scene_presentations']==2592
         and endpoint['final_adapter_unchanged_during_evaluation'] is endpoint['full_peft_config_exact'] is True
         and final==analysis['checkpoints']['final'] and final['step']==324
         and analysis['profile']==main['profile'] and analysis['profile_audit']==main['profile_audit'],
         'Exact audited final324 adapter ownership differs')
    p.bind(final['file'],bindings,final['sha256'])
    descriptor=dict(file=str(path),sha256=sha(path),analysis_file=summary['analysis_file'],analysis_sha256=summary['analysis_sha256'],
        main_summary_file=str(main_path),main_summary_sha256=sha(main_path),final_adapter_file=final['file'],final_adapter_sha256=final['sha256'])
    return dict(analysis=analysis,main=main,final=final,descriptor=descriptor)


def inputs(stage_path,main_audit_path,bindings):
    stage_path=audit.summary_path(stage_path)
    need(stage_path==PREPARATION and sha(stage_path)==PREPARATION_SHA,'Exact passed N16 preparation443925 required')
    p.bind(stage_path,bindings,PREPARATION_SHA);plan=preparation.verify_stage(stage_path,streaming=True);proof=read(stage_path)
    need(proof['plan_sha256']==PREPARATION_PLAN_SHA,'Fixed N16 prepared plan changed')
    p.bind(proof['plan_file'],bindings,proof['plan_sha256']);parent=verify_main_audit(main_audit_path,bindings)
    need(parent['main']['confirmation_report']==plan['stage_report'] and parent['main']['training_stage']==plan['training_stage']
         and parent['main']['native_identity']==plan['native_identity'] and parent['main']['native_identity_sha256']==plan['native_identity_sha256']
         and parent['main']['profile']==plan['profile'] and parent['main']['profile_audit']=={k:plan['profile_audit'][k] for k in ('file','sha256')},
         'Prepared unused confirmation and actual final adapter must share exact ancestry')
    for file,digest in plan['input_bindings'].items():p.bind(file,bindings,digest)
    for file,digest in plan['files'].items():p.bind(file,bindings,digest)
    envelope=read(plan['envelope_file']);timing=read(plan['timing_estimate_file'])
    need(envelope['passed'] is plan['envelope_passed'] is True and len(envelope['rows'])==270
         and all(r['comparison']['passed'] for r in envelope['rows']),'Every actual N16 input must fit the measured envelope')
    estimate=timing['setup_seconds']+1.25*270*timing['per_scene_four_token_seconds']+60.
    need(estimate==timing['projected_seconds'] and math.isfinite(estimate) and 0<estimate<=POLICY['run_seconds']
         and timing['source_profile_audit']==plan['profile_audit'] and timing['cap_seconds'] is None,'Separately reviewed N16 runtime bound required')
    rows=read(plan['rows_file']);preparation.cohort(rows,16);prepared=read(plan['prepared_file'])
    descriptor=dict(file=str(stage_path),sha256=sha(stage_path),plan_file=proof['plan_file'],plan_sha256=proof['plan_sha256'])
    return plan,rows,prepared,parent,descriptor,dict(timing,passed=True,cap_seconds=POLICY['run_seconds'],separate_inference_resource_gate=True)


def load_packet(torch,item):
    need(sha(item['file'])==item['sha256'],'Consumed fresh native bundle changed');packet=torch.load(item['file'],map_location='cpu',weights_only=True)
    p.validate_joint(torch,packet['bundle']);width=packet['bundle']['metadata']['prompt_width']
    need(set(packet)=={'bundle','layout','position_ids'} and packet['bundle']['metadata']==item['metadata']
         and packet['layout']['metadata']==item['layout_metadata'] and p.tensor_info(packet['position_ids'])==item['position_identity']
         and packet['position_ids'].shape==(4,1,width) and torch.equal(packet['position_ids'][1:],packet['layout']['position_ids'])
         and torch.equal(packet['position_ids'][0,0],torch.arange(width)),'Exact prompt-only native layout changed')
    return packet


def allocations(raw):
    result=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=9 or fields[1] not in (JOB,'identity_join_joint_lora_confirmation_run'):continue
        job,name,partition,state,code,seconds,tres,start,end=fields
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        result.append(dict(job_id=job,name=name,partition=partition,state=state,exit_code=code,seconds=int(seconds),
            gpus=int(generic[0]) if generic else sum(map(int,typed)),start=start,end=end))
    return result


def accounting(out,completed=False,job_id=None):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(raw);rows=allocations(raw)
    need(len(rows)==1 and rows[0]['name']==JOB and rows[0]['partition']=='gpu' and rows[0]['gpus']==1
         and rows[0]['job_id']==(job_id if completed else os.environ['SLURM_JOB_ID'])
         and rows[0]['seconds']<=POLICY['run_seconds'],'Exactly one bounded N16 allocation, including failed attempts, is allowed')
    if completed:need(rows[0]['state']=='COMPLETED' and rows[0]['exit_code']=='0:0','Completed N16 allocation required')
    else:need(os.environ.get('SLURM_JOB_NAME')==JOB,'Exact N16 allocation name required')
    return dict(passed=True,rows=rows,allocated_gpu_seconds=sum(r['seconds'] for r in rows),cap_seconds=POLICY['run_seconds'],
        max_attempts=1,file=str(file),sha256=sha(file),command=command)


def run(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);bindings={};plan,rows,prepared,parent,stage_ref,projection=inputs(args.prepared_stage,args.main_audit,bindings)
    allocation=accounting(out);save(out/'launch_accounting.json',allocation)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=loaded.model.eval().requires_grad_(False)
    hardware=p.frozen_joint.live_identity(torch,loaded,plan);refs=list(model.named_parameters());base=p.base_metadata(refs)
    save(out/'base_before.json',base);need(p.installed({})==plan['packages'],'Bound native/PEFT software changed')
    contract=p.install_lora(torch,model,out/'actual_peft_config_before_contract.json');params=p.adapter_parameters(model)
    final=parent['final'];checkpoint=torch.load(final['file'],map_location='cpu',weights_only=True)
    need(checkpoint['step']==324 and checkpoint['epoch']==12 and checkpoint['full_training_final'] is True
         and checkpoint['seed']==24 and checkpoint['profile_initialization']==plan['profile'] and set(checkpoint['adapter'])==set(params)
         and audit.semantic_contract(checkpoint['contract'])==audit.semantic_contract(contract)
         and checkpoint['peft_config']==final['peft_config']==p.canonical_peft_config(model),'Only actual audited final adapter/config may be evaluated')
    with torch.no_grad():
        for name,value in params.items():
            need(p.tensor_info(checkpoint['adapter'][name])==final['tensors'][name],'Audited adapter tensor changed');value.copy_(checkpoint['adapter'][name])
    need({name:p.tensor_info(value) for name,value in params.items()}==final['tensors'],'Loaded final adapter bytes differ')
    p.adapter_contract(torch,model);p.check_base(refs,base,model);del checkpoint
    model.eval().requires_grad_(False);versions={name:v._version for name,v in model.named_parameters()}
    config=dict(protocol=PROTOCOL,policy=POLICY,phase='N16',source_sha256=frozen,inherited_source_sha256=inherited_sources(),
        prepared_stage=stage_ref,main_audit=parent['descriptor'],final_adapter=final,profile=plan['profile'],profile_audit=plan['profile_audit'],
        stage_report=plan['stage_report'],training_stage=plan['training_stage'],orientation_plan=plan['orientation_plan'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],hardware=hardware,packages=plan['packages'],
        projection=projection,data_directory=str(data),adapter_contract=contract,no_optimizer_created=True,no_training=True,no_checkpoint_selection=True)
    save(out/'config.json',config);torch.cuda.synchronize();setup=time.perf_counter()-started;records=[];counts=Counter()
    for index,row in enumerate(rows):
        need(time.perf_counter()-started<=POLICY['run_seconds'],'Fixed confirmation cap exceeded')
        tick=time.perf_counter();item=prepared[row['sid']];p.bind(item['file'],bindings,item['sha256']);packet=load_packet(torch,item);evidence={}
        try:result=p.generate_joint(model,loaded.processor,packet['bundle'],native_identity_sha256=plan['native_identity_sha256'],capture_head=True,evidence=evidence)
        except BaseException:
            torch.save(p.cpu_tree(torch,dict(sid=row['sid'],evidence=evidence)),data/f'eval_{index:03d}_partial.pt');raise
        file=data/f'eval_{index:03d}.pt';torch.save(dict(sid=row['sid'],result=result),file);digest=sha(file);scored=audit.score(result['generated_ids'],row['target_ids'])
        record=dict(index=index,sid=row['sid'],panel=row['panel'],n_frames=16,contrast_id=row['contrast_id'],variant=row['variant'],
            file=str(file),sha256=digest,generated_ids=result['generated_ids'],target_ids=row['target_ids'],gold=row['gold'],**scored,
            text=result['text'],raw_text=result['raw_text'],counters=result['counters'])
        records.append(record);counts.update(result['counters']);save(out/f'eval_{index:03d}.json',record)
        p.check_base(refs,base,model);torch.cuda.synchronize();save(out/f'timing_{index:03d}.json',dict(index=index,seconds=time.perf_counter()-tick))
    outcomes=[dict(row,**{k:r[k] for k in ('exact','first_token_correct','generated_ids')}) for row,r in zip(rows,records)]
    result=preparation.criterion(outcomes);tokens=sum(len(r['generated_ids']) for r in records)
    need(counts['model']==counts['language']==counts['norm']==counts['head']==tokens<=1080 and counts['visual']==270
         and all(counts.get(k,0)==0 for k in ('broadcast','fusion','conditioning','probe_head')),'Complete270 natural-call inventory differs')
    need(versions=={name:v._version for name,v in model.named_parameters()} and all(not v.requires_grad and v.grad is None for v in model.parameters())
         and {name:p.tensor_info(value) for name,value in params.items()}==final['tensors'] and p.canonical_peft_config(model)==final['peft_config'],
         'Natural confirmation changed frozen base/adapter/configuration')
    p.check_base(refs,base,model)
    for name,value in (('evaluations.json',records),('criterion.json',result),('input_bindings.json',bindings)):save(out/name,value)
    files={str(file):sha(file) for file in sorted(data.rglob('*')) if file.is_file()};save(out/'artifacts.json',files)
    endpoint=dict(passed=True,counters=dict(counts),contexts=270,native_head_rows=tokens,criterion=result,
        base_parameters_unchanged=True,adapter_parameters_unchanged=True,all_parameters_frozen=True,full_peft_config_exact=True,
        base_before_file=str(out/'base_before.json'),base_before_sha256=sha(out/'base_before.json'))
    save(out/'endpoint.json',endpoint)
    return dict(**config,passed=True,completed=True,criterion=result,counters=dict(counts),native_head_rows=tokens,setup_seconds=setup,
        endpoint_file=str(out/'endpoint.json'),endpoint_sha256=sha(out/'endpoint.json'),evaluations_file=str(out/'evaluations.json'),evaluations_sha256=sha(out/'evaluations.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
        longer_lengths_eligible=result['qualifies_A_and_B'],longer_lengths_released=False,valid_negative_outcomes_retained=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared-stage',type=Path,required=True);parser.add_argument('--main-audit',type=Path,required=True);args=parser.parse_args()
    p.native.require_slurm(gpu=True);out=OUT/f'run_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(prepared_stage=str(args.prepared_stage.resolve()),main_audit=str(args.main_audit.resolve()),source_sha256=frozen))
    try:
        value=run(args,out,frozen,started);elapsed=time.perf_counter()-started
        need(elapsed<=POLICY['run_seconds'] and sources()==frozen,'Confirmation cap/source changed')
        save(out/'summary.json',dict(value,elapsed_seconds=elapsed))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()
