"""DRAFT fixed joint-image LoRA main; execution remains separately held.

A passed V2 profile, independent profile audit and positive measured forecast
are prerequisites. Twelve complete epochs precede the sole training evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora_main'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_lora_main')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_joint_lora_main')
PROTOCOL='identity_join_joint_lora_main'
JOB='identity_join_joint_lora_main'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_LORA_MAIN_PROPOSAL.md'
PROPOSAL_SHA='d2e404b6f8c3f1299d1ecc8e6c1899e06f3eb919d1a471a8256126ff9ca48eeb'
OWN=('scripts/train_native_identity_join_joint_lora.py',PROPOSAL,'slurm/native_identity_join_joint_lora_main.sbatch')
POLICY=dict(protocol=PROTOCOL,contexts=216,epochs=12,scene_presentations=2592,gradient_accumulation=8,updates=324,
    rank=16,alpha=32,dropout=.05,seed=24,trainable_parameters=10092544,adapter_tensors=224,
    learning_rate=2e-4,betas=[.9,.999],epsilon=1e-8,weight_decay=.01,clip=1.,checkpointing=False,
    maximum_new_tokens=4,main_seconds=3600,main_attempts=1,model_calls_cap=3456,vision_calls=2808,
    training_head_rows=5760,evaluation_head_rows_cap=864,backwards=2592,
    no_validation_inference=True,no_checkpoint_selection=True,no_resume=True,no_early_stopping=True,
    no_prepare_model_for_kbit_training=True,base_storage_dtypes_preserved=True,train_adapter_dtype='torch.float32',
    ordinary_joint_images_first=True,no_prompt_wrapper=True,no_broadcast=True,
    profile_trained_checkpoint_forbidden=True,unrecorded_profile_rng_equality_claimed=False)


def auditor():
    from scripts import report_native_identity_join_joint_lora as report
    return report


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held main protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    result={}
    for mapping in (p.sources(),p.inherited_sources(),auditor().sources()):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Conflicting inherited source');result[name]=digest
    need(len(result)==168 and all(sha(REPO/name)==digest for name,digest in result.items()),'V2 plus independent audit source closure changed')
    return result


def snapshot(out):
    own=sources();(out/'source').mkdir()
    for name,digest in own.items():
        file=out/'source'/name.replace('/','_');file.write_bytes((REPO/name).read_bytes());need(sha(file)==digest,'Main source copy changed')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return own


def verify_profile(summary_path,bindings=None):
    """Hash/JSON joins only. Independent numerical profile audit is separate."""
    bindings={} if bindings is None else bindings;path=Path(summary_path).resolve();p.bind(path,bindings);s=read(path);root=path.parent
    need(s['protocol']==p.PROTOCOL and s['phase']=='profile' and s['policy']==p.POLICY and s['passed'] is s['completed'] is True
         and s['source_sha256']==p.sources() and s['inherited_source_sha256']==p.inherited_sources()
         and s['profile_checkpoint_must_not_initialize_main'] is s['future_main_restarts_seed24'] is True
         and s['no_main_or_validation_execution'] is True and not (root/'failure.json').exists(),'Passed V2 software profile required')
    plan_file=Path(s['plan_file']);p.bind(plan_file,bindings,s['plan_sha256']);plan=p.verify_plan(plan_file,streaming=True);p.bind(plan_file.parent/'summary.json',bindings)
    for name,digest in p.sources().items():p.bind(root/'source'/name.replace('/','_'),bindings,digest)
    for name in ('source_hashes.json','inherited_sources.json','request.json','config.json','launch_accounting.json','launch_sacct.psv'):p.bind(root/name,bindings)
    need(read(root/'source_hashes.json')==p.sources() and read(root/'inherited_sources.json')==dict(source_sha256=p.inherited_sources())
         and read(root/'request.json')==dict(phase='profile',plan=str(plan_file),source_sha256=p.sources()),'Profile source/request ownership differs')
    configuration=read(root/'config.json');need(all(s[key]==value for key,value in configuration.items()),'Profile config join differs')
    p.bind(p.OUT/'source_release.json',bindings);release=read(p.OUT/'source_release.json')
    need(release['status']=='frozen_before_execution' and release['source_sha256']==p.sources()
         and release['inherited_source_sha256']==p.inherited_sources(),'V2 source release changed')
    for field in ('endpoint','measurements','artifacts'):p.bind(s[field+'_file'],bindings,s[field+'_sha256'])
    endpoint=read(s['endpoint_file']);measure=read(s['measurements_file']);artifacts=read(s['artifacts_file'])
    for file,digest in artifacts.items():p.bind(file,bindings,digest)
    need(endpoint['passed'] is endpoint['zero_lora_exact'] is endpoint['roundtrip_logits_exact'] is endpoint['base_parameters_unchanged'] is True,'Profile fidelity failed')
    forecast=json.loads(json.dumps(p.project(measure['setup_seconds'],measure['microbatches'],measure['optimizer_seconds'],measure['natural'],measure['checkpoint_seconds'])))
    need(forecast==measure['projection']==s['projection'] and s['main_implementation_eligible']==forecast['passed']
         and forecast['main_release'] is False,'Original V2 forecast must remain unchanged, including a failed cost gate')
    main_projection=amended_projection(measure,forecast)
    counts=measure['counters'];natural=sum(r['generated_tokens'] for r in measure['natural'])
    need(len(measure['natural'])==8 and len(measure['microbatches'])==16 and len(measure['optimizer_seconds'])==2
         and counts==endpoint['counters']==s['counters'] and counts['model']==counts['language']==counts['norm']==counts['head']==natural+20<=52
         and counts['visual']==28 and counts['backward']==16 and counts['optimizer']==2
         and all(counts.get(k,0)==0 for k in ('broadcast','fusion','conditioning','probe_head')),'Complete fixed profile counts required')
    p.bind(root/'zero_lora_parity.json',bindings);parity=read(root/'zero_lora_parity.json');p.bind(root/'roundtrip.json',bindings);roundtrip=read(root/'roundtrip.json')
    need(len(parity)==4 and all(r['exact_logits'] and r['exact_generated_ids'] and r['exact_shapes'] for r in parity)
         and roundtrip['passed'] and all(roundtrip['tensors_exact'].values()) and all(roundtrip['original_reset_exact'].values())
         and roundtrip['full_peft_config_exact'] and roundtrip['same_parameter_objects'],'Exact zero-LoRA/serialization proof required')
    p.bind(root/'initial_adapter.json',bindings);initial=read(root/'initial_adapter.json')
    for field in ('file','config_file'):p.bind(initial[field],bindings,initial['sha256' if field=='file' else 'config_sha256'])
    need(artifacts[initial['file']]==initial['sha256'] and artifacts[initial['config_file']]==initial['config_sha256']
         and Path(initial['file']).name=='initial_adapter.pt','Only actual unfitted profile initialization may be consumed')
    p.bind(endpoint['base_before_file'],bindings,endpoint['base_before_sha256'])
    descriptor=dict(file=str(path),sha256=sha(path),plan_file=str(plan_file),plan_sha256=sha(plan_file),
        initial_adapter_file=initial['file'],initial_adapter_sha256=initial['sha256'],adapter_config_file=initial['config_file'],adapter_config_sha256=initial['config_sha256'])
    return dict(summary=s,plan=plan,descriptor=descriptor,initial=initial,projection=forecast,main_projection=main_projection,input_bindings=bindings)


def amended_projection(measure,original_projection):
    """New explicit accounting for a main that never invokes torch.profiler."""
    micro=measure['microbatches'];keys=[(r['step'],r['microbatch']) for r in micro]
    need(keys==[(step,j) for step in (1,2) for j in range(1,9)],'All sixteen fixed profile timing observations required')
    instrumented=[r for r in micro if (r['step'],r['microbatch']) in ((1,1),(1,2))]
    normal=[r for r in micro if (r['step'],r['microbatch']) not in ((1,1),(1,2))]
    need([r['n_frames'] for r in instrumented]==[8,16] and Counter(r['n_frames'] for r in normal)=={8:7,16:7},
         'Only the two predeclared profiler calls may be priced once instead of per example')
    surcharge=sum(r['seconds'] for r in instrumented)
    value=json.loads(json.dumps(p.project(measure['setup_seconds']+surcharge,normal,measure['optimizer_seconds'],measure['natural'],measure['checkpoint_seconds'])))
    return dict(value,accounting_rule='uninstrumented_main_with_once_only_profile_instrumentation',original_projection=original_projection,
        original_setup_seconds=measure['setup_seconds'],instrumented_cold_seconds=surcharge,
        instrumented_microbatches=[{key:r[key] for key in ('step','microbatch','n_frames','seconds')} for r in instrumented],normal_microbatch_count=14)


def accounting(out,profile):
    need(os.environ.get('SLURM_JOB_NAME')==JOB,'Exact main allocation name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;(out/'launch_sacct.psv').write_text(raw);rows=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=9 or fields[1] not in (JOB,p.JOB):continue
        job,name,partition,state,code,seconds,tres,start,end=fields
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=job,name=name,partition=partition,state=state,exit_code=code,seconds=int(seconds),gpus=int(generic[0]) if generic else sum(map(int,typed))))
    current=[r for r in rows if r['name']==JOB];prior=[r for r in rows if r['name']==p.JOB];profile_job=Path(profile['file']).parent.name.removeprefix('profile_')
    need(len(current)==len(prior)==1 and current[0]['job_id']==os.environ['SLURM_JOB_ID'] and current[0]['seconds']<=3600
         and prior[0]['job_id']==profile_job and prior[0]['state']=='COMPLETED' and prior[0]['exit_code']=='0:0' and prior[0]['seconds']<=240
         and all(r['partition']=='gpu' and r['gpus']==1 for r in rows),'One passed profile and one bounded fresh main allocation required')
    save(out/'launch_accounting.json',dict(command=command,file=str(out/'launch_sacct.psv'),sha256=sha(out/'launch_sacct.psv'),matching_rows=rows,main_cap=3600,max_main_attempts=1))


def contract_semantics(contract):
    return {key:({name:{k:v for k,v in info.items() if k!='object_id'} for name,info in value.items()} if key=='tensors' else value) for key,value in contract.items()}


def rng_state(torch):return dict(python=random.getstate(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())


def optimizer_contract(torch,optimizer,params,step):
    group=optimizer.param_groups;need(len(group)==1 and [id(v) for v in group[0]['params']]==[id(v) for v in params.values()],'Exact224 optimizer ownership changed')
    group=group[0];need(group['lr']==2e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01,'Fixed AdamW changed')
    need(len(optimizer.state)==(224 if step else 0),'Adam state ownership differs')
    for parameter in params.values():
        if not step:continue
        state=optimizer.state[parameter]
        need(set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==step
             and all(state[key].shape==parameter.shape and state[key].dtype==torch.float32 and bool(state[key].isfinite().all()) for key in ('exp_avg','exp_avg_sq')),
             'Finite FP32 Adam moments/steps differ')
    return dict(parameter_names=list(params),step=step,state_tensors=224 if step else 0,learning_rate=2e-4,betas=[.9,.999],epsilon=1e-8,weight_decay=.01)


def save_checkpoint(torch,file,model,params,optimizer,step,profile):
    packet=dict(adapter={name:v.detach().cpu().clone() for name,v in params.items()},contract=p.adapter_contract(torch,model),
        peft_config=p.canonical_peft_config(model),optimizer=p.cpu_tree(torch,optimizer.state_dict()),rng=rng_state(torch),step=step,epoch=step//27,
        seed=24,profile_initialization=profile,full_training_final=step==324,no_resume=True,no_checkpoint_selection=True)
    torch.save(packet,file);info={name:p.tensor_info(value) for name,value in packet['adapter'].items()};restored=torch.load(file,map_location='cpu',weights_only=True)
    need(restored['peft_config']==packet['peft_config'] and restored['step']==step
         and {name:p.tensor_info(value) for name,value in restored['adapter'].items()}==info,'Restricted checkpoint serialization changed adapter bytes')
    return dict(file=str(file),sha256=sha(file),step=step,tensors=info,peft_config=packet['peft_config'],optimizer=optimizer_contract(torch,optimizer,params,step),
        rng_recorded=True,restricted_reload_exact=True)


def criterion(rows,records):
    need(len(rows)==len(records)==216 and [r['sid'] for r in rows]==[r['sid'] for r in records],'All216 final outcomes in original order required')
    orient=Counter();families=defaultdict(list);whole=first=0
    for row,record in zip(rows,records):
        correct=record['generated_ids']==row['target_ids'];first_correct=record['generated_ids'][0]==row['target_ids'][0]
        need(record['whole_correct']==correct and record['first_correct']==first_correct,'Outcome/target join differs')
        whole+=correct;first+=first_correct;orient[row['orientation_version']]+=correct;families[row['base_contrast_id']].append(correct)
    need(len(families)==18 and all(len(v)==12 for v in families.values()),'Exactly18 twelve-context families required')
    complete=sum(all(v) for v in families.values());passed=whole>=206 and all(orient[k]>=103 for k in ('original','flipped')) and complete>=16
    return dict(passed=passed,whole_correct=whole,first_correct=first,contexts=216,orientation_correct={k:orient[k] for k in ('original','flipped')},
        complete_families=complete,families=18,thresholds=dict(pooled=206,per_orientation=103,complete_families=16),natural_whole_answer_and_eos=True,training_only=True)


def append_json(file,value):file.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');file.flush()


def run(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);dependency=verify_profile(args.profile);profile=dependency['descriptor'];plan=dependency['plan']
    # JSON-only independent audit handoff; its final literal schema is pending peer review.
    audit=auditor().verify_profile_audit(args.audit,args.profile)
    need(audit['profile_audit_passed'] is True and audit['main_projection_passed'] is True
         and audit['main_projection']==dependency['main_projection'] and dependency['main_projection']['passed'] is True,
         'Passed independent profile and amended resource proof required')
    p.bind(args.audit,dependency['input_bindings']);accounting(out,profile);save(out/'input_bindings.json',dependency['input_bindings'])
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','Exactly one B200 required')
    rows=read(plan['rows_file']);order=read(plan['order_file']);prepared=read(plan['prepared_file'])
    need(len(rows)==216 and order==p.epoch_order(rows) and len(order)==2592 and plan['head_rows']['training']==5760
         and sum(len(rows[item['index']]['target_ids']) for item in order)==5760,'Complete12-epoch schedule/head counts differ')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=loaded.model.eval().requires_grad_(False)
    hardware=p.frozen_joint.live_identity(torch,loaded,plan);refs=list(model.named_parameters());base_before=p.base_metadata(refs)
    save(out/'base_before.json',base_before);need(p.installed({})==plan['packages'],'Actual PEFT/native software changed')
    config=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),profile=profile,
        profile_audit=dict(file=str(args.audit.resolve()),sha256=sha(args.audit)),plan_file=profile['plan_file'],plan_sha256=profile['plan_sha256'],
        profile_projection=dependency['projection'],main_projection=dependency['main_projection'],hardware=hardware,orientation_plan=plan['orientation_plan'],training_stage=plan['training_stage'],
        confirmation_report=plan['confirmation_report'],native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        packages=plan['packages'],data_directory=str(data),checkpoint_directory=str(ckpt),confirmation_predictions_accessed=False,validation_inference=False)
    save(out/'config.json',config)
    contract=p.install_lora(torch,model,out/'actual_peft_config_before_contract.json');params=p.adapter_parameters(model)
    initial=torch.load(profile['initial_adapter_file'],map_location='cpu',weights_only=True);initial_info=dependency['initial']
    need(initial['seed']==24 and initial['profile_initialization_only'] is True and set(initial['adapter'])==set(params)
         and contract_semantics(contract)==contract_semantics(initial['contract'])
         and p.canonical_peft_config(model)==initial['peft_config']==read(profile['adapter_config_file']),'Fresh adapter shape/config differs from passed profile')
    fresh_info={name:p.tensor_info(value) for name,value in params.items()}
    need(fresh_info==initial_info['tensors']=={name:p.tensor_info(value) for name,value in initial['adapter'].items()},'Fresh seed24 adapter must already equal exact profile original bytes')
    with torch.no_grad():
        for name,value in params.items():value.copy_(initial['adapter'][name])
    need({name:p.tensor_info(value) for name,value in params.items()}==fresh_info,'Copied unfitted adapter changed bytes')
    p.check_base(refs,base_before,model);del initial
    # install_lora seeds and consumes the stream. No reseeding occurs here.
    model.train();model.model.visual.eval();optimizer=torch.optim.AdamW(params.values(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    optimizer.zero_grad(set_to_none=True);p.adapter_contract(torch,model)
    initial_record=save_checkpoint(torch,ckpt/'initial_adapter.pt',model,params,optimizer,0,profile);save(out/'initial_adapter.json',initial_record)
    save(out/'initialization.json',dict(fresh_seed24_bytes_exact=True,profile_original_tensors=initial_info['tensors'],full_config_exact=True,
        trainable_parameters=10092544,trainable_tensors=224,no_seed_reset_after_install=True,main_rng_recorded=True,profile_rng_equality_claimed=False,
        profile_two_update_weights_used_for_initialization=False))
    counts=Counter();training_head_rows=training_rows=updates=0
    with (out/'training.jsonl').open('x') as training_log,(out/'optimizer.jsonl').open('x') as optimizer_log:
        for index,item in enumerate(order):
            need(time.perf_counter()-started<=3600,'Fixed main cap exceeded')
            row=rows[item['index']];step=index//8+1;micro=index%8+1;need(row['sid']==item['sid'],'Frozen scene order changed')
            tick=time.perf_counter();packet=p.load_packet(torch,prepared[row['sid']],row)
            capture,calls=p.retained_teacher(torch,model,packet,row,data/f'train_{index+1:04d}_partial.pt',backward=True,profile_kernels=False)
            counts.update(calls);counts['backward']+=1;training_head_rows+=len(row['target_ids']);training_rows+=1
            capture.update(order_index=index,epoch=item['epoch'],epoch_position=item['position'],step=step,microbatch=micro,n_frames=row['n_frames'],counters=calls)
            file=data/f'train_{index+1:04d}.pt';torch.save(capture,file);digest=sha(file)
            grads={name:value.grad for name,value in params.items()};nonnull=all(value is not None for value in grads.values())
            finite=nonnull and bool(torch.stack([value.isfinite().all() for value in grads.values()]).all())
            norms=({letter:float(torch.sqrt(torch.stack([value.float().square().sum() for name,value in grads.items() if f'.lora_{letter}.' in name]).sum()))
                for letter in ('A','B')} if nonnull else None)
            record=dict(order_index=index,epoch=item['epoch'],epoch_position=item['position'],step=step,microbatch=micro,sid=row['sid'],n_frames=row['n_frames'],
                file=str(file),sha256=digest,loss=capture['loss'],position_ce=capture['position_ce'],target_ids=row['target_ids'],target_positions=capture['target_positions'],
                target_rows=len(row['target_ids']),all_gradients_non_none=nonnull,all_gradients_finite=finite,gradient_tensors=sum(v is not None for v in grads.values()),
                accumulated_gradient_norms=norms,counters=calls,backward_coefficient=1/8)
            append_json(training_log,record);need(nonnull and finite,'All224 gradients must exist and be finite; zero norms are permitted')
            p.check_base(refs,base_before,model);torch.cuda.synchronize();save(out/f'timing_train_{index+1:04d}.json',dict(order_index=index,seconds=time.perf_counter()-tick))
            if micro==8:
                tick=time.perf_counter();versions={name:value._version for name,value in params.items()}
                grad_norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.);need(bool(torch.isfinite(grad_norm)),'Nonfinite adapter gradient norm')
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;counts['optimizer']+=1
                need(all(value._version>versions[name] and bool(value.isfinite().all()) and value.grad is None for name,value in params.items()),'Finite adapter-only update required')
                proof=optimizer_contract(torch,optimizer,params,step);p.check_base(refs,base_before,model)
                append_json(optimizer_log,dict(step=step,completed_scene_presentations=index+1,unclipped_gradient_norm=float(grad_norm),clip=1.,optimizer=proof,base_frozen=True))
                torch.cuda.synchronize();save(out/f'timing_optimizer_{step:03d}.json',dict(step=step,seconds=time.perf_counter()-tick))
    need(training_rows==2592 and updates==324 and training_head_rows==5760,'Complete fixed training horizon required')
    final_record=save_checkpoint(torch,ckpt/'final_adapter.pt',model,params,optimizer,324,profile);save(out/'final_adapter.json',final_record)
    need(final_record['peft_config']==initial_record['peft_config'],'Adapter configuration changed during training')
    p.check_base(refs,base_before,model);final_versions={name:value._version for name,value in model.named_parameters()}
    model.eval().requires_grad_(False);evaluations=[];evaluation_head_rows=0
    for index,row in enumerate(rows):
        need(time.perf_counter()-started<=3600,'Fixed cap exceeded before final evaluation')
        tick=time.perf_counter();packet=p.load_packet(torch,prepared[row['sid']],row);evidence={}
        try:result=p.generate_joint(model,loaded.processor,packet['bundle'],native_identity_sha256=plan['native_identity_sha256'],capture_head=True,evidence=evidence)
        except BaseException:
            file=data/f'eval_{index:03d}_partial.pt';torch.save(p.cpu_tree(torch,dict(sid=row['sid'],evidence=evidence)),file);raise
        file=data/f'eval_{index:03d}.pt';torch.save(dict(sid=row['sid'],result=result),file);digest=sha(file)
        counts.update(result['counters']);evaluation_head_rows+=len(result['generated_ids'])
        record=dict(index=index,sid=row['sid'],n_frames=row['n_frames'],orientation_version=row['orientation_version'],base_contrast_id=row['base_contrast_id'],
            file=str(file),sha256=digest,generated_ids=result['generated_ids'],target_ids=row['target_ids'],gold=row['gold'],
            whole_correct=result['generated_ids']==row['target_ids'],first_correct=result['generated_ids'][0]==row['target_ids'][0],
            completed=result['completed'],truncated=result['truncated'],text=result['text'],raw_text=result['raw_text'],counters=result['counters'])
        evaluations.append(record);save(out/f'eval_{index:03d}.json',record);p.check_base(refs,base_before,model);torch.cuda.synchronize()
        save(out/f'timing_eval_{index:03d}.json',dict(index=index,seconds=time.perf_counter()-tick))
    save(out/'evaluations.json',evaluations);competence=criterion(rows,evaluations);save(out/'competence.json',competence)
    need(counts['model']==counts['language']==counts['norm']==counts['head']==2592+evaluation_head_rows<=3456
         and counts['visual']==2808 and counts['backward']==2592 and counts['optimizer']==324
         and all(counts.get(key,0)==0 for key in ('broadcast','fusion','conditioning','probe_head')),'Actual complete main call inventory differs')
    need(216<=evaluation_head_rows<=864 and final_versions=={name:value._version for name,value in model.named_parameters()}
         and all(not value.requires_grad and value.grad is None for value in model.parameters()),'Final frozen evaluation changed state')
    p.check_base(refs,base_before,model)
    need({name:p.tensor_info(value) for name,value in params.items()}==final_record['tensors'] and p.canonical_peft_config(model)==final_record['peft_config'],
         'Final natural evaluation changed saved endpoint/config')
    files={str(file):sha(file) for directory in (data,ckpt) for file in sorted(directory.rglob('*')) if file.is_file()};save(out/'artifacts.json',files)
    endpoint=dict(passed=True,computational_passed=True,competence=competence,counters=dict(counts),training_head_rows=training_head_rows,
        evaluation_head_rows=evaluation_head_rows,total_head_rows=training_head_rows+evaluation_head_rows,completed_scene_presentations=2592,completed_updates=324,
        base_parameters_unchanged=True,all_adapter_gradients_finite=True,final_adapter_frozen=True,final_adapter_unchanged_during_evaluation=True,full_peft_config_exact=True,
        base_before_file=str(out/'base_before.json'),base_before_sha256=sha(out/'base_before.json'),
        initial_adapter_file=str(out/'initial_adapter.json'),initial_adapter_sha256=sha(out/'initial_adapter.json'),
        final_adapter_file=str(out/'final_adapter.json'),final_adapter_sha256=sha(out/'final_adapter.json'))
    save(out/'endpoint.json',endpoint)
    return dict(**config,passed=True,completed=True,phase='main',computational_passed=True,competence=competence,
        endpoint_file=str(out/'endpoint.json'),endpoint_sha256=sha(out/'endpoint.json'),training_file=str(out/'training.jsonl'),training_sha256=sha(out/'training.jsonl'),
        optimizer_file=str(out/'optimizer.jsonl'),optimizer_sha256=sha(out/'optimizer.jsonl'),evaluations_file=str(out/'evaluations.json'),evaluations_sha256=sha(out/'evaluations.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),counters=dict(counts),training_head_rows=training_head_rows,
        evaluation_head_rows=evaluation_head_rows,total_head_rows=training_head_rows+evaluation_head_rows,downstream_eligibility=competence['passed'],
        downstream_release=False,valid_negative_outcomes_retained=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True);args=parser.parse_args()
    p.native.require_slurm(gpu=True);out=OUT/f'run_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out);save(out/'request.json',dict(phase='main',profile=str(args.profile.resolve()),audit=str(args.audit.resolve()),source_sha256=frozen))
    try:
        value=run(args,out,frozen,started);need(sources()==frozen and inherited_sources()==value['inherited_source_sha256'],'Frozen main sources changed')
        elapsed=time.perf_counter()-started;need(elapsed<=3600,'Fixed main cap exceeded');save(out/'summary.json',dict(value,elapsed_seconds=elapsed))
        print(json.dumps(dict(passed=True,directory=str(out),competence=value['competence'])),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()
