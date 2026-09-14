"""Actual original-training-width profile and independent CPU audit.

No optimizer update, feature extraction, benchmark scoring or fit release.
All tensor/model operations require the appropriate Slurm allocation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
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
from scripts import profile_mmred_official_native_memory as original
from scripts import report_mmred_official_native_memory as independent
from scripts import stage_mmred_official_training as compact
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=original.p
runtime=original.runtime
PROTOCOL='mmred_official_training_boundary'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_TRAINING_BOUNDARY_PROFILE.md'
PROPOSAL_SHA='45f5b3e55bf80af12fc9a891374d995a4926a9341438a2da6a14a3d78f74da25'
OWN=('scripts/profile_mmred_official_training_boundary.py',PROPOSAL,
     'slurm/mmred_official_training_boundary_profile.sbatch','slurm/mmred_official_training_boundary_report.sbatch')
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_training_boundary'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_training_boundary')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/mmred_official_training_boundary')
AUDIT=original.OUT/'report_443990/summary.json'
AUDIT_SHA='183087eb0bd6ec18a0df4bb19b4dda009b20163863d25db39658460552a3ea2b'
COMPACT=compact.OUT/'merge_443989/summary.json'
COMPACT_SHA='ece343e085068195b66f4ab316f92d13c160022e396555eaebabc1fd088a5320'
LENGTHS=(1,2,4,8,16)
ARMS=original.ARMS
WIDTHS=compact.WIDTHS
JOB='mmred_official_training_boundary_profile'
POLICY=dict(protocol=PROTOCOL,gpu_seconds=900,cpu_report_seconds=1800,cpu_cores=4,memory_gib=16,
    maximum_attempts=1,gpu_count=1,maximum_selected_cases=25,natural_trajectories=15,maximum_new_tokens=50,
    maximum_model_head_calls=825,maximum_cpu_head_calls=825,maximum_cpu_head_rows=1350,maximum_target_rows=8,
    maximum_backwards=75,optimizer_steps=0,vision_calls=0,extra_GPU_heads=0,backward_loss_divisor=8,
    no_fit=True,no_efficacy=True,no_full_fit_release=True,no_prefix_reuse=True,
    native_tv_max=.02,ce_absolute_tolerance=2e-6,core_atol=2e-4,core_rtol=2e-4,cpu_argmax_gate=False)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Boundary proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Boundary input changed: '+str(path));bindings[str(path)]=digest
    return digest


def descriptor(path,bindings):
    path=Path(path).resolve();bind(path,bindings);summary=read(path)
    result=dict(file=str(path),sha256=sha(path))
    for key in ('plan','analysis'):
        if key+'_file' in summary:
            result.update({key+'_file':summary[key+'_file'],key+'_sha256':bind(summary[key+'_file'],bindings,summary[key+'_sha256'])})
    return result


def selection(records):
    need(len(records)==4000 and [r['index'] for r in records]==list(range(4000)) and len({r['sid'] for r in records})==4000,
         'Canonical4000 ordered training descriptors required')
    witnesses=[];chosen=set();natural={arm:[] for arm in ARMS}
    for n in LENGTHS:
        group=[r for r in records if r['n']==n];need(len(group)==800,'Each training length requires800 rows')
        owners={}
        for key in WIDTHS:
            value=max(r[key] for r in group);row=next(r for r in group if r[key]==value)
            owners[key]=row['index'];chosen.add(row['index'])
            witnesses.append(dict(n=n,metric=key,maximum=value,index=row['index'],sid=row['sid']))
        for arm in ARMS:natural[arm].append(owners['ordinary_prompt' if arm=='ordinary' else 'compressed_prompt'])
    selected=sorted(chosen)
    need(5<=len(selected)<=25 and max(r['target_rows'] for r in records)<=8,'Bounded union/target length differs')
    return dict(indices=selected,witnesses=witnesses,natural=natural,coordinate_maxima_only=True,
                all_descriptor_sha256=object_sha(records),no_joint_dominating_example_assumed=True)


def self_test():
    rows=[]
    for n in LENGTHS:
        for i in range(800):
            row=dict(index=len(rows),sid=str(len(rows)),n=n,**{k:2 for k in WIDTHS})
            if i in (3,4):row['ordinary_prompt']=7
            if i==5:row['compressed_teacher']=9
            rows.append(row)
    result=selection(rows)
    need(result['indices']==[base+i for base in range(0,4000,800) for i in (0,3,5)]
         and result['natural']['ordinary']==[base+3 for base in range(0,4000,800)]
         and result['natural']['mass']==list(range(0,4000,800)),'First ties/union/original ordering fixture failed')
    need(3*25+15*50==825 and 3*25*8+15*50==1350,'Replay bound fixture failed')
    old=dict(setup_seconds=2.,arms={});fresh={}
    for arm in ARMS:
        old['arms'][arm]=dict(microbatches=[dict(n=n,seconds=10. if n==2 else 1.,max_allocated_bytes=100) for n in LENGTHS],
            natural=[dict(n=n,kind='ordinary' if arm=='ordinary' else 'cold',fifty_token_seconds=12. if n==2 else 1.) for n in LENGTHS],
            optimizer_seconds=[.2,.3],checkpoint_seconds=.5)
        fresh[arm]=dict(teachers=[dict(n=n,seconds=4.,max_allocated_bytes=200) for n in LENGTHS],
            natural=[dict(n=n,fifty_token_seconds=8.) for n in LENGTHS])
    proof=projection(old,3.,fresh)
    need(all(a['microbatch_seconds_by_n']=={str(n):10. if n==2 else 4. for n in LENGTHS}
         and a['generation_fifty_token_seconds_by_n']=={str(n):12. if n==2 else 8. for n in LENGTHS}
         and a['setup_seconds']==3. and a['original_optimizer_seconds']==.3 for a in proof['arms'].values()),
         'Actual projection must preserve an older maximum and accept a larger boundary maximum')
    return dict(passed=True,groups=3,model_calls=0,head_calls=0)


def dependencies(features_path,bindings):
    from scripts import harvest_mmred_official_training_features as harvest
    bind(AUDIT,bindings,AUDIT_SHA);audit=independent.verify_report(AUDIT)
    need(audit['cold_training_components_passed'] is True and audit['original_profile_passed'] is False
         and audit['cache_parity_passed'] is False and audit['no_full_fit_release'],'Exact cold-only443978 independent audit required')
    bind(COMPACT,bindings,COMPACT_SHA);prepared=compact.verify_stage(COMPACT,streaming=True)
    cache=harvest.verify_stage(features_path,streaming=True)
    feature_ref=descriptor(features_path,bindings)
    refs=dict(feature_stage={k:feature_ref[k] for k in ('file','sha256')},compact_stage=descriptor(COMPACT,bindings),
              independent_profile_audit=descriptor(AUDIT,bindings))
    need(cache['compact_stage']==refs['compact_stage'] and cache['native_identity']==prepared['native_identity']
         and cache['native_identity_sha256']==prepared['native_identity_sha256']
         and cache['rows_file']==prepared['rows_file'] and cache['cases_file']==prepared['cases_file']
         and cache['independent_profile_audit']==refs['independent_profile_audit'],
         'Exact feature/compact/native ancestry differs')
    records=read(prepared['cases_file']);features=read(cache['feature_index_file'])
    bind(prepared['cases_file'],bindings);bind(cache['feature_index_file'],bindings)
    need(len(features)==4000 and [(r['index'],r['sid'],r['n']) for r in features]==[(r['index'],r['sid'],r['n']) for r in records],
         'Complete feature/compact row join differs')
    old=read(audit['profile']['analysis_file']);oldconfig=read(audit['profile']['config_file'])
    bind(audit['profile']['analysis_file'],bindings,audit['profile']['analysis_sha256'])
    bind(audit['profile']['config_file'],bindings,audit['profile']['config_sha256'])
    need(oldconfig['native_identity']==cache['native_identity'] and oldconfig['hardware']['precision']==cache['precision'],
         'Boundary/cache/original native model or precision differs')
    inherited={}
    for plan in (cache,prepared,read(AUDIT)):
        for name,digest in {**plan['source_sha256'],**plan['inherited_source_sha256']}.items():
            need(name not in inherited or inherited[name]==digest,'Conflicting boundary ancestor');inherited[name]=digest
    need(all(sha(REPO/name)==digest for name,digest in inherited.items()),'Boundary ancestor source changed')
    return cache,records,features,selection(records),refs,old,oldconfig,inherited


def archive(out,inherited):
    (out/'source').mkdir()
    for name,digest in {**sources(),**inherited}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Boundary source archive changed')


def load_pair(torch,index,records,features,native_identity,bindings):
    item=records[index];ref=features[index]
    for entry in (item,ref):bind(entry['file'],bindings,entry['sha256'])
    packet=torch.load(item['file'],map_location='cpu',weights_only=True);feature=torch.load(ref['file'],map_location='cpu',weights_only=True)
    need(set(packet)=={'case','omitted_pixel_values'} and packet['omitted_pixel_values']==item['omitted_pixel_values'],
         'Exact compact packet required')
    case=packet['case'];need('pixel_values' not in case['inputs'] and compact.widths(case)=={k:item[k] for k in WIDTHS}
         and object_sha(compact.tensor_tree(torch,p,case))==item['compact_identity_sha256']
         and case['metadata']['sid']==item['sid'] and case['metadata']['n']==item['n'],'Compact case identity differs')
    need(ref['compact']==dict(file=item['file'],sha256=item['sha256'],compact_identity_sha256=item['compact_identity_sha256'])
         and ref['input_identity']==dict(pixel_values=packet['omitted_pixel_values'],image_grid_thw=p.tensor_info(case['inputs']['image_grid_thw']))
         and ref['coordinate_identity']==original.tensor_state(case['coordinates']) and ref['finite'] is True,'Feature source/coordinate join differs')
    need(feature['index']==index and feature['sid']==item['sid'] and feature['input_identity']==ref['input_identity']
         and feature['compact']==ref['compact'] and feature['native_identity_sha256']==native_identity
         and feature['n']==item['n'] and feature['coordinate_identity']==ref['coordinate_identity'],
         'Saved native feature owner differs')
    values=feature['features'];need(p.tensor_info(values)==ref['tensor'] and values.dtype==torch.float16
         and values.shape==(item['n']*196,3584) and bool(values.isfinite().all()),'Exact finite native feature rows required')
    return case,values


def allocation(out,expected_job,finished=False):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    raw=subprocess.run(command,text=True,capture_output=True,check=True).stdout
    file=out/('completed_sacct.psv' if finished else 'launch_sacct.psv');file.write_text(raw);rows=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=8 or fields[1]!=JOB:continue
        identifier,name,partition,state,code,seconds,tres,limit=fields
        gpu=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=code,seconds=int(seconds),
                         gpus=int(gpu[0]) if gpu else sum(map(int,typed)),limit_minutes=int(limit)))
    need(len(rows)==1 and rows[0]['job_id']==expected_job and rows[0]['partition']=='gpu' and rows[0]['gpus']==1
         and rows[0]['seconds']<=900 and rows[0]['limit_minutes']==15,'One900-second boundary allocation required')
    if finished:need(rows[0]['state']=='COMPLETED' and rows[0]['exit_code']=='0:0','Completed boundary producer required')
    return dict(rows=rows,file=str(file),sha256=sha(file))


def projection(old,setup,arms):
    result=dict(training_only=True,main_release=False,shared_feature_harvest_included=False,
        diagnostic100_generation_included=False,validation_test_included=False,no_N32_estimate=True,arms={})
    for arm in ARMS:
        a=arms[arm];parent=old['arms'][arm]
        T={n:max([r['seconds'] for r in parent['microbatches'] if r['n']==n]+[r['seconds'] for r in a['teachers'] if r['n']==n]) for n in LENGTHS}
        G={n:max([r['fifty_token_seconds'] for r in parent['natural'] if r['n']==n and r['kind'] in ('ordinary','cold')]
                 +[r['fifty_token_seconds'] for r in a['natural'] if r['n']==n]) for n in LENGTHS}
        O=max(parent['optimizer_seconds']);C=parent['checkpoint_seconds'];S=max(setup,old['setup_seconds'])
        params=10092544+(0 if arm=='ordinary' else 469504);tensors=224+(0 if arm=='ordinary' else 3)
        reserve=2*4*params+4*tensors
        oldpeak=max(r['max_allocated_bytes'] for r in parent['microbatches'])
        peak=max(r['max_allocated_bytes'] for r in a['teachers'])
        result['arms'][arm]=dict(microbatch_seconds_by_n=T,generation_fifty_token_seconds_by_n=G,
            original_optimizer_seconds=O,original_checkpoint_seconds=C,setup_seconds=S,
            training_without_features_or_diagnostics_seconds=S+1.25*(sum(2400*T[n] for n in LENGTHS)+1500*O+2*C)+60,
            old_teacher_peak_allocated_bytes=oldpeak,boundary_teacher_peak_allocated_bytes=peak,
            exact_AdamW_persistent_state_reserve_bytes=reserve,estimated_training_peak_allocated_bytes=max(oldpeak,peak+reserve),
            optimizer_peak_not_newly_measured=True,all_old_and_new_timing_observations_retained=True)
    return json.loads(json.dumps(result))



def profile(args,out,data,ckpt,started,bindings,progress):
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.native_visual_memory import NativeVisualMemory
    torch.set_num_threads(4)
    need(os.environ.get('SLURM_JOB_NAME')==JOB,'Exact boundary GPU job required')
    resources=allocation(out,os.environ['SLURM_JOB_ID'])
    cache,records,features,selected,refs,old,oldconfig,inherited=dependencies(args.features,bindings)
    archive(out,inherited);tests=self_test();save(out/'tests.json',tests)
    save(out/'selection.json',selected)
    data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    loaded=load_runtime(str(original.preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    hardware=p.frozen_joint.live_identity(torch,loaded,cache)
    need(p.installed({})==cache['packages'],'Exact package implementations required')
    base_refs=list(model.named_parameters());base=p.base_metadata(base_refs);save(out/'base_before.json',base)
    contract=p.install_lora(torch,model,out/'actual_peft_config.json');params=p.adapter_parameters(model)
    config=p.canonical_peft_config(model)
    bind(oldconfig['initial']['file'],bindings,oldconfig['initial']['sha256'])
    oldinitial=torch.load(oldconfig['initial']['file'],map_location='cpu',weights_only=True)
    initial=original.cpu_state(torch,params);memory_initial=NativeVisualMemory().state_dict()
    need(config==oldconfig['peft_config']==oldinitial['peft_config']
         and original.tensor_state(initial)==oldconfig['initial']['adapter_tensors']==original.tensor_state(oldinitial['adapter'])
         and original.tensor_state(memory_initial)==oldconfig['initial']['memory_tensors']==original.tensor_state(oldinitial['memory']),
         'Fresh original profile initialization/configuration must match exactly')
    rng=dict(cpu=torch.random.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all())
    initial_file=ckpt/'initial.pt';torch.save(dict(adapter=initial,memory=memory_initial,peft_config=config,contract=contract,
        rng=rng,seed=24,updates=0),initial_file)
    initial_ref=dict(file=str(initial_file),sha256=sha(initial_file),adapter_tensors=original.tensor_state(initial),
                     memory_tensors=original.tensor_state(memory_initial))
    configuration=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=sources(),inherited_source_sha256=inherited,
        **refs,selection=selected,initial=initial_ref,peft_config=config,hardware=hardware,
        native_identity=cache['native_identity'],native_identity_sha256=cache['native_identity_sha256'],
        precision=cache['precision'],data_directory=str(data),checkpoint_directory=str(ckpt),
        old_initial=oldconfig['initial'],no_full_fit_release=True)
    save(out/'config.json',configuration)
    counts=Counter(dict.fromkeys(('model','backbone','visual','language','norm','head','backward','optimizer'),0))
    layers=[0]*28;arms={};progress.update(counters=counts,arms=arms)
    def check(memory=None):
        p.check_base(base_refs,base,model)
        need(original.tensor_state(params)==initial_ref['adapter_tensors'],'Profile modified initialized adapter bytes')
        if memory is not None:need(original.tensor_state(dict(memory.named_parameters()))==initial_ref['memory_tensors'],'Profile modified initialized memory bytes')
        need(time.perf_counter()-started<900,'Boundary GPU cap exceeded')
    def zero(memory):
        for value in params.values():value.grad=None
        if memory is not None:
            for value in memory.parameters():value.grad=None
    with ExitStack() as hooks:
        def bump(key):
            def callback(*_):counts[key]+=1
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.visual,'visual'),
            (model.model.language_model,'language'),(model.model.language_model.norm,'norm'),(model.lm_head,'head')):
            hooks.callback(module.register_forward_pre_hook(bump(key)).remove)
        for i,layer in enumerate(model.model.language_model.layers):
            def layer_hook(module,args,i=i):layers[i]+=1
            hooks.callback(layer.register_forward_pre_hook(layer_hook).remove)
        torch.cuda.synchronize();setup=time.perf_counter()-started
        for arm in ARMS:
            original.restore(torch,params,initial)
            memory=None if arm=='ordinary' else NativeVisualMemory().to(model.device)
            torch.random.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
            original.model_mode(model,params,True);zero(memory)
            provenance=dict(native_identity_sha256=cache['native_identity_sha256'],lora=initial_ref,
                memory=None if memory is None else dict(checkpoint=initial_ref,tensors=initial_ref['memory_tensors']))
            entry=dict(teachers=[],natural=[],initial=initial_ref,provenance=provenance,updates=0,
                common_initial_bytes_exact=True,post_install_rng_restored=True)
            arms[arm]=entry
            versions={name:value._version for name,value in params.items()}
            memory_versions={} if memory is None else {name:value._version for name,value in memory.named_parameters()}
            for index in selected['indices']:
                zero(memory);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();before_layers=list(layers)
                evidence=dict(parameter_state=initial_ref,feature_packet=features[index],compact_packet=records[index])
                file=data/f'{arm}_teacher_{index}.pt'
                try:
                    case,values=load_pair(torch,index,records,features,cache['native_identity_sha256'],bindings)
                    values=values.to(model.device)
                    if memory is None:result=original.ordinary_teacher(torch,model,case,values,evidence=evidence)
                    else:
                        capture=memory(values,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
                        evidence.update(memory_capture=capture,coordinate_identity=original.tensor_state(case['coordinates']))
                        result=runtime.teacher_forward(model,case['text'],capture['tokens'].unsqueeze(0),evidence=evidence)
                    original.retain(torch,file,index,arm,'teacher',evidence,backward_completed=False)
                    evidence['backward_started']=True;(result['loss']/8).backward();counts['backward']+=1
                    evidence['backward_completed']=True
                    ref=original.retain(torch,file,index,arm,'teacher',evidence,backward_completed=True)
                    gradients=original.grad_record(torch,params,memory)
                    need(gradients['all_present'] and gradients['all_finite'] and gradients['norms']['A']==0,
                         'Fresh zero-B backward must retain finite gradients and zero-A gradient')
                    if arm=='normalized':need(gradients['norms']['mass_direction']==0,'Normalized control cannot acquire mass gradient')
                    check(memory)
                    need(versions=={k:v._version for k,v in params.items()}
                         and memory_versions==({} if memory is None else {k:v._version for k,v in memory.named_parameters()}),
                         'A no-step call changed parameter versions')
                    row=dict(index=index,sid=records[index]['sid'],n=records[index]['n'],arm=arm,**ref,
                        loss=evidence['loss'],target_rows=len(case['target_ids']),gradients=gradients,
                        layer_forward_calls=[a-b for a,b in zip(layers,before_layers)],
                        max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
                    need(row['layer_forward_calls']==[1]*28,'Teacher decoder recomputation changed')
                    save(out/f'{arm}_teacher_{index}.json',row);torch.cuda.synchronize();row['seconds']=time.perf_counter()-tick
                    save(out/f'{arm}_teacher_{index}_timing.json',row);entry['teachers'].append(row)
                    del result,evidence,case,values
                except BaseException:
                    original.retain(torch,file,index,arm,'teacher',evidence,partial=True);raise
            zero(memory);original.model_mode(model,params,False)
            if memory is not None:memory.eval().requires_grad_(False)
            for index in selected['natural'][arm]:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                evidence=dict(parameter_state=initial_ref,feature_packet=features[index],compact_packet=records[index])
                file=data/f'{arm}_natural_{index}.pt';kind='ordinary' if arm=='ordinary' else 'cold'
                try:
                    case,values=load_pair(torch,index,records,features,cache['native_identity_sha256'],bindings);values=values.to(model.device)
                    with torch.inference_mode():
                        if memory is None:result=original.ordinary_generate(torch,model,loaded.processor,case,values,evidence)
                        else:
                            capture=memory(values,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
                            evidence.update(memory_capture=capture,coordinate_identity=original.tensor_state(case['coordinates']))
                            result=runtime.generate_cold(model,loaded.processor,case['text'],capture['tokens'].unsqueeze(0),
                                provenance=provenance,max_tokens=50,evidence=evidence)
                    ref=original.retain(torch,file,index,arm,kind,evidence,result=result,text_case_index=None)
                    check(memory);need(versions=={k:v._version for k,v in params.items()}
                        and memory_versions==({} if memory is None else {k:v._version for k,v in memory.named_parameters()}),'Natural call changed parameter versions')
                    row=dict(index=index,sid=records[index]['sid'],n=records[index]['n'],arm=arm,kind=kind,**ref,
                        tokens=len(result['generated_ids']),max_allocated_bytes=torch.cuda.max_memory_allocated(),
                        max_reserved_bytes=torch.cuda.max_memory_reserved(),efficacy_scored=False)
                    save(out/f'{arm}_natural_{index}.json',row);torch.cuda.synchronize();row['seconds']=time.perf_counter()-tick
                    row['fifty_token_seconds']=row['seconds']*50/row['tokens']
                    save(out/f'{arm}_natural_{index}_timing.json',row);entry['natural'].append(row)
                    del result,evidence,case,values
                except BaseException:
                    original.retain(torch,file,index,arm,kind,evidence,partial=True);raise
            save(out/f'{arm}_measurements.json',entry);check(memory);del memory
    generated=sum(r['tokens'] for arm in arms.values() for r in arm['natural']);D=len(selected['indices'])
    expected=dict(model=3*D+generated,backbone=3*D+generated,visual=0,language=3*D+generated,norm=3*D+generated,
                  head=3*D+generated,backward=3*D,optimizer=0)
    need(dict(counts)==expected and layers==[expected['model']]*28 and generated<=750 and expected['head']<=825,'Complete boundary call inventory differs')
    check();save(out/'base_after.json',p.base_metadata(base_refs));save(out/'input_bindings.json',bindings)
    result=dict(protocol=PROTOCOL,passed=True,completed=True,arms=arms,counters=expected,decoder_layer_calls=layers,
        setup_seconds=setup,projection=projection(old,setup,arms),selection=selected,
        teacher_head_rows=3*sum(records[i]['target_rows'] for i in selected['indices']),natural_head_rows=generated,
        resources=resources,parameter_bytes_and_versions_unchanged=True,optimizer_steps=0,vision_calls=0,
        original_profile_passed=False,original_cache_parity_passed=False,no_efficacy=True,no_full_fit_release=True)
    save(out/'analysis.json',result)
    return dict(configuration,passed=True,completed=True,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'))



def natural_audit(torch,raw,case,entry,initial,config,modules,data,key):
    e=raw['evidence'];result=raw['result'];arm=raw['arm'];ids=result['generated_ids'];T=len(ids)
    need(1<=T<=50 and all(type(v) is int and 0<=v<152064 for v in ids)
         and not any(v in runtime.EOS_IDS for v in ids[:-1]) and result['completed']==(ids[-1] in runtime.EOS_IDS)
         and result['truncated']==(ids[-1] not in runtime.EOS_IDS) and (result['completed'] or T==50),'Natural EOS/length differs')
    expected=dict(model=T,visual=0,language=T,norm=T,head=T,prefix_decoder=0)
    need(result['counters']==e['counters']==expected and result['rope_restored'] and result['hooks_removed']
         and e['rope_restored'] and e['hooks_removed'] and e['generated_ids']==ids
         and len(e['profile_head'])==len(e['native_inputs'])==len(e['native_positions'])==len(e['shapes'])==T,
         'Natural actual call/cleanup inventory differs')
    need(result['raw_logits'].shape==(T,152064) and result['raw_logits'].dtype==torch.float32
         and raw['text_case_index'] is None,'Own-question complete natural vectors required')
    if arm=='ordinary':
        need(raw['phase']=='ordinary' and result['native_positions_preserved'],'Original-position ordinary generation required')
        W=case['metadata']['prompt_width'];first=case['position_ids']
    else:
        need(raw['phase']=='cold' and result['prefix_bytes_unchanged'] is False and result['parameter_versions_unchanged']
             and result['metadata']==e['metadata'],'Only cold native memory generation required')
        W=case['text']['prefix']['opening_ids'].shape[1]+33+case['text']['suffix']['input_ids'].shape[1]
        first=torch.arange(W).view(1,1,W).expand(4,1,W)
        need(result['generation']==dict(max_new_tokens=50,do_sample=False,num_beams=1,native_eos_token_ids=[151645,151643],
             repetition_penalty=1.,vocabulary_mask=False,other_logits_processors=False,logits_to_keep=1)
             and e['metadata']['caller_provenance']==entry['provenance']
             and e['metadata']['actual_lora_tensors']==original.tensor_state(initial['adapter']),'Initial native generation policy differs')
    heads=[]
    for i in range(T):
        start=0 if i==0 else W+i-1;length=W if i==0 else 1
        pos=first if i==0 else torch.full((4,1,1),start,dtype=torch.long)
        if arm=='ordinary' and i:pos[1:]+=case['rope_deltas'].reshape(1,1,1)
        inp=e['native_inputs'][i];cap=e['profile_head'][i];shape=e['shapes'][i]
        need(torch.equal(e['native_positions'][i],pos) and torch.equal(result['native_positions'][i],pos)
             and inp['past_length']==start and not inp['has_pixels'] and inp['attention_mask'].tolist()==[[1]*(start+length)]
             and inp['cache_position'].tolist()==list(range(start,start+length)),'Natural positions/mask/cache differs')
        need(shape['norm_input_shape']==shape['norm_output_shape']==[1,length,3584]
             and shape['head_input_shape']==[1,1,3584] and shape['head_output_shape']==[1,1,152064],'Actual natural head shape differs')
        if i:need(inp['input_ids'].tolist()==[[ids[i-1]]] and inp['inputs_embeds_identity'] is None,'Natural trajectory consumed another prefix')
        else:
            need(inp['input_ids'] is None and inp['inputs_embeds_identity'] is not None,'Cold generation must consume assembled native embeddings')
            if arm!='ordinary':need(inp['inputs_embeds_identity']==e['metadata']['full_embedding_identity'],'Actual memory embedding identity differs')
        need(torch.equal(result['raw_logits'][i],cap['head_logits'][0,-1].float()) and ids[i]==int(result['raw_logits'][i].argmax())
             and original.tensor_state(cap)==original.tensor_state(result['profile_head'][i]),'Raw logit/native head/greedy ID differs')
        heads.append(independent.head_audit(torch,cap,modules,data,key+f'_token_{i}'))
    return dict(passed=all(x['passed'] for x in heads),index=raw['case_index'],arm=arm,generated_ids=ids,
        completed=result['completed'],truncated=result['truncated'],head_audits=heads,cpu_head_calls=T,cpu_head_rows=T,efficacy_scored=False)


def report(args,out,data,started,bindings,progress):
    import torch
    torch.set_num_threads(4);directory=args.report.resolve().parent;summary=read(args.report)
    need(directory.parent==OUT and directory.name.startswith('profile_') and summary['passed'] is summary['completed'] is True
         and summary['source_sha256']==sources() and summary['policy']==POLICY and not (directory/'failure.json').exists(),
         'Passed completed boundary producer required')
    bind(args.report,bindings);config=read(directory/'config.json');bind(directory/'config.json',bindings)
    bind(summary['analysis_file'],bindings,summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    cache,records,features,selected,refs,old,oldconfig,inherited=dependencies(config['feature_stage']['file'],bindings)
    need(config['selection']==analysis['selection']==selected and all(config[k]==v for k,v in refs.items())
         and config['source_sha256']==sources() and config['inherited_source_sha256']==inherited
         and summary['inherited_source_sha256']==inherited and config['policy']==POLICY,
         'Boundary selection/ancestry differs')
    need(config['native_identity']==cache['native_identity'] and config['native_identity_sha256']==cache['native_identity_sha256']
         and config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['precision']==cache['precision'],
         'Boundary actual native identity differs')
    for field in ('input_bindings','artifacts'):
        bind(summary[field+'_file'],bindings,summary[field+'_sha256'])
        for file,digest in read(summary[field+'_file']).items():bind(file,bindings,digest)
    independent.archive(directory,sources(),inherited,bindings);archive(out,inherited);data.mkdir(parents=True,exist_ok=False)
    initial=independent.initial_audit(torch,config['initial'],config['peft_config'],bindings)
    need(config['old_initial']==oldconfig['initial'] and config['initial']['adapter_tensors']==oldconfig['initial']['adapter_tensors']
         and config['initial']['memory_tensors']==oldconfig['initial']['memory_tensors']
         and config['peft_config']==oldconfig['peft_config'],'Boundary initial bytes/config differ from original profile')
    need(read(directory/'actual_peft_config.json')==config['peft_config']
         and read(directory/'base_before.json')==read(directory/'base_after.json'),'Native base/configuration changed')
    modules=independent.native_modules(torch,cache,bindings);tests=self_test();save(out/'tests.json',tests)
    teachers=[];naturals=[];memories=[];progress.update(teachers=teachers,naturals=naturals,memory=memories)
    def load(index):return load_pair(torch,index,records,features,cache['native_identity_sha256'],bindings)
    def raw_packet(row):
        bind(row['file'],bindings,row['sha256']);return torch.load(row['file'],map_location='cpu',weights_only=True)
    def owned(raw,index,arm,case):
        need(raw['case_index']==index and raw['arm']==arm and raw['evidence']['parameter_state']==config['initial']
             and raw['evidence']['feature_packet']==features[index] and raw['evidence']['compact_packet']==records[index],
             'Every capture needs exact initial/feature/compact ownership')
        if arm!='ordinary':return independent.memory_links(torch,raw['evidence'],case,features[index],config['initial'])
    def publish():
        save(out/f'progress_{len(teachers)}_{len(naturals)}.json',dict(teachers=teachers,naturals=naturals,memory=memories))
    for arm in ARMS:
        entry=analysis['arms'][arm]
        need(entry['initial']==config['initial'] and entry['updates']==0 and entry['common_initial_bytes_exact']
             and entry['post_install_rng_restored'] and [r['index'] for r in entry['teachers']]==selected['indices']
             and [r['index'] for r in entry['natural']]==selected['natural'][arm],'Exact boundary arm/case population differs')
        provenance=dict(native_identity_sha256=cache['native_identity_sha256'],lora=config['initial'],
            memory=None if arm=='ordinary' else dict(checkpoint=config['initial'],tensors=config['initial']['memory_tensors']))
        need(entry['provenance']==provenance,'Actual initial provenance differs')
        for row in entry['teachers']:
            index=row['index'];case,values=load(index);raw=raw_packet(row);capture=owned(raw,index,arm,case);key=f'{arm}_teacher_{index}'
            metric=independent.teacher_audit(torch,raw,case,arm,index,modules,data,key,True,True)
            need(row['arm']==arm and row['sid']==records[index]['sid'] and row['n']==records[index]['n'] and row['target_rows']==len(case['target_ids'])
                 and row['loss']==raw['evidence']['loss'] and row['layer_forward_calls']==[1]*28
                 and row['max_reserved_bytes']>=row['max_allocated_bytes']>0 and math.isfinite(row['seconds']) and row['seconds']>0,
                 'Teacher timing/input ownership differs')
            gradient=row['gradients'];expected=set(initial['adapter'])|(set() if arm=='ordinary' else {'memory.'+k for k in initial['memory']})
            need(set(gradient['present'])==set(gradient['finite'])==expected and all(gradient['present'].values())
                 and all(gradient['finite'].values()) and gradient['all_present'] and gradient['all_finite']
                 and gradient['norms']['A']==0,'Zero-B per-row finite gradient records differ')
            if arm=='ordinary':
                need(raw['evidence']['embedding_identity']==raw['evidence']['actual_language_embeddings']
                     ==raw['evidence']['native_inputs'][0]['inputs_embeds_identity'],'Actual ordinary teacher embeddings differ')
            else:
                need(raw['evidence']['decoder_training'] and raw['evidence']['gradient_enabled']
                     and raw['evidence']['native_inputs'][0]['inputs_embeds_identity']==raw['evidence']['metadata']['full_embedding_identity'],
                     'Live cold memory teacher input/mode differs')
                memories.append(dict(arm=arm,index=index,kind='teacher',**independent.memory_audit(torch,capture,values,
                    case['coordinates'],initial['memory'],arm,data,key)))
                if arm=='normalized':need(gradient['norms']['mass_direction']==0,'Normalized gradient cannot include mass')
            teachers.append(dict(metric,index=index));publish()
        for row in entry['natural']:
            index=row['index'];case,values=load(index);raw=raw_packet(row);capture=owned(raw,index,arm,case);key=f'{arm}_natural_{index}'
            need(row['arm']==arm and row['kind']==('ordinary' if arm=='ordinary' else 'cold')
                 and row['sid']==records[index]['sid'] and row['n']==records[index]['n'] and row['tokens']==len(raw['result']['generated_ids'])
                 and row['max_reserved_bytes']>=row['max_allocated_bytes']>0 and row['seconds']>0 and math.isfinite(row['seconds'])
                 and row['fifty_token_seconds']==row['seconds']*50/row['tokens'] and row['efficacy_scored'] is False,
                 'Natural timing/input population differs')
            metric=natural_audit(torch,raw,case,entry,initial,config,modules,data,key)
            if arm!='ordinary':memories.append(dict(arm=arm,index=index,kind='natural',**independent.memory_audit(
                torch,capture,values,case['coordinates'],initial['memory'],arm,data,key)))
            naturals.append(metric);publish()
    D=len(selected['indices']);G=sum(x['cpu_head_calls'] for x in naturals);L=sum(x['target_rows'] for x in teachers)
    expected=dict(model=3*D+G,backbone=3*D+G,visual=0,language=3*D+G,norm=3*D+G,head=3*D+G,backward=3*D,optimizer=0)
    need(len(teachers)==3*D and len(naturals)==15 and len(memories)==2*D+10
         and analysis['counters']==expected and analysis['decoder_layer_calls']==[3*D+G]*28
         and analysis['teacher_head_rows']==L==3*sum(records[i]['target_rows'] for i in selected['indices'])
         and analysis['natural_head_rows']==G and 3*D+G<=825 and L+G<=1350
         and analysis['parameter_bytes_and_versions_unchanged'] and analysis['optimizer_steps']==analysis['vision_calls']==0,
         'Complete bounded replay/call inventory differs')
    projected=projection(old,analysis['setup_seconds'],analysis['arms'])
    need(projected==analysis['projection'],'Retained-old-plus-boundary timing reconstruction differs')
    resources=allocation(out,directory.name.removeprefix('profile_'),finished=True)
    passed=all(x['passed'] for x in teachers+naturals+memories)
    result=dict(protocol=PROTOCOL,passed=passed,completed=True,cpu_numerical_passed=passed,
        boundary_profile=descriptor(args.report,bindings),**refs,selection=selected,
        projection=projected,teacher_audits=teachers,natural_audits=naturals,memory_audits=memories,
        counters=expected,cpu_head_calls=3*D+G,cpu_head_rows=L+G,resources=resources,tests=tests,
        source_sha256=sources(),inherited_source_sha256=inherited,native_identity=cache['native_identity'],
        native_identity_sha256=cache['native_identity_sha256'],all_scheduled_numerical_evidence_collected=True,
        original_profile_passed=False,original_cache_parity_passed=False,no_efficacy=True,no_full_fit_release=True)
    save(out/'analysis.json',result)
    return result


def verify_report(summary_path):
    """JSON/hash-only passed CPU-boundary handoff, with no automatic fit release."""
    path=Path(summary_path).resolve();summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['phase']=='report' and summary['passed'] is summary['completed'] is True
         and summary['source_sha256']==sources() and summary['policy']==POLICY and not (path.parent/'failure.json').exists(),
         'Passed exact boundary CPU report required')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'],'Boundary analysis changed')
    result=read(summary['analysis_file'])
    need(result['passed'] and result['cpu_numerical_passed'] and result['all_scheduled_numerical_evidence_collected']
         and result['no_full_fit_release'] and not result['original_profile_passed'] and not result['original_cache_parity_passed'],
         'Complete boundary audit must preserve original cache failure')
    for field in ('artifacts','input_bindings'):
        need(sha(summary[field+'_file'])==summary[field+'_sha256'],'Boundary manifest changed')
        for file,digest in read(summary[field+'_file']).items():need(sha(file)==digest,'Bound boundary evidence changed')
    for name,digest in {**summary['source_sha256'],**summary['inherited_source_sha256']}.items():
        need(sha(REPO/name)==digest and sha(path.parent/'source'/name.replace('/','_'))==digest,'Boundary source archive changed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--profile',action='store_true');mode.add_argument('--report',type=Path)
    parser.add_argument('--features',type=Path);args=parser.parse_args()
    p.native.require_slurm(gpu=args.profile)
    need(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four CPU cores required')
    if not args.profile:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu','Independent audit is CPU-only')
    else:need(args.features is not None,'Passed shared feature stage required')
    started=time.perf_counter();phase='profile' if args.profile else 'report'
    out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    data=DATA/out.name;ckpt=CKPT/out.name;bindings={};progress={}
    save(out/'request.json',dict(protocol=PROTOCOL,phase=phase,features=None if args.features is None else str(args.features.resolve()),
         report=None if args.report is None else str(args.report.resolve()),policy=POLICY,source_sha256=sources()))
    try:
        result=profile(args,out,data,ckpt,started,bindings,progress) if args.profile else report(args,out,data,started,bindings,progress)
        elapsed=time.perf_counter()-started;need(elapsed<=(900 if args.profile else 1800),'Fixed boundary allocation cap exceeded')
        if not args.profile:save(out/'input_bindings.json',bindings)
        (out/'REPORT.md').write_text('Boundary '+phase+' completed. No optimizer update, feature extraction or efficacy score. '
            'Original cached-prefix failure remains failed. CPU numerical status: '+str(None if args.profile else result['passed'])+
            '. No full fit or inference is released.\n')
        artifacts={str(f):sha(f) for root in (out,data,ckpt) if root.exists() for f in root.iterdir() if f.is_file()}
        save(out/'artifacts.json',artifacts)
        summary=dict(protocol=PROTOCOL,phase=phase,passed=result['passed'],completed=True,policy=POLICY,
            source_sha256=result['source_sha256'],inherited_source_sha256=result['inherited_source_sha256'],
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
            artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),elapsed_seconds=elapsed,no_full_fit_release=True)
        save(out/'summary.json',summary)
        need(result['passed'],'Boundary CPU numerical gate failed after complete evidence publication')
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,phase=phase,type=type(exc).__name__,message=str(exc),
            elapsed_seconds=time.perf_counter()-started,source_sha256=sources(),policy=POLICY,progress=progress,
            partial_evidence_retained=True,no_full_fit_release=True));raise


if __name__=='__main__':main()

