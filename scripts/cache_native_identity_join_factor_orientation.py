"""Held bridge from the fixed orientation inventory to a small native state bank.

Reused rows retain their exact original hashes. Only missing actual causal input
keys may be harvested; there is no branch, label gate, fitting or outcome filter.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import stage_native_identity_join_factor_orientation_v2 as training
from scripts.stage_native_vision_v10_features import pack,row_inputs,native_api,runtime_identity
from scripts.stage_native_vision_v6_teacher import MODEL,model_metadata
need=training.need;read=training.read;save=training.save;sha=training.sha;oid=training.oid
OUT=training.OUT/'features';DATA=training.DATA/'features'
PROTOCOL='identity_join_factor_orientation_native_feature_bridge'
OWN=('scripts/cache_native_identity_join_factor_orientation.py','slurm/native_identity_join_factor_orientation_feature_prepare.sbatch',
     'slurm/native_identity_join_factor_orientation_feature_harvest.sbatch','slurm/native_identity_join_factor_orientation_feature_merge.sbatch')
POLICY=dict(protocol=PROTOCOL,batch_size=64,phases=['local_empty','local_prefix'],maximum_missing_rows=1440,
    cpu_prepare_seconds=300,gpu_harvest_seconds=150,cpu_merge_seconds=300,maximum_gpu_attempts=1,
    native_dtype='torch.float16',hidden_size=3584,vocabulary_size=152064,logits_to_keep=1,use_cache=False,
    cpu_replay_all_missing_rows=True,head_tv_max=.02,head_argmax_exact=True,no_extra_gpu_heads=True,
    no_gate_or_local_labels=True,no_fitting=True,zero_missing_requires_zero_gpu=True)
GPU_JOB='identity_join_factor_orientation_feature_harvest'
_READ={}


def tensor_info(torch,value):
    v=value.detach().cpu().contiguous();return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or expected==h,'Changed bridge input: '+str(path));bindings[str(path)]=h;_READ[str(path)]=h;return h


def source_maps():
    new,old=training.source_maps();inherited={**old,**new};return {name:sha(REPO/name) for name in OWN},inherited


def sources():return source_maps()[0]


def inherited_sources():return source_maps()[1]


def snapshot(out):
    frozen,inherited=source_maps();(out/'source').mkdir()
    for name,h in frozen.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==h,'Bridge source copy changed')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited));return frozen,inherited


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(torch,v) for v in value)
    return value


def batch_inventory(inventory):
    missing=inventory['missing'];need(missing==sorted(set(missing)) and len(missing)<=1440,'Fixed sorted missing-key set required');batches=[]
    for phase in POLICY['phases']:
        ids=[fid for fid in missing if inventory['features'][fid]['phase']==phase]
        for offset in range(0,len(ids),64):batches.append(dict(index=len(batches),phase=phase,feature_ids=ids[offset:offset+64]))
    need(sorted(fid for batch in batches for fid in batch['feature_ids'])==missing,'Every and only missing local key must be harvested');return batches


def projection(profile,observations,batches):
    phases={phase:max(row['complete_batch_seconds'] for row in observations if row['route']=='feature' and row['phase']==phase) for phase in POLICY['phases']}
    load=profile['model_load_seconds'];pixels=profile['pixel_load_verify_seconds'];counts=Counter(row['phase'] for row in batches)
    need(all(math.isfinite(v) and v>0 for v in [load,pixels]+list(phases.values())),'Original complete-batch timing invalid')
    seconds=0. if not batches else load+pixels+1.5*sum(counts[p]*phases[p] for p in counts)+30.
    return dict(passed=seconds<=150,projected_seconds=seconds,gpu_cap_seconds=150,original_model_load_seconds=load,
        original_pixel_load_seconds=pixels,complete_batch_seconds=phases,phase_batches=dict(counts),margin=1.5,reserve_seconds=30.,
        no_gpu_needed=not batches,measured_original_native_feature_profile=True)


def self_test(torch):
    import struct
    a=torch.tensor(1.,dtype=torch.float32);need(tensor_info(torch,a)==dict(shape=[],dtype='torch.float32',sha256=hashlib.sha256(struct.pack('<f',1.)).hexdigest()),'Scalar identity differs')
    inv=dict(missing=['a','b'],features={'a':dict(phase='local_empty'),'b':dict(phase='local_prefix')})
    need(batch_inventory(inv)==[dict(index=0,phase='local_empty',feature_ids=['a']),dict(index=1,phase='local_prefix',feature_ids=['b'])],
         'Separate strict-prefix batch inventory differs')
    empty=dict(missing=[],features={});need(batch_inventory(empty)==[],'Empty missing set must not create native calls')
    before=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16);selected=before[torch.tensor([1,0])].clone()
    need(tensor_info(torch,selected[0])==tensor_info(torch,before[1]) and tensor_info(torch,selected[1])==tensor_info(torch,before[0]),'Exact reused-row gather identity differs')
    logits=torch.tensor([[[10.,0.]]],dtype=torch.float16)
    need(replay_metric(torch,logits,logits.clone())['passed'] and not replay_metric(torch,logits,logits.flip(-1))['passed'],
         'Native replay must preserve full-vocabulary argmax/TV')
    return dict(passed=True,groups=5,scalar_hash=True,missing_only_batches=True,zero_missing_zero_gpu=True,exact_gather=True,native_replay_gate=True)


def prepare(args,out,frozen,inherited):
    import torch
    torch.set_num_threads(4);value=training.verify_stage(args.training_stage);parent=value['plan'];bindings={}
    bind(value['summary_file'],bindings,value['summary_sha256']);bind(value['summary']['plan_file'],bindings,value['summary']['plan_sha256'])
    bind(parent['feature_inventory_file'],bindings,parent['files'][parent['feature_inventory_file']]);inventory=read(parent['feature_inventory_file'])
    bind(parent['cache_file'],bindings,parent['cache_sha256']);cache=read(parent['cache_file'])
    bind(parent['cache_plan_file'],bindings,parent['cache_plan_sha256']);original=read(parent['cache_plan_file'])
    batches=batch_inventory(inventory);profile_file=Path(cache['profile_directory'])/'summary.json';bind(profile_file,bindings,cache['profile_summary_sha256']);profile=read(profile_file)
    need(profile['passed'] is profile['completed'] is True and profile['native_identity']==parent['native_identity'],'Passed original native feature measurement required')
    bind(profile['observations_file'],bindings,profile['observations_sha256']);forecast=projection(profile,read(profile['observations_file']),batches)
    save(out/'missing_inventory.json',dict(missing=inventory['missing'],counts=inventory['counts'],batches=batches,projection=forecast))
    need(forecast['passed'],'Fixed150-second harvest projection failed; no larger cap or retry')
    input_plan=dict(features={fid:inventory['features'][fid] for fid in inventory['missing']},pairs=inventory['image_questions'],
        layouts={inventory['features'][fid]['layout_id']:original['layouts'][inventory['features'][fid]['layout_id']] for fid in inventory['missing']})
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);pixel_file=None;pixel_sha=None;pixel_info={};layouts={};positions=[]
    if batches:
        import transformers
        from transformers import AutoProcessor
        from scripts.probe_native_vision_v2_prefix import fingerprint
        processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
        owner,rope,api=native_api(processor)
        need(api==parent['native_identity']['native_api'] and fingerprint(processor,str(transformers.__version__))==parent['native_identity']['processor']
             and runtime_identity()==parent['native_identity']['runtime'] and model_metadata()==parent['native_identity']['model'],'Current original native input API differs')
        bind(original['pixels_file'],bindings,original['pixels_sha256']);blob=torch.load(original['pixels_file'],map_location='cpu',weights_only=True)
        wanted={inventory['image_questions'][inventory['features'][fid]['pair_id']]['image_sha256'] for fid in inventory['missing']}
        need(wanted<=set(original['pixel_info']) and wanted<=set(blob['pixels']),'Missing input image has no exact original native pixels')
        pixels={key:blob['pixels'][key].clone() for key in sorted(wanted)};del blob
        for key,pixel in pixels.items():need(tensor_info(torch,pixel)==original['pixel_info'][key],'Copied actual native pixels changed')
        pixel_info={k:tensor_info(torch,v) for k,v in pixels.items()};pixel_file=data/'pixels.pt';torch.save(dict(schema_version=1,pixels=pixels),pixel_file);pixel_sha=bind(pixel_file,bindings)
        for batch in batches:
            ids=batch['feature_ids'];inputs=pack(torch,[row_inputs(torch,input_plan,pixels,fid) for fid in ids],original['pad_token_id'])
            pos,delta=rope(owner,input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'])
            for i,fid in enumerate(ids):
                layout=input_plan['layouts'][input_plan['features'][fid]['layout_id']];length=len(layout['input_ids'])
                need(torch.equal(inputs['input_ids'][i,-length:],torch.tensor(layout['input_ids'])) and torch.equal(pos[:,i,-length:],torch.tensor(layout['position_ids'])[:,0]),
                     'Missing actual strict-prefix native layout differs')
            batch.update(batch_size=len(ids),input_identity={k:tensor_info(torch,v) for k,v in inputs.items()},position_ids=tensor_info(torch,pos),rope_deltas=delta.tolist())
            layouts[batch['index']]=dict(position_ids=pos.clone(),rope_deltas=delta.clone(),input_ids=inputs['input_ids'].clone(),attention_mask=inputs['attention_mask'].clone())
    layout_file=data/'batch_layouts.pt';torch.save(layouts,layout_file);bind(layout_file,bindings);save(out/'input_plan.json',input_plan);bind(out/'input_plan.json',bindings)
    for key in ('rows_file','pairs_file','order_file','feature_inventory_file'):bind(parent[key],bindings,parent['files'][parent[key]])
    bind(cache['native_model_file'],bindings,cache['native_model_sha256'])
    factor=read(parent['original_factor_plan']['file']);software_reference=profile['software']
    bind(software_reference['file'],bindings,software_reference['sha256']);software_report=read(software_reference['file'])
    need(software_report['passed'] is software_report['completed'] is True and software_report['native_identity']==parent['native_identity']
         and software_report['native_identity_sha256']==parent['native_identity_sha256'],'Original native software proof differs')
    software=dict(file=software_report['plan_file'],sha256=software_report['plan_sha256']);bind(software['file'],bindings,software['sha256'])
    tests=self_test(torch);save(out/'selftests.json',tests)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        training_stage=dict(file=value['summary_file'],sha256=value['summary_sha256'],plan_file=value['summary']['plan_file'],plan_sha256=value['summary']['plan_sha256'],
            independent_audit_file=value['summary']['independent_audit_file'],independent_audit_sha256=value['summary']['independent_audit_sha256'],
            native_identity_sha256=parent['native_identity_sha256'],source_sha256=value['source_sha256'],inherited_source_sha256=value['inherited_source_sha256']),
        training_plan=parent,inventory_file=parent['feature_inventory_file'],inventory_sha256=parent['files'][parent['feature_inventory_file']],
        native_identity=parent['native_identity'],native_identity_sha256=parent['native_identity_sha256'],native_model_file=cache['native_model_file'],native_model_sha256=cache['native_model_sha256'],
        native_module_identity=factor['native_module_identity'],precision=read(software['file'])['precision'],native_software_plan=software,
        native_software_report=dict(file=software_reference['file'],sha256=software_reference['sha256']),profile_reference=dict(file=str(profile_file),sha256=sha(profile_file),
            observations_file=profile['observations_file'],observations_sha256=profile['observations_sha256']),projection=forecast,batches=batches,
        missing_feature_ids=inventory['missing'],missing_count=len(inventory['missing']),gpu_required=bool(batches),
        input_plan_file=str(out/'input_plan.json'),pixels_file=None if pixel_file is None else str(pixel_file),pixels_sha256=pixel_sha,pixel_info=pixel_info,
        layouts_file=str(layout_file),layouts_sha256=sha(layout_file),pad_token_id=original['pad_token_id'],input_bindings=bindings,tests=tests,
        no_model_or_head_forward=True,no_feature_harvest_yet=True,no_fit_release=True)
    save(out/'plan.json',plan)
    return dict(passed=True,completed=True,phase='prepare',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),missing_count=plan['missing_count'],native_batches=len(batches),
        gpu_required=bool(batches),projection=forecast,tests=tests,no_model_or_head_forward=True,no_fit_release=True)


def verify_plan(path,ancestors=True):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json');frozen,inherited=source_maps()
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='prepare' and summary['plan_file']==str(path) and summary['plan_sha256']==sha(path)
         and plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==frozen and plan['inherited_source_sha256']==inherited,'Passed bridge preparation required')
    for name,h in frozen.items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Prepared source snapshot changed')
    for file,h in plan['input_bindings'].items():need(sha(file)==h,'Bound prepared input changed: '+file)
    if ancestors:need(training.verify_stage(plan['training_stage']['file'])['plan']==plan['training_plan'],'Training stage changed')
    return plan


def gpu_accounting(out,current=None,require_complete=False):
    import re
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;path=out/'gpu_accounting.psv';path.write_text(raw);rows=[]
    for line in raw.splitlines():
        f=line.split('|');need(len(f)==9,'Accounting field count differs');job,name,partition,state,code,elapsed,tres,start,end=f
        if name!=GPU_JOB:continue
        values=dict(v.split('=',1) for v in tres.split(',') if '=' in v);typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed)
        need(not typed or sum(typed)==gpus,'GPU accounting types conflict');rows.append(dict(job_id=job,name=name,partition=partition,state=state,exit_code=code,
            seconds=int(elapsed),gpus=gpus,gpu_seconds=int(elapsed)*gpus,start=start,end=end))
    need(len(rows)==(1 if current is not None else 0),'All failed/zero allocations count; at most one registered missing-only harvest')
    if current is not None:
        row=rows[0];need(row['job_id']==str(current) and row['partition']=='gpu' and row['gpus']==1 and 0<=row['seconds']<=150,'Registered harvest identity/cap differs')
        if require_complete:need(row['state']=='COMPLETED' and row['exit_code']=='0:0','Harvest did not complete successfully')
    return dict(passed=True,jobs=rows,gpu_seconds=sum(r['gpu_seconds'] for r in rows),file=str(path),sha256=sha(path),single_attempt=True,failed_zero_attempts_included=True)


def harvest(args,out,frozen,inherited,started):
    import torch,transformers
    import inspect
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from scripts import native_vision_v7_runtime as native
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    torch.set_num_threads(4);plan=verify_plan(args.plan,ancestors=False)
    need(plan['gpu_required'] is True and plan['missing_count']>0 and os.environ.get('SLURM_JOB_NAME')==GPU_JOB,'Only a required missing-only GPU harvest may run')
    accounting=gpu_accounting(out,current=os.environ['SLURM_JOB_ID']);need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    pixels=torch.load(plan['pixels_file'],map_location='cpu',weights_only=True)['pixels']
    need({k:tensor_info(torch,v) for k,v in pixels.items()}==plan['pixel_info'],'Prepared compact native pixels differ')
    layouts=torch.load(plan['layouts_file'],map_location='cpu',weights_only=True);inputs_plan=read(plan['input_plan_file'])
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model,None);head=model.lm_head
    identity=plan['native_identity'];need(fingerprint(loaded.processor,str(transformers.__version__))==identity['processor'] and native_api(loaded.processor)[2]==identity['native_api']
        and runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Actual native feature runtime identity differs')
    weights=dict(norm=tensor_info(torch,norm.weight),head=tensor_info(torch,head.weight))
    need(weights==dict(norm=identity['norm_weight'],head=identity['head_weight']) and norm.weight.dtype==head.weight.dtype==torch.float16
         and float(norm.variance_epsilon)==identity['rms_norm_eps'] and sha(inspect.getfile(type(norm)))==identity['norm_source_sha256'],'Actual native norm/head differs')
    precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision())
    need(precision==plan['precision'],'Native floating-point backend differs');quant=model.config.quantization_config.to_dict()
    need(quant['load_in_4bit'] is quant['bnb_4bit_use_double_quant'] is True and quant['bnb_4bit_quant_type']=='nf4'
         and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','Actual NF4/double/BF16 feature backend differs')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);config=dict(protocol=PROTOCOL,phase='harvest',policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),slurm_job_id=os.environ['SLURM_JOB_ID'],native_identity_sha256=plan['native_identity_sha256'],
        native_weights=weights,precision=precision,quantization=quant,gpu=torch.cuda.get_device_name(0),data_directory=str(data),missing_count=plan['missing_count'])
    save(out/'config.json',config);counts=dict(model=0,vision=0,language=0,norm=0,head=0,extra_heads=0,core=0);captured={};records=[]
    versions={name:p._version for name,p in model.named_parameters()};rope_before=getattr(model.model,'rope_deltas',None);rope=get_rope_index_fn(model)
    def count(key):
        def hook(*_):counts[key]+=1
        return hook
    def before_language(module,args,kw):
        counts['language']+=1;captured['actual_positions']=kw['position_ids'].detach().cpu().clone();captured['actual_mask']=kw['attention_mask'].detach().cpu().clone()
    def before_norm(module,args):
        counts['norm']+=1;captured['norm_input_shape']=list(args[0].shape);captured['states']=args[0][:,-1:,:].detach().cpu().clone()
    def after_norm(module,args,result):captured['normalized']=result[:,-1:,:].detach().cpu().clone()
    def after_head(module,args,result):
        counts['head']+=1;captured['head_input']=args[0].detach().cpu().clone();captured['native_logits']=result.detach().cpu().clone()
    try:
        with ExitStack() as stack:
            stack.callback(setattr,model.model,'rope_deltas',rope_before)
            for handle in (model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('vision')),
                model.model.language_model.register_forward_pre_hook(before_language,with_kwargs=True),norm.register_forward_pre_hook(before_norm),
                norm.register_forward_hook(after_norm),head.register_forward_hook(after_head)):stack.callback(handle.remove)
            for batch in plan['batches']:
                tick=time.perf_counter();ids=batch['feature_ids'];inputs=pack(torch,[row_inputs(torch,inputs_plan,pixels,fid) for fid in ids],plan['pad_token_id'])
                need({k:tensor_info(torch,v) for k,v in inputs.items()}==batch['input_identity'],'Actual feature pixels/tokens changed')
                pos,delta=rope(input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'])
                expected=layouts[batch['index']];need(torch.equal(pos,expected['position_ids']) and torch.equal(delta,expected['rope_deltas']),'Actual loaded native mRoPE differs')
                captured.clear();before=dict(counts)
                with torch.inference_mode():output=model(**move_to_device(inputs,model.device),use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize();raw=dict(schema_version=1,batch=batch,input_identity=batch['input_identity'],capture=cpu_tree(torch,captured),
                    input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],rope_deltas=model.model.rope_deltas.detach().cpu().clone())
                path=data/f'batch_{batch["index"]:03d}.pt';torch.save(raw,path);record=dict(batch=batch,raw_file=str(path),raw_sha256=sha(path));records.append(record)
                save(out/f'progress_{batch["index"]:03d}.json',record)
                n=len(ids);need({k:counts[k]-before[k] for k in counts}==dict(model=1,vision=1,language=1,norm=1,head=1,extra_heads=0,core=0)
                    and captured['states'].shape==(n,1,3584) and captured['native_logits'].shape==(n,1,152064)
                    and captured['states'].dtype==captured['native_logits'].dtype==torch.float16 and output.past_key_values is None
                    and torch.equal(output.logits.detach().cpu(),captured['native_logits']) and torch.equal(captured['actual_positions'],pos)
                    and torch.equal(captured['actual_mask'],inputs['attention_mask']) and torch.equal(raw['rope_deltas'],delta),
                    'Actual native missing-feature shapes/positions/calls differ; raw evidence retained')
                record['complete_batch_seconds']=time.perf_counter()-tick
    except BaseException:
        torch.save(cpu_tree(torch,captured),data/'failure_capture.pt');raise
    finally:save(out/'observations.json',records)
    need(versions=={name:p._version for name,p in model.named_parameters()} and all(not p.requires_grad and p.grad is None for p in model.parameters())
         and weights==dict(norm=tensor_info(torch,norm.weight),head=tensor_info(torch,head.weight)) and getattr(model.model,'rope_deltas',None) is rope_before,'Frozen model/rope ownership changed')
    need(time.perf_counter()-started<=150,'Fixed harvest cap exceeded')
    return dict(**config,passed=True,completed=True,counters=counts,observations_file=str(out/'observations.json'),observations_sha256=sha(out/'observations.json'),
        native_weights_unchanged=True,native_parameter_versions_unchanged=True,hooks_removed=True,rope_restored=True,accounting=accounting,
        no_extra_gpu_heads=True,no_gate_or_local_labels=True,no_fit_release=True)


def replay_metric(torch,actual,replayed):
    need(actual.shape==replayed.shape and actual.ndim==3 and actual.shape[1]==1 and actual.dtype==replayed.dtype==torch.float16,
         'Actual native-shaped FP16 head replay required')
    a=actual[:,0];b=replayed[:,0];need(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'Nonfinite native head replay')
    tv=.5*(torch.softmax(a.double(),-1)-torch.softmax(b.double(),-1)).abs().sum(-1);same=a.argmax(-1)==b.argmax(-1)
    return dict(passed=bool((tv<=.02).all() and same.all()),tv=tv.tolist(),argmax_exact=same.tolist(),maximum_tv=float(tv.max()),
        actual_shape=list(actual.shape),selected_query_norm_replay=True,full_prompt_norm_not_replayed=True)


def audited_missing(torch,args,out,plan):
    ids=plan['missing_feature_ids'];states={};audits=[]
    if not ids:
        need(args.harvest_summary is None and plan['batches']==[] and plan['gpu_required'] is False,'Zero missing features prohibit a GPU harvest')
        return states,dict(passed=True,cpu_head_calls=0,cpu_head_rows=0,gpu_calls=0,raw=[],audits=[],accounting=gpu_accounting(out),zero_missing_zero_gpu=True)
    need(args.harvest_summary is not None,'Nonempty missing set requires the one completed exact native harvest')
    path=Path(args.harvest_summary).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path);config=read(path.parent/'config.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='harvest' and all(summary[k]==v for k,v in config.items())
         and config['source_sha256']==plan['source_sha256'] and config['inherited_source_sha256']==plan['inherited_source_sha256']
         and config['plan_file']==str(args.plan.resolve()) and config['plan_sha256']==sha(args.plan) and config['missing_count']==len(ids)
         and config['native_identity_sha256']==plan['native_identity_sha256'],'Completed exact missing-only harvest required')
    need(path.parent==OUT/f'harvest_{config["slurm_job_id"]}' and config['data_directory']==str(DATA/path.parent.name)
         and config['policy']==POLICY and config['gpu']=='NVIDIA B200' and config['precision']==plan['precision']
         and math.isfinite(summary['elapsed_seconds']) and 0<summary['elapsed_seconds']<=150,'Harvest location/hardware/fixed cap differs')
    quant=config['quantization'];need(quant['load_in_4bit'] is quant['bnb_4bit_use_double_quant'] is True and quant['bnb_4bit_quant_type']=='nf4'
        and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','Harvest native quantization metadata differs')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Harvest source snapshot changed')
    need(sha(summary['observations_file'])==summary['observations_sha256'] and summary['native_weights_unchanged'] is summary['native_parameter_versions_unchanged']
         is summary['hooks_removed'] is summary['rope_restored'] is summary['no_extra_gpu_heads'] is summary['no_gate_or_local_labels'] is True,'Native harvest publication/ownership differs')
    records=read(summary['observations_file']);need(len(records)==len(plan['batches']),'Complete missing native batch inventory required')
    layouts=torch.load(plan['layouts_file'],map_location='cpu',weights_only=True)
    from scripts import diagnose_native_identity_join_readout as oracle
    need(oracle.native_module_identity(torch,plan['native_identity'])==plan['native_module_identity'],'Actual native norm/head implementation changed')
    packet=torch.load(plan['native_model_file'],map_location='cpu',weights_only=True);norm,head=oracle.native_modules(torch,packet,plan['native_identity'],'cpu');del packet
    before=dict(norm=tensor_info(torch,norm.weight),head=tensor_info(torch,head.weight));need(before==config['native_weights'],'Actual harvested native model weights differ')
    for batch,record in zip(plan['batches'],records):
        need(record['batch']==batch and sha(record['raw_file'])==record['raw_sha256'] and Path(record['raw_file'])==Path(config['data_directory'])/f'batch_{batch["index"]:03d}.pt',
             'Native missing input/raw record ownership differs')
        raw=torch.load(record['raw_file'],map_location='cpu',weights_only=True);cap=raw['capture'];expected=layouts[batch['index']];n=len(batch['feature_ids'])
        need(raw['schema_version']==1 and raw['batch']==batch and raw['input_identity']==batch['input_identity']
             and torch.equal(raw['input_ids'],expected['input_ids']) and torch.equal(raw['attention_mask'],expected['attention_mask'])
             and torch.equal(raw['rope_deltas'],expected['rope_deltas']) and torch.equal(cap['actual_positions'],expected['position_ids'])
             and torch.equal(cap['actual_mask'],expected['attention_mask']),'Actual native strict-prefix mask/position ownership differs')
        h=cap['states'];normal=cap['normalized'];native_logits=cap['native_logits'];head_input=cap['head_input']
        need(h.shape==normal.shape==head_input.shape==(n,1,3584) and native_logits.shape==(n,1,152064)
             and cap['norm_input_shape']==[n,expected['input_ids'].shape[1],3584]
             and all(v.dtype==torch.float16 and not v.requires_grad and bool(torch.isfinite(v).all()) for v in (h,normal,native_logits,head_input))
             and torch.equal(normal,head_input),'Actual native pre-final-norm/head capture differs')
        with torch.inference_mode():replayed=head(norm(h))
        metrics=replay_metric(torch,native_logits,replayed);file=out/f'head_replay_{batch["index"]:03d}.json'
        save(file,dict(batch=batch,metrics=metrics,raw_file=record['raw_file'],raw_sha256=record['raw_sha256']))
        audits.append(dict(file=str(file),sha256=sha(file)));need(metrics['passed'],'Native CPU head replay exceeded fixedTV/argmax gate; evidence retained')
        for fid,value in zip(batch['feature_ids'],h[:,0]):need(fid not in states,'Repeated harvested key');states[fid]=value.clone()
    count=len(records);need(summary['counters']==dict(model=count,vision=count,language=count,norm=count,head=count,extra_heads=0,core=0)
        and set(states)==set(ids),'Only complete missing-key native calls are allowed')
    need(before==dict(norm=tensor_info(torch,norm.weight),head=tensor_info(torch,head.weight)) and all(not p.requires_grad and p.grad is None for m in (norm,head) for p in m.parameters()),
         'CPU audit changed frozen norm/head weights')
    return states,dict(passed=True,cpu_head_calls=count,cpu_head_rows=len(ids),gpu_calls=count,raw=records,audits=audits,
        harvest_summary_file=str(path),harvest_summary_sha256=sha(path),accounting=gpu_accounting(out,current=summary['slurm_job_id'],require_complete=True),
        zero_missing_zero_gpu=False,native_replay_scope='Actual Bx1 query head rows; selected-query native norm replay, full-prompt norm shapes observed')


def merge(args,out,frozen,inherited):
    import torch
    torch.set_num_threads(4);plan=verify_plan(args.plan);inventory=read(plan['inventory_file']);missing,proof=audited_missing(torch,args,out,plan)
    gathered={};owners={};groups=defaultdict(list)
    for fid,pointer in inventory['reused'].items():groups[pointer['file']].append((fid,pointer))
    for file,values in sorted(groups.items()):
        digests={p['file_sha256'] for _,p in values};need(len(digests)==1 and sha(file)==next(iter(digests)),'Consumed original native shard changed')
        packet=torch.load(file,map_location='cpu',weights_only=True);bank=packet['states']
        need(bank.dtype==torch.float16 and bank.ndim==2 and bank.shape[1]==3584 and len(packet['feature_ids'])==len(bank),'Original native state bank shape differs')
        for fid,pointer in values:
            row=pointer['row'];need(0<=row<len(bank) and packet['feature_ids'][row]==fid,'Original pointer/actual feature ID differs')
            value=bank[row];info=tensor_info(torch,value);need(info['sha256']==pointer['state_sha256'] and bool(torch.isfinite(value).all()),'Original native row bytes differ')
            gathered[fid]=value.clone();owners[fid]=dict(kind='reused',source_pointer=pointer,state_sha256=info['sha256'])
        del bank,packet
    need(set(gathered).isdisjoint(missing),'Reused and harvested ownership overlap')
    for fid,value in missing.items():gathered[fid]=value;owners[fid]=dict(kind='harvested',state_sha256=tensor_info(torch,value)['sha256'])
    ids=sorted(inventory['features']);need(sorted(gathered)==ids,'Complete exact small feature bank required')
    states=torch.stack([gathered[fid] for fid in ids]);index={fid:i for i,fid in enumerate(ids)};data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    features=data/'features.pt';torch.save(dict(schema_version=1,states=states,feature_ids=ids,index=index),features);file_sha=sha(features);info=tensor_info(torch,states)
    del gathered,states,missing
    reloaded=torch.load(features,map_location='cpu',weights_only=True)
    need(reloaded['feature_ids']==ids and reloaded['index']==index and tensor_info(torch,reloaded['states'])==info,'Restricted state-bank roundtrip differs')
    for fid,value in zip(ids,reloaded['states']):need(tensor_info(torch,value)['sha256']==owners[fid]['state_sha256'],'Gathered exact native row changed')
    parent=plan['training_plan'];scenes=inventory['scenes'];rows=read(parent['rows_file']);pairs=read(parent['pairs_file']);order=read(parent['order_file'])
    used={f for scene in scenes.values() for part in scene['local_feature_ids'] for f in part}|{f for scene in scenes.values() for f in scene['global_feature_ids']}
    need(used==set(ids) and len(rows)==len(scenes)==216 and len(pairs)==108 and len(order)==48000,'Training scene/prefix/bank coverage differs')
    pointers={fid:dict(file=str(features),file_sha256=file_sha,row=index[fid],state_sha256=owners[fid]['state_sha256']) for fid in ids}
    cache=dict(schema_version=1,protocol=PROTOCOL,complete=True,training_only=True,source_sha256=frozen,inherited_source_sha256=inherited,
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],native_model_file=plan['native_model_file'],native_model_sha256=plan['native_model_sha256'],
        features=pointers,scenes=scenes,training_pairs=pairs,features_file=str(features),features_sha256=file_sha,feature_tensor=info,feature_ids=ids,index=index,
        state_dtype='torch.float16',read_boundary='Actual native pre-final-norm current-query hidden',plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),
        training_stage=plan['training_stage'],no_local_labels=True,no_fit_release=True)
    cache_path=data/'feature_cache.json';save(cache_path,cache);save(out/'ownership.json',owners);save(out/'native_audit.json',proof)
    return dict(passed=True,completed=True,phase='merge',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=inherited,
        plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),cache_file=str(cache_path),cache_sha256=sha(cache_path),
        features_file=str(features),features_sha256=file_sha,feature_tensor=info,feature_count=len(ids),reused_count=len(inventory['reused']),missing_count=plan['missing_count'],
        native_audit_file=str(out/'native_audit.json'),native_audit_sha256=sha(out/'native_audit.json'),ownership_file=str(out/'ownership.json'),ownership_sha256=sha(out/'ownership.json'),
        restricted_roundtrip_passed=True,all_row_hashes_exact=True,cpu_head_calls=proof['cpu_head_calls'],cpu_head_rows=proof['cpu_head_rows'],gpu_calls=proof['gpu_calls'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],no_fit_release=True)


def verify_cache(summary_path):
    path=Path(summary_path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path);frozen,inherited=source_maps()
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='merge' and summary['protocol']==PROTOCOL
         and summary['source_sha256']==frozen and summary['inherited_source_sha256']==inherited,'Completed current native bridge required')
    plan=verify_plan(summary['plan_file'],ancestors=False);need(summary['plan_sha256']==sha(summary['plan_file']),'Merged plan changed')
    for name,h in frozen.items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Merged source snapshot changed')
    for key in ('cache','features','native_audit','ownership'):need(sha(summary[key+'_file'])==summary[key+'_sha256'],'Merged '+key+' changed')
    cache=read(summary['cache_file']);audit=read(summary['native_audit_file']);need(cache['complete'] is cache['training_only'] is audit['passed'] is True
         and cache['source_sha256']==frozen and cache['inherited_source_sha256']==inherited and cache['training_stage']==plan['training_stage']
         and cache['native_identity']==summary['native_identity']==plan['native_identity'] and cache['native_identity_sha256']==plan['native_identity_sha256']
         and cache['features_file']==summary['features_file'] and cache['features_sha256']==summary['features_sha256']
         and cache['feature_tensor']==summary['feature_tensor'] and summary['all_row_hashes_exact'] is summary['restricted_roundtrip_passed'] is True,'Complete native bank identity/roundtrip differs')
    for record in audit['raw']:need(sha(record['raw_file'])==record['raw_sha256'],'Bound harvested raw record changed')
    for record in audit['audits']:need(sha(record['file'])==record['sha256'] and read(record['file'])['metrics']['passed'] is True,'Bound native replay audit changed')
    parent=plan['training_plan'];inventory=read(plan['inventory_file'])
    need(cache['scenes']==inventory['scenes'] and set(cache['features'])==set(inventory['features']) and cache['feature_ids']==sorted(inventory['features'])
         and cache['index']=={fid:i for i,fid in enumerate(cache['feature_ids'])},'Complete canonical feature/scene map differs')
    return dict(plan=plan,cache=cache,summary_file=str(path),summary_sha256=sha(path),features_file=summary['features_file'],features_sha256=summary['features_sha256'],
        feature_tensor=summary['feature_tensor'],feature_ids=cache['feature_ids'],index=cache['index'],scenes=cache['scenes'],pairs=read(parent['pairs_file']),
        rows=read(parent['rows_file']),order=read(parent['order_file']),source_sha256=frozen,inherited_source_sha256=inherited,
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--harvest',action='store_true');action.add_argument('--merge',action='store_true')
    parser.add_argument('--training-stage',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--harvest-summary',type=Path);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four-core Slurm allocation required')
    need((os.environ.get('SLURM_JOB_PARTITION')=='gpu') if args.harvest else (os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS')),'Registered CPU/GPU phase differs')
    need((args.training_stage is not None and args.plan is args.harvest_summary is None) if args.prepare else
         (args.training_stage is None and args.plan is not None and (args.harvest_summary is None if args.harvest else True)),'Exact phase inputs required')
    phase='prepare' if args.prepare else 'harvest' if args.harvest else 'merge';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen,inherited=snapshot(out)
    save(out/'request.json',dict(phase=phase,training_stage=None if args.training_stage is None else str(args.training_stage),
        plan=None if args.plan is None else str(args.plan),harvest_summary=None if args.harvest_summary is None else str(args.harvest_summary),source_sha256=frozen))
    try:
        result=prepare(args,out,frozen,inherited) if args.prepare else harvest(args,out,frozen,inherited,started) if args.harvest else merge(args,out,frozen,inherited)
        elapsed=time.perf_counter()-started;need(elapsed<=(150 if args.harvest else 300) and source_maps()==(frozen,inherited),'Bridge fixed resource/source contract changed')
        result['elapsed_seconds']=elapsed;save(out/'summary.json',result)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,elapsed_seconds=time.perf_counter()-started,
            accessed_input_bindings=_READ,partial_outputs_retained=True));raise


if __name__=='__main__':main()
