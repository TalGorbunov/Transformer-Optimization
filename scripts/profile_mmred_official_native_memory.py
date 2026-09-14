"""Bounded ordinary/native-memory software and cost profile on original MMReD.

One B200 allocation only. Two diagnostic updates per arm are not a released
benchmark fit. All tensor/image/model work is Slurm-only.
"""
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
from scripts import prepare_mmred_official_native_v3 as preparation
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts import native_visual_memory_runtime as runtime
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
PROTOCOL='mmred_official_native_memory_profile'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_MMRED_OFFICIAL_NATIVE_MEMORY_PROFILE.md'
PROPOSAL_SHA='22619da97f1874f1ac32732933e058b8ae9234721b17ca1aeead1af0ba3540fa'
OWN=('scripts/profile_mmred_official_native_memory.py',PROPOSAL,'slurm/mmred_official_native_memory_profile.sbatch')
CORE_REPORT=REPO/'outputs/native_aggregation_vlm/native_visual_memory/check_443957/summary.json'
CORE_REPORT_SHA='e06e95f992dfc5218a7e695938a79767501d40ce6d31549cab3efbab7b83089f'
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native_memory'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_memory')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/mmred_official_native_memory')
ARMS=('ordinary','normalized','mass')
POLICY=dict(protocol=PROTOCOL,gpu_seconds=900,gpu_count=1,maximum_attempts=1,cpu_cores=4,memory_gib=16,
    cases=8,arms=list(ARMS),updates_per_arm=2,accumulation=8,backwards=48,optimizer_steps=6,
    feature_extractions=8,parity_teacher_calls=4,roundtrip_teacher_calls=12,natural_trajectories=29,
    memory_headless_prefills=2,maximum_new_tokens=50,maximum_model_head_calls=1514,
    maximum_decoder_norm_calls=1516,vision_calls=10,extra_probe_heads=0,
    lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_norm=1.,
    future_epochs=3,future_scene_presentations=12000,future_updates=1500,
    numerical_tv_limit=.02,no_full_fit_release=True,no_validation_or_test_predictions=True)


def read(path):return json.loads(Path(path).read_text())


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Profile proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path);need(expected is None or digest==expected,'Bound profile input changed')
    bindings[str(path)]=digest;return digest


def dependencies(path):
    path=Path(path).resolve();summary=read(path);bindings={};bind(path,bindings)
    need(summary['passed'] is summary['completed'] is True and summary['protocol']==preparation.PROTOCOL
         and summary['source_sha256']==preparation.sources() and not (path.parent/'failure.json').exists(),
         'Passed original-MMReD CPU preparation required')
    bind(summary['plan_file'],bindings,summary['plan_sha256']);plan=read(summary['plan_file'])
    need(plan['source_sha256']==preparation.sources() and plan['policy']==preparation.POLICY
         and plan['no_model_forward'] is plan['no_training_release'] is True and len(plan['cases'])==8,
         'Exact eight-case input-only preparation required')
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():bind(file,bindings,digest)
    inherited={**plan['source_sha256'],**plan['inherited_source_sha256']}
    for name,digest in inherited.items():
        need(sha(REPO/name)==digest,'Preparation ancestor source changed')
        bind(path.parent/'source'/name.replace('/','_'),bindings,digest)
    bind(CORE_REPORT,bindings,CORE_REPORT_SHA);core=read(CORE_REPORT)
    need(core['passed'] is core['completed'] is True and core['protocol']=='native_visual_memory_core_check'
         and core['tests_run']==8 and core['parameters']==469504
         and not (CORE_REPORT.parent/'failure.json').exists(),'Passed fixed native-width core fixtures required')
    for name,digest in {**core['source_sha256'],**core['inherited_source_sha256']}.items():
        need(sha(REPO/name)==digest,'Core source changed')
        bind(CORE_REPORT.parent/'source'/name.replace('/','_'),bindings,digest)
        need(name not in inherited or inherited[name]==digest,'Conflicting core/preparation source closure');inherited[name]=digest
    for field in ('test_results','report'):bind(core[field+'_file'],bindings,core[field+'_sha256'])
    need(read(core['test_results_file'])==dict(tests_run=8,failures=[],errors=[],skipped=[]),'Complete passing core fixture report required')
    expected=[(n,'char_at_frame') for n in (1,2,4,8,16)]+[(16,q) for q in preparation.QTYPES[1:]]
    need([(r['n'],r['qtype']) for r in plan['cases']]==expected and [r['index'] for r in plan['cases']]==list(range(8)),
         'Input-only profile case order changed')
    for item in plan['cases']:bind(item['file'],bindings,item['sha256'])
    rows=read(plan['rows_file']);selected=preparation.select_cases(rows)
    need([r['sid'] for r in selected]==[r['sid'] for r in plan['cases']],'Original training-case selection changed')
    diagnostics=[next_rows[:5] for n in (1,2,4,8,16) for q in preparation.QTYPES
        for next_rows in [[r for r in rows if r['pilot_role']=='train' and r['n']==n and r['qtype']==q]]]
    need(all(len(group)==5 for group in diagnostics),'Five fixed future diagnostics per type/length required')
    return plan,inherited,bindings,dict(file=str(path),sha256=sha(path),plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256']),[r['sid'] for group in diagnostics for r in group]


def load_case(torch,item):
    need(sha(item['file'])==item['sha256'],'Consumed case bytes changed')
    packet=torch.load(item['file'],map_location='cpu',weights_only=True)
    need(packet['metadata']==item['metadata'] and packet['target_ids']==item['target_ids']
         and packet['metadata']['sid']==item['sid'] and packet['metadata']['n']==item['n'], 'Prepared packet ownership changed')
    return packet


def model_mode(model,params,training):
    for name,value in model.named_parameters():value.requires_grad_(name in params if training else False)
    model.train(training);model.model.visual.eval()


def cpu_state(torch,params):return {name:value.detach().cpu().clone() for name,value in params.items()}


def tensor_state(params):return {name:p.tensor_info(value) for name,value in params.items()}


def restore(torch,params,state):
    need(set(params)==set(state),'Checkpoint parameter set differs')
    with torch.no_grad():
        for name,value in params.items():value.copy_(state[name].to(value.device))
    need(tensor_state(params)==tensor_state(state),'Restored parameter bytes differ')


def extract_features(torch,model,pixel_values,image_grid_thw):
    """This boundary has no question, answer, text packet or target arguments."""
    need(not model.model.visual.training and not any(v.requires_grad for v in model.model.visual.parameters()),'Frozen eval vision required')
    with torch.no_grad():
        items=model.model.get_image_features(pixel_values.to(model.device),image_grid_thw.to(model.device))
        need(len(items)==len(image_grid_thw) and all(x.shape==(196,3584) and x.dtype==torch.float16 for x in items),
             'Native ordered14x14 postmerger values required')
        return torch.cat(items,dim=0).detach()


def ordinary_packet(torch,model,case,features,teacher):
    ids=case['teacher_input_ids'] if teacher else case['inputs']['input_ids']
    ids=ids.to(model.device);values=model.get_input_embeddings()(ids)
    mask=ids==model.config.image_token_id
    need(features.dtype==values.dtype==torch.float16 and features.shape==(int(mask.sum()),3584),'Exact cached native image-placeholder rows required')
    embeddings=values.masked_scatter(mask[:,:,None].expand_as(values),features.reshape(-1))
    positions=case['teacher_position_ids' if teacher else 'position_ids'].to(model.device)
    return dict(inputs_embeds=embeddings,attention_mask=torch.ones_like(ids),position_ids=positions,
        cache_position=torch.arange(ids.shape[1],device=model.device),input_ids=ids,
        prompt_width=case['metadata']['prompt_width'],rope_deltas=case['rope_deltas'])


def ordinary_teacher(torch,model,case,features,*,pixel_route=False,evidence):
    L=len(case['target_ids']);packet=ordinary_packet(torch,model,case,features,True)
    evidence.update(expected_positions=[packet['position_ids'].detach().cpu().clone()],target_ids=case['target_ids'],
        target_positions=case['target_positions'],pixel_route=pixel_route,backward_completed=False,
        embedding_identity=p.tensor_info(packet['inputs_embeds']))
    common={k:packet[k] for k in ('attention_mask','position_ids','cache_position')}
    if pixel_route:
        common.update(input_ids=packet['input_ids'],pixel_values=case['inputs']['pixel_values'].to(model.device),
                      image_grid_thw=case['inputs']['image_grid_thw'].to(model.device))
    else:common['inputs_embeds']=packet['inputs_embeds']
    def actual_embeddings(module,args,kwargs):
        evidence['actual_language_embeddings']=p.tensor_info(kwargs['inputs_embeds'])
    handle=model.model.language_model.register_forward_pre_hook(actual_embeddings,with_kwargs=True)
    try:
        with runtime._observe(model,evidence,L,rope_deltas=packet['rope_deltas']) as counts:
            output=model(**common,use_cache=False,logits_to_keep=L,return_dict=True)
    finally:handle.remove()
    logits=output.logits
    ce=torch.nn.functional.cross_entropy(logits[0].float(),torch.tensor(case['target_ids'],device=model.device),reduction='none')
    evidence.update(position_ce=ce.detach().cpu().tolist(),loss=float(ce.mean().detach()),prompt_width=packet['prompt_width'])
    need(output.past_key_values is None and counts==dict(model=1,visual=int(pixel_route),language=1,norm=1,head=1,prefix_decoder=0)
         and bool(torch.isfinite(ce).all()),'One finite native cold ordinary teacher required')
    return dict(loss=ce.mean(),logits=logits,capture=evidence,counters=counts)


def ordinary_generate(torch,model,processor,case,features,evidence):
    need(not model.training and all(not value.requires_grad and value.grad is None for value in model.parameters()),'Frozen ordinary generation required')
    need(model.generation_config.eos_token_id==list(runtime.EOS_IDS) and processor.tokenizer.eos_token_id==151645,
         'Original unrestricted native EOS configuration required')
    model_before=runtime._model_state(model)
    packet=ordinary_packet(torch,model,case,features,False);W=packet['prompt_width'];delta=packet['rope_deltas'].to(model.device)
    ids=[];raw=[];expected=[];cache=None
    evidence.update(generated_ids=ids,raw_vectors=raw,expected_positions=expected,rope_deltas=case['rope_deltas'])
    with runtime._observe(model,evidence,1,rope_deltas=delta) as counts:
        with torch.inference_mode():
            for step in range(50):
                start=0 if step==0 else W+step-1;length=W if step==0 else 1
                pos=packet['position_ids'] if step==0 else torch.cat((torch.tensor([[[start]]],device=model.device),
                    (delta.reshape(1,1,1)+start).expand(3,1,1)),dim=0)
                expected.append(pos.detach().cpu().clone())
                inp=dict(inputs_embeds=packet['inputs_embeds']) if step==0 else dict(input_ids=torch.tensor([[ids[-1]]],device=model.device))
                output=model(**inp,attention_mask=torch.ones((1,start+length),dtype=torch.long,device=model.device),
                    position_ids=pos,cache_position=torch.arange(start,start+length,device=model.device),past_key_values=cache,
                    use_cache=True,logits_to_keep=1,return_dict=True)
                need(cache is None or output.past_key_values is cache,'Ordinary cache ownership changed')
                cache=output.past_key_values;need(cache.get_seq_length()==start+length,'Ordinary native cache length changed')
                vector=output.logits[0,-1].detach().float().cpu();raw.append(vector);ids.append(int(vector.argmax()))
                if ids[-1] in runtime.EOS_IDS:break
    T=len(ids)
    need(counts==dict(model=T,visual=0,language=T,norm=T,head=T,prefix_decoder=0)
         and len(evidence['profile_head'])==T and all(bool(torch.isfinite(v).all()) for v in raw),'Ordinary natural call/head counts differ')
    for i,(observed,head) in enumerate(zip(evidence['native_inputs'],evidence['profile_head'])):
        begin=0 if i==0 else W+i-1;length=W if i==0 else 1
        need(observed['past_length']==begin and not observed['has_pixels']
             and observed['attention_mask'].shape==(1,begin+length) and bool((observed['attention_mask']==1).all())
             and observed['cache_position'].tolist()==list(range(begin,begin+length))
             and torch.equal(head['head_logits'][0,-1].float(),raw[i]),'Ordinary input/head history differs')
        if i:need(observed['input_ids'].tolist()==[[ids[i-1]]],'Ordinary generation did not consume its own output')
    need(runtime._model_state(model)==model_before,'Ordinary generation changed model/adapter state')
    return dict(generated_ids=ids,raw_logits=torch.stack(raw),completed=ids[-1] in runtime.EOS_IDS,
        truncated=ids[-1] not in runtime.EOS_IDS,text=processor.tokenizer.decode(ids,skip_special_tokens=True),
        raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False),counters=counts,
        native_inputs=evidence['native_inputs'],native_positions=evidence['native_positions'],profile_head=evidence['profile_head'],
        shapes=evidence['shapes'],rope_restored=True,hooks_removed=True,native_positions_preserved=True)


def grad_record(torch,params,memory):
    all_params={**params,**({} if memory is None else {'memory.'+k:v for k,v in memory.named_parameters()})}
    present={name:value.grad is not None for name,value in all_params.items()}
    finite={name:False if value.grad is None else bool(torch.isfinite(value.grad).all()) for name,value in all_params.items()}
    def norm(names):
        terms=[all_params[name].grad.float().square().sum() for name in names if all_params[name].grad is not None]
        return 0. if not terms else float(torch.stack(terms).sum().sqrt())
    norms={letter:norm([name for name in params if '.lora_'+letter+'.' in name]) for letter in ('A','B')}
    if memory is not None:
        norms.update({key:norm(['memory.'+key]) for key in ('key_weight','queries','mass_direction')})
    return dict(all_present=all(present.values()),all_finite=all(finite.values()),present=present,finite=finite,norms=norms)


def retain(torch,file,case_index,arm,phase,evidence,**extra):
    torch.save(p.cpu_tree(torch,dict(case_index=case_index,arm=arm,phase=phase,evidence=evidence,**extra)),file)
    return dict(file=str(file),sha256=sha(file))


def compare(torch,a,b,*,exact=False,generation=False):
    def logits(value):
        return value['raw_logits'] if generation else value['evidence']['profile_head'][0]['head_logits'][0].float()
    x,y=logits(a),logits(b);same_shape=x.shape==y.shape
    equal=same_shape and torch.equal(x,y)
    tv=float((x.double().softmax(-1)-y.double().softmax(-1)).abs().sum(-1).max()/2) if same_shape else None
    ids=(a['generated_ids']==b['generated_ids']) if generation else same_shape and torch.equal(x.argmax(-1),y.argmax(-1))
    return dict(shape_equal=same_shape,raw_logits_equal=equal,ids_equal=ids,maximum_tv=tv,
        exact_required=exact,passed=bool(same_shape and ids and tv<=.02 and (equal if exact else True)))


def single_attempt(out):
    import re
    import subprocess
    job='mmred_official_native_memory_profile'
    need(os.environ.get('SLURM_JOB_NAME')==job,'Exact registered memory profile job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout
    (out/'launch_sacct.psv').write_text(raw);rows=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=9 or fields[1]!=job:continue
        identifier,name,partition,state,exit_code,seconds,tres,start,end=fields
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=exit_code,
                         seconds=int(seconds),gpus=int(generic[0]) if generic else sum(map(int,typed)),tres=tres))
    save(out/'launch_accounting.json',dict(command=command,rows=rows,maximum_attempts=1,gpu_seconds_cap=900))
    need(len(rows)==1 and rows[0]['job_id']==os.environ['SLURM_JOB_ID'] and rows[0]['partition']=='gpu'
         and rows[0]['gpus']==1 and rows[0]['seconds']<=900,'One bounded900-second GPU attempt required')


def forecasts(setup,feature_records,arms):
    lengths=(1,2,4,8,16)
    F={n:max(r['seconds'] for r in feature_records if r['n']==n) for n in lengths}
    shared=sum(800*F[n] for n in lengths)
    result=dict(training_only=True,shared_feature_preparation_seconds=shared,feature_seconds_by_n=F,
        feature_cache_built_once_for_all_arms=True,training_pairs=4000,epochs=3,scene_presentations=12000,
        optimizer_steps=1500,validation_test_cost_included=False,no_N32_generation_estimate=True,
        empirical_estimate_not_guarantee=True,main_release=False,arms={})
    for arm in ARMS:
        a=arms[arm];T={n:max(r['seconds'] for r in a['microbatches'] if r['n']==n) for n in lengths}
        O=max(a['optimizer_seconds']);C=a['checkpoint_seconds']
        estimate=setup+1.25*(sum(2400*T[n] for n in lengths)+1500*O+2*C)+60
        natural=[r for r in a['natural'] if r['kind']=='ordinary' or r['kind']=='cold']
        G={n:max(r['fifty_token_seconds'] for r in natural if r['n']==n) for n in sorted({r['n'] for r in natural})}
        result['arms'][arm]=dict(training_without_shared_features_seconds=estimate,
            training_with_shared_features_if_run_alone_seconds=estimate+shared,microbatch_seconds_by_n=T,
            optimizer_seconds=O,checkpoint_seconds=C,generation_fifty_token_seconds_by_n=G,
            formula='setup+1.25*(sum_N2400*T_arm_N+1500*O_arm+2*C_arm)+60',
            future_training_generation_cases=100,diagnostic_generation_cost_is_separate=True)
    return result


def profile(args,out,started,frozen,progress):
    import torch
    from gnnformer.runtime import load_runtime
    from gnnformer.native_visual_memory import NativeVisualMemory
    torch.set_num_threads(4);single_attempt(out)
    plan,inherited,bindings,prepared_ref,diagnostic_sids=dependencies(args.prepared)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','Exactly one B200 required')
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir()
    for name,digest in {**frozen,**inherited}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source archive copy differs')
    save(out/'input_bindings.json',bindings);save(out/'future_training_diagnostic_sids.json',diagnostic_sids)
    loaded=load_runtime(str(preparation.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=p.frozen_joint.live_identity(torch,loaded,plan)
    refs=list(model.named_parameters());base=p.base_metadata(refs);save(out/'base_before.json',base)
    need(p.installed({})==plan['packages'],'Pinned PEFT/bnb package implementation changed')
    contract=p.install_lora(torch,model,out/'actual_peft_config_before_contract.json');params=p.adapter_parameters(model)
    config=p.canonical_peft_config(model);p.check_base(refs,base,model)
    initial=cpu_state(torch,params);initial_memory=NativeVisualMemory().state_dict()
    initial_rng=dict(cpu=torch.random.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all())
    initial_file=ckpt/'initial.pt'
    torch.save(dict(adapter=initial,memory=initial_memory,peft_config=config,contract=contract,rng=initial_rng,seed=24,updates=0),initial_file)
    initial_ref=dict(file=str(initial_file),sha256=sha(initial_file),adapter_tensors=tensor_state(initial),memory_tensors=tensor_state(initial_memory))
    configuration=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited,
        preparation=prepared_ref,core_report=dict(file=str(CORE_REPORT),sha256=sha(CORE_REPORT)),
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],hardware=hardware,
        initial=initial_ref,peft_config=config,data_directory=str(data),checkpoint_directory=str(ckpt),
        cases=plan['cases'],future_training_diagnostic_sids=diagnostic_sids,no_full_fit_release=True)
    save(out/'config.json',configuration)
    counts=Counter(dict.fromkeys(('model','backbone','visual','language','norm','head','backward','optimizer'),0));layers=[0]*28
    feature_records=[];feature_index={};texts={};arms={};comparisons=[];parity_refs={};artifacts={}
    progress.update(counters=counts,decoder_layer_calls=layers)
    save(out/'live_counters_initial.json',dict(counts))
    def record_file(file):
        artifacts[str(file)]=sha(file);return dict(file=str(file),sha256=artifacts[str(file)])
    def check():
        p.check_base(refs,base,model)
        need(time.perf_counter()-started<900,'Fixed900-second profile cap exceeded')
    def load_inputs(index):
        item=plan['cases'][index];case=load_case(torch,item);entry=feature_index[index]
        need(sha(entry['file'])==entry['sha256'],'Cached native features changed')
        saved=torch.load(entry['file'],map_location='cpu',weights_only=True)
        need(saved['case_index']==index and saved['input_identity']==entry['input_identity']
             and p.tensor_info(saved['features'])==entry['tensor'],'Cached feature ownership differs')
        return case,saved['features'].to(model.device)
    def memory_tokens(memory,case,features,arm):
        result=memory(features,**{k:v.to(model.device) for k,v in case['coordinates'].items()},retain_mass=arm=='mass')
        return result['tokens'].unsqueeze(0),result
    def teacher(arm,memory,index,file,backward):
        evidence={};case,features=load_inputs(index);before=features._version
        evidence['parameter_state']=arms[arm]['active_state']
        evidence['feature_packet']=feature_index[index]
        try:
            if arm=='ordinary':
                result=ordinary_teacher(torch,model,case,features,evidence=evidence)
            else:
                tokens,capture=memory_tokens(memory,case,features,arm);evidence['memory_capture']=capture
                evidence['coordinate_identity']={k:p.tensor_info(v) for k,v in case['coordinates'].items()}
                result=runtime.teacher_forward(model,case['text'],tokens,evidence=evidence)
            # Raw evidence is durable before backward or any gradient guard.
            retain(torch,file,index,arm,'teacher',evidence,backward_completed=False)
            if backward:
                evidence['backward_started']=True
                (result['loss']/8).backward();counts['backward']+=1;evidence['backward_completed']=True
            need(features._version==before,'Memory modified cached feature values')
            retain(torch,file,index,arm,'teacher',evidence,backward_completed=backward)
            return case,evidence,record_file(file)
        except BaseException:
            retain(torch,file,index,arm,'teacher',evidence,partial=True);raise
    def natural(arm,kind,index,memory,*,snapshot=None,text_case_index=None):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        case_meta=plan['cases'][index]['metadata']
        if snapshot is None:case,features=load_inputs(index)
        question_text=texts[index if text_case_index is None else text_case_index]
        evidence=dict(parameter_state=arms[arm]['active_state'],feature_packet=feature_index[index])
        file=data/f'{arm}_{kind}_{index}_{text_case_index}.pt'
        try:
            if arm=='ordinary':
                with torch.inference_mode():result=ordinary_generate(torch,model,loaded.processor,case,features,evidence)
            elif snapshot is None:
                with torch.inference_mode():tokens,capture=memory_tokens(memory,case,features,arm)
                evidence['memory_capture']=capture
                evidence['coordinate_identity']={k:p.tensor_info(v) for k,v in case['coordinates'].items()}
                result=runtime.generate_cold(model,loaded.processor,question_text,tokens,
                    provenance=arms[arm]['provenance'],max_tokens=50,evidence=evidence)
            else:
                result=runtime.generate_split(model,loaded.processor,snapshot,question_text['suffix'],max_tokens=50,evidence=evidence)
            retain(torch,file,index,arm,kind,evidence,result=result,text_case_index=text_case_index)
        except BaseException:
            retain(torch,file,index,arm,kind,evidence,partial=True,text_case_index=text_case_index);raise
        descriptor=record_file(file);record=dict(arm=arm,kind=kind,case_index=index,text_case_index=text_case_index,
            sid=case_meta['sid'],n=case_meta['n'],tokens=len(result['generated_ids']),**descriptor,
            transferred_answers_scored=False,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
        check();save(out/f'{file.stem}.json',record);torch.cuda.synchronize();record['seconds']=time.perf_counter()-tick
        # Deliberately scales the complete operation, including fixed setup,
        # as a conservative allowance; no decomposition or accuracy selection.
        record['fifty_token_seconds']=record['seconds']*50/record['tokens']
        save(out/f'{file.stem}_timing.json',record);arms[arm]['natural'].append(record)
        return descriptor
    with ExitStack() as hooks:
        def bump(key):
            def callback(*_):counts[key]+=1
            return callback
        for module,key in ((model,'model'),(model.model,'backbone'),(model.model.visual,'visual'),
            (model.model.language_model,'language'),(model.model.language_model.norm,'norm'),(model.lm_head,'head')):
            hooks.callback(module.register_forward_pre_hook(bump(key)).remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def observed(module,args,index=index):layers[index]+=1
            hooks.callback(layer.register_forward_pre_hook(observed).remove)
        model_mode(model,params,False);torch.cuda.synchronize();setup=time.perf_counter()-started
        for index,item in enumerate(plan['cases']):
            torch.cuda.synchronize();tick=time.perf_counter();case=load_case(torch,item);texts[index]=case['text']
            features=extract_features(torch,model,case['inputs']['pixel_values'],case['inputs']['image_grid_thw'])
            identity={key:p.tensor_info(case['inputs'][key]) for key in ('pixel_values','image_grid_thw')}
            file=data/f'features_{index}.pt';torch.save(dict(case_index=index,input_identity=identity,features=features.cpu()),file)
            entry=dict(index=index,n=item['n'],input_identity=identity,tensor=p.tensor_info(features),**record_file(file))
            feature_index[index]=entry;check();save(out/f'feature_{index}.json',entry)
            torch.cuda.synchronize();entry['seconds']=time.perf_counter()-tick;feature_records.append(entry)
            save(out/f'feature_{index}_timing.json',entry);del features,case
        for index in (0,4):
            for route in ('pixels','features'):
                case,features=load_inputs(index);evidence=dict(parameter_state=initial_ref,feature_packet=feature_index[index]);file=data/f'parity_{index}_{route}.pt'
                try:
                    with torch.no_grad():ordinary_teacher(torch,model,case,features,pixel_route=route=='pixels',evidence=evidence)
                    retain(torch,file,index,'ordinary','initial_parity',evidence)
                except BaseException:
                    retain(torch,file,index,'ordinary','initial_parity',evidence,partial=True);raise
                parity_refs[(index,route)]=record_file(file)
        for arm in ARMS:
            restore(torch,params,initial)
            for value in params.values():value.grad=None
            memory=None if arm=='ordinary' else NativeVisualMemory().to(model.device)
            if memory is not None:need(tensor_state(dict(memory.named_parameters()))==tensor_state(initial_memory),'Common private memory initialization differs')
            torch.random.set_rng_state(initial_rng['cpu']);torch.cuda.set_rng_state_all(initial_rng['cuda'])
            model_mode(model,params,True);need(p.adapter_contract(torch,model)==contract and p.canonical_peft_config(model)==config,'Fresh arm adapter configuration changed')
            live=list(params.values())+([] if memory is None else list(memory.parameters()))
            optimizer=torch.optim.AdamW(live,lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            optimizer.zero_grad(set_to_none=True)
            arm_record=dict(microbatches=[],optimizer_seconds=[],natural=[],roundtrip={},fresh_initial_adapter_exact=True,
                fresh_initial_memory_exact=memory is not None,post_install_rng_restored=True,
                parameter_states={'0':initial_ref},active_state=initial_ref,software_state_save_seconds=0.)
            arms[arm]=arm_record
            for step in (1,2):
                for index in range(8):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                    before_layers=list(layers);file=data/f'{arm}_training_{step}_{index}.pt'
                    case,evidence,descriptor=teacher(arm,memory,index,file,True)
                    gradients=grad_record(torch,params,memory)
                    record=dict(arm=arm,step=step,index=index,n=case['metadata']['n'],sid=case['metadata']['sid'],
                        loss=evidence['loss'],target_rows=len(case['target_ids']),gradients=gradients,**descriptor,
                        layer_forward_calls=[a-b for a,b in zip(layers,before_layers)],
                        max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
                    save(out/f'{arm}_training_{step}_{index}.json',record);arm_record['microbatches'].append(record)
                    need(gradients['all_present'] and gradients['all_finite'] and record['layer_forward_calls']==[1]*28,
                         'Missing/nonfinite gradient or decoder recomputation; raw forward retained')
                    check();torch.cuda.synchronize();record['seconds']=time.perf_counter()-tick
                    save(out/f'{arm}_training_{step}_{index}_timing.json',record)
                norms=gradients['norms'];need(norms['B']>0 and (norms['A']==0 if step==1 else norms['A']>0),'Expected zero-B/live-A LoRA paths differ')
                if memory is not None:
                    need(norms['key_weight']>0 and norms['queries']>0 and (norms['mass_direction']==0 if arm=='normalized' else norms['mass_direction']>0),
                         'Expected live memory keys/queries/mass paths differ')
                all_params={**params,**({} if memory is None else {'memory.'+k:v for k,v in memory.named_parameters()})}
                grad_file=ckpt/f'{arm}_gradient_{step}.pt';torch.save({k:v.grad.detach().cpu().clone() for k,v in all_params.items()},grad_file);record_file(grad_file)
                torch.cuda.synchronize();tick=time.perf_counter();before={id(v):v._version for v in live}
                unclipped=torch.nn.utils.clip_grad_norm_(live,1.);need(bool(torch.isfinite(unclipped)),'Nonfinite unclipped gradient norm')
                optimizer.step();optimizer.zero_grad(set_to_none=True);counts['optimizer']+=1
                need(all(v._version>before[id(v)] and bool(torch.isfinite(v).all()) for v in live),'Optimizer parameter update/finiteness differs')
                check();save(out/f'{arm}_optimizer_{step}.json',dict(step=step,gradient_norm=float(unclipped),lr=2e-4,
                    parameter_steps=[float(optimizer.state[v]['step']) for v in live]))
                torch.cuda.synchronize();seconds=time.perf_counter()-tick;arm_record['optimizer_seconds'].append(seconds)
                save(out/f'{arm}_optimizer_{step}_timing.json',dict(seconds=seconds))
                audit_tick=time.perf_counter();state_file=ckpt/f'{arm}_state_{step}.pt'
                torch.save(dict(arm=arm,updates=step,adapter=cpu_state(torch,params),
                    memory=None if memory is None else cpu_state(torch,dict(memory.named_parameters())),peft_config=config),state_file)
                state_ref=record_file(state_file);arm_record['parameter_states'][str(step)]=state_ref
                arm_record['active_state']=state_ref;arm_record['software_state_save_seconds']+=time.perf_counter()-audit_tick
            model.eval();model.model.visual.eval()
            if memory is not None:memory.eval()
            roundtrip={}
            for phase in ('before','after'):
                if phase=='after':
                    torch.cuda.synchronize();tick=time.perf_counter();final_adapter=cpu_state(torch,params)
                    final_memory=None if memory is None else cpu_state(torch,dict(memory.named_parameters()))
                    checkpoint=ckpt/f'{arm}_after_two_updates.pt'
                    torch.save(dict(arm=arm,adapter=final_adapter,memory=final_memory,peft_config=config,contract=contract,
                        optimizer=optimizer.state_dict(),rng=dict(cpu=torch.random.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
                        updates=2,profile_only=True,must_not_initialize_full_fit=True),checkpoint)
                    descriptor=record_file(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    need(saved['peft_config']==config==p.canonical_peft_config(model) and saved['contract']==contract and saved['updates']==2,'Checkpoint config/step differs')
                    restore(torch,params,initial)
                    if memory is not None:restore(torch,dict(memory.named_parameters()),initial_memory)
                    restore(torch,params,saved['adapter'])
                    if memory is not None:restore(torch,dict(memory.named_parameters()),saved['memory'])
                    need(p.adapter_contract(torch,model)==contract,'Checkpoint restore replaced adapter objects/configuration')
                    check();arm_record['checkpoint']=descriptor
                    arm_record['provenance']=dict(native_identity_sha256=plan['native_identity_sha256'],lora=descriptor,
                        memory=None if memory is None else dict(checkpoint=descriptor,tensors=tensor_state(dict(memory.named_parameters()))))
                    save(out/f'{arm}_checkpoint.json',dict(**descriptor,original_reset_exact=True,reload_exact=True,
                        adapter_tensors=tensor_state(params),memory_tensors=None if memory is None else tensor_state(dict(memory.named_parameters())),
                        optimizer_state_saved=True,rng_saved=True,full_peft_config_exact=True))
                    torch.cuda.synchronize();arm_record['checkpoint_seconds']=time.perf_counter()-tick
                    save(out/f'{arm}_checkpoint_timing.json',dict(seconds=arm_record['checkpoint_seconds']))
                for index in (0,4):
                    file=data/f'{arm}_roundtrip_{phase}_{index}.pt'
                    with torch.no_grad():_,_,descriptor=teacher(arm,memory,index,file,False)
                    roundtrip[(phase,index)]=descriptor
            arm_record['roundtrip']={phase:[roundtrip[(phase,index)] for index in (0,4)] for phase in ('before','after')}
            model_mode(model,params,False)
            if memory is not None:memory.requires_grad_(False)
            if arm=='ordinary':
                for index in range(5):natural(arm,'ordinary',index,None)
            else:
                case,features=load_inputs(4);prefixes=[texts[i]['prefix'] for i in range(4,8)]
                need(len({plan['cases'][i]['metadata']['question'] for i in range(4,8)})==4
                     and all(r['identity']==prefixes[0]['identity'] for r in prefixes),'Four distinct questions need one identical question-free opening')
                with torch.inference_mode():tokens,capture=memory_tokens(memory,case,features,arm)
                evidence=dict(memory_capture=capture,parameter_state=arm_record['active_state'],feature_packet=feature_index[4],
                    coordinate_identity={k:p.tensor_info(v) for k,v in case['coordinates'].items()})
                file=data/f'{arm}_prefix.pt'
                try:
                    snapshot,evidence=runtime.prefill_memory(model,prefixes[0],tokens,provenance=arm_record['provenance'],evidence=evidence)
                    retain(torch,file,4,arm,'prefix',evidence,snapshot=runtime.export_snapshot(torch,snapshot))
                except BaseException:
                    retain(torch,file,4,arm,'prefix',evidence,partial=True);raise
                arm_record['prefix']=record_file(file)
                for index in range(4,8):natural(arm,'cold',4,memory,text_case_index=index)
                for index in range(4,8):natural(arm,'split',4,memory,snapshot=snapshot,text_case_index=index)
                for index in reversed(range(4,8)):natural(arm,'reverse',4,memory,snapshot=snapshot,text_case_index=index)
                del snapshot,tokens,features
            save(out/f'{arm}_measurements.json',arm_record);save(out/f'live_counters_{arm}.json',dict(counts));check()
            del optimizer,memory
    # Collect every scheduled comparison only after all raw trajectories exist.
    for index in (0,4):
        a=torch.load(parity_refs[(index,'pixels')]['file'],map_location='cpu',weights_only=True)
        b=torch.load(parity_refs[(index,'features')]['file'],map_location='cpu',weights_only=True)
        metric=compare(torch,a,b,exact=True)
        metric['actual_language_embeddings_equal']=a['evidence']['actual_language_embeddings']==b['evidence']['actual_language_embeddings']
        metric['passed']=metric['passed'] and metric['actual_language_embeddings_equal']
        comparisons.append(dict(kind='native_pixel_feature_teacher',case_index=index,**metric))
    for arm in ARMS:
        for index,slot in zip((0,4),range(2)):
            a=torch.load(arms[arm]['roundtrip']['before'][slot]['file'],map_location='cpu',weights_only=True)
            b=torch.load(arms[arm]['roundtrip']['after'][slot]['file'],map_location='cpu',weights_only=True)
            comparisons.append(dict(kind='trained_checkpoint_teacher',arm=arm,case_index=index,**compare(torch,a,b,exact=True)))
        if arm!='ordinary':
            lookup={(r['kind'],r['text_case_index']):r for r in arms[arm]['natural']}
            for index in range(4,8):
                packets={kind:torch.load(lookup[(kind,index)]['file'],map_location='cpu',weights_only=True)['result'] for kind in ('cold','split','reverse')}
                comparisons.append(dict(kind='memory_cold_split',arm=arm,text_case_index=index,
                    **compare(torch,packets['cold'],packets['split'],generation=True)))
                comparisons.append(dict(kind='memory_reverse_order',arm=arm,text_case_index=index,
                    **compare(torch,packets['split'],packets['reverse'],exact=True,generation=True)))
    save(out/'comparisons.json',comparisons)
    need(len(comparisons)==24 and sum(len(a['natural']) for a in arms.values())==29,'Complete scheduled comparison/trajectory inventory required')
    generated=sum(r['tokens'] for a in arms.values() for r in a['natural'])
    target_lengths=[len(r['target_ids']) for r in plan['cases']]
    expected=dict(model=64+generated,backbone=66+generated,visual=10,language=66+generated,norm=66+generated,
                  head=64+generated,backward=48,optimizer=6)
    need(dict(counts)==expected and generated<=1450 and layers==[expected['language']]*28,
         'Complete fixed native call inventory differs')
    projection=forecasts(setup,feature_records,arms);save(out/'projection.json',projection)
    numerical_pass=all(r['passed'] for r in comparisons)
    analysis=dict(passed=numerical_pass,all_scheduled_comparisons_collected=True,comparison_count=len(comparisons),comparisons=comparisons,
        counters=dict(counts),decoder_layer_calls=layers,head_rows=8*(target_lengths[0]+target_lengths[4])+6*sum(target_lengths)+generated,
        natural_trajectories=29,feature_records=feature_records,arms=arms,setup_seconds=setup,projection=projection,
        profile_only=True,no_full_fit_release=True,no_validation_or_test_predictions=True)
    save(out/'analysis.json',analysis);check();save(out/'base_after.json',p.base_metadata(refs))
    need(numerical_pass,'Native pixel/feature, trained checkpoint or cold/cache parity failed; all scheduled raw evidence retained')
    need(sources()==frozen and all(sha(REPO/name)==value for name,value in inherited.items()),'Profile source closure changed')
    artifacts[str(initial_file)]=sha(initial_file)
    artifacts.update({str(file):sha(file) for file in out.glob('*.json')})
    save(out/'artifacts.json',artifacts)
    (out/'REPORT.md').write_text('# Official native memory software/cost profile\n\nThe fixed three-arm software profile passed its native pixel/feature, checkpoint and cold/cache parity checks. All29 natural trajectories and48 teacher backward calls are retained. No benchmark competence or aggregation gain is established. Training-only forecasts include shared vision preparation separately; no full fit, validation/test inference or N32 generation cost is released.\n')
    return dict(**configuration,passed=True,completed=True,counters=dict(counts),analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),report_file=str(out/'REPORT.md'),report_sha256=sha(out/'REPORT.md'),
        projection=projection,all_scheduled_comparisons_collected=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True);args=parser.parse_args()
    p.native.require_slurm(gpu=True)
    need(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'Four CPU cores required')
    started=time.perf_counter();out=OUT/f'profile_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=sources();save(out/'request.json',dict(prepared=str(args.prepared.resolve()),policy=POLICY,source_sha256=frozen))
    progress={}
    try:
        result=profile(args,out,started,frozen,progress)
        save(out/'summary.json',dict(result,elapsed_seconds=time.perf_counter()-started))
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            policy=POLICY,elapsed_seconds=time.perf_counter()-started,progress=progress,partial_evidence_retained=True,no_full_fit_release=True));raise


if __name__=='__main__':main()
