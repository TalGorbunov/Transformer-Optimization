"""Draft bounded independent audit for the original-MMReD cold training pilot.

Each arm has one bounded CPU report. The fixed scalar trace is checked in full;
independent loss reconstruction uses six saved teacher examples. All100 final
training diagnostic trajectories are audited without model or training replay.
Nothing executes on import; numerical work requires a CPU Slurm allocation.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import mmred_official_answer as answer

PROTOCOL='mmred_official_native_training_independent_audit'
POLICY=dict(cpu_seconds_per_arm=3600,cpu_cores=4,memory_gib=16,
    training_presentations=12000,optimizer_updates=1500,training_raw_indices=[0,11999],
    pre_last_update_snapshot_before_index=11992,pre_last_update_state=1499,
    roundtrip_teacher_examples=4,teacher_replay_examples=6,final_training_diagnostics=100,
    cpu_head_calls_cap_per_arm=5006,cpu_head_rows_cap_per_arm=5300,
    ce_absolute_tolerance=2e-6,cpu_head_tv_max=.02,core_atol=2e-4,core_rtol=2e-4,
    no_full_training_reexecution=True,no_validation_or_test=True,no_inference_release=True)


def score_native_output(tokenizer,generated_ids,atype,typed_gold):
    """Keep every generated non-EOS token visible to the full-response parser."""
    ids=list(generated_ids);content=ids[:-1] if ids and ids[-1] in answer.EOS_IDS else ids
    text=tokenizer.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return dict(scoring_text=text,**answer.score_answer(text,atype,typed_gold,ids))


def parser_proof():
    """Literal policy checks, independent of the producer's reported scores."""
    contract=answer.self_test()
    cases=[
        ('{"answer":"Bathroom"}','room','Bathroom',[9,151645],True,True,True,False),
        ('{"answer":0}','number',0,[9,151643],True,True,True,False),
        ('{"answer":false}','number',0,[9,151645],False,False,False,False),
        ('{"answer":"John","ans\\u0077er":"John"}','person','John',[9,151645],False,False,False,False),
        ('{"answer":"Mary"}','person','John',[9,151645],True,False,False,False),
        ('{"answer":"Kitchen"}','room','Kitchen',[9]*50,True,True,False,True),
        ('<|im_start|>{"answer":"Kitchen"}','room','Kitchen',[9,151645],False,False,False,False),
    ]
    for text,atype,gold,ids,valid,semantic,correct,truncated in cases:
        got=answer.score_answer(text,atype,gold,ids)
        if (got['format_valid'],got['semantic_match'],got['correct'],got['truncated'])!=(valid,semantic,correct,truncated):
            raise AssertionError(('Strict independent parser policy',text,got))
    class RecordingTokenizer:
        def __init__(self):self.calls=[]
        def decode(self,ids,**kwargs):
            self.calls.append((ids,kwargs));return '{"answer":"Kitchen"}'
    tokenizer=RecordingTokenizer()
    score_native_output(tokenizer,[9,151645],'room','Kitchen')
    score_native_output(tokenizer,[9]*50,'room','Kitchen')
    expected=dict(skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if tokenizer.calls!=[([9],expected),([9]*50,expected)]:raise AssertionError('Only final native EOS may be removed')
    return dict(passed=True,shared_contract=contract,independent_policy_cases=len(cases),decode_boundary_cases=2,
                model_calls=0,head_calls=0,tokenizer_model_loaded=False)


import argparse
from collections import Counter,defaultdict
import json
import math
import os
import random
import re
import subprocess
import time
from scripts import train_mmred_official_native_memory as producer
from scripts import report_mmred_official_native_memory as reference
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=reference.p
OUT=producer.OUT
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_training_audit')
PROPOSAL='docs/paper/MMRED_OFFICIAL_NATIVE_TRAINING_AUDIT_PROPOSAL.md'
PROPOSAL_SHA='ea2787741965ae966c91f690112eb46a3b3624b4b0b72470b27879514ec82567'
OWN=('scripts/report_mmred_official_native_training.py',PROPOSAL,'slurm/mmred_official_native_training_report.sbatch')
read=reference.read
bind=reference.bind
record=reference.record
packet=reference.packet


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held main audit proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def snapshot(out,inherited):
    own=sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        need(sha(REPO/name)==digest,'Main audit source changed')
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Main audit source copy differs')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',inherited);return own


def expected_order():
    rng=random.Random(24);result=[]
    for epoch in range(3):
        indices=list(range(4000));rng.shuffle(indices)
        result.extend(dict(micro_index=epoch*4000+j,epoch=epoch,index=index,updates_before=(epoch*4000+j)//8) for j,index in enumerate(indices))
    return result


def inputs(torch,path,bindings):
    path=Path(path).resolve();directory=path if path.is_dir() else path.parent
    need(directory.parent==OUT and re.fullmatch(r'run_\d+_[012]',directory.name),'Registered main array/arm directory required')
    config=record(directory/'config.json',bindings);analysis=record(directory/'analysis.json',bindings)
    need(config['protocol']==producer.PROTOCOL and config['policy']==producer.POLICY and config['source_sha256']==producer.sources()
         and config['inherited_source_sha256']==producer.inherited_sources() and config['arm'] in producer.profile.ARMS,
         'Exact frozen main producer policy/source required')
    arm=config['arm'];need(directory.name==f'run_{config["array_job_id"]}_{producer.profile.ARMS.index(arm)}','Run directory/arm differs')
    reference.archive(directory,config['source_sha256'],config['inherited_source_sha256'],bindings)
    bind(config['plan_file'],bindings,config['plan_sha256']);plan=producer.verify_plan(config['plan_file'])
    reference.archive(Path(config['plan_file']).parent,plan['source_sha256'],plan['inherited_source_sha256'],bindings)
    for mapping in (plan['input_bindings'],plan['artifacts']):
        for file,digest in mapping.items():bind(file,bindings,digest)
    need(analysis['protocol']==producer.PROTOCOL and analysis['arm']==arm and analysis['completed']
         and analysis['no_validation_test_inference'] and analysis['requires_independent_cpu_audit'] and not analysis['objective_achieved'],
         'Complete final training diagnostic analysis required')
    if (directory/'summary.json').exists():
        summary=record(directory/'summary.json',bindings)
        need(summary['passed'] and summary['completed'] and summary['phase']=='run'
             and summary['analysis_sha256']==sha(directory/'analysis.json') and summary['counters']==analysis['counters'], 'Main summary differs')
    else:
        failure=record(directory/'failure.json',bindings)
        need(not analysis['passed'] and failure['message']=='Final checkpoint teacher parity failed after complete evidence collection', 'Only completed final numerical failure is auditable')
    artifacts=record(directory/'artifacts.json',bindings)
    for file,digest in artifacts.items():bind(file,bindings,digest)
    need(record(directory/'base_before.json',bindings)==record(directory/'base_after.json',bindings),'Frozen native base changed')
    need(Path(config['data_directory'])==producer.DATA/directory.name and Path(config['checkpoint_directory'])==producer.CKPT/directory.name,
         'Main artifacts outside designated roots')
    items=record(plan['cases_file'],bindings);order=record(plan['order_file'],bindings)
    need(len(items)==4000 and order==expected_order() and all(item['compact']['index']==i for i,item in enumerate(items)), 'Exact main cohort/order differs')
    cached=producer.feature_module().verify_stage(plan['feature_stage']['file'],streaming=True)
    prepared=producer.compact.verify_stage(plan['compact_stage']['file'],streaming=True)
    compact_rows=read(prepared['cases_file']);raw_rows=read(prepared['rows_file']);features=read(cached['feature_index_file'])
    need(items==[dict(compact=c,features=f,row=r) for c,f,r in zip(compact_rows,features,raw_rows)]
         and len(compact_rows)==len(features)==len(raw_rows)==4000 and cached['compact_stage']==plan['compact_stage'], 'Main exact compact/feature/source join differs')
    for i,item in enumerate(items):
        c,f,r=item['compact'],item['features'],item['row']
        need(c['sid']==f['sid']==r['sid'] and c['index']==f['index']==i and f['compact']==dict(file=c['file'],sha256=c['sha256'],compact_identity_sha256=c['compact_identity_sha256'])
             and f['input_identity']==dict(pixel_values=c['omitted_pixel_values'],image_grid_thw=c['grid_identity'])
             and f['tensor']['shape']==[196*c['n'],3584] and f['tensor']['dtype']=='torch.float16' and f['finite'], 'Actual shared feature ownership differs')
    diagnostics=[next(i for i,item in enumerate(items) if item['compact']['sid']==sid) for sid in read(prepared['diagnostic_sids_file'])]
    need(diagnostics==plan['diagnostic_indices'] and len(diagnostics)==len(set(diagnostics))==100
         and Counter((items[i]['compact']['n'],items[i]['compact']['qtype']) for i in diagnostics)==
             Counter({(n,q):5 for n in (1,2,4,8,16) for q in ('char_at_frame','steps_in_room','spend_together','where_spend')}), 'Frozen100 training diagnostics differ')
    for stage in (cached,prepared):
        for mapping in (stage['input_bindings'],stage['artifacts']):
            for file,digest in mapping.items():bind(file,bindings,digest)
    profile_proof=producer.audit.verify_report(plan['profile_audit']['file'])
    need(profile_proof['cold_training_components_passed'] and not profile_proof['original_profile_passed'] and not profile_proof['cache_parity_passed'],
         'Preserved cache failure and passed cold evidence required')
    need(config['initial']==plan['initial'] and config['peft_config']==plan['peft_config'],'Actual unfitted initialization differs')
    boundary=producer.boundary_module().verify_report(plan['boundary_report']['file'])
    need(all(boundary['feature_stage'][k]==v for k,v in plan['feature_stage'].items()), 'Boundary/shared feature stage differs')
    projected=project_main(boundary,config['hardware']['total_memory_bytes'])
    need(projected==plan['projection'] and all(v['passed'] for v in projected.values()), 'Independent complete main resource projection differs')
    from transformers import AutoProcessor,__version__ as version
    processor=AutoProcessor.from_pretrained(str(producer.profile.preparation.MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    need(p.fingerprint(processor,str(version))==plan['native_identity']['processor'] and processor.tokenizer.eos_token_id==151645,'Actual tokenizer ownership differs')
    return directory,config,analysis,plan,items,order,processor,{**config['source_sha256'],**config['inherited_source_sha256']}



def project_main(boundary,device_bytes):
    result={}
    for arm in ('ordinary','normalized','mass'):
        row=boundary['projection']['arms'][arm]
        T={int(k):v for k,v in row['microbatch_seconds_by_n'].items()};G={int(k):v for k,v in row['generation_fifty_token_seconds_by_n'].items()}
        need(set(T)==set(G)=={1,2,4,8,16},'Every input length requires measured training/generation timing')
        setup=row['setup_seconds'];O=row['original_optimizer_seconds'];C=row['original_checkpoint_seconds']
        need(all(math.isfinite(v) and v>0 for v in [setup,O,C,*T.values(),*G.values()]),'Finite complete main timing population required')
        training=sum(2400*T[n] for n in T);generation=sum(20*G[n] for n in G);reload=4*max(T[1],T[16])
        estimate=setup+1.25*(training+1500*O+4*C+reload+generation)+60
        peak=row['estimated_training_peak_allocated_bytes'];memory=peak<.9*device_bytes
        result[arm]=dict(passed=estimate<=7200 and memory,seconds=estimate,setup_seconds=setup,training_seconds=training,
            optimizer_seconds=1500*O,checkpoint_seconds=4*C,reload_teacher_seconds=reload,diagnostic_generation_seconds=generation,
            microbatch_seconds_by_n=T,generation_fifty_token_seconds_by_n=G,cap_seconds=7200,shared_feature_cost_separate=True,
            empirical_not_guarantee=True,estimated_peak_allocated_bytes=peak,device_total_bytes=device_bytes,
            memory_eligible=memory,optimizer_peak_not_newly_measured=True)
    return json.loads(json.dumps(result))


def load_case(torch,item,bindings):
    descriptor=item['compact'];feature=item['features'];saved=packet(torch,descriptor,bindings);case=saved['case']
    cached=packet(torch,feature,bindings);values=cached['features']
    ownership=('index','sid','n','input_identity','compact','coordinate_identity','native_identity_sha256')
    need({k:cached[k] for k in ownership}=={k:feature[k] for k in ownership}
         and feature['coordinate_identity']==reference.state_identity(case['coordinates']), 'Selected feature metadata/coordinates differ')
    need(object_sha(producer.compact.tensor_tree(torch,p,case))==descriptor['compact_identity_sha256']
         and saved['omitted_pixel_values']==descriptor['omitted_pixel_values'] and p.tensor_info(values)==feature['tensor']
         and values.dtype==torch.float16 and values.shape==(196*descriptor['n'],3584) and bool(values.isfinite().all())
         and case['metadata']['sid']==descriptor['sid']==feature['sid']==item['row']['sid'], 'Selected native input bytes differ')
    return case,values


def state_audit(torch,config,analysis,plan,bindings):
    arm=config['arm'];refs=analysis['states'];need(set(refs)=={'0','1499','1500'} and refs['0']==plan['initial']
        and refs['1500']==analysis['final_checkpoint'],'Exact initial/pre-last/final state inventory required')
    initial=reference.initial_audit(torch,refs['0'],plan['peft_config'],bindings);states={'0':initial}
    expected_contract={k:v for k,v in initial['contract'].items() if k!='tensors'}
    need({k:v for k,v in config['contract'].items() if k!='tensors'}==expected_contract
         and {k:{a:b for a,b in v.items() if a!='object_id'} for k,v in config['contract']['tensors'].items()}==
             {k:{a:b for a,b in v.items() if a!='object_id'} for k,v in initial['contract']['tensors'].items()}, 'Actual main adapter configuration differs')
    for step in ('1499','1500'):
        value=packet(torch,refs[step],bindings);reference.adapter_state(torch,value['adapter'])
        need(value['arm']==arm and value['updates']==int(step) and refs[step]['updates']==int(step) and value['peft_config']==plan['peft_config'], 'Checkpoint state owner differs')
        if arm=='ordinary':need(value['memory'] is None,'Ordinary arm acquired memory parameters')
        else:
            reference.memory_state(torch,value['memory'])
            if arm=='normalized':need(bool((value['memory']['mass_direction']==0).all()),'Normalized arm acquired a mass contribution')
        states[step]=value
    final=states['1500'];need(final['final_checkpoint'] and final['rng']['cpu'].dtype==torch.uint8 and len(final['rng']['cuda'])==1,'Final RNG/endpoint packet incomplete')
    optimizer=final['optimizer'];group=optimizer['param_groups'];need(len(group)==1,'One AdamW group required');group=group[0]
    values=list(final['adapter'].values())+([] if arm=='ordinary' else list(final['memory'].values()))
    need(len(values)==(224 if arm=='ordinary' else 227) and len(group['params'])==len(values)
         and set(optimizer['state'])==set(group['params']) and group['lr']==2e-4 and tuple(group['betas'])==(.9,.999)
         and group['eps']==1e-8 and group['weight_decay']==.01,'Final optimizer policy/parameter set differs')
    for identifier,value in zip(group['params'],values):
        item=optimizer['state'][identifier]
        need(set(item)=={'step','exp_avg','exp_avg_sq'} and float(item['step'])==1500
             and all(item[k].shape==value.shape and item[k].dtype==torch.float32 and bool(item[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')),
             'Complete final1500 AdamW moments differ')
    expected=dict(native_identity_sha256=plan['native_identity_sha256'],lora=refs['1500'],
        memory=None if arm=='ordinary' else dict(checkpoint=refs['1500'],tensors=reference.state_identity(final['memory'])))
    need(analysis['provenance']==expected,'Final native/memory/adapter provenance differs')
    return states,dict(passed=True,states=refs,trainable_tensors=len(values),optimizer_updates=1500,
                      profile_fitted_weights_used=False,optimizer_reexecuted=False)


def trace_audit(analysis,items,order,arm,bindings):
    bind(analysis['training_trace_file'],bindings,analysis['training_trace_sha256']);bind(analysis['updates_file'],bindings,analysis['updates_sha256'])
    retained=[];per_epoch=defaultdict(lambda:dict(examples=0,sum_mean_ce=0.,head_rows=0));total_rows=0
    with Path(analysis['training_trace_file']).open() as stream:
        count=0
        for count,line in enumerate(stream,1):
            need(count<=12000,'Extra training microcall');row=json.loads(line);entry=order[count-1];item=items[entry['index']];c=item['compact']
            need(all(row[k]==v for k,v in entry.items()) and row['arm']==arm and row['sid']==c['sid'] and row['n']==c['n']
                 and row['target_rows']==len(c['target_ids']) and math.isfinite(row['mean_ce']) and row['mean_ce']>=0
                 and row['backward_loss']==row['mean_ce']/8 and row['gradients_all_present'] and row['gradients_all_finite']
                 and row['layer_forward_calls']==[1]*28 and math.isfinite(row['seconds']) and row['seconds']>0,'Complete scalar training trace differs')
            expected_state=analysis['states']['0'] if count==1 else analysis['states']['1499'] if count==12000 else None
            if expected_state is None:need(row['retained'] is None,'Unreleased training raw retention selection differs')
            else:retained.append(dict(row,parameter_state=expected_state))
            e=per_epoch[entry['epoch']];e['examples']+=1;e['sum_mean_ce']+=row['mean_ce'];e['head_rows']+=row['target_rows'];total_rows+=row['target_rows']
    need(count==12000 and retained==analysis['retained_training'] and total_rows==88665,'Exact retained endpoints/training head inventory differs')
    parameter_count=224 if arm=='ordinary' else 227
    with Path(analysis['updates_file']).open() as stream:
        count=0
        for count,line in enumerate(stream,1):
            need(count<=1500,'Extra optimizer update');row=json.loads(line)
            need(row['update']==count and row['last_micro_index']==count*8-1 and row['lr']==2e-4 and row['betas']==[.9,.999]
                 and row['eps']==1e-8 and row['weight_decay']==.01 and row['clip_norm']==1.
                 and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0 and row['parameter_steps']==[float(count)]*parameter_count
                 and math.isfinite(row['seconds']) and row['seconds']>0,'Exact1500 optimizer trace differs')
    need(count==1500 and all(v['examples']==4000 and v['head_rows']==29555 for v in per_epoch.values()),'Complete three-epoch schedule required')
    return dict(passed=True,examples=12000,updates=1500,head_rows=total_rows,per_epoch=dict(per_epoch),
                scalar_trace_only=True,all_training_losses_independently_recomputed=False,retained_indices=[0,11999])


def teacher_audit(torch,raw,case,item,arm,state_ref,state,modules,data,key,backward):
    e=raw['evidence'];need(e['parameter_state']==state_ref and e['feature_packet']==item['features'],'Teacher endpoint/feature ownership differs')
    value=reference.teacher_audit(torch,raw,case,arm,item['compact']['index'],modules,data,key,True,backward)
    core=None
    if arm=='ordinary':
        need(e['embedding_identity']==e['actual_language_embeddings']==e['native_inputs'][0]['inputs_embeds_identity']
             and not e.get('pixel_route',False),'Actual cold ordinary feature route differs')
    else:
        observed=reference.memory_links(torch,e,case,item['features'],state_ref)
        need(e['native_inputs'][0]['inputs_embeds_identity']==e['metadata']['full_embedding_identity']
             and e['decoder_training']==backward and e['gradient_enabled']==backward,'Teacher actual memory/autograd ownership differs')
        core=reference.memory_audit(torch,observed,item['_features'],case['coordinates'],state['memory'],arm,data,key)
    return dict(**value,key=key,memory=core,all_checks_passed=value['passed'] and (core is None or core['passed']))


def natural_audit(torch,raw,row,case,item,arm,analysis,final,processor,modules,data,key):
    e=raw['evidence'];result=raw['result'];ids=result['generated_ids'];T=len(ids);index=item['compact']['index']
    need(raw['case_index']==index and raw['arm']==arm and raw['phase']=='cold'
         and e['parameter_state']==analysis['final_checkpoint'] and e['feature_packet']==item['features'], 'Natural endpoint/input owner differs')
    scored=score_native_output(processor.tokenizer,ids,case['metadata']['atype'],json.loads(case['target_text'])['answer'])
    text=scored.pop('scoring_text')
    need(text==row['primary_text']==raw['primary_text'] and scored==row['score']==raw['score']
         and result['completed']==scored['completed'] and result['truncated']==scored['truncated']
         and row['generated_ids']==e['generated_ids']==ids,'Actual native strict parser/EOS result differs')
    need(result['text']==processor.tokenizer.decode(ids,skip_special_tokens=True)
         and result['raw_text']==processor.tokenizer.decode(ids,skip_special_tokens=False), 'Saved descriptive decoded text differs')
    expected=dict(model=T,visual=0,language=T,norm=T,head=T,prefix_decoder=0)
    need(result['counters']==e['counters']==expected and result['rope_restored'] and result['hooks_removed']
         and e['rope_restored'] and e['hooks_removed'] and len(e['native_inputs'])==len(e['native_positions'])==len(e['profile_head'])==len(e['shapes'])==T
         and result['raw_logits'].shape==(T,152064) and result['raw_logits'].dtype==torch.float32,'Actual cold natural call/head inventory differs')
    if arm=='ordinary':W=case['metadata']['prompt_width'];first_positions=case['position_ids'];need(result['native_positions_preserved'],'Native image positions changed')
    else:
        observed=reference.memory_links(torch,e,case,item['features'],analysis['final_checkpoint'])
        need(set(observed)=={'tokens','normalized_value','log_mass','mass_feature'} and all(bool(v.isfinite().all()) for v in observed.values()),'Complete finite actual memory capture required')
        meta=e['metadata'];need(meta['actual_lora_tensors']==reference.state_identity(final['adapter']) and meta['caller_provenance']==analysis['provenance']
             and result['metadata']==meta and result['parameter_versions_unchanged'] and not result['prefix_bytes_unchanged'], 'Final cold memory/LoRA ownership differs')
        need(result['generation']==dict(max_new_tokens=50,do_sample=False,num_beams=1,native_eos_token_ids=[151645,151643],
             repetition_penalty=1.,vocabulary_mask=False,other_logits_processors=False,logits_to_keep=1), 'Unrestricted fixed native decoding differs')
        W=case['text']['prefix']['opening_ids'].shape[1]+33+case['text']['suffix']['input_ids'].shape[1]
        first_positions=torch.arange(W).view(1,1,W).expand(4,1,W)
    heads=[]
    for i,cap in enumerate(e['profile_head']):
        start=0 if i==0 else W+i-1;length=W if i==0 else 1
        positions=first_positions if i==0 else torch.full((4,1,1),start,dtype=torch.long)
        if arm=='ordinary' and i:positions[1:]+=case['rope_deltas'].reshape(1,1,1)
        inp=e['native_inputs'][i];shape=e['shapes'][i]
        need(torch.equal(e['native_positions'][i],positions) and torch.equal(result['native_positions'][i],positions)
             and inp['past_length']==start and inp['attention_mask'].tolist()==[[1]*(start+length)]
             and inp['cache_position'].tolist()==list(range(start,start+length)) and not inp['has_pixels'], 'Natural full-prefix/history positions differ')
        need(shape==dict(norm_input_shape=[1,length,3584],norm_output_shape=[1,length,3584],head_input_shape=[1,1,3584],head_output_shape=[1,1,152064]),
             'Actual natural native norm/head shape differs')
        if i:need(inp['input_ids'].tolist()==[[ids[i-1]]] and inp['inputs_embeds_identity'] is None,'Natural continuation did not consume its own output')
        else:
            need(inp['input_ids'] is None and inp['inputs_embeds_identity'] is not None,'Cold path must consume complete actual embeddings')
            if arm!='ordinary':need(inp['inputs_embeds_identity']==e['metadata']['full_embedding_identity'],'Cold input embedding identity differs')
        vector=result['raw_logits'][i]
        need(torch.equal(vector,cap['head_logits'][0,0].float()) and ids[i]==int(vector.argmax())
             and reference.state_identity(cap)==reference.state_identity(result['profile_head'][i]),'Raw head vector or native greedy output differs')
        heads.append(reference.head_audit(torch,cap,modules,data,key+f'_token_{i}'))
    return dict(passed=all(v['passed'] for v in heads),index=index,sid=row['sid'],n=row['n'],qtype=row['qtype'],
        primary_text=text,generated_ids=ids,score=scored,head_audits=heads,head_calls=T,head_rows=T,
        memory_reconstruction_scope='fixed_teacher_examples_only',training_diagnostic_only=True)


def resources(config,directory,out,original_passed):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobID%40,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'sacct.psv';file.write_text(raw)
    rows=[];array=str(config['array_job_id']);own=array+'_'+str(producer.profile.ARMS.index(config['arm']))
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=8 or fields[1]!='mmred_official_native_training':continue
        identifier,name,partition,state,exit_code,seconds,tres,limit=fields
        need(re.fullmatch(re.escape(array)+r'_[012]',identifier) is not None,'Additional main array/allocation is not released')
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        rows.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=exit_code,seconds=int(seconds),
            gpus=int(generic[0]) if generic else sum(map(int,typed)),time_limit_minutes=int(limit),tres=tres))
    need(len(rows)==3 and {r['job_id'] for r in rows}=={array+'_'+str(i) for i in range(3)} and all(r['partition']=='gpu'
         and r['seconds']<=7200 and r['time_limit_minutes']==120 and r['gpus'] in (0,1) for r in rows), 'Exact three-arm bounded main allocation required')
    current=next(r for r in rows if r['job_id']==own);expected=('COMPLETED','0:0') if original_passed else ('FAILED','1:0')
    need((current['state'],current['exit_code'])==expected and current['gpus']==1,'Own complete main allocation/state differs')
    return dict(passed=True,array_job_id=array,own=current,all_array_rows=rows,maximum_per_arm_gpu_seconds=7200,
                maximum_total_gpu_seconds=21600,raw_file=str(file),raw_sha256=sha(file))


def audit(torch,args,out,data,bindings,progress):
    directory,config,original,plan,items,order,processor,inherited=inputs(torch,args.run,bindings)
    own=snapshot(out,inherited);arm=config['arm'];tests=parser_proof();save(out/'parser_proof.json',tests)
    states,state_proof=state_audit(torch,config,original,plan,bindings);trace=trace_audit(original,items,order,arm,bindings)
    parent=record(plan['profile_preparation']['plan_file'],bindings,plan['profile_preparation']['plan_sha256'])
    modules=reference.native_modules(torch,parent,bindings);teachers=[];natural=[];comparisons=[]
    progress.update(teacher_audits=teachers,natural_audits=natural,comparisons=comparisons)
    def selected_teacher(ref,index,state_key,backward,key):
        raw=packet(torch,ref,bindings);case,features=load_case(torch,items[index],bindings)
        item=dict(items[index],_features=features)
        metric=teacher_audit(torch,raw,case,item,arm,original['states'][state_key],states[state_key],modules,data,key,backward)
        metric.update(file=ref['file'],sha256=ref['sha256']);teachers.append(metric)
        save(out/f'teacher_{len(teachers)}.json',metric);return raw
    for row in original['retained_training']:
        index=row['index'];key='0' if row['micro_index']==0 else '1499'
        raw=selected_teacher(row['retained'],index,key,True,f'training_{row["micro_index"]}')
        need(raw['evidence']['loss']==row['mean_ce'],'Retained raw CE and scalar trace differ')
    need(set(original['roundtrip'])=={'before','after'} and all([r['index'] for r in original['roundtrip'][phase]]==plan['roundtrip_indices'] for phase in ('before','after')),
         'Fixed final checkpoint teacher selection differs')
    for slot,index in enumerate(plan['roundtrip_indices']):
        pair={phase:selected_teacher(original['roundtrip'][phase][slot],index,'1500',False,f'roundtrip_{phase}_{index}') for phase in ('before','after')}
        metric=dict(index=index,**reference.gpu_comparison(torch,pair['before'],pair['after'],exact=True));comparisons.append(metric)
    need(len(teachers)==6 and comparisons==original['comparisons'],'Exact six teacher replay cases/checkpoint comparisons differ')
    need([r['index'] for r in original['natural']]==plan['diagnostic_indices'],'Only frozen100 final training diagnostics allowed')
    for row in original['natural']:
        index=row['index'];item=items[index];raw=packet(torch,row,bindings);case,features=load_case(torch,item,bindings)
        need(row['sid']==item['compact']['sid'] and row['n']==item['compact']['n'] and row['qtype']==item['compact']['qtype']
             and row['max_reserved_bytes']>=row['max_allocated_bytes']>0 and row['seconds']>0 and math.isfinite(row['seconds']), 'Natural metadata/cost ownership differs')
        result=natural_audit(torch,raw,row,case,item,arm,original,states['1500'],processor,modules,data,f'natural_{index}')
        result.update(file=row['file'],sha256=row['sha256']);natural.append(result);save(out/f'natural_{index}.json',result)
    G=sum(r['head_rows'] for r in natural);teacher_rows=sum(r['head']['rows'] for r in teachers)
    expected=dict(model=12004+G,backbone=12004+G,visual=0,language=12004+G,norm=12004+G,head=12004+G,backward=12000,optimizer=1500)
    expected_teacher_rows=sum(len(items[order[m]['index']]['compact']['target_ids']) for m in (0,11999))+2*sum(len(items[i]['compact']['target_ids']) for i in plan['roundtrip_indices'])
    need(G<=5000 and len(natural)==100 and teacher_rows==expected_teacher_rows and teacher_rows+G<=5300 and 6+G<=5006
         and original['counters']==expected and original['decoder_layer_calls']==[12004+G]*28
         and original['head_rows']==88665+2*sum(len(items[i]['compact']['target_ids']) for i in plan['roundtrip_indices'])+G,
         'Exact main native/CPU head and decoder inventory differs')
    tables=defaultdict(Counter);total=Counter()
    for row in natural:
        score=row['score'];counts=dict(contexts=1,correct=int(score['correct']),format_valid=int(score['format_valid']),
            semantic_match=int(score['semantic_match']),truncated=int(score['truncated']))
        tables[f'{row["n"]}:{row["qtype"]}'].update(counts);total.update(counts)
    need(total['contexts']==100 and total['correct']==original['training_diagnostic_correct'] and all(v['contexts']==5 for v in tables.values()), 'Diagnostic denominator/scoring differs')
    allocation=resources(config,directory,out,original['passed'])
    need(original['training_peak']['max_reserved_bytes']>=original['training_peak']['max_allocated_bytes']>0, 'Actual training peak memory record required')
    allocation.update(main_projection=plan['projection'][arm],training_peak=original['training_peak'],
        total_diagnostic_seconds=sum(r['seconds'] for r in original['natural']),shared_feature_harvest_is_separate=True)
    numerical=all(v['all_checks_passed'] for v in teachers) and all(v['passed'] for v in natural) and all(v['passed'] for v in comparisons)
    result=dict(protocol=PROTOCOL,passed=numerical,completed=True,arm=arm,policy=POLICY,source_sha256=own,inherited_source_sha256=inherited,
        run=dict(directory=str(directory),config_file=str(directory/'config.json'),config_sha256=sha(directory/'config.json'),
            analysis_file=str(directory/'analysis.json'),analysis_sha256=sha(directory/'analysis.json'),plan_file=config['plan_file'],plan_sha256=config['plan_sha256']),
        original_run_passed=original['passed'],original_cache_profile_passed=False,cache_reuse_qualified=False,
        state_audit=state_proof,scalar_trace_audit=trace,teacher_audits=teachers,natural_audits=natural,comparisons=comparisons,
        counters=expected,cpu_head_calls=6+G,cpu_head_rows=teacher_rows+G,final_checkpoint=original['final_checkpoint'],
        training_diagnostics=dict(total=dict(total),by_length_task={k:dict(v) for k,v in sorted(tables.items())}),
        resources=allocation,tests=tests,all_scheduled_numerical_evidence_collected=True,no_validation_or_test=True,
        no_inference_release=True,no_checkpoint_selection=True)
    save(out/'analysis.json',result);return result


def verify_report(path):
    path=Path(path).resolve();summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['passed'] is summary['completed'] is True
         and summary['source_sha256']==sources() and summary['no_inference_release'],'Passed independent main audit required')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'],'Main audit analysis changed');result=read(summary['analysis_file'])
    need(result['passed'] and result['arm']==summary['arm'] and result['run']==summary['run'] and result['no_inference_release']
         and result['all_scheduled_numerical_evidence_collected'],'Complete main audit handoff differs')
    for key in ('input_bindings','artifacts'):
        need(sha(summary[key+'_file'])==summary[key+'_sha256'],'Main audit manifest changed')
        for file,digest in read(summary[key+'_file']).items():need(sha(file)==digest,'Main audit bound input/artifact changed')
    reference.archive(path.parent,summary['source_sha256'],summary['inherited_source_sha256'],{})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'CPU-only four-core audit required')
    import torch
    torch.set_num_threads(4);started=time.perf_counter();out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';data=DATA/out.name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);bindings={};progress={}
    save(out/'request.json',dict(run=str(args.run.resolve()),policy=POLICY,source_sha256=sources()))
    try:
        result=audit(torch,args,out,data,bindings,progress);elapsed=time.perf_counter()-started
        need(elapsed<=3600,'Fixed3600-second per-arm CPU audit cap exceeded');save(out/'input_bindings.json',bindings)
        (out/'REPORT.md').write_text('Independent main CPU audit: '+str(result['passed'])+'.\n\nAll100 fixed final training diagnostics retained; '+
            str(result['training_diagnostics']['total']['correct'])+'/100 correct under the strict typed JSON/EOS parser. These are training outcomes, not a validation or test release.\n')
        artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()};save(out/'artifacts.json',artifacts)
        summary=dict(protocol=PROTOCOL,passed=result['passed'],completed=True,arm=result['arm'],policy=POLICY,run=result['run'],
            source_sha256=result['source_sha256'],inherited_source_sha256=result['inherited_source_sha256'],analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            no_inference_release=True,elapsed_seconds=elapsed)
        save(out/'summary.json',summary);need(result['passed'],'Main CPU numerical audit failed after complete evidence collection')
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,
            input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),progress=progress,no_inference_release=True));raise


if __name__=='__main__':main()
