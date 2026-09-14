"""Four bare Cosmos prefills comparing original local readout prompts.

Source preparation only until separately released. No branch, generation, fit,
Qwen features, or global-answer scoring. All numerical execution requires Slurm.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import hashlib
import inspect
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
from scripts import profile_native_vision_reasoning_stream as ancestor
from scripts import native_vision_reasoning_stream as stream
from gnnformer.data import build_count_prompt
native=stream.base
need,read,sha,save=ancestor.need,ancestor.read,ancestor.sha,ancestor.save
MODEL=ancestor.MODEL
DATA=Path('/mnt/data/gabriele/gnn_transformer/reasoning_semantic_gate')
OUT=REPO/'outputs/native_aggregation_vlm/reasoning_semantic_gate'
SOURCE_PLAN=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/check_442830/plan.json'
SOURCE_PLAN_SHA='45921222241820700a8e3cbd6637c819f331c4c9f27348b8896e1a233655b8aa'
COSMOS_PROOF=REPO/'outputs/native_aggregation_vlm/reasoning_stream/profile_441994/summary.json'
MODES=('reasoning_local','direct_local')
PROTOCOL='cosmos_original_local_gate_prompt_audit'
POLICY=dict(protocol=PROTOCOL,modes=list(MODES),lengths=[16,64],model_calls=4,vision_calls=4,
    ordinary_norm_calls=4,ordinary_head_calls=4,extra_norm_replays=4,extra_head_replays=4,
    head_rows=164,scored_local_rows=160,global_rows_scored=0,resize=392,native_dtype='torch.float16',
    probability_dtype='torch.float64',gate_dtype='torch.float32',gate_rule='max(0,p1-p0) with full-vocabulary normalization',
    tv_max=.02,top1_exact=True,probability_rtol=1e-12,probability_atol=1e-15,
    direct_feasibility=dict(all_positive_gates_strictly_positive=True,all_negative_gates_exactly_zero=True,minimum_digit_mass=.95),
    per_gpu_job_seconds=180,campaign_gpu_seconds=540,maximum_concurrent_gpus=1,
    no_generation=True,no_branch=True,no_fit=True,no_global_answer_score=True,labels_offline_only=True,
    predecessor_main_array='442892')
OWN=('scripts/probe_reasoning_semantic_gate.py','slurm/reasoning_semantic_gate_check.sbatch',
     'slurm/reasoning_semantic_gate_run.sbatch','slurm/reasoning_semantic_gate_report.sbatch','gnnformer/data.py')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def oid(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sources():return {**ancestor.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Cosmos original local gate audit\n\n[Summary](summary.json) · [Sources](source_hashes.json). Four bare prefills only.\n')
    return frozen


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or h==expected,'Bound input changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==h,'Conflicting input identity');bindings[str(path)]=h
    return h


def labels(record,bindings):
    """QA is read only by CPU preparation/report; never by the model preparer."""
    path=Path(record['path'])/'qa.txt';bind(path,bindings,record['qa_sha256']);lines=path.read_text().splitlines()
    a,b=lines.index('question:'),lines.index('answer:');body=[x.strip() for x in lines[a+1:b] if x.strip()]
    states=[ast.literal_eval(x) for x in body if x.startswith('{')]
    need([x for x in body if not x.startswith('{')]==[record['question']] and int(lines[b+1])==record['gold']
         and oid(dict(states=states,question=record['question']))==record['content_sha256'],'QA identity changed')
    match=re.fullmatch(r'How many frames show (\w+) in the (\w+)\?',record['question']);need(match is not None,'Unexpected exact question')
    character,room=match.groups();need((character,room)==(record['target_character'],record['target_room']),'Question target differs')
    need(len(states)==record['n_frames']==len(record['image_files']),'QA/image occurrence count differs');rows=[]
    for i,(state,image) in enumerate(zip(states,record['image_files'])):
        need(state['step_id']==i+1 and len(state['rooms'])==1,'Physical Step or room count differs')
        place,people=next(iter(state['rooms'].items()));need(len(people)==1,'Exactly one person per frame required')
        c,r=people[0]==character,place==room;category='positive' if c and r else 'char_only' if c else 'room_only' if r else 'neither'
        bind(image['path'],bindings,image['sha256'])
        rows.append(dict(row=i,step=i+1,step_group='at_most16' if i<16 else 'above16',category=category,label=int(c and r),
                         image_sha256=image['sha256'],pair_id=oid([image['sha256'],record['question']])))
    need(sum(r['label'] for r in rows)==record['gold'],'Independent local label recount differs');return rows


def prepare_scene(processor,record,mode):
    """Consume only image/question/identity inputs, with global system fixed."""
    import torch
    from PIL import Image
    need(set(record)=={'sid','n_frames','question','image_files'} and mode in MODES,'Model inputs may not contain labels or answer metadata')
    frames=[];rows=[];conversations=[];padding=processor.tokenizer.padding_side
    try:
        for item in record['image_files']:
            need(sha(item['path'])==item['sha256'],'Original image changed')
            with Image.open(item['path']) as original:
                rgb=original.convert('RGB')
                try:frames.append(rgb.resize((392,392)))
                finally:rgb.close()
        n=len(frames);need(n==record['n_frames'] and n in (16,64),'Fixed old scene length differs')
        for i,frame in enumerate(frames+[None]):
            reasoning=i==n or mode=='reasoning_local'
            text=stream.prompt(record['question']) if reasoning else build_count_prompt(record['question'],1)
            content=[] if frame is None else [dict(type='image',image=frame)]
            content.append(dict(type='text',text=text))
            messages=([dict(role='system',content=[dict(type='text',text=stream.SYSTEM)])] if reasoning else [])
            conversations.append(messages+[dict(role='user',content=content)])
        for conversation in conversations:
            rows.append(dict(processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,
                                                           return_dict=True,return_tensors='pt')))
        packed=native.pack_rows(rows,processor.tokenizer.pad_token_id);processor.tokenizer.padding_side='left'
        ordinary=dict(processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=True,
                                                    return_dict=True,return_tensors='pt',padding=True))
        need(set(packed)==set(ordinary) and all(torch.equal(packed[k],ordinary[k]) for k in packed),'Ordinary mixed processor parity failed')
        width=packed['input_ids'].shape[1]
        meta=dict(arm='parallel',sid=record['sid'],n_frames=n,question=record['question'],question_sha256=oid(record['question']),
            local_mode=mode,local_system=stream.SYSTEM if mode=='reasoning_local' else 'native default helpful system',
            global_system=stream.SYSTEM,local_prompt=stream.prompt(record['question']) if mode=='reasoning_local' else build_count_prompt(record['question'],1),
            global_prompt=stream.prompt(record['question']),row_count=n+1,row_kinds=['local']*n+['global'],global_row=n,local_elements=n,
            original_prompt_width=width,prompt_width=width,prefix_ids=[],row_prompt_tokens=[r['input_ids'].shape[1] for r in rows],
            image_sha256=[r['sha256'] for r in record['image_files']],image_paths=[r['path'] for r in record['image_files']],resize=392,
            processor_parity_checked=True,input_identity={k:native.tensor_info(v) for k,v in packed.items()})
        return dict(inputs=packed,row_inputs=rows,metadata=meta)
    finally:
        processor.tokenizer.padding_side=padding
        for frame in frames:frame.close()


def probabilities(torch,logits):
    need(logits.ndim==2 and logits.shape[1]>16 and logits.dtype==torch.float16 and bool(torch.isfinite(logits).all()),'Finite native FP16 vocabulary logits required')
    x=logits.double();normalizer=torch.logsumexp(x,-1);p0=(x[:,15]-normalizer).exp();p1=(x[:,16]-normalizer).exp()
    gate=(p1-p0).clamp_min(0).float();mass=p0+p1
    return dict(p0=p0,p1=p1,digit_mass=mass,gate=gate,logit_gap=x[:,16]-x[:,15],
                binary_prediction=(x[:,16]>x[:,15]).long(),top1_id=x.argmax(-1),log_normalizer=normalizer)


def replay_metrics(torch,a,b):
    need(a.shape==b.shape and a.ndim==2 and a.dtype==b.dtype==torch.float16,'Native/replay head shapes or dtypes differ')
    pa=a.double().softmax(-1);pb=b.double().softmax(-1);tv=.5*(pa-pb).abs().sum(-1);top=a.argmax(-1)==b.argmax(-1)
    return [dict(row=i,full_vocabulary_tv=float(tv[i]),top1_equal=bool(top[i]),passed=bool(top[i] and tv[i]<=.02)) for i in range(len(a))]


def local_metrics(rows):
    need(rows,'Cannot summarize an empty local partition');positive=[r for r in rows if r['label']];negative=[r for r in rows if not r['label']]
    stats=lambda xs:dict(n=len(xs),minimum=min(xs) if xs else None,maximum=max(xs) if xs else None,mean=math.fsum(xs)/len(xs) if xs else None)
    return dict(n=len(rows),positives=len(positive),negatives=len(negative),
        native_top1_correct=sum(r['top1_id']==15+r['label'] for r in rows),binary_sign_correct=sum(r['binary_prediction']==r['label'] for r in rows),
        false_negative_closed_gate=sum(r['gate']==0 for r in positive),false_positive_open_gate=sum(r['gate']>0 for r in negative),
        digit_mass=stats([r['digit_mass'] for r in rows]),positive_gate=stats([r['gate'] for r in positive]),
        positive_attenuation=stats([1-r['gate'] for r in positive]),negative_leakage=stats([r['gate'] for r in negative]))


def feasible(rows):
    positive=[r for r in rows if r['label']];negative=[r for r in rows if not r['label']]
    need(positive and negative,'Feasibility requires both labels')
    a=all(r['gate']>0 for r in positive);b=all(r['gate']==0 for r in negative);c=min(r['digit_mass'] for r in rows)>=.95
    return dict(passed=a and b and c,all_positive_gates_strictly_positive=a,all_negative_gates_exactly_zero=b,minimum_digit_mass_at_least95pct=c,
                scope='Eligibility on the fixed old local examples only; no aggregate accuracy or adaptive reasoning claim')


def self_test():
    import torch
    x=torch.zeros(3,24,dtype=torch.float16);x[0,16]=2;x[1,15]=2;x[2,15]=x[2,16]=2
    p=probabilities(torch,x);need(p['p0'].dtype==torch.float64 and p['gate'].dtype==torch.float32 and p['gate'][0]>0 and p['gate'][1]==p['gate'][2]==0,'Full mass/sign/tie/dtype rule failed')
    need(float(p['digit_mass'][0])<.5,'Binary normalization replaced full vocabulary')
    need(all(r['passed'] and r['full_vocabulary_tv']==0 for r in replay_metrics(torch,x,x)),'Identical native head replay failed')
    rows=[dict(label=1,gate=.2,digit_mass=.95,top1_id=16,binary_prediction=1),dict(label=0,gate=0.,digit_mass=1.,top1_id=15,binary_prediction=0)]
    need(feasible(rows)['passed'] and not feasible([dict(rows[0],gate=0.),rows[1]])['passed']
         and not feasible([rows[0],dict(rows[1],gate=.001)])['passed'] and not feasible([dict(rows[0],digit_mass=.949),rows[1]])['passed'],'Fixed feasibility boundary changed')
    m=local_metrics(rows);need(m['false_negative_closed_gate']==m['false_positive_open_gate']==0 and m['positive_attenuation']['mean']==.8,'Local denominators/attenuation differ')
    sample='1|reasoning_gate_run|gpu|FAILED|1:0|7|gres/gpu=1,gres/gpu:b200=1|2026-09-11T10:00:00|2026-09-11T10:00:07'
    zero='2|reasoning_gate_run|gpu|CANCELLED|0:0|0||Unknown|Unknown'
    need(parse_accounting(sample+'\n'+zero)['allocated_gpu_seconds']==7,'Failed and zero allocation accounting differs')
    for bad in (sample+'\n'+sample,sample.replace('|7|','|181|'),sample.replace('|FAILED|','|RUNNING|')):
        try:parse_accounting(bad)
        except ValueError:pass
        else:raise ValueError('Invalid resource ledger passed')
    return dict(passed=True,tests=['failed_zero_duplicate_and_duration_accounting','full_vocabulary_mass','strict_sign_and_tie','FP64_to_FP32','native_replay_identity','fixed_feasibility_boundaries','local_denominators'],model_calls=0,head_calls=0)


def check(out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    torch.set_num_threads(4);bindings={};unit=self_test();bind(SOURCE_PLAN,bindings,SOURCE_PLAN_SHA);original=read(SOURCE_PLAN)
    bind(COSMOS_PROOF,bindings);prior=read(COSMOS_PROOF)
    need(prior['completed'] and prior['computational_integrity_passed'] and prior['actual_native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),'Pinned Cosmos software ancestor differs')
    mirror=ancestor.mirror_identity();need(mirror==prior['mirror'],'Cosmos mirror ancestry changed')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=ancestor.native_api(processor);fingerprint=ancestor.fingerprint(processor,str(transformers.__version__))
    need(stream.SYSTEM in (MODEL/'README.md').read_text().replace('\\n','\n'),'Official pinned reasoning system differs')
    for digit,token in [('0',15),('1',16)]:
        need(processor.tokenizer(digit,add_special_tokens=False)['input_ids']==[token]
             and processor.tokenizer.decode([token],skip_special_tokens=False)==digit and token not in processor.tokenizer.all_special_ids,'ASCII gate token identity changed')
    cases=[];bundles={};offline={};items=original['cases']
    need([(c['n_frames'],c['sample']['gold']) for c in items]==[(16,3),(64,6)],'Fixed old software cases differ')
    for case in items:
        sample=case['sample'];offline[sample['sid']]=labels(sample,bindings);pair=[]
        restricted={k:sample[k] for k in ('sid','n_frames','question','image_files')}
        for mode in MODES:
            bundle=prepare_scene(processor,restricted,mode);layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle)
            key=f'N{sample["n_frames"]}_{mode}';bundles[key]=bundle;pair.append(bundle)
            cases.append(dict(case_id=key,mode=mode,n_frames=sample['n_frames'],sid=sample['sid'],sample=sample,
                              metadata=bundle['metadata'],layout=layout['metadata']))
        for name in ('pixel_values','image_grid_thw'):
            need(torch.equal(pair[0]['inputs'][name],pair[1]['inputs'][name]),'Prompt variant changed image inputs')
        need(all(torch.equal(pair[0]['row_inputs'][-1][k],pair[1]['row_inputs'][-1][k]) for k in pair[0]['row_inputs'][-1]),'Global reasoning prompt changed across local modes')
    need(sum(c['n_frames']+1 for c in cases)==164 and len(cases)==4,'Fixed prefill inventory differs')
    destination=DATA/out.name;destination.mkdir(parents=True,exist_ok=False);file=destination/'prepared.pt'
    torch.save(dict(schema_version=1,bundles=bundles),file);save(out/'offline_labels.json',offline)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,mirror=mirror,
        runtime=ancestor.versions(),processor=fingerprint,native_api=api,count_token_ids={'0':15,'1':16},
        norm_source_sha256=sha(inspect.getfile(Qwen2RMSNorm)),rms_norm_eps=float(owner.config.text_config.rms_norm_eps),
        cases=cases,prepared_file=str(file),prepared_sha256=sha(file),offline_labels_file=str(out/'offline_labels.json'),
        offline_labels_sha256=sha(out/'offline_labels.json'),tests=unit,no_model_loaded=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=unit,
                source_sha256=frozen,no_model_loaded=True,cases=4,head_rows=164,scored_local_rows=160)


def verify_plan(path):
    path=Path(path).resolve();need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'CPU plan sidecar changed');plan=read(path)
    need(plan['policy']==POLICY and plan['source_sha256']==sources(),'Frozen policy/source changed')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'CPU source snapshot changed')
    for file,h in plan['artifact_bindings'].items():need(sha(file)==h,'Source input changed')
    need(sha(plan['prepared_file'])==plan['prepared_sha256'] and sha(plan['offline_labels_file'])==plan['offline_labels_sha256'],'Prepared bundle or offline labels changed')
    summary=read(path.parent/'summary.json');need(summary['passed'] and summary['plan_sha256']==sha(path),'Successful completed CPU preparation required')
    return plan


def predecessor_finished(out):
    command=['sacct','-X','-j',POLICY['predecessor_main_array'],'--parsable2','--noheader','--format=JobID,State,Partition']
    result=subprocess.run(command,capture_output=True,text=True,check=True);(out/'predecessor_sacct.psv').write_text(result.stdout)
    rows=[x.split('|') for x in result.stdout.splitlines() if x.strip()];expected={POLICY['predecessor_main_array']+'_'+str(i) for i in range(4)}
    selected=[r for r in rows if r[0] in expected]
    need({r[0] for r in selected}==expected and all(r[1].split()[0].rstrip('+') in TERMINAL and r[2]=='gpu' for r in selected),
         'All four V18 main allocations must finish before this GPU audit')
    return dict(passed=True,array=POLICY['predecessor_main_array'],file=str(out/'predecessor_sacct.psv'),sha256=sha(out/'predecessor_sacct.psv'),command=command)


def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    torch.set_num_threads(4);started=time.perf_counter();plan=verify_plan(args.plan);predecessor=predecessor_finished(out)
    reserve=accounting(out,exclude_job=os.environ['SLURM_JOB_ID'])
    need(reserve['allocated_gpu_seconds']+180<=540,'A full new180-second reservation exceeds the campaign cap')
    reserve.update(reserved_seconds=180,projected_campaign_seconds=reserve['allocated_gpu_seconds']+180)
    save(out/'resource_reservation.json',reserve)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One registered B200 required')
    need(ancestor.mirror_identity()==plan['mirror'] and ancestor.versions()==plan['runtime'],'Cosmos/runtime identity changed')
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True);need(blob['schema_version']==1,'Prepared packet schema differs')
    load_started=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=runtime.model.eval().requires_grad_(False)
    norm=stream.contract(model);head=model.lm_head;torch.cuda.synchronize();load_seconds=time.perf_counter()-load_started
    need(norm.weight.dtype==head.weight.dtype==torch.float16,'Actual native FP16 norm/head required; never recast')
    need(ancestor.fingerprint(runtime.processor,str(transformers.__version__))==plan['processor']
         and ancestor.native_api(runtime.processor)[2]==plan['native_api'],'Actual loaded native processor/API differs')
    identity=dict(schema_version=1,mirror=plan['mirror'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        norm_weight=native.tensor_info(norm.weight),head_weight=native.tensor_info(head.weight),
        norm_source_sha256=sha(inspect.getfile(type(norm))),rms_norm_eps=float(norm.variance_epsilon),native_dtype='torch.float16',count_token_ids=plan['count_token_ids'],
        backend=dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),cuda=str(torch.version.cuda),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),
            quantization='nf4_double_bf16',attention='sdpa'))
    need(identity['norm_source_sha256']==plan['norm_source_sha256'] and identity['rms_norm_eps']==plan['rms_norm_eps'],
         'Actual native normalization source/epsilon differs from CPU binding')
    versions={n:p._version for n,p in model.named_parameters()};counts=dict(model=0,vision=0,language=0,norm=0,head=0);capture={};records=[];extra=0
    destination=DATA/out.name;destination.mkdir(parents=True,exist_ok=False);save(out/'native_identity.json',identity)
    def count(key):
        def hook(*_):counts[key]+=1
        return hook
    def before_norm(module,args):capture['hidden']=args[0][:,-1:,:].detach().clone();counts['norm']+=1
    def after_norm(module,args,output):capture['normalized']=output[:,-1:,:].detach().clone()
    def before_language(module,args,kwargs):
        counts['language']+=1;capture['position_ids']=kwargs['position_ids'].detach().clone();capture['attention_mask']=kwargs['attention_mask'].detach().clone()
    failure=None
    try:
        for case in plan['cases']:
            case_started=time.perf_counter();key=case['case_id'];bundle=blob['bundles'][key];need(bundle['metadata']==case['metadata'],'Prepared metadata changed')
            need({k:native.tensor_info(v) for k,v in bundle['inputs'].items()}==case['metadata']['input_identity'],'Actual input tensor identity differs')
            layout=native.audit_layout(get_rope_index_fn(model),bundle);need(layout['metadata']==case['layout'],'Native logical positions changed')
            capture.clear();old_rope=getattr(model.model,'rope_deltas',None)
            with ExitStack() as stack:
                stack.callback(setattr,model.model,'rope_deltas',old_rope)
                for handle in [model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('vision')),
                    model.model.language_model.register_forward_pre_hook(before_language,with_kwargs=True),norm.register_forward_pre_hook(before_norm),
                    norm.register_forward_hook(after_norm),head.register_forward_pre_hook(count('head'))]:stack.callback(handle.remove)
                with torch.inference_mode():output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
                need(output.past_key_values is None and output.logits.shape==(case['n_frames']+1,1,152064),'Bare prefill returned cache or wrong logit shape')
                need(torch.equal(capture['position_ids'].cpu(),layout['position_ids']) and torch.equal(capture['attention_mask'].cpu(),bundle['inputs']['attention_mask'])
                     and torch.equal(model.model.rope_deltas.cpu(),layout['rope_deltas']),'Actual native mask/positions changed')
            with torch.inference_mode():
                replay_norm=norm.forward(capture['hidden']);replay=head.forward(replay_norm);extra+=1
                metrics=replay_metrics(torch,output.logits[:,0],replay[:,0]);p=probabilities(torch,output.logits[:,0])
            need(capture['hidden'].shape==capture['normalized'].shape==(case['n_frames']+1,1,3584)
                 and capture['hidden'].dtype==capture['normalized'].dtype==output.logits.dtype==torch.float16,'Original native state dtype/shape differs')
            raw=dict(schema_version=1,metadata=case['metadata'],hidden=capture['hidden'].cpu(),normalized=capture['normalized'].cpu(),
                logits=output.logits.detach().cpu(),replay_normalized=replay_norm.cpu(),replay_logits=replay.cpu(),
                position_ids=capture['position_ids'].cpu(),attention_mask=capture['attention_mask'].cpu(),
                probabilities={k:v.cpu() for k,v in p.items()})
            rawfile=destination/(key+'.pt');torch.save(raw,rawfile)
            record=dict(case_id=key,mode=case['mode'],sid=case['sid'],n_frames=case['n_frames'],raw_file=str(rawfile),raw_sha256=sha(rawfile),
                tensors={k:native.tensor_info(v) for k,v in raw.items() if isinstance(v,torch.Tensor)},
                probability_tensors={k:native.tensor_info(v) for k,v in raw['probabilities'].items()},replay=metrics,
                replay_normalized_exact=torch.equal(capture['normalized'],replay_norm),native_identity_sha256=oid(identity),
                seconds=time.perf_counter()-case_started)
            records.append(record);save(out/f'record_{len(records)-1:03d}.json',record)
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        unchanged=versions=={n:p._version for n,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters())
        complete=len(records)==4;numeric=complete and all(r['passed'] for record in records for r in record['replay'])
        expected=dict(model=4,vision=4,language=4,norm=4,head=4)
        passed=complete and numeric and unchanged and counts==expected and extra==4 and failure is None
        save(out/'records.json',records)
        save(out/'summary.json',dict(schema_version=1,protocol=PROTOCOL,completed=complete,passed=passed,computational_integrity_passed=passed,
            native_replay_passed=numeric,counts=counts,extra_norm_calls=extra,extra_head_calls=extra,records_file=str(out/'records.json'),records_sha256=sha(out/'records.json'),
            native_identity_file=str(out/'native_identity.json'),native_identity_sha256=oid(identity),native_identity_file_sha256=sha(out/'native_identity.json'),
            native_parameters_unchanged=unchanged,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
            predecessor=predecessor,resource_reservation=reserve,model_load_seconds=load_seconds,seconds=time.perf_counter()-started,
            failure=failure,no_generation=True,no_branch=True,no_fit=True,no_global_answer_score=True,slurm_job_id=os.environ['SLURM_JOB_ID']))
    need(passed,'Four complete native prefill/replay bindings failed; all available raw records retained')
    return None


def parse_accounting(raw,exclude_job=None):
    rows=[];events=[]
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==9,'Unexpected accounting field count')
        job,name,partition,state,exitcode,seconds,tres,start,end=fields
        if partition!='gpu' or name!='reasoning_gate_run' or job==exclude_job:continue
        state=state.split()[0].rstrip('+');need(state in TERMINAL and job.isdigit(),'Nonterminal or nonallocation audit row')
        entries=dict(x.split('=',1) for x in tres.split(',') if '=' in x);typed=[int(v) for k,v in entries.items() if k.startswith('gres/gpu:')]
        gpus=int(entries['gres/gpu']) if 'gres/gpu' in entries else sum(typed);seconds=int(seconds)
        need(gpus in (0,1) and seconds>=0 and (not typed or sum(typed)==gpus) and (gpus==0 or seconds<=180),'Per-job GPU count/cap exceeded')
        if gpus and start!=end:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(b>=a,'Invalid GPU interval');events.extend([(a,1),(b,-1)])
        rows.append(dict(job_id=job,state=state,exit_code=exitcode,gpus=gpus,elapsed_seconds=seconds,gpu_seconds=gpus*seconds,start=start,end=end))
    need(len({r['job_id'] for r in rows})==len(rows),'Duplicate GPU allocation');total=sum(r['gpu_seconds'] for r in rows);running=maximum=0
    for _,change in sorted(events):running+=change;need(running>=0,'Invalid allocation ordering');maximum=max(maximum,running)
    need(running==0 and maximum<=1 and total<=540,'Campaign GPU cap or concurrency exceeded')
    return dict(passed=True,jobs=rows,allocated_gpu_seconds=total,maximum_concurrent_gpus=maximum,all_failures_retained=True)


def accounting(out,*,exclude_job=None):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);file=out/'all_user_sacct.psv';file.write_text(result.stdout)
    return dict(parse_accounting(result.stdout,exclude_job),raw_file=str(file),raw_sha256=sha(file),command=command)


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);directory=Path(args.run_directory).resolve();summary=read(directory/'summary.json');plan=verify_plan(summary['plan_file'])
    need(summary['passed'] and summary['completed'] and summary['source_sha256']==frozen==plan['source_sha256']
         and sha(summary['plan_file'])==summary['plan_sha256'],'Complete exact-source GPU result required')
    for name,h in frozen.items():need(sha(directory/'source'/name.replace('/','_'))==h,'GPU source snapshot changed')
    need(sha(summary['records_file'])==summary['records_sha256'] and sha(summary['native_identity_file'])==summary['native_identity_file_sha256'],'GPU record/native identity changed')
    identity=read(summary['native_identity_file']);need(oid(identity)==summary['native_identity_sha256'] and all(identity[k]==plan[k] for k in ('mirror','runtime','processor','native_api')),'Actual native identity differs')
    need(identity['norm_source_sha256']==plan['norm_source_sha256'] and identity['rms_norm_eps']==plan['rms_norm_eps']
         and identity['backend']['gpu']=='NVIDIA B200' and identity['backend']['quantization']=='nf4_double_bf16'
         and identity['backend']['attention']=='sdpa' and identity['native_dtype']=='torch.float16' and identity['norm_weight']['shape']==[3584]
         and identity['head_weight']['shape']==[152064,3584] and identity['norm_weight']['dtype']==identity['head_weight']['dtype']=='torch.float16','Native norm/head tensor identity malformed')
    records=read(summary['records_file']);need(len(records)==4 and summary['counts']==dict(model=4,vision=4,language=4,norm=4,head=4)
         and summary['extra_norm_calls']==summary['extra_head_calls']==4 and summary['native_parameters_unchanged'],'Fixed four-prefill invocation inventory differs')
    stored_labels=read(plan['offline_labels_file']);rows=[];bindings={};all_metrics=[];max_probability_difference=0.
    for case,record in zip(plan['cases'],records):
        need(all(record[k]==case[k] for k in ('case_id','sid','mode','n_frames')) and record['native_identity_sha256']==oid(identity),'Raw case ownership differs')
        truth=labels(case['sample'],bindings);need(truth==stored_labels[case['sid']],'Independent offline labels differ')
        need(sha(record['raw_file'])==record['raw_sha256'],'Raw native head archive changed');raw=torch.load(record['raw_file'],map_location='cpu',weights_only=True)
        need(raw['schema_version']==1 and raw['metadata']==case['metadata'],'Raw input metadata changed')
        for key,info in record['tensors'].items():need(native.tensor_info(raw[key])==info,'Raw tensor bytes changed')
        for key,info in record['probability_tensors'].items():need(native.tensor_info(raw['probabilities'][key])==info,'Probability tensor bytes changed')
        n=case['n_frames'];need(raw['hidden'].shape==raw['normalized'].shape==raw['replay_normalized'].shape==(n+1,1,3584) and raw['logits'].shape==raw['replay_logits'].shape==(n+1,1,152064),'Raw all-row coverage differs')
        need(torch.equal(raw['normalized'],raw['replay_normalized'])==record['replay_normalized_exact'],'Descriptive normalization equality differs')
        need(native.tensor_info(raw['position_ids'])==case['layout']['position_ids']
             and native.tensor_info(raw['attention_mask'])==case['metadata']['input_identity']['attention_mask'],'Archived native positions/mask differ from prepared input')
        need(all(raw[k].dtype==torch.float16 and bool(torch.isfinite(raw[k]).all()) for k in ('hidden','normalized','logits','replay_normalized','replay_logits')),'Raw native dtype/finiteness differs')
        metrics=replay_metrics(torch,raw['logits'][:,0],raw['replay_logits'][:,0]);need(len(metrics)==len(record['replay'])==n+1,'Replay row inventory differs')
        for actual,saved in zip(metrics,record['replay']):
            need(actual['row']==saved['row'] and actual['top1_equal']==saved['top1_equal'] and actual['passed']==saved['passed']
                 and math.isclose(actual['full_vocabulary_tv'],saved['full_vocabulary_tv'],rel_tol=1e-9,abs_tol=1e-12),'Native replay decision differs')
        all_metrics.extend(metrics);p=probabilities(torch,raw['logits'][:,0])
        for key,value in p.items():
            saved=raw['probabilities'][key];need(value.shape==saved.shape and value.dtype==saved.dtype,'Probability/gate scalar schema differs')
            if key=='gate':
                need(torch.equal(saved,(raw['probabilities']['p1']-raw['probabilities']['p0']).clamp_min(0).float()),'Stored gate is not exact savedFP64-gap toFP32')
                # CPU/GPU probability reduction roundoff is audited separately;
                # preserve the actual deployed FP32 gate for all decisions.
                p[key]=saved
            elif key in ('binary_prediction','top1_id'):need(torch.equal(value,saved),'Exact native sign/top1 differs')
            else:
                need(torch.allclose(value,saved,rtol=1e-12,atol=1e-15),'FP64 probability reduction differs beyond fixed roundoff')
                max_probability_difference=max(max_probability_difference,float((value-saved).abs().max()))
        for i,label in enumerate(truth):
            rows.append(dict(case_id=case['case_id'],mode=case['mode'],sid=case['sid'],n_frames=n,**label,
                             **{k:(int(v[i]) if k in ('binary_prediction','top1_id') else float(v[i])) for k,v in p.items()}))
    need(len(rows)==160 and len(all_metrics)==164 and all(m['passed'] for m in all_metrics),'Complete binding/local denominator audit failed')
    aggregate={}
    for mode in MODES:
        selected=[r for r in rows if r['mode']==mode];parts={'all':selected}
        for n in (16,64):parts[f'N{n}']=[r for r in selected if r['n_frames']==n]
        for step in ('at_most16','above16'):parts['Step_'+step]=[r for r in selected if r['step_group']==step]
        for cat in ('positive','char_only','room_only','neither'):parts[cat]=[r for r in selected if r['category']==cat]
        aggregate[mode]={key:local_metrics(value) for key,value in parts.items() if value}
    budget=accounting(out);own=[r for r in budget['jobs'] if r['job_id']==str(summary['slurm_job_id'])]
    need(len(own)==1 and own[0]['state']=='COMPLETED' and own[0]['exit_code']=='0:0' and own[0]['gpus']==1,'Successful GPU allocation absent')
    decision=feasible([r for r in rows if r['mode']=='direct_local']);save(out/'rows.json',rows)
    analysis=dict(schema_version=1,protocol=PROTOCOL,passed=True,completed=True,source_sha256=frozen,plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        run_directory=str(directory),run_summary_sha256=sha(directory/'summary.json'),native_identity=identity,native_identity_sha256=oid(identity),
        rows_file=str(out/'rows.json'),rows_sha256=sha(out/'rows.json'),scored_local_rows=160,head_rows=164,metrics=aggregate,direct_local_feasibility=decision,
        native_binding_rows=164,max_native_replay_tv=max(x['full_vocabulary_tv'] for x in all_metrics),maximum_probability_roundoff=max_probability_difference,
        accounting=budget,no_generation=True,no_fit=True,no_global_answer_score=True,no_adaptive_reasoning_claim=True,
        interpretation='Direct feasibility is separate from execution correctness; old software scenes do not establish aggregate accuracy or broad local generalization.')
    save(out/'analysis.json',analysis)
    (out/'REPORT.md').write_text('# Cosmos original local readout\n\nDirect-local feasibility: **'+str(decision['passed'])+'**. All160 local rows retained. No generated/global answers scored.\n\n[Per-mode/category/Step results and provenance](analysis.json).\n')
    return dict(passed=True,completed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
                direct_local_feasibility=decision,allocated_gpu_seconds=budget['allocated_gpu_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--report',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--run-directory',type=Path);args=parser.parse_args()
    native.require_slurm(gpu=args.run)
    if not args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'Preparation/report requires CPU Slurm')
    label='check' if args.check else 'run' if args.run else 'report';out=OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out);started=time.perf_counter()
    try:
        if args.check:result=check(out,frozen)
        elif args.run:
            need(args.plan is not None,'Supply exact completed CPU plan');result=run(args,out,frozen)
        else:
            need(args.run_directory is not None,'Supply completed four-prefill GPU directory');result=report(args,out,frozen)
        need(sources()==frozen,'Frozen source changed during execution')
        if result is not None:
            result.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
