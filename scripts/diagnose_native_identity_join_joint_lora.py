"""Source-held ordinary joint-image LoRA CPU preparation and bounded GPU profile.

No main fit or validation inference is implemented. Existing NF4 base storage
and native generation are preserved; only the exact language q/k/v/o adapters
are trainable in the two profile updates.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_native_identity_join_joint_baseline as frozen_joint
from scripts import native_vision_v7_runtime as native
from scripts import stage_native_vision_v10_features as backend
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder
from scripts.probe_native_vision_v2_prefix import fingerprint
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha,model_metadata
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_lora'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_lora')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_joint_lora')
PROTOCOL='identity_join_joint_lora'
PARENT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_orientation_training/check_443676/plan.json'
PARENT_SHA='d299d42a4444520f148f86252e2931ad599e03afac9edadbe9201836229bf097'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_LORA_PROPOSAL.md'
PROPOSAL_SHA='550dd4503973a82ff95af3f62921ee61df2d8a70f7e727524d9a7f178e28ecf1'
JOB='identity_join_joint_lora_profile'
OWN=('scripts/diagnose_native_identity_join_joint_lora.py',PROPOSAL,
    'slurm/native_identity_join_joint_lora_check.sbatch','slurm/native_identity_join_joint_lora_profile.sbatch')
TARGETS=tuple(f'model.language_model.layers.{i}.self_attn.{projection}_proj' for i in range(28) for projection in ('q','k','v','o'))
POLICY=dict(protocol=PROTOCOL,contexts=216,image_counts=[8,16],epochs=12,scene_presentations=2592,gradient_accumulation=8,updates=324,
    rank=16,alpha=32,dropout=.05,seed=24,trainable_parameters=10092544,adapter_tensors=224,language_layers=28,
    learning_rate=2e-4,betas=[.9,.999],epsilon=1e-8,weight_decay=.01,clip=1.,checkpointing=False,
    profile_seconds=240,cpu_check_seconds=300,intended_main_seconds=3600,main_implemented=False,
    profile_natural_trajectories=8,profile_native_calls_cap=52,profile_vision_calls=28,profile_backwards=16,profile_updates=2,
    roundtrip_eval_calls=4,maximum_new_tokens=4,no_validation_inference=True,no_native_feature_cache=True,
    no_prepare_model_for_kbit_training=True,base_storage_dtypes_preserved=True,train_adapter_dtype='torch.float32',
    ordinary_joint_images_first=True,no_prompt_wrapper=True,no_broadcast=True)
bind=frozen_joint.bind
tensor_info=frozen_joint.tensor_info
cpu_tree=frozen_joint.cpu_tree


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held joint LoRA protocol changed')
    return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    need(sha(PARENT)==PARENT_SHA,'Fixed216 parent CPU plan changed');parent=read(PARENT);result={}
    for mapping in (frozen_joint.sources(),frozen_joint.inherited_sources(),parent['source_sha256'],parent['inherited_source_sha256']):
        for name,digest in mapping.items():
            need(name not in result or result[name]==digest,'Inherited source conflict');result[name]=digest
    need(all(sha(REPO/name)==digest for name,digest in result.items()),'Inherited source changed')
    return result


def snapshot(out):
    result=sources();(out/'source').mkdir()
    for name,digest in result.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==digest,'Source snapshot differs')
    save(out/'source_hashes.json',result);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return result


def model_view(row):return {key:row[key] for key in ('sid','n_frames','question','image_files')}

def prepare_joint(processor,sample):
    native.require_slurm();need(set(sample)=={'sid','n_frames','question','image_files'} and sample['n_frames'] in (8,16)
        and len(sample['image_files'])==sample['n_frames'] and isinstance(sample['question'],str) and bool(sample['question'].strip()),'Exact four-key N8/N16 input required')
    import torch
    from PIL import Image
    from gnnformer.data import build_prompt_inputs
    frames=[];side=processor.tokenizer.padding_side
    try:
        for image in sample['image_files']:
            path=Path(image['path']);need(sha(path)==image['sha256'],'Original joint image changed')
            with Image.open(path) as picture:
                need(picture.mode=='RGB' and picture.size==(512,512),'Original canonical RGB format differs');frames.append(picture.resize((392,392)))
        row=build_prompt_inputs(processor,frames,sample['question']);packed=native.pack_rows([row],processor.tokenizer.pad_token_id)
        conversation=[dict(role='user',content=[dict(type='image',image=frame) for frame in frames]+[dict(type='text',text=sample['question'])])]
        processor.tokenizer.padding_side='left';ordinary=dict(processor.apply_chat_template([conversation],add_generation_prompt=True,
            tokenize=True,return_dict=True,return_tensors='pt',padding=True))
        need(set(ordinary)==set(packed) and all(torch.equal(ordinary[k],packed[k]) for k in packed),'Ordinary one-row processor parity failed')
        width=packed['input_ids'].shape[1]
        meta=dict(arm='joint',sid=sample['sid'],n_frames=sample['n_frames'],question=sample['question'],question_sha256=object_sha(sample['question']),
            global_prompt=sample['question'],local_prompt=None,row_kinds=['joint'],row_count=1,global_row=0,local_elements=0,
            row_prompt_tokens=[width],original_prompt_width=width,prompt_width=width,prefix_ids=[],resize=392,processor_parity_checked=True,
            image_paths=[i['path'] for i in sample['image_files']],image_sha256=[i['sha256'] for i in sample['image_files']],
            input_identity={k:tensor_info(v) for k,v in packed.items()},ordinary_images_first=True,no_prompt_wrapper=True)
        bundle=dict(inputs=packed,row_inputs=[row],metadata=meta);validate_joint(torch,bundle);return bundle
    finally:
        processor.tokenizer.padding_side=side
        for frame in frames:frame.close()

def validate_joint(torch,bundle):
    x=bundle['inputs'];m=bundle['metadata'];rows=bundle['row_inputs']
    need(set(x)=={'input_ids','attention_mask','pixel_values','image_grid_thw'} and len(rows)==1 and set(rows[0])==set(x)
         and all(torch.equal(x[k],rows[0][k]) for k in x) and x['input_ids'].shape==x['attention_mask'].shape==(1,m['prompt_width'])
         and bool((x['attention_mask']==1).all()) and x['image_grid_thw'].shape==(m['n_frames'],3),'One complete ordinary joint row required')
    need(m['n_frames'] in (8,16) and m['row_count']==1 and m['global_row']==0 and m['row_kinds']==['joint'] and m['local_elements']==0
         and m['prefix_ids']==[] and m['global_prompt']==m['question'] and m['local_prompt'] is None and m['resize']==392
         and m['processor_parity_checked'] is m['ordinary_images_first'] is m['no_prompt_wrapper'] is True
         and m['input_identity']=={k:tensor_info(v) for k,v in x.items()},'Joint input semantics or identity differs')

def generation_policy(model,tokenizer):
    from transformers import GenerationConfig
    eos=model.generation_config.eos_token_id
    need(eos==[151645,151643] and tokenizer.eos_token_id==151645,'Native EOS differs')
    config=GenerationConfig(max_new_tokens=4,do_sample=False,num_beams=1,num_return_sequences=1,repetition_penalty=1.,use_cache=True,
        bos_token_id=model.generation_config.bos_token_id,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,output_logits=False,output_scores=False)
    return config,dict(max_new_tokens=4,do_sample=False,num_beams=1,repetition_penalty=1.,use_cache=True,native_eos_token_ids=eos,
        target_eos_token_id=151645,pad_token_id=tokenizer.pad_token_id,output_logits=False,output_scores=False,
        vocabulary_mask=False,other_logits_processors=False,logits_to_keep=1,ordinary_joint=True,broadcast=False)

def generate_joint(model,processor,bundle,*,native_identity_sha256,capture_head=False,evidence=None):
    native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn,move_to_device
    norm=native.native_contract(model,None);head=model.lm_head;validate_joint(torch,bundle)
    need(not norm._forward_pre_hooks and not head._forward_pre_hooks,'Ordinary joint run cannot inherit intervention hooks')
    layout=native.audit_layout(get_rope_index_fn(model),bundle);m=bundle['metadata'];width=m['prompt_width'];config,policy=generation_policy(model,processor.tokenizer)
    evidence={} if evidence is None else evidence
    counts=dict(model=0,visual=0,language=0,norm=0,head=0);inputs=[];positions=[];shapes=[];heads=[];latest={}
    recorder=GlobalLogitRecorder(max_steps=4,full_vectors=True)
    evidence.update(counters=counts,native_inputs=inputs,native_positions=positions,shapes=shapes,profile_head=heads,raw_vectors=recorder.vectors,logit_records=recorder.records)
    versions={k:v._version for k,v in model.named_parameters()};rope_before=getattr(model.model,'rope_deltas',None)
    def before_model(module,args,kw):
        t=counts['model'];counts['model']+=1;ids=kw['input_ids'];mask=kw['attention_mask'];cache=kw.get('past_key_values')
        inputs.append(dict(input_ids=ids.detach().cpu().clone(),attention_mask=mask.detach().cpu().clone(),has_pixels=kw.get('pixel_values') is not None,
            past_length=0 if cache is None else cache.get_seq_length()))
        need(ids.shape==(1,width if t==0 else 1) and mask.shape==(1,width+t) and bool((mask==1).all()),'Native joint input/cache mask differs')
        need(torch.equal(ids.cpu(),bundle['inputs']['input_ids']) and kw.get('pixel_values') is not None if t==0 else
            kw.get('pixel_values') is None and cache is not None and cache.get_seq_length()==width+t-1,'Joint image/cache ownership differs')
    def visual(*_):counts['visual']+=1
    def language(module,args,kw):
        t=counts['language'];counts['language']+=1;pos=kw['position_ids'].detach().cpu().clone();positions.append(pos)
        mask=kw['attention_mask'].detach().cpu();text=mask.long().cumsum(-1)-1
        expected=layout['position_ids'] if t==0 else (layout['rope_deltas'].view(1,1,1)+width+t-1).expand(3,-1,-1)
        need(pos.shape==(4,1,width if t==0 else 1) and torch.equal(pos[1:],expected)
             and torch.equal(pos[0],text if t==0 else text[:,-1:]),'Actual native joint text/mRoPE differs')
    def before_norm(module,args):
        counts['norm']+=1;latest['norm_input_shape']=list(args[0].shape)
        if capture_head:latest['norm_query_input']=args[0][:,-1:,:].detach().cpu().clone()
    def after_norm(module,args,output):
        latest['norm_output_shape']=list(output.shape)
        if capture_head:latest['normalized_query']=output[:,-1:,:].detach().cpu().clone()
    def after_head(module,args,output):
        t=counts['head'];counts['head']+=1
        shape=dict(norm_input_shape=latest['norm_input_shape'],norm_output_shape=latest['norm_output_shape'],head_input_shape=list(args[0].shape),head_output_shape=list(output.shape),
            norm_dtype=str(norm.weight.dtype),head_input_dtype=str(args[0].dtype),head_output_dtype=str(output.dtype));shapes.append(shape)
        if capture_head:heads.append(dict(norm_query_input=latest['norm_query_input'],normalized_query=latest['normalized_query'],
            head_input=args[0].detach().cpu().clone(),head_logits=output.detach().cpu().clone()))
        need(shape['norm_input_shape']==shape['norm_output_shape']==[1,width if t==0 else 1,3584]
             and shape['head_input_shape']==[1,1,3584] and shape['head_output_shape']==[1,1,152064]
             and args[0].dtype==output.dtype==torch.float16,'Actual joint norm/head shape differs; profile cannot silently expand scope')
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope_before)
        for handle in (model.register_forward_pre_hook(before_model,with_kwargs=True),model.register_forward_hook(recorder),
            model.model.visual.register_forward_pre_hook(visual),model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),
            norm.register_forward_pre_hook(before_norm),norm.register_forward_hook(after_norm),head.register_forward_hook(after_head)):
            stack.callback(handle.remove)
        with torch.inference_mode():output=model.generate(**move_to_device(bundle['inputs'],model.device),generation_config=config,logits_to_keep=1)
        torch.cuda.synchronize();ids=output.sequences[0,width:].detach().cpu().tolist();t=len(ids);raw=torch.stack(recorder.vectors)
        need(getattr(output,'logits',None) is None and getattr(output,'scores',None) is None and 1<=t<=4
             and counts==dict(model=t,visual=1,language=t,norm=t,head=t) and len(shapes)==t
             and len(heads)==(t if capture_head else 0) and raw.shape==(t,152064) and raw.dtype==torch.float32
             and torch.equal(raw,raw.half().float()) and raw.argmax(-1).tolist()==ids,'Native ordinary call/raw argmax inventory differs')
        need(all(bool((inputs[i]['input_ids']==ids[i-1]).all()) for i in range(1,t)),'Actual cached joint prefix differs from emitted history')
        completed=ids[-1] in policy['native_eos_token_ids'];need(not any(v in policy['native_eos_token_ids'] for v in ids[:-1]) and (completed or t==4),'Unexpected joint EOS stopping')
    need(versions=={k:v._version for k,v in model.named_parameters()} and all(not v.requires_grad and v.grad is None for v in model.parameters())
         and getattr(model.model,'rope_deltas',None) is rope_before,'Joint generation changed frozen weights/rope ownership')
    return dict(generated_ids=ids,raw_logits=raw,logit_records=recorder.records,text=processor.tokenizer.decode(ids,skip_special_tokens=True),
        raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False),completed=completed,truncated=not completed,finish_reason='eos' if completed else 'length',
        metadata=dict(m,generation=policy,layout=layout['metadata'],native_identity_sha256=native_identity_sha256,
            scene_input_identity=object_sha(dict(metadata=m,native_identity_sha256=native_identity_sha256)),generation_position_ids=[tensor_info(v) for v in positions]),
        counters=dict(counts,broadcast=0,fusion=0,conditioning=0,probe_head=0),native_inputs=inputs,native_positions=positions,shapes=shapes,
        profile_head=heads if capture_head else None,parameter_versions_unchanged=True,hooks_removed=True,rope_restored=True)


def teacher_bundle(torch,bundle,target_ids,rope):
    validate_joint(torch,bundle);need(2<=len(target_ids)<=4 and target_ids[-1]==151645 and 151645 not in target_ids[:-1],'Full name plus single EOS required')
    prefix=torch.tensor([target_ids[:-1]],dtype=torch.long);inputs=dict(bundle['inputs'])
    inputs['input_ids']=torch.cat((inputs['input_ids'],prefix),dim=1)
    inputs['attention_mask']=torch.ones_like(inputs['input_ids']);width=inputs['input_ids'].shape[1]
    meta=dict(bundle['metadata'],prompt_width=width,prefix_ids=list(target_ids[:-1]),
        input_identity={k:tensor_info(v) for k,v in inputs.items()})
    tf=dict(inputs=inputs,row_inputs=[inputs],metadata=meta);layout=native.audit_layout(rope,tf)
    positions=torch.cat((torch.arange(width).view(1,1,width),layout['position_ids']),dim=0)
    return dict(inputs=inputs,position_ids=positions,layout=layout,original_prompt_width=bundle['metadata']['prompt_width'],
        target_ids=list(target_ids),logits_to_keep=len(target_ids),target_positions=list(range(width-len(target_ids),width)),
        input_identity={k:tensor_info(v) for k,v in inputs.items()},position_identity=tensor_info(positions))


def validate_teacher(torch,bundle,tf,row):
    validate_joint(torch,bundle);L=len(row['target_ids']);W=bundle['metadata']['prompt_width'];S=W+L-1
    need(tf['target_ids']==row['target_ids'] and tf['original_prompt_width']==W and tf['logits_to_keep']==L
         and tf['target_positions']==list(range(W-1,S)) and tf['inputs']['input_ids'].shape==(1,S)
         and torch.equal(tf['inputs']['input_ids'][:,:W],bundle['inputs']['input_ids'])
         and tf['inputs']['input_ids'][0,W:].tolist()==row['target_ids'][:-1]
         and bool((tf['inputs']['attention_mask']==1).all()) and tf['position_ids'].shape==(4,1,S)
         and torch.equal(tf['inputs']['pixel_values'],bundle['inputs']['pixel_values'])
         and torch.equal(tf['inputs']['image_grid_thw'],bundle['inputs']['image_grid_thw'])
         and tf['input_identity']=={k:tensor_info(v) for k,v in tf['inputs'].items()}
         and tf['position_identity']==tensor_info(tf['position_ids']),'Full native teacher-forcing layout changed')


def epoch_order(rows):
    rng=random.Random(24);result=[]
    for epoch in range(12):
        indices=list(range(len(rows)));rng.shuffle(indices)
        result.extend(dict(epoch=epoch+1,position=j,index=i,sid=rows[i]['sid'],target_length=len(rows[i]['target_ids'])) for j,i in enumerate(indices))
    return result


def select_cases(rows,prepared):
    sandra=[r for r in rows if r['gold']=='Sandra'];base=next(r['base_pair_id'] for r in sandra if r['orientation_version']=='original')
    sentinels=[]
    for orientation in ('original','flipped'):
        for n in (8,16):
            selected=[r for r in sandra if r['base_pair_id']==base and r['orientation_version']==orientation and r['n_frames']==n]
            need(len(selected)==1,'Exactly one Sandra sentinel per orientation/length required');sentinels.append(selected[0]['sid'])
    training=[]
    for n in (8,16):
        choices=[r for r in rows if r['n_frames']==n]
        width=max(prepared[r['sid']]['teacher_width'] for r in choices)
        training.append(next(r['sid'] for r in choices if prepared[r['sid']]['teacher_width']==width))
    return dict(parity_sids=sentinels,training_sids=training,microbatch_sids=training*4,roundtrip_sids=training)


def lora_config(targets=TARGETS):
    from peft import LoraConfig
    return LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=list(targets),bias='none',
        init_lora_weights=True,use_rslora=False,use_dora=False,modules_to_save=None)


def installed(bindings):
    import peft
    names=('peft.mapping','peft.tuners.tuners_utils','peft.tuners.lora.config','peft.tuners.lora.model',
        'peft.tuners.lora.layer','peft.tuners.lora.bnb','bitsandbytes.nn.modules','torch.profiler.profiler','torch.optim.adamw')
    result={}
    for name in names:
        module=importlib.import_module(name);file=Path(inspect.getfile(module)).resolve();result[name]=dict(file=str(file),sha256=bind(file,bindings))
    return dict(peft_version=peft.__version__,files=result)


def cpu_fixtures(torch):
    from peft import inject_adapter_in_model
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.q_proj=torch.nn.Linear(8,8,bias=False)
        def forward(self,x):return self.q_proj(x)
    with torch.random.fork_rng():
        torch.manual_seed(24);model=Tiny().eval().requires_grad_(False);x=torch.arange(16,dtype=torch.float32).reshape(2,8)/16
        original=model(x).detach();parameter=model.q_proj.weight;dtype=parameter.dtype;pointer=parameter.data_ptr()
        model=inject_adapter_in_model(lora_config(['q_proj']),model);model.eval()
        need(model.q_proj.base_layer.weight is parameter and parameter.dtype==dtype and parameter.data_ptr()==pointer
             and torch.equal(model(x),original),'PEFT zero adapter/base storage fixture failed')
        adapters={n:p for n,p in model.named_parameters() if '.lora_' in n};need(len(adapters)==2,'Tiny exact A/B scope differs')
        opt=torch.optim.AdamW(adapters.values(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        model.train();loss=model(x).square().mean();loss.backward()
        need(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapters.values())
             and all(torch.count_nonzero(p.grad)==0 for n,p in adapters.items() if '.lora_A.' in n)
             and sum(p.grad.abs().sum() for n,p in adapters.items() if '.lora_B.' in n)>0,'Zero-B first-gradient fixture failed')
        opt.step();opt.zero_grad(set_to_none=True);model(x).square().mean().backward()
        need(sum(p.grad.abs().sum() for n,p in adapters.items() if '.lora_A.' in n)>0 and parameter.grad is None,'Second-step A/base gradient fixture failed')
    rows=[dict(sid=str(i),target_ids=[i,151645]) for i in range(216)];order=epoch_order(rows)
    need(len(order)==2592 and len({r['sid'] for r in order[:216]})==216 and all(Counter(r['sid'] for r in order[e*216:(e+1)*216])==Counter(r['sid'] for r in rows) for e in range(12)),'Twelve complete shuffled epochs required')
    return dict(passed=True,groups=['ordinary_peft_zero_parity','base_storage_identity','zero_B_then_live_A_gradients','twelve_epoch_order'],native_model_calls=0)


def check(args,out,frozen):
    import torch
    from transformers import AutoProcessor,__version__ as transformers_version
    torch.set_num_threads(4);bindings={};bind(PARENT,bindings,PARENT_SHA);parent=read(PARENT);proof=read(PARENT.parent/'summary.json')
    need(proof['passed'] is proof['completed'] is True and proof['plan_sha256']==PARENT_SHA,'Fixed passed216 CPU parent required')
    bind(PARENT.parent/'summary.json',bindings);bind(parent['rows_file'],bindings,parent['runtime_bindings'][parent['rows_file']]);rows=read(parent['rows_file'])
    need(len(rows)==216 and len({r['sid'] for r in rows})==216 and Counter(r['n_frames'] for r in rows)=={8:108,16:108}
         and Counter(r['orientation_version'] for r in rows)=={'original':108,'flipped':108},'Exact216 orientation training rows required')
    stage_ref=parent['training_stage'];stage_plan=read(stage_ref['plan_file'])
    for key in ('file','plan_file','independent_audit_file'):bind(stage_ref[key],bindings,stage_ref['sha256' if key=='file' else key.replace('_file','_sha256')])
    bind(stage_plan['rows_file'],bindings,stage_plan['files'][stage_plan['rows_file']])
    need([{k:v for k,v in r.items() if k not in ('first_token_id','question_index')} for r in rows]==read(stage_plan['rows_file']),'Original published training ownership differs')
    confirmation=parent['confirmation_report']
    for key in ('file','manifest_file'):bind(confirmation[key],bindings,confirmation['sha256' if key=='file' else 'manifest_sha256'])
    uniform=parent['uniform_plan'];bind(uniform['file'],bindings,uniform['sha256']);native_plan=read(uniform['file'])['native_software_plan']
    bind(native_plan['file'],bindings,native_plan['sha256']);software=read(native_plan['file']);identity=parent['native_identity']
    need(software['native_identity']==identity,'Original native model identity differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,rope,api=backend.native_api(processor)
    need(api==identity['native_api'] and fingerprint(processor,str(transformers_version))==identity['processor']
         and backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Bound processor/model/API changed')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);prepared={}
    for i,row in enumerate(rows):
        need(row['split']=='train' and processor.tokenizer.encode(row['gold'],add_special_tokens=False)+[151645]==row['target_ids'],'Canonical name/EOS target changed')
        bind(Path(row['path'])/'qa.txt',bindings,row['qa_sha256'])
        for image in row['image_files']:bind(image['path'],bindings,image['sha256'])
        conversation=[dict(role='user',content=[dict(type='image') for _ in row['image_files']]+[dict(type='text',text=row['question'])])]
        prompt_text=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
        full_text=processor.apply_chat_template(conversation+[dict(role='assistant',content=[dict(type='text',text=row['gold'])])],add_generation_prompt=False,tokenize=False)
        text_prompt=processor.tokenizer.encode(prompt_text,add_special_tokens=False);full_ids=processor.tokenizer.encode(full_text,add_special_tokens=False)
        need(full_ids[:len(text_prompt)+len(row['target_ids'])]==text_prompt+row['target_ids'],'Actual assistant boundary changes canonical target tokens')
        bundle=prepare_joint(processor,model_view(row));tf=teacher_bundle(torch,bundle,row['target_ids'],lambda **kw:rope(owner,**kw));validate_teacher(torch,bundle,tf,row)
        file=data/f'bundle_{i:03d}.pt';torch.save(dict(bundle=bundle,teacher=tf),file)
        prepared[row['sid']]=dict(file=str(file),sha256=sha(file),metadata=bundle['metadata'],teacher_width=tf['inputs']['input_ids'].shape[1],
            teacher_position_identity=tf['position_identity'],teacher_input_identity=tf['input_identity'],target_ids=row['target_ids'],head_rows=len(row['target_ids']))
    order=epoch_order(rows);cases=select_cases(rows,prepared);by_sid={r['sid']:r for r in rows}
    head_rows=sum(len(by_sid[s]['target_ids']) for s in cases['training_sids'])
    counts=dict(training=sum(v['target_length'] for v in order),by_update=[sum(v['target_length'] for v in order[i:i+8]) for i in range(0,2592,8)],
        profile_training=8*head_rows,profile_roundtrip=2*head_rows,profile_natural_cap=32,profile_total_cap=10*head_rows+32)
    tests=cpu_fixtures(torch);packages=installed(bindings)
    for name,value in (('rows.json',rows),('prepared.json',prepared),('order.json',order),('profile_cases.json',cases),('head_rows.json',counts),('selftests.json',tests)):save(out/name,value)
    runtime={}
    for name in ('rows.json','prepared.json','order.json','profile_cases.json','head_rows.json'):bind(out/name,runtime)
    for item in prepared.values():bind(item['file'],runtime,item['sha256'])
    plan=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),
        orientation_plan=dict(file=str(PARENT),sha256=PARENT_SHA),training_stage=stage_ref,confirmation_report=confirmation,
        confirmation_reserved_seed=91726342,confirmation_predictions_accessed=False,native_identity=identity,native_identity_sha256=parent['native_identity_sha256'],
        native_software_plan=native_plan,precision=software['precision'],packages=packages,rows_file=str(out/'rows.json'),prepared_file=str(out/'prepared.json'),
        order_file=str(out/'order.json'),order_sha256=sha(out/'order.json'),order_object_sha256=object_sha(order),profile_cases=cases,head_rows=counts,
        lora_targets=list(TARGETS),input_bindings=bindings,runtime_bindings=runtime,tests=tests,no_native_model_or_head_forward=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),contexts=216,tests=tests,profile_cases=cases,head_rows=counts,no_native_model_or_head_forward=True)


def verify_plan(path,streaming=False):
    path=Path(path).resolve();plan=read(path);proof=read(path.parent/'summary.json')
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources()
         and plan['inherited_source_sha256']==inherited_sources() and sha(path)==path.with_suffix('.sha256').read_text().strip()
         and proof['passed'] is proof['completed'] is True and proof['phase']=='check' and proof['plan_sha256']==sha(path), 'Passed CPU preparation required')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Source snapshot changed')
    need(read(path.parent/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'Inherited map changed')
    need(sha(plan['prepared_file'])==plan['runtime_bindings'][plan['prepared_file']],'Prepared index changed')
    prepared=read(plan['prepared_file']);files={v['file'] for v in prepared.values()}
    need(len(prepared)==len(files)==216 and all(plan['runtime_bindings'][v['file']]==v['sha256'] for v in prepared.values()),'Complete216 bundle map differs')
    for file,digest in plan['runtime_bindings'].items():
        if streaming and file in files:continue
        need(sha(file)==digest,'Consumed runtime input changed')
    for file,digest in plan['input_bindings'].items():need(sha(file)==digest,'Bound preparation input changed')
    need(object_sha(read(plan['order_file']))==plan['order_object_sha256'] and read(plan['order_file'])==epoch_order(read(plan['rows_file'])),'Frozen complete epoch order changed')
    return plan


def base_metadata(refs):
    return {name:dict(shape=list(p.shape),dtype=str(p.dtype),version=p._version,storage=p.data_ptr(),object_id=id(p)) for name,p in refs}


def check_base(refs,expected,model):
    live={id(p) for name,p in model.named_parameters() if '.lora_A.default.weight' not in name and '.lora_B.default.weight' not in name}
    need(live=={id(p) for _,p in refs},'Actual live base Parameter objects were replaced')
    need(base_metadata(refs)==expected and all(not p.requires_grad and p.grad is None for _,p in refs),'Original base weights/storage/dtypes/gradients changed')


def adapter_parameters(model):return {name:p for name,p in model.named_parameters() if '.lora_A.default.weight' in name or '.lora_B.default.weight' in name}


def canonical_peft_config(model):
    def convert(value):
        if isinstance(value,dict):return {key:convert(item) for key,item in sorted(value.items())}
        if isinstance(value,set):return sorted(convert(item) for item in value)
        if isinstance(value,(list,tuple)):return [convert(item) for item in value]
        return value
    return json.loads(json.dumps(convert(model.peft_config['default'].to_dict()),sort_keys=True))


def adapter_contract(torch,model):
    from peft.tuners.lora.bnb import Linear4bit
    modules=dict(model.named_modules());wrapped={name:m for name,m in modules.items() if isinstance(m,Linear4bit)}
    need(set(wrapped)==set(TARGETS),'Only exact28-language-layer q/k/v/o modules may be adapted')
    for name,m in wrapped.items():
        need(m.r=={'default':16} and m.lora_alpha=={'default':32} and m.scaling=={'default':2.}
             and m.active_adapters==['default'] and not m.disable_adapters and not m.merged
             and set(m.lora_A)==set(m.lora_B)==set(m.lora_dropout)=={'default'}
             and isinstance(m.lora_dropout['default'],torch.nn.Dropout) and m.lora_dropout['default'].p==.05,
             'Exact active default LoRA configuration differs: '+name)
    params=adapter_parameters(model)
    expected={target+suffix for target in TARGETS for suffix in ('.lora_A.default.weight','.lora_B.default.weight')}
    need(set(params)==expected and len(params)==224 and sum(p.numel() for p in params.values())==10092544
         and all(p.dtype==torch.float32 and p.is_cuda for p in params.values()),'Exact FP32 language adapter parameter budget differs')
    need(all(p.requires_grad==(name in params) for name,p in model.named_parameters()),'Only exact adapter tensors may train')
    cfg=model.peft_config['default'].to_dict()
    need(cfg['r']==16 and cfg['lora_alpha']==32 and cfg['lora_dropout']==.05 and cfg['bias']=='none'
         and cfg['init_lora_weights'] is True and not cfg['use_rslora'] and not cfg['use_dora']
         and cfg['modules_to_save'] is None and set(cfg['target_modules'])==set(TARGETS),'Saved PEFT configuration differs')
    return dict(targets=list(TARGETS),trainable_parameters=10092544,tensors={name:dict(shape=list(p.shape),dtype=str(p.dtype),object_id=id(p)) for name,p in params.items()},
        active_adapter='default',scaling=2.,rank=16,alpha=32,dropout=.05)


def install_lora(torch,model):
    from bitsandbytes.nn import Linear4bit
    from peft import inject_adapter_in_model
    modules=dict(model.named_modules())
    for name in TARGETS:
        m=modules[name];need(isinstance(m,Linear4bit) and m.in_features==3584
            and m.out_features==(512 if name.endswith(('k_proj','v_proj')) else 3584),'Expected native language projection differs')
    cls=type(model);generate=model.generate.__func__;torch.manual_seed(24);torch.cuda.manual_seed_all(24)
    result=inject_adapter_in_model(lora_config(),model)
    need(result is model and type(model) is cls and model.generate.__func__ is generate,'PEFT changed the native model/generate API')
    for name,p in model.named_parameters():
        p.requires_grad_('.lora_A.default.weight' in name or '.lora_B.default.weight' in name)
        if p.requires_grad:p.data=p.data.to(torch.float32)
    model.gradient_checkpointing_disable()
    need(not model.is_gradient_checkpointing and all(torch.count_nonzero(p)==0 for name,p in adapter_parameters(model).items() if '.lora_B.' in name),'Zero-B/checkpointing-off initialization differs')
    return adapter_contract(torch,model)


def teacher_forward(torch,model,packet,row,backward=False,evidence=None):
    from gnnformer.runtime import move_to_device
    bundle,tf=packet['bundle'],packet['teacher'];validate_teacher(torch,bundle,tf,row);L=len(row['target_ids']);S=tf['inputs']['input_ids'].shape[1]
    evidence={} if evidence is None else evidence;counts=dict.fromkeys(('model','visual','language','norm','head'),0)
    captured={'decoder_layer_forward_calls':[0]*28};evidence.update(capture=captured,counters=counts,backward_started=False,backward_completed=False);rope_before=model.model.rope_deltas
    def top(*_):counts['model']+=1
    def vision(*_):counts['visual']+=1
    def language(module,args,kwargs):
        counts['language']+=1;positions=kwargs['position_ids'].detach().cpu()
        captured['position_ids']=positions.clone();need(torch.equal(positions,tf['position_ids']),'Actual native teacher positions changed')
    def before_norm(module,args):
        counts['norm']+=1;captured['norm_query_input']=args[0][:,-L:,:].detach().cpu().clone();captured['norm_input_shape']=list(args[0].shape)
        need(args[0].shape==(1,S,3584),'Full native teacher normalization shape differs')
    def after_norm(module,args,output):captured['normalized_query']=output[:,-L:,:].detach().cpu().clone()
    def head(module,args,output):
        counts['head']+=1;captured['head_input']=args[0].detach().cpu().clone();captured['head_logits']=output.detach().cpu().clone()
        need(args[0].shape==(1,L,3584) and output.shape==(1,L,152064)
             and args[0].dtype==output.dtype==torch.float16,'Actual native selected head shape/dtype differs')
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope_before)
        for handle in (model.register_forward_pre_hook(top),model.model.visual.register_forward_pre_hook(vision),
            model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),model.model.language_model.norm.register_forward_pre_hook(before_norm),
            model.model.language_model.norm.register_forward_hook(after_norm),model.lm_head.register_forward_hook(head)):stack.callback(handle.remove)
        for index,layer in enumerate(model.model.language_model.layers):
            def observe_layer(module,args,index=index):captured['decoder_layer_forward_calls'][index]+=1
            stack.callback(layer.register_forward_pre_hook(observe_layer).remove)
        device_inputs=move_to_device(tf['inputs'],model.device)
        kwargs=model.prepare_inputs_for_generation(**device_inputs,cache_position=torch.arange(S,device=model.device),use_cache=False,logits_to_keep=L)
        need(torch.equal(kwargs['position_ids'].detach().cpu(),tf['position_ids']) and kwargs.get('past_key_values') is None
             and kwargs.get('pixel_values') is not None and kwargs['use_cache'] is False and kwargs['logits_to_keep']==L,
             'Ordinary native full-prefix preparation differs')
        with torch.set_grad_enabled(backward):
            output=model(**kwargs);logits=output.logits
            need(logits.shape==(1,L,152064) and output.past_key_values is None,'One native full-prefix head only required')
            labels=torch.tensor(row['target_ids'],device=logits.device);position_ce=torch.nn.functional.cross_entropy(logits[0].float(),labels,reduction='none')
            loss=position_ce.mean();need(bool(torch.isfinite(loss)),'Nonfinite native teacher CE')
            if backward:
                evidence['backward_started']=True;(loss/8).backward();evidence['backward_completed']=True
        captured.update(sid=row['sid'],target_ids=row['target_ids'],target_positions=tf['target_positions'],teacher_input_identity=tf['input_identity'],
            loss=float(loss.detach()),position_ce=position_ce.detach().cpu().tolist(),backward=backward,backward_coefficient=1/8 if backward else 0.)
    need(captured['decoder_layer_forward_calls']==[1]*28 and not model.is_gradient_checkpointing,'Unexpected decoder recomputation/checkpointing')
    need(dict(counts)==dict(model=1,visual=1,language=1,norm=1,head=1)
         and torch.equal(captured['normalized_query'],captured['head_input']) and model.model.rope_deltas is rope_before,'Teacher call/normalization/cleanup identity differs')
    return captured,dict(counts)


def retained_teacher(torch,model,packet,row,failure_file,backward=False,profile_kernels=False):
    evidence={}
    try:
        if profile_kernels:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False,profile_memory=False,with_stack=False) as trace:
                result=teacher_forward(torch,model,packet,row,backward=backward,evidence=evidence)
            events=trace.events();inventory=dict(
                sdpa_operators=sorted({e.name for e in events if 'scaled_dot_product' in e.name}),
                cuda_kernels=sorted({e.name for e in events if e.device_type==torch.autograd.DeviceType.CUDA}),
                event_count=len(events),existing_training_call_only=True)
            evidence['kernel_inventory']=inventory;result[0]['kernel_inventory']=inventory
            need('aten::scaled_dot_product_attention' in inventory['sdpa_operators'] and bool(inventory['cuda_kernels']),
                 'Actual SDPA/CUDA kernel observations unavailable; profile held')
            return result
        return teacher_forward(torch,model,packet,row,backward=backward,evidence=evidence)
    except BaseException:
        torch.save(cpu_tree(torch,dict(sid=row['sid'],evidence=evidence,partial_outputs_retained=True)),failure_file)
        save(failure_file.with_suffix('.json'),dict(file=str(failure_file),sha256=sha(failure_file),sid=row['sid'],partial_outputs_retained=True));raise


def load_packet(torch,item,row):
    need(sha(item['file'])==item['sha256'],'Actual consumed image bundle changed');packet=torch.load(item['file'],map_location='cpu',weights_only=True)
    validate_teacher(torch,packet['bundle'],packet['teacher'],row)
    need(packet['bundle']['metadata']==item['metadata'] and packet['teacher']['position_identity']==item['teacher_position_identity'], 'Prepared teacher metadata changed')
    return packet


def project(setup,micro,optim,natural,checkpoint):
    T={n:max(r['seconds'] for r in micro if r['n_frames']==n) for n in (8,16)}
    G={n:max(T[n],max(r['four_token_seconds'] for r in natural if r['n_frames']==n)) for n in (8,16)}
    O=max(optim);C=checkpoint
    need(all(math.isfinite(v) and v>0 for v in [setup,O,C,*T.values(),*G.values()]),'All measured resource segments must be finite/positive')
    value=setup+1.25*(1296*T[8]+1296*T[16]+324*O+108*G[8]+108*G[16]+2*C)+60.
    return dict(passed=value<=3600,setup_seconds=setup,per_N_micro_seconds=T,optimizer_seconds=O,per_N_generation_seconds=G,
        checkpoint_roundtrip_seconds=C,projected_seconds=value,cap_seconds=3600,multiplier=1.25,reserve_seconds=60,
        empirical_projection_not_latency_guarantee=True,main_implemented=False,main_release=False)


def single_attempt_guard(out):
    need(os.environ.get('SLURM_JOB_NAME')==JOB,'Exact registered LoRA profile job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;(out/'launch_sacct.psv').write_text(raw)
    rows=[line.split('|') for line in raw.splitlines() if line.split('|')[1]==JOB]
    need(len(rows)==1 and rows[0][0]==os.environ['SLURM_JOB_ID'] and rows[0][2]=='gpu'
         and int(rows[0][5])<=240,'Exactly one profile attempt, including failed allocations, is allowed')
    save(out/'launch_accounting.json',dict(command=command,file=str(out/'launch_sacct.psv'),sha256=sha(out/'launch_sacct.psv'),matching_rows=rows))


def profile(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);single_attempt_guard(out);plan=verify_plan(args.plan,streaming=True)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','Exactly one B200 required')
    rows=read(plan['rows_file']);by_sid={r['sid']:r for r in rows};prepared=read(plan['prepared_file']);cases=plan['profile_cases']
    data=DATA/out.name;ckpt=CKPT/out.name;data.mkdir(parents=True,exist_ok=False);ckpt.mkdir(parents=True,exist_ok=False)
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=loaded.model.eval().requires_grad_(False)
    hardware=frozen_joint.live_identity(torch,loaded,plan);refs=list(model.named_parameters());base_before=base_metadata(refs)
    save(out/'base_before.json',base_before);live_packages=installed({});need(live_packages==plan['packages'],'Actual PEFT/bnb source changed')
    config=dict(protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(args.plan.resolve()),plan_sha256=sha(args.plan),hardware=hardware,cases=cases,
        orientation_plan=plan['orientation_plan'],training_stage=plan['training_stage'],confirmation_report=plan['confirmation_report'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],packages=live_packages,
        data_directory=str(data),checkpoint_directory=str(ckpt),profile_only=True,main_release=False,confirmation_predictions_accessed=False)
    save(out/'config.json',config);counts=Counter();records=[];micro=[];optim=[];raw={};memory=[];setup=time.perf_counter()-started
    for state in ('base','zero_lora'):
        if state=='zero_lora':
            tick=time.perf_counter();contract=install_lora(torch,model);params=adapter_parameters(model)
            check_base(refs,base_before,model);save(out/'adapter_contract.json',contract)
            initial={name:p.detach().cpu().clone() for name,p in params.items()};initial_file=ckpt/'initial_adapter.pt'
            peft_config=canonical_peft_config(model);peft_config_file=ckpt/'adapter_config.json';save(peft_config_file,peft_config)
            torch.save(dict(adapter=initial,contract=contract,peft_config=peft_config,seed=24,profile_initialization_only=True),initial_file)
            save(out/'initial_adapter.json',dict(file=str(initial_file),sha256=sha(initial_file),config_file=str(peft_config_file),config_sha256=sha(peft_config_file),tensors={k:tensor_info(v) for k,v in initial.items()}))
            model.eval().requires_grad_(False);torch.cuda.synchronize();setup+=time.perf_counter()-tick
        for index,sid in enumerate(cases['parity_sids']):
            row=by_sid[sid];tick=time.perf_counter();packet=load_packet(torch,prepared[sid],row);prep=time.perf_counter()-tick;work=time.perf_counter();evidence={}
            try:result=generate_joint(model,loaded.processor,packet['bundle'],native_identity_sha256=plan['native_identity_sha256'],capture_head=True,evidence=evidence)
            except BaseException:
                file=data/f'{state}_{index:02d}_partial.pt';torch.save(cpu_tree(torch,dict(sid=sid,evidence=evidence)),file);raise
            file=data/f'{state}_{index:02d}.pt';torch.save(dict(sid=sid,state=state,result=result),file);digest=sha(file)
            record=dict(sid=sid,state=state,n_frames=row['n_frames'],file=str(file),sha256=digest,generated_tokens=len(result['generated_ids']),
                preparation_seconds=prep,counters=result['counters'])
            counts.update(result['counters']);raw[(state,sid)]=result
            save(out/f'{state}_{index:02d}.json',record);check_base(refs,base_before,model);torch.cuda.synchronize()
            record['work_seconds']=time.perf_counter()-work;record['four_token_seconds']=prep+record['work_seconds']*4/record['generated_tokens'];records.append(record)
            save(out/f'{state}_{index:02d}_timing.json',record)
    setup_tick=time.perf_counter();parity=[]
    for sid in cases['parity_sids']:
        a=raw[('base',sid)];b=raw[('zero_lora',sid)];equal=torch.equal(a['raw_logits'],b['raw_logits'])
        same_ids=a['generated_ids']==b['generated_ids'];same_shapes=a['shapes']==b['shapes']
        tv=0. if equal else float((a['raw_logits'].double().softmax(-1)-b['raw_logits'].double().softmax(-1)).abs().sum(-1).max()/2) if a['raw_logits'].shape==b['raw_logits'].shape else None
        parity.append(dict(sid=sid,exact_logits=equal,exact_generated_ids=same_ids,exact_shapes=same_shapes,maximum_tv=tv))
    save(out/'zero_lora_parity.json',parity)
    need(all(r['exact_logits'] and r['exact_generated_ids'] and r['exact_shapes'] for r in parity),'Zero LoRA must preserve exact native logits and IDs; no tolerance fallback')
    del raw
    for name,p in model.named_parameters():p.requires_grad_(name in params)
    model.train();model.model.visual.eval();adapter_contract(torch,model)
    optimizer=torch.optim.AdamW(params.values(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();setup+=time.perf_counter()-setup_tick;training_records=[]
    for step in (1,2):
        for j,sid in enumerate(cases['microbatch_sids']):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();row=by_sid[sid]
            packet=load_packet(torch,prepared[sid],row);capture,calls=retained_teacher(torch,model,packet,row,data/f'training_{step}_{j+1:02d}_partial.pt',backward=True,profile_kernels=step==1 and j<2);counts.update(calls);counts['backward']+=1
            grads={name:p.grad for name,p in params.items()};need(all(g is not None for g in grads.values()),'Every intended LoRA tensor must participate in backward')
            finite=bool(torch.stack([g.isfinite().all() for g in grads.values()]).all())
            norms={letter:float(torch.sqrt(torch.stack([g.float().square().sum() for name,g in grads.items() if f'.lora_{letter}.' in name]).sum())) for letter in ('A','B')}
            capture.update(step=step,microbatch=j+1,n_frames=row['n_frames'],all_adapter_gradients_finite=finite,accumulated_gradient_norms=norms)
            file=data/f'training_{step}_{j+1:02d}.pt';torch.save(capture,file);digest=sha(file);check_base(refs,base_before,model)
            record=dict(step=step,microbatch=j+1,sid=sid,n_frames=row['n_frames'],file=str(file),sha256=digest,loss=capture['loss'],
                target_rows=len(row['target_ids']),all_gradients_finite=finite,accumulated_gradient_norms=norms)
            micro.append(record);training_records.append(record);memory.append(dict(step=step,microbatch=j+1,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved()))
            save(out/f'training_{step}_{j+1:02d}.json',record)
            need(finite and norms['B']>0 and (norms['A']==0 if step==1 else norms['A']>0),'Expected zero-B then live A/B gradients differ')
            torch.cuda.synchronize();record['seconds']=time.perf_counter()-tick
            save(out/f'training_{step}_{j+1:02d}_timing.json',record)
        tick=time.perf_counter();before={name:p._version for name,p in params.items()};grad_norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        need(bool(torch.isfinite(grad_norm)),'Nonfinite unclipped adapter gradient norm');optimizer.step();optimizer.zero_grad(set_to_none=True)
        counts['optimizer']+=1
        need(all(p._version>before[name] and bool(torch.isfinite(p).all()) for name,p in params.items()),'Optimizer must update only finite adapter tensors')
        check_base(refs,base_before,model);save(out/f'optimizer_{step}.json',dict(step=step,unclipped_gradient_norm=float(grad_norm),
            learning_rate=2e-4,gradient_accumulation=8,optimizer_steps=[float(optimizer.state[p]['step']) for p in params.values()],base_frozen=True))
        torch.cuda.synchronize();seconds=time.perf_counter()-tick;optim.append(seconds);save(out/f'optimizer_{step}_timing.json',dict(step=step,seconds=seconds))
    model.eval();before_eval=[];after_eval=[]
    for phase,destination in (('before',before_eval),('after',after_eval)):
        if phase=='after':
            tick=time.perf_counter();contract_before=adapter_contract(torch,model);selected={name:p.detach().cpu().clone() for name,p in params.items()}
            file=ckpt/'profile_after_two_updates.pt';torch.save(dict(adapter=selected,contract=contract_before,peft_config=canonical_peft_config(model),updates=2,profile_only=True),file);checkpoint_sha=sha(file)
            loaded_packet=torch.load(file,map_location='cpu',weights_only=True)
            need(loaded_packet['contract']==contract_before and loaded_packet['peft_config']==peft_config==canonical_peft_config(model)
                 and read(peft_config_file)==peft_config and set(loaded_packet['adapter'])==set(params),'Restricted adapter reload metadata differs')
            with torch.no_grad():
                for name,p in params.items():p.copy_(initial[name])
            original_reset={name:tensor_info(p)==tensor_info(initial[name]) for name,p in params.items()}
            need(all(original_reset.values()) and canonical_peft_config(model)==peft_config,'Exact original adapter/config reset failed')
            with torch.no_grad():
                for name,p in params.items():p.copy_(loaded_packet['adapter'][name])
            exact={name:tensor_info(p)==tensor_info(selected[name]) for name,p in params.items()}
            need(all(exact.values()) and adapter_contract(torch,model)==contract_before and canonical_peft_config(model)==peft_config,'Adapter parameter objects/configuration/bytes not restored')
            check_base(refs,base_before,model)
            save(out/'roundtrip.json',dict(passed=True,file=str(file),sha256=checkpoint_sha,tensors_exact=exact,original_reset_exact=original_reset,
                original_file=str(initial_file),original_sha256=sha(initial_file),contract_restored=True,
                config_file=str(peft_config_file),config_sha256=sha(peft_config_file),full_peft_config_exact=True,
                same_parameter_objects=True,base_frozen=True,eval_forwards_excluded=True))
            torch.cuda.synchronize();checkpoint_seconds=time.perf_counter()-tick;save(out/'roundtrip_timing.json',dict(seconds=checkpoint_seconds,eval_forwards_excluded=True))
        for index,sid in enumerate(cases['roundtrip_sids']):
            tick=time.perf_counter();packet=load_packet(torch,prepared[sid],by_sid[sid]);capture,calls=retained_teacher(torch,model,packet,by_sid[sid],data/f'roundtrip_{phase}_{index}_partial.pt');counts.update(calls)
            file=data/f'roundtrip_{phase}_{index}.pt';torch.save(capture,file);digest=sha(file);torch.cuda.synchronize()
            destination.append(capture['head_logits']);save(out/f'roundtrip_{phase}_{index}.json',dict(sid=sid,file=str(file),sha256=digest,seconds=time.perf_counter()-tick))
    need(len(before_eval)==len(after_eval)==2 and all(torch.equal(a,b) for a,b in zip(before_eval,after_eval)), 'Serialized adapter must restore exact native full-target logits')
    expected_natural=sum(r['generated_tokens'] for r in records)
    need(counts['model']==counts['language']==counts['norm']==counts['head']==expected_natural+20<=52
         and counts['visual']==28 and counts['backward']==16 and counts['optimizer']==2
         and all(counts[k]==0 for k in ('broadcast','fusion','conditioning','probe_head')), 'Fixed full native profile call inventory differs')
    expected_rows=plan['head_rows']['profile_training']+plan['head_rows']['profile_roundtrip']+expected_natural
    actual_rows=sum(r['target_rows'] for r in training_records)+2*sum(len(by_sid[s]['target_ids']) for s in cases['roundtrip_sids'])+expected_natural
    need(actual_rows==expected_rows<=plan['head_rows']['profile_total_cap'],'Profile native head-row inventory differs')
    check_base(refs,base_before,model);contract=adapter_contract(torch,model);need(all(p.grad is None for p in params.values()),'Profile must clear accumulated gradients')
    forecast=project(setup,micro,optim,records,checkpoint_seconds)
    save(out/'profile_measurements.json',dict(setup_seconds=setup,natural=records,microbatches=micro,optimizer_seconds=optim,
        checkpoint_seconds=checkpoint_seconds,memory=memory,projection=forecast,counters=dict(counts),head_rows=actual_rows))
    files={str(p):sha(p) for directory in (data,ckpt) for p in sorted(directory.rglob('*')) if p.is_file()};save(out/'artifacts.json',files)
    endpoint=dict(passed=True,zero_lora_exact=True,roundtrip_logits_exact=True,base_parameters_unchanged=True,
        base_before_file=str(out/'base_before.json'),base_before_sha256=sha(out/'base_before.json'),adapter_contract=contract,counters=dict(counts),head_rows=actual_rows)
    save(out/'endpoint.json',endpoint)
    return dict(**config,passed=True,completed=True,phase='profile',endpoint_file=str(out/'endpoint.json'),endpoint_sha256=sha(out/'endpoint.json'),
        measurements_file=str(out/'profile_measurements.json'),measurements_sha256=sha(out/'profile_measurements.json'),
        artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),projection=forecast,main_implementation_eligible=forecast['passed'],
        profile_checkpoint_must_not_initialize_main=True,future_main_restarts_seed24=True,no_main_or_validation_execution=True,counters=dict(counts),head_rows=actual_rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--profile',action='store_true');parser.add_argument('--plan',type=Path);args=parser.parse_args()
    native.require_slurm(gpu=args.profile)
    need((args.plan is None) if args.check else (args.plan is not None),'CPU check takes no plan; profile requires frozen plan')
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU preparation requires CPU partition')
    phase='check' if args.check else 'profile';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();frozen=snapshot(out);save(out/'request.json',dict(phase=phase,plan=None if args.plan is None else str(args.plan.resolve()),source_sha256=frozen))
    try:
        value=check(args,out,frozen) if args.check else profile(args,out,frozen,started)
        need(sources()==frozen and inherited_sources()==value['inherited_source_sha256'],'Sources changed during bounded profile')
        elapsed=time.perf_counter()-started;need(elapsed<=POLICY['cpu_check_seconds' if args.check else 'profile_seconds'],'Fixed phase budget exceeded')
        save(out/'summary.json',dict(value,elapsed_seconds=elapsed));print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()
