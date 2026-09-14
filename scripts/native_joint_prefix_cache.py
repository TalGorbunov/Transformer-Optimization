"""DRAFT ordinary native image-prefix cache and independent question branches.

No compressor, fitting, labels or question-conditioned memory construction.
All tensor/image/model work is Slurm-only. The explicit-position branch API also
separates a future prepared continuous prefix from question decoding.
"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from scripts import diagnose_native_identity_join_joint_lora_v2 as p
from scripts.stage_native_vision_v6_teacher import need,sha,object_sha


@dataclass(frozen=True)
class PrefixSnapshot:
    layers: tuple
    prefix_ids: object
    position_ids: object
    rope_deltas: object
    identity: dict
    native_identity_sha256: str

    @property
    def length(self):return self.prefix_ids.shape[1]


def prepare_image_prefix(processor,image_files,rope):
    """Only ordered image descriptors enter this constructor; no question arg."""
    p.native.require_slurm()
    import torch
    from PIL import Image
    frames=[];side=processor.tokenizer.padding_side
    try:
        for item in image_files:
            need(sha(item['path'])==item['sha256'],'Canonical image changed')
            with Image.open(item['path']) as frame:
                need(frame.mode=='RGB' and frame.size==(512,512),'Canonical image geometry differs');frames.append(frame.resize((392,392)))
        processor.tokenizer.padding_side='left'
        inputs=dict(processor.apply_chat_template([dict(role='user',content=[dict(type='image',image=frame) for frame in frames])],
            add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt'))
        need(set(inputs)=={'input_ids','attention_mask','pixel_values','image_grid_thw'} and inputs['input_ids'].shape[0]==1,'Ordinary image-only processor inputs required')
        vision_end=processor.tokenizer.convert_tokens_to_ids('<|vision_end|>');indices=(inputs['input_ids'][0]==vision_end).nonzero(as_tuple=True)[0]
        need(len(indices)==len(image_files) and len(indices)>0,'Exactly one native vision_end per image required')
        length=int(indices[-1])+1;inputs['input_ids']=inputs['input_ids'][:,:length].clone();inputs['attention_mask']=inputs['attention_mask'][:,:length].clone()
        positions,delta=rope(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],image_grid_thw=inputs['image_grid_thw'])
        positions=torch.cat((torch.arange(length).view(1,1,length),positions),dim=0)
        need(bool((inputs['attention_mask']==1).all()) and positions.shape==(4,1,length),'Unpadded native image prefix required')
        metadata=dict(length=length,n_frames=len(image_files),image_sha256=[i['sha256'] for i in image_files],image_paths=[i['path'] for i in image_files],
            last_token_id=vision_end,input_identity={k:p.tensor_info(v) for k,v in inputs.items()},position_identity=p.tensor_info(positions),
            rope_deltas=delta.tolist(),question_input_absent=True,prefix_ends_last_vision_end=True)
        return dict(inputs=inputs,position_ids=positions,rope_deltas=delta,metadata=metadata)
    finally:
        processor.tokenizer.padding_side=side
        for frame in frames:frame.close()


def question_suffix(torch,prefix,bundle,rope):
    """Validation sees a question; it cannot modify the image-only packet."""
    p.validate_joint(torch,bundle);layout=p.native.audit_layout(rope,bundle);L=prefix['metadata']['length'];W=bundle['metadata']['prompt_width']
    positions=torch.cat((torch.arange(W).view(1,1,W),layout['position_ids']),dim=0)
    need(W>L and torch.equal(bundle['inputs']['input_ids'][:,:L],prefix['inputs']['input_ids'])
         and torch.equal(positions[:,:,:L],prefix['position_ids']) and torch.equal(layout['rope_deltas'],prefix['rope_deltas'])
         and all(torch.equal(bundle['inputs'][key],prefix['inputs'][key]) for key in ('pixel_values','image_grid_thw')),
         'Every ordinary full prompt must have exactly the same image tokens/pixels/mRoPE prefix')
    return dict(input_ids=bundle['inputs']['input_ids'][:,L:].clone(),position_ids=positions[:,:,L:].clone(),
        full_prompt_width=W,prefix_length=L,full_metadata=bundle['metadata'],full_layout=layout['metadata'])


def _identity(layers,prefix_ids,positions,delta):
    return dict(layers=[dict(key=p.tensor_info(k),value=p.tensor_info(v)) for k,v in layers],prefix_ids=p.tensor_info(prefix_ids),
        position_ids=p.tensor_info(positions),rope_deltas=p.tensor_info(delta))


def verify_snapshot(snapshot):
    p.native.require_slurm()
    need(_identity(snapshot.layers,snapshot.prefix_ids,snapshot.position_ids,snapshot.rope_deltas)==snapshot.identity,'Stored image-prefix bytes changed')


def export_snapshot(torch,snapshot):
    verify_snapshot(snapshot)
    return dict(layers=[dict(key=k.detach().cpu().clone(),value=v.detach().cpu().clone()) for k,v in snapshot.layers],
        prefix_ids=snapshot.prefix_ids.detach().cpu().clone(),position_ids=snapshot.position_ids.detach().cpu().clone(),
        rope_deltas=snapshot.rope_deltas.detach().cpu().clone(),identity=snapshot.identity,native_identity_sha256=snapshot.native_identity_sha256)


def prefill_images(model,packet,*,native_identity_sha256,evidence=None):
    p.native.require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import move_to_device
    from transformers.cache_utils import DynamicCache,DynamicLayer
    evidence={} if evidence is None else evidence;counts=dict(model=0,visual=0,language=0,norm=0,head=0,prefix_decoder=0)
    evidence.update(counters=counts,input_identity=packet['metadata']['input_identity'],position_ids=packet['position_ids'])
    need(not model.training and all(not v.requires_grad and v.grad is None for v in model.parameters()),'Frozen eval model required')
    versions={n:v._version for n,v in model.named_parameters()};rope_before=model.model.rope_deltas;length=packet['metadata']['length'];observed=[]
    def bump(key):
        def hook(*_):counts[key]+=1
        return hook
    def language(module,args,kw):
        counts['language']+=1;observed.append(kw['position_ids'].detach().cpu().clone())
        need(torch.equal(observed[-1],packet['position_ids']),'Actual image-prefix mRoPE changed')
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope_before)
        for h in (model.register_forward_pre_hook(bump('model')),model.model.register_forward_pre_hook(bump('prefix_decoder')),
            model.model.visual.register_forward_pre_hook(bump('visual')),model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),
            model.model.language_model.norm.register_forward_pre_hook(bump('norm')),model.lm_head.register_forward_pre_hook(bump('head'))):stack.callback(h.remove)
        with torch.inference_mode():
            output=model.model(**move_to_device(packet['inputs'],model.device),position_ids=packet['position_ids'].to(model.device),
                cache_position=torch.arange(length,device=model.device),use_cache=True,return_dict=True)
        cache=output.past_key_values
        need(type(cache) is DynamicCache and len(cache.layers)==28 and all(type(layer) is DynamicLayer for layer in cache.layers),
            'Exact full-attention28-layer native cache required; no sliding/offloaded cache')
        layers=tuple((layer.keys.detach().clone(),layer.values.detach().clone()) for layer in cache.layers)
        evidence['layers']=layers;evidence['native_position_ids']=observed
        need(counts==dict(model=0,visual=1,language=1,norm=1,head=0,prefix_decoder=1)
             and all(k.shape==v.shape==(1,4,length,128) and k.dtype==v.dtype==torch.float16 for k,v in layers),
             'One headless image prefill and native FP16 K/V geometry required')
    need(versions=={n:v._version for n,v in model.named_parameters()} and model.model.rope_deltas is rope_before,'Prefix prefill changed frozen model/rope state')
    identity=_identity(layers,packet['inputs']['input_ids'],packet['position_ids'],packet['rope_deltas'])
    snapshot=PrefixSnapshot(layers,packet['inputs']['input_ids'].clone(),packet['position_ids'].clone(),packet['rope_deltas'].clone(),identity,native_identity_sha256)
    return snapshot,dict(counters=counts,native_position_ids=observed,identity=identity,rope_restored=True,parameters_unchanged=True)


def fork_cache(model,snapshot):
    p.native.require_slurm()
    import torch
    from transformers.cache_utils import DynamicCache
    verify_snapshot(snapshot)
    cache=DynamicCache(ddp_cache_data=[(k.clone(),v.clone()) for k,v in snapshot.layers],config=model.config)
    need(len(cache.layers)==28 and cache.get_seq_length()==snapshot.length,'Independent full prefix cache required')
    for layer,(k,v) in zip(cache.layers,snapshot.layers):
        need(torch.equal(layer.keys,k) and torch.equal(layer.values,v) and layer.keys.data_ptr()!=k.data_ptr() and layer.values.data_ptr()!=v.data_ptr(),
             'Question fork must own exact independent K/V storage')
    return cache


def generate_branch(model,processor,snapshot,suffix,*,evidence=None):
    p.native.require_slurm(gpu=True)
    import torch
    from scripts.native_vision_reasoning_stream import GlobalLogitRecorder
    evidence={} if evidence is None else evidence;cache=fork_cache(model,snapshot);L=snapshot.length;Q=suffix['input_ids'].shape[1];W=L+Q
    need(not model.training and all(not v.requires_grad and v.grad is None for v in model.parameters()) and suffix['prefix_length']==L
         and suffix['full_prompt_width']==W and suffix['position_ids'].shape==(4,1,Q),'Frozen independent native question branch required')
    policy=p.generation_policy(model,processor.tokenizer)[1];versions={n:v._version for n,v in model.named_parameters()};rope_before=model.model.rope_deltas
    counts=dict(model=0,visual=0,language=0,norm=0,head=0);seen=[];positions=[];shapes=[];heads=[];latest={};ids=[]
    recorder=GlobalLogitRecorder(max_steps=4,full_vectors=True)
    evidence.update(counters=counts,native_inputs=seen,native_positions=positions,shapes=shapes,profile_head=heads,generated_ids=ids,
        raw_vectors=recorder.vectors,logit_records=recorder.records,prefix_identity=snapshot.identity)
    def top(module,args,kw):
        counts['model']+=1;seen.append(dict(input_ids=kw['input_ids'].detach().cpu().clone(),attention_mask=kw['attention_mask'].detach().cpu().clone(),
            cache_position=kw['cache_position'].detach().cpu().clone(),past_length=kw['past_key_values'].get_seq_length(),has_pixels=kw.get('pixel_values') is not None))
    def visual(*_):counts['visual']+=1
    def language(module,args,kw):counts['language']+=1;positions.append(kw['position_ids'].detach().cpu().clone())
    def norm_before(module,args):counts['norm']+=1;latest.update(norm_input_shape=list(args[0].shape),norm_query_input=args[0][:,-1:,:].detach().cpu().clone())
    def norm_after(module,args,value):latest.update(norm_output_shape=list(value.shape),normalized_query=value[:,-1:,:].detach().cpu().clone())
    def head_after(module,args,value):
        counts['head']+=1;shapes.append(dict(norm_input_shape=latest['norm_input_shape'],norm_output_shape=latest['norm_output_shape'],head_input_shape=list(args[0].shape),head_output_shape=list(value.shape)))
        heads.append(dict(norm_query_input=latest['norm_query_input'],normalized_query=latest['normalized_query'],head_input=args[0].detach().cpu().clone(),head_logits=value.detach().cpu().clone()))
        need(value.shape==(1,1,152064) and args[0].shape==(1,1,3584) and value.dtype==args[0].dtype==torch.float16,'Native single-query head required')
    with ExitStack() as stack:
        stack.callback(setattr,model.model,'rope_deltas',rope_before)
        for h in (model.register_forward_pre_hook(top,with_kwargs=True),model.register_forward_hook(recorder),model.model.visual.register_forward_pre_hook(visual),
            model.model.language_model.register_forward_pre_hook(language,with_kwargs=True),model.model.language_model.norm.register_forward_pre_hook(norm_before),
            model.model.language_model.norm.register_forward_hook(norm_after),model.lm_head.register_forward_hook(head_after)):stack.callback(h.remove)
        with torch.inference_mode():
            for step in range(4):
                start=L if step==0 else W+step-1;length=Q if step==0 else 1
                tokens=suffix['input_ids'].to(model.device) if step==0 else torch.tensor([[ids[-1]]],device=model.device)
                pos=suffix['position_ids'].to(model.device) if step==0 else torch.cat((torch.tensor([[[start]]],device=model.device),
                    (snapshot.rope_deltas.to(model.device).view(1,1,1)+start).expand(3,1,1)),dim=0)
                output=model(input_ids=tokens,attention_mask=torch.ones((1,start+length),device=model.device,dtype=torch.long),position_ids=pos,
                    cache_position=torch.arange(start,start+length,device=model.device),past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
                need(output.past_key_values is cache and cache.get_seq_length()==start+length,'Native question cache ownership/length changed')
                ids.append(int(output.logits[0,-1].argmax()))
                if ids[-1] in policy['native_eos_token_ids']:break
    raw=torch.stack(recorder.vectors);T=len(ids)
    need(counts==dict(model=T,visual=0,language=T,norm=T,head=T) and raw.shape==(T,152064) and raw.argmax(-1).tolist()==ids
         and all(torch.equal(head['normalized_query'],head['head_input']) for head in heads),'Actual branch call/head observations differ')
    for i,(item,pos,shape) in enumerate(zip(seen,positions,shapes)):
        start=L if i==0 else W+i-1;length=Q if i==0 else 1
        need(item['past_length']==start and not item['has_pixels'] and item['input_ids'].shape==(1,length)
             and item['attention_mask'].shape==(1,start+length) and bool((item['attention_mask']==1).all())
             and item['cache_position'].tolist()==list(range(start,start+length)) and shape['norm_input_shape']==shape['norm_output_shape']==[1,length,3584],
             'Question-only decode must use original prefix then emitted history')
        expected=suffix['position_ids'] if i==0 else torch.cat((torch.tensor([[[start]]]),(snapshot.rope_deltas.view(1,1,1)+start).expand(3,1,1)),dim=0)
        need(torch.equal(pos,expected) and (torch.equal(item['input_ids'],suffix['input_ids']) if i==0 else item['input_ids'].tolist()==[[ids[i-1]]]),'Actual branch mRoPE/history differs')
    for layer,(k,v) in zip(cache.layers,snapshot.layers):
        need(torch.equal(layer.keys[:,:,:L],k) and torch.equal(layer.values[:,:,:L],v),'Question/answer writes changed cached image-prefix bytes')
    verify_snapshot(snapshot)
    need(versions=={n:v._version for n,v in model.named_parameters()} and model.model.rope_deltas is rope_before,'Question branch changed model/rope state')
    complete=ids[-1] in policy['native_eos_token_ids']
    return dict(generated_ids=ids,raw_logits=raw,logit_records=recorder.records,completed=complete,truncated=not complete,finish_reason='eos' if complete else 'length',
        text=processor.tokenizer.decode(ids,skip_special_tokens=True),raw_text=processor.tokenizer.decode(ids,skip_special_tokens=False),counters=counts,
        native_inputs=seen,native_positions=positions,shapes=shapes,profile_head=heads,prefix_identity=snapshot.identity,
        prefix_bytes_unchanged=True,fork_storage_independent=True,rope_restored=True,parameter_versions_unchanged=True,hooks_removed=True,
        native_identity_sha256=snapshot.native_identity_sha256,generation=policy)


def self_test(torch):
    p.native.require_slurm()
    from types import SimpleNamespace
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    config=Qwen2_5_VLTextConfig(hidden_size=3584,num_hidden_layers=28,num_attention_heads=28,num_key_value_heads=4)
    layers=tuple((torch.full((1,4,2,128),float(i),dtype=torch.float16),torch.full((1,4,2,128),float(-i),dtype=torch.float16)) for i in range(28))
    ids=torch.tensor([[1,2]]);positions=torch.arange(2).view(1,1,2).expand(4,1,2).clone();delta=torch.zeros((1,1),dtype=torch.long)
    snapshot=PrefixSnapshot(layers,ids,positions,delta,_identity(layers,ids,positions,delta),'fixture')
    first=fork_cache(SimpleNamespace(config=config),snapshot)
    for index in range(28):first.update(torch.zeros((1,4,1,128),dtype=torch.float16),torch.ones((1,4,1,128),dtype=torch.float16),index)
    need(first.get_seq_length()==3 and snapshot.length==2,'Question append must not advance stored image prefix')
    second=fork_cache(SimpleNamespace(config=config),snapshot);need(second.get_seq_length()==2,'Second question inherited first question tokens')
    first.layers[1].keys.zero_();verify_snapshot(snapshot)
    need(torch.equal(second.layers[1].keys,layers[1][0]),'Independent fork changed')
    layers[0][1].add_(1)
    try:verify_snapshot(snapshot)
    except ValueError:rejected=True
    else:rejected=False
    need(rejected,'Changed original prefix bytes must be detected')
    return dict(passed=True,groups=['exact_independent28layer_fork','branch_append_reset','original_byte_guard'],native_model_calls=0,native_head_calls=0)
