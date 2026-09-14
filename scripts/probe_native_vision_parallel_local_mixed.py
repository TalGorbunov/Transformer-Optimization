"""Mixed independent local/text-only rows and two fixed broadcast cache steps.

Software only; no trained readout or aggregate answer. The global row has no
visual access. All numerical work, including CPU staging, requires Slurm.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import probe_native_vision_parallel_local as local
from scripts.stage_native_vision_v6_teacher import MODEL,need,prepared,read,save,sha
from scripts.cache_native_vision_v6_teacher import probabilities
DATA=Path('/mnt/data/gabriele/gnn_transformer/parallel_local_mixed')
OUT=REPO/'outputs/native_aggregation_vlm/parallel_local_mixed'
MANIFEST=Path('/mnt/data/gabriele/gnn_transformer/v4_diversity/profile_manifest.json')
CASES=(('test_N16','v2c_profile_test_software_N16_K3_0000_N16',16,3),
       ('test_N64','v2c_profile_test_software_N16_K6_0000_N64',64,6))
FORCED=('Therefore',':')
SEED=20260923
TOL=.02
OWN=('scripts/probe_native_vision_parallel_local_mixed.py',
     'slurm/native_vision_parallel_local_mixed_check.sbatch','slurm/native_vision_parallel_local_mixed.sbatch',
     'scripts/probe_native_vision_parallel_local.py','scripts/probe_native_vision_v5_local_readability.py')


def sources():return {name:sha(REPO/name) for name in OWN}


def tensor_info(tensor):
    import torch
    value=tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',sources())


def selected_sources():
    from scripts.probe_native_vision_v5_local_readability import semantic_labels
    manifest=read(MANIFEST);scenes=[]
    for split,sid,n,gold in CASES:
        found=[r for r in manifest['splits'][split]['samples'] if r['sid']==sid]
        need(len(found)==1,'Missing fixed software scene');row=found[0]
        need(row['n_frames']==n and row['gold']==gold and len(row['image_files'])==n,'Software scene law differs')
        labels=semantic_labels(row)
        for index,image in enumerate(row['image_files']):
            path=Path(image['path'])
            need(path==Path(row['path'])/f'{index:03d}.png' and path.stat().st_size==image['bytes']
                 and sha(path)==image['sha256'],'Software image ordering/hash differs')
        scenes.append(dict(row,semantic_labels=labels))
    return scenes,dict(path=str(MANIFEST),sha256=sha(MANIFEST))


def pack(torch,rows,pad):
    need(rows,'Empty mixed batch');length=max(r['input_ids'].shape[1] for r in rows)
    ids=torch.full((len(rows),length),pad,dtype=torch.long);mask=torch.zeros_like(ids);pixels=[];grids=[]
    for i,row in enumerate(rows):
        visual='pixel_values' in row
        need(set(row)==({'input_ids','attention_mask','pixel_values','image_grid_thw'} if visual else
                        {'input_ids','attention_mask'}) and row['input_ids'].shape[0]==1
             and row['attention_mask'].shape==row['input_ids'].shape and bool((row['attention_mask']==1).all()),
             'Expected complete unpadded N1 or text-only row')
        size=row['input_ids'].shape[1];ids[i,-size:]=row['input_ids'][0];mask[i,-size:]=1
        if visual:
            need(row['image_grid_thw'].shape==(1,3),'Local rows require one image')
            pixels.append(row['pixel_values']);grids.append(row['image_grid_thw'])
    value=dict(input_ids=ids,attention_mask=mask)
    if pixels:value.update(pixel_values=torch.cat(pixels),image_grid_thw=torch.cat(grids))
    return value


def rope(fn,owner,inputs):
    return fn(owner,input_ids=inputs['input_ids'],image_grid_thw=inputs.get('image_grid_thw'),attention_mask=inputs['attention_mask'])


def extend(torch,inputs,tokens):
    value=dict(inputs);b=inputs['input_ids'].shape[0]
    if tokens:
        tail=torch.tensor(tokens,dtype=torch.long).unsqueeze(0).expand(b,-1)
        value['input_ids']=torch.cat((inputs['input_ids'],tail),dim=1)
        value['attention_mask']=torch.cat((inputs['attention_mask'],torch.ones_like(tail)),dim=1)
    return value


def identity(inputs,positions,deltas):
    return dict(tensors={k:tensor_info(v) for k,v in inputs.items()},position_ids=tensor_info(positions),rope_deltas=deltas.tolist())


def full_processor(processor,scene):
    from PIL import Image
    from gnnformer.data import build_count_prompt
    frames=[];messages=[]
    try:
        for image in scene['image_files']:
            with Image.open(image['path']) as original:
                rgb=original.convert('RGB')
                try:frame=rgb.resize((392,392))
                finally:rgb.close()
            frames.append(frame);messages.append([dict(role='user',content=[dict(type='image',image=frame),
                dict(type='text',text=build_count_prompt(scene['question'],1))])])
        messages.append([dict(role='user',content=[dict(type='text',text=build_count_prompt(scene['question'],scene['n_frames']))])])
        return dict(processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                    return_dict=True,return_tensors='pt',padding=True))
    finally:
        for frame in frames:frame.close()


def compare(torch,current,reference,h_current,h_reference,policy):
    need(current.ndim==reference.ndim==1 and current.shape==reference.shape and h_current.shape==h_reference.shape==(3584,),
         'Comparison shape differs')
    a=current.double();b=reference.double();ha=h_current.double();hb=h_reference.double()
    need(all(bool(torch.isfinite(x).all()) for x in (a,b,ha,hb)),'Nonfinite logits or pre-final-RMS states')
    pa=torch.softmax(a,-1);pb=torch.softmax(b,-1);centered=(a-a.mean())-(b-b.mean());hdiff=ha-hb
    result=dict(policy=policy,full_vocabulary_tv=float(.5*(pa-pb).abs().sum()),top1_equal=int(a.argmax())==int(b.argmax()),
        maximum_absolute_logit_difference=float((a-b).abs().max()),centered_maximum_absolute_logit_difference=float(centered.abs().max()),
        centered_rms_logit_difference=float(centered.square().mean().sqrt()),hidden_maximum_absolute_difference=float(hdiff.abs().max()),
        hidden_rms_difference=float(hdiff.square().mean().sqrt()),hidden_relative_l2=float(hdiff.norm()/hb.norm()) if float(hb.norm()) else None,
        hidden_cosine=float(torch.dot(ha,hb)/(ha.norm()*hb.norm())) if float(ha.norm()*hb.norm()) else None,hidden_state_equality_claimed=False)
    if policy=='full_vocab':result['passed']=result['full_vocabulary_tv']<=TOL and result['top1_equal']
    return result


def mask_audit(torch,mask,key_mask,query_positions):
    # SDPA may unmask entirely padded query rows; check every valid query only.
    need(mask is not None and mask.ndim==4 and mask.shape[:2]==(key_mask.shape[0],1)
         and mask.shape[2:]==(query_positions.numel(),key_mask.shape[1]),'Expected mixed-row causal mask')
    allowed=mask if mask.dtype==torch.bool else mask==0
    expected=(torch.arange(key_mask.shape[1],device=key_mask.device).view(1,1,-1)<=query_positions.view(1,-1,1))
    expected=expected & key_mask.bool().unsqueeze(1);valid=key_mask.index_select(1,query_positions).bool()
    need(torch.equal(allowed[:,0][valid],expected.expand(key_mask.shape[0],-1,-1)[valid]),'Valid-query causal/padding mask differs')
    return dict(valid_query_rows=int(valid.sum()),padding_query_rows=int((~valid).sum()),mask_shape=list(mask.shape),
                mask_dtype=str(mask.dtype),valid_queries_exact=True)


def cache_snapshot(cache,batch,length,layers,clone=False):
    need(cache is not None and cache.__class__.__name__=='DynamicCache' and cache.get_seq_length()==length
         and len(cache.layers)==layers,'DynamicCache coverage/length differs')
    refs=[];metadata=[]
    for index,layer in enumerate(cache.layers):
        need(layer.__class__.__name__=='DynamicLayer','Unexpected sliding/static cache')
        k,v=layer.keys,layer.values
        need(k.shape==v.shape and k.ndim==4 and k.shape[0]==batch and k.shape[-2]==length
             and k.dtype==v.dtype and k.device==v.device and bool(k.isfinite().all()) and bool(v.isfinite().all()),'Cache state differs')
        if clone:refs.append((k.clone(),v.clone()))
        metadata.append(dict(layer=index,shape=list(k.shape),dtype=str(k.dtype)))
    return refs,metadata


def prefix_preserved(torch,cache,previous,length):
    exact=[bool(torch.equal(layer.keys[...,:length,:],k) and torch.equal(layer.values[...,:length,:],v))
           for layer,(k,v) in zip(cache.layers,previous)]
    need(len(exact)==len(cache.layers) and all(exact),'Broadcast modified existing cache prefix')
    return exact


def self_test(torch):
    a=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
           pixel_values=torch.tensor([[1.,2.]]),image_grid_thw=torch.tensor([[1,1,1]]))
    b=dict(input_ids=torch.tensor([[4]]),attention_mask=torch.ones(1,1,dtype=torch.long));batch=pack(torch,[a,b],0)
    need(batch['input_ids'].tolist()==[[1,2,3],[0,0,4]] and batch['image_grid_thw'].shape==(1,3),'Mixed row/padding failed')
    full=extend(torch,batch,[5,6]);need(full['input_ids'].tolist()==[[1,2,3,5,6],[0,0,4,5,6]],'Broadcast failed')
    pos=torch.arange(5);mask=(pos.view(1,1,-1)<=pos.view(1,-1,1)) & full['attention_mask'].bool().unsqueeze(1)
    mask_audit(torch,mask.unsqueeze(1),full['attention_mask'],pos);wrong=mask.clone();wrong[1,-1,0]=True
    try:mask_audit(torch,wrong.unsqueeze(1),full['attention_mask'],pos)
    except ValueError:pass
    else:raise AssertionError('Padding leakage accepted')
    x=torch.tensor([0.,1.,2.]);h=torch.ones(3584)
    need(compare(torch,x,x+100,h,h,'full_vocab')['passed'],'Common shift should pass')
    need(not compare(torch,x,x.flip(0),h,h,'full_vocab')['passed'],'Changed TV/top1 accepted')
    return dict(passed=True,tests=['mixed_padding','broadcast','causal_mask','padding_rejection','common_shift','top1_tv'])


def check():
    import torch
    import transformers
    from transformers import AutoProcessor
    from gnnformer.data import build_count_prompt,build_prompt_inputs
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter();tests=self_test(torch);frozen=sources()
    teacher,_,_,_,teacher_binding=local.teacher_inputs();scenes,manifest_binding=selected_sources()
    need(str(torch.__version__)==teacher['runtime']['torch_version'] and
         str(transformers.__version__)==teacher['runtime']['transformers_version'],'CPU runtime differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==teacher['processor'],'Frozen processor differs')
    batch_processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    batch_processor.tokenizer.padding_side='left';pad=processor.tokenizer.pad_token_id
    owner,fn,native=local.native_api(processor);token_ids=[]
    for word in FORCED:
        ids=processor.tokenizer(word,add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids and
             processor.tokenizer.decode(ids,skip_special_tokens=False)==word,'Fixed prefix is not one ordinary token')
        token_ids.append(ids[0])
    eos=read(MODEL/'generation_config.json')['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
    need(all(token not in eos for token in token_ids),'Fixed token is EOS')
    prepared_scenes={};records=[]
    for scene in scenes:
        rows=[];row_info=[];sid=scene['sid']
        for i,image in enumerate(scene['image_files']):
            inputs=prepared(processor,dict(image_path=image['path'],question=scene['question']));rows.append(inputs)
            row_info.append(dict(row_id=f'{sid}/local_{i:03d}',role='local',frame_index=i,image_path=image['path'],
                image_sha256=image['sha256'],question=scene['question'],semantic_class=scene['semantic_labels'][i],original_step=i+1))
        rows.append(build_prompt_inputs(processor,[],build_count_prompt(scene['question'],scene['n_frames'])))
        row_info.append(dict(row_id=f'{sid}/global',role='global',question=scene['question'],images=0))
        base=pack(torch,rows,pad);native_inputs=full_processor(batch_processor,scene)
        need(local.tensors_equal(torch,base,native_inputs),'Mixed packing differs from ordinary batch processor')
        lengths=[x['input_ids'].shape[1] for x in rows];padded=base['input_ids'].shape[1]
        need(len(set(lengths))>=2 and padded>lengths[-1] and bool((base['attention_mask'][-1,:padded-lengths[-1]]==0).all()),
             'Real heterogeneous padding must be exercised')
        identities=[];full_positions=[];full_deltas=[]
        for step in range(3):
            full=extend(torch,base,token_ids[:step]);positions,deltas=rope(fn,owner,full)
            for i,row in enumerate(rows):
                single=extend(torch,row,token_ids[:step]);sp,sd=rope(fn,owner,single);left=padded-lengths[i]
                need(torch.equal(positions[:,i,left:],sp[:,0,:]) and int(deltas[i,0])+left==int(sd[0,0]),
                     'Unpadded mixed mRoPE differs from complete isolated prefix')
            identities.append(identity(full,positions,deltas));full_positions.append(positions);full_deltas.append(deltas)
        need(all(torch.equal(full_deltas[0],x) for x in full_deltas),'Ordinary prefix changed rope deltas')
        prepared_scenes[sid]=dict(rows=rows,mixed=base,full_positions=full_positions,full_deltas=full_deltas)
        for info,row in zip(row_info,rows):
            info.update(prompt_tokens=row['input_ids'].shape[1],inputs={k:tensor_info(v) for k,v in row.items()},
                        input_ids=row['input_ids'][0].tolist())
        records.append(dict(sid=sid,n_frames=scene['n_frames'],audit_gold=scene['gold'],question=scene['question'],
            qa_path=str(Path(scene['path'])/'qa.txt'),qa_sha256=scene['qa_sha256'],rows=row_info,
            local_prompt_lengths=lengths[:-1],global_prompt_length=lengths[-1],global_left_padding=padded-lengths[-1],
            padded_prompt_length=padded,full_prefix_identities=identities,native_mixed_processor_exact=True))
    need(sources()==frozen and local.teacher_inputs()[4]==teacher_binding and sha(MANIFEST)==manifest_binding['sha256'],
         'Source/teacher/manifest changed during CPU stage')
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False);blob=data/'prepared_inputs.pt'
    torch.save(dict(schema_version=1,scenes=prepared_scenes),blob)
    plan=dict(schema_version=1,protocol='parallel_local_mixed_and_two_broadcast_tokens',source_sha256=frozen,
        teacher_binding=teacher_binding,manifest=manifest_binding,model=teacher['model'],runtime=teacher['runtime'],
        processor=teacher['processor'],native_api=native,image_processor_settings=teacher['image_processor_settings'],
        native_config_dtype=str(getattr(owner.config,'dtype',None)),
        dtype_policy='Preserve native tensor dtypes; record actual norm/head weights, pre-final-RMS hidden states and logits on GPU; config dtype is not an assertion about quantized runtime dtype',
        count_token_ids=teacher['count_token_ids'],pad_token_id=pad,forced_token_text=list(FORCED),forced_token_ids=token_ids,
        native_eos_token_ids=eos,seed=SEED,scenes=records,prepared_inputs_file=str(blob),prepared_inputs_sha256=sha(blob),
        expected_calls=dict(model=92,vision=86),expected_response_rows=492,
        numerical_policy=dict(local_prefill='conditional01 exact, p1/numeric-mass drift<=.02 vs fresh serial',
            global_prefill_and_all_cached_prefixes='full-vocabulary TV<=.02 and top1exact',hidden_errors='descriptive; finite required'),
        execution='per scene all local/global serial; mixed cached prefill; token1cached/full, token2cached/full',
        scope='software only; global row has no visual or aggregation input; no training or method accuracy',
        tests=tests,slurm_job_id=job,seconds=time.perf_counter()-started)
    path=out/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n');verify_plan(path)
    save(out/'summary.json',dict(passed=True,plan=str(path),plan_sha256=sha(path),source_sha256=frozen,
                               heterogeneous_padding_exercised=True,seconds=plan['seconds']))
    local.index(out,'Mixed CPU freeze',[('Plan','plan.json'),('Summary','summary.json'),('Sources','source/')])
    local.index(data,'Mixed prepared inputs',[('Tensors','prepared_inputs.pt')])
    print(json.dumps(dict(passed=True,plan=str(path),plan_sha256=sha(path),seconds=plan['seconds'])),flush=True)


def verify_plan(path):
    path=Path(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'Mixed plan sidecar differs')
    plan=read(path);teacher,_,_,_,binding=local.teacher_inputs()
    need(plan['schema_version']==1 and plan['protocol']=='parallel_local_mixed_and_two_broadcast_tokens'
         and plan['source_sha256']==sources() and plan['teacher_binding']==binding,'Frozen mixed source/teacher differs')
    for key in ('model','runtime','processor','image_processor_settings','count_token_ids'):
        need(plan[key]==teacher[key],'Frozen model/runtime differs: '+key)
    need(plan['manifest']==dict(path=str(MANIFEST),sha256=sha(MANIFEST)) and
         [(r['sid'],r['n_frames'],r['audit_gold']) for r in plan['scenes']]==[(s,n,k) for _,s,n,k in CASES],'Fixed scenes differ')
    need(plan['forced_token_text']==list(FORCED) and len(plan['forced_token_ids'])==2 and plan['seed']==SEED,'Fixed prefix differs')
    for name,value in plan['native_api']['source_sha256'].items():need(sha(name)==value,'Installed native source changed')
    need(sha(plan['prepared_inputs_file'])==plan['prepared_inputs_sha256'],'Prepared mixed tensors changed')
    return plan


def run(args):
    plan=verify_plan(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] is True and cpu['plan_sha256']==sha(args.plan) and cpu['source_sha256']==sources(),'Completed CPU freeze required')
    for name,value in sources().items():need(sha(Path(args.plan).parent/'source'/name.replace('/','_'))==value,'CPU snapshot differs')
    import torch
    import transformers
    from gnnformer.runtime import load_runtime,move_to_device,get_layers,get_rope_index_fn
    from gnnformer.data import build_count_prompt,build_prompt_inputs
    from scripts.probe_native_vision_v2_prefix import fingerprint
    need(torch.cuda.is_available() and str(torch.__version__)==plan['runtime']['torch_version']
         and str(transformers.__version__)==plan['runtime']['transformers_version'],'GPU runtime differs')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);tests=self_test(torch)
    started=time.perf_counter();job=os.environ['SLURM_JOB_ID'];out=OUT/f'run_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/f'run_{job}';data.mkdir(parents=True,exist_ok=False);(out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    local.index(out,'Mixed local/global broadcast probe',[('Configuration','config.json'),('Raw rows','rows.jsonl'),
        ('Summary after completion','summary.json'),('Report after completion','REPORT.md'),('Sources','source/')])
    local.index(data,'Pre-final-RMS states',[('State inventory after completion','states.json')])
    save(out/'config.json',dict(plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=sources(),
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],teacher_binding=plan['teacher_binding'],tests=tests))
    blob=torch.load(plan['prepared_inputs_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==1 and set(blob['scenes'])=={r['sid'] for r in plan['scenes']},'Mixed scene coverage differs')
    load_start=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.requires_grad_(False);model.eval()
    need(not any(p.requires_grad for p in model.parameters()) and fingerprint(runtime.processor,str(transformers.__version__))==plan['processor'],
         'Model is not frozen or processor differs')
    _,_,native=local.native_api(runtime.processor);need(native==plan['native_api'],'Native implementation differs')
    fn=get_rope_index_fn(model);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    language=model.model.language_model;vision=model.model.visual;layers=get_layers(model)
    weight_dtypes=dict(final_norm_weight=str(language.norm.weight.dtype),lm_head_weight=str(model.lm_head.weight.dtype),
                       final_norm_class=type(language.norm).__name__,lm_head_class=type(model.lm_head).__name__)
    save(out/'native_weight_dtypes.json',weight_dtypes)
    counts=dict(model=0,vision=0,language=0,norm=0);capture={};state_inventory=[];calls=[];raw_rows=[];violations=[]
    def model_hook(*_):counts['model']+=1
    def vision_hook(*_):counts['vision']+=1
    def language_hook(module,args,kwargs):
        counts['language']+=1;capture['positions']=kwargs['position_ids'].detach().clone()
        capture['key_mask']=kwargs['attention_mask'].detach().clone()
    def layer_hook(module,args,kwargs):
        capture['causal_mask']=kwargs['attention_mask'].detach().clone() if kwargs['attention_mask'] is not None else None
        capture['cache_position']=kwargs['cache_position'].detach().clone()
    def norm_hook(module,args):
        counts['norm']+=1;capture['hidden']=args[0][:,-1,:].detach().clone()
    handles=[model.register_forward_pre_hook(model_hook),vision.register_forward_pre_hook(vision_hook),
        language.register_forward_pre_hook(language_hook,with_kwargs=True),layers[0].register_forward_pre_hook(layer_hook,with_kwargs=True),
        language.norm.register_forward_pre_hook(norm_hook)]
    stream=(out/'rows.jsonl').open('x')

    def execute(scene,tag,inputs,expected_positions,expected_mask,*,cache=False,mixed=False,step=0):
        before=dict(counts);capture.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        with torch.inference_mode():result=model(**inputs,use_cache=cache is not False,logits_to_keep=1)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick;b=inputs['input_ids'].shape[0]
        actual={k:counts[k]-before[k] for k in counts};expected_vision=int(inputs.get('pixel_values') is not None)
        need(actual==dict(model=1,vision=expected_vision,language=1,norm=1),'Native forward/capture count differs')
        need(torch.equal(capture['positions'],expected_positions) and torch.equal(capture['key_mask'],expected_mask),
             'Executed logical mRoPE or key padding mask differs')
        mask_report=mask_audit(torch,capture['causal_mask'],expected_mask,capture['cache_position']) if mixed else None
        logits=result.logits[:,0,:].detach();hidden=capture['hidden']
        need(logits.shape[0]==b and hidden.shape==(b,3584) and bool(torch.isfinite(logits).all())
             and bool(torch.isfinite(hidden).all()),'Invalid final logits/pre-final-RMS state')
        path=data/f'{scene["sid"]}__{tag}.pt'
        row_ids=[r['row_id'] for r in scene['rows']] if mixed else [scene['active_row']['row_id']]
        torch.save(dict(schema_version=1,scene_id=scene['sid'],tag=tag,row_ids=row_ids,forced_token_ids=plan['forced_token_ids'][:step],
            pre_final_rms_hidden=hidden.detach().cpu(),native_next_token_logits=logits.detach().cpu(),
            native_weight_dtypes=weight_dtypes,native_position_ids=capture['positions'].detach().cpu(),
            attention_mask=capture['key_mask'].detach().cpu(),cache_position=capture['cache_position'].detach().cpu()),path)
        file_sha=sha(path)
        state_inventory.append(dict(scene_id=scene['sid'],tag=tag,path=str(path),sha256=file_sha,rows=row_ids,
                                    hidden_shape=list(hidden.shape),hidden_dtype=str(hidden.dtype),
                                    logits_shape=list(logits.shape),logits_dtype=str(logits.dtype),native_weight_dtypes=weight_dtypes))
        calls.append(dict(scene_id=scene['sid'],tag=tag,batch_size=b,forward_seconds=seconds,calls=actual,
            mask_audit=mask_report,hidden_dtype=str(hidden.dtype),logits_dtype=str(logits.dtype),
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),state_path=str(path),state_sha256=file_sha))
        for i,row_id in enumerate(row_ids):
            values=logits[i].double();a=float(values[plan['count_token_ids']['0']]);z=float(values[plan['count_token_ids']['1']]);normalizer=float(torch.logsumexp(values,-1))
            record=dict(scene_id=scene['sid'],tag=tag,row_id=row_id,step=step,forced_token_ids=plan['forced_token_ids'][:step],
                logit0=a,logit1=z,log_normalizer=normalizer,probabilities=probabilities(a,z,normalizer),top1_id=int(values.argmax()),
                top1_text=runtime.tokenizer.decode([int(values.argmax())],skip_special_tokens=False),state_path=str(path),state_row=i)
            raw_rows.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n')
        stream.flush();return logits,hidden,result.past_key_values

    def register(scene,tag,index,current,reference,hcurrent,href,policy):
        metrics=compare(torch,current,reference,hcurrent,href,policy)
        if policy=='local_prefill':
            ti=plan['count_token_ids']
            def values(logits):
                x=logits.double();return dict(logit0=float(x[ti['0']]),logit1=float(x[ti['1']]),log_normalizer=float(torch.logsumexp(x,-1)))
            a,b=(local.scalar_stats(values(x)) for x in (current,reference))
            metrics.update(binary_prediction_exact=a['binary_prediction']==b['binary_prediction'],
                conditional_p1_difference=abs(a['conditional_p1']-b['conditional_p1']),numeric_mass_difference=abs(a['numeric_mass']-b['numeric_mass']))
            metrics['passed']=metrics['binary_prediction_exact'] and metrics['conditional_p1_difference']<=TOL and metrics['numeric_mass_difference']<=TOL
        metrics.update(scene_id=scene['sid'],tag=tag,row_id=scene['rows'][index]['row_id'],row_index=index)
        if not metrics['passed']:violations.append(metrics)
        return metrics

    comparisons=[];cache_checks=[]
    try:
        for scene in plan['scenes']:
            entry=blob['scenes'][scene['sid']];base=entry['mixed'];b=len(scene['rows']);l=base['input_ids'].shape[1]
            need(sha(scene['qa_path'])==scene['qa_sha256'],'Software QA changed')
            serial_logits=[];serial_hidden=[]
            for i,(record,frozen_row) in enumerate(zip(scene['rows'],entry['rows'])):
                if record['role']=='local':
                    need(sha(record['image_path'])==record['image_sha256'],'Software image changed')
                    inputs=prepared(runtime.processor,dict(image_path=record['image_path'],question=record['question']))
                else:inputs=build_prompt_inputs(runtime.processor,[],build_count_prompt(record['question'],scene['n_frames']))
                need(local.tensors_equal(torch,inputs,frozen_row),'Fresh serial inputs differ from CPU freeze')
                item=move_to_device(inputs,runtime.device)
                positions,_=fn(input_ids=item['input_ids'],image_grid_thw=item.get('image_grid_thw'),attention_mask=item['attention_mask'])
                a,h,past=execute(dict(scene,active_row=record),f'serial_{i:03d}',item,positions,item['attention_mask'])
                need(past is None,'Serial reference returned a cache');serial_logits.append(a[0]);serial_hidden.append(h[0])
            serial_logits=torch.stack(serial_logits);serial_hidden=torch.stack(serial_hidden)
            item=move_to_device(base,runtime.device);positions=entry['full_positions'][0].to(runtime.device)
            need(identity(base,entry['full_positions'][0],entry['full_deltas'][0])==scene['full_prefix_identities'][0],'Mixed prefill identity differs')
            a,h,cache=execute(scene,'mixed_prefill',item,positions,item['attention_mask'],cache=True,mixed=True)
            need(torch.equal(model.model.rope_deltas.detach().cpu(),entry['full_deltas'][0]),'Prefill rope deltas differ')
            _,metadata=cache_snapshot(cache,b,l,len(layers));cache_checks.append(dict(scene_id=scene['sid'],tag='prefill',length=l,layers=metadata))
            for i in range(b):comparisons.append(register(scene,'mixed_prefill',i,a[i],serial_logits[i],h[i],serial_hidden[i],
                                                            'local_prefill' if i<b-1 else 'full_vocab'))
            del serial_logits,serial_hidden,a,h
            for step in (1,2):
                full=extend(torch,base,plan['forced_token_ids'][:step]);expected_pos=entry['full_positions'][step];expected_delta=entry['full_deltas'][step]
                need(identity(full,expected_pos,expected_delta)==scene['full_prefix_identities'][step],'Forced full-prefix identity differs')
                previous,_=cache_snapshot(cache,b,l+step-1,len(layers),clone=True);full_item=move_to_device(full,runtime.device)
                cp=torch.tensor([l+step-1],device=runtime.device)
                prepared_decode=model.prepare_inputs_for_generation(full_item['input_ids'],past_key_values=cache,
                    attention_mask=full_item['attention_mask'],cache_position=cp,use_cache=True,
                    pixel_values=full_item['pixel_values'],image_grid_thw=full_item['image_grid_thw'])
                need(prepared_decode['input_ids'].shape==(b,1) and bool((prepared_decode['input_ids']==plan['forced_token_ids'][step-1]).all())
                     and prepared_decode.get('pixel_values') is None and prepared_decode.get('past_key_values') is cache,
                     'Native broadcast preparation changed token/cache or reintroduced pixels')
                text_pos=(full_item['attention_mask'].long().cumsum(-1)-1)[:,-1:].unsqueeze(0)
                cache_pos=torch.cat((text_pos,expected_pos[:,:,-1:].to(runtime.device)),dim=0)
                need(torch.equal(prepared_decode['position_ids'],cache_pos),'Native text/mRoPE positions differ')
                need(prepared_decode.pop('use_cache',True) is True,'Native cache preparation disabled caching')
                ca,ch,next_cache=execute(scene,f'cached_step{step}',prepared_decode,cache_pos,full_item['attention_mask'],cache=True,mixed=True,step=step)
                need(next_cache is cache,'DynamicCache object changed');preserved=prefix_preserved(torch,cache,previous,l+step-1);del previous
                _,metadata=cache_snapshot(cache,b,l+step,len(layers))
                saved_deltas=model.model.rope_deltas.detach().clone()
                try:
                    fa,fh,no_cache=execute(scene,f'full_step{step}',full_item,expected_pos.to(runtime.device),full_item['attention_mask'],mixed=True,step=step)
                    need(no_cache is None and cache.get_seq_length()==l+step and
                         torch.equal(model.model.rope_deltas,saved_deltas) and torch.equal(saved_deltas.detach().cpu(),expected_delta),
                         'Full reference changed cache length or logical deltas')
                finally:model.model.rope_deltas=saved_deltas
                cache_checks.append(dict(scene_id=scene['sid'],tag=f'step{step}',length=l+step,layers=metadata,
                    old_prefix_exact_by_layer=preserved,broadcast_token_id=plan['forced_token_ids'][step-1],
                    pixel_values_absent_during_decode=True,logical_mrope_and_masks_exact=True,rope_deltas_preserved_around_reference=True))
                for i in range(b):comparisons.append(register(scene,f'cached_vs_full_step{step}',i,ca[i],fa[i],ch[i],fh[i],'full_vocab'))
                del ca,ch,fa,fh,full_item,prepared_decode
            del cache,item
            print(json.dumps(dict(completed_scene=scene['sid'],calls=counts,violations=len(violations))),flush=True)
    finally:
        stream.close()
        for handle in handles:handle.remove()
    need(counts==dict(model=92,vision=86,language=92,norm=92) and len(raw_rows)==492 and len(comparisons)==246,
         'Registered response/forward/comparison coverage differs')
    need(verify_plan(args.plan)==plan,'Frozen source/data changed during probe')
    save(data/'states.json',state_inventory);save(out/'comparisons.json',comparisons);save(out/'cache_checks.json',cache_checks)
    groups={name:[x for x in comparisons if x['policy']==name] for name in ('local_prefill','full_vocab')}
    numerical={name:dict(n=len(values),passed=all(x['passed'] for x in values),violations=sum(not x['passed'] for x in values),
        maximum_full_vocabulary_tv=max(x['full_vocabulary_tv'] for x in values),top1_mismatches=sum(not x['top1_equal'] for x in values),
        maximum_hidden_relative_l2=max(x['hidden_relative_l2'] for x in values if x['hidden_relative_l2'] is not None)) for name,values in groups.items()}
    numerical['local_prefill'].update(maximum_conditional_p1_difference=max(x['conditional_p1_difference'] for x in groups['local_prefill']),
        maximum_numeric_mass_difference=max(x['numeric_mass_difference'] for x in groups['local_prefill']))
    summary=dict(schema_version=1,completed=True,computational_integrity_passed=True,numerical_gate_passed=not violations,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=sources(),teacher_binding=plan['teacher_binding'],
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_weight_dtypes=weight_dtypes,
        actual_hidden_dtypes=sorted({x['hidden_dtype'] for x in state_inventory}),actual_logits_dtypes=sorted({x['logits_dtype'] for x in state_inventory}),
        calls=counts,response_rows=len(raw_rows),
        comparison_rows=len(comparisons),numerical_groups=numerical,violations=violations,per_call=calls,
        rows_file=str(out/'rows.jsonl'),rows_sha256=sha(out/'rows.jsonl'),comparisons_sha256=sha(out/'comparisons.json'),
        cache_checks_sha256=sha(out/'cache_checks.json'),state_inventory_file=str(data/'states.json'),state_inventory_sha256=sha(data/'states.json'),
        model_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,slurm_job_id=job,gpu=torch.cuda.get_device_name(0),
        limitations=['Software fidelity only: independent local rows never send evidence to the text-only global row.',
          'The Therefore/colon prefix is forced identically, not generated reasoning or a correctness signal.',
          'Hidden/centered-logit differences are descriptive; local output fidelity does not imply hidden equality.',
          'Cache prefix retention and mask/position correspondence are exact; cached/full distributions use fixed .02TV/top1 limits.',
          'Every violation is recorded. No tolerance search, training, aggregation or task-accuracy claim.',
          'State capture and numerical checks add overhead; timing is for this instrumented harness.'])
    save(out/'summary.json',summary)
    lines=['# Mixed local/text-only rows and broadcast cache','',
        f"Computational integrity: PASS. Registered numerical gate: {'PASS' if not violations else 'FAIL'}.",'',
        'Two software scenes;80local+2global serial references;92VLM/86visual calls;492rowoutputs.','',
        '| Comparison policy | Rows | Violations | Maximum full-vocabulary TV | Top1 mismatches |',
        '|---|---:|---:|---:|---:|']
    for name,x in numerical.items():lines.append(f"| {name} | {x['n']} | {x['violations']} | {x['maximum_full_vocabulary_tv']:.6g} | {x['top1_mismatches']} |")
    lines+=['','Local-prefill TV/top1 are descriptive; that gate uses conditional01/p1/numeric mass.',
            'Global-prefill and cached/full rows use full-vocabulary TV<=.02 and exacttop1.','',*summary['limitations'],'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),numerical_gate_passed=not violations,seconds=summary['total_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();need(bool(os.environ.get('SLURM_JOB_ID')),'All numerical work requires Slurm')
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS',''),'Use CPU wrapper')
        check()
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and args.plan is not None,'Use GPU wrapper with exact CPU plan')
        run(args)


if __name__=='__main__':main()
