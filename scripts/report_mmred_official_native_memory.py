"""Independent CPU audit of the original-MMReD native-memory software profile.

No model inference, fitting, backward pass, or benchmark efficacy score. Native
CPU norm/head replay covers the prospectively fixed rows only. All scheduled
numerical comparisons are retained before their final decision.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
from scripts import profile_mmred_official_native_memory as producer
from scripts.stage_mmred_official_recovery import need,sha,object_sha,save
p=producer.p
OUT=REPO/'outputs/native_aggregation_vlm/mmred_official_native_memory'
DATA=Path('/mnt/data/gabriele/gnn_transformer/mmred_official_native_memory_audit')
PROTOCOL='mmred_official_native_memory_independent_audit'
PROPOSAL='docs/paper/MMRED_OFFICIAL_NATIVE_MEMORY_AUDIT_PROPOSAL.md'
PROPOSAL_SHA='04eff9073031b3d1563873efd37f25f958eb71d8279514d71307c3be4a12e938'
OWN=('scripts/report_mmred_official_native_memory.py',PROPOSAL,'slurm/mmred_official_native_memory_report.sbatch')
POLICY=dict(cpu_seconds=1800,cpu_cores=4,memory_gib=16,cpu_head_calls_cap=1472,cpu_head_rows_cap=2550,
    teacher_replay_examples=22,natural_trajectories=29,memory_captures=42,core_atol=2e-4,core_rtol=2e-4,
    cpu_native_tv_max=.02,ce_absolute_tolerance=2e-6,cpu_argmax_gate=False,
    no_VLM_forward=True,no_backward=True,no_fit=True,no_efficacy_score=True,no_full_fit_release=True,
    all_scheduled_numerical_evidence_before_gate=True)


def read(path):return json.loads(Path(path).read_text())


def bind(path,bindings,expected=None):
    path=Path(path).resolve();digest=sha(path)
    need(expected is None or digest==expected,'Bound audit input changed: '+str(path))
    bindings[str(path)]=digest;return digest


def record(path,bindings,expected=None):bind(path,bindings,expected);return read(path)


def packet(torch,descriptor,bindings):
    bind(descriptor['file'],bindings,descriptor['sha256'])
    return torch.load(descriptor['file'],map_location='cpu',weights_only=True)


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held independent audit proposal changed')
    return {name:sha(REPO/name) for name in OWN}


def archive(directory,own,inherited,bindings):
    for name,digest in {**own,**inherited}.items():
        bind(REPO/name,bindings,digest);bind(directory/'source'/name.replace('/','_'),bindings,digest)


def snapshot(out,inherited):
    own=sources();(out/'source').mkdir()
    for name,digest in {**own,**inherited}.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Independent source archive changed')
    save(out/'source_hashes.json',own);save(out/'inherited_sources.json',dict(source_sha256=inherited));return own


def finite_number(value):
    value=float(value);return value if math.isfinite(value) else None


def functional_memory(torch,features,coordinates,weights,retain_mass):
    """Independent FP64 reconstruction; never invokes the learned core."""
    need(features.dtype==torch.float16 and features.ndim==2 and features.shape[1]==3584
         and set(weights)=={'key_weight','queries','mass_direction'},'Actual native feature/core state schema differs')
    need(weights['key_weight'].shape==(128,3608) and weights['queries'].shape==(32,128)
         and weights['mass_direction'].shape==(3584,) and all(v.dtype==torch.float32 and bool(v.isfinite().all()) for v in weights.values()),
         'Exact three-tensor FP32 memory state required')
    keys=('frame_index','raster_row','raster_col','grid_height','grid_width')
    need(set(coordinates)==set(keys) and all(coordinates[k].dtype==torch.int64 and coordinates[k].shape==(features.shape[0],) for k in keys),
         'Exact native-raster coordinate vectors required')
    frame,row,col,height,width=(coordinates[k].double() for k in keys)
    coords=torch.stack((frame/128,(row+.5)/height,(col+.5)/width),dim=1)
    phase=2*math.pi*coords[:,:,None]*torch.tensor([1.,2.,4.,8.],dtype=torch.float64)
    position=torch.stack((phase.sin(),phase.cos()),dim=-1).reshape(-1,24)
    value=features.double();normalized=value/(value.square().mean(-1,keepdim=True)+1e-6).sqrt()
    local=torch.cat((normalized,position),dim=-1)@weights['key_weight'].double().T
    score=(weights['queries'].double()@local.T)/math.sqrt(128)
    # Direct stable softmax is independent of the core's exposed merge state.
    log_mass=score.logsumexp(-1,keepdim=True);mu=score.softmax(-1)@value
    feature=log_mass if retain_mass else torch.zeros_like(log_mass)
    return dict(normalized_value=mu,log_mass=log_mass,mass_feature=feature,
                tokens=mu+feature*weights['mass_direction'].double()[None,:])


def memory_audit(torch,observed,features,coordinates,weights,arm,data,key):
    reference=functional_memory(torch,features,coordinates,weights,arm=='mass')
    need(set(observed)==set(reference),'Complete actual memory readout required')
    metrics={};passed=True
    for name,want in reference.items():
        got=observed[name]
        need(got.shape==want.shape and got.dtype==torch.float32 and not got.requires_grad,'Actual FP32 memory shape/dtype differs')
        finite=bool(got.isfinite().all()) and bool(want.isfinite().all())
        difference=(got.double()-want).abs();scale=want.abs()
        good=finite and bool((difference<=POLICY['core_atol']+POLICY['core_rtol']*scale).all())
        metrics[name]=dict(passed=good,finite=finite,max_absolute_error=finite_number(difference.max()),
            max_scaled_error=finite_number((difference/(POLICY['core_atol']+POLICY['core_rtol']*scale)).max()))
        passed=passed and good
    file=data/(key+'_memory_reference.pt');torch.save(reference,file)
    return dict(passed=passed,metrics=metrics,reference_file=str(file),reference_sha256=sha(file),
                arithmetic='FP64_from_actual_FP16_features_and_FP32_weights',input_normalization='keys_only_RMS_eps1e-6')


def native_modules(torch,plan,bindings):
    legacy=record(producer.preparation.IDENTITY_PLAN,bindings,producer.preparation.IDENTITY_SHA)
    need(legacy['native_identity']==plan['native_identity'],'Official preparation and original native identity differ')
    parent=record(legacy['orientation_plan']['file'],bindings,legacy['orientation_plan']['sha256'])
    bind(parent['native_model_file'],bindings,parent['native_model_sha256'])
    weights=torch.load(parent['native_model_file'],map_location='cpu',weights_only=True);identity=plan['native_identity']
    need(weights['schema_version']==1 and weights['native_identity']==identity and parent['native_identity']==identity
         and weights['rms_norm_eps']==identity['rms_norm_eps'],'Original frozen native weight packet differs')
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    bind(inspect.getfile(Qwen2RMSNorm),bindings,identity['norm_source_sha256'])
    for key in ('norm','head'):
        value=weights[key+'_weight']
        need(value.dtype==torch.float16 and not value.requires_grad and p.tensor_info(value)==identity[key+'_weight'],'Original native norm/head bytes differ')
    norm=Qwen2RMSNorm(3584,eps=weights['rms_norm_eps']);norm.weight=torch.nn.Parameter(weights['norm_weight'],requires_grad=False)
    head=torch.nn.Linear(3584,152064,bias=False,device='meta',dtype=torch.float16);head.weight=torch.nn.Parameter(weights['head_weight'],requires_grad=False)
    return norm.eval(),head.eval()


def head_identity(torch,cap):
    pre,normal,actual,logits=(cap[k] for k in ('norm_query_input','normalized_query','head_input','head_logits'))
    L=pre.shape[1]
    need(pre.shape==normal.shape==actual.shape==(1,L,3584) and logits.shape==(1,L,152064)
         and all(v.dtype==torch.float16 and not v.requires_grad for v in (pre,normal,actual,logits)), 'Actual selected native query tensors differ')
    return bool(torch.equal(normal,actual)) and all(bool(v.isfinite().all()) for v in (pre,normal,actual,logits))


def head_audit(torch,cap,modules,data,key):
    identity=head_identity(torch,cap);norm,head=modules
    with torch.inference_mode():normal=norm(cap['norm_query_input']);logits=head(normal)
    file=data/(key+'_head.pt');torch.save(dict(normalized_query=normal,head_logits=logits),file)
    finite=identity and bool(normal.isfinite().all()) and bool(logits.isfinite().all());metrics=[]
    for index in range(logits.shape[1]):
        gpu=cap['head_logits'][0,index];cpu=logits[0,index]
        tv=float((gpu.double().softmax(-1)-cpu.double().softmax(-1)).abs().sum()/2) if finite else None
        metrics.append(dict(row=index,full_vocabulary_tv=tv,gpu_argmax=int(gpu.argmax()),cpu_argmax=int(cpu.argmax()),
            argmax_exact=bool(gpu.argmax()==cpu.argmax()),passed=finite and tv<=.02))
    return dict(passed=all(v['passed'] for v in metrics),rows=len(metrics),metrics=metrics,normalized_equals_head_input=identity,
        norm_max_absolute_error=finite_number((normal.float()-cap['normalized_query'].float()).abs().max()),
        replay_file=str(file),replay_sha256=sha(file),cpu_argmax_gate=False)


def teacher_audit(torch,raw,case,arm,index,modules,data,key,replay,backward):
    need(raw['case_index']==index and raw['arm']==arm and raw['phase'] in ('teacher','initial_parity')
         and raw.get('backward_completed',False) is backward,'Teacher owner/phase/backward record differs')
    e=raw['evidence'];L=len(case['target_ids'])
    if arm=='ordinary':positions=case['teacher_position_ids'];width=case['metadata']['teacher_width']
    else:
        text=case['text'];width=text['prefix']['opening_ids'].shape[1]+32+1+text['suffix']['input_ids'].shape[1]+L-1
        positions=torch.arange(width).view(1,1,width).expand(4,1,width)
    need(e['target_ids']==case['target_ids'] and e['target_positions']==list(range(width-L,width))
         and len(e['native_positions'])==len(e['native_inputs'])==len(e['profile_head'])==len(e['shapes'])==1
         and torch.equal(e['native_positions'][0],positions), 'Teacher target/position ownership differs')
    expected=dict(model=1,visual=int(e.get('pixel_route',False)),language=1,norm=1,head=1,prefix_decoder=0)
    need(e['counters']==expected and e['shapes'][0]['norm_input_shape']==e['shapes'][0]['norm_output_shape']==[1,width,3584], 'Teacher actual call/normalization shape differs')
    item=e['native_inputs'][0]
    need(item['past_length']==0 and item['attention_mask'].shape==(1,width) and bool((item['attention_mask']==1).all())
         and item['cache_position'].tolist()==list(range(width)) and item['has_pixels']==bool(e.get('pixel_route',False)), 'Cold teacher cache/mask ownership differs')
    cap=e['profile_head'][0];identity=head_identity(torch,cap)
    logits=cap['head_logits'][0].double();targets=torch.tensor(case['target_ids'])
    ce=logits.logsumexp(-1)-logits[torch.arange(L),targets]
    need(len(e['position_ce'])==L,'Every teacher target loss is required')
    position_error=max(abs(float(x)-y) for x,y in zip(ce,e['position_ce']));mean_error=abs(float(ce.mean())-e['loss'])
    head=head_audit(torch,cap,modules,data,key) if replay else None
    passed=identity and math.isfinite(position_error) and math.isfinite(mean_error) and max(position_error,mean_error)<=2e-6
    return dict(passed=passed and (head is None or head['passed']),case_index=index,arm=arm,head=head,counters=expected,
        target_rows=L,normalized_equals_head_input=identity,maximum_position_ce_error=finite_number(position_error),
        mean_ce_error=finite_number(mean_error),backward_reexecuted=False)


def state_identity(values):return {name:p.tensor_info(value) for name,value in values.items()}


def adapter_state(torch,values):
    names={target+suffix for target in p.TARGETS for suffix in ('.lora_A.default.weight','.lora_B.default.weight')}
    need(set(values)==names and len(values)==224,'Exact224 language-only adapter tensors required')
    for name,value in values.items():
        projection=name.split('.self_attn.')[1].split('_proj')[0]
        width=512 if projection in ('k','v') else 3584
        expected=(16,3584) if '.lora_A.' in name else (width,16)
        need(tuple(value.shape)==expected and value.dtype==torch.float32 and not value.requires_grad
             and bool(value.isfinite().all()),'Adapter shape/dtype/finiteness differs')
    need(sum(v.numel() for v in values.values())==10092544,'Adapter parameter budget differs')


def memory_state(torch,values):
    need(set(values)=={'key_weight','queries','mass_direction'},'Exact three-field memory state required')
    shapes={'key_weight':(128,3608),'queries':(32,128),'mass_direction':(3584,)}
    need(all(tuple(v.shape)==shapes[k] and v.dtype==torch.float32 and not v.requires_grad
             and bool(v.isfinite().all()) for k,v in values.items()),'Memory state shape/dtype/finiteness differs')


def initial_audit(torch,descriptor,config,bindings):
    initial=packet(torch,descriptor,bindings);adapter_state(torch,initial['adapter']);memory_state(torch,initial['memory'])
    need(initial['seed']==24 and initial['updates']==0 and initial['peft_config']==config
         and state_identity(initial['adapter'])==descriptor['adapter_tensors']
         and state_identity(initial['memory'])==descriptor['memory_tensors'],'Actual initial packet differs')
    need(all(bool((v==0).all()) for k,v in initial['adapter'].items() if '.lora_B.' in k)
         and bool((initial['memory']['mass_direction']==0).all()),'Original zero-B/zero-mass initialization required')
    contract=initial['contract']
    need(contract['targets']==list(p.TARGETS) and contract['trainable_parameters']==10092544
         and set(contract['tensors'])==set(initial['adapter']),'Actual language-only contract differs')
    need(config['r']==16 and config['lora_alpha']==32 and config['lora_dropout']==.05
         and config['bias']=='none' and not config['use_dora'] and not config['use_rslora']
         and config['modules_to_save'] is None,'Fixed LoRA recipe differs')
    need(initial['rng']['cpu'].dtype==torch.uint8 and len(initial['rng']['cuda'])==1
         and initial['rng']['cuda'][0].dtype==torch.uint8,'Saved post-install RNG packet required')
    return initial


def parameter_audit(torch,arm,entry,initial,config,directory,bindings):
    need(set(entry['parameter_states'])=={'0','1','2'} and entry['parameter_states']['0']==config['initial']
         and entry['active_state']==entry['parameter_states']['2'] and entry['fresh_initial_adapter_exact']
         and entry['fresh_initial_memory_exact']==(arm!='ordinary') and entry['post_install_rng_restored'],
         'Fresh arm/init or complete state ledger differs')
    states={'0':initial};gradient_records=[]
    for step in (1,2):
        state=packet(torch,entry['parameter_states'][str(step)],bindings);adapter_state(torch,state['adapter'])
        need(state['arm']==arm and state['updates']==step and state['peft_config']==config['peft_config'],'Parameter-state owner differs')
        if arm=='ordinary':need(state['memory'] is None,'Ordinary arm cannot have trainable memory')
        else:
            memory_state(torch,state['memory'])
            if arm=='normalized':need(bool((state['memory']['mass_direction']==0).all()),'Normalized control acquired mass contribution')
        states[str(step)]=state
        gradfile=Path(config['checkpoint_directory'])/f'{arm}_gradient_{step}.pt'
        gradients=packet(torch,dict(file=str(gradfile),sha256=bind(gradfile,bindings)),bindings)
        live={**state['adapter'],**({} if arm=='ordinary' else {'memory.'+k:v for k,v in state['memory'].items()})}
        need(set(gradients)==set(live) and all(v.shape==live[k].shape and v.dtype==torch.float32
             and not v.requires_grad and bool(v.isfinite().all()) for k,v in gradients.items()),'Complete unclipped gradient packet differs')
        norms={letter:float(sum((v.double().square().sum() for k,v in gradients.items() if '.lora_'+letter+'.' in k)).sqrt()) for letter in ('A','B')}
        need(norms['B']>0 and (norms['A']==0 if step==1 else norms['A']>0),'Saved LoRA gradient path differs')
        if arm!='ordinary':
            norms.update({k:float(gradients['memory.'+k].double().norm()) for k in state['memory']})
            need(norms['key_weight']>0 and norms['queries']>0 and (norms['mass_direction']==0 if arm=='normalized' else norms['mass_direction']>0),
                 'Saved memory gradient path differs')
        log=record(directory/f'{arm}_optimizer_{step}.json',bindings)
        need(log['step']==step and log['lr']==2e-4 and log['parameter_steps']==[float(step)]*len(live)
             and math.isfinite(log['gradient_norm']),'Saved optimizer call inventory differs')
        gradient_records.append(dict(step=step,tensors=len(live),norms=norms,file=str(gradfile),sha256=sha(gradfile)))
    final=packet(torch,entry['checkpoint'],bindings)
    need(final['arm']==arm and final['updates']==2 and final['profile_only'] and final['must_not_initialize_full_fit']
         and final['peft_config']==config['peft_config'] and final['contract']==initial['contract']
         and state_identity(final['adapter'])==state_identity(states['2']['adapter'])
         and (final['memory'] is None if arm=='ordinary' else state_identity(final['memory'])==state_identity(states['2']['memory'])),
         'Final diagnostic endpoint/state or no-fit policy differs')
    optimizer=final['optimizer'];need(len(optimizer['param_groups'])==1,'One fixed optimizer group required')
    group=optimizer['param_groups'][0];count=224 if arm=='ordinary' else 227
    need(group['lr']==2e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01
         and len(group['params'])==count and set(optimizer['state'])==set(group['params']),'Optimizer scope or recipe differs')
    values=list(final['adapter'].values())+([] if arm=='ordinary' else list(final['memory'].values()))
    for identifier,value in zip(group['params'],values):
        item=optimizer['state'][identifier]
        need(float(item['step'])==2 and set(item)=={'step','exp_avg','exp_avg_sq'}
             and all(item[k].shape==value.shape and item[k].dtype==torch.float32 and bool(item[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')),
             'Actual final AdamW state differs')
    checkpoint=record(directory/f'{arm}_checkpoint.json',bindings)
    need(all(checkpoint[k] is True for k in ('original_reset_exact','reload_exact','optimizer_state_saved','rng_saved','full_peft_config_exact'))
         and checkpoint['adapter_tensors']==state_identity(final['adapter'])
         and checkpoint['memory_tensors']==(None if arm=='ordinary' else state_identity(final['memory'])),'Saved checkpoint roundtrip proof differs')
    provenance=dict(native_identity_sha256=config['native_identity_sha256'],lora=entry['checkpoint'],
        memory=None if arm=='ordinary' else dict(checkpoint=entry['checkpoint'],tensors=state_identity(final['memory'])))
    need(entry['provenance']==provenance,'Actual native/memory/LoRA endpoint provenance differs')
    return states,dict(passed=True,parameter_count=10092544+(0 if arm=='ordinary' else 469504),gradients=gradient_records,
                      checkpoint=entry['checkpoint'],optimizer_updates=2,optimizer_reexecuted=False)


def memory_links(torch,evidence,case,feature,state_ref,*,metadata=True):
    need(evidence['parameter_state']==state_ref and evidence['feature_packet']==feature
         and evidence['coordinate_identity']==state_identity(case['coordinates']),'Memory input/parameter/coordinate ownership differs')
    observed=evidence['memory_capture'];tokens=observed['tokens'].unsqueeze(0)
    if metadata:
        meta=evidence['metadata'];prefix=case['text']['prefix']
        need(meta['memory_input']==p.tensor_info(tokens) and meta['memory_native']==p.tensor_info(tokens.half())
             and meta['prefix_identity']==prefix['identity'] and meta['slots']==32
             and meta['prefix_length']==prefix['opening_ids'].shape[1]+33
             and meta['layout_ids_only'] and meta['question_input_absent'],'Actual memory cast/prefix metadata differs')
    return observed


def gpu_comparison(torch,left,right,*,exact=False,generation=False):
    a=left['raw_logits'] if generation else left['evidence']['profile_head'][0]['head_logits'][0].float()
    b=right['raw_logits'] if generation else right['evidence']['profile_head'][0]['head_logits'][0].float()
    shape=a.shape==b.shape;equal=shape and torch.equal(a,b)
    tv=float((a.double().softmax(-1)-b.double().softmax(-1)).abs().sum(-1).max()/2) if shape else None
    ids=left['generated_ids']==right['generated_ids'] if generation else shape and torch.equal(a.argmax(-1),b.argmax(-1))
    return dict(shape_equal=shape,raw_logits_equal=equal,ids_equal=ids,maximum_tv=finite_number(tv) if tv is not None else None,
        exact_required=exact,passed=bool(shape and ids and tv is not None and math.isfinite(tv) and tv<=.02 and (equal if exact else True)))


def prefix_audit(torch,raw,case,entry,features,states,config,data,arm):
    need(raw['case_index']==4 and raw['arm']==arm and raw['phase']=='prefix','Headless prefix ownership differs')
    e=raw['evidence'];observed=memory_links(torch,e,case,features,entry['active_state'])
    snap=raw['snapshot'];prefix=snap['prefix'];meta=snap['metadata'];P=case['text']['prefix']['opening_ids'].shape[1]+33
    positions=torch.arange(P).view(1,1,P).expand(4,1,P)
    layout=torch.cat((case['text']['prefix']['opening_ids'],torch.full((1,32),case['text']['prefix']['image_pad_id']),case['text']['prefix']['end_ids']),dim=1)
    need(len(prefix['layers'])==28 and torch.equal(prefix['prefix_ids'],layout)
         and torch.equal(prefix['position_ids'],positions) and torch.equal(prefix['rope_deltas'],torch.zeros((1,1),dtype=torch.long))
         and prefix['native_identity_sha256']==config['native_identity_sha256'],'Headless prefix layout/native positions differ')
    layers=[]
    for saved,actual in zip(prefix['layers'],e['layers']):
        k,v=saved['key'],saved['value']
        need(k.shape==v.shape==(1,4,P,128) and k.dtype==v.dtype==torch.float16 and bool(k.isfinite().all()) and bool(v.isfinite().all())
             and torch.equal(k,actual[0]) and torch.equal(v,actual[1]),'Actual28-layer prefix K/V bytes differ')
        layers.append(dict(key=p.tensor_info(k),value=p.tensor_info(v)))
    identity=dict(layers=layers,prefix_ids=p.tensor_info(layout),position_ids=p.tensor_info(positions),rope_deltas=p.tensor_info(prefix['rope_deltas']))
    need(prefix['identity']==identity and meta['caller_provenance']==entry['provenance']
         and meta['actual_lora_tensors']==state_identity(states['2']['adapter'])
         and meta['position_identity']==p.tensor_info(positions) and meta['rope_delta']==0,'Prefix byte or fitted-state identity differs')
    need(e['counters']==dict(model=0,visual=0,language=1,norm=1,head=0,prefix_decoder=1)
         and not e['profile_head'] and len(e['native_inputs'])==len(e['native_positions'])==1
         and torch.equal(e['native_positions'][0],positions) and e['rope_restored'] and e['hooks_removed'], 'Headless native call inventory differs')
    inp=e['native_inputs'][0];emb=e['prefix_embeddings'];offset=case['text']['prefix']['opening_ids'].shape[1]
    need(emb.shape==(1,P,3584) and emb.dtype==torch.float16 and p.tensor_info(emb)==meta['prefix_embeddings']
         and torch.equal(emb[:,offset:offset+32],observed['tokens'].unsqueeze(0).half())
         and inp['input_ids'] is None and inp['inputs_embeds_identity']==p.tensor_info(emb)
         and inp['past_length']==0 and not inp['has_pixels'] and inp['attention_mask'].tolist()==[[1]*P]
         and inp['cache_position'].tolist()==list(range(P)),'Actual continuous prefix embeddings or cache start differ')
    model=snap['model_state'];need(model['peft_config']==config['peft_config'] and set(model['lora_runtime'])==set(p.TARGETS)
         and all(not x['requires_grad'] for x in model['parameters'].values()),'Frozen prefix model/config differs')
    for target,state in model['lora_runtime'].items():
        need(state==dict(active_adapters=['default'],disable_adapters=False,merged_adapters=[],scaling={'default':2.},
            rank={'default':16},alpha={'default':32},dropout={'default':dict(p=.05,training=False)}),'Prefix active LoRA runtime differs')
    for name,value in states['2']['adapter'].items():
        actual=model['parameters'][name]
        need(actual['object_id']==config['_initial_contract']['tensors'][name]['object_id'] and actual['shape']==list(value.shape)
             and actual['dtype']=='torch.float32','Prefix adapter objects differ')
    return dict(passed=True,prefix_length=P,kv_bytes=sum(k['key'].numel()*k['key'].element_size()+k['value'].numel()*k['value'].element_size() for k in prefix['layers']),
                question_input_absent=True,identity=identity),observed


def natural_audit(torch,raw,case,text_case,entry,states,feature,prefix,modules,data,key):
    arm=raw['arm'];kind=raw['phase'];e=raw['evidence'];result=raw['result'];ids=result['generated_ids'];T=len(ids)
    need(1<=T<=50 and all(type(v) is int and 0<=v<152064 for v in ids)
         and not any(v in (151645,151643) for v in ids[:-1]) and result['completed']==(ids[-1] in (151645,151643))
         and result['truncated']==(ids[-1] not in (151645,151643)) and (result['completed'] or T==50), 'Natural EOS/length stopping differs')
    need(e['parameter_state']==entry['active_state'] and e['feature_packet']==feature and e['generated_ids']==ids,
         'Natural feature/endpoint ownership differs')
    expected=dict(model=T,visual=0,language=T,norm=T,head=T,prefix_decoder=0)
    need(result['counters']==e['counters']==expected and result['rope_restored'] and result['hooks_removed']
         and e['rope_restored'] and e['hooks_removed'] and len(e['native_inputs'])==len(e['profile_head'])==len(e['shapes'])==len(e['native_positions'])==T,
         'Natural actual counters/cleanup differ')
    need(result['raw_logits'].shape==(T,152064) and result['raw_logits'].dtype==torch.float32,'Complete raw natural vocabulary vectors required')
    if arm=='ordinary':
        need(kind=='ordinary' and raw['text_case_index'] is None and result['native_positions_preserved'],'Ordinary natural ownership differs')
        W=case['metadata']['prompt_width'];start=0;first_positions=case['position_ids']
    else:
        need(kind in ('cold','split','reverse') and raw['case_index']==4 and raw['text_case_index'] in range(4,8), 'Memory question reuse ownership differs')
        P=text_case['text']['prefix']['opening_ids'].shape[1]+33;W=P+text_case['text']['suffix']['input_ids'].shape[1]
        start=P if kind in ('split','reverse') else 0
        first_positions=torch.arange(start,W).view(1,1,W-start).expand(4,1,W-start)
        need(result['generation']==dict(max_new_tokens=50,do_sample=False,num_beams=1,native_eos_token_ids=[151645,151643],
             repetition_penalty=1.,vocabulary_mask=False,other_logits_processors=False,logits_to_keep=1)
             and result['parameter_versions_unchanged'] and result['prefix_bytes_unchanged']==(start>0)
             and result['metadata']==e['metadata'],'Memory generation/config or prefix immutability differs')
        meta=e['metadata'];need(meta['caller_provenance']==entry['provenance'] and meta['actual_lora_tensors']==state_identity(states['2']['adapter'])
             and meta['prefix_identity']==text_case['text']['prefix']['identity'] and meta['memory_input']==prefix['snapshot']['metadata']['memory_input']
             and meta['memory_native']==prefix['snapshot']['metadata']['memory_native'],'Natural prefix memory/adapter identity differs')
        if kind=='cold':
            observed=memory_links(torch,e,case,feature,entry['active_state'])
            need(state_identity(observed)==state_identity(prefix['evidence']['memory_capture']),'Redundant cold capture differs from same frozen prefix readout')
    audits=[];identity=True
    for i in range(T):
        begin=start if i==0 else W+i-1;length=W-start if i==0 else 1
        positions=first_positions if i==0 else torch.arange(begin,begin+1).view(1,1,1).expand(4,1,1).clone()
        if arm=='ordinary' and i:positions[1:]+=case['rope_deltas'].reshape(1,1,1)
        inp=e['native_inputs'][i];cap=e['profile_head'][i];shape=e['shapes'][i]
        need(torch.equal(e['native_positions'][i],positions) and torch.equal(result['native_positions'][i],positions)
             and inp['past_length']==begin and not inp['has_pixels'] and inp['attention_mask'].tolist()==[[1]*(begin+length)]
             and inp['cache_position'].tolist()==list(range(begin,begin+length)), 'Native natural positions/mask/cache differ')
        need(shape['norm_input_shape']==shape['norm_output_shape']==[1,length,3584]
             and shape['head_input_shape']==[1,1,3584] and shape['head_output_shape']==[1,1,152064], 'Actual natural norm/head shape differs')
        if i:need(inp['input_ids'].tolist()==[[ids[i-1]]] and inp['inputs_embeds_identity'] is None,'Natural branch consumed another trajectory')
        elif start:need(torch.equal(inp['input_ids'],text_case['text']['suffix']['input_ids']) and inp['inputs_embeds_identity'] is None,'Cached branch did not consume the exact original question suffix')
        else:
            need(inp['input_ids'] is None and inp['inputs_embeds_identity'] is not None,'Cold branch must use actual assembled embeddings')
            if arm!='ordinary':need(inp['inputs_embeds_identity']==e['metadata']['full_embedding_identity'],'Actual cold full embedding identity differs')
        vector=result['raw_logits'][i]
        need(torch.equal(vector,cap['head_logits'][0,-1].float()) and ids[i]==int(vector.argmax())
             and state_identity(cap)==state_identity(result['profile_head'][i]),'Saved raw vector/head or greedy ID differs')
        identity=identity and head_identity(torch,cap)
        audits.append(head_audit(torch,cap,modules,data,key+f'_token_{i}'))
    return dict(passed=identity and all(v['passed'] for v in audits),arm=arm,kind=kind,case_index=raw['case_index'],
        text_case_index=raw['text_case_index'],generated_ids=ids,completed=result['completed'],truncated=result['truncated'],
        counters=expected,cpu_head_calls=T,cpu_head_rows=T,head_audits=audits,efficacy_scored=False)


def timing_audit(analysis):
    lengths=(1,2,4,8,16);features=analysis['feature_records'];setup=analysis['setup_seconds'];arms=analysis['arms']
    F={n:max(x['seconds'] for x in features if x['n']==n) for n in lengths};shared=sum(800*F[n] for n in lengths)
    expected=dict(training_only=True,shared_feature_preparation_seconds=shared,feature_seconds_by_n=F,
        feature_cache_built_once_for_all_arms=True,training_pairs=4000,epochs=3,scene_presentations=12000,optimizer_steps=1500,
        validation_test_cost_included=False,no_N32_generation_estimate=True,empirical_estimate_not_guarantee=True,main_release=False,arms={})
    need(math.isfinite(setup) and setup>0 and all(math.isfinite(x) and x>0 for x in F.values()),'Finite setup/feature timings required')
    for arm in ('ordinary','normalized','mass'):
        a=arms[arm];T={n:max(x['seconds'] for x in a['microbatches'] if x['n']==n) for n in lengths}
        O=max(a['optimizer_seconds']);C=a['checkpoint_seconds']
        need(len(a['optimizer_seconds'])==2 and all(math.isfinite(x) and x>0 for x in [O,C,*T.values()]),'Finite fixed profile timing population required')
        estimate=setup+1.25*(sum(2400*T[n] for n in lengths)+1500*O+2*C)+60
        nat=[x for x in a['natural'] if x['kind'] in ('ordinary','cold')]
        for x in a['natural']:
            need(math.isfinite(x['seconds']) and x['seconds']>0 and x['fifty_token_seconds']==x['seconds']*50/x['tokens'], 'All measured natural timing bounds required')
        G={n:max(x['fifty_token_seconds'] for x in nat if x['n']==n) for n in sorted({x['n'] for x in nat})}
        expected['arms'][arm]=dict(training_without_shared_features_seconds=estimate,training_with_shared_features_if_run_alone_seconds=estimate+shared,
            microbatch_seconds_by_n=T,optimizer_seconds=O,checkpoint_seconds=C,generation_fifty_token_seconds_by_n=G,
            formula='setup+1.25*(sum_N2400*T_arm_N+1500*O_arm+2*C_arm)+60',future_training_generation_cases=100,diagnostic_generation_cost_is_separate=True)
    expected=json.loads(json.dumps(expected));need(expected==analysis['projection'],'Independent training-only forecast reconstruction differs')
    return dict(passed=True,projection=expected,full_4000_prepared_input_envelope_validated=False,
        fixed_100_diagnostic_generation_cost_included=False,full_fit_execution_release=False,
        limitations=['Eight fixed cases cover length strata, not the full4000 prompt/teacher-width envelope.',
                    'Forecasts exclude the100 diagnostic generations and all validation/test inference.',
                    'The fixed1.25 multiplier is an empirical reserve, not a physical latency bound.'])


def input_audit(torch,path,out,bindings):
    path=Path(path).resolve();directory=path if path.is_dir() else path.parent
    need(directory.parent==OUT and directory.name.startswith('profile_'),'Registered profile directory required')
    config=record(directory/'config.json',bindings);analysis=record(directory/'analysis.json',bindings)
    need(config['protocol']==producer.PROTOCOL and config['policy']==producer.POLICY and config['source_sha256']==producer.sources()
         and config['no_full_fit_release'] and analysis['profile_only'] and analysis['no_full_fit_release']
         and analysis['no_validation_or_test_predictions'] and analysis['all_scheduled_comparisons_collected']
         and analysis['comparison_count']==24 and analysis['natural_trajectories']==29,'Complete fixed software profile evidence required')
    original_passed=analysis['passed'];failure=None
    if (directory/'failure.json').exists():
        failure=record(directory/'failure.json',bindings)
        need(not original_passed and not (directory/'summary.json').exists() and failure['type']=='ValueError'
             and failure['message']=='Native pixel/feature, trained checkpoint or cold/cache parity failed; all scheduled raw evidence retained'
             and failure['policy']==producer.POLICY and failure['source_sha256']==config['source_sha256']
             and failure['progress']['counters']==analysis['counters'] and failure['no_full_fit_release'],
             'Only the complete preserved final numerical-gate failure is auditable')
    else:
        summary=record(directory/'summary.json',bindings)
        need(summary['passed'] is summary['completed'] is True and original_passed
             and summary['analysis_sha256']==sha(directory/'analysis.json') and summary['counters']==analysis['counters'], 'Passed profile summary differs')
    release_path=OUT/'source_release.json';release=record(release_path,bindings)
    need(release['status']=='frozen_before_execution' and release['source_sha256']==config['source_sha256']
         and release['inherited_source_sha256']==config['inherited_source_sha256'],'Pre-execution profile source release differs')
    archive(directory,config['source_sha256'],config['inherited_source_sha256'],bindings)
    plan,ancestors,parent_bindings,preparation,diagnostics=producer.dependencies(config['preparation']['file'])
    need(preparation==config['preparation'] and ancestors==config['inherited_source_sha256'] and diagnostics==config['future_training_diagnostic_sids']
         and config['cases']==plan['cases'] and config['native_identity']==plan['native_identity']
         and config['native_identity_sha256']==plan['native_identity_sha256'],'Profile input/source/native handoff differs')
    bindings.update(parent_bindings)
    recorded=record(directory/'input_bindings.json',bindings)
    for file,digest in recorded.items():bind(file,bindings,digest)
    need(record(directory/'base_before.json',bindings)==record(directory/'base_after.json',bindings),'Actual frozen base storage/dtypes/versions changed')
    need(config['hardware']['gpu']=='NVIDIA B200' and config['hardware']['precision']==plan['precision']
         and config['hardware']['quantization']['load_in_4bit'] and config['hardware']['quantization']['bnb_4bit_use_double_quant']
         and config['hardware']['quantization']['bnb_4bit_quant_type']=='nf4' and config['hardware']['quantization']['bnb_4bit_compute_dtype']=='bfloat16','Native B200/precision/NF4 identity differs')
    data=Path(config['data_directory']);ckpt=Path(config['checkpoint_directory'])
    need(data==producer.DATA/directory.name and ckpt==producer.CKPT/directory.name,'Profile artifacts outside designated roots')
    inventory={}
    for base in (directory,data,ckpt):
        for file in sorted(base.iterdir()):
            if file.is_file():inventory[str(file.resolve())]=bind(file,bindings)
    save(out/'profile_artifact_inventory.json',inventory)
    if (directory/'artifacts.json').exists():
        for file,digest in record(directory/'artifacts.json',bindings).items():bind(file,bindings,digest)
    own={**config['source_sha256'],**config['inherited_source_sha256']}
    cases={};features={};feature_records={r['index']:r for r in analysis['feature_records']}
    need(len(feature_records)==8 and list(feature_records)==list(range(8)),'Exactly eight ordered feature extractions required')
    for item in plan['cases']:
        index=item['index'];case=packet(torch,item,bindings)
        need(case['metadata']==item['metadata'] and case['target_ids']==item['target_ids'] and 1<len(case['target_ids'])<=50
             and case['target_ids'][-1]==151645 and not any(v in (151645,151643) for v in case['target_ids'][:-1]), 'Prepared teacher identity/target boundary differs')
        record_feature=feature_records[index];feature=packet(torch,record_feature,bindings)
        identity={k:p.tensor_info(case['inputs'][k]) for k in ('pixel_values','image_grid_thw')}
        need(feature['case_index']==index and feature['input_identity']==record_feature['input_identity']==identity
             and p.tensor_info(feature['features'])==record_feature['tensor'] and feature['features'].dtype==torch.float16
             and feature['features'].shape==(196*item['n'],3584) and bool(feature['features'].isfinite().all()),'Question-free native feature ownership differs')
        n=item['n'];coords=dict(frame_index=torch.arange(n).repeat_interleave(196),raster_row=torch.arange(14).repeat_interleave(14).repeat(n),
            raster_col=torch.arange(14).repeat(n*14),grid_height=torch.full((n*196,),14),grid_width=torch.full((n*196,),14))
        need(state_identity(case['coordinates'])==state_identity(coords),'Actual global native-raster coordinates differ')
        cases[index]=case;features[index]=feature['features']
    descriptor=dict(directory=str(directory),config_file=str(directory/'config.json'),config_sha256=sha(directory/'config.json'),
        analysis_file=str(directory/'analysis.json'),analysis_sha256=sha(directory/'analysis.json'),
        failure_file=str(directory/'failure.json') if failure is not None else None,
        failure_sha256=sha(directory/'failure.json') if failure is not None else None,
        original_profile_passed=original_passed,source_release_file=str(release_path),source_release_sha256=sha(release_path))
    return directory,config,analysis,plan,own,cases,features,feature_records,descriptor


def allocation_audit(directory,out,original_passed):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw']
    raw=subprocess.run(command,text=True,capture_output=True,check=True).stdout
    file=out/'sacct.psv';file.write_text(raw);selected=[];expected_job=directory.name.removeprefix('profile_')
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)!=8 or fields[1]!='mmred_official_native_memory_profile':continue
        identifier,name,partition,state,exit_code,seconds,tres,limit=fields
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres);typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        selected.append(dict(job_id=identifier,name=name,partition=partition,state=state,exit_code=exit_code,
            seconds=int(seconds),gpus=int(generic[0]) if generic else sum(map(int,typed)),time_limit_minutes=int(limit),tres=tres))
    need(len(selected)==1 and selected[0]['job_id']==expected_job and selected[0]['partition']=='gpu' and selected[0]['gpus']==1
         and selected[0]['seconds']<=900 and selected[0]['time_limit_minutes']==15,'Exactly one fixed900-second profile allocation required')
    expected_state=('COMPLETED','0:0') if original_passed else ('FAILED','1:0')
    need((selected[0]['state'],selected[0]['exit_code'])==expected_state,'Preserved original Slurm outcome differs')
    return dict(passed=True,profile=selected[0],raw_file=str(file),raw_sha256=sha(file),extra_GPU_calls=0)


def self_test(torch):
    # Closed-form uniform score fixture; no memory-core or native-head call.
    features=torch.zeros((2,3584),dtype=torch.float16);features[0,0]=2.;features[1,0]=6.
    coords=dict(frame_index=torch.tensor([0,1]),raster_row=torch.zeros(2,dtype=torch.long),raster_col=torch.zeros(2,dtype=torch.long),
                grid_height=torch.ones(2,dtype=torch.long),grid_width=torch.ones(2,dtype=torch.long))
    weights=dict(key_weight=torch.zeros((128,3608)),queries=torch.zeros((32,128)),mass_direction=torch.zeros(3584))
    weights['mass_direction'][1]=3.
    normal=functional_memory(torch,features,coords,weights,False);mass=functional_memory(torch,features,coords,weights,True)
    need(bool((normal['tokens'][:,0]==4).all()) and bool((normal['tokens'][:,1:]==0).all())
         and bool(torch.allclose(mass['log_mass'],torch.full((32,1),math.log(2),dtype=torch.float64),atol=1e-14,rtol=0))
         and bool(torch.allclose(mass['tokens'][:,1],torch.full((32,),3*math.log(2),dtype=torch.float64),atol=1e-14,rtol=0)),
         'Independent uniform-score memory fixture failed')
    teachers=[(arm,step,index) for arm in ('ordinary','normalized','mass') for step in (1,2) for index in range(8)]
    selected=[r for r in teachers if r[1:] in ((1,0),(2,7))]
    need(len(teachers)==48 and len(selected)==6 and 4+12+len(selected)==22 and 29*50+22==1472 and 29*50+22*50==2550,
         'Prospectively fixed replay inventory fixture failed')
    return dict(passed=True,groups=2,native_head_calls=0,model_calls=0)


def audit(torch,args,out,data,bindings,progress):
    directory,config,original,plan,inherited,cases,features,feature_records,descriptor=input_audit(torch,args.profile,out,bindings)
    own=snapshot(out,inherited);save(out/'profile_binding.json',descriptor)
    tests=self_test(torch);save(out/'tests.json',tests)
    modules=native_modules(torch,plan,bindings)
    initial=initial_audit(torch,config['initial'],config['peft_config'],bindings);config['_initial_contract']=initial['contract']
    need(config['peft_config']==record(directory/'actual_peft_config_before_contract.json',bindings),'Actual saved full PEFT configuration differs')
    teacher_results=[];memory_results=[];natural_results=[];comparisons=[];parameters={};prefix_results=[]
    progress.update(teacher_audits=teacher_results,memory_audits=memory_results,natural_audits=natural_results,comparisons=comparisons)
    def local_packet(file):
        return packet(torch,dict(file=str(file),sha256=bind(file,bindings)),bindings)
    def save_progress():
        # Exclusive files retain completed rows even if a later structural check fails.
        save(out/f'progress_{len(teacher_results)}_{len(memory_results)}_{len(natural_results)}.json',
             dict(teacher_audits=teacher_results,memory_audits=memory_results,natural_audits=natural_results))
    def teacher(raw,index,arm,key,replay,backward,state_ref,states):
        e=raw['evidence'];need(e['parameter_state']==state_ref and e['feature_packet']==feature_records[index], 'Teacher endpoint/native feature owner differs')
        result=teacher_audit(torch,raw,cases[index],arm,index,modules,data,key,replay,backward)
        if arm=='ordinary':
            need(e['embedding_identity']==e['actual_language_embeddings'],'Ordinary replacement differs from actual native language input')
            if not e.get('pixel_route',False):need(e['native_inputs'][0]['inputs_embeds_identity']==e['embedding_identity'],'Ordinary embedding path differs')
            else:need(torch.equal(e['native_inputs'][0]['input_ids'],cases[index]['teacher_input_ids']),'Pixel teacher original full prompt differs')
        else:
            observed=memory_links(torch,e,cases[index],feature_records[index],state_ref)
            need(e['native_inputs'][0]['inputs_embeds_identity']==e['metadata']['full_embedding_identity']
                 and e['decoder_training']==backward and e['gradient_enabled']==backward,'Actual memory embedding/autograd mode differs')
            state=next(value for k,value in states.items() if state_ref==original['arms'][arm]['parameter_states'][k])
            metric=memory_audit(torch,observed,features[index],cases[index]['coordinates'],state['memory'],arm,data,key)
            metric.update(arm=arm,case_index=index,parameter_state=state_ref,capture_file=key);memory_results.append(metric)
        result['key']=key;teacher_results.append(result);save_progress()
        return result
    for index in (0,4):
        pair={}
        for route in ('pixels','features'):
            key=f'parity_{index}_{route}';raw=local_packet(Path(config['data_directory'])/(key+'.pt'));pair[route]=raw
            teacher(raw,index,'ordinary',key,True,False,config['initial'],{'0':initial})
        metric=gpu_comparison(torch,pair['pixels'],pair['features'],exact=True)
        metric['actual_language_embeddings_equal']=pair['pixels']['evidence']['actual_language_embeddings']==pair['features']['evidence']['actual_language_embeddings']
        metric['passed']=metric['passed'] and metric['actual_language_embeddings_equal']
        comparisons.append(dict(kind='native_pixel_feature_teacher',case_index=index,**metric))
    for arm in ('ordinary','normalized','mass'):
        entry=original['arms'][arm];states,parameters[arm]=parameter_audit(torch,arm,entry,initial,config,directory,bindings)
        need([(r['arm'],r['step'],r['index']) for r in entry['microbatches']]==[(arm,s,i) for s in (1,2) for i in range(8)], 'Exact two accum8 updates required')
        for row in entry['microbatches']:
            index=row['index'];step=row['step'];key=f'{arm}_training_{step}_{index}';raw=packet(torch,row,bindings)
            need(row['n']==cases[index]['metadata']['n'] and row['sid']==cases[index]['metadata']['sid']
                 and row['loss']==raw['evidence']['loss'] and row['target_rows']==len(cases[index]['target_ids'])
                 and row['layer_forward_calls']==[1]*28 and row['max_reserved_bytes']>=row['max_allocated_bytes']>0
                 and math.isfinite(row['seconds']) and row['seconds']>0,'Measured teacher input/forward inventory differs')
            gradient=row['gradients'];expected=set(initial['adapter'])|(set() if arm=='ordinary' else {'memory.'+k for k in initial['memory']})
            need(set(gradient['present'])==set(gradient['finite'])==expected and gradient['all_present'] and gradient['all_finite']
                 and all(gradient['present'].values()) and all(gradient['finite'].values()),'Every profiled backward must retain complete finite gradients')
            teacher(raw,index,arm,key,(step,index) in ((1,0),(2,7)),True,entry['parameter_states'][str(step-1)],states)
        for index,slot in ((0,0),(4,1)):
            pair={}
            for phase in ('before','after'):
                ref=entry['roundtrip'][phase][slot];raw=packet(torch,ref,bindings);pair[phase]=raw
                teacher(raw,index,arm,f'{arm}_roundtrip_{phase}_{index}',True,False,entry['active_state'],states)
            comparisons.append(dict(kind='trained_checkpoint_teacher',arm=arm,case_index=index,**gpu_comparison(torch,pair['before'],pair['after'],exact=True)))
        prefix=None
        if arm!='ordinary':
            prefix=packet(torch,entry['prefix'],bindings)
            proof,observed=prefix_audit(torch,prefix,cases[4],entry,feature_records[4],states,config,data,arm)
            prefix_results.append(dict(arm=arm,**proof))
            metric=memory_audit(torch,observed,features[4],cases[4]['coordinates'],states['2']['memory'],arm,data,arm+'_prefix')
            metric.update(arm=arm,case_index=4,parameter_state=entry['active_state'],capture_file=entry['prefix']['file']);memory_results.append(metric)
        expected=[('ordinary',i,None) for i in range(5)] if arm=='ordinary' else [(kind,4,i) for kind,order in [('cold',range(4,8)),('split',range(4,8)),('reverse',reversed(range(4,8)))] for i in order]
        need([(r['kind'],r['case_index'],r['text_case_index']) for r in entry['natural']]==expected,'Exact natural/cold/split/reverse case order differs')
        lookup={}
        for row in entry['natural']:
            index=row['case_index'];text_index=index if row['text_case_index'] is None else row['text_case_index'];raw=packet(torch,row,bindings)
            need(row['arm']==arm and row['sid']==cases[index]['metadata']['sid'] and row['n']==cases[index]['metadata']['n']
                 and row['tokens']==len(raw['result']['generated_ids']) and row['transferred_answers_scored'] is False
                 and row['max_reserved_bytes']>=row['max_allocated_bytes']>0,'Natural input/metric ownership differs')
            key=f'{arm}_{row["kind"]}_{index}_{row["text_case_index"]}'
            result=natural_audit(torch,raw,cases[index],cases[text_index],entry,states,feature_records[index],prefix,modules,data,key)
            result.update(file=row['file'],sha256=row['sha256']);natural_results.append(result);save_progress()
            lookup[(row['kind'],row['text_case_index'])]=row
        if arm!='ordinary':
            for index in range(4,8):
                cold=packet(torch,lookup[('cold',index)],bindings)['result'];split=packet(torch,lookup[('split',index)],bindings)['result']
                reverse=packet(torch,lookup[('reverse',index)],bindings)['result']
                comparisons.append(dict(kind='memory_cold_split',arm=arm,text_case_index=index,**gpu_comparison(torch,cold,split,generation=True)))
                comparisons.append(dict(kind='memory_reverse_order',arm=arm,text_case_index=index,**gpu_comparison(torch,split,reverse,exact=True,generation=True)))
        del states,prefix
    need(len(teacher_results)==64 and len(memory_results)==42 and len(natural_results)==29 and len(comparisons)==24,'Complete fixed independent audit inventory differs')
    for observed,saved in zip(comparisons,original['comparisons']):
        need({k:v for k,v in observed.items() if k!='maximum_tv'}=={k:v for k,v in saved.items() if k!='maximum_tv'}
             and ((observed['maximum_tv'] is None and saved['maximum_tv'] is None) or
                  abs(observed['maximum_tv']-saved['maximum_tv'])<=1e-12),'Independent GPU parity reproduction differs')
    generated=sum(r['cpu_head_calls'] for r in natural_results)
    replay=[r for r in teacher_results if r['head'] is not None];teacher_rows=sum(r['head']['rows'] for r in replay)
    lengths=[len(cases[i]['target_ids']) for i in range(8)]
    need(len(replay)==22 and teacher_rows==8*(lengths[0]+lengths[4])+3*(lengths[0]+lengths[7]),'Exact predetermined teacher replay subset differs')
    counters=dict(model=64+generated,backbone=66+generated,visual=10,language=66+generated,norm=66+generated,
                  head=64+generated,backward=48,optimizer=6)
    need(counters==original['counters'] and original['decoder_layer_calls']==[66+generated]*28 and generated<=1450
         and original['head_rows']==8*(lengths[0]+lengths[4])+6*sum(lengths)+generated
         and 22+generated<=POLICY['cpu_head_calls_cap'] and teacher_rows+generated<=POLICY['cpu_head_rows_cap'], 'Actual GPU/CPU row/call accounting differs')
    timing=timing_audit(original);resources=allocation_audit(directory,out,descriptor['original_profile_passed'])
    cpu_passed=all(x['passed'] for x in teacher_results+memory_results+natural_results)
    cold_passed=all(x['passed'] for x in comparisons if x['kind'] in ('native_pixel_feature_teacher','trained_checkpoint_teacher'))
    cache_passed=all(x['passed'] for x in comparisons if x['kind'] in ('memory_cold_split','memory_reverse_order'))
    need(original['passed']==all(x['passed'] for x in comparisons),'Preserved combined producer gate differs')
    result=dict(protocol=PROTOCOL,passed=cpu_passed,completed=True,profile=descriptor,policy=POLICY,
        original_profile_passed=descriptor['original_profile_passed'],original_failure_preserved=not descriptor['original_profile_passed'],
        cpu_numerical_passed=cpu_passed,cold_training_components_passed=cpu_passed and cold_passed,cache_parity_passed=cache_passed,
        combined_software_parity_passed=cold_passed and cache_passed,all_scheduled_numerical_evidence_collected=True,
        teacher_audits=teacher_results,memory_audits=memory_results,natural_audits=natural_results,comparisons=comparisons,
        parameter_audits=parameters,prefix_audits=prefix_results,tests=tests,counters=counters,
        cpu_head_calls=22+generated,cpu_head_rows=teacher_rows+generated,teacher_replay_examples=22,
        timing=timing,resources=resources,no_efficacy_score=True,no_full_fit_release=True,
        source_sha256=own,inherited_source_sha256=inherited)
    save(out/'analysis.json',result)
    return result


def verify_report(summary_path):
    """JSON/hash-only handoff. CPU audit success never overrides cache failure."""
    path=Path(summary_path).resolve();summary=read(path)
    need(summary['protocol']==PROTOCOL and summary['passed'] is summary['completed'] is True
         and summary['source_sha256']==sources() and summary['no_full_fit_release'],'Passed independent CPU audit required')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'],'Independent analysis bytes changed')
    result=read(summary['analysis_file'])
    need(result['passed'] and result['cpu_numerical_passed'] and result['profile']==summary['profile']
         and result['no_full_fit_release'] and result['all_scheduled_numerical_evidence_collected'],'Complete CPU audit result required')
    for mapping_path,digest in ((summary['input_bindings_file'],summary['input_bindings_sha256']),(summary['artifacts_file'],summary['artifacts_sha256'])):
        need(sha(mapping_path)==digest,'Audit hash manifest changed')
        for file,expected in read(mapping_path).items():need(sha(file)==expected,'Bound audit evidence changed')
    for name,digest in {**summary['source_sha256'],**summary['inherited_source_sha256']}.items():
        need(sha(REPO/name)==digest and sha(path.parent/'source'/name.replace('/','_'))==digest,'Audit source closure changed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True);args=parser.parse_args()
    p.native.require_slurm();need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'CPU-only four-core audit required')
    import torch
    torch.set_num_threads(4);started=time.perf_counter();out=OUT/f'report_{os.environ["SLURM_JOB_ID"]}';data=DATA/out.name
    out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False);bindings={};progress={}
    save(out/'request.json',dict(profile=str(args.profile.resolve()),policy=POLICY,source_sha256=sources()))
    try:
        result=audit(torch,args,out,data,bindings,progress)
        elapsed=time.perf_counter()-started;need(elapsed<=1800,'Fixed1800-second CPU audit cap exceeded')
        save(out/'input_bindings.json',bindings)
        (out/'REPORT.md').write_text('Independent CPU numerical audit: '+str(result['passed'])+'.\n\nOriginal profile gate: '+str(result['original_profile_passed'])+
            '; cache parity: '+str(result['cache_parity_passed'])+'; cold/training components: '+str(result['cold_training_components_passed'])+
            '. The original failure and thresholds are preserved. All fixed captures were audited; no benchmark efficacy or full-fit release is inferred.\n')
        artifacts={str(file):sha(file) for root in (out,data) for file in root.iterdir() if file.is_file()}
        save(out/'artifacts.json',artifacts)
        summary=dict(protocol=PROTOCOL,passed=result['passed'],completed=True,profile=result['profile'],policy=POLICY,
            source_sha256=result['source_sha256'],inherited_source_sha256=result['inherited_source_sha256'],
            analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
            input_bindings_file=str(out/'input_bindings.json'),input_bindings_sha256=sha(out/'input_bindings.json'),
            artifacts_file=str(out/'artifacts.json'),artifacts_sha256=sha(out/'artifacts.json'),
            original_profile_passed=result['original_profile_passed'],cpu_numerical_passed=result['cpu_numerical_passed'],
            cold_training_components_passed=result['cold_training_components_passed'],cache_parity_passed=result['cache_parity_passed'],
            no_full_fit_release=True,elapsed_seconds=elapsed)
        save(out/'summary.json',summary)
        need(result['passed'],'Independent CPU numerical audit failed after complete evidence publication')
    except BaseException as exc:
        save(out/'failure_input_bindings.json',bindings)
        save(out/'failure.json',dict(protocol=PROTOCOL,passed=False,type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-started,
             source_sha256=sources(),input_bindings_file=str(out/'failure_input_bindings.json'),input_bindings_sha256=sha(out/'failure_input_bindings.json'),
             progress=progress,no_full_fit_release=True));raise


if __name__=='__main__':main()
