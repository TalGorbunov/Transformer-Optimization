"""Held ordinary frozen joint-image reference on the fixed270 fresh N16 scenes.

One exact images-first question, one ordinary native stream, no adapter or
broadcast. Profile observes actual full-norm versus selected-head shapes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
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
from scripts import native_vision_v7_runtime as native
from scripts import stage_native_identity_join_factor_fresh as stage
from scripts import stage_native_vision_v10_features as backend
from scripts.native_vision_reasoning_stream import GlobalLogitRecorder
from scripts.probe_native_vision_v2_prefix import fingerprint
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,object_sha,model_metadata
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_joint_baseline'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_joint_baseline')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_joint_baseline')
PROTOCOL='native_identity_join_frozen_joint_baseline'
NATIVE_REPORT=REPO/'outputs/native_aggregation_vlm/identity_join_factor_native_v2/report_443592/summary.json'
NATIVE_REPORT_SHA='d2f7a6ebe32020960be9e6811fb622f42c8effc3e35cd30e0578a419103d5a59'
PROPOSAL='docs/paper/NATIVE_AGGREGATION_JOINT_BASELINE_PROPOSAL.md'
PROPOSAL_SHA='891851f5ac930b09b9ba207a7aed476dad70b558129e64ffcaa630608baef94f'
JOBS={'profile':'identity_join_joint_baseline_profile','main':'identity_join_joint_baseline_main'}
POLICY=dict(protocol=PROTOCOL,n_frames=16,contexts=270,panels={'A':108,'B':108,'C':54},maximum_new_tokens=4,
    logits_to_keep=1,profile_trajectories=3,profile_call_cap=12,profile_vision_calls=3,main_call_cap=1080,main_vision_calls=270,
    profile_seconds=120,main_seconds=1200,campaign_seconds=1320,max_concurrent_gpus=1,
    cpu_check_seconds=300,cpu_release_seconds=300,cpu_report_seconds=300,head_tv_max=.02,head_argmax_exact=True,
    profile_rule='largest actual prepared width per A/B/C, first manifest row on ties',
    no_fitting=True,trainable_parameters=0,no_adapter=True,no_broadcast=True,no_prompt_wrapper=True,
    frozen_comparator=True,adaptation_matched=False,equal_calls_do_not_imply_equal_compute=True,
    no_extra_gpu_heads=True,main_cpu_heads=0,profile_cpu_head_cap=12,no_longer_lengths=True)
OWN=('scripts/evaluate_native_identity_join_joint_baseline.py','scripts/report_native_identity_join_joint_baseline.py',
    'slurm/native_identity_join_joint_baseline_check.sbatch','slurm/native_identity_join_joint_baseline_profile.sbatch',
    'slurm/native_identity_join_joint_baseline_release.sbatch','slurm/native_identity_join_joint_baseline_main.sbatch',
    'slurm/native_identity_join_joint_baseline_report.sbatch',PROPOSAL)


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or expected==h,'Bound joint input changed: '+str(path));bindings[str(path)]=h;return h


def sources():
    need(sha(REPO/PROPOSAL)==PROPOSAL_SHA,'Held ordinary joint protocol changed');return {name:sha(REPO/name) for name in OWN}


def inherited_sources():
    need(sha(NATIVE_REPORT)==NATIVE_REPORT_SHA,'Fixed native identity proof changed');proof=read(NATIVE_REPORT)
    values={**proof['inherited_source_sha256'],**proof['source_sha256']};need(len(values)==123,'Native source123 differs')
    for name,h in stage.source_hashes().items():need(name not in values or values[name]==h,'Inherited source conflict');values[name]=h
    for name,h in values.items():need(sha(REPO/name)==h,'Inherited source changed: '+name)
    return values


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        path=out/'source'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes());need(sha(path)==h,'Source snapshot changed')
    save(out/'source_hashes.json',frozen);save(out/'inherited_sources.json',dict(source_sha256=inherited_sources()));return frozen


def tensor_info(value):
    import torch
    v=value.detach().cpu().contiguous()
    return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(torch,v) for v in value)
    return value


def prepare_joint(processor,sample):
    native.require_slurm();need(set(sample)=={'sid','n_frames','question','image_files'} and sample['n_frames']==16
        and len(sample['image_files'])==16 and isinstance(sample['question'],str) and bool(sample['question'].strip()),'Exact four-key N16 input required')
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
        meta=dict(arm='joint',sid=sample['sid'],n_frames=16,question=sample['question'],question_sha256=object_sha(sample['question']),
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
         and bool((x['attention_mask']==1).all()) and x['image_grid_thw'].shape==(16,3),'One complete ordinary joint row required')
    need(m['n_frames']==16 and m['row_count']==1 and m['global_row']==0 and m['row_kinds']==['joint'] and m['local_elements']==0
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


def live_identity(torch,loaded,plan):
    from transformers import __version__ as transformers_version
    model=loaded.model;norm=native.native_contract(model,None);expected=plan['native_identity']
    need(fingerprint(loaded.processor,str(transformers_version))==expected['processor'] and backend.runtime_identity()==expected['runtime']
         and backend.native_api(loaded.processor)[2]==expected['native_api'] and model_metadata()==expected['model'],'Live joint backbone/processor/API differs')
    need(tensor_info(norm.weight)==expected['norm_weight'] and tensor_info(model.lm_head.weight)==expected['head_weight']
         and sha(Path(inspect.getfile(type(norm))))==expected['norm_source_sha256'] and float(norm.variance_epsilon)==expected['rms_norm_eps'],
         'Actual native norm/head identity differs')
    quant=model.config.quantization_config;quant=quant.to_dict() if hasattr(quant,'to_dict') else dict(quant)
    need(quant['load_in_4bit'] and quant['bnb_4bit_use_double_quant'] and quant['bnb_4bit_quant_type']=='nf4'
         and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','Actual NF4/double/BF16 backend differs')
    precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision());need(precision==plan['precision'],'Bound native precision changed')
    return dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=str(torch.version.cuda),quantization=quant,precision=precision)


def select_profiles(rows,prepared):
    result=[]
    for panel in ('A','B','C'):
        group=[row for row in rows if row['panel']==panel];need(len(group)==POLICY['panels'][panel],'Profile panel coverage differs')
        width=max(prepared[row['sid']]['metadata']['prompt_width'] for row in group)
        row=next(row for row in group if prepared[row['sid']]['metadata']['prompt_width']==width)
        result.append(dict(sid=row['sid'],panel=panel,n_frames=16,prompt_width=width,manifest_index=next(i for i,r in enumerate(rows) if r['sid']==row['sid'])))
    return result


def project(setup,times):
    need(len(times)==3 and all(math.isfinite(x) and x>0 for x in [setup]+times),'Three positive measured joint timings required')
    total=setup+1.25*270*max(times)+60.
    return dict(passed=total<=1200,setup_seconds=setup,maximum_four_token_seconds=max(times),multiplier=1.25,contexts=270,
        reserve_seconds=60.,projected_seconds=total,cap_seconds=1200,joint_only=True)


def self_test(torch):
    from gnnformer.data import build_prompt_inputs
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            self.messages=messages;self.kwargs=kwargs;return {'fixture':True}
    processor=Processor();frames=['image1','image2'];question='unchanged name question'
    need(build_prompt_inputs(processor,frames,question)=={'fixture':True} and processor.messages==[dict(role='user',content=
        [dict(type='image',image=x) for x in frames]+[dict(type='text',text=question)])],'Exact images-first helper changed the question')
    rows=[dict(sid=f'{panel}{i}',panel=panel,n_frames=16) for panel,count in POLICY['panels'].items() for i in range(count)]
    prepared={row['sid']:dict(metadata=dict(prompt_width=10 if row['sid'].endswith('0') else 20)) for row in rows}
    selected=select_profiles(rows,prepared);need([x['sid'] for x in selected]==['A1','B1','C1'],'Widest input/first tie selection differs')
    need(project(30.,[1.,1.,1.])['passed'] and not project(30.,[1.,4.,1.])['passed'],'Joint-only measured projection fixture failed')
    example=dict(sid='s',n_frames=16,question=question,image_files=[dict(path='p',sha256='0'*64,dimensions=[512,512],mode='RGB')]*16)
    need(stage.runtime_view(dict(example,gold='secret',states=['offline']))==example,'Gold entered ordinary joint inputs')
    return dict(passed=True,groups=4,unchanged_images_first_question=True,profile_selection_input_only=True,joint_timing_only=True,strict_model_input_boundary=True)


def check(args,out,frozen):
    import torch
    from transformers import AutoProcessor,__version__ as transformers_version
    torch.set_num_threads(4);bindings={};stage_path=args.stage_report.resolve();stage_path=stage_path/'summary.json' if stage_path.is_dir() else stage_path
    manifest=stage.verify_stage(stage_path);summary=read(stage_path);bind(stage_path,bindings);bind(summary['manifest_file'],bindings,summary['manifest_sha256'])
    need(manifest['native_report']['file']==str(NATIVE_REPORT) and manifest['native_report']['sha256']==NATIVE_REPORT_SHA,'Joint comparator must use the same fixed native/data proof')
    bind(NATIVE_REPORT,bindings,NATIVE_REPORT_SHA);proof=read(NATIVE_REPORT)
    for key in ('analysis','plan'):bind(proof[key+'_file'],bindings,proof[key+'_sha256'])
    parent=read(proof['plan_file']);native_ref=dict(file=str(NATIVE_REPORT),sha256=NATIVE_REPORT_SHA,
        analysis_file=proof['analysis_file'],analysis_sha256=proof['analysis_sha256'],plan_file=proof['plan_file'],plan_sha256=proof['plan_sha256'])
    stage_ref=dict(file=str(stage_path),sha256=sha(stage_path),manifest_file=summary['manifest_file'],manifest_sha256=summary['manifest_sha256'])
    rows=[row for panel in ('A','B','C') for row in manifest['splits'][f'fresh_{panel}_N16']['samples']]
    need(len(rows)==len({row['sid'] for row in rows})==270 and Counter(r['panel'] for r in rows)==POLICY['panels'],'All270 fixed fresh N16 rows required')
    save(out/'rows.json',rows);save(out/'dependencies.json',dict(native_report=native_ref,stage_report=stage_ref,input_bindings=bindings))
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    identity=parent['native_identity'];owner,rope,api=backend.native_api(processor)
    need(api==identity['native_api'] and fingerprint(processor,str(transformers_version))==identity['processor']
         and backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'Joint CPU native processor/runtime identity differs')
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);prepared={};layouts={}
    for i,row in enumerate(rows):
        need(row['n_frames']==16 and processor.tokenizer.encode(row['gold'],add_special_tokens=False)+[151645]==row['target_ids'],'Canonical name targets differ')
        for image in row['image_files']:bind(image['path'],bindings,image['sha256'])
        bundle=prepare_joint(processor,stage.runtime_view(row));layout=native.audit_layout(lambda **kw:rope(owner,**kw),bundle)
        path=data/f'bundle_{i:03d}.pt';torch.save(cpu_tree(torch,bundle),path)
        prepared[row['sid']]=dict(file=str(path),sha256=sha(path),metadata=bundle['metadata'],layout=layout['metadata']);layouts[row['sid']]=layout
        save(out/f'prepared_{i:03d}.json',dict(sid=row['sid'],file=str(path),sha256=sha(path),prompt_width=bundle['metadata']['prompt_width']))
    save(out/'prepared.json',prepared);layout_file=data/'layouts.pt';torch.save(layouts,layout_file);cases=select_profiles(rows,prepared);save(out/'profile_cases.json',cases)
    source_plan=parent['native_software_plan'];bind(source_plan['file'],bindings,source_plan['sha256']);precision=read(source_plan['file'])['precision']
    bind(parent['native_model_file'],bindings,parent['native_model_sha256'])
    tests=dict(driver=self_test(torch))
    from scripts.report_native_identity_join_joint_baseline import self_test as report_self_test
    tests['reporter']=report_self_test(torch);save(out/'selftests.json',tests)
    runtime_bindings={}
    for path in (out/'rows.json',out/'prepared.json',out/'profile_cases.json',layout_file):bind(path,runtime_bindings)
    for item in prepared.values():bind(item['file'],runtime_bindings,item['sha256'])
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,inherited_source_sha256=inherited_sources(),tests=tests,
        stage_report=stage_ref,native_report=native_ref,rows_file=str(out/'rows.json'),prepared_file=str(out/'prepared.json'),
        layout_file=str(layout_file),layout_sha256=sha(layout_file),profile_cases=cases,native_identity=identity,native_identity_sha256=parent['native_identity_sha256'],
        native_model_file=parent['native_model_file'],native_model_sha256=parent['native_model_sha256'],native_module_identity=parent['native_module_identity'],
        precision=precision,native_software_plan=source_plan,input_bindings=bindings,runtime_bindings=runtime_bindings,
        maximum_joint_prompt_width=max(v['metadata']['prompt_width'] for v in prepared.values()),no_model_or_head_forward=True,
        frozen_comparator=True,adaptation_matched=False,no_parallel_factor_input=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,contexts=270,profile_cases=cases,no_model_or_head_forward=True)


def verify_plan(path,ancestors=False,streaming=False):
    path=Path(path).resolve();plan=read(path);proof=read(path.parent/'summary.json')
    need(plan['protocol']==PROTOCOL and plan['policy']==POLICY and plan['source_sha256']==sources() and plan['inherited_source_sha256']==inherited_sources()
         and sha(path)==path.with_suffix('.sha256').read_text().strip() and proof['passed'] is proof['completed'] is True
         and proof['phase']=='check' and proof['plan_sha256']==sha(path) and proof['source_sha256']==plan['source_sha256'],'Passed frozen joint CPU plan required')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Joint source snapshot changed')
    need(read(path.parent/'inherited_sources.json')==dict(source_sha256=plan['inherited_source_sha256']),'Joint inherited source map changed')
    need(sha(Path(plan['prepared_file']))==plan['runtime_bindings'][plan['prepared_file']],'Prepared metadata changed')
    prepared=read(plan['prepared_file']);bundle_files={item['file'] for item in prepared.values()}
    need(len(prepared)==len(bundle_files)==270 and all(plan['runtime_bindings'][item['file']]==item['sha256'] for item in prepared.values()),
         'Complete270 bundle identity map differs')
    for file,h in plan['runtime_bindings'].items():
        if streaming and file in bundle_files:continue  # Verified immediately before the actual load, within scene timing.
        need(sha(Path(file))==h,'Consumed joint input changed: '+file)
    if ancestors:
        for file,h in plan['input_bindings'].items():need(sha(Path(file))==h,'Bound joint ancestry changed: '+file)
    return plan


def allocations(raw):
    result=[]
    for line in raw.splitlines():
        fields=line.split('|');need(len(fields)==9,'Unexpected accounting columns')
        job,name,partition,state,exit_code,seconds,tres,start,end=fields
        if name not in JOBS.values():continue
        generic=re.findall(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',tres)
        fallback=re.findall(r'(?:^|,)gpu(?::[^=,]+)?=(\d+)(?=,|$)',tres)
        gpus=int(generic[0]) if generic else sum(map(int,typed)) if typed else sum(map(int,fallback))
        result.append(dict(job_id=job,name=name,partition=partition,state=state,exit_code=exit_code,seconds=int(seconds),gpus=gpus,
            gpu_seconds=int(seconds)*gpus,alloc_tres=tres,start=start,end=end))
    return result


def single_attempt_guard(out,phase):
    need(os.environ.get('SLURM_JOB_NAME')==JOBS[phase],'Exact registered joint job name required')
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
        '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    raw=subprocess.run(command,capture_output=True,text=True,check=True).stdout;file=out/'launch_sacct.psv';file.write_text(raw)
    save(out/'launch_accounting.json',dict(command=command,file=str(file),sha256=sha(file)));rows=allocations(raw);counts=Counter(r['name'] for r in rows)
    need(all(v==1 for v in counts.values()) and all(r['partition']=='gpu' and r['gpus']==1 for r in rows)
         and all(r['job_id']==os.environ['SLURM_JOB_ID'] for r in rows if r['name']==JOBS[phase]),'Failed/zero attempts count; duplicate joint phase forbidden')
    need(sum(r['gpu_seconds'] for r in rows)<=1320,'Joint campaign cap exceeded')


def verify_release(path,plan_path,plan):
    path=Path(path).resolve();release=read(path);summary=read(path.parent/'summary.json')
    need(summary['passed'] is summary['completed'] is True and summary['phase']=='release' and summary['release_file']==str(path)
         and summary['release_sha256']==sha(path) and release['plan_file']==str(Path(plan_path).resolve()) and release['plan_sha256']==sha(Path(plan_path))
         and release['source_sha256']==plan['source_sha256'] and release['inherited_source_sha256']==plan['inherited_source_sha256']
         and release['projection']['passed'] is True and release['projection']['projected_seconds']<=1200
         and release['reserved_main_gpu_seconds']==1200,'Passed measured joint profile/release required')
    return dict(file=str(path),sha256=sha(path),summary_file=str(path.parent/'summary.json'),summary_sha256=sha(path.parent/'summary.json'))


def run(args,out,frozen,started):
    import torch
    from gnnformer.runtime import load_runtime
    torch.set_num_threads(4);phase='profile' if args.profile else 'main';single_attempt_guard(out,phase)
    args.plan=args.plan.resolve();plan=verify_plan(args.plan,streaming=True);release=None if args.profile else verify_release(args.main_release,args.plan,plan)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required for actual joint measurement')
    loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=loaded.model.eval().requires_grad_(False)
    hardware=json.loads(json.dumps(live_identity(torch,loaded,plan)));norm=model.model.language_model.norm;head=model.lm_head
    native_table=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight));versions={k:v._version for k,v in model.named_parameters()}
    rows=read(plan['rows_file']);prepared=read(plan['prepared_file']);data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    cases=plan['profile_cases'] if args.profile else [dict(sid=r['sid'],panel=r['panel'],n_frames=16) for r in rows]
    config=dict(protocol=PROTOCOL,policy=POLICY,phase=phase,run_id=out.name,slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=frozen,inherited_source_sha256=plan['inherited_source_sha256'],plan_file=str(args.plan),plan_sha256=sha(args.plan),
        native_report=plan['native_report'],stage_report=plan['stage_report'],native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
        native_weight_identity=native_table,hardware=hardware,cases=cases,data_directory=str(data),main_release=release,
        frozen_comparator=True,adaptation_matched=False,equal_calls_do_not_imply_equal_compute=True,no_adapter=True)
    save(out/'config.json',config);torch.cuda.synchronize();setup=time.perf_counter()-started;records=[];counts=Counter()
    for i,case in enumerate(cases):
        tick=time.perf_counter();item=prepared[case['sid']];need(sha(Path(item['file']))==item['sha256'],'Prepared joint image bundle changed')
        bundle=torch.load(item['file'],map_location='cpu',weights_only=True);validate_joint(torch,bundle);prep=time.perf_counter()-tick
        work=time.perf_counter();evidence={}
        try:result=generate_joint(model,loaded.processor,bundle,native_identity_sha256=plan['native_identity_sha256'],capture_head=args.profile,evidence=evidence)
        except BaseException:
            path=data/f'trajectory_{i:03d}_partial.pt';torch.save(cpu_tree(torch,dict(case=case,evidence=evidence,partial_outputs_retained=True)),path)
            save(out/f'trajectory_{i:03d}_failure.json',dict(case=case,file=str(path),sha256=sha(path),partial_outputs_retained=True));raise
        path=data/f'trajectory_{i:03d}.pt';torch.save(cpu_tree(torch,dict(schema_version=1,case=case,result=result)),path)
        digest=sha(path);save(path.with_suffix('.json'),dict(case=case,raw_file=str(path),raw_sha256=digest,validation_pending=True))
        need(result['metadata']['layout']==item['layout'] and result['metadata']['sid']==case['sid'] and result['metadata']['n_frames']==16,
             'Actual ordinary joint source/layout differs')
        steps=len(result['generated_ids']);expected=dict(model=steps,visual=1,language=steps,norm=steps,head=steps,broadcast=0,fusion=0,conditioning=0,probe_head=0)
        need(result['counters']==expected and result['parameter_versions_unchanged'] is result['hooks_removed'] is result['rope_restored'] is True,'Joint execution changed model/controller state')
        counts.update(expected);save(out/f'progress_{i+1:03d}.json',dict(processed=i+1,case=case,raw_file=str(path),raw_sha256=digest,accuracy_not_scored=True))
        work_seconds=time.perf_counter()-work
        records.append(dict(index=i,case=case,raw_file=str(path),raw_sha256=digest,prepared_file=item['file'],prepared_sha256=item['sha256'],generated_tokens=steps,
            preprocessing_seconds=prep,work_seconds=work_seconds,four_token_seconds=prep+work_seconds*4/steps))
    expected_scenes=3 if args.profile else 270;cap=12 if args.profile else 1080
    need(len(records)==counts['visual']==expected_scenes and counts['model']<=cap and counts['fusion']==counts['conditioning']==counts['broadcast']==0,
         'Joint phase call inventory differs')
    need(versions=={k:v._version for k,v in model.named_parameters()} and all(not v.requires_grad and v.grad is None for v in model.parameters())
         and native_table==dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight)),'Joint phase changed the frozen native model')
    save(out/'raw_manifest.json',dict(completed=True,n=len(records),rows=records,all_raw_retained_before_scoring=True))
    endpoint=dict(passed=True,native_weights_before=native_table,native_weights_after=dict(norm=tensor_info(norm.weight),head=tensor_info(head.weight)),
        native_parameter_versions_unchanged=True,counters=dict(counts),no_optimizer_or_training=True,no_adapter=True)
    save(out/'endpoint.json',endpoint);need(time.perf_counter()-started<=POLICY[phase+'_seconds'],'Joint native phase exceeded its fixed cap')
    return dict(**config,passed=True,completed=True,raw_manifest_file=str(out/'raw_manifest.json'),raw_manifest_sha256=sha(out/'raw_manifest.json'),
        endpoint_file=str(out/'endpoint.json'),endpoint_sha256=sha(out/'endpoint.json'),counters=dict(counts),setup_seconds=setup,trajectories=len(records),
        all_raw_retained_before_scoring=True,no_accuracy_selection=True,extra_gpu_head_calls=0,no_fitting=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true');action.add_argument('--profile',action='store_true');action.add_argument('--main',action='store_true')
    parser.add_argument('--stage-report',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--main-release',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=not args.check)
    if args.check:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS') and args.stage_report is not None
        and args.plan is args.main_release is None,'CPU preparation requires the passed fresh stage only')
    else:need(args.plan is not None and args.stage_report is None and (args.main_release is None if args.profile else args.main_release is not None),
        'Profile/main must consume the exact plan and applicable measured release')
    phase='check' if args.check else 'profile' if args.profile else 'main';out=OUT/f'{phase}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen=snapshot(out)
    save(out/'request.json',dict(phase=phase,stage_report=None if args.stage_report is None else str(args.stage_report),
        plan=None if args.plan is None else str(args.plan),main_release=None if args.main_release is None else str(args.main_release),source_sha256=frozen))
    try:
        value=check(args,out,frozen) if args.check else run(args,out,frozen,started)
        elapsed=time.perf_counter()-started;need(elapsed<=POLICY['cpu_check_seconds' if args.check else phase+'_seconds'],'Registered joint phase cap exceeded')
        need(sources()==frozen and inherited_sources()==value['inherited_source_sha256'],'Joint source changed during execution')
        value['elapsed_seconds']=elapsed;save(out/'summary.json',value);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,
            elapsed_seconds=time.perf_counter()-started,partial_outputs_retained=True));raise


if __name__=='__main__':main()
