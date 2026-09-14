"""Frozen N1 teacher computations in ordinary independent native batch rows.

This is a software oracle, not an aggregate-count method. Complete original
one-image prompts are left padded; images/grids stay in row order. No hidden
stream fencing, branch, generation, cache reuse, or learned parameter is added.
All numerical work requires Slurm, including the config-only CPU mRoPE check.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.stage_native_vision_v6_teacher import (
    CATEGORIES,MODEL,actual_layout,model_metadata,need,object_sha,prepared,read,save,
    sha,tensor_sha,verify_inputs,verify_plan as verify_teacher_plan,
)
from scripts.cache_native_vision_v6_teacher import probabilities

TEACHER=Path('/mnt/data/gabriele/gnn_transformer/v6_local_teacher/teacher_cache.json')
DATA=Path('/mnt/data/gabriele/gnn_transformer/parallel_local')
OUT=REPO/'outputs/native_aggregation_vlm/parallel_local'
OWN=('scripts/probe_native_vision_parallel_local.py',
     'slurm/native_vision_parallel_local_check.sbatch','slurm/native_vision_parallel_local.sbatch')
SEED=20260922
TOLERANCE=.02
INPUT_KEYS={'input_ids','attention_mask','pixel_values','image_grid_thw'}


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',sources())


def index(out,title,entries):
    (out/'INDEX.md').write_text('# '+title+'\n\n'+'\n'.join(f'- [{label}]({path})' for label,path in entries)+'\n')


def teacher_inputs():
    need(sha(TEACHER)==TEACHER.with_suffix('.sha256').read_text().strip(),'Teacher cache sidecar differs')
    cache=read(TEACHER);plan_path=Path(cache['plan_file']);teacher=verify_teacher_plan(plan_path)
    need(cache['schema_version']==1 and cache['passed_quality_gate'] is True
         and cache['plan_sha256']==sha(plan_path),'Require the complete eligible immutable V6 teacher')
    for name in ('source_sha256','source_files','model','processor','runtime','image_processor_settings',
                 'resize','quantization','attention','temperature'):
        need(cache[name]==teacher[name],'Teacher cache/plan policy differs: '+name)
    need(set(cache['targets'])==set(teacher['pairs']) and len(cache['targets'])==9980
         and cache['scenes']=={sid:x['pair_ids'] for sid,x in teacher['scenes'].items()},
         'Incomplete canonical training teacher cache')
    audit=read(teacher['audit_file']);labels=audit['labels']
    need(set(labels)==set(teacher['pairs']),'Teacher audit coverage differs')
    by_category={name:sorted(pid for pid,entry in labels.items() if entry['category']==name) for name in CATEGORIES}
    need(all(len(x)>=16 for x in by_category.values()),'Insufficient fixed training category coverage')
    ids=[by_category[name][i] for i in range(16) for name in CATEGORIES]
    need(len(set(ids))==64 and Counter(labels[pid]['category'] for pid in ids)==Counter({name:16 for name in CATEGORIES}),
         'Expected64 unique pairs,16 per category')
    for pid in ids:
        pair=teacher['pairs'][pid];target=cache['targets'][pid]
        need(target['image_sha256']==pair['image_sha256'] and target['question']==pair['question']
             and target['question_sha256']==object_sha(pair['question']) and target['layout_id']==pair['layout_id']
             and target['input_ids_sha256']==teacher['layouts'][pair['layout_id']]['input_ids_sha256'],
             'Archived target/input identity differs')
        expected=probabilities(target['logit0'],target['logit1'],target['log_normalizer'])
        need(len(target['probabilities'])==3 and all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
             for a,b in zip(expected,target['probabilities'])),'Archived raw probability identity differs')
    binding=dict(cache_file=str(TEACHER),cache_sha256=sha(TEACHER),plan_file=str(plan_path),plan_sha256=sha(plan_path),
                 audit_file=teacher['audit_file'],audit_sha256=teacher['audit_sha256'],source_sha256=teacher['source_sha256'])
    return teacher,cache,labels,ids,binding


def native_api(processor):
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    config=AutoConfig.from_pretrained(str(MODEL),trust_remote_code=True)
    need(config.model_type=='qwen2_5_vl','Wrong native model class')
    fn=Qwen2_5_VLModel.get_rope_index
    paths={str(Path(inspect.getfile(x)).resolve()) for x in (Qwen2_5_VLModel,type(processor),type(processor.image_processor))}
    binding=dict(source_sha256={path:sha(path) for path in sorted(paths)},
                 rope_method=fn.__qualname__,config_sha256=sha(MODEL/'config.json'))
    return SimpleNamespace(config=config),fn,binding


def rope(fn,owner,inputs):
    return fn(owner,input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],
              attention_mask=inputs['attention_mask'])


def pack(torch,records,pad_id):
    need(len(records)>0,'Empty independent batch')
    length=max(x['input_ids'].shape[1] for x in records)
    ids=torch.full((len(records),length),pad_id,dtype=torch.int64)
    mask=torch.zeros_like(ids)
    for i,row in enumerate(records):
        need(set(row)==INPUT_KEYS and row['input_ids'].ndim==2 and row['input_ids'].shape[0]==1
             and row['attention_mask'].shape==row['input_ids'].shape
             and bool((row['attention_mask']==1).all()) and row['image_grid_thw'].shape==(1,3),
             'Only complete unpadded N1 teacher rows may be packed')
        n=row['input_ids'].shape[1];ids[i,-n:]=row['input_ids'][0];mask[i,-n:]=1
    return dict(input_ids=ids,attention_mask=mask,pixel_values=torch.cat([x['pixel_values'] for x in records]),
                image_grid_thw=torch.cat([x['image_grid_thw'] for x in records]))


def tensors_equal(torch,a,b):
    return set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)


def packed_identity(torch,inputs,ids,rows,positions,deltas,single_positions,single_deltas,pad_id):
    batch,length=inputs['input_ids'].shape
    need(batch==len(ids) and positions.shape==(3,batch,length) and deltas.shape==(batch,1),'Native mRoPE batch shape differs')
    details=[];cursor=0
    for i,pid in enumerate(ids):
        original=rows[pid];n=original['input_ids'].shape[1];left=length-n
        need(torch.equal(inputs['input_ids'][i,left:],original['input_ids'][0])
             and bool((inputs['input_ids'][i,:left]==pad_id).all())
             and bool((inputs['attention_mask'][i,:left]==0).all())
             and bool((inputs['attention_mask'][i,left:]==1).all()),'Left-padding/complete prompt identity differs')
        need(torch.equal(positions[:,i,left:],single_positions[pid][:,0,:])
             and bool((positions[:,i,:left]==1).all())
             and int(deltas[i,0])+left==int(single_deltas[pid][0,0]),'Unpadded native mRoPE identity differs')
        count=original['pixel_values'].shape[0]
        need(torch.equal(inputs['pixel_values'][cursor:cursor+count],original['pixel_values'])
             and torch.equal(inputs['image_grid_thw'][i],original['image_grid_thw'][0]),
             'Image patch/grid row concatenation differs')
        details.append(dict(pair_id=pid,left_padding=left,prompt_tokens=n,pixel_slice=[cursor,cursor+count],
                            unpadded_input_ids_sha256=object_sha(original['input_ids'][0].tolist()),
                            unpadded_position_ids_sha256=tensor_sha(positions[:,i,left:]),
                            native_rope_delta=int(deltas[i,0]),single_rope_delta=int(single_deltas[pid][0,0])))
        cursor+=count
    need(cursor==inputs['pixel_values'].shape[0] and bool((inputs['attention_mask'][:,-1]==1).all()),
         'Unused pixels or padding at next-token readout')
    return dict(batch_size=batch,padded_tokens=length,rows=details,
                tensors={name:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value))
                         for name,value in inputs.items()},position_ids_sha256=tensor_sha(positions),
                rope_deltas=deltas.tolist(),all_unpadded_tokens_pixels_positions_exact=True)


def processor_batch(processor,pairs):
    from PIL import Image
    from gnnformer.data import build_count_prompt
    frames=[];messages=[]
    try:
        for pair in pairs:
            with Image.open(pair['image_path']) as original:
                rgb=original.convert('RGB')
                try:frame=rgb.resize((392,392))
                finally:rgb.close()
            frames.append(frame)
            messages.append([dict(role='user',content=[dict(type='image',image=frame),
                dict(type='text',text=build_count_prompt(pair['question'],1))])])
        return dict(processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                    return_dict=True,return_tensors='pt',padding=True))
    finally:
        for frame in frames:frame.close()


def scalar_stats(row):
    difference=row['logit1']-row['logit0']
    conditional=1/(1+math.exp(-difference)) if difference>=0 else math.exp(difference)/(1+math.exp(difference))
    probs=probabilities(row['logit0'],row['logit1'],row['log_normalizer'])
    return dict(binary_prediction=int(difference>0),conditional_p1=conditional,numeric_mass=math.fsum(probs[:2]))


def comparison(row,reference,gold):
    a,b=scalar_stats(row),scalar_stats(reference)
    delta=abs(a['conditional_p1']-b['conditional_p1']);mass=abs(a['numeric_mass']-b['numeric_mass'])
    prediction_match=a['binary_prediction']==b['binary_prediction']
    truth_correct=a['binary_prediction']==int(gold)
    return dict(binary_prediction=a['binary_prediction'],reference_binary_prediction=b['binary_prediction'],
                binary_prediction_exact=prediction_match,truth_correct=truth_correct,
                conditional_p1=a['conditional_p1'],reference_conditional_p1=b['conditional_p1'],
                conditional_p1_absolute_difference=delta,numeric_mass=a['numeric_mass'],
                reference_numeric_mass=b['numeric_mass'],numeric_mass_absolute_difference=mass,
                logit0_difference=row['logit0']-reference['logit0'],
                logit1_difference=row['logit1']-reference['logit1'],
                log_normalizer_difference=row['log_normalizer']-reference['log_normalizer'],
                passed=prediction_match and delta<=TOLERANCE and mass<=TOLERANCE)


def gate(comparisons):
    need(comparisons,'No comparisons')
    return dict(passed=all(x['passed'] for x in comparisons),n=len(comparisons),
                binary_prediction_mismatches=sum(not x['binary_prediction_exact'] for x in comparisons),
                truth_errors=sum(not x['truth_correct'] for x in comparisons),
                maximum_conditional_p1_difference=max(x['conditional_p1_absolute_difference'] for x in comparisons),
                maximum_numeric_mass_difference=max(x['numeric_mass_absolute_difference'] for x in comparisons),
                conditional_p1_tolerance=TOLERANCE,numeric_mass_tolerance=TOLERANCE)


def self_test(torch):
    first=dict(input_ids=torch.tensor([[4,5]]),attention_mask=torch.ones(1,2,dtype=torch.long),
               pixel_values=torch.tensor([[1.,2.],[3.,4.]]),image_grid_thw=torch.tensor([[1,1,2]]))
    second=dict(input_ids=torch.tensor([[6,7,8]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                pixel_values=torch.tensor([[5.,6.]]),image_grid_thw=torch.tensor([[1,1,1]]))
    mixed=pack(torch,[first,second],0)
    need(mixed['input_ids'].tolist()==[[0,4,5],[6,7,8]] and mixed['attention_mask'].tolist()==[[0,1,1],[1,1,1]],
         'Left-padding test failed')
    reverse=pack(torch,[second,first],0)
    need(reverse['input_ids'].tolist()==[[6,7,8],[0,4,5]] and
         reverse['pixel_values'].tolist()==[[5.,6.],[1.,2.],[3.,4.]],'Reverse image/prompt order failed')
    reference=dict(logit0=0.,logit1=4.,log_normalizer=4.1)
    same=comparison(reference,reference,1);need(same['passed'],'Identity should pass')
    shifted={name:value+100 for name,value in reference.items()}
    need(comparison(shifted,reference,1)['passed'],'Common full-vocabulary logit shift should preserve gate')
    wrong=dict(reference,logit0=8.,log_normalizer=8.1)
    need(not comparison(wrong,reference,1)['passed'],'Changed binary judgment accepted')
    badmass=dict(reference,log_normalizer=5.)
    need(not comparison(badmass,reference,1)['passed'],'Changed numeric mass accepted')
    other_probability=dict(logit0=math.log(.25),logit1=math.log(.75),log_normalizer=0.)
    reference_probability=dict(logit0=math.log(.3),logit1=math.log(.7),log_normalizer=0.)
    need(not comparison(other_probability,reference_probability,1)['passed'],
         'Conditional probability drift with unchanged judgment/mass was accepted')
    boundary=dict(same,conditional_p1_absolute_difference=.02,numeric_mass_absolute_difference=.02)
    boundary['passed']=boundary['binary_prediction_exact'] and boundary['truth_correct'] and \
        boundary['conditional_p1_absolute_difference']<=TOLERANCE and boundary['numeric_mass_absolute_difference']<=TOLERANCE
    need(gate([boundary])['passed'],'Inclusive fixed boundaries failed')
    return dict(passed=True,tests=['left_padding','reverse_pixel_order','identity','common_logit_shift',
                                  'binary_mismatch','numeric_mass_mismatch','conditional_probability_drift','inclusive_bounds'])


def check():
    import torch
    import transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter();tests=self_test(torch)
    teacher,cache,labels,ids,binding=teacher_inputs();frozen=sources()
    need(str(torch.__version__)==teacher['runtime']['torch_version'] and
         str(transformers.__version__)==teacher['runtime']['transformers_version'],'CPU runtime differs from teacher')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==teacher['processor'],'CPU processor differs from teacher')
    # A separate processor owns temporary padding configuration for parity checks.
    batched_processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(batched_processor,str(transformers.__version__))==teacher['processor'],'Batch processor differs')
    batched_processor.tokenizer.padding_side='left'
    pad_id=processor.tokenizer.pad_token_id;need(type(pad_id)is int,'Native tokenizer has no padding ID')
    owner,fn,native=native_api(processor)
    need(pad_id not in (owner.config.image_token_id,owner.config.video_token_id,owner.config.vision_start_token_id),
         'Padding ID is a visual marker')
    rows={};positions={};deltas={};records={}
    for pid in ids:
        pair=teacher['pairs'][pid];path=Path(pair['image_path'])
        need(path.stat().st_size==pair['image_bytes'] and sha(path)==pair['image_sha256'],'Selected source image changed')
        inputs=prepared(processor,pair);verify_inputs(inputs,pair,teacher)
        need(set(inputs)==INPUT_KEYS,'Unexpected processor inputs')
        pos,delta=rope(fn,owner,inputs);rows[pid]=inputs;positions[pid]=pos;deltas[pid]=delta
        records[pid]=dict(pair,category=labels[pid]['category'],audit_gold=labels[pid]['gold'],
                         archived_target=cache['targets'][pid],layout=teacher['layouts'][pair['layout_id']],
                         position_ids_sha256=tensor_sha(pos),single_rope_delta=delta.tolist())
    cases=[dict(case=f'batch_{n}',pair_ids=ids[:n]) for n in (1,8,16,64)]
    cases.append(dict(case='batch_64_reverse',pair_ids=ids[::-1]))
    for case in cases:
        order=case['pair_ids'];inputs=pack(torch,[rows[pid] for pid in order],pad_id)
        native_inputs=processor_batch(batched_processor,[teacher['pairs'][pid] for pid in order])
        need(tensors_equal(torch,inputs,native_inputs),'Manual independent-row packing differs from native batch processor')
        pos,delta=rope(fn,owner,inputs)
        case['identity']=packed_identity(torch,inputs,order,rows,pos,delta,positions,deltas,pad_id)
        case['ordinary_batched_processor_exact']=True
    need(sources()==frozen and teacher_inputs()[4]==binding,'Sources/teacher changed during CPU check')
    job=os.environ['SLURM_JOB_ID'];out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False);blob=data/'prepared_inputs.pt'
    torch.save(dict(schema_version=1,pair_ids=ids,inputs=rows,position_ids=positions,rope_deltas=deltas),blob)
    plan=dict(schema_version=1,protocol='native_parallel_complete_N1_teacher_prompts',source_sha256=frozen,
              teacher_binding=binding,model=teacher['model'],runtime=teacher['runtime'],processor=teacher['processor'],
              image_processor_settings=teacher['image_processor_settings'],native_api=native,
              resize=392,quantization='nf4_double_bf16',attention='sdpa',seed=SEED,
              count_token_ids=teacher['count_token_ids'],pad_token_id=pad_id,padding_side='left',
              selected_pair_ids=ids,records=records,cases=cases,fresh_serial_pair_ids=ids[:4],
              execution_order='four fresh serial references, then batch1,8,16,64,reverse64',
              prepared_inputs_file=str(blob),prepared_inputs_sha256=sha(blob),
              selection='first16 lexicographically sorted training pair IDs in each semantic category, interleaved positive/char_only/room_only/neither; no output selection',
              threshold=dict(binary_prediction_exact=True,conditional_p1_absolute_difference=.02,
                             numeric_mass_absolute_difference=.02,scope='every batch/archived and serial/archived comparison; truth and fresh-serial overlap are descriptive'),
              tests=tests,slurm_job_id=job,seconds=time.perf_counter()-started,
              scope='software fidelity of independent complete N1 prompts; no aggregate count, generation or cache claim')
    path=out/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    verify_frozen(path)
    save(out/'summary.json',dict(passed=True,plan=str(path),plan_sha256=sha(path),pairs=64,
                               batch_responses=153,fresh_serial_responses=4,source_sha256=frozen,seconds=plan['seconds']))
    index(out,'Parallel local CPU freeze',[('Plan','plan.json'),('Summary','summary.json'),('Sources','source/')])
    index(data,'Frozen local prompt tensors',[('Prepared inputs','prepared_inputs.pt')])
    print(json.dumps(dict(passed=True,plan=str(path),plan_sha256=sha(path),seconds=plan['seconds'])),flush=True)


def verify_frozen(path):
    path=Path(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'Parallel plan sidecar differs')
    plan=read(path)
    need(plan['schema_version']==1 and plan['protocol']=='native_parallel_complete_N1_teacher_prompts'
         and plan['source_sha256']==sources(),'Parallel plan/source differs')
    teacher,cache,labels,ids,binding=teacher_inputs()
    need(binding==plan['teacher_binding'] and ids==plan['selected_pair_ids'] and plan['fresh_serial_pair_ids']==ids[:4],
         'Teacher/selection changed since CPU freeze')
    for key in ('model','runtime','processor','image_processor_settings','count_token_ids'):
        need(plan[key]==teacher[key],'Parallel teacher policy differs: '+key)
    expected=[(f'batch_{n}',ids[:n]) for n in (1,8,16,64)]+[('batch_64_reverse',ids[::-1])]
    need([(x['case'],x['pair_ids']) for x in plan['cases']]==expected,'Batch schedule changed')
    need(plan['padding_side']=='left' and plan['resize']==392 and plan['quantization']=='nf4_double_bf16'
         and plan['attention']=='sdpa' and plan['seed']==SEED,'Registered runtime policy differs')
    for pid in ids:
        record=plan['records'][pid];pair=teacher['pairs'][pid]
        need(all(record[k]==v for k,v in pair.items()) and record['archived_target']==cache['targets'][pid]
             and record['category']==labels[pid]['category'] and record['audit_gold']==labels[pid]['gold'],
             'Frozen input/reference/audit changed')
    for filename,digest in plan['native_api']['source_sha256'].items():need(sha(filename)==digest,'Installed native source changed')
    need(sha(plan['prepared_inputs_file'])==plan['prepared_inputs_sha256'],'Prepared CPU tensor blob changed')
    return plan,teacher


def gpu(args):
    plan,teacher=verify_frozen(args.plan)
    cpu_summary=read(Path(args.plan).parent/'summary.json')
    need(cpu_summary['passed'] is True and cpu_summary['plan_sha256']==sha(args.plan)
         and cpu_summary['source_sha256']==plan['source_sha256']
         and read(Path(args.plan).parent/'source_hashes.json')==plan['source_sha256'],
         'A completed matching CPU freeze is required before GPU execution')
    for name,digest in plan['source_sha256'].items():
        need(sha(Path(args.plan).parent/'source'/name.replace('/','_'))==digest,'CPU source snapshot changed')
    import torch
    import transformers
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn
    need(torch.cuda.is_available(),'Expected one Slurm GPU')
    need(str(torch.__version__)==plan['runtime']['torch_version'] and
         str(transformers.__version__)==plan['runtime']['transformers_version'],'GPU runtime differs')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    tests=self_test(torch);started=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'run_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=DATA/f'run_{job}';data.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    index(out,'Independent local batched native prefills',[('Configuration','config.json'),('Raw comparisons','rows.jsonl'),
                ('Summary after completion','summary.json'),('Report after completion','REPORT.md'),('Frozen plan','plan.json')])
    index(data,'Parallel local oracle artifacts',[('Actual input identities','input_identities.json')])
    save(out/'config.json',dict(schema_version=1,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        source_sha256=plan['source_sha256'],teacher_binding=plan['teacher_binding'],model=plan['model'],
        runtime=plan['runtime'],processor=plan['processor'],slurm_job_id=job,tests=tests))
    blob=torch.load(plan['prepared_inputs_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==1 and blob['pair_ids']==plan['selected_pair_ids'],'Prepared blob order differs')
    rows=blob['inputs'];single_positions=blob['position_ids'];single_deltas=blob['rope_deltas']
    for pid in plan['selected_pair_ids']:
        pair=teacher['pairs'][pid];verify_inputs(rows[pid],pair,teacher)
        need(sha(pair['image_path'])==pair['image_sha256'] and
             tensor_sha(single_positions[pid])==plan['records'][pid]['position_ids_sha256'],
             'Image or single mRoPE identity changed')
    load_start=time.perf_counter();runtime=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.requires_grad_(False);model.eval()
    need(not any(p.requires_grad for p in model.parameters()),'Oracle model must be frozen')
    need(fingerprint(runtime.processor,str(transformers.__version__))==plan['processor'],'GPU processor differs')
    _,_,native=native_api(runtime.processor);need(native==plan['native_api'],'Native mRoPE implementation differs')
    fn=get_rope_index_fn(model);need(fn.__func__.__qualname__==native['rope_method'],'Actual model mRoPE method differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    vision=model.model.visual;language=model.model.language_model
    counters=dict(model=0,vision=0,language=0);captured={}
    def model_hook(*_):counters['model']+=1
    def vision_hook(*_):counters['vision']+=1
    def language_hook(module,args,kwargs):
        counters['language']+=1
        need(kwargs.get('position_ids') is not None and kwargs.get('attention_mask') is not None,
             'Native language forward omitted actual positions/mask')
        captured['position_ids']=kwargs['position_ids'].detach().cpu().clone()
        captured['attention_mask']=kwargs['attention_mask'].detach().cpu().clone()
    hooks=[model.register_forward_pre_hook(model_hook),vision.register_forward_pre_hook(vision_hook),
           language.register_forward_pre_hook(language_hook,with_kwargs=True)]
    serial={};all_rows=[];case_reports=[];identities=[]
    todo=[dict(case=f'fresh_serial_{i}',pair_ids=[pid],fresh_serial=True)
          for i,pid in enumerate(plan['fresh_serial_pair_ids'])]+[dict(x,fresh_serial=False) for x in plan['cases']]
    raw=out/'rows.jsonl'
    try:
        with raw.open('x') as stream:
            for case in todo:
                begin=time.perf_counter();order=case['pair_ids']
                if case['fresh_serial']:
                    inputs=prepared(runtime.processor,teacher['pairs'][order[0]])
                    need(tensors_equal(torch,inputs,rows[order[0]]),'Fresh serial processor differs from frozen CPU input')
                else:inputs=pack(torch,[rows[pid] for pid in order],plan['pad_token_id'])
                expected_positions,expected_deltas=fn(input_ids=inputs['input_ids'],image_grid_thw=inputs['image_grid_thw'],
                                                     attention_mask=inputs['attention_mask'])
                identity=packed_identity(torch,inputs,order,rows,expected_positions,expected_deltas,
                                         single_positions,single_deltas,plan['pad_token_id'])
                if not case['fresh_serial']:need(identity==case['identity'],'Actual batch differs from CPU freeze')
                item=move_to_device(inputs,runtime.device);torch.cuda.synchronize()
                prep_seconds=time.perf_counter()-begin;before=dict(counters);captured.clear()
                torch.cuda.reset_peak_memory_stats();forward_start=time.perf_counter()
                with torch.inference_mode():output=model(**item,use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize();forward_seconds=time.perf_counter()-forward_start
                need({k:counters[k]-before[k] for k in counters}==dict(model=1,vision=1,language=1),
                     'Each case must perform exactly one native VLM, visual and language forward')
                need(torch.equal(captured['position_ids'],expected_positions) and
                     torch.equal(captured['attention_mask'],inputs['attention_mask']) and
                     torch.equal(model.model.rope_deltas.detach().cpu(),expected_deltas),
                     'Actually executed native mRoPE/mask differs from independent CPU positions')
                need(output.logits.shape[:2]==(len(order),1) and output.past_key_values is None,
                     'Expected only final prompt logits and no cache')
                values=output.logits[:,0].double();need(bool(torch.isfinite(values).all()),'Nonfinite full-vocabulary logits')
                normalizers=torch.logsumexp(values,dim=-1);token_ids=plan['count_token_ids'];case_rows=[]
                for i,pid in enumerate(order):
                    record=plan['records'][pid];a=float(values[i,token_ids['0']]);b=float(values[i,token_ids['1']]);norm=float(normalizers[i])
                    top=int(values[i].argmax());row=dict(case=case['case'],batch_size=len(order),batch_row=i,pair_id=pid,
                        category=record['category'],audit_gold=record['audit_gold'],image_sha256=record['image_sha256'],
                        question=record['question'],input_ids_sha256=record['layout']['input_ids_sha256'],
                        prompt_tokens=record['layout']['prompt_tokens'],count_token_ids=token_ids,
                        logit0=a,logit1=b,log_normalizer=norm,probabilities=probabilities(a,b,norm),
                        top1_id=top,top1_text=runtime.tokenizer.decode([top],skip_special_tokens=False),
                        archived_comparison=comparison(dict(logit0=a,logit1=b,log_normalizer=norm),
                                                       record['archived_target'],record['audit_gold']))
                    if case['fresh_serial']:serial[pid]=row
                    elif pid in serial:row['fresh_serial_comparison']=comparison(row,serial[pid],record['audit_gold'])
                    case_rows.append(row);all_rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush();report=dict(case=case['case'],fresh_serial=case['fresh_serial'],batch_size=len(order),
                    pair_ids=order,preparation_seconds=prep_seconds,forward_seconds=forward_seconds,
                    total_case_seconds=time.perf_counter()-begin,seconds_per_row=forward_seconds/len(order),
                    native_model_forwards=1,native_visual_forwards=1,native_language_forwards=1,
                    peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                    archived_gate=gate([r['archived_comparison'] for r in case_rows]))
                fresh=[r['fresh_serial_comparison'] for r in case_rows if 'fresh_serial_comparison' in r]
                if fresh:report['fresh_serial_gate']=gate(fresh)
                case_reports.append(report);identities.append(dict(case=case['case'],**identity))
                print(json.dumps(report),flush=True)
                del inputs,item,output,values,normalizers
    finally:
        for hook in hooks:hook.remove()
    need(len(all_rows)==157 and counters==dict(model=9,vision=9,language=9),'Final response/forward coverage differs')
    need(verify_frozen(args.plan)[0]==plan,'Frozen source/model/teacher changed during GPU run')
    save(data/'input_identities.json',identities)
    archived=gate([r['archived_comparison'] for r in all_rows]);fresh=gate([r['fresh_serial_comparison'] for r in all_rows
                                                                       if 'fresh_serial_comparison' in r])
    summary=dict(schema_version=1,completed=True,computational_integrity_passed=True,
        numerical_gate_passed=archived['passed'],archived_gate=archived,fresh_serial_gate=fresh,
        fresh_serial_gate_is_descriptive=True,truth_accuracy_is_descriptive=True,
        distinct_training_pairs=64,batch_responses=153,fresh_serial_responses=4,rows=157,cases=case_reports,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=plan['source_sha256'],
        teacher_binding=plan['teacher_binding'],native_api=plan['native_api'],
        rows_file=str(raw),rows_sha256=sha(raw),input_identities_file=str(data/'input_identities.json'),
        input_identities_sha256=sha(data/'input_identities.json'),model_load_seconds=load_seconds,
        total_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(0),slurm_job_id=job,
        limitations=['These are repeated training-only local judgments, not aggregation accuracy or unseen-data generalization.',
          'A single native call processes independent complete batch rows; it does not fuse them into one answer.',
          'No generation, cached decoding, hidden-stream fencing or learned aggregation branch was tested.',
          'Probability tolerances are frozen. Common logit shifts are reported and need not change probabilities.',
          'Fresh serial references cover only the first four fixed pairs, one per category; all64 compare to the archived teacher. Fresh-overlap bounds and audit truth are descriptive, not extra eligibility criteria.',
          'Numerical failures are retained without threshold changes or selection.',
          'Forward timing includes the actual-position capture hook; CPU staging and model loading are reported separately.'])
    save(out/'summary.json',summary)
    lines=['# Independent complete local prompt batching','',
           f"Computational integrity: PASS. Frozen numerical gate: {'PASS' if summary['numerical_gate_passed'] else 'FAIL'}.",'',
           '64 fixed training pairs;153 batched responses plus4 fresh serial references. No aggregate answer is produced.','',
           '| Case | Rows | Forward seconds | Max conditional p1 difference | Max numeric mass difference | Gate |',
           '|---|---:|---:|---:|---:|---|']
    for case in case_reports:
        g=case['archived_gate'];lines.append(f"| {case['case']} | {case['batch_size']} | {case['forward_seconds']:.4f} | "
            f"{g['maximum_conditional_p1_difference']:.6g} | {g['maximum_numeric_mass_difference']:.6g} | {'PASS' if g['passed'] else 'FAIL'} |")
    lines+=['',f"Descriptive fresh-serial overlap bounds: {'PASS' if fresh['passed'] else 'FAIL'} ({fresh['n']} comparisons).",'',*summary['limitations'],'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),numerical_gate_passed=summary['numerical_gate_passed'],seconds=summary['total_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--check',action='store_true');modes.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path);args=parser.parse_args()
    need(bool(os.environ.get('SLURM_JOB_ID')),'All CPU and GPU numerical work requires Slurm')
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS',''),'Use the CPU check wrapper')
        check()
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and args.plan is not None,'Use the GPU wrapper with exact frozen plan')
        gpu(args)


if __name__=='__main__':main()
