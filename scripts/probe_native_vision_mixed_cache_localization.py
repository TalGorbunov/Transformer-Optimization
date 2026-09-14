"""Prospective unchanged-backend localization of one failed mixed cache row.

Same N64 local062 prompt alone, then the exact original65-row mixed batch.
Each performs prefill and Therefore/colon cached-versus-full checks. No original
failure, threshold, source, prompt, native kernel or model weight is changed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import probe_native_vision_parallel_local_mixed as original
from scripts.stage_native_vision_v6_teacher import need,read,save,sha
BASE=REPO/'outputs/native_aggregation_vlm/parallel_local_mixed'
PRIOR=BASE/'run_441729'
SID='v2c_profile_test_software_N16_K6_0000_N64'
INDEX=62
TAGS=('mixed_prefill','cached_step1','full_step1','cached_step2','full_step2')
OWN=('scripts/probe_native_vision_mixed_cache_localization.py',
     'slurm/native_vision_mixed_cache_localization_check.sbatch','slurm/native_vision_mixed_cache_localization.sbatch',
     'scripts/probe_native_vision_parallel_local_mixed.py')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',sources())


def prior_binding():
    summary=read(PRIOR/'summary.json');parent=original.verify_plan(summary['plan_file'])
    need(summary['completed'] is True and summary['computational_integrity_passed'] is True and
         summary['numerical_gate_passed'] is False,'Expected completed original failed gate')
    failures=summary['violations']
    need(len(failures)==1 and failures[0]['scene_id']==SID and failures[0]['row_index']==INDEX
         and failures[0]['tag']=='cached_vs_full_step2' and failures[0]['full_vocabulary_tv']>.02
         and failures[0]['top1_equal'] is True,'Registered failing row changed')
    scene=next(x for x in parent['scenes'] if x['sid']==SID)
    need(scene['n_frames']==64 and len(scene['rows'])==65 and scene['rows'][INDEX]['frame_index']==INDEX,'Selected source row differs')
    inventory=read(summary['state_inventory_file']);prior_states={}
    need(sha(summary['state_inventory_file'])==summary['state_inventory_sha256'],'Prior state inventory changed')
    for tag in (*TAGS,f'serial_{INDEX:03d}'):
        items=[x for x in inventory if x['scene_id']==SID and x['tag']==tag]
        need(len(items)==1 and sha(items[0]['path'])==items[0]['sha256'],'Prior native-vector binding differs')
        prior_states[tag]=dict(path=items[0]['path'],sha256=items[0]['sha256'])
    for name,digest in parent['source_sha256'].items():
        need(sha(REPO/name)==digest and sha(PRIOR/'source'/name.replace('/','_'))==digest,'Prior source snapshot differs')
    return parent,scene,dict(directory=str(PRIOR),summary_sha256=sha(PRIOR/'summary.json'),
        plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],state_inventory_sha256=summary['state_inventory_sha256'],
        prior_states=prior_states,original_failure=failures[0])


def check():
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from types import SimpleNamespace
    torch.set_num_threads(4);tests=original.self_test(torch);started=time.perf_counter()
    parent,scene,binding=prior_binding();blob=torch.load(parent['prepared_inputs_file'],map_location='cpu',weights_only=True)
    entry=blob['scenes'][SID];owner=SimpleNamespace(config=AutoConfig.from_pretrained(str(original.MODEL),trust_remote_code=True))
    fn=Qwen2_5_VLModel.get_rope_index;cases=[]
    for name,inputs,rows in (('single',entry['rows'][INDEX],[scene['rows'][INDEX]]),('mixed',entry['mixed'],scene['rows'])):
        identities=[]
        for step in range(3):
            full=original.extend(torch,inputs,parent['forced_token_ids'][:step]);pos,delta=original.rope(fn,owner,full)
            identities.append(original.identity(full,pos,delta))
        if name=='mixed':need(identities==scene['full_prefix_identities'],'Original mixed prefix identity differs')
        cases.append(dict(case=name,row_ids=[x['row_id'] for x in rows],batch_size=len(rows),prefix_identities=identities))
    job=os.environ['SLURM_JOB_ID'];out=BASE/f'localization_check_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    plan=dict(schema_version=1,protocol='unchanged_native_single_vs_mixed_cache_localization',source_sha256=sources(),
        prior=binding,model=parent['model'],runtime=parent['runtime'],processor=parent['processor'],native_api=parent['native_api'],
        selected_scene=SID,selected_local_row=INDEX,cases=cases,forced_token_ids=parent['forced_token_ids'],forced_token_text=parent['forced_token_text'],
        seed=parent['seed'],expected_calls=dict(model=10,vision=6),expected_response_rows=330,
        numerical_policy='All cached/full rows use unchanged full-vocabulary TV<=.02 and top1exact; repeat and cross-batch errors descriptive; original failed gate remains failed',
        tests=tests,slurm_job_id=job,seconds=time.perf_counter()-started)
    path=out/'plan.json';save(path,plan);path.with_suffix('.sha256').write_text(sha(path)+'\n');verify(path)
    save(out/'summary.json',dict(passed=True,plan=str(path),plan_sha256=sha(path),source_sha256=sources(),seconds=plan['seconds']))
    original.local.index(out,'Fixed mixed-cache localization CPU freeze',[('Plan','plan.json'),('Summary','summary.json'),('Sources','source/')])
    print(json.dumps(dict(passed=True,plan=str(path),plan_sha256=sha(path))),flush=True)


def verify(path):
    path=Path(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip(),'Localization plan sidecar differs')
    plan=read(path);parent,scene,binding=prior_binding()
    need(plan['schema_version']==1 and plan['protocol']=='unchanged_native_single_vs_mixed_cache_localization'
         and plan['source_sha256']==sources() and plan['prior']==binding,'Frozen localization source/prior differs')
    for key in ('model','runtime','processor','native_api','forced_token_ids','forced_token_text','seed'):
        need(plan[key]==parent[key],'Frozen policy differs: '+key)
    need(plan['selected_scene']==SID and plan['selected_local_row']==INDEX and
         [(x['case'],x['batch_size']) for x in plan['cases']]==[('single',1),('mixed',65)],'Fixed localization cases differ')
    return plan,parent,scene


def mask_check(torch,mask,key_mask,positions):
    if mask is not None:return original.mask_audit(torch,mask,key_mask,positions)
    need(bool((key_mask==1).all()) and (positions.numel()==key_mask.shape[1] or
         (positions.numel()==1 and int(positions[0])==key_mask.shape[1]-1)),
         'Implicit SDPA mask is only valid for unpadded full prefill or its final cached token')
    return dict(mode='native_sdpa_implicit_causal_or_single_final_query',valid_query_rows=key_mask.shape[0]*len(positions),
                padding_query_rows=0,valid_queries_exact=True)


def run(args):
    plan,parent,scene=verify(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] is True and cpu['plan_sha256']==sha(args.plan) and cpu['source_sha256']==sources(),'Completed CPU freeze required')
    for name,value in sources().items():need(sha(Path(args.plan).parent/'source'/name.replace('/','_'))==value,'CPU source snapshot differs')
    import torch
    import transformers
    from gnnformer.runtime import load_runtime,move_to_device,get_rope_index_fn,get_layers
    from scripts.probe_native_vision_v2_prefix import fingerprint
    need(torch.cuda.is_available() and str(torch.__version__)==plan['runtime']['torch_version']
         and str(transformers.__version__)==plan['runtime']['transformers_version'],'GPU runtime differs')
    torch.set_num_threads(4);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    started=time.perf_counter();job=os.environ['SLURM_JOB_ID'];out=BASE/f'localization_{job}';out.mkdir(parents=True,exist_ok=False);snapshot(out)
    data=original.DATA/f'localization_{job}';data.mkdir(parents=True,exist_ok=False);(out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    save(out/'config.json',dict(plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=sources(),
        prior=plan['prior'],model=plan['model'],runtime=plan['runtime'],processor=plan['processor']))
    original.local.index(out,'Fixed mixed-cache localization',[('Plan','plan.json'),('Raw rows','rows.jsonl'),('Summary after completion','summary.json')])
    original.local.index(data,'Native localization vectors',[('State inventory after completion','states.json')])
    entry=torch.load(parent['prepared_inputs_file'],map_location='cpu',weights_only=True)['scenes'][SID]
    load_start=time.perf_counter();runtime=load_runtime(str(original.MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=runtime.model;model.requires_grad_(False);model.eval()
    need(not any(x.requires_grad for x in model.parameters()) and fingerprint(runtime.processor,str(transformers.__version__))==plan['processor'],
         'Frozen model/processor differs')
    _,_,native=original.local.native_api(runtime.processor);need(native==plan['native_api'],'Native source differs')
    fn=get_rope_index_fn(model);language=model.model.language_model;layers=get_layers(model)
    dtypes=dict(final_norm_weight=str(language.norm.weight.dtype),lm_head_weight=str(model.lm_head.weight.dtype))
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    counts=dict(model=0,vision=0,language=0,norm=0);capture={};calls=[];state_inventory=[];saved_states={};raw_rows=[]
    def mh(*_):counts['model']+=1
    def vh(*_):counts['vision']+=1
    def lh(module,args,kwargs):
        counts['language']+=1;capture['positions']=kwargs['position_ids'].detach().clone();capture['mask']=kwargs['attention_mask'].detach().clone()
    def bh(module,args,kwargs):
        capture['causal']=kwargs['attention_mask'].detach().clone() if kwargs['attention_mask'] is not None else None
        capture['cache_position']=kwargs['cache_position'].detach().clone()
    def nh(module,args):counts['norm']+=1;capture['hidden']=args[0][:,-1,:].detach().clone()
    hooks=[model.register_forward_pre_hook(mh),model.model.visual.register_forward_pre_hook(vh),language.register_forward_pre_hook(lh,with_kwargs=True),
           layers[0].register_forward_pre_hook(bh,with_kwargs=True),language.norm.register_forward_pre_hook(nh)]
    stream=(out/'rows.jsonl').open('x')

    def execute(case,tag,inputs,expected_positions,expected_mask,cache_enabled,step):
        before=dict(counts);capture.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        with torch.inference_mode():result=model(**inputs,use_cache=cache_enabled,logits_to_keep=1)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        call_count={k:counts[k]-before[k] for k in counts};nv=int(inputs.get('pixel_values') is not None)
        need(call_count==dict(model=1,vision=nv,language=1,norm=1),'Native call count differs')
        need(torch.equal(capture['positions'],expected_positions) and torch.equal(capture['mask'],expected_mask),'Native logical positions/mask differ')
        masks=mask_check(torch,capture['causal'],expected_mask,capture['cache_position'])
        logits=result.logits[:,0].detach();hidden=capture['hidden'];b=case['batch_size']
        need(logits.shape[0]==b and hidden.shape==(b,3584) and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(hidden).all()),
             'Nonfinite/invalid native vectors')
        state=dict(schema_version=1,case=case['case'],tag=tag,row_ids=case['row_ids'],forced_token_ids=plan['forced_token_ids'][:step],
            native_next_token_logits=logits.detach().cpu(),pre_final_rms_hidden=hidden.detach().cpu(),
            native_position_ids=capture['positions'].detach().cpu(),attention_mask=capture['mask'].detach().cpu(),
            cache_position=capture['cache_position'].detach().cpu(),native_weight_dtypes=dtypes)
        path=data/f'{case["case"]}__{tag}.pt';torch.save(state,path);file_sha=sha(path);saved_states[(case['case'],tag)]=state
        state_inventory.append(dict(case=case['case'],tag=tag,path=str(path),sha256=file_sha,row_ids=case['row_ids'],
                                    hidden_dtype=str(hidden.dtype),logits_dtype=str(logits.dtype)))
        calls.append(dict(case=case['case'],tag=tag,calls=call_count,forward_seconds=seconds,mask_audit=masks,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),state_path=str(path),state_sha256=file_sha))
        for i,row_id in enumerate(case['row_ids']):
            x=logits[i].double();a=float(x[parent['count_token_ids']['0']]);z=float(x[parent['count_token_ids']['1']]);normalizer=float(torch.logsumexp(x,0))
            r=dict(case=case['case'],tag=tag,row_id=row_id,row_index=i,logit0=a,logit1=z,log_normalizer=normalizer,
                   probabilities=original.probabilities(a,z,normalizer),top1_id=int(x.argmax()),state_path=str(path))
            raw_rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n')
        stream.flush();return logits,hidden,result.past_key_values

    comparisons=[];cache_records=[]
    try:
        for case in plan['cases']:
            base=entry['rows'][INDEX] if case['case']=='single' else entry['mixed'];b=case['batch_size'];l=base['input_ids'].shape[1]
            pos,delta=fn(input_ids=base['input_ids'],image_grid_thw=base['image_grid_thw'],attention_mask=base['attention_mask'])
            need(original.identity(base,pos,delta)==case['prefix_identities'][0],'Frozen prefill differs')
            item=move_to_device(base,runtime.device)
            _,_,cache=execute(case,'prefill',item,pos.to(runtime.device),item['attention_mask'],True,0)
            need(torch.equal(model.model.rope_deltas.detach().cpu(),delta),'Prefill rope deltas differ')
            _,metadata=original.cache_snapshot(cache,b,l,len(layers));cache_records.append(dict(case=case['case'],tag='prefill',length=l,layers=metadata))
            for step in (1,2):
                full=original.extend(torch,base,plan['forced_token_ids'][:step])
                pos,delta=fn(input_ids=full['input_ids'],image_grid_thw=full['image_grid_thw'],attention_mask=full['attention_mask'])
                need(original.identity(full,pos,delta)==case['prefix_identities'][step],'Frozen forced prefix differs')
                previous,_=original.cache_snapshot(cache,b,l+step-1,len(layers),clone=True);full_item=move_to_device(full,runtime.device)
                cp=torch.tensor([l+step-1],device=runtime.device)
                prepared=model.prepare_inputs_for_generation(full_item['input_ids'],past_key_values=cache,attention_mask=full_item['attention_mask'],
                    cache_position=cp,use_cache=True,pixel_values=full_item['pixel_values'],image_grid_thw=full_item['image_grid_thw'])
                expected=torch.cat(((full_item['attention_mask'].cumsum(-1)-1)[:,-1:].unsqueeze(0),pos[:,:,-1:].to(runtime.device)),0)
                need(prepared['input_ids'].shape==(b,1) and bool((prepared['input_ids']==plan['forced_token_ids'][step-1]).all())
                     and prepared.get('past_key_values') is cache and prepared.get('pixel_values') is None and
                     torch.equal(prepared['position_ids'],expected) and prepared.pop('use_cache',True) is True,'Broadcast native preparation differs')
                ca,ch,past=execute(case,f'cached_step{step}',prepared,expected,full_item['attention_mask'],True,step)
                need(past is cache,'Cache object changed');preserved=original.prefix_preserved(torch,cache,previous,l+step-1);del previous
                _,metadata=original.cache_snapshot(cache,b,l+step,len(layers));saved_delta=model.model.rope_deltas.detach().clone()
                try:
                    fa,fh,unused=execute(case,f'full_step{step}',full_item,pos.to(runtime.device),full_item['attention_mask'],False,step)
                    need(unused is None and cache.get_seq_length()==l+step and torch.equal(model.model.rope_deltas,saved_delta)
                         and torch.equal(saved_delta.detach().cpu(),delta),'Reference changed logical cache/deltas')
                finally:model.model.rope_deltas=saved_delta
                cache_records.append(dict(case=case['case'],tag=f'step{step}',length=l+step,layers=metadata,
                    old_prefix_exact_by_layer=preserved,logical_positions_and_masks_exact=True,pixel_values_absent_during_decode=True))
                for i,row_id in enumerate(case['row_ids']):
                    result=original.compare(torch,ca[i],fa[i],ch[i],fh[i],'full_vocab')
                    result.update(case=case['case'],step=step,row_id=row_id,row_index=i);comparisons.append(result)
                del ca,ch,fa,fh,full_item,prepared
            del cache,item
            print(json.dumps(dict(completed_case=case['case'],calls=counts)),flush=True)
    finally:
        stream.close()
        for hook in hooks:hook.remove()
    need(counts==dict(model=10,vision=6,language=10,norm=10) and len(raw_rows)==330 and len(comparisons)==132,
         'Registered localization coverage differs')
    repeated=[];cross_batch=[];serial_prefill=[]
    for tag in ('prefill','cached_step1','full_step1','cached_step2','full_step2'):
        archived_tag='mixed_prefill' if tag=='prefill' else tag
        archived=torch.load(plan['prior']['prior_states'][archived_tag]['path'],map_location='cpu',weights_only=True)
        current=saved_states[('mixed',tag)];single=saved_states[('single',tag)]
        need(archived['row_ids']==current['row_ids']==plan['cases'][1]['row_ids'],'Repeat row order differs')
        for i,row_id in enumerate(current['row_ids']):
            a=current['native_next_token_logits'][i];b=archived['native_next_token_logits'][i]
            ha=current['pre_final_rms_hidden'][i];hb=archived['pre_final_rms_hidden'][i]
            result=original.compare(torch,a.to(runtime.device),b.to(runtime.device),ha.to(runtime.device),hb.to(runtime.device),'full_vocab')
            result.update(tag=tag,row_id=row_id,row_index=i,raw_logits_exact=bool(torch.equal(a,b)),hidden_exact=bool(torch.equal(ha,hb)),
                          descriptive_only=True);repeated.append(result)
        result=original.compare(torch,single['native_next_token_logits'][0].to(runtime.device),current['native_next_token_logits'][INDEX].to(runtime.device),
            single['pre_final_rms_hidden'][0].to(runtime.device),current['pre_final_rms_hidden'][INDEX].to(runtime.device),'full_vocab')
        result.update(tag=tag,row_id=scene['rows'][INDEX]['row_id'],descriptive_only=True);cross_batch.append(result)
    archived_serial=torch.load(plan['prior']['prior_states'][f'serial_{INDEX:03d}']['path'],map_location='cpu',weights_only=True)
    current=saved_states[('single','prefill')]
    serial_prefill=original.compare(torch,current['native_next_token_logits'][0].to(runtime.device),archived_serial['native_next_token_logits'][0].to(runtime.device),
        current['pre_final_rms_hidden'][0].to(runtime.device),archived_serial['pre_final_rms_hidden'][0].to(runtime.device),'full_vocab')
    serial_prefill['descriptive_only']=True
    need(verify(args.plan)[0]==plan,'Source/prior changed during localization')
    save(data/'states.json',state_inventory);save(out/'cached_full_comparisons.json',comparisons)
    save(out/'repeat_comparisons.json',repeated);save(out/'cross_batch_comparisons.json',cross_batch);save(out/'cache_checks.json',cache_records)
    focus=[x for x in comparisons if (x['case']=='single' and x['row_index']==0) or (x['case']=='mixed' and x['row_index']==INDEX)]
    summary=dict(schema_version=1,completed=True,computational_integrity_passed=True,numerical_gate_passed=all(x['passed'] for x in comparisons),
        original_numerical_gate_passed=False,original_failure=plan['prior']['original_failure'],source_sha256=sources(),
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),prior=plan['prior'],model=plan['model'],runtime=plan['runtime'],
        processor=plan['processor'],native_weight_dtypes=dtypes,calls=counts,rows=330,cached_full_comparisons=132,
        cache_comparison_violations=[x for x in comparisons if not x['passed']],focus_row=focus,
        mixed_repeat=dict(rows=325,raw_logits_exact_rows=sum(x['raw_logits_exact'] for x in repeated),
            hidden_exact_rows=sum(x['hidden_exact'] for x in repeated),maximum_full_vocabulary_tv=max(x['full_vocabulary_tv'] for x in repeated),
            top1_mismatches=sum(not x['top1_equal'] for x in repeated),descriptive_only=True),
        single_vs_mixed=cross_batch,single_prefill_vs_archived_serial=serial_prefill,per_call=calls,
        files={name:dict(path=str(out/name),sha256=sha(out/name)) for name in
               ('rows.jsonl','cached_full_comparisons.json','repeat_comparisons.json','cross_batch_comparisons.json','cache_checks.json')},
        state_inventory_file=str(data/'states.json'),state_inventory_sha256=sha(data/'states.json'),model_load_seconds=load_seconds,
        total_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(0),slurm_job_id=job,
        limitations=['The original441729failed gate remains failed, regardless of this localization.',
          'One row was chosen because it failed. This is numerical localization, not representative model accuracy.',
          'Same native NF4/bfloat16-compute backend, actual FP16 native states/head; no kernel or precision intervention.',
          'Single-versus-mixed and original-repeat differences are descriptive, not replacement eligibility criteria.',
          'No training, aggregate answer, new prompt search or claimed reasoning competence.'])
    save(out/'summary.json',summary)
    lines=['# Native single-versus-mixed cache localization','',
        'Original441729numerical gate remains **FAIL**. Every original artifact and threshold is unchanged.','',
        f"New132cached/full comparisons: {'PASS' if summary['numerical_gate_passed'] else 'FAIL'} under the same.02TV/top1 rule.",'',
        '| Case | Forced prefix step | Focus-row TV | Top1 equal | Gate |','|---|---:|---:|---|---|']
    for x in focus:lines.append(f"| {x['case']} | {x['step']} | {x['full_vocabulary_tv']:.9g} | {x['top1_equal']} | {'PASS' if x['passed'] else 'FAIL'} |")
    lines+=['',f"Repeated mixed raw-logit rows exact: {summary['mixed_repeat']['raw_logits_exact_rows']}/325.",'',*summary['limitations'],'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),numerical_gate_passed=summary['numerical_gate_passed'],seconds=summary['total_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();need(os.environ.get('SLURM_JOB_ID'),'All numerical work requires Slurm')
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS',''),'Use CPU wrapper');check()
    else:
        need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and args.plan is not None,'Use GPU wrapper with exact plan');run(args)


if __name__=='__main__':main()
