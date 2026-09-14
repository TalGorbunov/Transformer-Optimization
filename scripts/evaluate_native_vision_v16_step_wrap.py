"""V16 diagnostic Step-label replacement on four immutable V14 models.

Original V15 inputs/results are the comparator. Four original-image sentinel
traces precede68 wrapped traces per model. Both learned/bank means keep N+25
streams; no training, output mask, data selection or new efficacy criterion.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import evaluate_native_vision_v15_null_comparison as old
need,read,sha,save,bind,object_sha=old.need,old.read,old.sha,old.save,old.bind,old.object_sha
MODEL=old.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/v16/step_wrap_study'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v16_step_wrap_study')
ORIGINAL=REPO/'outputs/native_aggregation_vlm/v15/null_comparison_study/report_442711/analysis.json'
SOFTWARE=REPO/'outputs/native_aggregation_vlm/v15/null_comparison_software/profile_442641/summary.json'
CORE_KEYS=old.CORE_KEYS;MODES=old.MODES
OWN=('scripts/evaluate_native_vision_v16_step_wrap.py',
     'slurm/native_vision_v16_step_wrap_study_check.sbatch','slurm/native_vision_v16_step_wrap_study.sbatch')
POLICY=dict(protocol='v16_fixed_v14_step_label_wrap_diagnostic',core_keys=list(CORE_KEYS),modes=list(MODES),
    intervention='Only actual-image bottom Step labels i become1+((i-1)%16); original reference images',
    wrapped_scenes=34,sentinels=[[3,32],[6,64]],sentinel_traces_per_model=4,
    wrapped_traces_per_model=68,total_traces_per_model=72,total_traces=288,
    wrapped_total=272,sentinel_total=16,maximum_model_forwards=1152,visual_forwards=288,
    maximum_model_forwards_per_model=288,visual_forwards_per_model=72,
    native_rows='N actual +24 original training-reference images +one global text stream',
    correction_coefficients={'centered':'actual N','offset':1},anchor=None,
    max_new_tokens=4,native_eos=list(old.POLICY['native_eos']),target_eos=151645,
    capture_all_generated_queries=True,reference_mean='FP64 occurrence mean followed by FP32',
    scene_order='Original K3N32 thenK6N64 sentinels first; then all34 originalV15 cases in their fixed order; learned thenbank',
    sentinel_token_ids_exact=True,sentinel_each_prefix_top1_exact=True,sentinel_each_prefix_tv_max=.02,
    sentinel_probability_precision='FP64 softmax of saved raw FP32 native logits',
    sentinel_gate_before_wrapped=True,per_job_gpu_seconds_cap=900,campaign_gpu_seconds_cap=5400,
    maximum_concurrent_gpus=4,software_reused_charged_to_v15=True,
    projection='software_load +1.25*72*max_both_modes_N64_four_token_time +120',
    no_fit=True,no_model_selection=True,no_accuracy_eligibility=True,no_practical_milestone_claim=True)


def stage_module():return importlib.import_module('scripts.stage_native_vision_v16_step_wrap')


def sources():
    stage=stage_module()
    return {**old.sources(),**stage.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    frozen=sources();(out/'code').mkdir()
    for name,digest in frozen.items():
        path=out/'code'/name.replace('/','_');path.write_bytes((REPO/name).read_bytes())
        need(sha(path)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V16 Step-label intervention\n\n[Summary](summary.json) · [Sources](source_hashes.json) · [Progress](progress.json).\n')
    index=OUT/'INDEX.md'
    if not index.exists():index.write_text('# V16 Step-label diagnostic\n\n')
    with index.open('a') as stream:stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


def check_sources(value,directory):
    need(value==sources(),'V16 source changed')
    for name,digest in value.items():need(sha(Path(directory)/'code'/name.replace('/','_'))==digest,'Source snapshot changed')


def projection(gate):
    load,bound=gate['load_seconds'],gate['T64']
    need(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in (load,bound)),
         'Invalid measured native timing')
    value=load+1.25*72*bound+120
    return dict(passed=value<=900,projected_seconds_per_model=value,load_seconds=load,T64=bound,
        generations=72,all_lengths_charged_N64=True,per_model_cap=900,four_max_allocations=3600,campaign_cap=5400)


def parse_allocations(raw):
    rows=[];seen=set();events=[]
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}
    for line in raw.splitlines():
        if not line.strip():continue
        fields=line.split('|');need(len(fields)==9,'Malformed Slurm allocation row')
        job,name,partition,state,exit_code,elapsed,tres,start,end=fields
        if not name.startswith('v16_') or partition!='gpu':continue
        need(name=='v16_step_study' and job not in seen and elapsed.isdigit(),'Unexpected V16 allocation role/ID')
        seen.add(job);need(state.split()[0] in terminal and re.fullmatch(r'[0-9]+:[0-9]+',exit_code),
                           'V16 GPU allocations must be terminal')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed)
        need(gpus in (0,1) and (not typed or sum(typed)==gpus),'Unexpected GPU count')
        seconds=int(elapsed);need(seconds<=900,'V16 per-job cap exceeded')
        if gpus and seconds:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(b>=a,'Invalid allocation interval')
            events.extend(((a,1),(b,-1)))
        rows.append(dict(job_id=job,job_name=name,state=state,exit_code=exit_code,elapsed_seconds=seconds,
            gpus=gpus,gpu_seconds=seconds*gpus,start=start,end=end))
    active=maximum=0
    for _,change in sorted(events,key=lambda x:(x[0],x[1])):active+=change;maximum=max(maximum,active)
    total=sum(r['gpu_seconds'] for r in rows)
    need(active==0 and maximum<=4 and total<=5400,'V16 campaign/concurrency cap exceeded')
    return dict(jobs=rows,allocated_gpu_seconds=total,maximum_concurrent_gpus=maximum,
        failed_allocations_included=True,zero_allocation_attempts_retained=True,
        all_user_v16_prefix_jobs_discovered=True,reused_v15_software_not_charged_again=True)


def allocation_ledger(out):
    import subprocess
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    path=out/'all_user_sacct.psv';path.write_text(result.stdout)
    return dict(parse_allocations(result.stdout),command=command,raw_file=str(path),raw_sha256=sha(path))


def sentinel_metrics(torch,current,prior,ids,prior_ids):
    need(current.ndim==prior.ndim==2 and current.shape[1]==prior.shape[1]
         and current.shape[0]==len(ids) and prior.shape[0]==len(prior_ids)
         and bool(torch.isfinite(current).all()) and bool(torch.isfinite(prior).all()),'Invalid sentinel logits')
    prefix=old.prefix_relation(ids,prior_ids);comparisons=[]
    for t in range(prefix['shared_executed_prefixes']):
        a,b=current[t].double(),prior[t].double();tv=float((a.softmax(-1)-b.softmax(-1)).abs().sum()/2)
        same=int(a.argmax())==int(b.argmax())
        comparisons.append(dict(position=t,observed_prefix_ids=ids[:t],tv=tv,top1_equal=same,
            raw_max_abs=float((a-b).abs().max()),passed=tv<=.02 and same))
    passed=ids==prior_ids and len(comparisons)==len(ids) and all(r['passed'] for r in comparisons)
    return dict(passed=passed,generated_ids_equal=ids==prior_ids,comparison_count=len(comparisons),
        comparisons=comparisons,prefix_relation=prefix,threshold_tv=.02)


def sentinel_indices(cases):
    result=[]
    for gold,n in POLICY['sentinels']:
        matches=[i for i,c in enumerate(cases) if c['record']['gold']==gold and c['record']['n_frames']==n]
        need(len(matches)==1,'Fixed original sentinel is missing or ambiguous');result.extend(matches)
    return result


def self_test():
    import torch
    inherited=old.self_test();a=torch.tensor([[4.,0.,-1.],[0.,4.,-1.]])
    good=sentinel_metrics(torch,a,a.clone(),[0,1],[0,1]);need(good['passed'],'Identical sentinel rejected')
    bad=sentinel_metrics(torch,a,torch.tensor([[1.,0.,0.],[0.,4.,-1.]]),[0,1],[0,1])
    need(not bad['passed'] and bad['comparisons'][0]['top1_equal'],'Large same-top1 probability drift accepted')
    different=sentinel_metrics(torch,a,a,[0,1],[0,0]);need(not different['passed'],'Changed generated IDs accepted')
    cases=[dict(record=dict(gold=k,n_frames=n)) for k in range(17) for n in (32,64)]
    need(sentinel_indices(cases)==[6,13],'Sentinel K/N ordering changed')
    raw='1|v16_step_study|gpu|FAILED|1:0|3|gres/gpu=1|2026-09-11T00:00:00|2026-09-11T00:00:03'
    need(parse_allocations(raw)['allocated_gpu_seconds']==3,'Failed GPU seconds omitted')
    need(projection(dict(load_seconds=10.,T64=2.))['projected_seconds_per_model']==310.,'72-trace projection differs')
    return dict(passed=True,inherited=inherited,tests=['exact_sentinel','same_argmax_large_TV_rejected',
        'token_divergence_rejected','fixed_sentinel_order','failed_allocation_included','72_trace_projection'])


def verify_original(bindings):
    path=ORIGINAL;summary=read(path.parent/'summary.json');analysis=read(path)
    need(all(analysis[k] is True and summary[k] is True for k in ('passed','completed','audit_passed'))
         and summary['analysis_file']==str(path) and summary['analysis_sha256']==sha(path)
         and analysis['records']==272 and analysis['captures_all_prefixes'] is True,'Completed independent original V15 report required')
    old.check_sources(analysis['source_sha256'],path.parent);bind(path,bindings);bind(path.parent/'summary.json',bindings)
    plan=old.verify_plan(analysis['plan_file']);bind(analysis['plan_file'],bindings,analysis['plan_sha256'])
    need(plan['source_sha256']==analysis['source_sha256'] and analysis['policy']==old.POLICY,'Original study plan differs')
    bindings.update(plan['artifact_bindings']);bind(analysis['rescored_file'],bindings,analysis['rescored_sha256'])
    rows=read(analysis['rescored_file']);need(len(rows)==272 and set(analysis['runs'])==set(CORE_KEYS),'Missing original models/rows')
    lookup={}
    for key in CORE_KEYS:
        run=analysis['runs'][key];directory=Path(run['directory'])
        for name,field in (('summary.json','summary_sha256'),('config.json','config_sha256')):bind(directory/name,bindings,run[field])
        summary=read(directory/'summary.json');old.check_sources(summary['source_sha256'],directory)
        selected=plan['selected_models'][CORE_KEYS.index(key)]
        need(summary['passed'] is True and summary['completed'] is True and summary['computational_integrity_passed'] is True
             and summary['selected_model']==run['selected_model']==selected,'Original selected model/run changed')
        bind(summary['predictions_file'],bindings,summary['predictions_sha256']);original=read(summary['predictions_file'])
        group=[r for r in rows if r['core_key']==key];need(len(group)==len(original)==68,'Original model rows incomplete')
        for index,(saved,row) in enumerate(zip(original,group)):
            need(all(row[k]==v for k,v in saved.items()) and row['index']==index and row['mode']==MODES[index%2]
                 and row['case_index']==index//2,'Original record order changed')
            bind(row['raw_file'],bindings,row['raw_sha256'])
            identity=(key,row['case_index'],row['mode']);need(identity not in lookup,'Duplicate original record');lookup[identity]=row
    return analysis,plan,rows,lookup


def check(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    from scripts.probe_native_vision_parallel_local import native_api
    started=time.perf_counter();bindings={};tests=self_test();original,prior,old_rows,_=verify_original(bindings)
    stage=stage_module();manifest=stage.verify_stage(args.stage_summary);stage_summary=read(args.stage_summary)
    bind(args.stage_summary,bindings);need(len(manifest['cases'])==34,'Require all34 stage cases')
    # The stage verifier checks all footer-only RGB proofs, original QA and source ancestry.
    for key,value in stage_summary.items():
        if key.endswith('_file') and isinstance(value,str) and key[:-5]+'_sha256' in stage_summary:
            bind(value,bindings,stage_summary[key[:-5]+'_sha256'])
    bindings.update(manifest['source_bindings'])
    bind(manifest['stage_plan_file'],bindings,manifest['stage_plan_sha256'])
    bind(Path(manifest['stage_plan_file']).with_suffix('.sha256'),bindings)
    need(manifest['reference_banks']==prior['reference_banks']
         and manifest['source_v15_plan_file']==original['plan_file']
         and manifest['source_v15_plan_sha256']==original['plan_sha256'],'Stage used another source plan/reference bank')
    gate,software_plan=old.verify_software(SOFTWARE,bindings)
    need(str(gate['software_job_id'])=='442641','Require the registered unchanged V15 software gate')
    runtime,_,background,training,auditor,_=old.modules()
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==prior['processor']==software_plan['processor'], 'Processor differs')
    owner,fn,api=native_api(processor);need(api==prior['native_api']==software_plan['native_api'],'Installed native helper differs')
    need(all(prior[k]==software_plan[k] for k in ('model','runtime','processor')),'Original/software native identity differs')
    banks=prior['reference_banks'];prepared=[];data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    original_indices=sentinel_indices(prior['cases'])
    for index in original_indices:
        source=prior['cases'][index];bind(source['file'],bindings,source['sha256'])
        prepared.append(dict(source,role='original_sentinel',original_case_index=index,source_record=source['record']))
    for index,(stage_case,source) in enumerate(zip(manifest['cases'],prior['cases'])):
        need(stage_case['source_record']==source['record'],'Stage selected another original scene/order')
        sample=stage_case['wrapped_record'];original_record=source['record']
        need(all(sample[k]==v for k,v in original_record.items() if k not in ('path','source_path','image_files')),
             'Wrapped record changed protected original semantics')
        background.verify_qa(sample,bindings)
        for im in sample['image_files']:bind(im['path'],bindings,im['sha256'])
        bind(source['file'],bindings,source['sha256']);before=torch.load(source['file'],map_location='cpu',weights_only=True)
        bundle=runtime.prepare_scene(processor,sample,banks[sample['question']],verify_processor_parity=True)
        layout=runtime.native.audit_layout(lambda **kwargs:fn(owner,**kwargs),bundle)
        n=sample['n_frames'];need(set(bundle['inputs'])==set(before['inputs']) and layout['metadata']==source['layout'],
            'Wrapping changed native input fields/mRoPE layout')
        for name,tensor in bundle['inputs'].items():
            old_tensor=before['inputs'][name]
            need(tensor.shape==old_tensor.shape and tensor.dtype==old_tensor.dtype,'Wrapping changed input shapes/dtypes')
            if name!='pixel_values':need(torch.equal(tensor,old_tensor),'Wrapping changed nonpixel native input')
        for row_index in list(range(16))+list(range(n,n+25)):
            need(set(bundle['row_inputs'][row_index])==set(before['row_inputs'][row_index])
                 and all(torch.equal(v,before['row_inputs'][row_index][k]) for k,v in bundle['row_inputs'][row_index].items()),
                 'First16 actual/reference/global processor rows changed')
        need(bundle['metadata']['reference_bank']==source['metadata']['reference_bank'] and bundle['metadata']['prefix_ids']==[],
             'Reference bank or initial answer prefix changed')
        path=data/f'wrapped_{index:03d}.pt';torch.save(bundle,path);bind(path,bindings)
        prepared.append(dict(record=sample,source_record=original_record,role='wrapped',original_case_index=index,
            file=str(path),sha256=sha(path),metadata=bundle['metadata'],layout=layout['metadata']))
        print(json.dumps(dict(prepared=index+1,total=34,sid=sample['sid'])),flush=True)
    need(len(prepared)==36 and [x['role'] for x in prepared]==['original_sentinel']*2+['wrapped']*34,'Study case inventory differs')
    resource=projection(gate);allocation=allocation_ledger(out);reserve=allocation['allocated_gpu_seconds']+3600<=5400
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,tests=tests,
        stage_summary=dict(file=str(Path(args.stage_summary).resolve()),sha256=sha(args.stage_summary)),
        stage_manifest=manifest,original_report=dict(file=str(ORIGINAL),sha256=sha(ORIGINAL)),
        original_plan=dict(file=original['plan_file'],sha256=original['plan_sha256']),
        original_records=dict(file=original['rescored_file'],sha256=original['rescored_sha256']),
        selected_models=prior['selected_models'],cases=prepared,reference_banks=banks,
        model=prior['model'],runtime=prior['runtime'],processor=prior['processor'],native_api=api,
        native_dtypes=prior['native_dtypes'],target_token_ids=prior['target_token_ids'],
        software_gate=gate,projection=resource,prior_allocation=allocation,reserve_passed=reserve,
        no_fit=True,no_selection=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    need(sources()==frozen,'Sources changed during V16 preparation')
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    save(out/'summary.json',dict(passed=resource['passed'] and reserve,completed=True,no_model_loaded=True,
        source_sha256=frozen,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,
        cases=36,wrapped_cases=34,sentinel_cases=2,projection=resource,prior_allocation=allocation,
        reserve_passed=reserve,seconds=time.perf_counter()-started))
    need(resource['passed'] and reserve,'V16 timing/resource release failed; evidence retained')


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);summary=read(path.parent/'summary.json')
    need(path.is_relative_to(OUT) and path.with_suffix('.sha256').read_text().strip()==sha(path)
         and summary['passed'] is True and summary['completed'] is True and summary['no_model_loaded'] is True
         and summary['plan_sha256']==sha(path) and summary['source_sha256']==plan['source_sha256'] and plan['policy']==POLICY and plan['reserve_passed'] is True,'Require passed exact CPU plan')
    check_sources(plan['source_sha256'],path.parent)
    for name,digest in plan['artifact_bindings'].items():need(sha(name)==digest,'Frozen V16 artifact changed: '+name)
    stage_ref=plan['stage_summary'];need(sha(stage_ref['file'])==stage_ref['sha256'],'Staging summary changed')
    need(stage_module().verify_stage(stage_ref['file'])==plan['stage_manifest'],'Footer proof/stage inputs changed')
    gate,_=old.verify_software(SOFTWARE,{})
    need(gate==plan['software_gate'] and projection(gate)==plan['projection'] and plan['projection']['passed'],'Inherited software/timing changed')
    need([r['key'] for r in plan['selected_models']]==list(CORE_KEYS) and len(plan['cases'])==36
         and [r['role'] for r in plan['cases']]==['original_sentinel']*2+['wrapped']*34,'Fixed model/case inventory differs')
    for case in plan['cases']:need(sha(case['file'])==case['sha256'],'Prepared input changed')
    return plan


def original_lookup(plan):
    ref=plan['original_records'];need(sha(ref['file'])==ref['sha256'],'Original records changed')
    rows=read(ref['file']);lookup={(r['core_key'],r['case_index'],r['mode']):r for r in rows}
    need(len(rows)==len(lookup)==272,'Original comparator inventory differs')
    return lookup


def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    from scripts.probe_native_vision_v2_prefix import fingerprint
    runtime,_,background,training,auditor,_=old.modules();started=time.perf_counter();plan=verify_plan(args.plan)
    need(torch.cuda.device_count()==1 and args.model_index in range(4)
         and torch.cuda.get_device_name(0)=='NVIDIA B200','Require one registered B200/model index')
    prior=original_lookup(plan);selected=plan['selected_models'][args.model_index];key=selected['key']
    checkpoint=selected['selected']['checkpoint'];load_start=time.perf_counter()
    native=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda');model=native.model.eval().requires_grad_(False)
    need(background.feature.model_metadata()==plan['model'] and background.feature.runtime_identity()==plan['runtime']
         and fingerprint(native.processor,str(transformers.__version__))==plan['processor'],'Live model/runtime/processor changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(saved['config']==selected['config'] and saved['step']==4590 and saved['student_step']==8000,'Fixed checkpoint metadata changed')
    branch=ParallelLocalAggregation().eval().requires_grad_(False);branch.load_state_dict(saved['branch'],strict=True);branch.to(model.device)
    predictor=ConditionalNullMean().eval().requires_grad_(False);predictor.load_state_dict(saved['predictor'],strict=True);predictor.to(model.device)
    auditor.validate_state(torch,saved,selected['selected']['parameter_sha256']);del saved
    identity=training.state_info(branch,predictor);need(object_sha(identity)==selected['selected']['parameter_sha256'],'Loaded selected tensors differ')
    runtime.native.native_contract(model,branch);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    (data/'INDEX.md').write_text('# V16 complete raw native trajectories\n\nFour original sentinels then68 wrapped-image trajectories.\n')
    config=dict(schema_version=1,policy=POLICY,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
        source_sha256=frozen,model_index=args.model_index,core_key=key,selected_model=selected,
        checkpoint_sha256=sha(checkpoint),model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],
        hardware=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda),model_load_seconds=load_seconds,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'config.json',config);records=[];sentinels=[];totals=Counter()
    versions={name:p._version for name,p in model.named_parameters()}
    learned_versions={(kind,name):p._version for kind,module in [('core',branch),('predictor',predictor)] for name,p in module.named_parameters()}
    for case_index,case in enumerate(plan['cases']):
        if case_index==2:
            sentinel_gate=dict(passed=len(sentinels)==4 and all(r['passed'] for r in sentinels),records=4,
                observations=sentinels,exact_generated_ids_required=True,each_prefix_tv_max=.02)
            save(out/'sentinel_gate.json',sentinel_gate)
            old.atomic_json(out/'progress.json',dict(completed=False,records=len(records),sentinel_gate=sentinel_gate))
            need(sentinel_gate['passed'],'Original-image sentinel gate failed; allfour raw traces retained, no wrapped inference')
        need(sha(case['file'])==case['sha256'],'Prepared scene changed')
        bundle=torch.load(case['file'],map_location='cpu',weights_only=True)
        need(bundle['metadata']==case['metadata'] and runtime.validate_bundle(bundle)['passed'],'Prepared metadata differs')
        need(runtime.native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['layout'],'Native input/mRoPE changed')
        sample=case['record']
        for mode in MODES:
            index=len(records);result=runtime.generate_native(model,native.processor,branch,predictor,bundle,
                condition=selected['condition'],mean_source=mode,core_key=key,max_new_tokens=4,capture=True)
            raw=result['raw_logits'].detach().cpu();raw_path=data/f'raw_{index:03d}.pt'
            torch.save(dict(schema_version=1,raw_logits=raw,captures=result['captures']),raw_path)
            need(raw.dtype==torch.float32 and torch.equal(raw,raw.half().float()) and bool(torch.isfinite(raw).all()),
                 'Raw logits must be losslessly promoted native FP16')
            scored=old.score_ids(native.processor.tokenizer,result['generated_ids'],sample['gold'])
            need(all(result[k]==scored[k] for k in ('raw_text','text','completed','truncated','finish_reason')),'Runtime text/stop differs')
            record=dict(index=index,case_index=case_index,original_case_index=case['original_case_index'],role=case['role'],
                core_key=key,condition=selected['condition'],seed=selected['seed'],mode=mode,
                sid=sample['sid'],pair_id=sample['pair_id'],gold=sample['gold'],n_frames=sample['n_frames'],
                content_sha256=sample['content_sha256'],**scored,raw_file=str(raw_path),raw_sha256=sha(raw_path),
                raw_dtype='torch.float32',metadata=result['metadata'],counters=result['counters'],model_seconds=result['model_seconds'],
                capture_identities=old.audit_captures(torch,result['captures'],result['generated_ids'],case,selected,mode))
            old.audit_output(torch,native.processor.tokenizer,record,raw,case,selected)
            original=prior[key,case['original_case_index'],mode]
            record['original_record']=dict(raw_file=original['raw_file'],raw_sha256=original['raw_sha256'],
                generated_ids=original['generated_ids'],index=original['index'])
            if case['role']=='original_sentinel':
                need(sha(original['raw_file'])==original['raw_sha256'],'Original sentinel raw changed')
                old_raw=torch.load(original['raw_file'],map_location='cpu',weights_only=True)['raw_logits']
                comparison=sentinel_metrics(torch,raw,old_raw,record['generated_ids'],original['generated_ids'])
                record['sentinel_comparison']=comparison;sentinels.append(dict(index=index,sid=sample['sid'],mode=mode,**comparison));del old_raw
            save(out/f'record_{index:03d}.json',record);records.append(record);totals.update(record['counters'])
            old.atomic_json(out/'progress.json',dict(completed=False,records=len(records),expected=72,core_key=key,
                counters=dict(totals),last_record_file=str(out/f'record_{index:03d}.json'),elapsed_seconds=time.perf_counter()-started))
            if len(records)%6==0:print(json.dumps(dict(core_key=key,records=len(records),expected=72)),flush=True)
        del bundle
    need(len(records)==72 and totals['visual']==72 and 72<=totals['model']<=288,'V16 trace/call inventory differs')
    need(versions=={name:p._version for name,p in model.named_parameters()}
         and learned_versions=={(kind,name):p._version for kind,module in [('core',branch),('predictor',predictor)] for name,p in module.named_parameters()}
         and not any(p.requires_grad or p.grad is not None for p in model.parameters())
         and not any(p.requires_grad or p.grad is not None for module in (branch,predictor) for p in module.parameters())
         and training.state_info(branch,predictor)==identity,'Frozen backbone/learned tensors changed')
    need(sha(checkpoint)==config['checkpoint_sha256'] and sources()==frozen,'Checkpoint/source changed during run')
    save(out/'predictions.json',records)
    save(out/'summary.json',dict(config,passed=True,completed=True,computational_integrity_passed=True,no_fit=True,
        records=72,sentinel_records=4,wrapped_records=68,sentinel_gate_file=str(out/'sentinel_gate.json'),
        sentinel_gate_sha256=sha(out/'sentinel_gate.json'),predictions_file=str(out/'predictions.json'),
        predictions_sha256=sha(out/'predictions.json'),counters=dict(totals),
        generation_seconds=sum(r['model_seconds'] for r in records),elapsed_seconds=time.perf_counter()-started,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved()))
    old.atomic_json(out/'progress.json',dict(completed=True,records=72,summary_file=str(out/'summary.json'),counters=dict(totals)))


def report(args,out,frozen):
    import torch
    from transformers import AutoTokenizer
    started=time.perf_counter();plan=verify_plan(args.plan);prior=original_lookup(plan);tests=self_test()
    need(len(args.report)==len({str(Path(p).resolve()) for p in args.report})==4,'Require allfour distinct runs')
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),use_fast=False,local_files_only=True)
    config=read(MODEL/'config.json');vocab=config.get('text_config',config)['vocab_size']
    allrows=[];runs={};job_ids=set();all_sentinels=[]
    for item in args.report:
        directory=Path(item).resolve();need(directory.is_relative_to(OUT),'V16 run path differs')
        summary=read(directory/'summary.json');config=read(directory/'config.json');key=summary['core_key']
        need(key in CORE_KEYS and key not in runs and summary['passed'] is True and summary['completed'] is True
             and summary['computational_integrity_passed'] is True and summary['no_fit'] is True
             and all(summary[k]==v for k,v in config.items()) and summary['policy']==POLICY
             and summary['plan_sha256']==sha(args.plan) and summary['source_sha256']==frozen
             and summary['hardware']['name']=='NVIDIA B200','Run completion/source/native hardware differs')
        check_sources(summary['source_sha256'],directory);selected=plan['selected_models'][CORE_KEYS.index(key)]
        need(summary['selected_model']==selected and summary['checkpoint_sha256']==selected['selected']['checkpoint_sha256']
             and sha(selected['selected']['checkpoint'])==summary['checkpoint_sha256'],'Selected model identity changed')
        need(sha(summary['predictions_file'])==summary['predictions_sha256']
             and sha(summary['sentinel_gate_file'])==summary['sentinel_gate_sha256'],'Run archive changed')
        gate=read(summary['sentinel_gate_file']);records=read(summary['predictions_file'])
        need(len(records)==summary['records']==72 and summary['sentinel_records']==4 and summary['wrapped_records']==68,'Incomplete trace inventory')
        rescored=[];counts=Counter();sentinels=[]
        for index,row in enumerate(records):
            case=plan['cases'][index//2];mode=MODES[index%2]
            need(row['index']==index and row['case_index']==index//2 and row['mode']==mode and row['role']==case['role']
                 and row['original_case_index']==case['original_case_index'] and read(directory/f'record_{index:03d}.json')==row,
                 'Fixed trace order/durable record differs')
            need(sha(row['raw_file'])==row['raw_sha256'],'V16 raw archive changed')
            raw=torch.load(row['raw_file'],map_location='cpu',weights_only=True)
            need(raw['schema_version']==1 and raw['raw_logits'].shape[1]==vocab,'Raw native vocabulary differs')
            checked=old.audit_output(torch,tokenizer,row,raw['raw_logits'],case,selected)
            need(row['capture_identities']==old.audit_captures(torch,raw['captures'],row['generated_ids'],case,selected,mode),
                 'All-prefix capture identity differs')
            original=prior[key,case['original_case_index'],mode]
            need(row['original_record']==dict(raw_file=original['raw_file'],raw_sha256=original['raw_sha256'],
                generated_ids=original['generated_ids'],index=original['index']),'Wrong original comparator')
            if case['role']=='original_sentinel':
                need(index<4 and sha(original['raw_file'])==original['raw_sha256'],'Original sentinel source/order changed')
                original_raw=torch.load(original['raw_file'],map_location='cpu',weights_only=True)['raw_logits']
                comparison=sentinel_metrics(torch,raw['raw_logits'],original_raw,row['generated_ids'],original['generated_ids'])
                need(comparison==row['sentinel_comparison'] and comparison['passed'],'Independent original sentinel gate failed')
                sentinels.append(dict(index=index,sid=row['sid'],mode=mode,**comparison));del original_raw
            else:need(index>=4 and 'sentinel_comparison' not in row,'Wrapped trace used as sentinel')
            rescored.append(checked);counts.update(row['counters']);del raw
        expected_gate=dict(passed=True,records=4,observations=sentinels,exact_generated_ids_required=True,each_prefix_tv_max=.02)
        need(gate==expected_gate and len(sentinels)==4,'Saved sentinel gate differs')
        need(dict(counts)==summary['counters'] and counts['visual']==72 and 72<=counts['model']<=288
             and math.isclose(sum(r['model_seconds'] for r in rescored),summary['generation_seconds'],rel_tol=1e-12),
             'Aggregate native counters/timing differs')
        allrows.extend(rescored);all_sentinels.extend(dict(core_key=key,**r) for r in sentinels);job_ids.add(str(summary['slurm_job_id']))
        runs[key]=dict(directory=str(directory),summary_sha256=sha(directory/'summary.json'),config_sha256=sha(directory/'config.json'),
            selected_model=selected,counters=dict(counts),sentinel_gate_file=summary['sentinel_gate_file'],
            sentinel_gate_sha256=summary['sentinel_gate_sha256'],generation_seconds=summary['generation_seconds'],
            model_load_seconds=summary['model_load_seconds'],elapsed_seconds=summary['elapsed_seconds'])
    wrapped=[r for r in allrows if r['role']=='wrapped']
    need(set(runs)==set(CORE_KEYS) and len(allrows)==288 and len(wrapped)==272 and len(all_sentinels)==16,'Incomplete campaign inventory')
    cells={};by_k={};comparisons={};transitions=[]
    scopes=(('all',range(17)),('K0_8',range(9)),('K9_15',range(9,16)),('K16',[16]),
            ('K9_16',range(9,17)),('K0_9',range(10)),('K10_16',range(10,17)))
    for key in CORE_KEYS:
        for n in (32,64):
            for mode in MODES:
                rows=[r for r in wrapped if r['core_key']==key and r['n_frames']==n and r['mode']==mode]
                need(len(rows)==17 and {r['gold'] for r in rows}==set(range(17)),'Missing complete K support')
                before=[prior[key,r['original_case_index'],mode] for r in rows];label=f'{key}/N{n}/{mode}'
                cells[label]={'original':{s:old.cell_metrics([r for r in before if r['gold'] in ks]) for s,ks in scopes},
                    'wrapped':{s:old.cell_metrics([r for r in rows if r['gold'] in ks]) for s,ks in scopes}}
                tally=Counter()
                for a,b in zip(before,rows):
                    need(all(a[k]==b[k] for k in ('sid','pair_id','gold','n_frames','content_sha256')),'Comparator semantics differ')
                    tally[a['exact'],b['exact']]+=1
                    transition=dict(core_key=key,mode=mode,sid=b['sid'],pair_id=b['pair_id'],gold=b['gold'],n_frames=n,
                        original_generated_ids=a['generated_ids'],wrapped_generated_ids=b['generated_ids'],
                        original_exact=a['exact'],wrapped_exact=b['exact'],original_parseable=a['parseable'],wrapped_parseable=b['parseable'],
                        original_completed=a['completed'],wrapped_completed=b['completed'],
                        original_raw_file=a['raw_file'],original_raw_sha256=a['raw_sha256'],wrapped_raw_file=b['raw_file'],wrapped_raw_sha256=b['raw_sha256'],
                        prefix_relation=old.prefix_relation(a['generated_ids'],b['generated_ids']))
                    transitions.append(transition);by_k[label+f'/K{b["gold"]}']=transition
                comparisons[label]=dict(n=17,wrapped_minus_original_correct=sum(r['exact'] for r in rows)-sum(r['exact'] for r in before),
                    both_correct=tally[True,True],original_only=tally[True,False],wrapped_only=tally[False,True],neither=tally[False,False])
    allocation=allocation_ledger(out);scheduler={r['job_id']:r for r in allocation['jobs']}
    need(len(job_ids)==4 and job_ids<=set(scheduler) and all(scheduler[j]['job_name']=='v16_step_study'
         and scheduler[j]['state']=='COMPLETED' and scheduler[j]['gpus']==1
         and scheduler[j]['exit_code']=='0:0' for j in job_ids),'Successful run allocation evidence missing')
    need(allocation['allocated_gpu_seconds']<=5400 and sources()==frozen,'Final resource/source gate failed')
    save(out/'rescored_predictions.json',allrows)
    analysis=dict(schema_version=1,passed=True,completed=True,audit_passed=True,policy=POLICY,source_sha256=frozen,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),stage_summary=plan['stage_summary'],
        original_report=plan['original_report'],software_gate=plan['software_gate'],runs=runs,cells=cells,by_k=by_k,
        comparisons=comparisons,transitions=transitions,sentinel_checks=all_sentinels,all_sentinels_passed=True,
        allocation=allocation,records=288,wrapped_records=272,sentinel_records=16,captures_all_prefixes=True,
        capture_count=sum(len(r['generated_ids']) for r in allrows),
        rescored_file=str(out/'rescored_predictions.json'),rescored_sha256=sha(out/'rescored_predictions.json'),
        no_fit=True,no_model_selection=True,no_practical_milestone_claim=True,self_tests=tests,
        limitations=[
            'This is a diagnostic on34 previously evaluated scenes, allfour fixed models; no fresh-confirmation or method-success claim.',
            'The intervention changes only actual-image Step-footer pixels, preserving frame contents/order, question and counting labels.',
            'Repeated visible Step IDs alter the original unique ordinal grammar; rendered task semantics are not claimed identical.',
            'Original versus cyclic1..16 does not distinguish training-range familiarity, glyph magnitude/length and repeated-ID effects.',
            'Original reference images and the N+25 augmented native stream structure are held fixed; reference inference remains diagnostic only.',
            'Original-image sentinels bound observed rerun differences on two fixed scenes, not exact equality on every possible input.',
            'Later original/wrapped or learned/bank traces may have different generated prefixes; they are not same-state causal comparisons.',
            'No output filtering, accuracy eligibility, new fitting, checkpoint selection, attention novelty or reasoning-composition claim.'])
    save(out/'analysis.json',analysis)
    lines=['# V16 Step-label input intervention','',
        'Independent rescoring and all16 original-image sentinel gates passed. Every wrapped trajectory is retained.','',
        '| Model | N | Mean | Original whole answer | Wrapped whole answer | Change |',
        '|---|---:|---|---:|---:|---:|']
    for label,value in cells.items():
        key,n,mode=label.split('/');a=value['original']['all']['correct'];b=value['wrapped']['all']['correct']
        lines.append(f'| {key} | {n[1:]} | {mode} | {a}/17 | {b}/17 | {b-a:+d} |')
    lines+=['','The counting labels and nonfooter image content are held fixed. Cyclic IDs introduce repetitions and therefore change ordinal grammar; no range-specific mechanism or restored general aggregation is identified.','',
        '[All K cells, transitions, raw captures, sentinels and allocated cost](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'summary.json',dict(passed=True,completed=True,audit_passed=True,source_sha256=frozen,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),records=288,wrapped_records=272,
        sentinel_records=16,all_sentinels_passed=True,allocated_gpu_seconds=allocation['allocated_gpu_seconds'],
        no_practical_milestone_claim=True,seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--check',action='store_true')
    mode.add_argument('--run',action='store_true');mode.add_argument('--report',type=Path,nargs=4)
    parser.add_argument('--stage-summary',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--model-index',type=int)
    args=parser.parse_args();need(os.environ.get('SLURM_JOB_ID'),'All work requires Slurm')
    if args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='gpu' and os.environ.get('SLURM_JOB_GPUS'),'Native study requires GPU Slurm')
    else:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'Preparation/tests/report require CPU Slurm')
    import torch
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))))
    label='run' if args.run else 'check' if args.check else 'report' if args.report else 'selftest'
    out=OUT/f'{label}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    if args.self_test:save(out/'summary.json',dict(passed=True,completed=True,tests=self_test(),source_sha256=frozen,no_model_loaded=True))
    elif args.check:
        need(args.stage_summary is not None,'Passed wrapped-image stage summary required');check(args,out,frozen)
    elif args.run:
        need(args.plan is not None and args.model_index is not None,'Require exact CPU plan and model index');run(args,out,frozen)
    else:
        need(args.plan is not None,'Exact common CPU plan required');report(args,out,frozen)
    need(sources()==frozen,'Source changed during execution')
    print(json.dumps(dict(directory=str(out),summary_file=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()
