"""V15 matched-null native software profile; no training or efficacy scoring.

Fixed-prefix structural audits copy complete states/KV. Natural timings retain
all runtime query captures, matching the proposed study, plus a minimal observer
of full-batch LAST-query fused hidden/global logits. No full hidden or KV copies. Every actual model invocation has
exactly one later native-shaped captured-state norm/head replay.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
import math
import re
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from scripts import native_vision_null_comparison_runtime as runtime
from scripts import profile_native_vision_v13_null as learned
from scripts import profile_native_vision_v7_runtime as old
from scripts import profile_native_vision_v12_reference as ancestor
from scripts.profile_native_vision_v11_memory import cpu_copy,runtime_identity,snapshot_kv,cached_input
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,model_metadata
OUT=REPO/'outputs/native_aggregation_vlm/v15/null_comparison_software'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v15_null_comparison_software')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/v15_null_comparison_software')
ANCESTOR=REPO/'outputs/native_aggregation_vlm/v7/runtime/profile_441845/summary.json'
FORCED=('Therefore',':');MODES=('learned','bank');PREDICTOR_SEED=learned.PREDICTOR_SEED
REFERENCE_ANCESTOR=REPO/'outputs/native_aggregation_vlm/v12/reference_software/profile_442314/summary.json'
BUDGET=dict(fixed_model=32,fixed_visual=16,natural_generations=4,maximum_natural_model=16,
    maximum_model=48,visual=20,maximum_native_head_replays=48,extra_last_block_forwards=0)
OWN=tuple(dict.fromkeys(('scripts/profile_native_vision_v15_null_comparison.py',
    'slurm/native_vision_v15_null_comparison_check.sbatch','slurm/native_vision_v15_null_comparison_profile.sbatch',
    *runtime.OWN,*learned.OWN)))


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==digest,'Source changed while copying')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V15 matched-null software\n\n[Summary](summary.json) · [Plan](plan.json) · [Sources](source_hashes.json).\n')
    index=OUT/'INDEX.md'
    if not index.exists():index.write_text('# V15 matched-null software\n\n')
    with index.open('a') as stream:stream.write(f'- [{out.name}]({out.name}/INDEX.md)\n')
    return frozen


predictor_state=learned.predictor_state


def check():
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    command=[sys.executable,'tests/test_parallel_local_null_comparison.py']
    unit=subprocess.run(command,capture_output=True,text=True,cwd=REPO)
    (out/'unit.stdout.txt').write_text(unit.stdout);(out/'unit.stderr.txt').write_text(unit.stderr)
    need(unit.returncode==0 and re.search(r'Ran 7 tests',unit.stderr),'Seven comparison unit tests failed; logs retained')
    hooks=dict(passed=True,tests=7,coefficient_modes=['centered','offset'])
    reference_previous=ancestor.verify_profile(REFERENCE_ANCESTOR)
    previous=read(ANCESTOR);need(previous['computational_integrity_passed'] and previous['zero_identity_passed'],'Native ancestor changed')
    parent=old.verify(Path(previous['plan_file']))
    need(all(reference_previous[k]==parent[k] for k in ('model','runtime','processor','native_api')),
         'Previously verified augmented native backend differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==parent['processor'] and runtime_identity()==parent['runtime'],
        'Processor/runtime differs')
    owner,fn,api=old.local.native_api(processor);need(api==parent['native_api'],'Installed native API changed')
    def rope(**kwargs):return fn(owner,**kwargs)
    forced=[]
    for word in FORCED:
        ids=processor.tokenizer(word,add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Fixed software prefix differs');forced+=ids
    scenes,manifest=old.mixed.selected_sources()
    need(len(scenes)==2 and [s['n_frames'] for s in scenes]==[16,64],'Fixed software cases differ')
    weights=torch.load(parent['initial_file'],map_location='cpu',weights_only=True)
    need(weights['schema_version']==1 and old.state_identity(weights['states'])==parent['initial_state_identity'],'Original core states changed')
    rng=torch.random.get_rng_state().clone();prediction=predictor_state(torch)
    need(torch.equal(rng,torch.random.get_rng_state()),'Software predictor initialization changed caller RNG')
    bundles={};cases=[]
    for sample in scenes:
        bank=runtime.reference_bank(sample['question'])
        base=runtime.prepare_scene(processor,sample,bank,verify_processor_parity=True);case_id=f'parallel_N{sample["n_frames"]}';variants=[]
        for step in range(3):
            bundle=runtime.append_observed_prefix(base,forced[:step]);runtime.validate_bundle(bundle);layout=native.audit_layout(rope,bundle)
            bundles[f'{case_id}_t{step}']=bundle
            variants.append(dict(input_identity=bundle['metadata']['input_identity'],layout=layout['metadata']))
        cases.append(dict(case_id=case_id,sid=sample['sid'],n_frames=sample['n_frames'],question=sample['question'],
            prompt_width=base['metadata']['original_prompt_width'],sample=sample,reference_bank=bank,reference_bank_sha256=bank['bank_sha256'],
            metadata=base['metadata'],variants=variants))
    data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False)
    weights_dir=CKPT/f'check_{job}';weights_dir.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema_version=1,bundles=bundles),data/'prepared.pt')
    torch.save(dict(schema_version=1,states=weights['states'],predictor=prediction),weights_dir/'initial.pt')
    plan=dict(schema_version=1,protocol='v15_null_comparison_software_only',source_sha256=frozen,
        ancestor_file=str(ANCESTOR),ancestor_sha256=sha(ANCESTOR),
        reference_ancestor_file=str(REFERENCE_ANCESTOR),reference_ancestor_sha256=sha(REFERENCE_ANCESTOR),
        reference_manifest=dict(file=str(runtime.REFERENCE_MANIFEST),sha256=sha(runtime.REFERENCE_MANIFEST)),parent_plan_file=previous['plan_file'],parent_plan_sha256=sha(previous['plan_file']),
        original_mixed_failure=parent['original_mixed_failure'],model=model_metadata(),runtime=runtime_identity(),processor=parent['processor'],native_api=api,
        initial_file=str(weights_dir/'initial.pt'),initial_sha256=sha(weights_dir/'initial.pt'),initial_state_identity=parent['initial_state_identity'],
        predictor_state_identity=old.state_identity({'predictor':prediction})['predictor'],predictor_seed=PREDICTOR_SEED,
        predictor_initialization='privateCPU seed20261104; C1 default; C2 weight Normal(0,.001), bias0; no fitting',
        prepared_file=str(data/'prepared.pt'),prepared_sha256=sha(data/'prepared.pt'),source_manifest=manifest,
        cases=cases,forced_text=list(FORCED),forced_ids=forced,modes=list(MODES),maximum_calls=BUDGET,
        parameters=dict(core=1041600,predictor=18624,total=1060224),reference_rows=24,condition='centered',core_key='software',
        reference_mean_rule='FP64 occurrence mean of FP32 messages cast to FP32',coefficient_rule='actualN for centered;1 for offset; no anchor',
        natural_timing_observer='runtime query captures enabled plus full-batch last-query fused h/global logits; no full hidden/KV copying',
        numerical_policy='captured full-batch native head TV<=.02/exacttop1 binding; cached/full descriptive',
        no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True,controller_tests=hooks,
        unit_command=command,unit_stdout_sha256=sha(out/'unit.stdout.txt'),unit_stderr_sha256=sha(out/'unit.stderr.txt'),slurm_job_id=job)
    need(sources()==frozen,'Sources changed during CPU freeze')
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify(out/'plan.json')
    save(out/'summary.json',dict(passed=True,tests_passed=True,controller_tests=hooks,source_sha256=frozen,plan_file=str(out/'plan.json'),
        plan_sha256=sha(out/'plan.json'),no_model_loaded=True,seconds=time.perf_counter()-begin,slurm_job_id=job))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify(path):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(OUT) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Plan path/sidecar differs')
    need(plan['schema_version']==1 and plan['protocol']=='v15_null_comparison_software_only' and plan['source_sha256']==sources()
        and plan['model']==model_metadata() and plan['runtime']==runtime_identity() and plan['modes']==list(MODES)
        and plan['maximum_calls']==BUDGET and plan['predictor_seed']==PREDICTOR_SEED
        and plan['parameters']==dict(core=1041600,predictor=18624,total=1060224),'Source/native/policy identity differs')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Source copy changed')
    for name in ('ancestor','reference_ancestor','parent_plan','initial','prepared'):need(sha(plan[name+'_file'])==plan[name+'_sha256'],'Bound artifact changed')
    need(sha(plan['source_manifest']['path'])==plan['source_manifest']['sha256'],'Original software data manifest changed')
    need([c['n_frames'] for c in plan['cases']]==[16,64] and plan['reference_rows']==24 and plan['condition']=='centered'
         and plan['core_key']=='software' and sha(plan['reference_manifest']['file'])==plan['reference_manifest']['sha256'],
         'Software/reference rows or centered coefficient differ')
    need(plan['forced_text']==list(FORCED) and len(plan['forced_ids'])==2
         and plan['controller_tests']==dict(passed=True,tests=7,coefficient_modes=['centered','offset']),
         'CPU software test/prefix contract differs')
    for label in ('stdout','stderr'):
        need(sha(path.parent/('unit.'+label+'.txt'))==plan['unit_'+label+'_sha256'],'CPU test log changed')
    for case in plan['cases']:
        need(runtime.reference_bank(case['question'])==case['reference_bank'] and case['reference_bank_sha256']==case['reference_bank']['bank_sha256'],
             'Ordered reference bank changed')
    return plan


def verify_fusion(raw,controller,query,stream):
    """Replay only the small core on the same device; no VLM or head calls."""
    import torch
    from gnnformer.conditional_null_mean import projected_null_readout
    c=raw['fusion'];h=raw['native_norm_input'];n=controller.n_actual_rows
    need(c is not None and c['query_indices']==query and c['stream_positions']==stream
         and c['n_actual_rows']==n and c['n_reference_rows']==24
         and c['condition']==controller.condition=='centered' and c['mean_source']==controller.mean_source,
         'Explicit query, coefficient or augmented partition differs')
    need(torch.equal(c['actual_states'],h[:n,query]) and torch.equal(c['reference_states'],h[n:n+24,query])
         and torch.equal(c['global_states'],h[-1,query]),'Common raw query states differ')
    core=controller.core;device=core.up.weight.device
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        g=c['global_states'].to(device)
        actual=core.encode(c['actual_states'].to(device),g);refs=core.encode(c['reference_states'].to(device),g)
        aggregate=core.aggregate(actual);q=core.query(core.rms(g));pred=controller.predictor(q);bank=refs.double().mean(0).float()
        used=pred if controller.mean_source=='learned' else bank
        result=projected_null_readout(core,aggregate,g,used,n_elements=n,mode='centered',output_dtype=torch.float32)
    expected_values=dict(actual_messages=actual,reference_messages=refs,aggregate=aggregate,query=q,predicted_mean=pred,bank_mean=bank,
        used_mean=used,preactivation=result['preactivation'],projected_null=result['projected_null'],
        corrected_preactivation=result['corrected_preactivation'],delta=result['delta_float32'])
    need(all(torch.equal(c[k],v.cpu()) for k,v in expected_values.items()),'Same-device mean/query/core replay differs')
    expected=h.clone();expected[-1,query]=h[-1,query]+c['delta'].half()
    need(c['coefficient']==n and c['delta'].dtype==torch.float32 and torch.equal(expected,raw['fused_norm_input'])
         and torch.equal(expected[-1,query],c['fused_global']),'Native cast/write or actualN coefficient differs')
    return dict(passed=True,query_indices=query,stream_positions=stream,mode=controller.mean_source,condition='centered',n_frames=n,
        coefficient=n,delta_nonzero=bool(c['delta'].ne(0).any()),actual_reference_rows_untouched=True,small_core_replay_exact=True)


MinimalTimingAudit=learned.MinimalTimingAudit


def replay_heads(torch,model,records):
    fixed=[r for r in records if r.get('capture_kind')!='last_query_only']
    natural=[r for r in records if r.get('capture_kind')=='last_query_only']
    # Immutable V12 replay for fixed captures, V13 minimal form for timing.
    a,araw=ancestor.replay_heads(torch,model,fixed)
    b,braw=learned.replay_heads(torch,model,natural)
    return a+b,araw+braw


def run(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.conditional_null_mean import ConditionalNullMean
    from gnnformer.parallel_local_null_comparison import ParallelLocalNullComparison
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'profile_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    data=DATA/f'profile_{job}';data.mkdir(parents=True,exist_ok=False)
    (data/'INDEX.md').write_text('# V15 comparison software raw evidence\n\n[Head replays](head_replays.pt).\n')
    plan=verify(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] and cpu['tests_passed'] and cpu['controller_tests']['passed'] and cpu['plan_sha256']==sha(args.plan),
        'Require completed exact CPU gate')
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==weights['schema_version']==1 and old.state_identity(weights['states'])==plan['initial_state_identity']
        and old.state_identity({'predictor':weights['predictor']})['predictor']==plan['predictor_state_identity']
        and all(torch.equal(weights['predictor'][k],v) for k,v in predictor_state(torch).items()),'Initial states or prepared schema differ')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model)
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and runtime_identity()==plan['runtime'],
        'Loaded processor/runtime differs')
    _,_,api=old.local.native_api(loaded.processor);need(api==plan['native_api'],'Loaded native API differs')
    core=ParallelLocalAggregation().eval().requires_grad_(False).to(model.device)
    predictor=ConditionalNullMean().eval().requires_grad_(False);predictor.load_state_dict(weights['predictor'],strict=True);predictor.to(model.device)
    native.native_contract(model,core)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    versions={k:p._version for k,p in model.named_parameters()};predictor_versions={k:p._version for k,p in predictor.named_parameters()}
    torch.cuda.reset_peak_memory_stats();checks=[];zero=[];comparisons=[];controller_counts=[];timings=[];fixed_records=[];natural_records=[]
    fixed_counts={};natural_counts={};all_records=[];natural_capture_checks=[]
    try:
        with ancestor.NativeAudit(model,data) as audit:
            try:
                for case in plan['cases']:
                    cid=case['case_id'];n=case['n_frames'];width=case['prompt_width'];tokens=plan['forced_ids']
                    bundles=[blob['bundles'][f'{cid}_t{t}'] for t in range(3)]
                    for t,bundle in enumerate(bundles):
                        need(runtime.validate_bundle(bundle)['passed'] and bundle['metadata']['input_identity']==case['variants'][t]['input_identity']
                            and native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['variants'][t]['layout'],
                            'Frozen native layout differs')
                    baseline=[];baseline_kv=[];native_full={};cache=None;audit.controller=None
                    for step,bundle in enumerate(bundles):
                        inputs,positions,layout=cached_input(model,bundle,cache,step,tokens,model.device)
                        audit.context=dict(label=f'{cid}__native__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,condition='native')
                        with torch.inference_mode():
                            output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                        cache=output.past_key_values;raw=audit.last_raw
                        need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],raw['fused_norm_input']),
                            'Bare native positions/readout differ')
                        baseline.append(raw);baseline_kv.append(snapshot_kv(cache))
                    del cache,output
                    for condition,mode in [('zero','bank'),('active','learned'),('active','bank')]:
                        core.load_state_dict(weights['states'][condition],strict=True)
                        need(old.state_identity({condition:core.state_dict()})[condition]==plan['initial_state_identity'][condition],
                            'Fixed original core changed')
                        cache=None
                        controller=ParallelLocalNullComparison(norm,core,predictor,n_actual_rows=n,condition='centered',mean_source=mode,
                            query_indices=[width-1],stream_positions=[width-1],capture=True)
                        with controller:
                            audit.controller=controller
                            for step,bundle in enumerate(bundles):
                                query=[width-1] if step==0 else [0];stream=[width+step-1];controller.configure_queries(query,stream)
                                inputs,positions,layout=cached_input(model,bundle,cache,step,tokens,model.device)
                                audit.context=dict(label=f'{cid}__{condition}__{mode}__cached{step}',case_id=cid,n_frames=n,
                                    phase='cached',step=step,condition=condition,mode=mode)
                                with torch.inference_mode():
                                    output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                                cache=output.past_key_values;raw=audit.last_raw
                                need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],baseline[step]['native_norm_input']),
                                    'Learned-null mode changed fixed-prefix native hidden or positions')
                                checked=verify_fusion(raw,controller,query,stream)
                                checked.update(label=audit.context['label'],native_hidden_exact=True,
                                    all_layer_kv_exact=ancestor.all_kv_equal(cache,baseline_kv[step]));checks.append(checked)
                                if condition=='zero':
                                    need(not checked['delta_nonzero'] and torch.equal(raw['native_logits'],baseline[step]['native_logits']),
                                        'Zero U changed native output')
                                    zero.append(dict(label=audit.context['label'],passed=True,raw_logits_exact=True))
                                else:
                                    need(checked['delta_nonzero'],'Active software core produced no residual')
                                    if step>0:
                                        history=list(range(width-1,width+step));controller.configure_queries(history,history)
                                        audit.context=dict(label=f'{cid}__{condition}__{mode}__full{step}',case_id=cid,n_frames=n,
                                            phase='full',step=step,condition=condition,mode=mode)
                                        rope=cpu_copy(model.model.rope_deltas)
                                        try:
                                            with torch.inference_mode():full_output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
                                        finally:model.model.rope_deltas=rope.to(model.device)
                                        full=audit.last_raw
                                        need(full_output.past_key_values is None and torch.equal(full['position_ids'],layout['position_ids']),
                                            'Full prefix positions/cache differ')
                                        if mode=='learned':native_full[step]=full['native_norm_input']
                                        else:need(torch.equal(native_full[step],full['native_norm_input']),'Mode changed full-prefix common native reads')
                                        checked=verify_fusion(full,controller,history,history)
                                        checked.update(label=audit.context['label'],native_hidden_exact=True,all_layer_kv_exact=None);checks.append(checked)
                                        for row in range(n+25):
                                            metric=old.metric(torch,raw['native_logits'][row,-1],full['native_logits'][row,-1])
                                            metric.update(case_id=cid,n_frames=n,mode=mode,step=step,row_index=row,
                                                role='actual' if row<n else 'reference' if row<n+24 else 'global',binding=False);comparisons.append(metric)
                                        del full,full_output
                            wanted=3 if condition=='zero' else 5
                            need(controller.calls==wanted,'Controller fixed invocation count differs')
                            controller_counts.append(dict(case_id=cid,condition=condition,mode=mode,calls=controller.calls,passed=True))
                        audit.controller=None;need(not controller.active,'Controller survived cleanup');del cache,output
                    del baseline,baseline_kv,native_full,bundles,raw
                need(audit.counts==dict(model=32,visual=16,language=32,norm=32,last_block=32),'Fixed call inventory differs')
            finally:
                fixed_records=audit.records;fixed_counts=dict(audit.counts)
                save(out/'fixed_forwards.json',fixed_records);save(out/'fixed_calls.json',fixed_counts)
                save(out/'fixed_structural_checks.json',dict(checks=checks,zero_checks=zero,controller_counts=controller_counts))
                save(out/'cached_full_comparisons.json',comparisons)
        need(len(zero)==6 and len(checks)==26 and len(controller_counts)==6 and len(comparisons)==520
            and sum(c['all_layer_kv_exact'] is not None for c in checks)==18,'Fixed structural coverage differs')
        need(all(c['all_layer_kv_exact'] is None or (type(c['all_layer_kv_exact']) is list and len(c['all_layer_kv_exact'])==28
             and all(v is True for v in c['all_layer_kv_exact'])) for c in checks),'Require complete literal native KV equality lists')
        core.load_state_dict(weights['states']['active'],strict=True)
        with MinimalTimingAudit(model,data) as timing_audit:
            try:
                for case in plan['cases']:
                    for mode in MODES:
                        tick=time.perf_counter();bundle=runtime.prepare_scene(loaded.processor,case['sample'],case['reference_bank'])
                        prep=time.perf_counter()-tick;start_record=len(timing_audit.records)
                        timing_audit.context=dict(label=f'{case["case_id"]}__{mode}__natural',case_id=case['case_id'],n_frames=case['n_frames'],mode=mode)
                        tick=time.perf_counter()
                        generated=runtime.generate_native(model,loaded.processor,core,predictor,bundle,condition='centered',mean_source=mode,core_key='software',capture=True)
                        torch.cuda.synchronize();generation_seconds=time.perf_counter()-tick
                        records=timing_audit.records[start_record:];steps=len(generated['generated_ids'])
                        need(len(records)==len(generated['captures'])==steps and 1<=steps<=4,
                            'Natural timing must retain each query capture, matching study computation')
                        need(generated['raw_logits'].dtype==torch.float32
                            and torch.equal(generated['raw_logits'],generated['raw_logits'].half().float()),
                            'Generated global vectors are not native FP16 losslessly promoted to FP32')
                        for t,record in enumerate(records):
                            raw=torch.load(record['path'],map_location='cpu',weights_only=True)
                            need(torch.equal(generated['raw_logits'][t],raw['native_global_logits'])
                                and record['retained_hidden_shape']==[case['n_frames']+25,1,3584]
                                and record['kv_snapshots']==record['full_hidden_copies']==0,'Minimal timing observation differs')
                            c=generated['captures'][t];n=case['n_frames'];width=bundle['metadata']['prompt_width']
                            query=[width-1] if t==0 else [0];stream=[width+t-1]
                            before=torch.cat((c['actual_states'],c['reference_states'],c['global_states'].unsqueeze(0)),dim=0)
                            current=ParallelLocalNullComparison(norm,core,predictor,n_actual_rows=n,condition='centered',mean_source=mode,
                                query_indices=[0],stream_positions=stream)
                            # Runtime prefill physical index is width-1, while the
                            # minimal archive contains only that single query.
                            compact=dict(c,query_indices=[0])
                            check=verify_fusion(dict(fusion=compact,native_norm_input=before,fused_norm_input=raw['fused_query_hidden']),
                                current,[0],stream)
                            need(c['query_indices']==query and c['stream_positions']==stream,'Natural capture prefix ownership differs')
                            check.update(case_id=case['case_id'],step=t,physical_query_indices=query,label=record['label'])
                            natural_capture_checks.append(check)
                        path=data/f'{case["case_id"]}__{mode}__natural.pt';torch.save(generated,path)
                        timings.append(dict(sid=case['sid'],n_frames=case['n_frames'],mode=mode,generated_tokens=steps,
                            preparation_seconds=prep,generation_seconds=generation_seconds,runtime_internal_seconds=generated['model_seconds'],
                            four_token_seconds_bound=prep+generation_seconds*4/steps,raw_file=str(path),raw_sha256=sha(path),
                            counters=generated['counters'],query_captures_retained=True,no_efficacy_scoring=True,no_full_hidden_or_kv_copies=True))
            finally:
                natural_records=timing_audit.records;natural_counts=dict(timing_audit.counts)
                save(out/'natural_forwards.json',natural_records);save(out/'natural_calls.json',natural_counts)
                save(out/'natural_timings.json',timings)
                save(out/'natural_capture_checks.json',natural_capture_checks)
        steps=sum(t['generated_tokens'] for t in timings)
        need(len(timings)==4 and 4<=steps<=16 and natural_counts==dict(model=steps,visual=4,language=steps,norm=steps,last_block=steps),
            'Natural timing call inventory differs')
        all_records=fixed_records+natural_records
        metrics,raw_metrics=replay_heads(torch,model,all_records)
        need(len(metrics)==32+steps<=48,'One extra native-shaped head replay per actual model call required')
        torch.save(raw_metrics,data/'head_replays.pt');save(out/'head_replay_metrics.json',metrics)
    finally:
        save(out/'actual_calls.json',dict(fixed=fixed_counts,natural=natural_counts,
            total={k:fixed_counts.get(k,0)+natural_counts.get(k,0) for k in ('model','visual','language','norm','last_block')}))
    need(versions=={k:p._version for k,p in model.named_parameters()} and predictor_versions=={k:p._version for k,p in predictor.named_parameters()}
        and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone/predictor changed')
    need(old.state_identity({'active':core.state_dict()})['active']==plan['initial_state_identity']['active']
        and old.state_identity({'predictor':predictor.state_dict()})['predictor']==plan['predictor_state_identity'],'Fixed software parameters changed')
    need(sources()==frozen,'Source changed during profile')
    failures=[m for m in metrics if not m['numerical_rule_passed']];descriptive=[m for m in comparisons if not m['numerical_rule_passed']]
    timing=dict(rows=timings,by_mode={mode:{str(t['n_frames']):t['four_token_seconds_bound'] for t in timings if t['mode']==mode} for mode in MODES},
        bound_rule='preparation_seconds + measured_whole_generation_call_seconds*4/generated_tokens',
        observer='runtime query captures enabled plus full-batch last-query fused h/global logits; no full hidden/KV copies',maximum_new_tokens=4)
    save(out/'timing.json',timing)
    artifacts={p.name:dict(path=str(p),sha256=sha(p)) for p in sorted(out.glob('*.json'))}
    artifacts['head_replays.pt']=dict(path=str(data/'head_replays.pt'),sha256=sha(data/'head_replays.pt'))
    totals={k:fixed_counts[k]+natural_counts[k] for k in fixed_counts}
    summary=dict(schema_version=1,protocol=plan['protocol'],condition='centered',modes=list(MODES),reference_rows=24,completed=True,passed=not failures,computational_integrity_passed=True,
        zero_identity_passed=True,fixed_prefix_native_kv_unchanged=True,fixed_prefix_hidden_unchanged=True,native_head_replay_passed=not failures,
        numerical_failures=failures,cached_full_failures=descriptive,cached_full_differences_descriptive=True,
        original_mixed_failure=plan['original_mixed_failure'],original_mixed_numerical_gate_passed=False,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        initial_state_identity=plan['initial_state_identity'],predictor_state_identity=plan['predictor_state_identity'],parameters=plan['parameters'],
        calls=totals,fixed_calls=fixed_counts,natural_calls=natural_counts,native_head_replay_calls=len(metrics),maximum_calls=BUDGET,
        extra_last_block_forwards=0,small_same_device_core_replays=26+steps,natural_query_capture_checks=steps,fixed_prefix_kv_checks=18,fixed_prefix_fusion_checks=26,zero_identity_checks=6,cached_full_row_comparisons=520,
        timing=timing,load_seconds=load_seconds,seconds=time.perf_counter()-begin,slurm_job_id=job,
        native_dtypes=dict(norm=str(norm.weight.dtype),head=str(model.lm_head.weight.dtype)),
        hardware=dict(gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        artifacts=artifacts,frozen_backbone_unchanged=True,core_unfitted_unchanged=True,predictor_unfitted_unchanged=True,
        no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True)
    save(out/'summary.json',summary)
    print(json.dumps(dict(directory=str(out),passed=summary['passed'],calls=totals,binding_failures=len(failures),
        descriptive_failures=len(descriptive),timing=timing)),flush=True)
    need(summary['passed'],'Captured native head replay failed; preserve all observations')


def verify_profile(path):
    path=Path(path).resolve();summary=read(path)
    need(path.is_relative_to(OUT) and path.name=='summary.json' and summary.get('schema_version')==1
        and summary.get('protocol')=='v15_null_comparison_software_only' and all(summary.get(k) is True for k in
        ('completed','passed','computational_integrity_passed','zero_identity_passed','fixed_prefix_native_kv_unchanged',
         'fixed_prefix_hidden_unchanged','native_head_replay_passed','frozen_backbone_unchanged','core_unfitted_unchanged',
         'predictor_unfitted_unchanged','no_fit','no_efficacy_scoring')),'Require completed successful native V15 profile')
    need(summary['condition']=='centered' and summary['modes']==list(MODES) and summary['reference_rows']==24
         and summary['cached_full_differences_descriptive'] is True
         and summary['original_mixed_numerical_gate_passed'] is False,'Mean/coefficient or retained numerical scope differs')
    plan=verify(summary['plan_file'])
    need(sha(summary['plan_file'])==summary['plan_sha256'] and summary['source_sha256']==plan['source_sha256'],'CPU/GPU source binding differs')
    for name,digest in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'GPU source copy differs')
    for field in ('model','runtime','processor','native_api','initial_state_identity','predictor_state_identity','parameters'):
        need(summary[field]==plan[field],'Native/software identity differs: '+field)
    for item in summary['artifacts'].values():need(sha(item['path'])==item['sha256'],'Software artifact changed')
    need(read(path.parent/'timing.json')==summary['timing'],'Timing summary copy changed')
    rows=summary['timing']['rows'];need([(r['n_frames'],r['mode']) for r in rows]==[(n,m) for n in (16,64) for m in MODES],
        'Natural timing coverage/order differs')
    for row in rows:
        need(all(math.isfinite(row[k]) and row[k]>0 for k in ('preparation_seconds','generation_seconds','four_token_seconds_bound')),
             'Nonfinite or nonpositive natural timing')
        case=plan['cases'][0 if row['n_frames']==16 else 1]
        need(row['sid']==case['sid'] and 1<=row['generated_tokens']<=4 and row['preparation_seconds']>0 and row['generation_seconds']>0
            and row['four_token_seconds_bound']==row['preparation_seconds']+row['generation_seconds']*4/row['generated_tokens']
            and summary['timing']['by_mode'][row['mode']][str(row['n_frames'])]==row['four_token_seconds_bound']
            and row['no_full_hidden_or_kv_copies'] and row['query_captures_retained'] is True and sha(row['raw_file'])==row['raw_sha256'],'Natural timing binding differs')
    steps=sum(r['generated_tokens'] for r in rows);total=32+steps
    need(summary['small_same_device_core_replays']==26+steps and summary['natural_query_capture_checks']==steps,
         'Natural query capture/replay coverage differs')
    need(total<=48 and summary['calls']==dict(model=total,visual=20,language=total,norm=total,last_block=total)
        and summary['fixed_calls']==dict(model=32,visual=16,language=32,norm=32,last_block=32)
        and summary['natural_calls']==dict(model=steps,visual=4,language=steps,norm=steps,last_block=steps)
        and summary['native_head_replay_calls']==total and summary['extra_last_block_forwards']==0 and summary['maximum_calls']==BUDGET
        and summary['fixed_prefix_kv_checks']==18 and summary['fixed_prefix_fusion_checks']==26 and summary['zero_identity_checks']==6
        and summary['cached_full_row_comparisons']==520 and not summary['numerical_failures'],'Software call/structural coverage differs')
    fixed=read(path.parent/'fixed_forwards.json');natural=read(path.parent/'natural_forwards.json');metrics=read(path.parent/'head_replay_metrics.json')
    need(len(fixed)==32 and len(natural)==steps and len(metrics)==total
        and all(r['numerical_rule_passed'] and r['top1_equal'] and r['full_vocabulary_tv']<=.02 for r in metrics),'Head/forward evidence incomplete')
    for row in fixed+natural:need(sha(row['path'])==row['sha256'],'Raw native evidence changed')
    need(all(r['kv_snapshots']==r['full_hidden_copies']==0 and r['retained_hidden_shape'][1]==1 for r in natural),
        'Timing retained full hidden states or K/V')
    captures=read(path.parent/'natural_capture_checks.json')
    need(len(captures)==steps and all(r['passed'] is True and r['small_core_replay_exact'] is True for r in captures),
         'Natural query mean/delta capture checks incomplete')
    structure=read(path.parent/'fixed_structural_checks.json');checks=structure['checks'];zero=structure['zero_checks'];controls=structure['controller_counts']
    need(len(checks)==26 and len(zero)==6 and len(controls)==6 and all(r['passed'] is True for r in checks+zero+controls)
         and sum(r['all_layer_kv_exact'] is None for r in checks)==8
         and sum(type(r['all_layer_kv_exact']) is list for r in checks)==18
         and all(r['native_hidden_exact'] is True and r['small_core_replay_exact'] is True
             and (r['all_layer_kv_exact'] is None or (len(r['all_layer_kv_exact'])==28 and all(v is True for v in r['all_layer_kv_exact']))) for r in checks),
         'Fixed query/KV/zero/core-replay evidence is incomplete')
    need(Counter((r['case_id'],r['condition'],r['mode'],r['calls']) for r in controls)==Counter(
         (f'parallel_N{n}',kind,mode,3 if kind=='zero' else 5) for n in (16,64)
         for kind,mode in [('zero','bank'),('active','learned'),('active','bank')]),'Controller calls or conditions differ')
    cached=read(path.parent/'cached_full_comparisons.json')
    need(len(cached)==520 and Counter((r['n_frames'],r['mode'],r['step'],r['row_index']) for r in cached)==Counter(
         (n,mode,t,i) for n in (16,64) for mode in MODES for t in (1,2) for i in range(n+25)),
         'Cached/full row coverage missing or duplicated')
    for r in metrics+cached:
        need(type(r['numerical_rule_passed']) is bool and type(r['top1_equal']) is bool
             and math.isfinite(r['full_vocabulary_tv']) and 0<=r['full_vocabulary_tv']<=1
             and r['numerical_rule_passed']==(r['top1_equal'] and r['full_vocabulary_tv']<=.02),
             'Stored native numerical decision differs')
    need(all(r['binding'] is False and r['role']==('actual' if r['row_index']<r['n_frames'] else
         'reference' if r['row_index']<r['n_frames']+24 else 'global') for r in cached)
         and summary['cached_full_failures']==[r for r in cached if not r['numerical_rule_passed']],
         'Descriptive cached/full failures or actual/reference/global ownership changed')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU freeze only');check()
    else:
        need(args.plan is not None,'Pass the exact completed CPU plan');run(args)


if __name__=='__main__':main()
