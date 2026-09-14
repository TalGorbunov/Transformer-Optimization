"""Bounded V12 native reference-stream software check; no fitting or accuracy.

Every mode executes N actual +24 fixed K0 training reference image rows and one
text-only global row. References alter only the final readout; native K/V and
current actual/reference/global states remain unchanged at fixed token prefixes.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from scripts import native_vision_v12_runtime as reference
from scripts import profile_native_vision_v7_runtime as old
from scripts.profile_native_vision_v11_memory import cpu_copy,runtime_identity,snapshot_kv,cached_input
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,model_metadata
OUT=REPO/'outputs/native_aggregation_vlm/v12/reference_software'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v12_reference_software')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/v12_reference_software')
ANCESTOR=REPO/'outputs/native_aggregation_vlm/v7/runtime/profile_441845/summary.json'
FORCED=('Therefore',':')
MODES=('base','background','sham')
BUDGET=dict(fixed_model=42,fixed_visual=22,natural_generations=2,maximum_natural_model=8,
    maximum_model=50,visual=24,maximum_native_head_replays=50,extra_last_block_forwards=0)
OWN=tuple(dict.fromkeys(('scripts/profile_native_vision_v12_reference.py','scripts/native_vision_v12_runtime.py',
    'gnnformer/parallel_local_reference.py','tests/test_parallel_local_reference.py',
    'slurm/native_vision_v12_reference_check.sbatch','slurm/native_vision_v12_reference_profile.sbatch',
    'scripts/profile_native_vision_v11_memory.py',*old.OWN)))


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes())
        need(sha(p)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen);return frozen


def index(out,title,links):
    (out/'INDEX.md').write_text('# '+title+'\n\n'+' · '.join('['+label+']('+path+')' for label,path in links)+'\n')


def check():
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    index(out,'V12 reference software CPU freeze',[('Plan','plan.json'),('Summary','summary.json'),('Source','source_hashes.json')])
    command=[sys.executable,'tests/test_parallel_local_reference.py'];unit=subprocess.run(command,capture_output=True,text=True,cwd=REPO)
    (out/'unit.stdout.txt').write_text(unit.stdout);(out/'unit.stderr.txt').write_text(unit.stderr)
    need(unit.returncode==0,'Reference controller tests failed; preserve all logs')
    previous=read(ANCESTOR);need(previous['computational_integrity_passed'] and previous['zero_identity_passed'],'Old native software gate changed')
    parent=old.verify(Path(previous['plan_file']));processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==parent['processor'] and runtime_identity()==parent['runtime'],'Native processor/runtime differs')
    owner,fn,api=old.local.native_api(processor);need(api==parent['native_api'],'Installed native API changed')
    def rope(**kwargs):return fn(owner,**kwargs)
    forced=[]
    for word in FORCED:
        ids=processor.tokenizer(word,add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Fixed forced software prefix differs');forced+=ids
    scenes,manifest=old.mixed.selected_sources();need(len(scenes)==2 and [s['n_frames'] for s in scenes]==[16,64],'Old fixed software cases differ')
    states=torch.load(parent['initial_file'],map_location='cpu',weights_only=True)
    need(states['schema_version']==1 and old.state_identity(states['states'])==parent['initial_state_identity'],'Fixed unfitted branch states changed')
    bundles={};cases=[]
    for sample in scenes:
        bank=reference.reference_bank(sample['question']);base=reference.prepare_scene(processor,sample,bank,verify_processor_parity=True)
        case_id=f'parallel_N{sample["n_frames"]}';variants=[]
        for step in range(3):
            bundle=old.prefixed_bundle(base,forced[:step]);reference.validate_bundle(bundle);layout=native.audit_layout(rope,bundle)
            bundles[f'{case_id}_t{step}']=bundle
            variants.append(dict(input_identity=bundle['metadata']['input_identity'],layout=layout['metadata']))
        cases.append(dict(case_id=case_id,sid=sample['sid'],actual_n_frames=sample['n_frames'],question=sample['question'],
            prompt_width=base['metadata']['original_prompt_width'],reference_bank=bank,reference_bank_sha256=bank['bank_sha256'],
            sample=sample,metadata=base['metadata'],variants=variants))
    data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False)
    weights=CKPT/f'check_{job}';weights.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema_version=1,bundles=bundles),data/'prepared.pt');torch.save(states,weights/'initial.pt')
    index(data,'V12 immutable augmented software inputs',[('Prepared tensors','prepared.pt')]);index(weights,'V12 fixed unfitted branch states',[('States','initial.pt')])
    plan=dict(schema_version=1,protocol='v12_native_reference_software_only',source_sha256=frozen,
        ancestor_file=str(ANCESTOR),ancestor_sha256=sha(ANCESTOR),parent_plan_file=previous['plan_file'],parent_plan_sha256=sha(previous['plan_file']),
        original_mixed_failure=parent['original_mixed_failure'],model=model_metadata(),runtime=runtime_identity(),processor=parent['processor'],native_api=api,
        initial_file=str(weights/'initial.pt'),initial_sha256=sha(weights/'initial.pt'),initial_state_identity=parent['initial_state_identity'],
        prepared_file=str(data/'prepared.pt'),prepared_sha256=sha(data/'prepared.pt'),source_manifest=manifest,
        reference_manifest=dict(file=str(reference.REFERENCE_MANIFEST),sha256=sha(reference.REFERENCE_MANIFEST)),
        cases=cases,forced_text=list(FORCED),forced_ids=forced,core_key='software',anchor_n=16,modes=list(MODES),maximum_calls=BUDGET,
        reference_mean_dtype='FP64 mean of FP32 reference messages followed by FP32 cast',
        sham_key_rule='[core_key,exact_question], seed prefix20261101; independent N/prefix/mode/order',
        input_rule='actual image rows then all24 ordered K0 training reference occurrences then global text; no reference/QA/gold tokens',
        query_rule='cached current query; full prefix all historical assistant prediction positions',
        numerical_policy='same captured native full-batch norm/head TV<=.02 plus exacttop1 binding; cached/full differences descriptive',
        no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True,
        unit_command=command,unit_stdout_sha256=sha(out/'unit.stdout.txt'),unit_stderr_sha256=sha(out/'unit.stderr.txt'),slurm_job_id=job)
    need(sources()==frozen,'Sources changed during CPU freeze')
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify(out/'plan.json')
    save(out/'summary.json',dict(passed=True,source_sha256=frozen,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        tests_passed=True,no_model_loaded=True,maximum_calls=BUDGET,seconds=time.perf_counter()-begin,slurm_job_id=job))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify(path):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(OUT) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'V12 plan path/sidecar differs')
    need(plan['schema_version']==1 and plan['protocol']=='v12_native_reference_software_only' and plan['source_sha256']==sources()
        and plan['model']==model_metadata() and plan['runtime']==runtime_identity() and plan['maximum_calls']==BUDGET
        and plan['modes']==list(MODES) and plan['anchor_n']==16 and plan['core_key']=='software','V12 source/native/policy identity differs')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'Frozen source copy differs')
    for name in ('ancestor','parent_plan','initial','prepared'):need(sha(plan[name+'_file'])==plan[name+'_sha256'],'Frozen artifact changed: '+name)
    need(sha(plan['source_manifest']['path'])==plan['source_manifest']['sha256']
        and sha(plan['reference_manifest']['file'])==plan['reference_manifest']['sha256'],'Software/reference source manifest changed')
    need([c['actual_n_frames'] for c in plan['cases']]==[16,64],'Fixed software lengths differ')
    for case in plan['cases']:
        need(reference.reference_bank(case['question'])==case['reference_bank'],'Ordered training reference bank changed')
    return plan


class NativeAudit:
    """Observe each actual model invocation; replay heads only after controllers close."""
    def __init__(self,model,data):
        self.model,self.data=model,data;self.records=[];self.handles=[];self.per_call=[]
        self.context={};self.controller=None;self.counts=dict(model=0,visual=0,language=0,norm=0,last_block=0)
    def __enter__(self):
        from gnnformer.runtime import get_layers
        self.layers=get_layers(self.model)
        def increment(name):
            def callback(*_):self.counts[name]+=1
            return callback
        self.handles=[self.model.register_forward_pre_hook(self.before,with_kwargs=True),self.model.register_forward_hook(self.after),
            self.model.model.visual.register_forward_pre_hook(increment('visual')),
            self.layers[-1].register_forward_pre_hook(increment('last_block'))]
        return self
    def __exit__(self,*_):
        for h in self.per_call+self.handles:h.remove()
        return False
    def before(self,module,args,kwargs):
        self.counts['model']+=1;self.started=time.perf_counter();self.raw=dict(input_ids=cpu_copy(kwargs['input_ids']),
            input_attention_mask=cpu_copy(kwargs['attention_mask']),context=copy.deepcopy(self.context))
        cache=kwargs.get('past_key_values');self.old_length=0 if cache is None else cache.get_seq_length()
        self.previous=None if not self.old_length else snapshot_kv(cache)
        self.raw['visual']=kwargs.get('pixel_values') is not None
        language=self.model.model.language_model;norm=language.norm
        self.per_call=[language.register_forward_pre_hook(self.language,with_kwargs=True),
            self.layers[0].register_forward_pre_hook(self.block,with_kwargs=True),
            norm.register_forward_pre_hook(self.norm_before,prepend=True),norm.register_forward_pre_hook(self.norm_after)]
    def language(self,module,args,kwargs):
        self.counts['language']+=1;self.raw['position_ids']=cpu_copy(kwargs['position_ids']);self.raw['attention_mask']=cpu_copy(kwargs['attention_mask'])
    def block(self,module,args,kwargs):
        self.raw['causal_mask']=cpu_copy(kwargs.get('attention_mask'));self.raw['cache_position']=cpu_copy(kwargs['cache_position'])
    def norm_before(self,module,args):self.raw['native_norm_input']=cpu_copy(args[0])
    def norm_after(self,module,args):self.counts['norm']+=1;self.raw['fused_norm_input']=cpu_copy(args[0])
    def after(self,module,args,output):
        import torch
        for h in self.per_call:h.remove()
        self.per_call=[];torch.cuda.synchronize();raw=self.raw
        raw['native_logits']=cpu_copy(output.logits)
        raw['fusion']=None if self.controller is None else self.controller.export_last_capture(cpu=True)
        cache=output.past_key_values;mask=raw['attention_mask']
        raw['old_cache_length']=self.old_length;raw['cache_length']=None if cache is None else cache.get_seq_length()
        raw['existing_kv_prefix_exact']=None if self.previous is None else old.mixed.prefix_preserved(torch,cache,self.previous,self.old_length)
        need(cache is None or cache.get_seq_length()==mask.shape[1],'Actual native cache length differs')
        need(torch.equal(mask,raw['input_attention_mask']),'Native language mask changed')
        need(raw['native_norm_input'].dtype==raw['fused_norm_input'].dtype==raw['native_logits'].dtype==torch.float16
            and bool(torch.isfinite(raw['native_logits']).all()),'Native dtype/logits differ')
        raw['mask_audit']=old.mask_check(torch,raw['causal_mask'],mask,raw['cache_position'])
        label=self.context['label']+f'__call{len(self.records):03d}';path=self.data/(label+'.pt')
        torch.save(raw,path)
        self.records.append(dict(label=label,context=copy.deepcopy(self.context),path=str(path),sha256=sha(path),
            model=1,visual=int(raw['visual']),query_tokens=raw['input_ids'].shape[1],key_tokens=mask.shape[1],
            cache_length=raw['cache_length'],existing_kv_prefix_exact=raw['existing_kv_prefix_exact'],mask_audit=raw['mask_audit'],
            seconds=time.perf_counter()-self.started))
        self.last_raw=raw;self.previous=None


def verify_fusion(raw,controller,query,stream):
    import torch
    capture=raw['fusion'];h=raw['native_norm_input'];n=controller.n_actual_rows
    need(capture is not None and capture['query_indices']==query and capture['stream_positions']==stream
        and capture['n_actual_rows']==n and capture['n_reference_rows']==24 and capture['mode']==controller.mode,
        'Current/history reference query or row partition differs')
    need(torch.equal(capture['actual_states'],h[:n,query]) and torch.equal(capture['reference_states'],h[n:n+24,query])
        and torch.equal(capture['global_states'],h[-1,query]),'Reference read states do not match actual native queries')
    need(capture['delta'].dtype==torch.float32 and bool(torch.isfinite(capture['delta']).all()),'Reference residual is not finite FP32')
    expected=h.clone();expected[-1,query]=h[-1,query]+capture['delta'].half()
    need(torch.equal(expected,raw['fused_norm_input']) and torch.equal(expected[-1,query],capture['fused_global']),
        'Readout changed actual/reference rows or wrong global query positions')
    need(torch.equal(capture['reference_mean'],capture['reference_messages'].double().mean(0).float()),
        'Reference mean no longer matches frozen calibration arithmetic')
    return dict(passed=True,query_indices=query,stream_positions=stream,mode=controller.mode,
        actual_reference_untouched=True,only_declared_global_queries_changed=True,delta_nonzero=bool(capture['delta'].ne(0).any()),
        actual_n_frames=n,reference_rows=24,sham_seed=capture['sham_seed'])


def all_kv_equal(cache,baseline):
    import torch
    result=[torch.equal(layer.keys,k) and torch.equal(layer.values,v) for layer,(k,v) in zip(cache.layers,baseline)]
    need(len(result)==len(cache.layers)==len(baseline) and all(result),'Readout reference mode changed native K/V at fixed tokens')
    return result


def replay_heads(torch,model,records):
    result=[];logits=[]
    for record in records:
        raw=torch.load(record['path'],map_location='cpu',weights_only=True)
        with torch.no_grad():value=model.lm_head.forward(model.model.language_model.norm.forward(raw['fused_norm_input'][:,-1:,:].to(model.device)))
        metric=old.metric(torch,value[-1,-1].detach().cpu(),raw['native_logits'][-1,-1]);metric.update(label=record['label'],binding=True)
        result.append(metric);logits.append(dict(label=record['label'],replayed=cpu_copy(value),native=raw['native_logits']))
    return result,logits


def run(args):
    import torch,transformers
    from types import SimpleNamespace
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.parallel_local_reference import ParallelLocalReference
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'profile_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    index(out,'V12 native reference software profile',[('Summary','summary.json'),('Forwards','forwards.json'),('Structural checks','structural_checks.json')])
    data=DATA/f'profile_{job}';data.mkdir(parents=True,exist_ok=False)
    index(data,'V12 raw native/reference software evidence',[('Head replays','head_replays.pt')])
    plan=verify(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] and cpu['tests_passed'] and cpu['plan_sha256']==sha(args.plan),'Completed exact CPU software freeze required')
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==weights['schema_version']==1 and old.state_identity(weights['states'])==plan['initial_state_identity'],
        'Prepared input or fixed unfitted branch identity differs')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model;model.eval();model.requires_grad_(False);native.native_contract(model)
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and runtime_identity()==plan['runtime'],
        'Loaded processor/runtime differs')
    _,_,api=old.local.native_api(loaded.processor);need(api==plan['native_api'],'Loaded native source differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick;device=loaded.device
    branch=ParallelLocalAggregation().to(device=device).eval();native.native_contract(model,branch)
    versions={name:p._version for name,p in model.named_parameters()};torch.cuda.reset_peak_memory_stats()
    checks=[];comparisons=[];timings=[];zero_checks=[];controller_counts=[]
    with NativeAudit(model,data) as audit:
        try:
            for case in plan['cases']:
                case_id=case['case_id'];n=case['actual_n_frames'];width=case['prompt_width'];tokens=plan['forced_ids']
                bundles=[blob['bundles'][f'{case_id}_t{t}'] for t in range(3)]
                for t,bundle in enumerate(bundles):
                    need(reference.validate_bundle(bundle)['passed'] and bundle['metadata']['input_identity']==case['variants'][t]['input_identity']
                        and native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['variants'][t]['layout'],
                        'Frozen augmented native layout differs')
                baseline=[];baseline_kv=[];native_full={};base_outputs={};cache=None
                audit.controller=None
                for step,bundle in enumerate(bundles):
                    inputs,positions,layout=cached_input(model,bundle,cache,step,tokens,device)
                    audit.context=dict(label=f'{case_id}__native__cached{step}',case_id=case_id,actual_n_frames=n,phase='cached',step=step,condition='native')
                    with torch.inference_mode():output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                    cache=output.past_key_values;raw=audit.last_raw
                    need(torch.equal(raw['position_ids'],positions.cpu()),'Bare native positions differ')
                    need(torch.equal(raw['native_norm_input'],raw['fused_norm_input']),'Bare native readout changed')
                    baseline.append(raw);baseline_kv.append(snapshot_kv(cache))
                del cache,output
                for condition,mode in [('zero','background')]+[('active',mode) for mode in MODES]:
                    branch.load_state_dict(weights['states'][condition],strict=True)
                    need(old.state_identity({condition:branch.state_dict()})[condition]==plan['initial_state_identity'][condition], 'Fixed branch state changed')
                    cache=None
                    controller=ParallelLocalReference(model.model.language_model.norm,branch,n_actual_rows=n,n_reference_rows=24,
                        mode=mode,anchor_n=16,sham_key=['software',case['question']],query_indices=[width-1],stream_positions=[width-1],capture=True)
                    with controller:
                        audit.controller=controller
                        for step,bundle in enumerate(bundles):
                            query=[width-1] if step==0 else [0];stream=[width+step-1];controller.configure_queries(query,stream)
                            inputs,positions,layout=cached_input(model,bundle,cache,step,tokens,device)
                            audit.context=dict(label=f'{case_id}__{condition}__{mode}__cached{step}',case_id=case_id,actual_n_frames=n,
                                phase='cached',step=step,condition=condition,mode=mode)
                            with torch.inference_mode():output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                            cache=output.past_key_values;raw=audit.last_raw
                            need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],baseline[step]['native_norm_input']),
                                'Reference mode changed fixed-prefix native hidden states or positions')
                            verified=verify_fusion(raw,controller,query,stream)
                            verified.update(label=audit.context['label'],native_hidden_exact=True,all_layer_kv_exact=all_kv_equal(cache,baseline_kv[step]))
                            checks.append(verified)
                            if condition=='zero':
                                need(not verified['delta_nonzero'] and torch.equal(raw['native_logits'],baseline[step]['native_logits']),
                                    'Zero-U reference fusion differs from bare augmented native execution')
                                zero_checks.append(dict(label=audit.context['label'],passed=True,raw_logits_exact=True,delta_exact_zero=True))
                            else:
                                need(verified['delta_nonzero'],'Fixed active branch gave no residual')
                                if mode=='base':base_outputs[step]=raw['native_logits']
                                elif n==16:need(torch.equal(raw['native_logits'],base_outputs[step]),'N16 anchor changed matched active base logits')
                                if step>0:
                                    history=list(range(width-1,width+step));controller.configure_queries(history,history)
                                    audit.context=dict(label=f'{case_id}__{condition}__{mode}__full{step}',case_id=case_id,actual_n_frames=n,
                                        phase='full',step=step,condition=condition,mode=mode)
                                    rope=cpu_copy(model.model.rope_deltas)
                                    try:
                                        with torch.inference_mode():full_output=model(**move_to_device(bundle['inputs'],device),use_cache=False,logits_to_keep=1)
                                    finally:model.model.rope_deltas=rope.to(device)
                                    full=audit.last_raw;need(full_output.past_key_values is None and torch.equal(full['position_ids'],layout['position_ids']),
                                        'Full reference cache/positions differ')
                                    if mode=='base':native_full[step]=full['native_norm_input']
                                    else:need(torch.equal(full['native_norm_input'],native_full[step]),'Reference mode changed full-prefix native hidden states')
                                    verified=verify_fusion(full,controller,history,history)
                                    verified.update(label=audit.context['label'],native_hidden_exact=True,all_layer_kv_exact=None);checks.append(verified)
                                    for row in range(n+25):
                                        metric=old.metric(torch,raw['native_logits'][row,-1],full['native_logits'][row,-1])
                                        metric.update(case_id=case_id,actual_n_frames=n,mode=mode,step=step,row_index=row,
                                            role='actual' if row<n else 'reference' if row<n+24 else 'global',binding=False)
                                        comparisons.append(metric)
                                    del full,full_output
                        expected=3 if condition=='zero' else 5
                        need(controller.calls==controller.actual_reads==controller.reference_reads==expected,'Controller invocation/read coverage differs')
                        controller_counts.append(dict(case_id=case_id,condition=condition,mode=mode,calls=controller.calls,passed=True))
                    audit.controller=None;need(not controller.active,'Reference fusion hook survived context');del cache,output
                del baseline,baseline_kv,native_full,base_outputs,bundles,raw
            need(audit.counts==dict(model=42,visual=22,language=42,norm=42,last_block=42),'Fixed native software call inventory differs')
            branch.load_state_dict(weights['states']['active'],strict=True)
            for case in plan['cases']:
                tick=time.perf_counter();bundle=reference.prepare_scene(loaded.processor,case['sample'],case['reference_bank'])
                preprocessing=time.perf_counter()-tick;start_record=len(audit.records)
                audit.context=dict(label=case['case_id']+'__active__background__natural',case_id=case['case_id'],
                    actual_n_frames=case['actual_n_frames'],phase='natural',condition='active',mode='background')
                audit.controller=None
                generated=reference.generate_native(model,loaded.processor,branch,bundle,mode='background',core_key='software',capture=True)
                step_records=audit.records[start_record:];steps=len(generated['generated_ids'])
                need(len(step_records)==len(generated['captures'])==steps and 1<=steps<=4,'Natural generation/capture coverage differs')
                for t,(record,capture) in enumerate(zip(step_records,generated['captures'])):
                    raw=torch.load(record['path'],map_location='cpu',weights_only=True);raw['fusion']=capture
                    query=[raw['native_norm_input'].shape[1]-1];stream=[case['prompt_width']+t-1]
                    verified=verify_fusion(raw,SimpleNamespace(n_actual_rows=case['actual_n_frames'],mode='background'),query,stream)
                    need(torch.equal(generated['raw_logits'][t],raw['native_logits'][-1,-1]),'Natural returned raw logits changed')
                    verified.update(label=record['label'],native_hidden_exact=None,all_layer_kv_exact=None);checks.append(verified)
                path=data/(case['case_id']+'__natural.pt');torch.save(generated,path)
                timings.append(dict(sid=case['sid'],actual_n_frames=case['actual_n_frames'],generated_tokens=steps,
                    preprocessing_seconds=preprocessing,generation_seconds=generated['model_seconds'],
                    four_token_seconds_bound=preprocessing+generated['model_seconds']*4/steps,
                    raw_file=str(path),raw_sha256=sha(path),counters=generated['counters'],no_accuracy_scoring=True,native_eos_retained=True))
            total=42+sum(r['generated_tokens'] for r in timings)
            need(total<=50 and audit.counts==dict(model=total,visual=24,language=total,norm=total,last_block=total),'Complete software call cap differs')
            head_metrics,head_raw=replay_heads(torch,model,audit.records)
            need(len(head_metrics)==total,'One native-shaped captured head replay per model call required')
            torch.save(head_raw,data/'head_replays.pt')
        finally:
            save(out/'forwards.json',audit.records);save(out/'actual_calls.json',audit.counts)
            save(out/'structural_checks.json',dict(checks=checks,zero_checks=zero_checks,controller_counts=controller_counts))
            save(out/'cached_full_comparisons.json',comparisons)
    need(len(zero_checks)==6 and len(controller_counts)==8 and len(comparisons)==780
        and sum(r['all_layer_kv_exact'] is not None for r in checks)==24,'Structural comparison coverage differs')
    need(not any(p.requires_grad or p.grad is not None for p in model.parameters())
        and versions=={name:p._version for name,p in model.named_parameters()},'Frozen backbone changed')
    need(old.state_identity({'active':branch.state_dict()})['active']==plan['initial_state_identity']['active'],'Unfitted branch changed')
    need(sources()==frozen,'Frozen V12 sources changed during software profile')
    save(out/'head_replay_metrics.json',head_metrics)
    timing=dict(rows=timings,T16=timings[0]['four_token_seconds_bound'],T64=timings[1]['four_token_seconds_bound'],
        bound_rule='preprocessing_seconds + generation_seconds*4/generated_tokens',maximum_new_tokens=4)
    save(out/'timing.json',timing)
    failures=[r for r in head_metrics if not r['numerical_rule_passed']]
    descriptive=[r for r in comparisons if not r['numerical_rule_passed']]
    artifacts={p.name:dict(path=str(p),sha256=sha(p)) for p in sorted(out.glob('*.json'))}
    artifacts['head_replays.pt']=dict(path=str(data/'head_replays.pt'),sha256=sha(data/'head_replays.pt'))
    summary=dict(schema_version=1,protocol=plan['protocol'],completed=True,passed=not failures,computational_integrity_passed=True,
        zero_identity_passed=True,fixed_prefix_native_kv_unchanged=True,fixed_prefix_hidden_unchanged=True,native_head_replay_passed=not failures,
        numerical_failures=failures,cached_full_failures=descriptive,cached_full_differences_descriptive=True,
        original_mixed_failure=plan['original_mixed_failure'],original_mixed_numerical_gate_passed=False,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        initial_state_identity=plan['initial_state_identity'],reference_manifest=plan['reference_manifest'],
        calls=audit.counts,native_head_replay_calls=len(head_metrics),extra_last_block_forwards=0,maximum_calls=BUDGET,
        fixed_prefix_kv_checks=24,fixed_prefix_fusion_checks=36,zero_identity_checks=6,cached_full_row_comparisons=780,
        timing=timing,load_seconds=load_seconds,seconds=time.perf_counter()-begin,slurm_job_id=job,
        native_dtypes=dict(norm=str(model.model.language_model.norm.weight.dtype),head=str(model.lm_head.weight.dtype)),
        hardware=dict(gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        artifacts=artifacts,frozen_backbone_unchanged=True,branch_unfitted_unchanged=True,no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True)
    save(out/'summary.json',summary)
    print(json.dumps(dict(directory=str(out),passed=summary['passed'],calls=audit.counts,binding_failures=len(failures),
        descriptive_failures=len(descriptive),timing=timing)),flush=True)
    need(summary['passed'],'Same-captured native head replay failed; preserve all numerical evidence')


def verify_profile(path):
    path=Path(path).resolve();summary=read(path)
    need(path.is_relative_to(OUT) and path.name=='summary.json' and summary.get('schema_version')==1
        and summary.get('protocol')=='v12_native_reference_software_only' and all(summary.get(k) is True for k in
        ('completed','passed','computational_integrity_passed','zero_identity_passed','fixed_prefix_native_kv_unchanged',
         'fixed_prefix_hidden_unchanged','native_head_replay_passed','frozen_backbone_unchanged','branch_unfitted_unchanged','no_fit','no_efficacy_scoring')),
        'Require completed successful GPU reference software profile, not CPU check')
    plan=verify(summary['plan_file']);need(sha(summary['plan_file'])==summary['plan_sha256'] and summary['source_sha256']==plan['source_sha256'],
        'GPU summary differs from frozen CPU plan')
    for name,digest in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'GPU source copy changed')
    for field in ('model','runtime','processor','native_api','reference_manifest','initial_state_identity'):
        need(summary[field]==plan[field],'GPU native/reference identity differs: '+field)
    for item in summary['artifacts'].values():need(sha(item['path'])==item['sha256'],'GPU software evidence changed')
    timings=summary['timing']['rows'];need([r['actual_n_frames'] for r in timings]==[16,64],'Native augmented timing cases differ')
    for i,row in enumerate(timings):
        need(row['sid']==plan['cases'][i]['sid'] and 1<=row['generated_tokens']<=4
            and row['four_token_seconds_bound']==row['preprocessing_seconds']+row['generation_seconds']*4/row['generated_tokens']
            and row['four_token_seconds_bound']>0 and sha(row['raw_file'])==row['raw_sha256'],'Augmented timing bound or raw evidence differs')
    total=42+sum(r['generated_tokens'] for r in timings)
    need(total<=50 and summary['calls']==dict(model=total,visual=24,language=total,norm=total,last_block=total)
        and summary['native_head_replay_calls']==total and summary['extra_last_block_forwards']==0 and summary['maximum_calls']==BUDGET
        and summary['fixed_prefix_kv_checks']==24 and summary['fixed_prefix_fusion_checks']==36 and summary['zero_identity_checks']==6
        and summary['cached_full_row_comparisons']==780 and not summary['numerical_failures'],'Software coverage/call inventory differs')
    need(summary['timing']['T16']==timings[0]['four_token_seconds_bound'] and summary['timing']['T64']==timings[1]['four_token_seconds_bound'],
        'Published augmented timing bounds differ')
    forwards=read(path.parent/'forwards.json');metrics=read(path.parent/'head_replay_metrics.json')
    need(len(forwards)==len(metrics)==total and all(r['numerical_rule_passed'] and r['top1_equal'] and r['full_vocabulary_tv']<=.02 for r in metrics),
        'Native forward/head replay evidence incomplete')
    for row in forwards:need(sha(row['path'])==row['sha256'],'Native raw forward evidence changed')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU freeze only');check()
    else:
        need(args.plan is not None,'Exact completed CPU plan required');run(args)


if __name__=='__main__':main()
