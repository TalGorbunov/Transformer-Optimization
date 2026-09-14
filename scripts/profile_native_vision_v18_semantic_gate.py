"""HELD V18 native semantic-gate software profile; no fitting or efficacy release.

CPU preparation validates ancestry and saves the only tensors consumed on GPU.
GPU verification does not reread ancestor caches. Direct origin probes and
native-shaped replay heads have explicit counters because .forward bypasses
hooks. Cached/full numerical differences are retained descriptively.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
from scripts import native_vision_semantic_gate_runtime as runtime
from scripts import profile_native_vision_v7_runtime as old
from scripts import profile_native_vision_v12_reference as ancestor
from scripts import profile_native_vision_v15_null_comparison as previous
from scripts.profile_native_vision_v11_memory import cpu_copy,runtime_identity,snapshot_kv,cached_input
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,model_metadata
from scripts.stage_native_vision_v10_features import native_api
OUT=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v18_semantic_gate_software')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/v18_semantic_gate_software')
V17_REPORT=REPO/'outputs/native_aggregation_vlm/v17/local_readout/report_442798/summary.json'
V17_PLAN=REPO/'outputs/native_aggregation_vlm/v17/local_readout/check_442792/plan.json'
ANCESTOR=REPO/'outputs/native_aggregation_vlm/v7/runtime/profile_441845/summary.json'
PROTOCOL='v18_semantic_gate_software_only';SEED=20261118
FORCED=('Therefore',':');MODES=('all_open','native_gate')
BUDGET=dict(fixed_model=32,fixed_visual=16,natural_generations=4,maximum_natural_model=16,
    maximum_model=48,visual=20,fixed_origin_probes=6,natural_origin_probes=4,maximum_origin_probes=10,
    maximum_native_head_replays=48,maximum_standalone_head_calls=58,extra_last_block_forwards=0,
    fixed_origin_head_rows=246,cached_full_rows=328)
OWN=('scripts/profile_native_vision_v18_semantic_gate.py','slurm/native_vision_v18_semantic_gate_check.sbatch',
     'slurm/native_vision_v18_semantic_gate_profile.sbatch')


def sources():
    from scripts import audit_native_vision_v17_local_readout as v17
    names=set(OWN)|set(runtime.OWN)|set(previous.sources())|set(v17.sources())
    return {name:sha(REPO/name) for name in sorted(names)}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,digest in frozen.items():
        dest=out/'source'/name.replace('/','_');dest.write_bytes((REPO/name).read_bytes());need(sha(dest)==digest,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V18 semantic-gate software\n\n[Summary](summary.json) · [Plan](plan.json) · [Sources](source_hashes.json).\n')
    return frozen


def initial_states(torch):
    from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
    with torch.random.fork_rng():
        torch.manual_seed(SEED);core=ParallelLocalSemanticAggregation(hidden_size=3584,rank=96)
        zero={k:v.detach().clone() for k,v in core.state_dict().items()}
        with torch.no_grad():core.up.weight.normal_(0,.001)
        active={k:v.detach().clone() for k,v in core.state_dict().items()}
    need(sum(x.numel() for x in zero.values())==1041600 and not bool(zero['up.weight'].any())
         and all(torch.equal(zero[k],active[k]) for k in zero if k!='up.weight'),'Fixed semantic initialization differs')
    return dict(zero=zero,active=active)


def v17_binding(*,ancestors):
    from scripts import audit_native_vision_v17_local_readout as v17
    summary=read(V17_REPORT);analysis=read(summary['analysis_file'])
    need(summary['completed'] is True and summary['passed'] is True and sha(summary['analysis_file'])==summary['analysis_sha256']
        and analysis['completed'] is True and analysis['passed'] is True and not analysis['binding_failures']
        and Path(analysis['plan_file'])==V17_PLAN and sha(V17_PLAN)==analysis['plan_sha256'],'Canonical V17 native readout proof differs')
    plan=v17.verify_plan(V17_PLAN,inputs=False,ancestors=ancestors)
    gpu=read(analysis['gpu_summary_file'])
    need(sha(analysis['gpu_summary_file'])==analysis['gpu_summary_sha256'] and gpu['completed'] is True and gpu['passed'] is True
        and gpu['initial_weight_identity']==gpu['final_weight_identity'],'Native readout weights/proof changed')
    bindings={str(V17_REPORT):sha(V17_REPORT),str(V17_PLAN):sha(V17_PLAN),summary['analysis_file']:sha(summary['analysis_file']),
              analysis['gpu_summary_file']:sha(analysis['gpu_summary_file'])}
    return plan,gpu['initial_weight_identity'],bindings


def check():
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    tests=[]
    for filename,count in [('tests/test_parallel_local_semantic_aggregation.py',11),('tests/test_parallel_local_semantic_gate.py',9)]:
        command=[sys.executable,filename];result=subprocess.run(command,capture_output=True,text=True,cwd=REPO)
        stem=Path(filename).stem
        (out/(stem+'.stdout.txt')).write_text(result.stdout);(out/(stem+'.stderr.txt')).write_text(result.stderr)
        need(result.returncode==0 and re.search(r'Ran '+str(count)+r' tests',result.stderr),'Semantic source tests failed; preserve logs')
        tests.append(dict(file=filename,tests=count,passed=True,command=command,
            stdout_sha256=sha(out/(stem+'.stdout.txt')),stderr_sha256=sha(out/(stem+'.stderr.txt'))))
    vp,weights,bindings=v17_binding(ancestors=True)
    prior=read(ANCESTOR);parent=read(prior['plan_file'])
    need(prior['computational_integrity_passed'] and prior['zero_identity_passed']
        and all(parent[k]==vp[k] for k in ('model','runtime','processor')),'Original native software identity differs')
    bindings.update({str(ANCESTOR):sha(ANCESTOR),prior['plan_file']:sha(prior['plan_file'])})
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==parent['processor'] and runtime_identity()==parent['runtime'],'Processor/runtime differs')
    owner,fn,api=native_api(processor)
    cache=read('/mnt/data/gabriele/gnn_transformer/v10_parallel_local/feature_cache.json')
    _,_,base_api=old.local.native_api(processor)
    need(api==cache['native_api'] and base_api==parent['native_api'],'Original/extended native API differs')
    def rope(**kwargs):return fn(owner,**kwargs)
    token_ids={}
    for value in ('0','1'):
        ids=processor.tokenizer(value,add_special_tokens=False)['input_ids']
        need(ids==[vp['count_token_ids'][value]] and ids[0] not in processor.tokenizer.all_special_ids,'Native ASCII0/1 IDs differ')
        token_ids[value]=ids[0]
    forced=[]
    for word in FORCED:
        ids=processor.tokenizer(word,add_special_tokens=False)['input_ids'];need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Fixed prefix differs');forced+=ids
    from gnnformer.parallel_local_semantic_gate import GATE_RULE
    identity_payload=dict(schema_version=1,model=vp['model'],runtime=vp['runtime'],processor=vp['processor'],native_api=api,
        native_dtypes=vp['native_dtypes'],norm_weight=weights['norm'],head_weight=weights['head'],
        norm_source_sha256=vp['norm_source_sha256'],rms_norm_eps=vp['rms_norm_eps'],count_token_ids=token_ids,gate_rule=GATE_RULE)
    identity_sha=native.object_sha(identity_payload)
    scenes,manifest=old.mixed.selected_sources();need([s['n_frames'] for s in scenes]==[16,64],'Fixed old cases differ')
    bundles={};cases=[]
    for sample in scenes:
        base=runtime.prepare_scene(processor,sample,verify_processor_parity=True);cid=f'parallel_N{sample["n_frames"]}';variants=[]
        origin=runtime.origin_identity(base,identity_sha)
        for step in range(3):
            bundle=runtime.append_observed_prefix(base,forced[:step]);runtime.validate_bundle(bundle);layout=native.audit_layout(rope,bundle)
            need(runtime.origin_identity(bundle,identity_sha)==origin,'Observed prefix changed original input binding')
            bundles[f'{cid}_t{step}']=bundle;variants.append(dict(input_identity=bundle['metadata']['input_identity'],layout=layout['metadata']))
        cases.append(dict(case_id=cid,sid=sample['sid'],n_frames=sample['n_frames'],question=sample['question'],sample=sample,
                          prompt_width=base['metadata']['original_prompt_width'],metadata=base['metadata'],origin_identity=origin,variants=variants))
    states=initial_states(torch);data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False)
    weights_dir=CKPT/f'check_{job}';weights_dir.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema_version=1,bundles=bundles),data/'prepared.pt');torch.save(dict(schema_version=1,states=states),weights_dir/'initial.pt')
    plan=dict(schema_version=1,protocol=PROTOCOL,source_sha256=frozen,ancestor_bindings=bindings,
        original_mixed_failure=parent['original_mixed_failure'],model=model_metadata(),runtime=runtime_identity(),processor=parent['processor'],native_api=api,
        native_identity_payload=identity_payload,native_identity_sha256=identity_sha,token_ids=token_ids,
        initial_file=str(weights_dir/'initial.pt'),initial_sha256=sha(weights_dir/'initial.pt'),initial_state_identity=old.state_identity(states),
        initial_seed=SEED,active_initialization='privateCPU seed20261118; same semantic core; active U Normal(0,.001); no fitting',
        prepared_file=str(data/'prepared.pt'),prepared_sha256=sha(data/'prepared.pt'),source_manifest=manifest,cases=cases,
        forced_text=list(FORCED),forced_ids=forced,modes=list(MODES),maximum_calls=BUDGET,parameters=dict(core=1041600,total=1041600),
        gate_rule='max(0,full_vocabulary_p1-full_vocabulary_p0); FP64 probabilities then detached FP32',
        original_gate_reused_for_all_prefixes=True,reference_rows=0,no_predictor=True,
        numerical_policy='all fixed origin probe rows and captured native-shaped head TV<=.02/exacttop1 binding; cached/full descriptive',
        no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True,controller_tests=tests,
        ancestry_tensor_validation='CPU only; GPU hashes consumed prepared/initial tensors and actual loaded norm/head once',slurm_job_id=job)
    need(sources()==frozen,'Sources changed during CPU preparation');save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    verify(out/'plan.json',ancestors=False)
    save(out/'summary.json',dict(passed=True,completed=True,tests_passed=True,controller_tests=tests,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),no_model_loaded=True,seconds=time.perf_counter()-started,slurm_job_id=job))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'))),flush=True)


def verify(path,*,ancestors=True):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(OUT) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'Plan path/sidecar differs')
    need(plan['schema_version']==1 and plan['protocol']==PROTOCOL and plan['source_sha256']==sources()
        and plan['model']==model_metadata() and plan['runtime']==runtime_identity() and plan['modes']==list(MODES)
        and plan['maximum_calls']==BUDGET and plan['initial_seed']==SEED and plan['parameters']==dict(core=1041600,total=1041600)
        and plan['reference_rows']==0 and plan['no_predictor'] is True,'Native/source/software policy identity differs')
    for name,digest in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'CPU source snapshot differs')
    for filename,digest in plan['ancestor_bindings'].items():need(sha(filename)==digest,'Bound small ancestor proof changed')
    if ancestors:v17_binding(ancestors=True)
    for name in ('prepared','initial'):need(sha(plan[name+'_file'])==plan[name+'_sha256'],'Consumed software tensor artifact changed')
    need(sha(plan['source_manifest']['path'])==plan['source_manifest']['sha256']
         and [c['n_frames'] for c in plan['cases']]==[16,64] and plan['forced_text']==list(FORCED) and len(plan['forced_ids'])==2
         and plan['native_identity_sha256']==native.object_sha(plan['native_identity_payload']),'Input/prefix/origin identity differs')
    for row in plan['controller_tests']:
        stem=Path(row['file']).stem
        for stream in ('stdout','stderr'):need(sha(path.parent/(stem+'.'+stream+'.txt'))==row[stream+'_sha256'],'CPU test log differs')
    need([(r['tests'],r['passed']) for r in plan['controller_tests']]==[(11,True),(9,True)],'Require both complete semantic test suites')
    return plan


class NativeAudit(ancestor.NativeAudit):
    def before(self,module,args,kwargs):
        super().before(module,args,kwargs)
        def norm_output(_module,_args,output):self.raw['native_normalized_query']=cpu_copy(output[:,-1:,:])
        self.per_call.append(self.model.model.language_model.norm.register_forward_hook(norm_output))


def controller_for(model,core,plan,case,mode,query,stream,artifact=None,*,capture=True):
    from gnnformer.parallel_local_semantic_gate import ParallelLocalSemanticGate
    return ParallelLocalSemanticGate(native.native_contract(model,core),model.lm_head,core,n_local_rows=case['n_frames'],mode=mode,
        zero_token_id=plan['token_ids']['0'],one_token_id=plan['token_ids']['1'],origin_identity=case['origin_identity'],
        query_indices=query,stream_positions=stream,origin_artifact=artifact,capture=capture)


def verify_fusion(raw,controller,query,stream):
    import torch
    c=raw['fusion'];h=raw['native_norm_input'];n=controller.n_local_rows
    need(c is not None and c['query_indices']==query and c['stream_positions']==stream and c['n_local_rows']==n
        and c['mode']==controller.mode and c['origin_identity']==controller.origin_identity,'Semantic query/origin ownership differs')
    need(torch.equal(c['local_states'],h[:n,query]) and torch.equal(c['global_states'],h[-1,query]),'Common native query states differ')
    core=controller.core;device=core.up.weight.device
    gates=c['native_gates'].to(device);applied=gates if controller.mode=='native_gate' else torch.ones_like(gates)
    expanded=applied[:,None].expand(n,len(query))
    with torch.inference_mode():delta,expected=core(c['local_states'].to(device),c['global_states'].to(device),gates=expanded,output_dtype=torch.float32,capture=True)
    need(all(torch.equal(c[k],v.detach().cpu()) for k,v in expected.items()) and torch.equal(c['applied_gates'],expanded.cpu()),
         'Same-device semantic payload/gate/readout replay differs')
    fused=h.clone();fused[-1,query]=h[-1,query]+delta.detach().cpu().half()
    need(torch.equal(fused,raw['fused_norm_input']) and torch.equal(fused[-1,query],c['fused_global']),'Native query-only residual cast/write differs')
    return dict(passed=True,query_indices=query,stream_positions=stream,mode=controller.mode,n_frames=n,
        delta_nonzero=bool(delta.ne(0).any()),local_rows_untouched=True,small_core_replay_exact=True,
        origin_source=c['origin_source'],probe_norm_calls=c['probe_norm_calls'],probe_head_calls=c['probe_head_calls'],probability_calls=c['probability_calls'])


def check_origin(torch,artifact,baseline,*,case_id,condition,mode,path):
    """No additional head calls: compare the six saved probes with bare prefill."""
    h=baseline['native_norm_input'][:,-1:,:];logits=baseline['native_logits'];n=h.shape[0]-1
    need(artifact['origin_hidden'].dtype==artifact['origin_normalized'].dtype==artifact['origin_logits'].dtype==torch.float16
         and torch.equal(artifact['origin_hidden'],h)
         and torch.equal(artifact['origin_normalized'],baseline['native_normalized_query']),
         'Origin probe must preserve exact bare native hidden and normalized FP16 rows')
    rows=[];digest=sha(path)
    for index in range(n+1):
        metric=old.metric(torch,artifact['origin_logits'][index,0],logits[index,-1])
        metric.update(case_id=case_id,condition=condition,mode=mode,row_index=index,role='actual' if index<n else 'global',
            binding=True,origin_file=str(path),origin_sha256=digest,hidden_exact=True,normalized_exact=True)
        rows.append(metric)
    return rows


MinimalTimingAudit=previous.MinimalTimingAudit
replay_heads=previous.replay_heads


def run(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'profile_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    data=DATA/f'profile_{job}';data.mkdir(parents=True,exist_ok=False)
    # Consume only this CPU-prepared payload; no ancestor cache/model-head replay.
    plan=verify(args.plan,ancestors=False);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] is True and cpu['completed'] is True and cpu['tests_passed'] is True
         and cpu['plan_sha256']==sha(args.plan) and cpu['source_sha256']==frozen,'Completed exact CPU source gate required')
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==weights['schema_version']==1
         and old.state_identity(weights['states'])==plan['initial_state_identity']
         and old.state_identity(initial_states(torch))==plan['initial_state_identity'],'Fixed semantic initialization/payload differs')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model)
    need(torch.cuda.get_device_name()=='NVIDIA B200','Registered software hardware requires NVIDIA B200')
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and runtime_identity()==plan['runtime'],'Loaded processor/runtime differs')
    _,_,api=native_api(loaded.processor);need(api==plan['native_api'],'Actual installed native API differs')
    native_weights=dict(norm=native.tensor_info(norm.weight),head=native.tensor_info(model.lm_head.weight))
    need(native_weights==dict(norm=plan['native_identity_payload']['norm_weight'],head=plan['native_identity_payload']['head_weight']),
         'Actual loaded native FP16 norm/head differs from V17 identity')
    core=ParallelLocalSemanticAggregation(hidden_size=3584,rank=96).eval().requires_grad_(False).to(model.device)
    native.native_contract(model,core);torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    versions={k:p._version for k,p in model.named_parameters()};torch.cuda.reset_peak_memory_stats()
    checks=[];zero=[];comparisons=[];controller_counts=[];timings=[];origins=[];probe_metrics=[];natural_capture_checks=[]
    fixed_records=[];natural_records=[];fixed_counts={};natural_counts={};metrics=[]
    try:
        with NativeAudit(model,data) as audit:
            try:
                for case in plan['cases']:
                    cid=case['case_id'];n=case['n_frames'];width=case['prompt_width'];tokens=plan['forced_ids']
                    bundles=[blob['bundles'][f'{cid}_t{t}'] for t in range(3)];layouts=[]
                    for t,bundle in enumerate(bundles):
                        layout=native.audit_layout(get_rope_index_fn(model),bundle);layouts.append(layout)
                        need(runtime.validate_bundle(bundle)['passed'] and bundle['metadata']['input_identity']==case['variants'][t]['input_identity']
                            and layout['metadata']==case['variants'][t]['layout']
                            and runtime.origin_identity(bundle,plan['native_identity_sha256'])==case['origin_identity'],'Frozen original/native prefix layout differs')
                    baseline=[];baseline_kv=[];native_full={};cache=None;audit.controller=None;origin_reference=None
                    for step,bundle in enumerate(bundles):
                        inputs,positions,_=cached_input(model,bundle,cache,step,tokens,model.device)
                        audit.context=dict(label=f'{cid}__native__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,condition='native')
                        with torch.inference_mode():
                            output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                        cache=output.past_key_values;raw=audit.last_raw
                        need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],raw['fused_norm_input']),
                             'Bare native positions/readout differ')
                        baseline.append(raw);baseline_kv.append(snapshot_kv(cache))
                    del cache,output
                    for condition,mode in [('zero','native_gate'),('active','all_open'),('active','native_gate')]:
                        core.load_state_dict(weights['states'][condition],strict=True)
                        need(old.state_identity({condition:core.state_dict()})[condition]==plan['initial_state_identity'][condition],'Fixed core state changed')
                        cache=None;cached=[];controller=controller_for(model,core,plan,case,mode,[width-1],[width-1])
                        with controller:
                            audit.controller=controller
                            for step,bundle in enumerate(bundles):
                                query=[width-1] if step==0 else [0];stream=[width+step-1];controller.configure_queries(query,stream,origin_identity=case['origin_identity'])
                                inputs,positions,_=cached_input(model,bundle,cache,step,tokens,model.device)
                                audit.context=dict(label=f'{cid}__{condition}__{mode}__cached{step}',case_id=cid,n_frames=n,phase='cached',step=step,condition=condition,mode=mode)
                                with torch.inference_mode():
                                    output=model(**inputs,use_cache=True,logits_to_keep=1) if 'use_cache' not in inputs else model(**inputs,logits_to_keep=1)
                                cache=output.past_key_values;raw=audit.last_raw;cached.append(raw)
                                if step==0:
                                    artifact=controller.export_origin_artifact(cpu=True);path=data/f'{cid}__{condition}__{mode}__origin.pt';torch.save(artifact,path)
                                    origins.append(dict(case_id=cid,n_frames=n,condition=condition,mode=mode,path=str(path),sha256=sha(path),artifact_sha256=artifact['artifact_sha256']))
                                    probe_metrics+=check_origin(torch,artifact,baseline[0],case_id=cid,condition=condition,mode=mode,path=path)
                                    keys=('origin_hidden','origin_normalized','origin_logits','p0','p1','native_gates')
                                    if origin_reference is None:origin_reference=artifact
                                    else:need(all(torch.equal(artifact[k],origin_reference[k]) for k in keys),'Matched modes changed raw original probe/probability/gates')
                                need(torch.equal(raw['position_ids'],positions.cpu()) and torch.equal(raw['native_norm_input'],baseline[step]['native_norm_input']),
                                     'Semantic mode changed fixed-prefix native hidden/positions')
                                checked=verify_fusion(raw,controller,query,stream)
                                checked.update(label=audit.context['label'],native_hidden_exact=True,all_layer_kv_exact=ancestor.all_kv_equal(cache,baseline_kv[step]));checks.append(checked)
                                need(checked['probe_norm_calls']==checked['probe_head_calls']==checked['probability_calls']==1,'Cached prefixes repeated an origin probe')
                                need(torch.equal(raw['fusion']['native_gates'],artifact['native_gates']),'Cached gate differs from original artifact')
                                if condition=='zero':
                                    need(not checked['delta_nonzero'] and torch.equal(raw['native_logits'],baseline[step]['native_logits']),'Zero U changed native output')
                                    zero.append(dict(label=audit.context['label'],passed=True,raw_logits_exact=True))
                                else:need(checked['delta_nonzero'],'Active software core produced no residual')
                        audit.controller=None;need(not controller.active,'Cached controller survived cleanup')
                        status=controller.assert_complete();need(status['calls']==3,'Cached controller inventory differs')
                        controller_counts.append(dict(case_id=cid,condition=condition,mode=mode,phase='cached',**status));del cache,output
                        if condition=='active':
                            for step in (1,2):
                                history=list(range(width-1,width+step));bundle=bundles[step]
                                full_controller=controller_for(model,core,plan,case,mode,history,history,artifact)
                                audit.context=dict(label=f'{cid}__{condition}__{mode}__full{step}',case_id=cid,n_frames=n,phase='full',step=step,condition=condition,mode=mode)
                                rope=cpu_copy(model.model.rope_deltas)
                                try:
                                    with full_controller:
                                        audit.controller=full_controller
                                        with torch.inference_mode():full_output=model(**move_to_device(bundle['inputs'],model.device),use_cache=False,logits_to_keep=1)
                                finally:model.model.rope_deltas=rope.to(model.device);audit.controller=None
                                full=audit.last_raw
                                need(full_output.past_key_values is None and torch.equal(full['position_ids'],layouts[step]['position_ids']),
                                     'Full-prefix native positions/cache differ')
                                if mode=='all_open':native_full[step]=full['native_norm_input']
                                else:need(torch.equal(native_full[step],full['native_norm_input']),'Gate mode changed full-prefix native hidden')
                                checked=verify_fusion(full,full_controller,history,history)
                                checked.update(label=audit.context['label'],native_hidden_exact=True,all_layer_kv_exact=None);checks.append(checked)
                                need(checked['origin_source']=='bound_artifact'
                                    and checked['probe_norm_calls']==checked['probe_head_calls']==checked['probability_calls']==0
                                    and torch.equal(full['fusion']['native_gates'],artifact['native_gates']),'Full-prefix replay reclassified or changed origin gates')
                                status=full_controller.assert_complete();need(status['calls']==1 and not full_controller.active,'Full-prefix replay controller count/cleanup differs')
                                controller_counts.append(dict(case_id=cid,condition=condition,mode=mode,phase='full',step=step,**status))
                                for index in range(n+1):
                                    metric=old.metric(torch,cached[step]['native_logits'][index,-1],full['native_logits'][index,-1])
                                    metric.update(case_id=cid,n_frames=n,mode=mode,step=step,row_index=index,role='actual' if index<n else 'global',binding=False);comparisons.append(metric)
                                del full,full_output
                        del cached,artifact,controller
                    del baseline,baseline_kv,native_full,bundles,raw,origin_reference
                need(audit.counts==dict(model=32,visual=16,language=32,norm=32,last_block=32),'Fixed model/vision inventory differs')
            finally:
                fixed_records=audit.records;fixed_counts=dict(audit.counts)
                save(out/'fixed_forwards.json',fixed_records);save(out/'fixed_calls.json',fixed_counts)
                save(out/'fixed_structural_checks.json',dict(checks=checks,zero_checks=zero,controller_counts=controller_counts))
                save(out/'cached_full_comparisons.json',comparisons);save(out/'origin_artifacts.json',origins);save(out/'origin_probe_metrics.json',probe_metrics)
        need(len(zero)==6 and len(checks)==26 and len(controller_counts)==14 and len(comparisons)==328 and len(probe_metrics)==246 and len(origins)==6,
             'Fixed structural/origin coverage differs')
        need(sum(c['all_layer_kv_exact'] is not None for c in checks)==18
             and all(c['all_layer_kv_exact'] is None or (type(c['all_layer_kv_exact']) is list and len(c['all_layer_kv_exact'])==28
             and all(v is True for v in c['all_layer_kv_exact'])) for c in checks),'Require all28-layer native KV equalities for18cached fusions')
        core.load_state_dict(weights['states']['active'],strict=True)
        with MinimalTimingAudit(model,data) as timing_audit:
            try:
                for case in plan['cases']:
                    for mode in MODES:
                        tick=time.perf_counter();bundle=runtime.prepare_scene(loaded.processor,case['sample']);prep=time.perf_counter()-tick
                        need(runtime.origin_identity(bundle,plan['native_identity_sha256'])==case['origin_identity'],'Natural original bundle differs from frozen input')
                        start_record=len(timing_audit.records)
                        timing_audit.context=dict(label=f'{case["case_id"]}__{mode}__natural',case_id=case['case_id'],n_frames=case['n_frames'],mode=mode)
                        tick=time.perf_counter()
                        generated=runtime.generate_native(model,loaded.processor,core,bundle,mode=mode,
                            native_identity_sha256=plan['native_identity_sha256'],capture=True)
                        torch.cuda.synchronize();seconds=time.perf_counter()-tick
                        records=timing_audit.records[start_record:];steps=len(generated['generated_ids']);artifact=generated['origin_artifact']
                        need(len(records)==len(generated['captures'])==steps and 1<=steps<=4,'Natural native query capture coverage differs')
                        need(generated['raw_logits'].dtype==torch.float32 and torch.equal(generated['raw_logits'],generated['raw_logits'].half().float()),
                             'Global generation vectors are not losslessly promoted native FP16')
                        need(all(generated['counters'][k]==1 for k in ('probe_norm','probe_head','probability')),'Natural origin must be probed exactly once')
                        for t,record in enumerate(records):
                            raw=torch.load(record['path'],map_location='cpu',weights_only=True)
                            need(torch.equal(generated['raw_logits'][t],raw['native_global_logits'])
                                and record['retained_hidden_shape']==[case['n_frames']+1,1,3584]
                                and record['kv_snapshots']==record['full_hidden_copies']==0,'Minimal native timing capture differs')
                            c=generated['captures'][t];n=case['n_frames'];width=bundle['metadata']['prompt_width']
                            query=[width-1] if t==0 else [0];stream=[width+t-1]
                            before=torch.cat((c['local_states'],c['global_states'].unsqueeze(0)),dim=0)
                            current=controller_for(model,core,plan,case,mode,[0],stream,artifact,capture=False)
                            compact=dict(c,query_indices=[0])
                            checked=verify_fusion(dict(fusion=compact,native_norm_input=before,fused_norm_input=raw['fused_query_hidden']),current,[0],stream)
                            need(c['query_indices']==query and c['stream_positions']==stream and torch.equal(c['native_gates'],artifact['native_gates'])
                                 and c['probe_norm_calls']==c['probe_head_calls']==c['probability_calls']==1,'Natural origin/prefix ownership differs')
                            checked.update(case_id=case['case_id'],step=t,physical_query_indices=query,label=record['label']);natural_capture_checks.append(checked)
                        path=data/f'{case["case_id"]}__{mode}__natural.pt';torch.save(generated,path)
                        timings.append(dict(sid=case['sid'],n_frames=case['n_frames'],mode=mode,generated_tokens=steps,preparation_seconds=prep,
                            generation_seconds=seconds,runtime_internal_seconds=generated['model_seconds'],four_token_seconds_bound=prep+seconds*4/steps,
                            raw_file=str(path),raw_sha256=sha(path),origin_artifact_sha256=artifact['artifact_sha256'],counters=generated['counters'],
                            query_captures_retained=True,no_efficacy_scoring=True,no_full_hidden_or_kv_copies=True))
            finally:
                natural_records=timing_audit.records;natural_counts=dict(timing_audit.counts)
                save(out/'natural_forwards.json',natural_records);save(out/'natural_calls.json',natural_counts)
                save(out/'natural_timings.json',timings);save(out/'natural_capture_checks.json',natural_capture_checks)
        steps=sum(t['generated_tokens'] for t in timings)
        need(len(timings)==4 and 4<=steps<=16 and natural_counts==dict(model=steps,visual=4,language=steps,norm=steps,last_block=steps),'Natural call inventory differs')
        metrics,raw_metrics=replay_heads(torch,model,fixed_records+natural_records)
        need(len(metrics)==32+steps<=48,'One standalone native-shaped head replay per model call required')
        torch.save(raw_metrics,data/'head_replays.pt');save(out/'head_replay_metrics.json',metrics)
    finally:
        save(out/'actual_calls.json',dict(fixed=fixed_counts,natural=natural_counts,
            total={k:fixed_counts.get(k,0)+natural_counts.get(k,0) for k in ('model','visual','language','norm','last_block')},
            fixed_origin_probes=len(origins),natural_origin_probes=len(timings),completed_native_head_replays=len(metrics)))
    need(versions=={k:p._version for k,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen backbone changed')
    need(old.state_identity({'active':core.state_dict()})['active']==plan['initial_state_identity']['active'] and sources()==frozen,'Software core/source changed')
    head_failures=[r for r in metrics if not r['numerical_rule_passed']];probe_failures=[r for r in probe_metrics if not r['numerical_rule_passed']]
    descriptive=[r for r in comparisons if not r['numerical_rule_passed']]
    timing=dict(rows=timings,by_mode={mode:{str(t['n_frames']):t['four_token_seconds_bound'] for t in timings if t['mode']==mode} for mode in MODES},
        bound_rule='preparation_seconds + measured_whole_generation_call_seconds*4/generated_tokens',maximum_new_tokens=4,
        observer='runtime query/origin captures plus full-batch last-query fused h/global logits; no full hidden/KV copies')
    save(out/'timing.json',timing)
    artifacts={p.name:dict(path=str(p),sha256=sha(p)) for p in sorted(out.glob('*.json'))}
    artifacts['head_replays.pt']=dict(path=str(data/'head_replays.pt'),sha256=sha(data/'head_replays.pt'))
    totals={k:fixed_counts[k]+natural_counts[k] for k in fixed_counts}
    summary=dict(schema_version=1,protocol=PROTOCOL,completed=True,passed=not head_failures and not probe_failures,
        computational_integrity_passed=True,zero_identity_passed=True,fixed_prefix_native_kv_unchanged=True,fixed_prefix_hidden_unchanged=True,
        native_head_replay_passed=not head_failures,origin_probe_native_equivalence_passed=not probe_failures,matched_origin_gate_values_passed=True,
        original_gate_reuse_passed=True,numerical_failures=head_failures,origin_probe_failures=probe_failures,cached_full_failures=descriptive,
        cached_full_differences_descriptive=True,original_mixed_failure=plan['original_mixed_failure'],original_mixed_numerical_gate_passed=False,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,model=plan['model'],runtime=plan['runtime'],
        processor=plan['processor'],native_api=plan['native_api'],native_identity_sha256=plan['native_identity_sha256'],native_weight_identity=native_weights,
        initial_state_identity=plan['initial_state_identity'],parameters=plan['parameters'],modes=list(MODES),reference_rows=0,no_predictor=True,
        calls=totals,fixed_calls=fixed_counts,natural_calls=natural_counts,native_head_replay_calls=len(metrics),maximum_calls=BUDGET,
        origin_probe_calls=dict(norm=10,head=10,probability=10),standalone_head_calls=10+len(metrics),extra_last_block_forwards=0,
        small_same_device_core_replays=26+steps,natural_query_capture_checks=steps,fixed_prefix_kv_checks=18,fixed_prefix_fusion_checks=26,
        zero_identity_checks=6,cached_full_row_comparisons=328,origin_probe_row_comparisons=246,
        timing=timing,load_seconds=load_seconds,seconds=time.perf_counter()-started,slurm_job_id=job,
        native_dtypes=dict(norm=str(norm.weight.dtype),head=str(model.lm_head.weight.dtype)),
        hardware=dict(gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),artifacts=artifacts,
        frozen_backbone_unchanged=True,core_unfitted_unchanged=True,no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True)
    save(out/'summary.json',summary);print(json.dumps(dict(directory=str(out),passed=summary['passed'],calls=totals,
        origin_probe_failures=len(probe_failures),head_failures=len(head_failures),descriptive_failures=len(descriptive))),flush=True)
    need(summary['passed'],'Binding native probe/head gate failed; all observations retained')


def verify_profile(path):
    """Bind completed raw evidence and decisions; never execute a native head."""
    path=Path(path).resolve();summary=read(path)
    required=('completed','passed','computational_integrity_passed','zero_identity_passed','fixed_prefix_native_kv_unchanged',
              'fixed_prefix_hidden_unchanged','native_head_replay_passed','origin_probe_native_equivalence_passed',
              'matched_origin_gate_values_passed','original_gate_reuse_passed','frozen_backbone_unchanged',
              'core_unfitted_unchanged','no_fit','no_efficacy_scoring')
    need(path.is_relative_to(OUT) and path.name=='summary.json' and summary.get('schema_version')==1
         and summary.get('protocol')==PROTOCOL and all(summary.get(k) is True for k in required),'Completed successful V18 software profile required')
    plan=verify(summary['plan_file'],ancestors=False)
    need(sha(summary['plan_file'])==summary['plan_sha256'] and summary['source_sha256']==plan['source_sha256'],'CPU/GPU source binding differs')
    for name,digest in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==digest,'GPU source snapshot differs')
    for field in ('model','runtime','processor','native_api','initial_state_identity','parameters','modes','native_identity_sha256'):
        need(summary[field]==plan[field],'Native/software identity differs: '+field)
    need(summary['native_weight_identity']==dict(norm=plan['native_identity_payload']['norm_weight'],head=plan['native_identity_payload']['head_weight'])
         and summary['reference_rows']==0 and summary['no_predictor'] is True and summary['cached_full_differences_descriptive'] is True
         and summary['original_mixed_numerical_gate_passed'] is False,'Native weights or scope differs')
    for item in summary['artifacts'].values():need(sha(item['path'])==item['sha256'],'Software artifact changed')
    need(read(path.parent/'timing.json')==summary['timing'],'Timing summary copy differs')
    rows=summary['timing']['rows']
    need([(r['n_frames'],r['mode']) for r in rows]==[(n,m) for n in (16,64) for m in MODES],'Natural mode/length timing coverage differs')
    for row in rows:
        case=plan['cases'][0 if row['n_frames']==16 else 1];steps=row['generated_tokens'];counts=row['counters']
        need(all(math.isfinite(row[k]) and row[k]>0 for k in ('preparation_seconds','generation_seconds','four_token_seconds_bound')),
             'Invalid natural timing')
        need(row['sid']==case['sid'] and 1<=steps<=4 and row['four_token_seconds_bound']==row['preparation_seconds']+row['generation_seconds']*4/steps
             and summary['timing']['by_mode'][row['mode']][str(row['n_frames'])]==row['four_token_seconds_bound']
             and row['no_full_hidden_or_kv_copies'] is True and row['query_captures_retained'] is True
             and sha(row['raw_file'])==row['raw_sha256'],'Natural timing/input artifact binding differs')
        need(counts==dict(model=steps,visual=1,language=steps,norm=steps,head=steps,fusion=steps,broadcast=steps,
                         probe_norm=1,probe_head=1,probability=1),'Natural model versus direct probe counter separation differs')
    steps=sum(r['generated_tokens'] for r in rows);total=32+steps
    need(total<=48 and summary['calls']==dict(model=total,visual=20,language=total,norm=total,last_block=total)
        and summary['fixed_calls']==dict(model=32,visual=16,language=32,norm=32,last_block=32)
        and summary['natural_calls']==dict(model=steps,visual=4,language=steps,norm=steps,last_block=steps)
        and summary['native_head_replay_calls']==total and summary['origin_probe_calls']==dict(norm=10,head=10,probability=10)
        and summary['standalone_head_calls']==10+total<=58 and summary['extra_last_block_forwards']==0 and summary['maximum_calls']==BUDGET
        and summary['small_same_device_core_replays']==26+steps and summary['natural_query_capture_checks']==steps
        and summary['fixed_prefix_kv_checks']==18 and summary['fixed_prefix_fusion_checks']==26 and summary['zero_identity_checks']==6
        and summary['cached_full_row_comparisons']==328 and summary['origin_probe_row_comparisons']==246
        and not summary['numerical_failures'] and not summary['origin_probe_failures'],'Native/probe/replay inventory differs')
    fixed=read(path.parent/'fixed_forwards.json');natural=read(path.parent/'natural_forwards.json');metrics=read(path.parent/'head_replay_metrics.json')
    probes=read(path.parent/'origin_probe_metrics.json');origins=read(path.parent/'origin_artifacts.json')
    need(len(fixed)==32 and len(natural)==steps and len(metrics)==total and len(probes)==246 and len(origins)==6,'Raw model/probe/head coverage differs')
    for row in fixed+natural+origins:need(sha(row['path'])==row['sha256'],'Raw native evidence changed')
    need(all(r['kv_snapshots']==r['full_hidden_copies']==0 and r['retained_hidden_shape'][1]==1 for r in natural),'Natural timing copied full hidden/KV')
    expected_origins=Counter((f'parallel_N{n}',kind,mode) for n in (16,64)
                            for kind,mode in [('zero','native_gate'),('active','all_open'),('active','native_gate')])
    need(Counter((r['case_id'],r['condition'],r['mode']) for r in origins)==expected_origins,'Origin mode coverage differs')
    origin_lookup={(r['case_id'],r['condition'],r['mode']):r for r in origins}
    need(Counter((r['case_id'],r['condition'],r['mode'],r['row_index']) for r in probes)==Counter(
        (f'parallel_N{n}',kind,mode,i) for n in (16,64) for kind,mode in [('zero','native_gate'),('active','all_open'),('active','native_gate')]
        for i in range(n+1)),'All246 original actual/global probe rows required')
    for row in probes:
        original=origin_lookup[(row['case_id'],row['condition'],row['mode'])];n=original['n_frames']
        need(row['origin_file']==original['path'] and row['origin_sha256']==original['sha256'] and row['hidden_exact'] is True
             and row['normalized_exact'] is True and row['binding'] is True
             and row['role']==('actual' if row['row_index']<n else 'global'),'Probe row identity or native normalization claim differs')
    captures=read(path.parent/'natural_capture_checks.json')
    need(len(captures)==steps and all(r['passed'] is True and r['small_core_replay_exact'] is True for r in captures),'Natural query capture/core checks incomplete')
    structure=read(path.parent/'fixed_structural_checks.json');checks=structure['checks'];zero=structure['zero_checks'];controls=structure['controller_counts']
    need(len(checks)==26 and len(zero)==6 and len(controls)==14 and all(r['passed'] is True for r in checks+zero+controls)
        and sum(r['all_layer_kv_exact'] is None for r in checks)==8
        and sum(type(r['all_layer_kv_exact']) is list for r in checks)==18
        and all(r['native_hidden_exact'] is True and r['small_core_replay_exact'] is True
                and (r['all_layer_kv_exact'] is None or (len(r['all_layer_kv_exact'])==28 and all(v is True for v in r['all_layer_kv_exact']))) for r in checks),
        'Fixed query/KV/zero/core evidence incomplete')
    need(Counter((r['case_id'],r['condition'],r['mode'],r['phase'],r['calls']) for r in controls)==Counter(
        [(f'parallel_N{n}',kind,mode,'cached',3) for n in (16,64) for kind,mode in [('zero','native_gate'),('active','all_open'),('active','native_gate')]]+
        [(f'parallel_N{n}','active',mode,'full',1) for n in (16,64) for mode in MODES for _ in (1,2)]),'Controller history/probe invocation coverage differs')
    for row in controls:
        expected=1 if row['phase']=='cached' else 0
        need(row['probe_norm_calls']==row['probe_head_calls']==row['probability_calls']==expected
             and row['origin_source']==('native_prefill' if expected else 'bound_artifact'),'Full-prefix controllers must reuse bound origin with zero probes')
    cached=read(path.parent/'cached_full_comparisons.json')
    need(len(cached)==328 and Counter((r['n_frames'],r['mode'],r['step'],r['row_index']) for r in cached)==Counter(
        (n,mode,t,i) for n in (16,64) for mode in MODES for t in (1,2) for i in range(n+1)),'Cached/full row coverage differs')
    for row in metrics+probes+cached:
        need(type(row['numerical_rule_passed']) is bool and type(row['top1_equal']) is bool
             and math.isfinite(row['full_vocabulary_tv']) and 0<=row['full_vocabulary_tv']<=1
             and row['numerical_rule_passed']==(row['top1_equal'] and row['full_vocabulary_tv']<=.02),'Native numerical decision differs')
    need(all(r['numerical_rule_passed'] and r['binding'] is True for r in metrics+probes)
         and all(r['binding'] is False and r['role']==('actual' if r['row_index']<r['n_frames'] else 'global') for r in cached)
         and summary['cached_full_failures']==[r for r in cached if not r['numerical_rule_passed']],'Binding or descriptive failure scope differs')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU preparation only');check()
    else:
        need(args.plan is not None,'Pass the exact completed CPU plan');run(args)


if __name__=='__main__':main()
