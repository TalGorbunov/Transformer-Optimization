"""Bounded Cosmos direct-local/reasoning-global semantic-gate software proof.

Six natural max8 trajectories plus every active full prefix. No fitted weights,
answer accuracy, long reasoning or adaptive-gating claim. All tensor work Slurm.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_vision_reasoning_semantic_stream as runtime
from scripts import probe_reasoning_semantic_gate as probe
from scripts import profile_native_vision_reasoning_stream as ancestor
from scripts import profile_native_vision_v7_runtime as old
from scripts import native_vision_reasoning_stream as stream
from scripts.probe_native_vision_mixed_cache_localization import mask_check
native=runtime.native
need,read,sha,save=probe.need,probe.read,probe.sha,probe.save
MODEL=probe.MODEL
OUT=REPO/'outputs/native_aggregation_vlm/reasoning_semantic_stream'
DATA=Path('/mnt/data/gabriele/gnn_transformer/reasoning_semantic_stream')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/reasoning_semantic_stream')
PROOF=REPO/'outputs/native_aggregation_vlm/reasoning_semantic_gate/report_442926/summary.json'
PROTOCOL='cosmos_mixed_prompt_semantic_stream_software'
JOB_NAME='reasoning_semantic_stream'
SEED=20261118
POLICY=dict(protocol=PROTOCOL,seed=SEED,max_new_tokens=8,conditions=['bare','zero','active'],lengths=[16,64],
    maximum_model_calls=64,maximum_visual_calls=22,origin_probes=4,origin_rows=164,
    maximum_captured_head_replays=64,maximum_extra_head_calls=68,per_gpu_seconds=180,campaign_gpu_seconds=540,
    maximum_campaign_gpus=1,maximum_user_gpus=4,native_dtype='torch.float16',branch_dtype='torch.float32',
    head_tv_max=.02,head_top1_exact=True,cache_full_descriptive=True,no_fit=True,no_accuracy=True,
    no_adaptive_gate=True,no_trained_weight_transfer=True)
OWN=('scripts/native_vision_reasoning_semantic_stream.py','scripts/profile_native_vision_reasoning_semantic_stream.py',
    'slurm/reasoning_semantic_stream_check.sbatch','slurm/reasoning_semantic_stream_run.sbatch','slurm/reasoning_semantic_stream_report.sbatch')


def sources():return {**probe.sources(),**{n:sha(REPO/n) for n in set(OWN)|set(runtime.semantic.OWN)}}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Cosmos mixed semantic streaming software\n\n[Summary](summary.json) · [Sources](source_hashes.json). No fit or answer scoring.\n')
    return frozen


def initial_states(torch):
    from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);core=ParallelLocalSemanticAggregation()
        zero={k:v.detach().clone() for k,v in core.state_dict().items()}
        with torch.no_grad():core.up.weight.normal_(0,.001)
        active={k:v.detach().clone() for k,v in core.state_dict().items()}
    need(sum(v.numel() for v in zero.values())==1041600 and not bool(zero['up.weight'].any())
         and all(torch.equal(zero[k],active[k]) for k in zero if k!='up.weight'),'Untrained software initialization differs')
    return dict(zero=zero,active=active)


def state_info(states):return {name:{k:native.tensor_info(v) for k,v in state.items()} for name,state in states.items()}


def inventory(observations):
    need(len(observations)==6 and [(r['n_frames'],r['condition']) for r in observations]==[(n,c) for n in (16,64) for c in ('bare','zero','active')],'Six fixed trajectories required')
    need(all(type(r['tokens']) is int and 1<=r['tokens']<=8 for r in observations),'Natural token budget differs')
    g=sum(r['tokens'] for r in observations);a=sum(r['tokens'] for r in observations if r['condition']=='active')
    return dict(generated_tokens=g,active_prefixes=a,model=g+a,visual=6+a,origin_probes=4,
        head_replays=g+a,extra_heads=g+a+4,kv_transitions=g-6,
        gated_forwards=sum(r['tokens'] for r in observations if r['condition']!='bare')+a)


def active_gpu_count(raw):
    total=0
    for line in raw.splitlines():
        if not line.strip():continue
        job,partition,state,gres=[x.strip() for x in line.split('|')]
        if partition=='gpu' and state in ('RUNNING','COMPLETING'):
            pieces=[p for p in gres.split(',') if p.startswith(('gpu:','gres/gpu:'))]
            need(pieces,'Cannot verify active user GPU count')
            total+=sum(int(p.rsplit(':',1)[1].split('(')[0]) for p in pieces)
    return total


def self_test(out):
    import torch
    unit=probe.self_test()
    need(active_gpu_count('1|gpu|RUNNING|gres/gpu:1\n2|gpu|COMPLETING|gpu:b200:2\n3|gpu|PENDING|gpu:4')==3,'Active user GPU parsing differs')
    rows=[dict(n_frames=n,condition=c,tokens=8) for n in (16,64) for c in ('bare','zero','active')]
    need(inventory(rows)==dict(generated_tokens=48,active_prefixes=16,model=64,visual=22,origin_probes=4,head_replays=64,extra_heads=68,kv_transitions=42,gated_forwards=48),'Maximum call algebra differs')
    rows[0]['tokens']=1;need(inventory(rows)['model']==57,'Early stopping inventory differs')
    recorder=stream.GlobalLogitRecorder(max_steps=8,full_vectors=True)
    from types import SimpleNamespace
    value=torch.zeros(2,1,20,dtype=torch.float16);value[-1,0,17]=1;before=value.clone()
    need(recorder(None,None,SimpleNamespace(logits=value)) is None and torch.equal(value,before)
         and recorder.records[0]['top1_token_id']==17 and recorder.vectors[0].dtype==torch.float32,'Recorder mutation/promotion differs')
    command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_parallel_local_semantic*.py']
    result=subprocess.run(command,capture_output=True,text=True);(out/'unit_tests.log').write_text(result.stdout+result.stderr)
    need(result.returncode==0,'Frozen semantic core/controller CPU tests failed')
    return dict(passed=True,unit=unit,maximum_inventory=True,early_stop_inventory=True,recorder_nonmutating=True,
                semantic_tests_file=str(out/'unit_tests.log'),semantic_tests_sha256=sha(out/'unit_tests.log'),model_calls=0,head_calls=0)


def check(out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4);tests=self_test(out);bindings={}
    probe.bind(PROOF,bindings);summary=read(PROOF);probe.bind(summary['analysis_file'],bindings,summary['analysis_sha256']);analysis=read(summary['analysis_file'])
    need(summary['passed'] and analysis['passed'] and analysis['direct_local_feasibility']['passed'],'Completed direct-local feasibility required')
    original=probe.verify_plan(analysis['plan_file']);probe.bind(analysis['plan_file'],bindings,analysis['plan_sha256'])
    for p,h in original['artifact_bindings'].items():probe.bind(p,bindings,h)
    runfile=Path(analysis['run_directory'])/'summary.json';probe.bind(runfile,bindings,analysis['run_summary_sha256']);prior=read(runfile)
    probe.bind(prior['native_identity_file'],bindings,prior['native_identity_file_sha256']);identity=read(prior['native_identity_file'])
    need(identity==analysis['native_identity'] and probe.oid(identity)==analysis['native_identity_sha256'],'Actual Cosmos identity differs')
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=ancestor.native_api(processor)
    need(api==original['native_api'] and ancestor.fingerprint(processor,str(transformers.__version__))==original['processor'],'CPU processor/API differs')
    packet=torch.load(original['prepared_file'],map_location='cpu',weights_only=True);probe.bind(original['prepared_file'],bindings,original['prepared_sha256'])
    cases=[];bundles={}
    for item in original['cases']:
        if item['mode']!='direct_local':continue
        key=item['case_id'];bundle=packet['bundles'][key];runtime.semantic.validate_bundle(bundle)
        need(bundle['metadata']==item['metadata'] and {k:native.tensor_info(v) for k,v in bundle['inputs'].items()}==item['metadata']['input_identity'],'Original prepared inputs differ')
        layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle);need(layout['metadata']==item['layout'],'CPU native positions differ')
        cases.append(dict(case_id=key,n_frames=item['n_frames'],metadata=item['metadata'],layout=item['layout'],
            origin_identity=runtime.semantic.origin_identity(bundle,probe.oid(identity))))
        bundles[key]=bundle
    need([c['n_frames'] for c in cases]==[16,64],'Old fixed case ownership differs')
    destination=DATA/out.name;destination.mkdir(parents=True,exist_ok=False)
    weight_directory=CKPT/out.name;weight_directory.mkdir(parents=True,exist_ok=False)
    prepared=destination/'prepared.pt';initial=weight_directory/'initial.pt';states=initial_states(torch)
    torch.save(dict(schema_version=1,bundles=bundles),prepared);torch.save(states,initial)
    plan=dict(schema_version=1,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,mirror=original['mirror'],
        runtime=original['runtime'],processor=original['processor'],native_api=original['native_api'],native_identity=identity,
        native_identity_sha256=probe.oid(identity),cases=cases,prepared_file=str(prepared),prepared_sha256=sha(prepared),
        initial_file=str(initial),initial_sha256=sha(initial),initial_state_identity=state_info(states),tests=tests,no_model_loaded=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,source_sha256=frozen,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,no_model_loaded=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path)
    need(sha(path)==path.with_suffix('.sha256').read_text().strip() and plan['policy']==POLICY
         and plan['source_sha256']==sources(),'Frozen plan/source/policy differs')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'CPU source snapshot differs')
    for file,h in plan['artifact_bindings'].items():need(sha(file)==h,'Bound ancestor input differs')
    for name in ('prepared','initial'):need(sha(plan[name+'_file'])==plan[name+'_sha256'],'Consumed software packet differs')
    need(probe.oid(plan['native_identity'])==plan['native_identity_sha256'],'Native identity digest differs')
    summary=read(path.parent/'summary.json');need(summary['passed'] and summary['plan_sha256']==sha(path),'Completed CPU check required')
    return plan


def accounting(out,exclude_job=None):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);file=out/'all_user_sacct.psv';file.write_text(result.stdout)
    own=[]
    for line in result.stdout.splitlines():
        if not line.strip():continue
        parts=line.split('|');need(len(parts)==9,'Unexpected scheduler field count')
        if parts[1]!=JOB_NAME:continue
        parts[1]='reasoning_gate_run';own.append('|'.join(parts))
    return dict(probe.parse_accounting('\n'.join(own),exclude_job),raw_file=str(file),raw_sha256=sha(file),command=command)


class Audit(old.NativeAudit):
    """Observe old native mask/KV machinery; keep global vectors plus origins.

    Same-shaped replay bypasses observation hooks. Extra head calls are counted
    separately; no vocabulary outputs are substituted into the native result.
    """
    def __init__(self,model,data):
        super().__init__(model,data);self.controller=None;self.head_replays=0;self.ordinary_heads=0
    def set_controller(self,value):self.controller=value
    def __enter__(self):
        super().__enter__()
        def normalized(module,args,output):self.capture['normalized']=output[:,-1:,:].detach().cpu().clone()
        def head(*_):self.ordinary_heads+=1
        self.handles.extend([self.model.model.language_model.norm.register_forward_hook(normalized),self.model.lm_head.register_forward_pre_hook(head)])
        return self
    def after(self,module,args,output):
        import torch
        value=self.capture;counts=self.counts;controller=self.controller
        need(all(counts[k]-value['before_counts'][k]==1 for k in ('language','norm','attention'))
             and counts['visual']-value['before_counts']['visual']==int(value['visual_expected']),'Native submodule inventory differs')
        masks=mask_check(torch,value['causal_mask'],value['attention_mask'],value['cache_position'])
        hidden=value['pre_final_rms_hidden'].unsqueeze(1);fused=hidden.clone();fusion=None;algebra=None
        if controller is not None:
            fusion=controller.export_last_capture(cpu=True);c=fusion;device=controller.core.up.weight.device
            need(torch.equal(c['local_states'],hidden[:-1]) and torch.equal(c['global_states'],hidden[-1]),'Current query states differ')
            with torch.inference_mode():delta,expected=controller.core(c['local_states'].to(device),c['global_states'].to(device),gates=c['applied_gates'].to(device),output_dtype=torch.float32,capture=True)
            need(torch.equal(delta.cpu(),c['delta']) and all(torch.equal(v.cpu(),c[k]) for k,v in expected.items()),'Same-device semantic payload replay differs')
            fused[-1]=hidden[-1]+delta.detach().cpu().half()
            need(torch.equal(fused[-1],c['fused_global']) and torch.equal(c['native_gates'],controller._artifact['native_gates'].cpu()),'Native cast-before-add/origin reuse differs')
            need(bool((c['messages'][c['applied_gates']==0]==0).all()),'Closed gate leaked payload coordinates')
            algebra=dict(passed=True,delta_nonzero=bool(delta.ne(0).any()),closed_coordinates_exact=True,local_rows_untouched=True,
                query_indices=c['query_indices'],stream_positions=c['stream_positions'],origin_source=c['origin_source'])
        with torch.inference_mode():
            normalized=self.model.model.language_model.norm.forward(fused.to(self.model.device))
            replay=self.model.lm_head.forward(normalized);self.head_replays+=1
        metric=old.metric(torch,output.logits[-1,-1].float().cpu(),replay[-1,-1].float().cpu())
        cache=output.past_key_values;retained=None;cache_meta=None
        if value['use_cache']:
            _,cache_meta=ancestor.mixed.cache_snapshot(cache,hidden.shape[0],value['attention_mask'].shape[1],len(self.layers))
            if self.previous is not None:retained=ancestor.mixed.prefix_preserved(torch,cache,self.previous,value['old_length'])
        else:need(cache is None and self.previous is None,'Uncached replay returned/used KV')
        self.previous=None
        bare_origin=self.context['condition']=='bare' and value['old_length']==0
        logits=output.logits[:,-1].detach().cpu().clone() if bare_origin else output.logits[-1:,-1].detach().cpu().clone()
        raw=dict(schema_version=1,context=dict(self.context),native_logits=logits,pre_final_rms_hidden=hidden[:,0],
            normalized=value['normalized'],replay_normalized=normalized.cpu(),replay_global_logits=replay[-1,-1].float().cpu(),
            fusion=fusion,**{k:value[k] for k in ('input_ids','attention_mask','position_ids','causal_mask','cache_position')})
        file=self.data/f'forward_{len(self.records):03d}.pt';torch.save(raw,file)
        record=dict(context=dict(self.context),forward_index=len(self.records),path=str(file),sha256=sha(file),
            batch_size=hidden.shape[0],query_tokens=value['input_ids'].shape[1],key_tokens=value['attention_mask'].shape[1],visual=int(value['visual_expected']),
            mask_audit=masks,cache_layers=cache_meta,previous_prefix_exact_by_layer=retained,expected_visual_identity=value['expected_visual_identity'],
            head_replay=metric,normalized_exact=torch.equal(value['normalized'],normalized.cpu()),fusion=algebra,
            seconds=time.perf_counter()-value['started'])
        self.records.append(record)
        save(self.data/f'forward_{len(self.records)-1:03d}.json',record)


def compare_origin(torch,artifact,bare,case):
    need(torch.equal(artifact['origin_hidden'],bare['pre_final_rms_hidden'].unsqueeze(1))
         and torch.equal(artifact['origin_normalized'],bare['normalized']),'Original gate hidden/normalization differs from bare prefill')
    rows=[dict(case_id=case['case_id'],row=i,**old.metric(torch,a,b))
          for i,(a,b) in enumerate(zip(artifact['origin_logits'][:,0],bare['native_logits']))]
    need(len(rows)==case['n_frames']+1,'Origin all-row coverage differs');return rows


def run(args,out,frozen):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_rope_index_fn
    from gnnformer.parallel_local_semantic_aggregation import ParallelLocalSemanticAggregation
    torch.set_num_threads(4);started=time.perf_counter();plan=verify_plan(args.plan)
    budget=accounting(out,os.environ['SLURM_JOB_ID']);need(budget['allocated_gpu_seconds']+180<=540,'New reservation exceeds cumulative cap')
    save(out/'resource_reservation.json',dict(budget,reserved_seconds=180))
    queue=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%P|%T|%b'],capture_output=True,text=True,check=True)
    (out/'live_user_queue.psv').write_text(queue.stdout);active_gpus=active_gpu_count(queue.stdout)
    need(active_gpus<=4,'More than four user GPUs active')
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One B200 required')
    need(ancestor.mirror_identity()==plan['mirror'] and ancestor.versions()==plan['runtime'],'Cosmos/runtime changed')
    packet=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True);weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(state_info(weights)==plan['initial_state_identity'],'Fixed untrained software states differ')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);norm=native.native_contract(model)
    need(ancestor.fingerprint(loaded.processor,str(transformers.__version__))==plan['processor']
         and ancestor.native_api(loaded.processor)[2]==plan['native_api'],'Live processor/API differs')
    expected=plan['native_identity']
    need(native.tensor_info(norm.weight)==expected['norm_weight'] and native.tensor_info(model.lm_head.weight)==expected['head_weight']
         and sha(inspect.getfile(type(norm)))==expected['norm_source_sha256'] and float(norm.variance_epsilon)==expected['rms_norm_eps'],
         'Actual original Cosmos norm/head weights/source differ')
    live_backend=dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),cuda=str(torch.version.cuda),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),
        quantization='nf4_double_bf16',attention='sdpa')
    need(live_backend==expected['backend'],'Native backend changed')
    core=ParallelLocalSemanticAggregation().to(model.device).eval().requires_grad_(False);torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    versions={n:p._version for n,p in model.named_parameters()};observations=[];comparisons=[];origins=[];zero_checks=[]
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);audit=Audit(model,data);failure=None
    try:
        with audit:
            for case in plan['cases']:
                key=case['case_id'];bundle=packet['bundles'][key]
                need(bundle['metadata']==case['metadata'] and native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['layout']
                     and runtime.semantic.origin_identity(bundle,plan['native_identity_sha256'])==case['origin_identity'],'Native input/layout/origin differs')
                baseline=None;bare_origin=None
                for condition in ('bare','zero','active'):
                    if condition!='bare':core.load_state_dict(weights[condition],strict=True)
                    audit.context=dict(case_id=key,n_frames=case['n_frames'],condition=condition,phase='generation');offset=len(audit.records)
                    modules=(model,model.model.visual,model.model.language_model,norm,model.lm_head)
                    hooks=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules];old_rope=getattr(model.model,'rope_deltas',None)
                    result=runtime.generate_native(model,loaded.processor,None if condition=='bare' else core,bundle,
                        native_identity_sha256=plan['native_identity_sha256'],controller_observer=audit.set_controller)
                    need(hooks==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules]
                         and getattr(model.model,'rope_deltas',None) is old_rope and audit.controller is None,'Generation hook/rope cleanup differs')
                    records=audit.records[offset:];old.audit_sequence(torch,model,bundle,records,result['generated_ids'])
                    file=data/f'{key}_{condition}.pt';torch.save(result,file)
                    observation=dict(case_id=key,n_frames=case['n_frames'],condition=condition,tokens=len(result['generated_ids']),
                        generated_ids=result['generated_ids'],completed=result['completed'],truncated=result['truncated'],
                        counters=result['counters'],file=str(file),sha256=sha(file),model_seconds=result['model_seconds'],forward_indices=[r['forward_index'] for r in records])
                    observations.append(observation);save(out/f'trajectory_{len(observations)-1:03d}.json',observation)
                    for step,record in enumerate(records):
                        raw=torch.load(record['path'],map_location='cpu',weights_only=True)
                        need(torch.equal(raw['native_logits'][-1].float(),result['raw_logits'][step]),'Streamed global logits differ from native output')
                    if condition=='bare':
                        baseline=result;bare_origin=torch.load(records[0]['path'],map_location='cpu',weights_only=True);continue
                    artifact=result['origin_artifact'];origin_rows=compare_origin(torch,artifact,bare_origin,case)
                    origins.append(dict(case_id=key,condition=condition,artifact_sha256=artifact['artifact_sha256'],trajectory_file=str(file),trajectory_sha256=sha(file),rows=origin_rows))
                    need(all(torch.equal(c['native_gates'],artifact['native_gates']) for c in result['captures']),'Original gates changed during generation')
                    if condition=='zero':
                        need(result['generated_ids']==baseline['generated_ids'] and torch.equal(result['raw_logits'],baseline['raw_logits']), 'Zero-U native identity failed')
                        zero_checks.append(dict(case_id=key,passed=True,tokens=result['generated_ids'],raw_logits_exact=True));continue
                    need(any(r['fusion']['delta_nonzero'] for r in records),'Active untrained core never produced a residual')
                    for step in range(len(result['generated_ids'])):
                        prefix=result['generated_ids'][:step];prefixed=runtime.semantic.append_observed_prefix(bundle,prefix)
                        audit.context=dict(case_id=key,n_frames=case['n_frames'],condition='active',phase='full_prefix',step=step,prefix_ids=prefix)
                        old_rope=getattr(model.model,'rope_deltas',None)
                        reference=runtime.forward_native(model,loaded.processor,core,prefixed,native_identity_sha256=plan['native_identity_sha256'],
                            origin_artifact=artifact,controller_observer=audit.set_controller)
                        full_record=audit.records[-1];full=torch.load(full_record['path'],map_location='cpu',weights_only=True)
                        layout=native.audit_layout(get_rope_index_fn(model),prefixed)
                        need(torch.equal(full['input_ids'],prefixed['inputs']['input_ids']) and torch.equal(full['attention_mask'],prefixed['inputs']['attention_mask'])
                             and torch.equal(full['position_ids'],layout['position_ids']) and reference['layout']==layout['metadata']
                             and reference['fusion_audit']['probe_head_calls']==0 and audit.controller is None
                             and getattr(model.model,'rope_deltas',None) is old_rope,'Actual full-prefix input/position/origin/cleanup differs')
                        comparisons.append(dict(case_id=key,step=step,prefix_ids=prefix,cached_index=records[step]['forward_index'],full_index=full_record['forward_index'],
                            **old.metric(torch,result['raw_logits'][step],reference['global_logits']),descriptive_only=True))
                print(json.dumps(dict(case=key,model_calls=audit.counts['model'])),flush=True)
        expected_counts=inventory(observations)
        need(audit.counts==dict(model=expected_counts['model'],visual=expected_counts['visual'],language=expected_counts['model'],norm=expected_counts['model'],attention=expected_counts['model'])
             and audit.ordinary_heads==audit.head_replays==expected_counts['model'] and len(comparisons)==expected_counts['active_prefixes']
             and len(origins)==4 and sum(len(x['rows']) for x in origins)==164 and len(zero_checks)==2,'Exact native/probe/replay inventory differs')
        need(versions=={n:p._version for n,p in model.named_parameters()} and not any(p.grad is not None or p.requires_grad for p in model.parameters())
             and state_info({'active':core.state_dict()})['active']==plan['initial_state_identity']['active'],'Model or software weights changed')
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        for name,value in [('forwards',audit.records),('observations',observations),('comparisons',comparisons),('origins',origins),('zero_checks',zero_checks)]:save(out/(name+'.json'),value)
        complete=len(observations)==6 and failure is None
        head_failures=[r for r in audit.records if not r['head_replay']['numerical_rule_passed']]
        origin_failures=[r for origin in origins for r in origin['rows'] if not r['numerical_rule_passed']]
        numeric=complete and not head_failures and not origin_failures
        save(out/'summary.json',dict(schema_version=1,protocol=PROTOCOL,completed=complete,passed=complete and numeric,
            computational_integrity_passed=complete,native_replay_passed=numeric,zero_identity_passed=len(zero_checks)==2,
            plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,native_identity_sha256=plan['native_identity_sha256'],
            counts=audit.counts,ordinary_head_calls=audit.ordinary_heads,captured_head_replays=audit.head_replays,origin_probes=len(origins),
            origin_rows=sum(len(x['rows']) for x in origins),head_failures=head_failures,origin_failures=origin_failures,
            cache_full_failures=[x for x in comparisons if not x['numerical_rule_passed']],
            files={name:dict(file=str(out/(name+'.json')),sha256=sha(out/(name+'.json'))) for name in ('forwards','observations','comparisons','origins','zero_checks')},
            resource_reservation=budget,model_load_seconds=load_seconds,seconds=time.perf_counter()-started,
            failure=failure,slurm_job_id=os.environ['SLURM_JOB_ID'],no_fit=True,no_accuracy=True,no_reasoning_composition_claim=True))
    need(numeric,'Native origin/head binding failed; all raw observations retained')
    return None


def metric_matches(actual,saved):
    return all(actual[k]==saved[k] if isinstance(actual[k],bool) else math.isclose(actual[k],saved[k],rel_tol=1e-9,abs_tol=1e-12) for k in actual)


def verify_profile(path):
    """Read source/JSON/hash bindings only; raw tensor rescoring is CPU --report."""
    path=Path(path);path=path/'summary.json' if path.is_dir() else path;summary=read(path);plan=verify_plan(summary['plan_file'])
    need(summary['passed'] and summary['completed'] and summary['computational_integrity_passed'] and summary['native_replay_passed']
         and summary['zero_identity_passed'] and summary['source_sha256']==plan['source_sha256']==sources()
         and summary['plan_sha256']==sha(summary['plan_file']) and summary['native_identity_sha256']==plan['native_identity_sha256'],'Incomplete software/source binding')
    for name,h in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'GPU source snapshot changed')
    artifacts={}
    for name,item in summary['files'].items():need(sha(item['file'])==item['sha256'],'Raw index changed');artifacts[name]=read(item['file'])
    inv=inventory(artifacts['observations'])
    need(summary['counts']==dict(model=inv['model'],visual=inv['visual'],language=inv['model'],norm=inv['model'],attention=inv['model'])
         and summary['captured_head_replays']==summary['ordinary_head_calls']==len(artifacts['forwards'])==inv['model']
         and summary['origin_probes']==len(artifacts['origins'])==4 and summary['origin_rows']==164
         and len(artifacts['comparisons'])==inv['active_prefixes'] and len(artifacts['zero_checks'])==2,'Software inventory changed')
    for r in artifacts['forwards']:need(sha(r['path'])==r['sha256'],'Native raw state changed')
    for r in artifacts['observations']:need(sha(r['file'])==r['sha256'],'Trajectory archive changed')
    return plan,summary,artifacts


def report(args,out,frozen):
    import torch
    from transformers import AutoProcessor
    from gnnformer.parallel_local_semantic_gate import artifact_digest
    torch.set_num_threads(4);plan,summary,artifacts=verify_profile(args.run_directory)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=ancestor.native_api(processor);need(api==plan['native_api'],'CPU native position API differs')
    packet=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    records=artifacts['forwards'];observations=artifacts['observations'];cache={};trajectories={};origins={};head_rows=[];kv=0;fusions=0
    for obs in observations:
        result=torch.load(obs['file'],map_location='cpu',weights_only=True);key=(obs['case_id'],obs['condition']);trajectories[key]=result
        need(result['generated_ids']==obs['generated_ids'] and len(result['generated_ids'])==obs['tokens']
             and result['raw_logits'].shape==(obs['tokens'],152064) and result['raw_logits'].dtype==torch.float32
             and torch.equal(result['raw_logits'],result['raw_logits'].half().float())
             and result['raw_logits'].argmax(-1).tolist()==result['generated_ids'],'Streamed raw FP16-promoted argmax/history differs')
        eos=(151645,151643);ids=result['generated_ids'];need(not any(x in eos for x in ids[:-1])
             and result['completed']==(ids[-1] in eos) and result['truncated']==(ids[-1] not in eos)
             and (result['completed'] or len(ids)==8),'Natural stopping differs')
        need(result['counters']==dict(model=len(ids),visual=1,language=len(ids),norm=len(ids),head=len(ids),broadcast=len(ids),
            fusion=0 if obs['condition']=='bare' else len(ids),probe_norm=0 if obs['condition']=='bare' else 1,
            probe_head=0 if obs['condition']=='bare' else 1,probability=0 if obs['condition']=='bare' else 1),'Trajectory invocation/probe inventory differs')
        need(len(result['captures'])==(0 if obs['condition']=='bare' else len(ids)),'Query capture coverage differs')
        if obs['condition']=='bare':need(result['origin_artifact'] is None,'Bare trajectory unexpectedly has a gate');continue
        artifact=result['origin_artifact'];origins[key]=artifact
        case=next(c for c in plan['cases'] if c['case_id']==obs['case_id'])
        need(artifact_digest(artifact)==artifact['artifact_sha256'] and artifact['origin_identity']==case['origin_identity']
             and artifact['mode']=='native_gate' and artifact['n_local_rows']==case['n_frames'],'Original gate artifact identity differs')
        probability=probe.probabilities(torch,artifact['origin_logits'][:,0]);n=case['n_frames']
        need(torch.allclose(probability['p0'][:n],artifact['p0'],rtol=1e-12,atol=1e-15)
             and torch.allclose(probability['p1'][:n],artifact['p1'],rtol=1e-12,atol=1e-15)
             and torch.equal(artifact['native_gates'],(artifact['p1']-artifact['p0']).clamp_min(0).float()),'Original full-vocabulary probability/gate differs')
        for capture in result['captures']:need(torch.equal(capture['native_gates'],artifact['native_gates']),'Natural gate reuse differs')
    for index,record in enumerate(records):
        need(record['forward_index']==index,'Native forward order differs');raw=torch.load(record['path'],map_location='cpu',weights_only=True);cache[index]=raw
        context=record['context'];case=next(c for c in plan['cases'] if c['case_id']==context['case_id']);bundle=packet['bundles'][context['case_id']]
        trajectory=trajectories[(context['case_id'],context['condition'])]
        if context['phase']=='generation':
            obs=next(o for o in observations if (o['case_id'],o['condition'])==(context['case_id'],context['condition']));step=obs['forward_indices'].index(index)
        else:need(context['phase']=='full_prefix' and context['condition']=='active','Unexpected native forward ownership');step=context['step']
        prefix=trajectory['generated_ids'][:step];full=runtime.semantic.append_observed_prefix(bundle,prefix)
        pos,delta=fn(owner,input_ids=full['inputs']['input_ids'],image_grid_thw=full['inputs']['image_grid_thw'],attention_mask=full['inputs']['attention_mask'])
        width=case['metadata']['prompt_width'];n=case['n_frames'];is_full=context['phase']=='full_prefix';prefill=is_full or step==0
        expected_ids=full['inputs']['input_ids'] if prefill else torch.full((n+1,1),prefix[-1],dtype=torch.long)
        need(raw['context']==context and torch.equal(raw['input_ids'],expected_ids)
             and torch.equal(raw['attention_mask'],full['inputs']['attention_mask']) and record['visual']==int(prefill),'Executed tokens/key mask/vision differ')
        actual=raw['position_ids']
        if is_full:need(torch.equal(actual,pos) and record['previous_prefix_exact_by_layer'] is None and record['cache_layers'] is None,'Full-prefix mRoPE/cache differs')
        else:
            text=raw['attention_mask'].long().cumsum(-1)-1
            need(actual.shape==(4,n+1,width if step==0 else 1)
                 and torch.equal(actual[1:],pos if step==0 else pos[:,:,-1:]),'Generated native mRoPE differs')
            need(torch.equal(actual[0][raw['attention_mask'].bool()],text[raw['attention_mask'].bool()]) if step==0
                 else torch.equal(actual[0],text[:,-1:]),'Generated text positions differ')
            if step:
                flags=record['previous_prefix_exact_by_layer'];need(type(flags) is list and len(flags)==28 and all(x is True for x in flags),'All-layer past KV was not preserved');kv+=1
        need(mask_check(torch,raw['causal_mask'],raw['attention_mask'],raw['cache_position'])==record['mask_audit'],'Actual causal mask audit differs')
        h=raw['pre_final_rms_hidden'];need(h.shape==(n+1,3584) and h.dtype==torch.float16 and bool(torch.isfinite(h).all()),'Native query state shape/dtype differs')
        native_global=raw['native_logits'][-1].float();metric=old.metric(torch,native_global,raw['replay_global_logits'])
        need(metric_matches(metric,record['head_replay']) and metric['numerical_rule_passed'],'Same-shaped head reconstruction differs')
        need(raw['normalized'].shape==raw['replay_normalized'].shape==(n+1,1,3584)
             and torch.equal(raw['normalized'],raw['replay_normalized'])==record['normalized_exact'],'Descriptive normalized-state equality differs')
        head_rows.append(metric)
        if not is_full:need(torch.equal(native_global,trajectory['raw_logits'][step]),'Recorder/native global vector differs')
        c=raw['fusion']
        if context['condition']=='bare':need(c is None and record['fusion'] is None,'Bare path changed');continue
        artifact=origins[(context['case_id'],context['condition'])];query=[width+step-1 if prefill else 0];stream_pos=[width+step-1]
        need(c['query_indices']==query and c['stream_positions']==stream_pos and c['origin_identity']==case['origin_identity']
             and torch.equal(c['local_states'],h[:-1,None]) and torch.equal(c['global_states'],h[-1:]),'Fusion query positions/current states differ')
        gates=artifact['native_gates'][:,None]
        need(torch.equal(c['native_gates'],artifact['native_gates']) and torch.equal(c['applied_gates'],gates)
             and torch.equal(c['messages'],gates.unsqueeze(-1)*c['payload']) and bool((c['messages'][gates==0]==0).all())
             and torch.equal(c['fused_global'],h[-1:]+c['delta'].half()),'Gate-coordinate product or FP16 write differs')
        if context['condition']=='zero':need(not bool(c['delta'].any()),'Zero-U generated a residual')
        expected=0 if is_full else 1
        need(c['probe_head_calls']==c['probe_norm_calls']==c['probability_calls']==expected
             and c['origin_source']==('bound_artifact' if is_full else 'native_prefill'),'Prefix replay reclassified or lost origin')
        fusions+=1
    for case in plan['cases']:
        a=trajectories[(case['case_id'],'bare')];b=trajectories[(case['case_id'],'zero')]
        need(a['generated_ids']==b['generated_ids'] and torch.equal(a['raw_logits'],b['raw_logits']),'Independent zero/bare identity failed')
    origin_rows=[]
    for item in artifacts['origins']:
        key=(item['case_id'],item['condition']);artifact=origins[key];case=next(c for c in plan['cases'] if c['case_id']==key[0])
        bare=next(o for o in observations if o['case_id']==key[0] and o['condition']=='bare');raw=cache[bare['forward_indices'][0]]
        actual=compare_origin(torch,artifact,raw,case)
        need(item['artifact_sha256']==artifact['artifact_sha256'] and len(actual)==len(item['rows']) and all(metric_matches({k:a[k] for k in b if k not in ('case_id','row')},{k:b[k] for k in b if k not in ('case_id','row')})
            and a['row']==b['row'] and a['case_id']==b['case_id'] and a['numerical_rule_passed'] for a,b in zip(actual,item['rows'])),'Origin native all-row comparison differs')
        origin_rows.extend(actual)
    for item in artifacts['comparisons']:
        a,b=cache[item['cached_index']],cache[item['full_index']]
        actual=old.metric(torch,a['native_logits'][-1],b['native_logits'][-1]);need(metric_matches(actual,item) and item['descriptive_only'] is True,'Descriptive cache/full metric differs')
        need(item['prefix_ids']==trajectories[(item['case_id'],'active')]['generated_ids'][:item['step']],'Replay selected another prefix')
    inv=inventory(observations);need(kv==inv['kv_transitions'] and fusions==inv['gated_forwards'] and len(origin_rows)==164,'KV/fusion/origin coverage differs')
    budget=accounting(out);own=[x for x in budget['jobs'] if x['job_id']==str(summary['slurm_job_id'])]
    need(len(own)==1 and own[0]['state']=='COMPLETED' and own[0]['exit_code']=='0:0' and own[0]['gpus']==1,'Successful GPU allocation absent')
    analysis=dict(schema_version=1,protocol=PROTOCOL,passed=True,completed=True,source_sha256=frozen,plan_file=summary['plan_file'],plan_sha256=summary['plan_sha256'],
        run_directory=str(Path(args.run_directory).resolve()),run_summary_sha256=sha(Path(args.run_directory)/'summary.json'),native_identity_sha256=plan['native_identity_sha256'],
        inventory=inv,origin_rows=164,kv_transitions=kv,head_replays=len(head_rows),maximum_head_tv=max(x['full_vocabulary_tv'] for x in head_rows),
        cache_full_failures=[x for x in artifacts['comparisons'] if not x['numerical_rule_passed']],accounting=budget,
        zero_identity_passed=True,original_gate_reuse_passed=True,no_fit=True,no_accuracy=True,no_adaptive_reasoning_claim=True)
    save(out/'analysis.json',analysis)
    (out/'REPORT.md').write_text('# Mixed Cosmos semantic streaming software\n\nNative/gate/head/zero integrity passed. Cache/full differences remain descriptive.\n\n[Evidence and resource ledger](analysis.json). No answer accuracy or reasoning-composition claim.\n')
    return dict(passed=True,completed=True,source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),allocated_gpu_seconds=budget['allocated_gpu_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true');group.add_argument('--report',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--run-directory',type=Path);args=parser.parse_args();native.require_slurm(gpu=args.run)
    if not args.run:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU check/report requires CPU Slurm')
    kind='check' if args.check else 'run' if args.run else 'report';out=OUT/f'{kind}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out);started=time.perf_counter()
    try:
        if args.check:result=check(out,frozen)
        elif args.run:need(args.plan is not None,'Supply exact CPU plan');result=run(args,out,frozen)
        else:need(args.run_directory is not None,'Supply completed GPU directory');result=report(args,out,frozen)
        need(sources()==frozen,'Sources changed during execution')
        if result is not None:result.update(seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
