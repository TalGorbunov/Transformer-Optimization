"""Held native learned-selection software audit, without answer scoring.

Fourteen training-derived natural trajectories and every active strict prefix.
Every native call has one native-shaped head replay. CPU preparation/report own
all input/source/raw checks; no benchmark trajectory or fitted state is loaded.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import native_learned_selection_runtime as runtime
from scripts import profile_native_vision_v7_runtime as old
from scripts import stage_native_identity_join_learned as stage
from scripts import stage_native_vision_v10_features as backend
from scripts.probe_native_vision_mixed_cache_localization import mask_check
from scripts.probe_native_vision_v2_prefix import fingerprint
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,model_metadata
native=runtime.native
MODES=('clip','sigmoid','softmax');SEED=20261124
OUT=REPO/'outputs/native_aggregation_vlm/identity_join_learned/software'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_learned_software')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_learned_software')
STAGE=REPO/'outputs/native_aggregation_vlm/identity_join_learned/data_staging/render_442988/summary.json'
BACKBONE_PLAN=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/check_442830/plan.json'
BACKBONE_PROOF=REPO/'outputs/native_aggregation_vlm/v18/semantic_gate_software/profile_442831/summary.json'
PROTOCOL='native_learned_selection_software';JOB_NAME='v19_selection_software'
POLICY=dict(protocol=PROTOCOL,seed=SEED,modes=list(MODES),maximum_new_tokens=4,parameters=1041697,
    cases='lowest SID train_N16; same images plus48 canonical outside-room atoms at Steps17..64',
    local_prompt='unaltered complete global question',maximum_generated_tokens=56,maximum_active_prefixes=24,
    trajectories=14,maximum_model_calls=80,maximum_visual_calls=38,maximum_extra_head_calls=80,origin_probes=0,
    head_tv_max=.02,head_top1_exact=True,native_dtype='torch.float16',branch_dtype='torch.float32',
    per_gpu_seconds=300,campaign_gpu_seconds=900,maximum_campaign_gpus=1,maximum_user_gpus=4,
    zero_identity_exact=True,active_clip_sigmoid_identity_exact=True,cache_full_descriptive=True,
    no_fit=True,no_accuracy=True,no_test_trajectory=True,no_persistent_kv_write=True)
OWN=('scripts/profile_native_learned_selection.py','slurm/native_learned_selection_check.sbatch',
     'slurm/native_learned_selection_profile.sbatch','slurm/native_learned_selection_report.sbatch',
     'scripts/stage_native_vision_v10_features.py','gnnformer/constants.py')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def sources():
    names=set(OWN)|set(runtime.OWN)|set(old.OWN)|set(stage.source_hashes())
    return {name:sha(REPO/name) for name in sorted(names)}


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or h==expected,'Bound input changed: '+str(path));bindings[str(path)]=h;return h


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        p=out/'source'/name.replace('/','_');p.write_bytes((REPO/name).read_bytes());need(sha(p)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Learned selection native software\n\n[Summary](summary.json) · [Sources](source_hashes.json). No answer scoring or fit.\n')
    return frozen


def backbone_identity():
    bindings={}
    bind(BACKBONE_PLAN,bindings,'45921222241820700a8e3cbd6637c819f331c4c9f27348b8896e1a233655b8aa')
    bind(BACKBONE_PROOF,bindings,'212567976c10b4c998b8368411a9694ffa66e410db03860f860a653c264510b1')
    plan=read(BACKBONE_PLAN);proof=read(BACKBONE_PROOF);prior=plan['native_identity_payload']
    need(proof['passed'] and proof['completed'] and proof['native_head_replay_passed']
         and proof['plan_sha256']==sha(BACKBONE_PLAN) and proof['native_identity_sha256']==plan['native_identity_sha256'],
         'Native backbone proof differs')
    identity=dict(schema_version=1,**{k:prior[k] for k in ('model','runtime','processor','native_api','native_dtypes',
        'norm_weight','head_weight','norm_source_sha256','rms_norm_eps')})
    need(identity['model']['path']==str(MODEL) and identity['native_dtypes']==dict(norm='torch.float16',lm_head='torch.float16'),
         'Require the actual frozen Qwen backbone, not a different model or recast head')
    return identity,bindings


def initial_states(torch):
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);core=ParallelLocalLearnedSelection()
        zero={k:v.detach().clone() for k,v in core.state_dict().items()}
        with torch.no_grad():core.up.weight.normal_(0,.001)
        active={k:v.detach().clone() for k,v in core.state_dict().items()}
    need(sum(v.numel() for v in zero.values())==1041697 and not bool(zero['up.weight'].any())
         and all(torch.equal(zero[k],active[k]) for k in zero if k!='up.weight'),'Untrained core initialization differs')
    return dict(zero=zero,active=active)


def inventory(rows):
    expected=[(n,c,m) for n in (16,64) for c,m in [('bare',None)]+[(c,m) for c in ('zero','active') for m in MODES]]
    need([(r['n_frames'],r['condition'],r['mode']) for r in rows]==expected,'Fourteen prescribed trajectories required')
    need(all(type(r['tokens']) is int and 1<=r['tokens']<=4 for r in rows),'Natural token bound differs')
    g=sum(r['tokens'] for r in rows);a=sum(r['tokens'] for r in rows if r['condition']=='active')
    return dict(generated_tokens=g,active_prefixes=a,model=g+a,visual=14+a,head_replays=g+a,extra_heads=g+a,
        kv_transitions=g-14,selection_calls=sum(r['tokens'] for r in rows if r['condition']!='bare')+a,origin_probes=0)


def parse_accounting(raw,exclude_job=None):
    jobs=[];events=[]
    for line in raw.splitlines():
        if not line.strip():continue
        f=line.split('|');need(len(f)==9,'Unexpected scheduler record')
        job,name,partition,state,code,seconds,tres,start,end=f
        if partition!='gpu' or name!=JOB_NAME or job==exclude_job:continue
        state=state.split()[0].rstrip('+');need(state in TERMINAL and job.isdigit(),'Prior software allocation is not terminal')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed);seconds=int(seconds)
        need(gpus in (0,1) and seconds>=0 and (not typed or sum(typed)==gpus) and (not gpus or seconds<=300),'Per-allocation cap/count differs')
        if gpus and start!=end:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(b>=a,'Invalid allocation interval');events.extend([(a,1),(b,-1)])
        jobs.append(dict(job_id=job,name=name,state=state,exit_code=code,gpus=gpus,elapsed_seconds=seconds,gpu_seconds=gpus*seconds))
    need(len({j['job_id'] for j in jobs})==len(jobs),'Duplicate GPU allocation')
    total=sum(j['gpu_seconds'] for j in jobs);running=maximum=0
    for _,change in sorted(events):running+=change;need(running>=0,'Invalid interval order');maximum=max(maximum,running)
    need(running==0 and maximum<=1 and total<=900,'Software cumulative resource cap exceeded')
    return dict(passed=True,jobs=jobs,allocated_gpu_seconds=total,maximum_concurrent_gpus=maximum,failed_and_zero_allocations_retained=True)


def accounting(out,exclude_job=None):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);path=out/'all_user_sacct.psv';path.write_text(result.stdout)
    return dict(parse_accounting(result.stdout,exclude_job),file=str(path),sha256=sha(path),command=command)


def project_concurrency(out):
    result=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i|%P|%T|%b'],capture_output=True,text=True,check=True)
    path=out/'live_user_queue.psv';path.write_text(result.stdout);total=0
    for line in result.stdout.splitlines():
        if not line.strip():continue
        _,partition,state,gres=[x.strip() for x in line.split('|')]
        if partition=='gpu' and state in ('RUNNING','COMPLETING'):
            entries=[x for x in gres.split(',') if x.startswith(('gpu:','gres/gpu:'))];need(entries,'Unknown active GPU allocation')
            total+=sum(int(x.rsplit(':',1)[1].split('(')[0]) for x in entries)
    need(total<=4,'More than four user GPUs active');return dict(passed=True,active_user_gpus=total,file=str(path),sha256=sha(path))


def self_test(out):
    command=[sys.executable,'tests/test_parallel_local_learned_selection.py']
    result=subprocess.run(command,capture_output=True,text=True,cwd=REPO);log=out/'unit_tests.log';log.write_text(result.stdout+result.stderr)
    need(result.returncode==0 and re.search(r'Ran 10 tests',result.stderr),'Learned core/runtime CPU tests failed')
    rows=[dict(n_frames=n,condition=c,mode=m,tokens=4) for n in (16,64) for c,m in
          [('bare',None)]+[(c,m) for c in ('zero','active') for m in MODES]]
    need(inventory(rows)==dict(generated_tokens=56,active_prefixes=24,model=80,visual=38,head_replays=80,extra_heads=80,
        kv_transitions=42,selection_calls=72,origin_probes=0),'Maximum software inventory differs')
    rows[0]['tokens']=1;need(inventory(rows)['model']==77,'Early-stop inventory differs')
    row='1|'+JOB_NAME+'|gpu|FAILED|1:0|7|gres/gpu=1,gres/gpu:b200=1|2026-09-11T10:00:00|2026-09-11T10:00:07'
    zero='2|'+JOB_NAME+'|gpu|CANCELLED|0:0|0||Unknown|Unknown'
    need(parse_accounting(row+'\n'+zero)['allocated_gpu_seconds']==7,'Failed/zero accounting differs')
    for gres in ('gres/gpu=1','gres/gpu:b200=1'):
        need(parse_accounting(row.replace('gres/gpu=1,gres/gpu:b200=1',gres))['allocated_gpu_seconds']==7,'GPU TRES forms differ')
    for bad in (row+'\n'+row,row.replace('|7|','|301|'),row.replace('|FAILED|','|RUNNING|'),row.replace('gpu:b200=1','gpu:b200=2')):
        try:parse_accounting(bad)
        except ValueError:pass
        else:raise ValueError('Invalid resource fixture passed')
    return dict(passed=True,core_runtime_tests=10,inventory_resource_tests=True,log_file=str(log),log_sha256=sha(log),model_calls=0,head_calls=0)


def software_cases(manifest):
    base=min(manifest['splits']['train_N16']['samples'],key=lambda r:r['sid'])
    need(base['split']=='train' and base['n_frames']==16,'Software base must be training-only')
    view=stage.runtime_view(base);extended=dict(view,sid=view['sid']+'__software_N64',n_frames=64,image_files=list(view['image_files']))
    atoms=read(manifest['render_cache_file']);rng=random.Random(SEED);chosen=[]
    for step in range(17,65):
        candidates=sorted([a for a in atoms.values() if a['atom'][2]==step and a['atom'][1] not in base['room_pair']],key=lambda a:a['atom'])
        need(candidates,'No canonical outside-room atom for software extension')
        atom=candidates[rng.randrange(len(candidates))];chosen.append(atom)
        extended['image_files'].append({k:atom[k] for k in ('path','sha256','dimensions','mode')})
    return [view,extended],dict(parent_sid=base['sid'],parent_split='train',parent_n_frames=16,
        room_pair=base['room_pair'],seed=SEED,appended_atoms=chosen,no_test_scene_used=True)


def check(out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4);tests=self_test(out);manifest=stage.verify_stage(STAGE);identity,bindings=backbone_identity()
    bind(STAGE,bindings);published=read(STAGE);bind(published['manifest_file'],bindings,published['manifest_sha256'])
    for key in ('plan','parent_plan','runtime_inputs','samples','audit','render_cache'):
        bind(manifest[key+'_file'],bindings,manifest[key+'_sha256'])
    samples,derivation=software_cases(manifest)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=backend.native_api(processor)
    need(api==identity['native_api'] and fingerprint(processor,str(transformers.__version__))==identity['processor']
         and backend.runtime_identity()==identity['runtime'] and model_metadata()==identity['model'],'CPU native/processor/source identity differs')
    bundles={};cases=[]
    for sample in samples:
        for image in sample['image_files']:bind(image['path'],bindings,image['sha256'])
        bundle=runtime.prepare_scene(processor,sample,verify_processor_parity=True)
        layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle);key='train_derived_N'+str(sample['n_frames'])
        bundles[key]=bundle;cases.append(dict(case_id=key,sample=sample,metadata=bundle['metadata'],layout=layout['metadata']))
    destination=DATA/out.name;destination.mkdir(parents=True,exist_ok=False)
    weights_dir=CKPT/out.name;weights_dir.mkdir(parents=True,exist_ok=False)
    prepared=destination/'prepared.pt';initial=weights_dir/'initial.pt';states=initial_states(torch)
    torch.save(dict(schema_version=1,bundles=bundles),prepared);torch.save(states,initial)
    precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   float32_matmul_precision=torch.get_float32_matmul_precision())
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,
        native_identity=identity,native_identity_sha256=native.object_sha(identity),precision=precision,
        cases=cases,case_derivation=derivation,prepared_file=str(prepared),prepared_sha256=sha(prepared),
        initial_file=str(initial),initial_sha256=sha(initial),initial_state_identity=old.state_identity(states),tests=tests,no_model_loaded=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=tests,no_model_loaded=True,
        native_identity=identity,native_identity_sha256=plan['native_identity_sha256'])


def verify_plan(path):
    path=Path(path).resolve();plan=read(path)
    need(sha(path)==path.with_suffix('.sha256').read_text().strip() and plan['policy']==POLICY
         and plan['source_sha256']==sources(),'Plan/source/policy differs')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'CPU source snapshot differs')
    for filename,h in plan['artifact_bindings'].items():need(sha(filename)==h,'Bound input differs')
    for key in ('prepared','initial'):need(sha(plan[key+'_file'])==plan[key+'_sha256'],'Consumed software tensor packet differs')
    need(native.object_sha(plan['native_identity'])==plan['native_identity_sha256'],'Native identity digest differs')
    need(sha(plan['tests']['log_file'])==plan['tests']['log_sha256'] and plan['tests']['passed'],'CPU test proof differs')
    for filename,h in plan['native_identity']['native_api']['source_sha256'].items():need(sha(filename)==h,'Installed native source differs')
    summary=read(path.parent/'summary.json');need(summary['passed'] and summary['completed'] and summary['plan_sha256']==sha(path),'Complete CPU check required')
    return plan


def fusion_algebra(torch,hidden,capture,weights):
    """Independent same-device functional readout, not a second core forward."""
    import torch.nn.functional as F
    from gnnformer.parallel_local_learned_selection import selection_weights
    device=weights['up.weight'].device;local=hidden[:-1].to(device);global_states=hidden[-1].to(device)
    def rms(x):x=x.float();return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        query=F.linear(rms(global_states),weights['query.weight'])
        payload=torch.tanh(F.linear(rms(local),weights['local.weight'])+query.unsqueeze(0)+weights['local_bias'])
        scores=F.linear(payload,weights['selection_weight'].unsqueeze(0)).squeeze(-1)+weights['selection_bias']
        gates=selection_weights(scores,capture['mode'])
        messages=gates.unsqueeze(-1)*payload;aggregate=messages.sum(0)
        preactivation=F.linear(aggregate,weights['aggregate_projection.weight'],weights['aggregate_projection.bias'])+query
        delta=F.linear(F.silu(preactivation),weights['up.weight'])
    expected=dict(query=query,payload=payload,scores=scores,gates=gates,messages=messages,aggregate=aggregate,preactivation=preactivation,delta=delta)
    need(all(torch.equal(v.cpu(),capture[k]) for k,v in expected.items()),'Captured learned selection algebra differs on the native device')
    need(torch.equal(capture['native_query_hidden'],hidden) and torch.equal(capture['local_states'],hidden[:-1])
         and torch.equal(capture['global_states'],hidden[-1]),'Current raw native query ownership differs')
    fused=hidden.clone();fused[-1]=hidden[-1]+delta.cpu().half()
    need(torch.equal(capture['native_delta'],delta.cpu().half()) and torch.equal(capture['fused_query_hidden'],fused)
         and torch.equal(capture['fused_global'],fused[-1]),'Global-only native FP16 cast/write differs')
    return fused,dict(passed=True,mode=capture['mode'],functional_algebra_exact=True,delta_nonzero=bool(delta.ne(0).any()),
        closed_coordinates_exact=bool((messages[gates==0]==0).all()),local_rows_untouched=True,core_forward_recomputed=False)


class Audit(old.NativeAudit):
    def __init__(self,model,data):
        super().__init__(model,data);self.controller=None;self.branch_hook=None
        self.head_replays=0;self.ordinary_heads=0;self.branch_calls=0
    def set_controller(self,value):
        if self.branch_hook is not None:self.branch_hook.remove();self.branch_hook=None
        self.controller=value
        if value is not None:
            def branch(*_):self.branch_calls+=1
            self.branch_hook=value.branch.register_forward_pre_hook(branch)
    def __enter__(self):
        super().__enter__()
        need(len(self.layers)==28,'Expected all28 frozen native decoder layers')
        def normalized(module,args,output):self.capture['normalized']=output[:,-1:,:].detach().cpu().clone()
        def head(*_):self.ordinary_heads+=1
        self.handles.extend([self.model.model.language_model.norm.register_forward_hook(normalized),
                             self.model.lm_head.register_forward_pre_hook(head)])
        return self
    def __exit__(self,*args):
        self.set_controller(None);return super().__exit__(*args)
    def after(self,module,args,output):
        import torch
        v=self.capture;c=self.counts;hidden=v['pre_final_rms_hidden'].unsqueeze(1)
        native_logits=output.logits[:,-1:,:].detach().cpu().clone();fusion=None;fused=hidden.clone();algebra=None
        if self.controller is not None:
            fusion=self.controller.export_last_capture(cpu=True);fusion['mode']=self.controller.selection_mode
            fused=fusion['fused_query_hidden']
        # Retain the native evidence even if a subsequent replay or assertion fails.
        raw=dict(schema_version=1,context=dict(self.context),native_logits=native_logits,native_query_hidden=hidden,
                 normalized=v['normalized'],fusion=fusion,
                 **{k:v[k] for k in ('input_ids','attention_mask','position_ids','causal_mask','cache_position')})
        index=len(self.records);partial=self.data/f'partial_forward_{index:03d}.pt';torch.save(raw,partial)
        need(all(c[k]-v['before_counts'][k]==1 for k in ('language','norm','attention'))
             and c['visual']-v['before_counts']['visual']==int(v['visual_expected']),'Native submodule inventory differs')
        masks=mask_check(torch,v['causal_mask'],v['attention_mask'],v['cache_position'])
        if fusion is not None:fused,algebra=fusion_algebra(torch,hidden,fusion,self.controller.branch.state_dict())
        with torch.inference_mode():
            normalized=self.model.model.language_model.norm.forward(fused.to(self.model.device))
            replay=self.model.lm_head.forward(normalized);self.head_replays+=1
        need(native_logits.dtype==replay.dtype==hidden.dtype==normalized.dtype==torch.float16
             and native_logits.shape==replay.shape==(hidden.shape[0],1,152064),'Native head shape/dtype differs')
        metrics=[old.metric(torch,a,b) for a,b in zip(native_logits[:,0],replay[:,0].cpu())]
        cache=output.past_key_values;retained=None;cache_meta=None
        if v['use_cache']:
            _,cache_meta=old.mixed.cache_snapshot(cache,hidden.shape[0],v['attention_mask'].shape[1],len(self.layers))
            if self.previous is not None:retained=old.mixed.prefix_preserved(torch,cache,self.previous,v['old_length'])
        else:need(cache is None and self.previous is None,'Uncached reference returned/used KV')
        self.previous=None
        raw.update(replay_logits=replay.cpu(),replay_normalized=normalized.cpu(),fused_query_hidden=fused)
        path=self.data/f'forward_{index:03d}.pt';torch.save(raw,path)
        record=dict(context=dict(self.context),forward_index=index,path=str(path),sha256=sha(path),partial_file=str(partial),partial_sha256=sha(partial),
            batch_size=hidden.shape[0],query_tokens=v['input_ids'].shape[1],key_tokens=v['attention_mask'].shape[1],visual=int(v['visual_expected']),
            mask_audit=masks,cache_layers=cache_meta,previous_prefix_exact_by_layer=retained,expected_visual_identity=v['expected_visual_identity'],
            head_replay=metrics,normalized_exact=torch.equal(v['normalized'],normalized.cpu()),fusion=algebra,
            seconds=time.perf_counter()-v['started'])
        self.records.append(record);save(self.data/f'forward_{index:03d}.json',record)


def live_identity(torch,loaded,plan):
    from transformers import __version__ as transformers_version
    model=loaded.model;norm=native.native_contract(model,None);expected=plan['native_identity']
    need(fingerprint(loaded.processor,str(transformers_version))==expected['processor']
         and backend.runtime_identity()==expected['runtime'] and backend.native_api(loaded.processor)[2]==expected['native_api']
         and model_metadata()==expected['model'],'Live backbone/runtime/processor/API differs')
    need(native.tensor_info(norm.weight)==expected['norm_weight'] and native.tensor_info(model.lm_head.weight)==expected['head_weight']
         and sha(inspect.getfile(type(norm)))==expected['norm_source_sha256']
         and float(norm.variance_epsilon)==expected['rms_norm_eps'],'Native norm/head weight/source identity differs')
    quant=model.config.quantization_config;quant=quant.to_dict() if hasattr(quant,'to_dict') else dict(quant)
    need(quant['load_in_4bit'] and quant['bnb_4bit_use_double_quant'] and quant['bnb_4bit_quant_type']=='nf4'
         and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','Actual NF4/double/BF16 backend differs')
    precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   float32_matmul_precision=torch.get_float32_matmul_precision())
    need(precision==plan['precision'],'Frozen native precision flags changed')
    return dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=str(torch.version.cuda),quantization=quant,precision=precision)


def run(args,out,frozen):
    import torch
    from gnnformer.runtime import load_runtime,get_rope_index_fn
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    torch.set_num_threads(4);started=time.perf_counter();plan=verify_plan(args.plan)
    ledger=accounting(out,os.environ['SLURM_JOB_ID']);need(ledger['allocated_gpu_seconds']+300<=900,'Software GPU reservation exceeds cap')
    save(out/'resource_reservation.json',dict(ledger,reserved_seconds=300));concurrency=project_concurrency(out)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','Require one B200 GPU')
    packet=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(old.state_identity(weights)==plan['initial_state_identity']==old.state_identity(initial_states(torch)),'Untrained software state changed')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=live_identity(torch,loaded,plan)
    cores={m:ParallelLocalLearnedSelection(mode=m).to(model.device).eval().requires_grad_(False) for m in MODES}
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    versions={n:p._version for n,p in model.named_parameters()};data=DATA/out.name;data.mkdir(parents=True,exist_ok=False)
    observations=[];comparisons=[];zero_checks=[];matching_checks=[];audit=Audit(model,data);failure=None
    try:
        with audit:
            for case in plan['cases']:
                key=case['case_id'];bundle=packet['bundles'][key];n=case['metadata']['n_frames']
                need(bundle['metadata']==case['metadata'] and native.audit_layout(get_rope_index_fn(model),bundle)['metadata']==case['layout'],'Prepared native scene/layout differs')
                baseline=None;active_clip=None
                for condition,mode in [('bare',None)]+[(c,m) for c in ('zero','active') for m in MODES]:
                    core=None if mode is None else cores[mode]
                    if core is not None:core.load_state_dict(weights[condition],strict=True)
                    audit.context=dict(case_id=key,n_frames=n,condition=condition,mode=mode,phase='generation');offset=len(audit.records)
                    modules=(model,model.model.visual,model.model.language_model,model.model.language_model.norm,model.lm_head)
                    hooks=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules];rope=getattr(model.model,'rope_deltas',None)
                    result=runtime.generate_native(model,loaded.processor,core,bundle,selection_mode=mode,
                        native_identity_sha256=plan['native_identity_sha256'],capture=True,controller_observer=audit.set_controller)
                    need(hooks==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules]
                         and getattr(model.model,'rope_deltas',None) is rope and audit.controller is None,'Runtime hook/rope cleanup differs')
                    records=audit.records[offset:];old.audit_sequence(torch,model,bundle,records,result['generated_ids'])
                    path=data/f'{key}_{condition}_{mode or "none"}.pt';torch.save(result,path)
                    row=dict(case_id=key,n_frames=n,condition=condition,mode=mode,tokens=len(result['generated_ids']),
                        generated_ids=result['generated_ids'],completed=result['completed'],truncated=result['truncated'],counters=result['counters'],
                        file=str(path),sha256=sha(path),model_seconds=result['model_seconds'],four_token_seconds_bound=result['model_seconds']*4/len(result['generated_ids']),
                        forward_indices=[r['forward_index'] for r in records])
                    observations.append(row);save(out/f'trajectory_{len(observations)-1:03d}.json',row)
                    for step,r in enumerate(records):
                        raw=torch.load(r['path'],map_location='cpu',weights_only=True)
                        need(torch.equal(raw['native_logits'][-1,0].float(),result['raw_logits'][step]),'Streamed native global logits differ')
                    if condition=='bare':baseline=result;continue
                    if condition=='zero':
                        need(result['generated_ids']==baseline['generated_ids'] and torch.equal(result['raw_logits'],baseline['raw_logits']), 'Zero-U exact native identity failed')
                        zero_checks.append(dict(case_id=key,mode=mode,passed=True,tokens=result['generated_ids'],raw_logits_exact=True));continue
                    need(any(r['fusion']['delta_nonzero'] for r in records),'Active untrained core never produced a residual')
                    if mode=='clip':active_clip=result
                    if mode=='sigmoid':
                        need(result['generated_ids']==active_clip['generated_ids'] and torch.equal(result['raw_logits'],active_clip['raw_logits']),
                             'Clip/sigmoid initial learned maps must produce exact native identity')
                        matching_checks.append(dict(case_id=key,passed=True,raw_logits_exact=True,tokens=result['generated_ids']))
                    for step in range(len(result['generated_ids'])):
                        prefix=result['generated_ids'][:step];prefixed=runtime.append_observed_prefix(bundle,prefix)
                        audit.context=dict(case_id=key,n_frames=n,condition=condition,mode=mode,phase='full_prefix',step=step,prefix_ids=prefix)
                        rope=getattr(model.model,'rope_deltas',None)
                        reference=runtime.forward_native(model,prefixed,core,selection_mode=mode,native_identity_sha256=plan['native_identity_sha256'],
                            capture=True,cpu=True,controller_observer=audit.set_controller)
                        full_record=audit.records[-1];full=torch.load(full_record['path'],map_location='cpu',weights_only=True)
                        layout=native.audit_layout(get_rope_index_fn(model),prefixed)
                        need(torch.equal(full['input_ids'],prefixed['inputs']['input_ids']) and torch.equal(full['attention_mask'],prefixed['inputs']['attention_mask'])
                             and torch.equal(full['position_ids'],layout['position_ids']) and reference['metadata']['layout']==layout['metadata']
                             and reference['counters']['probe_head']==0 and audit.controller is None and getattr(model.model,'rope_deltas',None) is rope,
                             'Actual full-prefix native input/layout/cleanup differs')
                        comparisons.append(dict(case_id=key,mode=mode,step=step,prefix_ids=prefix,cached_index=records[step]['forward_index'],
                            full_index=full_record['forward_index'],**old.metric(torch,result['raw_logits'][step],reference['global_logits']),descriptive_only=True))
                print(json.dumps(dict(case=key,model_calls=audit.counts['model'])),flush=True)
        inv=inventory(observations)
        need(audit.counts==dict(model=inv['model'],visual=inv['visual'],language=inv['model'],norm=inv['model'],attention=inv['model'])
             and audit.head_replays==audit.ordinary_heads==inv['model'] and audit.branch_calls==inv['selection_calls']
             and len(comparisons)==inv['active_prefixes'] and len(zero_checks)==6 and len(matching_checks)==2,'Exact software invocation inventory differs')
        need(versions=={n:p._version for n,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen model changed')
        need(all(old.state_identity({'active':c.state_dict()})['active']==plan['initial_state_identity']['active'] for c in cores.values()),'Untrained core changed')
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        for name,value in [('forwards',audit.records),('observations',observations),('comparisons',comparisons),('zero_checks',zero_checks),('matching_checks',matching_checks)]:save(out/(name+'.json'),value)
        complete=len(observations)==14 and failure is None
        head_failures=[dict(forward_index=r['forward_index'],row=i,**m) for r in audit.records for i,m in enumerate(r['head_replay']) if not m['numerical_rule_passed']]
        passed=complete and not head_failures
        save(out/'summary.json',dict(schema_version=1,protocol=PROTOCOL,phase='profile',completed=complete,passed=passed,
            computational_integrity_passed=complete,native_replay_passed=passed,zero_identity_passed=len(zero_checks)==6,
            initial_clip_sigmoid_identity_passed=len(matching_checks)==2,plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),
            source_sha256=frozen,native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],
            hardware=hardware,counts=audit.counts,ordinary_head_calls=audit.ordinary_heads,captured_head_replays=audit.head_replays,
            core_forward_calls=audit.branch_calls,origin_probes=0,head_failures=head_failures,
            maximum_head_tv=max((m['full_vocabulary_tv'] for r in audit.records for m in r['head_replay']),default=None),
            cache_full_failures=[x for x in comparisons if not x['numerical_rule_passed']],
            files={name:dict(file=str(out/(name+'.json')),sha256=sha(out/(name+'.json'))) for name in ('forwards','observations','comparisons','zero_checks','matching_checks')},
            resource_reservation=ledger,project_concurrency=concurrency,model_load_seconds=load_seconds,seconds=time.perf_counter()-started,
            failure=failure,slurm_job_id=os.environ['SLURM_JOB_ID'],no_fit=True,no_accuracy=True))
    need(passed,'Native binding failed; all available raw observations retained')


def metric_matches(actual,saved):
    return all(actual[k]==saved[k] if isinstance(actual[k],bool) else
               math.isclose(actual[k],saved[k],rel_tol=1e-9,abs_tol=1e-12) for k in actual)


def _verify_run(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path
    summary=read(path);plan=verify_plan(summary['plan_file'])
    need(summary['phase']=='profile' and summary['protocol']==PROTOCOL and summary['passed'] and summary['completed']
         and summary['computational_integrity_passed'] and summary['native_replay_passed'] and summary['zero_identity_passed']
         and summary['initial_clip_sigmoid_identity_passed'] and summary['source_sha256']==plan['source_sha256']==sources()
         and summary['plan_sha256']==sha(summary['plan_file']) and summary['native_identity']==plan['native_identity']
         and summary['native_identity_sha256']==plan['native_identity_sha256'],'Completed native software/source gates required')
    for name,h in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'GPU source snapshot differs')
    artifacts={}
    for name,item in summary['files'].items():need(sha(item['file'])==item['sha256'],'GPU record ledger differs');artifacts[name]=read(item['file'])
    inv=inventory(artifacts['observations'])
    need(summary['counts']==dict(model=inv['model'],visual=inv['visual'],language=inv['model'],norm=inv['model'],attention=inv['model'])
         and summary['captured_head_replays']==summary['ordinary_head_calls']==len(artifacts['forwards'])==inv['model']
         and summary['core_forward_calls']==inv['selection_calls'] and summary['origin_probes']==0
         and len(artifacts['comparisons'])==inv['active_prefixes'] and len(artifacts['zero_checks'])==6
         and len(artifacts['matching_checks'])==2,'Software inventory differs')
    need([r['forward_index'] for r in artifacts['forwards']]==list(range(inv['model'])),'Native forward indices differ')
    for r in artifacts['forwards']:
        need(sha(r['path'])==r['sha256'] and sha(r['partial_file'])==r['partial_sha256'],'Native raw evidence differs')
    for r in artifacts['observations']:need(sha(r['file'])==r['sha256'],'Natural trajectory archive differs')
    return path,plan,summary,artifacts


def verify_profile(path):
    """Return the passed independent CPU-report summary; no tensor/model loads."""
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path;summary=read(path)
    need(summary['phase']=='report' and summary['protocol']==PROTOCOL and summary['passed'] and summary['completed']
         and summary['independent_raw_audit_passed'] and summary['source_sha256']==sources(),'Independent software CPU report required')
    need(sha(summary['analysis_file'])==summary['analysis_sha256'],'Independent analysis changed')
    analysis=read(summary['analysis_file']);_,plan,gpu,artifacts=_verify_run(analysis['gpu_summary_file'])
    need(sha(analysis['gpu_summary_file'])==analysis['gpu_summary_sha256'] and analysis['passed'] and analysis['completed']
         and analysis['native_identity']==summary['native_identity']==plan['native_identity']
         and analysis['native_identity_sha256']==summary['native_identity_sha256']==plan['native_identity_sha256']
         and analysis['inventory']==inventory(artifacts['observations'])
         and summary['plan_file']==analysis['plan_file'] and summary['plan_sha256']==analysis['plan_sha256']==sha(summary['plan_file']),
         'Software report/native identity differs')
    for name,h in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Report source snapshot differs')
    return summary


def report(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    from gnnformer.parallel_local_learned_selection import selection_weights
    torch.set_num_threads(4);gpu_path,plan,gpu,artifacts=_verify_run(args.run_directory)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=backend.native_api(processor);need(api==plan['native_identity']['native_api']
        and fingerprint(processor,str(transformers.__version__))==plan['native_identity']['processor'],'CPU native processor/API differs')
    packet=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    trajectories={};records=artifacts['forwards'];raw_by_index={};owned=[];metrics=[];kv=0;fusions=0;algebra=[]
    for obs in artifacts['observations']:
        result=torch.load(obs['file'],map_location='cpu',weights_only=True);key=(obs['case_id'],obs['condition'],obs['mode']);trajectories[key]=result
        ids=result['generated_ids'];steps=len(ids);bundle=packet['bundles'][obs['case_id']];n=obs['n_frames'];batch=n+1
        need(ids==obs['generated_ids'] and steps==obs['tokens'] and result['raw_logits'].shape==(steps,152064)
             and result['raw_logits'].dtype==torch.float32 and torch.equal(result['raw_logits'],result['raw_logits'].half().float())
             and result['raw_logits'].argmax(-1).tolist()==ids,'Native raw argmax/history differs')
        eos=result['metadata']['generation']['native_eos_token_ids']
        need(eos==[151645,151643] and not any(i in eos for i in ids[:-1]) and result['completed']==(ids[-1] in eos)
             and result['truncated']==(ids[-1] not in eos) and (result['completed'] or steps==4),'Native EOS/stopping differs')
        selection=0 if obs['condition']=='bare' else steps
        need(result['counters']==obs['counters']==dict(model=steps,visual=1,language=steps,norm=steps,head=steps,
             broadcast=steps,selection=selection,probe_head=0) and len(result['captures'])==selection,'Trajectory selection/probe counts differ')
        need(result['metadata']['selection_mode']==obs['mode'] and result['metadata']['local_prompt']==result['metadata']['global_prompt']==bundle['metadata']['question']
             and result['metadata']['scene_input_identity']==runtime.input_identity(bundle,plan['native_identity_sha256']), 'Native prompt/input identity differs')
        need(len(obs['forward_indices'])==steps,'Trajectory native forward ownership differs')
        for t,index in enumerate(obs['forward_indices']):
            record=records[index];raw=torch.load(record['path'],map_location='cpu',weights_only=True)
            raw_by_index[index]=raw;owned.append(index);context=record['context']
            need(context==dict(case_id=obs['case_id'],n_frames=n,condition=obs['condition'],mode=obs['mode'],phase='generation')
                 and raw['context']==context and torch.equal(raw['native_logits'][-1,0].float(),result['raw_logits'][t]),'Raw call/trajectory identity differs')
            prefixed=runtime.append_observed_prefix(bundle,ids[:t]);layout=native.audit_layout(lambda **kw:fn(owner,**kw),prefixed)
            mask=prefixed['inputs']['attention_mask'];text=mask.cumsum(-1)-1;pos=raw['position_ids'];width=bundle['metadata']['prompt_width']
            need(torch.equal(raw['attention_mask'],mask) and pos.shape==(4,batch,width if t==0 else 1),'Observed key mask/position shape differs')
            if t==0:
                need(torch.equal(raw['input_ids'],bundle['inputs']['input_ids']) and torch.equal(pos[1:],layout['position_ids'])
                     and torch.equal(pos[0][mask.bool()],text[mask.bool()]) and record['visual']==1
                     and record['expected_visual_identity']=={k:bundle['metadata']['input_identity'][k] for k in ('pixel_values','image_grid_thw')},'Actual native prefill differs')
            else:
                flags=record['previous_prefix_exact_by_layer']
                need(type(flags) is list and len(flags)==28 and all(type(x) is bool and x for x in flags)
                     and raw['input_ids'].shape==(batch,1) and bool((raw['input_ids']==ids[t-1]).all())
                     and torch.equal(pos[1:],layout['position_ids'][:,:,-1:]) and torch.equal(pos[0],text[:,-1:])
                     and record['visual']==0,'All-layer native cached prefix/positions differ');kv+=1
            if selection:
                cap=result['captures'][t];need(all(torch.equal(v,raw['fusion'][k]) for k,v in cap.items()),'Trajectory/current selection capture differs')
    for comp in artifacts['comparisons']:
        result=trajectories[(comp['case_id'],'active',comp['mode'])];prefix=result['generated_ids'][:comp['step']]
        need(prefix==comp['prefix_ids'],'Full replay was not the actual generated strict prefix')
        index=comp['full_index'];record=records[index];raw=torch.load(record['path'],map_location='cpu',weights_only=True)
        raw_by_index[index]=raw;owned.append(index);bundle=packet['bundles'][comp['case_id']]
        prefixed=runtime.append_observed_prefix(bundle,prefix);layout=native.audit_layout(lambda **kw:fn(owner,**kw),prefixed)
        expected=dict(case_id=comp['case_id'],n_frames=bundle['metadata']['n_frames'],condition='active',mode=comp['mode'],
                      phase='full_prefix',step=comp['step'],prefix_ids=prefix)
        need(record['context']==raw['context']==expected and torch.equal(raw['input_ids'],prefixed['inputs']['input_ids'])
             and torch.equal(raw['attention_mask'],prefixed['inputs']['attention_mask']) and torch.equal(raw['position_ids'],layout['position_ids'])
             and record['visual']==1 and record['cache_layers'] is None and record['previous_prefix_exact_by_layer'] is None,'Uncached strict-prefix native layout differs')
        cached=raw_by_index[comp['cached_index']]
        actual=old.metric(torch,cached['native_logits'][-1,0],raw['native_logits'][-1,0])
        need(metric_matches(actual,comp),'Descriptive cached/full metric differs')
    need(sorted(owned)==list(range(len(records))),'Missing or multiply owned native calls')
    for record in records:
        raw=raw_by_index[record['forward_index']];h=raw['native_query_hidden'];batch=record['batch_size']
        need(h.shape==(batch,1,3584) and h.dtype==torch.float16
             and raw['native_logits'].shape==raw['replay_logits'].shape==(batch,1,152064)
             and raw['native_logits'].dtype==raw['replay_logits'].dtype==torch.float16
             and raw['normalized'].shape==raw['replay_normalized'].shape==h.shape
             and raw['normalized'].dtype==raw['replay_normalized'].dtype==torch.float16,'Raw native shapes/dtypes differ')
        need(record['normalized_exact']==torch.equal(raw['normalized'],raw['replay_normalized']),'Normalization equality record differs')
        need(mask_check(torch,raw['causal_mask'],raw['attention_mask'],raw['cache_position'])==record['mask_audit'],'Actual mask audit differs')
        expected=h.clone();c=raw['fusion']
        if c is not None:
            fusions+=1;need(c['mode']==record['context']['mode'] and torch.equal(c['native_query_hidden'],h)
                and torch.equal(c['local_states'],h[:-1]) and torch.equal(c['global_states'],h[-1]),'Current prefix selection ownership differs')
            gate=selection_weights(c['scores'],c['mode'])
            need(torch.allclose(gate,c['gates'],rtol=1e-6,atol=1e-7) and torch.equal(c['messages'],c['gates'].unsqueeze(-1)*c['payload'])
                 and bool((c['messages'][c['gates']==0]==0).all()),'Selection or closed-coordinate arithmetic differs')
            expected[-1]=h[-1]+c['delta'].half()
            need(torch.equal(c['native_delta'],c['delta'].half()) and torch.equal(c['fused_query_hidden'],expected)
                 and torch.equal(c['fused_global'],expected[-1]) and record['fusion']['functional_algebra_exact']
                 and not record['fusion']['core_forward_recomputed'],'Native cast/write or GPU algebra proof differs')
            algebra.append(dict(forward_index=record['forward_index'],cpu_gate_maximum_difference=float((gate-c['gates']).abs().max()),
                cpu_sum_maximum_difference=float((c['messages'].sum(0)-c['aggregate']).abs().max()),sum_reduction_difference_descriptive=True))
        need(torch.equal(expected,raw['fused_query_hidden']),'Bare/global-only native carry differs')
        actual=[old.metric(torch,a,b) for a,b in zip(raw['native_logits'][:,0],raw['replay_logits'][:,0])]
        need(len(actual)==len(record['head_replay'])==batch and all(metric_matches(a,b) for a,b in zip(actual,record['head_replay'])),'Native head replay rescore differs')
        metrics.extend(dict(forward_index=record['forward_index'],row=i,**m) for i,m in enumerate(actual))
    for case in plan['cases']:
        key=case['case_id'];bare=trajectories[(key,'bare',None)]
        for mode in MODES:
            zero=trajectories[(key,'zero',mode)]
            need(zero['generated_ids']==bare['generated_ids'] and torch.equal(zero['raw_logits'],bare['raw_logits']),'Zero native identity differs')
        clip,sigmoid=trajectories[(key,'active','clip')],trajectories[(key,'active','sigmoid')]
        need(clip['generated_ids']==sigmoid['generated_ids'] and torch.equal(clip['raw_logits'],sigmoid['raw_logits']),'Initial clip/sigmoid native identity differs')
    inv=inventory(artifacts['observations']);need(kv==inv['kv_transitions'] and fusions==inv['selection_calls'],'KV/selection coverage differs')
    ledger=accounting(out);success=[r for r in ledger['jobs'] if r['job_id']==gpu['slurm_job_id']]
    need(len(success)==1 and success[0]['state']=='COMPLETED' and success[0]['exit_code']=='0:0' and success[0]['gpus']==1,'Software allocation did not complete successfully')
    failures=[r for r in metrics if not r['numerical_rule_passed']]
    timing={mode:{str(r['n_frames']):r['four_token_seconds_bound'] for r in artifacts['observations'] if r['condition']=='active' and r['mode']==mode} for mode in MODES}
    analysis=dict(schema_version=1,protocol=PROTOCOL,completed=True,passed=not failures,source_sha256=frozen,
        gpu_summary_file=str(gpu_path),gpu_summary_sha256=sha(gpu_path),plan_file=gpu['plan_file'],plan_sha256=gpu['plan_sha256'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],inventory=inv,
        native_head_rows=len(metrics),head_metrics=metrics,head_failures=failures,maximum_head_tv=max(r['full_vocabulary_tv'] for r in metrics),
        kv_transitions=kv,kv_layers_per_transition=28,selection_calls=fusions,origin_probes=0,zero_checks=artifacts['zero_checks'],
        initial_clip_sigmoid_checks=artifacts['matching_checks'],cache_full=artifacts['comparisons'],
        cache_full_failures=[r for r in artifacts['comparisons'] if not r['numerical_rule_passed']],cpu_selection_checks=algebra,
        timing_by_mode=timing,timing_scope='observed native generation including full software observer/replay/save; scaled4/tokens',
        hardware=gpu['hardware'],model_load_seconds=gpu['model_load_seconds'],resources=ledger,no_fit=True,no_accuracy=True)
    save(out/'analysis.json',analysis)
    need(not failures,'Native head replay failed; independent raw report retained')
    return dict(passed=True,completed=True,phase='report',protocol=PROTOCOL,independent_raw_audit_passed=True,
        source_sha256=frozen,analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),
        plan_file=gpu['plan_file'],plan_sha256=gpu['plan_sha256'],
        native_identity=plan['native_identity'],native_identity_sha256=plan['native_identity_sha256'],inventory=inv,
        timing_by_mode=timing,model_load_seconds=gpu['model_load_seconds'],hardware=gpu['hardware'],no_fit=True,no_accuracy=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--profile',action='store_true');group.add_argument('--report',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--run-directory',type=Path);args=parser.parse_args()
    native.require_slurm(gpu=args.profile)
    if not args.profile:need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU allocation required')
    kind='check' if args.check else 'profile' if args.profile else 'report';out=OUT/f'{kind}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out);start=time.perf_counter()
    try:
        if args.check:result=check(out,frozen)
        elif args.profile:need(args.plan is not None,'Supply the exact CPU plan');result=run(args,out,frozen)
        else:need(args.run_directory is not None,'Supply the completed software GPU directory');result=report(args,out,frozen)
        need(sources()==frozen,'Source changed during execution')
        if result is not None:result.update(seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID']);save(out/'summary.json',result)
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen,seconds=time.perf_counter()-start,partial_outputs_retained=True));raise


if __name__=='__main__':main()
