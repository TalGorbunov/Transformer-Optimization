"""Fixed-token, unfitted Qwen/Cosmos native learned-memory software audit.

New source only. Read block26; write before block27 or before final norm.
43 fixed calls/20 vision +2 natural max8 runs +2 differentiable block replays/model.
Fixed continuations plus two bounded natural runs; no fit, accuracy, old features or layer search.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import hashlib
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
from scripts import native_learned_memory_runtime as runtime
from scripts import profile_native_vision_v11_memory as v11
from scripts import profile_native_learned_selection as selection
from scripts import profile_native_vision_reasoning_stream as cosmos
from scripts import stage_native_vision_v10_features as backend
from scripts.probe_native_vision_v2_prefix import fingerprint
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,model_metadata
native=runtime.native
OUT=REPO/'outputs/native_aggregation_vlm/learned_memory/software'
DATA=Path('/mnt/data/gabriele/gnn_transformer/learned_memory_software')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/learned_memory_software')
TRAIN=REPO/'outputs/native_aggregation_vlm/identity_join_learned/training/train_check_443056/plan.json'
TRAIN_SHA='739477cc4a9b5e8278d8e632f0691543fd02ce42673585a349015970c251d79f'
TRAIN_PROOF=REPO/'outputs/native_aggregation_vlm/learned_memory/training_check_443081/summary.json'
TRAIN_PROOF_SHA='6a5d2ca9c88c1423d395c89caed797cb55ca31fa5bed495c2cef7eb7307e1b4f'
COSMOS_PROOF=REPO/'outputs/native_aggregation_vlm/reasoning_semantic_gate/report_442926/summary.json'
MODELS=('qwen','cosmos');PLACEMENTS=('pre_last','post_last');SEED=20261124
PROTOCOL='native_learned_memory_fixed_token_software';JOB_NAME='learned_memory_software'
POLICY=dict(protocol=PROTOCOL,models=list(MODELS),selection_mode='clip',seed=SEED,parameters=1041697,
    lengths=[16,64],forced_text=['Therefore',':'],read_block=26,write_block=27,placements=list(PLACEMENTS),
    fixed_model_calls=43,fixed_vision_calls=20,natural_trajectories=2,natural_max_tokens=8,
    maximum_model_calls=59,maximum_vision_calls=22,maximum_captured_head_replays=59,
    standalone_block_calls=2,standalone_loss_heads=2,standalone_native_shape_heads=2,maximum_extra_head_calls=63,
    initialization='seed20261124 constructor; active U then selection_weight Normal(0,.001), all other tensors unchanged',
    native_dtype='torch.float16',core_dtype='torch.float32',head_tv_max=.02,head_top1_exact=True,
    zero_identity_exact=True,common_read_exact=True,cached_full_descriptive=True,
    perturbation='.1*RMS(common global first query)*unit-RMS sin(arange(H)); no tuning',
    restore='Only layer27 global first-query K and V, with identical later applied writes and fixed tokens',
    gradient_loss='Fixed colon token at middle query; independent historical/current/future delta leaves and one zero local leaf',
    per_gpu_seconds=600,campaign_gpu_seconds=1200,maximum_campaign_gpus=2,maximum_project_gpus=4,
    no_fit=True,no_accuracy=True,no_test_trajectory=True,no_reasoning_performance_claim=True)
OWN=('scripts/profile_native_learned_memory.py','scripts/native_learned_memory_training.py',
     'slurm/native_learned_memory_check.sbatch','slurm/native_learned_memory_profile.sbatch',
     'slurm/native_learned_memory_report.sbatch','tests/test_native_learned_memory_training.py',
     'tests/test_parallel_local_learned_memory.py','slurm/native_learned_memory_controller_check.sbatch',
     'slurm/native_learned_memory_training_check.sbatch')
TERMINAL=selection.TERMINAL


def sources():
    names=set(OWN)|set(runtime.sources())|set(v11.sources())|set(cosmos.sources())|set(selection.sources())
    return {n:sha(REPO/n) for n in sorted(names)}


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,h in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# Native learned-memory software\n\nFixed tokens; no fit or answer scoring. [Summary](summary.json) · [Sources](source_hashes.json).\n')
    return frozen


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or h==expected,'Bound artifact differs: '+str(path));bindings[str(path)]=h;return h


def source_proof(path,bindings,protocol,count,expected=None):
    bind(path,bindings,expected);value=read(path);need(value['passed'] and value['completed']
        and value['protocol']==protocol and value['tests_run']==count,'Complete frozen CPU fixture proof required')
    for name,h in value['source_sha256'].items():
        need(sha(REPO/name)==h,'CPU fixture source changed')
        copy=Path(path).parent/('code/'+name if protocol=='learned_memory_controller_cpu_fixtures' else 'source/'+name.replace('/','_'))
        # The selected-loss fixture stores its original basenames, without path prefixes.
        if not copy.exists():copy=Path(path).parent/'source'/Path(name).name
        bind(copy,bindings,h)
    bind(value['log_file'],bindings,value['log_sha256']);return value


def parse_accounting(raw,exclude_job=None,allow_active=False):
    jobs=[];events=[];active=[]
    for line in raw.splitlines():
        if not line.strip():continue
        f=line.split('|');need(len(f)==9,'Unexpected scheduler fields')
        job,name,partition,state,code,seconds,tres,start,end=f
        if name!=JOB_NAME or partition!='gpu' or job==exclude_job:continue
        need(job.isdigit(),'GPU allocation ID must be concrete');state=state.split()[0].rstrip('+')
        values=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
        gpus=int(values['gres/gpu']) if 'gres/gpu' in values else sum(typed);seconds=int(seconds)
        need(gpus in (0,1) and seconds>=0 and (not typed or sum(typed)==gpus),'GPU TRES/count differs')
        terminal=state in TERMINAL
        need(terminal or allow_active and state in ('RUNNING','COMPLETING','PENDING','CONFIGURING'),'Nonterminal allocation in completed ledger')
        if gpus:need(seconds<=600,'Per-allocation limit exceeded')
        row=dict(job_id=job,name=name,state=state,exit_code=code,gpus=gpus,elapsed_seconds=seconds,
                 gpu_seconds=gpus*seconds if terminal else 0,terminal=terminal)
        jobs.append(row)
        if not terminal:
            if gpus:active.append(job)
        elif gpus and start!=end:
            a,b=datetime.fromisoformat(start),datetime.fromisoformat(end);need(b>=a,'Invalid allocation interval');events.extend(((a,1),(b,-1)))
    need(len({r['job_id'] for r in jobs})==len(jobs),'Duplicate allocation')
    running=maximum=0
    for _,d in sorted(events):running+=d;need(running>=0,'Invalid overlap order');maximum=max(maximum,running)
    total=sum(r['gpu_seconds'] for r in jobs)
    need(running==0 and maximum<=2 and len(active)<=2 and total+600*len(active)<=1200,'Campaign resource cap exceeded')
    return dict(passed=True,jobs=jobs,allocated_gpu_seconds=total,active_jobs=active,
        reserved_active_gpu_seconds=600*len(active),maximum_concurrent_gpus=maximum,failed_and_zero_allocations_retained=True)


def accounting(out,exclude_job=None,allow_active=False):
    command=['sacct','-X','-u',os.environ['USER'],'--starttime=2026-09-01','--parsable2','--noheader',
             '--format=JobIDRaw,JobName%100,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End']
    result=subprocess.run(command,capture_output=True,text=True,check=True);path=out/'all_user_sacct.psv';path.write_text(result.stdout)
    return dict(parse_accounting(result.stdout,exclude_job,allow_active),file=str(path),sha256=sha(path),command=command)


def project_concurrency(out):
    command=['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%P|%T|%b']
    result=subprocess.run(command,capture_output=True,text=True,check=True);path=out/'live_project_queue.psv';path.write_text(result.stdout)
    count=0;own=0
    for line in result.stdout.splitlines():
        if not line.strip():continue
        job,name,partition,state,gres=[x.strip() for x in line.split('|')]
        if partition!='gpu' or state not in ('RUNNING','COMPLETING') or name not in (JOB_NAME,'identity_join_learned_main'):continue
        entries=[x for x in gres.split(',') if x.startswith(('gpu:','gres/gpu:'))];need(entries,'Unknown active project GPU allocation')
        n=sum(int(x.rsplit(':',1)[1].split('(')[0]) for x in entries);count+=n
        if name==JOB_NAME:own+=n
    need(count<=4 and own<=2,'Combined current fits/memory software GPU limit exceeded')
    return dict(passed=True,active_project_gpus=count,active_memory_gpus=own,file=str(path),sha256=sha(path))


def self_test():
    row='1|'+JOB_NAME+'|gpu|FAILED|1:0|7|gres/gpu=1,gres/gpu:b200=1|2026-09-11T10:00:00|2026-09-11T10:00:07'
    zero='2|'+JOB_NAME+'|gpu|CANCELLED|0:0|0||Unknown|Unknown'
    need(parse_accounting(row+'\n'+zero)['allocated_gpu_seconds']==7,'Failed/zero accounting differs')
    for tres in ('gres/gpu=1','gres/gpu:b200=1'):
        need(parse_accounting(row.replace('gres/gpu=1,gres/gpu:b200=1',tres))['allocated_gpu_seconds']==7,'Generic/typed accounting differs')
    need(parse_accounting(row.replace('|FAILED|','|RUNNING|'),allow_active=True)['reserved_active_gpu_seconds']==600,'Active reservation differs')
    for bad in (row+'\n'+row,row.replace('|7|','|601|'),row.replace('|FAILED|','|RUNNING|'),row.replace('gpu:b200=1','gpu:b200=2')):
        try:parse_accounting(bad)
        except ValueError:pass
        else:raise ValueError('Invalid allocation fixture passed')
    need(2*(3+2*3+2*2)==26 and 2*(3+4*3+2*2)+5==43,'Fixed software call arithmetic differs')
    return dict(passed=True,resource_fixtures=True,inventory=True,no_model_loaded=True)


def initial_states(torch):
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);core=ParallelLocalLearnedSelection(mode='clip')
        zero={k:v.detach().clone() for k,v in core.state_dict().items()}
        with torch.no_grad():core.up.weight.normal_(0,.001);core.selection_weight.normal_(0,.001)
        active={k:v.detach().clone() for k,v in core.state_dict().items()}
    need(sum(v.numel() for v in zero.values())==1041697 and not bool(zero['up.weight'].any())
         and not bool(zero['selection_weight'].any()) and bool(active['up.weight'].ne(0).any())
         and bool(active['selection_weight'].ne(0).any()) and all(torch.equal(zero[k],active[k])
         for k in zero if k not in ('up.weight','selection_weight')), 'Private unfitted initialization differs')
    return dict(zero=zero,active=active)


def installed_sources():
    import bitsandbytes.autograd._functions as bnb_functions
    import transformers.masking_utils as masking
    import transformers.cache_utils as cache
    import torch.serialization as serialization
    return {str(Path(inspect.getfile(m)).resolve()):sha(inspect.getfile(m)) for m in (bnb_functions,masking,cache,serialization)}


def identities(bindings):
    q,prior=selection.backbone_identity();bindings.update(prior)
    bind(COSMOS_PROOF,bindings);proof=read(COSMOS_PROOF);need(proof['passed'] and proof['completed'],'Complete Cosmos native proof required')
    bind(proof['analysis_file'],bindings,proof['analysis_sha256']);analysis=read(proof['analysis_file']);old_identity=analysis['native_identity']
    need(analysis['passed'] and cosmos.mirror_identity()==old_identity['mirror'],'Cosmos mirror proof differs')
    c=dict(schema_version=1,model=old_identity['mirror'],runtime=old_identity['runtime'],processor=old_identity['processor'],
        native_api=old_identity['native_api'],native_dtypes=dict(norm='torch.float16',lm_head='torch.float16'),
        **{k:old_identity[k] for k in ('norm_weight','head_weight','norm_source_sha256','rms_norm_eps')})
    return dict(qwen=q,cosmos=c)


def check(args,out,frozen):
    import torch,transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4);bindings={};unit=self_test()
    source_proof(TRAIN_PROOF,bindings,'native_learned_memory_selected_loss_cpu_check',9,TRAIN_PROOF_SHA)
    source_proof(args.controller_proof,bindings,'learned_memory_controller_cpu_fixtures',8)
    bind(TRAIN,bindings,TRAIN_SHA);training=read(TRAIN)
    proof=read(TRAIN.parent/'summary.json');need(proof['passed'] and proof['completed'] and proof['plan_sha256']==sha(TRAIN),'Training-only case plan incomplete')
    bind(TRAIN.parent/'summary.json',bindings);bind(training['timing_cases_file'],bindings,training['timing_cases_sha256'])
    timing=read(training['timing_cases_file']);selected=[c for c in timing['cases'] if c['variant']==0 and c['n_frames'] in (16,64)]
    need([c['n_frames'] for c in selected]==[16,64] and timing['derivation']['no_dev_or_test_scene_used'],'Wrong training-only software cases')
    ids=identities(bindings);bundles={};models={}
    for key in MODELS:
        path=MODEL if key=='qwen' else cosmos.MODEL
        processor=AutoProcessor.from_pretrained(str(path),trust_remote_code=True,use_fast=False,local_files_only=True)
        owner,fn,api=backend.native_api(processor) if key=='qwen' else cosmos.native_api(processor)
        identity=ids[key]
        need(api==identity['native_api'] and fingerprint(processor,str(transformers.__version__))==identity['processor']
             and backend.runtime_identity()==identity['runtime'],'CPU model-specific processor/API/runtime differs')
        tokens=[]
        for text in POLICY['forced_text']:
            encoded=processor.tokenizer(text,add_special_tokens=False)['input_ids']
            need(len(encoded)==1 and encoded[0] not in processor.tokenizer.all_special_ids,'Fixed continuation must be one ordinary token per string');tokens+=encoded
        cases=[]
        for item in selected:
            sample=item['sample'];need(set(sample)=={'sid','n_frames','question','image_files'},'Unrestricted software input view')
            for image in sample['image_files']:bind(image['path'],bindings,image['sha256'])
            base=runtime.prepare_scene(processor,sample,verify_processor_parity=True);variants=[]
            for t in range(3):
                bundle=runtime.append_observed_prefix(base,tokens[:t]);layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle)
                case_id=f'{key}_N{item["n_frames"]}';bundles[f'{case_id}_t{t}']=bundle
                variants.append(dict(metadata=bundle['metadata'],layout=layout['metadata']))
            cases.append(dict(case_id=case_id,n_frames=item['n_frames'],sample=sample,parent_sid=item['parent_sid'],
                variant=0,prompt_width=base['metadata']['original_prompt_width'],variants=variants))
        models[key]=dict(path=str(path),native_identity=identity,native_identity_sha256=native.object_sha(identity),forced_ids=tokens,cases=cases)
    destination=DATA/out.name;destination.mkdir(parents=True,exist_ok=False);weights=CKPT/out.name;weights.mkdir(parents=True,exist_ok=False)
    states=initial_states(torch);initial=weights/'initial.pt';prepared=destination/'prepared.pt'
    torch.save(states,initial);torch.save(dict(schema_version=1,bundles=bundles),prepared)
    plan=dict(schema_version=1,protocol=PROTOCOL,policy=POLICY,source_sha256=frozen,artifact_bindings=bindings,
        models=models,prepared_file=str(prepared),prepared_sha256=sha(prepared),initial_file=str(initial),initial_sha256=sha(initial),
        initial_state_identity=v11.old.state_identity(states),installed_source_sha256=installed_sources(),
        precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision()),tests=unit,no_model_loaded=True)
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    return dict(passed=True,completed=True,phase='check',protocol=PROTOCOL,source_sha256=frozen,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),tests=unit,no_model_loaded=True)


def verify_plan(path):
    path=Path(path).resolve();plan=read(path);need(sha(path)==path.with_suffix('.sha256').read_text().strip()
        and plan['policy']==POLICY and plan['source_sha256']==sources(),'Frozen software plan/source differs')
    for n,h in plan['source_sha256'].items():need(sha(path.parent/'source'/n.replace('/','_'))==h,'Source snapshot differs')
    for p,h in plan['artifact_bindings'].items():need(sha(p)==h,'Bound CPU input/proof differs')
    for p,h in plan['installed_source_sha256'].items():need(sha(p)==h,'Installed replay/backend source differs')
    for k in ('prepared','initial'):need(sha(plan[k+'_file'])==plan[k+'_sha256'],'Prepared software packet differs')
    for key,entry in plan['models'].items():
        need(native.object_sha(entry['native_identity'])==entry['native_identity_sha256'],'Model identity digest differs')
        for p,h in entry['native_identity']['native_api']['source_sha256'].items():need(sha(p)==h,'Native model API source differs')
    proof=read(path.parent/'summary.json');need(proof['passed'] and proof['completed'] and proof['plan_sha256']==sha(path),'Complete CPU source freeze required')
    return plan


def live_identity(torch,loaded,entry,plan):
    norm=runtime.memory_contract(loaded.model,None,None,None);expected=entry['native_identity'];key=entry['key']
    api=backend.native_api(loaded.processor)[2] if key=='qwen' else cosmos.native_api(loaded.processor)[2]
    actual_model=model_metadata() if key=='qwen' else cosmos.mirror_identity()
    need(actual_model==expected['model'] and api==expected['native_api'] and backend.runtime_identity()==expected['runtime']
         and fingerprint(loaded.processor,expected['runtime']['transformers_version'])==expected['processor'],'Loaded model/processor/API differs')
    need(native.tensor_info(norm.weight)==expected['norm_weight'] and native.tensor_info(loaded.model.lm_head.weight)==expected['head_weight']
         and sha(inspect.getfile(type(norm)))==expected['norm_source_sha256'] and float(norm.variance_epsilon)==expected['rms_norm_eps'],
         'Actual frozen FP16 native head/norm identity differs')
    quant=loaded.model.config.quantization_config;quant=quant.to_dict() if hasattr(quant,'to_dict') else dict(quant)
    need(quant['load_in_4bit'] and quant['bnb_4bit_use_double_quant'] and quant['bnb_4bit_quant_type']=='nf4'
         and str(quant['bnb_4bit_compute_dtype']).removeprefix('torch.')=='bfloat16','NF4 double/BF16 execution differs')
    precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   float32_matmul_precision=torch.get_float32_matmul_precision())
    need(precision==plan['precision'],'Native precision flags changed')
    return json.loads(json.dumps(dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=str(torch.version.cuda),quantization=quant,precision=precision)))


def tensor_equal(torch,a,b):
    return a is None and b is None or isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a,b)


def kv_evidence(cache,first_write,old_length):
    """Hash actual full/local/prompt/retained KV views on all28 layers.

    Full tensors are archived separately only for the three N16 intervention
    forks; these per-call digests allow independent equality joins without
    repeatedly storing tens of GB of identical lower-layer cache entries.
    """
    result=[]
    for i,layer in enumerate(cache.layers):
        row=dict(layer=i)
        for name,value in (('key',layer.keys),('value',layer.values)):
            value=v11.cpu_copy(value);row[name]=native.tensor_info(value)
            row['local_'+name]=native.tensor_info(value[:-1])
            row['prompt_'+name]=native.tensor_info(value[-1,:,:first_write])
            row['first_'+name]=native.tensor_info(value[-1,:,first_write:first_write+1])
            row['retained_'+name]=None if not old_length else native.tensor_info(value[:,:,:old_length])
        result.append(row)
    need(len(result)==28,'Require every native KV layer');return result


def kv_compare(actual,baseline,placement,condition):
    need(len(actual)==len(baseline)==28,'KV comparison coverage differs')
    whole=[];local=[];prompt=[];first=[];first_fields=[]
    for a,b in zip(actual,baseline):
        whole.append(all(a[k]==b[k] for k in ('key','value')))
        local.append(all(a['local_'+k]==b['local_'+k] for k in ('key','value')))
        prompt.append(all(a['prompt_'+k]==b['prompt_'+k] for k in ('key','value')))
        fields={k:a['first_'+k]==b['first_'+k] for k in ('key','value')};first_fields.append(fields);first.append(all(fields.values()))
    need(all(whole[:-1]) and all(local) and all(prompt),'Write changed lower/local/earlier-prompt KV')
    need(all(whole) if condition=='zero' or placement=='post_last' else not any(first_fields[-1].values()),
         'KV write-site or zero-identity contrast differs')
    return dict(passed=True,per_layer_all_rows_exact=whole,per_layer_local_rows_exact=local,
        per_layer_prior_prompt_exact=prompt,per_layer_first_global_exact=first,first_global_field_exact=first_fields)


def functional_capture(torch,raw,weights):
    import torch.nn.functional as F
    from gnnformer.parallel_local_learned_selection import selection_weights
    c=raw['fusion'];indices=c['query_indices'];device=weights['up.weight'].device
    h=raw['common_penultimate_hidden'][:,indices].to(device);local,global_=h[:-1],h[-1]
    def rms(x):x=x.float();return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        q=F.linear(rms(global_),weights['query.weight'])
        p=torch.tanh(F.linear(rms(local),weights['local.weight'])+q.unsqueeze(0)+weights['local_bias'])
        s=F.linear(p,weights['selection_weight'].unsqueeze(0)).squeeze(-1)+weights['selection_bias']
        g=selection_weights(s,'clip');m=g.unsqueeze(-1)*p;z=m.sum(0)
        t=F.linear(z,weights['aggregate_projection.weight'],weights['aggregate_projection.bias'])+q
        delta=F.linear(F.silu(t),weights['up.weight'])
    expected=dict(query=q,payload=p,scores=s,gates=g,messages=m,aggregate=z,preactivation=t)
    need(all(torch.equal(v.cpu(),c[k]) for k,v in expected.items()),'Actual core intermediates differ from same-device functional algebra')
    original=c['delta'] if raw['intervention'] is None else raw['intervention']['original_delta']
    need(torch.equal(delta.cpu(),original),'Ordinary core delta differs from functional algebra')
    return dict(passed=True,core_forward_recomputed=False,ordinary_formula_exact=True,
                applied_override=raw['intervention'] is not None)


@contextmanager
def override_delta(core,applied,holder):
    """Explicit profile-only do(delta); no model/core/controller monkeypatch."""
    import torch
    def hook(module,args,output):
        original,values=output;chosen=applied(args,original) if callable(applied) else applied.to(original.device)
        need(chosen.shape==original.shape and chosen.dtype==torch.float32 and bool(torch.isfinite(chosen).all()),'Invalid explicit delta intervention')
        holder.clear();holder.update(original_delta=v11.cpu_copy(original),applied_delta=v11.cpu_copy(chosen),
                                    intervention_only=True,ordinary_core_formula_not_claimed_for_applied_delta=True)
        return chosen,dict(values,delta=chosen)
    handle=core.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def execute(model,inputs,*,cached,expected_positions,expected_mask,controller,data,label,records,
            context,first_write,override=None,weights=None):
    """One fixed-token call; archive partial raw states before numerical gates."""
    import torch
    item=dict(inputs);item.pop('use_cache',None);previous=item.get('past_key_values')
    old_length=0 if previous is None else previous.get_seq_length()
    before=None if not old_length else v11.snapshot_kv(previous);started=time.perf_counter();normalized={}
    def after_norm(module,args,output):normalized['value']=v11.cpu_copy(output[:,-1:,:])
    with v11.Capture(model) as observed:
        handle=model.model.language_model.norm.register_forward_hook(after_norm)
        try:
            with torch.inference_mode():output=model(**item,use_cache=cached,logits_to_keep=1)
            torch.cuda.synchronize()
        finally:handle.remove()
    raw=observed.value;cache=output.past_key_values
    raw.update(schema_version=1,label=label,context=context,input_ids=v11.cpu_copy(item['input_ids']),
        native_logits=v11.cpu_copy(output.logits),native_normalized=normalized.get('value'),
        fusion=None if controller is None else controller.export_last_capture(cpu=True),
        intervention=None if override is None else v11.cpu_copy(override),use_cache=cached,old_cache_length=old_length,
        expected_positions=v11.cpu_copy(expected_positions),expected_mask=v11.cpu_copy(expected_mask),
        visual_input_identity={k:native.tensor_info(v) for k,v in item.items() if k in ('pixel_values','image_grid_thw') and v is not None})
    partial=data/(label+'_partial.pt');torch.save(raw,partial)
    record=dict(label=label,context=context,partial_file=str(partial),partial_sha256=sha(partial),passed=False,
        model_calls=1,visual_calls=int(item.get('pixel_values') is not None),old_cache_length=old_length,use_cache=cached)
    records.append(record)
    need(torch.equal(raw['position_ids'],raw['expected_positions']) and torch.equal(raw['attention_mask'],raw['expected_mask']),
         'Actual native positions/padding differ')
    need(raw['common_penultimate_hidden'].dtype==raw['actual_norm_input'].dtype==raw['native_logits'].dtype==torch.float16
         and raw['native_logits'].shape[:2]==(raw['input_ids'].shape[0],1) and bool(torch.isfinite(raw['native_logits']).all()),'Actual native FP16 output differs')
    mask=v11.old.mask_check(torch,raw['last_block_mask'],raw['attention_mask'],raw['cache_position'])
    if cached:
        need(cache is not None and cache.get_seq_length()==expected_mask.shape[1],'Actual KV length differs')
        retained=None if before is None else v11.old.mixed.prefix_preserved(torch,cache,before,old_length)
        need(retained is None or len(retained)==28 and all(retained),'Previous native KV was modified')
        raw['kv']=kv_evidence(cache,first_write,old_length)
    else:
        need(cache is None and previous is None,'Uncached full reference used KV');retained=None;raw['kv']=None
    if controller is not None:
        controller.assert_complete();v11.verify_write(raw,context['placement'],list(controller.query_indices),list(controller.stream_positions))
        formula=functional_capture(torch,raw,weights)
    else:formula=None
    hidden=raw['actual_norm_input'][:,-1:,:].to(model.device)
    with torch.inference_mode():normed=model.model.language_model.norm.forward(hidden);replay=model.lm_head.forward(normed)
    torch.cuda.synchronize();raw.update(replay_logits=v11.cpu_copy(replay),replay_normalized=v11.cpu_copy(normed))
    metrics=[v11.old.metric(torch,raw['replay_logits'][i,0],raw['native_logits'][i,0]) for i in range(hidden.shape[0])]
    path=data/(label+'.pt');torch.save(raw,path)
    record.update(path=str(path),sha256=sha(path),head_replay=metrics,normalized_exact=torch.equal(raw['replay_normalized'],raw['native_normalized']),
        mask_audit=mask,previous_prefix_exact_by_layer=retained,kv=raw['kv'],formula=formula,
        query_tokens=item['input_ids'].shape[1],key_tokens=expected_mask.shape[1],seconds=time.perf_counter()-started,passed=all(m['numerical_rule_passed'] for m in metrics))
    save(data/f'after_{len(records):03d}.json',records)
    need(record['passed'],'Captured native-shaped head replay failed; raw retained')
    return cache,raw


def copy_cache(cache):
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(tuple(v11.snapshot_kv(cache)))


def save_cache(cache,path):
    import torch
    values=v11.cpu_copy(v11.snapshot_kv(cache));torch.save(values,path)
    return dict(file=str(path),sha256=sha(path),layers=28)


def persistence(model,core,bundles,case,baseline,baseline_caches,data,records,device):
    import torch
    from gnnformer.parallel_local_learned_memory import ParallelLocalLearnedMemory
    width=case['prompt_width'];tokens=case['forced_ids'];layers=model.model.language_model.layers;norm=model.model.language_model.norm
    results=[];artifacts={}
    # These forks were captured at the exact active prefill, before later appends.
    for placement in PLACEMENTS:
        base_raw=baseline[placement][0];base_next=baseline[placement][1];reference=baseline_caches[placement]
        saved_rope=v11.cpu_copy(model.model.rope_deltas);holder={};cache=None
        def perturbed(args,original):
            common=args[1].float();direction=torch.sin(torch.arange(common.shape[-1],device=common.device,dtype=torch.float32))
            direction=direction*torch.rsqrt(direction.square().mean());perturb=.1*common.square().mean(-1,keepdim=True).sqrt()*direction
            return original+perturb
        control=ParallelLocalLearnedMemory(layers[-2],norm,core,n_local_rows=16,write_location=placement,
            query_indices=[width-1],stream_positions=[width-1],capture=True,selection_mode='clip')
        try:
            with control:
                inputs,pos,_=v11.cached_input(model,bundles[0],None,0,tokens,device)
                with override_delta(core,perturbed,holder):
                    cache,raw=execute(model,inputs,cached=True,expected_positions=pos,expected_mask=bundles[0]['inputs']['attention_mask'],
                        controller=control,data=data,label=f'N16_{placement}_perturb_prefill',records=records,
                        context=dict(n_frames=16,condition='intervention',placement=placement,step=0,kind='perturb_prefill'),first_write=width-1,override=holder,weights=core.state_dict())
                need(torch.equal(raw['common_penultimate_hidden'],base_raw['common_penultimate_hidden']),'Intervention changed common lower read')
                altered=copy_cache(cache);restore=copy_cache(cache)
                if placement=='pre_last':
                    artifacts['baseline']=save_cache(reference,data/'persistence_baseline.pt')
                    artifacts['altered']=save_cache(altered,data/'persistence_altered.pt')
                    with torch.no_grad():
                        for name in ('keys','values'):
                            getattr(restore.layers[27],name)[-1,:,width-1:width]=getattr(reference.layers[27],name)[-1,:,width-1:width]
                    artifacts['restored']=save_cache(restore,data/'persistence_restored.pt')
                    need(all(torch.equal(a,b) for paira,pairb in zip(v11.snapshot_kv(restore),v11.snapshot_kv(reference)) for a,b in zip(paira,pairb)),
                         'Restoring only the first layer27 global slot did not recover the full baseline cache')
                else:
                    need(all(torch.equal(a,b) for paira,pairb in zip(v11.snapshot_kv(altered),v11.snapshot_kv(reference)) for a,b in zip(paira,pairb)),
                         'Post-last perturbation changed native KV')
                outcomes=[]
                variants=[('altered',altered)]+([('restored',restore)] if placement=='pre_last' else [])
                for tag,probe_cache in variants:
                    control.configure_queries([0],[width]);inputs,pos,_=v11.cached_input(model,bundles[1],probe_cache,1,tokens,device)
                    with override_delta(core,base_next['fusion']['delta'],holder):
                        _,result=execute(model,inputs,cached=True,expected_positions=pos,expected_mask=bundles[1]['inputs']['attention_mask'],
                            controller=control,data=data,label=f'N16_{placement}_{tag}_next',records=records,
                            context=dict(n_frames=16,condition='intervention',placement=placement,step=1,kind=tag+'_next'),first_write=width-1,override=holder,weights=core.state_dict())
                    same=torch.equal(result['native_logits'],base_next['native_logits'])
                    need(torch.equal(result['fusion']['delta'],base_next['fusion']['delta'])
                         and torch.equal(result['common_penultimate_hidden'],base_next['common_penultimate_hidden']),
                         'Later applied write or lower read differs across the fixed-token intervention')
                    need(same if tag=='restored' or placement=='post_last' else not same,'Memory intervention/restore did not isolate the registered causal route')
                    outcomes.append(dict(label=result['label'],kind=tag,all_row_logits_equal_baseline=same,
                        metric=v11.old.metric(torch,result['native_logits'][-1,0],base_next['native_logits'][-1,0]),passed=True))
                results.append(dict(placement=placement,prefill_label=raw['label'],baseline_prefill_label=base_raw['label'],
                    baseline_next_label=base_next['label'],outcomes=outcomes,passed=True))
        finally:model.model.rope_deltas=None if saved_rope is None else saved_rope.to(device)
    save(data/'persistence.json',dict(results=results,cache_artifacts=artifacts,first_write=width-1))
    return dict(results=results,cache_artifacts=artifacts,first_write=width-1,file=str(data/'persistence.json'),sha256=sha(data/'persistence.json'))


def gradient_replays(model,records,data,target):
    import torch
    from scripts.native_learned_memory_training import replay_learned_memory
    language=model.model.language_model;results=[]
    for placement in PLACEMENTS:
        source=next(r for r in records if r['context']==dict(n_frames=16,condition='active',placement=placement,step=2,kind='full'))
        raw=torch.load(source['path'],map_location='cpu',weights_only=True)
        # CPU archive loads are ordinary tensors, then cloned outside inference.
        h=raw['common_penultimate_hidden'].to(model.device).clone();positions=raw['fusion']['query_indices']
        need(not torch.is_inference(h),'Autograd replay must not receive inference tensors')
        writes=[[positions[1]]]+[[] for _ in range(h.shape[0]-2)]+[positions]
        losses=[[] for _ in range(h.shape[0]-1)]+[[positions[1]]]
        delta=torch.cat((torch.zeros((1,h.shape[-1]),device=model.device),raw['fusion']['delta'].to(model.device)),0).requires_grad_()
        tick=time.perf_counter()
        value=replay_learned_memory(language.layers[-1],language.norm,model.lm_head,language.rotary_emb,
            hidden_states=h,input_ids=raw['input_ids'].to(model.device),attention_mask=raw['attention_mask'].to(model.device),
            position_ids=raw['position_ids'].to(model.device),write_indices=writes,loss_indices=losses,deltas=delta,placement=placement)
        loss=torch.nn.functional.cross_entropy(value['logits'].float(),torch.tensor([target],device=model.device))
        gradient=torch.autograd.grad(loss,delta)[0]
        with torch.inference_mode():
            normalized=language.norm.forward(value['final_norm_input'][:,-1:,:].detach())
            replay=model.lm_head.forward(normalized)
        torch.cuda.synchronize()
        evidence=dict(schema_version=1,placement=placement,source=source,write_indices=writes,loss_indices=losses,target_token_id=target,
            deltas=v11.cpu_copy(delta),gradient=v11.cpu_copy(gradient),loss=float(loss.detach()),loss_logits=v11.cpu_copy(value['logits']),
            native_shape_logits=v11.cpu_copy(replay),native_shape_normalized=v11.cpu_copy(normalized),
            block_input=v11.cpu_copy(value['block_input']),final_norm_input=v11.cpu_copy(value['final_norm_input']),
            attention_mask=v11.cpu_copy(value['attention_mask']),position_embeddings=v11.cpu_copy(value['position_embeddings']),
            cache_position=v11.cpu_copy(value['cache_position']),layout=value['layout'],loss_to_write_indices=value['loss_to_write_indices'].tolist())
        path=data/('gradient_'+placement+'.pt');torch.save(evidence,path)
        metrics=[v11.old.metric(torch,evidence['native_shape_logits'][i,0],raw['native_logits'][i,0]) for i in range(h.shape[0])]
        checks=dict(finite=bool(torch.isfinite(gradient).all()),local_zero=bool((gradient[0]==0).all()),
            earlier_nonzero=bool(gradient[1].ne(0).any()),current_nonzero=bool(gradient[2].ne(0).any()),future_zero=bool((gradient[3]==0).all()),
            block_input_exact=torch.equal(evidence['block_input'],raw['actual_last_block_input']),
            mask_exact=tensor_equal(torch,evidence['attention_mask'],raw['last_block_mask']),
            positions_exact=torch.equal(evidence['cache_position'],raw['cache_position']),
            rotary_exact=all(torch.equal(a,b) for a,b in zip(evidence['position_embeddings'],raw['position_embeddings'])))
        item=dict(placement=placement,file=str(path),sha256=sha(path),source_label=source['label'],checks=checks,
            head_replay=metrics,seconds=time.perf_counter()-tick,gradient_norms=[float(x.double().norm()) for x in evidence['gradient']],
            block_calls=1,selected_loss_heads=1,native_shape_heads=1)
        results.append(item);save(data/('gradient_'+placement+'.json'),item)
        need(checks['earlier_nonzero']==(placement=='pre_last') and all(v for k,v in checks.items() if k!='earlier_nonzero')
             and all(m['numerical_rule_passed'] for m in metrics),'Actual quantized causal-gradient or same-state replay failed; raw retained')
        need(not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen native parameters received gradients')
        del value,delta,gradient,h,raw
    return results


def natural_runs(model,processor,core,bundle,entry,data):
    import torch
    results=[]
    for placement in PLACEMENTS:
        tick=time.perf_counter();rope=v11.cpu_copy(getattr(model.model,'rope_deltas',None));pointer={};partials=[]
        def observe_control(value):pointer['control']=value
        def retain_call(module,args,output):
            control=pointer.get('control')
            cap=None if control is None else control.export_last_capture(cpu=True)
            raw=dict(native_logits=v11.cpu_copy(output.logits),capture=cap,step=len(partials))
            path=data/f'natural_{placement}_partial_{len(partials)}.pt';torch.save(raw,path)
            partials.append(dict(file=str(path),sha256=sha(path)))
        hook=model.register_forward_hook(retain_call)
        try:
            result=runtime.generate_native(model,processor,core,bundle,selection_mode='clip',write_location=placement,
                native_identity_sha256=entry['native_identity_sha256'],max_new_tokens=8,capture=True,full_vectors=True,
                controller_observer=observe_control)
        finally:hook.remove();save(data/('natural_'+placement+'_partials.json'),partials)
        path=data/('natural_'+placement+'.pt');torch.save(result,path)
        replays=[];formulas=[];steps=len(result['generated_ids'])
        need(1<=steps<=8 and len(result['captures'])==len(partials)==steps and result['raw_logits'].shape[0]==steps,'Natural captured history is incomplete')
        for t,cap in enumerate(result['captures']):
            actual=torch.load(partials[t]['file'],map_location='cpu',weights_only=True)
            formulas.append(functional_capture(torch,dict(common_penultimate_hidden=cap['common_query_hidden'],
                fusion=dict(cap,query_indices=[0]),intervention=None),core.state_dict()))
            with torch.inference_mode():
                hidden=cap['fused_query_hidden'].to(model.device)
                normed=model.model.language_model.norm.forward(hidden);logits=model.lm_head.forward(normed)
            need(hidden.shape==(17,1,3584),'Natural replay must retain full native batch shape')
            metric=v11.old.metric(torch,logits[-1,0].cpu(),result['raw_logits'][t].half())
            rows=[v11.old.metric(torch,logits[i,0].cpu(),actual['native_logits'][i,0]) for i in range(17)]
            replays.append(dict(position=t,metric=metric,all_rows=rows,logits=v11.cpu_copy(logits),normalized=v11.cpu_copy(normed)))
        replay_path=data/('natural_'+placement+'_heads.pt');torch.save(replays,replay_path);torch.cuda.synchronize()
        restored=tensor_equal(torch,rope,v11.cpu_copy(getattr(model.model,'rope_deltas',None)))
        item=dict(placement=placement,file=str(path),sha256=sha(path),replay_file=str(replay_path),replay_sha256=sha(replay_path),
            partials=partials,formulas=formulas,tokens=steps,counters=result['counters'],rope_restored=restored,
            head_replay=[r['metric'] for r in replays],all_row_head_replay=[r['all_rows'] for r in replays],seconds=time.perf_counter()-tick)
        results.append(item);save(data/('natural_'+placement+'.json'),item)
        need(restored and all(r['metric']['numerical_rule_passed'] and all(m['numerical_rule_passed'] for m in r['all_rows']) for r in replays),
             'Natural runtime/head/rope check failed; raw retained')
    return results


def run(args,out,frozen):
    import torch
    from gnnformer.runtime import load_runtime,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_learned_selection import ParallelLocalLearnedSelection
    from gnnformer.parallel_local_learned_memory import ParallelLocalLearnedMemory
    torch.set_num_threads(4);begin=time.perf_counter();plan=verify_plan(args.plan);key=args.model
    entry=dict(plan['models'][key],key=key);reserve=accounting(out,os.environ['SLURM_JOB_ID'],allow_active=True)
    need(reserve['allocated_gpu_seconds']+reserve['reserved_active_gpu_seconds']+600<=1200
         and len(reserve['active_jobs'])<2,'A full new600-second reservation exceeds the campaign cap')
    reserve.update(reserved_this_job=600);save(out/'resource_reservation.json',reserve);concurrency=project_concurrency(out)
    need(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA B200','One actual B200 is required')
    packet=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True);states=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(packet['schema_version']==1 and v11.old.state_identity(states)==plan['initial_state_identity'],'CPU software packet changed')
    tick=time.perf_counter();loaded=load_runtime(entry['path'],use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model.eval().requires_grad_(False);hardware=live_identity(torch,loaded,entry,plan);torch.cuda.synchronize();load_seconds=time.perf_counter()-tick
    need(hardware['gpu']=='NVIDIA B200' and hardware['capability']==[10,0],'Actual registered GPU backend differs')
    core=ParallelLocalLearnedSelection(mode='clip').to(loaded.device).eval().requires_grad_(False)
    layers=model.model.language_model.layers;norm=model.model.language_model.norm;versions={n:p._version for n,p in model.named_parameters()}
    data=DATA/out.name;data.mkdir(parents=True,exist_ok=False);counts=dict(model=0,visual=0,last_block=0,head=0,core=0)
    records=[];zeros=[];writes=[];kv=[];comparisons=[];persistence_result=None;gradients=[];naturals=[];failure=None
    def count(name):
        def hook(*_):counts[name]+=1
        return hook
    handles=[model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('visual')),
        layers[-1].register_forward_pre_hook(count('last_block')),model.lm_head.register_forward_pre_hook(count('head')),
        core.register_forward_pre_hook(count('core'))]
    original_rope=v11.cpu_copy(getattr(model.model,'rope_deltas',None));torch.cuda.reset_peak_memory_stats()
    try:
        for case in entry['cases']:
            n=case['n_frames'];width=case['prompt_width'];tokens=entry['forced_ids'];bundles=[packet['bundles'][case['case_id']+f'_t{t}'] for t in range(3)]
            for t,b in enumerate(bundles):
                runtime.validate_bundle(b);layout=native.audit_layout(get_rope_index_fn(model),b)
                need(b['metadata']==case['variants'][t]['metadata'] and layout['metadata']==case['variants'][t]['layout']
                     and {k:native.tensor_info(v) for k,v in b['inputs'].items()}==b['metadata']['input_identity'],'Actual prepared input/layout identity differs')
            baseline=[];baseline_cache=[];cache=None
            for t,b in enumerate(bundles):
                inputs,pos,_=v11.cached_input(model,b,cache,t,tokens,loaded.device)
                cache,raw=execute(model,inputs,cached=True,expected_positions=pos,expected_mask=b['inputs']['attention_mask'],controller=None,
                    data=data,label=f'N{n}_bare_cached{t}',records=records,context=dict(n_frames=n,condition='bare',placement=None,step=t,kind='cached'),first_write=width-1)
                baseline.append(raw);baseline_cache.append(raw['kv'])
            del cache
            active={};active_cache={}
            for condition in ('zero','active'):
                core.load_state_dict(states[condition],strict=True)
                need(v11.old.state_identity({condition:core.state_dict()})[condition]==plan['initial_state_identity'][condition],'Unfitted state restoration differs')
                for placement in PLACEMENTS:
                    cache=None;active.setdefault(placement,[])
                    controller=ParallelLocalLearnedMemory(layers[-2],norm,core,n_local_rows=n,write_location=placement,
                        query_indices=[width-1],stream_positions=[width-1],capture=True,selection_mode='clip')
                    with controller:
                        for t,b in enumerate(bundles):
                            q=[width-1] if t==0 else [0];s=[width-1+t];controller.configure_queries(q,s)
                            inputs,pos,layout=v11.cached_input(model,b,cache,t,tokens,loaded.device)
                            label=f'N{n}_{condition}_{placement}_cached{t}';context=dict(n_frames=n,condition=condition,placement=placement,step=t,kind='cached')
                            cache,raw=execute(model,inputs,cached=True,expected_positions=pos,expected_mask=b['inputs']['attention_mask'],
                                controller=controller,data=data,label=label,records=records,context=context,first_write=width-1,weights=core.state_dict())
                            need(torch.equal(raw['common_penultimate_hidden'],baseline[t]['common_penultimate_hidden']),'Write changed its common lower read')
                            writes.append(dict(label=label,common_read_exact=True,**v11.verify_write(raw,placement,q,s)))
                            kv.append(dict(label=label,baseline_label=baseline[t]['label'],**kv_compare(raw['kv'],baseline_cache[t],placement,condition)))
                            if condition=='zero':
                                need(torch.equal(raw['native_logits'],baseline[t]['native_logits']) and not bool(raw['fusion']['delta'].any()),'Zero U changed native output')
                                zeros.append(dict(label=label,baseline_label=baseline[t]['label'],all_row_logits_exact=True,delta_zero=True,passed=True))
                            else:
                                need(bool(raw['fusion']['delta'].ne(0).any()) and bool(raw['fusion']['scores'].ne(.5).any()),'Fixed active U/selector did not exercise nonconstant native selection')
                                if n==16:active[placement].append(raw)
                                if n==16 and t==0:active_cache[placement]=copy_cache(cache)
                                if t:
                                    history=list(range(width-1,width+t));controller.configure_queries(history,history);rope=v11.cpu_copy(model.model.rope_deltas)
                                    try:
                                        _,full=execute(model,move_to_device(b['inputs'],loaded.device),cached=False,expected_positions=layout['position_ids'],
                                            expected_mask=b['inputs']['attention_mask'],controller=controller,data=data,
                                            label=f'N{n}_active_{placement}_full{t}',records=records,
                                            context=dict(n_frames=n,condition='active',placement=placement,step=t,kind='full'),first_write=width-1,weights=core.state_dict())
                                    finally:model.model.rope_deltas=rope.to(loaded.device)
                                    writes.append(dict(label=full['label'],common_read_exact=None,**v11.verify_write(full,placement,history,history)))
                                    comparisons.extend(dict(n_frames=n,placement=placement,step=t,row=i,cached_label=label,full_label=full['label'],
                                        binding=False,**v11.old.metric(torch,raw['native_logits'][i,0],full['native_logits'][i,0])) for i in range(n+1))
                        controller.assert_complete();need(controller.core_calls==controller.calls==(3 if condition=='zero' else 5),'Core/read/write call mismatch')
            if n==16:
                persistence_result=persistence(model,core,bundles,dict(case,forced_ids=tokens),active,active_cache,data,records,loaded.device)
            del baseline,baseline_cache,active,active_cache
        need(counts==dict(model=43,visual=20,last_block=43,head=43,core=37) and len(records)==43,'Fixed call inventory differs')
        gradients=gradient_replays(model,records,data,entry['forced_ids'][1])
        n16=next(c for c in entry['cases'] if c['n_frames']==16);base=packet['bundles'][n16['case_id']+'_t0']
        naturals=natural_runs(model,loaded.processor,core,base,entry,data);natural_tokens=sum(r['tokens'] for r in naturals)
        need(counts==dict(model=43+natural_tokens,visual=22,last_block=45+natural_tokens,head=45+natural_tokens,core=37+natural_tokens),
             'Actual native/block/head/core call counters differ')
        need(len(zeros)==12 and len(kv)==24 and len(writes)==32 and len(comparisons)==328,'Fixed structural/description coverage differs')
        need(versions=={n:p._version for n,p in model.named_parameters()} and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Native parameter state changed')
        need(v11.old.state_identity({'active':core.state_dict()})['active']==plan['initial_state_identity']['active'],'Unfitted core changed during software audit')
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        for h in handles:h.remove()
        model.model.rope_deltas=None if original_rope is None else original_rope.to(loaded.device)
        for name,value in dict(forwards=records,zero_checks=zeros,write_checks=writes,kv_checks=kv,comparisons=comparisons,
            persistence=persistence_result,gradients=gradients,naturals=naturals,actual_counts=counts).items():save(out/(name+'.json'),value)
        save(out/'execution.json',dict(completed=failure is None,failure=failure,counts=counts,hardware=hardware,load_seconds=load_seconds,
            elapsed_seconds=time.perf_counter()-begin,source_sha256=frozen,model_key=key,partial_raw_retained=True))
    elapsed=time.perf_counter()-begin;need(elapsed<=600,'Per-model software elapsed cap exceeded')
    files={p.stem:dict(file=str(p),sha256=sha(p)) for p in out.glob('*.json') if p.name!='summary.json'}
    return dict(passed=True,completed=True,phase='profile',protocol=PROTOCOL,model_key=key,source_sha256=frozen,
        plan_file=str(args.plan),plan_sha256=sha(args.plan),native_identity=entry['native_identity'],native_identity_sha256=entry['native_identity_sha256'],
        counts=counts,fixed_calls=43,natural_tokens=natural_tokens,extra_head_calls=47+natural_tokens,hardware=hardware,
        elapsed_seconds=elapsed,load_seconds=load_seconds,maximum_memory_bytes=torch.cuda.max_memory_allocated(),files=files,
        no_fit=True,no_accuracy=True,full_runtime_forward_api_not_executed=True,natural_runtime_generate_api_executed=True)


def load_bound(torch,item):
    need(sha(item['file'])==item['sha256'],'Raw evidence changed');return torch.load(item['file'],map_location='cpu',weights_only=True)


def audit_capture(torch,raw,placement,positions,streams):
    c=raw['fusion'];result=v11.verify_write(raw,placement,positions,streams)
    need(c['selection_mode']=='clip' and torch.equal(c['common_query_hidden'],raw['common_penultimate_hidden'][:,positions])
         and torch.equal(c['final_block_input_queries'],raw['actual_last_block_input'][:,positions]),'Common/write query capture differs')
    need(torch.equal(c['native_delta'],c['delta'].half()) and torch.equal(c['messages'],c['gates'].unsqueeze(-1)*c['payload'])
         and torch.equal(c['gates'],c['scores'].clamp(0,1)),'Current selector/cast/multiplication differs')
    for k in ('query','payload','scores','gates','messages','aggregate','preactivation','delta'):
        need(c[k].dtype==torch.float32 and bool(torch.isfinite(c[k]).all()),'Nonfinite/non-FP32 learned capture')
    intervention=raw['intervention']
    if intervention is not None:need(torch.equal(intervention['applied_delta'],c['delta']) and intervention['intervention_only'], 'Unlabelled or mismatched applied delta')
    return result


def verify_run(path,plan):
    directory=Path(path).resolve();summary=read(directory/'summary.json')
    need(summary['passed'] and summary['completed'] and summary['phase']=='profile' and summary['protocol']==PROTOCOL
         and summary['source_sha256']==plan['source_sha256']==sources() and summary['plan_sha256']==sha(summary['plan_file'])
         and read(summary['plan_file'])==plan,
         'Complete exact-source software run required')
    for n,h in summary['source_sha256'].items():need(sha(directory/'source'/n.replace('/','_'))==h,'GPU source copy differs')
    for item in summary['files'].values():need(sha(item['file'])==item['sha256'],'GPU result index changed')
    entry=plan['models'][summary['model_key']]
    need(summary['native_identity']==entry['native_identity'] and summary['native_identity_sha256']==entry['native_identity_sha256']
         and summary['hardware']['gpu']=='NVIDIA B200' and summary['hardware']['capability']==[10,0], 'Actual model/GPU identity differs')
    return directory,summary


def audit_run(torch,directory,summary,plan,packet):
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(plan['models'][summary['model_key']]['path'],trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=backend.native_api(processor) if summary['model_key']=='qwen' else cosmos.native_api(processor)
    need(api==plan['models'][summary['model_key']]['native_identity']['native_api'],'Independent report native API differs')
    records=read(directory/'forwards.json');need(len(records)==43 and len({r['label'] for r in records})==43,'Fixed native call coverage differs')
    by_label={r['label']:r for r in records};entry=plan['models'][summary['model_key']];cases={c['n_frames']:c for c in entry['cases']}
    head_metrics=[];normalization_exact=[];cache_comparisons=[];writes=0;raw_headers={}
    def raw_for(label):
        record=by_label[label];need(sha(record['path'])==record['sha256'],'Forward raw changed');return torch.load(record['path'],map_location='cpu',weights_only=True)
    expected=set()
    for n in (16,64):
        expected.update(f'N{n}_bare_cached{t}' for t in range(3))
        expected.update(f'N{n}_{c}_{p}_cached{t}' for c in ('zero','active') for p in PLACEMENTS for t in range(3))
        expected.update(f'N{n}_active_{p}_full{t}' for p in PLACEMENTS for t in (1,2))
    expected.update(('N16_pre_last_perturb_prefill','N16_pre_last_altered_next','N16_pre_last_restored_next',
                     'N16_post_last_perturb_prefill','N16_post_last_altered_next'))
    need(set(by_label)==expected,'Prescribed fixed-call labels differ')
    for record in records:
        need(record['passed'] and sha(record['partial_file'])==record['partial_sha256'],'Missing pre-gate raw artifact')
        raw=raw_for(record['label']);context=record['context'];n,t=context['n_frames'],context['step'];case=cases[n]
        bundle=packet['bundles'][case['case_id']+f'_t{t}'];width=case['prompt_width'];mask=bundle['inputs']['attention_mask'];layout=case['variants'][t]['layout']
        actual_layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle);need(actual_layout['metadata']==layout,'Independent CPU native mRoPE layout differs')
        positions=raw['position_ids'];cached=record['use_cache'];query=[width-1] if t==0 else [0]
        ids=bundle['inputs']['input_ids'] if not cached or t==0 else bundle['inputs']['input_ids'][:,-1:]
        need(raw['label']==record['label'] and raw['context']==context and torch.equal(raw['input_ids'],ids)
             and torch.equal(raw['attention_mask'],mask) and torch.equal(raw['expected_mask'],mask),'Raw input/current-prefix ownership differs')
        expected_positions=actual_layout['position_ids'] if not cached or t==0 else torch.cat(((mask.cumsum(-1)-1)[:,-1:].unsqueeze(0),actual_layout['position_ids'][:,:,-1:]),0)
        need(torch.equal(positions,expected_positions) and torch.equal(positions,raw['expected_positions']) and positions.shape==(3 if not cached or t==0 else 4,n+1,ids.shape[1]),'Actual position-axis layout differs')
        # CPU recomputation uses the exact model-specific installed rope helper, no model.
        need(native.tensor_info(raw['common_penultimate_hidden'])['shape']==[n+1,ids.shape[1],3584]
             and raw['native_logits'].shape==(n+1,1,152064) and raw['native_logits'].dtype==torch.float16,'Native shape/dtype differs')
        masks=v11.old.mask_check(torch,raw['last_block_mask'],raw['attention_mask'],raw['cache_position'])
        need(masks==record['mask_audit'],'Native causal/padding mask audit differs')
        expected_visual={k:native.tensor_info(bundle['inputs'][k]) for k in ('pixel_values','image_grid_thw')} if not cached or t==0 else {}
        # Native cached preparation may retain image_grid_thw, but never pixels.
        need(all(raw['visual_input_identity'].get(k)==v for k,v in expected_visual.items())
             and (not cached or t==0 or 'pixel_values' not in raw['visual_input_identity']), 'Actual image ownership differs')
        metrics=[v11.old.metric(torch,raw['replay_logits'][i,0],raw['native_logits'][i,0]) for i in range(n+1)]
        need(metrics==record['head_replay'] and all(m['numerical_rule_passed'] for m in metrics),'Independent same-state native head replay failed')
        head_metrics+=metrics
        norm_exact=torch.equal(raw['native_normalized'],raw['replay_normalized'])
        need(raw['native_normalized'].shape==raw['replay_normalized'].shape==(n+1,1,3584)
             and norm_exact==record['normalized_exact'],'Normalized replay identity evidence differs')
        normalization_exact.append(norm_exact)
        if raw['fusion'] is not None:
            history=query if cached else list(range(width-1,width+t));stream=[width-1+t] if cached else history
            audit_capture(torch,raw,context['placement'],history,stream);writes+=1
            need(record['formula']['passed'] and record['formula']['ordinary_formula_exact']
                 and record['formula']['applied_override']==(raw['intervention'] is not None),'GPU functional core binding differs')
        if cached:
            need(raw['kv']==record['kv'] and len(raw['kv'])==28,'Actual KV evidence incomplete')
            for layer in raw['kv']:
                need(all(layer[k]['shape']==[n+1,4,mask.shape[1],128] and layer[k]['dtype']=='torch.float16' for k in ('key','value')),'Actual all-layer native KV shape differs')
            if record['old_cache_length']:
                need(record['previous_prefix_exact_by_layer']==[True]*28,'Past KV was modified')
                if context['condition']!='intervention':
                    previous=by_label[record['label'][:-1]+str(t-1)]['kv']
                    need(all(a['retained_'+k]==b[k] for a,b in zip(raw['kv'],previous) for k in ('key','value')),'Retained-prefix tensor digests differ')
            if context['condition'] in ('zero','active'):
                baseline=raw_for(f'N{n}_bare_cached{t}')
                need(torch.equal(raw['common_penultimate_hidden'],baseline['common_penultimate_hidden']),'Common lower read differs by placement')
                check=kv_compare(raw['kv'],baseline['kv'],context['placement'],context['condition'])
                saved=next(x for x in read(directory/'kv_checks.json') if x['label']==record['label'])
                need(all(saved[k]==v for k,v in check.items()),'KV contrast metadata differs')
                if context['condition']=='zero':need(torch.equal(raw['native_logits'],baseline['native_logits'])
                    and torch.equal(raw['actual_norm_input'],baseline['actual_norm_input']) and not bool(raw['fusion']['delta'].any()),'Zero identity failed independently')
                del baseline
        if context['kind']=='full':
            cached_raw=raw_for(f'N{n}_active_{context["placement"]}_cached{t}')
            cache_comparisons.extend(dict(n_frames=n,placement=context['placement'],step=t,row=i,
                cached_label=cached_raw['label'],full_label=raw['label'],binding=False,
                **v11.old.metric(torch,cached_raw['native_logits'][i,0],raw['native_logits'][i,0])) for i in range(n+1))
            other=raw_for(f'N{n}_active_{PLACEMENTS[1] if context["placement"]==PLACEMENTS[0] else PLACEMENTS[0]}_full{t}')
            need(torch.equal(raw['common_penultimate_hidden'],other['common_penultimate_hidden']),'Full-prefix pre/post common read differs')
            del cached_raw,other
        raw_headers[record['label']]=dict(query_tokens=ids.shape[1],key_tokens=mask.shape[1]);del raw
    saved_comparisons=read(directory/'comparisons.json')
    key=lambda x:(x['n_frames'],x['placement'],x['step'],x['row'])
    need(sorted(cache_comparisons,key=key)==sorted(saved_comparisons,key=key) and len(cache_comparisons)==328 and writes==37,'Fixed prefix/write coverage differs')
    persistence_data=read(directory/'persistence.json');need(sha(persistence_data['file'])==persistence_data['sha256'],'Persistence index differs')
    banks={k:load_bound(torch,v) for k,v in persistence_data['cache_artifacts'].items()};first=persistence_data['first_write']
    need(set(banks)=={'baseline','altered','restored'} and all(len(x)==28 for x in banks.values())
         and first==cases[16]['prompt_width']-1,'Persistence cache fork coverage/position differs')
    for bank,label in (('baseline','N16_active_pre_last_cached0'),('altered','N16_pre_last_perturb_prefill')):
        expected=by_label[label]['kv']
        need(all(native.tensor_info(value)==expected[i][field] for i,pair in enumerate(banks[bank])
                 for field,value in zip(('key','value'),pair)), 'Archived cache fork is not the actual bound native prefill')
    need([x['placement'] for x in persistence_data['results']]==list(PLACEMENTS),'Missing persistence placement')
    for item in persistence_data['results']:
        place=item['placement'];tags=['altered','restored'] if place=='pre_last' else ['altered']
        need(item['prefill_label']==f'N16_{place}_perturb_prefill'
             and item['baseline_prefill_label']==f'N16_active_{place}_cached0'
             and item['baseline_next_label']==f'N16_active_{place}_cached1'
             and [x['kind'] for x in item['outcomes']]==tags
             and [x['label'] for x in item['outcomes']]==[f'N16_{place}_{tag}_next' for tag in tags],
             'Persistence causal comparison/cardinality differs')
    for i,(base,altered,restored) in enumerate(zip(banks['baseline'],banks['altered'],banks['restored'])):
        for a,b,c in zip(base,altered,restored):
            need(torch.equal(a,c),'Restoration did not recover every baseline cache byte')
            if i<27:need(torch.equal(a,b),'Perturbation changed a lower-layer cache')
            else:
                need(torch.equal(a[:-1],b[:-1]) and torch.equal(a[-1,:,:first],b[-1,:,:first])
                     and torch.equal(a[-1,:,first+1:],b[-1,:,first+1:]) and not torch.equal(a[-1,:,first],b[-1,:,first]),
                     'Perturbation escaped the one first-query global K/V slot')
    del banks
    for result in persistence_data['results']:
        changed=raw_for(result['prefill_label']);baseline=raw_for(result['baseline_prefill_label'])
        common=changed['fusion']['global_states'].float();direction=torch.sin(torch.arange(3584,dtype=torch.float32));direction*=torch.rsqrt(direction.square().mean())
        expected=changed['intervention']['original_delta']+.1*common.square().mean(-1,keepdim=True).sqrt()*direction
        need(torch.allclose(expected,changed['fusion']['delta'],rtol=1e-5,atol=1e-6)
             and torch.equal(changed['intervention']['original_delta'],baseline['fusion']['delta']), 'Fixed intervention magnitude/ordinary delta differs')
        base_next=raw_for(result['baseline_next_label'])
        for item in result['outcomes']:
            actual=raw_for(item['label']);same=torch.equal(actual['native_logits'],base_next['native_logits'])
            need(same==item['all_row_logits_equal_baseline'] and same==(item['kind']=='restored' or result['placement']=='post_last')
                 and torch.equal(actual['native_logits'][:-1],base_next['native_logits'][:-1])
                 and torch.equal(actual['fusion']['delta'],base_next['fusion']['delta'])
                 and torch.equal(actual['common_penultimate_hidden'],base_next['common_penultimate_hidden'])
                 and v11.old.metric(torch,actual['native_logits'][-1,0],base_next['native_logits'][-1,0])==item['metric'],'Fixed-token persistence result differs')
            del actual
        del changed,baseline,base_next
    gradients=read(directory/'gradients.json');need([x['placement'] for x in gradients]==list(PLACEMENTS),'Missing placement gradient replay')
    for item in gradients:
        value=load_bound(torch,item);raw=raw_for(item['source_label']);gradient=value['gradient'];placement=item['placement'];q=raw['fusion']['query_indices']
        need(gradient.shape==value['deltas'].shape==(4,3584) and gradient.dtype==torch.float32 and bool(torch.isfinite(gradient).all())
             and bool((gradient[0]==0).all()) and bool(gradient[1].ne(0).any())==(placement=='pre_last')
             and bool(gradient[2].ne(0).any()) and bool((gradient[3]==0).all()),'Quantized temporal/future/crossscene derivative differs')
        need(value['write_indices']==[[q[1]]]+[[]]*15+[q] and value['loss_indices']==[[]]*16+[[q[1]]]
             and value['loss_to_write_indices']==[2] and bool((value['deltas'][0]==0).all())
             and torch.equal(value['deltas'][1:],raw['fusion']['delta']), 'Selected-loss/all-history replay layout differs')
        need(torch.equal(value['block_input'],raw['actual_last_block_input'])
             and tensor_equal(torch,value['attention_mask'],raw['last_block_mask'])
             and torch.equal(value['cache_position'],raw['cache_position'])
             and all(torch.equal(a,b) for a,b in zip(value['position_embeddings'],raw['position_embeddings'])),'Actual native replay inputs differ')
        metrics=[v11.old.metric(torch,value['native_shape_logits'][i,0],raw['native_logits'][i,0]) for i in range(17)]
        need(metrics==item['head_replay'] and all(m['numerical_rule_passed'] for m in metrics),'Independent full-batch final-block/head replay failed')
        head_metrics+=metrics
        loss=torch.nn.functional.cross_entropy(value['loss_logits'].float(),torch.tensor([entry['forced_ids'][1]]))
        need(value['loss_logits'].shape==(1,152064) and math.isclose(float(loss),value['loss'],rel_tol=1e-6,abs_tol=1e-6), 'Selected-only loss projection differs')
        del raw,value
    naturals=read(directory/'naturals.json');need([x['placement'] for x in naturals]==list(PLACEMENTS),'Two N16 natural trajectories required')
    natural_tokens=0
    for item in naturals:
        value=load_bound(torch,item);heads=load_bound(torch,dict(file=item['replay_file'],sha256=item['replay_sha256']))
        tokens=value['generated_ids'];steps=len(tokens);natural_tokens+=steps;n16=cases[16];width=n16['prompt_width']
        need(1<=steps<=8 and len(value['captures'])==len(heads)==len(item['partials'])==len(item['formulas'])==steps
             and all(x['passed'] and x['ordinary_formula_exact'] and not x['applied_override'] for x in item['formulas']) and item['rope_restored'], 'Natural runtime coverage/cleanup differs')
        need(value['metadata']['native_identity_sha256']==entry['native_identity_sha256']
             and value['metadata']['scene_input_identity']==runtime.input_identity(packet['bundles'][n16['case_id']+'_t0'],entry['native_identity_sha256'])
             and value['counters']==dict(model=steps,visual=1,language=steps,norm=steps,head=steps,broadcast=steps,selection=steps,probe_head=0), 'Natural runtime/input/native identity differs')
        need(torch.equal(value['raw_logits'],value['raw_logits'].half().float()) and value['raw_logits'].shape==(steps,152064)
             and value['raw_logits'].argmax(-1).tolist()==tokens and [x['top1_token_id'] for x in value['logit_records']]==tokens,
             'Natural unfiltered global argmax differs')
        eos=value['metadata']['generation']['native_eos_token_ids']
        need(not any(t in eos for t in tokens[:-1]) and value['completed']==(tokens[-1] in eos)
             and value['truncated']==(tokens[-1] not in eos) and (value['completed'] or steps==8),'Natural EOS/length status differs')
        for t,(cap,replay,ref) in enumerate(zip(value['captures'],heads,item['partials'])):
            actual=load_bound(torch,ref)
            need(cap['query_indices']==([width-1] if t==0 else [0]) and cap['stream_positions']==[width-1+t]
                 and cap['write_location']==item['placement'] and cap['selection_mode']=='clip'
                 and torch.equal(cap['fused_query_hidden'],actual['capture']['fused_query_hidden'])
                 and torch.equal(value['raw_logits'][t],actual['native_logits'][-1,0].float()),'Natural current-write/capture/recorder differs')
            need(torch.equal(cap['messages'],cap['gates'].unsqueeze(-1)*cap['payload']) and torch.equal(cap['gates'],cap['scores'].clamp(0,1))
                 and torch.equal(cap['native_delta'],cap['delta'].half()) and torch.equal(cap['write_output'],cap['write_input']+cap['native_delta']), 'Natural learned selector/cast differs')
            metrics=[v11.old.metric(torch,replay['logits'][i,0],actual['native_logits'][i,0]) for i in range(17)]
            need(metrics==replay['all_rows']==item['all_row_head_replay'][t] and all(m['numerical_rule_passed'] for m in metrics), 'Natural full-batch native head evidence differs')
            head_metrics+=metrics
        del value,heads
    need(summary['counts']==dict(model=43+natural_tokens,visual=22,last_block=45+natural_tokens,head=45+natural_tokens,core=37+natural_tokens)
         and summary['extra_head_calls']==47+natural_tokens<=63 and natural_tokens<=16,'Independent complete call count differs')
    return dict(passed=True,model_key=summary['model_key'],slurm_job_id=summary['slurm_job_id'],native_identity=entry['native_identity'],
        native_identity_sha256=entry['native_identity_sha256'],counts=summary['counts'],extra_head_calls=47+natural_tokens,
        all_row_head_comparisons=len(head_metrics),maximum_head_tv=max(m['full_vocabulary_tv'] for m in head_metrics),
        normalized_exact_count=sum(normalization_exact),fixed_normalization_comparisons=len(normalization_exact),
        zero_checks=12,common_read_checks=24,all_layer_kv_contrasts=24,persistence_restoration_passed=True,
        actual_quantized_temporal_gradients_passed=True,cached_full_comparisons=328,
        cached_full_failures=[x for x in cache_comparisons if not x['numerical_rule_passed']],
        natural_tokens=natural_tokens,natural_global_prefixes_retained=True,
        kv_scope='All28-layer equality/past-prefix digest checks bind fixed calls; natural runtime binds native masks/cache lengths/current token history',
        same_state_replay_passed=True,no_accuracy=True,no_fit=True)


def report(args,out,frozen):
    import torch
    torch.set_num_threads(4);plan=verify_plan(args.plan);packet=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    runs=[verify_run(p,plan) for p in args.runs];need({s['model_key'] for _,s in runs}==set(MODELS) and len(runs)==2,'Exactly one complete run of each native model required')
    results=[]
    for directory,summary in runs:
        results.append(audit_run(torch,directory,summary,plan,packet))
    resources=accounting(out);successful={r['job_id']:r for r in resources['jobs']}
    for _,summary in runs:
        row=successful[summary['slurm_job_id']];need(row['state']=='COMPLETED' and row['exit_code']=='0:0' and row['gpus']==1,'Successful source-bound software allocation missing')
    analysis=dict(passed=True,completed=True,protocol=PROTOCOL,source_sha256=frozen,plan_file=str(args.plan),plan_sha256=sha(args.plan),
        runs=[dict(directory=str(d),summary_sha256=sha(d/'summary.json'),audit=a) for (d,_),a in zip(runs,results)],resources=resources,
        no_model_or_head_execution=True,no_fit=True,no_accuracy=True,
        conclusion_scope='Native software, write persistence and fixed-token continuous credit only; no beneficial memory or reasoning claim')
    save(out/'analysis.json',analysis)
    return dict(passed=True,completed=True,phase='report',protocol=PROTOCOL,source_sha256=frozen,plan_file=str(args.plan),plan_sha256=sha(args.plan),
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),models=list(MODELS),resources=resources,
        computational_integrity_passed=True,native_replay_passed=True,temporal_gradient_passed=True,persistence_restoration_passed=True)


def verify_profile(path):
    """Read-only JSON/source/hash verifier of the completed two-model CPU report."""
    path=Path(path);path=path/'summary.json' if path.is_dir() else path;summary=read(path);plan=verify_plan(summary['plan_file'])
    need(summary['passed'] and summary['completed'] and summary['phase']=='report' and summary['protocol']==PROTOCOL
         and summary['source_sha256']==sources() and summary['plan_sha256']==sha(summary['plan_file'])
         and sha(summary['analysis_file'])==summary['analysis_sha256'],'Completed independent software report required')
    analysis=read(summary['analysis_file']);need(analysis['passed'] and analysis['completed'] and analysis['source_sha256']==sources(),'Independent report differs')
    for name,h in summary['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'Report source snapshot differs')
    for item in analysis['runs']:
        directory=Path(item['directory']);need(sha(directory/'summary.json')==item['summary_sha256'],'Audited GPU summary changed');verify_run(directory,plan)
    return summary


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true');modes.add_argument('--profile',action='store_true');modes.add_argument('--report',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--model',choices=MODELS);parser.add_argument('--controller-proof',type=Path)
    parser.add_argument('--runs',nargs=2,type=Path);args=parser.parse_args();native.require_slurm(gpu=args.profile)
    if args.check:need(args.controller_proof is not None,'Explicit completed controller CPU proof is required')
    if args.profile:need(args.plan is not None and args.model in MODELS,'Exact CPU plan and model are required')
    if args.report:need(args.plan is not None and args.runs is not None,'Exact plan and both GPU run directories are required')
    phase='check' if args.check else 'profile' if args.profile else 'report';job=os.environ['SLURM_JOB_ID']
    suffix='_'+args.model if args.profile else '';out=OUT/(phase+'_'+job+suffix);out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);start=time.perf_counter()
    save(out/'request.json',dict(argv=sys.argv,slurm_job_id=job,phase=phase,source_sha256=frozen,policy=POLICY))
    try:
        result=check(args,out,frozen) if args.check else run(args,out,frozen) if args.profile else report(args,out,frozen)
        need(sources()==frozen,'Sources changed during execution');result.update(slurm_job_id=job,seconds=time.perf_counter()-start)
        save(out/'summary.json',result);print(json.dumps(dict(passed=True,directory=str(out),phase=phase)),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),slurm_job_id=job,source_sha256=frozen,
            seconds=time.perf_counter()-start,partial_outputs_retained=True));raise


if __name__=='__main__':main()
