"""V11 software-only native aggregation write-location and temporal-gradient probe.

Fixed old software cases and one fixed K10 training scene, no fitting or
accuracy evaluation. Both write sites read the same penultimate hidden states.
All native cached/full historical writes and actual native masks are audited.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import profile_native_vision_v7_runtime as old
from scripts import native_vision_v7_runtime as native
from scripts.stage_native_vision_v6_teacher import MODEL,need,read,save,sha,model_metadata
OUT=REPO/'outputs/native_aggregation_vlm/v11/memory_software'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v11_memory_software')
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/v11_memory_software')
ANCESTOR=REPO/'outputs/native_aggregation_vlm/v7/runtime/profile_441845/summary.json'
TRAIN=Path('/mnt/data/gabriele/gnn_transformer/v10_balanced/main_manifest.json')
PLACEMENTS=('pre_last','post_last')
FORCED=('Therefore',':')
OWN=tuple(dict.fromkeys(('scripts/profile_native_vision_v11_memory.py',
    'slurm/native_vision_v11_memory_check.sbatch','slurm/native_vision_v11_memory_profile.sbatch',
    'gnnformer/parallel_local_memory.py','tests/test_parallel_local_memory.py',
    'scripts/native_vision_v11_last_block.py','tests/test_native_vision_v11_last_block.py',*old.OWN)))
BUDGET=dict(model=39,visual=19,last_block_replay=16,native_head_replay_calls=39,native_head_replay_global_positions=41)


def sources():return {p:sha(REPO/p) for p in OWN}


def snapshot(out):
    (out/'source').mkdir()
    frozen=sources()
    for p,h in frozen.items():
        (out/'source'/p.replace('/','_')).write_bytes((REPO/p).read_bytes())
        need(sha(out/'source'/p.replace('/','_'))==h,'Source changed while snapshotting')
    save(out/'source_hashes.json',frozen)
    return frozen


def index(out,title):
    (out/'INDEX.md').write_text('# '+title+'\n\n[Plan](plan.json) · [Summary](summary.json) · [Sources](source_hashes.json)\n')


def runtime_identity():
    import torch,transformers
    return dict(torch_version=str(torch.__version__),transformers_version=str(transformers.__version__),
        bitsandbytes_version=importlib.metadata.version('bitsandbytes'))


def cpu_copy(value):
    import torch
    if isinstance(value,torch.Tensor):
        with torch.inference_mode(False),torch.no_grad():return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def check():
    import torch,transformers
    from transformers import AutoProcessor
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);started=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'check_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);index(out,'V11 memory software CPU freeze')
    command=[sys.executable,'-m','unittest','tests.test_parallel_local_memory','tests.test_native_vision_v11_last_block']
    tests=subprocess.run(command,check=False,capture_output=True,text=True,cwd=REPO)
    (out/'unit.stdout.txt').write_text(tests.stdout);(out/'unit.stderr.txt').write_text(tests.stderr)
    need(tests.returncode==0,'New controller or last-block replay CPU tests failed; preserve logs')
    previous=read(ANCESTOR)
    need(previous['computational_integrity_passed'] is True and previous['zero_identity_passed'] is True,'Old native software ancestor incomplete')
    parent=old.verify(Path(previous['plan_file']))
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,use_fast=False)
    need(fingerprint(processor,str(transformers.__version__))==parent['processor'] and runtime_identity()==parent['runtime'],
        'Old native processor/runtime differs')
    owner,fn,api=old.local.native_api(processor)
    need(api==parent['native_api'],'Installed native code differs')
    def rope(**kwargs):return fn(owner,**kwargs)
    forced=[]
    for text in FORCED:
        ids=processor.tokenizer(text,add_special_tokens=False)['input_ids']
        need(len(ids)==1 and ids[0] not in processor.tokenizer.all_special_ids,'Inherited forced software token is not one ordinary token')
        forced.extend(ids)
    old_inputs=torch.load(parent['prepared_file'],map_location='cpu',weights_only=True)
    states=torch.load(parent['initial_file'],map_location='cpu',weights_only=True)
    need(states['schema_version']==1 and old.state_identity(states['states'])==parent['initial_state_identity'],'Fixed untrained branch states differ')
    manifest=read(TRAIN)
    rows=sorted((r for r in manifest['splits']['train_N16']['samples'] if r['gold']==10),key=lambda r:r['sid'])
    need(rows,'Missing registered K10 training support');sample=rows[0]
    need(sha(Path(sample['path'])/'qa.txt')==sample['qa_sha256'] and all(sha(im['path'])==im['sha256'] for im in sample['image_files']),
        'Fixed K10 source QA/images changed')
    targets=native.encode_target(processor.tokenizer,10)
    need(len(targets)==3 and targets[-1]==151645,'K10 must have two numeral tokens plus EOS')
    bundles={};cases=[]
    for case_id in ('parallel_N16','parallel_N64','gradient_N16_K10'):
        if case_id.startswith('parallel_'):base=old_inputs['bundles'][case_id];tokens=forced
        else:base=native.prepare_scene(processor,sample,'parallel',verify_processor_parity=True);tokens=targets[:-1]
        variants=[]
        for step in range(3):
            bundle=old.prefixed_bundle(base,tokens[:step]);layout=native.audit_layout(rope,bundle)
            variants.append(dict(input_identity=bundle['metadata']['input_identity'],layout=layout['metadata']))
            bundles[f'{case_id}_t{step}']=bundle
        cases.append(dict(case_id=case_id,sid=base['metadata']['sid'],n_frames=base['metadata']['n_frames'],
            question=base['metadata']['question'],prompt_width=base['metadata']['original_prompt_width'],
            forced_prefix_ids=tokens,targets=targets if case_id.startswith('gradient') else None,
            variants=variants,scope='K10 training gradient fixture' if case_id.startswith('gradient') else 'unchanged old software case'))
    data=DATA/f'check_{job}';data.mkdir(parents=True,exist_ok=False)
    weights=CKPT/f'check_{job}';weights.mkdir(parents=True,exist_ok=False)
    torch.save(dict(schema_version=1,bundles=bundles),data/'prepared.pt');torch.save(states,weights/'initial.pt')
    index(data,'V11 fixed native input tensors');index(weights,'V11 fixed untrained branch tensors')
    plan=dict(schema_version=1,protocol='v11_native_memory_software_only',source_sha256=frozen,
        ancestor_file=str(ANCESTOR),ancestor_sha256=sha(ANCESTOR),parent_plan_file=previous['plan_file'],parent_plan_sha256=sha(previous['plan_file']),
        model=model_metadata(),runtime=runtime_identity(),processor=parent['processor'],native_api=api,
        original_mixed_failure=parent['original_mixed_failure'],initial_state_identity=parent['initial_state_identity'],
        initial_file=str(weights/'initial.pt'),initial_sha256=sha(weights/'initial.pt'),
        prepared_file=str(data/'prepared.pt'),prepared_sha256=sha(data/'prepared.pt'),
        training_manifest=dict(file=str(TRAIN),sha256=sha(TRAIN)),gradient_scene=dict(sid=sample['sid'],qa_sha256=sample['qa_sha256'],gold=10),
        cases=cases,forced_text=list(FORCED),forced_ids=forced,maximum_calls=BUDGET,write_locations=list(PLACEMENTS),
        query_rule='Every historical assistant prediction position in full replay; current query only in native cached calls',
        read_boundary='Output of native penultimate decoder block, shared by both write locations',
        initialization='Unchanged V7 fixed seed20260927 zero and active normal(.001) output projection; no fit',
        gradient_objective='Native CE at second numeral(t1,target0), independent FP32 delta[3,H] leaves: rows0/1zero, row2=.1*linspace(-1,1,H); prior/future derivatives and unchanged earlier logits checked',
        numerical_policy='Same-captured-state native head and samebatch lastblock TV<=.02/exacttop1 binding; cached/full and global-only replay differences descriptive',
        no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True,
        unit_command=command,unit_stdout_sha256=sha(out/'unit.stdout.txt'),unit_stderr_sha256=sha(out/'unit.stderr.txt'),
        slurm_job_id=job)
    need(sources()==frozen,'Sources changed during CPU check')
    save(out/'plan.json',plan);(out/'plan.sha256').write_text(sha(out/'plan.json')+'\n');verify(out/'plan.json')
    save(out/'summary.json',dict(passed=True,source_sha256=frozen,plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),
        tests_passed=True,maximum_calls=BUDGET,no_model_loaded=True,seconds=time.perf_counter()-started,slurm_job_id=job))
    print(json.dumps(dict(passed=True,plan_file=str(out/'plan.json'),maximum_calls=BUDGET)),flush=True)


def verify(path):
    path=Path(path).resolve();plan=read(path)
    need(path.is_relative_to(OUT) and sha(path)==path.with_suffix('.sha256').read_text().strip(),'V11 plan path/sidecar differs')
    need(plan['schema_version']==1 and plan['protocol']=='v11_native_memory_software_only' and plan['source_sha256']==sources()
         and plan['model']==model_metadata() and plan['runtime']==runtime_identity() and plan['maximum_calls']==BUDGET,'Frozen V11 source/model/runtime/budget differs')
    for name,h in plan['source_sha256'].items():need(sha(path.parent/'source'/name.replace('/','_'))==h,'CPU source copy differs')
    for name in ('ancestor','parent_plan','prepared','initial'):need(sha(plan[name+'_file'])==plan[name+'_sha256'],'Frozen V11 artifact changed: '+name)
    need(sha(plan['training_manifest']['file'])==plan['training_manifest']['sha256'],'K10 training manifest changed')
    need([c['case_id'] for c in plan['cases']]==['parallel_N16','parallel_N64','gradient_N16_K10'] and plan['write_locations']==list(PLACEMENTS),
        'Fixed software cases/sites differ')
    return plan


class Capture:
    """One actual model call; raw penultimate output is captured before fusion."""
    def __init__(self,model):self.model=model;self.value={};self.handles=[]
    def __enter__(self):
        from gnnformer.runtime import get_layers
        layers=get_layers(self.model)
        self.handles=[self.model.model.language_model.register_forward_pre_hook(self.language,with_kwargs=True),
            layers[-2].register_forward_hook(self.penultimate,prepend=True),
            layers[-1].register_forward_pre_hook(self.last_input,with_kwargs=True),
            self.model.model.language_model.norm.register_forward_pre_hook(self.norm)]
        return self
    def __exit__(self,*_):
        for h in self.handles:h.remove()
        return False
    def language(self,module,args,kwargs):
        self.value['position_ids']=cpu_copy(kwargs['position_ids']);self.value['attention_mask']=cpu_copy(kwargs['attention_mask'])
    def penultimate(self,module,args,output):
        import torch
        hidden=output if isinstance(output,torch.Tensor) else output[0]
        self.value['common_penultimate_hidden']=cpu_copy(hidden)
    def last_input(self,module,args,kwargs):
        hidden=args[0] if args else kwargs['hidden_states']
        self.value['actual_last_block_input']=cpu_copy(hidden)
        self.value['last_block_mask']=cpu_copy(kwargs.get('attention_mask'))
        self.value['cache_position']=cpu_copy(kwargs['cache_position'])
        self.value['position_embeddings']=cpu_copy(kwargs['position_embeddings'])
    def norm(self,module,args):self.value['actual_norm_input']=cpu_copy(args[0])


def snapshot_kv(cache):
    import torch
    with torch.inference_mode(False),torch.no_grad():
        return [(layer.keys.detach().clone(),layer.values.detach().clone()) for layer in cache.layers]


def kv_structure(cache,baseline,placement,condition,first_write):
    import torch
    need(len(cache.layers)==len(baseline),'KV layer coverage differs')
    lower=[];local=[]
    for i,(layer,(k,v)) in enumerate(zip(cache.layers,baseline)):
        need(layer.keys.shape==k.shape and layer.values.shape==v.shape,'KV shape differs')
        lower.append(torch.equal(layer.keys,k) and torch.equal(layer.values,v))
        local.append(torch.equal(layer.keys[:-1],k[:-1]) and torch.equal(layer.values[:-1],v[:-1]))
    layer=cache.layers[-1];k,v=baseline[-1]
    prior=torch.equal(layer.keys[-1,:,:first_write],k[-1,:,:first_write]) and torch.equal(layer.values[-1,:,:first_write],v[-1,:,:first_write])
    key_changed=not torch.equal(layer.keys[-1,:,first_write],k[-1,:,first_write])
    value_changed=not torch.equal(layer.values[-1,:,first_write],v[-1,:,first_write])
    need(all(lower[:-1]) and all(local) and prior,'Memory write changed lower-layer/local-row or earlier prompt KV')
    if condition=='zero' or placement=='post_last':need(all(lower),'Zero/post-last write changed native KV')
    else:need(key_changed and value_changed,'Pre-last active write did not change the earliest written global K/V')
    return dict(lower_layers_exact=all(lower[:-1]),local_rows_all_layers_exact=all(local),
        final_layer_global_before_first_write_exact=prior,all_layers_exact=all(lower),
        first_written_global_key_changed=key_changed,first_written_global_value_changed=value_changed,
        first_write_position=first_write,per_layer_all_rows_exact=lower,per_layer_local_rows_exact=local)


def execute(model,inputs,*,cached,expected_positions,expected_mask,controller,data,label,records,keep=1,gradient_targets=None):
    import torch
    item=dict(inputs);item.pop('use_cache',None);previous=item.get('past_key_values');old_length=0 if previous is None else previous.get_seq_length()
    previous_kv=None if not old_length else snapshot_kv(previous)
    started=time.perf_counter()
    with Capture(model) as observed:
        with torch.inference_mode():output=model(**item,use_cache=cached,logits_to_keep=keep)
    torch.cuda.synchronize();raw=observed.value
    need(torch.equal(raw['position_ids'],expected_positions.cpu()) and torch.equal(raw['attention_mask'],expected_mask.cpu()),
        'Actual native positions/key mask differ from frozen layout')
    need(raw['common_penultimate_hidden'].dtype==raw['actual_norm_input'].dtype==output.logits.dtype==torch.float16,
        'Native memory pipeline dtype changed')
    mask=old.mask_check(torch,raw['last_block_mask'],raw['attention_mask'],raw['cache_position'])
    cache=output.past_key_values
    if cached:
        need(cache is not None and cache.get_seq_length()==expected_mask.shape[1],'Native cache length differs')
        prefix_exact=None if previous_kv is None else old.mixed.prefix_preserved(torch,cache,previous_kv,old_length)
    else:need(cache is None,'Full reference returned a cache');prefix_exact=None
    raw['input_ids']=cpu_copy(item['input_ids']);raw['native_logits']=cpu_copy(output.logits)
    need(bool(torch.isfinite(raw['native_logits']).all()),'Native logits are nonfinite')
    if controller is not None:
        controller.assert_complete();raw['fusion']=controller.export_last_capture(cpu=True)
    else:raw['fusion']=None
    raw.update(schema_version=1,label=label,use_cache=cached,old_cache_length=old_length)
    if gradient_targets is not None:raw['gradient_targets']=list(gradient_targets)
    path=data/(label+'.pt');torch.save(raw,path)
    records.append(dict(label=label,path=str(path),sha256=sha(path),use_cache=cached,model_calls=1,
        visual_calls=int(item.get('pixel_values') is not None),query_tokens=item['input_ids'].shape[1],key_tokens=expected_mask.shape[1],
        native_dtype=str(output.logits.dtype),mask_audit=mask,previous_prefix_exact_by_layer=prefix_exact,
        seconds=time.perf_counter()-started,controller_read_calls=None if controller is None else controller.read_calls,
        controller_write_calls=None if controller is None else controller.calls))
    save(data/f'forwards_after_{len(records):03d}.json',records)
    return cache,raw


def cached_input(model,bundle,cache,step,tokens,device):
    import torch
    from gnnformer.runtime import move_to_device,get_rope_index_fn
    layout=native.audit_layout(get_rope_index_fn(model),bundle);full=move_to_device(bundle['inputs'],device)
    if step==0:return full,layout['position_ids'],layout
    width=bundle['metadata']['original_prompt_width'];position=torch.tensor([width+step-1],device=device)
    prepared=model.prepare_inputs_for_generation(full['input_ids'],past_key_values=cache,attention_mask=full['attention_mask'],
        cache_position=position,use_cache=True,pixel_values=full['pixel_values'],image_grid_thw=full['image_grid_thw'])
    expected=torch.cat(((full['attention_mask'].cumsum(-1)-1)[:,-1:].unsqueeze(0),layout['position_ids'][:,:,-1:].to(device)),dim=0)
    need(prepared['input_ids'].shape==(full['input_ids'].shape[0],1) and bool((prepared['input_ids']==tokens[step-1]).all())
         and prepared.get('past_key_values') is cache and prepared.get('pixel_values') is None
         and torch.equal(prepared['position_ids'],expected),'Native cached shared-token preparation differs')
    return prepared,expected,layout


def verify_write(raw,placement,positions,stream_positions):
    import torch
    fusion=raw['fusion'];need(fusion is not None and fusion['write_location']==placement
        and fusion['query_indices']==positions and fusion['stream_positions']==stream_positions,'Controller omitted or shifted a historical write')
    h=raw['common_penultimate_hidden'];local=h[:-1,positions,:];global_=h[-1,positions,:]
    need(torch.equal(fusion['local_states'],local) and torch.equal(fusion['global_states'],global_),'Write locations do not share actual penultimate read states')
    delta=fusion['delta'];need(delta.dtype==torch.float32 and delta.shape==global_.shape and bool(torch.isfinite(delta).all()),'FP32 delta contract differs')
    expected=h.clone()
    if placement=='pre_last':expected[-1,positions,:]=global_+delta.half()
    need(torch.equal(raw['actual_last_block_input'],expected),'Actual final-block input omitted/moved a memory write')
    need(torch.equal(fusion['write_output'],fusion['write_input']+delta.half()),'Native FP16 cast/add order differs')
    need(torch.equal(raw['actual_norm_input'][-1,positions,:],fusion['norm_input_after_write']),
        'Actual native final norm did not receive the captured fused query states')
    return dict(passed=True,write_location=placement,query_indices=positions,stream_positions=stream_positions,
        historical_write_count=len(positions),delta_nonzero=bool(delta.ne(0).any()),native_cast_add_verified=True)


def head_replays(model,records):
    import torch
    result=[]
    norm,head=model.model.language_model.norm,model.lm_head
    for record in records:
        raw=torch.load(record['path'],map_location='cpu',weights_only=True);q=raw['native_logits'].shape[1]
        hidden=raw['actual_norm_input'][:,-q:,:].to(model.device)
        with torch.no_grad():logits=head.forward(norm.forward(hidden))
        for t in range(q):
            measured=old.metric(torch,logits[-1,t].detach().cpu(),raw['native_logits'][-1,t]);measured.update(label=record['label'],position=t,scope='same captured fused norm input; native full batch shape',binding=True)
            result.append(measured)
    need(len(result)==41,'Native captured-state head replay coverage differs')
    return result


def last_block_replays(model,records,replay_labels,gradient_label):
    import torch
    from gnnformer.runtime import get_layers
    from scripts.native_vision_v11_last_block import replay_last_block
    layers=get_layers(model);language=model.model.language_model;norm=language.norm;head=model.lm_head
    results=[];raw_results={};calls=0
    def replay(raw,placement,deltas,*,global_only=False):
        nonlocal calls
        h=raw['common_penultimate_hidden'].to(model.device);ids=raw['input_ids'].to(model.device)
        mask=raw['attention_mask'].to(model.device);positions=raw['position_ids'].to(model.device)
        indices=raw['fusion']['query_indices'] if raw['fusion'] is not None else list(range(h.shape[1]-3,h.shape[1]))
        if global_only:
            h=h[-1:];ids=ids[-1:];mask=mask[-1:];positions=positions[:,-1:,:];queries=[indices]
        else:queries=[[] for _ in range(h.shape[0]-1)]+[indices]
        calls+=1
        return replay_last_block(layers[-1],norm,head,language.rotary_emb,hidden_states=h,
            input_ids=ids,attention_mask=mask,position_ids=positions,query_indices=queries,deltas=deltas,placement=placement)
    for record in records:
        if record['label'] not in replay_labels:continue
        raw=torch.load(record['path'],map_location='cpu',weights_only=True);placement=raw['fusion']['write_location'];delta=raw['fusion']['delta'].to(model.device)
        with torch.no_grad():
            value=replay(raw,placement,delta)
            logits=head.forward(norm.forward(value['final_norm_input'][:,-1:,:]))
        measured=old.metric(torch,logits[-1,-1].cpu(),raw['native_logits'][-1,-1]);measured.update(label=record['label'],scope='same actual full batch final block and current-query native head shape',binding=True)
        def same(a,b):return a is None and b is None or isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a.detach().cpu(),b.detach().cpu())
        mask_exact=same(value['attention_mask'],raw['last_block_mask'])
        rotary_exact=all(same(a,b) for a,b in zip(value['position_embeddings'],raw['position_embeddings']))
        cache_position_exact=same(value['cache_position'],raw['cache_position'])
        need(mask_exact and rotary_exact and cache_position_exact,'Same-batch replay native mask/rotary/cache positions differ')
        measured.update(native_mask_exact=mask_exact,native_rotary_exact=rotary_exact,native_cache_positions_exact=cache_position_exact)
        results.append(measured)
        raw_results[record['label']]=dict(replayed_global_logits=cpu_copy(logits[-1,-1]),native_global_logits=raw['native_logits'][-1,-1],
            replayed_last_block_input=cpu_copy(value['block_input']),actual_last_block_input=raw['actual_last_block_input'])
        need(torch.equal(value['block_input'].detach().cpu(),raw['actual_last_block_input']),'Full native last-block replay did not retain every historical write')
        del value,raw
    need(len(results)==12,'Same-batch last-block replay coverage differs')
    record=next(r for r in records if r['label']==gradient_label);raw=torch.load(record['path'],map_location='cpu',weights_only=True)
    gradients=[];descriptive=[]
    for placement in PLACEMENTS:
        zeros=torch.zeros((3,3584),dtype=torch.float32,device=model.device,requires_grad=True)
        # Match the differentiated replay's grad-enabled native kernel route.
        value=replay(raw,placement,zeros,global_only=True)
        per_position=[old.metric(torch,value['logits'][t].detach().cpu(),raw['native_logits'][-1,t]) for t in range(3)]
        descriptive.append(dict(write_location=placement,scope='global-only zero-write training replay versus native full batch; descriptive',positions=per_position))
        zero_logits=cpu_copy(value['logits']);del value
        delta=zeros.detach().clone();delta[2]=.1*torch.linspace(-1,1,3584,device=model.device,dtype=torch.float32)
        delta.requires_grad_(True)
        value=replay(raw,placement,delta,global_only=True)
        earlier_logits_exact=torch.equal(value['logits'][:2].detach().cpu(),zero_logits[:2])
        future_logits_changed=not torch.equal(value['logits'][2].detach().cpu(),zero_logits[2])
        need(earlier_logits_exact and future_logits_changed,'A future write changed earlier logits or had no current-position effect')
        # Independent delta leaves isolate the native causal path. The fixed
        # future perturbation cannot influence the second numeral objective.
        # No optimizer or branch fitting occurs.
        target=raw['gradient_targets'][1]
        loss=torch.nn.functional.cross_entropy(value['logits'][1:2].float(),torch.tensor([target],device=model.device))
        gradient=torch.autograd.grad(loss,delta,allow_unused=False)[0]
        need(bool(torch.isfinite(gradient).all()) and bool(gradient[1].ne(0).any()),'Second numeral loss has no finite current-write gradient')
        earlier=bool(gradient[0].ne(0).any());future=bool(gradient[2].ne(0).any())
        need(earlier==(placement=='pre_last') and not future,'Native temporal derivative does not match registered pre/post causal structure')
        need(not any(p.grad is not None or p.requires_grad for p in model.parameters()),'Frozen backbone gradient state changed')
        gradients.append(dict(write_location=placement,loss=float(loss.detach()),target_token_id=target,target_position=1,
            earlier_delta_nonzero=earlier,current_delta_nonzero=True,future_delta_nonzero=future,
            future_perturbation='delta[2]=.1*linspace(-1,1,3584); delta[0:2]=0',
            earlier_logits_bitexact_under_future_write=earlier_logits_exact,future_logits_changed=future_logits_changed,
            gradient_norms=[float(torch.linalg.vector_norm(g.float())) for g in gradient],passed=True))
        raw_results['gradient_'+placement]=dict(delta=cpu_copy(delta),gradient=cpu_copy(gradient),logits=cpu_copy(value['logits']),
            native_global_logits=raw['native_logits'][-1],zero_delta_global_only_logits=zero_logits,target_ids=raw['gradient_targets'])
        del value,gradient,delta
    need(calls==16,'Last-block-only call budget differs')
    return results,descriptive,gradients,raw_results,calls


def run(args):
    import torch,transformers
    from gnnformer.runtime import load_runtime,get_layers,get_rope_index_fn,move_to_device
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.parallel_local_memory import ParallelLocalMemory
    from scripts.probe_native_vision_v2_prefix import fingerprint
    torch.set_num_threads(4);begin=time.perf_counter();job=os.environ['SLURM_JOB_ID']
    out=OUT/f'profile_{job}';out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out);index(out,'V11 native memory software profile')
    data=DATA/f'profile_{job}';data.mkdir(parents=True,exist_ok=False);index(data,'V11 raw native memory software evidence')
    plan=verify(args.plan);cpu=read(Path(args.plan).parent/'summary.json')
    need(cpu['passed'] and cpu['tests_passed'] and cpu['plan_sha256']==sha(args.plan),'Completed exact CPU freeze required')
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    blob=torch.load(plan['prepared_file'],map_location='cpu',weights_only=True)
    weights=torch.load(plan['initial_file'],map_location='cpu',weights_only=True)
    need(blob['schema_version']==weights['schema_version']==1 and old.state_identity(weights['states'])==plan['initial_state_identity'],
        'Fixed prepared inputs or unfitted branch states differ')
    tick=time.perf_counter();loaded=load_runtime(str(MODEL),use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model=loaded.model;model.eval();model.requires_grad_(False);native.native_contract(model)
    need(fingerprint(loaded.processor,str(transformers.__version__))==plan['processor'] and runtime_identity()==plan['runtime'],
        'Loaded native processor/runtime differs')
    _,_,api=old.local.native_api(loaded.processor);need(api==plan['native_api'],'Loaded native source differs')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-tick;device=loaded.device
    layers=get_layers(model);norm=model.model.language_model.norm
    branch=ParallelLocalAggregation().to(device=device).eval();native.native_contract(model,branch)
    parameter_versions={name:p._version for name,p in model.named_parameters()}
    counts=dict(model=0,visual=0,last_block=0)
    def count(name):
        def hook(module,args):counts[name]+=1
        return hook
    handles=[model.register_forward_pre_hook(count('model')),model.model.visual.register_forward_pre_hook(count('visual')),
        layers[-1].register_forward_pre_hook(count('last_block'))]
    records=[];zero_checks=[];write_checks=[];kv_checks=[];comparisons=[];controller_calls=[];replay_labels=set()
    torch.cuda.reset_peak_memory_stats()
    try:
        for case in plan['cases'][:2]:
            case_id=case['case_id'];width=case['prompt_width'];tokens=case['forced_prefix_ids'];n=case['n_frames']
            bundles=[blob['bundles'][f'{case_id}_t{step}'] for step in range(3)]
            for step,bundle in enumerate(bundles):
                layout=native.audit_layout(get_rope_index_fn(model),bundle)
                need(bundle['metadata']['input_identity']==case['variants'][step]['input_identity']
                    and layout['metadata']==case['variants'][step]['layout'],'Frozen old software layout differs')
            baseline=[];baseline_kv=[];cache=None
            for step,bundle in enumerate(bundles):
                inputs,positions,layout=cached_input(model,bundle,cache,step,tokens,device)
                cache,raw=execute(model,inputs,cached=True,expected_positions=positions,expected_mask=bundle['inputs']['attention_mask'],
                    controller=None,data=data,label=f'{case_id}__native__cached{step}',records=records)
                baseline.append(raw);baseline_kv.append(snapshot_kv(cache))
            del cache
            for condition in ('zero','active'):
                branch.load_state_dict(weights['states'][condition],strict=True)
                need(old.state_identity({condition:branch.state_dict()})[condition]==plan['initial_state_identity'][condition],
                    'Fixed branch initialization changed')
                for placement in PLACEMENTS:
                    cache=None
                    controller=ParallelLocalMemory(layers[-2],norm,branch,n_local_rows=n,write_location=placement,
                        query_indices=[width-1],stream_positions=[width-1],capture=True)
                    with controller:
                        for step,bundle in enumerate(bundles):
                            query=[width-1] if step==0 else [0];stream=[width-1+step]
                            controller.configure_queries(query,stream)
                            inputs,positions,layout=cached_input(model,bundle,cache,step,tokens,device)
                            label=f'{case_id}__{condition}__{placement}__cached{step}'
                            cache,raw=execute(model,inputs,cached=True,expected_positions=positions,expected_mask=bundle['inputs']['attention_mask'],
                                controller=controller,data=data,label=label,records=records)
                            common_exact=torch.equal(raw['common_penultimate_hidden'],baseline[step]['common_penultimate_hidden'])
                            need(common_exact,'A final-block/post-last write changed its common penultimate read')
                            write=verify_write(raw,placement,query,stream);write.update(label=label,common_read_matches_native=common_exact)
                            write_checks.append(write)
                            kv=kv_structure(cache,baseline_kv[step],placement,condition,width-1);kv.update(label=label,passed=True);kv_checks.append(kv)
                            if condition=='zero':
                                exact=torch.equal(raw['native_logits'],baseline[step]['native_logits'])
                                need(exact and not write['delta_nonzero'],'Zero-initialized controller is not exact native identity')
                                zero_checks.append(dict(label=label,raw_all_row_logits_exact=True,delta_exact_zero=True,passed=True))
                            else:
                                need(write['delta_nonzero'],'Fixed active branch produced no software perturbation')
                                if step==0:replay_labels.add(label)
                                if step>0:
                                    # Recompute the complete prefix with every earlier write.
                                    # Preserve native generation mRoPE state for the next cached step.
                                    history=list(range(width-1,width+step));controller.configure_queries(history,history)
                                    rope_before=cpu_copy(model.model.rope_deltas)
                                    full_label=f'{case_id}__{condition}__{placement}__full{step}'
                                    full_inputs=move_to_device(bundle['inputs'],device)
                                    try:
                                        _,full=execute(model,full_inputs,cached=False,expected_positions=layout['position_ids'],
                                            expected_mask=bundle['inputs']['attention_mask'],controller=controller,data=data,label=full_label,records=records)
                                    finally:model.model.rope_deltas=rope_before.to(device)
                                    write=verify_write(full,placement,history,history);write.update(label=full_label,common_read_matches_native=None)
                                    write_checks.append(write);replay_labels.add(full_label)
                                    for row in range(n+1):
                                        metric=old.metric(torch,raw['native_logits'][row,-1],full['native_logits'][row,-1])
                                        metric.update(case_id=case_id,write_location=placement,step=step,row_index=row,
                                            role='global' if row==n else 'local',binding=False,scope='native cached versus complete full prefix with all historical writes')
                                        comparisons.append(metric)
                                    del full
                        controller.assert_complete()
                        expected_calls=3 if condition=='zero' else 5
                        need(controller.calls==controller.read_calls==controller.norm_calls==expected_calls,'Memory hook counts differ')
                        controller_calls.append(dict(case_id=case_id,condition=condition,write_location=placement,
                            read_calls=controller.read_calls,write_calls=controller.calls,norm_calls=controller.norm_calls,passed=True))
                    need(not controller.active,'Fusion hooks survived their context')
                    del cache
            del baseline,baseline_kv,bundles,raw
            save(out/f'forwards_{case_id}.json',records)
            save(out/f'structural_checks_{case_id}.json',dict(zero=zero_checks,writes=write_checks,kv=kv_checks,controllers=controller_calls))
            save(out/f'cached_full_comparisons_{case_id}.json',comparisons)
        gradient_case=plan['cases'][2];gradient_label='gradient_N16_K10__native__full2'
        gradient_bundle=blob['bundles']['gradient_N16_K10_t2'];layout=native.audit_layout(get_rope_index_fn(model),gradient_bundle)
        need(layout['metadata']==gradient_case['variants'][2]['layout']
            and gradient_bundle['metadata']['input_identity']==gradient_case['variants'][2]['input_identity'],'Fixed K10 gradient layout differs')
        _,raw=execute(model,move_to_device(gradient_bundle['inputs'],device),cached=False,expected_positions=layout['position_ids'],
            expected_mask=gradient_bundle['inputs']['attention_mask'],controller=None,data=data,label=gradient_label,records=records,
            keep=3,gradient_targets=gradient_case['targets'])
        del raw
        need(counts==dict(model=39,visual=19,last_block=39) and len(records)==39 and len(replay_labels)==12,'Fixed native forward coverage differs')
        head_metrics=head_replays(model,records)
        block_metrics,global_descriptive,gradient_checks,replay_raw,replay_calls=last_block_replays(model,records,replay_labels,gradient_label)
        torch.save(replay_raw,data/'last_block_replays.pt');del replay_raw
        need(counts==dict(model=39,visual=19,last_block=55),'Actual native/replay call counters differ')
    finally:
        for handle in handles:handle.remove()
        save(out/'forwards.json',records)
        save(out/'structural_checks.json',dict(zero=zero_checks,writes=write_checks,kv=kv_checks,controllers=controller_calls))
        save(out/'cached_full_comparisons.json',comparisons);save(out/'actual_call_counters.json',counts)
    need(len(zero_checks)==12 and len(kv_checks)==24 and len(write_checks)==32 and len(controller_calls)==8 and len(comparisons)==328,
        'Native structural or descriptive coverage differs')
    need(not any(p.requires_grad or p.grad is not None for p in model.parameters())
        and parameter_versions=={name:p._version for name,p in model.named_parameters()},'Frozen native model changed')
    need(old.state_identity({'active':branch.state_dict()})['active']==plan['initial_state_identity']['active'],'Untrained branch changed')
    need(sources()==frozen,'Frozen V11 sources changed during profile')
    native.native_contract(model,branch);torch.cuda.synchronize()
    save(out/'head_replay_metrics.json',head_metrics);save(out/'last_block_replay_metrics.json',block_metrics)
    save(out/'global_only_replay_descriptive.json',global_descriptive);save(out/'gradient_checks.json',gradient_checks)
    binding_failures=[x for x in head_metrics+block_metrics if not x['numerical_rule_passed']]
    cached_failures=[x for x in comparisons if not x['numerical_rule_passed']]
    global_failures=[dict(write_location=x['write_location'],position=i,**m) for x in global_descriptive
        for i,m in enumerate(x['positions']) if not m['numerical_rule_passed']]
    artifacts={p.name:dict(path=str(p),sha256=sha(p)) for p in sorted(out.glob('*.json'))}
    artifacts['last_block_replays.pt']=dict(path=str(data/'last_block_replays.pt'),sha256=sha(data/'last_block_replays.pt'))
    summary=dict(schema_version=1,protocol=plan['protocol'],completed=True,passed=not binding_failures,
        computational_integrity_passed=True,zero_identity_passed=True,temporal_gradient_passed=True,
        binding_numerical_gate_passed=not binding_failures,binding_numerical_failures=binding_failures,
        cache_full_differences_descriptive=True,cache_full_failures=cached_failures,global_only_replay_failures=global_failures,
        original_mixed_failure=plan['original_mixed_failure'],original_mixed_numerical_gate_passed=False,
        plan_file=str(Path(args.plan).resolve()),plan_sha256=sha(args.plan),source_sha256=frozen,
        model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=plan['native_api'],
        native_dtypes=dict(norm=str(norm.weight.dtype),head=str(model.lm_head.weight.dtype)),
        hardware=dict(gpu=torch.cuda.get_device_name(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        calls=counts,last_block_replay_calls=replay_calls,native_head_replay_calls=len(records),native_head_replay_global_positions=len(head_metrics),
        maximum_calls=BUDGET,controller_sequences=len(controller_calls),native_forward_artifacts=len(records),
        same_batch_last_block_replays=len(block_metrics),gradient_checks=gradient_checks,
        artifacts=artifacts,raw_data_directory=str(data),frozen_backbone_unchanged=True,branch_unfitted_unchanged=True,
        no_fit=True,no_efficacy_scoring=True,no_reasoning_composition_claim=True,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        load_seconds=load_seconds,seconds=time.perf_counter()-begin,slurm_job_id=job)
    save(out/'summary.json',summary)
    (out/'README.md').write_text('# V11 native memory software profile\n\nSoftware gates: '+str(summary['passed'])+'. '
        'No fitting or aggregation accuracy was measured. Cache/full and global-only replay differences remain descriptive, '
        'including all failures. Temporal gradients use independent residual leaves and second-numeral CE; a fixed future-only write also tests causal value preservation.\n')
    print(json.dumps(dict(directory=str(out),passed=summary['passed'],calls=counts,binding_failures=len(binding_failures),
        descriptive_cache_failures=len(cached_failures),global_only_failures=len(global_failures))),flush=True)
    need(summary['passed'],'Native same-state or same-full-batch replay numerical gate failed; preserve all evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path);args=parser.parse_args();native.require_slurm(gpu=args.run)
    if args.check:
        need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),'CPU freeze only')
        check()
    else:
        need(args.plan is not None,'Exact completed frozen CPU plan is required');run(args)


if __name__=='__main__':main()
