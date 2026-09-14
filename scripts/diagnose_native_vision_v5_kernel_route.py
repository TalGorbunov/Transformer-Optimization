"""Fixed-checkpoint V5 kernel-route intervention; diagnostic only, no new gate.

Compare native bitsandbytes dispatch with MatMul4Bit.apply for both cached and
full paths. Same native-ON first token for both routes/modes. No updates/weights
saved. All compute, including checkpoint and image audits, requires Slurm.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
DATA=Path('/mnt/data/gabriele/gnn_transformer')
OUT=REPO/'outputs/native_aggregation_vlm/v5/kernel_route_diagnostic'
RUN=REPO/'outputs/native_aggregation_vlm/v5/profile/mean/mean_seed4_20260910_200035_441450_2903314'
OWN=('scripts/diagnose_native_vision_v5_kernel_route.py',
     'slurm/native_aggregation_vision_v5_kernel_check.sbatch',
     'slurm/native_aggregation_vision_v5_kernel_diagnostic.sbatch',
     'scripts/diagnose_native_vision_v5_cache.py')


def ensure(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def own_sources():
    return {name:sha(REPO/name) for name in OWN}


def installed_sources():
    root=Path(importlib.util.find_spec('bitsandbytes').origin).parent
    return {str(root/name):sha(root/name) for name in
            ('autograd/_functions.py','nn/modules.py','functional.py','__init__.py')}


def make_plan(torch):
    from scripts.diagnose_native_vision_v5_cache import selected_epoch
    from scripts.native_aggregation_vlm_v4 import sample_metadata
    from scripts.profile_native_vision_v5_checks import logit_parity_metrics
    check=logit_parity_metrics(torch.tensor([1.,2.,3.,4.,5.])+8,torch.tensor([1.,2.,3.,4.,5.]))
    ensure(check['centered_rms_logit_difference']==0 and check['total_variation']==0 and check['raw_top1_equal'],
           'Metric shift invariance check failed')
    config=json.loads((RUN/'config.json').read_text())
    history=json.loads((RUN/'training.json').read_text())
    ensure(config['condition']=='mean' and config['profile'] and config['seed']==4,'Wrong failed profile')
    for name,digest in config['code_sha256'].items():
        ensure(sha(REPO/name)==digest,f'Frozen source changed: {name}')
    checkpoint=Path(config['checkpoint_root'])/'native_aggregation_vlm_v5'/config['run_id']/'best.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    epoch=selected_epoch(history)
    ensure(saved['architecture']==config['architecture'] and saved['epoch']==epoch
           and saved['step']==epoch and saved['dev']==history[epoch-1]['dev'],'Selected checkpoint metadata differs')
    ensure(float(saved['branch']['up.weight'].float().norm())>0,'Expected active U')
    manifest_path=DATA/'v4_diversity/profile_manifest.json'
    ensure(sha(manifest_path)==config['manifest_sha256'],'Manifest differs')
    manifest=json.loads(manifest_path.read_text())
    staged=json.loads((RUN/'data_manifest.json').read_text())
    records=[]
    for n in (16,64):
        row=dict(manifest['splits'][f'test_N{n}']['samples'][1])
        expected=staged[f'length_N{n}']['samples'][1]
        ensure(all(row[k]==expected[k] for k in ('sid','gold','qa_sha256','path')),'Unused C/D case changed')
        parsed=sample_metadata(Path(row['path']),n)
        ensure(all(row[k]==parsed[k] for k in ('sid','gold','qa_sha256')),'QA metadata differs')
        row['question']=parsed['question']
        ensure(sha(Path(row['path'])/'qa.txt')==row['qa_sha256'],'QA checksum differs')
        for picture in row['image_files']:
            ensure(sha(picture['path'])==picture['sha256'],'Image checksum differs')
        records.append(row)
    return dict(schema_version=1,source_sha256=own_sources(),frozen_profile_source_sha256=config['code_sha256'],
        installed_source_sha256=installed_sources(),bitsandbytes_version=importlib.metadata.version('bitsandbytes'),
        config=config,config_path=str(RUN/'config.json'),config_sha256=sha(RUN/'config.json'),
        training_path=str(RUN/'training.json'),training_sha256=sha(RUN/'training.json'),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_epoch=epoch,
        manifest=str(manifest_path),manifest_sha256=sha(manifest_path),records=records,
        scope='Diagnostic kernel intervention only; original V5 cache gate stays failed. No thresholds, efficacy or training.',
        intervention='Temporarily replace bitsandbytes.matmul_4bit dispatch with MatMul4Bit.apply for BOTH cached and full paths; restore in finally',
        prefix='Native-route enabled-branch first token held fixed across native/forced routes and ON/OFF',
        limitation='A reduction localizes a contributor in four-bit linear dispatch; remaining differences can arise in other kernels. This is not a proposed production change.')


def execute(torch,plan,plan_path):
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from transformers import LogitsProcessor,LogitsProcessorList,__version__ as transformers_version
    from scripts.diagnose_native_vision_v5_cache import Trace,compare_traces,cpu_tree,tensor_difference
    from scripts.profile_native_vision_v5_checks import logit_parity_metrics
    from gnnformer.runtime import load_runtime,move_to_device,get_layers
    from gnnformer.data import load_mmred_sample,build_prompt_inputs,build_count_prompt
    from gnnformer.carriers import attach_lora
    from gnnformer.independent_vision_aggregation import attach_independent_vision_aggregation
    ensure(torch.cuda.is_available(),'CUDA required')
    job=os.environ['SLURM_JOB_ID']; started=time.monotonic()
    output=OUT/f'run_{job}'; tensor_root=DATA/'v5_cache_kernel_diagnostics'/f'run_{job}'
    output.mkdir(parents=True,exist_ok=False); tensor_root.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    config=plan['config']
    ensure(str(torch.__version__)==config['torch_version'] and str(transformers_version)==config['transformers_version'],'Runtime versions changed')
    runtime=load_runtime(config['model'],use_4bit=True,attn_implementation='sdpa',device_map='cuda')
    model,processor=runtime.model,runtime.processor
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval(); layers=get_layers(model)
    settings={name:getattr(processor.image_processor,name,None) for name in
        ('size','min_pixels','max_pixels','patch_size','temporal_patch_size','merge_size')}
    ensure(settings==config['image_processor_settings'],'Processor settings changed')
    if runtime.tokenizer.pad_token_id is None:
        runtime.tokenizer.pad_token_id=runtime.tokenizer.eos_token_id
    branch=attach_independent_vision_aggregation(model,layer_index=14,rank=96,merge='mean')
    branch.capture_last_query_messages=True
    lora=attach_lora(layers,len(layers)-4,rank=8,alpha=16.,device=runtime.device)
    saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    ensure(saved['architecture']==config['architecture'] and saved['epoch']==plan['selected_epoch'],'Checkpoint metadata changed')
    branch.load_state_dict(saved['branch'],strict=True)
    ensure(set(saved['lora'])=={f'{i}.{name}' for i,name in lora.params},'LoRA keys differ')
    with torch.no_grad():
        for (index,name),(a,b) in lora.params.items():
            a.copy_(saved['lora'][f'{index}.{name}'][0].to(a))
            b.copy_(saved['lora'][f'{index}.{name}'][1].to(b))
    del saved
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    original_dispatch=bnb.matmul_4bit
    observations=[]

    class ForceFirst(LogitsProcessor):
        def __init__(self,length,token):
            self.length,self.token=length,token
        def __call__(self,ids,scores):
            if ids.shape[1]==self.length:
                forced=torch.full_like(scores,-torch.inf)
                forced[:,self.token]=0
                return forced
            return scores

    class Route:
        def __init__(self,name):
            self.name=name
            self.counts=dict(native_gemv=0,native_matmul=0,forced_matmul=0)
        def __enter__(self):
            ensure(bnb.matmul_4bit is original_dispatch,'Unexpected existing dispatch replacement')
            def dispatch(A,B,quant_state,out=None,bias=None):
                if self.name=='forced_matmul':
                    self.counts['forced_matmul']+=1
                    return MatMul4Bit.apply(A,B,out,bias,quant_state)
                vector=(A.numel()==A.shape[-1] and not A.requires_grad
                        and A.device.type!='hpu' and A.shape[-1]%quant_state.blocksize==0)
                self.counts['native_gemv' if vector else 'native_matmul']+=1
                return original_dispatch(A,B,quant_state=quant_state,out=out,bias=bias)
            bnb.matmul_4bit=dispatch
            return self
        def __exit__(self,*exc):
            bnb.matmul_4bit=original_dispatch

    try:
        with torch.no_grad():
            for record in plan['records']:
                sid,frames,question,_states,answer=load_mmred_sample(Path(record['path']))
                ensure(sid==record['sid'] and question==record['question'] and int(answer)==record['gold'],'Sample changed')
                resized=[]
                try:
                    resized=[frame.resize((392,392)) for frame in frames]
                    inputs=move_to_device(build_prompt_inputs(processor,resized,build_count_prompt(question,len(resized))),runtime.device)
                finally:
                    for frame in frames+resized:
                        frame.close()
                first_token=None; routes={}; tensors={}
                for route_name in ('native','forced_matmul'):
                    trace=Trace(torch,model,branch,layers[14].self_attn)
                    modes={}; route_tensors={}
                    try:
                        with Route(route_name) as route:
                            for mode in ('all','off'):
                                branch.mode=mode
                                processors=None if first_token is None else LogitsProcessorList([ForceFirst(inputs['input_ids'].shape[1],first_token)])
                                generated=model.generate(**inputs,do_sample=False,use_cache=True,min_new_tokens=2,max_new_tokens=2,
                                    repetition_penalty=1.,logits_processor=processors,return_dict_in_generate=True,output_logits=True)
                                ids=generated.sequences[0,inputs['input_ids'].shape[1]:]
                                ensure(len(ids)==2 and len(generated.logits)==2,'Expected two native generation steps')
                                if first_token is None:
                                    first_token=int(ids[0])
                                ensure(int(ids[0])==first_token,'Intervention changed controlled prefix')
                                cached=generated.logits[1][0].detach().clone()
                                prefill=generated.logits[0][0].detach().clone()
                                cached_trace=trace.last()
                                full=dict(inputs)
                                full['input_ids']=torch.cat((inputs['input_ids'],ids[:1][None]),dim=1)
                                for key,value in (('attention_mask',1),('token_type_ids',0)):
                                    if key in full:
                                        full[key]=torch.cat((full[key],torch.full_like(ids[:1][None],value)),dim=1)
                                uncached=model(**full,use_cache=False,logits_to_keep=1).logits[0,-1].detach().float().clone()
                                full_trace=trace.last()
                                modes[mode]=dict(logits=logit_parity_metrics(cached,uncached),trace=compare_traces(torch,cached_trace,full_trace))
                                route_tensors[mode]=dict(cached_logits=cached,full_logits=uncached,prefill_first_logits=prefill,
                                    cached_trace=cached_trace,full_trace=full_trace,
                                    raw_logit_difference=cached.float()-uncached.float(),generated_ids=ids.detach().clone())
                                del generated,full
                        ensure(bnb.matmul_4bit is original_dispatch,'Dispatch was not restored')
                    finally:
                        trace.remove()
                        bnb.matmul_4bit=original_dispatch
                    branch.mode='all'
                    modes['branch_replay']={}
                    for phase in ('cached','full'):
                        point=route_tensors['all'][phase+'_trace']
                        delta=branch(point['hidden'][None,None],point['raw_memory'],point['image_ends'],
                            point['query_positions'][-1:],point['language_mask'][-1:])[0,-1].detach().clone()
                        modes['branch_replay'][phase]=tensor_difference(torch,point['delta'],delta)
                        route_tensors['all'][phase+'_replay_delta']=delta
                    if route_name=='native':
                        ensure(route.counts['native_gemv']>0 and route.counts['native_matmul']>0,'Native route did not exercise both kernels')
                    else:
                        ensure(route.counts['forced_matmul']>0 and route.counts['native_gemv']==0,'Forced route not applied')
                    modes['dispatch_call_counts']=route.counts
                    routes[route_name]=modes; tensors[route_name]=route_tensors
                contrasts={}
                for mode in ('all','off'):
                    contrasts[mode]={}
                    native=tensors['native'][mode]; forced=tensors['forced_matmul'][mode]
                    contrasts[mode]['prefill_first_logits']=tensor_difference(torch,native['prefill_first_logits'],forced['prefill_first_logits'])
                    for phase in ('cached','full'):
                        contrasts[mode][phase+'_logits']=logit_parity_metrics(native[phase+'_logits'],forced[phase+'_logits'])
                        contrasts[mode][phase+'_upstream_hidden']=tensor_difference(torch,native[phase+'_trace']['hidden'],forced[phase+'_trace']['hidden'])
                tensors_path=tensor_root/f'{sid}.pt'
                torch.save(cpu_tree(torch,tensors),tensors_path)
                observation=dict(sid=sid,n_frames=record['n_frames'],first_token=first_token,routes=routes,
                    native_versus_forced=contrasts,tensors_path=str(tensors_path),tensors_sha256=sha(tensors_path),
                    original_dispatch_restored=bnb.matmul_4bit is original_dispatch)
                observations.append(observation)
                save_json(output/'observations.json',observations)
                print(json.dumps(dict(sid=sid,n_frames=record['n_frames'],routes={name:{mode:data[mode]['logits'] for mode in ('all','off')} for name,data in routes.items()})),flush=True)
                del tensors,route_tensors,cached_trace,full_trace,native,forced,point,inputs
                branch.reset_memory(); gc.collect()
    finally:
        bnb.matmul_4bit=original_dispatch
        branch.remove(); lora.remove()
    torch.cuda.synchronize()
    summary=dict(schema_version=1,plan_sha256=sha(plan_path),source_sha256=own_sources(),
        slurm_job_id=job,gpu=torch.cuda.get_device_name(0),elapsed_seconds=time.monotonic()-started,
        scope=plan['scope'],limitation=plan['limitation'],observations=observations,
        original_dispatch_restored=bnb.matmul_4bit is original_dispatch)
    save_json(output/'summary.json',summary)
    (output/'INDEX.md').write_text('# V5 kernel-route diagnosis\n\n[Summary](summary.json) · [Frozen plan](plan.json).\n\n'
        'Same checkpoint/prefix, native versus forced four-bit matmul dispatch. Original strict gate remains failed; no new acceptance threshold.\n')
    print(json.dumps(dict(output=str(output),elapsed_seconds=summary['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true'); group.add_argument('--plan',type=Path)
    args=parser.parse_args()
    ensure(os.environ.get('SLURM_JOB_ID'),'All diagnostic compute requires Slurm')
    ensure(os.environ.get('SLURM_JOB_PARTITION')==('cpu' if args.check else 'gpu'),'Wrong Slurm partition')
    ensure(bool(os.environ.get('SLURM_JOB_GPUS',''))!=args.check,'Wrong GPU allocation')
    import torch
    torch.set_num_threads(max(1,min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))))
    if args.check:
        plan=make_plan(torch)
        output=OUT/f'check_{os.environ["SLURM_JOB_ID"]}'
        output.mkdir(parents=True,exist_ok=False)
        for name in plan['source_sha256']:
            target=output/'code'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((REPO/name).read_bytes())
        save_json(output/'plan.json',plan)
        (output/'plan.sha256').write_text(sha(output/'plan.json')+'\n')
        (output/'INDEX.md').write_text('# V5 kernel diagnostic CPU freeze\n\n- [Plan](plan.json)\n- [Checksum](plan.sha256)\n- [Source snapshots](code/)\n')
        print(json.dumps(dict(passed=True,plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'))),flush=True)
    else:
        ensure(sha(args.plan)==args.plan.with_suffix('.sha256').read_text().strip(),'Plan checksum differs')
        plan=json.loads(args.plan.read_text())
        ensure(plan['source_sha256']==own_sources(),'Diagnostic source changed')
        ensure(plan['installed_source_sha256']==installed_sources(),'Installed bitsandbytes sources changed')
        ensure(plan['bitsandbytes_version']==importlib.metadata.version('bitsandbytes'),'Bitsandbytes version changed')
        for name,digest in plan['frozen_profile_source_sha256'].items():
            ensure(sha(REPO/name)==digest,'Frozen V5 source changed')
        for path_key,hash_key in (('checkpoint','checkpoint_sha256'),('manifest','manifest_sha256'),
            ('config_path','config_sha256'),('training_path','training_sha256')):
            ensure(sha(plan[path_key])==plan[hash_key],f'{path_key} changed')
        for row in plan['records']:
            ensure(sha(Path(row['path'])/'qa.txt')==row['qa_sha256'],'QA changed')
            for picture in row['image_files']:
                ensure(sha(picture['path'])==picture['sha256'],'Image changed')
        execute(torch,plan,args.plan)


if __name__=='__main__':
    main()
